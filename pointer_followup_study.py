"""Answer margins of approximate K/V on saved inputs, against the dense reference path.

margin: sparse prefill methods on mixed benchmarks. Prompts where a method and dense emit identical
tokens compare two greedy runs on one path, so the margin difference has no path-selection bias.
quant: uniform K/V cache quantization after dense prefill (keys per channel, values per token, recent
tokens kept at full precision), a simplified KIVI-style scheme applied only to the K/V decode reads.
"""

import argparse
import csv
import json
from pathlib import Path
import subprocess

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

from attention import AttentionBackend
from evaluate import _eos_ids
from pointer_patch_study import forced_margins, free_run, prefill_cache
from scoring import score_prediction


RESIDUAL_TOKENS = 256
MARGIN_FIELDS = ["benchmark", "sample_id", "condition", "step", "margin", "gold_logprob", "gold_token", "competitor_token"]


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("margin", "quant"))
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--methods", nargs="+", default=["fp_v1:0.01", "fp_v1:0.08", "fp_row_norm:0.08"])
    parser.add_argument("--bits", type=int, nargs="+", default=[4, 3, 2, 1])
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def quantize(tensor, bits, dim):
    values = tensor.float()
    body = values[:, :, :-RESIDUAL_TOKENS]
    low = body.amin(dim=dim, keepdim=True)
    step = (body.amax(dim=dim, keepdim=True) - low).clamp_min(1e-12) / (2**bits - 1)
    values[:, :, :-RESIDUAL_TOKENS] = ((body - low) / step).round() * step + low
    return values.to(tensor.dtype)


def quantized_cache(keys, values, bits):
    return [quantize(k, bits, 2) for k in keys], [quantize(v, bits, 3) for v in values]


@torch.inference_mode()
def main():
    args = arguments()
    args.out.mkdir(parents=True, exist_ok=True)
    inputs = []
    for path in args.inputs:
        inputs.extend(torch.load(path, map_location="cpu", weights_only=False))
    inputs = inputs[:args.limit]
    backend = AttentionBackend()
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, attn_implementation="sdpa", device_map="cuda"
    ).eval()
    model.requires_grad_(False)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    eos_ids = _eos_ids(model, tokenizer)
    (args.out / "metadata.json").write_text(json.dumps({
        "mode": args.mode,
        "inputs": [str(path) for path in args.inputs],
        "methods": args.methods,
        "bits": args.bits,
        "residual_tokens": RESIDUAL_TOKENS,
        "model_commit": model.config._commit_hash,
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "git_dirty": subprocess.check_output(["git", "status", "--short"], text=True),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
    }, indent=2), encoding="utf-8")
    margin_file = (args.out / "margins.csv").open("w", newline="", encoding="utf-8")
    margin_writer = csv.DictWriter(margin_file, MARGIN_FIELDS, extrasaction="ignore")
    margin_writer.writeheader()
    generations = (args.out / "generations.jsonl").open("w", encoding="utf-8")

    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        for sample in inputs:
            ids = torch.tensor([sample["input_ids"]], dtype=torch.long, device="cuda")
            last_token = sample["input_ids"][-1]
            position = len(sample["input_ids"]) - 1
            benchmark = sample["task_label"]
            reference = None

            def run(condition, keys, values):
                nonlocal reference
                generated = free_run(
                    model, keys, values, last_token, position, eos_ids, sample["max_new_tokens"], "cuda"
                )
                if reference is None:
                    reference = generated
                text = tokenizer.decode(generated, skip_special_tokens=True)
                scored = score_prediction(sample, text)
                generations.write(json.dumps({
                    "benchmark": benchmark,
                    "sample_id": sample["sample_id"],
                    "condition": condition,
                    "generated_token_ids": generated,
                    "raw_text": text,
                    "score": scored["score"],
                    "exact_match": scored["exact_match"],
                    "identical_to_dense": generated == reference,
                }) + "\n")
                generations.flush()
                rows = forced_margins(model, tokenizer, keys, values, last_token, position, reference, "cuda")
                for row in rows:
                    margin_writer.writerow({
                        "benchmark": benchmark,
                        "sample_id": sample["sample_id"],
                        "condition": condition,
                        **row,
                    })
                margin_file.flush()
                print(f"{sample['sample_id']} {condition}: score {scored['score']:.1f} identical {generated == reference}", flush=True)

            dense_keys, dense_values = prefill_cache(model, backend, "dense", None, ids)
            run("dense", dense_keys, dense_values)
            if args.mode == "margin":
                for spec in args.methods:
                    method, alpha = spec.split(":")
                    keys, values = prefill_cache(model, backend, method, float(alpha), ids)
                    run(f"{method}__a{float(alpha):g}", keys, values)
            else:
                keys, values = prefill_cache(model, backend, "fp_v1", 0.08, ids)
                run("fp_v1__a0.08", keys, values)
                for bits in args.bits:
                    run(f"kv_quant_{bits}bit", *quantized_cache(dense_keys, dense_values, bits))
            del ids

    margin_file.close()
    generations.close()


if __name__ == "__main__":
    main()
