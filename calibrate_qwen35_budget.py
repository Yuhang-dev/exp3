"""Measure complete Full/V1 generations on fixed CL-bench budget probes."""

import argparse
import importlib.metadata
import json
import time

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers import AutoModelForImageTextToText, AutoTokenizer
from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
from fla.ops.gated_delta_rule import chunk_gated_delta_rule, fused_recurrent_gated_delta_rule

from .attention import AttentionBackend
from .clbench_qwen35 import generate, validate_architecture, write_json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=32768)
    parser.add_argument("--decoding", choices=["greedy", "qwen35"], default="greedy")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(42)
    torch.set_num_threads(1)
    inputs = torch.load(args.inputs, map_location="cpu", weights_only=False)
    longest = max(range(len(inputs)), key=lambda i: inputs[i]["prompt_tokens"])
    indices = list(dict.fromkeys([longest, 0, 3]))
    metadata = {
        "status": "running",
        "inputs": str(args.inputs),
        "selection": "longest prompt, first finance task, first game-mechanics task",
        "sample_indices": indices,
        "max_new_tokens": args.max_new_tokens,
        "decoding": args.decoding,
        "torch": torch.__version__,
        "linear_kernels": {p: importlib.metadata.version(p) for p in ["fla-core", "causal-conv1d"]},
    }
    write_json(args.out / "metadata.json", metadata)
    backend = AttentionBackend(alpha=0.08)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation="sdpa",
        use_kernels=False, device_map="cuda",
    ).eval()
    model.requires_grad_(False)
    validate_architecture(model)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    records = []
    with (args.out / "generations.jsonl").open("w", encoding="utf-8") as output:
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            for index in indices:
                sample = inputs[index]
                input_ids = torch.tensor([sample["input_ids"]], device="cuda")
                for method in ["dense", "fp_v1"]:
                    backend.configure(method, 0.08)
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats()
                    torch.cuda.synchronize()
                    started = time.perf_counter()
                    print(f"Probe sample={index} method={method} prompt={sample['prompt_tokens']}", flush=True)
                    generated = generate(
                        model, tokenizer, input_ids, args.max_new_tokens,
                        thinking=True, progress_every=2048,
                        decoding=args.decoding, seed=42 + index,
                    )
                    torch.cuda.synchronize()
                    metrics = {
                        "sample_index": index,
                        "sample_id": sample["sample_id"],
                        "input_sha256": sample["input_sha256"],
                        "prompt_tokens": sample["prompt_tokens"],
                        "method": method,
                        "generated_tokens": len(generated["generated_token_ids"]),
                        "elapsed_seconds": time.perf_counter() - started,
                        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
                        "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
                        "end_reason": generated["end_reason"],
                        "thinking_status": generated["thinking_status"],
                        "final_answer_nonempty": bool(generated["final_text"].strip()),
                    }
                    output.write(json.dumps({**metrics, **generated}, ensure_ascii=False) + "\n")
                    output.flush()
                    records.append(metrics)
                    write_json(args.out / "summary.json", records)
                    print(json.dumps(metrics), flush=True)
    metadata["status"] = "complete"
    metadata["all_finished"] = all(
        r["end_reason"] == "eos" and r["thinking_status"] == "closed"
        and r["final_answer_nonempty"] for r in records
    )
    write_json(args.out / "metadata.json", metadata)


if __name__ == "__main__":
    main()
