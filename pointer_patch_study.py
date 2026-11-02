"""K/V patching, teacher-forced answer margins, and K/V drift on saved synthetic inputs.

Every method prefills input[:-1]; the final prompt token and all answer tokens then run through
dense single-token decode against a chosen K/V cache, so the only thing that differs between
conditions is the K/V the answer reads.
"""

import argparse
import csv
import json
from pathlib import Path
import subprocess

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

from attention import AttentionBackend
from evaluate import _eos_ids
from scoring import score_synthetic


BLOCK = 128
LAYERS = range(28)
# name, base cache, donor cache, position region, layers, parts restored from the donor
PATCHES = [
    ("dense_base", "dense", None, None, None, None),
    ("v1_base", "v1", None, None, None, None),
    ("v1_fix_all", "v1", "dense", "all", LAYERS, "kv"),
    ("v1_fix_all_k", "v1", "dense", "all", LAYERS, "k"),
    ("v1_fix_all_v", "v1", "dense", "all", LAYERS, "v"),
    ("v1_fix_needle", "v1", "dense", "needle", LAYERS, "kv"),
    ("v1_fix_needle_k", "v1", "dense", "needle", LAYERS, "k"),
    ("v1_fix_needle_v", "v1", "dense", "needle", LAYERS, "v"),
    ("v1_fix_needle_block", "v1", "dense", "needle_block", LAYERS, "kv"),
    ("v1_fix_filler", "v1", "dense", "filler", LAYERS, "kv"),
    ("v1_fix_question", "v1", "dense", "question", LAYERS, "kv"),
    ("v1_fix_L01_09", "v1", "dense", "all", range(1, 10), "kv"),
    ("v1_fix_L10_18", "v1", "dense", "all", range(10, 19), "kv"),
    ("v1_fix_L19_27", "v1", "dense", "all", range(19, 28), "kv"),
    ("v1_fix_L27", "v1", "dense", "all", range(27, 28), "kv"),
    ("dense_break_needle", "dense", "v1", "needle", LAYERS, "kv"),
    ("dense_break_filler", "dense", "v1", "filler", LAYERS, "kv"),
    ("dense_break_question", "dense", "v1", "question", LAYERS, "kv"),
]
EXTRA_METHODS = [
    ("fp_v1", 0.0),
    ("mean_native", 0.08),
    ("fp_row_norm", 0.08),
    ("mean_balanced", 0.08),
]
MARGIN_FIELDS = [
    "sample_id", "variant", "condition", "step", "target_index", "segment", "gold_id", "gold_token",
    "margin", "gold_logprob", "top1_id", "top1_token", "competitor_id", "competitor_token", "top5",
]


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def prefill_cache(model, backend, method, alpha, ids):
    backend.configure(method, alpha)
    torch.compiler.cudagraph_mark_step_begin()
    cache = model.model(input_ids=ids[:, :-1], use_cache=True).past_key_values
    return [k.cpu() for k in cache.key_cache], [v.cpu() for v in cache.value_cache]


def regions(sample, tokenizer, length):
    text = tokenizer.decode(sample["input_ids"], skip_special_tokens=False)
    offsets = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)["offset_mapping"]
    question_char = text.rindex(sample["question"])
    question = next(index for index, (_, right) in enumerate(offsets) if right > question_char)
    needle = torch.zeros(length, dtype=torch.bool)
    needle_block = torch.zeros(length, dtype=torch.bool)
    for evidence in sample["evidence_positions"]:
        start, end = evidence["token_span"]
        needle[start:end] = True
        needle_block[start // BLOCK * BLOCK:((end - 1) // BLOCK + 1) * BLOCK] = True
    question_mask = torch.zeros(length, dtype=torch.bool)
    question_mask[question:] = True
    return {
        "all": torch.ones(length, dtype=torch.bool),
        "needle": needle,
        "needle_block": needle_block,
        "filler": ~needle & ~question_mask,
        "question": question_mask,
    }


def patched(base, donor, mask, layers):
    output = list(base)
    for layer in layers:
        output[layer] = base[layer].clone()
        output[layer][:, :, mask] = donor[layer][:, :, mask]
    return output


def build_condition(caches, region_masks, base, donor, region, layers, parts):
    keys, values = caches[base]
    if donor is None:
        return keys, values
    donor_keys, donor_values = caches[donor]
    mask = region_masks[region]
    if "k" in parts:
        keys = patched(keys, donor_keys, mask, layers)
    if "v" in parts:
        values = patched(values, donor_values, mask, layers)
    return keys, values


def to_cache(keys, values):
    cache = DynamicCache()
    cache.key_cache = [k.cuda() for k in keys]
    cache.value_cache = [v.cuda() for v in values]
    cache._seen_tokens = keys[0].shape[2]
    return cache


def decode_step(model, cache, token, position, device):
    output = model(
        input_ids=torch.tensor([[token]], device=device),
        past_key_values=cache,
        use_cache=True,
        cache_position=torch.tensor([position], device=device),
        logits_to_keep=1,
    )
    return output.logits[0, -1].float()


def free_run(model, keys, values, last_token, position, eos_ids, limit, device):
    cache = to_cache(keys, values)
    generated = []
    token = last_token
    while len(generated) < limit and (not generated or generated[-1] not in eos_ids):
        token = decode_step(model, cache, token, position + len(generated), device).argmax().item()
        generated.append(token)
    return generated


def forced_margins(model, tokenizer, keys, values, last_token, position, gold, device):
    cache = to_cache(keys, values)
    rows = []
    token = last_token
    for step, gold_id in enumerate(gold):
        logits = decode_step(model, cache, token, position + step, device)
        top = logits.topk(6).indices.tolist()
        competitor = next(candidate for candidate in top if candidate != gold_id)
        rows.append({
            "step": step,
            "gold_id": gold_id,
            "gold_token": tokenizer.decode([gold_id]),
            "margin": (logits[gold_id] - logits[competitor]).item(),
            "gold_logprob": logits.log_softmax(-1)[gold_id].item(),
            "top1_id": top[0],
            "top1_token": tokenizer.decode([top[0]]),
            "competitor_id": competitor,
            "competitor_token": tokenizer.decode([competitor]),
            "top5": json.dumps([tokenizer.decode([candidate]) for candidate in top[:5]]),
        })
        token = gold_id
    return rows


def answer_segments(tokenizer, gold, sample):
    """Label each gold token by the character it starts on: key, prefix, index, suffix, or other."""
    pieces = [tokenizer.decode([token]) for token in gold]
    text = "".join(pieces)
    labels = [(-1, "other")] * len(text)
    for target_index, key in enumerate(sample["target_keys"]):
        value = sample["answers"][key]
        key_start = text.index(key)
        value_start = text.index(value, key_start)
        index_char = value_start + len(value) - 9
        for char in range(key_start, key_start + len(key)):
            labels[char] = (target_index, "key")
        for char in range(value_start, value_start + len(value)):
            segment = "prefix" if char < index_char else "index" if char == index_char else "suffix"
            labels[char] = (target_index, segment)
    output = []
    start = 0
    for piece in pieces:
        output.append(labels[start] if start < len(labels) else (-1, "other"))
        start += len(piece)
    return output


def drift_rows(sample, name, reference, other, region_masks):
    rows = []
    for layer in LAYERS:
        for part, index in (("k", 0), ("v", 1)):
            difference = other[index][layer].float() - reference[index][layer].float()
            base = reference[index][layer].float()
            for region, mask in region_masks.items():
                rows.append({
                    "sample_id": sample["sample_id"],
                    "comparison": name,
                    "layer": layer,
                    "part": part,
                    "region": region,
                    "rel_l2": (
                        difference[:, :, mask].norm() / base[:, :, mask].norm()
                    ).item(),
                })
    return rows


@torch.inference_mode()
def main():
    args = arguments()
    args.out.mkdir(parents=True, exist_ok=True)
    inputs = torch.load(args.inputs, map_location="cpu", weights_only=False)[:args.limit]
    backend = AttentionBackend()
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map="cuda",
    ).eval()
    model.requires_grad_(False)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    eos_ids = _eos_ids(model, tokenizer)
    (args.out / "metadata.json").write_text(json.dumps({
        "inputs": str(args.inputs),
        "model_commit": model.config._commit_hash,
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "git_dirty": subprocess.check_output(["git", "status", "--short"], text=True),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "patches": [patch[0] for patch in PATCHES],
        "extra_methods": EXTRA_METHODS,
    }, indent=2), encoding="utf-8")

    margin_file = (args.out / "margins.csv").open("w", newline="", encoding="utf-8")
    margin_writer = csv.DictWriter(margin_file, MARGIN_FIELDS)
    margin_writer.writeheader()
    drift_file = (args.out / "kv_drift.csv").open("w", newline="", encoding="utf-8")
    drift_writer = csv.DictWriter(
        drift_file, ["sample_id", "comparison", "layer", "part", "region", "rel_l2"]
    )
    drift_writer.writeheader()
    generations = (args.out / "generations.jsonl").open("w", encoding="utf-8")

    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        for sample in inputs:
            ids = torch.tensor([sample["input_ids"]], dtype=torch.long, device="cuda")
            last_token = sample["input_ids"][-1]
            position = len(sample["input_ids"]) - 1
            caches = {
                "dense": prefill_cache(model, backend, "dense", None, ids),
                "v1": prefill_cache(model, backend, "fp_v1", 0.08, ids),
            }
            region_masks = regions(sample, tokenizer, position)
            drift_writer.writerows(drift_rows(sample, "v1_vs_dense", caches["dense"], caches["v1"], region_masks))
            gold = None
            segments = None

            def evaluate_condition(name, keys, values):
                nonlocal gold, segments
                generated = free_run(
                    model, keys, values, last_token, position, eos_ids, sample["max_new_tokens"], "cuda"
                )
                if gold is None:
                    gold = generated
                    segments = answer_segments(tokenizer, gold, sample)
                text = tokenizer.decode(generated, skip_special_tokens=True)
                scored = score_synthetic(text, sample["answers"], sample["target_keys"])
                generations.write(json.dumps({
                    "sample_id": sample["sample_id"],
                    "variant": sample["variant"],
                    "condition": name,
                    "generated_token_ids": generated,
                    "text": text,
                    "parsed_answer": scored["parsed_answer"],
                    "target_correct": scored["target_correct"],
                    "answers": sample["answers"],
                    "target_keys": sample["target_keys"],
                }) + "\n")
                generations.flush()
                for row, (target_index, segment) in zip(
                    forced_margins(model, tokenizer, keys, values, last_token, position, gold, "cuda"),
                    segments,
                ):
                    margin_writer.writerow({
                        "sample_id": sample["sample_id"],
                        "variant": sample["variant"],
                        "condition": name,
                        "target_index": target_index,
                        "segment": segment,
                        **row,
                    })
                margin_file.flush()
                print(f"{sample['sample_id']} {name}: correct {sum(scored['target_correct'].values())}/3 {text}", flush=True)

            for name, base, donor, region, layers, parts in PATCHES:
                keys, values = build_condition(caches, region_masks, base, donor, region, layers, parts)
                evaluate_condition(name, keys, values)

            for method, alpha in EXTRA_METHODS:
                name = f"{method}__a{alpha:g}"
                extra = prefill_cache(model, backend, method, alpha, ids)
                drift_writer.writerows(drift_rows(sample, f"{name}_vs_dense", caches["dense"], extra, region_masks))
                evaluate_condition(name, *extra)
            drift_file.flush()
            del ids, caches

    margin_file.close()
    drift_file.close()
    generations.close()


if __name__ == "__main__":
    main()
