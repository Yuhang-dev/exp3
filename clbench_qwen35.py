"""Paired Qwen3.5-4B Full/V1 generation for a fixed CL-bench panel."""

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
import csv
import hashlib
import json
from pathlib import Path
import platform
import shutil
import subprocess
import time
import traceback

import numpy as np
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
import transformers
from transformers import AutoModelForImageTextToText, AutoTokenizer
import triton

from .attention import AttentionBackend
from .upstream import flashprefill_native_forward as upstream


MODEL_ID = "Qwen/Qwen3.5-4B"
MODEL_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
DATASET_ID = "tencent/CL-bench"
DATASET_REVISION = "b28a5832a09b0d96c0cf4c22e90d7c60ede25b80"
DATASET_SHA256 = "d5fc88d4b2eea75c61dd40862021b6ae2fba26bd21b58e8c5e18377a763943be"
EVAL_REPOSITORY = "Tencent-Hunyuan/CL-bench"
EVAL_COMMIT = "16bffd1cfa05927e72ec75c835177d6e23e82172"
EXPECTED_FULL_LAYERS = [3, 7, 11, 15, 19, 23, 27, 31]


@dataclass(frozen=True)
class Candidate:
    method: str
    alpha: float | None
    config_id: str


def alpha_label(alpha):
    return f"{alpha:g}".replace("-", "m").replace(".", "p")


def candidates(alpha):
    return [
        Candidate("dense", None, "dense"),
        Candidate("fp_v1", float(alpha), f"fp_v1__a{alpha_label(alpha)}"),
    ]


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="models/Qwen3.5-4B")
    parser.add_argument("--data", type=Path, default=Path("datasets/cl_bench/CL-bench.jsonl"))
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--context-limit",
        type=int,
        default=65536,
        help="Maximum prompt plus generation tokens; inputs are never truncated.",
    )
    parser.add_argument("--min-prompt-tokens", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--alpha", type=float, default=0.08)
    parser.add_argument("--inputs", type=Path, help="Reuse an exact inputs.pt snapshot.")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--selector-chunk-tiles", type=int, default=8)
    thinking = parser.add_mutually_exclusive_group()
    thinking.add_argument("--thinking", dest="thinking", action="store_true")
    thinking.add_argument("--no-thinking", dest="thinking", action="store_false")
    parser.set_defaults(thinking=True)
    parser.add_argument("--out", type=Path, default=Path("results/clbench_qwen35_full100"))
    return parser.parse_args()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value):
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def token_sha256(token_ids):
    return hashlib.sha256(np.asarray(token_ids, dtype="<i4").tobytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def read_jsonl(path):
    rows = []
    with Path(path).open(encoding="utf-8") as source:
        for line in source:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path, rows):
    with Path(path).open("w", encoding="utf-8") as output:
        for row in rows:
            output.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path, fields, rows):
    with Path(path).open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


class CsvSink:
    def __init__(self, path, fields, mode):
        path = Path(path)
        has_rows = path.is_file() and path.stat().st_size > 0
        self.file = path.open(mode, newline="", encoding="utf-8")
        self.writer = csv.DictWriter(self.file, fieldnames=fields)
        if mode == "w" or not has_rows:
            self.writer.writeheader()

    def write(self, row):
        self.writer.writerow(row)
        self.file.flush()

    def close(self):
        self.file.close()


def encode_messages(tokenizer, messages, thinking):
    token_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        return_dict=False,
        add_generation_prompt=True,
        enable_thinking=thinking,
    )
    if isinstance(token_ids, torch.Tensor):
        token_ids = token_ids.tolist()
    if token_ids and isinstance(token_ids[0], list):
        token_ids = token_ids[0]
    return [int(token) for token in token_ids]


def length_bucket(tokens):
    if tokens < 8192:
        return "<8K"
    if tokens < 16384:
        return "8K-16K"
    if tokens < 32768:
        return "16K-32K"
    if tokens < 65536:
        return "32K-64K"
    return "64K+"


def interleave_buckets(items_by_key):
    keys = sorted(items_by_key)
    positions = {key: 0 for key in keys}
    output = []
    while True:
        added = False
        for key in keys:
            position = positions[key]
            bucket = items_by_key[key]
            if position < len(bucket):
                output.append(bucket[position])
                positions[key] += 1
                added = True
        if not added:
            return output


def select_balanced(candidates_, count, seed):
    pair_buckets = defaultdict(list)
    for candidate in candidates_:
        pair_buckets[(candidate["context_category"], candidate["sub_category"])].append(candidate)
    for bucket in pair_buckets.values():
        bucket.sort(
            key=lambda row: hashlib.sha256(
                f"{seed}:{row['task_id']}".encode("utf-8")
            ).hexdigest()
        )
    category_buckets = defaultdict(dict)
    for (category, sub_category), bucket in pair_buckets.items():
        category_buckets[category][sub_category] = bucket
    category_sequences = {
        category: interleave_buckets(subcategories)
        for category, subcategories in category_buckets.items()
    }
    ordered = interleave_buckets(category_sequences)
    if len(ordered) < count:
        raise ValueError(f"only {len(ordered)} eligible CL-bench rows for requested {count}")
    return ordered[:count]


def prepare_inputs(args, tokenizer):
    actual_sha = sha256_file(args.data)
    if actual_sha != DATASET_SHA256:
        raise ValueError(f"CL-bench SHA256 mismatch: {actual_sha}")
    raw_lines = args.data.read_bytes().splitlines()
    candidates_ = []
    exclusions = Counter()
    task_ids = set()
    for row_index, raw_line in enumerate(raw_lines):
        row = json.loads(raw_line)
        metadata = row["metadata"]
        task_id = str(metadata["task_id"])
        if task_id in task_ids:
            raise ValueError(f"duplicate CL-bench task_id: {task_id}")
        task_ids.add(task_id)
        input_ids = encode_messages(tokenizer, row["messages"], args.thinking)
        prompt_tokens = len(input_ids)
        if prompt_tokens < args.min_prompt_tokens:
            exclusions["below_min_prompt_tokens"] += 1
            continue
        if prompt_tokens + args.max_new_tokens > args.context_limit:
            exclusions["prompt_plus_generation_over_context_limit"] += 1
            continue
        candidates_.append({
            "row_index": row_index,
            "task_id": task_id,
            "context_id": str(metadata["context_id"]),
            "context_category": metadata["context_category"],
            "sub_category": metadata["sub_category"],
            "prompt_tokens": prompt_tokens,
            "source_row_sha256": hashlib.sha256(raw_line).hexdigest(),
        })

    chosen = select_balanced(candidates_, args.samples, args.seed)
    inputs = []
    for selected_index, candidate in enumerate(chosen):
        raw_line = raw_lines[candidate["row_index"]]
        row = json.loads(raw_line)
        input_ids = encode_messages(tokenizer, row["messages"], args.thinking)
        if len(input_ids) != candidate["prompt_tokens"]:
            raise ValueError(f"non-deterministic tokenization for {candidate['task_id']}")
        inputs.append({
            "sample_index": selected_index,
            "sample_id": candidate["task_id"],
            "task_id": candidate["task_id"],
            "context_id": candidate["context_id"],
            "context_category": candidate["context_category"],
            "sub_category": candidate["sub_category"],
            "source_row_index": candidate["row_index"],
            "source_row_sha256": candidate["source_row_sha256"],
            "messages_sha256": canonical_sha256(row["messages"]),
            "rubrics_sha256": canonical_sha256(row["rubrics"]),
            "input_sha256": token_sha256(input_ids),
            "input_ids": input_ids,
            "prompt_tokens": len(input_ids),
            "length_bucket": length_bucket(len(input_ids)),
            "max_new_tokens": args.max_new_tokens,
            "thinking": args.thinking,
            "messages": row["messages"],
            "rubrics": row["rubrics"],
            "metadata": row["metadata"],
        })

    eligible_by_category = Counter(row["context_category"] for row in candidates_)
    selected_by_category = Counter(row["context_category"] for row in chosen)
    selected_by_subcategory = Counter(row["sub_category"] for row in chosen)
    manifest = {
        "algorithm": (
            "SHA256(seed:task_id) order within sub-category; round-robin sub-categories "
            "within category; round-robin categories"
        ),
        "seed": args.seed,
        "source_rows": len(raw_lines),
        "eligible_rows": len(candidates_),
        "selected_rows": len(inputs),
        "exclusions": dict(sorted(exclusions.items())),
        "eligible_by_category": dict(sorted(eligible_by_category.items())),
        "selected_by_category": dict(sorted(selected_by_category.items())),
        "selected_by_sub_category": dict(sorted(selected_by_subcategory.items())),
        "task_ids": [sample["task_id"] for sample in inputs],
        "input_sha256": [sample["input_sha256"] for sample in inputs],
    }
    return inputs, manifest


def validate_inputs(args, inputs):
    if len(inputs) != args.samples:
        raise ValueError(f"input snapshot has {len(inputs)} rows, expected {args.samples}")
    seen = set()
    for index, sample in enumerate(inputs):
        if sample["sample_index"] != index:
            raise ValueError("input snapshot order/index mismatch")
        if sample["sample_id"] in seen:
            raise ValueError(f"duplicate input sample: {sample['sample_id']}")
        seen.add(sample["sample_id"])
        if token_sha256(sample["input_ids"]) != sample["input_sha256"]:
            raise ValueError(f"input token hash mismatch: {sample['sample_id']}")
        if canonical_sha256(sample["messages"]) != sample["messages_sha256"]:
            raise ValueError(f"message hash mismatch: {sample['sample_id']}")
        if canonical_sha256(sample["rubrics"]) != sample["rubrics_sha256"]:
            raise ValueError(f"rubric hash mismatch: {sample['sample_id']}")
        if len(sample["input_ids"]) != sample["prompt_tokens"]:
            raise ValueError(f"prompt length mismatch: {sample['sample_id']}")
        if sample["prompt_tokens"] + args.max_new_tokens > args.context_limit:
            raise ValueError(f"input exceeds requested context limit: {sample['sample_id']}")
        if sample["max_new_tokens"] != args.max_new_tokens:
            raise ValueError(f"generation cap differs from input snapshot: {sample['sample_id']}")
        if sample["thinking"] != args.thinking:
            raise ValueError(f"thinking mode differs from input snapshot: {sample['sample_id']}")


def save_inputs(folder, inputs, selection):
    torch.save(inputs, folder / "inputs.pt")
    public_rows = []
    for sample in inputs:
        public = {key: value for key, value in sample.items() if key != "input_ids"}
        public["exact_token_ids_artifact"] = "inputs.pt"
        public_rows.append(public)
    write_jsonl(folder / "inputs.jsonl", public_rows)
    write_json(folder / "input_manifest.json", selection)


def load_or_prepare_inputs(args, tokenizer):
    if sha256_file(args.data) != DATASET_SHA256:
        raise ValueError("the CL-bench source file does not match the pinned SHA256")
    if args.inputs:
        inputs = torch.load(args.inputs, map_location="cpu", weights_only=False)
        manifest_path = args.inputs.parent / "input_manifest.json"
        selection = json.loads(manifest_path.read_text(encoding="utf-8"))
        selection = {**selection, "reused_inputs": str(args.inputs.resolve())}
    else:
        inputs, selection = prepare_inputs(args, tokenizer)
    validate_inputs(args, inputs)
    if not args.resume:
        save_inputs(args.out, inputs, selection)
    return inputs, selection


def gpu_metadata():
    properties = torch.cuda.get_device_properties(0)
    return {
        "name": torch.cuda.get_device_name(0),
        "total_memory_gib": properties.total_memory / 2**30,
        "cuda": torch.version.cuda,
        "capability": list(torch.cuda.get_device_capability(0)),
        "nvidia_smi": subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total",
                "--format=csv,noheader",
            ],
            text=True,
        ).strip(),
    }


def source_hashes():
    root = Path(__file__).parent
    names = [
        "clbench_qwen35.py",
        "attention.py",
        "kernels.py",
        "upstream/flashprefill_native_forward.py",
    ]
    return {name: sha256_file(root / name) for name in names}


def make_metadata(args, configs, inputs, selection):
    return {
        "status": "prepared" if args.prepare_only else "running",
        "quality_status": "pending_official_judge",
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "candidates": [asdict(config) for config in configs],
        "input_count": len(inputs),
        "input_signatures": [
            [sample["sample_id"], sample["input_sha256"]] for sample in inputs
        ],
        "selection": selection,
        "model_source": {"id": MODEL_ID, "revision": MODEL_REVISION},
        "data_source": {
            "id": DATASET_ID,
            "revision": DATASET_REVISION,
            "file": str(args.data),
            "sha256": DATASET_SHA256,
        },
        "official_evaluator": {
            "repository": EVAL_REPOSITORY,
            "commit": EVAL_COMMIT,
            "judge_model": "gpt-5.1",
            "reasoning_effort": "low",
            "primary_metric": "strict binary task success: every rubric must pass",
        },
        "generation": {
            "decoding": "greedy",
            "thinking": args.thinking,
            "judge_text": "final answer only; raw reasoning and token IDs remain in generations.jsonl",
            "max_new_tokens": args.max_new_tokens,
        },
        "attention": {
            "method_scope": "V1 replaces only the eight full-attention prefills",
            "linear_layers": "24 Qwen3.5 Gated Delta layers remain unchanged",
            "block_size": 128,
            "sink_blocks": 2,
            "window_blocks": 4,
            "last_query_blocks_full": 2,
            "decode": "dense SDPA for full-attention layers",
            "batch_size": 1,
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "triton": triton.__version__,
            "package_entrypoint": "python -m exp3.clbench_qwen35",
        },
        "source_sha256": source_hashes(),
        "upstream_flashprefill": {
            "repository": "qhfan/FlashPrefill",
            "commit": "baa612047433a992a00d07dc178205eed065ae14",
            "score_implementation": upstream.SCORE_IMPL,
            "lse_implementation": upstream.LSE_IMPL,
        },
    }


def validate_architecture(model):
    config = model.config.text_config
    layer_types = list(config.layer_types)
    full_layers = [index for index, kind in enumerate(layer_types) if kind == "full_attention"]
    linear_layers = [index for index, kind in enumerate(layer_types) if kind == "linear_attention"]
    observed = {
        "num_hidden_layers": config.num_hidden_layers,
        "num_attention_heads": config.num_attention_heads,
        "num_key_value_heads": config.num_key_value_heads,
        "head_dim": config.head_dim,
        "full_attention_layers": full_layers,
        "linear_attention_layers": linear_layers,
    }
    expected = {
        "num_hidden_layers": 32,
        "num_attention_heads": 16,
        "num_key_value_heads": 4,
        "head_dim": 256,
        "full_attention_layers": EXPECTED_FULL_LAYERS,
        "linear_attention_count": 24,
    }
    if observed["num_hidden_layers"] != expected["num_hidden_layers"]:
        raise ValueError(f"unexpected Qwen3.5 layer count: {observed}")
    if observed["num_attention_heads"] != expected["num_attention_heads"]:
        raise ValueError(f"unexpected Qwen3.5 query heads: {observed}")
    if observed["num_key_value_heads"] != expected["num_key_value_heads"]:
        raise ValueError(f"unexpected Qwen3.5 KV heads: {observed}")
    if observed["head_dim"] != expected["head_dim"]:
        raise ValueError(f"unexpected Qwen3.5 head dimension: {observed}")
    if observed["full_attention_layers"] != expected["full_attention_layers"]:
        raise ValueError(f"unexpected Qwen3.5 full-attention layers: {observed}")
    if len(linear_layers) != expected["linear_attention_count"]:
        raise ValueError(f"unexpected Qwen3.5 linear-attention count: {observed}")
    actual_layers = model.model.language_model.layers
    actual_types = [layer.block_type for layer in actual_layers]
    if actual_types != layer_types:
        raise ValueError("loaded decoder layer types differ from text_config.layer_types")
    return {"observed": observed, "expected": expected, "validated": True}


@torch.inference_mode()
def prefill(model, input_ids, measure):
    torch.compiler.cudagraph_mark_step_begin()
    if measure:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
    output = model(input_ids=input_ids, use_cache=True, logits_to_keep=1)
    first_token = output.logits[:, -1].argmax(dim=-1, keepdim=True)
    if measure:
        torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - started) * 1000
        metrics = {
            "prefill_ms": elapsed_ms,
            "prefill_tokens_s": input_ids.shape[1] / (elapsed_ms / 1000),
            "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
            "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
            "first_token_id": first_token.item(),
        }
    else:
        metrics = None
    del output, first_token
    return metrics


def eos_ids(model, tokenizer):
    value = model.generation_config.eos_token_id
    if value is None:
        value = tokenizer.eos_token_id
    if isinstance(value, int):
        return {value}
    return set(value or [])


def split_qwen_response(raw_text, thinking):
    if not thinking:
        return "", raw_text.strip(), "disabled"
    if "</think>" not in raw_text:
        return raw_text.strip(), "", "unclosed"
    reasoning, final = raw_text.split("</think>", 1)
    if reasoning.lstrip().startswith("<think>"):
        reasoning = reasoning.lstrip()[len("<think>"):]
    return reasoning.strip(), final.strip(), "closed"


@torch.inference_mode()
def generate(model, tokenizer, input_ids, max_new_tokens, thinking):
    torch.compiler.cudagraph_mark_step_begin()
    output = model(input_ids=input_ids, use_cache=True, logits_to_keep=1)
    token = output.logits[:, -1].argmax(dim=-1, keepdim=True)
    cache = output.past_key_values
    del output
    generated = [token.item()]
    terminal_ids = eos_ids(model, tokenizer)
    while len(generated) < max_new_tokens and generated[-1] not in terminal_ids:
        cache_position = torch.tensor(
            [input_ids.shape[1] + len(generated) - 1],
            dtype=torch.long,
            device=input_ids.device,
        )
        output = model(
            input_ids=token,
            past_key_values=cache,
            use_cache=True,
            cache_position=cache_position,
            logits_to_keep=1,
        )
        token = output.logits[:, -1].argmax(dim=-1, keepdim=True)
        cache = output.past_key_values
        generated.append(token.item())
        del output
    end_reason = "eos" if generated[-1] in terminal_ids else "max_new_tokens"
    decoded_ids = list(generated)
    while decoded_ids and decoded_ids[-1] in terminal_ids:
        decoded_ids.pop()
    raw_text = tokenizer.decode(
        decoded_ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    reasoning_text, final_text, thinking_status = split_qwen_response(raw_text, thinking)
    del cache, token
    return {
        "generated_token_ids": generated,
        "raw_text": raw_text,
        "reasoning_text": reasoning_text,
        "final_text": final_text,
        "thinking_status": thinking_status,
        "end_reason": end_reason,
    }


@torch.inference_mode()
def profile_prefill(model, backend, input_ids):
    torch.compiler.cudagraph_mark_step_begin()
    backend.start_profile()
    output = model(input_ids=input_ids, use_cache=True, logits_to_keep=1)
    output.logits[:, -1].argmax(dim=-1)
    rows = backend.finish_profile()
    del output
    layers = [row["layer"] for row in rows]
    if layers != EXPECTED_FULL_LAYERS:
        raise ValueError(f"profile did not contain exactly the eight full-attention layers: {layers}")
    return rows


def rotate(items, offset):
    offset %= len(items)
    return items[offset:] + items[:offset]


def sample_key(sample):
    return {
        "sample_index": sample["sample_index"],
        "sample_id": sample["sample_id"],
        "task_id": sample["task_id"],
        "source_row_index": sample["source_row_index"],
        "context_category": sample["context_category"],
        "sub_category": sample["sub_category"],
        "prompt_tokens": sample["prompt_tokens"],
        "length_bucket": sample["length_bucket"],
        "input_sha256": sample["input_sha256"],
    }


def common_fields():
    return [
        "method", "config_id", "alpha", "sample_index", "sample_id", "task_id",
        "source_row_index", "context_category", "sub_category", "prompt_tokens",
        "length_bucket", "input_sha256",
    ]


def timing_fields():
    return common_fields() + [
        "repeat", "prefill_ms", "prefill_tokens_s", "peak_allocated_gib",
        "peak_reserved_gib", "first_token_id",
    ]


def profile_fields():
    return common_fields() + [
        "layer", "descriptor_ms", "selector_ms", "indices_ms", "exact_ms",
        "mean_ms", "merge_ms", "attention_ms", "effective_exact_token_pair_ratio",
        "exact_token_pairs", "causal_token_pairs", "legacy_block_density",
        "selected_block_entries", "causal_block_entries", "exact_physical_qk_tiles",
        "mean_proxy_entries", "selector_proxy_entries", "mean_executed_logit_entries",
        "mean_executed_value_entries", "selector_executed_dot_entries",
        "selector_physical_qk_tiles", "q_tile_size", "k_tile_size", "score_k_tile_size",
    ]


def expected_generation_keys(inputs, configs):
    return [
        (sample["sample_id"], config.config_id)
        for sample_number, sample in enumerate(inputs)
        for config in rotate(configs, sample_number)
    ]


def expected_timing_keys(inputs, configs, repeats):
    return [
        (sample["sample_id"], config.config_id, str(repeat))
        for sample_number, sample in enumerate(inputs)
        for repeat in range(repeats)
        for config in rotate(configs, repeat + sample_number)
    ]


def archive_resume(args):
    attempts = args.out / "resume_attempts"
    attempts.mkdir(exist_ok=True)
    number = 1
    while (attempts / f"{number:03d}").exists():
        number += 1
    destination = attempts / f"{number:03d}"
    destination.mkdir()
    for name in ("metadata.json", "generations.jsonl", "timings.csv", "profile.csv"):
        source = args.out / name
        if source.is_file():
            shutil.copy2(source, destination / f"{Path(name).stem}_before_resume{Path(name).suffix}")
    return destination


def prepare_resume(args, configs, inputs, attempt_dir):
    generation_path = args.out / "generations.jsonl"
    records = read_jsonl(generation_path) if generation_path.is_file() else []
    expected_generations = expected_generation_keys(inputs, configs)
    actual_generations = [(row["sample_id"], row["config_id"]) for row in records]
    if actual_generations != expected_generations[:len(actual_generations)]:
        raise ValueError("saved generations are not an exact prefix of the fixed schedule")

    timing_path = args.out / "timings.csv"
    with timing_path.open(newline="", encoding="utf-8") as source:
        timing_rows = list(csv.DictReader(source))
    expected_timings = expected_timing_keys(inputs, configs, args.repeats)
    actual_timings = [
        (row["sample_id"], row["config_id"], row["repeat"]) for row in timing_rows
    ]
    if actual_timings != expected_timings[:len(actual_timings)]:
        raise ValueError("saved timing rows are not an exact prefix of the fixed schedule")
    rows_per_sample = len(configs) * args.repeats
    timed_sample_prefix = len(timing_rows) // rows_per_sample
    kept_timing_rows = timing_rows[:timed_sample_prefix * rows_per_sample]
    discarded_timing_rows = timing_rows[timed_sample_prefix * rows_per_sample:]
    write_csv(timing_path, timing_fields(), kept_timing_rows)
    if discarded_timing_rows:
        write_csv(
            attempt_dir / "discarded_partial_timings.csv",
            timing_fields(),
            discarded_timing_rows,
        )

    profile_path = args.out / "profile.csv"
    profile_rows = []
    if profile_path.is_file():
        with profile_path.open(newline="", encoding="utf-8") as source:
            profile_rows = list(csv.DictReader(source))
    grouped = defaultdict(list)
    for row in profile_rows:
        grouped[row["config_id"]].append(row)
    kept_profile_rows = []
    profiled = set()
    discarded_profile_rows = []
    for config in configs:
        rows = grouped.get(config.config_id, [])
        layers = [int(row["layer"]) for row in rows]
        if layers == EXPECTED_FULL_LAYERS:
            kept_profile_rows.extend(rows)
            profiled.add(config.config_id)
        else:
            discarded_profile_rows.extend(rows)
    write_csv(profile_path, profile_fields(), kept_profile_rows)
    if discarded_profile_rows:
        write_csv(
            attempt_dir / "discarded_partial_profile.csv",
            profile_fields(),
            discarded_profile_rows,
        )

    manifest = {
        "generation_records": len(records),
        "generation_total": len(expected_generations),
        "timed_sample_prefix": timed_sample_prefix,
        "kept_timing_rows": len(kept_timing_rows),
        "discarded_partial_timing_rows": len(discarded_timing_rows),
        "profiled_configs": sorted(profiled),
        "discarded_partial_profile_rows": len(discarded_profile_rows),
    }
    write_json(attempt_dir / "resume_manifest.json", manifest)
    return records, timed_sample_prefix, profiled, manifest


def export_official_inputs(folder, inputs, configs, records):
    by_key = {(row["sample_id"], row["config_id"]): row for row in records}
    official_dir = folder / "official_inputs"
    official_dir.mkdir(exist_ok=True)
    files = {}
    for config in configs:
        rows = []
        for sample in inputs:
            generation = by_key[(sample["sample_id"], config.config_id)]
            rows.append({
                "idx": sample["task_id"],
                "messages": sample["messages"],
                "model_output": generation["final_text"],
                "rubrics": sample["rubrics"],
                "metadata": sample["metadata"],
            })
        path = official_dir / f"{config.config_id}.jsonl"
        write_jsonl(path, rows)
        files[config.config_id] = {
            "path": str(path),
            "sha256": sha256_file(path),
            "rows": len(rows),
        }
    manifest = {
        "status": "quality_pending",
        "judge": {
            "repository": EVAL_REPOSITORY,
            "commit": EVAL_COMMIT,
            "model": "gpt-5.1",
            "reasoning_effort": "low",
        },
        "generation_sha256": sha256_file(folder / "generations.jsonl"),
        "files": files,
    }
    write_json(folder / "judge_manifest.json", manifest)
    return manifest


def run(args, configs, inputs, metadata, attempt_dir=None):
    if args.resume:
        generation_records, timed_prefix, profiled, resume_manifest = prepare_resume(
            args, configs, inputs, attempt_dir
        )
        metadata["active_resume"] = resume_manifest
    else:
        generation_records = []
        timed_prefix = 0
        profiled = set()
    generated_prefix = len(generation_records)
    metadata["completed_generation_prefix"] = generated_prefix
    write_json(args.out / "metadata.json", metadata)

    backend = AttentionBackend(
        alpha=args.alpha,
        selector_chunk_tiles=args.selector_chunk_tiles,
    )
    model = AutoModelForImageTextToText.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map="cuda",
    ).eval()
    model.requires_grad_(False)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    metadata["architecture"] = validate_architecture(model)
    metadata["model_commit"] = getattr(model.config, "_commit_hash", None)
    metadata["model_config"] = model.config.to_dict()
    metadata["gpu"] = gpu_metadata()
    write_json(args.out / "metadata.json", metadata)

    mode = "a" if args.resume else "w"
    timings = CsvSink(args.out / "timings.csv", timing_fields(), mode)
    profiles = CsvSink(args.out / "profile.csv", profile_fields(), mode)
    generations = (args.out / "generations.jsonl").open(mode, encoding="utf-8")
    expected_generations = expected_generation_keys(inputs, configs)
    generation_position = generated_prefix

    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        for sample_number, sample in enumerate(inputs):
            input_ids = torch.tensor(
                [sample["input_ids"]], dtype=torch.long, device="cuda"
            )
            key = sample_key(sample)
            if sample_number >= timed_prefix:
                for config in configs:
                    backend.configure(config.method, config.alpha)
                    print(
                        f"Warmup {config.config_id} {sample['sample_id']} "
                        f"n={sample['prompt_tokens']}",
                        flush=True,
                    )
                    prefill(model, input_ids, measure=False)
                for repeat in range(args.repeats):
                    for config in rotate(configs, repeat + sample_number):
                        backend.configure(config.method, config.alpha)
                        metrics = prefill(model, input_ids, measure=True)
                        timings.write({
                            "method": config.method,
                            "config_id": config.config_id,
                            "alpha": config.alpha,
                            **key,
                            "repeat": repeat,
                            **metrics,
                        })
                        print(
                            f"Timing {config.config_id:16s} {sample['sample_id']} "
                            f"rep={repeat} prefill={metrics['prefill_ms']:.1f} ms",
                            flush=True,
                        )

            for config in rotate(configs, sample_number):
                schedule_key = (sample["sample_id"], config.config_id)
                if generation_position < len(expected_generations) and (
                    expected_generations[generation_position] == schedule_key
                ):
                    backend.configure(config.method, config.alpha)
                    generated = generate(
                        model,
                        tokenizer,
                        input_ids,
                        sample["max_new_tokens"],
                        args.thinking,
                    )
                    record = {
                        "method": config.method,
                        "config_id": config.config_id,
                        "alpha": config.alpha,
                        **key,
                        "source_row_sha256": sample["source_row_sha256"],
                        "messages_sha256": sample["messages_sha256"],
                        "rubrics_sha256": sample["rubrics_sha256"],
                        **generated,
                    }
                    generations.write(json.dumps(record, ensure_ascii=False) + "\n")
                    generations.flush()
                    generation_records.append(record)
                    generation_position += 1
                    print(
                        f"Generation {config.config_id:16s} {sample['sample_id']} "
                        f"tokens={len(generated['generated_token_ids'])} "
                        f"think={generated['thinking_status']} end={generated['end_reason']}",
                        flush=True,
                    )
                elif schedule_key in expected_generations[:generation_position]:
                    pass
                else:
                    raise ValueError(f"generation resume cursor mismatch at {schedule_key}")

                if args.profile and config.config_id not in profiled:
                    backend.configure(config.method, config.alpha)
                    for layer_row in profile_prefill(model, backend, input_ids):
                        profiles.write({
                            "method": config.method,
                            "config_id": config.config_id,
                            "alpha": config.alpha,
                            **key,
                            **layer_row,
                        })
                    profiled.add(config.config_id)
            del input_ids

    timings.close()
    profiles.close()
    generations.close()
    if generation_position != len(expected_generations):
        raise ValueError("generation schedule ended before all Full/V1 records were saved")
    judge_manifest = export_official_inputs(args.out, inputs, configs, generation_records)
    metadata["status"] = "generated"
    metadata["quality_status"] = "pending_official_judge"
    metadata["completed_generation_prefix"] = len(generation_records)
    metadata["judge_manifest"] = judge_manifest
    metadata["generated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    metadata.pop("failure", None)
    write_json(args.out / "metadata.json", metadata)


def validate_arguments(args):
    if transformers.__version__ != "5.17.0":
        raise RuntimeError(
            f"Qwen3.5 runner requires transformers==5.17.0, found {transformers.__version__}"
        )
    for name in ("samples", "context_limit", "max_new_tokens", "repeats"):
        if getattr(args, name) < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.min_prompt_tokens < 0:
        raise ValueError("--min-prompt-tokens must be non-negative")
    if args.alpha < 0:
        raise ValueError("--alpha must be non-negative")
    if args.max_new_tokens >= args.context_limit:
        raise ValueError("--max-new-tokens must be smaller than --context-limit")
    if args.prepare_only and args.resume:
        raise ValueError("--prepare-only and --resume cannot be combined")


def validate_resume_arguments(previous, current):
    if previous["candidates"] != current["candidates"]:
        raise ValueError("resume candidates differ from the interrupted run")
    if previous["input_signatures"] != current["input_signatures"]:
        raise ValueError("resume inputs differ from the interrupted run")
    for key in (
        "model", "samples", "seed", "context_limit", "min_prompt_tokens",
        "max_new_tokens", "repeats", "alpha", "thinking", "profile",
        "selector_chunk_tiles",
    ):
        if previous["arguments"].get(key) != current["arguments"].get(key):
            raise ValueError(f"resume argument changed: {key}")


def main():
    args = arguments()
    validate_arguments(args)
    configs = candidates(args.alpha)
    if args.resume:
        if not (args.out / "metadata.json").is_file():
            raise FileNotFoundError(f"resume metadata missing in {args.out}")
        previous_metadata = json.loads(
            (args.out / "metadata.json").read_text(encoding="utf-8")
        )
        attempt_dir = archive_resume(args)
    else:
        if (args.out / "metadata.json").exists():
            raise FileExistsError(f"refusing to overwrite existing run: {args.out}")
        args.out.mkdir(parents=True, exist_ok=True)
        previous_metadata = None
        attempt_dir = None

    torch.manual_seed(args.seed)
    torch.set_num_threads(1)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    inputs, selection = load_or_prepare_inputs(args, tokenizer)
    metadata = make_metadata(args, configs, inputs, selection)
    if previous_metadata is not None:
        validate_resume_arguments(previous_metadata, metadata)
        history = list(previous_metadata.get("resume_history", []))
        history.append({
            "attempt": len(history) + 1,
            "resumed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "previous_status": previous_metadata.get("status"),
            "previous_failure": previous_metadata.get("failure"),
            "artifact_directory": str(attempt_dir),
        })
        metadata["status"] = "resuming"
        metadata["resume_history"] = history
    write_json(args.out / "metadata.json", metadata)
    if args.prepare_only:
        print(f"Prepared {len(inputs)} fixed CL-bench inputs in {args.out}", flush=True)
        return
    try:
        run(args, configs, inputs, metadata, attempt_dir)
    except Exception as error:
        metadata["status"] = "failed"
        metadata["failure"] = {
            "type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc(),
        }
        write_json(args.out / "metadata.json", metadata)
        raise
    print(
        f"Generation complete; run the official judge on {args.out / 'official_inputs'}",
        flush=True,
    )


if __name__ == "__main__":
    main()
