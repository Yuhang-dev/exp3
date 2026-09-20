"""End-to-end task quality and full-model prefill evaluation for exp3."""

import argparse
import csv
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import platform
import shutil
import subprocess
import time
import traceback

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
import triton

from attention import AttentionBackend, METHODS
from data import RULER_TASKS, input_hash, prepare_inputs
from scoring import SCORER_VERSION, score_prediction
from upstream import flashprefill_native_forward as upstream


@dataclass(frozen=True)
class Candidate:
    method: str
    alpha: float | None
    config_id: str


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument(
        "--candidate",
        action="append",
        default=[],
        help="Explicit method or method:alpha; repeat for a small mixed-alpha run.",
    )
    parser.add_argument("--config", type=Path, help="JSON file containing a candidates list.")
    parser.add_argument(
        "--split",
        choices=("quick", "calibration", "holdout", "ruler", "modern"),
        default="quick",
    )
    parser.add_argument(
        "--tasks",
        "--task",
        nargs="+",
        choices=("synthetic_kv_retrieval", "hotpotqa", "longbench_v2", *RULER_TASKS),
        default=["synthetic_kv_retrieval"],
    )
    parser.add_argument(
        "--lengths",
        "--length",
        nargs="+",
        type=int,
        default=[4096],
        help="Synthetic total context budgets, including generation reserve.",
    )
    parser.add_argument(
        "--prompt-budgets",
        nargs="+",
        type=int,
        help="Optional explicit synthetic prompt budgets; overrides --lengths.",
    )
    parser.add_argument("--samples", type=int, default=2)
    parser.add_argument("--synthetic-samples", type=int)
    parser.add_argument("--hotpot-samples", type=int)
    parser.add_argument("--ruler-samples", type=int, help="Samples per selected RULER task and length.")
    parser.add_argument("--longbench-v2-samples", type=int)
    parser.add_argument(
        "--longbench-v2-file",
        type=Path,
        default=Path("datasets/longbench_v2/data.json"),
    )
    parser.add_argument(
        "--longbench-v2-min-tokens",
        type=int,
        default=16384,
        help="Minimum model-tokenized prompt length for the native-32K subset.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--alpha", type=float, default=0.08)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--inputs", type=Path, help="Reuse a previously saved inputs.pt.")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume an interrupted run whose saved generations are an exact sample prefix.",
    )
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--selector-chunk-tiles", type=int, default=8)
    parser.add_argument("--out", type=Path, default=Path("results/quick"))
    return parser.parse_args()


def _alpha_label(alpha):
    return f"{alpha:g}".replace("-", "m").replace(".", "p")


def _candidate(method, alpha):
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}")
    if method == "dense":
        return Candidate(method, None, "dense")
    alpha = float(alpha)
    if alpha < 0:
        raise ValueError("candidate alpha must be non-negative")
    return Candidate(method, alpha, f"{method}__a{_alpha_label(alpha)}")


def parse_candidates(args):
    specs = []
    if args.config:
        payload = json.loads(args.config.read_text(encoding="utf-8"))
        specs = payload["candidates"]
    elif args.candidate:
        specs = args.candidate
    else:
        specs = [{"method": method, "alpha": args.alpha} for method in args.methods]

    candidates = []
    for spec in specs:
        if isinstance(spec, str):
            if ":" in spec:
                method, alpha = spec.rsplit(":", 1)
            else:
                method, alpha = spec, args.alpha
        else:
            method = spec["method"]
            alpha = spec.get("alpha", args.alpha)
        candidates.append(_candidate(method, alpha))
    config_ids = [candidate.config_id for candidate in candidates]
    if len(config_ids) != len(set(config_ids)):
        raise ValueError("candidate configuration IDs must be unique")
    return candidates


class CsvSink:
    def __init__(self, path, fields, mode="w"):
        has_content = path.is_file() and path.stat().st_size > 0
        self.file = path.open(mode, newline="", encoding="utf-8")
        self.writer = csv.DictWriter(self.file, fieldnames=fields)
        if mode == "w" or not has_content:
            self.writer.writeheader()

    def write(self, row):
        self.writer.writerow(row)
        self.file.flush()

    def close(self):
        self.file.close()


def save_inputs(inputs, folder):
    torch.save(inputs, folder / "inputs.pt")
    with (folder / "inputs.jsonl").open("w", encoding="utf-8") as output:
        for sample in inputs:
            public = {key: value for key, value in sample.items() if key != "input_ids"}
            output.write(json.dumps(public, ensure_ascii=False) + "\n")


def load_or_prepare_inputs(args, tokenizer):
    if args.inputs:
        inputs = torch.load(args.inputs, map_location="cpu", weights_only=False)
        revisions = {"reused_inputs": str(args.inputs.resolve())}
        source_metadata = args.inputs.parent / "metadata.json"
        if source_metadata.exists():
            source = json.loads(source_metadata.read_text(encoding="utf-8"))
            revisions["source_data_revisions"] = source.get("data_revisions")
    else:
        inputs, revisions = prepare_inputs(
            tokenizer,
            args.tasks,
            args.split,
            args.seed,
            args.lengths,
            args.prompt_budgets,
            args.samples,
            args.synthetic_samples,
            args.hotpot_samples,
            args.ruler_samples,
            args.max_new_tokens,
            args.longbench_v2_samples,
            args.longbench_v2_file,
            args.longbench_v2_min_tokens,
        )
    stored_splits = {sample["split"] for sample in inputs}
    if stored_splits != {args.split}:
        raise ValueError(
            f"input split {sorted(stored_splits)} does not match requested split {args.split!r}"
        )
    for sample in inputs:
        if len(sample["input_ids"]) != sample["actual_tokens"]:
            raise ValueError(f"stored length mismatch for {sample['sample_id']}")
        if input_hash(sample["input_ids"]) != sample["input_sha256"]:
            raise ValueError(f"stored input hash mismatch for {sample['sample_id']}")
    if args.resume:
        saved_path = args.out / "inputs.pt"
        if not saved_path.is_file():
            raise FileNotFoundError(f"resume input snapshot is missing: {saved_path}")
        saved_inputs = torch.load(saved_path, map_location="cpu", weights_only=False)
        current_signature = [
            (sample["sample_id"], sample["input_sha256"]) for sample in inputs
        ]
        saved_signature = [
            (sample["sample_id"], sample["input_sha256"]) for sample in saved_inputs
        ]
        if saved_signature != current_signature:
            raise ValueError("resume inputs differ from the saved input snapshot")
    else:
        save_inputs(inputs, args.out)
    return inputs, revisions


def write_metadata(path, metadata):
    path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")


def source_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def base_metadata(args, candidates, inputs, revisions):
    task_counts = {}
    for sample in inputs:
        task_counts[sample["task_label"]] = task_counts.get(sample["task_label"], 0) + 1
    return {
        "status": "prepared" if args.prepare_only else "running",
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "candidates": [asdict(candidate) for candidate in candidates],
        "input_count": len(inputs),
        "task_counts": task_counts,
        "data_revisions": revisions,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "triton": triton.__version__,
        "method_formula_version": "quality-first-v2-2026-09-16",
        "scoring": {
            "version": SCORER_VERSION,
            "implementation": "scoring.py",
            "implementation_sha256": source_hash(Path(__file__).with_name("scoring.py")),
            "raw_generation_artifact": "generations.jsonl; flushed before scoring",
            "execution_order": "finish all GPU generations, then score from saved raw records",
            "offline_rescore_entrypoint": "rescore.py",
        },
        "upstream": {
            "repository": "qhfan/FlashPrefill",
            "commit": "baa612047433a992a00d07dc178205eed065ae14",
            "score_implementation": upstream.SCORE_IMPL,
            "lse_implementation": upstream.LSE_IMPL,
        },
        "attention": {
            "block_size": 128,
            "sink_blocks": 2,
            "window_blocks": 4,
            "last_query_blocks_full": 2,
            "query_tile_routing": 128,
            "dtype": "bfloat16",
            "statistics_and_softmax_state": "float32",
            "rope": "model original",
            "prefill_cache": True,
            "decode": "dense PyTorch Flash SDPA",
            "ruler_quality_generation": (
                "mirror upstream SelfdefinedModel: sparse prefill input[:-1], then process "
                "the final prompt token and generated tokens through dense single-token decode"
            ),
            "batch_size": 1,
        },
        "prompt_and_scorer": {
            "synthetic": "exp3 synthetic_kv_retrieval v1",
            "hotpotqa_prompt": (
                "THUDM/LongBench LongBench/config/dataset2prompt.json at "
                "2e00731f8d0bff23dc4325161044d0ed8af94c1e"
            ),
            "hotpotqa_scorer": (
                "THUDM/LongBench LongBench/metrics.py qa_f1_score; max over answers"
            ),
            "ruler_prompt": (
                "qhfan/FlashPrefill ruler/data.py load_ruler at "
                "baa612047433a992a00d07dc178205eed065ae14; use_chat_template=false"
            ),
            "ruler_scorer": (
                "case-insensitive answer substring recall from the same pinned load_ruler"
            ),
            "longbench_v2_prompt": (
                "THUDM/LongBench-v2 prompts/0shot.txt at "
                "2e00731f8d0bff23dc4325161044d0ed8af94c1e"
            ),
            "longbench_v2_scorer": (
                "THUDM/LongBench-v2 pred.py exact extracted A/B/C/D accuracy"
            ),
        },
    }


def gpu_metadata():
    properties = torch.cuda.get_device_properties(0)
    return {
        "name": torch.cuda.get_device_name(0),
        "total_memory_gib": properties.total_memory / 2**30,
        "cuda": torch.version.cuda,
        "capability": list(torch.cuda.get_device_capability(0)),
        "nvidia_smi": subprocess.check_output([
            "nvidia-smi",
            "--query-gpu=name,driver_version,memory.total",
            "--format=csv,noheader",
        ], text=True).strip(),
    }


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


def _eos_ids(model, tokenizer):
    value = model.generation_config.eos_token_id
    if value is None:
        value = tokenizer.eos_token_id
    if isinstance(value, int):
        return {value}
    return set(value or [])


@torch.inference_mode()
def generate(model, tokenizer, input_ids, max_new_tokens, split_last_prompt_token=False):
    torch.compiler.cudagraph_mark_step_begin()
    if split_last_prompt_token:
        prefill_output = model.model(input_ids=input_ids[:, :-1], use_cache=True)
        cache = prefill_output.past_key_values
        del prefill_output
        cache_position = torch.tensor(
            [input_ids.shape[1] - 1],
            dtype=torch.long,
            device=input_ids.device,
        )
        output = model(
            input_ids=input_ids[:, -1:],
            past_key_values=cache,
            use_cache=True,
            cache_position=cache_position,
            logits_to_keep=1,
        )
    else:
        output = model(input_ids=input_ids, use_cache=True, logits_to_keep=1)
    token = output.logits[:, -1].argmax(dim=-1, keepdim=True)
    cache = output.past_key_values
    del output
    generated = [token.item()]
    eos_ids = _eos_ids(model, tokenizer)
    reason = "eos" if generated[-1] in eos_ids else "max_new_tokens"
    while len(generated) < max_new_tokens and generated[-1] not in eos_ids:
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
    if generated[-1] in eos_ids:
        reason = "eos"
    del cache, token
    text = tokenizer.decode(generated, skip_special_tokens=True)
    return generated, text, reason


@torch.inference_mode()
def profile_prefill(model, backend, input_ids):
    torch.compiler.cudagraph_mark_step_begin()
    backend.start_profile()
    output = model(input_ids=input_ids, use_cache=True, logits_to_keep=1)
    output.logits[:, -1].argmax(dim=-1)
    rows = backend.finish_profile()
    del output
    return rows


def sample_key(sample):
    length_label = sample.get("length_label")
    if length_label is None:
        length_label = (
            str(sample["total_context_budget"])
            if sample["task"] in ("synthetic_kv_retrieval", "ruler")
            else "actual"
        )
    return {
        "task": sample["task_label"],
        "split": sample["split"],
        "length_label": length_label,
        "total_context_budget": sample["total_context_budget"],
        "prompt_budget": sample["prompt_budget"],
        "actual_tokens": sample["actual_tokens"],
        "sample_id": sample["sample_id"],
        "source_id": sample["source_id"],
        "domain": sample.get("domain"),
        "sub_domain": sample.get("sub_domain"),
        "difficulty": sample.get("difficulty"),
        "source_length_category": sample.get("source_length_category"),
    }


def rotate(items, offset):
    offset %= len(items)
    return items[offset:] + items[:offset]


def write_quality(rows, path):
    dense = {
        row["sample_id"]: row["score"]
        for row in rows
        if row["method"] == "dense"
    }
    for row in rows:
        reference = dense.get(row["sample_id"])
        row["delta_vs_dense"] = "" if reference is None else row["score"] - reference
    fields = [
        "method", "config_id", "alpha", "task", "split", "length_label",
        "total_context_budget", "prompt_budget", "actual_tokens", "sample_id",
        "source_id", "domain", "sub_domain", "difficulty", "source_length_category",
        "metric", "scorer_version", "score", "delta_vs_dense",
        "exact_match", "target_accuracy", "all_target_em", "raw_substring_score",
        "normalized_substring_score", "parsed_answer",
    ]
    sink = CsvSink(path, fields)
    for row in rows:
        sink.write({key: row.get(key) for key in fields})
    sink.close()


def read_jsonl(path):
    with path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def write_csv_rows(path, fields, rows):
    sink = CsvSink(path, fields)
    for row in rows:
        sink.write({key: row.get(key) for key in fields})
    sink.close()


def prepare_resume_prefix(
    args,
    candidates,
    inputs,
    timing_fields,
    profile_fields,
    attempt_dir,
):
    generations_path = args.out / "generations.jsonl"
    timings_path = args.out / "timings.csv"
    profile_path = args.out / "profile.csv"
    if not generations_path.is_file() or not timings_path.is_file():
        raise FileNotFoundError("resume requires generations.jsonl and timings.csv")

    for path in (generations_path, timings_path, profile_path):
        if path.is_file():
            shutil.copy2(path, attempt_dir / f"{path.stem}_before_resume{path.suffix}")

    records = read_jsonl(generations_path)
    expected_generation_keys = [
        (sample["sample_id"], candidate.config_id, sample["input_sha256"])
        for sample_number, sample in enumerate(inputs)
        for candidate in rotate(candidates, sample_number)
    ]
    actual_generation_keys = [
        (record["sample_id"], record["config_id"], record["input_sha256"])
        for record in records
    ]
    if actual_generation_keys != expected_generation_keys[:len(actual_generation_keys)]:
        raise ValueError("saved generations are not an exact prefix of the requested schedule")
    if len(records) % len(candidates):
        raise ValueError("resume requires saved generations to end at a sample boundary")
    completed_samples = len(records) // len(candidates)

    with timings_path.open(newline="", encoding="utf-8") as source:
        timing_rows = list(csv.DictReader(source))
    expected_timing_keys = [
        (sample["sample_id"], candidate.config_id, str(repeat))
        for sample_number, sample in enumerate(inputs)
        for repeat in range(args.repeats)
        for candidate in rotate(candidates, repeat + sample_number)
    ]
    actual_timing_keys = [
        (row["sample_id"], row["config_id"], row["repeat"])
        for row in timing_rows
    ]
    if actual_timing_keys != expected_timing_keys[:len(actual_timing_keys)]:
        raise ValueError("saved timings are not an exact prefix of the requested schedule")
    kept_timing_count = completed_samples * args.repeats * len(candidates)
    if len(timing_rows) < kept_timing_count:
        raise ValueError("saved timings do not cover every completed generation sample")
    kept_timing_rows = timing_rows[:kept_timing_count]
    discarded_timing_rows = timing_rows[kept_timing_count:]
    if discarded_timing_rows:
        write_csv_rows(
            attempt_dir / "discarded_partial_timings.csv",
            timing_fields,
            discarded_timing_rows,
        )
    write_csv_rows(timings_path, timing_fields, kept_timing_rows)

    profile_rows = []
    if profile_path.is_file():
        with profile_path.open(newline="", encoding="utf-8") as source:
            profile_rows = list(csv.DictReader(source))
    profiled_groups = {
        (row["config_id"], row["task"], row["length_label"])
        for row in profile_rows
    }
    manifest = {
        "completed_samples": completed_samples,
        "input_samples": len(inputs),
        "completed_generation_records": len(records),
        "kept_timing_rows": len(kept_timing_rows),
        "discarded_partial_timing_rows": len(discarded_timing_rows),
        "next_sample_id": (
            inputs[completed_samples]["sample_id"]
            if completed_samples < len(inputs)
            else None
        ),
        "generations_before_resume_sha256": source_hash(
            attempt_dir / "generations_before_resume.jsonl"
        ),
        "timings_before_resume_sha256": source_hash(
            attempt_dir / "timings_before_resume.csv"
        ),
    }
    write_metadata(attempt_dir / "resume_manifest.json", manifest)
    return records, completed_samples, profiled_groups, manifest


def run(args, candidates, inputs, metadata, resume_attempt_dir=None):
    common_fields = [
        "method", "config_id", "alpha", "task", "split", "length_label",
        "total_context_budget", "prompt_budget", "actual_tokens", "sample_id", "source_id",
        "domain", "sub_domain", "difficulty", "source_length_category",
    ]
    timing_fields = common_fields + [
        "repeat", "prefill_ms", "prefill_tokens_s", "peak_allocated_gib",
        "peak_reserved_gib", "first_token_id",
    ]
    profile_fields = common_fields + [
        "layer", "descriptor_ms", "selector_ms", "indices_ms", "exact_ms",
        "mean_ms", "merge_ms", "attention_ms", "effective_exact_token_pair_ratio",
        "exact_token_pairs", "causal_token_pairs", "legacy_block_density",
        "selected_block_entries", "causal_block_entries", "exact_physical_qk_tiles",
        "mean_proxy_entries", "selector_proxy_entries", "mean_executed_logit_entries",
        "mean_executed_value_entries", "selector_executed_dot_entries",
        "selector_physical_qk_tiles", "q_tile_size", "k_tile_size", "score_k_tile_size",
    ]
    if args.resume:
        generation_records, start_sample, profiled, resume_manifest = prepare_resume_prefix(
            args,
            candidates,
            inputs,
            timing_fields,
            profile_fields,
            resume_attempt_dir,
        )
        metadata["active_resume"] = resume_manifest
    else:
        generation_records = []
        start_sample = 0
        profiled = set()
    metadata["completed_sample_prefix"] = start_sample
    write_metadata(args.out / "metadata.json", metadata)

    backend = AttentionBackend(alpha=args.alpha, selector_chunk_tiles=args.selector_chunk_tiles)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map="cuda",
    ).eval()
    model.requires_grad_(False)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    metadata["model_commit"] = model.config._commit_hash
    metadata["model_config"] = model.config.to_dict()
    metadata["gpu"] = gpu_metadata()
    write_metadata(args.out / "metadata.json", metadata)

    output_mode = "a" if args.resume else "w"
    timing_sink = CsvSink(args.out / "timings.csv", timing_fields, mode=output_mode)
    profile_sink = CsvSink(args.out / "profile.csv", profile_fields, mode=output_mode)
    generations = (args.out / "generations.jsonl").open(output_mode, encoding="utf-8")
    warmed = set()

    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        for sample_number, sample in enumerate(inputs[start_sample:], start=start_sample):
            input_ids = torch.tensor([sample["input_ids"]], dtype=torch.long, device="cuda")
            key = sample_key(sample)
            for candidate in candidates:
                warm_key = (candidate.config_id, sample["actual_tokens"])
                if warm_key not in warmed:
                    backend.configure(candidate.method, candidate.alpha)
                    print(
                        f"Warmup {candidate.config_id} n={sample['actual_tokens']}: "
                        "compile/autotune excluded",
                        flush=True,
                    )
                    prefill(model, input_ids, measure=False)
                    warmed.add(warm_key)

            for repeat in range(args.repeats):
                for candidate in rotate(candidates, repeat + sample_number):
                    backend.configure(candidate.method, candidate.alpha)
                    metrics = prefill(model, input_ids, measure=True)
                    row = {
                        "method": candidate.method,
                        "config_id": candidate.config_id,
                        "alpha": candidate.alpha,
                        **key,
                        "repeat": repeat,
                        **metrics,
                    }
                    timing_sink.write(row)
                    print(
                        f"Timing {candidate.config_id:28s} {sample['sample_id']} "
                        f"rep={repeat} prefill={metrics['prefill_ms']:.1f} ms",
                        flush=True,
                    )

            for candidate in rotate(candidates, sample_number):
                backend.configure(candidate.method, candidate.alpha)
                token_ids, text, reason = generate(
                    model,
                    tokenizer,
                    input_ids,
                    sample["max_new_tokens"],
                    split_last_prompt_token=sample["task"] == "ruler",
                )
                generation_record = {
                    **key,
                    "method": candidate.method,
                    "config_id": candidate.config_id,
                    "alpha": candidate.alpha,
                    "input_sha256": sample["input_sha256"],
                    "generated_token_ids": token_ids,
                    "text": text,
                    "raw_text": text,
                    "end_reason": reason,
                    "generation_path": (
                        "upstream_ruler_split_final_prompt_token"
                        if sample["task"] == "ruler"
                        else "full_prompt_prefill"
                    ),
                }
                generations.write(json.dumps(generation_record, ensure_ascii=False) + "\n")
                generations.flush()
                generation_records.append(generation_record)
                print(
                    f"Generation {candidate.config_id:25s} {sample['sample_id']} "
                    f"tokens={len(token_ids)} end={reason}",
                    flush=True,
                )

                profile_group = (
                    candidate.config_id,
                    sample["task_label"],
                    key["length_label"],
                )
                if args.profile and profile_group not in profiled:
                    rows = profile_prefill(model, backend, input_ids)
                    for layer_row in rows:
                        profile_sink.write({
                            "method": candidate.method,
                            "config_id": candidate.config_id,
                            "alpha": candidate.alpha,
                            **key,
                            **layer_row,
                        })
                    profiled.add(profile_group)
            del input_ids

    timing_sink.close()
    profile_sink.close()
    generations.close()

    inputs_by_id = {sample["sample_id"]: sample for sample in inputs}
    predictions = (args.out / "predictions.jsonl").open("w", encoding="utf-8")
    quality_rows = []
    for generation_record in generation_records:
        sample = inputs_by_id[generation_record["sample_id"]]
        text = generation_record["raw_text"]
        scored = score_prediction(sample, text)
        parsed = json.dumps(scored["parsed_answer"], ensure_ascii=False)
        quality_row = {
            **{field: generation_record[field] for field in common_fields},
            "metric": scored["metric"],
            "scorer_version": scored["scorer_version"],
            "score": scored["score"],
            "exact_match": scored["exact_match"],
            "target_accuracy": scored["target_accuracy"],
            "all_target_em": scored["all_target_em"],
            "raw_substring_score": scored.get("raw_substring_score"),
            "normalized_substring_score": scored.get("normalized_substring_score"),
            "parsed_answer": parsed,
        }
        quality_rows.append(quality_row)
        prediction = {
            **generation_record,
            "scorer_text": scored.get("scorer_text", text),
            "parsed_answer": scored["parsed_answer"],
            "scorer_version": scored["scorer_version"],
            "score": scored,
        }
        predictions.write(json.dumps(prediction, ensure_ascii=False) + "\n")
        predictions.flush()
        print(
            f"Quality {generation_record['config_id']:28s} {sample['sample_id']} "
            f"{scored['metric']}={scored['score']:.2f}",
            flush=True,
        )
    predictions.close()
    write_quality(quality_rows, args.out / "quality.csv")
    metadata["status"] = "complete"
    metadata.pop("failure", None)
    metadata["completed_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    write_metadata(args.out / "metadata.json", metadata)
    from report import make_report
    make_report(args.out)


def main():
    args = arguments()
    if args.repeats < 1 or args.samples < 1 or args.max_new_tokens < 1:
        raise ValueError("repeats, samples, and max-new-tokens must be positive")
    if args.synthetic_samples is not None and args.synthetic_samples < 1:
        raise ValueError("synthetic-samples must be positive")
    if args.hotpot_samples is not None and args.hotpot_samples < 1:
        raise ValueError("hotpot-samples must be positive")
    if args.ruler_samples is not None and args.ruler_samples < 1:
        raise ValueError("ruler-samples must be positive")
    if args.longbench_v2_samples is not None and args.longbench_v2_samples < 1:
        raise ValueError("longbench-v2-samples must be positive")
    if args.longbench_v2_min_tokens < 1:
        raise ValueError("longbench-v2-min-tokens must be positive")
    if args.alpha < 0:
        raise ValueError("alpha must be non-negative")
    if args.prepare_only and args.resume:
        raise ValueError("--prepare-only and --resume cannot be combined")
    previous_metadata = None
    resume_attempt_dir = None
    if args.resume:
        metadata_path = args.out / "metadata.json"
        if not metadata_path.is_file():
            raise FileNotFoundError(f"resume metadata is missing: {metadata_path}")
        previous_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    args.out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    torch.set_num_threads(1)
    candidates = parse_candidates(args)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    inputs, revisions = load_or_prepare_inputs(args, tokenizer)
    metadata = base_metadata(args, candidates, inputs, revisions)
    if previous_metadata is not None:
        if previous_metadata["candidates"] != metadata["candidates"]:
            raise ValueError("resume candidates differ from the interrupted run")
        if previous_metadata["input_count"] != metadata["input_count"]:
            raise ValueError("resume input count differs from the interrupted run")
        for key in (
            "model",
            "split",
            "tasks",
            "max_new_tokens",
            "repeats",
            "profile",
            "selector_chunk_tiles",
        ):
            if previous_metadata["arguments"].get(key) != metadata["arguments"].get(key):
                raise ValueError(f"resume argument changed: {key}")
        attempts_root = args.out / "resume_attempts"
        attempts_root.mkdir(parents=True, exist_ok=True)
        attempt_number = 1
        while (attempts_root / f"{attempt_number:03d}").exists():
            attempt_number += 1
        resume_attempt_dir = attempts_root / f"{attempt_number:03d}"
        resume_attempt_dir.mkdir()
        shutil.copy2(
            args.out / "metadata.json",
            resume_attempt_dir / "metadata_before_resume.json",
        )
        history = list(previous_metadata.get("resume_history", []))
        history.append({
            "attempt": len(history) + 1,
            "resumed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "previous_status": previous_metadata.get("status"),
            "previous_failure": previous_metadata.get("failure"),
            "artifact_directory": str(resume_attempt_dir),
        })
        metadata["status"] = "resuming"
        metadata["resume_history"] = history
    write_metadata(args.out / "metadata.json", metadata)
    write_metadata(args.out / "scorer_manifest.json", {
        **metadata["scoring"],
        "prompt_and_scorer": metadata["prompt_and_scorer"],
    })
    if args.prepare_only:
        print(f"Prepared {len(inputs)} fixed inputs in {args.out}", flush=True)
        return
    try:
        run(
            args,
            candidates,
            inputs,
            metadata,
            resume_attempt_dir=resume_attempt_dir,
        )
    except Exception as error:
        metadata["status"] = "failed"
        metadata["failure"] = {
            "type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc(),
        }
        write_metadata(args.out / "metadata.json", metadata)
        raise
    print(f"Completed: {args.out / 'REPORT.md'}", flush=True)


if __name__ == "__main__":
    main()
