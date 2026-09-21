"""Audit official CL-bench judge outputs and build paired quality/latency reports."""

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import time

import numpy as np
import torch


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--dense-graded", type=Path, required=True)
    parser.add_argument("--v1-graded", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=10000)
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


def read_jsonl(path):
    rows = []
    with Path(path).open(encoding="utf-8") as source:
        for line in source:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def write_csv(path, fields, rows):
    with Path(path).open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def task_id(row):
    return str(row["metadata"]["task_id"])


def load_unique(path, expected_ids):
    rows = read_jsonl(path)
    by_id = {}
    for row in rows:
        identifier = task_id(row)
        if identifier in by_id:
            raise ValueError(f"duplicate judge row {identifier} in {path}")
        by_id[identifier] = row
    if set(by_id) != set(expected_ids):
        missing = sorted(set(expected_ids) - set(by_id))
        extra = sorted(set(by_id) - set(expected_ids))
        raise ValueError(f"judge IDs differ for {path}: missing={missing}, extra={extra}")
    return by_id


def rubric_satisfaction(row, rubric_count):
    statuses = row.get("requirement_status", [])
    passed = sum(str(value).strip().lower() == "yes" for value in statuses)
    return passed, rubric_count, passed / rubric_count if rubric_count else 0.0


def validate_and_score(run, inputs, generations, config_id, graded_path):
    expected_ids = [sample["task_id"] for sample in inputs]
    graded = load_unique(graded_path, expected_ids)
    generated = {
        row["sample_id"]: row
        for row in generations
        if row["config_id"] == config_id
    }
    if set(generated) != set(expected_ids):
        raise ValueError(f"generation IDs are incomplete for {config_id}")
    rows = []
    for sample in inputs:
        identifier = sample["task_id"]
        judge = graded[identifier]
        generation = generated[identifier]
        if canonical_sha256(judge["messages"]) != sample["messages_sha256"]:
            raise ValueError(f"judge messages differ for {config_id}/{identifier}")
        if canonical_sha256(judge["rubrics"]) != sample["rubrics_sha256"]:
            raise ValueError(f"judge rubrics differ for {config_id}/{identifier}")
        if judge["model_output"] != generation["final_text"]:
            raise ValueError(f"judge model_output differs for {config_id}/{identifier}")
        score = int(judge["score"])
        if score not in (0, 1):
            raise ValueError(f"non-binary official score for {config_id}/{identifier}: {score}")
        passed, total, fraction = rubric_satisfaction(judge, len(sample["rubrics"]))
        rows.append({
            "method": generation["method"],
            "config_id": config_id,
            "alpha": generation["alpha"],
            "sample_index": sample["sample_index"],
            "sample_id": identifier,
            "context_category": sample["context_category"],
            "sub_category": sample["sub_category"],
            "prompt_tokens": sample["prompt_tokens"],
            "length_bucket": sample["length_bucket"],
            "task_success": score,
            "score_percent": score * 100.0,
            "rubrics_passed": passed,
            "rubrics_total": total,
            "rubric_satisfaction": fraction,
            "end_reason": generation["end_reason"],
            "thinking_status": generation["thinking_status"],
            "generated_tokens": len(generation["generated_token_ids"]),
            "judge_file": str(graded_path),
        })
    return rows


def paired_bootstrap(dense, sparse, iterations):
    difference = np.asarray(sparse, dtype=np.float64) - np.asarray(dense, dtype=np.float64)
    rng = np.random.default_rng(20260921)
    draws = rng.integers(0, len(difference), size=(iterations, len(difference)))
    samples = difference[draws].mean(axis=1) * 100
    return [float(value) for value in np.quantile(samples, [0.025, 0.975])]


def exact_mcnemar(dense_only, sparse_only):
    discordant = dense_only + sparse_only
    if discordant == 0:
        return 1.0
    smaller = min(dense_only, sparse_only)
    tail = sum(math.comb(discordant, k) for k in range(smaller + 1)) / (2 ** discordant)
    return min(1.0, 2 * tail)


def load_timing_summary(run, configs, inputs):
    with (run / "timings.csv").open(newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
    expected = len(configs) * len(inputs)
    grouped = defaultdict(list)
    peak = defaultdict(list)
    for row in rows:
        grouped[(row["sample_id"], row["config_id"])].append(float(row["prefill_ms"]))
        peak[(row["sample_id"], row["config_id"])].append(float(row["peak_allocated_gib"]))
    if len(grouped) != expected:
        raise ValueError(f"timing groups {len(grouped)} != expected {expected}")
    medians = {key: statistics.median(values) for key, values in grouped.items()}
    peaks = {key: max(values) for key, values in peak.items()}
    result = {}
    sample_ids = [sample["sample_id"] for sample in inputs]
    for config in configs:
        config_id = config["config_id"]
        config_times = [medians[(identifier, config_id)] for identifier in sample_ids]
        speedups = [
            medians[(identifier, "dense")] / medians[(identifier, config_id)]
            for identifier in sample_ids
        ]
        result[config_id] = {
            "median_prefill_ms": statistics.median(config_times),
            "median_paired_speedup": statistics.median(speedups),
            "median_peak_allocated_gib": statistics.median(
                peaks[(identifier, config_id)] for identifier in sample_ids
            ),
        }
    return result


def load_profile_summary(run, configs):
    with (run / "profile.csv").open(newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
    result = {}
    for config in configs:
        config_id = config["config_id"]
        selected = [row for row in rows if row["config_id"] == config_id]
        if not selected:
            result[config_id] = {}
            continue
        layers = [int(row["layer"]) for row in selected]
        if layers != [3, 7, 11, 15, 19, 23, 27, 31]:
            raise ValueError(f"profile layers differ for {config_id}: {layers}")
        result[config_id] = {
            "profile_sample_id": selected[0]["sample_id"],
            "profile_full_attention_layers": len(selected),
            "median_exact_pair_ratio": statistics.median(
                float(row["effective_exact_token_pair_ratio"]) for row in selected
            ),
            "median_attention_ms_per_full_layer": statistics.median(
                float(row["attention_ms"]) for row in selected
            ),
            "median_selector_ms_per_full_layer": statistics.median(
                float(row["selector_ms"]) for row in selected
            ),
            "sum_attention_ms_eight_full_layers": sum(
                float(row["attention_ms"]) for row in selected
            ),
        }
    return result


def aggregate_quality(rows, timing, profile, configs):
    result = []
    dense_by_id = {
        row["sample_id"]: row for row in rows if row["config_id"] == "dense"
    }
    for config in configs:
        config_id = config["config_id"]
        selected = [row for row in rows if row["config_id"] == config_id]
        success_rate = statistics.mean(row["task_success"] for row in selected) * 100
        rubric_rate = (
            sum(row["rubrics_passed"] for row in selected)
            / sum(row["rubrics_total"] for row in selected)
            * 100
        )
        dense_rate = statistics.mean(
            dense_by_id[row["sample_id"]]["task_success"] for row in selected
        ) * 100
        result.append({
            "method": config["method"],
            "config_id": config_id,
            "alpha": config["alpha"],
            "n": len(selected),
            "task_successes": sum(row["task_success"] for row in selected),
            "success_rate_percent": success_rate,
            "delta_vs_dense_pp": success_rate - dense_rate,
            "rubric_satisfaction_percent": rubric_rate,
            "max_token_endings": sum(row["end_reason"] == "max_new_tokens" for row in selected),
            "unclosed_thinking": sum(row["thinking_status"] == "unclosed" for row in selected),
            **timing[config_id],
            **profile.get(config_id, {}),
        })
    return result


def subgroup_rows(quality, configs, dimensions):
    rows = []
    for dimension in dimensions:
        labels = sorted({row[dimension] for row in quality})
        for label in labels:
            dense = {
                row["sample_id"]: row
                for row in quality
                if row["config_id"] == "dense" and row[dimension] == label
            }
            for config in configs:
                config_id = config["config_id"]
                selected = [
                    row for row in quality
                    if row["config_id"] == config_id and row[dimension] == label
                ]
                if not selected:
                    continue
                rate = statistics.mean(row["task_success"] for row in selected) * 100
                dense_rate = statistics.mean(
                    dense[row["sample_id"]]["task_success"] for row in selected
                ) * 100
                rows.append({
                    "dimension": dimension,
                    "group": label,
                    "method": config["method"],
                    "config_id": config_id,
                    "n": len(selected),
                    "successes": sum(row["task_success"] for row in selected),
                    "success_rate_percent": rate,
                    "delta_vs_dense_pp": rate - dense_rate,
                    "rubric_satisfaction_percent": (
                        sum(row["rubrics_passed"] for row in selected)
                        / sum(row["rubrics_total"] for row in selected)
                        * 100
                    ),
                })
    return rows


def paired_outcomes(quality, dimensions):
    by_config = defaultdict(dict)
    for row in quality:
        by_config[row["config_id"]][row["sample_id"]] = row
    dense = by_config["dense"]
    sparse_id = next(config_id for config_id in by_config if config_id != "dense")
    sparse = by_config[sparse_id]
    rows = []
    group_specs = [("all", "all", sorted(dense))]
    for dimension in dimensions:
        labels = sorted({row[dimension] for row in dense.values()})
        for label in labels:
            identifiers = sorted(
                identifier for identifier, row in dense.items() if row[dimension] == label
            )
            group_specs.append((dimension, label, identifiers))
    for dimension, label, identifiers in group_specs:
        counts = Counter()
        for identifier in identifiers:
            pair = (dense[identifier]["task_success"], sparse[identifier]["task_success"])
            names = {(1, 1): "both", (1, 0): "dense_only", (0, 1): "v1_only", (0, 0): "neither"}
            counts[names[pair]] += 1
        rows.append({
            "dimension": dimension,
            "group": label,
            "n": len(identifiers),
            "both_success": counts["both"],
            "dense_only": counts["dense_only"],
            "v1_only": counts["v1_only"],
            "neither_success": counts["neither"],
            "net_v1_minus_dense": counts["v1_only"] - counts["dense_only"],
        })
    return rows


def markdown_table(headers, rows):
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(str(value) for value in row) + " |" for row in rows)
    return "\n".join(lines)


def report_text(metadata, summary, subgroups, paired, interval, p_value, judge_audit):
    global_pair = next(row for row in paired if row["dimension"] == "all")
    quality_table = markdown_table(
        ["Config", "Solved", "Rate", "Δ Full", "Rubric pass*", "Max-token", "Unclosed think"],
        [
            [
                row["config_id"],
                f"{row['task_successes']}/{row['n']}",
                f"{row['success_rate_percent']:.2f}%",
                f"{row['delta_vs_dense_pp']:+.2f} pp",
                f"{row['rubric_satisfaction_percent']:.2f}%",
                row["max_token_endings"],
                row["unclosed_thinking"],
            ]
            for row in summary
        ],
    )
    latency_table = markdown_table(
        ["Config", "Median prefill", "Paired speedup", "Peak GiB", "Exact pairs†"],
        [
            [
                row["config_id"],
                f"{row['median_prefill_ms']:.1f} ms",
                f"{row['median_paired_speedup']:.3f}×",
                f"{row['median_peak_allocated_gib']:.2f}",
                (
                    f"{row['median_exact_pair_ratio'] * 100:.1f}%"
                    if row.get("median_exact_pair_ratio") is not None
                    else "not profiled"
                ),
            ]
            for row in summary
        ],
    )

    def subgroup_table(dimension):
        grouped = defaultdict(dict)
        for row in subgroups:
            if row["dimension"] == dimension:
                grouped[row["group"]][row["config_id"]] = row
        body = []
        for label in sorted(grouped):
            dense = grouped[label]["dense"]
            sparse_id = next(key for key in grouped[label] if key != "dense")
            sparse = grouped[label][sparse_id]
            body.append([
                label,
                dense["n"],
                f"{dense['success_rate_percent']:.1f}%",
                f"{sparse['success_rate_percent']:.1f}%",
                f"{sparse['delta_vs_dense_pp']:+.1f} pp",
            ])
        return markdown_table([dimension, "n", "Full", "V1", "Δ"], body)

    lines = [
        "# Qwen3.5-4B Full vs FlashPrefill V1 — CL-bench",
        "",
        "## Scope",
        "",
        (
            f"This report covers a deterministic, category-balanced {metadata['input_count']}-task "
            "panel drawn from CL-bench. It is a paired diagnostic subset, not the official 1,899-task "
            "leaderboard result. Prompts were not truncated."
        ),
        "",
        "## Official task quality",
        "",
        quality_table,
        "",
        (
            "Primary score is the pinned official strict binary GPT-5.1 judge: a task passes only "
            "when every rubric passes. *Rubric pass is a secondary diagnostic, not the official metric."
        ),
        "",
        (
            f"Paired outcomes: both {global_pair['both_success']}, Full-only "
            f"{global_pair['dense_only']}, V1-only {global_pair['v1_only']}, neither "
            f"{global_pair['neither_success']}. V1−Full paired bootstrap 95% interval: "
            f"[{interval[0]:+.2f}, {interval[1]:+.2f}] percentage points; exact McNemar "
            f"p={p_value:.4g}. These are descriptive for this fixed panel."
        ),
        "",
        "## Full-model prefill latency",
        "",
        latency_table,
        "",
        (
            "Timing spans all 32 model layers, the final LM head, first-token argmax, and CUDA "
            "synchronization. Each sample/config is the median of repeated measurements; paired "
            "speedup is computed per sample and then summarized. †Exact-pair ratio comes from one "
            "independent profile sample and only the eight full-attention layers; the 24 Gated Delta "
            "layers are unchanged."
        ),
        "",
        "## Category breakdown",
        "",
        subgroup_table("context_category"),
        "",
        "## Sub-category breakdown",
        "",
        subgroup_table("sub_category"),
        "",
        "## Prompt-length breakdown",
        "",
        subgroup_table("length_bucket"),
        "",
        "## Integrity and scoring contract",
        "",
        f"- Official evaluator: `{metadata['official_evaluator']['repository']}` at `{metadata['official_evaluator']['commit']}`.",
        f"- Judge: `{judge_audit['judge_model']}`, reasoning effort `{judge_audit['reasoning_effort']}`.",
        f"- Dense graded SHA256: `{judge_audit['graded_files']['dense']['sha256']}`.",
        f"- V1 graded SHA256: `{judge_audit['graded_files'][judge_audit['v1_config_id']]['sha256']}`.",
        "- Raw generated token IDs, thinking text, final text, and termination reason were saved before judging.",
        "- Only final-answer text was sent to the official judge; saved generations permit rescoring without GPU inference.",
        "",
    ]
    return "\n".join(lines)


def main():
    args = arguments()
    if args.bootstrap < 1:
        raise ValueError("--bootstrap must be positive")
    metadata_path = args.run / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata["status"] not in {"generated", "complete"}:
        raise ValueError(f"generation status must be generated or complete, found {metadata['status']}")
    if metadata["official_evaluator"]["judge_model"] != "gpt-5.1":
        raise ValueError("run metadata does not pin the official GPT-5.1 judge")
    inputs = torch.load(args.run / "inputs.pt", map_location="cpu", weights_only=False)
    generations = read_jsonl(args.run / "generations.jsonl")
    configs = metadata["candidates"]
    v1_config = next(config for config in configs if config["method"] == "fp_v1")

    quality = []
    quality.extend(validate_and_score(args.run, inputs, generations, "dense", args.dense_graded))
    quality.extend(
        validate_and_score(
            args.run,
            inputs,
            generations,
            v1_config["config_id"],
            args.v1_graded,
        )
    )
    dense_score = {
        row["sample_id"]: row["task_success"] for row in quality if row["config_id"] == "dense"
    }
    for row in quality:
        row["delta_vs_dense"] = row["task_success"] - dense_score[row["sample_id"]]

    timing = load_timing_summary(args.run, configs, inputs)
    profile = load_profile_summary(args.run, configs)
    summary = aggregate_quality(quality, timing, profile, configs)
    dimensions = ["context_category", "sub_category", "length_bucket"]
    subgroups = subgroup_rows(quality, configs, dimensions)
    paired = paired_outcomes(quality, dimensions)
    dense_values = [
        dense_score[sample["sample_id"]] for sample in inputs
    ]
    v1_by_id = {
        row["sample_id"]: row["task_success"]
        for row in quality
        if row["config_id"] == v1_config["config_id"]
    }
    v1_values = [v1_by_id[sample["sample_id"]] for sample in inputs]
    interval = paired_bootstrap(dense_values, v1_values, args.bootstrap)
    global_pair = next(row for row in paired if row["dimension"] == "all")
    p_value = exact_mcnemar(global_pair["dense_only"], global_pair["v1_only"])

    quality_fields = [
        "method", "config_id", "alpha", "sample_index", "sample_id",
        "context_category", "sub_category", "prompt_tokens", "length_bucket",
        "task_success", "score_percent", "delta_vs_dense", "rubrics_passed",
        "rubrics_total", "rubric_satisfaction", "end_reason", "thinking_status",
        "generated_tokens", "judge_file",
    ]
    summary_fields = [
        "method", "config_id", "alpha", "n", "task_successes",
        "success_rate_percent", "delta_vs_dense_pp", "rubric_satisfaction_percent",
        "max_token_endings", "unclosed_thinking", "median_prefill_ms",
        "median_paired_speedup", "median_peak_allocated_gib", "profile_sample_id",
        "profile_full_attention_layers", "median_exact_pair_ratio",
        "median_attention_ms_per_full_layer", "median_selector_ms_per_full_layer",
        "sum_attention_ms_eight_full_layers",
    ]
    subgroup_fields = [
        "dimension", "group", "method", "config_id", "n", "successes",
        "success_rate_percent", "delta_vs_dense_pp", "rubric_satisfaction_percent",
    ]
    paired_fields = [
        "dimension", "group", "n", "both_success", "dense_only", "v1_only",
        "neither_success", "net_v1_minus_dense",
    ]
    write_csv(args.run / "quality.csv", quality_fields, quality)
    write_csv(args.run / "summary.csv", summary_fields, summary)
    write_csv(args.run / "subgroups.csv", subgroup_fields, subgroups)
    write_csv(args.run / "paired_outcomes.csv", paired_fields, paired)

    judge_audit = {
        "status": "complete",
        "judge_model": "gpt-5.1",
        "reasoning_effort": "low",
        "evaluator_commit": metadata["official_evaluator"]["commit"],
        "rows_per_config": len(inputs),
        "v1_config_id": v1_config["config_id"],
        "graded_files": {
            "dense": {"path": str(args.dense_graded), "sha256": sha256_file(args.dense_graded)},
            v1_config["config_id"]: {
                "path": str(args.v1_graded),
                "sha256": sha256_file(args.v1_graded),
            },
        },
        "paired_bootstrap_iterations": args.bootstrap,
        "v1_minus_dense_95_percent_interval_pp": interval,
        "exact_mcnemar_p": p_value,
    }
    write_json(args.run / "judge_audit.json", judge_audit)
    (args.run / "REPORT.md").write_text(
        report_text(metadata, summary, subgroups, paired, interval, p_value, judge_audit),
        encoding="utf-8",
    )
    metadata["status"] = "complete"
    metadata["quality_status"] = "official_judge_complete"
    metadata["judge_audit"] = judge_audit
    metadata["completed_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    write_json(metadata_path, metadata)
    print(f"Complete CL-bench report: {args.run / 'REPORT.md'}", flush=True)


if __name__ == "__main__":
    main()
