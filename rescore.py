"""Re-score saved raw generations without loading the model or touching GPU artifacts."""

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
import time

from scoring import SCORER_VERSION, score_prediction


QUALITY_FIELDS = [
    "method", "config_id", "alpha", "task", "split", "length_label",
    "total_context_budget", "prompt_budget", "actual_tokens", "sample_id",
    "source_id", "metric", "scorer_version", "score", "delta_vs_dense",
    "exact_match", "target_accuracy", "all_target_em", "raw_substring_score",
    "normalized_substring_score", "parsed_answer", "stored_score", "score_changed",
]


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--tag", default="current", help="Derived snapshot name under rescoring/.")
    return parser.parse_args()


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path):
    with path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def numeric_score(value):
    return float(value["score"] if isinstance(value, dict) else value)


def sample_key(sample):
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
    }


def main():
    args = arguments()
    inputs_path = args.run_dir / "inputs.jsonl"
    predictions_path = args.run_dir / "predictions.jsonl"
    generations_path = args.run_dir / "generations.jsonl"
    inputs = {sample["sample_id"]: sample for sample in read_jsonl(inputs_path)}
    predictions = read_jsonl(predictions_path) if predictions_path.exists() else []
    generations = read_jsonl(generations_path) if generations_path.exists() else predictions
    stored_predictions = {
        (prediction["sample_id"], prediction["config_id"]): prediction
        for prediction in predictions
    }
    output_dir = args.run_dir / "rescoring" / args.tag
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    score_records = []
    mismatch_count = 0
    compared_count = 0
    for generation in generations:
        sample = inputs[generation["sample_id"]]
        raw_text = generation.get("raw_text", generation["text"])
        scored = score_prediction(sample, raw_text)
        stored_prediction = stored_predictions.get(
            (generation["sample_id"], generation["config_id"])
        )
        stored = numeric_score(stored_prediction["score"]) if stored_prediction else None
        changed = None if stored is None else abs(stored - float(scored["score"])) > 1e-9
        compared_count += int(stored is not None)
        mismatch_count += int(changed or False)
        row = {
            "method": generation["method"],
            "config_id": generation["config_id"],
            "alpha": generation.get("alpha"),
            **sample_key(sample),
            "metric": scored["metric"],
            "scorer_version": scored["scorer_version"],
            "score": scored["score"],
            "exact_match": scored["exact_match"],
            "target_accuracy": scored["target_accuracy"],
            "all_target_em": scored["all_target_em"],
            "raw_substring_score": scored.get("raw_substring_score"),
            "normalized_substring_score": scored.get("normalized_substring_score"),
            "parsed_answer": json.dumps(scored["parsed_answer"], ensure_ascii=False),
            "stored_score": "" if stored is None else stored,
            "score_changed": "" if changed is None else int(changed),
        }
        rows.append(row)
        score_records.append({
            "sample_id": generation["sample_id"],
            "config_id": generation["config_id"],
            "method": generation["method"],
            "input_sha256": sample["input_sha256"],
            "raw_text_sha256": hashlib.sha256(raw_text.encode("utf-8")).hexdigest(),
            "stored_score": None if stored_prediction is None else stored_prediction["score"],
            "rescored": scored,
        })

    dense = {
        row["sample_id"]: row["score"]
        for row in rows
        if row["method"] == "dense"
    }
    for row in rows:
        reference = dense.get(row["sample_id"])
        row["delta_vs_dense"] = "" if reference is None else row["score"] - reference

    with (output_dir / "quality.csv").open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=QUALITY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    with (output_dir / "scores.jsonl").open("w", encoding="utf-8") as output:
        for record in score_records:
            output.write(json.dumps(record, ensure_ascii=False) + "\n")

    groups = defaultdict(list)
    for row in rows:
        groups[(row["task"], row["length_label"], row["config_id"])].append(row)
    lines = [
        "# Offline rescoring snapshot",
        "",
        f"- Scorer: `{SCORER_VERSION}`",
        f"- Raw generations rescored: {len(rows)}",
        f"- Compared with a stored score: {compared_count}",
        f"- Changed among stored scores: {mismatch_count}",
        "- No model inference or generation was run.",
        "",
        "| Task | Length | Config | n | Stored | Rescored | Changed |",
        "| --- | ---: | --- | ---: | ---: | ---: | ---: |",
    ]
    for (task, length, config_id), group in sorted(groups.items()):
        stored_values = [row["stored_score"] for row in group if row["stored_score"] != ""]
        stored_mean = (
            f"{sum(stored_values) / len(stored_values):.2f}"
            if stored_values
            else "—"
        )
        rescored_mean = sum(row["score"] for row in group) / len(group)
        changed = sum(
            row["score_changed"]
            for row in group
            if row["score_changed"] != ""
        )
        lines.append(
            f"| {task} | {length} | {config_id} | {len(group)} | "
            f"{stored_mean} | {rescored_mean:.2f} | {changed} |"
        )
    (output_dir / "SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    scorer_path = Path(__file__).with_name("scoring.py")
    manifest = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "run_dir": str(args.run_dir.resolve()),
        "scorer_version": SCORER_VERSION,
        "scoring_py_sha256": sha256(scorer_path),
        "inputs_jsonl_sha256": sha256(inputs_path),
        "raw_generation_file": generations_path.name if generations_path.exists() else predictions_path.name,
        "raw_generation_sha256": sha256(
            generations_path if generations_path.exists() else predictions_path
        ),
        "predictions_jsonl_sha256": sha256(predictions_path) if predictions_path.exists() else None,
        "input_count": len(inputs),
        "prediction_count": len(predictions),
        "raw_generation_count": len(generations),
        "stored_score_comparison_count": compared_count,
        "changed_score_count": mismatch_count,
        "generation_rerun": False,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Saved offline rescoring snapshot to {output_dir}")


if __name__ == "__main__":
    main()
