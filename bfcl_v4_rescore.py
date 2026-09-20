"""Re-run the pinned BFCL state checker from saved trajectories without inference."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import time

from bfcl_v4_agent import (
    BFCL_VERSION,
    episode_row,
    load_bfcl,
    score_record,
)


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--tag", default="current")
    parser.add_argument(
        "--bfcl-root",
        type=Path,
        default=Path("third_party/bfcl_eval_2025_12_17"),
    )
    parser.add_argument(
        "--bfcl-wheel",
        type=Path,
        default=Path("third_party/downloads/bfcl_eval-2025.12.17-py3-none-any.whl"),
    )
    return parser.parse_args()


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def write_jsonl(path, rows):
    with Path(path).open("w", encoding="utf-8") as output:
        for row in rows:
            output.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def main():
    args = arguments()
    if not args.tag or args.tag in {".", ".."} or "/" in args.tag or "\\" in args.tag:
        raise ValueError("tag must be one path component")
    raw_path = args.run_dir / "generations.jsonl"
    predictions_path = args.run_dir / "predictions.jsonl"
    metadata_path = args.run_dir / "metadata.json"
    records = read_jsonl(raw_path)
    stored = {
        (row["id"], row["config_id"]): row.get("checker_result")
        for row in read_jsonl(predictions_path)
    } if predictions_path.is_file() else {}
    run_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    categories = {record["category"] for record in records}
    if len(categories) != 1:
        raise ValueError(f"saved trajectories contain mixed BFCL categories: {categories}")
    category = categories.pop()
    metadata_category = run_metadata["arguments"].get(
        "category", "multi_turn_long_context"
    )
    if category != metadata_category:
        raise ValueError(
            f"trajectory category {category} differs from metadata {metadata_category}"
        )

    loader_args = SimpleNamespace(
        bfcl_root=args.bfcl_root,
        bfcl_wheel=args.bfcl_wheel,
        category=category,
        samples=200,
        seed=int(run_metadata["arguments"]["seed"]),
        selected_cases=None,
    )
    entries, sources, multi_turn_utils, multi_turn_checker = load_bfcl(loader_args)
    entries_by_id = {entry["id"]: entry for entry in entries}
    missing = sorted({record["id"] for record in records} - set(entries_by_id))
    if missing:
        raise ValueError(f"saved BFCL IDs are absent from the pinned package: {missing}")

    output_dir = args.run_dir / "rescoring" / args.tag
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty snapshot: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    quality_rows = []
    episode_rows = []
    score_rows = []
    changed = 0
    compared = 0
    for record in records:
        entry = entries_by_id[record["id"]]
        if record["source_row_sha256"] != entry["source_row_sha256"]:
            raise ValueError(f"source row hash mismatch for {record['id']}")
        checker_result, checker_decoded_result = score_record(
            record,
            entry,
            multi_turn_utils,
            multi_turn_checker,
        )
        stored_result = stored.get((record["id"], record["config_id"]))
        score_changed = None
        if stored_result is not None:
            compared += 1
            score_changed = bool(stored_result["valid"]) != bool(checker_result["valid"])
            changed += int(score_changed)
        quality_rows.append({
            "method": record["method"],
            "config_id": record["config_id"],
            "alpha": record["alpha"],
            "sample_id": record["id"],
            "success": int(checker_result["valid"]),
            "stored_success": "" if stored_result is None else int(stored_result["valid"]),
            "score_changed": "" if score_changed is None else int(score_changed),
            "error_type": checker_result.get("error_type"),
            "error_message": checker_result.get("error_message"),
        })
        episode_rows.append(episode_row(record, checker_result["valid"]))
        score_rows.append({
            "id": record["id"],
            "config_id": record["config_id"],
            "source_row_sha256": record["source_row_sha256"],
            "stored_checker_result": stored_result,
            "checker_decoded_result": checker_decoded_result,
            "rescored_checker_result": checker_result,
        })

    quality_fields = [
        "method", "config_id", "alpha", "sample_id", "success", "stored_success",
        "score_changed", "error_type", "error_message",
    ]
    with (output_dir / "quality.csv").open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=quality_fields)
        writer.writeheader()
        writer.writerows(quality_rows)
    episode_fields = [
        "method", "config_id", "alpha", "sample_id", "success", "expected_turns",
        "completed_turns", "steps", "tool_calls_executed", "prompt_token_instances",
        "max_prompt_tokens", "total_prefill_ms", "total_generation_ms",
        "force_terminated", "termination_reason",
    ]
    with (output_dir / "episodes.csv").open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=episode_fields)
        writer.writeheader()
        writer.writerows(episode_rows)
    write_jsonl(output_dir / "scores.jsonl", score_rows)

    groups = {}
    for row in quality_rows:
        groups.setdefault(row["config_id"], []).append(row)
    lines = [
        "# BFCL V4 offline rescoring snapshot",
        "",
        f"- Category: `{category}`",
        f"- Pinned checker package: `{BFCL_VERSION}`",
        f"- Raw trajectories rescored: {len(records)}",
        f"- Compared with stored checker results: {compared}",
        f"- Changed success decisions: {changed}",
        "- Model inference and tool-trajectory generation were not rerun.",
        "",
        "| Config | n | Stored success | Rescored success | Changed |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for config_id, rows in sorted(groups.items()):
        stored_success = sum(
            int(row["stored_success"])
            for row in rows
            if row["stored_success"] != ""
        )
        lines.append(
            f"| {config_id} | {len(rows)} | {stored_success} | "
            f"{sum(row['success'] for row in rows)} | "
            f"{sum(int(row['score_changed']) for row in rows if row['score_changed'] != '')} |"
        )
    (output_dir / "SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    manifest = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "run_dir": str(args.run_dir.resolve()),
        "bfcl_package_version": BFCL_VERSION,
        "category": category,
        "generation_rerun": False,
        "raw_generation_sha256": file_hash(raw_path),
        "stored_predictions_sha256": (
            file_hash(predictions_path) if predictions_path.is_file() else None
        ),
        "original_metadata_sha256": file_hash(metadata_path),
        "adapter_source_sha256": file_hash(Path(__file__).with_name("bfcl_v4_agent.py")),
        "rescore_source_sha256": file_hash(Path(__file__)),
        "bfcl_sources": sources,
        "raw_trajectory_count": len(records),
        "stored_comparison_count": compared,
        "changed_success_count": changed,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Saved BFCL offline rescoring snapshot to {output_dir}")


if __name__ == "__main__":
    main()
