"""Save a fixed real-RULER subset from the completed 32K pilot inputs."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import torch

from data import input_hash


# Four score-discordant cases plus one correct control for each RULER task.
# Controls for mk_1, mk_3, and mq have nearby answer-text positions.
SELECTION = (
    ("ruler_niah_mk_1:32768:71", "v1_wrong_dense_correct", 100.0, 0.0),
    ("ruler_niah_mk_1:32768:56", "both_correct_control", 100.0, 100.0),
    ("ruler_niah_mk_2:32768:59", "both_correct_control", 100.0, 100.0),
    ("ruler_niah_mk_3:32768:50", "v1_wrong_dense_correct", 100.0, 0.0),
    ("ruler_niah_mk_3:32768:27", "v1_correct_dense_wrong", 0.0, 100.0),
    ("ruler_niah_mk_3:32768:56", "both_correct_control", 100.0, 100.0),
    ("ruler_niah_mq:32768:59", "v1_partial_dense_correct", 100.0, 75.0),
    ("ruler_niah_mq:32768:21", "both_correct_control", 100.0, 100.0),
)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite subset: {args.out}")

    source_inputs = args.source / "inputs.pt"
    source_quality = args.source / "quality.csv"
    inputs = torch.load(source_inputs, map_location="cpu", weights_only=False)
    by_id = {sample["sample_id"]: sample for sample in inputs}
    scores = {}
    with source_quality.open(newline="", encoding="utf-8") as source:
        for row in csv.DictReader(source):
            if row["method"] in ("dense", "fp_v1"):
                scores.setdefault(row["sample_id"], {})[row["method"]] = float(
                    row["score"]
                )

    chosen = []
    records = []
    for sample_id, role, expected_dense, expected_v1 in SELECTION:
        sample = by_id[sample_id]
        pair = scores[sample_id]
        if (pair["dense"], pair["fp_v1"]) != (expected_dense, expected_v1):
            raise ValueError(f"saved RULER score pair changed: {sample_id}")
        if input_hash(sample["input_ids"]) != sample["input_sha256"]:
            raise ValueError(f"stored input hash mismatch: {sample_id}")
        chosen.append(sample)
        prompt = sample["prompt_text"]
        answer_positions = [
            prompt.find(str(answer)) / len(prompt)
            for answer in sample["answers"]
        ]
        records.append({
            "sample_id": sample_id,
            "role": role,
            "task_label": sample["task_label"],
            "actual_tokens": sample["actual_tokens"],
            "dense_score": pair["dense"],
            "fp_v1_score": pair["fp_v1"],
            "answer_first_occurrence_char_fractions": answer_positions,
            "input_sha256": sample["input_sha256"],
            "prompt_sha256": sample["prompt_sha256"],
            "source_file_sha256": sample["source_file_sha256"],
            "source_row_sha256": sample["source_row_sha256"],
        })

    args.out.mkdir(parents=True)
    subset_pt = args.out / "inputs.pt"
    subset_jsonl = args.out / "inputs.jsonl"
    torch.save(chosen, subset_pt)
    with subset_jsonl.open("w", encoding="utf-8") as output:
        for sample in chosen:
            public = {key: value for key, value in sample.items() if key != "input_ids"}
            output.write(json.dumps(public, ensure_ascii=False) + "\n")
    manifest = {
        "purpose": "Outcome-stratified mechanism subset, not a benchmark estimate",
        "source": str(args.source.resolve()),
        "source_sha256": {
            "inputs.pt": sha256(source_inputs),
            "quality.csv": sha256(source_quality),
        },
        "selection": records,
        "subset_sha256": {
            "inputs.pt": sha256(subset_pt),
            "inputs.jsonl": sha256(subset_jsonl),
        },
    }
    (args.out / "selection.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[ruler-subset] saved {len(chosen)} original inputs to {args.out}")
    for record in records:
        print(
            f"  {record['sample_id']}: {record['role']}, "
            f"dense={record['dense_score']:g}, V1={record['fp_v1_score']:g}"
        )


if __name__ == "__main__":
    main()
