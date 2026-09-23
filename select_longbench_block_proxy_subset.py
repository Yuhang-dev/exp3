"""Select an outcome-blind, domain-stratified natural-text mechanism panel."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import torch

from data import input_hash


DOMAINS = (
    "Single-Document QA",
    "Multi-Document QA",
    "Long-dialogue History Understanding",
    "Code Repository Understanding",
)
SELECTION_SALT = "block-proxy-natural-v1:"


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
    parser.add_argument("--min-tokens", type=int, default=24576)
    parser.add_argument("--per-domain", type=int, default=2)
    args = parser.parse_args()

    source_inputs = args.source / "inputs.pt"
    source_quality = args.source / "quality.csv"
    inputs = torch.load(source_inputs, map_location="cpu", weights_only=False)
    scores = {}
    with source_quality.open(newline="", encoding="utf-8") as source:
        for row in csv.DictReader(source):
            if row["method"] in ("dense", "fp_v1"):
                scores.setdefault(row["sample_id"], {})[row["method"]] = float(
                    row["score"]
                )

    chosen = []
    records = []
    for domain in DOMAINS:
        eligible = [
            sample for sample in inputs
            if sample["domain"] == domain
            and sample["actual_tokens"] >= args.min_tokens
        ]
        eligible.sort(key=lambda sample: hashlib.sha256(
            (SELECTION_SALT + sample["sample_id"]).encode("utf-8")
        ).hexdigest())
        if len(eligible) < args.per_domain:
            raise ValueError(f"{domain}: only {len(eligible)} eligible inputs")
        for sample in eligible[:args.per_domain]:
            assert input_hash(sample["input_ids"]) == sample["input_sha256"]
            chosen.append(sample)
            records.append({
                "sample_id": sample["sample_id"],
                "domain": domain,
                "sub_domain": sample["sub_domain"],
                "actual_tokens": sample["actual_tokens"],
                "dense_score": scores[sample["sample_id"]]["dense"],
                "fp_v1_score": scores[sample["sample_id"]]["fp_v1"],
                "input_sha256": sample["input_sha256"],
                "source_row_sha256": sample["source_row_sha256"],
            })

    args.out.mkdir(parents=True)
    output_inputs = args.out / "inputs.pt"
    torch.save(chosen, output_inputs)
    with (args.out / "inputs.jsonl").open("w", encoding="utf-8") as output:
        for sample in chosen:
            public = {key: value for key, value in sample.items() if key != "input_ids"}
            output.write(json.dumps(public, ensure_ascii=False) + "\n")
    manifest = {
        "purpose": "Outcome-blind natural-text mechanism panel, not a benchmark estimate",
        "selection_rule": {
            "domains": DOMAINS,
            "min_tokens": args.min_tokens,
            "per_domain": args.per_domain,
            "order": "ascending SHA256 of salt plus sample_id",
            "salt": SELECTION_SALT,
            "quality_scores_used_for_selection": False,
        },
        "source": str(args.source.resolve()),
        "source_sha256": {
            "inputs.pt": sha256(source_inputs),
            "quality.csv": sha256(source_quality),
        },
        "selection": records,
        "subset_sha256": {
            "inputs.pt": sha256(output_inputs),
            "inputs.jsonl": sha256(args.out / "inputs.jsonl"),
        },
    }
    (args.out / "selection.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[natural-subset] saved {len(chosen)} inputs to {args.out}")
    for record in records:
        print(
            f"  {record['domain']}: {record['sample_id']} "
            f"{record['actual_tokens']} tokens, "
            f"dense={record['dense_score']:g}, V1={record['fp_v1_score']:g}"
        )


if __name__ == "__main__":
    main()
