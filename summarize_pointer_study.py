"""Write SUMMARY.md for a pointer-study result root (alpha, length, patch, factors, holdout)."""

import json
import math
from pathlib import Path
import re
import sys

import pandas as pd

from scoring import parse_synthetic


NEAR_LOW, NEAR_HIGH = 21, 120


def table(frame, digits=3):
    frame = frame.round(digits)
    head = "| " + " | ".join(map(str, frame.columns)) + " |"
    rule = "|" + " --- |" * len(frame.columns)
    rows = ["| " + " | ".join(map(str, row)) + " |" for row in frame.itertuples(index=False)]
    return "\n".join([head, rule, *rows]) + "\n"


def sign_test(better, worse):
    total = better + worse
    if total == 0:
        return 1.0
    tail = sum(math.comb(total, k) for k in range(0, min(better, worse) + 1)) / 2**total
    return min(1.0, 2 * tail)


def read_jsonl(path):
    return [json.loads(line) for line in path.open(encoding="utf-8")]


def quality_section(folder, by_length=False):
    quality = pd.read_csv(folder / "quality.csv")
    quality["variant"] = quality.task.str.split("/").str[1]
    keys = ["length_label", "config_id"] if by_length else ["config_id"]
    wide = quality.pivot_table(index=keys, columns="variant", values="target_accuracy", aggfunc="mean") * 100
    wide["all_targets"] = quality.groupby(keys).target_accuracy.mean() * 100
    wide["all_correct_prompts"] = quality.groupby(keys).all_target_em.sum()
    wide["prompts"] = quality.groupby(keys).all_target_em.count()
    profile_path = folder / "profile.csv"
    if profile_path.exists() and profile_path.stat().st_size:
        profile = pd.read_csv(profile_path)
        if len(profile):
            wide["exact_pair_ratio_pct"] = profile.groupby("config_id").effective_exact_token_pair_ratio.mean() * 100
    return table(wide.reset_index())


def paired_section(folder):
    quality = pd.read_csv(folder / "quality.csv")
    prompt = quality.pivot_table(index="sample_id", columns="config_id", values="target_accuracy")
    rows = []
    for config in prompt.columns:
        for reference in ("dense", "fp_v1__a0p08"):
            if config == reference:
                continue
            difference = prompt[config] - prompt[reference]
            better, worse = int((difference > 0).sum()), int((difference < 0).sum())
            rows.append({
                "config": config,
                "versus": reference,
                "mean_pp": 100 * difference.mean(),
                "prompts_better": better,
                "prompts_worse": worse,
                "sign_p": sign_test(better, worse),
            })
    return table(pd.DataFrame(rows))


def target_rows(records):
    rows = []
    for record in records:
        for index, key in enumerate(record["target_keys"]):
            rows.append({
                "sample_id": record["sample_id"],
                "condition": record["condition"],
                "target_index": index,
                "correct": record["target_correct"][key],
            })
    return pd.DataFrame(rows)


def patch_section(folder):
    records = read_jsonl(folder / "generations.jsonl")
    targets = target_rows(records)
    identifier = ["sample_id", "target_index"]
    failed = targets[(targets.condition == "v1_base") & (targets.correct == 0)][identifier]
    rescued = targets.merge(failed, on=identifier).groupby("condition").correct.agg(["sum", "count"])
    held = targets[(targets.condition == "dense_base") & (targets.correct == 1)][identifier]
    kept = targets.merge(held, on=identifier).groupby("condition").correct.agg(["sum", "count"])
    summary = pd.DataFrame({
        "correct": targets.groupby("condition").correct.sum(),
        "targets": targets.groupby("condition").correct.count(),
        "v1_failures_rescued": rescued["sum"],
        "v1_failures": rescued["count"],
        "dense_correct_kept": kept["sum"],
    }).reset_index()
    output = ["### Task outcome by condition\n", table(summary)]

    margins = pd.read_csv(folder / "margins.csv", keep_default_na=False)
    value_rows = margins[margins.segment.isin(["prefix", "index", "suffix"])]
    by_segment = value_rows.groupby(["condition", "segment"]).margin.agg(
        mean="mean",
        median="median",
        minimum="min",
        negative=lambda values: (values < 0).sum(),
        steps="count",
    ).reset_index()
    output += ["### Teacher-forced margin by value segment\n", table(by_segment)]

    negative = margins[(margins.condition == "v1_base") & (margins.margin < 0)]
    first = negative.sort_values("step").groupby(["sample_id", "target_index"], as_index=False).first()
    output += [
        "### v1_base: first negative-margin token per target\n",
        table(first[["sample_id", "target_index", "segment", "step", "gold_token", "competitor_token", "margin"]]),
    ]

    drift = pd.read_csv(folder / "kv_drift.csv")
    drift["band"] = pd.cut(drift.layer, [-1, 0, 9, 18, 27], labels=["L0", "L1-9", "L10-18", "L19-27"]).astype(str)
    pivot = drift.pivot_table(index=["comparison", "part", "region"], columns="band", values="rel_l2").reset_index()
    output += ["### K/V relative L2 drift against dense\n", table(pivot, 5)]
    return "\n".join(output)


def factor_targets(folder):
    samples = {sample["sample_id"]: sample for sample in read_jsonl(folder / "inputs.jsonl")}
    rows = []
    for record in read_jsonl(folder / "generations.jsonl"):
        sample = samples[record["sample_id"]]
        parsed = parse_synthetic(record["raw_text"], sample["target_keys"])
        for index, key in enumerate(sample["target_keys"]):
            expected = sample["answers"][key]
            got = parsed.get(key, "")
            index_char = len(expected) - 9
            start = sample["evidence_positions"][index]["token_span"][0]
            rows.append({
                "config_id": record["config_id"],
                "factor": sample["factor"],
                "base": sample["base_sample_id"],
                "variant": sample["variant"],
                "target_index": index,
                "correct": int(got == expected),
                "emitted_index": got[index_char] if len(got) > index_char else "",
                "right_suffix": int(got.endswith(expected[-8:])),
                "offset": start % 128,
            })
    return pd.DataFrame(rows)


def factors_section(folder):
    targets = factor_targets(folder)
    accuracy = targets.pivot_table(index="factor", columns="config_id", values="correct", aggfunc="mean") * 100
    output = ["### Target accuracy by factor\n", table(accuracy.reset_index())]

    sparse = targets[targets.config_id == "fp_v1__a0p08"]
    collapse = sparse[(sparse.correct == 0) & (sparse.right_suffix == 1)]
    counts = collapse.groupby(["factor", "emitted_index"]).size().reset_index(name="errors")
    output += ["### V1 wrong-index errors: emitted index character\n", table(counts)]

    shifts = sparse[sparse.factor.str.startswith("shift_") | (sparse.factor == "baseline")].copy()
    shifts["bin"] = shifts.offset // 16 * 16
    shifts["near"] = (shifts.offset <= NEAR_LOW) | (shifts.offset >= NEAR_HIGH)
    output += [
        "### V1 failure rate by needle offset in its block\n",
        table(shifts.groupby("bin").correct.agg(fail=lambda values: 100 * (1 - values.mean()), targets="count").reset_index()),
    ]
    per_target = shifts.pivot_table(index=["base", "target_index"], columns="near", values="correct", aggfunc="mean")
    per_target = per_target.dropna()
    better = int((per_target[False] > per_target[True]).sum())
    worse = int((per_target[False] < per_target[True]).sum())
    output.append(
        f"Same needle, near vs far from a block boundary (offset <= {NEAR_LOW} or >= {NEAR_HIGH}): "
        f"far better for {better} targets, near better for {worse}, sign p = {sign_test(better, worse):.4f}; "
        f"near failure {100 * (1 - shifts[shifts.near].correct.mean()):.1f}%, "
        f"far failure {100 * (1 - shifts[~shifts.near].correct.mean()):.1f}%.\n"
    )
    return "\n".join(output)


def main():
    root = Path(sys.argv[1])
    sections = ["# Pointer study summary\n"]
    if (root / "alpha/quality.csv").exists():
        sections += ["## Alpha sweep, 32K calibration\n", quality_section(root / "alpha")]
    if (root / "length/quality.csv").exists():
        sections += ["## Length dependence\n", quality_section(root / "length", by_length=True)]
    if (root / "patch/generations.jsonl").exists():
        sections += ["## K/V patching and margins\n", patch_section(root / "patch")]
    if (root / "factors/generations.jsonl").exists():
        sections += ["## Counterfactual factors\n", factors_section(root / "factors")]
    if (root / "holdout/quality.csv").exists():
        sections += [
            "## Holdout, frozen alpha 0.08\n",
            quality_section(root / "holdout"),
            paired_section(root / "holdout"),
        ]
    text = "\n".join(sections)
    (root / "SUMMARY.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
