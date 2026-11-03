"""Write FOLLOWUP.md: margin erosion, K/V quantization, and small-model dense failures with error classes."""

from collections import Counter
import json
from pathlib import Path
import sys

import pandas as pd
import torch
from transformers import AutoTokenizer

from scoring import parse_synthetic
from summarize_pointer_study import read_jsonl, sign_test, table


def longest_in_prompt(tail, prompt):
    for size in range(len(tail), 0, -1):
        if any(tail[start:start + size] in prompt for start in range(len(tail) - size + 1)):
            return size
    return 0


def classify(got, expected, answers, prompt):
    if got == expected:
        return "correct"
    if not got:
        return "missing"
    if got in answers:
        return "other_target_verbatim"
    if got in prompt or f"F{got}" in prompt:
        return "other_record_verbatim"
    if got[-8:] == expected[-8:]:
        return "right_suffix_wrong_index"
    if expected.startswith(got):
        return "truncated_correct"
    shared = 0
    while shared < min(len(got), len(expected)) and got[shared] == expected[shared]:
        shared += 1
    if longest_in_prompt(got[shared:], prompt) >= 6:
        return "other_record_fragment"
    return "other"


def error_classes(records, samples, tokenizer, config_key, prompts):
    counts = {}
    for record in records:
        sample = samples.get(record["sample_id"])
        if sample is None or sample["task"] != "synthetic_kv_retrieval":
            continue
        prompt = prompts.setdefault(
            sample["sample_id"], tokenizer.decode(sample["input_ids"], skip_special_tokens=False)
        )
        parsed = parse_synthetic(record["raw_text"], sample["target_keys"])
        answers = set(sample["answers"].values())
        for key in sample["target_keys"]:
            label = classify(parsed.get(key, ""), sample["answers"][key], answers, prompt)
            counts.setdefault((sample["task_label"].split("/")[0], record[config_key]), Counter())[label] += 1
    rows = []
    for (benchmark, config), counter in sorted(counts.items()):
        rows.append({"benchmark": benchmark, "config": config, "targets": sum(counter.values()), **counter})
    return table(pd.DataFrame(rows).fillna(0))


def margin_table(folder):
    records = pd.DataFrame(read_jsonl(folder / "generations.jsonl"))
    margins = pd.read_csv(folder / "margins.csv", keep_default_na=False)
    minimum = margins.groupby(["sample_id", "condition"]).margin.min().unstack()
    rows = []
    for (benchmark, condition), group in records[records.condition != "dense"].groupby(["benchmark", "condition"]):
        dense = records[(records.benchmark == benchmark) & (records.condition == "dense")]
        same = group[group.identical_to_dense].sample_id
        difference = (minimum[condition] - minimum["dense"]).loc[same]
        down, up = int((difference < 0).sum()), int((difference > 0).sum())
        rows.append({
            "benchmark": benchmark,
            "condition": condition,
            "prompts": len(group),
            "dense_score": dense.score.mean(),
            "score": group.score.mean(),
            "identical_output": len(same),
            "median_min_margin_dense": minimum["dense"].loc[same].median(),
            "median_change": difference.median(),
            "margin_lower": down,
            "margin_higher": up,
            "sign_p": sign_test(down, up),
        })
    return table(pd.DataFrame(rows))


def main():
    root = Path(sys.argv[1])
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
    samples = {}
    for path in [root / "holdout/inputs.pt", root / "followup_inputs/logs.pt", root / "followup_inputs/ruler/inputs.pt", root / "followup_inputs/longbench/inputs.pt"]:
        for sample in torch.load(path, map_location="cpu", weights_only=False):
            samples[sample["sample_id"]] = sample
    prompts = {}
    sections = ["# Follow-up experiments\n", "## 1. Margin on real tasks\n", margin_table(root / "margin")]
    records = read_jsonl(root / "margin/generations.jsonl")
    sections += ["### Error classes on the log task\n", error_classes(records, samples, tokenizer, "condition", prompts)]
    sections += ["## 2. K/V cache quantization on the holdout prompts\n", margin_table(root / "quant")]
    records = read_jsonl(root / "quant/generations.jsonl")
    sections += ["### Error classes\n", error_classes(records, samples, tokenizer, "condition", prompts)]
    sections.append("## 3. Dense on smaller models, holdout prompts\n")
    for name in sorted(path.name for path in (root / "small").iterdir()):
        records = read_jsonl(root / "small" / name / "generations.jsonl")
        for record in records:
            record["config_id"] = name
        sections.append(error_classes(records, samples, tokenizer, "config_id", prompts))
    text = "\n".join(sections)
    (root / "FOLLOWUP.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
