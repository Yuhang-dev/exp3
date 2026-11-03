"""Write BATCH2.md: inference-time temperature, shared-namespace competitor sweep, and weight compression."""

import json
from pathlib import Path
import sys

import pandas as pd
import torch
from transformers import AutoTokenizer

from summarize_followup import error_classes
from summarize_pointer_study import read_jsonl, sign_test, table


def scores(folder, label=None):
    quality = pd.read_csv(folder / "quality.csv")
    quality["benchmark"] = quality.task.str.split("/").str[0]
    if label:
        quality["config_id"] = label
    return quality


def by_benchmark(quality):
    wide = quality.pivot_table(index="config_id", columns="benchmark", values="score", aggfunc="mean")
    return table(wide.reset_index())


def paired(quality, reference):
    prompt = quality.pivot_table(index="sample_id", columns="config_id", values="score")
    rows = []
    for config in prompt.columns:
        if config == reference:
            continue
        difference = (prompt[config] - prompt[reference]).dropna()
        better, worse = int((difference > 0).sum()), int((difference < 0).sum())
        rows.append({"config": config, "versus": reference, "mean_pp": difference.mean(),
                     "prompts_better": better, "prompts_worse": worse, "sign_p": sign_test(better, worse)})
    return table(pd.DataFrame(rows))


def generations(folder, label=None):
    records = read_jsonl(folder / "generations.jsonl")
    if label:
        for record in records:
            record["config_id"] = label
    return records


def main():
    root = Path(sys.argv[1])
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
    samples = {}
    for path in (root / "batch2_inputs/combined.pt", root / "batch2_inputs/competitors.pt"):
        for sample in torch.load(path, map_location="cpu", weights_only=False):
            samples[sample["sample_id"]] = sample
    prompts = {}
    sections = ["# Batch 2\n"]

    temperature = scores(root / "temperature")
    sections += ["## 1. Inference-time sharpening (dense)\n", by_benchmark(temperature),
                 paired(temperature, "dense"), "### Error classes\n",
                 error_classes(generations(root / "temperature"), samples, tokenizer, "config_id", prompts)]

    sweep = []
    records = []
    for name in ("competitors_7b", "competitors_0p5b"):
        frame = scores(root / name)
        if name.endswith("0p5b"):
            frame["config_id"] = "qwen2.5-0.5b_dense"
        sweep.append(frame)
        records += generations(root / name, "qwen2.5-0.5b_dense" if name.endswith("0p5b") else None)
    sweep = pd.concat(sweep)
    sweep["level"] = sweep.sample_id.str.split("-synthetic").str[0]
    wide = sweep.pivot_table(index="config_id", columns="level", values="score", aggfunc="mean")
    for record in records:
        record["config_id"] = f"{record['config_id']}|{record['sample_id'].split('-synthetic')[0]}"
    sections += ["## 2. Share of fillers in the target namespace (holdout prompts; shared_1 is the holdout run)\n",
                 table(wide.reset_index()), "### Error classes\n",
                 error_classes(records, samples, tokenizer, "config_id", prompts)]

    compressed = [scores(root / "temperature").query("config_id == 'dense'")]
    records = [record for record in generations(root / "temperature") if record["config_id"] == "dense"]
    for folder in sorted((root / "weights").iterdir()):
        compressed.append(scores(folder, folder.name))
        records += generations(folder, folder.name)
    compressed = pd.concat(compressed)
    perplexity = pd.DataFrame([json.loads(path.read_text()) for path in sorted((root / "ppl").glob("*.json"))])
    sections += ["## 3. Weight compression\n", by_benchmark(compressed), paired(compressed, "dense"),
                 "### C4-validation perplexity\n", table(perplexity),
                 "### Error classes\n", error_classes(records, samples, tokenizer, "config_id", prompts)]
    text = "\n".join(sections)
    (root / "BATCH2.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
