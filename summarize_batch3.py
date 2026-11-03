"""Write BATCH3.md: exact copying of real identifiers (HDFS, OpenStack, wheel SHA-256 manifests) under weight
compression and sparse prefill, with strict whole-value error classes and the line offset of copied values.
Usage: summarize_batch3.py ROOT [RUNS INPUTS REPORT]; prompts are grouped by sample_id without its index."""

from collections import Counter
from pathlib import Path
import re
import sys

import pandas as pd
import torch

from prepare_copy_tasks import SOURCES
from scoring import parse_synthetic
from summarize_pointer_study import read_jsonl, sign_test, table


def label(got, expected, values, lines, target_line):
    if got == expected:
        return "correct", None
    if not got:
        return "missing", None
    if got in values:
        offsets = [index - target_line for index, line in enumerate(lines) if got in line]
        return "copy_other_line", min(offsets, key=abs)
    if expected.startswith(got):
        return "truncated", None
    if len(got) == len(expected) and sum(a != b for a, b in zip(got, expected)) <= 2:
        return "near_miss", None
    return "other", None


def main():
    root = Path(sys.argv[1])
    runs, inputs, report = sys.argv[2:5] if len(sys.argv) > 2 else ("batch3", "batch3_inputs/copy.pt", "BATCH3.md")
    samples = {s["sample_id"]: s for s in torch.load(root / inputs, weights_only=False)}
    quality, counts, offsets = [], {}, {}
    for folder in sorted(path for path in (root / runs).iterdir() if path.is_dir()):
        frame = pd.read_csv(folder / "quality.csv")
        frame = frame[frame.sample_id.isin(samples)]
        frame["config"] = folder.name + "|" + frame.config_id
        quality.append(frame)
        for record in read_jsonl(folder / "generations.jsonl"):
            if record["sample_id"] not in samples:
                continue
            sample = samples[record["sample_id"]]
            config, source = folder.name + "|" + record["config_id"], record["sample_id"].rsplit("-", 1)[0]
            lines = sample["context_lines"]
            values = {value for line in lines for value in re.findall(SOURCES[sample["variant"]][1], line)}
            parsed = parse_synthetic(record["raw_text"], sample["target_keys"])
            for key, evidence in zip(sample["target_keys"], sample["evidence_positions"]):
                kind, offset = label(parsed.get(key, ""), sample["answers"][key], values, lines,
                                     lines.index(evidence["record"]))
                counts.setdefault((source, config), Counter())[kind] += 1
                if offset is not None:
                    offsets.setdefault((source, config), []).append(offset)
    quality = pd.concat(quality)
    quality["source"] = quality.sample_id.str.rsplit("-", n=1).str[0]
    sections = [f"# {runs}: exact copying of real identifiers\n",
                table(quality.pivot_table(index="config", columns="source", values="score", aggfunc="mean").reset_index())]
    rows = []
    reference = "bf16|dense" if "bf16|dense" in set(quality.config) else "bf16_none|dense"
    for source, group in quality.groupby("source"):
        prompt = group.pivot_table(index="sample_id", columns="config", values="score")
        for config in prompt.columns.drop(reference):
            difference = prompt[config] - prompt[reference]
            better, worse = int((difference > 0).sum()), int((difference < 0).sum())
            rows.append({"source": source, "config": config, "mean_pp": difference.mean(),
                         "prompts_better": better, "prompts_worse": worse, "sign_p": sign_test(better, worse)})
    sections += [f"## Paired against {reference}\n", table(pd.DataFrame(rows))]
    rows = []
    for (source, config), counter in sorted(counts.items()):
        distance = pd.Series([abs(offset) for offset in offsets.get((source, config), [])], dtype=float)
        rows.append({"source": source, "config": config, "targets": sum(counter.values()), **counter,
                     "copy_within_10_lines": int((distance <= 10).sum()), "copy_median_lines": distance.median()})
    sections += ["## Strict error classes\n", table(pd.DataFrame(rows).fillna(0))]
    text = "\n".join(sections)
    (root / report).write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
