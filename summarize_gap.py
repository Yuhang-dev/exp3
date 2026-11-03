"""Write GAP.md: does the gold line's attention advantage behave as in the log-n scaling picture?

For each target, head and retrieval layer at the divergence step, with g the attention mass on the gold line and
c the mass on all context lines, the gold log-odds against the other lines is lo = log g - log(c - g) and the
length-free score gap is gap = lo + log(n_other), n_other being the number of other context-line tokens: if the
scores of competing tokens do not depend on the length, gap stays constant while lo falls by log n. Per target we
use the retrieval head with the largest BF16 log-odds. Tests: (1) BF16 lo falls with log n while gap does not;
(2) compression lowers gap by an amount that does not depend on length; (3) after conditioning on BF16 lo,
length adds nothing to the chance of a flip, i.e. longer contexts hurt only by pushing targets toward the threshold.
Usage: summarize_gap.py FOLDER (copy_mechanism.py output with per-head masses)."""

import math
from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd

from summarize_mechanism import auc
from summarize_predecode import bootstrap
from summarize_pointer_study import read_jsonl, table


HEADS = 8
LENGTHS = {"len8k": "8K", "len16k": "16K", "copy": "32K"}


def logistic(features, label, steps=50):
    """Newton fit with an intercept; returns coefficients and log-likelihood."""
    x = np.column_stack([np.ones(len(label)), *features])
    w = np.zeros(x.shape[1])
    for _ in range(steps):
        p = 1 / (1 + np.exp(-x @ w))
        hessian = x.T @ (x * (p * (1 - p))[:, None]) + 1e-6 * np.eye(len(w))
        w += np.linalg.solve(hessian, x.T @ (label - p))
    p = np.clip(1 / (1 + np.exp(-x @ w)), 1e-12, 1 - 1e-12)
    return w, float(np.sum(label * np.log(p) + (1 - label) * np.log(1 - p)))


def bootstrap_median(values, prompts, draws=2000, seed=0):
    rng = np.random.default_rng(seed)
    names = np.unique(prompts)
    members = {name: np.flatnonzero(prompts == name) for name in names}
    medians = [np.median(values[np.concatenate([members[n] for n in rng.choice(names, len(names))])])
               for _ in range(draws)]
    return np.percentile(medians, [2.5, 97.5])


def main():
    folder = Path(sys.argv[1])
    rows = pd.DataFrame(read_jsonl(folder / "targets.jsonl"))
    rows["target"] = rows.sample_id + "|" + rows.key
    rows["length"] = rows.sample_id.str.split("-").str[0].map(LENGTHS)
    gold = np.array(rows.gold_head_mass.tolist())
    context = np.array(rows.context_head_mass.tolist())
    other_tokens = (rows.context_tokens - rows.gold_tokens).values
    log_odds = np.log(np.clip(gold, 1e-12, None)) - np.log(np.clip(context - gold, 1e-12, None))
    first = 28 - gold.shape[1]
    is_bf16 = (rows.condition == "bf16").values
    if len(sys.argv) > 2:
        heads = [(int(layer) - first, int(head)) for layer, head in
                 (re.fullmatch(r"L(\d+)H(\d+)", name).groups() for name in sys.argv[2].split(","))]
        ranking = np.array([layer * gold.shape[2] + head for layer, head in heads])
    else:
        ranking = np.argsort(-gold[is_bf16].mean(0), axis=None)[:HEADS]
        heads = [divmod(int(index), gold.shape[2]) for index in ranking]
    flat = log_odds.reshape(len(rows), -1)[:, ranking]
    head_of = dict(zip(rows.target[is_bf16], np.argmax(flat[is_bf16], axis=1)))
    rows["lo"] = flat[np.arange(len(rows)), rows.target.map(head_of).values]
    rows["log_n"] = np.log(other_tokens)
    rows["gap"] = rows.lo + rows.log_n
    reference = rows[is_bf16].set_index("target")
    for column in ("lo", "gap", "teacher_forced_correct", "context_nll_mean", "divergence_margin"):
        rows[f"bf16_{column}"] = rows.target.map(reference[column])
    rows["gap_drop"] = rows.bf16_gap - rows.gap
    rows["context_nll_rise"] = rows.context_nll_mean - rows.bf16_context_nll_mean
    rows["flipped"] = rows.bf16_teacher_forced_correct & ~rows.teacher_forced_correct
    order = ["8K", "16K", "32K"]

    sections = [f"# Gold log-odds and the log-n picture ({rows.sample_id.nunique()} prompts, "
                f"{len(reference)} targets)\n",
                ("Retrieval heads fixed in advance: " if len(sys.argv) > 2 else
                 "Retrieval heads (layer, head), top by BF16 gold mass at the divergence step: ")
                + ", ".join(f"L{first + layer}H{head}" for layer, head in heads) + ". Per target the head with the "
                "largest BF16 log-odds is used throughout.\n"]

    base = rows[is_bf16]
    by_length = base.groupby(["source", "length"]).agg(
        targets=("lo", "size"), teacher_forced_accuracy=("teacher_forced_correct", "mean"),
        median_log_n=("log_n", "median"), median_lo=("lo", "median"), median_gap=("gap", "median"))
    by_length["teacher_forced_accuracy"] *= 100
    logs = base[base.source == "hdfs"]
    slope_lo = np.polyfit(logs.log_n, logs.lo, 1)[0]
    slope_gap = np.polyfit(logs.log_n, logs.gap, 1)[0]
    sections += ["## 1. BF16: log-odds against length\n", table(by_length.reset_index()),
                 f"Least-squares slope over HDFS targets: d(lo)/d(log n) = {slope_lo:.2f} (log-n picture: -1), "
                 f"d(gap)/d(log n) = {slope_gap:.2f} (log-n picture: 0). BF16 accuracy against BF16 lo: AUC "
                 f"{auc(base.lo.values, base.teacher_forced_correct.values):.2f}.\n"]

    drops, tests, bins, aucs = [], [], [], []
    for condition, frame in rows[~is_bf16].groupby("condition"):
        frame = frame[frame.bf16_teacher_forced_correct.astype(bool)]
        label = frame.flipped.values.astype(float)
        weights, only_lo = logistic([frame.bf16_lo.values], label)
        expected = 1 / (1 + np.exp(-(weights[0] + weights[1] * frame.bf16_lo.values)))
        for length in order:
            inside = (frame.length == length).values
            part = frame[inside]
            low, high = bootstrap_median(part.gap_drop.values, part.sample_id.values)
            q75, q90 = np.quantile(part.gap_drop, [0.75, 0.9])
            drops.append({"condition": condition, "length": length, "targets": len(part),
                          "flips": int(part.flipped.sum()), "flips expected from bf16 lo": expected[inside].sum(),
                          "median_gap_drop": part.gap_drop.median(), "95% interval": f"[{low:.2f}, {high:.2f}]",
                          "q75 drop": q75, "q90 drop": q90, "mean drop": part.gap_drop.mean(),
                          "median_bf16_lo": part.bf16_lo.median(), "median_lo": part.lo.median()})
        _, with_length = logistic([frame.bf16_lo.values, frame.log_n.values], label)
        _, only_length = logistic([frame.log_n.values], label)
        weights, _ = logistic([frame.bf16_lo.values, frame.log_n.values], label)
        _, null = logistic([], label)
        _, only_compressed = logistic([frame.lo.values], label)
        _, compressed_length = logistic([frame.lo.values, frame.log_n.values], label)
        statistic = 2 * (with_length - only_lo)
        flipped, prompts = frame.flipped.values, frame.sample_id.values
        scores = {"bf16 lo (low = risky)": -frame.bf16_lo.values, "compressed lo (low = risky)": -frame.lo.values,
                  "context KL": frame.context_kl_mean.values, "context NLL rise": frame.context_nll_rise.values,
                  "bf16 divergence margin (low = risky)": -frame.bf16_divergence_margin.values}
        aucs.append({"condition": condition, "targets": len(frame), "flips": int(label.sum()),
                     **{name: f"{auc(score, flipped):.2f} [{low:.2f}, {high:.2f}]" for name, score in scores.items()
                        for low, high in [bootstrap(score, flipped, prompts)]},
                     **{f"compressed lo, {source}": f"{auc(-part.lo.values, part.flipped.values):.2f}"
                        for source, part in frame.groupby("source")}})
        tests.append({
            "condition": condition, "targets": len(frame), "flips": int(label.sum()),
            "AUC bf16 lo (low = risky)": auc(-frame.bf16_lo.values, frame.flipped.values),
            "AUC compressed lo": auc(-frame.lo.values, frame.flipped.values),
            "AUC gap drop": auc(frame.gap_drop.values, frame.flipped.values),
            "AUC log n": auc(frame.log_n.values, frame.flipped.values),
            "length alone: LR p": math.erfc(math.sqrt(max(2 * (only_length - null), 0) / 2)),
            "length given bf16 lo: LR p": math.erfc(math.sqrt(max(statistic, 0) / 2)),
            "length given compressed lo: LR p": math.erfc(
                math.sqrt(max(2 * (compressed_length - only_compressed), 0) / 2)),
            "coef bf16 lo": weights[1], "coef log n": weights[2],
            "gap drop vs log n slope": np.polyfit(frame.log_n, frame.gap_drop, 1)[0],
        })
        edges = np.quantile(base.lo, [0, 0.25, 0.5, 0.75, 1])
        frame = frame.assign(bin=pd.cut(frame.bf16_lo, edges, include_lowest=True))
        for (interval, length), part in frame.groupby(["bin", "length"], observed=True):
            bins.append({"condition": condition, "bf16 lo bin": str(interval), "length": length,
                         "targets": len(part), "flip rate": part.flipped.mean() * 100})
    sections += ["## Flip prediction (AUC [95% prompt-bootstrap interval]; targets BF16 gets right)\n",
                 table(pd.DataFrame(aucs).T.reset_index()),
                 "## 2. Compression: drop of the length-free gap (BF16-correct targets; interval resamples prompts; "
                 "expected flips from a logistic model on BF16 log-odds pooled over lengths)\n",
                 table(pd.DataFrame(drops)),
                 "## 3. Is the length effect explained by the BF16 log-odds?\n",
                 "'length given bf16 lo' is the likelihood-ratio test of adding log n to a logistic model of the flip "
                 "on BF16 log-odds; a large p means length matters only through the log-odds.\n",
                 table(pd.DataFrame(tests)),
                 "### Flip rate (%) by BF16 log-odds quartile and length\n",
                 table(pd.DataFrame(bins).pivot_table(index=["condition", "bf16 lo bin"], columns="length",
                                                      values="flip rate").reindex(columns=order).reset_index())]
    text = "\n".join(sections)
    (folder / "GAP.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
