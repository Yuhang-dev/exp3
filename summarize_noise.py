"""Write NOISE.md: is the compression loss of the gold log-odds (delta) the convexity gain of score noise?

At each target's retrieval head (the fixed head with the largest BF16 gold log-odds), delta = rest_lse - gold_lse
splits exactly into a systematic part (weighted mean change of the other lines' scores minus the gold line's)
and two convexity gains (log sum w e^eta minus its mean). For Gaussian score noise of variance sigma^2 the
other lines' gain is sigma^2 / 2 once enough of them share the attention (large effective number n_eff).
Also: median delta per method against its teacher-forced accuracy drop (and the implied effective-length factor
e^delta), and the attention profile over the ten lines on either side of the gold line.
Usage: summarize_noise.py FOLDER HEADS (copy_mechanism.py output with --noise-heads HEADS)."""

from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd

from summarize_mechanism import auc
from summarize_pointer_study import read_jsonl, table


LENGTHS = {"len8k": "8K", "len16k": "16K", "copy": "32K"}
ORDER = ["8K", "16K", "32K"]


def spearman(a, b):
    return np.corrcoef(pd.Series(a).rank(), pd.Series(b).rank())[0, 1]


def main():
    folder = Path(sys.argv[1])
    rows = pd.DataFrame(read_jsonl(folder / "targets.jsonl"))
    rows["target"] = rows.sample_id + "|" + rows.key
    rows["length"] = rows.sample_id.str.split("-").str[0].map(LENGTHS)
    base = rows[rows.condition == "bf16"].set_index("target")
    heads = [tuple(map(int, re.fullmatch(r"L(\d+)H(\d+)", name).groups())) for name in sys.argv[2].split(",")]
    gold = np.array(base.gold_head_mass.tolist())
    context = np.array(base.context_head_mass.tolist())
    first = 28 - gold.shape[1]
    log_odds = np.stack([np.log(gold[:, layer - first, h]) - np.log(context[:, layer - first, h] - gold[:, layer - first, h])
                         for layer, h in heads], axis=1)
    best = dict(zip(base.index, log_odds.argmax(1)))
    head = rows.target.map(best).values
    pick = lambda column: np.array([values[h] for values, h in zip(rows[column], head)])
    for name in ("rest_lse", "gold_lse", "rest_mean", "gold_mean", "rest_var", "gold_var", "rest_neff",
                 "rest_var_unweighted"):
        rows[name] = pick(f"noise_{name}")
    rows["delta"] = rows.rest_lse - rows.gold_lse
    rows["systematic"] = rows.rest_mean - rows.gold_mean
    rows["rest_gain"] = rows.rest_lse - rows.rest_mean
    rows["gold_gain"] = rows.gold_lse - rows.gold_mean
    rows["half_var"] = rows.rest_var / 2
    rows["bf16_correct"] = rows.target.map(base.teacher_forced_correct).astype(bool)
    rows["flipped"] = rows.bf16_correct & ~rows.teacher_forced_correct
    compressed = rows[(rows.condition != "bf16") & rows.bf16_correct]

    sections = [f"# Score noise and the gold log-odds loss ({rows.sample_id.nunique()} prompts, {len(base)} targets)\n",
                "delta = systematic + rest_gain - gold_gain exactly; the log-n picture predicts rest_gain ~ "
                "half_var (sigma^2 / 2) when n_eff is large. Medians over targets BF16 copies correctly.\n"]
    split = compressed.groupby(["condition", "length"]).agg(
        targets=("delta", "size"), flips=("flipped", "sum"), delta=("delta", "median"),
        systematic=("systematic", "median"), rest_gain=("rest_gain", "median"), half_var=("half_var", "median"),
        gold_gain=("gold_gain", "median"), rest_neff=("rest_neff", "median"), mean_delta=("delta", "mean"),
        mean_systematic=("systematic", "mean"), mean_rest_gain=("rest_gain", "mean"),
        mean_gold_gain=("gold_gain", "mean")).reset_index()
    split["length"] = pd.Categorical(split.length, ORDER)
    sections += ["## 1. Decomposition of delta\n", table(split.sort_values(["condition", "length"]))]

    checks = []
    for condition, frame in compressed.groupby("condition"):
        checks.append({
            "condition": condition, "targets": len(frame), "flips": int(frame.flipped.sum()),
            "share of mean delta from rest_gain": frame.rest_gain.mean() / frame.delta.mean(),
            "share from systematic": frame.systematic.mean() / frame.delta.mean(),
            "rest_gain / half_var (median)": (frame.rest_gain / frame.half_var).median(),
            "spearman(delta, half_var)": spearman(frame.delta, frame.half_var),
            "spearman(delta, systematic)": spearman(frame.delta, frame.systematic),
            "spearman(rest_neff, log n)": spearman(frame.rest_neff, np.log(frame.context_tokens)),
            "AUC delta": auc(frame.delta.values, frame.flipped.values),
            "AUC half_var": auc(frame.half_var.values, frame.flipped.values),
            "AUC systematic": auc(frame.systematic.values, frame.flipped.values),
        })
    sections += ["## 2. Does the convexity gain explain delta?\n", table(pd.DataFrame(checks))]

    ranking = []
    accuracy = rows.groupby(["condition", "length"]).teacher_forced_correct.mean().unstack() * 100
    for condition, frame in compressed.groupby("condition"):
        long = frame[frame.length == "32K"]
        ranking.append({"condition": condition, "median delta (all)": frame.delta.median(),
                        "median delta 32K": long.delta.median(),
                        "effective length factor e^-delta (32K)": np.exp(-long.delta.median()),
                        "teacher-forced accuracy 32K": accuracy.loc[condition, "32K"],
                        "drop vs BF16 32K": accuracy.loc["bf16", "32K"] - accuracy.loc[condition, "32K"],
                        "flip rate 32K (%)": long.flipped.mean() * 100})
    ranking = pd.DataFrame(ranking).sort_values("median delta 32K")
    sections += ["## 3. Methods ranked by delta\n", table(ranking),
                 f"Spearman over methods, median delta 32K vs flip rate 32K: "
                 f"{spearman(ranking['median delta 32K'], ranking['flip rate 32K (%)']):.2f}\n"]

    shapes = []
    for (condition, length), frame in rows[rows.bf16_correct].groupby(["condition", "length"]):
        aligned = []
        for values, h, c in zip(frame.line_profile, frame.target.map(best), frame.profile_center):
            line = np.full(2 * 10 + 1, np.nan)
            line[10 - c:10 - c + len(values[h])] = values[h]
            aligned.append(line)
        shapes.append({"condition": condition, "length": length,
                       **{f"{offset:+d}": value for offset, value in
                          zip(range(-10, 11), np.nanmean(np.array(aligned), axis=0))}})
    shapes = pd.DataFrame(shapes)
    shapes["length"] = pd.Categorical(shapes.length, ORDER)
    shapes = shapes.sort_values(["length", "condition"])
    sections += ["## 4. Attention profile around the gold line (mean mass per line at the target's retrieval head; "
                 "offset 0 is the gold line)\n", table(shapes[["condition", "length"] +
                                                            [f"{o:+d}" for o in range(-5, 6)]], digits=4)]
    shapes.to_csv(folder / "line_profile.csv", index=False)
    compressed.drop(columns=[c for c in compressed.columns if c.startswith("noise_") or c in
                             ("line_profile", "gold_head_mass", "context_head_mass", "gold_line_mass",
                              "rival_line_mass", "neighbour_mass", "gold_line_head_max", "rival_line_head_max",
                              "attended_offset")]).to_csv(folder / "delta_targets.csv", index=False)
    text = "\n".join(sections)
    (folder / "NOISE.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
