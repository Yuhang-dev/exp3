"""Figures for the noise run: (1) gold log-odds against context length per condition, (2) the decomposition of
delta at 32K into systematic shift and convexity gain, (3) attention around the gold line at 32K.
Usage: plot_noise.py FOLDER HEADS."""

import json
from pathlib import Path
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


LENGTHS = {"len8k": 8, "len16k": 16, "copy": 32}
SHOWN = [("bf16", "BF16 (dense)", "#2a78d6", "o-"), ("nf4sim:all", "NF4", "#eb6834", "^--"),
         ("nf4sim:!attn", "NF4 + BF16 attention (repair)", "#1baf7a", "s-"),
         ("wanda:all", "Wanda 50%", "#6250d6", "D:")]


def main():
    folder, heads = Path(sys.argv[1]), sys.argv[2].split(",")
    rows = pd.DataFrame([json.loads(line) for line in open(folder / "targets.jsonl")])
    rows["target"] = rows.sample_id + "|" + rows.key
    rows["length"] = rows.sample_id.str.split("-").str[0].map(LENGTHS)
    gold = np.array(rows.gold_head_mass.tolist())
    context = np.array(rows.context_head_mass.tolist())
    first = 28 - gold.shape[1]
    pairs = [tuple(map(int, re.fullmatch(r"L(\d+)H(\d+)", name).groups())) for name in heads]
    log_odds = np.stack([np.log(gold[:, l - first, h]) - np.log(context[:, l - first, h] - gold[:, l - first, h])
                         for l, h in pairs], axis=1)
    base = rows.condition == "bf16"
    best = dict(zip(rows.target[base], log_odds[base.values].argmax(1)))
    rows["lo"] = log_odds[np.arange(len(rows)), rows.target.map(best).values]
    correct = dict(zip(rows.target[base], rows.teacher_forced_correct[base]))
    rows = rows[rows.target.map(correct)]

    figure, axis = plt.subplots(figsize=(5.2, 3.6))
    for condition, label, colour, style in SHOWN:
        part = rows[rows.condition == condition].groupby("length").lo.median()
        axis.plot(np.log(part.index * 1024 * 0.94), part.values, style, color=colour, label=label, lw=2, ms=6)
    axis.axhline(0, color="#898781", lw=1, ls=":")
    axis.set_xticks(np.log(np.array([8, 16, 32]) * 1024 * 0.94), ["8K", "16K", "32K"])
    axis.set_xlabel("context length (log scale)")
    axis.set_ylabel("gold log-odds at retrieval head (median)")
    axis.legend(frameon=False, fontsize=8)
    figure.tight_layout()
    figure.savefig(folder / "fig_advantage_vs_length.png", dpi=200)

    delta = pd.read_csv(folder / "delta_targets.csv")
    delta = delta[delta.length == "32K"]
    order = ["nf4sim:!attn", "nf4sim:!qk", "nf4sim:all", "rtn4:all", "wanda:all", "fp_v1:all"]
    names = ["NF4+BF16 attn", "NF4+BF16 Q,K", "NF4", "RTN4", "Wanda", "FlashPrefill"]
    means = delta.groupby("condition")[["systematic", "rest_gain", "gold_gain", "half_var", "delta"]].mean().loc[order]
    figure, axis = plt.subplots(figsize=(6, 3.6))
    x = np.arange(len(order))
    axis.bar(x - 0.2, means.rest_gain - means.gold_gain, 0.4, color="#eb6834", label="convexity gain (noise)")
    axis.bar(x + 0.2, means.systematic, 0.4, color="#2a78d6", label="systematic shift")
    axis.plot(x, means.half_var, "k_", ms=18, mew=2, label="sigma^2 / 2 (weighted)")
    axis.plot(x, means.delta, "o", color="#e34948", ms=7, label="delta (total)")
    axis.axhline(0, color="#898781", lw=1)
    axis.set_xticks(x, names, rotation=20, fontsize=8)
    axis.set_ylabel("loss of gold log-odds at 32K (mean)")
    axis.legend(frameon=False, fontsize=8)
    figure.tight_layout()
    figure.savefig(folder / "fig_delta_decomposition.png", dpi=200)

    profile = pd.read_csv(folder / "line_profile.csv")
    profile = profile[profile.length == "32K"].set_index("condition")
    offsets = [f"{o:+d}" for o in range(-5, 6)]
    figure, axes = plt.subplots(1, 2, figsize=(9, 3.4), gridspec_kw={"width_ratios": [1.3, 1]})
    for condition, label, colour, style in SHOWN:
        axes[0].plot(range(-5, 6), profile.loc[condition, offsets].values, style, color=colour, label=label, lw=1.8, ms=5)
        line = rows[rows.condition == condition]
        axes[1].plot([8, 16, 32], [1 / (1 + np.exp(-line[line.length == n].lo.median())) * 100 for n in (8, 16, 32)],
                     style, color=colour, lw=1.8, ms=5)
    axes[0].set_xlabel("line offset from the gold line (32K)")
    axes[0].set_ylabel("attention mass per line")
    axes[0].legend(frameon=False, fontsize=8)
    axes[1].set_xscale("log", base=2)
    axes[1].set_xticks([8, 16, 32], ["8K", "16K", "32K"])
    axes[1].set_ylabel("gold share among context lines (%)")
    axes[1].set_xlabel("context length")
    figure.tight_layout()
    figure.savefig(folder / "fig_line_profile.png", dpi=200)
    print("figures written")


if __name__ == "__main__":
    main()
