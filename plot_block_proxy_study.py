"""Render first-pass mechanism figures from block_proxy_study.py output."""

import argparse
import gzip
import json
from pathlib import Path
from zipfile import ZipFile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result", type=Path)
    return parser.parse_args()


def main():
    root = arguments().result
    frame = pd.read_csv(root / "summary.csv")
    figures = root / "figures"
    figures.mkdir(exist_ok=True)

    base = frame[frame.method == "mean_raw"]
    decomposition = base.assign(
        granularity_loss=base.token_oracle - base.row_block_oracle,
        sharing_loss=base.row_block_oracle - base.tile_block_oracle,
        proxy_loss=base.tile_block_oracle - base.retained_mass_mean,
    )
    decomposition.to_csv(root / "decomposition.csv", index=False)
    methods = ("mean_raw", "sub2_raw", "sub4_raw")
    fig, ax = plt.subplots(figsize=(8, 5))
    for field, label in (
        ("token_oracle", "token oracle"),
        ("row_block_oracle", "per-row block oracle"),
        ("tile_block_oracle", "shared block oracle"),
    ):
        curve = base.groupby("k_block_size")[field].mean()
        ax.plot(curve.index, curve.values, marker="o", label=label)
    for method in methods:
        curve = frame[frame.method == method].groupby(
            "k_block_size"
        ).retained_mass_mean.mean()
        ax.plot(curve.index, curve.values, marker="o", label=method)
    ax.set_xlabel("K block size")
    ax.set_ylabel("Mean retained attention mass")
    ax.set_title("Fixed exact-token budget; protected region held fixed")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(figures / "01_block_size_oracles.png", dpi=180)
    plt.close(fig)

    pieces = decomposition.groupby("k_block_size")[
        ["granularity_loss", "sharing_loss", "proxy_loss"]
    ].mean()
    fig, ax = plt.subplots(figsize=(8, 4))
    bottom = np.zeros(len(pieces))
    for field, label in (
        ("granularity_loss", "Full-block granularity"),
        ("sharing_loss", "Q-tile sharing"),
        ("proxy_loss", "Mean-key routing"),
    ):
        values = pieces[field].values * 100
        ax.bar(pieces.index.astype(str), values, bottom=bottom, label=label)
        bottom += values
    ax.set_xlabel("K block size")
    ax.set_ylabel("Mass gap to token oracle (pp)")
    ax.set_title("Attributing block-sparse selection loss")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figures / "01b_oracle_decomposition.png", dpi=180)
    plt.close(fig)

    regret = base.pivot_table(
        index="q_block_size", columns="k_block_size",
        values="block_regret", aggfunc="mean",
    ).sort_index().sort_index(axis=1)
    fig, ax = plt.subplots(figsize=(7, 4))
    image = ax.imshow(regret.values * 100, aspect="auto", cmap="magma")
    ax.set_xticks(np.arange(len(regret.columns)), regret.columns)
    ax.set_yticks(np.arange(len(regret.index)), regret.index)
    ax.set_xlabel("K block size")
    ax.set_ylabel("Q block size")
    ax.set_title("Mean proxy regret against shared-block oracle (pp)")
    for row in range(len(regret.index)):
        for col in range(len(regret.columns)):
            ax.text(
                col, row, f"{regret.iloc[row, col] * 100:.2f}",
                ha="center", va="center", color="white", fontsize=8,
            )
    fig.colorbar(image, ax=ax, label="Mass percentage points")
    fig.tight_layout()
    fig.savefig(figures / "02_qk_block_size_regret.png", dpi=180)
    plt.close(fig)

    samples = []
    for chunk in pd.read_csv(
        root / "block_stats.csv.gz",
        usecols=(
            "k_block_size", "mean_proxy_rank", "true_mass_rank",
            "mean_gap", "true_mass_mean", "mean_raw_selected",
            "oracle_selected", "query_jensen_gap",
        ),
        chunksize=100000,
    ):
        chosen = chunk[chunk.k_block_size == 128]
        if len(chosen):
            samples.append(
                chosen.sample(
                    min(len(chosen), 400), random_state=len(samples) + 31
                )
            )
    if samples:
        points = pd.concat(samples, ignore_index=True)
        rank_error = points.mean_proxy_rank - points.true_mass_rank
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        axes[0].scatter(
            points.mean_gap, rank_error, s=5, alpha=0.25,
        )
        axes[0].axhline(0, color="black", linewidth=0.8)
        axes[0].set_xlabel("Mean K-block Jensen gap")
        axes[0].set_ylabel("Mean rank − true-mass rank")
        axes[0].set_title("A large gap alone need not change rank")
        axes[1].scatter(
            points.query_jensen_gap, rank_error, s=5, alpha=0.25,
        )
        axes[1].axhline(0, color="black", linewidth=0.8)
        axes[1].set_xlabel("Q-tile Jensen gap")
        axes[1].set_ylabel("Mean rank − true-mass rank")
        axes[1].set_title("Query demand varies within a tile")
        fig.tight_layout()
        fig.savefig(figures / "03_gap_and_rank.png", dpi=180)
        plt.close(fig)

    with gzip.open(
        root / "miss_profiles.jsonl.gz", "rt", encoding="utf-8"
    ) as source:
        profiles = [json.loads(line) for _, line in zip(range(8), source)]
    if profiles:
        fig, axes = plt.subplots(
            len(profiles), 1, figsize=(10, 1.7 * len(profiles)),
            squeeze=False,
        )
        for index, profile in enumerate(profiles):
            ax = axes[index, 0]
            ax.plot(profile["centered_token_logits"], linewidth=1)
            ax.axhline(0, color="black", linewidth=0.6)
            ax.set_ylabel("s − mean")
            ax.set_title(
                f"{profile['sample_id']} layer {profile['layer']} "
                f"head {profile['query_head']} "
                f"K {profile['token_start']}:"
                f"{profile['token_start'] + profile['k_block_size']} "
                f"gap={profile['jensen_gap']:.2f}",
                fontsize=8,
            )
        axes[-1, 0].set_xlabel("Token offset within missed block")
        fig.tight_layout()
        fig.savefig(figures / "04_missed_block_profiles.png", dpi=180)
        plt.close(fig)

    with ZipFile(root / "block_proxy_report.zip", "w") as archive:
        for name in (
            "summary.csv", "v1_reference.csv", "block_stats.csv.gz", "row_block_stats.csv.gz",
            "miss_profiles.jsonl.gz", "metadata.json", "decomposition.csv",
        ):
            archive.write(root / name, name)
        for figure in sorted(figures.glob("*.png")):
            archive.write(figure, f"figures/{figure.name}")

    print(f"[proxy-plots] {figures}")


if __name__ == "__main__":
    main()
