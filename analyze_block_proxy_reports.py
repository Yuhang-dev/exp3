"""Read returned block-proxy report ZIPs and compare measured mechanisms."""

import argparse
import gzip
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", nargs="+", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    summaries, references, block_frames, coverage = [], [], [], []
    for path in args.reports:
        with ZipFile(path) as archive:
            meta = json.loads(archive.read("metadata.json"))
            panel = "LongBench" if "longbench" in meta["arguments"]["capture"] else "RULER"
            dest = args.out / panel
            dest.mkdir(exist_ok=True)
            (dest / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
            hashes = {}
            for name, expected in meta["artifacts"].items():
                digest = hashlib.sha256()
                with archive.open(name) as source:
                    for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
                        digest.update(chunk)
                hashes[name] = digest.hexdigest() == expected["sha256"]
                assert hashes[name], (path, name)
            frame = pd.read_csv(archive.open("summary.csv"))
            frame["panel"] = panel
            summaries.append(frame)
            reference = pd.read_csv(archive.open("v1_reference.csv"))
            reference["panel"] = panel
            references.append(reference)
            coverage.append({
                "panel": panel, "zip": str(path), "all_hashes_match": all(hashes.values()),
                "summary_rows": len(frame), "query_heads": sorted(frame.query_head.unique().tolist()),
                "source_samples": meta["source_samples"], "arguments": meta["arguments"],
            })
            kept = []
            with archive.open("block_stats.csv.gz") as member, gzip.GzipFile(fileobj=member) as source:
                for chunk in pd.read_csv(source, chunksize=150000):
                    kept.append(chunk[chunk.q_block_size == 128].copy())
            blocks = pd.concat(kept, ignore_index=True)
            blocks["panel"] = panel
            blocks["support_fraction"] = blocks.effective_support_mean / blocks.k_block_size
            blocks["rank_error"] = blocks.mean_proxy_rank - blocks.true_mass_rank
            blocks["missed"] = (blocks.oracle_selected == 1) & (blocks.mean_raw_selected == 0)
            blocks["hit"] = (blocks.oracle_selected == 1) & (blocks.mean_raw_selected == 1)
            blocks["sub4_gap_fraction"] = blocks.sub4_recovered_gap / blocks.mean_gap.replace(0, np.nan)
            block_frames.append(blocks)
            for name in archive.namelist():
                if name.startswith("figures/"):
                    target = dest / name
                    target.parent.mkdir(exist_ok=True)
                    target.write_bytes(archive.read(name))
            print(f"Loaded {panel}: {len(meta['source_samples'])} samples, {len(frame)} summary rows", flush=True)

    summary = pd.concat(summaries, ignore_index=True)
    reference = pd.concat(references, ignore_index=True)
    blocks = pd.concat(block_frames, ignore_index=True)
    (args.out / "coverage.json").write_text(json.dumps(coverage, indent=2), encoding="utf-8")
    base = summary[summary.method == "mean_raw"].copy()
    base["granularity_loss"] = base.token_oracle - base.row_block_oracle
    base["sharing_loss"] = base.row_block_oracle - base.tile_block_oracle
    base["proxy_loss"] = base.block_regret
    base["remote_capture_fraction"] = (
        (base.retained_mass_mean - base.fixed_mass_mean) / base.remote_mass_mean
    )
    metrics = ["retained_mass_mean", "token_oracle", "row_block_oracle", "tile_block_oracle",
               "granularity_loss", "sharing_loss", "proxy_loss", "fixed_mass_mean",
               "remote_mass_mean", "remote_capture_fraction", "mean_gap", "selected_overlap_oracle"]
    q128 = base[base.q_block_size == 128]
    q128.groupby(["panel", "layer", "k_block_size"])[metrics].mean().to_csv(args.out / "layer_blocksize.csv")
    base.groupby(["panel", "layer", "q_block_size", "k_block_size"])[metrics].mean().to_csv(args.out / "qk_blocksize.csv")
    core = q128[q128.k_block_size == 128]
    core.groupby(["panel", "sample_id", "layer"])[metrics].mean().to_csv(args.out / "sample_layer.csv")
    core.groupby(["panel", "layer", "query_head"])[metrics].mean().to_csv(args.out / "head_layer.csv")
    core.groupby(["panel", "layer"]).agg(
        configurations=("block_regret", "size"), mean_regret=("block_regret", "mean"),
        median_regret=("block_regret", "median"), p90_regret=("block_regret", lambda x: x.quantile(.9)),
        max_regret=("block_regret", "max"),
    ).to_csv(args.out / "regret_distribution.csv")
    methods = summary[(summary.q_block_size == 128) & (summary.k_block_size == 128)]
    methods.groupby(["panel", "layer", "method"])[
        ["retained_mass_mean", "block_regret", "selected_overlap_oracle"]
    ].mean().to_csv(args.out / "method_layer.csv")
    cfg = ["panel", "sample_id", "layer", "query_head", "q_tile_start"]
    wide = methods.pivot(index=cfg, columns="method", values="block_regret")
    comparisons = []
    for (panel, layer), frame in wide.groupby(level=["panel", "layer"]):
        for alternative in ["sub2_raw", "sub4_raw", "mean_balanced_remote", "mean_oracle_z", "exact_raw", "sub4_oracle_z"]:
            delta = frame.mean_raw - frame[alternative]
            comparisons.append({"panel": panel, "layer": layer, "alternative": alternative,
                                "mean_gain": delta.mean(), "improved_fraction": (delta > 1e-6).mean(),
                                "worsened_fraction": (delta < -1e-6).mean(),
                                "worst_gain": delta.min(), "best_gain": delta.max()})
    pd.DataFrame(comparisons).to_csv(args.out / "paired_method_gain.csv", index=False)
    reference.groupby(["panel", "layer"]).agg(
        selected_blocks=("selected_blocks", "mean"), retained_mean=("retained_mass_mean", "mean"),
        retained_p05=("retained_mass_mean", lambda x: x.quantile(.05)),
        worst_row=("retained_mass_min", "min"),
    ).to_csv(args.out / "v1_layer.csv")

    block_rows = []
    features = ["mean_gap", "effective_support_mean", "support_fraction", "max_minus_mean_mean",
                "sub4_gap_fraction", "query_jensen_gap"]
    for (panel, layer, size), frame in blocks.groupby(["panel", "layer", "k_block_size"]):
        for label, mask in [("all", np.ones(len(frame), dtype=bool)),
                            ("oracle", frame.oracle_selected == 1), ("hit", frame.hit), ("miss", frame.missed)]:
            values = frame.loc[mask]
            weights = values.true_mass_mean.to_numpy()
            row = {"panel": panel, "layer": layer, "k_block_size": size, "group": label, "blocks": len(values)}
            for feature in features:
                x = values[feature].to_numpy()
                row[feature + "_mean"] = np.mean(x)
                row[feature + "_median"] = np.median(x)
                row[feature + "_p90"] = np.quantile(x, .9)
                valid = np.isfinite(x)
                row[feature + "_mass_weighted"] = np.average(x[valid], weights=weights[valid]) if weights[valid].sum() else np.nan
            row["mass_sum"] = weights.sum()
            block_rows.append(row)
    pd.DataFrame(block_rows).to_csv(args.out / "block_distributions.csv", index=False)
    blocks[(blocks.k_block_size == 128) & blocks.missed].sort_values(
        "true_mass_mean", ascending=False
    ).head(30).to_csv(args.out / "largest_missed_blocks.csv", index=False)

    # Match layer coverage explicitly; do not combine unmatched layers into a panel score.
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharex=True, sharey=True)
    for row, panel in enumerate(["RULER", "LongBench"]):
        for col, layer in enumerate([0, 14, 27]):
            grouped = q128[(q128.panel == panel) & (q128.layer == layer)].groupby("k_block_size")[
                ["granularity_loss", "sharing_loss", "proxy_loss"]
            ].mean() * 100
            grouped.plot.bar(stacked=True, ax=axes[row, col], legend=False,
                             color=["#4378a8", "#6ba58d", "#d8894b"])
            axes[row, col].set_title(f"{panel}, layer {layer}")
            axes[row, col].set_xlabel("K block size")
            axes[row, col].set_ylabel("Gap to token oracle (pp)")
            axes[row, col].tick_params(axis="x", rotation=0)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, ["Block granularity", "Shared Q routing", "Mean proxy"],
               loc="upper center", bbox_to_anchor=(.5, .95), ncol=3)
    fig.suptitle("Fixed 2048 remote-token budget; Q block 128; all 28 Q heads", y=.99)
    fig.tight_layout(rect=[0, 0, 1, .89])
    fig.savefig(args.out / "01_loss_decomposition.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(12, 5), sharex=True)
    vmax = core.groupby(["panel", "layer", "query_head"]).proxy_loss.mean().max() * 100
    for ax, panel in zip(axes, ["RULER", "LongBench"]):
        heat = core[(core.panel == panel) & core.layer.isin([0, 14, 27])].pivot_table(
            index="layer", columns="query_head", values="proxy_loss", aggfunc="mean") * 100
        im = ax.imshow(heat, aspect="auto", vmin=0, vmax=vmax, cmap="magma")
        ax.set_yticks(range(len(heat.index)), heat.index)
        ax.set_xticks(range(28), range(28))
        ax.set_ylabel(f"{panel}\nLayer")
        for boundary in [6.5, 13.5, 20.5]:
            ax.axvline(boundary, color="white", linewidth=.6)
    axes[-1].set_xlabel("Query head (vertical lines separate shared KV groups)")
    fig.colorbar(im, ax=axes, label="Mean proxy regret (pp)", fraction=.025)
    fig.savefig(args.out / "02_head_regret.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    print("\nQ=K=128 loss decomposition (percentage points):")
    print(core.groupby(["panel", "layer"])[["granularity_loss", "sharing_loss", "proxy_loss", "retained_mass_mean"]].mean().mul(100).round(3).to_string())
    print("\nQ=K=128 method regret (percentage points):")
    print(methods.pivot_table(index=["panel", "layer"], columns="method", values="block_regret").mul(100).round(3).to_string())
    print("\nSaved:", args.out)


if __name__ == "__main__":
    main()
