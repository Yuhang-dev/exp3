"""Turn block_proxy_validation.py raw tables into the spec deliverables.

Writes e1-e8 CSVs, figures, summary.json (numbers for Q1-Q6 and every
pre-registered prediction), and a ZIP of the whole directory.
Units: attention-mass percentage points (pp), relative output error, never task accuracy.
"""

import argparse
import json
import zipfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


TILE = ["panel", "sample_id", "layer", "query_head", "q_start"]
NAMED_CASES = [  # (panel, sample_id, layer, head, block token_start)
    ("RULER", "ruler_niah_mq:32768:59", 14, 7, 9984),
    ("LongBench", "longbench-v2-66ebc0c95a08c7b9b35de7f3", 14, 7, 11008),
    ("LongBench", "longbench-v2-66ebc0c95a08c7b9b35de7f3", 14, 7, 13184),
]
GROUPS = ["hit", "missed", "false_pos", "boundary"]
RNG = np.random.default_rng(20260924)


def read(out, name):
    return pd.read_csv(out / "raw" / f"{name}.csv.gz")


def spearman(a, b):
    return a.rank().corr(b.rank())


def group_masks(frame):
    return {
        "hit": (frame.in_oracle == 1) & (frame.in_mean == 1),
        "missed": (frame.in_oracle == 1) & (frame.in_mean == 0),
        "false_pos": (frame.in_oracle == 0) & (frame.in_mean == 1),
        "boundary": frame.boundary == 1,
    }


def bootstrap_difference(frame, metric, first, second, reps=2000):
    """Tile-resampled CI of mean(metric|first) - mean(metric|second)."""
    tiles = frame.groupby(TILE, sort=False).ngroup().to_numpy()
    count = tiles.max() + 1
    x = frame[metric].to_numpy()
    sums = [np.bincount(tiles, weights=np.where(m, x, 0), minlength=count) for m in (first, second)]
    counts = [np.bincount(tiles, weights=m.astype(float), minlength=count) for m in (first, second)]
    draws = RNG.integers(0, count, size=(reps, count))
    values = []
    for draw in draws:
        a = sums[0][draw].sum() / counts[0][draw].sum()
        b = sums[1][draw].sum() / counts[1][draw].sum()
        values.append(a - b)
    point = x[first].mean() - x[second].mean()
    low, high = np.nanquantile(values, [0.025, 0.975])
    ranked = pd.Series(np.concatenate([x[first], x[second]])).rank().to_numpy()
    n1, n2 = first.sum(), second.sum()
    u = ranked[:n1].sum() - n1 * (n1 + 1) / 2
    cliff = 2 * u / (n1 * n2) - 1
    return point, low, high, cliff


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    out = args.out
    figures = out / "figures"
    figures.mkdir(exist_ok=True)
    summary = {
        "e0": json.loads((out / "e0_reproduction.json").read_text()),
        "checks": json.loads((out / "checks.json").read_text()),
    }
    rope_failed = {
        (item["panel"], item["sample_id"], item["layer"])
        for item in summary["checks"]["rope"] if not item["passed"]
    }
    summary["excluded_for_rope_check"] = sorted(map(list, rope_failed))

    # ------------------------------------------------------------ E1
    methods = read(out, "methods")
    k128 = methods[methods.k_block_size == 128]
    e1 = k128[k128.method.isin(["tile_block_oracle", "mean_raw", "exact_raw", "mean_var2"])]
    e1_table = e1.groupby(["panel", "layer", "method"]).agg(
        configs=("block_regret", "size"),
        retained_mass_mean=("retained_mass_mean", "mean"),
        block_regret_pp=("block_regret", lambda x: 100 * x.mean()),
        selected_overlap_oracle=("selected_overlap_oracle", "mean"),
    ).reset_index()
    sweep = methods[(methods.layer == 14) & methods.method.isin(["mean_raw", "exact_raw", "mean_var2"])]
    sweep_table = sweep.groupby(["panel", "layer", "k_block_size", "method"]).agg(
        configs=("block_regret", "size"), block_regret_pp=("block_regret", lambda x: 100 * x.mean()),
    ).reset_index()
    pd.concat([e1_table.assign(table="k128_all_layers"), sweep_table.assign(table="l14_k_sweep")]).to_csv(
        out / "e1_second_order.csv", index=False
    )

    blocks = read(out, "blocks")
    rows = []
    for (panel, layer), frame in blocks.groupby(["panel", "layer"]):
        local = group_masks(frame)
        for name in GROUPS:
            part = frame[local[name]]
            rows.append({
                "panel": panel, "layer": layer, "group": name, "blocks": len(part),
                **{f"median_abs_err_K{n}": part[f"cum_err_k{n}"].median() for n in range(1, 5)},
                "k4_worse_than_k2_row_frac": part.k4_worse_than_k2_frac.mean(),
                "k4_worse_than_k2_block_frac": (part.cum_err_k4 > part.cum_err_k2).mean(),
                "half_var_over_gap_block_frac": (0.5 * part["var"] > part.gap).mean(),
                "half_var_over_gap_row_frac": part.half_var_over_gap_frac.mean(),
                "median_gap": part.gap.median(), "median_var": part["var"].median(),
                "median_delta_eff": part.delta_eff.median(), "median_n_eff": part.n_eff.median(),
                "median_top1_share": part.top1_share.median(),
            })
    e1_cumulants = pd.DataFrame(rows)
    e1_cumulants.to_csv(out / "e1_cumulants.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    order = ["mean_raw", "mean_var2", "exact_raw"]
    for ax, panel in zip(axes, ["RULER", "LongBench"]):
        table = e1_table[(e1_table.panel == panel) & e1_table.method.isin(order)].pivot(
            index="layer", columns="method", values="block_regret_pp")[order]
        table.plot.bar(ax=ax, color=["#d8894b", "#8a6bb0", "#4378a8"])
        ax.set_title(f"{panel}: regret vs shared-block oracle (Q=K=128)")
        ax.set_ylabel("Attention-mass regret (pp)")
        ax.tick_params(axis="x", rotation=0)
    fig.tight_layout()
    fig.savefig(figures / "e1_regret_by_method.png", dpi=170)
    plt.close(fig)

    l14 = e1_table[e1_table.layer == 14].pivot(index="panel", columns="method", values="block_regret_pp")
    summary["q1"] = {
        "l14_regret_pp": l14.to_dict(orient="index"),
        "prediction_mean_var2_minus_exact_gt_5pp": {
            panel: bool(row.mean_var2 - row.exact_raw > 5) for panel, row in l14.iterrows()
        },
        "missed_vs_hit_k4_worse_than_k2_block_frac": e1_cumulants[
            e1_cumulants.group.isin(["hit", "missed"])
        ].pivot_table(index=["panel", "layer"], columns="group", values="k4_worse_than_k2_block_frac").to_dict(orient="index"),
    }

    # ------------------------------------------------------------ E2
    rows, tests = [], []
    for (panel, layer), frame in blocks.groupby(["panel", "layer"]):
        local = group_masks(frame)
        for ref in ("cb", "cm"):
            for name in GROUPS:
                part = frame[local[name]]
                rows.append({
                    "panel": panel, "layer": layer, "reference": ref, "group": name, "blocks": len(part),
                    **{
                        f"{stat}_{metric}": getattr(part[f"{ref}_{metric}"], stat)()
                        for metric in ("a_plus", "a_minus", "a_plus_top1", "mu_minus_c", "mu_minus1_minus_c")
                        for stat in ("median", "mean")
                    },
                    "median_top1_share": part.top1_share.median(),
                })
            for other in ("false_pos", "boundary"):
                second = local[other] & ~local["missed"]
                for metric in ("a_minus", "a_plus", "a_plus_top1"):
                    column = f"{ref}_{metric}"
                    valid = frame[column].notna().to_numpy()
                    point, low, high, cliff = bootstrap_difference(
                        frame[valid], column, local["missed"].to_numpy()[valid], second.to_numpy()[valid]
                    )
                    tests.append({
                        "panel": panel, "layer": layer, "reference": ref, "metric": metric,
                        "comparison": f"missed-{other}", "mean_difference": point,
                        "ci95_low": low, "ci95_high": high, "cliffs_delta": cliff,
                    })
    e2 = pd.DataFrame(rows)
    e2_tests = pd.DataFrame(tests)
    e2.merge(e2_tests.pivot_table(
        index=["panel", "layer", "reference"], columns=["metric", "comparison"],
        values=["mean_difference", "ci95_low", "ci95_high", "cliffs_delta"],
    ).pipe(lambda x: x.set_axis(["__".join(c) for c in x.columns], axis=1)).reset_index(),
        on=["panel", "layer", "reference"]).to_csv(out / "e2_dilution_cancellation.csv", index=False)
    e2_tests.to_csv(out / "e2_tests.csv", index=False)
    verdicts = {}
    for (panel, layer, ref), frame in e2_tests.groupby(["panel", "layer", "reference"]):
        minus = frame[frame.metric == "a_minus"]
        includes_zero = ((minus.ci95_low <= 0) & (minus.ci95_high >= 0)).all()
        larger = (minus.ci95_low > 0).all()
        top1 = e2[(e2.panel == panel) & (e2.layer == layer) & (e2.reference == ref) & (e2.group == "missed")]
        top1_share = float(top1.median_a_plus_top1.iloc[0])
        verdicts[f"{panel}/L{layer}/{ref}"] = {
            "a_minus_missed_vs_competitors_ci_includes_zero": bool(includes_zero),
            "a_minus_missed_larger": bool(larger),
            "median_a_plus_top1_share_missed": top1_share,
            "verdict": (
                "dilution" if includes_zero and top1_share > 0.5
                else "cancellation" if larger else "undetermined"
            ),
        }
    summary["q2_dilution_vs_cancellation"] = verdicts

    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    for col, panel in enumerate(["RULER", "LongBench"]):
        frame = blocks[(blocks.panel == panel) & (blocks.layer == 14)]
        local = group_masks(frame)
        for row, metric in enumerate(["cb_a_plus", "cb_a_minus"]):
            axes[row, col].boxplot(
                [frame.loc[local[g], metric].dropna() for g in GROUPS], showfliers=False,
            )
            axes[row, col].set_xticks(range(1, 5), GROUPS)
            axes[row, col].set_title(f"{panel} L14: {metric} (c = selection boundary)")
    fig.tight_layout()
    fig.savefig(figures / "e2_a_plus_minus.png", dpi=170)
    plt.close(fig)

    # ------------------------------------------------------------ E3
    blocks_rope = blocks[~blocks.set_index(["panel", "sample_id", "layer"]).index.isin(list(rope_failed))]
    rows = []
    for (panel, layer), frame in blocks_rope.groupby(["panel", "layer"]):
        local = group_masks(frame)
        for name in GROUPS:
            part = frame[local[name]]
            rows.append({
                "panel": panel, "layer": layer, "group": name, "blocks": len(part),
                **{f"mean_{c}": part[c].mean() for c in blocks.columns
                   if c.startswith(("var_share_", "delta_share_", "rho_"))},
                **{f"median_{c}": part[c].median() for c in blocks.columns
                   if c.startswith(("var_share_", "delta_share_"))},
            })
    e3 = pd.DataFrame(rows)
    e3.to_csv(out / "e3_bands.csv", index=False)
    attenuation = read(out, "attenuation")
    e3_att = attenuation.groupby(["panel", "layer", "rope", "pair", "band"]).agg(
        theta=("theta", "first"), a_theory_128=("a_theory_128", "first"),
        rho_mean=("rho_mean", "mean"), rho_std=("rho_mean", "std"),
    ).reset_index()
    e3_att.to_csv(out / "e3_attenuation.csv", index=False)
    summary["e3_band_pairs"] = e3_att[(e3_att.rope == "post")].drop_duplicates("pair").band.value_counts().to_dict()

    layers = sorted(e3_att.layer.unique())
    fig, axes = plt.subplots(1, len(layers), figsize=(3.6 * len(layers), 3.4), sharey=True)
    for ax, layer in zip(np.atleast_1d(axes), layers):
        for (panel, rope), frame in e3_att[e3_att.layer == layer].groupby(["panel", "rope"]):
            ax.plot(frame.pair, frame.rho_mean, label=f"{panel} {rope}-RoPE",
                    linestyle="-" if rope == "post" else ":")
        theory = e3_att[e3_att.layer == layer].drop_duplicates("pair")
        ax.plot(theory.pair, theory.a_theory_128, color="black", linewidth=1, label="a_f(128) theory")
        ax.set_title(f"Layer {layer}")
        ax.set_xlabel("Frequency pair f (0 = highest)")
    np.atleast_1d(axes)[0].set_ylabel("Block-mean attenuation rho_f")
    np.atleast_1d(axes)[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(figures / "e3_attenuation.png", dpi=170)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharey=True)
    for col, panel in enumerate(["RULER", "LongBench"]):
        frame = e3[(e3.panel == panel) & (e3.layer == 14) & e3.group.isin(["hit", "missed"])].set_index("group")
        for row, kind in enumerate(["var_share", "delta_share"]):
            frame[[f"mean_{kind}_{b}" for b in ("high", "mid", "low")]].plot.bar(
                stacked=True, ax=axes[row, col], legend=row == 0 and col == 0,
                color=["#b25136", "#d9b45a", "#4378a8"])
            axes[row, col].set_title(f"{panel} L14: {kind} by RoPE band")
            axes[row, col].tick_params(axis="x", rotation=0)
    fig.tight_layout()
    fig.savefig(figures / "e3_band_contributions.png", dpi=170)
    plt.close(fig)

    # ------------------------------------------------------------ E4
    peaks = read(out, "peaks")
    peaks["delta_ratio"] = peaks.delta_nope / peaks.delta_rope
    peaks["driver"] = np.where(
        (peaks.nope_rank == 0) & (peaks.delta_ratio >= 0.5), "content",
        np.where((peaks.nope_rank > 0) & (peaks.delta_ratio < 0.5), "position", "mixed"),
    )
    peaks["named_case"] = False
    for panel, sample, layer, head, start in NAMED_CASES:
        peaks.loc[
            (peaks.panel == panel) & (peaks.sample_id == sample) & (peaks.layer == layer)
            & (peaks.query_head == head) & (peaks.token_start == start), "named_case"
        ] = True
    top = peaks.sort_values("true_mass_mean", ascending=False).groupby(["panel", "layer"]).head(50)
    e4 = pd.concat([top, peaks[peaks.named_case]]).drop_duplicates(TILE + ["block"])
    e4.to_csv(out / "e4_nope_peaks.csv", index=False)
    summary["e4_driver_counts"] = e4.groupby(["panel", "layer"]).driver.value_counts().unstack(fill_value=0).to_dict(orient="index")
    summary["e4_named_cases"] = peaks[peaks.named_case][
        TILE + ["peak_position", "token", "nope_rank", "delta_rope", "delta_nope", "driver", "distance"]
    ].to_dict(orient="records")
    summary["missing_named_cases"] = [
        list(case) for case in NAMED_CASES
        if not ((peaks.panel == case[0]) & (peaks.sample_id == case[1]) & (peaks.layer == case[2])
                & (peaks.query_head == case[3]) & (peaks.token_start == case[4])).any()
    ]

    # ------------------------------------------------------------ E5
    sink = read(out, "sink")
    recurrence = peaks.groupby(["panel", "sample_id", "peak_position"]).apply(
        lambda x: len(set(zip(x.layer, x.query_head)))
    ).rename("peak_in_layer_heads").reset_index()
    e5 = e4[["panel", "sample_id", "layer", "kv_head", "query_head", "peak_position", "token"]].drop_duplicates(
        ["panel", "sample_id", "layer", "peak_position"]
    ).rename(columns={"query_head": "peak_head"})
    group_stats = sink.groupby(["panel", "sample_id", "layer", "kv_head", "position"]).agg(
        group_argmax_fraction_mean=("argmax_fraction", "mean"),
        group_argmax_fraction_min=("argmax_fraction", "min"),
    ).reset_index().rename(columns={"position": "peak_position"})
    own = sink.rename(columns={"position": "peak_position", "query_head": "peak_head"})
    e5 = e5.merge(own.drop(columns=["token"]), on=["panel", "sample_id", "layer", "kv_head", "peak_head", "peak_position"], how="left")
    e5 = e5.merge(group_stats, on=["panel", "sample_id", "layer", "kv_head", "peak_position"], how="left")
    e5 = e5.merge(recurrence, on=["panel", "sample_id", "peak_position"], how="left")
    e5["sink_like"] = e5.argmax_fraction > 0.8
    e5.to_csv(out / "e5_sink.csv", index=False)
    sink.to_csv(out / "e5_sink_all_heads.csv.gz", index=False)
    summary["q3_sink_like_fraction"] = e5.groupby(["panel", "layer"]).sink_like.mean().to_dict()

    # ------------------------------------------------------------ E6
    output = read(out, "output_rows")
    oracle_rows = output[output.method == "tile_block_oracle"][TILE + ["query_token", "retained_mass"]]
    output = output.merge(oracle_rows.rename(columns={"retained_mass": "oracle_retained"}), on=TILE + ["query_token"])
    output["mass_regret_row"] = output.oracle_retained - output.retained_mass
    output["mass_loss_row"] = 1 - output.retained_mass
    oproj = read(out, "oproj_rows")
    rows = []
    for (panel, layer, method), frame in output.groupby(["panel", "layer", "method"]):
        valid = frame.retained_mass.notna()
        rows.append({
            "panel": panel, "layer": layer, "method": method, "rows": len(frame),
            "mass_loss_pp_mean": 100 * frame.mass_loss_row.mean(),
            "mass_regret_pp_mean": 100 * frame.mass_regret_row.mean(),
            **{f"{stat}_{m}": getattr(frame[m], stat)() for m in ("rel_err", "e_par", "e_perp", "cos")
               for stat in ("mean", "median")},
            "p90_rel_err": frame.rel_err.quantile(0.9),
            "abs_e_par_gt_e_perp_frac": (frame.e_par.abs() > frame.e_perp).mean(),
            "spearman_mass_loss_rel_err": spearman(frame.mass_loss_row[valid], frame.rel_err[valid]) if valid.sum() > 2 else np.nan,
            "spearman_mass_regret_rel_err": spearman(frame.mass_regret_row[valid], frame.rel_err[valid]) if valid.sum() > 2 else np.nan,
        })
    e6 = pd.DataFrame(rows).merge(
        oproj.groupby(["panel", "layer", "method"]).agg(
            oproj_rel_err_mean=("oproj_rel_err", "mean"), oproj_rel_err_median=("oproj_rel_err", "median"),
            residual_rel_mean=("residual_rel", "mean"), residual_rel_median=("residual_rel", "median"),
            attn_out_over_residual_mean=("attn_out_over_residual", "mean"),
        ).reset_index(), on=["panel", "layer", "method"], how="left",
    )
    e6.to_csv(out / "e6_output_error.csv", index=False)
    drop = read(out, "block_drop")
    drop["error_per_mass"] = drop.drop_err_direct / drop.true_mass_mean
    drop.to_csv(out / "e6_block_drop.csv", index=False)
    summary["q4_output_error"] = e6[e6.method.isin(
        ["tile_block_oracle", "mean_raw", "mean_raw_v2comp", "exact_raw", "mean_var2", "thr_base_a0.08", "thr_base_a0.08_v2comp"]
    )][["panel", "layer", "method", "mass_regret_pp_mean", "mean_rel_err", "median_rel_err",
        "abs_e_par_gt_e_perp_frac", "spearman_mass_regret_rel_err", "oproj_rel_err_median", "residual_rel_median"]
      ].to_dict(orient="records")
    summary["q4_block_drop"] = drop.groupby(["panel", "layer"]).agg(
        blocks=("block", "size"), median_mass=("true_mass_mean", "median"),
        median_vbar_minus_o_rel=("vbar_minus_o_rel", "median"), median_cos_vbar_o=("cos_vbar_o", "median"),
        median_vbar_norm_rel=("vbar_norm_rel", "median"), median_drop_err=("drop_err_direct", "median"),
        median_error_per_mass=("error_per_mass", "median"),
    ).reset_index().to_dict(orient="records")

    base = output[output.method == "mean_raw"]
    layers = sorted(base.layer.unique())
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(layers)))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for ax, panel in zip(axes, ["RULER", "LongBench"]):
        for color, layer in zip(colors, layers):
            frame = base[(base.panel == panel) & (base.layer == layer)]
            ax.scatter(100 * frame.mass_regret_row, frame.rel_err, s=3, alpha=0.35, color=color, label=f"L{layer}")
        ax.set_title(f"{panel}: MeanPool, per sampled row and head")
        ax.set_xlabel("Row mass regret vs shared-block oracle (pp)")
        ax.set_yscale("log")
    axes[0].set_ylabel("Relative head-output error ||O_d - O_s|| / ||O_d||")
    axes[0].legend(markerscale=4, fontsize=8)
    fig.tight_layout()
    fig.savefig(figures / "e6_mass_vs_output_error.png", dpi=170)
    plt.close(fig)

    # ------------------------------------------------------------ E7
    rbs_names = [m for m in k128.method.unique() if m.startswith(("rbs_", "thr_base"))] + ["mean_raw", "tile_block_oracle"]
    e7 = k128[k128.method.isin(rbs_names)].groupby(["panel", "layer", "method"]).agg(
        retained_mass_mean=("retained_mass_mean", "mean"),
        block_regret_pp=("block_regret", lambda x: 100 * x.mean()),
        missed_recall_mass=("missed_recall_mass", "mean"),
        oracle_recall_mass=("oracle_recall_mass", "mean"),
        selected_remote_blocks=("selected_remote_blocks", "mean"),
        density=("density", "mean"),
        rescue_only_blocks=("rescue_only_blocks", "mean"),
        rescue_only_false_pos=("rescue_only_false_pos", "mean"),
    ).reset_index().merge(
        e6[["panel", "layer", "method", "mean_rel_err", "median_rel_err", "oproj_rel_err_median"]],
        on=["panel", "layer", "method"], how="left",
    )
    e7.to_csv(out / "e7_rbs.csv", index=False)
    named = []
    for panel, sample, layer, head, start in NAMED_CASES:
        frame = blocks[(blocks.panel == panel) & (blocks.sample_id == sample) & (blocks.layer == layer)
                       & (blocks.query_head == head) & (blocks.token_start == start)
                       & (blocks.in_oracle == 1) & (blocks.in_mean == 0)]
        for _, row in frame.iterrows():
            named.append({
                "case": f"{panel}/{sample}/L{layer}/H{head}/K@{start}", "q_start": int(row.q_start),
                **{c[4:]: int(row[c]) for c in blocks.columns if c.startswith("sel_rbs") or c.startswith("sel_thr")},
            })
    summary["q5_named_cases_rescued"] = named
    summary["q5_rbs"] = e7[e7.layer == 14].to_dict(orient="records")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for ax, panel in zip(axes, ["RULER", "LongBench"]):
        for layer, frame in e7[e7.panel == panel].groupby("layer"):
            budget = frame[frame.method.isin(["mean_raw"] + [f"rbs_budget_r{k}" for k in (2, 4, 8)])]
            budget = budget.assign(k=budget.method.map(lambda m: 0 if m == "mean_raw" else int(m.split("_r")[-1])))
            budget = budget.sort_values("k")
            ax.plot(budget.k, budget.missed_recall_mass, marker="o", label=f"L{layer} fixed 16 blocks")
            threshold = frame[frame.method.str.startswith("rbs_thr_b0.22")]
            base = frame[frame.method == "thr_base_a0.22"]
            ax.scatter(threshold.selected_remote_blocks - float(base.selected_remote_blocks.iloc[0]),
                       threshold.missed_recall_mass, marker="x")
        ax.set_title(f"{panel}: rescue of MeanPool-missed oracle blocks")
        ax.set_xlabel("Rescue blocks (fixed budget) / extra blocks over base alpha=0.22 (x)")
        ax.set_ylabel("Mass-weighted recall of missed blocks")
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(figures / "e7_rescue_recall.png", dpi=170)
    plt.close(fig)

    # ------------------------------------------------------------ E8
    needles = read(out, "needles")
    needles.to_csv(out / "e8_needle_blocks.csv", index=False)
    remote = needles[needles.region == "remote"]
    spikes = e4[e4.layer.isin(remote.layer.unique()) & (e4.panel == "RULER")]
    summary["q6_needles"] = {
        "rows": len(needles), "region_counts": needles.region.value_counts().to_dict(),
        "queried_remote": remote[remote.queried == 1].groupby("layer").agg(
            in_mean=("in_mean", "mean"), in_oracle=("in_oracle", "mean"),
            in_v1_like=("in_v1_like_a0.08", "mean"), median_mean_rank=("mean_raw_rank", "median"),
            median_true_rank=("true_rank", "median"), median_gap=("gap", "median"),
            median_n_eff=("n_eff", "median"), median_top1=("top1_share", "median"),
            peak_in_value=("peak_in_value_frac", "mean"),
        ).to_dict(orient="index"),
        "e4_spike_median_gap_row": spikes.groupby("layer").gap_row.median().to_dict(),
        "e4_spike_median_n_eff_row": spikes.groupby("layer").n_eff_row.median().to_dict(),
    }

    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    archive = out.with_name(out.name + "_report.zip")
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as bundle:
        for path in sorted(out.rglob("*")):
            if path.is_file():
                bundle.write(path, path.relative_to(out.parent))
    print(f"[validation-summary] {out / 'summary.json'}")
    print(f"[validation-summary] report: {archive}")


if __name__ == "__main__":
    main()
