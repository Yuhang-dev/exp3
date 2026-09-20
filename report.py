"""Aggregate paired task/timing results and render the quality-latency report."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


GROUP = ["task", "length_label", "method", "config_id"]


def _read_profile(path):
    frame = pd.read_csv(path)
    return frame if len(frame) else None


def _sum_available(values):
    return values.sum(min_count=1)


def make_report(folder):
    folder = Path(folder)
    timings = pd.read_csv(folder / "timings.csv")
    quality = pd.read_csv(folder / "quality.csv")
    profile = _read_profile(folder / "profile.csv")
    metadata = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))

    timing_sample = timings.groupby(GROUP + ["sample_id"], dropna=False).agg(
        alpha=("alpha", "first"),
        repeats=("repeat", "size"),
        actual_tokens=("actual_tokens", "first"),
        prefill_ms=("prefill_ms", "median"),
        prefill_min_ms=("prefill_ms", "min"),
        prefill_max_ms=("prefill_ms", "max"),
        peak_allocated_gib=("peak_allocated_gib", "max"),
        peak_reserved_gib=("peak_reserved_gib", "max"),
    ).reset_index()
    dense_timing = timing_sample[timing_sample.method == "dense"][
        ["sample_id", "prefill_ms"]
    ].rename(columns={"prefill_ms": "dense_prefill_ms"})
    timing_sample = timing_sample.merge(dense_timing, on="sample_id", how="left")
    timing_sample["paired_speedup_vs_dense"] = (
        timing_sample.dense_prefill_ms / timing_sample.prefill_ms
    )
    timing_summary = timing_sample.groupby(GROUP, dropna=False).agg(
        alpha=("alpha", "first"),
        timing_samples=("sample_id", "nunique"),
        timing_repeats=("repeats", "sum"),
        median_actual_tokens=("actual_tokens", "median"),
        min_actual_tokens=("actual_tokens", "min"),
        max_actual_tokens=("actual_tokens", "max"),
        prefill_ms=("prefill_ms", "median"),
        prefill_min_ms=("prefill_ms", "min"),
        prefill_max_ms=("prefill_ms", "max"),
        paired_speedup_vs_dense=("paired_speedup_vs_dense", "median"),
        peak_allocated_gib=("peak_allocated_gib", "max"),
        peak_reserved_gib=("peak_reserved_gib", "max"),
    ).reset_index()

    quality_summary = quality.groupby(GROUP, dropna=False).agg(
        quality_samples=("sample_id", "nunique"),
        metric=("metric", "first"),
        score=("score", "mean"),
        score_min=("score", "min"),
        score_max=("score", "max"),
        delta_vs_dense=("delta_vs_dense", "mean"),
        exact_match=("exact_match", "mean"),
        target_accuracy=("target_accuracy", "mean"),
        all_target_em=("all_target_em", "mean"),
    ).reset_index()
    summary = quality_summary.merge(timing_summary, on=GROUP, how="outer")

    baseline_rows = []
    for (task, length), group in summary.groupby(["task", "length_label"], dropna=False):
        means = group[group.method.isin(["mean_native", "mean_balanced"])]
        if len(means):
            strongest = means.sort_values(
                ["score", "prefill_ms"],
                ascending=[False, True],
            ).iloc[0]
            baseline_rows.append({
                "task": task,
                "length_label": length,
                "mean_baseline_config": strongest.config_id,
                "mean_baseline_prefill_ms": strongest.prefill_ms,
                "mean_baseline_score": strongest.score,
            })
    if baseline_rows:
        baselines = pd.DataFrame(baseline_rows)
        summary = summary.merge(baselines, on=["task", "length_label"], how="left")
        baseline_samples = timing_sample.merge(
            baselines[["task", "length_label", "mean_baseline_config"]],
            on=["task", "length_label"],
            how="inner",
        )
        baseline_samples = baseline_samples[
            baseline_samples.config_id == baseline_samples.mean_baseline_config
        ][["task", "length_label", "sample_id", "prefill_ms"]].rename(
            columns={"prefill_ms": "mean_baseline_sample_ms"}
        )
        paired_mean = timing_sample.merge(
            baseline_samples,
            on=["task", "length_label", "sample_id"],
            how="left",
        )
        paired_mean["paired_speedup_vs_mean_baseline"] = (
            paired_mean.mean_baseline_sample_ms / paired_mean.prefill_ms
        )
        paired_mean = paired_mean.groupby(GROUP, dropna=False)[
            "paired_speedup_vs_mean_baseline"
        ].median().reset_index()
        summary = summary.merge(paired_mean, on=GROUP, how="left")
        summary["delta_vs_mean_baseline"] = summary.score - summary.mean_baseline_score
    else:
        summary["mean_baseline_config"] = None
        summary["mean_baseline_prefill_ms"] = np.nan
        summary["mean_baseline_score"] = np.nan
        summary["paired_speedup_vs_mean_baseline"] = np.nan
        summary["delta_vs_mean_baseline"] = np.nan

    if profile is not None:
        stage_columns = [
            "descriptor_ms", "selector_ms", "indices_ms", "exact_ms",
            "mean_ms", "merge_ms", "attention_ms",
        ]
        profile_sample = profile.groupby(GROUP + ["sample_id"], dropna=False).agg(
            **{column: (column, "sum") for column in stage_columns},
            effective_exact_token_pair_ratio=("effective_exact_token_pair_ratio", "mean"),
            legacy_block_density=("legacy_block_density", "mean"),
            exact_physical_qk_tiles=("exact_physical_qk_tiles", _sum_available),
            mean_proxy_entries=("mean_proxy_entries", "sum"),
            selector_proxy_entries=("selector_proxy_entries", "sum"),
            mean_executed_logit_entries=("mean_executed_logit_entries", "sum"),
            mean_executed_value_entries=("mean_executed_value_entries", "sum"),
            selector_executed_dot_entries=("selector_executed_dot_entries", "sum"),
            selector_physical_qk_tiles=("selector_physical_qk_tiles", _sum_available),
        ).reset_index()
        profile_summary = profile_sample.groupby(GROUP, dropna=False).agg(
            profile_samples=("sample_id", "nunique"),
            **{column: (column, "median") for column in stage_columns},
            effective_exact_token_pair_ratio=("effective_exact_token_pair_ratio", "mean"),
            legacy_block_density=("legacy_block_density", "mean"),
            exact_physical_qk_tiles=("exact_physical_qk_tiles", "median"),
            mean_proxy_entries=("mean_proxy_entries", "median"),
            selector_proxy_entries=("selector_proxy_entries", "median"),
            mean_executed_logit_entries=("mean_executed_logit_entries", "median"),
            mean_executed_value_entries=("mean_executed_value_entries", "median"),
            selector_executed_dot_entries=("selector_executed_dot_entries", "median"),
            selector_physical_qk_tiles=("selector_physical_qk_tiles", "median"),
        ).reset_index()
        summary = summary.merge(profile_summary, on=GROUP, how="left")
    summary = summary.sort_values(["task", "length_label", "score", "prefill_ms"], ascending=[True, True, False, True])
    summary.to_csv(folder / "summary.csv", index=False)

    domain_summary = None
    if "domain" in quality.columns and quality["domain"].notna().any():
        domain_group = ["task", "length_label", "domain", "method", "config_id"]
        domain_quality = quality[quality["domain"].notna()].groupby(
            domain_group,
            dropna=False,
        ).agg(
            quality_samples=("sample_id", "nunique"),
            score=("score", "mean"),
            delta_vs_dense=("delta_vs_dense", "mean"),
        ).reset_index()
        sample_domains = quality[["sample_id", "domain"]].dropna().drop_duplicates()
        domain_timing_samples = timing_sample.merge(sample_domains, on="sample_id", how="inner")
        domain_timing = domain_timing_samples.groupby(domain_group, dropna=False).agg(
            prefill_ms=("prefill_ms", "median"),
            paired_speedup_vs_dense=("paired_speedup_vs_dense", "median"),
        ).reset_index()
        domain_summary = domain_quality.merge(domain_timing, on=domain_group, how="left")
        domain_summary = domain_summary.sort_values(
            ["task", "domain", "score", "prefill_ms"],
            ascending=[True, True, False, True],
        )
        domain_summary.to_csv(folder / "domain_summary.csv", index=False)

    groups = list(summary.groupby(["task", "length_label"], dropna=False))
    columns = min(2, len(groups))
    rows = int(np.ceil(len(groups) / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(6.4 * columns, 4.4 * rows), squeeze=False)
    colors = {
        "dense": "#3569a8",
        "fp_v1": "#7f7f7f",
        "mean_native": "#e27739",
        "mean_balanced": "#d9a441",
        "cgf_mean": "#27856d",
        "dispersion_mean": "#8064a2",
    }
    for axis, ((task, length), group) in zip(axes.flat, groups):
        for method, method_rows in group.groupby("method"):
            method_rows = method_rows.sort_values("alpha")
            axis.plot(
                method_rows.prefill_ms,
                method_rows.score,
                "o-" if len(method_rows) > 1 else "o",
                color=colors.get(method, "black"),
                label=method,
            )
            for item in method_rows.itertuples():
                axis.annotate(item.config_id, (item.prefill_ms, item.score), fontsize=7, xytext=(3, 3), textcoords="offset points")
        axis.set_title(f"{task} · {length}")
        axis.set_xlabel("Full-model prefill (ms; per-sample median)")
        axis.set_ylabel("Task score (0–100)")
        axis.grid(alpha=0.2)
        axis.legend(frameon=False, fontsize=8)
    for axis in axes.flat[len(groups):]:
        axis.axis("off")
    figure.tight_layout()
    figure.savefig(folder / "quality_latency.png", dpi=160)
    plt.close(figure)

    lines = [
        "# Sparse prefill quality–latency report",
        "",
        "每个任务和长度单独汇总；未把不同任务的原始分数平均成总分。质量每个样本只计算一次；"
        "prefill 先在样本内取重复测量中位数，再跨样本取中位数。",
        "",
        "| Task | Length | Config | n | Score | Δ dense | Prefill ms | Paired speedup | Peak GiB | Exact pair ratio | Mean baseline | Speedup vs mean |",
        "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |",
    ]
    for row in summary.itertuples():
        delta = "—" if pd.isna(row.delta_vs_dense) else f"{row.delta_vs_dense:+.2f}"
        speedup = "—" if pd.isna(row.paired_speedup_vs_dense) else f"{row.paired_speedup_vs_dense:.2f}×"
        exact_ratio = getattr(row, "effective_exact_token_pair_ratio", np.nan)
        exact_ratio = "—" if pd.isna(exact_ratio) else f"{exact_ratio:.1%}"
        mean_name = "—" if pd.isna(row.mean_baseline_config) else str(row.mean_baseline_config)
        mean_speedup = "—" if pd.isna(row.paired_speedup_vs_mean_baseline) else f"{row.paired_speedup_vs_mean_baseline:.2f}×"
        lines.append(
            f"| {row.task} | {row.length_label} | {row.config_id} | {int(row.quality_samples)} | "
            f"{row.score:.2f} | {delta} | {row.prefill_ms:.1f} | {speedup} | "
            f"{row.peak_allocated_gib:.2f} | {exact_ratio} | {mean_name} | {mean_speedup} |"
        )

    if domain_summary is not None:
        lines += [
            "",
            "## Domain breakdown",
            "",
            "该表只拆分同一 benchmark 内的 domain，不把不同 domain 的原始分数另行混合。",
            "",
            "| Task | Domain | Config | n | Score | Δ dense | Prefill ms | Paired speedup |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
        for row in domain_summary.itertuples():
            delta = "—" if pd.isna(row.delta_vs_dense) else f"{row.delta_vs_dense:+.2f}"
            speedup = (
                "—"
                if pd.isna(row.paired_speedup_vs_dense)
                else f"{row.paired_speedup_vs_dense:.2f}×"
            )
            lines.append(
                f"| {row.task} | {row.domain} | {row.config_id} | "
                f"{int(row.quality_samples)} | {row.score:.2f} | {delta} | "
                f"{row.prefill_ms:.1f} | {speedup} |"
            )

    if profile is not None:
        lines += [
            "",
            "## 独立 profile：阶段耗时",
            "",
            "以下时间来自不进入主计时的额外前向。每个已 profile 样本先跨层求和，再在同一任务、长度和配置内取中位数；`Attention` 还包含布局转换及统计等未单列开销，不能把各阶段时间当作完整模型 prefill 时间。",
            "",
            "| Task | Length | Config | n | Descriptor ms | Selector ms | Indices ms | Exact ms | Mean ms | Merge ms | Attention ms |",
            "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for row in summary.itertuples():
            profile_samples = getattr(row, "profile_samples", np.nan)
            if pd.isna(profile_samples):
                continue
            stage_values = [
                getattr(row, column)
                for column in (
                    "descriptor_ms", "selector_ms", "indices_ms", "exact_ms",
                    "mean_ms", "merge_ms", "attention_ms",
                )
            ]
            formatted = ["—" if pd.isna(value) else f"{value:.2f}" for value in stage_values]
            lines.append(
                f"| {row.task} | {row.length_label} | {row.config_id} | {int(profile_samples)} | "
                + " | ".join(formatted)
                + " |"
            )

        lines += [
            "",
            "## 独立 profile：执行量",
            "",
            "计数均为单次完整模型 prefill 的跨层合计。Proxy logical entries 表示有用块条目；executed entries 表示当前 dense PyTorch proxy 实际计算的条目。",
            "",
            "| Task | Length | Config | Exact pair ratio | Exact QK tiles | Mean logical | Mean logits executed | Mean values executed | Selector logical | Selector dots executed | Selector QK tiles |",
            "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        quantity_columns = (
            "exact_physical_qk_tiles",
            "mean_proxy_entries",
            "mean_executed_logit_entries",
            "mean_executed_value_entries",
            "selector_proxy_entries",
            "selector_executed_dot_entries",
            "selector_physical_qk_tiles",
        )
        for row in summary.itertuples():
            if pd.isna(getattr(row, "profile_samples", np.nan)):
                continue
            exact_ratio = getattr(row, "effective_exact_token_pair_ratio", np.nan)
            exact_ratio = "—" if pd.isna(exact_ratio) else f"{exact_ratio:.1%}"
            quantities = []
            for column in quantity_columns:
                value = getattr(row, column)
                quantities.append("—" if pd.isna(value) else f"{value:,.0f}")
            lines.append(
                f"| {row.task} | {row.length_label} | {row.config_id} | {exact_ratio} | "
                + " | ".join(quantities)
                + " |"
            )

    lines += ["", "## 逐项负结果", ""]
    negatives = []
    for (task, length), group in summary.groupby(["task", "length_label"], dropna=False):
        dense = group[group.method == "dense"]
        if not len(dense):
            continue
        dense = dense.iloc[0]
        for row in group[group.method != "dense"].itertuples():
            issues = []
            if row.score < dense.score:
                issues.append(f"score {row.score - dense.score:+.2f}")
            if row.prefill_ms > dense.prefill_ms:
                issues.append(f"prefill {row.prefill_ms / dense.prefill_ms:.2f}× dense")
            if issues:
                negatives.append(f"- {task} / {length} / `{row.config_id}`: " + "; ".join(issues) + ".")
    lines += negatives or ["- 本次结果中没有按上述简单条件标记的负项；这不构成统计非劣性结论。"]
    lines += [
        "",
        "## 口径与执行状态",
        "",
        f"- Run status: `{metadata.get('status', 'unknown')}`; model: `{metadata['arguments']['model']}`.",
        f"- Independent profile: {'已运行首个分组样本' if profile is not None else '未运行'}。profile 不进入主计时。",
        "- `prefill_ms` 从 GPU 上已有 token IDs 的完整模型前向开始，到最后位置 LM head 与首 token argmax 完成并同步；包含 descriptor、selector、indices、exact、mean 和 merge。",
        "- Dense 与每个稀疏配置使用相同输入；paired speedup 先逐样本计算。Mean baseline 在同一任务/长度下从 `mean_native` 与 `mean_balanced` 中按质量优先、延迟次优选出，仅作描述性比较。",
        "- `exact_physical_qk_tiles` 是已选 Triton micro-tile 配置下实际循环的 QK tile 数；profile 同时保留逻辑 proxy entries 与当前实现实际执行的 dense proxy entries，不把 block multiplicity 当成实际 KV-token FLOPs。",
        "- 本报告不把小样本差异解释为统计非劣性，也不把未测的速度或质量写成收益。",
        "",
        "![Quality–latency](quality_latency.png)",
        "",
    ]
    (folder / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    make_report(parser.parse_args().folder)
