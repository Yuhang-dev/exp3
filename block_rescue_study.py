"""Offline test: FlashPrefill block-mean selection + refined-Jensen rescue of missed blocks.

One query row, key block b of B tokens, logits s_j = q.k_j * scale:
  FlashPrefill (MeanPool):  Z_b = sum_j e^{s_j} >= B e^{mean_j s_j}           (Jensen, gap g_b)
  Refined, R subset of b:   Z_b >= sum_{j in R} e^{s_j} + (B-m) e^{mean_{j notin R} s_j} >= B e^{mean s}
                            0 <= log Z_b - log(refined) <= g_{b minus R}       (remainder gap)
R_b is chosen once per Q tile as the top-m tokens of qbar.k_j (qbar = tile-mean query). Because
sum_i e^{q_i.k_j} >= n e^{qbar.k_j}, this maximizes a Jensen lower bound of each token's tile mass.

Rescue modes: re-rank blocks by the refined bound, add k blocks by it on top of MeanPool's 16,
keep the R tokens of unselected blocks at token level, and optionally correct the remainder with
the FlashPrefill-V2 zero-order term. Every mode is compared with simply taking more MeanPool
blocks at the same exact-token cost. Quantities are frozen-activation, single-layer diagnostics
(attention mass and head-output error), not task accuracy.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path

import torch

from block_proxy_positions import query_rows, tile_starts
from block_proxy_study import block_arrays, selection_scores
from block_proxy_validation import (
    B, BUDGET_BLOCKS, BUDGET_TOKENS, FRACTIONS, REMOTE_START, ROWS_PER_TILE, Checks, Layer, Writer,
    cpu_lists, inverse_frequencies, panel_of, remote_end_of, threshold_mask, topk_mask,
)


MEAN_BUDGETS = (16, 17, 18, 20, 24)
PLUS_BLOCKS = (1, 2, 4)
THRESHOLDS = (0.08, 0.06, 0.04)


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--layers", nargs="+", type=int)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def expand(idx, rows):
    return idx if idx.dim() == 3 else idx.unsqueeze(0).expand(rows, -1, -1)


def refined_log(s_rem, idx, with_var=False):
    """Per-row log of the refined lower bound; s_rem is [rows, blocks, B]."""
    full = expand(idx, s_rem.shape[0])
    m = full.shape[-1]
    rest = torch.ones_like(s_rem).scatter_(2, full, 0.0)
    rest_mean = (s_rem * rest).sum(-1) / (B - m)
    rest_log = math.log(B - m) + rest_mean
    if with_var:  # exact remainder variance: diagnostic COBS-on-remainder upper bound
        rest_log = rest_log + 0.5 * ((s_rem - rest_mean[..., None]).square() * rest).sum(-1) / (B - m)
    return torch.logaddexp(torch.logsumexp(s_rem.gather(2, full), -1), rest_log)


def evaluate(layer, head, q_start, positions, writers, checks, common):
    device = layer.device
    kv = head // layer.group
    remote_end = remote_end_of(q_start)
    blocks = (remote_end - REMOTE_START) // B
    remote_slice = slice(REMOTE_START // B, remote_end // B)

    # Selection path identical to block_proxy_study.py (float32 logits).
    query, logits = layer.tile_logits(head, positions)
    data = block_arrays(logits, torch.softmax(logits, 1), REMOTE_START, remote_end, B)
    mean_score = selection_scores(data, torch.logsumexp(logits, 1))["mean_raw"]
    oracle = topk_mask(data["probability"].mean(0), BUDGET_BLOCKS)
    visible = q_start // B
    visible_log = torch.logsumexp(logits[:, :visible * B].reshape(-1, visible, B).mean(-1), 0)

    # Exact quantities in float64.
    prefix = logits.shape[1]
    q64 = query.double()
    k64 = layer.k[:prefix, kv].double()
    v64 = layer.v[:prefix, kv].double()
    s = q64 @ k64.T * layer.scale
    s.masked_fill_(torch.arange(prefix, device=device)[None, :] > positions.to(device)[:, None], float("-inf"))
    log_z = torch.logsumexp(s, 1)
    p = torch.exp(s - log_z[:, None])
    rows = p.shape[0]
    s_rem = s[:, REMOTE_START:remote_end].reshape(rows, blocks, B)
    p_rem = p[:, REMOTE_START:remote_end].reshape(rows, blocks, B)
    k_rem = k64[REMOTE_START:remote_end].reshape(blocks, B, -1)
    v_rem = v64[REMOTE_START:remote_end].reshape(blocks, B, -1)
    block_p = p_rem.sum(-1)
    block_w = torch.einsum("rbj,bjd->rbd", p_rem, v_rem)
    fixed_p = p[:, :REMOTE_START].sum(1) + p[:, remote_end:].sum(1)
    fixed_w = p[:, :REMOTE_START] @ v64[:REMOTE_START] + p[:, remote_end:] @ v64[remote_end:]
    dense = p @ v64
    dense_norm = torch.linalg.vector_norm(dense, dim=1)

    # Rescue tokens: tile-mean query (method) and per-row argmax (oracle upper bound).
    tile_score = k_rem @ q64.mean(0)
    tile_idx = {m: torch.topk(tile_score, m, dim=-1).indices for m in (1, 2)}
    row_idx = torch.topk(s_rem, 1, dim=-1).indices

    # Theory checks: MeanPool bound <= refined bound <= true mass, gap <= remainder gap.
    log_true = torch.logsumexp(s_rem, -1)
    mean_log = math.log(B) + s_rem.mean(-1)
    ref1 = refined_log(s_rem, tile_idx[1])
    rest = torch.ones_like(s_rem).scatter_(2, expand(tile_idx[1], rows), 0.0)
    rest_logits = s_rem.masked_fill(rest == 0, float("-inf"))
    rest_gap = torch.logsumexp(rest_logits, -1) - math.log(B - 1) - (s_rem * rest).sum(-1) / (B - 1)
    checks.update("refined_above_true", (ref1 - log_true).max().item(), 1e-9)
    checks.update("meanpool_above_refined", (mean_log - ref1).max().item(), 1e-9)
    checks.update("refined_gap_above_remainder_gap", ((log_true - ref1) - rest_gap).max().item(), 1e-9)

    scores = {
        "tile_m1": torch.logsumexp(ref1, 0),
        "tile_m2": torch.logsumexp(refined_log(s_rem, tile_idx[2]), 0),
        "tile_m1_var": torch.logsumexp(refined_log(s_rem, tile_idx[1], with_var=True), 0),
        "row_m1": torch.logsumexp(refined_log(s_rem, row_idx), 0),
    }
    mean16 = topk_mask(mean_score, BUDGET_BLOCKS)
    variants = {f"mean_k{k}": (topk_mask(mean_score, k), None, False) for k in MEAN_BUDGETS}
    variants["oracle_k16"] = (oracle, None, False)
    for name, score in scores.items():
        variants[f"refined_{name}_k16"] = (topk_mask(score, BUDGET_BLOCKS), None, False)
    for name in ("tile_m1", "row_m1"):
        remaining = scores[name].masked_fill(mean16, float("-inf"))
        for extra in PLUS_BLOCKS:
            variants[f"mean16_plus{extra}_{name}"] = (mean16 | topk_mask(remaining, extra), None, False)
    variants["mean16_v2"] = (mean16, None, True)
    variants["mean16_tok_tile_m1"] = (mean16, tile_idx[1], False)
    variants["mean16_tok_tile_m2"] = (mean16, tile_idx[2], False)
    variants["mean16_tok_row_m1"] = (mean16, row_idx, False)
    variants["mean16_tok_tile_m1_v2"] = (mean16, tile_idx[1], True)
    variants["mean16_plus2_tok_tile_m1"] = (variants["mean16_plus2_tile_m1"][0], tile_idx[1], False)
    for alpha in THRESHOLDS:
        variants[f"thr{alpha:.2f}"] = (threshold_mask(visible_log, alpha, remote_slice), None, False)
    threshold = variants["thr0.08"][0]
    variants["thr0.08_v2"] = (threshold, None, True)
    variants["thr0.08_tok_tile_m1"] = (threshold, tile_idx[1], False)
    variants["thr0.08_tok_tile_m1_v2"] = (threshold, tile_idx[1], True)

    missed = (oracle & ~mean16).double()
    oracle_retained = fixed_p + block_p @ oracle.double()
    missed_mass = block_p @ missed
    for name, (mask, idx, v2) in variants.items():
        selected = mask.double()
        unselected = 1.0 - selected
        numerator = fixed_w + torch.einsum("rbd,b->rd", block_w, selected)
        denominator = fixed_p + block_p @ selected
        rescued = torch.zeros_like(block_p)
        tokens = B * int(mask.sum())
        if idx is not None:
            full = expand(idx, rows)
            picked = p_rem.gather(2, full) * unselected[None, :, None]
            values = v_rem[torch.arange(blocks, device=device)[None, :, None], full]
            numerator = numerator + torch.einsum("rbm,rbmd->rd", picked, values)
            denominator = denominator + picked.sum((1, 2))
            rescued = picked.sum(-1)
            tokens += full.shape[-1] * int((~mask).sum())
        if v2:  # FlashPrefill-V2 zero-order term on whatever part of each unselected block is not exact
            if idx is None:
                count, k_rest, v_rest = B, k_rem.mean(1), v_rem.mean(1)
            else:
                m = idx.shape[-1]
                gather = idx.unsqueeze(-1).expand(-1, -1, k_rem.shape[-1])
                k_rest = (k_rem.sum(1) - k_rem.gather(1, gather).sum(1)) / (B - m)
                v_rest = (v_rem.sum(1) - v_rem.gather(1, gather).sum(1)) / (B - m)
                count = B - m
            correction = count * torch.exp(q64 @ k_rest.T * layer.scale - log_z[:, None]) * unselected[None, :]
            numerator = numerator + correction @ v_rest
            denominator = denominator + correction.sum(1)
        sparse = numerator / denominator[:, None]
        writers["rows"].rows(
            {**common, "method": name, "remote_tokens": tokens, "remote_blocks": blocks},
            cpu_lists({
                "query_token": positions,
                "retained": fixed_p + block_p @ selected + rescued.sum(1),
                "oracle_retained": oracle_retained,
                "missed_mass": missed_mass,
                "recovered_missed": block_p @ (selected * missed) + (rescued * missed).sum(1),
                "rel_err": torch.linalg.vector_norm(dense - sparse, dim=1) / dense_norm,
            }),
        )

    # Mechanism diagnostics by block group (weights = per-row block mass).
    g_full = log_true - mean_log
    gap_refined = log_true - ref1
    argmax_hit = (row_idx.squeeze(-1) == tile_idx[1].squeeze(-1)[None, :]).double()
    top1 = p_rem.amax(-1) / block_p.clamp_min(1e-300)
    for group, member in (
        ("hit", oracle & mean16), ("missed", oracle & ~mean16),
        ("false_pos", ~oracle & mean16), ("other", ~oracle & ~mean16),
    ):
        weight = block_p * member.double()[None, :]
        writers["diag"].row({
            **common, "group": group, "blocks": int(member.sum()), "weight": weight.sum().item(),
            "g_full": (weight * g_full).sum().item(),
            "gap_refined": (weight * gap_refined).sum().item(),
            "argmax_hit": (weight * argmax_hit).sum().item(),
            "top1_share": (weight * top1).sum().item(),
        })


def summarize(out):
    import numpy as np
    import pandas as pd

    rows = pd.read_csv(out / "rows.csv.gz")
    rows["regret_pp"] = 100 * (rows.oracle_retained - rows.retained)
    keys = ["panel", "layer", "method"]
    grouped = rows.groupby(keys)
    table = grouped.agg(
        block_equiv=("remote_tokens", lambda x: x.mean() / B),
        regret_pp=("regret_pp", "mean"),
        rel_err_median=("rel_err", "median"),
        rel_err_mean=("rel_err", "mean"),
        recovered_missed=("recovered_missed", "sum"),
        missed_mass=("missed_mass", "sum"),
    ).reset_index()
    table["missed_recall"] = table.recovered_missed / table.missed_mass
    table = table.drop(columns=["recovered_missed", "missed_mass"])

    # Same-cost comparison with the MeanPool frontier (16/17/18/20/24 blocks), linear in block count.
    frames = []
    for (panel, layer), frame in table.groupby(["panel", "layer"]):
        frame = frame.copy()
        frontier = frame[frame.method.str.fullmatch(r"mean_k\d+")].sort_values("block_equiv")
        inside = frame.block_equiv.between(frontier.block_equiv.min(), frontier.block_equiv.max())
        for metric in ("regret_pp", "rel_err_median", "rel_err_mean"):
            same_cost = np.interp(frame.block_equiv, frontier.block_equiv, frontier[metric])
            frame[f"meanpool_same_cost_{metric}"] = np.where(inside, same_cost, np.nan)
            frame[f"gain_vs_same_cost_{metric}"] = frame[f"meanpool_same_cost_{metric}"] - frame[metric]
            base = frame.loc[frame.method == "mean_k16", metric].iloc[0]
            frame[f"gain_vs_flashprefill16_{metric}"] = base - frame[metric]
        frames.append(frame)
    table = pd.concat(frames)
    table.to_csv(out / "summary_methods.csv", index=False)

    diag = pd.read_csv(out / "diagnostics.csv.gz")
    sums = diag.groupby(["panel", "layer", "group"])[
        ["blocks", "weight", "g_full", "gap_refined", "argmax_hit", "top1_share"]
    ].sum()
    for column in ("g_full", "gap_refined", "argmax_hit", "top1_share"):
        sums[column] = sums[column] / sums["weight"]
    sums.reset_index().to_csv(out / "summary_diagnostics.csv", index=False)

    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 500)
    show = ["panel", "layer", "method", "block_equiv", "regret_pp", "missed_recall", "rel_err_median",
            "gain_vs_same_cost_regret_pp", "gain_vs_same_cost_rel_err_median"]
    print(table[show].round(4).to_string(index=False))
    print(sums.round(4).to_string())


def main():
    args = arguments()
    if (args.out / "metadata.json").exists():
        raise FileExistsError(args.out)
    args.out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    checks = Checks()
    writers = {name: Writer(args.out / f"{name}.csv.gz") for name in ("rows", "diag")}
    for capture in args.capture:
        meta = json.loads((capture / "metadata.json").read_text(encoding="utf-8"))
        inv_freq = inverse_frequencies(meta["model_config"], device)
        for sample_dir in sorted((capture / "samples").iterdir()):
            sample_meta = json.loads((sample_dir / "metadata.json").read_text(encoding="utf-8"))
            for layer_id in sample_meta["captured_layers"]:
                if args.layers is not None and layer_id not in args.layers:
                    continue
                layer = Layer(sample_dir, sample_meta, layer_id, device, inv_freq)
                common = {"panel": panel_of(capture), "sample_id": sample_meta["sample_id"], "layer": layer_id}
                for q_start in tile_starts(layer.sequence, B, FRACTIONS):
                    if remote_end_of(q_start) - REMOTE_START <= BUDGET_TOKENS:
                        continue
                    positions = query_rows(q_start, B, layer.sequence, ROWS_PER_TILE)
                    for head in range(layer.heads):
                        evaluate(
                            layer, head, q_start, positions, writers, checks,
                            {**common, "query_head": head, "q_start": q_start},
                        )
                print(f"[rescue] {common} done", flush=True)
                del layer
                torch.cuda.empty_cache()
    for writer in writers.values():
        writer.close()
    report = checks.report()
    (args.out / "metadata.json").write_text(json.dumps({
        "captures": [str(path) for path in args.capture],
        "checks": report,
        "constants": {
            "block": B, "budget_blocks": BUDGET_BLOCKS, "rows_per_tile": ROWS_PER_TILE,
            "mean_budgets": MEAN_BUDGETS, "plus_blocks": PLUS_BLOCKS, "thresholds": THRESHOLDS,
        },
        "source_sha256": {
            name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
            for name in ("block_rescue_study.py", "block_proxy_validation.py", "block_proxy_study.py")
        },
        "notes": [
            "Tile-mean query uses the 16 sampled rows of each Q tile (a kernel would use all 128).",
            "V2 zero-order term uses float64 plain means (the V2 kernel rounds them to BF16).",
            "retained mass counts exact tokens only; V2 variants are judged by rel_err.",
        ],
    }, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=1), flush=True)
    summarize(args.out)


if __name__ == "__main__":
    main()
