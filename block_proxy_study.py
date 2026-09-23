"""Offline block-size and mean-key routing study from saved V1 Q/K captures.

All probabilities and oracles here are diagnostics, not a timed prefill method.
The Q/K tensors are the actual post-RoPE tensors captured during V1 prefill.
"""

import argparse
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F


SUMMARY_FIELDS = (
    "sample_id", "layer", "query_head", "kv_head", "q_block_size",
    "k_block_size", "q_tile_start", "sampled_rows", "remote_start",
    "remote_end", "remote_blocks", "budget_tokens", "budget_blocks",
    "fixed_mass_mean", "remote_mass_mean", "token_oracle",
    "row_block_oracle", "tile_block_oracle", "method",
    "retained_mass_mean", "retained_mass_p05", "block_regret",
    "selected_overlap_oracle", "cutoff_margin", "mean_gap",
    "gap_std_across_blocks", "gap_range_across_blocks",
    "query_gap_mean", "query_gap_std_across_blocks",
)

BLOCK_FIELDS = (
    "sample_id", "layer", "query_head", "q_block_size", "k_block_size",
    "q_tile_start", "block", "token_start", "token_end",
    "true_mass_mean", "true_mass_row_max", "mean_proxy_score",
    "mean_proxy_rank", "true_mass_rank", "mean_gap", "gap_row_max",
    "gap_row_std", "sub2_recovered_gap", "sub4_recovered_gap",
    "sub2_residual_gap", "sub4_residual_gap", "logit_std_mean",
    "max_minus_mean_mean", "effective_support_mean", "query_jensen_gap",
    "mean_raw_selected", "oracle_selected",
)

ROW_FIELDS = (
    "sample_id", "layer", "query_head", "q_block_size", "k_block_size",
    "q_tile_start", "query_token", "block", "token_start",
    "true_block_mass", "mean_logit", "jensen_gap", "sub2_recovered_gap",
    "sub4_recovered_gap", "logit_std", "max_minus_mean",
    "effective_support",
)

V1_FIELDS = (
    "sample_id", "layer", "query_head", "q_tile_start", "sampled_rows",
    "selected_blocks", "protected_blocks", "routed_blocks",
    "retained_mass_mean", "retained_mass_min",
)


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--sample-id", action="append", default=[])
    parser.add_argument("--layers", nargs="+", type=int)
    parser.add_argument("--heads", nargs="+", type=int, default=[0, 7, 14, 21])
    parser.add_argument("--q-block-sizes", nargs="+", type=int, default=[64, 128, 256])
    parser.add_argument("--k-block-sizes", nargs="+", type=int, default=[32, 64, 128, 256])
    parser.add_argument(
        "--tile-fractions", nargs="+", type=float,
        default=[0.25, 0.5, 0.75, 0.9],
    )
    parser.add_argument("--rows-per-tile", type=int, default=16)
    parser.add_argument("--budget-tokens", type=int, default=2048)
    parser.add_argument("--sink-tokens", type=int, default=256)
    parser.add_argument("--local-tokens", type=int, default=384)
    parser.add_argument("--profiles-per-config", type=int, default=3)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def align_up(value, size):
    return (value + size - 1) // size * size


def align_down(value, size):
    return value // size * size


def tile_starts(sequence, size, fractions):
    count = sequence // size
    return sorted({
        min(count - 1, max(0, int(fraction * count))) * size
        for fraction in fractions
    })


def query_rows(start, size, sequence, count):
    end = min(start + size, sequence)
    if count >= end - start:
        return torch.arange(start, end)
    return torch.linspace(start, end - 1, count).round().long().unique()


def ranks(values):
    order = torch.argsort(values, descending=True)
    rank = torch.empty_like(order)
    rank[order] = torch.arange(order.numel(), device=order.device)
    return rank


def block_arrays(logits, probabilities, remote_start, remote_end, block_size):
    rows = logits.shape[0]
    blocks = (remote_end - remote_start) // block_size
    scores = logits[:, remote_start:remote_end].reshape(rows, blocks, block_size)
    block_probability = probabilities[:, remote_start:remote_end].reshape(
        rows, blocks, block_size
    ).sum(dim=-1)

    mean_logit = scores.mean(dim=-1)
    log_mass = torch.logsumexp(scores, dim=-1)
    log_mean_mass = log_mass - math.log(block_size)
    gap = (log_mean_mass - mean_logit).clamp_min(0.0)
    log_l0 = mean_logit + math.log(block_size)
    log_l2 = torch.logsumexp(
        scores.reshape(rows, blocks, 2, block_size // 2).mean(dim=-1)
        + math.log(block_size // 2),
        dim=-1,
    )
    log_l4 = torch.logsumexp(
        scores.reshape(rows, blocks, 4, block_size // 4).mean(dim=-1)
        + math.log(block_size // 4),
        dim=-1,
    )
    std = scores.std(dim=-1, unbiased=False)
    max_minus_mean = scores.amax(dim=-1) - mean_logit
    effective_support = torch.exp(
        2 * log_mass - torch.logsumexp(2 * scores, dim=-1)
    )
    return {
        "probability": block_probability,
        "log_mass": log_mass,
        "log_l0": log_l0,
        "log_l2": log_l2,
        "log_l4": log_l4,
        "mean_logit": mean_logit,
        "gap": gap,
        "h2": log_l2 - log_l0,
        "h4": log_l4 - log_l0,
        "std": std,
        "max_minus_mean": max_minus_mean,
        "effective_support": effective_support,
        "scores": scores,
    }


def selection_scores(data, log_z):
    result = {}
    for name, key in (
        ("mean", "log_l0"), ("sub2", "log_l2"), ("sub4", "log_l4")
    ):
        values = data[key]
        result[f"{name}_raw"] = torch.logsumexp(values, dim=0)
        result[f"{name}_balanced_remote"] = torch.softmax(
            values, dim=1
        ).mean(dim=0)
        result[f"{name}_oracle_z"] = torch.exp(
            values - log_z[:, None]
        ).mean(dim=0)
    result["exact_raw"] = torch.logsumexp(data["log_mass"], dim=0)
    result["tile_block_oracle"] = data["probability"].mean(dim=0)
    return result


def write_config(
    sample_id, layer, head, kv_head, q_size, k_size, q_start, positions,
    remote_start, remote_end, budget_tokens, logits, probabilities, log_z,
    summary_writer, block_writer, row_writer, profile_file,
    profiles_per_config,
):
    data = block_arrays(
        logits, probabilities, remote_start, remote_end, k_size
    )
    block_mass = data["probability"]
    blocks = block_mass.shape[1]
    budget_blocks = budget_tokens // k_size
    fixed_mass = 1.0 - block_mass.sum(dim=1)
    fixed_mean = fixed_mass.mean()
    mean_mass = block_mass.mean(dim=0)
    row_log_probability = data["log_mass"] - log_z[:, None]
    query_gap = (
        torch.logsumexp(row_log_probability, dim=0)
        - math.log(positions.numel())
        - row_log_probability.mean(dim=0)
    ).clamp_min(0.0)
    oracle_ids = torch.topk(mean_mass, budget_blocks).indices
    oracle_mask = torch.zeros(blocks, dtype=torch.bool, device=logits.device)
    oracle_mask[oracle_ids] = True

    token_oracle = fixed_mean + torch.topk(
        probabilities[:, remote_start:remote_end],
        budget_tokens, dim=1,
    ).values.sum(dim=1).mean()
    row_block_oracle = fixed_mean + torch.topk(
        block_mass, budget_blocks, dim=1
    ).values.sum(dim=1).mean()
    tile_block_oracle = fixed_mean + mean_mass[oracle_ids].sum()
    scores = selection_scores(data, log_z)
    mean_mask = torch.zeros_like(oracle_mask)
    mean_mask[torch.topk(scores["mean_raw"], budget_blocks).indices] = True
    gap_mean = data["gap"].mean(dim=0)
    common = {
        "sample_id": sample_id,
        "layer": layer,
        "query_head": head,
        "kv_head": kv_head,
        "q_block_size": q_size,
        "k_block_size": k_size,
        "q_tile_start": q_start,
        "sampled_rows": positions.numel(),
        "remote_start": remote_start,
        "remote_end": remote_end,
        "remote_blocks": blocks,
        "budget_tokens": budget_tokens,
        "budget_blocks": budget_blocks,
        "fixed_mass_mean": fixed_mean.item(),
        "remote_mass_mean": block_mass.sum(dim=1).mean().item(),
        "token_oracle": token_oracle.item(),
        "row_block_oracle": row_block_oracle.item(),
        "tile_block_oracle": tile_block_oracle.item(),
        "mean_gap": gap_mean.mean().item(),
        "gap_std_across_blocks": gap_mean.std(unbiased=False).item(),
        "gap_range_across_blocks": (
            gap_mean.max() - gap_mean.min()
        ).item(),
        "query_gap_mean": query_gap.mean().item(),
        "query_gap_std_across_blocks": query_gap.std(
            unbiased=False
        ).item(),
    }

    for method, score in scores.items():
        indices = torch.topk(score, budget_blocks).indices
        chosen = torch.zeros_like(oracle_mask)
        chosen[indices] = True
        row_retained = fixed_mass + block_mass[:, indices].sum(dim=1)
        retained = row_retained.mean()
        sorted_scores = torch.sort(score, descending=True).values
        margin = (
            sorted_scores[budget_blocks - 1] - sorted_scores[budget_blocks]
            if budget_blocks < blocks else torch.tensor(0.0, device=logits.device)
        )
        summary_writer.writerow({
            **common,
            "method": method,
            "retained_mass_mean": retained.item(),
            "retained_mass_p05": torch.quantile(row_retained, 0.05).item(),
            "block_regret": (tile_block_oracle - retained).item(),
            "selected_overlap_oracle": (
                (chosen & oracle_mask).sum().item() / budget_blocks
            ),
            "cutoff_margin": margin.item(),
        })

    raw_rank = ranks(scores["mean_raw"])
    true_rank = ranks(mean_mass)
    saved = {
        name: value.detach().cpu()
        for name, value in data.items()
    }
    block_mass_cpu = saved["probability"]
    mean_mass_cpu = mean_mass.cpu()
    query_gap_cpu = query_gap.cpu()
    gap_mean_cpu = gap_mean.cpu()
    raw_rank_cpu = raw_rank.cpu()
    true_rank_cpu = true_rank.cpu()
    mean_raw_score_cpu = scores["mean_raw"].cpu()
    mean_mask_cpu = mean_mask.cpu()
    oracle_mask_cpu = oracle_mask.cpu()
    positions_cpu = positions.cpu()
    for block in range(blocks):
        token_start = remote_start + block * k_size
        block_writer.writerow({
            "sample_id": sample_id,
            "layer": layer,
            "query_head": head,
            "q_block_size": q_size,
            "k_block_size": k_size,
            "q_tile_start": q_start,
            "block": block,
            "token_start": token_start,
            "token_end": token_start + k_size,
            "true_mass_mean": mean_mass_cpu[block].item(),
            "true_mass_row_max": block_mass_cpu[:, block].max().item(),
            "mean_proxy_score": mean_raw_score_cpu[block].item(),
            "mean_proxy_rank": raw_rank_cpu[block].item(),
            "true_mass_rank": true_rank_cpu[block].item(),
            "mean_gap": gap_mean_cpu[block].item(),
            "gap_row_max": saved["gap"][:, block].max().item(),
            "gap_row_std": saved["gap"][:, block].std(
                unbiased=False
            ).item(),
            "sub2_recovered_gap": saved["h2"][:, block].mean().item(),
            "sub4_recovered_gap": saved["h4"][:, block].mean().item(),
            "sub2_residual_gap": (
                saved["log_mass"][:, block] - saved["log_l2"][:, block]
            ).mean().item(),
            "sub4_residual_gap": (
                saved["log_mass"][:, block] - saved["log_l4"][:, block]
            ).mean().item(),
            "logit_std_mean": saved["std"][:, block].mean().item(),
            "max_minus_mean_mean": (
                saved["max_minus_mean"][:, block].mean().item()
            ),
            "effective_support_mean": (
                saved["effective_support"][:, block].mean().item()
            ),
            "query_jensen_gap": query_gap_cpu[block].item(),
            "mean_raw_selected": int(mean_mask_cpu[block].item()),
            "oracle_selected": int(oracle_mask_cpu[block].item()),
        })

        for row in range(positions.numel()):
            row_writer.writerow({
                "sample_id": sample_id,
                "layer": layer,
                "query_head": head,
                "q_block_size": q_size,
                "k_block_size": k_size,
                "q_tile_start": q_start,
                "query_token": positions_cpu[row].item(),
                "block": block,
                "token_start": token_start,
                "true_block_mass": block_mass_cpu[row, block].item(),
                "mean_logit": saved["mean_logit"][row, block].item(),
                "jensen_gap": saved["gap"][row, block].item(),
                "sub2_recovered_gap": saved["h2"][row, block].item(),
                "sub4_recovered_gap": saved["h4"][row, block].item(),
                "logit_std": saved["std"][row, block].item(),
                "max_minus_mean": (
                    saved["max_minus_mean"][row, block].item()
                ),
                "effective_support": (
                    saved["effective_support"][row, block].item()
                ),
            })

    missed = oracle_mask_cpu & ~mean_mask_cpu
    missed_ids = torch.nonzero(missed).flatten()
    missed_ids = missed_ids[
        torch.argsort(mean_mass_cpu[missed_ids], descending=True)
    ][:profiles_per_config]
    for block in missed_ids.tolist():
        row = int(block_mass_cpu[:, block].argmax().item())
        token_start = remote_start + block * k_size
        profile_file.write(json.dumps({
            "sample_id": sample_id,
            "layer": layer,
            "query_head": head,
            "q_block_size": q_size,
            "k_block_size": k_size,
            "q_tile_start": q_start,
            "query_token": positions_cpu[row].item(),
            "token_start": token_start,
            "true_mass_mean": mean_mass_cpu[block].item(),
            "true_mass_row": block_mass_cpu[row, block].item(),
            "mean_proxy_rank": raw_rank_cpu[block].item(),
            "true_mass_rank": true_rank_cpu[block].item(),
            "mean_logit": saved["mean_logit"][row, block].item(),
            "jensen_gap": saved["gap"][row, block].item(),
            "centered_token_logits": (
                saved["scores"][row, block]
                - saved["mean_logit"][row, block]
            ).tolist(),
        }) + "\n")


def write_v1_reference(
    sample_id, layer, head, q_start, positions, probabilities, route,
    writer,
):
    selected = route["selected_mask"][q_start // 128, :, head]
    protected = route["protected_mask"][q_start // 128, :, head]
    routed = route["routed_mask"][q_start // 128, :, head]
    blocks = selected.numel()
    padded = F.pad(probabilities, (0, blocks * 128 - probabilities.shape[1]))
    masses = padded.reshape(positions.numel(), blocks, 128).sum(dim=-1)
    retained = masses[:, selected.to(probabilities.device)].sum(dim=1)
    writer.writerow({
        "sample_id": sample_id,
        "layer": layer,
        "query_head": head,
        "q_tile_start": q_start,
        "sampled_rows": positions.numel(),
        "selected_blocks": selected.sum().item(),
        "protected_blocks": protected.sum().item(),
        "routed_blocks": routed.sum().item(),
        "retained_mass_mean": retained.mean().item(),
        "retained_mass_min": retained.min().item(),
    })


def main():
    args = arguments()
    max_k = max(args.k_block_sizes)
    assert all(size % 4 == 0 and max_k % size == 0 for size in args.k_block_sizes)
    assert args.budget_tokens % max_k == 0
    assert args.rows_per_tile > 0
    assert all(0 <= fraction < 1 for fraction in args.tile_fractions)
    if (args.out / "metadata.json").exists():
        raise FileExistsError(args.out)
    args.out.mkdir(parents=True, exist_ok=True)

    sample_dirs = sorted((args.capture / "samples").iterdir())
    if args.sample_id:
        sample_dirs = [
            sample for sample in sample_dirs if sample.name in args.sample_id
        ]
    if not sample_dirs:
        raise ValueError("no captured samples selected")
    with (
        (args.out / "summary.csv").open("w", newline="", encoding="utf-8") as summary,
        (args.out / "v1_reference.csv").open("w", newline="", encoding="utf-8") as v1_output,
        gzip.open(args.out / "block_stats.csv.gz", "wt", newline="", encoding="utf-8") as blocks,
        gzip.open(args.out / "row_block_stats.csv.gz", "wt", newline="", encoding="utf-8") as rows,
        gzip.open(args.out / "miss_profiles.jsonl.gz", "wt", encoding="utf-8") as profiles,
    ):
        summary_writer = csv.DictWriter(summary, fieldnames=SUMMARY_FIELDS)
        v1_writer = csv.DictWriter(v1_output, fieldnames=V1_FIELDS)
        block_writer = csv.DictWriter(blocks, fieldnames=BLOCK_FIELDS)
        row_writer = csv.DictWriter(rows, fieldnames=ROW_FIELDS)
        for writer in (summary_writer, v1_writer, block_writer, row_writer):
            writer.writeheader()

        source_samples = []
        for sample_dir in sample_dirs:
            sample_meta = json.loads(
                (sample_dir / "metadata.json").read_text(encoding="utf-8")
            )
            if sample_meta["status"] != "complete":
                raise ValueError(f"incomplete capture: {sample_dir}")
            source_samples.append({
                "sample_id": sample_meta["sample_id"],
                "capture_input_sha256": sample_meta["capture_input_sha256"],
                "capture_layers": sample_meta["captured_layers"],
            })
            layer_dirs = sorted((sample_dir / "layers").glob("layer_*"))
            if args.layers is not None:
                layer_dirs = [
                    path for path in layer_dirs
                    if int(path.name.split("_")[-1]) in args.layers
                ]
            for layer_dir in layer_dirs:
                layer = int(layer_dir.name.split("_")[-1])
                q = torch.load(
                    layer_dir / "query_post_rope.pt",
                    map_location="cpu", weights_only=True,
                )
                k = torch.load(
                    layer_dir / "key_post_rope.pt",
                    map_location="cpu", weights_only=True,
                )
                route = torch.load(
                    layer_dir / "v1_route.pt",
                    map_location="cpu", weights_only=True,
                )
                sequence = q.shape[0]
                group = q.shape[1] // k.shape[1]
                scale = float(route["scale"])
                for head in args.heads:
                    kv_head = head // group
                    key = k[:, kv_head].to(args.device, dtype=torch.float32)
                    for q_size in args.q_block_sizes:
                        for q_start in tile_starts(
                            sequence, q_size, args.tile_fractions
                        ):
                            remote_start = align_up(args.sink_tokens, max_k)
                            remote_end = align_down(
                                q_start - args.local_tokens, max_k
                            )
                            if remote_end - remote_start <= args.budget_tokens:
                                continue
                            positions = query_rows(
                                q_start, q_size, sequence,
                                args.rows_per_tile,
                            )
                            query = q[positions, head].to(
                                args.device, dtype=torch.float32
                            )
                            prefix_end = int(positions[-1]) + 1
                            logits = query @ key[:prefix_end].T * scale
                            key_positions = torch.arange(
                                prefix_end, device=args.device
                            )
                            logits.masked_fill_(
                                key_positions[None, :]
                                > positions.to(args.device)[:, None],
                                float("-inf"),
                            )
                            log_z = torch.logsumexp(logits, dim=1)
                            probabilities = torch.softmax(logits, dim=1)
                            if q_size == 128:
                                write_v1_reference(
                                    sample_dir.name, layer, head, q_start,
                                    positions, probabilities, route, v1_writer,
                                )
                            for k_size in args.k_block_sizes:
                                write_config(
                                    sample_dir.name, layer, head, kv_head,
                                    q_size, k_size, q_start, positions,
                                    remote_start, remote_end,
                                    args.budget_tokens, logits,
                                    probabilities, log_z,
                                    summary_writer, block_writer, row_writer,
                                    profiles, args.profiles_per_config,
                                )
                            print(
                                f"[proxy-study] {sample_dir.name} layer {layer} "
                                f"head {head} Q={q_size} start={q_start}",
                                flush=True,
                            )
                    del key
                del q, k, route

    artifacts = {}
    for name in (
        "summary.csv", "v1_reference.csv", "block_stats.csv.gz",
        "row_block_stats.csv.gz", "miss_profiles.jsonl.gz",
    ):
        path = args.out / name
        artifacts[name] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
    metadata = {
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "source_samples": source_samples,
        "artifacts": artifacts,
        "interpretation": (
            "Offline frozen-Q/K mass diagnostics. Fixed/protected token mass is "
            "the complement of the aligned remote region; mean_balanced_remote "
            "normalizes across remote blocks only. oracle_z uses true full-row "
            "normalizers and is diagnostic, not an online selector."
        ),
    }
    (args.out / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[proxy-study] complete: {args.out}", flush=True)


if __name__ == "__main__":
    main()
