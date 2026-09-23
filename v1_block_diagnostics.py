"""Capture V1 block structure and dense-attention routing ground truth.

This is an intentionally offline diagnostic.  Its dense oracle, CPU copies, and
artifact writes are not part of the prefill latency measurement.
"""

import argparse
import csv
import hashlib
import json
import math
import platform
import re
import subprocess
import traceback
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
import triton

from attention import AttentionBackend
from data import input_hash


BLOCK_FIELDS = (
    "sample_id", "layer", "block", "token_start", "token_end", "token_count",
    "kv_head", "query_head_start", "query_head_end",
    "pre_mean_norm", "pre_token_norm_mean", "pre_token_norm_std",
    "pre_centered_rms", "pre_relative_mean_reconstruction_mse",
    "pre_directional_concentration", "pre_token_to_mean_cos_mean",
    "pre_token_to_mean_cos_std", "pre_token_to_mean_cos_min",
    "pre_adjacent_cos_mean", "pre_first_last_cos", "pre_half_mean_cos",
    "post_mean_norm", "post_token_norm_mean", "post_token_norm_std",
    "post_centered_rms", "post_relative_mean_reconstruction_mse",
    "post_directional_concentration", "post_token_to_mean_cos_mean",
    "post_token_to_mean_cos_std", "post_token_to_mean_cos_min",
    "post_adjacent_cos_mean", "post_first_last_cos", "post_half_mean_cos",
    "pre_post_mean_cos", "v1_mean_max_abs_error", "v1_mean_relative_l2_error",
    "causally_available_tile_heads", "selected_tile_heads", "routed_tile_heads",
    "selected_rate", "exact_attention_mass_sum", "missed_attention_mass_sum",
    "v1_proxy_score_sum",
    "mean_pool_jensen_gap_mean", "mean_pool_jensen_gap_row_max",
    "mean_pool_jensen_gap_exact_mass_weighted",
)

TILE_FIELDS = (
    "sample_id", "layer", "query_block", "query_token_start", "query_token_end",
    "query_token_count", "query_head", "kv_head", "selected_blocks",
    "protected_blocks", "routed_blocks", "causal_blocks",
    "exact_retained_mass_mean", "exact_retained_mass_min",
    "exact_omitted_mass_mean", "exact_omitted_mass_max",
    "oracle_same_budget_retained_mass", "oracle_route_overlap",
    "proxy_exact_cosine", "proxy_exact_l1", "proxy_top_block",
    "exact_top_block", "proxy_exact_remote_cosine", "proxy_exact_remote_l1",
    "proxy_remote_top_block", "exact_remote_top_block",
    "mean_pool_jensen_gap_exact_mass_weighted",
    "mean_pool_jensen_gap_missed_mass_weighted",
    "mean_pool_jensen_gap_tile_mean_max", "mean_pool_jensen_gap_row_max",
    "output_abs_l2_mean", "output_abs_l2_max",
    "output_relative_l2_mean", "output_relative_l2_max",
    "output_cosine_mean", "output_cosine_min",
)

LAYER_FIELDS = (
    "sample_id", "layer", "sequence_tokens", "blocks", "query_heads", "kv_heads",
    "selected_block_ratio", "routed_block_ratio", "exact_retained_mass_mean",
    "exact_retained_mass_p05", "exact_retained_mass_min",
    "oracle_same_budget_retained_mass_mean", "output_relative_l2_mean",
    "output_relative_l2_p95", "output_relative_l2_max", "output_cosine_mean",
    "proxy_exact_cosine_mean", "proxy_exact_remote_cosine_mean",
    "mean_pool_jensen_gap_exact_mass_weighted",
    "mean_pool_jensen_gap_tile_mean_max", "mean_pool_jensen_gap_row_max",
    "mean_pool_relative_l2_error_max",
)


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--sample-id", action="append", default=[])
    parser.add_argument("--sample-index", action="append", type=int, default=[])
    parser.add_argument("--all-samples", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--alpha", type=float, default=0.08)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--sink-blocks", type=int, default=2)
    parser.add_argument("--window-blocks", type=int, default=4)
    parser.add_argument("--last-full-blocks", type=int, default=2)
    parser.add_argument("--query-chunk", type=int, default=128)
    parser.add_argument(
        "--capture-only", action="store_true",
        help=(
            "Skip the dense oracle; save tensors selected by --save-q-layers "
            "and --save-v-layers alongside V1 routing."
        ),
    )
    parser.add_argument(
        "--include-ruler-final-token",
        action="store_true",
        help=(
            "Include the final RULER prompt token in sparse prefill. By default it is "
            "excluded to mirror the pinned upstream quality-generation boundary."
        ),
    )
    parser.add_argument(
        "--layers", nargs="+", default=["all"],
        help="Layer numbers, or 'all'.",
    )
    parser.add_argument(
        "--save-q-layers", nargs="+", default=[],
        help="Optionally save post-RoPE Q for these layers, or 'all'.",
    )
    parser.add_argument(
        "--save-pre-rope-q-layers", nargs="+", default=[],
        help="Save q_proj output before RoPE for these layers, or 'all'.",
    )
    parser.add_argument(
        "--save-layer-input-layers", nargs="+", default=[],
        help="Save input hidden states before each decoder layer, or 'all'.",
    )
    parser.add_argument(
        "--save-v-layers", nargs="+", default=[],
        help="Optionally save V for these layers, or 'all'.",
    )
    parser.add_argument(
        "--save-row-block-mass-layers", nargs="+", default=[],
        help="Save the large [tokens, heads, blocks] dense mass tensor for these layers.",
    )
    parser.add_argument(
        "--save-output-vector-layers", nargs="+", default=[],
        help="Save dense and selected-block output vectors for these layers.",
    )
    parser.add_argument(
        "--no-save-raw-k", dest="save_raw_k", action="store_false",
        help="Keep structural summaries but omit pre/post-RoPE token-level K tensors.",
    )
    parser.set_defaults(save_raw_k=True)
    return parser.parse_args()


def _jsonable(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _write_json(path, payload):
    path.write_text(
        json.dumps(_jsonable(payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _source_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _slug(value):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_") or "sample"


def _layer_set(words, layer_count, allow_empty=False):
    if not words:
        return set() if allow_empty else set(range(layer_count))
    if "all" in words:
        if len(words) != 1:
            raise ValueError("'all' cannot be combined with explicit layer numbers")
        return set(range(layer_count))
    layers = {int(word) for word in words}
    invalid = sorted(layer for layer in layers if layer < 0 or layer >= layer_count)
    if invalid:
        raise ValueError(f"invalid layers {invalid}; model has {layer_count} layers")
    return layers


def _select_samples(inputs, args):
    if args.all_samples:
        if args.sample_id or args.sample_index:
            raise ValueError("--all-samples cannot be combined with explicit samples")
        return list(inputs)
    by_id = {sample["sample_id"]: sample for sample in inputs}
    selected = []
    for sample_id in args.sample_id:
        if sample_id not in by_id:
            raise ValueError(f"sample ID not found: {sample_id}")
        selected.append(by_id[sample_id])
    for index in args.sample_index:
        selected.append(inputs[index])
    if not selected:
        selected = [inputs[0]]
    seen = set()
    unique = []
    for sample in selected:
        if sample["sample_id"] not in seen:
            unique.append(sample)
            seen.add(sample["sample_id"])
    return unique


def _gpu_metadata():
    properties = torch.cuda.get_device_properties(0)
    return {
        "name": torch.cuda.get_device_name(0),
        "total_memory_gib": properties.total_memory / 2**30,
        "cuda": torch.version.cuda,
        "capability": list(torch.cuda.get_device_capability(0)),
        "nvidia_smi": subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total",
                "--format=csv,noheader",
            ],
            text=True,
        ).strip(),
    }


class CsvSink:
    def __init__(self, path, fields):
        self.file = path.open("w", newline="", encoding="utf-8")
        self.writer = csv.DictWriter(self.file, fieldnames=fields)
        self.writer.writeheader()

    def write(self, row):
        self.writer.writerow({field: row.get(field) for field in self.writer.fieldnames})

    def close(self):
        self.file.flush()
        self.file.close()


def _quantile(values, q):
    return torch.quantile(values.float().reshape(-1), q).item()


def _key_structure(keys, block_size):
    """Return lossless first/second moments and compact sequence-structure summaries."""
    sequence, heads, width = keys.shape
    blocks = math.ceil(sequence / block_size)
    padded = torch.zeros(blocks * block_size, heads, width, dtype=torch.float32)
    padded[:sequence] = keys.float()
    values = padded.view(blocks, block_size, heads, width)
    counts = torch.full((blocks,), block_size, dtype=torch.int64)
    counts[-1] = sequence - (blocks - 1) * block_size
    positions = torch.arange(block_size).view(1, block_size, 1, 1)
    valid = positions < counts.view(blocks, 1, 1, 1)
    denominator = counts.float().view(blocks, 1, 1)

    mean = (values * valid).sum(dim=1) / denominator
    centered = (values - mean[:, None]) * valid
    variance = centered.square().sum(dim=1) / denominator
    centered_energy = variance.sum(dim=-1)
    token_sq = values.square().sum(dim=-1)
    token_norm = token_sq.sqrt()
    valid_tokens = valid.squeeze(-1)
    token_energy = (token_sq * valid_tokens).sum(dim=1) / counts.float().view(blocks, 1)
    token_norm_mean = (token_norm * valid_tokens).sum(dim=1) / counts.float().view(blocks, 1)
    token_norm_variance = (
        (token_norm - token_norm_mean[:, None]).square() * valid_tokens
    ).sum(dim=1) / counts.float().view(blocks, 1)

    mean_norm = mean.norm(dim=-1)
    cosine = (values * mean[:, None]).sum(dim=-1) / (
        token_norm * mean_norm[:, None] + 1e-12
    )
    cosine_mean = (cosine * valid_tokens).sum(dim=1) / counts.float().view(blocks, 1)
    cosine_variance = (
        (cosine - cosine_mean[:, None]).square() * valid_tokens
    ).sum(dim=1) / counts.float().view(blocks, 1)
    cosine_min = cosine.masked_fill(~valid_tokens, float("inf")).amin(dim=1)

    adjacent = F.cosine_similarity(values[:, :-1], values[:, 1:], dim=-1, eps=1e-12)
    pair_positions = torch.arange(block_size - 1).view(1, block_size - 1, 1)
    pair_valid = pair_positions < (counts - 1).clamp_min(0).view(blocks, 1, 1)
    pair_counts = (counts - 1).clamp_min(1).float().view(blocks, 1)
    adjacent_mean = (adjacent * pair_valid).sum(dim=1) / pair_counts

    first = values[:, 0]
    last_indices = (counts - 1).view(blocks, 1, 1).expand(blocks, heads, width)
    last = values.gather(1, last_indices[:, None]).squeeze(1)
    first_last = F.cosine_similarity(first, last, dim=-1, eps=1e-12)
    half_cosine = torch.empty(blocks, heads, dtype=torch.float32)
    for block in range(blocks):
        count = int(counts[block])
        split = max(1, count // 2)
        left = values[block, :split].mean(dim=0)
        right = values[block, split:count].mean(dim=0) if split < count else left
        half_cosine[block] = F.cosine_similarity(left, right, dim=-1, eps=1e-12)

    return {
        "counts": counts,
        "mean": mean,
        "variance_diag": variance,
        "mean_norm": mean_norm,
        "token_norm_mean": token_norm_mean,
        "token_norm_std": token_norm_variance.sqrt(),
        "centered_rms": centered_energy.sqrt(),
        "relative_mean_reconstruction_mse": centered_energy / (token_energy + 1e-12),
        "directional_concentration": mean_norm.square() / (token_energy + 1e-12),
        "token_to_mean_cos_mean": cosine_mean,
        "token_to_mean_cos_std": cosine_variance.sqrt(),
        "token_to_mean_cos_min": cosine_min,
        "adjacent_cos_mean": adjacent_mean,
        "first_last_cos": first_last,
        "half_mean_cos": half_cosine,
    }


def _protected_masks(selected, sink_blocks, window_blocks, last_full_blocks):
    query_blocks, key_blocks, heads = selected.shape
    query_ids = torch.arange(query_blocks).view(query_blocks, 1, 1)
    key_ids = torch.arange(key_blocks).view(1, key_blocks, 1)
    distance = query_ids - key_ids
    causal = key_ids <= query_ids
    protected = (
        (key_ids < sink_blocks)
        | ((distance >= 0) & (distance < window_blocks))
        | (query_ids >= query_blocks - last_full_blocks)
    ) & causal
    protected = protected.expand(query_blocks, key_blocks, heads).clone()
    routed = selected & ~protected
    return causal.expand_as(selected).clone(), protected, routed


@torch.inference_mode()
def _dense_oracle(
    q,
    k,
    v,
    mean_k,
    selected,
    scale,
    block_size,
    query_chunk,
    save_row_block_mass,
    save_output_vectors,
):
    """Compute dense token attention, then aggregate its probabilities by key block."""
    if q.shape[0] != 1:
        raise ValueError("V1 diagnostics require batch size 1")
    q = q[0]
    k = k[0]
    v = v[0]
    mean_k = mean_k[0]
    selected = selected[0]
    sequence, query_heads, width = q.shape
    kv_heads = k.shape[1]
    group = query_heads // kv_heads
    blocks = math.ceil(sequence / block_size)
    padded_sequence = blocks * block_size

    exact_sum = torch.zeros(blocks, blocks, query_heads, dtype=torch.float32)
    exact_max = torch.zeros_like(exact_sum)
    jensen_gap_sum = torch.zeros_like(exact_sum)
    jensen_gap_max = torch.zeros_like(exact_sum)
    logit_std_sum = torch.zeros_like(exact_sum)
    max_minus_mean_sum = torch.zeros_like(exact_sum)
    exact_aggregate_log_mass = torch.zeros_like(exact_sum)
    proxy_aggregate_log_mass = torch.zeros_like(exact_sum)
    row_retained = torch.empty(sequence, query_heads, dtype=torch.float32)
    row_abs_l2 = torch.empty_like(row_retained)
    row_relative_l2 = torch.empty_like(row_retained)
    row_output_cosine = torch.empty_like(row_retained)
    row_dense_output_norm = torch.empty_like(row_retained)
    row_selected_output_norm = torch.empty_like(row_retained)
    row_block_mass = (
        torch.empty(sequence, query_heads, blocks, dtype=torch.float16)
        if save_row_block_mass else None
    )
    dense_vectors = (
        torch.empty(sequence, query_heads, width, dtype=q.dtype)
        if save_output_vectors else None
    )
    sparse_vectors = torch.empty_like(dense_vectors) if save_output_vectors else None

    for kv_head in range(kv_heads):
        head_start = kv_head * group
        head_end = head_start + group
        key_head = k[:, kv_head].float()
        value_head = v[:, kv_head].float()
        mean_key_head = mean_k[:, kv_head].float()
        value_padded = torch.zeros(
            padded_sequence, width, dtype=torch.float32, device=v.device
        )
        value_padded[:sequence] = value_head
        value_blocks = value_padded.view(blocks, block_size, width)
        group_sum = torch.zeros(blocks, blocks, group, dtype=torch.float32, device=q.device)
        group_max = torch.zeros_like(group_sum)
        group_jensen_sum = torch.zeros_like(group_sum)
        group_jensen_max = torch.zeros_like(group_sum)
        group_logit_std_sum = torch.zeros_like(group_sum)
        group_max_minus_mean_sum = torch.zeros_like(group_sum)
        group_exact_log_mass = torch.zeros_like(group_sum)
        group_proxy_log_mass = torch.zeros_like(group_sum)

        for query_start in range(0, sequence, query_chunk):
            query_end = min(query_start + query_chunk, sequence)
            query_values = q[query_start:query_end, head_start:head_end].float()
            logits = torch.einsum("qhd,kd->qhk", query_values, key_head) * scale
            query_positions = torch.arange(query_start, query_end, device=q.device)
            key_positions = torch.arange(sequence, device=q.device)
            causal = key_positions.view(1, 1, sequence) <= query_positions.view(-1, 1, 1)
            logits.masked_fill_(~causal, float("-inf"))
            if padded_sequence != sequence:
                padded_logits = F.pad(
                    logits, (0, padded_sequence - sequence), value=float("-inf")
                )
            else:
                padded_logits = logits
            logit_blocks = padded_logits.view(
                query_end - query_start, group, blocks, block_size
            )
            for query_block in range(
                query_start // block_size, (query_end - 1) // block_size + 1
            ):
                if query_block == 0:
                    continue
                local_start = max(query_block * block_size, query_start) - query_start
                local_end = min((query_block + 1) * block_size, query_end) - query_start
                full_logits = logit_blocks[local_start:local_end, :, :query_block]
                block_logit_mean = full_logits.mean(dim=-1)
                block_log_mean_exp = (
                    torch.logsumexp(full_logits, dim=-1) - math.log(block_size)
                )
                block_jensen_gap = (
                    block_log_mean_exp - block_logit_mean
                ).clamp_min(0.0)
                block_logit_std = (
                    full_logits - block_logit_mean.unsqueeze(-1)
                ).square().mean(dim=-1).sqrt()
                block_max_minus_mean = full_logits.amax(dim=-1) - block_logit_mean
                group_jensen_sum[query_block, :query_block] += (
                    block_jensen_gap.sum(dim=0).transpose(0, 1)
                )
                group_jensen_max[query_block, :query_block] = torch.maximum(
                    group_jensen_max[query_block, :query_block],
                    block_jensen_gap.amax(dim=0).transpose(0, 1),
                )
                group_logit_std_sum[query_block, :query_block] += (
                    block_logit_std.sum(dim=0).transpose(0, 1)
                )
                group_max_minus_mean_sum[query_block, :query_block] += (
                    block_max_minus_mean.sum(dim=0).transpose(0, 1)
                )
                group_exact_log_mass[query_block, :query_block] = (
                    torch.logsumexp(block_log_mean_exp, dim=0).transpose(0, 1)
                )
                proxy_logits = torch.einsum(
                    "qhd,kd->qhk",
                    query_values[local_start:local_end],
                    mean_key_head[:query_block],
                ) * scale
                group_proxy_log_mass[query_block, :query_block] = (
                    torch.logsumexp(proxy_logits, dim=0).transpose(0, 1)
                )
            probabilities = torch.softmax(logits, dim=-1, dtype=torch.float32)
            if padded_sequence != sequence:
                probabilities = F.pad(probabilities, (0, padded_sequence - sequence))
            probability_blocks = probabilities.view(
                query_end - query_start, group, blocks, block_size
            )
            block_mass = probability_blocks.sum(dim=-1)
            query_block_ids = torch.div(query_positions, block_size, rounding_mode="floor")
            selected_rows = selected[
                query_block_ids, :, head_start:head_end
            ].permute(0, 2, 1)
            retained = (block_mass * selected_rows).sum(dim=-1)

            row_retained[query_start:query_end, head_start:head_end] = retained.cpu()
            if row_block_mass is not None:
                row_block_mass[query_start:query_end, head_start:head_end] = (
                    block_mass.to(torch.float16).cpu()
                )

            for query_block in range(
                query_start // block_size, (query_end - 1) // block_size + 1
            ):
                local_start = max(query_block * block_size, query_start) - query_start
                local_end = min((query_block + 1) * block_size, query_end) - query_start
                local_mass = block_mass[local_start:local_end]
                group_sum[query_block] += local_mass.sum(dim=0).transpose(0, 1)
                group_max[query_block] = torch.maximum(
                    group_max[query_block], local_mass.amax(dim=0).transpose(0, 1)
                )

            dense_output = torch.einsum(
                "qhk,kd->qhd", probabilities[..., :sequence], value_head
            )
            selected_numerator = torch.einsum(
                "qhbt,btd->qhd",
                probability_blocks * selected_rows.unsqueeze(-1),
                value_blocks,
            )
            sparse_output = selected_numerator / retained.clamp_min(1e-20).unsqueeze(-1)
            difference = sparse_output - dense_output
            absolute_l2 = difference.norm(dim=-1)
            dense_norm = dense_output.norm(dim=-1)
            relative_l2 = absolute_l2 / dense_norm.clamp_min(1e-12)
            output_cosine = F.cosine_similarity(
                dense_output, sparse_output, dim=-1, eps=1e-12
            )
            row_abs_l2[query_start:query_end, head_start:head_end] = absolute_l2.cpu()
            row_relative_l2[query_start:query_end, head_start:head_end] = relative_l2.cpu()
            row_output_cosine[query_start:query_end, head_start:head_end] = output_cosine.cpu()
            row_dense_output_norm[query_start:query_end, head_start:head_end] = dense_norm.cpu()
            row_selected_output_norm[query_start:query_end, head_start:head_end] = (
                sparse_output.norm(dim=-1).cpu()
            )
            if dense_vectors is not None:
                dense_vectors[query_start:query_end, head_start:head_end] = (
                    dense_output.to(q.dtype).cpu()
                )
                sparse_vectors[query_start:query_end, head_start:head_end] = (
                    sparse_output.to(q.dtype).cpu()
                )

        exact_sum[:, :, head_start:head_end] = group_sum.cpu()
        exact_max[:, :, head_start:head_end] = group_max.cpu()
        jensen_gap_sum[:, :, head_start:head_end] = group_jensen_sum.cpu()
        jensen_gap_max[:, :, head_start:head_end] = group_jensen_max.cpu()
        logit_std_sum[:, :, head_start:head_end] = group_logit_std_sum.cpu()
        max_minus_mean_sum[:, :, head_start:head_end] = (
            group_max_minus_mean_sum.cpu()
        )
        exact_aggregate_log_mass[:, :, head_start:head_end] = (
            group_exact_log_mass.cpu()
        )
        proxy_aggregate_log_mass[:, :, head_start:head_end] = (
            group_proxy_log_mass.cpu()
        )

    query_counts = torch.full((blocks,), block_size, dtype=torch.float32)
    query_counts[-1] = sequence - (blocks - 1) * block_size
    exact_mean = exact_sum / query_counts.view(blocks, 1, 1)
    jensen_gap_mean = jensen_gap_sum / query_counts.view(blocks, 1, 1)
    logit_std_mean = logit_std_sum / query_counts.view(blocks, 1, 1)
    max_minus_mean_mean = max_minus_mean_sum / query_counts.view(blocks, 1, 1)
    query_ids = torch.arange(blocks).view(blocks, 1, 1)
    key_ids = torch.arange(blocks).view(1, blocks, 1)
    fully_visible = (key_ids < query_ids).expand(blocks, blocks, query_heads)
    aggregate_log_mass_gap = torch.zeros_like(exact_aggregate_log_mass)
    aggregate_log_mass_gap[fully_visible] = (
        exact_aggregate_log_mass[fully_visible]
        - proxy_aggregate_log_mass[fully_visible]
    )
    return {
        "exact_block_probability_mean": exact_mean,
        "exact_block_probability_max": exact_max,
        "full_block_jensen_gap_mean": jensen_gap_mean,
        "full_block_jensen_gap_max": jensen_gap_max,
        "full_block_logit_std_mean": logit_std_mean,
        "full_block_max_minus_mean_mean": max_minus_mean_mean,
        "full_block_exact_aggregate_log_mean_mass": exact_aggregate_log_mass,
        "full_block_v1_proxy_aggregate_log_mass": proxy_aggregate_log_mass,
        "full_block_aggregate_log_mass_gap": aggregate_log_mass_gap,
        "fully_visible_block_mask": fully_visible,
        "row_retained_mass": row_retained,
        "row_output_abs_l2": row_abs_l2,
        "row_output_relative_l2": row_relative_l2,
        "row_output_cosine": row_output_cosine,
        "row_dense_output_norm": row_dense_output_norm,
        "row_selected_output_norm": row_selected_output_norm,
        "row_block_mass": row_block_mass,
        "dense_output": dense_vectors,
        "selected_output": sparse_vectors,
        "query_token_counts": query_counts.to(torch.int64),
    }


def _oracle_same_budget(exact_mean, protected, routed):
    query_blocks, key_blocks, heads = exact_mean.shape
    oracle = protected.clone()
    for query_block in range(query_blocks):
        causal = torch.arange(key_blocks) <= query_block
        for head in range(heads):
            budget = int(routed[query_block, :, head].sum())
            if budget == 0:
                continue
            eligible = causal & ~protected[query_block, :, head]
            candidates = torch.nonzero(eligible, as_tuple=False).flatten()
            budget = min(budget, candidates.numel())
            if budget == 0:
                continue
            top = torch.topk(exact_mean[query_block, candidates, head], budget).indices
            oracle[query_block, candidates[top], head] = True
    return oracle


def _validate_oracle(
    oracle, selected, causal, protected, routed, oracle_mask, counts, scores
):
    exact = oracle["exact_block_probability_mean"]
    retained = oracle["row_retained_mass"]
    probability_sum_error = (exact.sum(dim=1) - 1.0).abs().max().item()
    future_values = exact.masked_select(~causal)
    future_mass_max = future_values.abs().max().item() if future_values.numel() else 0.0
    selected_count_mismatch = int(
        (selected.sum(dim=1).to(counts.dtype) != counts).sum().item()
    )
    missing_protected = int((protected & ~selected).sum().item())
    routed_budget_mismatch = int(
        (
            (oracle_mask & ~protected).sum(dim=1)
            != routed.sum(dim=1)
        ).sum().item()
    )
    nonfinite = sum(
        int((~torch.isfinite(oracle[name])).sum().item())
        for name in (
            "exact_block_probability_mean",
            "exact_block_probability_max",
            "full_block_jensen_gap_mean",
            "full_block_jensen_gap_max",
            "full_block_logit_std_mean",
            "full_block_max_minus_mean_mean",
            "full_block_exact_aggregate_log_mean_mass",
            "full_block_v1_proxy_aggregate_log_mass",
            "full_block_aggregate_log_mass_gap",
            "row_retained_mass",
            "row_output_abs_l2",
            "row_output_relative_l2",
            "row_output_cosine",
            "row_dense_output_norm",
            "row_selected_output_norm",
        )
    )
    retained_min = retained.min().item()
    retained_max = retained.max().item()
    proxy_log_mass = oracle["full_block_v1_proxy_aggregate_log_mass"]
    proxy_distribution_errors = []
    for query_block in range(1, proxy_log_mass.shape[0]):
        reconstructed = torch.softmax(
            proxy_log_mass[query_block, :query_block], dim=0
        )
        actual = scores[query_block, :query_block].float()
        actual = actual / actual.sum(dim=0, keepdim=True).clamp_min(1e-20)
        proxy_distribution_errors.append((reconstructed - actual).abs().max())
    proxy_distribution_error = (
        torch.stack(proxy_distribution_errors).max().item()
        if proxy_distribution_errors else 0.0
    )
    fully_visible = oracle["fully_visible_block_mask"]
    aggregate_gap_min = oracle["full_block_aggregate_log_mass_gap"][
        fully_visible
    ].min().item() if fully_visible.any() else 0.0
    sanity = {
        "probability_sum_max_abs_error": probability_sum_error,
        "future_block_mass_max_abs": future_mass_max,
        "row_retained_mass_min": retained_min,
        "row_retained_mass_max": retained_max,
        "selected_count_mismatch": selected_count_mismatch,
        "missing_protected_entries": missing_protected,
        "oracle_routed_budget_mismatch": routed_budget_mismatch,
        "nonfinite_value_count": nonfinite,
        "v1_proxy_full_block_distribution_max_abs_error": proxy_distribution_error,
        "aggregate_log_mass_gap_min": aggregate_gap_min,
    }
    if (
        probability_sum_error > 2e-4
        or future_mass_max > 1e-7
        or retained_min < -2e-4
        or retained_max > 1.0002
        or selected_count_mismatch
        or missing_protected
        or routed_budget_mismatch
        or nonfinite
    ):
        raise RuntimeError(f"dense-oracle sanity check failed: {sanity}")
    return sanity


def _write_token_maps(sample_dir, input_ids, tokenizer, block_size, sample):
    ids = torch.as_tensor(input_ids, dtype=torch.long).tolist()
    token_strings = tokenizer.convert_ids_to_tokens(ids)
    decoded_tokens = tokenizer.batch_decode(
        [[token_id] for token_id in ids],
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    with (sample_dir / "tokens.jsonl").open("w", encoding="utf-8") as output:
        for position, (token_id, token_string, decoded) in enumerate(
            zip(ids, token_strings, decoded_tokens)
        ):
            row = {
                "position": position,
                "block": position // block_size,
                "offset_in_block": position % block_size,
                "token_id": token_id,
                "token": token_string,
                "decoded": decoded,
            }
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (sample_dir / "blocks.jsonl").open("w", encoding="utf-8") as output:
        for start in range(0, len(ids), block_size):
            end = min(start + block_size, len(ids))
            evidence = []
            for evidence_index, item in enumerate(sample.get("evidence_positions", [])):
                span = item.get("token_span")
                if span and start < span[1] and span[0] < end:
                    evidence.append({
                        "evidence_index": evidence_index,
                        "token_span": span,
                        "record": item.get("record"),
                    })
            row = {
                "block": start // block_size,
                "token_start": start,
                "token_end": end,
                "token_count": end - start,
                "is_evidence_block": bool(evidence),
                "evidence": evidence,
                "decoded_text": tokenizer.decode(
                    ids[start:end],
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=False,
                ),
            }
            output.write(json.dumps(row, ensure_ascii=False) + "\n")


class SampleCapture:
    def __init__(
        self, root, sample, capture_input_ids, capture_boundary, tokenizer, args,
        model, layer_sets,
    ):
        self.sample = sample
        self.input_ids = torch.as_tensor(capture_input_ids, dtype=torch.long).tolist()
        self.capture_boundary = capture_boundary
        self.args = args
        self.model = model
        self.layers = layer_sets["capture"]
        self.save_q_layers = layer_sets["q"]
        self.save_pre_q_layers = layer_sets["pre_q"]
        self.save_hidden_layers = layer_sets["hidden"]
        self.save_v_layers = layer_sets["v"]
        self.save_row_mass_layers = layer_sets["row_mass"]
        self.save_output_layers = layer_sets["outputs"]
        self.pending_pre_rope = {}
        self.pending_pre_q = {}
        self.pending_hidden = {}
        self.captured_layers = set()
        self.artifacts = []
        self.sample_dir = root / "samples" / _slug(sample["sample_id"])
        self.sample_dir.mkdir(parents=True)
        (self.sample_dir / "layers").mkdir()
        input_path = self.sample_dir / "input.pt"
        capture_input_path = self.sample_dir / "capture_input_ids.pt"
        torch.save(sample, input_path)
        torch.save(
            torch.as_tensor(self.input_ids, dtype=torch.long),
            capture_input_path,
        )
        public = {key: value for key, value in sample.items() if key != "input_ids"}
        input_json_path = self.sample_dir / "input.json"
        _write_json(input_json_path, public)
        _write_token_maps(
            self.sample_dir, self.input_ids, tokenizer, args.block_size, sample
        )
        self._record_artifact(input_path, "Complete source sample including exact input IDs.")
        self._record_artifact(
            capture_input_path, "Exact token IDs entering this sparse-prefill capture."
        )
        self._record_artifact(input_json_path, "Human-readable source sample metadata.")
        self._record_artifact(
            self.sample_dir / "tokens.jsonl", "Per-token ID, text, block, and offset map."
        )
        self._record_artifact(
            self.sample_dir / "blocks.jsonl", "Decoded text and token range for every block."
        )
        self.block_sink = CsvSink(self.sample_dir / "block_summary.csv", BLOCK_FIELDS)
        self.tile_sink = CsvSink(self.sample_dir / "tile_summary.csv", TILE_FIELDS)
        self.layer_sink = CsvSink(self.sample_dir / "layer_summary.csv", LAYER_FIELDS)
        self.metadata_path = self.sample_dir / "metadata.json"
        self.metadata = {
            "status": "running",
            "capture_only": args.capture_only,
            "sample_id": sample["sample_id"],
            "source_input_sha256": sample.get(
                "input_sha256", input_hash(sample["input_ids"])
            ),
            "capture_input_sha256": input_hash(self.input_ids),
            "source_tokens": len(sample["input_ids"]),
            "capture_tokens": len(self.input_ids),
            "capture_boundary": capture_boundary,
            "captured_layers": [],
            "artifact_contract": {
                "pre_rope_key": "k_proj output reshaped to [tokens, kv_heads, head_dim]",
                "pre_rope_query": "optional q_proj output before RoPE",
                "post_rope_key": "the K tensor consumed by V1",
                "layer_input": "optional hidden states before the decoder layer",
                "mean_k_v1": "the actual BF16 V1 block_mean_k output",
                "dense_truth": (
                    "omitted in capture-only mode; recomputed by block_proxy_study.py"
                    if args.capture_only else
                    "causal token-level softmax over every K token, aggregated only after "
                    "softmax to [query_block, key_block, query_head]"
                ),
                "selected_output": (
                    "omitted in capture-only mode; reconstructible from saved Q/K/V"
                    if args.capture_only else
                    "the same dense probabilities restricted to V1-selected blocks and "
                    "renormalized; V affects only this error audit, never routing"
                ),
            },
        }
        _write_json(self.metadata_path, self.metadata)

    def _record_artifact(self, path, description):
        self.artifacts.append({
            "path": str(path.relative_to(self.sample_dir)),
            "bytes": path.stat().st_size,
            "sha256": _source_hash(path),
            "description": description,
        })

    def _save(self, path, payload, description):
        torch.save(payload, path)
        self._record_artifact(path, description)

    def pre_rope_hook(self, layer, output):
        if layer not in self.layers:
            return
        sequence = output.shape[1]
        kv_heads = self.model.config.num_key_value_heads
        head_dim = output.shape[-1] // kv_heads
        self.pending_pre_rope[layer] = (
            output.detach()[0].reshape(sequence, kv_heads, head_dim).to("cpu")
        )

    def pre_rope_q_hook(self, layer, output):
        if layer not in self.save_pre_q_layers:
            return
        sequence = output.shape[1]
        query_heads = self.model.config.num_attention_heads
        head_dim = output.shape[-1] // query_heads
        self.pending_pre_q[layer] = (
            output.detach()[0].reshape(sequence, query_heads, head_dim).to("cpu")
        )

    def layer_input_hook(self, layer, hidden):
        if layer in self.save_hidden_layers:
            self.pending_hidden[layer] = hidden.detach()[0].to("cpu")

    def capture_layer(
        self, layer, q, k, v, mean_k, scores, indices, counts, selected, scale
    ):
        if layer not in self.layers:
            return
        print(
            f"[v1-diag] {self.sample['sample_id']} layer {layer}: " +
            (
                "saving attention features and V1 route"
                if self.args.capture_only else
                "saving K structure and computing dense oracle"
            ),
            flush=True,
        )
        layer_dir = self.sample_dir / "layers" / f"layer_{layer:02d}"
        layer_dir.mkdir()
        pre_key = self.pending_pre_rope.pop(layer)
        post_key = k.detach()[0].to("cpu")
        actual_mean = mean_k.detach()[0].to("cpu")
        score_cpu = scores.detach()[0].to("cpu")
        selected_cpu = selected.detach()[0].to("cpu")
        indices_cpu = indices.detach()[0].to("cpu")
        counts_cpu = counts.detach()[0].to("cpu")
        causal, protected, routed = _protected_masks(
            selected_cpu,
            self.args.sink_blocks,
            self.args.window_blocks,
            self.args.last_full_blocks,
        )

        if self.args.save_raw_k:
            self._save(
                layer_dir / "key_pre_rope.pt", pre_key,
                "Raw projected K before RoPE [tokens, kv_heads, head_dim].",
            )
            self._save(
                layer_dir / "key_post_rope.pt", post_key,
                "Raw K consumed by V1 after RoPE [tokens, kv_heads, head_dim].",
            )
        if layer in self.save_q_layers:
            self._save(
                layer_dir / "query_post_rope.pt", q.detach()[0].to("cpu"),
                "Post-RoPE Q [tokens, query_heads, head_dim].",
            )
        if layer in self.save_pre_q_layers:
            self._save(
                layer_dir / "query_pre_rope.pt", self.pending_pre_q.pop(layer),
                "Q before RoPE [tokens, query_heads, head_dim].",
            )
        if layer in self.save_hidden_layers:
            self._save(
                layer_dir / "layer_input.pt", self.pending_hidden.pop(layer),
                "Decoder-layer input [tokens, hidden_size] before layer norms.",
            )
        if layer in self.save_v_layers:
            self._save(
                layer_dir / "value.pt", v.detach()[0].to("cpu"),
                "V [tokens, kv_heads, head_dim]; V is not used by V1 routing.",
            )

        pre_stats = _key_structure(pre_key, self.args.block_size)
        post_stats = _key_structure(post_key, self.args.block_size)
        structure = {
            "pre_rope": pre_stats,
            "post_rope": post_stats,
            "pre_post_mean_cosine": F.cosine_similarity(
                pre_stats["mean"], post_stats["mean"], dim=-1, eps=1e-12
            ),
            "v1_mean_max_abs_error": (
                actual_mean.float() - post_stats["mean"]
            ).abs().amax(dim=-1),
            "v1_mean_relative_l2_error": (
                actual_mean.float() - post_stats["mean"]
            ).norm(dim=-1) / post_stats["mean"].norm(dim=-1).clamp_min(1e-12),
        }
        self._save(
            layer_dir / "key_structure.pt", structure,
            "Per-block FP32 means, diagonal variances, and sequence/coherence summaries.",
        )
        self._save(
            layer_dir / "mean_k_v1.pt", actual_mean,
            "Exact mean-pooled BF16 K used by V1.",
        )

        if self.args.capture_only:
            self._save(
                layer_dir / "v1_route.pt",
                {
                    "v1_proxy_score": score_cpu,
                    "selected_mask": selected_cpu,
                    "protected_mask": protected,
                    "routed_mask": routed,
                    "causal_mask": causal,
                    "indices": indices_cpu,
                    "counts": counts_cpu,
                    "alpha": self.args.alpha,
                    "scale": scale,
                },
                "V1 routing without a dense oracle; offline analysis uses raw Q/K/V.",
            )
            self.captured_layers.add(layer)
            self.metadata["captured_layers"] = sorted(self.captured_layers)
            _write_json(self.metadata_path, self.metadata)
            print(
                f"[v1-diag] {self.sample['sample_id']} layer {layer}: "
                "Q/K/V and route capture complete",
                flush=True,
            )
            return

        oracle = _dense_oracle(
            q,
            k,
            v,
            mean_k,
            selected,
            scale,
            self.args.block_size,
            self.args.query_chunk,
            layer in self.save_row_mass_layers,
            layer in self.save_output_layers,
        )
        exact_mean = oracle["exact_block_probability_mean"]
        oracle_mask = _oracle_same_budget(exact_mean, protected, routed)
        oracle["sanity"] = _validate_oracle(
            oracle,
            selected_cpu,
            causal,
            protected,
            routed,
            oracle_mask,
            counts_cpu,
            score_cpu,
        )
        route_payload = {
            "v1_proxy_score": score_cpu,
            "selected_mask": selected_cpu,
            "protected_mask": protected,
            "routed_mask": routed,
            "causal_mask": causal,
            "oracle_same_budget_mask": oracle_mask,
            "indices": indices_cpu,
            "counts": counts_cpu,
            "alpha": self.args.alpha,
            "scale": scale,
        }
        self._save(
            layer_dir / "v1_route.pt", route_payload,
            "V1 proxy score, selected/protected/routed masks, and dense same-budget oracle.",
        )
        dense_payload = {
            key: value for key, value in oracle.items()
            if key not in {"row_block_mass", "dense_output", "selected_output"}
        }
        self._save(
            layer_dir / "dense_block_oracle.pt", dense_payload,
            "Dense block probability truth plus per-row retained mass and output errors.",
        )
        if oracle["row_block_mass"] is not None:
            self._save(
                layer_dir / "dense_row_block_mass.pt", oracle["row_block_mass"],
                "FP16 dense mass [query_token, query_head, key_block].",
            )
        if oracle["dense_output"] is not None:
            self._save(
                layer_dir / "dense_output.pt", oracle["dense_output"],
                "Dense attention output [tokens, query_heads, head_dim].",
            )
            self._save(
                layer_dir / "selected_output.pt", oracle["selected_output"],
                "Exact attention restricted to V1-selected blocks.",
            )

        self._write_summaries(
            layer, pre_stats, post_stats, structure, score_cpu, selected_cpu,
            causal, protected, routed, oracle_mask, oracle,
        )
        self.captured_layers.add(layer)
        self.metadata["captured_layers"] = sorted(self.captured_layers)
        _write_json(self.metadata_path, self.metadata)
        print(
            f"[v1-diag] {self.sample['sample_id']} layer {layer}: complete",
            flush=True,
        )

    def _write_summaries(
        self, layer, pre, post, structure, score, selected, causal, protected,
        routed, oracle_mask, oracle,
    ):
        sequence = len(self.input_ids)
        blocks, _, query_heads = selected.shape
        kv_heads = post["mean"].shape[1]
        group = query_heads // kv_heads
        exact = oracle["exact_block_probability_mean"]
        jensen_gap = oracle["full_block_jensen_gap_mean"]
        jensen_gap_row_max = oracle["full_block_jensen_gap_max"]
        query_counts = oracle["query_token_counts"].float()
        row_retained = oracle["row_retained_mass"]
        row_abs = oracle["row_output_abs_l2"]
        row_relative = oracle["row_output_relative_l2"]
        row_cosine = oracle["row_output_cosine"]

        for block in range(blocks):
            token_start = block * self.args.block_size
            token_end = min(token_start + self.args.block_size, sequence)
            for kv_head in range(kv_heads):
                head_start = kv_head * group
                head_end = head_start + group
                possible = causal[:, block, head_start:head_end]
                chosen = selected[:, block, head_start:head_end]
                routed_here = routed[:, block, head_start:head_end]
                exact_here = exact[:, block, head_start:head_end]
                gap_here = jensen_gap[:, block, head_start:head_end]
                gap_row_max_here = jensen_gap_row_max[
                    :, block, head_start:head_end
                ]
                proxy_here = score[:, block, head_start:head_end]
                fully_visible = (
                    torch.arange(blocks) > block
                )[:, None].expand(blocks, group)
                gap_observation_weights = (
                    fully_visible * query_counts[:, None]
                )
                exact_gap_weights = exact_here * gap_observation_weights
                row = {
                    "sample_id": self.sample["sample_id"],
                    "layer": layer,
                    "block": block,
                    "token_start": token_start,
                    "token_end": token_end,
                    "token_count": token_end - token_start,
                    "kv_head": kv_head,
                    "query_head_start": head_start,
                    "query_head_end": head_end,
                    "pre_post_mean_cos": structure["pre_post_mean_cosine"][block, kv_head].item(),
                    "v1_mean_max_abs_error": structure["v1_mean_max_abs_error"][block, kv_head].item(),
                    "v1_mean_relative_l2_error": structure["v1_mean_relative_l2_error"][block, kv_head].item(),
                    "causally_available_tile_heads": int(possible.sum()),
                    "selected_tile_heads": int(chosen.sum()),
                    "routed_tile_heads": int(routed_here.sum()),
                    "selected_rate": chosen.sum().item() / max(1, possible.sum().item()),
                    "exact_attention_mass_sum": (
                        exact_here * query_counts[:, None]
                    ).sum().item(),
                    "missed_attention_mass_sum": (
                        exact_here * ~chosen * query_counts[:, None]
                    ).sum().item(),
                    "v1_proxy_score_sum": proxy_here.sum().item(),
                    "mean_pool_jensen_gap_mean": (
                        (gap_here * gap_observation_weights).sum().item()
                        / max(1, gap_observation_weights.sum().item())
                    ),
                    "mean_pool_jensen_gap_row_max": (
                        gap_row_max_here * fully_visible
                    ).max().item(),
                    "mean_pool_jensen_gap_exact_mass_weighted": (
                        (gap_here * exact_gap_weights).sum().item()
                        / max(1e-20, exact_gap_weights.sum().item())
                    ),
                }
                for prefix, stats in (("pre", pre), ("post", post)):
                    for name in (
                        "mean_norm", "token_norm_mean", "token_norm_std", "centered_rms",
                        "relative_mean_reconstruction_mse", "directional_concentration",
                        "token_to_mean_cos_mean", "token_to_mean_cos_std",
                        "token_to_mean_cos_min", "adjacent_cos_mean", "first_last_cos",
                        "half_mean_cos",
                    ):
                        row[f"{prefix}_{name}"] = stats[name][block, kv_head].item()
                self.block_sink.write(row)

        proxy_cosines = []
        remote_proxy_cosines = []
        for query_block in range(blocks):
            start = query_block * self.args.block_size
            end = min(start + self.args.block_size, sequence)
            for head in range(query_heads):
                chosen = selected[query_block, :, head]
                fixed = protected[query_block, :, head]
                route = routed[query_block, :, head]
                oracle_chosen = oracle_mask[query_block, :, head]
                exact_distribution = exact[query_block, :, head]
                gap_distribution = jensen_gap[query_block, :, head]
                gap_row_max_distribution = jensen_gap_row_max[
                    query_block, :, head
                ]
                exact_retained = (exact_distribution * chosen).sum().item()
                oracle_retained = (exact_distribution * oracle_chosen).sum().item()
                row_slice = slice(start, end)
                history = torch.arange(blocks) <= query_block
                proxy_distribution = score[query_block, :, head].float()
                exact_history = exact_distribution * history
                exact_history = exact_history / exact_history.sum().clamp_min(1e-12)
                proxy_cosine = F.cosine_similarity(
                    proxy_distribution[None], exact_history[None], dim=-1, eps=1e-12
                ).item()
                proxy_cosines.append(proxy_cosine)
                remote = history & ~fixed
                if remote.any():
                    remote_candidates = torch.nonzero(remote, as_tuple=False).flatten()
                    proxy_remote = proxy_distribution * remote
                    exact_remote = exact_distribution * remote
                    proxy_remote = proxy_remote / proxy_remote.sum().clamp_min(1e-12)
                    exact_remote = exact_remote / exact_remote.sum().clamp_min(1e-12)
                    proxy_remote_cosine = F.cosine_similarity(
                        proxy_remote[None], exact_remote[None], dim=-1, eps=1e-12
                    ).item()
                    proxy_remote_l1 = (proxy_remote - exact_remote).abs().sum().item()
                    proxy_remote_top = int(
                        remote_candidates[proxy_distribution[remote_candidates].argmax()]
                    )
                    exact_remote_top = int(
                        remote_candidates[exact_distribution[remote_candidates].argmax()]
                    )
                    remote_proxy_cosines.append(proxy_remote_cosine)
                else:
                    proxy_remote_cosine = float("nan")
                    proxy_remote_l1 = float("nan")
                    proxy_remote_top = -1
                    exact_remote_top = -1
                remote_exact_weight = exact_distribution * remote
                missed_remote = remote & ~chosen
                missed_exact_weight = exact_distribution * missed_remote
                gap_exact_weighted = (
                    (gap_distribution * remote_exact_weight).sum().item()
                    / max(1e-20, remote_exact_weight.sum().item())
                )
                gap_missed_weighted = (
                    (gap_distribution * missed_exact_weight).sum().item()
                    / max(1e-20, missed_exact_weight.sum().item())
                )
                gap_tile_mean_max = (
                    gap_distribution[remote].max().item() if remote.any() else 0.0
                )
                gap_row_max = (
                    gap_row_max_distribution[remote].max().item()
                    if remote.any() else 0.0
                )
                union = route | (oracle_chosen & ~fixed)
                intersection = route & oracle_chosen
                route_overlap = intersection.sum().item() / max(1, union.sum().item())
                self.tile_sink.write({
                    "sample_id": self.sample["sample_id"],
                    "layer": layer,
                    "query_block": query_block,
                    "query_token_start": start,
                    "query_token_end": end,
                    "query_token_count": end - start,
                    "query_head": head,
                    "kv_head": head // group,
                    "selected_blocks": int(chosen.sum()),
                    "protected_blocks": int(fixed.sum()),
                    "routed_blocks": int(route.sum()),
                    "causal_blocks": query_block + 1,
                    "exact_retained_mass_mean": exact_retained,
                    "exact_retained_mass_min": row_retained[row_slice, head].min().item(),
                    "exact_omitted_mass_mean": 1.0 - exact_retained,
                    "exact_omitted_mass_max": 1.0 - row_retained[row_slice, head].min().item(),
                    "oracle_same_budget_retained_mass": oracle_retained,
                    "oracle_route_overlap": route_overlap,
                    "proxy_exact_cosine": proxy_cosine,
                    "proxy_exact_l1": (proxy_distribution - exact_history).abs().sum().item(),
                    "proxy_top_block": int(proxy_distribution.argmax()),
                    "exact_top_block": int(exact_history.argmax()),
                    "proxy_exact_remote_cosine": proxy_remote_cosine,
                    "proxy_exact_remote_l1": proxy_remote_l1,
                    "proxy_remote_top_block": proxy_remote_top,
                    "exact_remote_top_block": exact_remote_top,
                    "mean_pool_jensen_gap_exact_mass_weighted": gap_exact_weighted,
                    "mean_pool_jensen_gap_missed_mass_weighted": gap_missed_weighted,
                    "mean_pool_jensen_gap_tile_mean_max": gap_tile_mean_max,
                    "mean_pool_jensen_gap_row_max": gap_row_max,
                    "output_abs_l2_mean": row_abs[row_slice, head].mean().item(),
                    "output_abs_l2_max": row_abs[row_slice, head].max().item(),
                    "output_relative_l2_mean": row_relative[row_slice, head].mean().item(),
                    "output_relative_l2_max": row_relative[row_slice, head].max().item(),
                    "output_cosine_mean": row_cosine[row_slice, head].mean().item(),
                    "output_cosine_min": row_cosine[row_slice, head].min().item(),
                })

        selected_ratio = selected.sum().item() / causal.sum().item()
        remotely_available = (causal & ~protected).sum().item()
        routed_ratio = routed.sum().item() / max(1, remotely_available)
        oracle_retained = (exact * oracle_mask).sum(dim=1)
        oracle_retained_mean = (
            oracle_retained * query_counts[:, None]
        ).sum().item() / (sequence * query_heads)
        query_ids = torch.arange(blocks).view(blocks, 1, 1)
        key_ids = torch.arange(blocks).view(1, blocks, 1)
        fully_visible = (key_ids < query_ids).expand_as(exact)
        gap_weights = exact * fully_visible * query_counts.view(blocks, 1, 1)
        gap_weighted_mean = (
            (jensen_gap * gap_weights).sum().item()
            / max(1e-20, gap_weights.sum().item())
        )
        self.layer_sink.write({
            "sample_id": self.sample["sample_id"],
            "layer": layer,
            "sequence_tokens": sequence,
            "blocks": blocks,
            "query_heads": query_heads,
            "kv_heads": kv_heads,
            "selected_block_ratio": selected_ratio,
            "routed_block_ratio": routed_ratio,
            "exact_retained_mass_mean": row_retained.mean().item(),
            "exact_retained_mass_p05": _quantile(row_retained, 0.05),
            "exact_retained_mass_min": row_retained.min().item(),
            "oracle_same_budget_retained_mass_mean": oracle_retained_mean,
            "output_relative_l2_mean": row_relative.mean().item(),
            "output_relative_l2_p95": _quantile(row_relative, 0.95),
            "output_relative_l2_max": row_relative.max().item(),
            "output_cosine_mean": row_cosine.mean().item(),
            "proxy_exact_cosine_mean": sum(proxy_cosines) / len(proxy_cosines),
            "proxy_exact_remote_cosine_mean": (
                sum(remote_proxy_cosines) / len(remote_proxy_cosines)
                if remote_proxy_cosines else float("nan")
            ),
            "mean_pool_jensen_gap_exact_mass_weighted": gap_weighted_mean,
            "mean_pool_jensen_gap_tile_mean_max": jensen_gap.max().item(),
            "mean_pool_jensen_gap_row_max": jensen_gap_row_max.max().item(),
            "mean_pool_relative_l2_error_max": structure[
                "v1_mean_relative_l2_error"
            ].max().item(),
        })

    def finish(self):
        missing = sorted(self.layers - self.captured_layers)
        if missing:
            raise RuntimeError(f"capture callback did not observe layers {missing}")
        if self.pending_pre_rope:
            raise RuntimeError(
                f"unconsumed pre-RoPE keys for layers {sorted(self.pending_pre_rope)}"
            )
        if self.pending_pre_q or self.pending_hidden:
            raise RuntimeError(
                "unconsumed feature hooks: "
                f"Q={sorted(self.pending_pre_q)}, "
                f"hidden={sorted(self.pending_hidden)}"
            )
        self.block_sink.close()
        self.tile_sink.close()
        self.layer_sink.close()
        self._record_artifact(
            self.sample_dir / "block_summary.csv",
            "Per-layer/key-block structural and routing summary.",
        )
        self._record_artifact(
            self.sample_dir / "tile_summary.csv",
            "Per-layer/query-tile/head routing and output-error summary.",
        )
        self._record_artifact(
            self.sample_dir / "layer_summary.csv",
            "Per-layer aggregate diagnostic summary.",
        )
        with (self.sample_dir / "artifacts.jsonl").open("w", encoding="utf-8") as output:
            for artifact in self.artifacts:
                output.write(json.dumps(artifact, ensure_ascii=False) + "\n")
        self.metadata["status"] = "complete"
        self.metadata["captured_layers"] = sorted(self.captured_layers)
        self.metadata["artifact_count"] = len(self.artifacts)
        self.metadata["artifact_bytes"] = sum(item["bytes"] for item in self.artifacts)
        _write_json(self.metadata_path, self.metadata)

    def fail(self, error):
        self.metadata["status"] = "failed"
        self.metadata["error"] = str(error)
        self.metadata["traceback"] = traceback.format_exc()
        _write_json(self.metadata_path, self.metadata)
        for sink in (self.block_sink, self.tile_sink, self.layer_sink):
            if not sink.file.closed:
                sink.close()


def main():
    args = arguments()
    if args.alpha < 0:
        raise ValueError("--alpha must be non-negative")
    if args.block_size != 128:
        raise ValueError("the pinned V1 kernels require --block-size 128")
    if args.query_chunk <= 0:
        raise ValueError("--query-chunk must be positive")
    if args.query_chunk % args.block_size:
        raise ValueError("--query-chunk must be a multiple of the 128-token block size")
    if (args.out / "metadata.json").exists():
        raise FileExistsError(f"refusing to overwrite existing capture: {args.out}")
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "samples").mkdir()

    inputs = torch.load(args.inputs, map_location="cpu", weights_only=False)
    for sample in inputs:
        if input_hash(sample["input_ids"]) != sample["input_sha256"]:
            raise ValueError(f"stored input hash mismatch for {sample['sample_id']}")
    samples = _select_samples(inputs, args)

    print(
        f"[v1-diag] loading {args.model}; selected {len(samples)} sample(s)",
        flush=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    backend = AttentionBackend(
        alpha=args.alpha,
        sink_blocks=args.sink_blocks,
        window_blocks=args.window_blocks,
        last_full_blocks=args.last_full_blocks,
        block_size=args.block_size,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map="cuda",
    ).eval()
    model.requires_grad_(False)
    layer_count = model.config.num_hidden_layers
    layer_sets = {
        "capture": _layer_set(args.layers, layer_count),
        "q": _layer_set(args.save_q_layers, layer_count, allow_empty=True),
        "pre_q": _layer_set(
            args.save_pre_rope_q_layers, layer_count, allow_empty=True
        ),
        "hidden": _layer_set(
            args.save_layer_input_layers, layer_count, allow_empty=True
        ),
        "v": _layer_set(args.save_v_layers, layer_count, allow_empty=True),
        "row_mass": _layer_set(
            args.save_row_block_mass_layers, layer_count, allow_empty=True
        ),
        "outputs": _layer_set(
            args.save_output_vector_layers, layer_count, allow_empty=True
        ),
    }
    for name in ("q", "pre_q", "hidden", "v", "row_mass", "outputs"):
        if not layer_sets[name] <= layer_sets["capture"]:
            raise ValueError(f"{name} layers must be included in --layers")

    metadata = {
        "status": "running",
        "purpose": (
            "V1 Q/K/V and routing capture for offline block-size study"
            if args.capture_only else
            "V1-only block structure and dense-routing ground-truth capture"
        ),
        "arguments": vars(args),
        "input_file": str(args.inputs.resolve()),
        "input_file_sha256": _source_hash(args.inputs),
        "selected_sample_ids": [sample["sample_id"] for sample in samples],
        "selected_sample_count": len(samples),
        "model_commit": model.config._commit_hash,
        "model_config": model.config.to_dict(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "triton": triton.__version__,
        "gpu": _gpu_metadata(),
        "source_hashes": {
            "v1_block_diagnostics.py": _source_hash(__file__),
            "attention.py": _source_hash(Path(__file__).with_name("attention.py")),
            "upstream/flashprefill_native_forward.py": _source_hash(
                Path(__file__).with_name("upstream") / "flashprefill_native_forward.py"
            ),
        },
        "upstream_flashprefill_commit": "baa612047433a992a00d07dc178205eed065ae14",
        "timing_warning": (
            "This run includes dense oracle computation, CPU copies, and disk writes. "
            "It is not a latency benchmark and must not enter prefill timing reports."
        ),
    }
    _write_json(args.out / "metadata.json", metadata)

    current_capture = {"value": None}
    handles = []
    for layer_number, layer_module in enumerate(model.model.layers):
        if layer_number not in layer_sets["capture"]:
            continue

        def hook(_module, _inputs, output, number=layer_number):
            current_capture["value"].pre_rope_hook(number, output)

        handles.append(layer_module.self_attn.k_proj.register_forward_hook(hook))
        if layer_number in layer_sets["pre_q"]:
            def q_hook(_module, _inputs, output, number=layer_number):
                current_capture["value"].pre_rope_q_hook(number, output)

            handles.append(
                layer_module.self_attn.q_proj.register_forward_hook(q_hook)
            )
        if layer_number in layer_sets["hidden"]:
            def hidden_hook(_module, inputs, number=layer_number):
                current_capture["value"].layer_input_hook(number, inputs[0])

            handles.append(layer_module.register_forward_pre_hook(hidden_hook))

    try:
        backend.configure("fp_v1", args.alpha)
        for sample in samples:
            capture_input_ids = sample["input_ids"]
            capture_boundary = "full_saved_prompt"
            if sample.get("task") == "ruler" and not args.include_ruler_final_token:
                capture_input_ids = capture_input_ids[:-1]
                capture_boundary = "upstream_ruler_sparse_prefill_excludes_final_prompt_token"
            print(
                f"[v1-diag] sample {sample['sample_id']}: "
                f"{len(capture_input_ids)} captured tokens, capture start",
                flush=True,
            )
            capture = SampleCapture(
                args.out, sample, capture_input_ids, capture_boundary, tokenizer,
                args, model, layer_sets,
            )
            current_capture["value"] = capture
            backend.start_capture(capture.capture_layer)
            try:
                input_ids = torch.as_tensor(
                    capture_input_ids, dtype=torch.long, device="cuda"
                ).unsqueeze(0)
                with torch.inference_mode(), sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                    output = model(
                        input_ids=input_ids,
                        use_cache=False,
                        logits_to_keep=1,
                    )
                del output, input_ids
                torch.cuda.synchronize()
                capture.finish()
                print(
                    f"[v1-diag] sample {sample['sample_id']}: complete; "
                    f"artifacts={capture.metadata['artifact_count']}, "
                    f"bytes={capture.metadata['artifact_bytes']}",
                    flush=True,
                )
            except Exception as error:
                capture.fail(error)
                raise
            finally:
                backend.finish_capture()
                current_capture["value"] = None
        metadata["status"] = "complete"
        print(f"[v1-diag] run complete: {args.out}", flush=True)
    except Exception as error:
        metadata["status"] = "failed"
        metadata["error"] = str(error)
        metadata["traceback"] = traceback.format_exc()
        raise
    finally:
        backend.finish_capture()
        for handle in handles:
            handle.remove()
        _write_json(args.out / "metadata.json", metadata)


if __name__ == "__main__":
    main()
