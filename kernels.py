"""GPU tensor paths for descriptors, selectors, mean correction, and accounting."""

from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F


BLOCK_SIZE = 128


@dataclass
class BlockDescriptors:
    mean_k: torch.Tensor
    mean_v: torch.Tensor
    var_k: torch.Tensor
    sigma_v: torch.Tensor
    counts: torch.Tensor


def block_descriptors(k: torch.Tensor, v: torch.Tensor, block_size: int = BLOCK_SIZE):
    """Compute valid-token block descriptors in FP32 for each KV head."""
    batch, sequence, kv_heads, head_dim = k.shape
    blocks = math.ceil(sequence / block_size)
    padded_length = blocks * block_size
    padding = padded_length - sequence
    if padding:
        k = F.pad(k, (0, 0, 0, 0, 0, padding))
        v = F.pad(v, (0, 0, 0, 0, 0, padding))
    k_blocks = k.reshape(batch, blocks, block_size, kv_heads, head_dim)
    v_blocks = v.reshape(batch, blocks, block_size, kv_heads, head_dim)
    counts = torch.full((blocks,), block_size, dtype=torch.int32, device=k.device)
    counts[-1] = sequence - (blocks - 1) * block_size
    denominator = counts.to(torch.float32).view(1, blocks, 1, 1)

    k_float = k_blocks.float()
    mean_k = k_float.sum(dim=2) / denominator
    var_k = (k_float.square().sum(dim=2) / denominator - mean_k.square()).clamp_min_(0)
    del k_float

    v_float = v_blocks.float()
    mean_v = v_float.sum(dim=2) / denominator
    var_v = (v_float.square().sum(dim=2) / denominator - mean_v.square()).clamp_min_(0)
    sigma_v = var_v.sum(dim=-1).sqrt_()
    return BlockDescriptors(mean_k, mean_v, var_k, sigma_v, counts)


def selector_scores(
    q: torch.Tensor,
    descriptors: BlockDescriptors,
    scale: float,
    selector: str,
    block_size: int = BLOCK_SIZE,
    chunk_tiles: int = 8,
):
    """Return [B, query-block, key-block, Q-head] scores.

    Normalization is performed for each real query row over full historical
    blocks, followed by an arithmetic mean over the real rows in its tile.
    """
    if selector not in {"mean_balanced", "cgf_mean", "dispersion_mean"}:
        raise ValueError(f"unsupported selector: {selector}")
    batch, sequence, q_heads, _ = q.shape
    blocks = descriptors.mean_k.shape[1]
    kv_heads = descriptors.mean_k.shape[2]
    group = q_heads // kv_heads
    mean_k = descriptors.mean_k.repeat_interleave(group, dim=2)
    var_k = descriptors.var_k.repeat_interleave(group, dim=2)
    sigma_v = descriptors.sigma_v.repeat_interleave(group, dim=2).permute(0, 2, 1)
    log_counts = descriptors.counts.to(torch.float32).log()
    key_ids = torch.arange(blocks, device=q.device)
    scores = torch.zeros(
        batch,
        blocks,
        blocks,
        q_heads,
        dtype=torch.float32,
        device=q.device,
    )

    for first_tile in range(0, blocks, chunk_tiles):
        tile_count = min(chunk_tiles, blocks - first_tile)
        row_start = first_tile * block_size
        row_end = min(sequence, (first_tile + tile_count) * block_size)
        q_chunk = q[:, row_start:row_end].float()
        row_tiles = torch.arange(row_start, row_end, device=q.device) // block_size
        visible = key_ids[None, :] < row_tiles[:, None]
        ell0 = torch.einsum("brhd,bkhd->brhk", q_chunk, mean_k)
        ell0.mul_(scale).add_(log_counts.view(1, 1, 1, blocks))

        if selector == "mean_balanced":
            logits = ell0
            dispersion = None
        else:
            u = torch.einsum("brhd,bkhd->brhk", q_chunk.square(), var_k)
            u.mul_(scale * scale).clamp_min_(0)
            logits = ell0 + 0.5 * u
            dispersion = u.sqrt_() if selector == "dispersion_mean" else None

        masked = logits.masked_fill(~visible.view(1, -1, 1, blocks), float("-inf"))
        normalizer = torch.logsumexp(masked, dim=-1, keepdim=True)
        probability = torch.where(
            visible.view(1, -1, 1, blocks),
            torch.exp(masked - normalizer),
            0.0,
        )
        if dispersion is not None:
            contribution = probability * dispersion * sigma_v[:, None]
        else:
            contribution = probability

        local_tiles = row_tiles - first_tile
        aggregate = torch.zeros(
            batch,
            tile_count,
            q_heads,
            blocks,
            dtype=torch.float32,
            device=q.device,
        )
        aggregate.index_add_(1, local_tiles, contribution)
        row_counts = torch.bincount(local_tiles, minlength=tile_count).to(torch.float32)
        aggregate.div_(row_counts.view(1, tile_count, 1, 1))
        scores[:, first_tile:first_tile + tile_count] = aggregate.permute(0, 1, 3, 2)
    return scores


def protected_selection(
    scores: torch.Tensor,
    alpha: float,
    sink_blocks: int,
    window_blocks: int,
    last_full_blocks: int,
):
    """Apply the specified threshold and common protected regions."""
    batch, query_blocks, key_blocks, _ = scores.shape
    query_ids = torch.arange(query_blocks, device=scores.device).view(1, query_blocks, 1, 1)
    key_ids = torch.arange(key_blocks, device=scores.device).view(1, 1, key_blocks, 1)
    causal = key_ids <= query_ids
    if alpha == 0:
        return causal.expand_as(scores).clone()
    maximum = scores.amax(dim=2, keepdim=True)
    routed = (scores > 0) & (scores >= alpha * maximum)
    distance = query_ids - key_ids
    protected = (
        (key_ids < sink_blocks)
        | ((distance >= 0) & (distance < window_blocks))
        | (query_ids >= query_blocks - last_full_blocks)
    )
    return (routed | protected) & causal


def indices_from_mask(selected: torch.Tensor):
    batch, query_blocks, key_blocks, heads = selected.shape
    key_ids = torch.arange(key_blocks, device=selected.device).view(1, 1, key_blocks, 1)
    indices = key_ids.expand(batch, query_blocks, key_blocks, heads)
    indices = indices.masked_fill(~selected, key_blocks).sort(dim=2).values
    counts = selected.sum(dim=2).to(torch.int32)
    return indices.contiguous(), counts.contiguous()


def mask_from_indices(indices: torch.Tensor, counts: torch.Tensor):
    batch, query_blocks, key_blocks, heads = indices.shape
    positions = torch.arange(key_blocks, device=indices.device).view(1, 1, key_blocks, 1)
    valid = positions < counts.unsqueeze(2)
    full = torch.zeros(
        batch,
        query_blocks,
        key_blocks + 1,
        heads,
        dtype=torch.bool,
        device=indices.device,
    )
    full.scatter_(2, indices, valid)
    return full[:, :, :key_blocks]


def mean_tail(
    q: torch.Tensor,
    descriptors: BlockDescriptors,
    selected: torch.Tensor,
    scale: float,
    block_size: int = BLOCK_SIZE,
    chunk_tiles: int = 8,
):
    """Compute the unselected full-history block-mean path in FP32."""
    batch, sequence, q_heads, head_dim = q.shape
    blocks = descriptors.mean_k.shape[1]
    group = q_heads // descriptors.mean_k.shape[2]
    mean_k = descriptors.mean_k.repeat_interleave(group, dim=2)
    mean_v = descriptors.mean_v.repeat_interleave(group, dim=2)
    log_counts = descriptors.counts.to(torch.float32).log()
    selected_by_head = selected.permute(0, 1, 3, 2)
    key_ids = torch.arange(blocks, device=q.device)
    output = torch.zeros(
        batch,
        sequence,
        q_heads,
        head_dim,
        dtype=torch.float32,
        device=q.device,
    )
    lse = torch.full(
        (batch, sequence, q_heads),
        float("-inf"),
        dtype=torch.float32,
        device=q.device,
    )

    for first_tile in range(0, blocks, chunk_tiles):
        tile_count = min(chunk_tiles, blocks - first_tile)
        row_start = first_tile * block_size
        row_end = min(sequence, (first_tile + tile_count) * block_size)
        row_tiles = torch.arange(row_start, row_end, device=q.device) // block_size
        visible = key_ids[None, :] < row_tiles[:, None]
        selected_rows = selected_by_head[:, row_tiles]
        tail = visible.view(1, -1, 1, blocks) & ~selected_rows
        logits = torch.einsum(
            "brhd,bkhd->brhk",
            q[:, row_start:row_end].float(),
            mean_k,
        )
        logits.mul_(scale).add_(log_counts.view(1, 1, 1, blocks))
        masked = logits.masked_fill(~tail, float("-inf"))
        chunk_lse = torch.logsumexp(masked, dim=-1)
        weights = torch.where(tail, torch.exp(masked - chunk_lse.unsqueeze(-1)), 0.0)
        output[:, row_start:row_end] = torch.einsum("brhk,bkhd->brhd", weights, mean_v)
        lse[:, row_start:row_end] = chunk_lse
    return output, lse


def merge_exact_mean(
    exact_output: torch.Tensor,
    exact_lse: torch.Tensor,
    mean_output: torch.Tensor,
    mean_lse: torch.Tensor,
):
    total_lse = torch.logaddexp(exact_lse, mean_lse)
    exact_weight = torch.exp(exact_lse - total_lse)
    mean_weight = torch.exp(mean_lse - total_lse)
    output = (
        exact_output.float() * exact_weight.unsqueeze(-1)
        + mean_output * mean_weight.unsqueeze(-1)
    )
    return output.to(exact_output.dtype), total_lse


def selection_accounting(
    selected: torch.Tensor,
    sequence: int,
    q_tile_size: int | None,
    k_tile_size: int | None,
    method: str,
    score_k_tile_size: int | None,
    block_size: int = BLOCK_SIZE,
):
    """Return scalar GPU tensors; callers synchronize only after profile forward."""
    batch, blocks, _, heads = selected.shape
    counts = torch.full((blocks,), block_size, dtype=torch.int64, device=selected.device)
    counts[-1] = sequence - (blocks - 1) * block_size
    query_ids = torch.arange(blocks, device=selected.device).view(blocks, 1)
    key_ids = torch.arange(blocks, device=selected.device).view(1, blocks)
    history_pairs = counts.view(-1, 1) * counts.view(1, -1)
    diagonal_pairs = counts * (counts + 1) // 2
    pair_weights = torch.where(
        key_ids < query_ids,
        history_pairs,
        torch.where(key_ids == query_ids, diagonal_pairs.view(-1, 1), 0),
    )
    exact_pairs = (selected.to(torch.int64) * pair_weights.view(1, blocks, blocks, 1)).sum()
    total_pairs = batch * heads * sequence * (sequence + 1) // 2
    causal_blocks = batch * heads * blocks * (blocks + 1) // 2
    selected_blocks = selected.sum()

    full_history = key_ids < query_ids
    tail = full_history.view(1, blocks, blocks, 1) & ~selected
    query_rows = counts.view(1, blocks, 1, 1)
    mean_proxy_entries = (tail.to(torch.int64) * query_rows).sum()
    selector_proxy_entries = (
        full_history.to(torch.int64) * counts.view(blocks, 1)
    ).sum() * batch * heads

    physical_tiles = None
    if q_tile_size is not None and k_tile_size is not None:
        q_programs = torch.div(counts + q_tile_size - 1, q_tile_size, rounding_mode="floor")
        k_iterations = math.ceil(block_size / k_tile_size)
        physical_tiles = (
            selected.to(torch.int64)
            * q_programs.view(1, blocks, 1, 1)
            * k_iterations
        ).sum()
    mean_method = method in {
        "mean_native", "mean_balanced", "cgf_mean", "dispersion_mean"
    }
    executed_mean_entries = batch * sequence * heads * blocks if mean_method else 0
    selector_executed_entries = 0
    selector_physical_tiles = None
    if method in {"mean_balanced", "cgf_mean", "dispersion_mean"}:
        descriptor_dots = 1 if method == "mean_balanced" else 2
        selector_executed_entries = batch * sequence * heads * blocks * descriptor_dots
    elif method in {"fp_v1", "mean_native"} and score_k_tile_size is not None:
        selector_physical_tiles = sum(
            math.ceil((query_block + 1) / score_k_tile_size)
            for query_block in range(blocks)
        ) * batch * heads
        selector_executed_entries = (
            selector_physical_tiles * score_k_tile_size * block_size
        )
    return {
        "effective_exact_token_pair_ratio": exact_pairs / total_pairs,
        "exact_token_pairs": exact_pairs,
        "causal_token_pairs": total_pairs,
        "legacy_block_density": selected_blocks / causal_blocks,
        "selected_block_entries": selected_blocks,
        "causal_block_entries": causal_blocks,
        "exact_physical_qk_tiles": physical_tiles,
        "mean_proxy_entries": mean_proxy_entries,
        "selector_proxy_entries": selector_proxy_entries,
        "mean_executed_logit_entries": executed_mean_entries,
        "mean_executed_value_entries": executed_mean_entries,
        "selector_executed_dot_entries": selector_executed_entries,
        "selector_physical_qk_tiles": selector_physical_tiles,
        "q_tile_size": q_tile_size,
        "k_tile_size": k_tile_size,
        "score_k_tile_size": score_k_tile_size,
    }
