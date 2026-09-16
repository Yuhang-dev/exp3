"""Audited FlashPrefill V1 kernels with an exact-path natural-log LSE output.

Source: qhfan/FlashPrefill, commit
baa612047433a992a00d07dc178205eed065ae14, adapted first in exp2.
See ../ORIGIN.md for provenance and the exact local changes.

All public entry points in this file use [batch, sequence, heads, dim].
"""

import torch
import triton
import triton.language as tl


SCORE_IMPL = "v1_kmean_qt_row_reduce"
LSE_IMPL = "natural_log_exact_lse"


def get_mean_configs():
    return [
        triton.Config({}, num_warps=warps, num_stages=stages)
        for warps in (4, 8)
        for stages in (2, 3, 4, 5)
    ]


@triton.autotune(
    configs=get_mean_configs(),
    key=["query_len", "num_q_heads", "BLOCK_SIZE"],
)
@triton.jit
def compute_mean_vector(
    Q_ptr,
    mQ_ptr,
    stride_qz,
    stride_qm,
    stride_qh,
    stride_qd,
    stride_mqz,
    stride_mqm,
    stride_mqh,
    stride_mqd,
    num_q_heads,
    query_len,
    BLOCK_SIZE: tl.constexpr,
    D_HEAD: tl.constexpr,
):
    query_tile_index = tl.program_id(0).to(tl.int64)
    offset_zh = tl.program_id(1).to(tl.int64)
    offset_batch = offset_zh // num_q_heads
    offset_q_head = offset_zh % num_q_heads

    q_base = Q_ptr + offset_batch * stride_qz + offset_q_head * stride_qh
    mean_base = mQ_ptr + offset_batch * stride_mqz + offset_q_head * stride_mqh
    offset_q = query_tile_index * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offset_d = tl.arange(0, D_HEAD)
    valid = (offset_q[:, None] < query_len) & (offset_d[None, :] < D_HEAD)
    q = tl.load(
        q_base + offset_q[:, None] * stride_qm + offset_d[None, :] * stride_qd,
        mask=valid,
        other=0.0,
    )
    count = tl.sum((offset_q < query_len).to(tl.int32)).to(q.dtype)
    mean = tl.where(count > 0, tl.sum(q, axis=0) / count, 0.0)
    tl.store(
        mean_base + query_tile_index * stride_mqm + offset_d * stride_mqd,
        mean,
        mask=offset_d < D_HEAD,
    )


def get_score_configs():
    configs = []
    for k_tile in (64, 128, 256):
        for warps in (4, 8):
            for stages in (2, 3, 4, 5):
                if k_tile == 256 and warps == 4:
                    continue
                configs.append(
                    triton.Config(
                        {"K_TILE_SIZE": k_tile},
                        num_warps=warps,
                        num_stages=stages,
                    )
                )
    return configs


@triton.autotune(
    configs=get_score_configs(),
    key=["query_len", "sub_key_len", "num_q_heads", "num_k_heads"],
)
@triton.jit
def compute_block_score(
    Q_ptr,
    K_ptr,
    scale,
    score_ptr,
    max_ptr,
    stride_qz,
    stride_qm,
    stride_qh,
    stride_qd,
    stride_kz,
    stride_kn,
    stride_kh,
    stride_kd,
    stride_scz,
    stride_scmb,
    stride_scnb,
    stride_sch,
    stride_mxz,
    stride_mxmb,
    stride_mxnb,
    stride_mxh,
    num_q_heads,
    num_k_heads,
    query_len,
    sub_key_len,
    BLOCK_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    K_STRIDE: tl.constexpr,
    D_HEAD: tl.constexpr,
):
    tl.static_assert(K_STRIDE == BLOCK_SIZE)
    query_tile = tl.program_id(0).to(tl.int64)
    offset_zh = tl.program_id(1).to(tl.int64)
    batch = offset_zh // num_q_heads
    q_head = offset_zh % num_q_heads
    kv_head = q_head // (num_q_heads // num_k_heads)

    q_base = Q_ptr + batch * stride_qz + q_head * stride_qh
    k_base = K_ptr + batch * stride_kz + kv_head * stride_kh
    score_base = score_ptr + batch * stride_scz + q_head * stride_sch
    max_base = max_ptr + batch * stride_mxz + q_head * stride_mxh

    offset_q = query_tile * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offset_d = tl.arange(0, D_HEAD)
    q = tl.load(
        q_base + offset_q[:, None] * stride_qm + offset_d[None, :] * stride_qd,
        mask=(offset_q[:, None] < query_len) & (offset_d[None, :] < D_HEAD),
        other=0.0,
    )
    hi = tl.minimum(query_tile + 1, sub_key_len)
    scale_log2 = scale * 1.4426950408889634

    for start in range(0, hi, K_TILE_SIZE):
        offset_k = start + tl.arange(0, K_TILE_SIZE)
        key_end = offset_k * K_STRIDE + K_STRIDE - 1
        k = tl.load(
            k_base + offset_k[:, None] * stride_kn + offset_d[None, :] * stride_kd,
            mask=(offset_k[:, None] < sub_key_len) & (offset_d[None, :] < D_HEAD),
            other=0.0,
        )
        kq = tl.dot(k, tl.trans(q)) * scale_log2
        causal = (
            (offset_q[None, :] >= key_end[:, None])
            & (offset_q[None, :] < query_len)
            & (offset_k[:, None] < sub_key_len)
        )
        kq = tl.where(causal, kq, float("-inf"))
        maximum = tl.max(kq, axis=1)
        mass = tl.where(causal, tl.exp2(kq - maximum[:, None]), 0.0)
        mass = tl.sum(mass, axis=1)
        valid_store = offset_k < sub_key_len
        tl.store(
            score_base + query_tile * stride_scmb + offset_k * stride_scnb,
            mass,
            mask=valid_store,
        )
        tl.store(
            max_base + query_tile * stride_mxmb + offset_k * stride_mxnb,
            maximum,
            mask=valid_store,
        )


def get_attention_configs():
    configs = []
    for q_tile, k_tile in ((64, 64), (128, 64), (64, 128)):
        for warps in (4, 8):
            for stages in (2, 3, 4, 5):
                if q_tile == 128 and warps == 4:
                    continue
                configs.append(
                    triton.Config(
                        {"Q_TILE_SIZE": q_tile, "K_TILE_SIZE": k_tile},
                        num_warps=warps,
                        num_stages=stages,
                    )
                )
    return configs


@triton.autotune(
    configs=get_attention_configs(),
    key=["query_len", "key_len", "num_q_heads", "num_k_heads", "BLOCK_SIZE"],
    prune_configs_by={
        "early_config_prune": lambda configs, named_args, **kwargs: [
            config
            for config in configs
            if config.kwargs["Q_TILE_SIZE"] <= kwargs["BLOCK_SIZE"]
            and config.kwargs["K_TILE_SIZE"] <= kwargs["BLOCK_SIZE"]
        ]
    },
)
@triton.jit
def _flash_forward(
    Q_ptr,
    K_ptr,
    V_ptr,
    O_ptr,
    LSE_ptr,
    index_ptr,
    valid_ptr,
    scale,
    stride_qz,
    stride_qm,
    stride_qh,
    stride_qd,
    stride_kz,
    stride_kn,
    stride_kh,
    stride_kd,
    stride_vz,
    stride_vn,
    stride_vh,
    stride_vd,
    stride_oz,
    stride_om,
    stride_oh,
    stride_od,
    stride_lz,
    stride_lm,
    stride_lh,
    stride_indexz,
    stride_indexm,
    stride_indexn,
    stride_indexh,
    stride_validz,
    stride_validm,
    stride_validh,
    query_len,
    key_len,
    num_q_heads,
    num_k_heads,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    D_HEAD: tl.constexpr,
):
    query_tile = tl.program_id(0).to(tl.int64)
    offset_zh = tl.program_id(1).to(tl.int64)
    batch = offset_zh // num_q_heads
    q_head = offset_zh % num_q_heads
    kv_head = q_head // (num_q_heads // num_k_heads)
    index_group_size = BLOCK_SIZE // Q_TILE_SIZE
    logical_query_block = query_tile // index_group_size

    q_base = Q_ptr + batch * stride_qz + q_head * stride_qh
    k_base = K_ptr + batch * stride_kz + kv_head * stride_kh
    v_base = V_ptr + batch * stride_vz + kv_head * stride_vh
    o_base = O_ptr + batch * stride_oz + q_head * stride_oh
    lse_base = LSE_ptr + batch * stride_lz + q_head * stride_lh
    index_base = (
        index_ptr
        + batch * stride_indexz
        + q_head * stride_indexh
        + logical_query_block * stride_indexm
    )
    valid_base = (
        valid_ptr
        + batch * stride_validz
        + q_head * stride_validh
        + logical_query_block * stride_validm
    )

    offset_q = query_tile * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)
    offset_d = tl.arange(0, D_HEAD)
    q = tl.load(
        q_base + offset_q[:, None] * stride_qm + offset_d[None, :] * stride_qd,
        mask=(offset_q[:, None] < query_len) & (offset_d[None, :] < D_HEAD),
        other=0.0,
    )
    scale_log2 = scale * 1.4426950408889634
    maximum = tl.full((Q_TILE_SIZE,), float("-inf"), dtype=tl.float32)
    denominator = tl.full((Q_TILE_SIZE,), 1.0, dtype=tl.float32)
    accumulator = tl.zeros((Q_TILE_SIZE, D_HEAD), dtype=tl.float32)
    count = tl.load(valid_base)
    block_index = tl.load(index_base)

    for position in range(0, count):
        next_block = tl.load(
            index_base + (position + 1) * stride_indexn,
            mask=position + 1 < count,
            other=0,
        )
        key_start = block_index * BLOCK_SIZE
        key_end = (block_index + 1) * BLOCK_SIZE
        diagonal = block_index == logical_query_block
        for start in range(key_start, key_end, K_TILE_SIZE):
            offset_k = start + tl.arange(0, K_TILE_SIZE)
            k = tl.load(
                k_base + offset_k[:, None] * stride_kn + offset_d[None, :] * stride_kd,
                mask=(offset_k[:, None] < key_len) & (offset_d[None, :] < D_HEAD),
                other=0.0,
            )
            v = tl.load(
                v_base + offset_k[:, None] * stride_vn + offset_d[None, :] * stride_vd,
                mask=(offset_k[:, None] < key_len) & (offset_d[None, :] < D_HEAD),
                other=0.0,
            )
            qk = tl.dot(q, tl.trans(k))
            if diagonal:
                qk = tl.where(
                    offset_q[:, None] >= offset_k[None, :],
                    qk,
                    float("-inf"),
                )
            qk *= scale_log2
            next_maximum = tl.maximum(maximum, tl.max(qk, axis=1))
            probabilities = tl.exp2(qk - next_maximum[:, None])
            tile_denominator = tl.sum(probabilities, axis=1)
            correction = tl.exp2(maximum - next_maximum)
            correction = tl.where(correction != correction, 1.0, correction)
            accumulator *= correction[:, None]
            denominator = denominator * correction + tile_denominator
            accumulator += tl.dot(probabilities.to(v.dtype), v)
            maximum = next_maximum
        block_index = next_block

    output = accumulator / denominator[:, None]
    valid_q = offset_q < query_len
    tl.store(
        o_base + offset_q[:, None] * stride_om + offset_d[None, :] * stride_od,
        output.to(q.dtype),
        mask=valid_q[:, None] & (offset_d[None, :] < D_HEAD),
    )
    natural_lse = maximum * 0.6931471805599453 + tl.log(denominator)
    tl.store(
        lse_base + offset_q * stride_lm,
        natural_lse,
        mask=valid_q,
    )


@torch.compile(mode="reduce-overhead")
def deal_output_score(
    score: torch.Tensor,
    attention_sink: int,
    window: int,
    alpha: float = 0.1,
    last_n_blocks_full: int = 2,
    min_budget: int = 0,
):
    batch, query_blocks, key_blocks, heads = score.shape
    key_ids = torch.arange(key_blocks, device=score.device).view(1, 1, key_blocks, 1)
    if min_budget > 0:
        top_values, top_indices = torch.topk(score, k=min_budget, dim=2)
        keep = score >= top_values[:, :, :1, :] * alpha
        keep.scatter_(2, top_indices, True)
    else:
        keep = score >= score.amax(dim=2, keepdim=True) * alpha
    query_ids = torch.arange(query_blocks, device=score.device).view(1, query_blocks, 1, 1)
    distance = query_ids - key_ids
    protected = (
        (key_ids < attention_sink)
        | ((distance >= 0) & (distance < window))
        | (query_ids >= query_blocks - last_n_blocks_full)
    )
    active = (keep | protected) & (distance >= 0)
    counts = active.sum(dim=2).to(torch.int32)
    indices = key_ids.expand(batch, query_blocks, key_blocks, heads)
    indices = indices.masked_fill(~active, key_blocks).sort(dim=2).values
    return indices.contiguous(), counts.contiguous()


@torch.compile
def normalize_scores(score: torch.Tensor, maximum: torch.Tensor):
    finite = maximum != float("-inf")
    global_maximum = maximum.amax(dim=2, keepdim=True)
    rescaled_maximum = torch.exp2(maximum - global_maximum)
    rescaled_maximum = torch.where(finite, rescaled_maximum, 1.0)
    score = torch.where(finite, score, 0.0) * rescaled_maximum
    return score / (score.sum(dim=2, keepdim=True) + 1e-9)


@torch.no_grad()
def block_mean_k(k: torch.Tensor, block_size: int = 128):
    batch, sequence, kv_heads, head_dim = k.shape
    blocks = triton.cdiv(sequence, block_size)
    mean = torch.empty(
        batch,
        blocks,
        kv_heads,
        head_dim,
        dtype=k.dtype,
        device=k.device,
    )
    compute_mean_vector[(blocks, batch * kv_heads, 1)](
        k,
        mean,
        *k.stride(),
        *mean.stride(),
        kv_heads,
        sequence,
        block_size,
        head_dim,
    )
    return mean


@torch.no_grad()
def v1_scores(q: torch.Tensor, mean_k: torch.Tensor, scale: float, block_size: int = 128):
    batch, sequence, q_heads, head_dim = q.shape
    blocks = mean_k.shape[1]
    kv_heads = mean_k.shape[2]
    score = torch.full(
        (batch, blocks, blocks, q_heads),
        float("-inf"),
        dtype=torch.float32,
        device=q.device,
    )
    maximum = torch.full_like(score, float("-inf"))
    compute_block_score[(blocks, batch * q_heads, 1)](
        q,
        mean_k,
        scale,
        score,
        maximum,
        *q.stride(),
        *mean_k.stride(),
        *score.stride(),
        *maximum.stride(),
        q_heads,
        kv_heads,
        sequence,
        blocks,
        block_size,
        K_STRIDE=block_size,
        D_HEAD=head_dim,
    )
    return normalize_scores(score, maximum)


@torch.no_grad()
def exact_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    indices: torch.Tensor,
    counts: torch.Tensor,
    scale: float,
    block_size: int = 128,
):
    batch, sequence, q_heads, head_dim = q.shape
    output = torch.empty_like(q)
    lse = torch.empty(
        batch,
        sequence,
        q_heads,
        dtype=torch.float32,
        device=q.device,
    )
    grid = lambda meta: (triton.cdiv(sequence, meta["Q_TILE_SIZE"]), batch * q_heads, 1)
    _flash_forward[grid](
        q,
        k,
        v,
        output,
        lse,
        indices,
        counts,
        scale,
        *q.stride(),
        *k.stride(),
        *v.stride(),
        *output.stride(),
        *lse.stride(),
        *indices.stride(),
        *counts.stride(),
        sequence,
        sequence,
        q_heads,
        k.shape[2],
        BLOCK_SIZE=block_size,
        D_HEAD=head_dim,
    )
    return output, lse


def attention_tile_sizes():
    config = getattr(_flash_forward, "best_config", None)
    if config is None:
        return None, None
    return config.kwargs["Q_TILE_SIZE"], config.kwargs["K_TILE_SIZE"]


def score_tile_size():
    config = getattr(compute_block_score, "best_config", None)
    return None if config is None else config.kwargs["K_TILE_SIZE"]
