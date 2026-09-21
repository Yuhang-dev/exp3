"""GPU gate for the Qwen3.5-4B V1 shape: 16:4 GQA with head dim 256."""

import argparse
import json
from pathlib import Path

import torch
import triton

from .attention import AttentionBackend
from . import kernels
from .upstream import flashprefill_native_forward as upstream


def compare(name, actual, expected, records, atol=0.03, rtol=0.03):
    delta = actual.float() - expected.float()
    relative = delta.norm() / expected.float().norm().clamp_min(1e-12)
    record = {
        "check": name,
        "max_abs": delta.abs().max().item(),
        "relative_l2": relative.item(),
        "atol": atol,
        "rtol": rtol,
    }
    torch.testing.assert_close(actual.float(), expected.float(), atol=atol, rtol=rtol)
    records.append(record)
    print(
        f"PASS {name}: max_abs={record['max_abs']:.6g}, "
        f"rel_l2={record['relative_l2']:.6g}",
        flush=True,
    )


def dense_reference(q, k, v, scale):
    group = q.shape[2] // k.shape[2]
    expanded_k = k.repeat_interleave(group, dim=2).float()
    expanded_v = v.repeat_interleave(group, dim=2).float()
    logits = torch.einsum("bqhd,bkhd->bhqk", q.float(), expanded_k) * scale
    positions = torch.arange(q.shape[1], device=q.device)
    logits.masked_fill_(positions[None, :] > positions[:, None], float("-inf"))
    lse = torch.logsumexp(logits, dim=-1).permute(0, 2, 1)
    output = torch.einsum("bhqk,bkhd->bqhd", logits.softmax(dim=-1), expanded_v)
    return output, lse


def mean_reference(k, block_size=128):
    return torch.stack(
        [k[:, start:start + block_size].float().mean(dim=1)
         for start in range(0, k.shape[1], block_size)],
        dim=1,
    )


def score_reference(q, mean_k, scale, block_size=128):
    batch, sequence, heads, _ = q.shape
    blocks = mean_k.shape[1]
    expanded_k = mean_k.repeat_interleave(heads // mean_k.shape[2], dim=2).float()
    logits = torch.einsum("bqhd,bkhd->bhqk", q.float(), expanded_k) * scale
    query_positions = torch.arange(sequence, device=q.device)
    key_ends = (torch.arange(blocks, device=q.device) + 1) * block_size - 1
    logits.masked_fill_(query_positions[:, None] < key_ends[None, :], float("-inf"))
    padding = blocks * block_size - sequence
    logits = torch.nn.functional.pad(logits, (0, 0, 0, padding), value=float("-inf"))
    logits = logits.reshape(batch, heads, blocks, block_size, blocks)
    maximum = logits.amax(dim=(3, 4), keepdim=True)
    maximum = torch.where(torch.isfinite(maximum), maximum, 0.0)
    mass = torch.exp(logits - maximum).sum(dim=3)
    normalized = mass / (mass.sum(dim=-1, keepdim=True) + 1e-9)
    return normalized.permute(0, 2, 3, 1).contiguous()


def selection_reference(scores, alpha, sink=2, window=4, last_full=2):
    batch, query_blocks, key_blocks, heads = scores.shape
    query_ids = torch.arange(query_blocks, device=scores.device).view(1, query_blocks, 1, 1)
    key_ids = torch.arange(key_blocks, device=scores.device).view(1, 1, key_blocks, 1)
    distance = query_ids - key_ids
    keep = scores >= scores.amax(dim=2, keepdim=True) * alpha
    protected = (
        (key_ids < sink)
        | ((distance >= 0) & (distance < window))
        | (query_ids >= query_blocks - last_full)
    )
    active = (keep | protected) & (distance >= 0)
    indices = key_ids.expand(batch, query_blocks, key_blocks, heads)
    indices = indices.masked_fill(~active, key_blocks).sort(dim=2).values
    return indices.contiguous(), active.sum(dim=2).to(torch.int32).contiguous()


class DummyAttention:
    layer_idx = 3


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("results/check_qwen35_math.json"))
    args = parser.parse_args()

    torch.manual_seed(35)
    torch.backends.cuda.matmul.allow_tf32 = False
    records = []
    length = 257
    q = torch.randn(1, length, 16, 256, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(1, length, 4, 256, device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    scale = 256 ** -0.5

    mean_k = upstream.block_mean_k(k)
    compare("qwen35_v1_block_mean", mean_k, mean_reference(k), records, atol=0.003, rtol=0.02)

    scores = upstream.v1_scores(q, mean_k, scale).clone()
    compare(
        "qwen35_v1_proxy_scores",
        scores,
        score_reference(q, mean_k, scale),
        records,
        atol=3e-4,
        rtol=0.008,
    )

    selection_input = scores.square().square()
    indices, counts = upstream.deal_output_score(selection_input, 2, 4, 0.8, 2, 0)
    expected_indices, expected_counts = selection_reference(selection_input, 0.8)
    torch.testing.assert_close(indices, expected_indices, atol=0, rtol=0)
    torch.testing.assert_close(counts, expected_counts, atol=0, rtol=0)
    records.append({"check": "qwen35_v1_threshold_selection", "exact_match": True})
    print("PASS qwen35_v1_threshold_selection", flush=True)

    all_scores = torch.zeros_like(scores)
    all_indices, all_counts = upstream.deal_output_score(all_scores, 2, 4, 0.0, 2, 0)
    exact_output, exact_lse = upstream.exact_attention(
        q, k, v, all_indices, all_counts, scale
    )
    exact_config = upstream._flash_forward.best_config
    exact_launch = {
        **exact_config.kwargs,
        "num_warps": exact_config.num_warps,
        "num_stages": exact_config.num_stages,
    }
    print(f"Exact attention launch: {exact_launch}", flush=True)
    dense_output, dense_lse = dense_reference(q, k, v, scale)
    compare("qwen35_all_exact_output_vs_dense", exact_output, dense_output, records)
    compare(
        "qwen35_all_exact_lse_vs_dense",
        exact_lse,
        dense_lse,
        records,
        atol=0.03,
        rtol=0.015,
    )

    backend = AttentionBackend(alpha=0.0)
    backend.configure("fp_v1", 0.0)
    backend_output, _ = backend.forward(
        DummyAttention(),
        q.transpose(1, 2).contiguous(),
        k.transpose(1, 2).contiguous(),
        v.transpose(1, 2).contiguous(),
        attention_mask=None,
        scaling=scale,
    )
    compare(
        "qwen35_attention_backend_alpha0",
        backend_output,
        dense_output,
        records,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {
                "status": "PASS",
                "shape": {"length": length, "q_heads": 16, "kv_heads": 4, "head_dim": 256},
                "gpu": torch.cuda.get_device_name(),
                "torch": torch.__version__,
                "triton": triton.__version__,
                "exact_attention_launch": exact_launch,
                "checks": records,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Qwen3.5 math gate passed: {args.out}", flush=True)


if __name__ == "__main__":
    main()
