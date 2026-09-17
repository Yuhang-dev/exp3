"""Small GPU checks for the exp3 exact, mean-corrected, and selector paths."""

import argparse
import json
from pathlib import Path

import torch
import triton

from attention import AttentionBackend
import kernels
from upstream import flashprefill_native_forward as upstream


def compare(name, actual, expected, records, atol=0.02, rtol=0.02):
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
        f"PASS {name}: max_abs={record['max_abs']:.6g}, rel_l2={record['relative_l2']:.6g}",
        flush=True,
    )


def dense_reference(q, k, v, scale):
    heads = q.shape[2]
    k = k.repeat_interleave(heads // k.shape[2], dim=2).float()
    v = v.repeat_interleave(heads // v.shape[2], dim=2).float()
    logits = torch.einsum("bqhd,bkhd->bhqk", q.float(), k) * scale
    positions = torch.arange(q.shape[1], device=q.device)
    logits.masked_fill_(positions[None, :] > positions[:, None], float("-inf"))
    lse = torch.logsumexp(logits, dim=-1).permute(0, 2, 1)
    output = torch.einsum("bhqk,bkhd->bqhd", logits.softmax(dim=-1), v)
    return output, lse


def v1_score_reference(q, mean_k, scale, block_size=128):
    """Independent FP32 expression for V1's query-tile/key-block mass."""
    batch, sequence, heads, _ = q.shape
    blocks = mean_k.shape[1]
    expanded_k = mean_k.repeat_interleave(heads // mean_k.shape[2], dim=2).float()
    logits = torch.einsum("bqhd,bkhd->bhqk", q.float(), expanded_k) * scale
    query_positions = torch.arange(sequence, device=q.device)
    key_ends = (torch.arange(blocks, device=q.device) + 1) * block_size - 1
    logits.masked_fill_(query_positions[:, None] < key_ends[None, :], float("-inf"))
    padded = blocks * block_size - sequence
    logits = torch.nn.functional.pad(logits, (0, 0, 0, padded), value=float("-inf"))
    logits = logits.reshape(batch, heads, blocks, block_size, blocks)
    maximum = logits.amax(dim=(3, 4), keepdim=True)
    maximum = torch.where(torch.isfinite(maximum), maximum, 0.0)
    mass = torch.exp(logits - maximum).sum(dim=3)
    normalized = mass / (mass.sum(dim=-1, keepdim=True) + 1e-9)
    return normalized.permute(0, 2, 3, 1).contiguous()


def v1_selection_reference(scores, sink_blocks, window_blocks, alpha, last_full_blocks):
    batch, query_blocks, key_blocks, heads = scores.shape
    query_ids = torch.arange(query_blocks, device=scores.device).view(1, query_blocks, 1, 1)
    key_ids = torch.arange(key_blocks, device=scores.device).view(1, 1, key_blocks, 1)
    distance = query_ids - key_ids
    keep = scores >= scores.amax(dim=2, keepdim=True) * alpha
    protected = (
        (key_ids < sink_blocks)
        | ((distance >= 0) & (distance < window_blocks))
        | (query_ids >= query_blocks - last_full_blocks)
    )
    active = (keep | protected) & (distance >= 0)
    indices = key_ids.expand(batch, query_blocks, key_blocks, heads)
    indices = indices.masked_fill(~active, key_blocks).sort(dim=2).values
    return indices.contiguous(), active.sum(dim=2).to(torch.int32).contiguous()


def explicit_mean_reference(q, k, v, selected, descriptors, scale):
    batch, sequence, heads, _ = q.shape
    blocks = selected.shape[1]
    group = heads // k.shape[2]
    k_expanded = k.repeat_interleave(group, dim=2).float()
    v_expanded = v.repeat_interleave(group, dim=2).float()
    mean_k = descriptors.mean_k.repeat_interleave(group, dim=2)
    mean_v = descriptors.mean_v.repeat_interleave(group, dim=2)
    exact_logits = torch.einsum("bqhd,bkhd->bqhk", q.float(), k_expanded) * scale
    query_positions = torch.arange(sequence, device=q.device)
    key_positions = torch.arange(sequence, device=q.device)
    query_tiles = query_positions // kernels.BLOCK_SIZE
    key_tiles = key_positions // kernels.BLOCK_SIZE
    selected_rows = selected.permute(0, 1, 3, 2)[:, query_tiles]
    exact_allowed = selected_rows[:, :, :, key_tiles]
    exact_allowed &= key_positions.view(1, 1, 1, -1) <= query_positions.view(1, -1, 1, 1)
    exact_logits.masked_fill_(~exact_allowed, float("-inf"))

    mean_logits = torch.einsum("bqhd,bkhd->bqhk", q.float(), mean_k) * scale
    mean_logits += descriptors.counts.float().log().view(1, 1, 1, blocks)
    full_history = torch.arange(blocks, device=q.device).view(1, 1, 1, blocks) < query_tiles.view(1, -1, 1, 1)
    mean_allowed = full_history & ~selected_rows
    mean_logits.masked_fill_(~mean_allowed, float("-inf"))

    total_lse = torch.logaddexp(
        torch.logsumexp(exact_logits, dim=-1),
        torch.logsumexp(mean_logits, dim=-1),
    )
    exact_weights = torch.exp(exact_logits - total_lse.unsqueeze(-1))
    mean_weights = torch.where(
        mean_allowed,
        torch.exp(mean_logits - total_lse.unsqueeze(-1)),
        0.0,
    )
    output = (
        torch.einsum("bqhk,bkhd->bqhd", exact_weights, v_expanded)
        + torch.einsum("bqhk,bkhd->bqhd", mean_weights, mean_v)
    )
    return output, total_lse


def selector_reference(q, descriptors, scale, selector):
    batch, sequence, heads, _ = q.shape
    blocks = descriptors.mean_k.shape[1]
    group = heads // descriptors.mean_k.shape[2]
    mean_k = descriptors.mean_k.repeat_interleave(group, dim=2)
    var_k = descriptors.var_k.repeat_interleave(group, dim=2)
    sigma_v = descriptors.sigma_v.repeat_interleave(group, dim=2).permute(0, 2, 1)
    scores = torch.zeros(batch, blocks, blocks, heads, device=q.device)
    for tile in range(blocks):
        left = tile * kernels.BLOCK_SIZE
        right = min(sequence, left + kernels.BLOCK_SIZE)
        if tile == 0:
            continue
        query = q[:, left:right].float()
        ell0 = torch.einsum("bqhd,bkhd->bqhk", query, mean_k[:, :tile]) * scale
        ell0 += descriptors.counts[:tile].float().log().view(1, 1, 1, tile)
        if selector == "mean_balanced":
            logits = ell0
            multiplier = 1.0
        else:
            u = torch.einsum("bqhd,bkhd->bqhk", query.square(), var_k[:, :tile])
            u = (u * scale * scale).clamp_min(0)
            logits = ell0 + 0.5 * u
            multiplier = 1.0 if selector == "cgf_mean" else u.sqrt() * sigma_v[:, None, :, :tile]
        contribution = logits.softmax(dim=-1) * multiplier
        scores[:, tile, :tile] = contribution.mean(dim=1).permute(0, 2, 1)
    return scores


def combined_path(q, k, v, selected, scale):
    descriptors = kernels.block_descriptors(k, v)
    indices, counts = kernels.indices_from_mask(selected)
    exact_output, exact_lse = upstream.exact_attention(q, k, v, indices, counts, scale)
    mean_output, mean_lse = kernels.mean_tail(q, descriptors, selected, scale)
    return (*kernels.merge_exact_mean(exact_output, exact_lse, mean_output, mean_lse), descriptors)


def zero_dispersion_counterexample(records):
    logits_a = torch.tensor([2.0, -2.0])
    logits_b = torch.tensor([5.0, 5.0])
    exact = logits_a.exp().sum() / (logits_a.exp().sum() + logits_b.exp().sum())
    mean_mass_a = 2 * logits_a.mean().exp()
    mean_mass_b = 2 * logits_b.mean().exp()
    approximate = mean_mass_a / (mean_mass_a + mean_mass_b)
    error = (exact - approximate).abs().item()
    assert error > 0 and torch.tensor([1.0, 1.0]).var(unbiased=False) == 0
    records.append({
        "check": "zero_value_dispersion_is_not_an_error_bound",
        "dispersion": 0.0,
        "exact_output": exact.item(),
        "mean_output": approximate.item(),
        "absolute_error": error,
        "status": "OBSERVED",
    })
    print(f"OBSERVED zero-dispersion counterexample: abs_error={error:.6g}", flush=True)


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("results/check_math.json"))
    args = parser.parse_args()
    torch.manual_seed(17)
    torch.backends.cuda.matmul.allow_tf32 = False
    records = []
    length = 257
    q = torch.randn(1, length, 28, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(1, length, 4, 128, device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    scale = 128 ** -0.5
    blocks = 3

    scores = torch.zeros(1, blocks, blocks, 28, device="cuda")
    all_selected = kernels.protected_selection(scores, 0.0, 2, 4, 2)
    indices, counts = kernels.indices_from_mask(all_selected)
    exact_output, exact_lse = upstream.exact_attention(q, k, v, indices, counts, scale)
    dense_output, dense_lse = dense_reference(q, k, v, scale)
    compare("all_exact_output_vs_dense", exact_output, dense_output, records)
    compare("all_exact_natural_lse_vs_dense", exact_lse, dense_lse, records, atol=0.02, rtol=0.01)

    selected = all_selected.clone()
    selected[:, 2, 1, ::2] = False
    combined, combined_lse, descriptors = combined_path(q, k, v, selected, scale)
    reference, reference_lse = explicit_mean_reference(q, k, v, selected, descriptors, scale)
    compare("fixed_mask_exact_plus_mean_output", combined, reference, records, atol=0.03, rtol=0.03)
    compare("fixed_mask_exact_plus_mean_lse", combined_lse, reference_lse, records, atol=0.025, rtol=0.015)

    for selector in ("mean_balanced", "cgf_mean", "dispersion_mean"):
        actual = kernels.selector_scores(q, descriptors, scale, selector, chunk_tiles=2)
        expected = selector_reference(q, descriptors, scale, selector)
        compare(f"selector_{selector}", actual, expected, records, atol=2e-5, rtol=2e-5)

    audit_length = 1089
    audit_q = torch.randn(1, audit_length, 28, 128, device="cuda", dtype=torch.bfloat16)
    audit_k = torch.randn(1, audit_length, 4, 128, device="cuda", dtype=torch.bfloat16)
    audit_mean = upstream.block_mean_k(audit_k)
    audit_mean_reference = torch.stack(
        [audit_k[:, start:start + 128].float().mean(dim=1) for start in range(0, audit_length, 128)],
        dim=1,
    )
    compare("v1_block_mean", audit_mean, audit_mean_reference, records, atol=0.002, rtol=0.02)
    audit_scores = upstream.v1_scores(audit_q, audit_mean, scale).clone()
    audit_score_reference = v1_score_reference(audit_q, audit_mean, scale)
    compare("v1_proxy_scores", audit_scores, audit_score_reference, records, atol=2e-4, rtol=0.005)

    selection_input = audit_scores.square().square()
    audit_indices, audit_counts = upstream.deal_output_score(selection_input, 2, 4, 0.8, 2, 0)
    reference_indices, reference_counts = v1_selection_reference(selection_input, 2, 4, 0.8, 2)
    torch.testing.assert_close(audit_indices, reference_indices, atol=0, rtol=0)
    torch.testing.assert_close(audit_counts, reference_counts, atol=0, rtol=0)
    records.append({"check": "v1_threshold_and_protected_selection", "exact_match": True})
    print("PASS v1_threshold_and_protected_selection", flush=True)

    backend = AttentionBackend(alpha=0.08)
    record = {"events": {}}
    _, fp_indices, fp_counts, _ = backend._v1_selection(q, k, scale, record, None)
    _, mean_indices, mean_counts, _ = backend._v1_selection(q, k, scale, record, v)
    torch.testing.assert_close(fp_indices, mean_indices, atol=0, rtol=0)
    torch.testing.assert_close(fp_counts, mean_counts, atol=0, rtol=0)
    records.append({"check": "mean_native_and_fp_v1_masks", "exact_match": True})
    print("PASS mean_native_and_fp_v1_masks", flush=True)

    cut = 200
    changed_k = k.clone()
    changed_v = v.clone()
    changed_k[:, cut:] *= 19
    changed_v[:, cut:] += 23
    first, _, _ = combined_path(q, k, v, selected, scale)
    second, _, _ = combined_path(q, changed_k, changed_v, selected, scale)
    compare("fixed_route_future_kv", second[:, :cut], first[:, :cut], records, atol=0, rtol=0)

    zero_dispersion_counterexample(records)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "status": "PASS",
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "triton": triton.__version__,
        "checks": records,
    }, indent=2), encoding="utf-8")
    print(f"Focused math checks passed: {args.out}", flush=True)


if __name__ == "__main__":
    main()
