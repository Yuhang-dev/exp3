"""Phase-1 block-proxy validation (E0-E8 of BLOCK_PROXY_VALIDATION_SPEC.md).

Offline analysis of saved V1 captures: frozen post/pre-RoPE Q/K, V, layer
inputs. Selection, oracle, and row sampling reuse block_proxy_study.py exactly
(float32 logits, identical tiles, rows, remote region, and top-k rules), so E0
reproduces the returned reports before any new quantity is computed.
Row-level mechanism and output quantities are recomputed in float64.

This script writes raw per-tile/per-row tables; summarize_block_proxy_validation.py
turns them into the deliverable CSVs and figures.
"""

import argparse
import csv
import gzip
import hashlib
import json
import math
import re
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open

from block_proxy_positions import query_rows, sampled_query_positions, tile_starts
from block_proxy_study import align_down, align_up, block_arrays, ranks, selection_scores


B = 128
BUDGET_TOKENS = 2048
BUDGET_BLOCKS = BUDGET_TOKENS // B
SINK_TOKENS = 256
LOCAL_TOKENS = 384
ALIGN = 256  # max K block size of the original study; fixes the remote region
REMOTE_START = align_up(SINK_TOKENS, ALIGN)
ROWS_PER_TILE = 16
FRACTIONS = (0.25, 0.5, 0.75, 0.9)
SWEEP_LAYER = 14
SWEEP_K = (32, 64, 128, 256)
RESCUE_BUDGETS = (2, 4, 8)
BASE_ALPHAS = (0.08, 0.12, 0.22)
RBS_BASE_ALPHAS = (0.08, 0.22)
RBS_RESCUE_ALPHAS = (0.10, 0.18, 0.30, 0.50)
BAND_NAMES = ("high", "mid", "low")

# block_regret (fraction) of the returned reports, Q=K=128, mean over all heads/tiles
# of exactly these samples.
E0_SAMPLES = {
    "ruler_niah_mk_1:32768:56", "ruler_niah_mk_1:32768:71", "ruler_niah_mq:32768:59",
    "longbench-v2-66ebc0c95a08c7b9b35de7f3",
}
E0_REFERENCE = {
    ("RULER", 0): {"mean_raw": 0.019269189752993084, "exact_raw": 0.015595435564007033},
    ("RULER", 7): {"mean_raw": 0.0038005868416456493, "exact_raw": 0.002274949192291179},
    ("RULER", 14): {"mean_raw": 0.07529690666567708, "exact_raw": 0.008614462578580456},
    ("RULER", 21): {"mean_raw": 0.0032181042645658283, "exact_raw": 0.0016715720828089971},
    ("RULER", 27): {"mean_raw": 0.007814574454511874, "exact_raw": 0.0033211942229951774},
    ("LongBench", 0): {"mean_raw": 0.02480242161878513, "exact_raw": 0.017771597951650574},
    ("LongBench", 14): {"mean_raw": 0.09820095449686048, "exact_raw": 0.01309527137449806},
    ("LongBench", 27): {"mean_raw": 0.010299191411052386, "exact_raw": 0.006593501993587998},
}
E0_TOLERANCE = 0.0005  # 0.05 percentage points

# Spec section 6 asked for max abs < 1e-3, which BF16 cannot meet for |k| up to ~400.
# Replaced (user-approved) by: every element within 2 BF16 ulps of its input pair norm,
# and fewer than 1e-3 of elements differing at all.
ROPE_MAX_ULP = 2.0
ROPE_MAX_MISMATCH = 1e-3

NEEDLE = re.compile(r"One of the special magic numbers for (.+?) is: (\d+)\.")


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--layers", nargs="+", type=int)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def panel_of(capture):
    return "LongBench" if "longbench" in capture.name else "RULER"


def remote_end_of(q_start):
    return align_down(q_start - LOCAL_TOKENS, ALIGN)


class Writer:
    def __init__(self, path):
        self.file = gzip.open(path, "wt", newline="", encoding="utf-8")
        self.writer = None

    def rows(self, common, columns):
        """Write len(columns[*]) rows: common fields plus one entry per column list."""
        names = list(columns)
        if self.writer is None:
            self.writer = csv.DictWriter(self.file, fieldnames=list(common) + names)
            self.writer.writeheader()
        for values in zip(*(columns[name] for name in names)):
            self.writer.writerow({**common, **dict(zip(names, values))})

    def row(self, record):
        self.rows({}, {key: [value] for key, value in record.items()})

    def close(self):
        self.file.close()


def cpu_lists(tensors):
    return {
        name: value.detach().cpu().tolist() if torch.is_tensor(value) else value
        for name, value in tensors.items()
    }


class Checks:
    def __init__(self):
        self.values = {}

    def update(self, name, value, limit):
        entry = self.values.setdefault(name, {"max": 0.0, "limit": limit, "count": 0})
        entry["max"] = max(entry["max"], float(value))
        entry["count"] += 1

    def report(self):
        return {
            name: {**entry, "passed": entry["max"] < entry["limit"]}
            for name, entry in self.values.items()
        }


# ---------------------------------------------------------------- RoPE

def inverse_frequencies(config, device):
    head_dim = config["hidden_size"] // config["num_attention_heads"]
    assert config.get("rope_scaling") is None, config.get("rope_scaling")
    exponent = torch.arange(0, head_dim, 2, dtype=torch.int64).to(device, torch.float32) / head_dim
    return 1.0 / (config["rope_theta"] ** exponent)


def apply_rope(x, positions, inv_freq):
    """HF Qwen2 apply_rotary_pos_emb in the tensor dtype; x is [tokens, heads, d]."""
    freqs = positions.to(torch.float32)[:, None] * inv_freq[None, :]
    emb = torch.cat((freqs, freqs), dim=-1)
    cos = emb.cos().to(x.dtype)[:, None, :]
    sin = emb.sin().to(x.dtype)[:, None, :]
    half = x.shape[-1] // 2
    rotated = torch.cat((-x[..., half:], x[..., :half]), dim=-1)
    return x * cos + rotated * sin


def rope_agreement(pre, post, positions, inv_freq):
    """Replicated BF16 rotation vs saved post-RoPE, in BF16 ulps of each input pair's norm.

    Differences come from rounding x*cos and rot*sin separately, so they scale with the
    input pair magnitude, not with the (possibly cancelled) output value.
    """
    rotated = apply_rope(pre, positions, inv_freq)
    diff = (rotated.float() - post.float()).abs()
    x = pre.float()
    half = x.shape[-1] // 2
    pair = torch.sqrt(x[..., :half].square() + x[..., half:].square()).repeat(1, 1, 2)
    ulp = torch.exp2(torch.floor(torch.log2(pair.clamp_min(1e-30)))) * 2 ** -7
    freqs = positions.to(torch.float32)[:, None] * inv_freq[None, :]
    cos = torch.repeat_interleave(freqs.cos(), 2, -1)[:, None, :]
    sin = torch.repeat_interleave(freqs.sin(), 2, -1)[:, None, :]
    interleaved = torch.stack((-x[..., 1::2], x[..., 0::2]), -1).flatten(-2)
    return {
        "max_abs": diff.max().item(),
        "max_ulp": (diff / ulp).max().item(),
        "mismatch_frac": (diff > 0).float().mean().item(),
        "interleaved_convention_max_abs": (x * cos + interleaved * sin - post.float()).abs().max().item(),
    }


def band_groups(inv_freq):
    theta = inv_freq.double()
    a = torch.abs(torch.sin(B * theta / 2) / (B * torch.sin(theta / 2)))
    group = torch.ones_like(a, dtype=torch.long)
    group[a < 0.5] = 0
    group[a > 0.9] = 2
    return a, group


# ---------------------------------------------------------------- selection

def topk_mask(score, count):
    mask = torch.zeros(score.numel(), dtype=torch.bool, device=score.device)
    mask[torch.topk(score, count).indices] = True
    return mask


def budget_rescue_mask(base_score, rescue_score, rescue_count):
    mask = topk_mask(base_score, BUDGET_BLOCKS - rescue_count)
    mask[torch.topk(rescue_score.masked_fill(mask, float("-inf")), rescue_count).indices] = True
    return mask


def threshold_mask(log_score, alpha, remote_slice):
    return (log_score >= math.log(alpha) + log_score.max())[remote_slice]


# ---------------------------------------------------------------- capture access

LAYER_FILES = (
    "query_post_rope.pt", "query_pre_rope.pt", "key_post_rope.pt", "key_pre_rope.pt",
    "value.pt", "layer_input.pt", "v1_route.pt",
)


class Layer:
    def __init__(self, sample_dir, sample_meta, layer, device, inv_freq):
        layer_dir = sample_dir / "layers" / f"layer_{layer:02d}"
        load = lambda name: torch.load(layer_dir / name, map_location="cpu", weights_only=True)
        self.files = [layer_dir / name for name in LAYER_FILES]
        self.q = load("query_post_rope.pt")
        self.q_pre = load("query_pre_rope.pt")
        self.k = load("key_post_rope.pt").to(device)
        self.k_pre = load("key_pre_rope.pt").to(device)
        self.v = load("value.pt").to(device)
        self.hidden = load("layer_input.pt")
        self.scale = float(load("v1_route.pt")["scale"])
        positions_path = sample_dir / "query_positions.pt"
        self.saved_positions = None
        if positions_path.exists():
            self.saved_positions = torch.load(positions_path, map_location="cpu", weights_only=True)
            self.files.append(positions_path)
        self.full_q = self.saved_positions is None
        self.sequence = sample_meta["capture_tokens"]
        self.device = device
        self.inv_freq = inv_freq
        self.group = self.q.shape[1] // self.k.shape[1]
        self.heads = self.q.shape[1]

    def rows(self, positions):
        if self.saved_positions is None:
            return positions
        index = torch.searchsorted(self.saved_positions, positions)
        assert torch.equal(self.saved_positions[index], positions), "query rows not captured"
        return index

    def tile_logits(self, head, positions):
        query = self.q[self.rows(positions), head].to(self.device, torch.float32)
        prefix = int(positions[-1]) + 1
        logits = query @ self.k[:prefix, head // self.group].float().T * self.scale
        future = torch.arange(prefix, device=self.device)[None, :] > positions.to(self.device)[:, None]
        return query, logits.masked_fill_(future, float("-inf"))


# ---------------------------------------------------------------- E0

def reproduce(plan, device, out):
    totals = {}
    for capture, meta, inv_freq, sample_dir, sample_meta, recorded, layer_id in plan:
        if sample_meta["sample_id"] not in E0_SAMPLES:
            continue
        layer = Layer(sample_dir, sample_meta, layer_id, device, inv_freq)
        for head in range(layer.heads):
            for q_start in tile_starts(layer.sequence, B, FRACTIONS):
                remote_end = remote_end_of(q_start)
                if remote_end - REMOTE_START <= BUDGET_TOKENS:
                    continue
                positions = query_rows(q_start, B, layer.sequence, ROWS_PER_TILE)
                _, logits = layer.tile_logits(head, positions)
                data = block_arrays(logits, torch.softmax(logits, 1), REMOTE_START, remote_end, B)
                scores = selection_scores(data, torch.logsumexp(logits, 1))
                mass = data["probability"].mean(0)
                oracle = mass[topk_mask(mass, BUDGET_BLOCKS)].sum()
                for method in ("mean_raw", "exact_raw"):
                    regret = oracle - mass[topk_mask(scores[method], BUDGET_BLOCKS)].sum()
                    totals.setdefault((panel_of(capture), layer_id, method), []).append(regret.item())
        del layer
    e0 = []
    for (panel, layer_id, method), values in sorted(totals.items()):
        measured = sum(values) / len(values)
        reference = E0_REFERENCE[(panel, layer_id)][method]
        e0.append({
            "panel": panel, "layer": layer_id, "method": method, "configs": len(values),
            "measured_pp": 100 * measured, "reference_pp": 100 * reference,
            "passed": abs(measured - reference) <= E0_TOLERANCE,
        })
    (out / "e0_reproduction.json").write_text(json.dumps(e0, indent=2), encoding="utf-8")
    print(json.dumps(e0, indent=1), flush=True)
    failed = [item for item in e0 if not item["passed"]]
    if failed:
        raise SystemExit(f"E0 reproduction failed; no further experiment is run: {failed}")


# ---------------------------------------------------------------- per tile/head

def select(layer, head, q_start, positions, query, logits, uplift, writers, common):
    """Selection sets at K=128 (and the L14 K sweep); writes the methods table."""
    remote_end = remote_end_of(q_start)
    log_z = torch.logsumexp(logits, 1)
    probabilities = torch.softmax(logits, 1)
    result = None
    for size in (SWEEP_K if common["layer"] == SWEEP_LAYER else (B,)):
        data = block_arrays(logits, probabilities, REMOTE_START, remote_end, size)
        budget = BUDGET_TOKENS // size
        block_mass = data["probability"]
        fixed_mass = 1.0 - block_mass.sum(1)
        mean_mass = block_mass.mean(0)
        oracle = topk_mask(mean_mass, budget)
        tile_oracle = fixed_mass.mean() + mean_mass[oracle].sum()
        scores = selection_scores(data, log_z)
        scores["mean_var2"] = torch.logsumexp(data["log_l0"] + 0.5 * data["std"].square(), 0)
        masks = {name: topk_mask(scores[name], budget) for name in ("mean_raw", "exact_raw", "mean_var2")}
        masks["tile_block_oracle"] = oracle
        rescue_only, density = {}, {}
        if size == B:
            q_norm = torch.linalg.vector_norm(query, dim=1)[:, None]
            remote_slice = slice(REMOTE_START // B, remote_end // B)
            rescue_score = torch.logsumexp(
                data["mean_logit"] + q_norm * uplift[remote_slice][None, :] * layer.scale, 0
            )
            for count in RESCUE_BUDGETS:
                name = f"rbs_budget_r{count}"
                masks[name] = budget_rescue_mask(scores["mean_raw"], rescue_score, count)
                rescue_only[name] = masks[name] & ~masks["mean_raw"]
            # Relative thresholds over every fully visible block of the tile (RBS Eqs. 10-14).
            visible = q_start // B
            visible_mean = logits[:, :visible * B].reshape(-1, visible, B).mean(-1)
            base_log = torch.logsumexp(visible_mean, 0)
            rescue_log = torch.logsumexp(
                visible_mean + q_norm * uplift[:visible][None, :] * layer.scale, 0
            )
            for alpha in BASE_ALPHAS:
                masks[f"thr_base_a{alpha:.2f}"] = threshold_mask(base_log, alpha, remote_slice)
            for alpha_base in RBS_BASE_ALPHAS:
                base = threshold_mask(base_log, alpha_base, remote_slice)
                for alpha_rescue in RBS_RESCUE_ALPHAS:
                    rescue = threshold_mask(rescue_log, alpha_rescue, remote_slice)
                    name = f"rbs_thr_b{alpha_base:.2f}_r{alpha_rescue:.2f}"
                    masks[name] = base | rescue
                    rescue_only[name] = rescue & ~base
            forced = REMOTE_START // B + (q_start - remote_end) // B + 1
            density = {name: (int(mask.sum()) + forced) / (visible + 1) for name, mask in masks.items()}
        missed = oracle & ~masks["mean_raw"]
        names = list(masks)
        stacked = torch.stack([masks[name] for name in names]).float()
        retained = fixed_mass[None, :] + stacked @ block_mass.T  # [methods, rows]
        writers["methods"].rows(
            {**common, "q_start": q_start, "k_block_size": size,
             "remote_blocks": block_mass.shape[1], "missed_blocks": int(missed.sum())},
            cpu_lists({
                "method": names,
                "selected_remote_blocks": stacked.sum(1),
                "retained_mass_mean": retained.mean(1),
                "block_regret": tile_oracle - retained.mean(1),
                "selected_overlap_oracle": (stacked @ oracle.float()) / budget,
                "oracle_recall_mass": (stacked @ (mean_mass * oracle)) / mean_mass[oracle].sum(),
                "missed_recall_mass": (stacked @ (mean_mass * missed)) / (mean_mass * missed).sum(),
                "density": [density.get(name, float("nan")) for name in names],
                "rescue_only_blocks": [int(rescue_only[n].sum()) if n in rescue_only else 0 for n in names],
                "rescue_only_false_pos": [
                    int((rescue_only[n] & ~oracle).sum()) if n in rescue_only else 0 for n in names
                ],
            }),
        )
        if size == B:
            result = {
                "data": data, "masks": masks, "oracle": oracle, "missed": missed,
                "mean_mass": mean_mass, "mean_raw_rank": ranks(scores["mean_raw"]),
                "true_rank": ranks(mean_mass),
            }
    return result


# ---------------------------------------------------------------- E6

def output_errors(layer, head, q_start, positions, query, selection, writers, checks, store, common):
    device = layer.device
    kv_head = head // layer.group
    remote_end = remote_end_of(q_start)
    prefix = int(positions[-1]) + 1
    q64 = query.double()
    k64 = layer.k[:prefix, kv_head].double()
    v64 = layer.v[:prefix, kv_head].double()
    logits = q64 @ k64.T * layer.scale
    future = torch.arange(prefix, device=device)[None, :] > positions.to(device)[:, None]
    logits.masked_fill_(future, float("-inf"))
    log_z = torch.logsumexp(logits, 1)
    p = torch.exp(logits - log_z[:, None])
    rows, blocks = p.shape[0], (remote_end - REMOTE_START) // B
    remote_p = p[:, REMOTE_START:remote_end].reshape(rows, blocks, B)
    block_p = remote_p.sum(-1)
    block_w = torch.einsum("rbj,bjd->rbd", remote_p, v64[REMOTE_START:remote_end].reshape(blocks, B, -1))
    fixed_w = p[:, :REMOTE_START] @ v64[:REMOTE_START] + p[:, remote_end:] @ v64[remote_end:]
    fixed_p = p[:, :REMOTE_START].sum(1) + p[:, remote_end:].sum(1)
    dense = p @ v64
    dense_norm = torch.linalg.vector_norm(dense, dim=1)
    checks.update(
        "dense_blockwise_vs_direct_rel",
        (torch.linalg.vector_norm(fixed_w + block_w.sum(1) - dense, dim=1) / dense_norm).max().item(),
        1e-5,
    )

    # FlashPrefill V2 zero-order correction: BF16 plain block means of K and V.
    k_bar = layer.k[REMOTE_START:remote_end, kv_head].float().reshape(blocks, B, -1).mean(1)
    v_bar = layer.v[REMOTE_START:remote_end, kv_head].float().reshape(blocks, B, -1).mean(1)
    k_bar = k_bar.to(layer.k.dtype).double()
    v_bar = v_bar.to(layer.v.dtype).double()
    comp_p = B * torch.exp(q64 @ k_bar.T * layer.scale - log_z[:, None])

    masks = selection["masks"]
    variants = [(name, mask, False) for name, mask in masks.items()]
    variants += [
        ("mean_raw_v2comp", masks["mean_raw"], True),
        ("thr_base_a0.08_v2comp", masks["thr_base_a0.08"], True),
    ]
    unit = dense / dense_norm[:, None]
    for name, mask, compensated in variants:
        weight = mask.double()
        numerator = fixed_w + torch.einsum("rbd,b->rd", block_w, weight)
        denominator = fixed_p + block_p @ weight
        retained = denominator.clone()
        if compensated:
            off = (1.0 - weight)[None, :] * comp_p
            numerator = numerator + off @ v_bar
            denominator = denominator + off.sum(1)
            retained.fill_(float("nan"))
        sparse = numerator / denominator[:, None]
        delta = dense - sparse
        parallel = (delta * unit).sum(1)
        perpendicular = torch.linalg.vector_norm(delta - parallel[:, None] * unit, dim=1)
        store.setdefault(name, {})[head] = sparse
        writers["output_rows"].rows(
            {**common, "q_start": q_start, "method": name},
            cpu_lists({
                "query_token": positions,
                "retained_mass": retained,
                "rel_err": torch.linalg.vector_norm(delta, dim=1) / dense_norm,
                "e_par": parallel / dense_norm,
                "e_perp": perpendicular / dense_norm,
                "cos": F.cosine_similarity(dense, sparse, dim=1),
            }),
        )
    store.setdefault("dense", {})[head] = dense

    missed = torch.nonzero(selection["missed"]).flatten()
    if missed.numel():
        mass = block_p[:, missed]  # [rows, M]
        v_mean = block_w[:, missed] / mass[..., None]
        distance = torch.linalg.vector_norm(v_mean - dense[:, None], dim=-1) / dense_norm[:, None]
        formula = mass / (1 - mass) * distance
        without = (dense[:, None] - block_w[:, missed]) / (1 - mass)[..., None]
        direct = torch.linalg.vector_norm(dense[:, None] - without, dim=-1) / dense_norm[:, None]
        # Rows where the block mass is below 1e-6 fall under float64 resolution of the direct recompute.
        resolved = mass >= 1e-6
        checks.update(
            "single_block_drop_formula_rel",
            ((formula - direct).abs() / direct)[resolved].max().item() if resolved.any() else 0.0, 1e-4,
        )
        weight = mass / mass.sum(0, keepdim=True)
        cosine = F.cosine_similarity(v_mean, dense[:, None].expand_as(v_mean), dim=-1)
        writers["block_drop"].rows(
            {**common, "q_start": q_start},
            cpu_lists({
                "block": missed,
                "token_start": REMOTE_START + missed * B,
                "true_mass_mean": mass.mean(0),
                "mean_raw_rank": selection["mean_raw_rank"][missed],
                "true_rank": selection["true_rank"][missed],
                "vbar_minus_o_rel": (weight * distance).sum(0),
                "cos_vbar_o": (weight * cosine).sum(0),
                "vbar_norm_rel": (weight * torch.linalg.vector_norm(v_mean, dim=-1) / dense_norm[:, None]).sum(0),
                "drop_err_formula": (weight * formula).sum(0),
                "drop_err_direct": (weight * direct).sum(0),
                "drop_err_rowmax": direct.max(0).values,
                "scale_if_vbar_zero": (weight / (1 - mass)).sum(0),
            }),
        )


def output_projection(layer, o_proj, q_start, positions, store, writers, common):
    rows = positions.numel()
    dense = torch.stack([store["dense"][h] for h in range(layer.heads)], 1).reshape(rows, -1)
    dense_out = torch.linalg.vector_norm(dense @ o_proj.T, dim=1)
    hidden = torch.linalg.vector_norm(
        layer.hidden[layer.rows(positions)].to(layer.device, torch.float64), dim=1
    )
    for name, per_head in store.items():
        if name == "dense":
            continue
        sparse = torch.stack([per_head[h] for h in range(layer.heads)], 1).reshape(rows, -1)
        delta = torch.linalg.vector_norm((dense - sparse) @ o_proj.T, dim=1)
        writers["oproj_rows"].rows(
            {**common, "q_start": q_start, "method": name},
            cpu_lists({
                "query_token": positions,
                "oproj_rel_err": delta / dense_out,
                "residual_rel": delta / hidden,
                "attn_out_over_residual": dense_out / hidden,
            }),
        )


# ---------------------------------------------------------------- E1-E4

def mechanism(layer, head, q_start, positions, query, selection, writers, checks, peaks, common):
    device = layer.device
    kv_head = head // layer.group
    data, masks, oracle = selection["data"], selection["masks"], selection["oracle"]
    mean_mask = masks["mean_raw"]
    rank = selection["mean_raw_rank"]
    boundary = (rank >= BUDGET_BLOCKS - 2) & (rank <= BUDGET_BLOCKS + 1)
    union = oracle | mean_mask | boundary | masks["rbs_budget_r4"] | masks["rbs_thr_b0.22_r0.18"]
    block_ids = torch.nonzero(union).flatten()
    starts = REMOTE_START + block_ids * B
    token_index = starts[:, None] + torch.arange(B, device=device)[None, :]

    q64 = query.double()
    k_blocks = layer.k[:, kv_head].double()[token_index]  # [U, B, d]
    s = torch.einsum("rd,ujd->ruj", q64, k_blocks) * layer.scale
    half = q64.shape[1] // 2
    band = (
        q64[:, None, None, :half] * k_blocks[None, :, :, :half]
        + q64[:, None, None, half:] * k_blocks[None, :, :, half:]
    ) * layer.scale  # [R, U, B, d/2]
    checks.update("band_sum_vs_logit_abs", (band.sum(-1) - s).abs().max().item(), 1e-4)
    checks.update(
        "float64_vs_float32_logit_abs", (s - data["scores"][:, block_ids].double()).abs().max().item(), 1e-2,
    )
    _, group = band_groups(layer.inv_freq)
    group_logit = torch.stack([band[..., group == g].sum(-1) for g in range(3)], -1)  # [R, U, B, 3]

    # E1: cumulants and their partial sums against the exact log mean mass.
    mu = s.mean(-1)
    centered = s - mu[..., None]
    k2 = centered.square().mean(-1)
    k3 = centered.pow(3).mean(-1)
    k4 = centered.pow(4).mean(-1) - 3 * k2.square()
    truth = torch.logsumexp(s, -1) - math.log(B)
    gap = truth - mu
    partial = torch.stack([mu, mu + k2 / 2, mu + k2 / 2 + k3 / 6, mu + k2 / 2 + k3 / 6 + k4 / 24], -1)
    cumulant_err = (partial - truth[..., None]).abs()
    r = torch.softmax(s, -1)
    top1 = r.max(-1).values
    n_eff = 1.0 / r.square().sum(-1)
    peak = s.argmax(-1)
    s_peak = s.max(-1).values
    delta = s_peak - mu

    weights = data["probability"][:, block_ids].double()
    weights = weights / weights.sum(0, keepdim=True)
    wmean = lambda x: (weights * x).sum(0)
    columns = {
        "block": block_ids, "token_start": starts,
        "true_mass_mean": selection["mean_mass"][block_ids],
        "mean_raw_rank": rank[block_ids], "true_rank": selection["true_rank"][block_ids],
        "in_oracle": oracle[block_ids].int(), "in_mean": mean_mask[block_ids].int(),
        "boundary": boundary[block_ids].int(),
        "mu": wmean(mu), "gap": wmean(gap), "var": wmean(k2),
        "kappa3": wmean(k3), "kappa4": wmean(k4),
        "half_var_over_gap_frac": wmean((0.5 * k2 > gap).double()),
        "top1_share": wmean(top1), "n_eff": wmean(n_eff),
        "delta_eff": wmean(torch.log(top1 / (1 - top1)) + math.log(B - 1)),
        "delta_peak": wmean(delta),
        "k4_worse_than_k2_frac": wmean((cumulant_err[..., 3] > cumulant_err[..., 1]).double()),
    }
    for n in range(4):
        columns[f"cum_err_k{n + 1}"] = wmean(cumulant_err[..., n])

    # E2: decomposition around two per-row reference levels.
    mean_logit = data["mean_logit"].double()
    mu_minus1 = (B * mu - s_peak) / (B - 1)
    for label, c in (
        ("cb", mean_logit[:, mean_mask].min(1).values),
        ("cm", mean_logit.median(1).values),
    ):
        a_plus = (s - c[:, None, None]).clamp_min(0).mean(-1)
        a_minus = (c[:, None, None] - s).clamp_min(0).mean(-1)
        checks.update("e2_identity_abs", ((mu - c[:, None]) - (a_plus - a_minus)).abs().max().item(), 1e-9)
        top_share = (s_peak - c[:, None]).clamp_min(0) / B / a_plus
        valid = ~top_share.isnan()
        columns[f"{label}_a_plus"] = wmean(a_plus)
        columns[f"{label}_a_minus"] = wmean(a_minus)
        columns[f"{label}_a_plus_top1"] = (
            (weights * top_share.nan_to_num()).sum(0) / (weights * valid).sum(0)
        )
        columns[f"{label}_mu_minus_c"] = wmean(mu - c[:, None])
        columns[f"{label}_mu_minus1_minus_c"] = wmean(mu_minus1 - c[:, None])

    # E3: band shares of within-block variance and of the peak advantage.
    group_centered = group_logit - group_logit.mean(2, keepdim=True)
    var_share = (group_centered * centered[..., None]).mean(2) / k2[..., None]
    peak_group = group_logit.gather(2, peak[..., None, None].expand(-1, -1, 1, 3)).squeeze(2)
    delta_share = (peak_group - group_logit.mean(2)) / delta[..., None]
    k_float = k_blocks.float()
    pair_mean = torch.sqrt(k_float[:, :, :half].mean(1).square() + k_float[:, :, half:].mean(1).square())
    pair_norm = torch.sqrt(k_float[:, :, :half].square() + k_float[:, :, half:].square()).mean(1)
    rho = pair_mean / pair_norm  # [U, d/2]
    for g, label in enumerate(BAND_NAMES):
        columns[f"var_share_{label}"] = wmean(var_share[..., g])
        columns[f"delta_share_{label}"] = wmean(delta_share[..., g])
        columns[f"rho_{label}"] = rho[:, group == g].mean(1)
    for name, mask in masks.items():
        columns[f"sel_{name}"] = mask[block_ids].int()
    writers["blocks"].rows({**common, "q_start": q_start}, cpu_lists(columns))

    # E4: every missed block, at its highest-mass sampled row, with NoPE logits.
    if not layer.rope_ok:
        return
    missed = torch.nonzero(selection["missed"]).flatten()
    if not missed.numel():
        return
    u = torch.searchsorted(block_ids, missed)
    row = data["probability"][:, missed].argmax(0)
    start = REMOTE_START + missed * B
    j = peak[row, u]
    q_pre = layer.q_pre[layer.rows(positions), head].to(device, torch.float64)[row]  # [M, d]
    k_pre = layer.k_pre[:, kv_head].double()[start[:, None] + torch.arange(B, device=device)[None, :]]
    s_nope = torch.einsum("md,mjd->mj", q_pre, k_pre) * layer.scale
    nope_peak = s_nope.gather(1, j[:, None]).squeeze(1)
    peak_position = start + j
    query_token = positions.to(device)[row]
    for kv, position in zip([kv_head] * missed.numel(), peak_position.tolist()):
        peaks.setdefault(kv, set()).add(position)
    tokens = [layer.tokens[p] for p in peak_position.tolist()]
    contexts = ["".join(layer.tokens[max(0, p - 12):p + 13]) for p in peak_position.tolist()]
    writers["peaks"].rows(
        {**common, "q_start": q_start},
        cpu_lists({
            "block": missed, "token_start": start,
            "true_mass_mean": selection["mean_mass"][missed],
            "true_mass_row": data["probability"][row, missed],
            "mean_raw_rank": rank[missed], "true_rank": selection["true_rank"][missed],
            "query_token": query_token, "peak_position": peak_position, "peak_offset": j,
            "distance": query_token - peak_position,
            "peak_share_rope": r[row, u, j],
            "delta_rope": delta[row, u],
            "delta_nope": nope_peak - s_nope.mean(1),
            "nope_rank": (s_nope > nope_peak[:, None]).sum(1),
            "nope_top_position": start + s_nope.argmax(1),
            "delta_share_high": delta_share[row, u, 0],
            "delta_share_mid": delta_share[row, u, 1],
            "delta_share_low": delta_share[row, u, 2],
            "gap_row": gap[row, u], "n_eff_row": n_eff[row, u],
            "token": tokens, "context": contexts,
        }),
    )


# ---------------------------------------------------------------- E5

def sink_features(layer, peaks, writers, common):
    device = layer.device
    positions = sampled_query_positions(layer.sequence, [64, 128, 256], FRACTIONS, ROWS_PER_TILE)
    index = layer.rows(positions)
    pos = positions.to(device)
    remote_end = ((pos // B) * B - LOCAL_TOKENS).div(ALIGN, rounding_mode="floor") * ALIGN
    full = layer.sequence // B * B
    key_norm = torch.linalg.vector_norm(layer.k.float(), dim=-1)  # [T, kv]
    value_norm = torch.linalg.vector_norm(layer.v.float(), dim=-1)
    future = torch.arange(layer.sequence, device=device)[None, :] > pos[:, None]
    masked = (future.sum(1))[:, None].double()
    for kv_head, points in sorted(peaks.items()):
        points = torch.tensor(sorted(points), device=device)
        block_tokens = points[:, None] // B * B + torch.arange(B, device=device)[None, :]
        norms = {}
        for label, table in (("key", key_norm[:, kv_head]), ("value", value_norm[:, kv_head])):
            norms[f"{label}_norm_pct"] = (table[None, :] < table[points][:, None]).double().mean(1)
            norms[f"{label}_norm_over_block_median"] = table[points] / table[block_tokens].median(1).values
        valid = (points[None, :] >= REMOTE_START) & (points[None, :] < remote_end[:, None])  # [rows, P]
        valid_count = valid.sum(0)
        key = layer.k[:, kv_head].float()
        for head in range(kv_head * layer.group, (kv_head + 1) * layer.group):
            query = layer.q[index, head].to(device, torch.float32)
            logits = (query @ key.T * layer.scale).masked_fill_(future, float("-inf"))
            block_argmax = logits[:, :full].reshape(pos.numel(), -1, B).argmax(-1)
            is_argmax = (block_argmax[:, points // B] == (points % B)[None, :]).double()
            below = torch.searchsorted(torch.sort(logits, 1).values, logits[:, points].contiguous())
            percentile = (below.double() - masked) / (pos + 1)[:, None].double()
            percentile_valid = percentile.masked_fill(~valid, float("nan"))
            writers["sink"].rows(
                {**common, "kv_head": kv_head, "query_head": head},
                cpu_lists({
                    "position": points, "token": [layer.tokens[p] for p in points.tolist()],
                    "valid_rows": valid_count,
                    "argmax_fraction": (is_argmax * valid).sum(0) / valid_count,
                    "row_percentile_mean": percentile_valid.nanmean(0),
                    "row_percentile_median": percentile_valid.nanmedian(0).values,
                    **norms,
                }),
            )


# ---------------------------------------------------------------- E3 attenuation

def attenuation_table(layer, writers, common):
    blocks = layer.sequence // B
    half = layer.k.shape[-1] // 2
    theory, group = band_groups(layer.inv_freq)
    for label, keys in (("post", layer.k), ("pre", layer.k_pre)):
        x = keys[:blocks * B].float().reshape(blocks, B, keys.shape[1], -1)
        pair_mean = torch.sqrt(x[..., :half].mean(1).square() + x[..., half:].mean(1).square())
        pair_norm = torch.sqrt(x[..., :half].square() + x[..., half:].square()).mean(1)
        rho = (pair_mean / pair_norm).mean(0)  # [kv, d/2]
        for kv_head in range(rho.shape[0]):
            writers["attenuation"].rows(
                {**common, "kv_head": kv_head, "rope": label},
                cpu_lists({
                    "pair": list(range(half)), "theta": layer.inv_freq,
                    "a_theory_128": theory, "band": [BAND_NAMES[g] for g in group.tolist()],
                    "rho_mean": rho[kv_head],
                }),
            )


# ---------------------------------------------------------------- E8

NEEDLE_FIELDS = (
    "mean_raw_rank", "true_rank", "in_oracle", "in_mean", "in_v1_like_a0.08", "gap", "n_eff",
    "top1_share", "peak_in_value_frac", "peak_in_needle_frac", "remote_blocks",
)


def needle_blocks(layer, sample, writers, checks, common):
    device = layer.device
    starts = [0]
    for token in layer.tokens:
        starts.append(starts[-1] + len(token))
    text = "".join(layer.tokens)
    starts_t = torch.tensor(starts)
    token_of = lambda char: int(torch.searchsorted(starts_t, char, right=True)) - 1
    needles = [
        {
            "key": match.group(1), "value": match.group(2),
            "queried": int(match.group(2) in sample["answers"]),
            "span_start": token_of(match.start()), "value_start": token_of(match.start(2)),
            "span_end": token_of(match.end(2) - 1) + 1,
        }
        for match in NEEDLE.finditer(text)
    ]
    checks.update("e8_answers_not_located", len(set(sample["answers"]) - {n["value"] for n in needles}), 1)
    q_start = (layer.sequence - 1) // B * B
    positions = torch.arange(q_start, layer.sequence)
    remote_end = remote_end_of(q_start)
    for head in range(layer.heads):
        _, logits = layer.tile_logits(head, positions)
        probabilities = torch.softmax(logits, 1)
        data = block_arrays(logits, probabilities, REMOTE_START, remote_end, B)
        scores = selection_scores(data, torch.logsumexp(logits, 1))
        mean_mass = data["probability"].mean(0)
        oracle = topk_mask(mean_mass, BUDGET_BLOCKS)
        mean_mask = topk_mask(scores["mean_raw"], BUDGET_BLOCKS)
        visible = q_start // B
        v1_like = threshold_mask(
            torch.logsumexp(logits[:, :visible * B].reshape(-1, visible, B).mean(-1), 0),
            0.08, slice(REMOTE_START // B, remote_end // B),
        )
        mean_rank, true_rank = ranks(scores["mean_raw"]), ranks(mean_mass)
        weights = data["probability"] / data["probability"].sum(0, keepdim=True)
        for needle in needles:
            for block_start in sorted({needle["span_start"] // B * B, (needle["span_end"] - 1) // B * B}):
                region = (
                    "sink" if block_start < REMOTE_START
                    else "remote" if block_start + B <= remote_end else "local"
                )
                row = {
                    **common, "query_head": head, "kv_head": head // layer.group, "q_start": q_start,
                    **needle, "block_start": block_start, "region": region,
                    "row_mass_mean": probabilities[:, block_start:block_start + B].sum(1).mean().item(),
                    **{field: float("nan") for field in NEEDLE_FIELDS},
                }
                if region == "remote":
                    b = (block_start - REMOTE_START) // B
                    peak_token = block_start + data["scores"][:, b].argmax(-1)
                    w = weights[:, b]
                    in_value = (peak_token >= needle["value_start"]) & (peak_token < needle["span_end"])
                    in_needle = (peak_token >= needle["span_start"]) & (peak_token < needle["span_end"])
                    row.update({
                        "mean_raw_rank": int(mean_rank[b]), "true_rank": int(true_rank[b]),
                        "in_oracle": int(oracle[b]), "in_mean": int(mean_mask[b]),
                        "in_v1_like_a0.08": int(v1_like[b]),
                        "gap": (w * data["gap"][:, b]).sum().item(),
                        "n_eff": (w * data["effective_support"][:, b]).sum().item(),
                        "top1_share": (w * torch.softmax(data["scores"][:, b], -1).max(-1).values).sum().item(),
                        "peak_in_value_frac": (w * in_value).sum().item(),
                        "peak_in_needle_frac": (w * in_needle).sum().item(),
                        "remote_blocks": data["probability"].shape[1],
                    })
                writers["needles"].row(row)


# ---------------------------------------------------------------- driver

def load_o_proj(meta, layer, device):
    model = meta["arguments"]["model"]
    local = Path(model)
    if not local.exists():
        from huggingface_hub import snapshot_download
        local = Path(snapshot_download(
            model, revision=meta["model_commit"], allow_patterns=["*.json", "*.safetensors"],
        ))
    index = json.loads((local / "model.safetensors.index.json").read_text())
    name = f"model.layers.{layer}.self_attn.o_proj.weight"
    with safe_open(local / index["weight_map"][name], framework="pt", device="cpu") as source:
        return source.get_tensor(name).to(device, torch.float64)


def rbs_uplift(layer):
    """r_b * beta_b per complete block and KV head (RBS Eqs. 4-7)."""
    blocks = layer.sequence // B
    keys = layer.k[:blocks * B].float().reshape(blocks, B, layer.k.shape[1], -1)
    radius = torch.linalg.vector_norm(keys - keys.mean(1, keepdim=True), dim=-1).amax(1)  # [blocks, kv]
    low = torch.quantile(radius, 0.5, dim=0)
    high = torch.quantile(radius, 0.9, dim=0)
    return radius * ((radius - low) / (high - low)).clamp(0, 1)


def main():
    args = arguments()
    if (args.out / "metadata.json").exists():
        raise FileExistsError(args.out)
    (args.out / "raw").mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    plan = []
    for capture in args.capture:
        meta = json.loads((capture / "metadata.json").read_text(encoding="utf-8"))
        inv_freq = inverse_frequencies(meta["model_config"], device)
        for sample_dir in sorted((capture / "samples").iterdir()):
            sample_meta = json.loads((sample_dir / "metadata.json").read_text(encoding="utf-8"))
            artifacts = sample_dir / "artifacts.jsonl"
            recorded = {}
            if artifacts.exists():
                for line in artifacts.read_text(encoding="utf-8").splitlines():
                    item = json.loads(line)
                    recorded[item["path"].replace("\\", "/")] = item["sha256"]
            for layer_id in sample_meta["captured_layers"]:
                if args.layers is None or layer_id in args.layers:
                    plan.append((capture, meta, inv_freq, sample_dir, sample_meta, recorded, layer_id))

    reproduce(plan, device, args.out)

    checks = Checks()
    names = (
        "methods", "blocks", "output_rows", "oproj_rows", "block_drop",
        "peaks", "sink", "attenuation", "needles",
    )
    writers = {name: Writer(args.out / "raw" / f"{name}.csv.gz") for name in names}
    inputs, rope_report = [], []
    for capture, meta, inv_freq, sample_dir, sample_meta, recorded, layer_id in plan:
        panel = panel_of(capture)
        common = {"panel": panel, "sample_id": sample_meta["sample_id"], "layer": layer_id}
        layer = Layer(sample_dir, sample_meta, layer_id, device, inv_freq)
        layer.tokens = [
            json.loads(line)["decoded"]
            for line in (sample_dir / "tokens.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        for path in layer.files:
            relative = path.relative_to(sample_dir).as_posix()
            digest = sha256(path)
            inputs.append({
                **common, "file": str(path), "sha256": digest,
                "recorded_match": digest == recorded[relative] if relative in recorded else None,
            })
        o_proj = load_o_proj(meta, layer_id, device)

        # RoPE convention (spec section 6): rotate the saved pre-RoPE tensors.
        token_positions = torch.arange(layer.sequence, device=device)
        q_positions = token_positions if layer.full_q else layer.saved_positions.to(device)
        rope = {**common}
        for label, pre, post, positions in (
            ("k", layer.k_pre, layer.k, token_positions),
            ("q", layer.q_pre.to(device), layer.q.to(device), q_positions),
        ):
            rope.update({f"{label}_{name}": value for name, value in rope_agreement(pre, post, positions, inv_freq).items()})
            checks.update(f"rope_rotation_{label}_ulp", rope[f"{label}_max_ulp"], ROPE_MAX_ULP + 1e-6)
            checks.update(f"rope_rotation_{label}_mismatch_frac", rope[f"{label}_mismatch_frac"], ROPE_MAX_MISMATCH)
        layer.rope_ok = all(
            rope[f"{label}_max_ulp"] <= ROPE_MAX_ULP and rope[f"{label}_mismatch_frac"] < ROPE_MAX_MISMATCH
            for label in ("k", "q")
        )
        rope["passed"] = layer.rope_ok
        rope_report.append(rope)
        print(f"[validation] rope {rope}", flush=True)
        if layer.rope_ok:
            attenuation_table(layer, writers, common)

        uplift = rbs_uplift(layer)
        peaks = {}
        for q_start in tile_starts(layer.sequence, B, FRACTIONS):
            if remote_end_of(q_start) - REMOTE_START <= BUDGET_TOKENS:
                continue
            positions = query_rows(q_start, B, layer.sequence, ROWS_PER_TILE)
            store = {}
            for head in range(layer.heads):
                head_common = {**common, "query_head": head, "kv_head": head // layer.group}
                query, logits = layer.tile_logits(head, positions)
                selection = select(
                    layer, head, q_start, positions, query, logits,
                    uplift[:, head // layer.group], writers, head_common,
                )
                output_errors(layer, head, q_start, positions, query, selection, writers, checks, store, head_common)
                mechanism(layer, head, q_start, positions, query, selection, writers, checks, peaks, head_common)
            output_projection(layer, o_proj, q_start, positions, store, writers, common)
            print(f"[validation] {common} tile {q_start} done", flush=True)
        if layer.rope_ok:
            sink_features(layer, peaks, writers, common)
        if panel == "RULER" and layer.full_q:
            sample = torch.load(sample_dir / "input.pt", map_location="cpu", weights_only=False)
            needle_blocks(layer, sample, writers, checks, common)
        del layer, o_proj
        torch.cuda.empty_cache()

    for writer in writers.values():
        writer.close()
    report = checks.report()
    (args.out / "checks.json").write_text(
        json.dumps({"checks": report, "rope": rope_report}, indent=2), encoding="utf-8"
    )
    (args.out / "metadata.json").write_text(json.dumps({
        "spec": "BLOCK_PROXY_VALIDATION_SPEC.md",
        "captures": [str(path) for path in args.capture],
        "constants": {
            "block": B, "budget_tokens": BUDGET_TOKENS, "sink_tokens": SINK_TOKENS,
            "local_tokens": LOCAL_TOKENS, "remote_alignment": ALIGN,
            "rows_per_tile": ROWS_PER_TILE, "tile_fractions": FRACTIONS,
            "sweep_layer": SWEEP_LAYER, "sweep_k": SWEEP_K, "rescue_budgets": RESCUE_BUDGETS,
            "base_alphas": BASE_ALPHAS, "rbs_base_alphas": RBS_BASE_ALPHAS,
            "rbs_rescue_alphas": RBS_RESCUE_ALPHAS,
        },
        "inputs": inputs,
        "source_sha256": {
            name: sha256(Path(__file__).with_name(name))
            for name in ("block_proxy_validation.py", "block_proxy_study.py", "block_proxy_positions.py")
        },
        "approximations": [
            "Selection logits stay float32 as in block_proxy_study.py so E0 is an identity; mechanism and output quantities use float64.",
            "RBS (arXiv 2609.20971 Eqs. 4-14) on post-RoPE keys; the paper does not state pre/post RoPE. Radius quantiles use all complete 128-token blocks of the prompt per layer and KV head. Offline scoring uses the 16 sampled rows of each Q tile, not all 128; threshold candidates are the fully visible blocks of the tile.",
            "rbs_budget_r* is an offline fixed-budget adaptation: (16-k) mean_raw blocks plus k blocks with the highest RBS rescue score.",
            "FlashPrefill V2 mean correction (FlashPrefillv2 75b58f2; prefill.py use_mean_correction defaults to False; mainloop_fwd_sm90_tma_gmma_ws.hpp L1531-1600; mean kernel in flash_block_sparse_index_triton.py): each unselected, fully visible block adds B*exp(q.kbar*scale) to the denominator and that weight times vbar to the numerator; kbar and vbar are plain token means rounded to BF16. Emulated offline in float64 on mean_raw and thr_base_a0.08 selections.",
            "thr_base_a* is an offline FlashPrefill-V1-style relative threshold on 16 sampled rows; the captured V1 route uses all 128 rows.",
        ],
    }, indent=2), encoding="utf-8")
    failed = [name for name, entry in report.items() if not entry["passed"]]
    print(f"[validation] failed checks: {failed}", flush=True)
    print(f"[validation] complete: {args.out}", flush=True)


if __name__ == "__main__":
    main()
