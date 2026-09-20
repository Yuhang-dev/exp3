"""Transformers 4.51 attention backend for dense, V1, and mean-corrected prefill."""

import torch
import torch.nn.functional as F
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

import kernels
from upstream import flashprefill_native_forward as upstream


METHODS = (
    "dense",
    "fp_v1",
    "mean_native",
    "mean_balanced",
    "cgf_mean",
    "dispersion_mean",
)
MEAN_METHODS = {
    "mean_native",
    "mean_balanced",
    "cgf_mean",
    "dispersion_mean",
}


class AttentionBackend:
    def __init__(
        self,
        alpha: float = 0.08,
        sink_blocks: int = 2,
        window_blocks: int = 4,
        last_full_blocks: int = 2,
        block_size: int = 128,
        selector_chunk_tiles: int = 8,
    ):
        self.method = "dense"
        self.alpha = alpha
        self.sink_blocks = sink_blocks
        self.window_blocks = window_blocks
        self.last_full_blocks = last_full_blocks
        self.block_size = block_size
        self.selector_chunk_tiles = selector_chunk_tiles
        self.record = False
        self.profile_records = []
        self.capture_callback = None
        ALL_ATTENTION_FUNCTIONS["sdpa"] = self.forward

    def configure(self, method: str, alpha: float | None = None):
        if method not in METHODS:
            raise ValueError(f"unknown method {method!r}; choose from {METHODS}")
        self.method = method
        if alpha is not None:
            self.alpha = alpha

    def _stage(self, record, name, function):
        if not self.record:
            return function()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        result = function()
        end.record()
        record["events"][name] = (start, end)
        return result

    def _v1_selection(
        self, q, k, scale, record, descriptors_needed, return_auxiliary=False
    ):
        def descriptor_work():
            descriptors = kernels.block_descriptors(k, descriptors_needed, self.block_size) \
                if isinstance(descriptors_needed, torch.Tensor) else None
            mean_k = upstream.block_mean_k(k, self.block_size)
            return descriptors, mean_k

        descriptors, mean_k = self._stage(record, "descriptor", descriptor_work)
        scores = self._stage(
            record,
            "selector",
            lambda: upstream.v1_scores(q, mean_k, scale, self.block_size),
        )

        def index_work():
            indices, counts = upstream.deal_output_score(
                scores,
                self.sink_blocks,
                self.window_blocks,
                self.alpha,
                self.last_full_blocks,
                0,
            )
            selected = kernels.mask_from_indices(indices, counts)
            return indices, counts, selected

        indices, counts, selected = self._stage(record, "indices", index_work)
        if return_auxiliary:
            return descriptors, mean_k, scores, indices, counts, selected
        return descriptors, indices, counts, selected

    def _new_selection(self, q, k, v, scale, record):
        descriptors = self._stage(
            record,
            "descriptor",
            lambda: kernels.block_descriptors(k, v, self.block_size),
        )
        if self.alpha == 0:
            blocks = descriptors.mean_k.shape[1]
            scores = torch.zeros(
                q.shape[0],
                blocks,
                blocks,
                q.shape[2],
                dtype=torch.float32,
                device=q.device,
            )
        else:
            scores = self._stage(
                record,
                "selector",
                lambda: kernels.selector_scores(
                    q,
                    descriptors,
                    scale,
                    self.method,
                    self.block_size,
                    self.selector_chunk_tiles,
                ),
            )

        def index_work():
            selected = kernels.protected_selection(
                scores,
                self.alpha,
                self.sink_blocks,
                self.window_blocks,
                self.last_full_blocks,
            )
            indices, counts = kernels.indices_from_mask(selected)
            return indices, counts, selected

        indices, counts, selected = self._stage(record, "indices", index_work)
        return descriptors, indices, counts, selected

    def forward(
        self,
        module,
        query,
        key,
        value,
        attention_mask,
        scaling,
        dropout=0.0,
        **kwargs,
    ):
        record = {
            "layer": module.layer_idx,
            "method": self.method,
            "events": {},
            "accounting": None,
        }
        overall_start = None
        overall_end = None
        if self.record:
            overall_start = torch.cuda.Event(enable_timing=True)
            overall_end = torch.cuda.Event(enable_timing=True)
            overall_start.record()

        is_prefill = query.shape[2] == key.shape[2]
        if self.method == "dense" or not is_prefill:
            output = F.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=None,
                dropout_p=0.0,
                is_causal=is_prefill,
                scale=scaling,
                enable_gqa=True,
            ).transpose(1, 2).contiguous()
            if self.record and is_prefill:
                sequence = query.shape[2]
                record["accounting"] = {
                    "effective_exact_token_pair_ratio": 1.0,
                    "exact_token_pairs": query.shape[0] * query.shape[1] * sequence * (sequence + 1) // 2,
                    "causal_token_pairs": query.shape[0] * query.shape[1] * sequence * (sequence + 1) // 2,
                    "legacy_block_density": 1.0,
                    "selected_block_entries": None,
                    "causal_block_entries": None,
                    "exact_physical_qk_tiles": None,
                    "mean_proxy_entries": 0,
                    "selector_proxy_entries": 0,
                    "mean_executed_logit_entries": 0,
                    "mean_executed_value_entries": 0,
                    "selector_executed_dot_entries": 0,
                    "selector_physical_qk_tiles": None,
                    "q_tile_size": None,
                    "k_tile_size": None,
                    "score_k_tile_size": None,
                }
        else:
            q = query.transpose(1, 2).contiguous()
            k = key.transpose(1, 2).contiguous()
            v = value.transpose(1, 2).contiguous()
            scale = float(scaling)
            capture_v1 = self.capture_callback is not None and self.method == "fp_v1"
            if self.method in {"fp_v1", "mean_native"}:
                descriptor_value = v if self.method == "mean_native" else None
                selection = self._v1_selection(
                    q,
                    k,
                    scale,
                    record,
                    descriptor_value,
                    return_auxiliary=capture_v1,
                )
                if capture_v1:
                    descriptors, mean_k, scores, indices, counts, selected = selection
                else:
                    descriptors, indices, counts, selected = selection
            else:
                descriptors, indices, counts, selected = self._new_selection(
                    q,
                    k,
                    v,
                    scale,
                    record,
                )

            if capture_v1:
                self.capture_callback(
                    layer=int(module.layer_idx),
                    q=q,
                    k=k,
                    v=v,
                    mean_k=mean_k,
                    scores=scores,
                    indices=indices,
                    counts=counts,
                    selected=selected,
                    scale=scale,
                )

            exact_output, exact_lse = self._stage(
                record,
                "exact",
                lambda: upstream.exact_attention(
                    q,
                    k,
                    v,
                    indices,
                    counts,
                    scale,
                    self.block_size,
                ),
            )
            q_tile_size, k_tile_size = upstream.attention_tile_sizes()
            score_k_tile_size = (
                upstream.score_tile_size()
                if self.method in {"fp_v1", "mean_native"}
                else None
            )
            if self.record:
                record["accounting"] = kernels.selection_accounting(
                    selected,
                    q.shape[1],
                    q_tile_size,
                    k_tile_size,
                    self.method,
                    score_k_tile_size,
                    self.block_size,
                )

            if self.method in MEAN_METHODS:
                mean_output, mean_lse = self._stage(
                    record,
                    "mean",
                    lambda: kernels.mean_tail(
                        q,
                        descriptors,
                        selected,
                        scale,
                        self.block_size,
                        self.selector_chunk_tiles,
                    ),
                )
                output, _ = self._stage(
                    record,
                    "merge",
                    lambda: kernels.merge_exact_mean(
                        exact_output,
                        exact_lse,
                        mean_output,
                        mean_lse,
                    ),
                )
            else:
                output = exact_output

        if self.record and is_prefill:
            overall_end.record()
            record["events"]["attention"] = (overall_start, overall_end)
            self.profile_records.append(record)
        return output, None

    def start_profile(self):
        self.profile_records.clear()
        self.record = True

    def start_capture(self, callback):
        """Attach a synchronous V1-prefill diagnostic callback."""
        self.capture_callback = callback

    def finish_capture(self):
        self.capture_callback = None

    def finish_profile(self):
        self.record = False
        torch.cuda.synchronize()
        rows = []
        stage_names = ("descriptor", "selector", "indices", "exact", "mean", "merge", "attention")
        for record in self.profile_records:
            row = {"layer": record["layer"], "method": record["method"]}
            for name in stage_names:
                event = record["events"].get(name)
                row[f"{name}_ms"] = event[0].elapsed_time(event[1]) if event else 0.0
            for key, value in (record["accounting"] or {}).items():
                if isinstance(value, torch.Tensor):
                    row[key] = value.item()
                else:
                    row[key] = value
            rows.append(row)
        self.profile_records.clear()
        return rows
