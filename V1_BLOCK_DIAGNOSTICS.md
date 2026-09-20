# FlashPrefill V1 block-structure capture

This diagnostic freezes the evidence needed to study V1 before proposing a new
selector. It does **not** introduce a new method and its runtime must never be
reported as prefill latency.

## What is saved

For every captured layer, the default capture writes:

- `key_pre_rope.pt`: token-level `k_proj` output, reshaped to
  `[tokens, kv_heads, head_dim]`. This is the appropriate view for studying
  content structure without absolute-position rotation.
- `key_post_rope.pt`: token-level K actually consumed by V1. V1 mean pooling and
  routing operate on this representation.
- `mean_k_v1.pt`: the exact BF16 result produced by the pinned V1
  `block_mean_k` kernel.
- `key_structure.pt`: FP32 block mean and diagonal variance, norm and centered
  energy, mean-reconstruction MSE, directional concentration, token-to-mean
  cosine, adjacent-token cosine, first/last cosine, and first-half/second-half
  mean cosine. Raw K remains the source of truth for any later statistic such as
  covariance spectra, clustering, or prototypes.
- `v1_route.pt`: V1 proxy scores, exact selected indices/counts, selected mask,
  protected mask, threshold-routed mask, causal mask, and a dense-oracle mask
  with the same non-protected block budget.
- `dense_block_oracle.pt`: full causal token-level attention probabilities,
  aggregated **after softmax** into mean/max block mass, plus every query row's
  retained mass and dense-vs-selected output error. For every fully visible
  query/key block pair it also saves the Mean Pool Jensen gap
  `logmeanexp(scale*qk) - mean(scale*qk)`, logit standard deviation, and
  max-minus-mean. It additionally preserves the exact and V1-proxy log mass
  after aggregating all query rows in the tile. These distinguish a poor
  exponential-mass approximation from a later threshold/aggregation failure.
- `block_summary.csv`, `tile_summary.csv`, and `layer_summary.csv`: flat tables
  joining intrinsic K structure to selection, missed dense mass, and output
  error.
- `input.pt`, `capture_input_ids.pt`, `input.json`, `tokens.jsonl`, and
  `blocks.jsonl`: the complete saved sample, exact tokens entering the captured
  sparse prefill, task metadata, token mapping, and decoded block text. For
  RULER, the final prompt token is excluded by default to mirror the pinned
  upstream quality-generation boundary; `--include-ruler-final-token` overrides
  this for a full-prompt diagnostic. Synthetic `blocks.jsonl` rows also mark
  overlaps with the saved target-evidence token spans.
- `metadata.json` and `artifacts.jsonl`: input/source hashes, per-tensor SHA256,
  versions, configuration, status, file descriptions, and sizes.

The dense oracle recomputes all causal QK logits in FP32 chunks. V is used only
to measure the output error caused by the already chosen V1 mask; V does not
participate in routing.

Each layer is rejected before being marked complete if block probabilities do
not sum to one, future blocks receive mass, selected counts disagree with the
saved mask, protected blocks are absent, the same-budget oracle changes the
routed budget, retained mass leaves `[0,1]`, or any core oracle value is not
finite. The recorded `sanity` dictionary preserves the measured tolerances.

Full `[tokens, tokens]` attention matrices are deliberately not saved. At 32K,
one FP32 matrix is 4 GiB for just one head and one layer. The default preserves
the useful sufficient views instead:

- full-attention block mass per query tile and head;
- full-attention retained mass and output error per query token and head;
- optional row-by-block mass where query-row aggregation itself is under study;
- optional Q/V so any new proxy can be recomputed without another model forward.

## Recommended first capture

Run a one-layer 4K smoke before committing to the 32K archive:

```bash
bash run_v1_block_diagnostics.sh \
  results/quick/inputs.pt \
  synthetic-quick-4096-0-multi_key \
  results/v1_blocks_smoke_4k \
  --layers 0
```

Successful completion requires both the root and sample `metadata.json` files to
say `"status": "complete"`; the first data check is
`samples/synthetic-quick-4096-0-multi_key/layer_summary.csv`.

Use one multi-key and one multi-query sample for lossless mechanism discovery.
Saving Q and V is recommended here because it permits later offline tests of a
new QK- or value-aware proxy without rerunning the model:

```bash
bash run_v1_block_diagnostics.sh \
  results/calibration_sweep_32k/inputs.pt \
  synthetic-calibration-32768-0-multi_key \
  results/v1_blocks_32k/multi_key_0 \
  --save-q-layers all \
  --save-v-layers all \
  --save-row-block-mass-layers 0 7 14 21 27
```

```bash
bash run_v1_block_diagnostics.sh \
  results/calibration_sweep_32k/inputs.pt \
  synthetic-calibration-32768-4-multi_query \
  results/v1_blocks_32k/multi_query_4 \
  --save-q-layers all \
  --save-v-layers all \
  --save-row-block-mass-layers 0 7 14 21 27
```

To capture several samples in one model load, call the Python entrypoint and
repeat `--sample-id`:

```bash
python -u v1_block_diagnostics.py \
  --inputs results/calibration_sweep_32k/inputs.pt \
  --sample-id synthetic-calibration-32768-0-multi_key \
  --sample-id synthetic-calibration-32768-4-multi_query \
  --out results/v1_blocks_32k/two_samples \
  --save-q-layers all \
  --save-v-layers all \
  --save-row-block-mass-layers 0 7 14 21 27
```

The output directory is intentionally non-overwriting. Use a new directory for
a changed configuration so evidence from different runs cannot be mixed.

## Avoiding the earlier small-sample problem

The two rich samples are for hypothesis formation only. They are not evidence
that a structural relation generalizes. After identifying candidate block
features, run a lighter cross-sample capture on all eight 32K calibration
examples, using five depth checkpoints:

```bash
python -u v1_block_diagnostics.py \
  --inputs results/calibration_sweep_32k/inputs.pt \
  --all-samples \
  --layers 0 7 14 21 27 \
  --no-save-raw-k \
  --out results/v1_blocks_32k/calibration8_summary
```

Only after a relation survives the multi-key/multi-query samples separately
should it be checked on the saved RULER inputs. Do not average raw task scores;
analyze structure/error correlations by task, sample, layer, and head first.

## Approximate 32K storage for Qwen2.5-7B

Per sample across all 28 layers:

| Artifact | Approximate size |
| --- | ---: |
| pre- and post-RoPE raw K | 1.75 GiB |
| V1 route + dense block oracle + per-row summaries | about 3 GiB |
| optional post-RoPE Q | 6.13 GiB |
| optional V | 0.88 GiB |
| optional row-by-block mass | 0.44 GiB per layer |
| optional dense + selected output vectors | 0.44 GiB per layer |

The default all-layer capture is therefore roughly 5 GiB, while the recommended
Q/V-rich capture is roughly 12–13 GiB before optional row mass.
Exact size is recorded in the manifest. Check available disk before using every
optional flag on every sample.

## Interpretation boundaries

- Pre-RoPE K describes projected content structure; post-RoPE K describes what
  V1 actually pools. A post-RoPE pattern must not automatically be called a
  semantic block pattern.
- `exact_block_probability_mean` is the mean of true row-wise dense attention
  mass, not softmax of an averaged logit.
- `oracle_same_budget_mask` retains the same protected blocks and replaces only
  V1's routed remote blocks with the highest dense block mass. It is diagnostic,
  not an implementable selector.
- Output error can be large even when omitted attention mass is small because V
  changes the weighted value. This diagnoses the consequence; it does not make
  V part of V1.
- Any method idea formed from the rich two-sample capture remains exploratory
  until it holds on the broader saved inputs.
