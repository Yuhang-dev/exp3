# exp3: task-quality-first sparse prefill

This directory implements the P0/P1 loop from `research/refinement-prefill-2026-09-16/IMPLEMENTATION_BRIEF.md` for frozen `Qwen/Qwen2.5-7B-Instruct`, BF16, batch 1, and one RTX 4090.

The primary outputs are paired final-answer quality and full-model prefill latency. Internal attention agreement is used only by the focused implementation checks.

## Implemented methods

| Method | Selector | Unselected full-history blocks |
| --- | --- | --- |
| `dense` | all causal tokens | n/a |
| `fp_v1` | audited FlashPrefill V1 routing | dropped |
| `mean_native` | exactly the same V1 routing as `fp_v1` | block mean K/V |
| `mean_balanced` | per-query normalized mean-mass, then tile mean | block mean K/V |
| `cgf_mean` | per-query normalized diagonal second-order mass, then tile mean | block mean K/V using first-order mass |
| `dispersion_mean` | second-order probability × `sqrt(u)` × Value dispersion | block mean K/V using first-order mass |

The mean-corrected path exports the exact path output and natural-log LSE, computes an unselected block-mean output/LSE, and combines them with `logaddexp`. Selected blocks are never added again through the mean path. `cgf_mean` uses the second-order term only for routing.

Common routing settings are block size 128, sink 2 blocks, window 4 blocks, and the final 2 query blocks fully exact. Decode is always dense Flash SDPA over each method's own complete KV cache.

## Remote environment

```bash
cd /root/autodl-tmp/exp3
source ./env.sh
```

`env.sh` reuses `/root/autodl-tmp/conda/envs/exp1` and its existing Hugging Face, Triton, and TorchInductor caches. It does not install or upgrade packages.

## 1. Prepare fixed inputs without running the model

Calibration inputs (4 samples for each synthetic variant and length, plus 8 complete hotpotqa prompts):

```bash
python -u evaluate.py \
  --prepare-only \
  --split calibration \
  --tasks synthetic_kv_retrieval hotpotqa \
  --lengths 16384 32768 \
  --synthetic-samples 4 \
  --hotpot-samples 8 \
  --max-new-tokens 128 \
  --out results/calibration_inputs
```

For synthetic data, `--lengths` is the total native context budget, including the generation reserve. Thus a 32768 run with 128 generation tokens constructs a prompt of at most 32640 tokens. `--prompt-budgets` can be used instead when an exact prompt budget is wanted.

The saved `inputs.pt` contains exact token IDs. `inputs.jsonl` records task, split, source/sample ID, seed, SHA256, actual token count, question, standards answers, and synthetic evidence positions. Calibration and holdout use disjoint synthetic seeds and disjoint hash buckets of LongBench source IDs.

## 2. P0 quick loop

```bash
bash run_quick.sh
```

This runs the focused GPU math checks and then two 4K synthetic samples with `dense`, `mean_native`, `cgf_mean`, and `dispersion_mean`, including one independent profile input per task/length/config. It is a code-path check, not a research result.

Equivalent direct command after the math check:

```bash
python -u evaluate.py \
  --split quick \
  --tasks synthetic_kv_retrieval \
  --lengths 4096 \
  --synthetic-samples 1 \
  --methods dense mean_native cgf_mean dispersion_mean \
  --alpha 0.08 \
  --max-new-tokens 128 \
  --repeats 3 \
  --profile \
  --out results/quick
```

## 3. P1 calibration

```bash
bash run_pilot.sh calibration
```

This runs all six methods at `alpha=0.08` on synthetic 16K/32K budgets and the complete-input `LongBench hotpotqa subset`. Synthetic calibration has 4 multi-key and 4 multi-query samples per length; hotpotqa has 8 samples. The official hotpotqa task cap is 32 generated tokens even though the global upper bound is 128.

To reuse already prepared inputs exactly:

```bash
python -u evaluate.py \
  --inputs results/calibration_inputs/inputs.pt \
  --split calibration \
  --tasks synthetic_kv_retrieval hotpotqa \
  --lengths 16384 32768 \
  --config initial_candidates.json \
  --repeats 3 \
  --profile \
  --out results/calibration
```

## 4. Freeze a configuration and run holdout

After reading the calibration `REPORT.md`, create a JSON file containing only the frozen configurations to carry forward. The schema is:

```json
{
  "candidates": [
    {"method": "dense"},
    {"method": "mean_native", "alpha": 0.08},
    {"method": "cgf_mean", "alpha": 0.04}
  ]
}
```

Then run the disjoint holdout (8 samples for each synthetic variant and length, plus 16 hotpotqa samples):

```bash
bash run_pilot.sh holdout results/calibration/frozen_selection.json results/holdout
```

The code does not tune from holdout scores and does not silently change a failed/OOM configuration. An exception is written into `metadata.json` and then re-raised.

## 5. Explicit small alpha sweep

The supplied sweep has one mean baseline and two new candidates at `alpha={0.04,0.08,0.16}`, plus dense:

```bash
bash run_pilot.sh sweep results/calibration_sweep
```

For a different two-candidate set, call the CLI explicitly with repeated entries such as:

```bash
python -u evaluate.py \
  --split calibration \
  --tasks synthetic_kv_retrieval hotpotqa \
  --lengths 16384 32768 \
  --synthetic-samples 4 --hotpot-samples 8 \
  --candidate dense \
  --candidate mean_balanced:0.04 --candidate mean_balanced:0.08 --candidate mean_balanced:0.16 \
  --candidate cgf_mean:0.04 --candidate cgf_mean:0.08 --candidate cgf_mean:0.16 \
  --candidate dispersion_mean:0.04 --candidate dispersion_mean:0.08 --candidate dispersion_mean:0.16 \
  --repeats 3 --profile \
  --out results/custom_sweep
```

## 6. Rebuild a report only

```bash
python -u report.py results/calibration
# or
bash run_pilot.sh report results/calibration
```

## Output contract

| File | Contents |
| --- | --- |
| `metadata.json` | environment/GPU, model revision/config, sources, formula version, dtype/RoPE/cache/routing settings, run arguments and failure status |
| `inputs.jsonl`, `inputs.pt` | human-readable provenance and exact token IDs |
| `predictions.jsonl` | one generation per config/sample, generated token IDs/text, parsed answer, end reason and task score |
| `quality.csv` | de-duplicated config/sample task score and paired dense delta |
| `timings.csv` | every synchronized full-model prefill repeat and peak allocated/reserved memory |
| `profile.csv` | per-layer independent descriptor/selector/indices/exact/mean/merge timing and density/execution counts |
| `summary.csv`, `REPORT.md` | per-task/per-length quality–latency comparison, sample counts, paired speedups and explicit negative results |
| `quality_latency.png` | per-task/per-length quality versus measured prefill latency |

Main timing starts after input IDs are already on the GPU and ends after the final-position LM head and first-token argmax synchronize. It includes descriptor construction, routing, exact/mean attention, merge, projections, norms, MLPs, and the complete KV cache. It excludes model loading, tokenization, H2D, compilation and autotune. Each target shape/config is warmed first; method order rotates across the three raw repeats.

The independent profile records three different quantities: effective exact causal token-pair ratio, actual Triton QK micro-tile loops for the selected exact path, and logical versus physically executed mean/selector proxy entries. The current PyTorch proxy GEMMs evaluate masked future/selected entries too, so both the useful entries and the executed dense proxy entries are reported. Mean block multiplicity is not counted as if the proxy executed one operation per original KV token.

## Focused correctness checks

```bash
python -u check_math.py --out results/check_math.json
```

The check covers an incomplete final block, Q padding, 28:4 GQA, model scaling, all-exact output and natural-log LSE against dense FP32, a hand-fixed exact+mean mask, all three new selector formulas, `mean_native`/`fp_v1` mask identity, and fixed-route future-K/V causality. It also records the zero-Value-dispersion counterexample showing that the dispersion heuristic is not an output-error bound.

## Current execution status and known cost

Local work has run Python bytecode compilation only. No CUDA kernel check, model generation, latency measurement, or task-quality experiment has been run locally; no speed or quality gain is claimed.

The first implementation deliberately favors an auditable closed loop:

- V1 exact attention remains Triton, but new descriptors, selectors, unselected-mean attention, and LSE merge are chunked PyTorch GPU tensor operations rather than a fused production kernel.
- `mean_native` computes the audited BF16 V1 mean-K route and separate FP32 K/V descriptors, so its measured cost honestly includes this duplicate descriptor work.
- Selector and mean paths scan all full historical proxy blocks. They avoid `[L,L]` and `[L,N_blocks,d_v]` tensors, but their proxy scan can still dominate at 32K; `profile.csv` is intended to expose that cost.
- The implementation is only for no-history full prefill followed by standard single-token dense decode. It does not implement chunked/paged serving, FP8, YaRN/128K, Value sketches, dual prototypes, or SGLang integration.

Source and modification details are in [ORIGIN.md](ORIGIN.md); live progress is tracked in [CHECKLIST.md](CHECKLIST.md).
