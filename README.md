# exp3: task-quality-first sparse prefill

This directory implements the P0/P1 loop from `research/refinement-prefill-2026-09-16/IMPLEMENTATION_BRIEF.md` for frozen `Qwen/Qwen2.5-7B-Instruct`, BF16, batch 1, and one RTX 4090. It also contains an isolated Qwen3.5-4B Full/V1 continuation for CL-bench; that path uses a separate environment and result namespace.

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

## 5. Focused 32K alpha sweep

The supplied sweep compares all four mean-corrected methods at
`alpha={0.04,0.08,0.16}`. Dense and `fp_v1:0.08` are rerun in the same batch as paired references.
It uses only the discriminating 32K synthetic multi-key and multi-query calibration inputs: 4
samples per variant, 8 inputs total, and 14 configurations (112 scored generations). The saturated
16K inputs are not repeated. The eight-row HotpotQA calibration subset is also not swept: its
apparent `dispersion_mean` gain came entirely from one sample, so choosing alpha against it would
retune to that case. The frozen continuation will instead test the chosen candidates on 16 disjoint
HotpotQA holdout samples. Although `cgf_mean` and `dispersion_mean` were slower and lower-quality
than `mean_balanced` on the 32K synthetic calibration at `alpha=0.08`, they remain in this sweep by
explicit experiment decision so their ranking can be checked across thresholds and actual densities.

```bash
bash run_pilot.sh sweep results/calibration_sweep_32k
```

The completed 32K sweep is audited in [SWEEP_32K_AUDIT.md](SWEEP_32K_AUDIT.md), including
all alpha results, per-target failures, measured latency/density, and preserved artifact hashes.
All 112 saved generations rescore unchanged. This is still the same eight custom calibration
inputs, not an expanded RULER evaluation; holdout selection/freeze remains pending.

The exact candidate list is frozen in `focused_sweep_candidates.json`. For a different candidate
set, call the CLI explicitly with repeated entries such as:

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

## 7. Re-score and diagnose saved generations

This does not load the model or rerun generation. It verifies every stored score and separates strict
`KEY=VALUE` failures from outputs that still contain the correct answer value:

```bash
python -u diagnose_quality.py results/calibration
```

It writes `quality_rescored.csv`, `target_diagnosis.csv`, and `QUALITY_DIAGNOSIS.md` into the run directory.
`Value recall` is a diagnostic only; the primary synthetic metric remains parsed per-target exact match.

## 8. Official RULER parity run

The parity run uses the four retrieval tasks most directly related to the custom failure:
`niah_multikey_1`, `niah_multikey_2`, `niah_multikey_3`, and `niah_multiquery`.
It pins the FlashPrefill evaluation code at commit
`baa612047433a992a00d07dc178205eed065ae14` and the
[`aldjalkdf/ruler`](https://huggingface.co/datasets/aldjalkdf/ruler) data at revision
`2a9d66ecfcdbcaa72d692b6e89d1fb3325e7d634`. Prompts are rebuilt with the pinned
`load_ruler` templates, no chat template, and the official 50/100-token task limits.
For quality generation it also mirrors the upstream `SelfdefinedModel` boundary: sparse prefill runs
on all but the final prompt token, then the final prompt token and generated tokens use the common
dense single-token decode path. The separately reported exp3 prefill latency retains this project's
existing full-prompt timing definition, so it is not relabeled as the paper's vLLM TTFT.

First run 20 samples per task with only dense and FlashPrefill V1:

```bash
bash run_ruler.sh pilot
```

If the parity result warrants the official cap, run 100 samples per task:

```bash
bash run_ruler.sh full
```

The defaults write to `results/ruler_32k_pilot20` and `results/ruler_32k_full100`.
The wrapper refuses to overwrite a directory that already contains run artifacts and saves
`check_math.log`, timestamped `run_*.log`, and timestamped `rescore_*.log`. It defaults only the
Hugging Face transport endpoint to `https://hf-mirror.com` for the remote AutoDL environment; the
dataset repository and pinned revision remain unchanged and every resolved file is hashed. Set
`HF_ENDPOINT` explicitly to override it. An optional second argument selects a new output directory.
If a previous attempt already produced `check_math.json`, rerunning the same pilot directory reuses
that completed check instead of spending GPU time on it again.

RULER's primary score is the pinned repository's case-insensitive answer-substring recall. Every
generation is flushed to the scorer-independent `generations.jsonl`, and the entire GPU generation
phase finishes before scoring starts. The scored copies then go to `predictions.jsonl`. The official
scorer text (which prepends the completion prefix),
per-answer decisions, normalized diagnostic score, generated token IDs, and scorer version are
stored separately. Each selected source row, full rebuilt prompt, standard
answers, exact input IDs, source file/row hashes, shuffle rank, and tokenizer-result hash are also
saved. Thus a scorer change does not require another model run.

Create another immutable derived scoring snapshot at any time:

```bash
python -u rescore.py results/ruler_32k_pilot20 --tag scorer-audit
# or
bash run_ruler.sh rescore results/ruler_32k_pilot20 scorer-audit
```

This writes `rescoring/scorer-audit/{quality.csv,scores.jsonl,manifest.json,SUMMARY.md}` and records
hashes of both raw artifact files and `scoring.py`. It never loads the model or modifies the original
scores and predictions.

The completed 20-sample-per-task 32K pilot, artifact hashes, paired failures, and interpretation are
recorded in [RULER_PILOT_AUDIT.md](RULER_PILOT_AUDIT.md).

## 9. V1 block-structure diagnostics

`v1_block_diagnostics.py` is a V1-only, non-timed capture path. It saves raw
pre-/post-RoPE K, the actual V1 mean pool and routing state, dense token-attention
block truth, per-row retained mass, and dense-vs-selected output error. Optional
Q, V, row-by-block mass, and output vectors let later hypotheses be tested
offline without another model forward.

```bash
bash run_v1_block_diagnostics.sh \
  results/calibration_sweep_32k/inputs.pt \
  synthetic-calibration-32768-0-multi_key \
  results/v1_blocks_32k/multi_key_0 \
  --save-q-layers all \
  --save-v-layers all \
  --save-row-block-mass-layers 0 7 14 21 27
```

The artifact definitions, storage estimates, second multi-query command, and
multi-sample validation stage are in
[V1_BLOCK_DIAGNOSTICS.md](V1_BLOCK_DIAGNOSTICS.md). These diagnostic runtimes
must not enter `prefill_ms`. The completed 4K/layer-0 capture gate and its
integrity/numerical audit are recorded in
[V1_BLOCK_SMOKE_AUDIT.md](V1_BLOCK_SMOKE_AUDIT.md).

## 10. Modern Full-vs-V1 benchmarks

The modern suite adds a fixed-prompt reasoning benchmark and a stateful agent
benchmark:

```bash
bash prepare_modern_benchmarks.sh
bash run_modern_benchmarks.sh longbench
bash run_modern_benchmarks.sh agent
```

An interrupted Agent suffix can be continued without regenerating completed
episodes via `bash run_modern_benchmarks.sh agent-resume`; the runner validates
the exact saved prefix and archives the failed attempt before continuing.

LongBench v2 uses a deterministic, six-domain round-robin 24-row subset whose
complete official zero-shot prompts fit Qwen's native 32K context without
truncation. BFCL V4 uses the official `multi_turn_long_context` data, simulated
tool backends, and state checker through a documented Qwen2.5 adapter. BFCL
reports pure prefill speedup only for dynamic steps whose prompt hashes still
match after the two methods' trajectories evolve. Pins, exact commands,
limitations, and output definitions are in
[MODERN_BENCHMARKS.md](MODERN_BENCHMARKS.md).
The completed pilot, integrity checks, resume accounting, and interpretation
limits are recorded in [MODERN_BENCHMARK_AUDIT.md](MODERN_BENCHMARK_AUDIT.md).

The final statistical panels contain 116 native-context LongBench v2 questions
and 100 BFCL V4 `multi_turn_base` episodes, each evaluated with Full and V1:

```bash
bash run_final_benchmarks.sh prepare
bash run_final_benchmarks.sh longbench
bash run_final_benchmarks.sh agent
```

The installed Transformers build requires the compatible Hub client pinned for
this suite:

```bash
python -m pip install --no-deps --force-reinstall \
  --index-url https://pypi.org/simple \
  -r requirements-modern.txt
```

The completed 116-question LongBench and 100-episode BFCL runs, independent
offline rescoring, paired uncertainty, sub-evaluations, interrupted-run
accounting, and archive hash are recorded in
[FINAL_100PLUS_AUDIT.md](FINAL_100PLUS_AUDIT.md). The derived tables can be
regenerated from the preserved archive with `analyze_final_benchmarks.py`.

## 11. Qwen3.5-4B Full vs V1 on CL-bench

This experiment uses Tencent-Hunyuan CL-bench, not the unrelated continual-learning
benchmark with the same abbreviation. Sources are pinned before preparation:

```bash
cd /root/autodl-tmp/exp3
export HF_ENDPOINT=https://hf-mirror.com

hf download Qwen/Qwen3.5-4B \
  --revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a \
  --local-dir models/Qwen3.5-4B

hf download tencent/CL-bench \
  --repo-type dataset \
  --revision b28a5832a09b0d96c0cf4c22e90d7c60ede25b80 \
  --local-dir datasets/cl_bench

git clone https://github.com/Tencent-Hunyuan/CL-bench.git third_party/CL-bench
git -C third_party/CL-bench checkout 16bffd1cfa05927e72ec75c835177d6e23e82172
sha256sum datasets/cl_bench/CL-bench.jsonl
```

The expected dataset hash is
`d5fc88d4b2eea75c61dd40862021b6ae2fba26bd21b58e8c5e18377a763943be`.
Create the isolated environment and execute the gates in order:

```bash
bash run_clbench_qwen35.sh setup
bash run_clbench_qwen35.sh prepare
bash run_clbench_qwen35.sh check
bash run_clbench_qwen35.sh smoke
bash run_clbench_qwen35.sh run
```

`setup` clones the existing CUDA/Torch environment to `exp35` and upgrades only
that clone to Transformers 5.17.0. The old Qwen2.5 environment is untouched.
The smoke stage stays online once so the official Hugging Face FLA and causal-conv
kernels used by Qwen3.5's linear layers can be cached; the 100-task run is then
offline and reproducible. If the terminal job is interrupted, continue its exact
saved prefix with:

```bash
bash run_clbench_qwen35.sh resume
```

Qwen3.5-4B has 24 Gated Delta layers and eight full-attention layers. Full and V1
use identical 24 linear layers; V1 replaces prefill only in full-attention layers
3, 7, 11, 15, 19, 23, 27, and 31. The head-dimension-256 Triton path must pass
`check_qwen35_math.py` before model inference. Reported prefill latency still covers
the complete 32-layer forward, LM head, and first-token argmax, while profile density
refers only to the eight full-attention layers.

The fixed panel contains 100 unique tasks selected by deterministic round-robin over
the four categories and their sub-categories after applying the native context bound.
No prompt is truncated. `inputs.pt` retains exact token IDs; every generation saves
raw token IDs, reasoning text, final text, and termination status before scoring.
This is a paired diagnostic subset, not an official full-1,899-task leaderboard score.

Official quality grading is a separate paid/API stage (200 judge calls total):

```bash
export OPENAI_API_KEY=...
export CLBENCH_JUDGE_WORKERS=4
bash run_clbench_qwen35.sh judge
```

The pinned official evaluator uses `gpt-5.1` with low reasoning effort and receives
only the final answer, as required for reasoning models. It can resume partial judge
files without rerunning Qwen. After both files are complete, `clbench_report.py`
validates their exact messages, rubrics, task IDs, and saved final text, then writes
`quality.csv`, `summary.csv`, `subgroups.csv`, `paired_outcomes.csv`, `judge_audit.json`,
and `REPORT.md`. The report separates all categories, sub-categories, and prompt-length
buckets and includes paired Full-only/V1-only outcomes.

## Output contract

| File | Contents |
| --- | --- |
| `metadata.json` | environment/GPU, model revision/config, sources, formula version, dtype/RoPE/cache/routing settings, run arguments and failure status |
| `inputs.jsonl`, `inputs.pt` | human-readable provenance and exact token IDs |
| `generations.jsonl` | scorer-independent generated token IDs and untouched raw text, flushed per record before the scoring phase |
| `predictions.jsonl` | scored copy of each generation with scorer text, parsed answer, end reason and complete score details |
| `scorer_manifest.json` | scorer version, implementation SHA256, pinned prompt/scorer provenance, and offline rescore entrypoint |
| `quality.csv` | de-duplicated config/sample task score and paired dense delta |
| `timings.csv` | every synchronized full-model prefill repeat and peak allocated/reserved memory |
| `profile.csv` | per-layer independent descriptor/selector/indices/exact/mean/merge timing and density/execution counts |
| `summary.csv`, `REPORT.md` | per-task/per-length quality–latency comparison, sample counts, paired speedups and explicit negative results |
| `quality_latency.png` | per-task/per-length quality versus measured prefill latency |
| `rescoring/<tag>/` | derived offline scores plus hashes of the exact inputs, raw predictions, and scorer implementation; no generation rerun |

Main timing starts after input IDs are already on the GPU and ends after the final-position LM head and first-token argmax synchronize. It includes descriptor construction, routing, exact/mean attention, merge, projections, norms, MLPs, and the complete KV cache. It excludes model loading, tokenization, H2D, compilation and autotune. Each target shape/config is warmed first; method order rotates across the three raw repeats.

The independent profile records three different quantities: effective exact causal token-pair ratio, actual Triton QK micro-tile loops for the selected exact path, and logical versus physically executed mean/selector proxy entries. The current PyTorch proxy GEMMs evaluate masked future/selected entries too, so both the useful entries and the executed dense proxy entries are reported. Mean block multiplicity is not counted as if the proxy executed one operation per original KV token.

## Focused correctness checks

```bash
python -u check_math.py --out results/check_math.json
```

The check covers an incomplete final block, Q padding, 28:4 GQA, model scaling, all-exact output and natural-log LSE against dense FP32, a hand-fixed exact+mean mask, all three new selector formulas, V1 block means/proxy scores/threshold selection against independent FP32/PyTorch references, `mean_native`/`fp_v1` mask identity, and fixed-route future-K/V causality. It also records the zero-Value-dispersion counterexample showing that the dispersion heuristic is not an output-error bound.

## Current execution status and known cost

Local work has run Python bytecode compilation only. No CUDA kernel check, model generation, latency measurement, or task-quality experiment has been run locally; no speed or quality gain is claimed.

The first implementation deliberately favors an auditable closed loop:

- V1 exact attention remains Triton, but new descriptors, selectors, unselected-mean attention, and LSE merge are chunked PyTorch GPU tensor operations rather than a fused production kernel.
- `mean_native` computes the audited BF16 V1 mean-K route and separate FP32 K/V descriptors, so its measured cost honestly includes this duplicate descriptor work.
- Selector and mean paths scan all full historical proxy blocks. They avoid `[L,L]` and `[L,N_blocks,d_v]` tensors, but their proxy scan can still dominate at 32K; `profile.csv` is intended to expose that cost.
- The implementation is only for no-history full prefill followed by standard single-token dense decode. It does not implement chunked/paged serving, FP8, YaRN/128K, Value sketches, dual prototypes, or SGLang integration.

Source and modification details are in [ORIGIN.md](ORIGIN.md); live progress is tracked in [CHECKLIST.md](CHECKLIST.md).
