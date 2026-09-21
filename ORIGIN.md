# Source and modification record

## FlashPrefill V1

- Upstream repository: `qhfan/FlashPrefill`.
- Pinned upstream commit: `baa612047433a992a00d07dc178205eed065ae14`.
- Upstream file: `ops/flashprefill_native_forward.py`.
- Immediate local source: the already audited copy at `D:/long-context/exp2/upstream/flashprefill_native_forward.py`.
- Vendored exp3 file: `upstream/flashprefill_native_forward.py`.

The pinned upstream repository exposes no `LICENSE` file through the GitHub license endpoint and the local pinned checkout also contains no license file. No license terms are invented here. The source attribution and commit are preserved; redistribution outside this experiment should first resolve the upstream licensing status.

The exp2 audit had already made these changes relative to upstream:

1. removed FLA-only wrappers and made BF16 contiguous inference explicit;
2. fixed the incomplete query-block validity mask;
3. replaced the failing `[Q,K,1]`/axis-0 scoring reduction with the mathematically equivalent `K_mean @ Q.T` row reduction used by `SCORE_IMPL=v1_kmean_qt_row_reduce`.

Exp3 keeps that V1 score, routing rule, protected regions, sorted-index convention, GQA mapping, and sparse QK/softmax/PV computation. Exp3 adds only the following kernel-facing changes:

1. `_flash_forward` writes the exact path log-sum-exp in **natural-log units** as well as its normalized output;
2. thin Python entry points expose V1 mean/scoring, selection, and exact attention separately;
3. the diagonal-block causal test is expressed as an equivalent predicated mask rather than a runtime Triton `if`, avoiding a scheduler diagnostic in the pinned remote environment;
4. compiled routing outputs are cloned in the uncompiled public wrapper so CUDA Graph buffer reuse cannot invalidate masks retained across calls;
5. the old monolithic `FlashPrefill.autograd.Function` wrapper is omitted because exp3 profiles and combines the stages explicitly;
6. Qwen3.5's head dimension 256 uses 32-by-32 compute tiles with four warps and one pipeline stage after the original launch exceeded the RTX 4090's 101376-byte shared-memory limit. Logical routing blocks remain 128 tokens, and each selected block is traversed in four K micro-tiles. Head dimension is included in the exact-path autotune key; the head-dimension-128 candidate set is unchanged. The remote math gate records the selected launch configuration.

The new block variance, Value dispersion, balanced/CGF/dispersion selectors, unselected-mean path, and log-domain merge live in `exp3/kernels.py`; they are project code and are not labeled as an official FlashPrefill V2 kernel.

## LongBench hotpotqa

- Dataset archive: `zai-org/LongBench`, `data.zip`, file `data/hotpotqa.jsonl`; the resolved Hugging Face snapshot is recorded per run.
- Prompt and scorer source: `THUDM/LongBench`, commit `2e00731f8d0bff23dc4325161044d0ed8af94c1e`:
  - `LongBench/config/dataset2prompt.json`
  - `LongBench/config/dataset2maxlen.json`
  - `LongBench/metrics.py`
  - `LongBench/eval.py`

Exp3 uses the official hotpotqa prompt, 32-token generation cap, normalized token-F1, and maximum over reference answers. It applies the Qwen chat template around the official task prompt and reports the result as `LongBench hotpotqa subset`, not as the official full benchmark score.

## RULER parity subset

- Evaluation code: `qhfan/FlashPrefill`, pinned commit `baa612047433a992a00d07dc178205eed065ae14`:
  - `ruler/configs/ruler_32k.yaml`
  - `ruler/data.py`, function `load_ruler`
  - `ruler/model_utils.py`, `tokenize` and `SelfdefinedModel.generate`
- Dataset: `aldjalkdf/ruler`, pinned revision `2a9d66ecfcdbcaa72d692b6e89d1fb3325e7d634`.
- Files: `niah_multikey_{1,2,3}/validation_32768.jsonl` and `niah_multiquery/validation_32768.jsonl`.

Exp3 rebuilds the prompt from `context`, `query`, and `type_needle_v` with the pinned runtime template instead of using the dataset's pre-rendered `input` field. Those strings differ at the separator before the completion prefix (runtime template newline versus stored-input space). It mirrors the official seeded dataset shuffle, no-chat tokenization/truncation, task-specific 50/100-token generation limits, final-prompt-token generation boundary, and case-insensitive answer-substring recall. Per-run metadata records the repository revisions, full source-file hashes, selected source rows and row hashes; `inputs.jsonl` and `inputs.pt` retain the exact rebuilt prompt and input IDs used by the model.

## LongBench v2 native-32K subset

- Dataset: `THUDM/LongBench-v2`, pinned HF revision
  `2b48e494f2c7a2f0af81aae178e05c7e1dde0fe9`, file `data.json`.
- Data SHA256:
  `15d61c22d92c96900b3c4948b6aeea218d3214b676a65df48e7b8555604c7fe2`.
- Prompt/scorer source: `THUDM/LongBench`, commit
  `2e00731f8d0bff23dc4325161044d0ed8af94c1e`, files
  `prompts/0shot.txt` and `pred.py`.

Exp3 uses the complete official zero-shot prompt wrapped in the Qwen chat
template, exact extracted A/B/C/D accuracy, and no prompt truncation. Because
Qwen is held at its native 32K context, the reported task is explicitly a
deterministic token-filtered subset rather than the official 503-row overall
score. Exp3 also keeps its common greedy paired decoding rather than the
upstream script's temperature-0.1 request.

## BFCL V4 agent subset

- Package: `bfcl-eval==2025.12.17`, wheel SHA256
  `8555bc9407a56682ceb7d969e87eb724f6b679deb0ef05114d9c6e786406b103`.
- Reproduction/leaderboard commit:
  `f7cf7359b7ac615a0b294831c5ba2bc95ee4a000`.
- Data: `BFCL_v4_multi_turn_long_context.json` and its official
  `possible_answer` file; the package contains 200 entries.
- Evaluation: official executable backends and `multi_turn_checker` from the
  extracted pinned wheel.

The official package does not register this exact Qwen2.5-7B-Instruct model as
a built-in handler. `bfcl_v4_agent.py` therefore provides a project adapter that
mirrors the package's Qwen `<tool_call>` protocol. The raw data, backend, and
checker remain official and pinned; the resulting comparison is not labeled an
official BFCL leaderboard submission.

## Qwen3.5 CL-bench paired subset

- Model: `Qwen/Qwen3.5-4B`, revision
  `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`.
- Dataset: `tencent/CL-bench`, revision
  `b28a5832a09b0d96c0cf4c22e90d7c60ede25b80`, `CL-bench.jsonl` SHA256
  `d5fc88d4b2eea75c61dd40862021b6ae2fba26bd21b58e8c5e18377a763943be`.
- Official inference/evaluation repository: `Tencent-Hunyuan/CL-bench`, commit
  `16bffd1cfa05927e72ec75c835177d6e23e82172`.
- Official quality judge: `gpt-5.1`, low reasoning effort, strict all-rubrics
  binary task success.

Exp3 uses the model's own thinking-enabled chat template, greedy local decoding,
and the unchanged official messages and rubrics. Only text after `</think>` is
exported to the official judge; raw generated IDs and reasoning remain in the
scorer-independent artifact. The experiment is a deterministic category-balanced
100-task panel under a 65,536-token prompt-plus-generation bound with no truncation,
not a claim of the official 1,899-task leaderboard score.

The model has 24 Gated Delta layers and eight full-attention layers. Exp3 registers
the existing V1 backend only at the standard full-attention interface; the linear
layers keep the Transformers/Hugging Face kernel path. Running the new entrypoints
as `python -m exp3...` is intentional: it lets project code import `exp3.kernels`
without shadowing the separate top-level `kernels` distribution required by
Qwen3.5's FLA and causal-convolution implementations.
