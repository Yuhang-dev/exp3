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
5. the old monolithic `FlashPrefill.autograd.Function` wrapper is omitted because exp3 profiles and combines the stages explicitly.

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
