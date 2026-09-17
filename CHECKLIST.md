# exp3 implementation checklist

Last updated: 2026-09-17

Status legend: `[x]` complete, `[~]` in progress, `[ ]` pending, `[!]` blocked/failed.

## Contract and scope

- [x] Read `D:/long-context/AGENTS.md`.
- [x] Read `research/refinement-prefill-2026-09-16/AGENT_PROMPT.md`.
- [x] Read `research/refinement-prefill-2026-09-16/IMPLEMENTATION_BRIEF.md`.
- [x] Confirm implementation is isolated to `exp3/`; do not modify `exp1/exp2` or existing results.
- [x] Record that local work is limited to necessary syntax/static checks; GPU checks and experiments remain remote.

## Phase 1 — audit and scaffold

- [x] Audit reusable `exp2` model integration, V1 kernel, data loading, timing, and reporting paths.
- [x] Copy the audited V1 kernel into `exp3/upstream/` and document provenance/modifications in `ORIGIN.md`.
- [x] Create the minimal `exp3` module and CLI layout.
- [x] Copy/adapt the fixed remote environment entrypoint without upgrading dependencies.

## Phase 2 — attention implementation

- [x] Preserve `dense` and `fp_v1` baselines with full-prefill cache and dense single-token decode.
- [x] Implement block descriptors (`mean K/V`, diagonal K variance, Value dispersion) with valid-token counts.
- [x] Implement selectors: `mean_native`, `mean_balanced`, `cgf_mean`, `dispersion_mean`.
- [x] Preserve identical masks for `fp_v1` and `mean_native`.
- [x] Implement selected-exact + unselected-mean output and natural-log LSE merge.
- [x] Implement exact/mean/selector accounting for effective token-pair density, physical tiles, and proxy entries.
- [x] Add opt-in independent layer/profile timing outside the main timing path.

## Phase 3 — tasks and scoring

- [x] Implement deterministic `synthetic_kv_retrieval` generation for multi-key and multi-query variants.
- [x] Enforce token-budget construction without truncating evidence or padding after the final question.
- [x] Implement parsed value exact match, all-target EM, and per-target accuracy.
- [x] Implement `LongBench hotpotqa subset` loading from the existing archive cache.
- [x] Verify and implement the task prompt and official token-F1 normalization/scorer.
- [x] Implement fixed, disjoint calibration/holdout splits and persist IDs, hashes, token IDs, and labels.

## Phase 4 — experiment loop and outputs

- [x] Implement one independent prefill/cache per method and dense greedy decode.
- [x] Implement rotated, warmed, synchronized full-model prefill timing with raw repeats and medians.
- [x] Record peak allocated/reserved memory and failures without silent fallback.
- [x] Emit `metadata.json`, `inputs.jsonl`, `inputs.pt`, `predictions.jsonl`, `quality.csv`, `timings.csv`, and `profile.csv`.
- [x] Implement report aggregation to `summary.csv`, `REPORT.md`, and `quality_latency.png`.
- [x] Implement quick, calibration, holdout, report-only, and small explicit alpha-sweep entrypoints.

## Phase 5 — focused checks and handoff

- [x] Add small exact-vs-dense, fixed-mask exact+mean, selector, V1 proxy-score/selection reference, mask-identity, and future-K/V checks.
- [x] Add the zero-dispersion explanatory counterexample.
- [x] Run local syntax checks: `python -m compileall -q D:\long-context\exp3` (PASS on 2026-09-16).
- [x] Document runnable remote quick/calibration/holdout/report/sweep commands in `README.md`.
- [x] Document known reference-path costs and all GPU checks/experiments not run locally.
- [x] Remote: rerun `check_math.py` on RTX 4090 after the CUDA Graph output-lifetime and Triton diagonal-branch fixes (PASS; completion of `run_quick.sh` confirms all focused checks returned successfully).
- [x] Remote: run the 4K quick loop and confirm all required output files are populated (`metadata.status=complete`; report and independent profile generated).
- [~] Remote: resolve the FlashPrefill V1 32K reproduction discrepancy before method selection: add explicit V1 score/mask parity against the pinned upstream implementation, then run dense and `fp_v1` on the official RULER multi-key/multi-query subset with the official prompts and scorer.
- [x] Re-score the saved remote calibration generations with `diagnose_quality.py` and inspect per-target binding errors, wrong values, omissions, and max-token endings.
- [ ] Remote: after the V1 parity check, choose any explicit alpha sweep, freeze selected configurations, then run the disjoint holdout.

## Progress log

- 2026-09-16: implementation contract read; isolated `exp3/` created; `exp2` audit started. No GPU checks or experiments run.
- 2026-09-16: P0/P1 code loop, focused GPU check script, reports, shell entrypoints, provenance, and README completed. Local bytecode compilation passed; CUDA/model runs remain pending on the remote RTX 4090.
- 2026-09-16: final interface audit completed. Synthetic counts are per variant, native context budgets reserve generation positions, and profile output separates logical proxy entries from physically executed dense proxy work. Only the three explicitly remote items above remain open.
- 2026-09-16: repository published to `https://github.com/Yuhang-dev/exp3`; local `main` tracks `origin/main`. Repository description records the task-quality-first sparse-prefill scope for subsequent experiments.
- 2026-09-16: first remote RTX 4090 quick run reached the focused checks. Exact output/LSE, exact+mean output/LSE, and all three selector checks passed. The run then exposed CUDA Graph reuse of compiled selector outputs; the public selector now clones both returned tensors outside `torch.compile`. The equivalent diagonal causal mask was also rewritten without a runtime Triton branch to remove the repeated scheduler diagnostic. Remote rerun remains pending.
- 2026-09-16: patched remote quick rerun completed end to end. Both 4K synthetic variants scored 100 with all tested methods, but this is one saturated sample per variant. Every mean-corrected method was slower than dense: `mean_native` used about 71–72% exact pairs and took 1.18–1.20x dense time; `cgf_mean` and `dispersion_mean` used about 65–68% exact pairs and took about 1.30–1.31x dense time. This is retained as the P0 negative latency result; 16K/32K calibration remains pending.
- 2026-09-16: quick report audit found that stage/profile quantities were present in `summary.csv` but absent from `REPORT.md`. The report now renders separate cross-layer stage-time and execution-quantity tables; existing quick artifacts can be rebuilt with the report-only command without rerunning the model.
- 2026-09-17: P1 calibration completed (`metadata.status=complete`). At 16K all synthetic configurations scored 100; `fp_v1` was about 1.10–1.11x dense while mean-corrected paths were 1.03–1.13x slower. At 32K, dense remained 100; `fp_v1` reached about 1.26x speedup but fell to 58.33/33.33, and `mean_native` reached about 1.06x but fell to 58.33/41.67. `mean_balanced` retained the most synthetic quality among mean methods (75/91.67) but was about 1.03x dense time. On 8 HotpotQA samples, `fp_v1` matched dense score at about 1.12x; dispersion's +3.57 score observation was slower and is not treated as a statistical gain.
- 2026-09-17: calibration profile localized the reference-path cost. At 32K, mean-tail took about 502–518 ms and merge about 143 ms; selectors took about 438–445 ms (`mean_balanced`), 655–659 ms (`cgf_mean`), and 821–825 ms (`dispersion_mean`). The existing dense PyTorch proxy computes substantially more entries than the logical unselected set. Candidate sweep/freeze and holdout remain pending.
- 2026-09-17: FlashPrefill V1 paper audit found a material unresolved reproduction gap. Its Qwen2.5-7B 32K RULER aggregate changes only from 90.14 to 88.25, whereas the custom 32K retrieval tasks fall from 100 to 58.33/33.33. The published `alpha=0.08`, 128-token blocks, sink/window settings, and about 20.8% density agree with this run's configuration and about 20.4% density. However, the official run uses RULER prompts without the chat template and substring-recall scoring, while this project uses denser same-format KV distractors, three required answers, the chat template, and parsed exact match. Until upstream numerical parity and an official-RULER subset run are complete, the observed failure is treated as task-local rather than a general V1 quality conclusion.
- 2026-09-17: added an offline saved-generation diagnosis. It independently recomputes every score and reports answer-value substring recall alongside strict parsing, classifying each failed target as case-only, correct-value-but-unparsed, wrong value after the requested key, key without a value, or complete omission. The remote calibration artifacts are required to determine which mechanism caused the reported 32K loss.
- 2026-09-17: analyzed the supplied calibration artifacts (`SHA256 25C74F...080811`). All 144 stored scores reproduce exactly; no synthetic output reached the generation limit, and value-only recall does not recover `fp_v1` (58.33 multi-key, 33.33 multi-query). Every failed target emits the requested key with a wrong value. Of `fp_v1`'s 13 failures, six preserve the correct random suffix under the wrong value index, two duplicate another target value, and five hallucinate within the shared namespace. This localizes the measured loss to exact copying/key-value binding after sparse prefill, while V1 numerical parity and official RULER parity remain pending.
