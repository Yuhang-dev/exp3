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
- [x] Re-run local Python compilation/diff checks, validate `run_ruler.sh` with Git Bash `bash -n`, and smoke-test `rescore.py` against all 144 saved calibration predictions (144/144 unchanged; PASS on 2026-09-17).
- [x] Document runnable remote quick/calibration/holdout/report/sweep commands in `README.md`.
- [x] Document known reference-path costs and all GPU checks/experiments not run locally.
- [x] Remote: rerun `check_math.py` on RTX 4090 after the CUDA Graph output-lifetime and Triton diagonal-branch fixes (PASS; completion of `run_quick.sh` confirms all focused checks returned successfully).
- [x] Remote: run the 4K quick loop and confirm all required output files are populated (`metadata.status=complete`; report and independent profile generated).
- [x] Remote: resolve the FlashPrefill V1 32K reproduction discrepancy before method selection.
  - [x] Add the pinned official RULER `mk1/mk2/mk3/mq` loader, no-chat prompt path, official task token limits, and substring-recall scorer.
  - [x] Flush every raw generation to a scorer-independent artifact before scoring; preserve full rebuilt prompts, exact input/generated token IDs, scorer text/details, dataset file/row hashes, scorer hash/version, console logs, and independent offline rescoring snapshots.
  - [x] Run the expanded V1 math check on the remote RTX 4090: block mean, proxy score, threshold/protected selection, `mean_native`/`fp_v1` mask identity, exact/mean outputs and causal checks all passed.
  - [x] Run and audit the dense/`fp_v1` 20-sample-per-task RULER pilot. The completed four-task macro average is 96.25 dense versus 94.69 V1 (-1.56); all 160 raw generations rescore identically, and artifact hashes match the saved manifest. See `RULER_PILOT_AUDIT.md`.
  - [x] Default the remote wrapper to the accessible Hugging Face mirror transport, retain the same pinned repository/revision and file hashing, reuse an existing passed `check_math.json`, and keep timestamped logs across attempts.
  - [x] Close the parity check without a 100-sample expansion by user decision. The 20-sample-per-task pilot is sufficient for the scoped diagnosis (paper-like small V1 delta and no scorer mismatch); it is not presented as a formal non-inferiority estimate.
- [x] Re-score the saved remote calibration generations with `diagnose_quality.py` and inspect per-target binding errors, wrong values, omissions, and max-token endings.
- [~] Remote: after the V1 parity check, choose an explicit alpha sweep, freeze selected configurations, then run the disjoint holdout.
  - [x] Retain all four mean-corrected methods in the alpha sweep by user decision, including `cgf_mean` and `dispersion_mean`, so their initial `alpha=0.08` ranking can be checked across thresholds and actual densities.
  - [x] Restrict the sweep to the discriminating 32K synthetic calibration set and freeze the 14 configurations in `focused_sweep_candidates.json` (dense, V1, and three alpha values for each mean-corrected method).
  - [x] Audit the apparent HotpotQA `dispersion_mean` gain: all +3.57 aggregate points come from one of eight calibration samples (0 to 28.57), while the other seven dense/dispersion pairs are unchanged. Preserve it as a candidate signal for the 16-sample disjoint holdout, not as a stable gain or an alpha-tuning target.
  - [ ] Remote: run `bash run_pilot.sh sweep results/calibration_sweep_32k`.
  - [ ] Select and freeze configurations from the sweep without consulting holdout scores.
  - [ ] Remote: run the disjoint 16K/32K synthetic plus HotpotQA holdout with the frozen file.

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
- 2026-09-17: audited the scale and timing disclosures of the V1 paper/repository. RULER uses 13 tasks, up to 100 samples per task, and six lengths, yielding a nominal 1,300 samples per model/method/length and 7,800 per model/method. The paper reports H20 single-request TTFT but not complete benchmark wall time, warmup/repetition protocol, or result logs; Qwen2.5-7B 32K TTFT is 4.735 s dense versus 3.534 s FlashPrefill.
- 2026-09-17: implemented the official 32K RULER parity entrypoint for `niah_multikey_{1,2,3}` and `niah_multiquery`, initially 20 samples per task and optionally 100. The data and FlashPrefill revisions are pinned; generation artifacts and scoring inputs are independently hashed and retained, and `rescore.py` can create a new scorer snapshot without model inference. Python compilation, diff checks, shell syntax, and a 144-prediction offline rescoring smoke test passed; no new GPU result exists yet.
- 2026-09-17: the first remote RULER pilot attempt completed every focused math check, including the newly added V1 mean/score/selection references, with all checks passing. It then stopped at the first RULER download because the official Hugging Face endpoint timed out; model generation and scoring never began. The runner now defaults to `hf-mirror.com` only as a transport, reuses the saved passing math result, and writes a new timestamped log for each attempt so the failure record is not overwritten.
- 2026-09-17: the retried 32K official-data RULER pilot completed on 20 rows each for `multikey_1/2/3` and `multiquery`. Dense scores are 100/95/90/100; V1 scores are 95/95/90/98.75, giving a four-task macro change of -1.56 and a median 1.23x paired full-prefill speedup. Only four of 80 paired samples change score (three regressions, one recovery), and the failures are wrong number/UUID or key-value binding rather than scorer disagreement. All 160 raw generations rescore unchanged, input/generation/prediction hashes match the manifest, and the source archive is preserved under the ignored `results/archives/` directory.
- 2026-09-17: user accepted the 20-sample-per-task parity evidence as sufficient for the present diagnostic and chose not to spend GPU time on the 100-sample expansion. The parity phase is closed: it supports a small paper-like V1 quality delta and rules out a scorer-only explanation for the custom-task collapse, but is not labeled a formal non-inferiority result. Candidate selection and holdout are the next phase.
- 2026-09-17: configured the focused P1b continuation on only the eight discriminating 32K synthetic calibration inputs, removing saturated 16K and non-discriminating HotpotQA repeats. By user decision, the sweep retains `mean_native`, `mean_balanced`, `cgf_mean`, and `dispersion_mean` at `alpha={0.04,0.08,0.16}`, plus dense and V1 references: 14 configurations and 112 scored generations in total.
- 2026-09-17: re-audited the tentative HotpotQA dispersion signal before pruning. Its +3.57 aggregate change is caused by one calibration sample improving from 0 to 28.57; the other seven paired scores do not change. `dispersion_mean` therefore stays in the 32K alpha sweep, while the real-task signal is reserved for confirmation on the 16 unseen HotpotQA holdout examples after configuration freeze.
