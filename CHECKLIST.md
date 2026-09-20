# exp3 implementation checklist

Last updated: 2026-09-20

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
  - [x] Remote: run `bash run_pilot.sh sweep results/calibration_sweep_32k` (completed 2026-09-17; archive audited 2026-09-18).
  - [x] Preserve the original sweep archive and audit all 112 raw generations, 336 timing repeats, and 784 profile rows. Offline rescoring changes 0 scores; independent aggregation reproduces all 28 summary rows. The eight old input hashes and all 48 common-config outputs match the prior calibration. See `SWEEP_32K_AUDIT.md`.
  - [x] Record the full alpha-dependent CGF/dispersion signal and per-sample failures without pruning either branch: both reach 100 on multi-key at alpha=0.04, while the best sparse multi-query score remains 91.67 from mean-native/balanced. This remains an eight-input custom calibration, not an expanded RULER run.
  - [ ] Select and freeze configurations from the sweep without consulting holdout scores.
  - [ ] Remote: run the disjoint 16K/32K synthetic plus HotpotQA holdout with the frozen file.

## Phase 6 — V1 block-structure diagnosis

- [x] Add a disabled-by-default V1 capture callback without changing the normal benchmark path.
- [x] Save token-level projected K both before RoPE and after RoPE, plus the exact BF16 V1 Mean Pool result.
- [x] Save lossless FP32 block first/second moments and compact within-block sequence/coherence features while retaining raw K for later spectra, clustering, and prototype analysis.
- [x] Recompute full causal token-level attention in bounded query chunks and aggregate it after softmax into block-level truth.
- [x] Preserve per-row and query-tile Mean Pool approximation evidence: Jensen gap, logit spread, exact/proxy aggregated log mass, and normalized V1 score parity.
- [x] Save V1 proxy scores, selected/protected/routed masks, exact indices/counts, and a protected same-budget dense oracle.
- [x] Save per-query-row retained attention mass and dense-vs-selected output error; keep Q, V, row-by-block mass, and output vectors as explicit archival options.
- [x] Join intrinsic block features, selection/miss statistics, and output errors into per-block, per-tile, and per-layer CSV summaries.
- [x] Preserve exact input IDs, token/block text mapping, source hashes, run state, artifact inventory, and a non-overwrite output contract.
- [x] Document a two-sample rich discovery stage and an eight-sample lightweight confirmation stage so the earlier small-sample mistake is not repeated.
- [x] Run local Python bytecode compilation for the capture path (PASS on 2026-09-20); no local CUDA/model run performed.
- [x] Remote: run a short 4K, one-layer smoke capture and inspect tensor shapes/probability sums/output-error sanity. Archive and audit are recorded in `V1_BLOCK_SMOKE_AUDIT.md`.
- [ ] Remote: capture one 32K multi-key and one 32K multi-query sample with Q/V archived and five layers of row-by-block mass.
- [~] Analyze block structure before specifying a new selector; treat the two rich samples as hypothesis formation only.
  - [x] Audit fixed-budget Mean Pool versus exact-block-mass routing on the saved 4K layer-0 tensors: 213 original oracle misses, 59 recovered but 41 new misses, leaving 195. Preserve the results and formulas in `V1_ROUTING_MISS_INVESTIGATION.md`.
  - [x] Separate shortlist coverage from reranking: +6 candidates per active query-tile/head leaves 9 oracle entries outside the candidate set, but exact mass with unchanged V1 query aggregation still misses 194. Save the complete sweep and tensor hashes under `results/audits/v1_routing_miss_4k_20260920.json`; do not call this a working repair.
  - [x] Derive the query-aggregation key-translation counterexample and a residual/margin view of actual rank flips; retain them as hypotheses, without implementing a new selector.
  - [ ] With archived Q/V, complete the estimate-versus-query-normalization 2x2 counterfactual and check residual/common-direction structure against actual recovered and newly lost blocks.
  - [ ] Evaluate any cheap reranking or append-repair rule separately from oracle candidate coverage, including final budget, output error, and measured cost.
- [ ] Confirm any structural relation separately across all eight calibration inputs and then on saved RULER inputs before claiming generality.

## Phase 7 — modern Full vs V1 benchmarks

- [x] Select complementary newer evaluations: LongBench v2 for realistic fixed-prompt long-context reasoning and BFCL V4 `multi_turn_long_context` for stateful agent behavior.
- [x] Pin the LongBench v2 dataset revision/data hash and the `bfcl-eval==2025.12.17` wheel/hash; keep BFCL extracted rather than installing it into the Torch environment.
- [x] Add a LongBench v2 native-32K loader using the official zero-shot prompt and exact A/B/C/D scorer, with no middle truncation and deterministic domain-round-robin selection subject to eligible-row availability.
- [x] Preserve full LongBench prompts, exact token IDs/hashes, source row/file hashes, difficulty/domain metadata, raw generations before scoring, and a domain-level report.
- [x] Add a Qwen2.5 BFCL adapter that mirrors the pinned package's Qwen tool-call protocol while retaining the official data, executable backends, ground truth, and `multi_turn_checker`.
- [x] Save BFCL raw/scored trajectories, generated IDs, decoded calls, tool returns, per-step prompt hashes/token counts, synchronized prefill repeats, overflow/termination state, and checker details; add immutable offline checker snapshots so scorer audits do not rerun the model or trajectory.
- [x] Separate agent episode workload from pure kernel speedup: only identical dynamic prompt hashes enter the paired Full/V1 prefill ratio.
- [x] Add pinned download, preparation, LongBench, and BFCL shell entrypoints plus provenance and interpretation documentation.
- [x] Run local static/parity checks: Python bytecode compilation, Git Bash syntax, pinned-wheel/extracted-file verification, 20-case BFCL selection/parser checks, exact BFCL Qwen prompt-format parity on multi-step fixtures, saved-raw reparsing, the LongBench v2 official answer extractor, and an official `multi_turn_checker` ground-truth self-test all pass.
- [x] Pin `huggingface-hub==0.36.0` for the existing Transformers environment after the remote machine exposed an incompatible Hub 1.x install; use the official PyPI index because the configured Aliyun mirror did not expose this wheel. Source downloads were already complete and hash-correct.
- [x] Remote: download and hash-check both sources, then prepare the 24 fixed LongBench v2 inputs and 20 fixed BFCL cases. The candidate scan emitted a tokenizer warning for an unselected 806,505-token source row; retained LongBench inputs remain constrained to 16,384–32,640 tokens and are not truncated.
- [x] Preserve and diagnose the interrupted BFCL pilot: 31 complete episode/config records and 831 associated repeat rows are intact; three timing rows belong to the incomplete `dense / multi_turn_long_context_147` episode. The actual failure is malformed model tool-call JSON missing `arguments`, not a context-position index error.
- [x] Treat malformed-schema calls as model decode errors without synthesizing arguments, and add exact-prefix Agent resume with failed-attempt archival, partial-timing removal, and official-result reconstruction. Targeted malformed-call and resume-filter tests pass locally.
- [x] Remote: run and audit the 24-row LongBench v2 Full/V1 pilot. Full scores 29.17 and V1 37.50 (four V1-only correct, two Full-only correct); V1 has 1.20× paired prefill speedup at 18.4% exact pairs. Treat the net two-question difference as small-pilot variance, not a quality gain.
- [x] Remote: complete and audit the 20-row BFCL V4 Full/V1 pilot. Both succeed on the same 2/20 episodes; only 38/189 V1 steps have identical Full prompts and their median prefill speedup is 1.01×. Three overflows per method, one step-limit failure per method, a 512-token generation cap, and the 10% Full baseline make this a floor-effect result rather than evidence of V1 parity.
- [x] Save immutable offline rescores (0/48 LongBench and 0/40 BFCL decisions changed), write `MODERN_BENCHMARK_AUDIT.md`, and preserve all inputs/results/resume artifacts in archive SHA256 `383d5c52db535ae53804f0213a265b9996e570bf1cd9dbd896fca2a59c3dd820`.

## Phase 8 — final 100+ paired panels

- [x] Do not mechanically expand the 10%-Full BFCL long-context pilot: retain it as a stress test and use BFCL V4 `multi_turn_base` for the final non-floor Agent panel.
- [x] Count the pinned sources before freezing scale: LongBench v2 has exactly 116 complete official prompts in `[8,192, 32,640]` native tokens; BFCL V4 provides 200 official `multi_turn_base` rows.
- [x] Prepare and hash the complete 116-row LongBench panel without truncation or duplicate sampling. Remote `inputs.pt` SHA256: `fb01a2dcf3934f1c2b05576fb8985e51bc4a636497ed148f9719b2284a54306b`.
- [x] Generalize the BFCL runner/rescorer from a hard-coded long-context category to pinned `multi_turn_base` or `multi_turn_long_context`, including the category-specific simulator mode, source hashes, checker category, result filename, reporting, and exact-prefix resume validation.
- [x] Add `run_final_benchmarks.sh` for fixed 116-row LongBench and 100-row BFCL base Full/V1 runs.
- [x] Remote: prepare and inspect the fixed 100-row BFCL V4 `multi_turn_base` manifest. It contains 100 unique base IDs, covers all 20 involved-class signatures, verifies 33 pinned wheel members, and passes an official-checker ground-truth self-test. Manifest SHA256: `6b17c4ecd6e515ff71394e0c0a78d9ef4cb5d5d394711cb22fa1ca8b955fe4fba`.
- [ ] Remote: run and audit the 116-row LongBench v2 Full/V1 panel.
- [ ] Remote: run and audit the 100-row BFCL V4 `multi_turn_base` Full/V1 panel, including dense success-floor check, trajectory divergence, generation-limit hits, overflow, same-prompt coverage, and paired prefill speedup.
- [ ] Save immutable offline rescores and a combined final archive before interpreting either quality gap.

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
- 2026-09-18: preserved and audited `calibration_sweep_32k.tar.gz` (SHA256 `B5C2DC0105D0B0ACF42D46D7BBD2A06CAF4AB0551B55069E52C1FD7409CE8FAB`). The completed 14-configuration sweep contains eight unchanged calibration inputs and 112 generations; all raw/stored scores agree, all 28 quality/timing summaries reproduce, and the 48 common-config outputs match the prior calibration verbatim. CGF:0.04 scores 100/75 and dispersion:0.04 scores 100/66.67 on multi-key/multi-query, versus mean_native:0.04 at 83.33/91.67; dispersion:0.08 scores 75/75 and dispersion:0.16 scores 83.33/58.33. No sparse candidate matches dense on both tasks. The 89 failed config-target judgments are wrong values or misbindings, not parser omissions or max-token endings. CGF/dispersion at 0.04 remain about 8%/11% slower than dense in the current implementation. Full results, density/profile costs, hashes, and proposed holdout candidates are recorded in `SWEEP_32K_AUDIT.md`; selection/freeze and remote holdout remain pending.
- 2026-09-20: added the V1-only structural capture path. It distinguishes pre-RoPE content K from the post-RoPE K actually mean-pooled by V1, stores dense block-mass truth and exact selected-mask consequences, and optionally archives Q/V for offline method work. The capture never enters benchmark timing. Local compilation passed; remote 4K smoke and 32K captures remain pending.
- 2026-09-20: audited `v1_blocks_smoke_4k.tar.gz` (SHA256 `058319745E8ED28CE3A69AF3092F0CE82E026A2848C759FAE20727CC7E446107`). Root/sample status is complete; 14/14 artifact hashes and all source hashes match. The 3962-token, layer-0 capture has correct `[31,31,28]` route/oracle tensors, probability-sum error `1.79e-7`, zero future mass, zero selection/protection/budget mismatches, and no non-finite values. This passes the capture gate but is only one 4K sample/layer. The audit also separated the formerly ambiguous Jensen-gap CSV maximum into tile-mean and row maxima before 32K collection; see `V1_BLOCK_SMOKE_AUDIT.md`.
- 2026-09-20: investigated actual routing misses using only the saved smoke tensors. Fixed-budget proxy and dense-oracle masks reconstruct exactly. Exact block mass with unchanged query aggregation changes 213 misses to 195, not to near zero. A +6 boundary shortlist covers all but 9 oracle entries, but costs 2388 extra candidate entries and still leaves 194 misses when reranked by exact unnormalized mass. Derived a key-translation invariance diagnostic and residual-boundary refinement questions. Findings and precise numbers are saved; no selector/benchmark code, GPU run, or new quality/latency claim was introduced. Further row-normalization and output tests require Q/V-rich captures.
- 2026-09-20: added the modern Full/V1 suite. LongBench v2 uses the pinned official data/prompt/scorer and a 24-row domain-round-robin native-32K subset without truncation. BFCL V4 uses the pinned 200-row multi-turn long-context source, official executable backends/state checker, and a Qwen2.5 adapter with trajectory-aware timing. Raw artifacts remain scorer-independent. Local compilation, shell syntax, exact prompt-format parity, and checker self-test pass.
- 2026-09-20: the remote source-preparation step verified both pinned SHA256 values, then input preparation stopped before tokenization because `huggingface-hub==1.32.0` violated the installed Transformers requirement `<1.0`. Added an exact Hub 0.36.0 pin and an official-index install command after the configured Aliyun mirror returned no matching release; no dataset or GPU work needs repeating.
- 2026-09-20: after installing the pinned Hub client, remote preparation completed: 24 fixed LongBench v2 native-32K inputs and 20 fixed BFCL V4 long-context cases were saved. The 806,505-token warning came from a discarded LongBench candidate during tokenization; no over-limit prompt was selected or sent to the model. Full/V1 GPU pilots remain pending.
- 2026-09-20: LongBench completed with status `complete` and all expected report artifacts. BFCL then stopped during episode 32/40 after saving 31 complete episodes. Saved metadata recovers `KeyError: 'arguments'`: a model emitted valid JSON with a tool name but no argument field, and the adapter incorrectly reinserted it as a structured call on the next official turn. The fix preserves it as raw invalid output (so the checker can penalize it), executes no tool, and resumes only the missing suffix. The original failed metadata, 31 records, 834 timing rows, and exact three-row partial episode were inspected remotely before mutation.
- 2026-09-20: the first resume invocation passed the real 31-record prefix validation and archived its inputs, then was stopped before generation while Hugging Face Hub waited on an unnecessary online model-config HEAD request. Resume now forces Hub offline mode because the exact model/tokenizer cache already produced both pilots; the offline invocation loaded all four shards and crossed the formerly crashing episode.
- 2026-09-20: both modern pilots and their offline scorer audits are complete. LongBench is 7/24 Full versus 9/24 V1 with 1.20× paired prefill speedup; paired outcomes show only a net two-question difference. BFCL is 2/20 for both methods with no success flips, a 1.01× same-prompt speedup, and longer/slower V1 trajectories. The BFCL result is non-discriminating because the Full baseline is at 10%, not evidence of quality parity. Complete artifacts were downloaded and hash-verified; see `MODERN_BENCHMARK_AUDIT.md`.
- 2026-09-20: user set the final evidence target to at least 100 fixed cases per benchmark with paired Full/V1 outputs. The complete eligible LongBench v2 panel is frozen at 116 rows (8K–32K native prompts); the final Agent panel uses 100 BFCL V4 `multi_turn_base` rows so the primary comparison is not dominated by the 2/20 Full floor observed in the long-context pilot. The prior 20-row long-context result remains a separate stress test.
- 2026-09-20: final input preparation passed remotely. BFCL base selected 100/200 unique rows across all 20 involved-class signatures; category metadata, source hashes, wheel-member verification, and the official state checker all pass. LongBench reuses the previously frozen 116-row tensor with SHA256 `fb01a2dcf3934f1c2b05576fb8985e51bc4a636497ed148f9719b2284a54306b`.
