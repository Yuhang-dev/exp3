# Modern Full-vs-V1 pilot audit

Date: 2026-09-20  
Model: `Qwen/Qwen2.5-7B-Instruct`  
Sparse configuration: FlashPrefill V1, `alpha=0.08`, block size 128

## Artifact integrity

- Both runs end with `metadata.status=complete`.
- LongBench saved 48 raw generations and 144 synchronized timing rows
  (`24 samples × 2 methods × 3 repeats`). Its run-time `inputs.pt` SHA256 is
  identical to the prepared input file:
  `8b9e73ba00da5c47588be21dbbcf6c09eb8f7b9d72b28f5e1367a42b0d5a24cb`.
- BFCL saved 40 unique episode/configuration records, 40 scored records, and
  1,023 timing rows. The latter exactly equals the sum of the three prefill
  repeats stored in every generated step. Each official-results file contains
  20 rows. Its run-time case manifest matches the prepared manifest:
  `83c9fcfa47d14b6e8373a2aed11c5c22aabb6e56ecc63f81c19e7e35e2b1ced7`.
- Offline rescoring changed 0/48 LongBench scores and 0/40 BFCL success
  decisions. Neither rescore loaded the model or regenerated a trajectory.
- The complete archive is
  `results/archives/modern_full_v1_20260920.tar.gz`, SHA256
  `383d5c52db535ae53804f0213a265b9996e570bf1cd9dbd896fca2a59c3dd820`.

## LongBench v2 native-32K pilot

The 24 prompts contain 16,488–31,283 tokens (median 25,038.5) and are selected
by deterministic round-robin across the six domains, subject to native-32K
eligibility. Availability produces domain counts of 5/5/5/5/3/1; this is
domain-stratified coverage, not an exactly balanced sample.

| Configuration | Correct | Score | Median prefill | Paired speedup | Exact pair ratio | Peak |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Full (`dense`) | 7/24 | 29.17 | 3,435.8 ms | 1.00× | 100.0% | 20.03 GiB |
| V1 (`fp_v1__a0p08`) | 9/24 | 37.50 | 2,860.2 ms | 1.20× | 18.4% | 20.03 GiB |

Paired outcomes are 13 both wrong, 5 both correct, 4 V1-only correct, and 2
Full-only correct. The apparent +8.33-point V1 result is therefore a net two
questions in a small pilot. It is not evidence that sparsity improves quality.
This native-32K, greedy subset is also not the official 503-row LongBench v2
overall score.

## BFCL V4 multi-turn long-context pilot

| Configuration | Success | Median steps | Median max prompt | Median summed prefill | Median generation |
| --- | ---: | ---: | ---: | ---: | ---: |
| Full (`dense`) | 2/20 (10.0%) | 7 | 6,786 | 4,776.4 ms | 13,621.0 ms |
| V1 (`fp_v1__a0p08`) | 2/20 (10.0%) | 8 | 6,954 | 6,011.6 ms | 17,966.1 ms |

Both methods succeed on the same two episodes, so there are no success-level
regressions or recoveries. Only 38 of V1's 189 generated steps retain an
identical complete prompt hash in Full; those steps have a median Full/V1
prefill speedup of only 1.01×. End-to-end V1 is slower because its trajectories
contain 189 steps versus Full's 152, and V1 routing has little opportunity to
amortize its overhead at the roughly 7K median maximum prompt length.

This pilot has a severe floor effect:

- each method has three native-32K trajectory overflows and one maximum-step
  termination;
- Full reaches the 512-token per-step generation cap four times and V1 three
  times, whereas the upstream BFCL local handler can request a larger
  context-dependent allowance;
- most remaining failures are backend state or execution-response mismatches;
- this project Qwen2.5 adapter uses official BFCL data, simulators, ground truth,
  and checker, but is not a built-in leaderboard model configuration.

Consequently, equal 10% success does **not** establish V1 quality parity. A
second Agent evaluation should first obtain a non-floor Full baseline, for
example a fixed BFCL V4 multi-turn base subset, before spending GPU time on a
larger long-context sample.

## Interrupted-run accounting

The first BFCL process stopped after 31 complete records because a syntactically
valid model tool-call object omitted `arguments`; the adapter reinserted the
malformed object as structured history and raised `KeyError` on the next turn.
The fix does not invent arguments or execute the malformed call. It preserves
the raw assistant text, records a decode error, and lets the checker penalize
the trajectory.

Before resume, all 31 complete records and the original 834 timing rows were
archived. Exactly three rows belonged to the incomplete episode; they were
saved separately and excluded before that episode restarted. The completed
file contains no duplicated timing rows. The original 31 records retain adapter
version V1 and the nine resumed records carry adapter version V2; prompt
formatting differs only at the malformed-call boundary. One resumed Full step
records a malformed call missing `name` and is scored as an error.
