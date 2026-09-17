# RULER 32K pilot audit

Audit date: 2026-09-17

## Preserved artifact

- Original archive: `C:\Users\Yuhang\Downloads\ruler_32k_pilot20.tar.gz`
- Project-local ignored copy: `results/archives/ruler_32k_pilot20_20260917_sha25F70CD7.tar.gz`
- Size: 8,559,931 bytes
- SHA256: `25F70CD73C1DCE0E69E55E5238746AAB8445A268E61DB1624D3049C193428D24`

The run completed on an RTX 4090 with `Qwen/Qwen2.5-7B-Instruct` at model commit
`a09a35458c702b33eeacc393d103063234e8bc28`. It used the pinned
`aldjalkdf/ruler` revision `2a9d66ecfcdbcaa72d692b6e89d1fb3325e7d634`, the FlashPrefill
evaluation code at `baa612047433a992a00d07dc178205eed065ae14`, no chat template, and
the pinned official substring-recall scorer.

## Artifact integrity

- `metadata.status` is `complete`.
- There are 80 fixed inputs: 20 samples for each of four tasks.
- There are 160 raw generations, 160 predictions, 160 quality rows, and 160 timing rows.
- Every dense/FlashPrefill pair has the same input SHA256.
- The actual hashes of `inputs.jsonl`, `generations.jsonl`, and `predictions.jsonl` match the
  hashes recorded in the immutable rescoring manifest.
- Offline rescoring compared all 160 stored scores and changed 0.
- The expanded GPU math check passed, including V1 block means, proxy scores, threshold/protected
  selection, and future-K/V causality.

## Quality and latency

| Task | Dense | FlashPrefill V1 | Delta | Paired prefill speedup |
| --- | ---: | ---: | ---: | ---: |
| `niah_multikey_1` | 100.00 | 95.00 | -5.00 | 1.24x |
| `niah_multikey_2` | 95.00 | 95.00 | 0.00 | 1.21x |
| `niah_multikey_3` | 90.00 | 90.00 | 0.00 | 1.23x |
| `niah_multiquery` | 100.00 | 98.75 | -1.25 | 1.24x |
| Four-task macro average | **96.25** | **94.69** | **-1.56** | **1.23x median over 80 pairs** |

The 80 paired speedups range from 1.207x to 1.245x. Peak allocated memory is effectively
unchanged at about 19.6--19.9 GiB because this path retains the complete KV cache. Timing has only
one measured repeat per sample, so it is descriptive rather than a production benchmark.

## Paired score changes

Only four of the 80 input pairs receive different scores:

| Sample | Dense | FlashPrefill V1 | Observation |
| --- | ---: | ---: | --- |
| `ruler_niah_mk_1:32768:71` | 100 | 0 | V1 returns the wrong number. |
| `ruler_niah_mk_3:32768:27` | 0 | 100 | V1 recovers a UUID that dense misses. |
| `ruler_niah_mk_3:32768:50` | 100 | 0 | V1 corrupts the final part of the UUID. |
| `ruler_niah_mq:32768:59` | 100 | 75 | V1 misses one of four key/value bindings. |

These are answer-selection or key/value-binding differences, not scorer-only differences. The
outputs usually reach the upstream task-specific 50/100-token cap, but the requested answer is
emitted at the beginning; the failed cases contain a wrong value rather than a correct value cut
off after the generation limit.

## Interpretation

This pilot resolves the main scorer concern for the custom 32K failure. Under the pinned
FlashPrefill/HELMET-style RULER prompt and scorer, V1 loses only 1.56 points on this four-task
subset. That is close to the paper's all-13-task 32K change for the same model (90.14 to 88.25,
-1.89), whereas the project's denser custom KV tasks lost much more. The custom loss therefore
cannot be explained by the scorer implementation alone.

This is not yet a reproduction of the paper's absolute RULER score: it covers only four retrieval
tasks and 20 of 100 rows per task, while the paper score aggregates all 13 tasks at the official
cap. The local runtime also uses Torch 2.6.0, Triton 3.2.0, Transformers 4.51.3, and an RTX 4090,
so its 1.23x full-prefill speedup must not be compared directly with the paper's H20/vLLM or
operator-only numbers.

## Decision

The pilot is internally valid and sufficiently close in quality delta to justify the already
planned 100-sample-per-task expansion of the same four-task parity check. The full run remains
necessary because the net -1.56 result is determined by only a few paired changes.
