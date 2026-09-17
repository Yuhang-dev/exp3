# 32K calibration quality diagnosis

Date: 2026-09-17  
Source artifact: `calibration_debug.zip`  
SHA256: `25C74F97479F7D162D124D80F8414A5561E267D15BBFD26B622EF34B3A080811`

## Result

The reported scores are reproducible from the saved generations. All 144 stored predictions match an
independent re-score. The 32K quality loss is not caused by report aggregation, duplicate scoring,
`max_new_tokens` truncation, or a general parser failure.

| 32K task | Config | Strict score | Answer-value recall | Failed targets / 12 |
| --- | --- | ---: | ---: | ---: |
| multi-key | dense | 100.00 | 100.00 | 0 |
| multi-key | fp_v1 | 58.33 | 58.33 | 5 |
| multi-key | mean_native | 58.33 | 66.67 | 5 |
| multi-key | mean_balanced | 75.00 | 75.00 | 3 |
| multi-key | cgf_mean | 75.00 | 75.00 | 3 |
| multi-key | dispersion_mean | 75.00 | 75.00 | 3 |
| multi-query | dense | 100.00 | 100.00 | 0 |
| multi-query | fp_v1 | 33.33 | 33.33 | 8 |
| multi-query | mean_native | 41.67 | 41.67 | 7 |
| multi-query | mean_balanced | 91.67 | 91.67 | 1 |
| multi-query | cgf_mean | 58.33 | 58.33 | 5 |
| multi-query | dispersion_mean | 75.00 | 75.00 | 3 |

The apparent HotpotQA improvement from `dispersion_mean` is concentrated in one of the eight
calibration samples. On that sample, dense, V1, `mean_native`, `mean_balanced`, and `cgf_mean` all
score 0, while `dispersion_mean` scores 28.57; the other seven dense/dispersion pairs have identical
scores. This is a useful candidate signal, not evidence of a stable +3.57 task-level gain. It should
be checked on the 16 disjoint HotpotQA holdout samples after the alpha configuration is frozen,
rather than used to tune repeatedly on the same single calibration case.

`mean_native` multi-key gains one target under value-only recall because the correct value appears under
the wrong key. This is a key/value binding error, not a formatting-only success. No relaxed value-occurrence
score recovers the `fp_v1` result.

## What the failures contain

All failed synthetic targets still contain the requested key followed by a generated value, and every
generation ends with EOS after 83–102 tokens. Across all sparse methods, the 43 failed targets divide into:

| Wrong-value morphology | Count |
| --- | ---: |
| Same-namespace hallucinated value | 16 |
| Correct random suffix attached to the wrong target index | 15 |
| Exact duplicate of another requested target's value | 10 |
| Truncated correct value | 2 |

For `fp_v1` alone, its 13 failures are six correct-suffix/wrong-index outputs, two duplicates of another
target, and five same-namespace hallucinations. This is a precise-copy and key/value-binding failure after
sparse prefill, rather than missing output syntax.

## Position pattern

The three target records occur near 8%, 50%, and 86% of the 32K prompt. For `fp_v1`, combining the two
task variants gives:

| Evidence position | Correct targets / 8 |
| ---: | ---: |
| 8% | 5 / 8 |
| 50% | 3 / 8 |
| 86% | 3 / 8 |

The failure is therefore not confined to the earliest context. Middle and late records are both affected,
and multi-query is substantially more fragile than multi-key.

## Interpretation and remaining discriminator

The custom values deliberately share a long namespace prefix, while the task requires exact assignment of
three high-entropy suffixes. Sparse prefill often preserves part of the requested value but corrupts which
key it belongs to. This explains why the custom benchmark can fall much more sharply than FlashPrefill's
aggregate RULER number, but dense scoring 100 shows that the examples remain solvable by the frozen model.

The saved generations establish a real quality difference but cannot by themselves distinguish an inherent
V1 robustness weakness from a local proxy-score/routing implementation divergence. The next required check
is the updated `check_math.py` V1 block-mean, proxy-score, and exact selection comparison on the RTX 4090,
followed by the official RULER multi-key/multi-query parity run.

## Scale of the published V1 evaluation

The pinned FlashPrefill RULER configs contain 13 tasks, cap each task at 100 samples, and evaluate six
lengths from 4K through 128K. Assuming every task file fills the cap, this is 1,300 generations for one
model/method/length and 7,800 for one model/method across all lengths. Table 4 contains three LLMs and six
methods, corresponding to a nominal 140,400 RULER generations.

The paper does not publish the total wall-clock duration, repetition/warmup count, or the saved RULER timing
logs. It reports single-request TTFT on H20 instead. For Qwen2.5-7B at 32K, dense TTFT is 4.735 s and
FlashPrefill TTFT is 3.534 s. Multiplying those figures by 1,300 gives prefill-only sequential lower bounds
of about 1.71 h and 1.28 h, respectively; the real quality evaluation also generates 50–100 tokens and adds
loading, compilation, tokenization, and scoring overhead, so these are not claimed as the authors' actual
run times.
