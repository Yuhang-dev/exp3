#!/usr/bin/env bash
# Task-level test of FlashPrefill V1 + refined-Jensen token rescue (see BLOCK_RESCUE_RESULTS_20260928.md).
# Configs, all on identical inputs: dense, fp_v1:0.08, fp_v1_rescue:0.08, fp_v1_rescue_v2:0.08.
# Usage:
#   bash run_rescue_benchmarks.sh check              # GPU math checks incl. the rescue path (~1 min)
#   bash run_rescue_benchmarks.sh ruler13 [N=20]     # RULER 32K, 13 tasks, N samples per task
#   bash run_rescue_benchmarks.sh longbench          # LongBench v2, the 116 prepared inputs
#   bash run_rescue_benchmarks.sh bfcl               # BFCL v4 multi_turn_base, the 100 prepared cases
set -euo pipefail
cd /root/autodl-tmp/exp3
source ./env.sh

STAGE="${1:-}"
TAG="$(date +%Y%m%d_%H%M%S)"
CANDIDATES=(--candidate dense --candidate fp_v1:0.08 --candidate fp_v1_rescue:0.08 --candidate fp_v1_rescue_v2:0.08)

case "$STAGE" in
  check)
    python -u check_math.py --out "results/check_math_rescue_${TAG}.json"
    ;;
  ruler13)
    SAMPLES="${2:-20}"
    OUT="results/ruler13_rescue_32k_n${SAMPLES}_${TAG}"
    mkdir -p "$OUT"
    export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
    python -u check_math.py --out "$OUT/check_math.json" 2>&1 | tee "$OUT/check_math.log"
    python -u evaluate.py \
      --split ruler \
      --tasks ruler_niah_s_1 ruler_niah_s_2 ruler_niah_s_3 \
              ruler_niah_mk_1 ruler_niah_mk_2 ruler_niah_mk_3 \
              ruler_niah_mv ruler_niah_mq \
              ruler_vt ruler_cwe ruler_fwe ruler_qa_1 ruler_qa_2 \
      --lengths 32768 \
      --ruler-samples "$SAMPLES" \
      "${CANDIDATES[@]}" \
      --max-new-tokens 128 \
      --repeats 1 \
      --profile \
      --out "$OUT" \
      2>&1 | tee "$OUT/run.log"
    python -u rescore.py "$OUT" --tag at-run 2>&1 | tee "$OUT/rescore.log"
    echo "Done: $OUT"
    ;;
  longbench)
    OUT="results/longbench_v2_rescue_${TAG}"
    test -f results/longbench_v2_native32k_inputs116/inputs.pt
    mkdir -p "$OUT"
    HF_HUB_OFFLINE=1 python -u evaluate.py \
      --inputs results/longbench_v2_native32k_inputs116/inputs.pt \
      --split modern \
      --tasks longbench_v2 \
      "${CANDIDATES[@]}" \
      --max-new-tokens 128 \
      --repeats 1 \
      --profile \
      --out "$OUT" \
      2>&1 | tee "$OUT/run.log"
    python -u rescore.py "$OUT" --tag at-run 2>&1 | tee "$OUT/rescore.log"
    echo "Done: $OUT"
    ;;
  bfcl)
    OUT="results/bfcl_v4_rescue_${TAG}"
    test -f results/bfcl_v4_multi_turn_base_inputs100/selected_cases.jsonl
    HF_HUB_OFFLINE=1 python -u bfcl_v4_agent.py \
      --category multi_turn_base \
      --bfcl-root third_party/bfcl_eval_2025_12_17 \
      --bfcl-wheel third_party/downloads/bfcl_eval-2025.12.17-py3-none-any.whl \
      --samples 100 \
      --selected-cases results/bfcl_v4_multi_turn_base_inputs100/selected_cases.jsonl \
      --methods dense fp_v1 fp_v1_rescue fp_v1_rescue_v2 \
      --alpha 0.08 \
      --repeats 1 \
      --max-new-tokens 1024 \
      --out "$OUT" \
      2>&1 | tee "$OUT.log"
    echo "Done: $OUT"
    ;;
  *)
    echo "usage: bash run_rescue_benchmarks.sh check|ruler13|longbench|bfcl [N]" >&2
    exit 2
    ;;
esac
