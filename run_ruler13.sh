#!/usr/bin/env bash
# Phase 2 G2: RULER 32K, all 13 tasks of the pinned FlashPrefill RULER data.
# Dense vs FlashPrefill V1 (alpha 0.08) vs V1 + V2-style zero-order mean correction
# (mean_native: PyTorch emulation of FlashPrefillv2 75b58f2 use_mean_correction; the sm90 kernel cannot run on RTX 4090).
# Usage: bash run_ruler13.sh [SAMPLES_PER_TASK=20] [OUT_DIR]
set -euo pipefail
cd /root/autodl-tmp/exp3
source ./env.sh
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

SAMPLES="${1:-20}"
OUT="${2:-results/ruler13_32k_n${SAMPLES}_$(date +%Y%m%d_%H%M%S)}"
if [[ -e "$OUT/metadata.json" || -e "$OUT/predictions.jsonl" ]]; then
  echo "Refusing to overwrite existing run artifacts in $OUT" >&2
  exit 2
fi
mkdir -p "$OUT"
python -u check_math.py --out "$OUT/check_math.json" 2>&1 | tee "$OUT/check_math.log"
python -u evaluate.py \
  --split ruler \
  --tasks ruler_niah_s_1 ruler_niah_s_2 ruler_niah_s_3 \
          ruler_niah_mk_1 ruler_niah_mk_2 ruler_niah_mk_3 \
          ruler_niah_mv ruler_niah_mq \
          ruler_vt ruler_cwe ruler_fwe ruler_qa_1 ruler_qa_2 \
  --lengths 32768 \
  --ruler-samples "$SAMPLES" \
  --candidate dense \
  --candidate fp_v1:0.08 \
  --candidate mean_native:0.08 \
  --max-new-tokens 128 \
  --repeats 1 \
  --out "$OUT" \
  2>&1 | tee "$OUT/run.log"
python -u rescore.py "$OUT" --tag at-run 2>&1 | tee "$OUT/rescore.log"
echo "RULER 13-task run complete: $OUT"
