#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
source ./env.sh

STAGE="${1:-pilot}"
if [[ "$STAGE" == "rescore" ]]; then
  RUN_DIR="${2:?usage: bash run_ruler.sh rescore RUN_DIR [TAG]}"
  TAG="${3:-current}"
  python -u rescore.py "$RUN_DIR" --tag "$TAG"
  exit 0
fi

case "$STAGE" in
  pilot)
    SAMPLES=20
    DEFAULT_OUT="results/ruler_32k_pilot20"
    ;;
  full)
    SAMPLES=100
    DEFAULT_OUT="results/ruler_32k_full100"
    ;;
  *)
    echo "usage: bash run_ruler.sh {pilot|full} [OUT_DIR]" >&2
    exit 2
    ;;
esac

OUT="${2:-$DEFAULT_OUT}"
if [[ -e "$OUT/metadata.json" || -e "$OUT/predictions.jsonl" ]]; then
  echo "Refusing to overwrite existing run artifacts in $OUT" >&2
  exit 2
fi
mkdir -p "$OUT"

python -u check_math.py --out "$OUT/check_math.json" 2>&1 | tee "$OUT/check_math.log"

python -u evaluate.py \
  --split ruler \
  --tasks ruler_niah_mk_1 ruler_niah_mk_2 ruler_niah_mk_3 ruler_niah_mq \
  --lengths 32768 \
  --ruler-samples "$SAMPLES" \
  --candidate dense \
  --candidate fp_v1:0.08 \
  --max-new-tokens 128 \
  --repeats 1 \
  --out "$OUT" \
  2>&1 | tee "$OUT/run.log"

python -u rescore.py "$OUT" --tag at-run 2>&1 | tee "$OUT/rescore.log"

echo "RULER $STAGE complete: $OUT"
