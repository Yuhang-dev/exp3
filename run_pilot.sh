#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
source ./env.sh

PHASE="${1:-calibration}"

case "$PHASE" in
  calibration)
    python -u evaluate.py \
      --split calibration \
      --tasks synthetic_kv_retrieval hotpotqa \
      --lengths 16384 32768 \
      --synthetic-samples 4 \
      --hotpot-samples 8 \
      --config initial_candidates.json \
      --max-new-tokens 128 \
      --repeats 3 \
      --profile \
      --out results/calibration
    ;;
  holdout)
    CONFIG="${2:?usage: bash run_pilot.sh holdout path/to/frozen_selection.json [output]}"
    OUTPUT="${3:-results/holdout}"
    python -u evaluate.py \
      --split holdout \
      --tasks synthetic_kv_retrieval hotpotqa \
      --lengths 16384 32768 \
      --synthetic-samples 8 \
      --hotpot-samples 16 \
      --config "$CONFIG" \
      --max-new-tokens 128 \
      --repeats 3 \
      --profile \
      --out "$OUTPUT"
    ;;
  sweep)
    OUTPUT="${2:-results/calibration_sweep_32k}"
    python -u evaluate.py \
      --split calibration \
      --tasks synthetic_kv_retrieval \
      --lengths 32768 \
      --synthetic-samples 4 \
      --config focused_sweep_candidates.json \
      --max-new-tokens 128 \
      --repeats 3 \
      --profile \
      --out "$OUTPUT"
    ;;
  report)
    OUTPUT="${2:?usage: bash run_pilot.sh report results/run_directory}"
    python -u report.py "$OUTPUT"
    ;;
  *)
    echo "unknown phase: $PHASE (expected calibration, holdout, sweep, or report)" >&2
    exit 2
    ;;
esac
