#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
source ./env.sh

python -u check_math.py --out results/quick/check_math.json
python -u evaluate.py \
  --split quick \
  --tasks synthetic_kv_retrieval \
  --lengths 4096 \
  --synthetic-samples 1 \
  --methods dense mean_native cgf_mean dispersion_mean \
  --alpha 0.08 \
  --max-new-tokens 128 \
  --repeats 3 \
  --profile \
  --out results/quick \
  "$@"
