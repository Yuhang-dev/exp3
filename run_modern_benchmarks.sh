#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
source ./env.sh

STAGE="${1:-all}"
LONGBENCH_OUT="${2:-results/longbench_v2_native32k_pilot24}"
AGENT_OUT="${3:-results/bfcl_v4_long_context_pilot20}"

require_sources() {
  test -f datasets/longbench_v2/data.json
  test -f third_party/bfcl_eval_2025_12_17/bfcl_eval/data/BFCL_v4_multi_turn_long_context.json
}

prepare_inputs() {
  require_sources
  if [[ -e results/longbench_v2_native32k_inputs24/metadata.json ]]; then
    echo "Reusing results/longbench_v2_native32k_inputs24"
  else
    python -u evaluate.py \
      --prepare-only \
      --split modern \
      --tasks longbench_v2 \
      --longbench-v2-file datasets/longbench_v2/data.json \
      --longbench-v2-samples 24 \
      --longbench-v2-min-tokens 16384 \
      --max-new-tokens 128 \
      --out results/longbench_v2_native32k_inputs24
  fi
  if [[ -e results/bfcl_v4_long_context_inputs20/metadata.json ]]; then
    echo "Reusing results/bfcl_v4_long_context_inputs20"
  else
    python -u bfcl_v4_agent.py \
      --prepare-only \
      --bfcl-root third_party/bfcl_eval_2025_12_17 \
      --bfcl-wheel third_party/downloads/bfcl_eval-2025.12.17-py3-none-any.whl \
      --samples 20 \
      --methods dense fp_v1 \
      --out results/bfcl_v4_long_context_inputs20
  fi
}

run_longbench() {
  require_sources
  test -f results/longbench_v2_native32k_inputs24/inputs.pt || {
    echo "Run: bash run_modern_benchmarks.sh prepare" >&2
    exit 1
  }
  if [[ -e "$LONGBENCH_OUT/metadata.json" ]]; then
    echo "Refusing to overwrite $LONGBENCH_OUT" >&2
    exit 1
  fi
  python -u evaluate.py \
    --inputs results/longbench_v2_native32k_inputs24/inputs.pt \
    --split modern \
    --tasks longbench_v2 \
    --methods dense fp_v1 \
    --alpha 0.08 \
    --max-new-tokens 128 \
    --repeats 3 \
    --profile \
    --out "$LONGBENCH_OUT"
}

run_agent() {
  require_sources
  test -f results/bfcl_v4_long_context_inputs20/selected_cases.jsonl || {
    echo "Run: bash run_modern_benchmarks.sh prepare" >&2
    exit 1
  }
  python -u bfcl_v4_agent.py \
    --bfcl-root third_party/bfcl_eval_2025_12_17 \
    --bfcl-wheel third_party/downloads/bfcl_eval-2025.12.17-py3-none-any.whl \
    --samples 20 \
    --selected-cases results/bfcl_v4_long_context_inputs20/selected_cases.jsonl \
    --methods dense fp_v1 \
    --alpha 0.08 \
    --repeats 3 \
    --max-new-tokens 512 \
    --out "$AGENT_OUT"
}

case "$STAGE" in
  download)
    bash ./prepare_modern_benchmarks.sh
    ;;
  prepare)
    prepare_inputs
    ;;
  longbench)
    run_longbench
    ;;
  agent)
    run_agent
    ;;
  all)
    run_longbench
    run_agent
    ;;
  *)
    echo "Usage: bash run_modern_benchmarks.sh {download|prepare|longbench|agent|all} [longbench_out] [agent_out]" >&2
    exit 2
    ;;
esac
