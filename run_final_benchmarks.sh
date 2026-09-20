#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
source ./env.sh

STAGE="${1:-all}"
LONGBENCH_INPUTS="results/longbench_v2_native32k_inputs116"
LONGBENCH_OUT="${2:-results/longbench_v2_native32k_full116}"
AGENT_INPUTS="results/bfcl_v4_multi_turn_base_inputs100"
AGENT_OUT="${3:-results/bfcl_v4_multi_turn_base_full100}"

require_sources() {
  test -f datasets/longbench_v2/data.json
  test -f third_party/bfcl_eval_2025_12_17/bfcl_eval/data/BFCL_v4_multi_turn_base.json
  test -f third_party/bfcl_eval_2025_12_17/bfcl_eval/data/possible_answer/BFCL_v4_multi_turn_base.json
}

prepare_inputs() {
  require_sources
  if [[ -e "$LONGBENCH_INPUTS/metadata.json" ]]; then
    echo "Reusing $LONGBENCH_INPUTS"
  else
    HF_HUB_OFFLINE=1 python -u evaluate.py \
      --prepare-only \
      --split modern \
      --tasks longbench_v2 \
      --longbench-v2-file datasets/longbench_v2/data.json \
      --longbench-v2-samples 116 \
      --longbench-v2-min-tokens 8192 \
      --max-new-tokens 128 \
      --out "$LONGBENCH_INPUTS"
  fi
  if [[ -e "$AGENT_INPUTS/metadata.json" ]]; then
    echo "Reusing $AGENT_INPUTS"
  else
    HF_HUB_OFFLINE=1 python -u bfcl_v4_agent.py \
      --prepare-only \
      --category multi_turn_base \
      --bfcl-root third_party/bfcl_eval_2025_12_17 \
      --bfcl-wheel third_party/downloads/bfcl_eval-2025.12.17-py3-none-any.whl \
      --samples 100 \
      --methods dense fp_v1 \
      --out "$AGENT_INPUTS"
  fi
}

run_longbench() {
  require_sources
  test -f "$LONGBENCH_INPUTS/inputs.pt" || {
    echo "Run: bash run_final_benchmarks.sh prepare" >&2
    exit 1
  }
  if [[ -e "$LONGBENCH_OUT/metadata.json" ]]; then
    echo "Refusing to overwrite $LONGBENCH_OUT" >&2
    exit 1
  fi
  HF_HUB_OFFLINE=1 python -u evaluate.py \
    --inputs "$LONGBENCH_INPUTS/inputs.pt" \
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
  test -f "$AGENT_INPUTS/selected_cases.jsonl" || {
    echo "Run: bash run_final_benchmarks.sh prepare" >&2
    exit 1
  }
  HF_HUB_OFFLINE=1 python -u bfcl_v4_agent.py \
    --category multi_turn_base \
    --bfcl-root third_party/bfcl_eval_2025_12_17 \
    --bfcl-wheel third_party/downloads/bfcl_eval-2025.12.17-py3-none-any.whl \
    --samples 100 \
    --selected-cases "$AGENT_INPUTS/selected_cases.jsonl" \
    --methods dense fp_v1 \
    --alpha 0.08 \
    --repeats 3 \
    --max-new-tokens 1024 \
    --out "$AGENT_OUT"
}

resume_agent() {
  require_sources
  test -f "$AGENT_OUT/generations.jsonl" || {
    echo "No interrupted Agent run found in $AGENT_OUT" >&2
    exit 1
  }
  HF_HUB_OFFLINE=1 python -u bfcl_v4_agent.py \
    --resume \
    --category multi_turn_base \
    --bfcl-root third_party/bfcl_eval_2025_12_17 \
    --bfcl-wheel third_party/downloads/bfcl_eval-2025.12.17-py3-none-any.whl \
    --samples 100 \
    --selected-cases "$AGENT_INPUTS/selected_cases.jsonl" \
    --methods dense fp_v1 \
    --alpha 0.08 \
    --repeats 3 \
    --max-new-tokens 1024 \
    --out "$AGENT_OUT"
}

case "$STAGE" in
  prepare)
    prepare_inputs
    ;;
  longbench)
    run_longbench
    ;;
  agent)
    run_agent
    ;;
  agent-resume)
    resume_agent
    ;;
  all)
    prepare_inputs
    run_longbench
    run_agent
    ;;
  *)
    echo "Usage: bash run_final_benchmarks.sh {prepare|longbench|agent|agent-resume|all} [longbench_out] [agent_out]" >&2
    exit 2
    ;;
esac
