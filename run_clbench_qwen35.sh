#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PARENT="$(dirname "$ROOT")"
STAGE="${1:-help}"
OUT="${2:-$ROOT/results/clbench_qwen35_full100}"
MODEL="$ROOT/models/Qwen3.5-4B"
DATA="$ROOT/datasets/cl_bench/CL-bench.jsonl"
INPUTS="$ROOT/results/clbench_qwen35_inputs100"
SMOKE_INPUTS="$ROOT/results/clbench_qwen35_smoke_inputs4"
SMOKE_OUT="$ROOT/results/clbench_qwen35_smoke4"
CHECK_OUT="$ROOT/results/check_qwen35_math.json"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

activate_env() {
  source "$ROOT/env_qwen35.sh"
}

run_module() {
  (cd "$PARENT" && python -u -m "$@")
}

require_sources() {
  test -f "$MODEL/config.json" || {
    echo "Missing model under $MODEL; run the pinned download command first." >&2
    exit 1
  }
  test -f "$DATA" || {
    echo "Missing $DATA; run the pinned dataset download command first." >&2
    exit 1
  }
  test -f "$ROOT/third_party/CL-bench/eval.py" || {
    echo "Missing pinned CL-bench evaluator under third_party/CL-bench." >&2
    exit 1
  }
  test "$(git -C "$ROOT/third_party/CL-bench" rev-parse HEAD)" = "16bffd1cfa05927e72ec75c835177d6e23e82172" || {
    echo "third_party/CL-bench is not at the pinned evaluator commit." >&2
    exit 1
  }
  echo "d5fc88d4b2eea75c61dd40862021b6ae2fba26bd21b58e8c5e18377a763943be  $DATA" | sha256sum --check --status
}

metadata_status() {
  python - "$1" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as source:
    print(json.load(source)["status"])
PY
}

prepare_main() {
  activate_env
  require_sources
  if [[ -e "$INPUTS/metadata.json" ]]; then
    echo "Reusing fixed inputs: $INPUTS"
    return
  fi
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 run_module exp3.clbench_qwen35 \
    --prepare-only \
    --model "$MODEL" \
    --data "$DATA" \
    --samples 100 \
    --seed 42 \
    --context-limit 65536 \
    --max-new-tokens 2048 \
    --thinking \
    --out "$INPUTS"
}

run_check() {
  activate_env
  run_module exp3.check_qwen35_math --out "$CHECK_OUT"
}

run_smoke() {
  activate_env
  require_sources
  test -f "$CHECK_OUT" || {
    echo "Run: bash run_clbench_qwen35.sh check" >&2
    exit 1
  }
  if [[ ! -e "$SMOKE_INPUTS/metadata.json" ]]; then
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 run_module exp3.clbench_qwen35 \
      --prepare-only \
      --model "$MODEL" \
      --data "$DATA" \
      --samples 4 \
      --seed 42 \
      --context-limit 32768 \
      --max-new-tokens 256 \
      --thinking \
      --out "$SMOKE_INPUTS"
  fi
  if [[ -e "$SMOKE_OUT/metadata.json" ]]; then
    echo "Refusing to overwrite $SMOKE_OUT" >&2
    exit 1
  fi
  # Keep the Hub online here so the official FLA/causal-conv kernels are cached.
  run_module exp3.clbench_qwen35 \
    --model "$MODEL" \
    --data "$DATA" \
    --inputs "$SMOKE_INPUTS/inputs.pt" \
    --samples 4 \
    --seed 42 \
    --context-limit 32768 \
    --max-new-tokens 256 \
    --repeats 1 \
    --alpha 0.08 \
    --thinking \
    --profile \
    --out "$SMOKE_OUT"
}

run_panel() {
  activate_env
  require_sources
  test -f "$INPUTS/inputs.pt" || {
    echo "Run: bash run_clbench_qwen35.sh prepare" >&2
    exit 1
  }
  test -f "$CHECK_OUT" || {
    echo "Run: bash run_clbench_qwen35.sh check" >&2
    exit 1
  }
  test "$(metadata_status "$SMOKE_OUT/metadata.json")" = "generated" || {
    echo "Run and inspect: bash run_clbench_qwen35.sh smoke" >&2
    exit 1
  }
  if [[ -e "$OUT/metadata.json" ]]; then
    echo "Refusing to overwrite $OUT" >&2
    exit 1
  fi
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 run_module exp3.clbench_qwen35 \
    --model "$MODEL" \
    --data "$DATA" \
    --inputs "$INPUTS/inputs.pt" \
    --samples 100 \
    --seed 42 \
    --context-limit 65536 \
    --max-new-tokens 2048 \
    --repeats 3 \
    --alpha 0.08 \
    --thinking \
    --profile \
    --out "$OUT"
}

resume_panel() {
  activate_env
  require_sources
  test -f "$OUT/generations.jsonl" || {
    echo "No interrupted run found in $OUT" >&2
    exit 1
  }
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 run_module exp3.clbench_qwen35 \
    --resume \
    --model "$MODEL" \
    --data "$DATA" \
    --inputs "$INPUTS/inputs.pt" \
    --samples 100 \
    --seed 42 \
    --context-limit 65536 \
    --max-new-tokens 2048 \
    --repeats 3 \
    --alpha 0.08 \
    --thinking \
    --profile \
    --out "$OUT"
}

judge_panel() {
  activate_env
  require_sources
  test -n "${OPENAI_API_KEY:-}" || {
    echo "Set OPENAI_API_KEY before running the official judge." >&2
    exit 1
  }
  test "$(metadata_status "$OUT/metadata.json")" = "generated" || {
    echo "Generation is not complete in $OUT" >&2
    exit 1
  }
  local grade_dir="$OUT/official_grades"
  mkdir -p "$grade_dir"
  python -u "$ROOT/third_party/CL-bench/eval.py" \
    --input "$OUT/official_inputs/dense.jsonl" \
    --output "$grade_dir/dense.jsonl" \
    --judge-model gpt-5.1 \
    --reasoning-effort low \
    --workers "${CLBENCH_JUDGE_WORKERS:-4}"
  python -u "$ROOT/third_party/CL-bench/eval.py" \
    --input "$OUT/official_inputs/fp_v1__a0p08.jsonl" \
    --output "$grade_dir/fp_v1__a0p08.jsonl" \
    --judge-model gpt-5.1 \
    --reasoning-effort low \
    --workers "${CLBENCH_JUDGE_WORKERS:-4}"
  run_module exp3.clbench_report \
    --run "$OUT" \
    --dense-graded "$grade_dir/dense.jsonl" \
    --v1-graded "$grade_dir/fp_v1__a0p08.jsonl"
}

case "$STAGE" in
  setup)
    bash "$ROOT/setup_qwen35_env.sh"
    ;;
  prepare)
    prepare_main
    ;;
  check)
    run_check
    ;;
  smoke)
    run_smoke
    ;;
  run)
    run_panel
    ;;
  resume)
    resume_panel
    ;;
  judge)
    judge_panel
    ;;
  all)
    prepare_main
    run_check
    run_smoke
    run_panel
    ;;
  *)
    echo "Usage: bash run_clbench_qwen35.sh {setup|prepare|check|smoke|run|resume|judge|all} [run_out]" >&2
    exit 2
    ;;
esac
