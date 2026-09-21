#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PARENT="$(dirname "$ROOT")"
STAGE="${1:-run}"
INPUTS="$ROOT/results/clbench_qwen35_inputs100_thinking32k"
OUT="$ROOT/results/clbench_qwen35_full100_thinking32k_sampling"
source "$ROOT/env_qwen35.sh"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
cd "$PARENT"

case "$STAGE" in
  run) RESUME=() ;;
  resume) RESUME=(--resume) ;;
  *) echo "Usage: bash run_clbench_qwen35_long.sh {run|resume}" >&2; exit 2 ;;
esac

test -f "$ROOT/results/check_qwen35_math.json"

if [[ ! -f "$INPUTS/inputs.pt" ]]; then
  python - "$ROOT" "$INPUTS" <<'PY'
import json
import sys
from pathlib import Path
from types import SimpleNamespace
import torch
from exp3.clbench_qwen35 import save_inputs, sha256_file, validate_inputs, write_json

root, target = map(Path, sys.argv[1:])
source = root / "results/clbench_qwen35_inputs100"
original = torch.load(source / "inputs.pt", map_location="cpu", weights_only=False)
validate_inputs(SimpleNamespace(samples=100, max_new_tokens=2048, context_limit=65536, thinking=True), original)
inputs = [{**row, "max_new_tokens": 32768} for row in original]
validate_inputs(SimpleNamespace(samples=100, max_new_tokens=32768, context_limit=98304, thinking=True), inputs)
selection = json.loads((source / "input_manifest.json").read_text())
selection["budget_change"] = {
    "source_inputs": str(source / "inputs.pt"),
    "source_file_sha256": sha256_file(source / "inputs.pt"),
    "original_max_new_tokens": 2048,
    "max_new_tokens": 32768,
    "original_context_limit": 65536,
    "context_limit": 98304,
    "selection_policy": "retain the original 100 tasks and exact prompt tokens; no resampling",
}
target.mkdir()
save_inputs(target, inputs, selection)
write_json(target / "budget_change.json", selection["budget_change"])
print(f"Preserved all 100 task IDs and prompt token hashes in {target}", flush=True)
PY
fi

python -u -m exp3.clbench_qwen35 \
  "${RESUME[@]}" \
  --model "$ROOT/models/Qwen3.5-4B" \
  --data "$ROOT/datasets/cl_bench/CL-bench.jsonl" \
  --inputs "$INPUTS/inputs.pt" \
  --samples 100 --seed 42 \
  --context-limit 98304 --max-new-tokens 32768 \
  --decoding qwen35 \
  --repeats 3 --alpha 0.08 --thinking --profile \
  --out "$OUT"
