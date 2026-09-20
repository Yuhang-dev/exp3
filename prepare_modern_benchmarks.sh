#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

if [[ -f ./env.sh ]]; then
  source ./env.sh
fi
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

LONGBENCH_DIR="datasets/longbench_v2"
LONGBENCH_FILE="$LONGBENCH_DIR/data.json"
LONGBENCH_REVISION="2b48e494f2c7a2f0af81aae178e05c7e1dde0fe9"
LONGBENCH_SHA256="15d61c22d92c96900b3c4948b6aeea218d3214b676a65df48e7b8555604c7fe2"
DOWNLOAD_DIR="third_party/downloads"
BFCL_WHEEL="$DOWNLOAD_DIR/bfcl_eval-2025.12.17-py3-none-any.whl"
BFCL_ROOT="third_party/bfcl_eval_2025_12_17"
BFCL_SHA256="8555bc9407a56682ceb7d969e87eb724f6b679deb0ef05114d9c6e786406b103"

mkdir -p "$LONGBENCH_DIR" "$DOWNLOAD_DIR" "$BFCL_ROOT"

if [[ ! -f "$LONGBENCH_FILE" ]]; then
  python - <<PY
from huggingface_hub import hf_hub_download

hf_hub_download(
    repo_id="THUDM/LongBench-v2",
    filename="data.json",
    repo_type="dataset",
    revision="$LONGBENCH_REVISION",
    local_dir="$LONGBENCH_DIR",
)
PY
fi

if [[ ! -f "$BFCL_WHEEL" ]]; then
  python -m pip download \
    --no-deps \
    --only-binary=:all: \
    --dest "$DOWNLOAD_DIR" \
    "bfcl-eval==2025.12.17"
fi

longbench_actual="$(sha256sum "$LONGBENCH_FILE" | cut -d ' ' -f 1)"
bfcl_actual="$(sha256sum "$BFCL_WHEEL" | cut -d ' ' -f 1)"
[[ "$longbench_actual" == "$LONGBENCH_SHA256" ]] || {
  echo "LongBench v2 SHA256 mismatch: $longbench_actual" >&2
  exit 1
}
[[ "$bfcl_actual" == "$BFCL_SHA256" ]] || {
  echo "BFCL wheel SHA256 mismatch: $bfcl_actual" >&2
  exit 1
}

if [[ ! -f "$BFCL_ROOT/bfcl_eval/data/BFCL_v4_multi_turn_long_context.json" ]]; then
  python -m zipfile -e "$BFCL_WHEEL" "$BFCL_ROOT"
fi

test -f "$BFCL_ROOT/bfcl_eval/data/possible_answer/BFCL_v4_multi_turn_long_context.json"
test -f "$BFCL_ROOT/bfcl_eval/data/BFCL_v4_multi_turn_base.json"
test -f "$BFCL_ROOT/bfcl_eval/data/possible_answer/BFCL_v4_multi_turn_base.json"
test -f "$BFCL_ROOT/bfcl_eval/eval_checker/multi_turn_eval/multi_turn_checker.py"

echo "Prepared pinned LongBench v2 and BFCL V4 sources."
sha256sum "$LONGBENCH_FILE" "$BFCL_WHEEL"
