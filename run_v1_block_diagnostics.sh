#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
source ./env.sh

if [[ $# -lt 3 ]]; then
  echo "usage: bash run_v1_block_diagnostics.sh INPUTS_PT SAMPLE_ID OUTPUT_DIR [extra args...]" >&2
  exit 2
fi

inputs=$1
sample_id=$2
output_dir=$3
shift 3

python -u v1_block_diagnostics.py \
  --inputs "$inputs" \
  --sample-id "$sample_id" \
  --out "$output_dir" \
  "$@"
