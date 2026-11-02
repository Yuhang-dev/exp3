#!/usr/bin/env bash
# Discriminating experiments for the sparse copying failure; see POINTER_STUDY.md.
# Usage: bash run_pointer_study.sh {check|prepare|smoke|alpha|length|patch|factors|holdout|summary|all}
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
source ./env.sh
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

ROOT="${POINTER_ROOT:-results/pointer}"
INPUTS="$ROOT/inputs_32k/inputs.pt"
mkdir -p "$ROOT"

check() {
  python -u check_math.py --out "$ROOT/check_math.json" 2>&1 | tee "$ROOT/check_math.log"
}

prepare() {
  python -u evaluate.py --split calibration --tasks synthetic_kv_retrieval --lengths 32768 \
    --synthetic-samples 4 --candidate dense --max-new-tokens 128 --prepare-only --out "$ROOT/inputs_32k"
  python - "$INPUTS" <<'PY'
import json, sys, torch
expected = json.load(open("pointer_calibration_hashes.json"))
actual = {s["sample_id"]: s["input_sha256"] for s in torch.load(sys.argv[1], weights_only=False)}
assert actual == expected, "regenerated prompts differ from the archived 32K sweep"
print("8 prompts match the archived 32K sweep")
PY
  python -u prepare_pointer_factors.py --inputs "$INPUTS" --out "$ROOT/factors_inputs/inputs.pt"
}

smoke() {
  python -u evaluate.py --split quick --tasks synthetic_kv_retrieval --lengths 4096 \
    --synthetic-samples 1 --candidate dense --max-new-tokens 128 --prepare-only --out "$ROOT/smoke_inputs"
  python -u pointer_patch_study.py --inputs "$ROOT/smoke_inputs/inputs.pt" --limit 1 --out "$ROOT/smoke_patch" \
    2>&1 | tee "$ROOT/smoke_patch.log"
  python -u evaluate.py --inputs "$ROOT/smoke_inputs/inputs.pt" --split quick --tasks synthetic_kv_retrieval \
    --candidate dense --candidate fp_row_norm:0.08 --max-new-tokens 128 --repeats 1 --profile --out "$ROOT/smoke_eval" \
    2>&1 | tee "$ROOT/smoke_eval.log"
}

alpha() {
  python -u evaluate.py --inputs "$INPUTS" --split calibration --tasks synthetic_kv_retrieval \
    --config pointer_candidates.json --max-new-tokens 128 --repeats 1 --profile --out "$ROOT/alpha" \
    2>&1 | tee "$ROOT/alpha.log"
}

length() {
  python -u evaluate.py --split calibration --tasks synthetic_kv_retrieval --lengths 8192 16384 24576 32768 \
    --synthetic-samples 4 --candidate dense --candidate fp_v1:0.08 --candidate fp_row_norm:0.08 \
    --candidate mean_balanced:0.08 --max-new-tokens 128 --repeats 1 --out "$ROOT/length" \
    2>&1 | tee "$ROOT/length.log"
}

patch() {
  python -u pointer_patch_study.py --inputs "$INPUTS" --out "$ROOT/patch" 2>&1 | tee "$ROOT/patch.log"
}

factors() {
  python -u evaluate.py --inputs "$ROOT/factors_inputs/inputs.pt" --split calibration \
    --tasks synthetic_kv_retrieval --candidate dense --candidate fp_v1:0.08 --max-new-tokens 128 --repeats 1 \
    --out "$ROOT/factors" 2>&1 | tee "$ROOT/factors.log"
}

holdout() {
  python -u evaluate.py --split holdout --tasks synthetic_kv_retrieval --lengths 32768 --synthetic-samples 16 \
    --candidate dense --candidate fp_v1:0.08 --candidate mean_native:0.08 --candidate fp_row_norm:0.08 \
    --candidate mean_balanced:0.08 --max-new-tokens 128 --repeats 1 --profile --out "$ROOT/holdout" \
    2>&1 | tee "$ROOT/holdout.log"
}

summary() {
  python -u summarize_pointer_study.py "$ROOT"
  tar --exclude='*.pt' -czf "$ROOT.tar.gz" -C "$(dirname "$ROOT")" "$(basename "$ROOT")"
  ls -lh "$ROOT.tar.gz"
}

case "${1:-}" in
  check|prepare|smoke|alpha|length|patch|factors|holdout|summary) "$1" ;;
  all) check; prepare; smoke; alpha; patch; length; factors; holdout; summary ;;
  *) echo "usage: bash run_pointer_study.sh {check|prepare|smoke|alpha|length|patch|factors|holdout|summary|all}" >&2; exit 2 ;;
esac
