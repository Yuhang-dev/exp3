#!/usr/bin/env bash
# Phase 1 of BLOCK_PROXY_VALIDATION_SPEC.md on the saved RULER and LongBench captures.
# Extra captures (e.g. the 7 remaining LongBench inputs) can be appended:
#   bash run_block_proxy_validation.sh --capture results/longbench_block_proxy_capture_remaining_<tag>
set -eo pipefail
cd /root/autodl-tmp/exp3
source ./env.sh
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

out="results/audits/block_proxy_validation_$(date +%Y%m%d_%H%M%S)"
python -u block_proxy_validation.py \
  --capture results/ruler_block_proxy_capture_32k_20260923_154550 \
  --capture results/longbench_block_proxy_capture_20260923_174053_355052878 \
  "$@" --out "$out"
python -u summarize_block_proxy_validation.py "$out"
echo "Report: ${out}_report.zip"
