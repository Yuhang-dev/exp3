#!/usr/bin/env bash
# Complete the seven inputs omitted by the 59b175d command's missing --all-samples.
set -eo pipefail
cd /root/autodl-tmp/exp3
source ./env.sh
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

subset_dir="results/longbench_block_proxy_subset_20260923_174053_355052878"
run_tag=$(date +%Y%m%d_%H%M%S_%N)
capture_dir="results/longbench_block_proxy_capture_remaining_${run_tag}"
study_dir="results/longbench_block_proxy_study_remaining_${run_tag}"

python -u v1_block_diagnostics.py \
  --inputs "$subset_dir/inputs.pt" --out "$capture_dir" \
  --sample-index 1 --sample-index 2 --sample-index 3 --sample-index 4 \
  --sample-index 5 --sample-index 6 --sample-index 7 \
  --layers 0 14 27 --capture-only \
  --save-q-layers 0 14 27 \
  --save-pre-rope-q-layers 0 14 27 \
  --save-layer-input-layers 0 14 27 \
  --sample-query-rows-per-tile 16 \
  --save-v-layers 0 14 27

python -u block_proxy_study.py \
  --capture "$capture_dir" --out "$study_dir" \
  --heads all --detail-heads 0 7 14 21 \
  --q-block-sizes 64 128 256 --k-block-sizes 32 64 128 256 \
  --budget-tokens 2048 --rows-per-tile 16
python plot_block_proxy_study.py "$study_dir"
echo "Report: $study_dir/block_proxy_report.zip"
