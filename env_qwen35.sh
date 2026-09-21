#!/usr/bin/env bash
source /root/autodl-tmp/exp1/env.sh
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$EXP1_STORAGE_ROOT/conda/envs/exp35"
export CUDA_VISIBLE_DEVICES=0
export TRITON_CACHE_DIR="$EXP1_STORAGE_ROOT/cache/triton-qwen35"
export TORCHINDUCTOR_CACHE_DIR="$EXP1_STORAGE_ROOT/cache/torchinductor-qwen35"

