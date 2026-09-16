#!/usr/bin/env bash
source /root/autodl-tmp/exp1/env.sh
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$EXP1_ENV_PREFIX"
export CUDA_VISIBLE_DEVICES=0
export TRITON_CACHE_DIR="$EXP1_STORAGE_ROOT/cache/triton"
export TORCHINDUCTOR_CACHE_DIR="$EXP1_STORAGE_ROOT/cache/torchinductor"
