#!/usr/bin/env bash
# Rebuild the exp3 Qwen2.5 environment on a fresh instance. Versions and the model commit are those
# recorded in the archived 32K sweep metadata. Usage: bash setup_instance.sh {env|model|ruler|longbench|all}
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

STORAGE="${EXP1_STORAGE_ROOT:-/root/autodl-tmp}"
PIP_INDEX="${PIP_INDEX:-https://mirrors.aliyun.com/pypi/simple}"
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu124}"
MODEL_COMMIT=a09a35458c702b33eeacc393d103063234e8bc28

write_env() {
  mkdir -p "$STORAGE/exp1"
  cat > "$STORAGE/exp1/env.sh" <<EOF
#!/usr/bin/env bash
export EXP1_STORAGE_ROOT=$STORAGE
export CONDA_ENVS_PATH="\$EXP1_STORAGE_ROOT/conda/envs"
export CONDA_PKGS_DIRS="\$EXP1_STORAGE_ROOT/conda/pkgs"
export EXP1_ENV_PREFIX="\$CONDA_ENVS_PATH/exp1"
export HF_HOME="\$EXP1_STORAGE_ROOT/huggingface"
export HF_HUB_CACHE="\$HF_HOME/hub"
export HF_DATASETS_CACHE="\$HF_HOME/datasets"
export PIP_CACHE_DIR="\$EXP1_STORAGE_ROOT/cache/pip"
export XDG_CACHE_HOME="\$EXP1_STORAGE_ROOT/cache"
export TMPDIR="\$EXP1_STORAGE_ROOT/tmp"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
mkdir -p "\$CONDA_ENVS_PATH" "\$CONDA_PKGS_DIRS" "\$HF_HUB_CACHE" "\$HF_DATASETS_CACHE" "\$PIP_CACHE_DIR" "\$TMPDIR"
EOF
}

env_stage() {
  write_env
  source "$STORAGE/exp1/env.sh"
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda create -y --prefix "$EXP1_ENV_PREFIX" python=3.11.* pip
  conda activate "$EXP1_ENV_PREFIX"
  python -m pip install torch==2.6.0 --index-url "$TORCH_INDEX"
  python -m pip install --index-url "$PIP_INDEX" \
    transformers==4.51.3 accelerate==1.6.0 huggingface-hub==0.30.2 numpy==2.2.5 pandas==2.2.3 \
    matplotlib==3.10.1 safetensors==0.5.3
  python -c "import torch, triton, transformers; print(torch.__version__, triton.__version__, transformers.__version__, torch.cuda.get_device_name(0))"
}

model_stage() {
  source ./env.sh
  export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
  unset HF_HUB_OFFLINE
  huggingface-cli download Qwen/Qwen2.5-7B-Instruct
  test -d "$HF_HUB_CACHE/models--Qwen--Qwen2.5-7B-Instruct/snapshots/$MODEL_COMMIT"
  echo "Model snapshot $MODEL_COMMIT ready"
}

ruler_stage() {
  source ./env.sh
  export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
  python - <<'PY'
from huggingface_hub import snapshot_download
from data import RULER_REPO, RULER_REVISION
print(snapshot_download(RULER_REPO, repo_type="dataset", revision=RULER_REVISION))
PY
}

case "${1:-}" in
  env) env_stage ;;
  model) model_stage ;;
  ruler) ruler_stage ;;
  longbench) bash prepare_modern_benchmarks.sh ;;
  all) env_stage; model_stage ;;
  *) echo "usage: bash setup_instance.sh {env|model|ruler|longbench|all}" >&2; exit 2 ;;
esac
