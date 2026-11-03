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
  conda create -y --override-channels -c conda-forge --prefix "$EXP1_ENV_PREFIX" python=3.11.* pip
  conda activate "$EXP1_ENV_PREFIX"
  python -m pip install torch==2.6.0 --index-url "$TORCH_INDEX"
  python -m pip install --index-url "$PIP_INDEX" \
    transformers==4.51.3 accelerate==1.6.0 huggingface-hub==0.30.2 numpy==2.2.5 pandas==2.2.3 \
    matplotlib==3.10.1 safetensors==0.5.3
  python -c "import torch, triton, transformers; print(torch.__version__, triton.__version__, transformers.__version__, torch.cuda.get_device_name(0))"
}

model_stage() {
  # ModelScope tested fastest from the instance; the weight SHA256s equal the LFS oids of MODEL_COMMIT.
  source ./env.sh
  local repo="$HF_HUB_CACHE/models--Qwen--Qwen2.5-7B-Instruct"
  local snapshot="$repo/snapshots/$MODEL_COMMIT"
  mkdir -p "$snapshot" "$repo/refs"
  while read -r sha name; do
    aria2c -x 8 -s 8 -k 64M -c --file-allocation=none --summary-interval=60 --console-log-level=warn \
      -d "$snapshot" -o "$name" "https://www.modelscope.cn/models/Qwen/Qwen2.5-7B-Instruct/resolve/master/$name"
    echo "$sha  $snapshot/$name" | sha256sum -c -
  done <<'EOF'
7463bb0ea78315365e6c6b74de4e73bbcc8359dfb0c5a737584e077d42c0b03c config.json
3a8f9087e486054c8a4a08dae2e5a3ba62e23da212b5b8c08bc42cb983c3459f generation_config.json
599bab54075088774b1733fde865d5bd747cbcc7a547c5bc12610e874e26f5e3 merges.txt
624bf7c47cd12468fdc16e38a47cf4f19e0415b859a223ba3c027eed2f0e1028 model.safetensors.index.json
c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539 tokenizer.json
5b5d4f65d0acd3b2d56a35b56d374a36cbc1c8fa5cf3b3febbbfabf22f359583 tokenizer_config.json
ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910 vocab.json
a1333e6293854747c481288ea83b348226af178dd565c49b6f9495ba1966aba7 model-00001-of-00004.safetensors
f5d25a2772cb825164a2a2c0fb6d51a87e282abf21e4dd75bc5cfb3cd0ea6185 model-00002-of-00004.safetensors
8efdec4c1bc12317ae1a38dc42b595ce777738a64deea3fcb8a0a91381bcdfd5 model-00003-of-00004.safetensors
1a72d403cdf0c1ec3cb7f289f17b394a01e64394c2e9b3c0f94dbce3faf879bd model-00004-of-00004.safetensors
EOF
  printf %s "$MODEL_COMMIT" > "$repo/refs/main"
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
