#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source /root/autodl-tmp/exp1/env.sh
source "$(conda info --base)/etc/profile.d/conda.sh"

TARGET_ENV="$EXP1_STORAGE_ROOT/conda/envs/exp35"
if [[ ! -x "$TARGET_ENV/bin/python" ]]; then
  conda create --yes --prefix "$TARGET_ENV" --clone "$EXP1_ENV_PREFIX"
fi

conda activate "$TARGET_ENV"
python -m pip install \
  --index-url "${QWEN35_PIP_INDEX_URL:-https://mirrors.aliyun.com/pypi/simple/}" \
  --upgrade \
  --requirement "$ROOT/requirements-qwen35.txt"

# Official binary for the experiment's Torch 2.6 / CUDA 12 / Python 3.11.
CONV_WHEEL_URL="$(python - <<'PY'
import torch

abi = str(torch._C._GLIBCXX_USE_CXX11_ABI).upper()
print(
    "https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.5.0.post8/"
    f"causal_conv1d-1.5.0.post8+cu12torch2.6cxx11abi{abi}-cp311-cp311-linux_x86_64.whl"
)
PY
)"
python -m pip install --no-deps "$CONV_WHEEL_URL"

python - <<'PY'
import importlib.metadata
import huggingface_hub
import openai
import tokenizers
import torch
import transformers
import triton
from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
from fla.ops.gated_delta_rule import chunk_gated_delta_rule, fused_recurrent_gated_delta_rule

print("python environment ready")
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("triton", triton.__version__)
print("transformers", transformers.__version__)
print("huggingface_hub", huggingface_hub.__version__)
print("tokenizers", tokenizers.__version__)
print("kernels", importlib.metadata.version("kernels"))
print("fla-core", importlib.metadata.version("fla-core"))
print("causal-conv1d", importlib.metadata.version("causal-conv1d"))
print("openai", openai.__version__)
assert transformers.__version__ == "5.17.0"
PY
