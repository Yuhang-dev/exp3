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
  --index-url https://pypi.org/simple \
  --upgrade \
  --requirement "$ROOT/requirements-qwen35.txt"

python - <<'PY'
import importlib.metadata
import huggingface_hub
import openai
import tokenizers
import torch
import transformers
import triton

print("python environment ready")
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("triton", triton.__version__)
print("transformers", transformers.__version__)
print("huggingface_hub", huggingface_hub.__version__)
print("tokenizers", tokenizers.__version__)
print("kernels", importlib.metadata.version("kernels"))
print("openai", openai.__version__)
assert transformers.__version__ == "5.17.0"
PY
