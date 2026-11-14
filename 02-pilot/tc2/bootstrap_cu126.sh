#!/bin/bash
# Prepare a separate runtime for long Conv3D inputs; keep the original env intact.
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --job-name=ltx-cu126-setup
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=28G
#SBATCH --time=01:00:00
#SBATCH --output=bootstrap_cu126_%j.out
#SBATCH --error=bootstrap_cu126_%j.err

set -euo pipefail
[[ -n "${SLURM_JOB_ID:-}" ]] || { printf 'Submit with sbatch.\n' >&2; exit 2; }
AI6132_ROOT="${AI6132_ROOT:-$HOME/ai6132-c}"
AI6132_LTX_ENV="${AI6132_LTX_ENV:-$AI6132_ROOT/env-cu126}"
[[ ! -e "$AI6132_LTX_ENV" ]] || { printf 'Target environment already exists: %s\n' "$AI6132_LTX_ENV" >&2; exit 2; }
export CONDA_PKGS_DIRS="$AI6132_ROOT/conda-pkgs"
export PIP_CACHE_DIR="$AI6132_ROOT/pip-cache"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export PYTHONUNBUFFERED=1
export TORCH_CUDNN_V8_API_DISABLED=0
# Budget a full clone plus 12 GB for upgraded packages and caches in the 100 GB Home.
AI6132_HOME_LIMIT_BYTES="${AI6132_HOME_LIMIT_BYTES:-100000000000}"
AI6132_HOME_USED_BYTES=$(du -s -B1 "$HOME" | cut -f1)
AI6132_CLONE_BYTES=$(du -s -B1 "$AI6132_ROOT/env" | cut -f1)
if (( AI6132_HOME_USED_BYTES + AI6132_CLONE_BYTES + 12000000000 > AI6132_HOME_LIMIT_BYTES )); then
  printf 'Insufficient Home quota budget for an isolated clone and package caches.\n' >&2
  exit 2
fi
module load miniconda3/25.5.1
set +u
eval "$(conda shell.bash hook)"
set -u
"$AI6132_ROOT/env/bin/python" -c 'import torch; print("Original runtime:", torch.__version__, "cuDNN", torch.backends.cudnn.version())'
conda create --prefix "$AI6132_LTX_ENV" --clone "$AI6132_ROOT/env" --yes --offline
set +u
conda activate "$AI6132_LTX_ENV"
set -u
python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu126
python -m pip check
python - <<'PY'
import torch
from ltx_video.inference import InferenceConfig, infer
print({"torch": torch.__version__, "cuda": torch.version.cuda,
       "cudnn": torch.backends.cudnn.version(), "gpu": torch.cuda.get_device_name(0),
       "bf16": torch.cuda.is_bf16_supported()})
assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
assert torch.backends.cudnn.version() >= 90300
x = torch.ones((256, 256), device="cuda", dtype=torch.bfloat16)
assert (x @ x).sum().item() > 0
torch.cuda.synchronize()
PY
mkdir -p "$AI6132_ROOT/environment"
python -m pip freeze > "$AI6132_ROOT/environment/ltx-cu126-pip-freeze.txt"
conda env export --prefix "$AI6132_LTX_ENV" > "$AI6132_ROOT/environment/ltx-cu126-conda-env.yml"
