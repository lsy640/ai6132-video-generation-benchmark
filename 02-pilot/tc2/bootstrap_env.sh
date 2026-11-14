#!/bin/bash
# First-stage setup on an allocated GPU compute node; requires assigned partition/QoS.
#SBATCH --job-name=ai6132-c-env
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=28G
#SBATCH --time=02:00:00
#SBATCH --output=bootstrap_env_%j.out
#SBATCH --error=bootstrap_env_%j.err

set -euo pipefail
if [ -z "${SLURM_JOB_ID:-}" ]; then
  printf 'Use sbatch to run setup on a compute node.\n' >&2
  exit 2
fi
AI6132_ROOT="${AI6132_ROOT:-$HOME/ai6132-c}"
mkdir -p "$AI6132_ROOT" "$AI6132_ROOT/environment"
export CONDA_PKGS_DIRS="$AI6132_ROOT/conda-pkgs"
export PIP_CACHE_DIR="$AI6132_ROOT/pip-cache"
module load miniconda3/25.5.1
eval "$(conda shell.bash hook)"
if [ ! -x "$AI6132_ROOT/env/bin/python" ]; then
  conda create --yes --prefix "$AI6132_ROOT/env" --override-channels \
    -c conda-forge python=3.10 pip ffmpeg
fi
conda activate "$AI6132_ROOT/env"
python -m pip install torch==2.5.1 torchvision==0.20.1 \
  --index-url https://download.pytorch.org/whl/cu121
python - <<'PY'
import json, sys, torch
if not torch.cuda.is_available():
    raise RuntimeError('PyTorch cannot access the allocated GPU')
p = torch.cuda.get_device_properties(0)
print(json.dumps({
    'python': sys.version, 'torch': torch.__version__,
    'torch_cuda': torch.version.cuda, 'gpu_name': p.name,
    'gpu_memory_bytes': p.total_memory,
}, indent=2))
x = torch.ones((16, 16), device='cuda', dtype=torch.bfloat16)
assert torch.isfinite(x @ x).all().item()
print('CUDA BF16 check passed')
PY
python -m pip freeze > "$AI6132_ROOT/environment/base-pip-freeze.txt"
conda env export > "$AI6132_ROOT/environment/base-conda-env.yml"
