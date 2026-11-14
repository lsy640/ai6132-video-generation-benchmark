#!/bin/bash
# Environment maintenance and import checks run only on a compute node.
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gpus=0
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --job-name=ai6132-b-env
#SBATCH --output=ai6132_b_env_%j.out
#SBATCH --error=ai6132_b_env_%j.err
set -euo pipefail
test -n "${SLURM_JOB_ID:-}"
: "${AI6132_CODE:?Set the benchmark repository path}"
AI6132_B_ROOT="${AI6132_B_ROOT:-$HOME/ai6132-b}"
AI6132_UPSTREAM="$AI6132_B_ROOT/HunyuanVideo-1.5"
AI6132_UPSTREAM_COMMIT=60783e704160023913bee78f0b47036d393d4dfa
test "$(git -C "$AI6132_UPSTREAM" rev-parse HEAD)" = "$AI6132_UPSTREAM_COMMIT"
test -z "$(git -C "$AI6132_UPSTREAM" status --porcelain --untracked-files=no)"
mkdir -p "$AI6132_B_ROOT/environment"
if [[ -e "$AI6132_B_ROOT/environment/READY.json" ]]; then
  printf 'Environment already frozen; refusing to reinstall.\n' >&2
  exit 2
fi
module load anaconda
set +u
eval "$(conda shell.bash hook)"
set -u
export CONDA_PKGS_DIRS="$HOME/.conda/pkgs"
export PIP_NO_CACHE_DIR=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
if [[ ! -x "$AI6132_B_ROOT/env/bin/python" ]]; then
  conda create --yes --prefix "$AI6132_B_ROOT/env" --override-channels \
    --strict-channel-priority -c conda-forge python=3.11 pip ffmpeg pkg-config
fi
set +u
conda activate "$AI6132_B_ROOT/env"
set -u
cd "$AI6132_CODE"
python -m pip install --index-url https://download.pytorch.org/whl/cu124 \
  torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0
python -m pip install -r "$AI6132_UPSTREAM/requirements.txt" \
  -c 03-generation/hunyuan_video/constraints.txt
python -m pip check
export PYTHONPATH="$AI6132_UPSTREAM:$AI6132_CODE${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
python - <<'PY'
import importlib.metadata
import json
import torch
from hyvideo.pipelines.hunyuan_video_pipeline import HunyuanVideo_1_5_Pipeline
from hyvideo.commons.parallel_states import initialize_parallel_state
from hyvideo.commons.infer_state import initialize_infer_state
assert torch.__version__.startswith('2.6.0')
print(json.dumps({'torch': torch.__version__, 'cuda_build': torch.version.cuda,
                  'upstream_imports': 'passed', 'gpu_check': 'not requested in CPU install job'}))
PY
python -m pip freeze > "$AI6132_B_ROOT/environment/pip-freeze.txt"
conda env export > "$AI6132_B_ROOT/environment/conda-environment.yml"
export AI6132_B_ROOT
python - <<'PY'
import hashlib
import json
import os
from pathlib import Path
root = Path(os.environ['AI6132_B_ROOT'])
files = {name: hashlib.sha256((root / 'environment' / name).read_bytes()).hexdigest()
         for name in ('pip-freeze.txt', 'conda-environment.yml')}
receipt = {'upstream_commit': '60783e704160023913bee78f0b47036d393d4dfa',
           'slurm_job_id': os.environ['SLURM_JOB_ID'], 'files': files,
           'gpu_verified': False, 'model_assets_downloaded': False}
temporary = root / 'environment/READY.json.tmp'
temporary.write_text(json.dumps(receipt, indent=2) + '\n')
temporary.replace(root / 'environment/READY.json')
PY
