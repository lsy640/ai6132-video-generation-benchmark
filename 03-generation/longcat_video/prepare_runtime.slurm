#!/bin/bash
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=28G
#SBATCH --time=02:00:00
#SBATCH --output=longcat_assets_%j.out
#SBATCH --error=longcat_assets_%j.err
set -euo pipefail
: "${SLURM_JOB_ID:?submit through sbatch}"
: "${LONGCAT_ROOT:?set runtime directory}"
: "${LONGCAT_BOOTSTRAP_PYTHON:?set Linux Python 3.11 executable}"
LONGCAT_UV="${LONGCAT_UV:-uv}"
cd "${AI6132_CODE:-$SLURM_SUBMIT_DIR}"
CODE="$PWD/03-generation/longcat_video"
mkdir -p "$LONGCAT_ROOT/code" "$LONGCAT_ROOT/tmp"
export TMPDIR="$LONGCAT_ROOT/tmp" HF_HUB_DISABLE_XET=1 HF_HUB_DOWNLOAD_TIMEOUT=120
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1
"$LONGCAT_BOOTSTRAP_PYTHON" -c 'import sys; assert sys.version_info[:2] == (3,11)'
PY="$LONGCAT_ROOT/.venv/bin/python"
if [ ! -x "$PY" ]; then "$LONGCAT_UV" venv --python "$LONGCAT_BOOTSTRAP_PYTHON" "$LONGCAT_ROOT/.venv"; fi
"$LONGCAT_UV" pip install --python "$PY" --no-cache torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
"$LONGCAT_UV" pip install --python "$PY" --no-cache --extra-index-url https://download.pytorch.org/whl/cu124 --index-strategy unsafe-best-match -r "$CODE/frozen/environment-freeze.txt"
cp "$CODE/frozen/asset-config.json" "$LONGCAT_ROOT/code/config.json"
"$PY" -c 'import torch, flash_attn; assert not torch._C._GLIBCXX_USE_CXX11_ABI'
"$LONGCAT_UV" pip freeze --python "$PY" > "$LONGCAT_ROOT/environment-freeze.txt"
date -Is > "$LONGCAT_ROOT/ENV_READY"
"$PY" -u "$CODE/frozen/prepare_assets.py"
# The CUDA pipeline import is checked by the GPU case job, not this CPU allocation.
