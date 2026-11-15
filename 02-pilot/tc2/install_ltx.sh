#!/bin/bash
#SBATCH --job-name=ai6132-c-install
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=28G
#SBATCH --time=02:00:00
#SBATCH --output=install_ltx_%j.out
#SBATCH --error=install_ltx_%j.err

set -euo pipefail
if [ -z "${SLURM_JOB_ID:-}" ]; then
  printf 'Use sbatch to run installation on a compute node.\n' >&2
  exit 2
fi
AI6132_ROOT="${AI6132_ROOT:-$HOME/ai6132-c}"
AI6132_CODE="${AI6132_CODE:-$AI6132_ROOT/code}"
AI6132_LTX_ASSETS="${AI6132_LTX_ASSETS:-$AI6132_ROOT/ltx_assets_13b}"
export CONDA_PKGS_DIRS="$AI6132_ROOT/conda-pkgs"
export PIP_CACHE_DIR="$AI6132_ROOT/pip-cache"
export HF_HOME="$AI6132_ROOT/hf-cache"
export PYTHONUNBUFFERED=1
module load miniconda3/25.5.1
eval "$(conda shell.bash hook)"
conda activate "$AI6132_ROOT/env"
bash "$AI6132_CODE/03-generation/ltx_video/prepare_environment.sh" "$AI6132_ROOT/LTX-Video"
python "$AI6132_CODE/03-generation/ltx_video/download_assets.py" --assets "$AI6132_LTX_ASSETS"
mkdir -p "$AI6132_ROOT/environment"
python -m pip freeze > "$AI6132_ROOT/environment/ltx-pip-freeze.txt"
conda env export > "$AI6132_ROOT/environment/ltx-conda-env.yml"
printf 'LTX environment and pinned assets are ready.\n'
