#!/bin/bash
# Query mytcinfo before submission; override resources only within your current quota.
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus=0
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --job-name=ai6132-b-preflight
#SBATCH --output=ai6132_preflight_%j.out
#SBATCH --error=ai6132_preflight_%j.err
set -euo pipefail
test -n "${SLURM_JOB_ID:-}"
: "${AI6132_CODE:?Set the benchmark repository path}"
: "${AI6132_CPU_ENV:?Set a personal Python 3.10+ environment with Pillow}"
: "${AI6132_PREFLIGHT_REPORT:?Set a new JSON report path outside the repository}"
module load anaconda
set +u
eval "$(conda shell.bash hook)"
conda activate "$AI6132_CPU_ENV"
set -u
cd "$AI6132_CODE"
export PYTHONPATH="$AI6132_CODE${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=""
python 02-pilot/tc2/preflight_hunyuan.py \
  --cases 01-dataset/cases.canonical.csv \
  --runtime-root "${AI6132_B_ROOT:-$HOME/ai6132-b}" \
  --report "$AI6132_PREFLIGHT_REPORT" "$@"
