#!/bin/bash
# Query current QoS and measure Home usage before each new download plan.
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gpus=0
#SBATCH --mem=4G
#SBATCH --time=03:00:00
#SBATCH --job-name=ai6132-b-assets
#SBATCH --output=ai6132_b_assets_%j.out
#SBATCH --error=ai6132_b_assets_%j.err
set -euo pipefail
test -n "${SLURM_JOB_ID:-}"
: "${AI6132_CODE:?Set the benchmark repository path}"
: "${AI6132_HOME_BYTES_BEFORE:?Measure Home usage before creating the download plan}"
: "${AI6132_HF_TOKEN_FILE:?Set only the private credential file path}"
: "${AI6132_AUTH_PREFLIGHT:?Set the successful authenticated access report}"
AI6132_B_ROOT="${AI6132_B_ROOT:-$HOME/ai6132-b}"
module load anaconda
set +u
eval "$(conda shell.bash hook)"
conda activate "$AI6132_B_ROOT/env"
set -u
cd "$AI6132_CODE"
export PYTHONPATH="$AI6132_CODE${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export HF_HUB_DISABLE_PROGRESS_BARS=1
export HF_HUB_DISABLE_XET=1
export HF_HUB_ETAG_TIMEOUT=60
export HF_HUB_DOWNLOAD_TIMEOUT=120
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE
python -m ai6132_benchmark.download_hunyuan \
  --preflight "$AI6132_AUTH_PREFLIGHT" --root "$AI6132_B_ROOT/assets" \
  --token-file "$AI6132_HF_TOKEN_FILE" --home-bytes-before "$AI6132_HOME_BYTES_BEFORE"
