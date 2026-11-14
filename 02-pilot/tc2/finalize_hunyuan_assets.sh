#!/bin/bash
# Run after the slow transport job has stopped and an identical copy is verified.
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gpus=0
#SBATCH --mem=2G
#SBATCH --time=00:45:00
#SBATCH --job-name=ai6132-b-assets-check
#SBATCH --output=ai6132_b_assets_check_%j.out
#SBATCH --error=ai6132_b_assets_check_%j.err
set -euo pipefail
test -n "${SLURM_JOB_ID:-}"
: "${AI6132_CODE:?Set the benchmark repository path}"
: "${AI6132_HF_TOKEN_FILE:?Set only the private credential file path}"
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
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE
python -m ai6132_benchmark.finalize_hunyuan_assets \
  --root "$AI6132_B_ROOT/assets" --plan "$AI6132_B_ROOT/download-plan.json" \
  --transport-receipt "$AI6132_B_ROOT/glyph-transport-receipt.json" \
  --glyph-git "$AI6132_B_ROOT/Glyph-SDXL-v2.git" --token-file "$AI6132_HF_TOKEN_FILE"
