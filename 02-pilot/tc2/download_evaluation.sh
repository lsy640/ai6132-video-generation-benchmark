#!/bin/bash
# Query current QoS and ensure AI6132_EVAL_ROOT is explicitly authorized storage.
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gpus=0
#SBATCH --mem=4G
#SBATCH --time=01:00:00
#SBATCH --job-name=ai6132-eval-assets
#SBATCH --output=ai6132_eval_assets_%j.out
#SBATCH --error=ai6132_eval_assets_%j.err
set -euo pipefail
test -n "${SLURM_JOB_ID:-}"
: "${AI6132_CODE:?Set the repository path}"
: "${AI6132_EVAL_ROOT:?Set the authorized evaluation storage}"
AI6132_B_ROOT="${AI6132_B_ROOT:-$HOME/ai6132-b}"
umask 077
module load anaconda
set +u
eval "$(conda shell.bash hook)"
conda activate "$AI6132_B_ROOT/env"
set -u
cd "$AI6132_CODE"
export PYTHONPATH="$AI6132_CODE${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"
python -m ai6132_benchmark.evaluation_assets --root "$AI6132_EVAL_ROOT" \
  --dino-source "$AI6132_B_ROOT/evaluation/cache/dino_model/facebookresearch_dino_main" "$@"
