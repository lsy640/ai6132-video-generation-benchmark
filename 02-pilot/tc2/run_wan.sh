#!/bin/bash
# One canonical case per allocation; query mytcinfo before every submission.
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=28G
#SBATCH --time=03:00:00
#SBATCH --job-name=ai6132-wan
#SBATCH --output=ai6132_wan_%j.out
#SBATCH --error=ai6132_wan_%j.err
set -euo pipefail
: "${SLURM_JOB_ID:?Submit through sbatch}"
: "${AI6132_WAN_ENV:?Set a personal, prepared Wan Conda environment}"
cd "${AI6132_CODE:-$SLURM_SUBMIT_DIR}"
module load anaconda
module load cuda/12.8.0
set +u
eval "$(conda shell.bash hook)"
conda activate "$AI6132_WAN_ENV"
set -u
export PYTHONPATH="$PWD" PYTHONUTF8=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
nvidia-smi --query-gpu=name,memory.total --format=csv
python 03-generation/Wan2.2/benchmark.py generate "$@"
