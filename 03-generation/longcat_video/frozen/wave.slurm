#!/bin/bash
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --nodelist=TC2N04
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=28G
#SBATCH --time=06:00:00
#SBATCH --output=logs/wave_%j.out
#SBATCH --error=logs/wave_%j.err
set -euo pipefail
export LONGCAT_ROOT="$HOME/ai6132-longcat"
BATCH="${LONGCAT_BATCH:?missing batch}"
cd "$BATCH"
export PYTHONPATH="$LONGCAT_ROOT/source"
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TMPDIR="$LONGCAT_ROOT/tmp"
module load cuda/12.8.0
hostname
nvidia-smi --query-gpu=name,memory.total --format=csv
"$LONGCAT_ROOT/.venv/bin/python" -u "$BATCH/code/worker.py" --batch "$BATCH" --wave "${LONGCAT_WAVE:?missing wave}"
