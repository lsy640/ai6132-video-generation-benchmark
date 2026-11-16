#!/bin/bash
# Resource defaults: query mytcinfo before every submission.
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus=0
#SBATCH --mem=2G
#SBATCH --time=00:30:00
#SBATCH --job-name=ai6132-wan-cpu
#SBATCH --output=ai6132_wan_cpu_%j.out
#SBATCH --error=ai6132_wan_cpu_%j.err
set -euo pipefail
: "${SLURM_JOB_ID:?Submit through sbatch}"
: "${AI6132_CPU_ENV:?Set a personal Conda environment}"
cd "${AI6132_CODE:-$SLURM_SUBMIT_DIR}"
module load anaconda
set +u
eval "$(conda shell.bash hook)"
conda activate "$AI6132_CPU_ENV"
set -u
export PYTHONPATH="$PWD" PYTHONUTF8=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
AI6132_TASK="${1:?prepare|assets|test}"; shift
case "$AI6132_TASK" in
  prepare) python 03-generation/Wan2.2/benchmark.py prepare "$@" ;;
  assets) python -m ai6132_benchmark.assets "$@" ;;
  test) python -m unittest discover -s 03-generation/Wan2.2/tests -p 'test_benchmark.py' -v ;;
  *) printf 'Unknown task: %s\n' "$AI6132_TASK" >&2; exit 2 ;;
esac
