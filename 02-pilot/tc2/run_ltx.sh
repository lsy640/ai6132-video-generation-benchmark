#!/bin/bash
# Args: smoke|pilot|formal /absolute/cases.csv /absolute/output [runner options]
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --job-name=ai6132-c-ltx
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=30G
#SBATCH --time=02:00:00
#SBATCH --output=run_ltx_%j.out
#SBATCH --error=run_ltx_%j.err

set -euo pipefail
if [ -z "${SLURM_JOB_ID:-}" ] || [ "$#" -lt 3 ]; then
  printf 'Submit via sbatch: run_ltx.sh smoke|pilot|formal /absolute/cases.csv /absolute/output [options]\n' >&2
  exit 2
fi
AI6132_PHASE="$1"
AI6132_CASES="$2"
AI6132_OUTPUT="$3"
shift 3
AI6132_ROOT="${AI6132_ROOT:-$HOME/ai6132-c}"
AI6132_CODE="${AI6132_CODE:-$AI6132_ROOT/code}"
AI6132_LTX_ENV="${AI6132_LTX_ENV:-$AI6132_ROOT/env-cu126}"
AI6132_LTX_ASSETS="${AI6132_LTX_ASSETS:-$AI6132_ROOT/ltx_assets_13b}"
AI6132_NODE="$(hostname -s)"
if ! scontrol show hostnames "${SLURM_JOB_NODELIST:?}" | sed 's/\..*//' | grep -Fxq "$AI6132_NODE"; then
  printf 'Run this script on its allocated compute node.\n' >&2
  exit 2
fi
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export HF_HOME="$AI6132_ROOT/hf-cache"
export PYTHONUNBUFFERED=1
export PYTHONUTF8=1
export PYTHONDONTWRITEBYTECODE=1
export TOKENIZERS_PARALLELISM=false
export TORCH_CUDNN_V8_API_DISABLED="${TORCH_CUDNN_V8_API_DISABLED:-0}"
module load miniconda3/25.5.1
set +u
eval "$(conda shell.bash hook)"
conda activate "$AI6132_LTX_ENV"
set -u
python "$AI6132_CODE/03-generation/ltx_video/run_cases.py" \
  --cases "$AI6132_CASES" --phase "$AI6132_PHASE" \
  --upstream "$AI6132_ROOT/LTX-Video" --assets "$AI6132_LTX_ASSETS" \
  --output "$AI6132_OUTPUT" --offload-to-cpu "$@"
