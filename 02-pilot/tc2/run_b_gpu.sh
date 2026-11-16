#!/bin/bash
# Resource defaults: re-query the submitting user's current QoS before sbatch.
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=28G
#SBATCH --time=02:00:00
#SBATCH --job-name=ai6132-b-eval
#SBATCH --output=ai6132_b_%j.out
#SBATCH --error=ai6132_b_%j.err
set -euo pipefail
if [[ -z "${SLURM_JOB_ID:-}" || $# -lt 1 ]]; then
  printf 'Submit via sbatch: run_b_gpu.sh hunyuan|qwen|vbench [arguments]\n' >&2
  exit 2
fi
AI6132_TASK="$1"
shift
case "$AI6132_TASK" in hunyuan|qwen|vbench) ;; *) exit 2 ;; esac
AI6132_CODE="${AI6132_CODE:-${SLURM_SUBMIT_DIR:?}}"
: "${AI6132_GPU_ENV:?Set an explicit personal environment for this evaluation stage}"
cd "$AI6132_CODE"
module load anaconda
set +u
eval "$(conda shell.bash hook)"
conda activate "$AI6132_GPU_ENV"
set -u
export PYTHONPATH="$AI6132_CODE${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export PYTHONUTF8=1
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
nvidia-smi
# qwen runs the reported protocol: qwen_v3 with the fast frame cache from qwen_fast.
[[ "$AI6132_TASK" == qwen ]] && AI6132_TASK=qwen_fast
python -m "ai6132_benchmark.$AI6132_TASK" "$@"
