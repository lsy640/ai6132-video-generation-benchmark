#!/bin/bash
# Submit from the repository root, or export AI6132_CODE before sbatch.
# These resource defaults must be checked against the submitter's current QoS.
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus=0
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --job-name=ai6132-cpu
#SBATCH --output=ai6132_cpu_%j.out
#SBATCH --error=ai6132_cpu_%j.err

set -euo pipefail
if [[ -z "${SLURM_JOB_ID:-}" || $# -lt 1 ]]; then
  printf 'Submit via sbatch: run_cpu.sh human-xlsx|human-eval|validate-cases|import-ltx|dataset|assets|media|vbench-cache [arguments]\n' >&2
  exit 2
fi
AI6132_TASK="$1"
shift
case "$AI6132_TASK" in
  human-xlsx|human-eval|validate-cases|import-ltx|dataset|media|vbench-cache|assets) ;;
  *) printf 'Unknown CPU task: %s\n' "$AI6132_TASK" >&2; exit 2 ;;
esac
AI6132_ROOT="${AI6132_ROOT:-$HOME/ai6132-c}"
AI6132_CODE="${AI6132_CODE:-${SLURM_SUBMIT_DIR:?}}"
AI6132_CPU_ENV="${AI6132_CPU_ENV:-$AI6132_ROOT/env}"
cd "$AI6132_CODE"
[[ -f PLAN.md && -f 05-human-evaluation/human_eval.py ]] || {
  printf 'AI6132_CODE must point to the benchmark repository root.\n' >&2
  exit 2
}
module load anaconda
set +u
eval "$(conda shell.bash hook)"
conda activate "$AI6132_CPU_ENV"
set -u
export PYTHONUNBUFFERED=1
export PYTHONUTF8=1
export PYTHONDONTWRITEBYTECODE=1
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export PYTHONPATH="$AI6132_CODE${PYTHONPATH:+:$PYTHONPATH}"
python -c 'import sys; assert sys.version_info >= (3, 10), "Python 3.10+ required"'
printf 'Job: %s; node: %s; task: %s; environment: %s\n' \
  "$SLURM_JOB_ID" "$(hostname)" "$AI6132_TASK" "$CONDA_PREFIX"
case "$AI6132_TASK" in
  human-xlsx) python 05-human-evaluation/human_workbooks.py "$@" ;;
  human-eval) python 05-human-evaluation/human_eval.py "$@" ;;
  validate-cases) python 03-generation/ltx_video/run_cases.py "$@" --validate-only ;;
  import-ltx) python 03-generation/ltx_video/import_completed.py "$@" ;;
  dataset) python -m ai6132_benchmark.dataset "$@" ;;
  assets) python -m ai6132_benchmark.assets "$@" ;;
  media) python -m ai6132_benchmark.media "$@" ;;
  vbench-cache)
    [[ " $* " == *" --receipt-cache "* ]] || { printf 'CPU mode only writes a prepared-cache receipt.\n' >&2; exit 2; }
    python -m ai6132_benchmark.vbench "$@"
    ;;
esac
