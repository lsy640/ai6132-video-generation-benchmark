#!/bin/bash
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --nodelist=TC2N04
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=28G
#SBATCH --time=03:00:00
#SBATCH --output=longcat_case_%j.out
#SBATCH --error=longcat_case_%j.err
set -euo pipefail
: "${SLURM_JOB_ID:?submit through sbatch}"
: "${LONGCAT_ROOT:?set prepared runtime}"
cd "${AI6132_CODE:-$SLURM_SUBMIT_DIR}"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
module load cuda/12.8.0
"$LONGCAT_ROOT/.venv/bin/python" 03-generation/longcat_video/run_case.py --runtime "$LONGCAT_ROOT" "$@"
