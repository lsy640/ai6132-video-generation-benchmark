#!/bin/bash
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus=0
#SBATCH --mem=3G
#SBATCH --time=00:20:00
#SBATCH --output=longcat_intake_%j.out
#SBATCH --error=longcat_intake_%j.err
set -euo pipefail
: "${SLURM_JOB_ID:?submit through sbatch}"
: "${AI6132_CPU_PYTHON:?absolute CPU environment Python executable}"
cd "${AI6132_CODE:-$SLURM_SUBMIT_DIR}"
export CUDA_VISIBLE_DEVICES=""
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1
"$AI6132_CPU_PYTHON" 03-generation/longcat_video/import_completed.py "$@"
