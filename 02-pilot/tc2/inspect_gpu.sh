#!/bin/bash
# Submit with sbatch --partition=<assigned_partition> --qos=<assigned_qos> inspect_gpu.sh
#SBATCH --job-name=ai6132-c-inspect
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem=1G
#SBATCH --time=00:03:00
#SBATCH --output=inspect_gpu_%j.out
#SBATCH --error=inspect_gpu_%j.err

set -eu
if [ -z "${SLURM_JOB_ID:-}" ]; then
  printf 'Use sbatch to allocate a compute node before running this script.\n' >&2
  exit 2
fi
printf 'UTC time: %s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
printf 'Job ID: %s\n' "$SLURM_JOB_ID"
printf 'Node: %s\n' "$(hostname)"
printf 'CUDA_VISIBLE_DEVICES: %s\n' "${CUDA_VISIBLE_DEVICES:-unset}"
printf 'SLURM_JOB_GPUS: %s\n' "${SLURM_JOB_GPUS:-unset}"
printf '\nGPU inventory (match indices to the allocation above):\n'
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv
printf '\nDriver and CUDA compatibility:\n'
nvidia-smi
printf '\nEnvironment modules:\n'
module avail 2>&1
