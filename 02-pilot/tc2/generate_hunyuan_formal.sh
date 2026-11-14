#!/bin/bash
# Direct formal generation explicitly requested by the user on 2026-11-14.
# Re-query mytcinfo and verify TC2N08 before submitting; node explicitly requested by the user.
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --nodelist=TC2N08
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=30G
#SBATCH --time=05:50:00
#SBATCH --job-name=ai6132-b-formal
#SBATCH --output=ai6132_b_formal_%j.out
#SBATCH --error=ai6132_b_formal_%j.err
set -euo pipefail
test -n "${SLURM_JOB_ID:-}"
: "${AI6132_CODE:?Set the benchmark repository path}"
AI6132_B_ROOT="${AI6132_B_ROOT:-$HOME/ai6132-b}"
AI6132_FORMAL_OUT="${AI6132_FORMAL_OUT:-$AI6132_B_ROOT/outputs/formal-20261114-vae64}"
export AI6132_GPU_ENV="$AI6132_B_ROOT/env"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
test -f "$AI6132_B_ROOT/assets/assets.json"
if [[ -z "${SLURM_JOB_END_TIME:-}" ]]; then
  # Match this script's 05:50:00 allocation on SLURM versions lacking the end-time variable.
  export SLURM_JOB_END_TIME="$(date -d '+350 minutes' +%s)"
fi
cd "$AI6132_CODE"
sha256sum ai6132_benchmark/*.py > "$AI6132_B_ROOT/environment/formal_sources_${SLURM_JOB_ID}.sha256"
bash 02-pilot/tc2/run_b_gpu.sh hunyuan \
  --cases 01-dataset/cases.canonical.csv --phase formal \
  --upstream "$AI6132_B_ROOT/HunyuanVideo-1.5" \
  --commit 60783e704160023913bee78f0b47036d393d4dfa \
  --assets "$AI6132_B_ROOT/assets" --out "$AI6132_FORMAL_OUT" \
  --skip-pilot-reason 'User explicitly requested: 不用试跑，直接开始正式生成 (2026-11-14)'
sha256sum --check "$AI6132_B_ROOT/environment/formal_sources_${SLURM_JOB_ID}.sha256"
