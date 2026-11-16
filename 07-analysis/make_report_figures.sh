#!/bin/bash
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=0
#SBATCH --mem=6G
#SBATCH --time=00:30:00
#SBATCH --job-name=ai6132-figures
#SBATCH --output=ai6132_figures_%j.out
#SBATCH --error=ai6132_figures_%j.err
# Rebuild the report figures. Submit from the repository root:
#   sbatch 07-analysis/make_report_figures.sh
# AI6132_PLOT_PYTHON needs matplotlib and Pillow; AI6132_HUNYUAN_DIR holds case_XXXX/native.mp4.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
PYTHON="${AI6132_PLOT_PYTHON:-/apps/anaconda3-2025.06-0/bin/python}"
HUNYUAN="${AI6132_HUNYUAN_DIR:-$HOME/ai6132-b/outputs/formal-20261114-vae64}"
FFMPEG="${AI6132_FFMPEG:-$HOME/ai6132-b/env/bin/ffmpeg}"
"$PYTHON" 07-analysis/make_report_figures.py "$HUNYUAN" "$FFMPEG"
