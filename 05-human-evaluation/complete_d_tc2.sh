#!/bin/bash
# Submit after querying mytcinfo. This job imports D into already-issued v3 packets.
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus=0
#SBATCH --mem=2G
#SBATCH --time=01:00:00
#SBATCH --nodelist=TC2N08
#SBATCH --job-name=ai6132-human-D
#SBATCH --output=ai6132_human_D_%j.out
#SBATCH --error=ai6132_human_D_%j.err
set -euo pipefail
test -n "${SLURM_JOB_ID:-}"
if [[ $# -ne 3 ]]; then
  printf 'Submit: sbatch 05-human-evaluation/complete_d_tc2.sh PACKET_ROOT D_RUNS_CSV NEW_PATCH_ZIP\n' >&2
  exit 2
fi
umask 077
AI6132_HUMAN_HOME_BYTES=$(du -s -B1 /home/msai/lius0131 | cut -f1)
if (( AI6132_HUMAN_HOME_BYTES + 1000000000 > 100000000000 )); then
  printf 'Home storage budget is insufficient for D videos and a media patch; retain existing materials.\n' >&2
  exit 2
fi
export AI6132_CODE="${AI6132_CODE:-/home/msai/lius0131/ai6132-video-generation-benchmark}"
export AI6132_CPU_ENV="${AI6132_CPU_ENV:-/home/msai/lius0131/XJIANG026/liu/ai6132-evaluation/human-tools-env}"
export AI6132_FFMPEG="${AI6132_FFMPEG:-/home/msai/lius0131/.conda/envs/repostguard/bin/ffmpeg}"
export AI6132_FFPROBE="${AI6132_FFPROBE:-/home/msai/lius0131/.conda/envs/repostguard/bin/ffprobe}"
cd "$AI6132_CODE"
bash 02-pilot/tc2/run_cpu.sh human-xlsx complete-d \
  --packet-root "$1" --runs "$2" --patch-out "$3"
