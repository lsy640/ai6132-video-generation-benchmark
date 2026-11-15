#!/bin/bash
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --nodelist=TC2N04
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=28G
#SBATCH --time=02:00:00
#SBATCH --job-name=longcat_prepare
#SBATCH --output=logs/prepare_%j.out
#SBATCH --error=logs/prepare_%j.err
set -euo pipefail
export LONGCAT_ROOT="$HOME/ai6132-longcat"
COG="$HOME/ai6132-cogvideox"
cd "$LONGCAT_ROOT"
export TMPDIR="$LONGCAT_ROOT/tmp"
export UV_CACHE_DIR="$LONGCAT_ROOT/tmp/uv-cache"
export HF_HOME="$LONGCAT_ROOT/tmp/hf"
export HF_HUB_DISABLE_XET=1
export HF_HUB_DOWNLOAD_TIMEOUT=120
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export UV_LINK_MODE=copy
UV="$COG/bootstrap/bin/uv"
PY="$LONGCAT_ROOT/.venv/bin/python"
mkdir -p "$TMPDIR"
if [ ! -x "$PY" ]; then "$UV" venv --python "$COG/.venv/bin/python" "$LONGCAT_ROOT/.venv"; fi
if [ ! -f ENV_READY ]; then
  "$UV" pip install --python "$PY" --no-cache torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
  "$UV" pip install --python "$PY" --no-cache numpy==1.26.4 transformers==4.41.0 diffusers==0.35.1 loguru==0.7.2 einops==0.8.0 ftfy==6.2.0 psutil==6.0.0 av==12.0.0 imageio==2.37.0 imageio-ffmpeg==0.6.0 accelerate==1.2.1 sentencepiece huggingface-hub==0.36.0 packaging ninja pillow safetensors setuptools==75.8.0 wheel==0.45.1
  "$UV" pip install --python "$PY" --no-deps --no-cache 'https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1%2Bcu12torch2.6cxx11abiFALSE-cp311-cp311-linux_x86_64.whl'
  "$PY" -c 'import torch,flash_attn; assert torch.__version__.startswith("2.6."); assert not torch._C._GLIBCXX_USE_CXX11_ABI; print(torch.__version__,flash_attn.__version__)'
  "$UV" pip freeze --python "$PY" > environment-freeze.txt
  date -Is > ENV_READY
fi
"$PY" -u code/prepare.py
export PYTHONPATH="$LONGCAT_ROOT/source"
"$PY" -c 'from longcat_video.pipeline_longcat_video import LongCatVideoPipeline; print("LONGCAT_IMPORT_OK")'
date -Is > PREPARE_OK
