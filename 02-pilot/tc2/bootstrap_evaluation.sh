#!/bin/bash
# Query mytcinfo before submitting. AI6132_EVAL_ROOT must be authorized storage.
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gpus=0
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --job-name=ai6132-eval-setup
#SBATCH --output=ai6132_eval_setup_%j.out
#SBATCH --error=ai6132_eval_setup_%j.err
set -euo pipefail
test -n "${SLURM_JOB_ID:-}"
: "${AI6132_CODE:?Set the repository path}"
: "${AI6132_EVAL_ROOT:?Set an explicitly authorized storage directory}"
AI6132_B_ROOT="${AI6132_B_ROOT:-$HOME/ai6132-b}"
AI6132_EVAL_ENV="${AI6132_EVAL_ENV:-$AI6132_EVAL_ROOT/env}"
umask 077
mkdir -p "$AI6132_EVAL_ROOT/receipts" "$AI6132_EVAL_ROOT/conda-pkgs"
printf 'Compute node: %s; evaluation storage: %s\n' "$(hostname)" "$AI6132_EVAL_ROOT"
df -h "$AI6132_EVAL_ROOT"
if [[ -e "$AI6132_EVAL_ENV" ]]; then
  [[ "${AI6132_EVAL_RESUME_SETUP:-0}" == 1 && -f "$AI6132_EVAL_ENV/conda-meta/history" ]] || {
    printf 'Use explicit setup resume only for a completed clone from this setup.\n' >&2; exit 2;
  }
fi
export CONDA_PKGS_DIRS="$AI6132_EVAL_ROOT/conda-pkgs:$HOME/.conda/pkgs"
export PIP_CACHE_DIR="$AI6132_EVAL_ROOT/pip-cache"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
module load anaconda
set +u
eval "$(conda shell.bash hook)"
set -u
if [[ ! -f "$AI6132_EVAL_ENV/conda-meta/history" ]]; then
  conda create --prefix "$AI6132_EVAL_ENV" --clone "$AI6132_B_ROOT/env" --yes
fi
AI6132_CONDA_DEPS_READY=0
if [[ "${AI6132_EVAL_USE_PYPI:-0}" != 1 ]]; then
  if conda install --prefix "$AI6132_EVAL_ENV" --override-channels -c conda-forge \
    --freeze-installed --yes 'scipy=1.15.3' 'opencv=4.11.*' 'numpy=1.26.4' easydict ftfy; then
    AI6132_CONDA_DEPS_READY=1
  fi
fi
set +u
conda activate "$AI6132_EVAL_ENV"
set -u
if [[ "$AI6132_CONDA_DEPS_READY" == 0 ]]; then
  printf 'Exact dependency combination unavailable from Conda; using pinned PyPI wheels in the isolated clone.\n'
  python -m pip install --no-cache-dir --no-deps \
    'scipy==1.15.3' 'opencv-python-headless==4.11.0.86' 'easydict==1.13' 'ftfy==6.3.1'
fi
# These exact upstream Python packages are isolated from the generation environment.
python -m pip install --no-cache-dir --no-deps \
  'timm==1.0.12' 'open-clip-torch==2.26.1' 'openai-clip==1.0.1' \
  'decord==0.6.0' 'qwen-vl-utils==0.0.14'
python -m pip install --no-cache-dir --no-deps "$AI6132_B_ROOT/evaluation/dreamsim-source"
if ! python -m pip check > "$AI6132_EVAL_ROOT/receipts/pip-check.txt" 2>&1; then
  # The upstream py3 Decord wheel embeds a cp36 WHEEL tag despite using ctypes.
  # Retain that metadata warning verbatim; every other dependency error fails.
  export AI6132_EVAL_ROOT
  python - <<'PY'
import os
from pathlib import Path
text = (Path(os.environ['AI6132_EVAL_ROOT']) / 'receipts/pip-check.txt').read_text().strip()
if text != 'decord 0.6.0 is not supported on this platform':
    raise RuntimeError(text)
print('Retained Decord upstream wheel-tag warning; validating the actual decoder runtime next.')
PY
fi
python - <<'PY'
import csv, hashlib, json, os, sys
from pathlib import Path
import cv2, decord, dreamsim, easydict, scipy, timm, torch, torchvision
from transformers import Qwen3VLForConditionalGeneration
from qwen_vl_utils import smart_resize
assert torch.__version__ == '2.6.0+cu124', torch.__version__
assert torchvision.__version__ == '0.21.0+cu124', torchvision.__version__
print({'torch': torch.__version__, 'torchvision': torchvision.__version__,
       'cv2': cv2.__version__, 'timm': timm.__version__, 'scipy': scipy.__version__})
if os.environ.get('AI6132_DIAGNOSTIC_DIR'):
    cohort = Path(os.environ['AI6132_DIAGNOSTIC_DIR'])
    with (cohort / 'handoff/anonymous_clips.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    verified = []
    for row in rows:
        if row['status'] != 'success':
            continue
        reader = decord.VideoReader(str(cohort / 'handoff' / row['video_path']), num_threads=1)
        frames = reader.get_batch(list(range(len(reader)))).asnumpy()
        assert frames.shape[1:] == (432, 768, 3), frames.shape
        assert reader.get_avg_fps() == 12
        verified.append({'clip_id': row['clip_id'], 'frames': len(reader)})
    receipt = {'python': sys.version, 'decoder_version': decord.__version__, 'full_decode': verified}
    target = Path(os.environ['AI6132_EVAL_ROOT']) / 'receipts/decord-runtime.json'
    target.write_text(json.dumps(receipt, indent=2) + '\n')
    print(f'Decord independently decoded {len(verified)} complete scoring videos.')
PY
python -m pip freeze > "$AI6132_EVAL_ROOT/receipts/pip-freeze.txt"
conda env export --prefix "$AI6132_EVAL_ENV" > "$AI6132_EVAL_ROOT/receipts/conda-environment.yml"
sha256sum "$AI6132_EVAL_ROOT/receipts/pip-freeze.txt" "$AI6132_EVAL_ROOT/receipts/conda-environment.yml" > "$AI6132_EVAL_ROOT/receipts/environment.sha256"
printf 'Evaluation environment ready.\n'
