#!/bin/bash
# Submit only after the formal generation job finishes successfully.
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gpus=0
#SBATCH --mem=2G
#SBATCH --time=00:30:00
#SBATCH --job-name=ai6132-b-verify
#SBATCH --output=ai6132_b_verify_%j.out
#SBATCH --error=ai6132_b_verify_%j.err
set -euo pipefail
test -n "${SLURM_JOB_ID:-}"
: "${AI6132_CODE:?Set the benchmark repository path}"
AI6132_B_ROOT="${AI6132_B_ROOT:-$HOME/ai6132-b}"
export AI6132_FORMAL_OUT="${AI6132_FORMAL_OUT:-$AI6132_B_ROOT/outputs/formal-20261114-vae64}"
module load anaconda
set +u
eval "$(conda shell.bash hook)"
conda activate "$AI6132_B_ROOT/env"
set -u
cd "$AI6132_CODE"
export PYTHONPATH="$AI6132_CODE${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export AI6132_B_ROOT
python -m pip freeze > "$AI6132_B_ROOT/environment/verification_env_${SLURM_JOB_ID}.txt"
python - <<'PY'
import json
import os
from pathlib import Path
from ai6132_benchmark.io import sha256
root = Path(os.environ['AI6132_B_ROOT']) / 'environment'
receipt = json.loads((root / 'READY.json').read_text())
current = root / f"verification_env_{os.environ['SLURM_JOB_ID']}.txt"
if sha256(current) != receipt['files']['pip-freeze.txt']:
    raise ValueError('Installed environment differs from the frozen complete dependency export')
print('Complete installed environment still matches the frozen receipt')
PY
python -m ai6132_benchmark.verify_hunyuan \
  --cases 01-dataset/cases.canonical.csv --out "$AI6132_FORMAL_OUT" \
  --report "$AI6132_B_ROOT/environment/formal_verification_${SLURM_JOB_ID}.json"
python - <<'PY'
import json
import os
from pathlib import Path
from statistics import median
from ai6132_benchmark.hunyuan import MODEL_ID
from ai6132_benchmark.io import read_csv, sha256, write_json
root = Path(os.environ['AI6132_B_ROOT'])
output = Path(os.environ['AI6132_FORMAL_OUT'])
verification_path = root / 'environment' / f"formal_verification_{os.environ['SLURM_JOB_ID']}.json"
verified = json.loads(verification_path.read_text())
batch = json.loads((output / 'batch.json').read_text())
_, rows, _ = read_csv(output / 'runs.csv')
public = Path('03-generation/hunyuan_video/FORMAL_VALIDATION.json')
if public.exists():
    raise ValueError('Retain the existing public validation summary; do not overwrite it')
summary = {'schema_version': 1, 'model_id': MODEL_ID, 'status': 'formal_generation_and_full_decode_passed',
           'success': verified['success'], 'short': verified['short'], 'long': verified['long'],
           'canonical_sha256': verified['canonical_sha256'],
           'assets_receipt_sha256': sha256(root / 'assets/assets.json'),
           'upstream_commit': batch['protocol']['upstream_commit'],
           'generator_source_sha256': batch['protocol']['source_sha256'],
           'settings': batch['protocol']['settings'], 'pilot_validation': verified['pilot_validation'],
           'verification_job_id': os.environ['SLURM_JOB_ID'],
           'verification_receipt_sha256': sha256(verification_path),
           'environment_receipt_sha256': sha256(root / 'environment/READY.json'),
           'gpu_models': sorted({r['gpu_model'] for r in rows}),
           'native_dimensions': sorted({(int(r['native_width']), int(r['native_height'])) for r in rows}),
           'median_elapsed_seconds': median(float(r['elapsed_seconds']) for r in rows),
           'max_peak_vram_allocated_mib': max(float(r['peak_vram_allocated_mib']) for r in rows),
           'max_peak_vram_reserved_mib': max(float(r['peak_vram_reserved_mib']) for r in rows)}
write_json(public, summary)
print(f'Created aggregate public validation summary: {public}')
PY
