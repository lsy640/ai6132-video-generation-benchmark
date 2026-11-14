#!/bin/bash
# This verifies the runtime only; it does not load weights or generate a video.
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=4G
#SBATCH --time=00:05:00
#SBATCH --job-name=ai6132-b-gpu-check
#SBATCH --output=ai6132_b_gpu_check_%j.out
#SBATCH --error=ai6132_b_gpu_check_%j.err
set -euo pipefail
test -n "${SLURM_JOB_ID:-}"
export AI6132_B_ROOT="${AI6132_B_ROOT:-$HOME/ai6132-b}"
test -f "$AI6132_B_ROOT/environment/READY.json"
module load anaconda
set +u
eval "$(conda shell.bash hook)"
conda activate "$AI6132_B_ROOT/env"
set -u
export PYTHONPATH="$AI6132_B_ROOT/HunyuanVideo-1.5${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1
nvidia-smi
python - <<'PY'
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
import torch
from hyvideo.commons.parallel_states import initialize_parallel_state
from hyvideo.commons.infer_state import initialize_infer_state
from hyvideo.pipelines.hunyuan_video_pipeline import HunyuanVideo_1_5_Pipeline
assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
torch.cuda.set_device(0)
environment = Path(os.environ['AI6132_B_ROOT']) / 'environment'
store = environment / f"nccl-check-{os.environ['SLURM_JOB_ID']}"
try:
    torch.distributed.init_process_group('nccl', rank=0, world_size=1,
                                        init_method=store.resolve().as_uri())
    initialize_parallel_state(sp=1)
    initialize_infer_state(SimpleNamespace(sage_blocks_range='0-53', no_cache_block_id='0',
        use_sageattn=False, enable_torch_compile=False, enable_cache=False, cache_type='deepcache',
        cache_start_step=0, cache_end_step=8, total_steps=8, cache_step_interval=1,
        use_fp8_gemm=False, quant_type='fp8-per-token-sgl', include_patterns='double_blocks'))
    value = torch.ones((16, 16), device='cuda', dtype=torch.bfloat16)
    result = value @ value
    torch.distributed.all_reduce(result)
    assert torch.isfinite(result).all().item() and result[0, 0].item() == 16
    torch.cuda.synchronize()
    gpu = torch.cuda.get_device_properties(0)
    report = {'slurm_job_id': os.environ['SLURM_JOB_ID'], 'torch': torch.__version__,
              'cuda_build': torch.version.cuda, 'gpu_name': gpu.name,
              'gpu_total_bytes': gpu.total_memory, 'bf16': 'passed', 'single_rank_nccl': 'passed',
              'official_initialization': 'passed', 'model_inference_verified': False,
              'environment_receipt_sha256': hashlib.sha256((environment / 'READY.json').read_bytes()).hexdigest()}
    destination = environment / f"gpu_check_{os.environ['SLURM_JOB_ID']}.json"
    destination.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
finally:
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
PY
