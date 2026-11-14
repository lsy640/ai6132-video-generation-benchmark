#!/usr/bin/env bash
# Run inside the allocated compute job, after activating a fresh Python 3.10
# environment. This installs software but does not download model weights.
set -euo pipefail
if [[ $# != 1 ]]; then
  echo "Usage: bash prepare_environment.sh /absolute/path/to/LTX-Video" >&2
  exit 2
fi
C_LTX_UPSTREAM="$1"
C_LTX_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
C_LTX_COMMIT="50f0aef1297574a4b29acbdb874e3a363a6d5f2e"
python - <<'PY'
import os,sys
if sys.version_info[:2] != (3,10):
    raise SystemExit('Activate a dedicated Python 3.10 environment first')
if sys.prefix == sys.base_prefix and not os.environ.get('CONDA_PREFIX'):
    raise SystemExit('Refusing installation outside a virtualenv/conda environment')
PY
if [[ ! -e "$C_LTX_UPSTREAM" ]]; then
  git clone https://github.com/Lightricks/LTX-Video.git "$C_LTX_UPSTREAM"
  git -C "$C_LTX_UPSTREAM" checkout --detach "$C_LTX_COMMIT"
elif [[ "$(git -C "$C_LTX_UPSTREAM" rev-parse HEAD)" != "$C_LTX_COMMIT" ]]; then
  echo "Existing upstream is not at $C_LTX_COMMIT; use a new directory." >&2
  exit 2
fi
if [[ -n "$(git -C "$C_LTX_UPSTREAM" status --porcelain --untracked-files=no)" ]]; then
  echo "Existing upstream has tracked modifications; use a clean pinned checkout." >&2
  exit 2
fi
python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -c "$C_LTX_SCRIPT_DIR/constraints.txt" -e "${C_LTX_UPSTREAM}[inference]" PyYAML Pillow accelerate
python -m pip check
python - <<'PY'
import torch,transformers,diffusers,av,yaml
from ltx_video.inference import InferenceConfig
print({'torch':torch.__version__, 'torch_cuda':torch.version.cuda,
       'transformers':transformers.__version__, 'diffusers':diffusers.__version__,
       'cuda_available':torch.cuda.is_available()})
if not torch.cuda.is_available():
    raise SystemExit('Environment import passed, but this process has no allocated CUDA GPU')
print({'gpu':torch.cuda.get_device_name(0), 'bf16':torch.cuda.is_bf16_supported()})
if not torch.cuda.is_bf16_supported():
    raise SystemExit('Frozen BF16 pipeline requires a BF16-capable GPU')
PY
echo "Imports passed. Weights have not been downloaded; run download_assets.py explicitly next."
