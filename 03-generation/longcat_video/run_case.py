"""Replay one shared case with the byte-preserved, measured LongCat engine."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from ai6132_benchmark.dataset import load_canonical
from ai6132_benchmark.io import sha256, write_json

HERE = Path(__file__).resolve().parent


def prepare_request(cases, case_id, runtime, output):
    cases, runtime, output = Path(cases).resolve(), Path(runtime).resolve(), Path(output).resolve()
    if output.exists():
        raise ValueError('Use a new case output; prior attempts must be preserved')
    rows, receipt = load_canonical(cases, decode=True)
    matches = [c for c in rows if c['case_id'] == case_id]
    if len(matches) != 1:
        raise ValueError('Case is not in the shared manifest')
    c = matches[0]
    frozen = json.loads((HERE/'frozen/batch.json').read_text())
    profile = frozen['profile']
    if (runtime/'WEIGHTS_READY').read_text().strip() != profile['model_revision']:
        raise ValueError('Runtime weights revision differs from the measured profile')
    engine = HERE/'frozen/engine.py'
    if sha256(engine) != frozen['code_sha256']['engine.py']:
        raise ValueError('Historical generation engine changed')
    frames = int(c['target_frames'])
    # An absolute RHS overrides the historical engine's home-directory prefix.
    request = dict(profile, trial_id='replay/'+case_id, source_case=case_id, case_id=case_id,
                   category=c['category'], prompt=c['prompt'], seed=int(c['seed']),
                   source_frame_num=c['source_frame_num'], num_frames=frames,
                   image_relative_path=str((cases.parent/c['image_path']).resolve()),
                   image_sha256=receipt['images'][case_id]['sha256'],
                   cpu_offload_block_count=frozen['memory']['long_offload_blocks' if frames == 513 else 'short_offload_blocks'])
    output.mkdir(parents=True)
    write_json(output/'case-config.json', request)
    write_json(output/'selection.json', {'canonical_sha256': receipt['canonical_sha256'], 'case_id': case_id,
                                       'engine_sha256': sha256(engine), 'phase': 'replay'})
    return output/'case-config.json'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cases', type=Path, required=True)
    p.add_argument('--case-id', required=True)
    p.add_argument('--runtime', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    cfg = prepare_request(a.cases, a.case_id, a.runtime, a.out)
    env = os.environ.copy()
    env.update(LONGCAT_ROOT=str(a.runtime.resolve()), LONGCAT_CASE_CONFIG=str(cfg),
               LONGCAT_CASE_OUTPUT=str(a.out.resolve()), HF_HUB_OFFLINE='1',
               PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True', TOKENIZERS_PARALLELISM='false',
               PYTHONPATH=str(a.runtime.resolve()/'source')+os.pathsep+env.get('PYTHONPATH', ''))
    raise SystemExit(subprocess.call([sys.executable, '-u', str(HERE/'frozen/engine.py')], env=env))


if __name__ == '__main__':
    main()
