"""Bind the measured two-wave recipe to new paths without editing old evidence."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

from ai6132_benchmark.dataset import load_canonical
from ai6132_benchmark.io import sha256, write_json

HERE = Path(__file__).resolve().parent


def prepare(cases, runtime, reused_long, output):
    cases, runtime, reused_long, output = map(lambda p: Path(p).resolve(), (cases, runtime, reused_long, output))
    if output.exists():
        raise ValueError('Use a new batch directory; never change an existing frozen batch')
    rows, receipt = load_canonical(cases, decode=True)
    first = next(c for c in rows if c['case_id'] == 'case_0001')
    trial = json.loads((reused_long/'run.json').read_text())
    if first['target_frames'] != '513':
        raise ValueError('This measured two-wave recipe requires case_0001 as its pre-completed long case')
    profile = json.loads((HERE/'frozen/batch.json').read_text())
    expected = dict(profile['profile'], prompt=first['prompt'], seed=int(first['seed']), num_frames=513,
                    image_sha256=receipt['images']['case_0001']['sha256'])
    keys = ('model_id','model_revision','source_revision','resolution','num_inference_steps','guidance_scale',
            'use_distill','native_fps','prompt','seed','num_frames','image_sha256','cpu_offload_block_count',
            'share_identical_rope_frequency_cache','generation_mode')
    if trial.get('status') != 'succeeded' or any(trial.get(k) != expected[k] for k in keys):
        raise ValueError('Reused long case differs from the frozen generation profile')
    if trial['native']['frames'] != 513 or trial['native']['fps'] != 15 or sha256(reused_long/'native.mp4') != trial['native']['sha256']:
        raise ValueError('Reused native is invalid or changed')
    if sha256(reused_long/'evaluation.mp4') != trial['evaluation']['sha256']:
        raise ValueError('Reused processing artifact changed')
    code = output/'code'; code.mkdir(parents=True)
    shutil.copyfile(HERE/'frozen/engine.py', code/'engine.py')
    worker = (HERE/'frozen/worker.py').read_text()
    old = "'image_relative_path': 'input/shared24/' + case['image_path']"
    if worker.count(old) != 1:
        raise ValueError('Unexpected historical worker layout')
    worker = worker.replace(old, "'image_relative_path': str((Path(config['input_root']) / case['image_path']).resolve())")
    (code/'worker.py').write_text(worker, encoding='utf-8', newline='\n')
    wave = (HERE/'frozen/wave.slurm').read_text()
    old = 'export LONGCAT_ROOT="$HOME/ai6132-longcat"'
    if wave.count(old) != 1:
        raise ValueError('Unexpected historical Slurm layout')
    (code/'wave.slurm').write_text(wave.replace(old, 'export LONGCAT_ROOT="${LONGCAT_ROOT:?set runtime}"'), encoding='utf-8', newline='\n')
    for name in ('cases.canonical.csv', 'cases.canonical.receipt.json'):
        source = cases if name.endswith('.csv') else cases.with_suffix('.receipt.json')
        shutil.copyfile(source, output/name)
    profile.update(batch_id=output.name, created_utc=datetime.now(timezone.utc).isoformat(), input_root=str(cases.parent), remote_root=str(runtime),
        canonical_sha256=receipt['canonical_sha256'], receipt_sha256=sha256(output/'cases.canonical.receipt.json'),
        short_order=[c['case_id'] for c in rows if c['target_frames']=='129'],
        long_order=[c['case_id'] for c in rows if c['target_frames']=='513'],
        reuse={'case_0001': str(reused_long)}, code_sha256={p.name:sha256(p) for p in code.iterdir()})
    write_json(output/'config.json', profile)
    (output/'logs').mkdir(); (output/'job_ids').mkdir()
    return profile


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cases', type=Path, required=True); p.add_argument('--runtime', type=Path, required=True)
    p.add_argument('--reuse-case1', type=Path, required=True); p.add_argument('--out', type=Path, required=True)
    a=p.parse_args(); result=prepare(a.cases,a.runtime,a.reuse_case1,a.out)
    print(json.dumps({'batch_id':result['batch_id'],'short':18,'remaining_long':5}))


if __name__=='__main__': main()
