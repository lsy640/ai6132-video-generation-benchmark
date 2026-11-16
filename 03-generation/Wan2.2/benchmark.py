"""Canonical Wan TI2V batches. All CLI operations on TC2 run through sbatch."""
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time

from ai6132_benchmark.dataset import load_canonical
from ai6132_benchmark.io import read_csv, sha256
from ai6132_benchmark.media import inspect_video

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
MODEL_REPO = 'Wan-AI/Wan2.2-TI2V-5B'
UPSTREAM = '1ea34ff48f87168174e12956e200b1d908b1c5ff'
FIELDS = ['case_id', 'category', 'model_id', 'phase', 'seed', 'source_frame_num',
          'target_frames', 'status', 'weight_version', 'weight_revision', 'config',
          'config_sha256', 'image_sha256', 'native_video', 'video_sha256',
          'native_width', 'native_height', 'native_frames', 'native_fps',
          'native_duration_seconds', 'elapsed_seconds', 'gpu_model',
          'peak_vram_allocated_mib', 'peak_vram_reserved_mib', 'job_id', 'failure_reason']
PROFILES = {'formal50': 50, 'diagnostic10': 10}


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + '.', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


@contextmanager
def batch_lock(batch):
    with (batch/'generation.lock').open('a') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def asset_snapshot(root, revision):
    """Check the actual local payload against an independently prepared receipt."""
    root = Path(root).resolve()
    if not re.fullmatch('[0-9a-f]{40}', revision):
        raise ValueError('Supply a full model revision, not a branch or tag')
    receipt = read(root/'assets.json')
    if receipt['repositories'].get(MODEL_REPO) != revision:
        raise ValueError('Model revision differs from the prepared asset receipt')
    files = receipt['files']
    actual = {p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()
              and p != root/'assets.json' and not any(part.startswith('.') for part in p.relative_to(root).parts)}
    if actual != files.keys():
        raise ValueError('Asset file set differs from the prepared receipt')
    required = {'config.json', 'models_t5_umt5-xxl-enc-bf16.pth', 'Wan2.2_VAE.pth',
                'google/umt5-xxl/tokenizer_config.json'}
    if not required <= files.keys() or not any(p.endswith('.safetensors') for p in files):
        raise ValueError('Incomplete TI2V-5B payload: model, T5, VAE and tokenizer are required')
    for name, expected in files.items():
        path = (root/name).resolve()
        if not path.is_relative_to(root) or not path.is_file() or sha256(path) != expected:
            raise ValueError('Asset bytes missing, changed or unsafe: ' + name)
    for index in root.glob('*.safetensors.index.json'):
        if index.name not in files or not set(read(index)['weight_map'].values()) <= files.keys():
            raise ValueError('Unreceipted or incomplete model shard index')
    return {'root': str(root), 'revision': revision, 'receipt_sha256': sha256(root/'assets.json'),
            'files': files}


def source_snapshot():
    sources = list((HERE/'wan').rglob('*.py')) + [HERE/'generate.py', HERE/'benchmark.py', HERE/'benchmark_worker.py']
    sources += [PROJECT/'ai6132_benchmark'/name for name in ('dataset.py', 'io.py', 'media.py')]
    return {p.relative_to(PROJECT).as_posix(): sha256(p) for p in sorted(sources)}


def prepare(cases, assets, revision, output, profile='formal50'):
    cases, output = Path(cases).resolve(), Path(output).resolve()
    if output.exists():
        raise ValueError('Use a new batch directory; never overwrite an existing batch')
    if profile not in PROFILES:
        raise ValueError('Unknown profile')
    rows, receipt = load_canonical(cases, decode=True)
    weights = asset_snapshot(assets, revision)
    selected = [dict(row, image_path=str((cases.parent/row['image_path']).resolve()),
                     image_sha256=receipt['images'][row['case_id']]['sha256']) for row in rows]
    settings = {'profile': profile, 'sampling_steps': PROFILES[profile], 'task': 'ti2v-5B',
                'max_area': 1280*704, 'size': [1280, 704], 'sample_solver': 'unipc',
                'shift': 5.0, 'guide_scale': 5.0, 'offload_model': True, 't5_cpu': True,
                'convert_model_dtype': True, 'fps': 24, 'prompt_extension': False}
    batch = {'schema_version': 1, 'created_utc': now(), 'upstream_commit': UPSTREAM,
             'model_id': 'wan2.2-ti2v-5b-' + profile, 'phase': 'formal' if profile == 'formal50' else 'diagnostic',
             'canonical_path': str(cases), 'canonical_sha256': receipt['canonical_sha256'],
             'cases': selected, 'assets': weights, 'settings': settings, 'source_sha256': source_snapshot(),
             'historical_A_delivery': False}
    output.mkdir(parents=True)
    (output/'cases.canonical.csv').write_bytes(cases.read_bytes())
    if sha256(output/'cases.canonical.csv') != receipt['canonical_sha256']:
        raise ValueError('Canonical CSV changed during preparation')
    atomic_json(output/'batch.json', batch)
    atomic_json(output/'batch_receipt.json', {'batch_sha256': sha256(output/'batch.json')})
    export_runs(output, batch)
    return batch


def load_batch(output):
    output = Path(output).resolve()
    if sha256(output/'batch.json') != read(output/'batch_receipt.json')['batch_sha256']:
        raise ValueError('Frozen batch configuration changed')
    batch = read(output/'batch.json')
    if source_snapshot() != batch['source_sha256']:
        raise ValueError('Source changed: use the frozen checkout or prepare a new batch')
    if asset_snapshot(batch['assets']['root'], batch['assets']['revision']) != batch['assets']:
        raise ValueError('Asset receipt changed')
    for path in (Path(batch['canonical_path']), output/'cases.canonical.csv'):
        if sha256(path) != batch['canonical_sha256']:
            raise ValueError('Canonical CSV changed')
    for case in batch['cases']:
        if sha256(case['image_path']) != case['image_sha256']:
            raise ValueError('Reference image changed: ' + case['case_id'])
    return batch


def export_runs(output, batch):
    import csv
    rows = []
    for case in batch['cases']:
        cid = case['case_id']
        result_path = output/'cases'/cid/'result.json'
        result = read(result_path) if result_path.exists() else {'status': 'not_started'}
        row = {key: '' for key in FIELDS}
        row.update({key: case[key] for key in ('case_id', 'category', 'seed', 'source_frame_num', 'target_frames', 'image_sha256')})
        row.update(model_id=batch['model_id'], phase=batch['phase'], status=result['status'],
                   weight_version=MODEL_REPO+'@'+batch['assets']['revision'], weight_revision=batch['assets']['revision'])
        row.update({key: result[key] for key in FIELDS if key in result})
        rows.append(row)
    fd, temporary = tempfile.mkstemp(prefix='runs.', suffix='.tmp', dir=output)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(rows)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, output/'runs.csv')
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    if all(row['status'] == 'success' for row in rows):
        atomic_json(output/'DONE.json', {'status': 'success', 'cases': len(rows), 'utc': now(),
                                       'runs_sha256': sha256(output/'runs.csv')})
    return rows


def validate_native(video, case):
    info = inspect_video(video)
    if (info['frames'], info['fps'], info['codec']) != (int(case['target_frames']), 24, 'h264'):
        raise ValueError('Native frames/fps/codec differ from the requested profile')
    # Canonical images are exactly 16:9; upstream best_output_size at 1280*704
    # area and a 32-pixel patch/stride grid selects 1248*704.
    if (info['width'], info['height']) != (1248, 704):
        raise ValueError('Unexpected TI2V output dimensions: expected 1248x704')
    if abs(info['duration']-int(case['target_frames'])/24) > 1/24+1e-6:
        raise ValueError('Native duration differs from the full requested timeline')
    return info


def generate_one(output, case_id):
    output = Path(output).resolve()
    with batch_lock(output):
        batch = load_batch(output)
        matches = [case for case in batch['cases'] if case['case_id'] == case_id]
        if len(matches) != 1:
            raise ValueError('Unknown canonical case')
        case = matches[0]
        folder = output/'cases'/case_id
        if folder.exists():
            result_path = folder/'result.json'
            result = read(result_path) if result_path.exists() else {}
            if result.get('status') == 'success' and sha256(folder/'native.mp4') == result['video_sha256']:
                validate_native(folder/'native.mp4', case)
                export_runs(output, batch)
                return result
            raise ValueError('Existing attempted case is retained; use a new batch for an explicit retry')
        folder.mkdir(parents=True)
        request = dict(case, **batch['settings'], assets=batch['assets'], model_id=batch['model_id'])
        atomic_json(folder/'request.json', request)
        result = {'status': 'incomplete', 'config': f'cases/{case_id}/request.json',
                  'config_sha256': sha256(folder/'request.json'), 'started_utc': now(),
                  'job_id': os.environ.get('SLURM_JOB_ID')}
        atomic_json(folder/'result.json', result)
        export_runs(output, batch)
        started = time.perf_counter()
        try:
            env = dict(os.environ, HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false')
            with (folder/'worker.log').open('xb') as log:
                subprocess.run([sys.executable, str(HERE/'benchmark_worker.py'), '--request', str(folder/'request.json'),
                                '--out', str(folder)], env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
            info = validate_native(folder/'native.mp4', case)
            performance = read(folder/'worker_result.json')
            result.update(status='success', native_video=f'cases/{case_id}/native.mp4',
                          video_sha256=sha256(folder/'native.mp4'), native_width=info['width'],
                          native_height=info['height'], native_frames=info['frames'], native_fps=info['fps'],
                          native_duration_seconds=info['duration'], **performance)
        except BaseException as exc:
            result.update(status='interrupted' if isinstance(exc, (KeyboardInterrupt, SystemExit)) else 'generation_failed',
                          failure_reason=f'{type(exc).__name__}: {exc}')
            raise
        finally:
            result.update(elapsed_seconds=time.perf_counter()-started, finished_utc=now())
            atomic_json(folder/'result.json', result)
            export_runs(output, batch)
        return result


def main():
    if not os.environ.get('SLURM_JOB_ID'):
        raise SystemExit('TC2: submit this operation through sbatch')
    parser = argparse.ArgumentParser(description=__doc__)
    tasks = parser.add_subparsers(dest='task', required=True)
    prepare_cli = tasks.add_parser('prepare')
    prepare_cli.add_argument('--cases', type=Path, required=True)
    prepare_cli.add_argument('--assets', type=Path, required=True)
    prepare_cli.add_argument('--revision', required=True)
    prepare_cli.add_argument('--out', type=Path, required=True)
    prepare_cli.add_argument('--profile', choices=PROFILES, default='formal50')
    generate_cli = tasks.add_parser('generate')
    generate_cli.add_argument('--batch', type=Path, required=True)
    generate_cli.add_argument('--case-id', required=True)
    args = parser.parse_args()
    if args.task == 'prepare':
        result = prepare(args.cases, args.assets, args.revision, args.out, args.profile)
        print(json.dumps({'model_id': result['model_id'], 'cases': len(result['cases'])}))
    else:
        print(json.dumps(generate_one(args.batch, args.case_id), ensure_ascii=False))


if __name__ == '__main__':
    main()
