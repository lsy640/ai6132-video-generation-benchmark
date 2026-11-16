"""Run new two-wave batches with terminal recovery; frozen/worker.py is historical."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tarfile
import time
import traceback

FIELDS = ['case_id', 'category', 'model_id', 'phase', 'seed', 'source_frame_num',
          'target_frames', 'status', 'origin', 'weight_revision', 'steps', 'resolution',
          'elapsed_seconds', 'generation_seconds', 'peak_memory_gib', 'job_id', 'attempt_count',
          'native_video', 'native_frames', 'native_width', 'native_height', 'native_fps',
          'native_duration_seconds', 'video_sha256', 'evaluation_video', 'evaluation_sha256', 'failure_reason']


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, indent=2), encoding='utf-8')
    tmp.replace(path)


def read(path, default=None):
    return json.loads(path.read_text()) if path.exists() else default


def load_batch(batch):
    batch = batch.resolve()
    config = read(batch / 'config.json')
    for name, expected in config['code_sha256'].items():
        if sha(batch / 'code' / name) != expected:
            raise RuntimeError('Frozen code changed: ' + name)
    assert sha(batch / 'cases.canonical.csv') == config['canonical_sha256']
    assert sha(batch / 'cases.canonical.receipt.json') == config['receipt_sha256']
    with (batch / 'cases.canonical.csv').open(encoding='utf-8', newline='') as f:
        cases = list(csv.DictReader(f))
    receipt = read(batch / 'cases.canonical.receipt.json')
    assert len(cases) == 24 and len({c['case_id'] for c in cases}) == 24
    assert config['short_order'] == [c['case_id'] for c in cases if c['target_frames'] == '129']
    assert config['long_order'] == [c['case_id'] for c in cases if c['target_frames'] == '513']
    for c in cases:
        assert sha(Path(config['input_root']) / c['image_path']) == receipt['images'][c['case_id']]['sha256']
    return config, cases, receipt


def save_runtime_provenance(batch, config):
    root = Path(config['remote_root'])
    weights = read(root / 'weights-manifest.json')
    assert weights['model_id'] == config['profile']['model_id']
    assert weights['revision'] == config['profile']['model_revision']
    assert (root / 'WEIGHTS_READY').read_text().strip() == weights['revision']
    record = {'weights_manifest_sha256': sha(root / 'weights-manifest.json'),
              'environment_sha256': sha(root / 'environment-freeze.txt'),
              'source_revision': config['profile']['source_revision'],
              'source_sha256': {p.relative_to(root / 'source').as_posix(): sha(p)
                                for p in sorted((root / 'source/longcat_video').rglob('*.py'))}}
    path = batch / 'provenance/runtime.json'
    if path.exists():
        assert read(path) == record, 'Runtime changed between waves'
    else:
        write(path, record)
        for name in ('weights-manifest.json', 'environment-freeze.txt'):
            shutil.copy2(root / name, path.parent / name)


def request_for(config, case, receipt, fallback=False):
    frames = int(case['target_frames'])
    blocks = config['memory']['long_offload_blocks' if frames == 513 else 'short_offload_blocks']
    if fallback:
        blocks = config['memory']['oom_fallback_blocks']
    return {**config['profile'], 'trial_id': config['batch_id'] + '/' + case['case_id'],
            'source_case': case['case_id'], 'case_id': case['case_id'], 'category': case['category'],
            'prompt': case['prompt'], 'seed': int(case['seed']), 'source_frame_num': case['source_frame_num'],
            'num_frames': frames, 'cpu_offload_block_count': blocks,
            'image_relative_path': str((Path(config['input_root']) / case['image_path']).resolve()),
            'image_sha256': receipt['images'][case['case_id']]['sha256']}


def validate_report(folder, report, expected, decode=False):
    assert report['status'] == 'succeeded'
    for key in ('model_id', 'model_revision', 'source_revision', 'prompt', 'seed', 'num_frames',
                'resolution', 'num_inference_steps', 'guidance_scale', 'use_distill', 'native_fps', 'image_sha256'):
        assert report[key] == expected[key], key
    assert report['native']['frames'] == expected['num_frames']
    assert report['native']['fps'] == expected['native_fps']
    assert (report['evaluation']['width'], report['evaluation']['height'], report['evaluation']['fps']) == (768, 432, 12)
    assert abs(report['native']['duration_seconds'] - report['evaluation']['duration_seconds']) <= 1/12 + .001
    for name in ('native', 'evaluation'):
        assert sha(folder / (name + '.mp4')) == report[name]['sha256']
        if decode:
            import av
            with av.open(str(folder / (name + '.mp4'))) as container:
                assert sum(1 for _ in container.decode(video=0)) == report[name]['frames']


def link_files(source, target):
    target.mkdir(parents=True, exist_ok=True)
    for p in source.iterdir():
        if not p.is_file() or p.name.endswith('.pt') or p.name in ('distributed-init', 'monitor-state.json', 'RESULT.md'):
            continue
        dest = target / p.name
        if dest.exists():
            assert sha(dest) == sha(p), 'Refusing to overwrite ' + str(dest)
            continue
        try:
            os.link(p, dest)
        except OSError:
            shutil.copy2(p, dest)


def import_prior(batch, config, cases, receipt):
    lookup = {c['case_id']: c for c in cases}
    for case_id, source_path in config['reuse'].items():
        folder = batch / 'cases' / case_id
        if read(folder / 'result.json', {}).get('status') == 'succeeded':
            continue
        source = Path(source_path)
        report = read(source / 'run.json')
        validate_report(source, report, request_for(config, lookup[case_id], receipt), decode=True)
        link_files(source, folder)
        write(folder / 'result.json', {'status': 'succeeded', 'origin': 'imported_validated_trial',
              'source': str(source), 'source_run_sha256': sha(source / 'run.json'),
              'attempt_count': 1, 'finished_utc': now(), 'slurm_job_id': report.get('slurm_job_id')})


def collect(batch, config, cases):
    rows = []
    for c in cases:
        folder = batch / 'cases' / c['case_id']
        terminal = read(folder / 'result.json', {})
        report = read(folder / 'run.json', {})
        n = report.get('native', {})
        ev = report.get('evaluation', {})
        generation = read(folder / 'denoise-result.json', {})
        row = {k: '' for k in FIELDS}
        row.update(case_id=c['case_id'], category=c['category'], model_id=config['profile']['model_id'],
                   phase='full24', seed=c['seed'], source_frame_num=c['source_frame_num'], target_frames=c['target_frames'],
                   status=terminal.get('status', 'not_started'), origin=terminal.get('origin', 'new_generation'),
                   weight_revision=config['profile']['model_revision'], steps=config['profile']['num_inference_steps'],
                   resolution=config['profile']['resolution'], elapsed_seconds=report.get('elapsed_seconds', ''),
                   generation_seconds=generation.get('seconds', ''), peak_memory_gib=generation.get('peak_gpu_gib', ''),
                   job_id=terminal.get('slurm_job_id', ''), attempt_count=terminal.get('attempt_count', 0),
                   failure_reason=terminal.get('error', ''))
        if n:
            row.update(native_video=f"cases/{c['case_id']}/native.mp4", native_frames=n['frames'],
                       native_width=n['width'], native_height=n['height'], native_fps=n['fps'],
                       native_duration_seconds=n['duration_seconds'], video_sha256=n['sha256'])
        if ev:
            row.update(evaluation_video=f"cases/{c['case_id']}/evaluation.mp4", evaluation_sha256=ev['sha256'])
        rows.append(row)
    tmp = batch / 'runs.csv.tmp'
    with tmp.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(batch / 'runs.csv')
    counts = {s: sum(r['status'] == s for r in rows) for s in sorted({r['status'] for r in rows})}
    status = {'updated_utc': now(), 'batch_id': config['batch_id'], 'case_count': 24, 'counts': counts,
              'succeeded': counts.get('succeeded', 0), 'all_succeeded': counts.get('succeeded', 0) == 24,
              'short_succeeded': sum(r['status'] == 'succeeded' and r['target_frames'] == '129' for r in rows),
              'long_succeeded': sum(r['status'] == 'succeeded' and r['target_frames'] == '513' for r in rows)}
    write(batch / 'STATUS.json', status)
    return status


def resource_failure(folder):
    text = '\n'.join(p.read_text(errors='replace')[-16000:] for p in folder.glob('*.log'))
    return any(x in text for x in ('OutOfMemoryError', 'CUDA out of memory', 'DefaultCPUAllocator: can\'t allocate memory', 'CalledProcessError(-9'))


def execute_engine(command, env, log, timeout):
    process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    try:
        return process.wait(timeout=timeout)
    except BaseException:
        # Every stage inherits this group. A timeout must release its GPU too.
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            pass
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()
        raise


def run_case(batch, config, case, receipt):
    folder = batch / 'cases' / case['case_id']
    terminal = read(folder / 'result.json', {})
    if terminal.get('status') == 'succeeded':
        validate_report(folder, read(folder / 'run.json'), request_for(config, case, receipt))
        return
    attempts = folder / 'attempts'
    attempts.mkdir(parents=True, exist_ok=True)
    existing = sorted(attempts.glob('attempt_*'))
    # A completed video is always retained, including recovery after a wave was killed.
    for attempt in existing:
        report = read(attempt / 'run.json', {})
        if report.get('status') == 'succeeded':
            validate_report(attempt, report, request_for(config, case, receipt))
            link_files(attempt, folder)
            write(folder / 'result.json', {'status': 'succeeded', 'origin': 'new_generation',
                  'attempt_count': len(existing), 'slurm_job_id': report.get('slurm_job_id'), 'finished_utc': now()})
            return
    if terminal.get('status') == 'failed' and (not terminal.get('resource_failure') or terminal.get('attempt_limit_exhausted')):
        return
    if len(existing) >= config['resource_attempt_limit']:
        error = 'Technical attempt limit exhausted without a validated successful output'
        if terminal.get('error'):
            error += ': ' + terminal['error']
        write(folder / 'result.json', {**terminal, 'status': 'failed',
              'origin': terminal.get('origin', 'new_generation'),
              'previous_status': terminal.get('status', 'not_started'),
              'attempt_count': len(existing), 'attempt_limit_exhausted': True,
              'error': error, 'finished_utc': now()})
        return
    first_index = len(existing) + 1
    for number in range(first_index, config['resource_attempt_limit'] + 1):
        request = request_for(config, case, receipt, fallback=number > 1)
        attempt = attempts / f'attempt_{number:02d}'
        attempt.mkdir()
        request_path = attempt / 'case-config.json'
        write(request_path, request)
        write(folder / 'result.json', {'status': 'running', 'origin': 'new_generation',
              'attempt_count': number, 'slurm_job_id': os.environ.get('SLURM_JOB_ID'), 'started_utc': now()})
        env = os.environ.copy()
        env.update(LONGCAT_CASE_CONFIG=str(request_path), LONGCAT_CASE_OUTPUT=str(attempt))
        error = ''
        try:
            with (attempt / 'worker.log').open('xb') as log:
                returncode = execute_engine([sys.executable, '-u', str(batch / 'code/engine.py')],
                    env, log, config['case_timeout_seconds'][case['target_frames']])
            report = read(attempt / 'run.json', {})
            if returncode == 0:
                validate_report(attempt, report, request)
                link_files(attempt, folder)
                write(folder / 'result.json', {'status': 'succeeded', 'origin': 'new_generation',
                      'attempt_count': number, 'slurm_job_id': os.environ.get('SLURM_JOB_ID'), 'finished_utc': now()})
                return
            error = report.get('error', f'Engine exited {returncode}')
        except subprocess.TimeoutExpired:
            error = 'Per-case time limit exceeded'
        except (KeyboardInterrupt, SystemExit):
            write(folder / 'result.json', {'status': 'interrupted', 'origin': 'new_generation',
                  'attempt_count': number, 'slurm_job_id': os.environ.get('SLURM_JOB_ID'),
                  'error': 'Wave interrupted; preserve attempt and recover in finish wave', 'finished_utc': now()})
            raise
        except Exception:
            error = traceback.format_exc()
        oom = resource_failure(attempt)
        write(folder / 'result.json', {'status': 'failed', 'origin': 'new_generation',
              'attempt_count': number, 'slurm_job_id': os.environ.get('SLURM_JOB_ID'),
              'error': error, 'resource_failure': oom, 'finished_utc': now()})
        if not oom:
            return


def finalize(batch, config, cases, receipt):
    for c in cases:
        folder = batch / 'cases' / c['case_id']
        terminal = read(folder / 'result.json', {})
        if terminal.get('status') == 'succeeded':
            validate_report(folder, read(folder / 'run.json'), request_for(config, c, receipt), decode=True)
        elif terminal.get('status') != 'failed':
            write(folder / 'result.json', {**terminal, 'status': 'failed',
                  'origin': terminal.get('origin', 'new_generation'),
                  'previous_status': terminal.get('status', 'not_started'),
                  'error': 'Finish wave ended without a terminal result' +
                           (': ' + terminal['error'] if terminal.get('error') else ''),
                  'finished_utc': now()})
    status = collect(batch, config, cases)
    write(batch / 'COMPLETION.json', {**status, 'generation_finished': True, 'finished_utc': now(),
          'note': 'All 24 rows retained. Original successful trial imported explicitly. Technical attempts preserved; no quality rerolls.'})
    exports = Path(config['remote_root']) / 'exports'
    exports.mkdir(exist_ok=True)
    archive = exports / (config['batch_id'] + '.tar.gz')
    tmp = archive.with_suffix('.partial')
    with tarfile.open(tmp, 'w:gz', compresslevel=1) as tf:
        for path in sorted(batch.rglob('*')):
            relative = path.relative_to(batch)
            if path.is_file() and path.suffix != '.pt' and relative.parts[0] != 'logs' and not path.name.endswith('.tmp'):
                tf.add(path, arcname=config['batch_id'] + '/' + relative.as_posix(), recursive=False)
    tmp.replace(archive)
    digest = sha(archive)
    archive.with_suffix(archive.suffix + '.sha256').write_text(digest + '  ' + archive.name + '\n')
    write(batch / 'EXPORT_READY.json', {'archive': str(archive), 'sha256': digest, 'bytes': archive.stat().st_size,
                                      'all_succeeded': status['all_succeeded'], 'finished_utc': now()})
    return status


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', type=Path, required=True)
    parser.add_argument('--wave', choices=['short', 'finish'], required=True)
    args = parser.parse_args()
    batch = args.batch.resolve()
    config, cases, receipt = load_batch(batch)
    save_runtime_provenance(batch, config)
    signal.signal(signal.SIGTERM, lambda signum, frame: sys.exit(128 + signum))
    import_prior(batch, config, cases, receipt)
    lookup = {c['case_id']: c for c in cases}
    order = config['short_order'] + (config['long_order'] if args.wave == 'finish' else [])
    collect(batch, config, cases)
    try:
        for case_id in order:
            write(batch / 'ACTIVE.json', {'case_id': case_id, 'wave': args.wave, 'job_id': os.environ.get('SLURM_JOB_ID'), 'utc': now()})
            run_case(batch, config, lookup[case_id], receipt)
            print(json.dumps(collect(batch, config, cases)), flush=True)
    finally:
        status = collect(batch, config, cases)
        write(batch / (args.wave.upper() + '_WAVE_DONE.json'), status)
    if args.wave == 'finish':
        status = finalize(batch, config, cases, receipt)
        return 0 if status['all_succeeded'] else 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
