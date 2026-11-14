"""Independently decode and verify every item in a completed formal Hunyuan batch."""
import argparse
import json
import os
from pathlib import Path

from .dataset import load_canonical
from .hunyuan import MODEL_ID, SETTINGS
from .io import read_csv, sha256, write_json
from .media import inspect_video


def verify(cases, output, report):
    output = Path(output).resolve()
    rows, inputs = load_canonical(cases, decode=True)
    frozen = json.loads((output / 'batch.json').read_text())
    if frozen['phase'] != 'formal' or frozen['inputs'] != inputs or frozen['cases'] != rows:
        raise ValueError('Formal frozen inputs differ from the canonical dataset')
    protocol = frozen['protocol']
    if protocol['settings'] != SETTINGS or protocol['source_sha256'] != sha256(Path(__file__).with_name('hunyuan.py')):
        raise ValueError('Generator/settings differ from the formal frozen configuration')
    if protocol['dataset_sha256'] != sha256(Path(__file__).with_name('dataset.py')):
        raise ValueError('Dataset contract changed')
    identifiers = [r['case_id'] for r in rows]
    if json.loads((output / 'selection.json').read_text()) != {'case_ids': identifiers}:
        raise ValueError('Selection does not preserve the entire formal manifest')
    _, runs, csv_hash = read_csv(output / 'runs.csv')
    if [r['case_id'] for r in runs] != identifiers or any(r['status'] != 'success' for r in runs):
        raise ValueError('Formal generation requires all 24 cases to succeed in canonical order')
    verified = []
    for case, row in zip(rows, runs):
        directory = output / case['case_id']
        request = json.loads((directory / 'request.json').read_text())
        result = json.loads((directory / 'result.json').read_text())
        if (row['model_id'] != MODEL_ID or row['phase'] != 'formal' or
                int(row['seed']) != int(case['seed']) or row['category'] != case['category'] or
                int(row['source_frame_num']) != int(case['source_frame_num']) or
                int(row['target_frames']) != int(case['target_frames']) or
                request['prompt'] != case['prompt'] or result['status'] != 'success' or
                row['failure_reason'] or result['failure_reason']):
            raise ValueError(f"Case/request/result mismatch: {case['case_id']}")
        native = directory / 'native.mp4'
        if Path(row['native_video']).resolve() != native or Path(result['native_video']).resolve() != native:
            raise ValueError('Native output path differs from the retained formal case')
        digest = sha256(native)
        if digest != row['video_sha256'] or digest != result['video_sha256']:
            raise ValueError('Native video differs from its terminal result')
        expected_image = inputs['images'][case['case_id']]['sha256']
        if row['image_sha256'] != expected_image or sha256(request['image']) != expected_image:
            raise ValueError('First image differs from the frozen input')
        info = inspect_video(native)  # Full decode, not just container metadata.
        if (info['frames'] != int(case['target_frames']) or info['fps'] != 24 or info['codec'] != 'h264' or
                info['width'] != int(row['native_width']) or info['height'] != int(row['native_height']) or
                int(row['native_frames']) != info['frames'] or float(row['native_fps']) != info['fps']):
            raise ValueError('Native media specifications differ from the frozen case')
        if abs(info['duration'] - info['frames'] / info['fps']) > 1 / 24:
            raise ValueError('Native duration differs from the full frame count')
        if not row['gpu_model'] or not row['driver_version'] or float(row['elapsed_seconds']) <= 0:
            raise ValueError('Missing measured inference hardware/timing')
        verified.append({'case_id': case['case_id'], 'video_sha256': digest, **info})
        print(json.dumps({'verified_case': case['case_id'], 'frames': info['frames']}), flush=True)
    receipt = {'slurm_job_id': os.environ.get('SLURM_JOB_ID'), 'status': 'passed', 'success': 24,
               'short': 18, 'long': 6, 'canonical_sha256': inputs['canonical_sha256'],
               'batch_sha256': sha256(output / 'batch.json'), 'runs_sha256': csv_hash,
               'pilot_validation': frozen.get('pilot_validation', {'status': 'receipt_required'}),
               'videos': verified}
    if Path(report).exists():
        raise ValueError('Use a new verification report; never overwrite evidence')
    write_json(report, receipt)
    print(json.dumps({'status': 'passed', 'success': 24, 'report': str(report)}))


def main():
    if not os.environ.get('SLURM_JOB_ID'):
        raise SystemExit('Submit verification through SLURM')
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cases', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--report', type=Path, required=True)
    args = p.parse_args()
    verify(args.cases, args.out, args.report)


if __name__ == '__main__':
    main()
