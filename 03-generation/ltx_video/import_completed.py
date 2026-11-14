"""Audit C's completed Shared24 delivery and derive canonical runs; submit via sbatch.

This is a read-only intake of the historical 24-case batch, never a generator or
resume mode. Original IDs, records, configuration and videos remain untouched.
"""
import argparse
import json
from pathlib import Path

from common import (MODEL_ID, MODEL_REVISION, TEXT_REVISION, UPSTREAM_COMMIT,
                    digest, object_digest, read_manifest_snapshot, select_cases, write_json)
from run_cases import CSV_FIELDS
from ai6132_benchmark.io import read_csv, write_csv
from ai6132_benchmark.media import inspect_video


def intake(cases, bundle, output):
    bundle, output = Path(bundle).resolve(), Path(output).resolve()
    if output.exists() or output.is_relative_to(bundle):
        raise ValueError('Use a new derived directory outside the original bundle')
    canonical, canonical_sha = read_manifest_snapshot(Path(cases).resolve())
    select_cases(canonical, 'formal')
    sources = {}

    def retained(relative):
        path = (bundle / relative).resolve()
        if not path.is_relative_to(bundle) or not path.is_file():
            raise ValueError(f'Missing or escaping retained artifact: {relative}')
        sources[str(path.relative_to(bundle))] = digest(path)
        return path

    def document(relative):
        return json.loads(retained(relative).read_text(encoding='utf-8'))

    frozen = document('batch.json')
    selection = document('selection.json')
    ids = [f'{i:02d}' for i in range(1, 25)]
    if (frozen.get('schema_version') != 1 or frozen.get('protocol') != 'shared24'
            or frozen.get('phase') != 'formal' or frozen.get('model_id') != MODEL_ID
            or frozen.get('upstream_commit') != UPSTREAM_COMMIT):
        raise ValueError('Only the completed formal Shared24 LTX batch is accepted')
    if (selection.get('schema_version') != 1 or selection.get('protocol') != 'shared24'
            or selection.get('phase') != 'formal' or selection.get('case_ids') != ids
            or selection.get('cases_csv_sha256') != frozen.get('cases_csv_sha256')):
        raise ValueError('Selection must retain all 24 frozen original IDs in order')
    old_cases = frozen['cases']
    if [c['case_id'] for c in old_cases] != ids:
        raise ValueError('Frozen cases must contain all 24 original IDs in order')
    settings = frozen['settings']
    if (settings.get('width'), settings.get('height'), settings.get('frame_rate'),
            settings.get('num_frames')) != (768, 432, 24, None):
        raise ValueError('Shared24 settings differ from 768x432 / 24 fps / per-case frames')
    pipeline = retained('pipeline.json')
    if json.loads(pipeline.read_text()) != frozen['pipeline']:
        raise ValueError('Retained pipeline differs from the frozen configuration')
    fields, runs, _ = read_csv(retained('runs.csv'))
    if [r['case_id'] for r in runs] != ids:
        raise ValueError('runs.csv must contain all 24 original IDs exactly once in order')
    derived, decoded = [], {}
    for old, case, row in zip(old_cases, canonical, runs):
        old_id = old['case_id']
        if (case['case_id'] != f'case_{int(old_id):04d}'
                or any(old.get(k) != case[k] for k in ('category', 'prompt', 'seed', 'image_sha256'))
                or old.get('target_frame_num') != case['source_frame_num']
                or old.get('requested_frame_num') != case['target_frames']):
            raise ValueError(f'Frozen input differs from the canonical case: {old_id}')
        request = document(f'cases/{old_id}/request.json')
        record = document(f'cases/{old_id}/result.json')
        expected_settings = dict(settings, num_frames=case['target_frames'])
        fingerprint = object_digest({'case': old, 'frozen': frozen})
        if (request['case'] != old or request['settings'] != expected_settings
                or request['record'].get('case_fingerprint') != fingerprint
                or record.get('case_fingerprint') != fingerprint):
            raise ValueError(f'Request or result is not bound to the frozen case: {old_id}')
        if any(row[k] != ('' if record.get(k) is None else str(record.get(k))) for k in fields):
            raise ValueError(f'runs.csv differs from the terminal result: {old_id}')
        expected = {'case_id': old_id, 'model_id': MODEL_ID, 'phase': 'formal', 'protocol': 'shared24',
                    'category': case['category'], 'seed': case['seed'], 'status': 'success',
                    'failure_reason': '', 'weight_version': MODEL_ID, 'weight_revision': MODEL_REVISION,
                    'text_encoder_revision': TEXT_REVISION, 'upstream_commit': UPSTREAM_COMMIT,
                    'image_sha256': case['image_sha256'], 'config_sha256': digest(pipeline),
                    'target_frame_num': case['source_frame_num'], 'requested_frame_num': case['target_frames']}
        if any(record.get(k) != v for k, v in expected.items()):
            raise ValueError(f'Incomplete or mismatched successful result: {old_id}')
        # Relocate only the basename within this case's delivered native directory.
        # Never open the original submitter's absolute path from a JSON record.
        native_dir = (bundle / 'cases' / old_id / 'native').resolve()
        video = retained(f'cases/{old_id}/native/{Path(record["native_video"]).name}')
        if len(list(native_dir.glob('*.mp4'))) != 1 or digest(video) != record['video_sha256']:
            raise ValueError(f'Missing, ambiguous or changed native video: {old_id}')
        info = inspect_video(video)  # Independently decode every frame in the CPU job.
        if ((info['width'], info['height'], info['frames'], info['fps'])
                != (768, 432, case['target_frames'], 24)
                or abs(info['duration'] - case['target_frames'] / 24) > 1 / 24 + 1e-6
                or (record['native_width'], record['native_height'], record['native_frames'], record['native_fps'])
                != (info['width'], info['height'], info['frames'], info['fps'])):
            raise ValueError(f'Full native decode differs from the frozen specification: {old_id}')
        decoded[case['case_id']] = {k: v for k, v in info.items() if k != 'timestamps'}
        derived.append(dict(record, case_id=case['case_id'], source_case_id=old_id,
                            source_frame_num=case['source_frame_num'], target_frames=case['target_frames'],
                            native_video=str(video), config=str(pipeline)))
    # Fail before publishing a derived handoff if any source changed during intake.
    if any(digest(bundle / name) != sha for name, sha in sources.items()):
        raise ValueError('Retained delivery changed during intake')
    output.mkdir(parents=True)
    write_csv(output / 'runs.csv', CSV_FIELDS + ['source_case_id'], derived)
    receipt = {'schema_version': 1, 'kind': 'completed_shared24_canonical_intake',
               'canonical_sha256': canonical_sha, 'source_bundle': str(bundle), 'source_files': sources,
               'runs_sha256': digest(output / 'runs.csv'), 'success': 24,
               'id_map': {old: new['case_id'] for old, new in zip(ids, canonical)}, 'full_decode': decoded}
    write_json(output / 'import_receipt.json', receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cases', type=Path, required=True)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    receipt = intake(args.cases, args.bundle, args.output)
    print(json.dumps({'success': receipt['success'], 'runs': str(args.output / 'runs.csv')}))


if __name__ == '__main__':
    main()
