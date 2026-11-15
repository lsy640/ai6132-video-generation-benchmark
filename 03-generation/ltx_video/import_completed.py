"""Audit a completed 13B canonical or historical 2B Shared24 delivery.

Derive a portable handoff with byte-identical native videos and relative paths.
The source bundle is read-only; execute this CPU intake through sbatch on TC2.
"""
import argparse
import json
import math
from pathlib import Path, PurePosixPath
import re
import shutil

from common import (MODEL_REVISION, TEXT_REVISION, TEXT_FILES, UPSTREAM_COMMIT,
                    digest, object_digest, read_manifest_snapshot, select_cases, write_json)
from run_cases import CSV_FIELDS
from ai6132_benchmark.io import read_csv, write_csv
from ai6132_benchmark.media import inspect_video

HISTORICAL_MODEL = 'ltxv-2b-0.9.8-distilled'
CANONICAL_MODEL = 'ltxv-13b-0.9.8-distilled'
MEMORY_STRATEGY = 'sequential_bf16_prompt_cache_final_transformer_release'
SOURCE_PATHS = {
    'runner': 'source/03-generation/ltx_video/run_cases.py',
    'common': 'source/03-generation/ltx_video/common.py',
    'memory_adapter': 'source/03-generation/ltx_video/memory_adapter.py',
    'dataset_contract': 'source/ai6132_benchmark/dataset.py',
    'pipeline_template': 'source/03-generation/ltx_video/ltxv-13b-0.9.8-no-enhance.json',
}


def verify_pipeline(pipeline, model_id):
    size = '13b' if model_id == CANONICAL_MODEL else '2b'
    expected = json.loads((Path(__file__).parent / f'ltxv-{size}-0.9.8-no-enhance.json').read_text())
    names = {'checkpoint_path': f'{model_id}.safetensors',
             'spatial_upscaler_model_path': 'ltxv-spatial-upscaler-0.9.8.safetensors',
             'text_encoder_model_name_or_path': 'text'}
    for key, basename in names.items():
        path = PurePosixPath(pipeline.get(key, ''))
        if not path.is_absolute() or '..' in path.parts or path.name != basename:
            raise ValueError(f'Unexpected frozen asset path: {key}')
        expected[key] = str(path)
    if pipeline != expected or (size == '13b' and pipeline['second_pass'].get('tone_map_compression_ratio') != 0.6):
        raise ValueError('Pipeline differs from the fixed official model configuration')


def verify_asset_manifest(frozen, model_id):
    assets = frozen.get('asset_manifest', {})
    expected_files = {f'ltx/{model_id}.safetensors', 'ltx/ltxv-spatial-upscaler-0.9.8.safetensors'}
    expected_files.update('text/' + name for name in TEXT_FILES)
    files = assets.get('files', {})
    if (assets.get('model_revision') != MODEL_REVISION or assets.get('text_revision') != TEXT_REVISION
            or set(files) != expected_files
            or any(not isinstance(value, str) or not re.fullmatch(r'[0-9a-f]{64}', value) for value in files.values())):
        raise ValueError('Asset receipt differs from the fixed model/upscaler/T5 revisions and files')


def relocated_video(record, request, source_id):
    config = PurePosixPath(record.get('config', ''))
    video = PurePosixPath(record.get('native_video', ''))
    if (not config.is_absolute() or config.name != 'pipeline.json' or '..' in config.parts
            or request.get('config') != str(config) or request['record'].get('config') != str(config)
            or not video.is_absolute() or '..' in video.parts or video.suffix.lower() != '.mp4'):
        raise ValueError(f'Invalid retained configuration/video path: {source_id}')
    try:
        relative = video.relative_to(config.parent)
    except ValueError as exc:
        raise ValueError(f'Video escapes its recorded batch: {source_id}') from exc
    if relative != PurePosixPath('cases', source_id, 'native', video.name):
        raise ValueError(f'Video escapes its recorded case: {source_id}')
    return relative.as_posix()


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
    historical = frozen.get('schema_version') == 1 and frozen.get('protocol') == 'shared24'
    current = frozen.get('schema_version') == 3
    model_id = HISTORICAL_MODEL if historical else CANONICAL_MODEL
    ids = [f'{i:02d}' if historical else f'case_{i:04d}' for i in range(1, 25)]
    if (not (historical or current) or frozen.get('phase') != 'formal' or frozen.get('model_id') != model_id
            or frozen.get('upstream_commit') != UPSTREAM_COMMIT):
        raise ValueError('Only the completed formal 13B canonical or historical 2B Shared24 batch is accepted')
    expected_schema, protocol = (1, 'shared24') if historical else (2, 'a-mixed-24-v1')
    if (selection.get('schema_version') != expected_schema or selection.get('protocol') != protocol
            or selection.get('phase') != 'formal' or selection.get('case_ids') != ids
            or selection.get('cases_csv_sha256') != frozen.get('cases_csv_sha256')):
        raise ValueError('Selection must retain all 24 frozen IDs in order')
    if current and frozen.get('cases_csv_sha256') != canonical_sha:
        raise ValueError('Canonical CSV hash differs from the frozen formal manifest')
    old_cases = frozen['cases']
    if [c['case_id'] for c in old_cases] != ids:
        raise ValueError('Frozen cases must contain all 24 IDs in order')
    if sorted(path.name for path in (bundle / 'cases').iterdir() if path.is_dir()) != sorted(ids):
        raise ValueError('Retained case directories must contain exactly all 24 selected IDs')
    settings = frozen['settings']
    if (settings.get('width'), settings.get('height'), settings.get('frame_rate'),
            settings.get('num_frames')) != (768, 432, 24, None):
        raise ValueError('Settings differ from 768x432 / 24 fps / per-case frames')
    verify_asset_manifest(frozen, model_id)
    pipeline = retained('pipeline.json')
    if json.loads(pipeline.read_text()) != frozen['pipeline']:
        raise ValueError('Retained pipeline differs from the frozen configuration')
    verify_pipeline(frozen['pipeline'], model_id)
    source_receipt, code_hashes = None, {}
    if current:
        fixed = {'memory_strategy': MEMORY_STRATEGY, 'conditioning_strength': 1.0,
                 'image_cond_noise_scale': 0.15, 'conditioning_start_frame': 0}
        if (set(settings) != {'width', 'height', 'frame_rate', 'offload_to_cpu', *fixed}
                or type(settings.get('offload_to_cpu')) is not bool
                or any(settings.get(key) != value for key, value in fixed.items())):
            raise ValueError('13B memory/conditioning settings differ from the frozen protocol')
        source_receipt = document('source_receipt.json')
        if (source_receipt != frozen.get('source_receipt') or source_receipt.get('schema_version') != 1
                or set(source_receipt.get('files', {})) != set(SOURCE_PATHS)):
            raise ValueError('Source receipt differs from the frozen batch')
        for name, relative in SOURCE_PATHS.items():
            item = source_receipt['files'][name]
            if item.get('path') != relative or digest(retained(relative)) != item.get('sha256'):
                raise ValueError(f'Source snapshot missing or changed: {name}')
            if name != 'pipeline_template':
                if frozen.get(name + '_sha256') != item['sha256']:
                    raise ValueError(f'Source snapshot differs from frozen code hash: {name}')
                code_hashes[name] = item['sha256']
        template = document(SOURCE_PATHS['pipeline_template'])
        if template != json.loads((Path(__file__).parent / 'ltxv-13b-0.9.8-no-enhance.json').read_text()):
            raise ValueError('Source pipeline template differs from the fixed 13B configuration')
    fields, runs, _ = read_csv(retained('runs.csv'))
    if [r['case_id'] for r in runs] != ids:
        raise ValueError('runs.csv must contain all 24 IDs exactly once in order')
    required_fields = {'case_id', 'model_id', 'status', 'seed', 'native_video', 'video_sha256',
                       'config', 'config_sha256', 'case_fingerprint', 'image_sha256'}
    if not required_fields <= set(fields):
        raise ValueError('runs.csv is missing required audit fields')
    derived, decoded, videos = [], {}, {}
    for old, case, row in zip(old_cases, canonical, runs):
        old_id = old['case_id']
        source_field, target_field = ('target_frame_num', 'requested_frame_num') if historical else ('source_frame_num', 'target_frames')
        canonical_id = f'case_{int(old_id):04d}' if historical else old_id
        if (case['case_id'] != canonical_id
                or any(old.get(k) != case[k] for k in ('category', 'prompt', 'seed', 'image_sha256'))
                or old.get(source_field) != case['source_frame_num']
                or old.get(target_field) != case['target_frames']):
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
        expected = {'case_id': old_id, 'model_id': model_id, 'phase': 'formal',
                    'category': case['category'], 'seed': case['seed'], 'status': 'success',
                    'failure_reason': '', 'weight_version': model_id, 'weight_revision': MODEL_REVISION,
                    'text_encoder_revision': TEXT_REVISION, 'upstream_commit': UPSTREAM_COMMIT,
                    'image_sha256': case['image_sha256'], 'config_sha256': digest(pipeline),
                    source_field: case['source_frame_num'], target_field: case['target_frames']}
        if historical:
            expected['protocol'] = 'shared24'
        else:
            expected.update(memory_strategy=MEMORY_STRATEGY,
                            memory_adapter_sha256=code_hashes['memory_adapter'],
                            source_receipt_sha256=sources['source_receipt.json'],
                            conditioning_strength=1.0, image_cond_noise_scale=0.15,
                            conditioning_start_frame=0, offload_requested=settings['offload_to_cpu'],
                            offload_effective=False)
            if (request.get('source_receipt') != source_receipt or request.get('code_sha256') != code_hashes
                    or request.get('batch_root') != str(PurePosixPath(record.get('config', '')).parent)):
                raise ValueError(f'Request code/source receipt differs from the frozen batch: {old_id}')
        if any(record.get(k) != v for k, v in expected.items()):
            raise ValueError(f'Incomplete or mismatched successful result: {old_id}')
        stable = {key: value for key, value in expected.items() if key not in {'status', 'failure_reason', 'offload_effective'}}
        if any(request['record'].get(key) != value for key, value in stable.items()):
            raise ValueError(f'Request record differs from the frozen model/input: {old_id}')
        # Resolve only within the downloaded bundle, never the remote paths.
        native_dir = (bundle / 'cases' / old_id / 'native').resolve()
        video = retained(relocated_video(record, request, old_id))
        if current:
            audit_relative = f'cases/{old_id}/memory_audit.json'
            expected_audit_path = str(PurePosixPath(request['batch_root']) / audit_relative)
            if record.get('memory_audit') != expected_audit_path or request['record'].get('memory_audit') != expected_audit_path:
                raise ValueError(f'Memory audit path differs from the retained case: {old_id}')
            audit = document(audit_relative)
            if (record.get('memory_audit_sha256') != sources[audit_relative]
                    or audit.get('version') != 1 or audit.get('strategy') != MEMORY_STRATEGY
                    or audit.get('cpu_offload_effective') is not False or audit.get('factory_calls') != 1):
                raise ValueError(f'Memory audit is missing or inconsistent: {old_id}')
            events = audit.get('events', [])
            if (any([event.get('pass_number') for event in events if event.get('event') == name] != [1, 2]
                    for name in ('pass_start', 'pass_complete'))
                    or sum(event.get('event') == 'final_transformer_released_before_vae_decode' for event in events) != 1):
                raise ValueError(f'Memory audit does not record both passes and final release: {old_id}')
        if len(list(native_dir.glob('*.mp4'))) != 1 or digest(video) != record['video_sha256']:
            raise ValueError(f'Missing, ambiguous or changed native video: {old_id}')
        info = inspect_video(video)  # Independently decode every frame in the CPU job.
        if ((info['width'], info['height'], info['frames'], info['fps'])
                != (768, 432, case['target_frames'], 24)
                or abs(info['duration'] - case['target_frames'] / 24) > 1 / 24 + 1e-6
                or (record['native_width'], record['native_height'], record['native_frames'], record['native_fps'])
                != (info['width'], info['height'], info['frames'], info['fps'])
                or info.get('codec') != 'h264'
                or record.get('native_codec') != info['codec']
                or not math.isfinite(record.get('native_duration_seconds', -1))
                or abs(record.get('native_duration_seconds', -1) - info['duration']) > 1 / 24 + 1e-6):
            raise ValueError(f'Full native decode differs from the frozen specification: {old_id}')
        decoded[case['case_id']] = {k: v for k, v in info.items() if k != 'timestamps'}
        videos[case['case_id']] = video
        derived.append(dict(record, case_id=case['case_id'], source_case_id=old_id,
                            source_frame_num=case['source_frame_num'], target_frames=case['target_frames'],
                            native_video=f'native/{case["case_id"]}.mp4', config='pipeline.json'))
    # Fail before publishing a derived handoff if any source changed during intake.
    if any(digest(bundle / name) != sha for name, sha in sources.items()):
        raise ValueError('Retained delivery changed during intake')
    if (digest(Path(cases).resolve()) != canonical_sha
            or any(digest(Path(case['image_path'])) != case['image_sha256'] for case in canonical)):
        raise ValueError('Canonical input changed during intake')
    output.mkdir(parents=True, exist_ok=False)
    (output / 'native').mkdir()
    shutil.copyfile(pipeline, output / 'pipeline.json')
    if digest(output / 'pipeline.json') != sources['pipeline.json']:
        raise ValueError('Pipeline changed while copying the handoff')
    for record in derived:
        target = output / record['native_video']
        shutil.copyfile(videos[record['case_id']], target)
        if digest(target) != record['video_sha256']:
            raise ValueError('Native video changed while copying the handoff')
    if any(digest(bundle / name) != sha for name, sha in sources.items()):
        raise ValueError('Retained delivery changed during handoff copying')
    write_csv(output / 'runs.csv', CSV_FIELDS + ['source_case_id'], derived)
    receipt = {'schema_version': 2, 'kind': 'completed_ltx_canonical_intake',
               'source_schema_version': frozen['schema_version'], 'model_id': model_id,
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
