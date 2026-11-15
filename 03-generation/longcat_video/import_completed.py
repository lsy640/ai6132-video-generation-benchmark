"""Validate the retained LongCat batch and export/verify a portable B handoff.

Run CPU intake through sbatch on TC2. Source records and video bytes are immutable.
"""
import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import shutil

from ai6132_benchmark.dataset import load_canonical
from ai6132_benchmark.io import read_csv, sha256, write_csv, write_json

MODEL_ID = 'longcat-video-480p-16step-distill'
WEIGHT_ID = 'meituan-longcat/LongCat-Video'
WEIGHT_REVISION = '03b55529b1d1d4045f5fbe14d65c8c6e8116b278'
SOURCE_REVISION = '58eb336edc5c8c63389838727e40b8c18434e54e'
REFERENCE = Path(__file__).parent / 'frozen'
FIELDS = ['case_id', 'category', 'model_id', 'phase', 'seed', 'source_frame_num', 'target_frames',
          'status', 'source_status', 'origin', 'weight_version', 'weight_revision', 'upstream_commit',
          'config', 'config_sha256', 'image_sha256', 'native_video', 'video_sha256',
          'native_width', 'native_height', 'native_frames', 'native_fps', 'native_duration_seconds',
          'native_codec', 'elapsed_seconds', 'generation_seconds', 'gpu_model',
          'peak_vram_allocated_mib', 'peak_vram_reserved_mib', 'cpu_offload_blocks',
          'source_formal_evaluation', 'job_id', 'failure_reason']


def require(condition, message):
    if not condition:
        raise ValueError(message)


def document(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def retained(root, relative):
    path = (root / relative).resolve()
    require(path.is_relative_to(root.resolve()) and path.is_file(), 'Missing or unsafe retained path: ' + str(relative))
    return path


def inspect_native(path, decoder='ffmpeg'):
    if decoder == 'ffmpeg':
        from ai6132_benchmark.media import inspect_video
        return inspect_video(path)
    import av
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        fps = float(stream.average_rate)
        require(math.isfinite(fps) and fps > 0, 'Invalid native fps')
        times, sizes = [], set()
        for frame in container.decode(stream):
            require(frame.time is not None, 'Missing native timestamp')
            times.append(float(frame.time))
            sizes.add((frame.width, frame.height))
        require(times and len(sizes) == 1 and all(b > a for a, b in zip(times, times[1:])), 'Invalid decoded video/timestamps')
        width, height = next(iter(sizes))
        duration = float(stream.duration * stream.time_base) if stream.duration is not None else times[-1]-times[0]+1/fps
        return dict(width=width, height=height, frames=len(times), fps=fps, duration=duration,
                    codec=stream.codec_context.name, timestamps=[t-times[0] for t in times])


def audit_source(raw, cases_csv, decoder='ffmpeg', videos=None):
    raw = Path(raw).resolve()
    canonical, receipt = load_canonical(cases_csv, decode=True)
    config_path = retained(raw, 'config.json')
    require(sha256(config_path) == sha256(REFERENCE / 'batch.json'), 'Batch differs from the published frozen source')
    batch = document(config_path)
    require(batch['canonical_sha256'] == receipt['canonical_sha256'], 'Canonical cases changed')
    require(sha256(retained(raw, 'cases.canonical.csv')) == receipt['canonical_sha256'], 'Retained canonical CSV changed')
    frozen_receipt_path = retained(raw, 'cases.canonical.receipt.json')
    require(sha256(frozen_receipt_path) == batch['receipt_sha256'], 'Retained image receipt changed')
    frozen_receipt = document(frozen_receipt_path)
    require(all(receipt['images'][c['case_id']]['sha256'] == frozen_receipt['images'][c['case_id']]['sha256'] for c in canonical),
            'Reference images differ from the frozen batch')
    profile = batch['profile']
    require((profile['model_id'], profile['model_revision'], profile['source_revision'], profile['resolution'],
             profile['num_inference_steps'], profile['guidance_scale'], profile['use_distill'], profile['native_fps']) ==
            (WEIGHT_ID, WEIGHT_REVISION, SOURCE_REVISION, '480p', 16, 1.0, True, 15), 'Unexpected model or sampling profile')
    for name, digest in batch['code_sha256'].items():
        require(sha256(retained(raw, 'code/' + name)) == digest == sha256(REFERENCE / name), 'Source code changed: ' + name)
    runtime = document(retained(raw, 'provenance/runtime.json'))
    require(runtime == document(REFERENCE / 'runtime.json'), 'Runtime provenance changed')
    for name, key in [('environment-freeze.txt', 'environment_sha256'), ('weights-manifest.json', 'weights_manifest_sha256')]:
        require(sha256(retained(raw, 'provenance/' + name)) == runtime[key], 'Runtime receipt changed: ' + name)
    completion = document(retained(raw, 'COMPLETION.json'))
    require(completion['all_succeeded'] and completion['counts'] == {'succeeded': 24}, 'Source batch is incomplete')
    _, source_rows, source_csv_sha = read_csv(retained(raw, 'runs.csv'))
    require([r['case_id'] for r in source_rows] == [c['case_id'] for c in canonical], 'Source must retain all 24 canonical rows in order')
    require(sorted(p.name for p in (raw/'cases').iterdir() if p.is_dir()) == sorted(c['case_id'] for c in canonical),
            'Case directories do not match the selected 24 cases')
    rows, evidence, sources = [], {}, {}
    stable = ('model_id', 'model_revision', 'source_revision', 'prompt', 'seed', 'resolution',
              'num_inference_steps', 'guidance_scale', 'use_distill', 'native_fps', 'num_frames',
              'image_sha256', 'cpu_offload_block_count', 'share_identical_rope_frequency_cache', 'generation_mode')
    for case, source_row in zip(canonical, source_rows):
        cid = case['case_id']
        folder = raw/'cases'/cid
        report, request, result = [document(retained(folder, name)) for name in ('run.json', 'request.json', 'result.json')]
        shape = document(retained(folder, 'generation-shape.json'))
        cost = document(retained(folder, 'denoise-result.json'))
        expected_blocks = batch['memory']['short_offload_blocks' if case['target_frames'] == '129' else 'long_offload_blocks']
        expected = dict(profile, prompt=case['prompt'], seed=int(case['seed']), num_frames=int(case['target_frames']),
                        image_sha256=receipt['images'][cid]['sha256'], cpu_offload_block_count=expected_blocks)
        require(report.get('source_case') == cid and all(report.get(k) == request.get(k) == expected.get(k) for k in stable),
                'Request/result differ from frozen inputs or configuration: ' + cid)
        require(source_row['status'] == report['status'] == result['status'] == 'succeeded', 'Case is not a completed success: ' + cid)
        require(source_row['category'] == case['category'] and int(source_row['seed']) == int(case['seed']) and
                source_row['target_frames'] == case['target_frames'] and source_row['source_frame_num'] == case['source_frame_num'] and
                source_row['model_id'] == WEIGHT_ID and source_row['weight_revision'] == WEIGHT_REVISION and
                int(source_row['steps']) == 16 and source_row['resolution'] == '480p', 'Source CSV differs from canonical data: ' + cid)
        require(sha256(retained(folder, 'input.jpg')) == expected['image_sha256'], 'Retained input image changed: ' + cid)
        require(result['attempt_count'] == 1 and source_row['attempt_count'] == '1', 'Unexpected full24 technical attempts: ' + cid)
        origin = 'imported_validated_trial' if cid in batch['reuse'] else 'new_generation'
        require(result['origin'] == source_row['origin'] == origin, 'Origin changed: ' + cid)
        if origin == 'imported_validated_trial':
            require(sha256(retained(folder, 'run.json')) == result['source_run_sha256'], 'Imported source report changed')
            proof = {}
            for line in retained(folder, 'code-and-environment.sha256').read_text().splitlines():
                digest, name = line.split(maxsplit=1)
                proof[Path(name.strip()).name] = digest
            require(proof.get('run_trial.py') == sha256(REFERENCE/'trial_engine.py') and
                    proof.get('config.json') == sha256(REFERENCE/'trial_config.json') and
                    proof.get('weights-manifest.json') == runtime['weights_manifest_sha256'] and
                    proof.get('environment-freeze.txt') == runtime['environment_sha256'], 'Imported trial source proof changed')
        steps = [json.loads(line) for line in retained(folder, 'steps.jsonl').read_text().splitlines() if line.strip()]
        require([s['step'] for s in steps] == list(range(1, 17)) and all(s['total_steps'] == 16 and math.isfinite(s['seconds']) and s['seconds'] > 0 for s in steps)
                and cost['completed_steps'] == 16, 'Missing/inconsistent 16-step record: ' + cid)
        require(shape['frames'] == int(case['target_frames']) and shape['offloaded_blocks'] == expected_blocks and
                shape['gpu'] == 'NVIDIA L40S', 'Hardware or effective offload mismatch: ' + cid)
        for key, value in [('elapsed_seconds', report['elapsed_seconds']), ('generation_seconds', cost['seconds']), ('peak_memory_gib', cost['peak_gpu_gib'])]:
            require(math.isfinite(float(value)) and float(value) > 0 and float(source_row[key]) == value, 'Cost record differs: ' + cid)
        source_relative = 'cases/' + cid + '/native.mp4'
        require(source_row['native_video'] == source_relative, 'Unexpected native path: ' + cid)
        video = retained(Path(videos).resolve(), cid+'.mp4') if videos else retained(raw, source_relative)
        video_hash = sha256(video)
        require(video_hash == report['native']['sha256'] == source_row['video_sha256'], 'Native bytes changed: ' + cid)
        info = inspect_native(video, decoder)
        require((info['frames'], info['width'], info['height'], info['fps'], info['codec']) ==
                (int(case['target_frames']), 832, 480, 15.0, 'h264'), 'Decoded native specification differs: ' + cid)
        require(abs(info['duration'] - info['frames']/15) <= 1/15+1e-6, 'Native duration differs: ' + cid)
        require(all(float(source_row[key]) == value == report['native'][report_key] for key, value, report_key in
                    [('native_frames', info['frames'], 'frames'), ('native_width', info['width'], 'width'),
                     ('native_height', info['height'], 'height'), ('native_fps', info['fps'], 'fps')]), 'Native metadata differs: ' + cid)
        require(float(source_row['native_duration_seconds']) == report['native']['duration_seconds'] and
                abs(report['native']['duration_seconds'] - info['duration']) <= 1/15+1e-6, 'Duration record differs: ' + cid)
        row = dict(case_id=cid, category=case['category'], model_id=MODEL_ID, phase=batch['phase'], seed=case['seed'],
            source_frame_num=case['source_frame_num'], target_frames=case['target_frames'], status='success', source_status='succeeded',
            origin=origin, weight_version=WEIGHT_ID, weight_revision=WEIGHT_REVISION, upstream_commit=SOURCE_REVISION,
            config='configs/'+cid+'.json', config_sha256=sha256(retained(folder, 'request.json')), image_sha256=expected['image_sha256'],
            native_video='native/'+cid+'.mp4', video_sha256=video_hash, native_width=info['width'], native_height=info['height'],
            native_frames=info['frames'], native_fps=info['fps'], native_duration_seconds=info['duration'], native_codec=info['codec'],
            elapsed_seconds=report['elapsed_seconds'], generation_seconds=cost['seconds'], gpu_model=shape['gpu'],
            peak_vram_allocated_mib=cost['peak_gpu_gib']*1024, peak_vram_reserved_mib='', cpu_offload_blocks=expected_blocks,
            source_formal_evaluation=str(report.get('formal_evaluation', False)).lower(), job_id=source_row['job_id'], failure_reason='')
        rows.append(row)
        evidence[cid] = {k: v for k, v in info.items() if k != 'timestamps'}
        sources[cid] = {'video': video, 'request': retained(folder, 'request.json')}
    return rows, {'canonical_sha256': receipt['canonical_sha256'], 'source_batch_sha256': sha256(config_path),
                  'source_runs_sha256': source_csv_sha, 'decoded': evidence}, sources


def export(raw, cases, output, decoder='ffmpeg'):
    output = Path(output).resolve()
    require(not output.exists(), 'Use a new delivery directory; source and previous exports are retained')
    rows, evidence, sources = audit_source(raw, cases, decoder)
    (output/'native').mkdir(parents=True)
    (output/'configs').mkdir()
    for row in rows:
        files = sources[row['case_id']]
        for key, src_key, hash_key in [('native_video', 'video', 'video_sha256'), ('config', 'request', 'config_sha256')]:
            shutil.copyfile(files[src_key], output/row[key])
            require(sha256(output/row[key]) == row[hash_key], 'Source changed during export')
    write_csv(output/'runs.csv', FIELDS, rows)
    write_json(output/'selection.json', {'model_id': MODEL_ID, 'phase': 'full24',
        'case_ids': [r['case_id'] for r in rows], 'canonical_sha256': evidence['canonical_sha256'],
        'exceptions': ['Three independent common pilot cases were not completed.',
                       'case_0001 reuses the one completed identical-profile technical trial.',
                       'Source formal_evaluation=false is preserved; B/group acceptance is pending.']})
    evidence.update(model_id=MODEL_ID, status='member_D_local_export_verified', independent_B_acceptance=False,
        decoder=decoder, utc=datetime.now(timezone.utc).isoformat(),
        files={p.relative_to(output).as_posix(): sha256(p) for p in sorted(output.rglob('*')) if p.is_file()})
    write_json(output/'receipt.json', evidence)
    return evidence


def verify(raw, cases, delivery, decoder='ffmpeg'):
    delivery = Path(delivery).resolve()
    rows, evidence, _ = audit_source(raw, cases, decoder, videos=delivery/'native')
    fields, actual, _ = read_csv(retained(delivery, 'runs.csv'))
    require(fields == FIELDS and len(actual) == 24, 'Unexpected delivery CSV schema/count')
    require(actual == [{k: str(row[k]) for k in FIELDS} for row in rows], 'Delivery table differs from retained source')
    receipt = document(retained(delivery, 'receipt.json'))
    for name, digest in receipt['files'].items():
        require(sha256(retained(delivery, name)) == digest, 'Delivery file changed: ' + name)
    for row in rows:
        require(sha256(retained(delivery, row['config'])) == row['config_sha256'], 'Exported request changed')
    require(sorted(p.name for p in (delivery/'native').iterdir()) == [r['case_id']+'.mp4' for r in rows], 'Extra/missing native files')
    return dict(evidence, model_id=MODEL_ID, status='verified', decoder=decoder, cases=24, adapter_sha256=sha256(__file__))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['export', 'verify'])
    p.add_argument('--raw', type=Path, required=True)
    p.add_argument('--cases', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True, help='new delivery output, or existing delivery for verify')
    p.add_argument('--decoder', choices=['ffmpeg', 'pyav'], default='ffmpeg')
    p.add_argument('--verification-out', type=Path)
    a = p.parse_args()
    result = export(a.raw, a.cases, a.out, a.decoder) if a.action == 'export' else verify(a.raw, a.cases, a.out, a.decoder)
    if a.verification_out:
        require(not a.verification_out.exists(), 'Verification output already exists')
        write_json(a.verification_out, result)
    print(json.dumps({'status': result['status'], 'cases': len(result['decoded']), 'model_id': MODEL_ID}))


if __name__ == '__main__':
    main()
