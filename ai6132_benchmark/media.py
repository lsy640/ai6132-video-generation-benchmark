"""Full-duration score copies and a sealed four-model handoff (run via sbatch)."""
import argparse
from fractions import Fraction
import json
import math
import os
from pathlib import Path
import random
import re
import secrets
import shutil
import subprocess

from .dataset import load_canonical
from .io import read_csv, sha256, write_csv, write_json

PUBLIC_FIELDS = ['clip_id', 'video_path', 'image_path', 'prompt', 'status', 'failure_reason']


def binary(name):
    value = os.environ.get('AI6132_' + name.upper(), name)
    found = shutil.which(value)
    if not found:
        raise RuntimeError(f'{name} is required in this job environment')
    return found


def inspect_video(path):
    """Fully decode; metadata alone cannot establish file integrity."""
    path = str(Path(path).resolve())
    subprocess.run([binary('ffmpeg'), '-v', 'error', '-xerror', '-err_detect', 'explode', '-threads', '1',
                    '-i', path, '-map', '0:v:0', '-an', '-f', 'null', '-'],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    data = json.loads(subprocess.check_output([
        binary('ffprobe'), '-v', 'error', '-threads', '1', '-select_streams', 'v:0', '-show_frames', '-show_streams',
        '-show_entries', 'frame=width,height,best_effort_timestamp_time:stream=width,height,avg_frame_rate,duration,codec_name,pix_fmt',
        '-of', 'json', path], text=True))
    stream = data['streams'][0]
    frames = data.get('frames', [])
    sizes = {(f['width'], f['height']) for f in frames}
    if not frames or len(sizes) != 1:
        raise ValueError('Empty video or changing frame dimensions')
    fps = float(Fraction(stream['avg_frame_rate']))
    times = [float(f['best_effort_timestamp_time']) for f in frames]
    if fps <= 0 or any(b <= a for a, b in zip(times, times[1:])):
        raise ValueError('Invalid fps or non-monotonic timestamps')
    duration = float(stream.get('duration', times[-1] - times[0] + 1 / fps))
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError('Invalid video duration')
    return dict(width=stream['width'], height=stream['height'], frames=len(frames), fps=fps,
                duration=duration, codec=stream['codec_name'], pix_fmt=stream.get('pix_fmt'),
                timestamps=[t - times[0] for t in times])


def normalize(source, destination):
    """Preserve full playback duration; never loop, speed up, or truncate."""
    destination = Path(destination)
    if destination.exists():
        raise ValueError('Refusing to replace an existing score copy')
    before = inspect_video(source)
    subprocess.run([binary('ffmpeg'), '-v', 'error', '-xerror', '-filter_threads', '1', '-i', str(source),
                    '-map', '0:v:0', '-an', '-sn', '-dn', '-map_metadata', '-1', '-map_metadata:s:v', '-1',
                    '-metadata:s:v:0', 'handler_name=VideoHandler', '-map_chapters', '-1',
                    '-vf', 'setpts=PTS-STARTPTS,scale=768:432:force_original_aspect_ratio=decrease:force_divisible_by=2,pad=768:432:(ow-iw)/2:(oh-ih)/2,setsar=1,fps=12',
                    '-c:v', 'libx264', '-crf', '18', '-preset', 'medium', '-pix_fmt', 'yuv420p',
                    '-threads', '1', '-movflags', '+faststart', '-n', str(destination)],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    after = inspect_video(destination)
    if (after['width'], after['height'], after['fps'], after['codec'], after['pix_fmt']) != (768, 432, 12, 'h264', 'yuv420p'):
        raise ValueError('Score copy specification mismatch')
    if abs(after['duration'] - before['duration']) > 1 / 12 + 1e-6:
        raise ValueError('Score copy did not preserve the full native duration')
    return before, after


def failure(row):
    text = row.get('failure_reason', '').lower()
    if row['status'] == 'decode_failed':
        return 'decode_failed', 'decode_error'
    if row['status'] in {'not_started', 'incomplete', 'interrupted'}:
        return 'generation_failed', 'missing_output'
    if 'out of memory' in text or 'outofmemory' in text:
        return 'generation_failed', 'out_of_memory'
    return 'generation_failed', 'generation_error'


def prepare(cases, runs, output, frozen_outputs=False):
    """Only call after the group freezes delivery; missing cases remain explicit."""
    if not frozen_outputs:
        raise ValueError('Use --outputs-frozen only once all deliveries are frozen')
    rows, case_receipt = load_canonical(cases, decode=True)
    by_case = {r['case_id']: r for r in rows}
    if len(runs) != 4:
        raise ValueError('Supply exactly four distinct MODEL=RUNS.csv inputs')
    output = Path(output).resolve()
    if output.exists():
        raise ValueError('Use a new handoff directory; IDs and retained originals must not be overwritten')
    attempts, source_hashes = {}, {}
    for model, filename in runs.items():
        if not re.fullmatch(r'[A-Za-z0-9_.-]+', model):
            raise ValueError('Invalid model_id')
        fields, records, digest = read_csv(filename)
        if not {'case_id', 'model_id', 'seed', 'status', 'weight_version', 'config'} <= set(fields):
            raise ValueError(f'Missing generation audit fields for {model}')
        source_hashes[model] = digest
        for record in records:
            key = (record['case_id'], model)
            if record['case_id'] not in by_case or key in attempts or record['model_id'] != model:
                raise ValueError('Duplicate, unknown case, or mismatched model')
            case = by_case[record['case_id']]
            if int(record['seed']) != int(case['seed']):
                raise ValueError('Seed differs from shared manifest')
            if record.get('image_sha256') and record['image_sha256'] != case_receipt['images'][case['case_id']]['sha256']:
                raise ValueError('First image differs from shared manifest')
            if record['status'] not in {'success', 'technical_failure', 'generation_failed', 'decode_failed', 'not_started', 'incomplete', 'interrupted'}:
                raise ValueError('Unknown generation status; supply an explicit adapter')
            if record['status'] == 'success':
                if not record['weight_version'] or not record['config']:
                    raise ValueError('Successful records require a weight version and configuration')
                video = Path(record.get('native_video', ''))
                if not video.is_absolute():
                    video = Path(filename).resolve().parent / video
                record['_video'] = str(video)
            attempts[key] = record
    # Public artifacts and the private mapping occupy distinct subdirectories.
    output.mkdir(parents=True, mode=0o700)
    private = output / 'private'
    public = output / 'handoff'
    private.mkdir(mode=0o700)
    (public / 'videos').mkdir(parents=True)
    (public / 'images').mkdir()
    seed = secrets.randbits(128)
    keys = [(r['case_id'], model) for r in rows for model in runs]
    random.Random(seed).shuffle(keys)
    mapping, anonymous, summary = [], [], []
    write_json(private / 'intake.json', {'cases': case_receipt, 'runs_sha256': source_hashes,
                                       'anonymous_seed': seed, 'expected_items': 96,
                                       'ffmpeg_version': subprocess.check_output([binary('ffmpeg'), '-version'], text=True).splitlines()[0]})
    for model, filename in runs.items():
        target = private / f'{model}.runs.csv'
        shutil.copyfile(filename, target)
        if sha256(target) != source_hashes[model]:
            raise ValueError('A delivered runs.csv changed while preparing the handoff')
    for index, (case_id, model) in enumerate(keys, 1):
        case = by_case[case_id]
        clip = f'clip_{index:04d}'
        mapping.append({'clip_id': clip, 'case_id': case_id, 'model_id': model,
                        'category': case['category'], 'target_frames': case['target_frames']})
        record = attempts.get((case_id, model), {'status': 'not_started', 'failure_reason': 'Missing delivered row'})
        audit = dict(record)
        audit.pop('_video', None)
        audit.update(clip_id=clip, case_id=case_id, model_id=model, target_frames=case['target_frames'])
        status, reason = failure(record)
        video_name = ''
        if record['status'] == 'success':
            source = Path(record['_video'])
            if not source.is_file():
                status, reason = 'generation_failed', 'missing_output'
            else:
                if record.get('video_sha256') and sha256(source) != record['video_sha256']:
                    raise ValueError('Delivered native video changed after generation')
                try:
                    info = inspect_video(source)
                except (subprocess.CalledProcessError, ValueError, KeyError, IndexError) as exc:
                    status, reason = 'decode_failed', 'decode_error'
                    audit['decode_error'] = f'{type(exc).__name__}: {exc}'
                else:
                    if info['frames'] != int(case['target_frames']):
                        raise ValueError('Native frame count differs from the frozen case protocol')
                    # Failures here are operational errors, never counted against a generator.
                    before, after = normalize(source, public / 'videos' / f'{clip}.mp4')
                    audit.update(native=before, score_copy=after, native_sha256=sha256(source),
                                 score_sha256=sha256(public / 'videos' / f'{clip}.mp4'))
                    status, reason, video_name = 'success', '', f'videos/{clip}.mp4'
        from PIL import Image
        reference = (Path(cases).resolve().parent / case['image_path']).resolve()
        if sha256(reference) != case_receipt['images'][case_id]['sha256']:
            raise ValueError('Shared reference changed during preprocessing')
        with Image.open(reference) as image:
            image.convert('RGB').save(public / 'images' / f'{clip}.png')
        anonymous.append(dict(zip(PUBLIC_FIELDS, [clip, video_name, f'images/{clip}.png', case['prompt'], status, reason])))
        audit.update(handoff_status=status, handoff_failure_reason=reason)
        summary.append(audit)
    write_csv(private / 'clip_map.csv', list(mapping[0]), mapping)
    write_json(private / 'generation_summary.json', summary)
    summary_fields = ['clip_id', 'case_id', 'model_id', 'target_frames', 'status', 'seed', 'weight_version',
                      'config', 'elapsed_seconds', 'worker_wall_seconds', 'gpu_model', 'peak_vram_allocated_mib',
                      'peak_vram_reserved_mib', 'failure_reason', 'handoff_status', 'handoff_failure_reason']
    write_csv(private / 'generation_summary.csv', summary_fields, summary)
    write_csv(public / 'anonymous_clips.csv', PUBLIC_FIELDS, anonymous)
    write_json(output / 'COMPLETE.json', {'items': 96, 'files': {
        str(p.relative_to(output)): sha256(p) for p in sorted(output.rglob('*')) if p.is_file()}})
    return {'items': 96, 'success': sum(r['status'] == 'success' for r in anonymous)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cases', type=Path, required=True)
    p.add_argument('--runs', nargs=4, required=True, metavar='MODEL=CSV')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--outputs-frozen', action='store_true')
    args = p.parse_args()
    runs = {}
    for entry in args.runs:
        name, sep, path = entry.partition('=')
        if not sep or not path or name in runs:
            p.error('Provide four distinct MODEL=CSV inputs')
        runs[name] = path
    print(json.dumps(prepare(args.cases, runs, args.out, args.outputs_frozen)))


if __name__ == '__main__':
    main()
