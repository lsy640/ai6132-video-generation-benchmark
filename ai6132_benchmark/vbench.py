"""Pinned VBench-I2V custom-input evaluation, retaining each applicable result."""
import argparse
import contextlib
import json
import math
import os
from pathlib import Path
import socket
import subprocess
import sys
from unittest.mock import patch

from .dataset import SUBJECT
from .io import evaluation_inputs, sha256, write_csv, write_json
from .media import inspect_video


def dimensions(category):
    return ['i2v_subject', 'motion_smoothness', 'dynamic_degree'] if category in SUBJECT else ['i2v_background']


def extract_result(raw, dimension, expected_video):
    value = raw[dimension]
    if not isinstance(value, list) or len(value) != 2 or len(value[1]) != 1:
        raise ValueError('Expected one per-video result, not an aggregate-only score')
    item = value[1][0]
    if Path(item['video_path']).resolve() != Path(expected_video).resolve():
        raise ValueError('VBench result points to the wrong video')
    score = item['video_results']
    if isinstance(score, bool):
        score = int(score)  # dynamic_degree legitimately reports a boolean.
    if not isinstance(score, (int, float)) or not math.isfinite(score):
        raise ValueError('Non-finite or non-numeric per-video result')
    return score


@contextlib.contextmanager
def no_downloads():
    """Prepared caches are mandatory; library convenience downloads are disallowed."""
    original = socket.socket.connect
    def connect(sock, address):
        if sock.family in (socket.AF_INET, socket.AF_INET6):
            raise RuntimeError('Evaluation is offline; populate and receipt model caches first')
        return original(sock, address)
    with patch.object(socket.socket, 'connect', connect), patch('subprocess.run', side_effect=RuntimeError('Subprocess downloads disabled during VBench inference')):
        yield


class OfficialVBench:
    def __init__(self, upstream, cache):
        self.upstream, self.cache = Path(upstream).resolve(), Path(cache).resolve()

    def __call__(self, video, image, dimension, directory):
        os.environ['VBENCH_CACHE_DIR'] = str(self.cache)
        os.environ['TORCH_HOME'] = str(self.cache / 'torch')
        os.environ['XDG_CACHE_HOME'] = str(self.cache)
        os.environ['HF_HOME'] = str(self.cache / 'huggingface')
        os.environ['HF_HUB_OFFLINE'] = '1'
        sys.path.insert(0, str(self.upstream))
        from vbench2_beta_i2v import VBenchI2V
        engine = VBenchI2V('cuda', str(self.upstream / 'vbench2_beta_i2v/vbench2_i2v_full_info.json'), str(directory))
        # DreamSim's default relative cache also stays inside the receipted tree.
        previous = Path.cwd()
        try:
            os.chdir(self.cache)
            with no_downloads():
                engine.evaluate(videos_path=str(video), name='result', dimension_list=[dimension],
                                custom_image_folder=str(image), mode='custom_input', local=True, resolution='16-9')
        finally:
            os.chdir(previous)
        return json.loads((directory / 'result_eval_results.json').read_text())


def cache_files(cache):
    return {str(p.relative_to(cache)): sha256(p) for p in sorted(cache.rglob('*'))
            if p.is_file() and '__pycache__' not in p.parts and p.suffix != '.pyc'
            and not any(part.startswith('.') and part != '.cache' for part in p.relative_to(cache).parts)}


def cache_facts(cache):
    return {str(p.relative_to(cache)): (p.stat().st_size, p.stat().st_mtime_ns)
            for p in cache.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix != '.pyc'
            and not any(part.startswith('.') and part != '.cache' for part in p.relative_to(cache).parts)}


def run(manifest, mapping, output, upstream, commit, cache, cache_receipt, phase='formal', backend=None):
    rows, links, receipt = evaluation_inputs(manifest, mapping, phase)
    upstream, cache, output = Path(upstream).resolve(), Path(cache).resolve(), Path(output).resolve()
    if subprocess.check_output(['git', '-C', str(upstream), 'rev-parse', 'HEAD'], text=True).strip() != commit:
        raise ValueError('VBench source must match the fixed commit')
    if subprocess.check_output(['git', '-C', str(upstream), 'status', '--porcelain', '--untracked-files=no'], text=True).strip():
        raise ValueError('VBench tracked source has local changes')
    assets = json.loads(Path(cache_receipt).read_text())
    if not assets or assets != cache_files(cache):
        raise ValueError('VBench cache does not match its prepared SHA256 receipt')
    initial_cache_facts = cache_facts(cache)
    frozen = {'inputs': receipt, 'upstream_commit': commit, 'assets': assets, 'phase': phase,
              'source_sha256': sha256(__file__), 'io_sha256': sha256(Path(__file__).with_name('io.py')),
              'media_sha256': sha256(Path(__file__).with_name('media.py')),
              'protocol': 'custom_input/full duration/768x432/12fps/one sample per case-model'}
    output.mkdir(parents=True, exist_ok=True, mode=0o700)
    import fcntl
    with (output / '.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        batch = output / 'batch.json'
        if batch.exists():
            if json.loads(batch.read_text()) != frozen:
                raise ValueError('VBench frozen batch changed')
        else:
            if any(p.name != '.lock' for p in output.iterdir()):
                raise ValueError('Prior artifacts have no batch receipt')
            write_json(batch, frozen)
        backend = backend or OfficialVBench(upstream, cache)
        tasks = [(row, dimension) for row in rows for dimension in dimensions(links[row['clip_id']]['category'])]
        results = []
        try:
            for row, dimension in tasks:
                directory = output / row['clip_id'] / dimension
                result = directory / 'result.json'
                if directory.exists():
                    continue
                directory.mkdir(parents=True)
                item = {**links[row['clip_id']], 'dimension': dimension}
                write_json(directory / 'request.json', item)
                if row['status'] != 'success':
                    write_json(result, {**item, 'status': 'missing_video', 'failure_reason': row['failure_reason']})
                    continue
                image = (Path(manifest).resolve().parent / row['image_path']).resolve()
                video = (Path(manifest).resolve().parent / row['video_path']).resolve()
                try:
                    if sha256(image) != receipt['files'][row['image_path']] or sha256(video) != receipt['files'][row['video_path']]:
                        raise ValueError('Scoring input changed during evaluation')
                    info = inspect_video(video)
                    if (info['width'], info['height'], info['fps'], info['codec']) != (768, 432, 12, 'h264'):
                        raise ValueError('VBench input is not a verified score copy')
                    # The official filename resolver maps "clip_0001-0" -> "clip_0001".
                    staged = directory / f"{row['clip_id']}-0.mp4"
                    staged.symlink_to(video)
                    reference = directory / Path(row['image_path']).name
                    reference.symlink_to(image)
                    raw_dir = directory / 'raw'
                    raw_dir.mkdir()
                    raw = backend(staged, reference, dimension, raw_dir)
                    if cache_facts(cache) != initial_cache_facts:
                        raise ValueError('Model cache changed during inference; outputs are not validated')
                    write_json(raw_dir / 'retained.json', raw)
                    write_json(result, {**item, 'value': extract_result(raw, dimension, staged), 'status': 'success'})
                except Exception as exc:
                    write_json(result, {**item, 'status': 'technical_failure', 'failure_reason': f'{type(exc).__name__}: {exc}'})
                    raise
        finally:
            for row, dimension in tasks:
                directory = output / row['clip_id'] / dimension
                result = directory / 'result.json'
                results.append(json.loads(result.read_text()) if result.exists() else {
                    **links[row['clip_id']], 'dimension': dimension,
                    'status': 'incomplete' if directory.exists() else 'not_started'})
            temporary = output / 'vbench_results.csv.new'
            write_csv(temporary, ['clip_id', 'case_id', 'model_id', 'category', 'target_frames',
                                  'dimension', 'value', 'status', 'failure_reason'], results)
            temporary.replace(output / 'vbench_results.csv')
        return 0 if all(r['status'] in {'success', 'missing_video'} for r in results) else 1


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--receipt-cache', type=Path, help='Write a SHA256 receipt of an already prepared cache')
    p.add_argument('--cache', type=Path, required=True)
    for name in ('manifest', 'clip-map', 'out', 'upstream', 'cache-receipt'):
        p.add_argument('--' + name, type=Path)
    p.add_argument('--commit')
    p.add_argument('--phase', choices=['formal', 'pilot'], default='formal')
    args = p.parse_args()
    if args.receipt_cache:
        if args.receipt_cache.exists():
            p.error('Never overwrite a cache receipt')
        files = cache_files(args.cache.resolve())
        if not files:
            p.error('Prepare the model caches first')
        write_json(args.receipt_cache, files)
    else:
        if not all([args.manifest, args.clip_map, args.out, args.upstream, args.commit, args.cache_receipt]):
            p.error('Missing evaluation inputs or pinned upstream/cache receipt')
        raise SystemExit(run(args.manifest, args.clip_map, args.out, args.upstream, args.commit,
                             args.cache, args.cache_receipt, args.phase))


if __name__ == '__main__':
    main()
