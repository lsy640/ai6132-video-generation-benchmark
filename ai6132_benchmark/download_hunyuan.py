"""Download only pinned Hunyuan I2V assets, within a checked Home budget."""
import argparse
import fcntl
import json
import logging
import os
from pathlib import Path
import re
import sys

from .auth import read_hf_token
from .io import sha256, write_json

GLYPH_REPO = 'AI-ModelScope/Glyph-SDXL-v2'
GLYPH_REVISION = 'b51a8604c03c9a4e71a389f4c7deeb526320f2e6'
GLYPH_FILES = ('assets/color_idx.json', 'assets/multilingual_10-lang_idx.json', 'checkpoints/byt5_model.pt')
DESTINATIONS = {'tencent/HunyuanVideo-1.5': '', 'Qwen/Qwen2.5-VL-7B-Instruct': 'text_encoder/llm',
                'google/byt5-small': 'text_encoder/byt5-small',
                'black-forest-labs/FLUX.1-Redux-dev': 'vision_encoder/siglip'}


def run(args):
    if not os.environ.get('SLURM_JOB_ID'):
        raise ValueError('Submit through SLURM')
    from huggingface_hub import HfApi, hf_hub_download
    from modelscope.hub.api import HubApi
    from modelscope.hub.file_download import model_file_download
    for name in ('huggingface_hub', 'urllib3', 'httpx', 'modelscope'):
        logging.getLogger(name).setLevel(logging.CRITICAL)
    token = read_hf_token(args.token_file)
    preflight = json.loads(args.preflight.read_text())
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    with (root / '.download.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        api, files, repositories = HfApi(token=token), [], {}
        for repo, prefix in DESTINATIONS.items():
            checked = preflight['repositories'][repo]
            revision = checked['revision']
            if checked.get('weight_access_http') != 200 or not re.fullmatch('[0-9a-f]{40}', revision):
                raise ValueError('Required repository access/revision has not been checked')
            metadata = api.model_info(repo, revision=revision, files_metadata=True)
            if metadata.sha != revision:
                raise ValueError('HF revision changed')
            available = {f.rfilename: f for f in metadata.siblings}
            repositories[repo] = revision
            for selected in checked['files']:
                name = selected['rfilename']
                item = available[name]
                if item.size != selected['size']:
                    raise ValueError('Pinned file size differs from preflight')
                lfs = item.lfs
                expected = lfs.get('sha256') if isinstance(lfs, dict) else getattr(lfs, 'sha256', None)
                files.append({'service': 'hf', 'repo': repo, 'revision': revision, 'file': name,
                              'prefix': prefix, 'bytes': item.size, 'upstream_sha256': expected})
        glyph = {f['Path']: f for f in HubApi().get_model_files(GLYPH_REPO, revision=GLYPH_REVISION)}
        repositories['modelscope:' + GLYPH_REPO] = GLYPH_REVISION
        for name in GLYPH_FILES:
            files.append({'service': 'modelscope', 'repo': GLYPH_REPO, 'revision': GLYPH_REVISION,
                          'file': name, 'prefix': 'text_encoder/Glyph-SDXL-v2',
                          'bytes': glyph[name]['Size'], 'upstream_sha256': None})
        plan = {'repositories': repositories, 'files': files,
                'total_bytes': sum(f['bytes'] for f in files), 'home_bytes_before': args.home_bytes_before,
                'home_limit_bytes': args.home_limit_bytes, 'reserve_bytes': args.reserve_bytes}
        if args.home_bytes_before + plan['total_bytes'] + args.reserve_bytes > args.home_limit_bytes:
            raise ValueError('Planned assets plus measured Home usage and reserve exceed the Home limit')
        plan_path = root.parent / 'download-plan.json'
        if plan_path.exists():
            if json.loads(plan_path.read_text()) != plan:
                raise ValueError('Download plan changed; inspect the retained plan before continuing')
        else:
            write_json(plan_path, plan)
        print(json.dumps({'event': 'pinned_plan', 'files': len(files), 'bytes': plan['total_bytes'],
                          'projected_home_bytes_with_reserve': args.home_bytes_before + plan['total_bytes'] + args.reserve_bytes}), flush=True)
        for number, item in enumerate(files, 1):
            destination = root / item['prefix'] / item['file']
            print(json.dumps({'event': 'download', 'number': number, 'total': len(files),
                              'file': str(destination.relative_to(root)), 'bytes': item['bytes']}), flush=True)
            if destination.is_file() and destination.stat().st_size == item['bytes']:
                if item['upstream_sha256'] and sha256(destination) != item['upstream_sha256']:
                    raise ValueError('Previously downloaded asset checksum mismatch; retain it for inspection')
            elif item['service'] == 'hf':
                hf_hub_download(item['repo'], item['file'], revision=item['revision'],
                                local_dir=str(root / item['prefix']), token=token)
            else:
                model_file_download(item['repo'], item['file'], revision=item['revision'],
                                    local_dir=str(root / item['prefix']))
            if destination.stat().st_size != item['bytes']:
                raise ValueError('Downloaded asset size mismatch')
            if item['upstream_sha256'] and sha256(destination) != item['upstream_sha256']:
                raise ValueError('Downloaded asset checksum differs from pinned upstream LFS')
        expected = {str(Path(f['prefix']) / f['file']) for f in files}
        actual = {str(p.relative_to(root)) for p in root.rglob('*') if p.is_file()
                  and p.name != 'assets.json' and not any(s.startswith('.') for s in p.relative_to(root).parts)}
        if expected != actual:
            raise ValueError('Asset tree differs from the complete pinned download plan')
        receipt = {'repositories': repositories, 'files': {name: sha256(root / name) for name in sorted(expected)}}
        target = root / 'assets.json'
        if target.exists():
            if json.loads(target.read_text()) != receipt:
                raise ValueError('Prepared asset receipt differs from retained assets')
        else:
            write_json(target, receipt)
        print(json.dumps({'event': 'ready', 'files': len(expected), 'assets_receipt_sha256': sha256(target)}), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--preflight', type=Path, required=True)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--token-file', type=Path, required=True)
    p.add_argument('--home-bytes-before', type=int, required=True)
    p.add_argument('--home-limit-bytes', type=int, default=100_000_000_000)
    p.add_argument('--reserve-bytes', type=int, default=4_000_000_000)
    args = p.parse_args()
    try:
        run(args)
    except Exception as exc:
        # Transport exceptions may contain signed URLs; never print their message/traceback.
        status = getattr(getattr(exc, 'response', None), 'status_code', None)
        print(json.dumps({'event': 'failed', 'error_type': type(exc).__name__, 'http_status': status}), file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
