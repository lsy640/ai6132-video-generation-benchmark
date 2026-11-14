"""Verify a complete pinned asset tree after a byte-identical transport recovery."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from .auth import read_hf_token
from .download_hunyuan import GLYPH_REPO, GLYPH_REVISION
from .io import sha256, write_json


def finalize(args):
    from huggingface_hub import HfApi
    root = args.root.resolve()
    plan = json.loads(args.plan.read_text())
    transport = json.loads(args.transport_receipt.read_text())
    if transport['source_repository'] != 'modelscope:' + GLYPH_REPO or transport['source_revision'] != GLYPH_REVISION:
        raise ValueError('Transport source differs from the pinned official repository')
    pointer = subprocess.check_output(['git', '--git-dir', str(args.glyph_git), 'show',
                                      GLYPH_REVISION + ':' + transport['source_file']], text=True)
    digest = next(line.split('sha256:')[1] for line in pointer.splitlines() if line.startswith('oid sha256:'))
    length = int(next(line.split()[1] for line in pointer.splitlines() if line.startswith('size ')))
    if (transport['sha256'], transport['bytes']) != (digest, length):
        raise ValueError('Transport receipt differs from the official Git LFS pointer')
    target = root / 'text_encoder/Glyph-SDXL-v2' / transport['source_file']
    if not target.exists():
        source = Path(transport['local_file'])
        if source.stat().st_size != length or sha256(source) != digest:
            raise ValueError('Recovered transport file is not byte-identical')
        target.parent.mkdir(parents=True, exist_ok=True)
        source.rename(target)  # Same Home filesystem; do not make another large copy.
    elif target.stat().st_size != length or sha256(target) != digest:
        raise ValueError('Retained target differs from the official weight')
    api = HfApi(token=read_hf_token(args.token_file))
    metadata = {}
    for repo, revision in plan['repositories'].items():
        if repo.startswith('modelscope:'):
            continue
        info = api.model_info(repo, revision=revision, files_metadata=True)
        if info.sha != revision:
            raise ValueError('Repository revision differs from the pinned plan')
        metadata[repo] = {f.rfilename: f for f in info.siblings}
    checked = {}
    for index, item in enumerate(plan['files'], 1):
        relative = str(Path(item['prefix']) / item['file'])
        path = root / relative
        size = path.stat().st_size
        if size != item['bytes']:
            raise ValueError('Asset size differs from the pinned plan')
        content_hash = hashlib.sha256()
        blob_hash = hashlib.sha1(f'blob {size}\0'.encode())
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                content_hash.update(block)
                if not item.get('upstream_sha256'):
                    blob_hash.update(block)
        actual = content_hash.hexdigest()
        if item['service'] == 'hf':
            record = metadata[item['repo']][item['file']]
            if record.lfs is not None:
                expected = record.lfs.sha256
                if actual != expected:
                    raise ValueError('HF asset differs from the pinned LFS SHA256')
            elif blob_hash.hexdigest() != record.blob_id:
                raise ValueError('HF configuration differs from the pinned Git blob')
        elif item['file'] == transport['source_file']:
            if actual != digest:
                raise ValueError('Glyph weight differs from the official LFS SHA256')
        else:
            official = subprocess.check_output(['git', '--git-dir', str(args.glyph_git), 'show',
                                               GLYPH_REVISION + ':' + item['file']])
            if actual != hashlib.sha256(official).hexdigest():
                raise ValueError('Glyph configuration differs from the pinned official Git blob')
        checked[relative] = actual
        print(json.dumps({'verified_file': index, 'total': len(plan['files']), 'file': relative}), flush=True)
    actual_names = {str(p.relative_to(root)) for p in root.rglob('*') if p.is_file()
                    and p.name != 'assets.json' and not any(s.startswith('.') for s in p.relative_to(root).parts)}
    if actual_names != set(checked):
        raise ValueError('Asset tree contains missing or unexpected files')
    repositories = dict(plan['repositories'])
    repositories['transport:huggingface:' + transport['transport_repository']] = transport['transport_revision']
    receipt = {'repositories': repositories, 'files': checked,
               'transport_receipt_sha256': sha256(args.transport_receipt)}
    destination = root / 'assets.json'
    if destination.exists():
        if json.loads(destination.read_text()) != receipt:
            raise ValueError('Refusing to overwrite a different prepared asset receipt')
    else:
        write_json(destination, receipt)
    print(json.dumps({'status': 'ready', 'files': len(checked), 'bytes': plan['total_bytes'],
                      'assets_sha256': sha256(destination)}), flush=True)


def main():
    if not os.environ.get('SLURM_JOB_ID'):
        raise SystemExit('Submit asset verification through SLURM')
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'plan', 'transport-receipt', 'glyph-git', 'token-file'):
        p.add_argument('--' + name, type=Path, required=True)
    args = p.parse_args()
    try:
        finalize(args)
    except Exception as exc:
        # Metadata requests can contain authentication headers; do not dump exceptions.
        print(json.dumps({'status': 'failed', 'error_type': type(exc).__name__}), file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
