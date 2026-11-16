"""Prepare fixed public evaluation weights in authorized storage, via sbatch."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import urllib.request
import zipfile

from .io import sha256, write_json

QWEN_REPO = 'Qwen/Qwen3-VL-8B-Instruct'
QWEN_REVISION = '0c351dd01ed87e9c1b53cbc748cba10e6187ff3b'
DINO_REVISION = '7c446df5b9f45747937fb0d72314eb9f7b66930a'
AMT_REVISION = '31d0420ef7788f6dc22aad3445148705007141bb'
DINO_URL = 'https://dl.fbaipublicfiles.com/dino/dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth'
DREAMSIM_URL = 'https://github.com/ssundaram21/dreamsim/releases/download/v0.2.0-checkpoints/dreamsim_ensemble_checkpoint.zip'
RAFT_URL = 'https://dl.dropboxusercontent.com/s/4j4z58wuv8o0mfz/models.zip'


def metadata(repo, revision='main'):
    url = f'https://huggingface.co/api/models/{repo}/revision/{revision}?blobs=true'
    with urllib.request.urlopen(url, timeout=90) as response:
        return json.load(response)


def download(url, destination):
    destination = Path(destination)
    if destination.is_file():
        return {'url': url, 'bytes': destination.stat().st_size, 'sha256': sha256(destination)}
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + '.part')
    with urllib.request.urlopen(url, timeout=180) as response, temporary.open('wb') as stream:
        shutil.copyfileobj(response, stream, length=4 * 1024 * 1024)
    temporary.replace(destination)
    return {'url': url, 'bytes': destination.stat().st_size, 'sha256': sha256(destination)}


def extract(archive, destination, only=None):
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as bundle:
        for member in bundle.infolist():
            if only and member.filename not in only:
                continue
            target = (destination / member.filename).resolve()
            if not target.is_relative_to(destination) or stat.S_ISLNK(member.external_attr >> 16):
                raise ValueError('Unsafe evaluation checkpoint archive member')
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_suffix(target.suffix + '.extracting')
            with bundle.open(member) as source, temporary.open('wb') as output:
                shutil.copyfileobj(source, output, length=4 * 1024 * 1024)
            if target.exists() and sha256(target) != sha256(temporary):
                raise ValueError('Retained checkpoint differs from its original archive')
            temporary.replace(target)


def hf_file(repo, revision, item, directory):
    from huggingface_hub import hf_hub_download
    filename = item['rfilename']
    path = Path(hf_hub_download(repo, filename, revision=revision, local_dir=directory, token=False))
    digest = sha256(path)
    if item.get('lfs'):
        if digest != item['lfs']['sha256'] or path.stat().st_size != item['lfs']['size']:
            raise ValueError(f'Official LFS mismatch: {filename}')
    else:
        data = path.read_bytes()
        git_hash = hashlib.sha1(f'blob {len(data)}\0'.encode() + data).hexdigest()
        if git_hash != item['blobId']:
            raise ValueError(f'Official Git blob mismatch: {filename}')
    print(json.dumps({'verified_asset': filename, 'bytes': path.stat().st_size}), flush=True)
    return {'repository': repo, 'revision': revision, 'filename': filename,
            'bytes': path.stat().st_size, 'sha256': digest}


def prepare_qwen(root):
    info = metadata(QWEN_REPO, QWEN_REVISION)
    if info['sha'] != QWEN_REVISION:
        raise ValueError('Qwen revision differs from its fixed source')
    target = root / 'qwen3-vl-8b'
    files = [f for f in info['siblings'] if '/' not in f['rfilename']
             and f['rfilename'].endswith(('.json', '.txt', '.safetensors', '.jinja'))]
    facts = [hf_file(QWEN_REPO, QWEN_REVISION, item, target) for item in files]
    hashes = {f['filename']: f['sha256'] for f in facts}
    receipt = {'repositories': {QWEN_REPO: QWEN_REVISION}, 'files': hashes}
    saved = target / 'assets.json'
    if saved.exists():
        if json.loads(saved.read_text()) != receipt:
            raise ValueError('Qwen asset receipt changed')
    else:
        write_json(saved, receipt)
    write_json(root / 'receipts/qwen-download.json', {'source': info['sha'], 'files': facts})


def prepare_vbench(root, dino_source):
    cache = root / 'vbench-cache'
    cache.mkdir(parents=True, exist_ok=True)
    if subprocess.check_output(['git', '-C', str(dino_source), 'rev-parse', 'HEAD'], text=True).strip() != DINO_REVISION:
        raise ValueError('DINO source differs from its fixed official commit')
    target = cache / 'dino_model/facebookresearch_dino_main'
    if not target.exists():
        shutil.copytree(dino_source, target, ignore=shutil.ignore_patterns('.git', '__pycache__'))
    downloads = root / 'downloads'
    facts = [download(DINO_URL, cache / 'dino_model/dino_vitbase16_pretrain.pth')]
    facts.append(download(DREAMSIM_URL, downloads / 'dreamsim_ensemble_checkpoint.zip'))
    extract(downloads / 'dreamsim_ensemble_checkpoint.zip', cache / 'models')
    facts.append(download(RAFT_URL, downloads / 'raft-models.zip'))
    extract(downloads / 'raft-models.zip', cache / 'raft_model', {'models/raft-things.pth'})
    amt = metadata('lalala125/AMT', AMT_REVISION)
    checkpoint = next(f for f in amt['siblings'] if f['rfilename'] == 'amt-s.pth')
    facts.append(hf_file('lalala125/AMT', amt['sha'], checkpoint, cache / 'amt_model'))
    write_json(root / 'receipts/vbench-download.json', {'dino_source_commit': DINO_REVISION, 'files': facts})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--dino-source', type=Path, required=True)
    parser.add_argument('--task', choices=['vbench', 'qwen', 'both'], default='both')
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        parser.error('Submit evaluation downloads through sbatch')
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    (root / 'receipts').mkdir(exist_ok=True)
    # Budget both complete weight sets and environment copies on this filesystem.
    if shutil.disk_usage(root).free < 40_000_000_000:
        parser.error('Evaluation storage requires a 40 GB free-space budget')
    os.environ['HF_HOME'] = str(root / 'huggingface')
    os.environ['HF_HUB_DISABLE_XET'] = '1'
    os.environ['HF_HUB_DOWNLOAD_TIMEOUT'] = '180'
    if args.task in {'vbench', 'both'}:
        prepare_vbench(root, args.dino_source.resolve())
    if args.task in {'qwen', 'both'}:
        prepare_qwen(root)


if __name__ == '__main__':
    main()
