"""Pin official source and stage the official inference BF16 weights, one shard at a time."""
import gc
import hashlib
import json
import os
from pathlib import Path
import tarfile
import urllib.request

import torch
from huggingface_hub import HfApi, hf_hub_download
from safetensors import safe_open
from safetensors.torch import save_file

ROOT = Path(os.environ['LONGCAT_ROOT'])
CONFIG = json.loads((ROOT / 'code/config.json').read_text())
MODEL = ROOT / 'model'
MODEL.mkdir(exist_ok=True)
INCOMING = ROOT / 'incoming'
INCOMING.mkdir(exist_ok=True)
manifest_path = ROOT / 'weights-manifest.json'
manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {
    'model_id': CONFIG['model_id'], 'revision': CONFIG['model_revision'],
    'conversion': 'Official inference uses torch.bfloat16; original F32 weights cast to BF16 without quantization.',
    'files': {}}


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def save_manifest():
    tmp = manifest_path.with_suffix('.tmp')
    tmp.write_text(json.dumps(manifest, indent=2))
    tmp.replace(manifest_path)


source = ROOT / 'source'
if not (source / 'longcat_video/pipeline_longcat_video.py').exists():
    archive = ROOT / 'source.tar.gz'
    urllib.request.urlretrieve('https://codeload.github.com/meituan-longcat/LongCat-Video/tar.gz/' + CONFIG['source_revision'], archive)
    source.mkdir(exist_ok=True)
    with tarfile.open(archive) as tf:
        for item in tf:
            pieces = Path(item.name).parts[1:]
            if not pieces or pieces[0] not in {'longcat_video', 'README.md', 'requirements.txt', 'LICENSE'}:
                continue
            target = source.joinpath(*pieces)
            if not target.resolve().is_relative_to(source.resolve()):
                raise ValueError('Source archive escapes target')
            if item.isdir():
                target.mkdir(parents=True, exist_ok=True)
            elif item.isfile():
                target.parent.mkdir(parents=True, exist_ok=True)
                with tf.extractfile(item) as src, target.open('wb') as dst:
                    import shutil
                    shutil.copyfileobj(src, dst)
    archive.unlink()
print('SOURCE_READY', CONFIG['source_revision'], flush=True)

info = HfApi().model_info(CONFIG['model_id'], revision=CONFIG['model_revision'], files_metadata=True)
assert info.sha == CONFIG['model_revision']
files = [x for x in info.siblings if x.rfilename.startswith(('dit/', 'text_encoder/', 'vae/', 'tokenizer/', 'scheduler/'))
         or x.rfilename in {'lora/cfg_step_lora.safetensors', 'config.json', 'model_index.json', 'LICENSE'}]
for entry in files:
    name = entry.rfilename
    target = MODEL / name
    if name in manifest['files'] and target.exists():
        if digest(target) == manifest['files'][name]['bf16_sha256']:
            print('VERIFIED', name, flush=True)
            continue
        raise RuntimeError('Previously converted weight changed: ' + name)
    print('DOWNLOADING', name, entry.size, flush=True)
    downloaded = Path(hf_hub_download(CONFIG['model_id'], name, revision=info.sha, local_dir=INCOMING))
    source_hash = digest(downloaded)
    lfs = entry.lfs
    expected = (lfs.get('sha256') if isinstance(lfs, dict) else getattr(lfs, 'sha256', None)) if lfs else None
    if expected and source_hash != expected:
        raise RuntimeError('Downloaded LFS hash mismatch: ' + name)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + '.tmp')
    if name.endswith('.safetensors'):
        tensors = {}
        with safe_open(downloaded, framework='pt', device='cpu') as sf:
            metadata = sf.metadata()
            for key in sf.keys():
                tensor = sf.get_tensor(key)
                tensors[key] = tensor.to(torch.bfloat16) if tensor.is_floating_point() else tensor.clone()
            del tensor
            save_file(tensors, tmp, metadata=metadata)
        del tensors
        gc.collect()
        tmp.replace(target)
        downloaded.unlink()
    else:
        import shutil
        shutil.copyfile(downloaded, tmp)
        tmp.replace(target)
        downloaded.unlink()
    manifest['files'][name] = {'original_bytes': entry.size, 'original_sha256': source_hash,
                               'bf16_bytes': target.stat().st_size, 'bf16_sha256': digest(target)}
    save_manifest()
    print('STAGED', name, target.stat().st_size, flush=True)

# The index's total_size describes the converted tensor storage, not the F32 download.
for folder in ('dit', 'text_encoder'):
    for path in (MODEL / folder).glob('*.index.json'):
        data = json.loads(path.read_text())
        total = 0
        for shard in set(data['weight_map'].values()):
            import struct
            with (path.parent / shard).open('rb') as f:
                header_len = struct.unpack('<Q', f.read(8))[0]
            total += (path.parent / shard).stat().st_size - 8 - header_len
        data.setdefault('metadata', {})['total_size'] = total
        path.write_text(json.dumps(data, indent=2))
        manifest['files'][str(path.relative_to(MODEL))]['bf16_sha256'] = digest(path)
save_manifest()
print('ALL_WEIGHTS_READY', sum(x['bf16_bytes'] for x in manifest['files'].values()), flush=True)
(ROOT / 'WEIGHTS_READY').write_text(info.sha + '\n')
