"""Single-GPU TI2V worker. No optional speech, animation or prompt extension."""
import argparse
from importlib import metadata
import json
import os
from pathlib import Path
import shutil
import subprocess


def encode_native(video, destination, fps, torch):
    """Write native pixels; encoder failures propagate to the parent receipt."""
    executable = shutil.which(os.environ.get('AI6132_FFMPEG', 'ffmpeg'))
    if not executable:
        raise RuntimeError('FFmpeg is required')
    _, frames, height, width = video.shape
    process = subprocess.Popen([executable, '-v', 'error', '-nostdin', '-n', '-f', 'rawvideo',
        '-pix_fmt', 'rgb24', '-s', f'{width}x{height}', '-r', str(fps), '-i', 'pipe:0', '-an',
        '-c:v', 'libx264', '-threads', '1', '-crf', '18', '-preset', 'medium', '-pix_fmt', 'yuv420p',
        str(destination)], stdin=subprocess.PIPE)
    try:
        for index in range(frames):
            pixels = ((video[:, index].float().clamp(-1, 1)+1)*127.5).to(torch.uint8)
            process.stdin.write(pixels.permute(1, 2, 0).contiguous().cpu().numpy().tobytes())
        process.stdin.close()
        if process.wait() != 0:
            raise RuntimeError('FFmpeg native writer failed')
    except BaseException:
        process.kill()
        process.wait()
        raise


def main():
    if not os.environ.get('SLURM_JOB_ID'):
        raise SystemExit('Submit through sbatch')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    cfg = json.loads(args.request.read_text(encoding='utf-8'))
    # Import GPU dependencies only inside the sbatch worker.
    import torch
    from PIL import Image
    from wan import WanTI2V
    from wan.configs import WAN_CONFIGS
    from wan.modules.attention import FLASH_ATTN_2_AVAILABLE, FLASH_ATTN_3_AVAILABLE
    from benchmark import atomic_json
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError('Exactly one scheduler-assigned GPU is required')
    if not (FLASH_ATTN_2_AVAILABLE or FLASH_ATTN_3_AVAILABLE):
        raise RuntimeError('TI2V model directly calls FlashAttention; SDPA fallback is not sufficient')
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError('This profile requires BF16')
    frames = int(cfg['target_frames'])
    if frames not in (129, 513) or cfg['sampling_steps'] not in (10, 50) or cfg['prompt_extension']:
        raise ValueError('Unexpected benchmark request')
    torch.cuda.set_device(0)
    torch.cuda.reset_peak_memory_stats()
    pipeline = WanTI2V(WAN_CONFIGS['ti2v-5B'], checkpoint_dir=cfg['assets']['root'], device_id=0,
                       rank=0, t5_fsdp=False, dit_fsdp=False, use_sp=False,
                       t5_cpu=cfg['t5_cpu'], convert_model_dtype=cfg['convert_model_dtype'])
    with Image.open(cfg['image_path']) as source:
        image = source.convert('RGB')
    video = pipeline.generate(cfg['prompt'], img=image, size=tuple(cfg['size']), max_area=cfg['max_area'],
                              frame_num=frames, shift=cfg['shift'], sample_solver=cfg['sample_solver'],
                              sampling_steps=cfg['sampling_steps'], guide_scale=cfg['guide_scale'],
                              seed=int(cfg['seed']), offload_model=cfg['offload_model'])
    if video.ndim != 4 or video.shape[:2] != (3, frames) or not torch.isfinite(video).all().item():
        raise ValueError('Invalid native tensor: expected finite [3, target_frames, height, width]')
    encode_native(video, args.out/'native.mp4', cfg['fps'], torch)
    versions = {}
    for name in ('torch', 'torchvision', 'transformers', 'diffusers', 'flash-attn', 'numpy', 'Pillow'):
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = 'not installed'
    atomic_json(args.out/'environment.json', {'packages': versions, 'cuda': torch.version.cuda,
                                            'gpu': torch.cuda.get_device_name(0)})
    atomic_json(args.out/'worker_result.json', {'gpu_model': torch.cuda.get_device_name(0),
        'peak_vram_allocated_mib': torch.cuda.max_memory_allocated()/2**20,
        'peak_vram_reserved_mib': torch.cuda.max_memory_reserved()/2**20})


if __name__ == '__main__':
    main()
