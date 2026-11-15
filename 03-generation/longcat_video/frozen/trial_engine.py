"""One LongCat I2V generation, with staged loading and auditable timing."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path(os.environ['LONGCAT_ROOT'])
CFG = json.loads((ROOT / 'code/config.json').read_text())
OUT = ROOT / 'outputs' / CFG['trial_id']
MODEL = ROOT / 'model'
OUT.mkdir(parents=True, exist_ok=True)


def utc():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2), encoding='utf-8')
    tmp.replace(path)


def file_hash(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def imports():
    import torch
    from longcat_video.pipeline_longcat_video import LongCatVideoPipeline
    from longcat_video.modules.autoencoder_kl_wan import AutoencoderKLWan
    return torch, LongCatVideoPipeline, AutoencoderKLWan


def encode():
    torch, Pipeline, _ = imports()
    from transformers import AutoTokenizer, UMT5EncoderModel
    torch.set_num_threads(4)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, subfolder='tokenizer', local_files_only=True)
    encoder = UMT5EncoderModel.from_pretrained(MODEL, subfolder='text_encoder',
        torch_dtype=torch.bfloat16, device_map={'': 0}, local_files_only=True).eval()
    pipe = Pipeline(tokenizer, encoder, None, None, None)
    with torch.inference_mode():
        embeds, mask = pipe._get_t5_prompt_embeds(CFG['prompt'], device='cuda', dtype=torch.bfloat16)
    torch.save({'embeds': embeds.cpu(), 'mask': mask.cpu()}, OUT / 'prompt.pt')
    print('PROMPT_ENCODED', tuple(embeds.shape), flush=True)


def denoise():
    torch, Pipeline, VAE = imports()
    import torch.distributed as dist
    import types
    from PIL import Image
    from longcat_video.context_parallel.context_parallel_util import init_context_parallel
    from longcat_video.modules.longcat_video_dit import LongCatVideoTransformer3DModel
    from longcat_video.modules.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
    torch.set_num_threads(4)
    torch.cuda.set_device(0)
    dist.init_process_group('nccl', init_method=(OUT / 'distributed-init').resolve().as_uri(), rank=0, world_size=1)
    init_context_parallel(1, 0, 1)
    try:
        dit = LongCatVideoTransformer3DModel.from_pretrained(MODEL, subfolder='dit',
            torch_dtype=torch.bfloat16, device_map={'': 0}, low_cpu_mem_usage=True,
            cp_split_hw=[1, 1], enable_flashattn2=True, enable_flashattn3=False,
            enable_xformers=False, enable_bsa=False, local_files_only=True).eval()
        dit.load_lora(str(MODEL / 'lora/cfg_step_lora.safetensors'), 'cfg_step_lora')
        dit.enable_loras(['cfg_step_lora'])
        # All 48 blocks use the same fixed RoPE frequencies. Reuse the identical
        # CPU cache instead of retaining 48 copies (~4.6 GiB for 513 frames).
        first_rope = dit.blocks[0].attn.rope_3d
        for block in dit.blocks:
            rope = block.attn.rope_3d
            assert (rope.head_dim, rope.base, rope.cp_split_hw) == (first_rope.head_dim, first_rope.base, first_rope.cp_split_hw)
            rope.freqs_dict = first_rope.freqs_dict
        # Move whole transformer blocks synchronously. LoRA adapters stay on GPU.
        # No weight quantization, token dropping, sparse attention, or continuation.
        count = min(CFG['cpu_offload_block_count'], len(dit.blocks))
        hooks = []
        def load_block(module, args):
            module.to('cuda')

        def unload_block(module, args, result):
            module.to('cpu')

        for block in list(dit.blocks)[-count:]:
            block.to('cpu')
            hooks.append(block.register_forward_pre_hook(load_block))
            hooks.append(block.register_forward_hook(unload_block))
        torch.cuda.empty_cache()
        vae = VAE.from_pretrained(MODEL, subfolder='vae', torch_dtype=torch.bfloat16,
                                 local_files_only=True).to('cuda').eval()
        scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(MODEL, subfolder='scheduler', local_files_only=True)
        pipe = Pipeline(None, None, vae, scheduler, dit)
        cache = torch.load(OUT / 'prompt.pt', map_location='cuda', weights_only=True)

        def cached_prompt(self, **kwargs):
            assert kwargs['prompt'] == CFG['prompt']
            assert not kwargs['do_classifier_free_guidance']
            return cache['embeds'], cache['mask'], None, None

        pipe.encode_prompt = types.MethodType(cached_prompt, pipe)
        original_forward = dit.forward
        step = 0
        generation_start = time.perf_counter()

        def timed_forward(*args, **kwargs):
            nonlocal step
            torch.cuda.synchronize()
            started = time.perf_counter()
            result = original_forward(*args, **kwargs)
            torch.cuda.synchronize()
            step += 1
            record = {'utc': utc(), 'step': step, 'total_steps': CFG['num_inference_steps'],
                      'seconds': time.perf_counter() - started,
                      'elapsed_seconds': time.perf_counter() - generation_start,
                      'peak_gpu_gib': torch.cuda.max_memory_allocated() / 2**30}
            with (OUT / 'steps.jsonl').open('a') as f:
                f.write(json.dumps(record) + '\n')
            print('STEP', json.dumps(record), flush=True)
            return result

        dit.forward = timed_forward
        image_path = Path.home() / 'ai6132-cogvideox' / CFG['image_relative_path']
        assert file_hash(image_path) == CFG['image_sha256']
        image = Image.open(image_path).convert('RGB')
        height, width = pipe.get_condition_shape(image, CFG['resolution'], scale_factor_spatial=16)
        write_json(OUT / 'generation-shape.json', {'width': width, 'height': height,
            'frames': CFG['num_frames'], 'offloaded_blocks': count, 'total_blocks': len(dit.blocks),
            'gpu': torch.cuda.get_device_name(0), 'started_utc': utc()})
        with torch.inference_mode():
            latents = pipe.generate_i2v(image=image, prompt=CFG['prompt'], resolution=CFG['resolution'],
                num_frames=CFG['num_frames'], num_inference_steps=CFG['num_inference_steps'],
                use_distill=True, guidance_scale=1.0, output_type='latent',
                generator=torch.Generator(device='cuda').manual_seed(CFG['seed']))
        assert step == CFG['num_inference_steps']
        torch.save(latents.cpu(), OUT / 'latents.pt')
        write_json(OUT / 'denoise-result.json', {'seconds': time.perf_counter() - generation_start,
            'completed_steps': step, 'latent_shape': list(latents.shape),
            'peak_gpu_gib': torch.cuda.max_memory_allocated() / 2**30})
    finally:
        dist.destroy_process_group()


def decode():
    torch, Pipeline, VAE = imports()
    import imageio_ffmpeg
    torch.set_num_threads(4)
    vae = VAE.from_pretrained(MODEL, subfolder='vae', torch_dtype=torch.bfloat16,
                             local_files_only=True).to('cuda').eval()
    pipe = Pipeline(None, None, vae, None, None)
    latents = torch.load(OUT / 'latents.pt', map_location='cuda', weights_only=True).to(torch.bfloat16)
    with torch.inference_mode():
        latents = pipe.denormalize_latents(latents)
        video = vae.decode(latents, return_dict=False)[0]
    assert video.shape[0] == 1 and video.shape[2] == CFG['num_frames'], tuple(video.shape)
    _, _, frames, height, width = video.shape
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    proc = subprocess.Popen([ffmpeg, '-nostdin', '-hide_banner', '-loglevel', 'error', '-n',
        '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{width}x{height}', '-r', str(CFG['native_fps']),
        '-i', 'pipe:0', '-an', '-c:v', 'libx264', '-crf', '18', '-preset', 'medium',
        '-pix_fmt', 'yuv420p', str(OUT / 'native.mp4')], stdin=subprocess.PIPE)
    try:
        for i in range(frames):
            frame = ((video[0, :, i].float() / 2 + 0.5).clamp(0, 1) * 255).to(torch.uint8)
            proc.stdin.write(frame.permute(1, 2, 0).contiguous().cpu().numpy().tobytes())
        proc.stdin.close()
        if proc.wait() != 0:
            raise RuntimeError('Native video encoder failed')
    except BaseException:
        proc.kill()
        proc.wait()
        raise
    subprocess.run([ffmpeg, '-nostdin', '-hide_banner', '-loglevel', 'error', '-n',
        '-i', str(OUT / 'native.mp4'), '-vf', 'scale=768:432:flags=lanczos,fps=12',
        '-an', '-c:v', 'libx264', '-crf', '18', '-preset', 'medium', '-pix_fmt', 'yuv420p',
        str(OUT / 'evaluation.mp4')], check=True)
    print('DECODED', frames, width, height, flush=True)


def probe_video(path):
    import av
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        metadata = {'width': stream.width, 'height': stream.height,
                    'fps': float(stream.average_rate), 'frames': sum(1 for _ in container.decode(stream))}
    metadata['duration_seconds'] = metadata['frames'] / metadata['fps']
    metadata['sha256'] = file_hash(path)
    return metadata


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--stage', choices=['encode', 'denoise', 'decode'])
    args = p.parse_args()
    if args.stage:
        return globals()[args.stage]()
    run_path = OUT / 'run.json'
    if run_path.exists():
        raise RuntimeError('Attempt already exists; never overwrite a trial')
    started = time.perf_counter()
    state = dict(CFG, status='running', started_utc=utc(), slurm_job_id=os.environ.get('SLURM_JOB_ID'), stages={})
    write_json(run_path, state)
    try:
        if not (ROOT / 'WEIGHTS_READY').exists() or not (ROOT / 'ENV_READY').exists():
            raise RuntimeError('Preparation did not finish')
        cog = Path.home() / 'ai6132-cogvideox'
        import shutil
        image_path = cog / CFG['image_relative_path']
        assert file_hash(image_path) == CFG['image_sha256']
        shutil.copy2(image_path, OUT / 'input.jpg')
        write_json(OUT / 'request.json', CFG)
        batch = cog / 'batches/formal30_20261114_overnight_v2'
        subprocess.run([str(cog / '.venv/bin/python'), str(batch / 'code/run_formal.py'),
                        '--batch', str(batch), '--collect'], check=True)
        for stage in ('encode', 'denoise', 'decode'):
            state['stage'] = stage
            write_json(run_path, state)
            before = time.perf_counter()
            with (OUT / (stage + '.log')).open('xb') as log:
                subprocess.run([sys.executable, '-u', __file__, '--stage', stage], stdout=log, stderr=subprocess.STDOUT, check=True)
            state['stages'][stage] = {'elapsed_seconds': time.perf_counter() - before}
        native = probe_video(OUT / 'native.mp4')
        evaluation = probe_video(OUT / 'evaluation.mp4')
        assert native['frames'] == CFG['num_frames']
        assert native['fps'] == CFG['native_fps']
        assert (evaluation['width'], evaluation['height'], evaluation['fps']) == (768, 432, 12)
        assert abs(evaluation['duration_seconds'] - native['duration_seconds']) <= 1/12 + 0.001
        state.update(status='succeeded', native=native, evaluation=evaluation)
    except BaseException as e:
        state.update(status='failed', error=repr(e), traceback=traceback.format_exc())
        raise
    finally:
        state.update(ended_utc=utc(), elapsed_seconds=time.perf_counter() - started)
        write_json(run_path, state)
        print(json.dumps(state, indent=2), flush=True)


if __name__ == '__main__':
    main()
