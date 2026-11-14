"""Fixed-seed native Hunyuan 480p step-distilled generation; no automatic rerolls."""
import argparse
from datetime import datetime, timezone
from importlib import metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from .dataset import load_canonical
from .io import fingerprint, sha256, write_csv, write_json
from .media import binary, inspect_video

MODEL_ID = 'HunyuanVideo-1.5-480P-I2V-step-distill'
SETTINGS = {'resolution': '480p', 'steps': 8, 'dtype': 'bfloat16', 'fps': 24,
            'offloading': True, 'group_offloading': True, 'overlap_group_offloading': False,
            'module_offloading': False, 'component_init_device': 'cuda', 'text_encoder_load_dtype': 'float16',
            'rewrite': False, 'sr': False, 'cache': False}
FIELDS = ['case_id', 'category', 'model_id', 'phase', 'seed', 'source_frame_num', 'target_frames',
          'status', 'failure_reason', 'weight_version', 'weight_revision', 'upstream_commit',
          'config', 'image_sha256', 'native_video', 'video_sha256', 'native_frames', 'native_width',
          'native_height', 'native_fps', 'native_duration_seconds', 'native_codec', 'elapsed_seconds',
          'worker_wall_seconds', 'gpu_model', 'driver_version', 'peak_vram_allocated_mib',
          'peak_vram_reserved_mib', 'started_at', 'finished_at', 'log_path']


def now():
    return datetime.now(timezone.utc).isoformat()


def create_pipeline(torch, pipeline_class, assets):
    """Keep host-resident weights within TC2's 30G limit without quantization."""
    import hyvideo.models.text_encoders as text_encoders
    original = text_encoders.load_text_encoder

    def load_at_final_precision(*args, **kwargs):
        previous = torch.get_default_dtype()
        try:
            # The official loader casts to fp16 after loading. Loading directly
            # in that same dtype avoids an unnecessary fp32 host allocation.
            torch.set_default_dtype(torch.float16)
            return original(*args, **kwargs)
        finally:
            torch.set_default_dtype(previous)

    text_encoders.load_text_encoder = load_at_final_precision
    try:
        pipe = pipeline_class.create_pipeline(
            pretrained_model_name_or_path=assets, transformer_version='480p_i2v_step_distilled',
            create_sr_pipeline=False, transformer_dtype=torch.bfloat16, device=torch.device('cuda'),
            transformer_init_device=torch.device('cuda'))
    finally:
        text_encoders.load_text_encoder = original
    return pipe


def worker(request_path):
    payload = json.loads(Path(request_path).read_text())
    row = dict(payload['record'])
    wall = time.perf_counter()
    torch = None
    infer_start = None
    try:
        os.environ['HF_HUB_OFFLINE'] = '1'
        os.environ['TRANSFORMERS_OFFLINE'] = '1'
        if sha256(payload['image']) != row['image_sha256']:
            raise ValueError('Shared input changed after freezing')
        import torch
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError('Allocated CUDA device must support BF16')
        row['gpu_model'] = torch.cuda.get_device_name(0)
        row['driver_version'] = subprocess.check_output([binary('nvidia-smi'), '--query-gpu=driver_version', '--format=csv,noheader'], text=True).splitlines()[0]
        sys.path.insert(0, payload['upstream'])
        from hyvideo.pipelines.hunyuan_video_pipeline import HunyuanVideo_1_5_Pipeline
        from hyvideo.commons.parallel_states import initialize_parallel_state
        from hyvideo.commons.infer_state import initialize_infer_state
        from types import SimpleNamespace
        torch.cuda.set_device(0)
        # The official pipeline uses device meshes even with one GPU. A private
        # per-attempt file store avoids relying on torchrun's inherited env/ports.
        torch.distributed.init_process_group('nccl', rank=0, world_size=1,
            init_method=(Path(request_path).parent / '.distributed-store').resolve().as_uri())
        initialize_parallel_state(sp=1)
        infer_state = initialize_infer_state(SimpleNamespace(
            sage_blocks_range='0-53', no_cache_block_id='0', use_sageattn=False,
            enable_torch_compile=False, enable_cache=False, cache_type='deepcache',
            cache_start_step=0, cache_end_step=8, total_steps=8, cache_step_interval=1,
            use_fp8_gemm=False, quant_type='fp8-per-token-sgl', include_patterns='double_blocks'))
        from PIL import Image
        import numpy as np
        import imageio.v2 as imageio
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        infer_start = time.perf_counter()
        pipe = create_pipeline(torch, HunyuanVideo_1_5_Pipeline, payload['assets'])
        pipe.apply_infer_optimization(infer_state=infer_state, enable_offloading=True,
                                      enable_group_offloading=True, overlap_group_offloading=False)
        # Keep the official Transformer group hooks active, but avoid moving the
        # encoders/VAE onto a host already holding the offloaded Transformer.
        pipe.enable_offloading = False
        with Image.open(payload['image']) as picture, torch.inference_mode():
            result = pipe(prompt=payload['prompt'], reference_image=picture.convert('RGB'), aspect_ratio='16:9',
                          video_length=int(row['target_frames']), num_inference_steps=8, enable_sr=False,
                          prompt_rewrite=False, seed=int(row['seed']), output_type='pt')
        video = result.videos
        if video.ndim == 5:
            if video.shape[0] != 1:
                raise ValueError('Expected exactly one generated video')
            video = video[0]
        destination = Path(request_path).parent / 'native.mp4'
        # Encode a frame at a time rather than allocating several full-length
        # float32/uint8 copies of a 513-frame video in host memory.
        with imageio.get_writer(destination, fps=24, macro_block_size=1, codec='libx264',
                                ffmpeg_params=['-threads', '1']) as writer:
            for frame_index in range(video.shape[1]):
                frame = video[:, frame_index].detach().cpu().clamp(0, 1).permute(1, 2, 0).float().numpy()
                writer.append_data((frame * 255).astype(np.uint8))
        torch.cuda.synchronize()
        row['elapsed_seconds'] = time.perf_counter() - infer_start
        info = inspect_video(destination)
        if info['frames'] != int(row['target_frames']) or info['fps'] != 24:
            raise ValueError('Native output frame count/fps differs from the frozen case')
        row.update(status='success', failure_reason='', native_video=str(destination), video_sha256=sha256(destination),
                   native_frames=info['frames'], native_width=info['width'], native_height=info['height'],
                   native_fps=info['fps'], native_duration_seconds=info['duration'], native_codec=info['codec'])
    except BaseException as exc:
        import traceback
        traceback.print_exc()
        row.update(status='interrupted' if isinstance(exc, (KeyboardInterrupt, SystemExit)) else 'technical_failure',
                   failure_reason=f'{type(exc).__name__}: {exc}')
    finally:
        if infer_start is not None and 'elapsed_seconds' not in row:
            row['elapsed_seconds'] = time.perf_counter() - infer_start
        if torch is not None and torch.cuda.is_available():
            try:
                row.update(peak_vram_allocated_mib=torch.cuda.max_memory_allocated() / 1024**2,
                           peak_vram_reserved_mib=torch.cuda.max_memory_reserved() / 1024**2)
            except RuntimeError:
                pass
        if torch is not None and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
        row.update(worker_wall_seconds=time.perf_counter() - wall, finished_at=now())
        write_json(Path(request_path).parent / 'result.json', row)
    return 0 if row['status'] == 'success' else 1


def generate(cases, upstream, commit, assets, output, phase='formal', pilot_receipt=None, max_new_cases=24,
             skip_pilot_reason=None, deadline=None):
    if skip_pilot_reason is not None and (phase != 'formal' or pilot_receipt is not None or not skip_pilot_reason.strip()):
        raise ValueError('Pilot skip requires an explicit reason, formal phase and no pilot receipt')
    rows, inputs = load_canonical(cases, decode=True, formal=phase == 'formal')
    if phase == 'pilot' and (len(rows) != 3 or {int(r['target_frames']) for r in rows} != {129, 513}):
        raise ValueError('Pilot requires 3 independent cases, including short and long')
    upstream, assets, output = [Path(p).resolve() for p in (upstream, assets, output)]
    if subprocess.check_output(['git', '-C', str(upstream), 'rev-parse', 'HEAD'], text=True).strip() != commit or subprocess.check_output(
            ['git', '-C', str(upstream), 'status', '--porcelain', '--untracked-files=no'], text=True).strip():
        raise ValueError('Use the clean, pinned official Hunyuan checkout')
    manifest = json.loads((assets / 'assets.json').read_text())
    import re
    if not manifest.get('repositories') or any(not re.fullmatch(r'[0-9a-f]{40}', v) for v in manifest['repositories'].values()):
        raise ValueError('assets.json must contain fixed repository revision SHAs')
    files = manifest.get('files', {})
    actual_files = {str(p.relative_to(assets)) for p in assets.rglob('*') if p.is_file()
                    and p.name != 'assets.json' and not any(s.startswith('.') for s in p.relative_to(assets).parts)}
    if set(files) != actual_files or not files or any(sha256(assets / name) != value for name, value in files.items()):
        raise ValueError('Prepared Hunyuan assets differ from their complete SHA256 receipt')
    for required in ('transformer/480p_i2v_step_distilled', 'vae', 'scheduler', 'text_encoder/llm',
                     'text_encoder/byt5-small', 'text_encoder/Glyph-SDXL-v2', 'vision_encoder/siglip'):
        if not (assets / required).is_dir():
            raise ValueError(f'Missing local model component: {required}')
    environment = {}
    for name in ('torch', 'transformers', 'diffusers', 'imageio', 'qwen-vl-utils'):
        try:
            environment[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            environment[name] = 'not installed'
    protocol = {'settings': SETTINGS, 'upstream_commit': commit, 'assets': manifest, 'environment': environment,
                'source_sha256': sha256(__file__), 'dataset_sha256': sha256(Path(__file__).with_name('dataset.py'))}
    if phase == 'formal' and skip_pilot_reason is None:
        if pilot_receipt is None:
            raise ValueError('Formal generation requires a completed independent short/long --pilot-receipt')
        pilot = json.loads(Path(pilot_receipt).read_text())
        if pilot.get('protocol_sha256') != fingerprint(protocol) or pilot.get('success') != 3:
            raise ValueError('Pilot did not pass with this exact configuration/environment')
        if set(pilot['image_sha256']) & {v['sha256'] for v in inputs['images'].values()}:
            raise ValueError('Pilot and formal images must be independent')
    frozen = {'protocol': protocol, 'inputs': inputs, 'cases': rows, 'phase': phase}
    if skip_pilot_reason is not None:
        frozen['pilot_validation'] = {'status': 'skipped_by_user', 'reason': skip_pilot_reason}
    output.mkdir(parents=True, exist_ok=True)
    import fcntl
    with (output / '.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        batch = output / 'batch.json'
        if batch.exists():
            if json.loads(batch.read_text()) != frozen:
                raise ValueError('Frozen Hunyuan batch changed; do not mix configurations or reroll')
        else:
            if any(p.name != '.lock' for p in output.iterdir()):
                raise ValueError('Prior artifacts without batch.json')
            write_json(batch, frozen)
            write_json(output / 'selection.json', {'case_ids': [r['case_id'] for r in rows]})
        base = [{**r, 'model_id': MODEL_ID, 'phase': phase, 'status': 'not_started',
                 'failure_reason': 'Not started in selected batch', 'weight_version': MODEL_ID,
                 'weight_revision': json.dumps(manifest['repositories'], sort_keys=True), 'upstream_commit': commit,
                 'config': str(batch), 'image_sha256': inputs['images'][r['case_id']]['sha256'],
                 'log_path': str(output / r['case_id'] / 'inference.log')} for r in rows]
        def export():
            records = []
            for item in base:
                directory = output / item['case_id']
                if (directory / 'result.json').exists():
                    records.append(json.loads((directory / 'result.json').read_text()))
                elif directory.exists():
                    records.append({**item, 'status': 'incomplete', 'failure_reason': 'Interrupted attempt; no automatic reroll'})
                else:
                    records.append(item)
            temporary = output / 'runs.csv.new'
            write_csv(temporary, FIELDS, records)
            temporary.replace(output / 'runs.csv')
            return records
        prior = export()
        if phase == 'formal' and any(r['status'] not in ('success', 'not_started') for r in prior):
            print('Retained formal batch has a failed/incomplete attempt; inspect before starting more cases', flush=True)
            return 1
        count = 0
        budget_stop = False
        try:
            for case, item in zip(rows, base):
                directory = output / case['case_id']
                if directory.exists():
                    result = directory / 'result.json'
                    if result.exists():
                        old = json.loads(result.read_text())
                        if old['status'] == 'success' and sha256(old['native_video']) != old['video_sha256']:
                            raise ValueError('Successful native output changed; refusing regeneration')
                    continue
                if count >= max_new_cases:
                    break
                if deadline is not None:
                    current = export()
                    profile = int(case['target_frames'])
                    observed = [float(r.get('worker_wall_seconds') or 0) for r in current
                                if r['status'] == 'success' and int(r['target_frames']) == profile]
                    estimate = max(900 if profile == 129 else 3600, max(observed, default=0) * 1.5)
                    if profile == 513 and not observed:
                        shorts = [float(r.get('worker_wall_seconds') or 0) for r in current if r['status'] == 'success']
                        estimate = max(estimate, max(shorts, default=0) * 3)
                    if time.time() + estimate + 600 >= deadline:
                        budget_stop = True
                        print('Stopping between cases before the SLURM deadline; retain successful outputs and continue unstarted cases', flush=True)
                        break
                directory.mkdir()
                record = {**item, 'status': 'running', 'failure_reason': '', 'started_at': now()}
                request = directory / 'request.json'
                write_json(request, {'record': record, 'image': str((Path(cases).resolve().parent / case['image_path']).resolve()),
                                     'prompt': case['prompt'], 'upstream': str(upstream), 'assets': str(assets)})
                export()
                with (directory / 'inference.log').open('x') as log:
                    subprocess.call([sys.executable, '-m', 'ai6132_benchmark.hunyuan', '--worker', str(request)],
                                    stdout=log, stderr=subprocess.STDOUT)
                count += 1
                current = export()
                if phase == 'formal' and any(r['status'] not in ('success', 'not_started') for r in current):
                    print('Formal generation stopped after the first technical failure; remaining cases are retained as not_started', flush=True)
                    break
        finally:
            results = export()
        success = sum(r['status'] == 'success' for r in results)
        if budget_stop:
            write_json(output / f"continuation_{os.environ.get('SLURM_JOB_ID', 'manual')}.json",
                       {'status': 'continue_unstarted', 'success': success, 'selected': len(rows),
                        'deadline': deadline, 'runs_sha256': sha256(output / 'runs.csv')})
            print(json.dumps({'selected': len(rows), 'success': success, 'new_attempts': count,
                              'continuation_required': True}), flush=True)
            return 0
        if success == len(rows) and phase == 'formal' and not (output / 'DONE.json').exists():
            write_json(output / 'DONE.json', {'success': success, 'runs_sha256': sha256(output / 'runs.csv'),
                                            'batch_sha256': sha256(output / 'batch.json')})
        if success == len(rows) and phase == 'pilot' and not (output / 'pilot_pass.json').exists():
            write_json(output / 'pilot_pass.json', {'protocol_sha256': fingerprint(protocol), 'success': 3,
                       'image_sha256': [r['image_sha256'] for r in base]})
        print(json.dumps({'selected': len(rows), 'success': success, 'new_attempts': count}))
        return 0 if success == len(rows) else 1


def main():
    if len(sys.argv) == 3 and sys.argv[1] == '--worker':
        raise SystemExit(worker(sys.argv[2]))
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('cases', 'upstream', 'assets', 'out'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--commit', required=True)
    p.add_argument('--phase', choices=['formal', 'pilot'], default='formal')
    p.add_argument('--pilot-receipt', type=Path)
    p.add_argument('--skip-pilot-reason', help='Record an explicit user instruction to start formal generation without pilot validation')
    p.add_argument('--max-new-cases', type=int, default=24)
    args = p.parse_args()
    if args.max_new_cases < 1:
        p.error('--max-new-cases must be positive')
    raise SystemExit(generate(args.cases, args.upstream, args.commit, args.assets, args.out,
                              args.phase, args.pilot_receipt, args.max_new_cases, args.skip_pilot_reason,
                              float(os.environ['SLURM_JOB_END_TIME']) if os.environ.get('SLURM_JOB_END_TIME') else None))


if __name__ == '__main__':
    main()
