"""Synthetic protocol/integration fixtures, never experimental video ratings."""
import csv
import contextlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import types
import unittest
from urllib.request import Request
from unittest.mock import patch

from PIL import Image
from ai6132_benchmark import auth, dataset, media, qwen, vbench, hunyuan
from ai6132_benchmark.io import human_module, read_csv, sha256, write_csv, write_json


class BenchmarkTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.source = self.root / 'cases.csv'
        self.canonical = self.root / 'cases.canonical.csv'
        categories = []
        for category in sorted(dataset.COUNTS):
            categories.extend([category] * dataset.COUNTS[category])
        self.source_rows = []
        for i, category in enumerate(categories, 1):
            name = f'{i:02d}_image.jpg'
            Image.new('RGB', (64, 36), (i * 7, 30, 100)).save(self.root / name)
            self.source_rows.append({'image_type': category, 'image_path': name,
                                     'prompt': f'  Original, prompt {i}\nsecond line  ', 'seed': '42',
                                     'frame_num': '512' if i <= 6 else '128'})
        write_csv(self.source, list(self.source_rows[0]), self.source_rows)

    def adapt(self):
        return dataset.canonicalize(self.source, self.canonical)

    def test_private_credential_file_permissions_and_symlinks(self):
        credential = self.root / 'private-credential'
        fixture = 'hf_' + 'synthetic_fixture_only'
        credential.write_text(fixture)
        credential.chmod(0o600)
        self.assertEqual(auth.read_hf_token(credential), fixture)
        credential.chmod(0o644)
        with self.assertRaisesRegex(ValueError, 'mode 600'):
            auth.read_hf_token(credential)
        credential.chmod(0o600)
        link = self.root / 'credential-link'
        link.symlink_to(credential)
        with self.assertRaisesRegex(ValueError, 'regular file'):
            auth.read_hf_token(link)

    def test_credential_is_removed_from_external_redirects(self):
        request = Request('https://huggingface.co/example', headers={'Authorization': 'Bearer fixture'})
        handler = auth.SafeRedirect()
        redirected = handler.redirect_request(request, None, 302, 'Found', {}, 'https://cdn.example/file')
        self.assertIsNone(redirected.get_header('Authorization'))
        same_host = handler.redirect_request(request, None, 302, 'Found', {}, 'https://huggingface.co/file')
        self.assertEqual(same_host.get_header('Authorization'), 'Bearer fixture')
        with self.assertRaisesRegex(ValueError, 'HTTPS'):
            handler.redirect_request(request, None, 302, 'Found', {}, 'http://cdn.example/file')

    def make_video(self, frames=129, fps=24):
        path = self.root / f'video_{frames}_{fps}.mp4'
        if not path.exists():
            subprocess.run([media.binary('ffmpeg'), '-v', 'error', '-f', 'lavfi', '-i', f'color=c=red:s=64x36:r={fps}',
                            '-frames:v', str(frames), '-c:v', 'libx264', '-threads', '1', '-pix_fmt', 'yuv420p', str(path)], check=True)
        return path

    def packet(self, frames=513):
        (self.root / 'images').mkdir()
        (self.root / 'videos').mkdir()
        Image.new('RGB', (64, 36)).save(self.root / 'images/clip_0001.png')
        media.normalize(self.make_video(frames), self.root / 'videos/clip_0001.mp4')
        manifest = self.root / 'anonymous.csv'
        mapping = self.root / 'map.csv'
        write_csv(manifest, media.PUBLIC_FIELDS, [dict(zip(media.PUBLIC_FIELDS,
                  ['clip_0001', 'videos/clip_0001.mp4', 'images/clip_0001.png', 'Original prompt', 'success', '']))])
        link = dict(clip_id='clip_0001', case_id='case_0001', model_id='model', category='single-human', target_frames=frames)
        write_csv(mapping, list(link), [link])
        b = self.root / 'B.csv'
        h = human_module()
        write_csv(b, h.RATING_FIELDS, [dict(rater_id='B', clip_id='clip_0001', reference_fidelity=3,
                                           motion_naturalness=3, prompt_adherence=3, comment='Synthetic fixture only')])
        return manifest, mapping, b

    def test_canonicalization_preserves_source_prompt_seed_order_and_receipt(self):
        before = self.source.read_bytes()
        receipt = self.adapt()
        rows, actual = dataset.load_canonical(self.canonical, decode=True)
        self.assertEqual(before, self.source.read_bytes())
        self.assertEqual(len(rows), 24)
        self.assertEqual(len({r['category'] for r in rows}), 11)
        self.assertEqual([r['target_frames'] for r in rows], ['513'] * 6 + ['129'] * 18)
        self.assertEqual(rows[0]['prompt'], self.source_rows[0]['prompt'])
        self.assertTrue(all(r['seed'] == '42' for r in rows))
        self.assertEqual(actual['images'], receipt['images'])
        with self.assertRaises(ValueError):
            self.adapt()

    def test_changed_image_or_canonical_csv_is_rejected_by_receipt(self):
        self.adapt()
        image = self.root / self.source_rows[0]['image_path']
        Image.new('RGB', (64, 36), 'green').save(image)
        with self.assertRaisesRegex(ValueError, 'frozen receipt'):
            dataset.load_canonical(self.canonical)

    def test_invalid_source_or_non_widescreen_input_does_not_publish_manifest(self):
        Image.new('RGB', (32, 32)).save(self.root / self.source_rows[0]['image_path'])
        with self.assertRaisesRegex(ValueError, '16:9'):
            self.adapt()
        self.assertFalse(self.canonical.exists())

    def test_full_video_conversion_preserves_both_lengths(self):
        for frames in (129, 513):
            with self.subTest(frames=frames):
                before, after = media.normalize(self.make_video(frames), self.root / f'score_{frames}.mp4')
                self.assertEqual(before['frames'], frames)
                self.assertLessEqual(abs(after['duration'] - before['duration']), 1 / 12 + 1e-6)
                self.assertEqual((after['width'], after['height'], after['fps']), (768, 432, 12))
                if frames == 513:
                    self.assertGreater(after['duration'], 21)

    def test_handoff_retains_96_items_and_missing_deliveries(self):
        self.adapt()
        rows, _ = dataset.load_canonical(self.canonical)
        runs = {}
        for i in range(4):
            model = f'model{i}'
            runs[model] = self.root / f'{model}.csv'
            records = []
            if i == 0:
                video = self.make_video(513)
                records = [dict(case_id=rows[0]['case_id'], model_id=model, seed=42, status='success',
                                weight_version='fixture', config='fixture-config', native_video=str(video), video_sha256=sha256(video))]
            write_csv(runs[model], ['case_id', 'model_id', 'seed', 'status', 'weight_version', 'config', 'native_video', 'video_sha256'], records)
        output = self.root / 'handoff-batch'
        self.assertEqual(media.prepare(self.canonical, runs, output, True), {'items': 96, 'success': 1})
        h = human_module()
        anonymous = h.load_manifest(output / 'handoff/anonymous_clips.csv', check_files=True)
        self.assertEqual(len(anonymous), 96)
        self.assertEqual(sum(r['status'] == 'generation_failed' for r in anonymous), 95)
        packet = self.root / 'review'
        receipt = h.prepare(output / 'handoff/anonymous_clips.csv', packet, ['A', 'B', 'C', 'D'], 6132, 96, 'formal')
        self.assertTrue(h.verify_packet(receipt, packet)['valid'])
        self.assertNotIn('case_id', (output / 'handoff/anonymous_clips.csv').read_text())
        self.assertTrue((output / 'COMPLETE.json').exists())
        with self.assertRaises(ValueError):
            media.prepare(self.canonical, runs, output, True)

    def test_qwen_sampling_keeps_full_span_and_absolute_window_times(self):
        info = {'duration': 22, 'fps': 12, 'frames': 264, 'timestamps': [i / 12 for i in range(264)]}
        tasks = qwen.requests_for({}, 513)
        self.assertEqual(len(tasks), 5)
        whole = qwen.sample_request(info, tasks[0])
        self.assertEqual(len(set(whole['indices'])), 128)
        self.assertEqual((whole['indices'][0], whole['indices'][-1]), (0, 263))
        seen = []
        for task in tasks[1:]:
            sample = qwen.sample_request(info, task)
            self.assertEqual(len(sample['indices']), 32)
            self.assertTrue(all(sample['start_seconds'] <= t < sample['end_seconds'] for t in sample['timestamps']))
            seen.extend(sample['indices'])
        self.assertEqual(len(seen), len(set(seen)))
        self.assertEqual(len(qwen.requests_for({}, 129)), 1)

    def test_qwen_parser_rejects_bool_range_and_missing_evidence(self):
        scores = {key: 3 for key in qwen.MAIN_SCORES}
        scores['evidence'] = {key: 'Visible evidence' for key in qwen.MAIN_SCORES}
        self.assertEqual(qwen.parse_reply(json.dumps(scores), 'overall')['prompt_adherence'], 3)
        for value in (True, 6, '3'):
            with self.subTest(value=value), self.assertRaises(ValueError):
                qwen.parse_reply(json.dumps({**scores, 'reference_fidelity': value}), 'overall')
        with self.assertRaises(ValueError):
            qwen.parse_reply(json.dumps({**scores, 'evidence': {}}), 'overall')

    def test_qwen_one_repair_per_request_and_resume_does_not_reroll(self):
        manifest, mapping, b = self.packet()
        model = self.root / 'model'
        model.mkdir()
        (model / 'config.json').write_text('{}')
        write_json(model / 'assets.json', {'repositories': {'Qwen/Qwen3-VL-8B-Instruct': '0' * 40},
                                         'files': {'config.json': sha256(model / 'config.json')}})
        calls = []
        def backend(image, video, prompt, request, previous):
            calls.append((request['request_id'], previous))
            if previous is None:
                return 'invalid JSON fixture', {'fixture': True}
            keys = qwen.MAIN_SCORES if request['kind'] == 'overall' else qwen.WINDOW_SCORES
            result = {k: 3 for k in keys}
            result['evidence'] = {k: 'Synthetic evidence, not a real grade' for k in keys}
            return json.dumps(result), {'fixture': True}
        out = self.root / 'qwen'
        self.assertEqual(qwen.run(manifest, mapping, out, model, '0' * 40, b, 'pilot', backend), 0)
        self.assertEqual(len(calls), 10)
        self.assertEqual(len(read_csv(out / 'qwen_ratings.csv')[1]), 1)
        windows = read_csv(out / 'qwen_windows.csv')[1]
        self.assertEqual(len(windows), 4)
        self.assertTrue(all(r['prompt_adherence'] == '' for r in windows))
        self.assertEqual(qwen.run(manifest, mapping, out, model, '0' * 40, b, 'pilot', backend), 0)
        self.assertEqual(len(calls), 10)
        self.assertTrue((out / 'clip_0001/overall/reply_1.json').exists())

    def test_qwen_operational_failure_retains_future_requests(self):
        manifest, mapping, b = self.packet()
        model = self.root / 'model'
        model.mkdir()
        (model / 'config.json').write_text('{}')
        write_json(model / 'assets.json', {'repositories': {'Qwen/Qwen3-VL-8B-Instruct': '0' * 40},
                                         'files': {'config.json': sha256(model / 'config.json')}})
        def fail(*_args):
            raise RuntimeError('Synthetic GPU failure')
        out = self.root / 'qwen'
        with self.assertRaisesRegex(RuntimeError, 'Synthetic GPU failure'):
            qwen.run(manifest, mapping, out, model, '0' * 40, b, 'pilot', fail)
        self.assertEqual(read_csv(out / 'qwen_ratings.csv')[1][0]['status'], 'technical_failure')
        self.assertEqual([r['status'] for r in read_csv(out / 'qwen_windows.csv')[1]], ['not_started'] * 4)

    def test_vbench_category_mapping_and_per_video_result_validation(self):
        self.assertEqual(len(vbench.dimensions('transportation')), 3)
        self.assertEqual(vbench.dimensions('abstract'), ['i2v_background'])
        path = self.root / 'clip_0001-0.mp4'
        raw = {'dynamic_degree': [0.5, [{'video_path': str(path), 'video_results': True}]]}
        self.assertEqual(vbench.extract_result(raw, 'dynamic_degree', path), 1)
        with self.assertRaises(ValueError):
            vbench.extract_result(raw, 'dynamic_degree', self.root / 'other.mp4')

    def test_vbench_custom_staging_and_resume(self):
        manifest, mapping, _b = self.packet(129)
        upstream, cache = self.root / 'upstream', self.root / 'cache'
        upstream.mkdir()
        cache.mkdir()
        (cache / 'fixture_weights').write_bytes(b'fixture')
        receipt = self.root / 'cache.json'
        write_json(receipt, vbench.cache_files(cache))
        calls = []
        def backend(video, image, dimension, directory):
            self.assertEqual(video.name, 'clip_0001-0.mp4')
            self.assertEqual(image.name, 'clip_0001.png')
            self.assertTrue(video.is_file() and image.is_file())
            calls.append(dimension)
            return {dimension: [0.5, [{'video_path': str(video), 'video_results': 0.5}]]}
        out = self.root / 'vbench'
        original_check_output = subprocess.check_output
        def fake_git(command, **kwargs):
            if command[0] == 'git':
                return 'a' * 40 if 'rev-parse' in command else ''
            return original_check_output(command, **kwargs)
        with patch.object(vbench.subprocess, 'check_output', side_effect=fake_git):
            self.assertEqual(vbench.run(manifest, mapping, out, upstream, 'a' * 40, cache, receipt, 'pilot', backend), 0)
            self.assertEqual(vbench.run(manifest, mapping, out, upstream, 'a' * 40, cache, receipt, 'pilot', backend), 0)
        self.assertEqual(len(calls), 3)
        self.assertEqual(len(read_csv(out / 'vbench_results.csv')[1]), 3)

    def test_hunyuan_pilot_gate_and_mixed_frame_resume(self):
        self.adapt()
        upstream, assets = self.root / 'upstream', self.root / 'assets'
        upstream.mkdir()
        assets.mkdir()
        required = ('transformer/480p_i2v_step_distilled', 'vae', 'scheduler', 'text_encoder/llm',
                    'text_encoder/byt5-small', 'text_encoder/Glyph-SDXL-v2', 'vision_encoder/siglip')
        files = {}
        for name in required:
            path = assets / name / 'fixture'
            path.parent.mkdir(parents=True)
            path.write_bytes(b'fake weight fixture')
            files[str(path.relative_to(assets))] = sha256(path)
        write_json(assets / 'assets.json', {'repositories': {'tencent/HunyuanVideo-1.5': 'a' * 40}, 'files': files})
        pilot = self.root / 'pilot.csv'
        pilot_rows = []
        for i in range(3):
            image = self.root / f'pilot_{i}.png'
            Image.new('RGB', (64, 36), (240, i * 80, 230)).save(image)
            pilot_rows.append(dict(case_id=f'case_900{i}', category='animal', image_path=image.name,
                                   prompt='Independent synthetic pilot', seed=42, source_frame_num=512 if i == 0 else 128,
                                   target_frames=513 if i == 0 else 129))
        write_csv(pilot, dataset.FIELDS, pilot_rows)
        calls = []
        force_failure = [False]
        def fake_worker(command, **_kwargs):
            request = Path(command[-1])
            payload = json.loads(request.read_text())
            row = payload['record']
            calls.append(int(row['target_frames']))
            native = request.parent / 'native.mp4'
            native.write_bytes(b'Synthetic worker fixture; not experimental video')
            write_json(request.parent / 'result.json', {**row, 'status': 'technical_failure' if force_failure[0] else 'success',
                       'native_video': str(native), 'video_sha256': sha256(native)})
            return 0
        def fake_git(command, **_kwargs):
            return 'b' * 40 if 'rev-parse' in command else ''
        with patch.object(hunyuan.subprocess, 'check_output', side_effect=fake_git), \
             patch.object(hunyuan.subprocess, 'call', side_effect=fake_worker):
            with self.assertRaisesRegex(ValueError, 'pilot-receipt'):
                hunyuan.generate(self.canonical, upstream, 'b' * 40, assets, self.root / 'formal')
            pilot_out = self.root / 'pilot-out'
            self.assertEqual(hunyuan.generate(pilot, upstream, 'b' * 40, assets, pilot_out, 'pilot'), 0)
            calls.clear()
            formal = self.root / 'formal'
            self.assertEqual(hunyuan.generate(self.canonical, upstream, 'b' * 40, assets, formal,
                                              pilot_receipt=pilot_out / 'pilot_pass.json', max_new_cases=3), 1)
            self.assertEqual(len(read_csv(formal / 'runs.csv')[1]), 24)
            self.assertEqual(len(calls), 3)
            self.assertEqual(hunyuan.generate(self.canonical, upstream, 'b' * 40, assets, formal,
                                              pilot_receipt=pilot_out / 'pilot_pass.json'), 0)
            self.assertEqual((calls.count(513), calls.count(129)), (6, 18))
            direct = self.root / 'direct-formal'
            calls.clear()
            reason = 'User explicitly requested direct formal generation without a pilot'
            self.assertEqual(hunyuan.generate(self.canonical, upstream, 'b' * 40, assets, direct,
                                             skip_pilot_reason=reason), 0)
            self.assertEqual((calls.count(513), calls.count(129)), (6, 18))
            self.assertEqual(json.loads((direct / 'batch.json').read_text())['pilot_validation'],
                             {'status': 'skipped_by_user', 'reason': reason})
            calls.clear()
            self.assertEqual(hunyuan.generate(self.canonical, upstream, 'b' * 40, assets, direct,
                                             skip_pilot_reason=reason), 0)
            self.assertEqual(calls, [])
            budget_batch = self.root / 'budget-formal'
            self.assertEqual(hunyuan.generate(self.canonical, upstream, 'b' * 40, assets, budget_batch,
                                             skip_pilot_reason=reason, deadline=1), 0)
            self.assertEqual(calls, [])
            self.assertEqual(sum(r['status'] == 'not_started' for r in read_csv(budget_batch / 'runs.csv')[1]), 24)
            self.assertFalse((budget_batch / 'DONE.json').exists())
            self.assertTrue(list(budget_batch.glob('continuation_*.json')))
            with self.assertRaisesRegex(ValueError, 'explicit reason'):
                hunyuan.generate(self.canonical, upstream, 'b' * 40, assets, self.root / 'invalid',
                                  skip_pilot_reason=' ')
            failed_batch = self.root / 'failed-formal'
            force_failure[0] = True
            calls.clear()
            self.assertEqual(hunyuan.generate(self.canonical, upstream, 'b' * 40, assets, failed_batch,
                                             skip_pilot_reason=reason), 1)
            self.assertEqual(len(calls), 1)
            retained = read_csv(failed_batch / 'runs.csv')[1]
            self.assertEqual(sum(r['status'] == 'not_started' for r in retained), 23)
            calls.clear()
            self.assertEqual(hunyuan.generate(self.canonical, upstream, 'b' * 40, assets, failed_batch,
                                             skip_pilot_reason=reason), 1)
            self.assertEqual(calls, [])

    def test_hunyuan_worker_initializes_single_rank_and_handles_official_5d_output(self):
        import numpy as np
        source = self.root / self.source_rows[0]['image_path']
        directory = self.root / 'worker'
        directory.mkdir()
        request = directory / 'request.json'
        record = {'case_id': 'case_0001', 'seed': 42, 'target_frames': 129,
                  'image_sha256': sha256(source), 'status': 'running'}
        write_json(request, {'record': record, 'image': str(source), 'prompt': 'Synthetic fixture',
                             'assets': str(self.root / 'assets'), 'upstream': str(self.root / 'upstream')})
        state = {'initialized': False}
        torch = types.ModuleType('torch')
        torch.bfloat16 = 'fixture-bf16'
        torch.float16 = 'fixture-fp16'
        state['default_dtype'] = 'fixture-fp32'
        torch.get_default_dtype = lambda: state['default_dtype']
        torch.set_default_dtype = lambda value: state.update(default_dtype=value)
        torch.device = lambda value: value
        torch.inference_mode = contextlib.nullcontext
        torch.cuda = types.SimpleNamespace(is_available=lambda: True, is_bf16_supported=lambda: True,
            get_device_name=lambda _i: 'fixture-gpu', set_device=lambda _i: None,
            synchronize=lambda: None, reset_peak_memory_stats=lambda: None,
            empty_cache=lambda: state.update(cache_released=True),
            max_memory_allocated=lambda: 1024, max_memory_reserved=lambda: 2048)
        def initialize(*_args, **kwargs):
            state.update(initialized=True, init_method=kwargs['init_method'])
        torch.distributed = types.SimpleNamespace(init_process_group=initialize,
            is_initialized=lambda: state['initialized'], destroy_process_group=lambda: state.update(initialized=False))
        class Video:
            def __init__(self, ndim=5):
                self.ndim = ndim
                self.shape = (1, 3, 129, 36, 64) if ndim == 5 else (3, 129, 36, 64)
            def __getitem__(self, index):
                return Frame() if isinstance(index, tuple) else Video(4)
        class Frame:
            def detach(self): return self
            def cpu(self): return self
            def clamp(self, *_args): return self
            def permute(self, *_args): return self
            def float(self): return self
            def numpy(self): return np.zeros((36, 64, 3), dtype=np.float32)
        class Latents:
            shape = (1, 32, 33, 3, 4)
            dtype = 'fixture-bf16'
            def detach(self): return self
            def cpu(self): return self
            def contiguous(self): return self
        class Pipeline:
            def __init__(self):
                self.vae = types.SimpleNamespace(decode=lambda *_args, **_kwargs: Video(),
                    set_tile_sample_min_size=lambda size, overlap: state.update(vae_tile=(size, overlap)),
                    disable_tile_parallelism=lambda: state.update(vae_parallel=False))
            @classmethod
            def create_pipeline(cls, **kwargs):
                state['transformer_version'] = kwargs['transformer_version']
                state['component_device'] = kwargs['device']
                state['transformer_init_device'] = kwargs['transformer_init_device']
                text_encoders.load_text_encoder()
                return cls()
            def apply_infer_optimization(self, **kwargs):
                state['optimization'] = kwargs
            def __call__(self, **kwargs):
                state['generation'] = kwargs
                state['module_offloading'] = self.enable_offloading
                return types.SimpleNamespace(videos=self.vae.decode(Latents()))
        modules = {}
        for name in ('hyvideo', 'hyvideo.pipelines', 'hyvideo.commons', 'hyvideo.models', 'imageio'):
            modules[name] = types.ModuleType(name)
            modules[name].__path__ = []
        pipeline = types.ModuleType('hyvideo.pipelines.hunyuan_video_pipeline')
        pipeline.HunyuanVideo_1_5_Pipeline = Pipeline
        parallel = types.ModuleType('hyvideo.commons.parallel_states')
        parallel.initialize_parallel_state = lambda **kwargs: state.update(parallel=kwargs)
        infer = types.ModuleType('hyvideo.commons.infer_state')
        infer.initialize_infer_state = lambda args: args
        imageio = types.ModuleType('imageio.v2')
        class Writer:
            def __init__(self, path, **kwargs):
                self.path = path
                state['saved_frames'] = 0
            def __enter__(self): return self
            def append_data(self, frame):
                self_test.assertEqual(frame.shape, (36, 64, 3))
                self_test.assertEqual(frame.dtype, np.uint8)
                state['saved_frames'] += 1
            def __exit__(self, *_args):
                Path(self.path).write_bytes(b'Synthetic encoded-video fixture')
        self_test = self
        imageio.get_writer = Writer
        text_encoders = types.ModuleType('hyvideo.models.text_encoders')
        original_loader = lambda: state.update(text_loader_dtype=state['default_dtype'])
        text_encoders.load_text_encoder = original_loader
        safetensors = types.ModuleType('safetensors')
        safetensors.__path__ = []
        safe_torch = types.ModuleType('safetensors.torch')
        safe_torch.save_file = lambda tensors, path: Path(path).write_bytes(b'Synthetic retained latent fixture')
        modules.update({'torch': torch, pipeline.__name__: pipeline, parallel.__name__: parallel,
                        infer.__name__: infer, text_encoders.__name__: text_encoders, 'imageio.v2': imageio,
                        'safetensors': safetensors, 'safetensors.torch': safe_torch})
        with patch.dict('sys.modules', modules), patch.object(hunyuan, 'binary', return_value='/mock/nvidia-smi'), \
             patch.object(hunyuan.subprocess, 'check_output', return_value='fixture-driver'), \
             patch.object(hunyuan, 'inspect_video', return_value=dict(frames=129, fps=24, width=64, height=36,
                                                                    duration=129 / 24, codec='h264')):
            self.assertEqual(hunyuan.worker(request), 0)
        self.assertEqual(state['saved_frames'], 129)
        self.assertEqual(state['parallel'], {'sp': 1})
        self.assertTrue(state['init_method'].startswith('file://'))
        self.assertFalse(state['initialized'])
        self.assertFalse(state['optimization']['overlap_group_offloading'])
        self.assertFalse(state['module_offloading'])
        self.assertEqual(state['component_device'], 'cuda')
        self.assertEqual(state['transformer_init_device'], 'cuda')
        self.assertEqual(state['text_loader_dtype'], 'fixture-fp16')
        self.assertEqual(state['default_dtype'], 'fixture-fp32')
        self.assertIs(text_encoders.load_text_encoder, original_loader)
        self.assertEqual(state['generation']['num_inference_steps'], 8)
        self.assertEqual(state['vae_tile'], (64, 0.25))
        self.assertFalse(state['vae_parallel'])
        self.assertFalse(state['generation']['enable_vae_tile_parallelism'])
        self.assertTrue(state['cache_released'])
        retained = json.loads((directory / 'vae_input_latents.json').read_text())
        self.assertTrue(retained['already_scaled_for_vae'])
        self.assertEqual(retained['target_frames'], 129)
        self.assertEqual(retained['sha256'], sha256(directory / 'vae_input_latents.safetensors'))
        self.assertEqual(json.loads((directory / 'result.json').read_text())['status'], 'success')


if __name__ == '__main__':
    unittest.main()
