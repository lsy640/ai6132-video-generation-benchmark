"""Synthetic runner/codec tests, not measured Wan GPU generation or quality scores."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image
from ai6132_benchmark import dataset, media
from ai6132_benchmark.io import read_csv, sha256, write_csv

HERE = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('wan_benchmark', HERE/'benchmark.py')
b = importlib.util.module_from_spec(spec)
spec.loader.exec_module(b)
spec = importlib.util.spec_from_file_location('wan_benchmark_worker', HERE/'benchmark_worker.py')
w = importlib.util.module_from_spec(spec)
spec.loader.exec_module(w)


class BenchmarkTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='SYNTHETIC-wan-')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.cases = self.root/'cases.canonical.csv'
        self.assets = self.root/'assets'; self.assets.mkdir()
        self.batch = self.root/'batch'
        self.revision = 'a'*40
        categories = [key for key, count in sorted(dataset.COUNTS.items()) for _ in range(count)]
        source_rows = []
        for index, category in enumerate(categories, 1):
            name = f'{index:02d}_reference.jpg'
            Image.new('RGB', (64, 36), (index*5, 30, 70)).save(self.root/name)
            source_rows.append({'image_type': category, 'image_path': name,
                                'prompt': f' Original prompt {index}\nsecond line ', 'seed': '42',
                                'frame_num': '512' if index <= 6 else '128'})
        source = self.root/'source.csv'
        write_csv(source, list(source_rows[0]), source_rows)
        dataset.canonicalize(source, self.cases)
        for name in ('config.json', 'models_t5_umt5-xxl-enc-bf16.pth', 'Wan2.2_VAE.pth',
                     'google/umt5-xxl/tokenizer_config.json', 'diffusion_pytorch_model.safetensors'):
            path = self.assets/name; path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b'SYNTHETIC placeholder; never load as a real checkpoint')
        self.receipt_assets()

    def receipt_assets(self):
        files = {p.relative_to(self.assets).as_posix(): sha256(p) for p in self.assets.rglob('*')
                 if p.is_file() and p.name != 'assets.json'}
        b.atomic_json(self.assets/'assets.json', {'repositories': {b.MODEL_REPO: self.revision}, 'files': files})

    def prepare(self, profile='formal50'):
        return b.prepare(self.cases, self.assets, self.revision, self.batch, profile)

    def fake_generation(self, command, **kwargs):
        if command[0] == sys.executable and command[1] == str(HERE/'benchmark_worker.py'):
            request = b.read(command[command.index('--request')+1])
            output = Path(command[command.index('--out')+1])
            self.original_run([media.binary('ffmpeg'), '-v', 'error', '-f', 'lavfi', '-i', 'color=c=red:s=1248x704:r=24',
                '-frames:v', str(request['target_frames']), '-c:v', 'libx264', '-threads', '1',
                '-preset', 'ultrafast', '-pix_fmt', 'yuv420p', str(output/'native.mp4')], check=True)
            b.atomic_json(output/'worker_result.json', {'gpu_model': 'SYNTHETIC-NO-GPU',
                'peak_vram_allocated_mib': 0, 'peak_vram_reserved_mib': 0})
            return subprocess.CompletedProcess(command, 0)
        return self.original_run(command, **kwargs)

    def test_formal_profile_preserves_all_cases_prompt_seed_and_129_513_frames(self):
        batch = self.prepare()
        self.assertEqual(len(batch['cases']), 24)
        self.assertEqual(batch['settings']['sampling_steps'], 50)
        self.assertEqual(batch['cases'][0]['target_frames'], '513')
        self.assertEqual(batch['cases'][-1]['target_frames'], '129')
        self.assertEqual(batch['cases'][0]['prompt'], ' Original prompt 1\nsecond line ')
        self.assertTrue(all(case['seed'] == '42' for case in batch['cases']))
        _, rows, _ = read_csv(self.batch/'runs.csv')
        self.assertEqual(len(rows), 24)
        self.assertTrue(all(row['status'] == 'not_started' for row in rows))
        self.assertFalse((self.batch/'DONE.json').exists())
        with self.assertRaisesRegex(ValueError, 'new batch'):
            self.prepare()

    def test_ten_step_profile_is_a_separate_diagnostic_and_does_not_floor_frames(self):
        batch = self.prepare('diagnostic10')
        self.assertEqual(batch['phase'], 'diagnostic')
        self.assertEqual(batch['settings']['sampling_steps'], 10)
        self.assertEqual(batch['cases'][0]['target_frames'], '513')
        self.assertFalse(batch['historical_A_delivery'])

    def test_asset_change_revision_and_unreceipted_payload_are_rejected(self):
        with self.assertRaisesRegex(ValueError, 'revision'):
            b.asset_snapshot(self.assets, 'main')
        with self.assertRaisesRegex(ValueError, 'revision'):
            b.asset_snapshot(self.assets, 'b'*40)
        path = self.assets/'diffusion_pytorch_model.safetensors'
        path.write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'Asset bytes'):
            self.prepare()
        self.receipt_assets()
        (self.assets/'extra.safetensors').write_bytes(b'unreceipted alternative model')
        with self.assertRaisesRegex(ValueError, 'file set'):
            self.prepare()
        self.assertFalse(self.batch.exists())

    def test_incomplete_shard_index_is_rejected(self):
        b.atomic_json(self.assets/'diffusion_pytorch_model.safetensors.index.json',
                      {'weight_map': {'synthetic': 'missing.safetensors'}})
        self.receipt_assets()
        with self.assertRaisesRegex(ValueError, 'shard index'):
            self.prepare()

    def test_batch_settings_and_inputs_cannot_drift(self):
        batch = self.prepare()
        batch['settings']['sampling_steps'] = 10
        b.atomic_json(self.batch/'batch.json', batch)
        with self.assertRaisesRegex(ValueError, 'configuration changed'):
            b.load_batch(self.batch)

    def test_reference_and_code_changes_are_rejected_before_generation(self):
        batch = self.prepare()
        with patch.object(b, 'source_snapshot', return_value={'changed': 'changed'}):
            with self.assertRaisesRegex(ValueError, 'Source changed'):
                b.generate_one(self.batch, 'case_0001')
        Path(batch['cases'][0]['image_path']).write_bytes(b'changed reference')
        with self.assertRaisesRegex(ValueError, 'Reference image changed'):
            b.generate_one(self.batch, 'case_0001')
        self.assertFalse((self.batch/'cases').exists())

    def test_short_and_long_native_decode_and_success_is_not_regenerated(self):
        self.prepare()
        self.original_run = subprocess.run
        with patch.object(b.subprocess, 'run', side_effect=self.fake_generation) as execute:
            long = b.generate_one(self.batch, 'case_0001')
            short = b.generate_one(self.batch, 'case_0024')
            before = len([call for call in execute.call_args_list if call.args[0][0] == sys.executable])
            again = b.generate_one(self.batch, 'case_0001')
            after = len([call for call in execute.call_args_list if call.args[0][0] == sys.executable])
        self.assertEqual((long['native_frames'], short['native_frames']), (513, 129))
        self.assertEqual(before, after)
        self.assertEqual(again['video_sha256'], long['video_sha256'])
        _, rows, _ = read_csv(self.batch/'runs.csv')
        self.assertEqual(len(rows), 24)
        self.assertEqual(sum(row['status'] == 'success' for row in rows), 2)
        self.assertTrue(all(not Path(row['config']).is_absolute() for row in rows))
        self.assertFalse((self.batch/'DONE.json').exists())
        (self.batch/'cases/case_0001/native.mp4').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'Existing attempted'):
            b.generate_one(self.batch, 'case_0001')

    def test_failed_attempt_is_retained_and_never_automatically_retried(self):
        self.prepare()
        with patch.object(b.subprocess, 'run', side_effect=subprocess.CalledProcessError(1, ['SYNTHETIC'])) as execute:
            with self.assertRaises(subprocess.CalledProcessError):
                b.generate_one(self.batch, 'case_0001')
            with self.assertRaisesRegex(ValueError, 'Existing attempted'):
                b.generate_one(self.batch, 'case_0001')
        self.assertEqual(execute.call_count, 1)
        _, rows, _ = read_csv(self.batch/'runs.csv')
        self.assertEqual(rows[0]['status'], 'generation_failed')
        self.assertEqual(sum(row['status'] == 'not_started' for row in rows), 23)
        self.assertTrue((self.batch/'cases/case_0001/worker.log').exists())

    def test_unfinished_case_and_concurrent_writer_are_rejected(self):
        self.prepare()
        with b.batch_lock(self.batch):
            with self.assertRaises(BlockingIOError):
                b.generate_one(self.batch, 'case_0001')
        (self.batch/'cases/case_0001').mkdir(parents=True)
        with self.assertRaisesRegex(ValueError, 'Existing attempted'):
            b.generate_one(self.batch, 'case_0001')

    def test_cli_rejects_a_head_node_invocation_before_loading_weights(self):
        with patch.dict(os.environ, {}, clear=True), self.assertRaisesRegex(SystemExit, 'sbatch'):
            b.main()


class LibraryTests(unittest.TestCase):
    def test_ti2v_lazy_import_does_not_load_audio_or_animation(self):
        spec = importlib.util.spec_from_file_location('wan_lazy_fixture', HERE/'wan/__init__.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        sentinel = type('TI2VFixture', (), {})
        with patch.object(module, 'import_module', return_value=type('OnlyTI2V', (), {'WanTI2V': sentinel})) as imported:
            self.assertIs(module.WanTI2V, sentinel)
            self.assertIs(module.WanTI2V, sentinel)
        imported.assert_called_once_with('.textimage2video', 'wan_lazy_fixture')

    def test_native_encoder_error_is_not_reported_as_success(self):
        class EmptyVideo:
            shape = (3, 0, 32, 64)
        with patch.object(w.shutil, 'which', return_value='synthetic-ffmpeg'), patch.object(w.subprocess, 'Popen') as popen:
            popen.return_value.wait.return_value = 7
            with self.assertRaisesRegex(RuntimeError, 'writer failed'):
                w.encode_native(EmptyVideo(), Path('never-created.mp4'), 24, None)


if __name__ == '__main__':
    unittest.main()
