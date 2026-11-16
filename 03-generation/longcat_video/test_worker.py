import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('longcat_worker', Path(__file__).parent / 'worker.py')
w = importlib.util.module_from_spec(spec)
spec.loader.exec_module(w)


class PreservationChecks(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='longcat-check-')
        self.batch = Path(self.temp.name)
        self.case = {'case_id': 'case_0002', 'category': 'human', 'prompt': 'walk forward', 'seed': '42',
                     'source_frame_num': '128', 'target_frames': '129', 'image_path': 'dataset/input.jpg'}
        self.receipt = {'images': {'case_0002': {'sha256': 'image-hash'}}}
        self.config = {'batch_id': 'test', 'resource_attempt_limit': 2, 'input_root': str(self.batch),
                       'case_timeout_seconds': {'129': 1800, '513': 7200},
                       'memory': {'short_offload_blocks': 0, 'long_offload_blocks': 40, 'oom_fallback_blocks': 40},
                       'profile': {'model_id': 'longcat', 'model_revision': 'model-sha', 'source_revision': 'source-sha',
                                   'resolution': '480p', 'num_inference_steps': 16, 'guidance_scale': 1.0,
                                   'use_distill': True, 'native_fps': 15}}

    def tearDown(self):
        self.temp.cleanup()

    def success(self, command, env, log, timeout):
        folder = Path(env['LONGCAT_CASE_OUTPUT'])
        request = w.read(Path(env['LONGCAT_CASE_CONFIG']))
        frames = request['num_frames']
        scoring_frames = round(frames / 15 * 12)
        for name in ('native', 'evaluation'):
            (folder / (name + '.mp4')).write_bytes(b'valid-for-orchestration-check-' + name.encode())
        report = {**request, 'status': 'succeeded', 'elapsed_seconds': 10,
                  'native': {'frames': frames, 'width': 832, 'height': 480, 'fps': 15, 'duration_seconds': frames/15,
                             'sha256': w.sha(folder / 'native.mp4')},
                  'evaluation': {'frames': scoring_frames, 'width': 768, 'height': 432, 'fps': 12,
                                 'duration_seconds': scoring_frames/12, 'sha256': w.sha(folder / 'evaluation.mp4')}}
        w.write(folder / 'run.json', report)
        return 0

    def test_completed_video_is_not_generated_again_and_tampering_is_rejected(self):
        with patch.object(w, 'execute_engine', side_effect=self.success) as run:
            w.run_case(self.batch, self.config, self.case, self.receipt)
            w.run_case(self.batch, self.config, self.case, self.receipt)
            self.assertEqual(run.call_count, 1)
            (self.batch / 'cases/case_0002/native.mp4').write_bytes(b'changed')
            with self.assertRaises(AssertionError):
                w.run_case(self.batch, self.config, self.case, self.receipt)
            self.assertEqual(run.call_count, 1)

    def test_oom_fallback_preserves_seed_and_failed_attempt(self):
        seen = []
        def execute(command, env, log, timeout):
            request = w.read(Path(env['LONGCAT_CASE_CONFIG']))
            seen.append(request)
            if len(seen) == 1:
                folder = Path(env['LONGCAT_CASE_OUTPUT'])
                (folder / 'denoise.log').write_text('CUDA out of memory')
                w.write(folder / 'run.json', {'status': 'failed', 'error': 'OOM'})
                return 1
            return self.success(command, env, log, timeout)
        with patch.object(w, 'execute_engine', side_effect=execute):
            w.run_case(self.batch, self.config, self.case, self.receipt)
        self.assertEqual([r['cpu_offload_block_count'] for r in seen], [0, 40])
        for key in ('prompt', 'seed', 'num_frames', 'num_inference_steps', 'image_sha256'):
            self.assertEqual(seen[0][key], seen[1][key])
        self.assertTrue((self.batch / 'cases/case_0002/attempts/attempt_01/denoise.log').exists())
        self.assertEqual(w.read(self.batch / 'cases/case_0002/result.json')['status'], 'succeeded')

    def test_failed_non_resource_attempt_is_retained_without_reroll(self):
        with patch.object(w, 'execute_engine', return_value=1) as run:
            w.run_case(self.batch, self.config, self.case, self.receipt)
            w.run_case(self.batch, self.config, self.case, self.receipt)
        self.assertEqual(run.call_count, 1)
        self.assertEqual(w.read(self.batch / 'cases/case_0002/result.json')['status'], 'failed')

    def test_success_is_recovered_after_interruption_before_promotion(self):
        attempt = self.batch / 'cases/case_0002/attempts/attempt_01'
        attempt.mkdir(parents=True)
        request = attempt / 'case-config.json'
        w.write(request, w.request_for(self.config, self.case, self.receipt))
        self.success([], {'LONGCAT_CASE_OUTPUT': str(attempt), 'LONGCAT_CASE_CONFIG': str(request)}, None, 1800)
        with patch.object(w, 'execute_engine') as run:
            w.run_case(self.batch, self.config, self.case, self.receipt)
        run.assert_not_called()
        self.assertEqual(w.read(self.batch / 'cases/case_0002/result.json')['status'], 'succeeded')

    def test_exhausted_interrupted_attempts_become_failed_without_reroll(self):
        folder = self.batch/'cases/case_0002'
        for number in (1, 2):
            attempt = folder/f'attempts/attempt_{number:02d}'
            attempt.mkdir(parents=True)
            (attempt/'worker.log').write_text('retained interrupted attempt')
        for status in ('running', 'interrupted'):
            with self.subTest(status=status):
                w.write(folder/'result.json', {'status': status, 'attempt_count': 2, 'slurm_job_id': 'original-job'})
                with patch.object(w, 'execute_engine') as run:
                    w.run_case(self.batch, self.config, self.case, self.receipt)
                    result = w.read(folder/'result.json')
                    w.run_case(self.batch, self.config, self.case, self.receipt)
                run.assert_not_called()
                self.assertEqual(result['status'], 'failed')
                self.assertEqual(result['previous_status'], status)
                self.assertEqual(result['slurm_job_id'], 'original-job')
                self.assertTrue(result['attempt_limit_exhausted'])
                self.assertEqual(result, w.read(folder/'result.json'))
        self.assertEqual(len(list((folder/'attempts').glob('attempt_*'))), 2)
        self.assertTrue((folder/'attempts/attempt_02/worker.log').exists())

    def test_validated_last_attempt_is_promoted_even_when_retry_limit_is_reached(self):
        folder = self.batch/'cases/case_0002'
        (folder/'attempts/attempt_01').mkdir(parents=True)
        attempt = folder/'attempts/attempt_02'
        attempt.mkdir()
        request = attempt/'case-config.json'
        w.write(request, w.request_for(self.config, self.case, self.receipt))
        self.success([], {'LONGCAT_CASE_OUTPUT': str(attempt), 'LONGCAT_CASE_CONFIG': str(request)}, None, 1800)
        w.write(folder/'result.json', {'status': 'interrupted', 'attempt_count': 2})
        with patch.object(w, 'execute_engine') as run:
            w.run_case(self.batch, self.config, self.case, self.receipt)
        run.assert_not_called()
        self.assertEqual(w.read(folder/'result.json')['status'], 'succeeded')

    def test_finish_wave_returns_nonzero_when_results_include_failures(self):
        for succeeded, expected in ((False, 1), (True, 0)):
            with self.subTest(all_succeeded=succeeded), \
                 patch.object(w.sys, 'argv', ['worker.py', '--batch', str(self.batch), '--wave', 'finish']), \
                 patch.object(w, 'load_batch', return_value=({'short_order': [], 'long_order': []}, [], {})), \
                 patch.object(w, 'save_runtime_provenance'), patch.object(w, 'import_prior'), \
                 patch.object(w, 'collect', return_value={}), patch.object(w, 'write'), \
                 patch.object(w.signal, 'signal'), \
                 patch.object(w, 'finalize', return_value={'all_succeeded': succeeded}):
                self.assertEqual(w.main(), expected)

    def test_export_retains_all_24_rows_and_usable_hardlinks_including_failure(self):
        import csv
        import tarfile
        runtime = self.batch / 'runtime'
        runtime.mkdir()
        self.config['remote_root'] = str(runtime)
        cases = [{**self.case, 'case_id': f'case_{i:04d}',
                  'target_frames': '513' if i <= 6 else '129',
                  'source_frame_num': '512' if i <= 6 else '128'} for i in range(1, 25)]
        receipt = {'images': {c['case_id']: {'sha256': 'image-hash'} for c in cases}}
        with patch.object(w, 'execute_engine', side_effect=self.success):
            for case in cases[:-1]:
                w.run_case(self.batch, self.config, case, receipt)
        w.write(self.batch / 'cases/case_0024/result.json', {'status': 'interrupted', 'error': 'retained interruption'})
        original = w.validate_report
        def validate(folder, report, expected, decode=False):
            return original(folder, report, expected, decode=False)
        with patch.object(w, 'validate_report', side_effect=validate):
            w.finalize(self.batch, self.config, cases, receipt)
        completion = w.read(self.batch / 'COMPLETION.json')
        self.assertFalse(completion['all_succeeded'])
        self.assertEqual(completion['counts'], {'failed': 1, 'succeeded': 23})
        failed = w.read(self.batch/'cases/case_0024/result.json')
        self.assertEqual(failed['previous_status'], 'interrupted')
        self.assertIn('retained interruption', failed['error'])
        export = w.read(self.batch / 'EXPORT_READY.json')
        self.assertEqual(w.sha(Path(export['archive'])), export['sha256'])
        extract = self.batch / 'extracted'
        extract.mkdir()
        with tarfile.open(export['archive']) as tf:
            tf.extractall(extract, filter='data')
        output = extract / 'test'
        with (output / 'runs.csv').open(newline='') as f:
            rows = list(csv.DictReader(f))
        self.assertEqual(len(rows), 24)
        self.assertIn('retained interruption', rows[-1]['failure_reason'])
        self.assertTrue((output / 'cases/case_0001/native.mp4').exists())
        self.assertEqual(w.sha(output / 'cases/case_0001/native.mp4'), rows[0]['video_sha256'])


if __name__ == '__main__':
    unittest.main()
