"""CPU protocol fixtures for the canonical mixed-length benchmark only."""
import contextlib
import csv
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import run_cases
from common import MODEL_ID, MODEL_REVISION, TEXT_REVISION, read_cases, select_cases, verify_assets, write_json
from ai6132_benchmark.dataset import COUNTS, FIELDS


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        (self.root / 'image.png').write_bytes(b'Synthetic protocol fixture; no image decoding')
        self.csv = self.root / 'cases.csv'

    def manifest(self, count=1, prompt='  Original, English prompt  '):
        categories = [category for category in sorted(COUNTS) for _ in range(COUNTS[category])]
        rows = []
        for i in range(count):
            long = (count == 24 and i < 6) or (count == 3 and i == 0)
            rows.append(dict(zip(FIELDS, [f'case_{i:04d}', categories[i], 'image.png', prompt,
                                         100 + i, 512 if long else 128, 513 if long else 129])))
        self.write_rows(rows)

    def write_rows(self, rows):
        with self.csv.open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(rows)

    def run_fixture(self, *, phase='smoke', failed=(), interrupted=(),
                    mutate_during_asset_check=None, before_worker=None):
        args = ['--cases', str(self.csv), '--phase', phase, '--upstream', str(self.root / 'source'),
                '--assets', str(self.root / 'weights'), '--output', str(self.root / 'result')]
        calls = []

        def fake_worker(command, **kwargs):
            request = Path(command[-1])
            payload = json.loads(request.read_text())
            if before_worker:
                before_worker(payload)
            calls.append(payload)
            result = dict(payload['record'])
            if result['case_id'] in interrupted:
                raise KeyboardInterrupt()
            if result['case_id'] in failed:
                result.update(status='technical_failure', failure_reason='Synthetic CUDA failure')
            else:
                clip = request.parent / 'native' / 'original.mp4'
                clip.parent.mkdir()
                clip.write_bytes(b'Synthetic retained native fixture')
                result.update(status='success', native_video=str(clip), video_sha256=run_cases.digest(clip))
            write_json(request.parent / 'result.json', result)
            return 0 if result['status'] == 'success' else 1

        def fake_asset_check(_root):
            if mutate_during_asset_check:
                mutate_during_asset_check()
            return {}

        with patch.object(run_cases, 'verify_upstream'), patch.object(run_cases, 'verify_assets', side_effect=fake_asset_check), \
             patch.object(run_cases.platform, 'platform', return_value='CPU protocol fixture'), \
             patch.object(run_cases.subprocess, 'check_output', return_value='fixture-package==0\n'), \
             patch.object(run_cases.subprocess, 'call', side_effect=fake_worker):
            status = run_cases.main(args)
        return status, calls

    def test_prompt_seed_and_all_24_case_order_are_preserved(self):
        self.manifest(24)
        rows = read_cases(self.csv)
        self.assertEqual(rows[0]['prompt'], '  Original, English prompt  ')
        self.assertEqual(rows[0]['seed'], 100)
        self.assertEqual(select_cases(rows, 'formal'), rows)
        self.assertEqual(len({r['category'] for r in rows}), 11)

    def test_canonical_schema_and_duplicate_ids_are_checked(self):
        self.manifest(3)
        self.csv.write_text(self.csv.read_text().replace('target_frames', 'frames'))
        with self.assertRaisesRegex(ValueError, 'target_frames'):
            read_cases(self.csv)
        self.manifest(3)
        with self.csv.open() as stream:
            rows = list(csv.DictReader(stream))
        self.write_rows(rows + [rows[0]])
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            read_cases(self.csv)

    def test_resume_never_rerolls_successful_or_failed_cases(self):
        self.manifest(3)
        status, first = self.run_fixture(phase='pilot', failed={'case_0001'})
        self.assertEqual((status, len(first)), (1, 3))
        status, second = self.run_fixture(phase='pilot')
        self.assertEqual((status, len(second)), (1, 0))
        with (self.root / 'result/runs.csv').open() as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 3)
        self.assertEqual(rows[1]['failure_reason'], 'Synthetic CUDA failure')
        self.assertEqual(first[0]['case']['seed'], 100)

    def test_changed_prompt_or_missing_original_blocks_rerun(self):
        self.manifest()
        self.assertEqual(self.run_fixture()[0], 0)
        self.manifest(prompt='A different prompt')
        with self.assertRaisesRegex(ValueError, 'Frozen batch changed'):
            self.run_fixture()
        self.manifest()
        next((self.root / 'result/cases/case_0000/native').glob('*.mp4')).unlink()
        with self.assertRaisesRegex(ValueError, 'refusing regeneration'):
            self.run_fixture()

    def test_csv_changed_during_asset_verification_keeps_one_snapshot(self):
        original = 'Original snapshot prompt'
        self.manifest(prompt=original)
        digest = run_cases.digest(self.csv)
        status, calls = self.run_fixture(mutate_during_asset_check=lambda: self.manifest(prompt='Changed prompt'))
        self.assertEqual(status, 0)
        frozen = json.loads((self.root / 'result/batch.json').read_text())
        self.assertEqual(frozen['cases_csv_sha256'], digest)
        self.assertNotEqual(digest, run_cases.digest(self.csv))
        self.assertEqual(frozen['cases'][0]['prompt'], original)
        self.assertEqual(calls[0]['case']['prompt'], original)
        with self.assertRaisesRegex(ValueError, 'Frozen batch changed'):
            self.run_fixture()

    def test_incomplete_attempt_is_preserved_while_remaining_cases_continue(self):
        self.manifest(3)
        self.run_fixture(phase='pilot', interrupted={'case_0001'})
        status, calls = self.run_fixture(phase='pilot')
        self.assertEqual((status, len(calls)), (1, 1))
        self.assertEqual(calls[0]['case']['case_id'], 'case_0002')
        self.assertEqual(run_cases.records(self.root / 'result')[1]['status'], 'incomplete')

    def test_interruption_exports_every_case_without_starting_pending_ones(self):
        self.manifest(3)
        status, calls = self.run_fixture(phase='pilot', interrupted={'case_0001'})
        self.assertEqual((status, len(calls)), (1, 2))
        output = self.root / 'result'
        with (output / 'runs.csv').open() as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual([r['case_id'] for r in rows], ['case_0000', 'case_0001', 'case_0002'])
        self.assertEqual([r['status'] for r in rows], ['success', 'incomplete', 'not_started'])
        self.assertEqual(rows[2]['seed'], '102')
        self.assertEqual(rows[2]['elapsed_seconds'], '')
        self.assertEqual(rows[2]['gpu_model'], '')
        self.assertFalse((output / 'cases/case_0002').exists())

    def test_full_denominator_is_exported_before_first_worker(self):
        self.manifest(3)
        snapshots = []
        def inspect_export(_payload):
            with (self.root / 'result/runs.csv').open() as stream:
                snapshots.append(list(csv.DictReader(stream)))
        self.run_fixture(phase='pilot', before_worker=inspect_export)
        self.assertEqual(len(snapshots[0]), 3)
        self.assertEqual([r['status'] for r in snapshots[0]], ['incomplete', 'not_started', 'not_started'])

    def test_formal_rejects_subset_or_edited_selection_receipt(self):
        self.manifest(24)
        self.assertEqual(self.run_fixture(phase='formal')[0], 0)
        output = self.root / 'result'
        before = (output / 'runs.csv').read_bytes()
        self.manifest(3)
        with self.assertRaisesRegex(ValueError, 'Formal manifest'):
            self.run_fixture(phase='formal')
        self.manifest(24)
        selection = json.loads((output / 'selection.json').read_text())
        selection['case_ids'] = selection['case_ids'][:12]
        write_json(output / 'selection.json', selection)
        with self.assertRaisesRegex(ValueError, 'complete canonical batch'):
            self.run_fixture(phase='formal')
        with self.assertRaisesRegex(ValueError, 'complete canonical batch'):
            run_cases.records(output)
        self.assertEqual((output / 'runs.csv').read_bytes(), before)

    def test_all_11_categories_and_per_case_frames_are_used_by_default(self):
        self.manifest(24)
        status, calls = self.run_fixture(phase='formal')
        self.assertEqual((status, len(calls)), (0, 24))
        self.assertEqual([p['settings']['num_frames'] for p in calls], [513] * 6 + [129] * 18)
        self.assertEqual(len({p['case']['category'] for p in calls}), 11)
        self.assertEqual({p['record']['model_id'] for p in calls}, {'ltxv-13b-0.9.8-distilled'})
        for payload in calls:
            run_cases.verify_worker_code(payload)
            options = run_cases.inference_arguments(payload)
            self.assertEqual(options['num_frames'], payload['case']['target_frames'])
            self.assertEqual(options['conditioning_start_frames'], [0])
            self.assertEqual(options['conditioning_strengths'], [1.0])
            self.assertEqual(options['image_cond_noise_scale'], 0.15)
            self.assertEqual(options['prompt'], payload['case']['prompt'])
            self.assertEqual(options['seed'], payload['case']['seed'])
        self.assertEqual(self.run_fixture(phase='formal'), (0, []))

    def test_13b_config_and_snapshot_preserve_the_validated_adapter(self):
        self.manifest()
        status, calls = self.run_fixture()
        self.assertEqual(status, 0)
        output = self.root / 'result'
        frozen = json.loads((output / 'batch.json').read_text())
        receipt = json.loads((output / 'source_receipt.json').read_text())
        self.assertEqual(frozen['schema_version'], 3)
        self.assertEqual(frozen['source_receipt'], receipt)
        self.assertEqual(set(receipt['files']), set(run_cases.SOURCE_FILES))
        self.assertEqual(frozen['memory_adapter_sha256'], 'ef26651b6d78861db9331eca86a94d9ba31821ddc3f56d710d7418f8e44aad7a')
        self.assertEqual(calls[0]['record']['source_receipt_sha256'], run_cases.digest(output / 'source_receipt.json'))
        template = json.loads((output / receipt['files']['pipeline_template']['path']).read_text())
        self.assertEqual(template['checkpoint_path'], 'ltxv-13b-0.9.8-distilled.safetensors')
        self.assertEqual(template['second_pass']['tone_map_compression_ratio'], 0.6)
        self.assertEqual(template['prompt_enhancement_words_threshold'], 0)
        self.assertEqual((len(template['first_pass']['timesteps']), len(template['second_pass']['timesteps'])), (7, 3))
        legacy = json.loads((run_cases.HERE / 'ltxv-2b-0.9.8-no-enhance.json').read_text())
        self.assertEqual(legacy['checkpoint_path'], 'ltxv-2b-0.9.8-distilled.safetensors')

    def test_changed_frozen_source_blocks_worker_and_resume(self):
        self.manifest()
        _, calls = self.run_fixture()
        payload = calls[0]
        receipt = payload['source_receipt']
        saved = self.root / 'result' / receipt['files']['memory_adapter']['path']
        saved.write_bytes(saved.read_bytes() + b'\n# changed\n')
        with self.assertRaisesRegex(ValueError, 'source snapshot'):
            run_cases.verify_worker_code(payload)
        with self.assertRaisesRegex(ValueError, 'source snapshot'):
            self.run_fixture()

    def test_worker_rejects_changed_code_model_and_conditioning(self):
        self.manifest()
        _, calls = self.run_fixture()
        payload = calls[0]
        original_sha = payload['code_sha256']['memory_adapter']
        payload['code_sha256']['memory_adapter'] = '0' * 64
        with self.assertRaisesRegex(ValueError, 'Frozen worker code changed'):
            run_cases.verify_worker_code(payload)
        payload['code_sha256']['memory_adapter'] = original_sha
        payload['record']['model_id'] = 'ltxv-2b-0.9.8-distilled'
        with self.assertRaisesRegex(ValueError, 'selected 13B'):
            run_cases.verify_worker_code(payload)
        payload['record']['model_id'] = MODEL_ID
        for key, changed in [('conditioning_strength', 0.8), ('image_cond_noise_scale', 0.05),
                             ('conditioning_start_frame', 1), ('memory_strategy', 'none')]:
            original = payload['settings'][key]
            payload['settings'][key] = changed
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'conditioning/memory'):
                run_cases.inference_arguments(payload)
            payload['settings'][key] = original

    def test_2b_assets_cannot_be_used_as_13b(self):
        root = self.root / 'assets'
        write_json(root / 'assets.json', {'model_revision': MODEL_REVISION, 'text_revision': TEXT_REVISION,
                                        'files': {'ltx/ltxv-2b-0.9.8-distilled.safetensors': '0' * 64}})
        with self.assertRaisesRegex(ValueError, 'required model'):
            verify_assets(root)

    def test_removed_cli_flags_are_rejected(self):
        base = ['--cases', str(self.csv), '--phase', 'formal']
        for flags in (['--per-category', '4'], ['--selection-mode', 'manifest'],
                      ['--num-frames', '121'], ['--protocol', 'legacy'], ['--protocol', 'shared24']):
            with self.subTest(flags=flags), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as caught:
                run_cases.parser().parse_args(base + flags)
            self.assertEqual(caught.exception.code, 2)

    def test_smoke_generator_writes_the_same_canonical_schema(self):
        out = self.root / 'synthetic'
        script = Path(__file__).resolve().parents[2] / '02-pilot/tc2/make_smoke_case.py'
        subprocess.run([sys.executable, str(script), '--out', str(out)], check=True)
        rows = read_cases(out / 'cases.csv')
        self.assertEqual(select_cases(rows, 'smoke')[0]['target_frames'], 129)

    def test_pilot_requires_short_and_long_profiles(self):
        self.manifest(3)
        rows = read_cases(self.csv)
        for row in rows:
            row.update(source_frame_num=128, target_frames=129)
        with self.assertRaisesRegex(ValueError, 'both 129-frame and 513-frame'):
            select_cases(rows, 'pilot')


if __name__ == '__main__':
    unittest.main()
