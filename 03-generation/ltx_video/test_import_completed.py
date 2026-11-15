"""Completed-delivery intake fixtures; no experimental results or model loading."""
import csv
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import import_completed as intake
from common import MODEL_REVISION, TEXT_REVISION, TEXT_FILES, UPSTREAM_COMMIT, digest, object_digest, write_json
from ai6132_benchmark.dataset import COUNTS, FIELDS
from ai6132_benchmark.io import write_csv, read_csv

MODEL_ID = intake.HISTORICAL_MODEL


class CompletedIntakeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.bundle, self.output = self.root / 'relocated', self.root / 'derived'
        self.bundle.mkdir()
        self.cases = self.root / 'canonical.csv'
        rows, old = [], []
        categories = [name for name, count in sorted(COUNTS.items()) for _ in range(count)]
        for i, category in enumerate(categories, 1):
            image = self.root / f'{i:02d}_reference.jpg'
            image.write_bytes(f'synthetic image bytes {i}'.encode())
            frames = 512 if i <= 6 else 128
            rows.append(dict(zip(FIELDS, [f'case_{i:04d}', category, image.name,
                                         f' Original prompt {i} ', 42, frames, frames + 1])))
            old.append({'case_id': f'{i:02d}', 'category': category, 'prompt': rows[-1]['prompt'],
                        'seed': 42, 'image_path': f'/unavailable/original/{image.name}',
                        'image_sha256': digest(image), 'target_frame_num': frames, 'requested_frame_num': frames + 1})
        write_csv(self.cases, FIELDS, rows)
        settings = dict(width=768, height=432, num_frames=None, frame_rate=24, offload_to_cpu=True)
        template = json.loads((Path(intake.__file__).parent / 'ltxv-2b-0.9.8-no-enhance.json').read_text())
        pipeline = dict(template, checkpoint_path=f'/assets/ltx/{MODEL_ID}.safetensors',
                        spatial_upscaler_model_path='/assets/ltx/ltxv-spatial-upscaler-0.9.8.safetensors',
                        text_encoder_model_name_or_path='/assets/text')
        assets = dict(model_revision=MODEL_REVISION, text_revision=TEXT_REVISION,
                      files={name: 'a' * 64 for name in [f'ltx/{MODEL_ID}.safetensors',
                      'ltx/ltxv-spatial-upscaler-0.9.8.safetensors'] + ['text/' + name for name in TEXT_FILES]})
        frozen = dict(schema_version=1, protocol='shared24', phase='formal', model_id=MODEL_ID,
                      upstream_commit=UPSTREAM_COMMIT, cases_csv_sha256='a' * 64, cases=old,
                      pipeline=pipeline, settings=settings, asset_manifest=assets)
        write_json(self.bundle / 'batch.json', frozen)
        write_json(self.bundle / 'selection.json', dict(schema_version=1, protocol='shared24', phase='formal',
                   cases_csv_sha256='a' * 64, case_ids=[c['case_id'] for c in old]))
        write_json(self.bundle / 'pipeline.json', frozen['pipeline'])
        self.records, self.infos = [], {}
        for case in old:
            original = case['case_id']
            directory = self.bundle / 'cases' / original
            (directory / 'native').mkdir(parents=True)
            video = directory / 'native' / f'original-{original}.mp4'
            video.write_bytes(f'synthetic native {original}'.encode())
            record = dict(case_id=original, category=case['category'], model_id=MODEL_ID, phase='formal',
                          protocol='shared24', seed=42, status='success', failure_reason='', weight_version=MODEL_ID,
                          weight_revision=MODEL_REVISION, text_encoder_revision=TEXT_REVISION,
                          upstream_commit=UPSTREAM_COMMIT, image_sha256=case['image_sha256'],
                          config='/unavailable/original/pipeline.json', config_sha256=digest(self.bundle / 'pipeline.json'),
                          case_fingerprint=object_digest({'case': case, 'frozen': frozen}),
                          target_frame_num=case['target_frame_num'], requested_frame_num=case['requested_frame_num'],
                          native_video=f'/unavailable/original/cases/{original}/native/{video.name}',
                          video_sha256=digest(video), native_width=768, native_height=432,
                          native_frames=case['requested_frame_num'], native_fps=24,
                          native_duration_seconds=case['requested_frame_num'] / 24, native_codec='h264')
            request = dict(case=case, settings=dict(settings, num_frames=case['requested_frame_num']),
                           config=record['config'], record=dict(record, status='running'))
            write_json(directory / 'request.json', request)
            write_json(directory / 'result.json', record)
            self.records.append(record)
            self.infos[video.name] = dict(width=768, height=432, frames=case['requested_frame_num'], fps=24,
                                         duration=case['requested_frame_num'] / 24, codec='h264')
        write_csv(self.bundle / 'runs.csv', list(self.records[0]), self.records)

    def make_current(self):
        frozen = json.loads((self.bundle / 'batch.json').read_text())
        frozen.update(schema_version=3, model_id=intake.CANONICAL_MODEL, cases_csv_sha256=digest(self.cases))
        frozen.pop('protocol')
        frozen['settings'].pop('num_frames')
        frozen['settings'].update(memory_strategy=intake.MEMORY_STRATEGY, conditioning_strength=1.0,
                                  image_cond_noise_scale=0.15, conditioning_start_frame=0)
        template = json.loads((Path(intake.__file__).parent / 'ltxv-13b-0.9.8-no-enhance.json').read_text())
        frozen['pipeline'] = dict(template, checkpoint_path=f'/assets/ltx/{intake.CANONICAL_MODEL}.safetensors',
                                  spatial_upscaler_model_path='/assets/ltx/ltxv-spatial-upscaler-0.9.8.safetensors',
                                  text_encoder_model_name_or_path='/assets/text')
        assets = frozen['asset_manifest']['files']
        assets[f'ltx/{intake.CANONICAL_MODEL}.safetensors'] = assets.pop(f'ltx/{MODEL_ID}.safetensors')
        entries = {}
        for name, relative in intake.SOURCE_PATHS.items():
            path = self.bundle / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            if name == 'pipeline_template':
                write_json(path, template)
            else:
                path.write_text(f'# Synthetic {name} fixture\n')
                frozen[name + '_sha256'] = digest(path)
            entries[name] = dict(path=relative, sha256=digest(path))
        receipt = frozen['source_receipt'] = dict(schema_version=1, files=entries)
        write_json(self.bundle / 'source_receipt.json', receipt)
        write_json(self.bundle / 'pipeline.json', frozen['pipeline'])
        for case in frozen['cases']:
            case.update(case_id=f"case_{int(case['case_id']):04d}",
                        source_frame_num=case.pop('target_frame_num'), target_frames=case.pop('requested_frame_num'))
        write_json(self.bundle / 'batch.json', frozen)
        write_json(self.bundle / 'selection.json', dict(schema_version=2, protocol='a-mixed-24-v1', phase='formal',
                   cases_csv_sha256=frozen['cases_csv_sha256'], case_ids=[case['case_id'] for case in frozen['cases']]))
        for case, record in zip(frozen['cases'], self.records):
            old_id, new_id = record['case_id'], case['case_id']
            directory = self.bundle / 'cases' / new_id
            (self.bundle / 'cases' / old_id).rename(directory)
            record.pop('protocol')
            record.update(case_id=new_id, model_id=intake.CANONICAL_MODEL, weight_version=intake.CANONICAL_MODEL,
                          source_frame_num=record.pop('target_frame_num'), target_frames=record.pop('requested_frame_num'),
                          config_sha256=digest(self.bundle / 'pipeline.json'), case_fingerprint=object_digest({'case': case, 'frozen': frozen}),
                          native_video=record['native_video'].replace(f'/cases/{old_id}/', f'/cases/{new_id}/'),
                          memory_strategy=intake.MEMORY_STRATEGY, memory_adapter_sha256=frozen['memory_adapter_sha256'],
                          source_receipt_sha256=digest(self.bundle / 'source_receipt.json'),
                          conditioning_strength=1.0, image_cond_noise_scale=0.15, conditioning_start_frame=0,
                          offload_requested=True, offload_effective=False,
                          memory_audit=f'/unavailable/original/cases/{new_id}/memory_audit.json')
            audit = dict(version=1, strategy=intake.MEMORY_STRATEGY, cpu_offload_effective=False, factory_calls=1,
                         events=[{'event': name, 'pass_number': number} for number in (1, 2)
                                 for name in ('pass_start', 'pass_complete')]
                                + [{'event': 'final_transformer_released_before_vae_decode'}])
            write_json(directory / 'memory_audit.json', audit)
            record['memory_audit_sha256'] = digest(directory / 'memory_audit.json')
            request_record = {key: value for key, value in record.items() if key != 'offload_effective'}
            request_record['status'] = 'running'
            request = dict(case=case, settings=dict(frozen['settings'], num_frames=case['target_frames']),
                           config=record['config'], record=request_record, batch_root='/unavailable/original',
                           source_receipt=receipt, code_sha256={name: frozen[name + '_sha256'] for name in entries if name != 'pipeline_template'})
            write_json(directory / 'request.json', request)
            write_json(directory / 'result.json', record)
        self.save_csv()

    def save_csv(self):
        (self.bundle / 'runs.csv').unlink()
        write_csv(self.bundle / 'runs.csv', list(self.records[0]), self.records)

    def run_intake(self):
        with patch.object(intake, 'inspect_video', side_effect=lambda p: self.infos[p.name]) as decode:
            result = intake.intake(self.cases, self.bundle, self.output)
        return result, decode.call_count

    def test_relocation_preserves_all_originals_and_derives_all_24_canonical_records(self):
        before = {p.relative_to(self.bundle): digest(p) for p in self.bundle.rglob('*') if p.is_file()}
        receipt, calls = self.run_intake()
        self.assertEqual((receipt['success'], calls), (24, 24))
        _, rows, _ = read_csv(self.output / 'runs.csv')
        self.assertEqual([r['case_id'] for r in rows], [f'case_{i:04d}' for i in range(1, 25)])
        self.assertEqual(sum(r['target_frames'] == '513' for r in rows), 6)
        self.assertTrue(all(not Path(r['native_video']).is_absolute() for r in rows))
        self.assertTrue(all(r['model_id'] == MODEL_ID for r in rows))
        moved = self.root / 'moved'
        shutil.move(self.output, moved)
        for row in rows:
            self.assertEqual(digest(moved / row['native_video']), row['video_sha256'])
            self.assertEqual(digest(moved / row['config']), row['config_sha256'])
        shutil.move(moved, self.output)
        self.assertEqual(before, {p.relative_to(self.bundle): digest(p) for p in self.bundle.rglob('*') if p.is_file()})
        with self.assertRaisesRegex(ValueError, 'new derived'):
            self.run_intake()

    def test_changed_canonical_prompt_or_image_is_rejected(self):
        with self.cases.open() as stream:
            rows = list(csv.DictReader(stream))
        original_prompt = rows[0]['prompt']
        rows[0]['prompt'] = 'a different experiment'
        self.cases.unlink()
        write_csv(self.cases, FIELDS, rows)
        with self.assertRaisesRegex(ValueError, 'Frozen input differs'):
            self.run_intake()
        rows[0]['prompt'] = original_prompt
        self.cases.unlink()
        write_csv(self.cases, FIELDS, rows)
        (self.root / rows[0]['image_path']).write_bytes(b'a different first image')
        with self.assertRaisesRegex(ValueError, 'Frozen input differs'):
            self.run_intake()
        self.assertFalse(self.output.exists())

    def test_incomplete_denominator_and_legacy_protocol_are_rejected(self):
        path = self.bundle / 'selection.json'
        data = json.loads(path.read_text())
        data['case_ids'].pop()
        write_json(path, data)
        with self.assertRaisesRegex(ValueError, 'all 24'):
            self.run_intake()
        path = self.bundle / 'batch.json'
        data = json.loads(path.read_text())
        data['protocol'] = 'legacy'
        write_json(path, data)
        with self.assertRaisesRegex(ValueError, 'Only the completed'):
            self.run_intake()

    def test_failed_or_forged_terminal_results_are_rejected(self):
        self.records[0]['status'] = 'technical_failure'
        write_json(self.bundle / 'cases/01/result.json', self.records[0])
        # The original success CSV cannot hide a failed terminal result.
        with self.assertRaisesRegex(ValueError, 'differs from the terminal'):
            self.run_intake()
        (self.bundle / 'runs.csv').unlink()
        write_csv(self.bundle / 'runs.csv', list(self.records[0]), self.records)
        with self.assertRaisesRegex(ValueError, 'mismatched successful result'):
            self.run_intake()

    def test_changed_native_hash_decode_and_external_symlinks_are_rejected(self):
        video = self.bundle / 'cases/01/native/original-01.mp4'
        original = video.read_bytes()
        video.write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'changed native'):
            self.run_intake()
        video.write_bytes(original)
        self.infos[video.name]['frames'] = 129
        with self.assertRaisesRegex(ValueError, 'Full native decode'):
            self.run_intake()
        outside = self.root / 'outside.mp4'
        outside.write_bytes(original)
        video.unlink()
        video.symlink_to(outside)
        with self.assertRaisesRegex(ValueError, 'escaping retained'):
            self.run_intake()

    def test_canonical_13b_source_snapshots_and_portable_copy(self):
        self.make_current()
        before = {p.relative_to(self.bundle): digest(p) for p in self.bundle.rglob('*') if p.is_file()}
        receipt, calls = self.run_intake()
        self.assertEqual((calls, receipt['success'], receipt['source_schema_version']), (24, 24, 3))
        self.assertEqual(receipt['model_id'], intake.CANONICAL_MODEL)
        _, rows, _ = read_csv(self.output / 'runs.csv')
        moved = self.root / 'moved13b'
        shutil.move(self.output, moved)
        for row in rows:
            self.assertEqual(row['model_id'], intake.CANONICAL_MODEL)
            self.assertEqual(row['source_case_id'], row['case_id'])
            self.assertFalse(Path(row['native_video']).is_absolute())
            self.assertEqual(digest(moved / row['native_video']), row['video_sha256'])
            self.assertEqual(digest(moved / row['config']), row['config_sha256'])
        self.assertEqual(before, {p.relative_to(self.bundle): digest(p) for p in self.bundle.rglob('*') if p.is_file()})

    def test_canonical_schema2_and_mixed_models_are_rejected(self):
        self.make_current()
        self.records[0]['model_id'] = MODEL_ID
        write_json(self.bundle / 'cases/case_0001/result.json', self.records[0])
        self.save_csv()
        with self.assertRaisesRegex(ValueError, 'mismatched successful result'):
            self.run_intake()
        path = self.bundle / 'batch.json'
        data = json.loads(path.read_text())
        data['schema_version'] = 2
        write_json(path, data)
        with self.assertRaisesRegex(ValueError, 'Only the completed'):
            self.run_intake()

    def test_13b_tone_map_is_part_of_the_frozen_configuration(self):
        self.make_current()
        pipeline = json.loads((self.bundle / 'pipeline.json').read_text())
        pipeline['second_pass'].pop('tone_map_compression_ratio')
        data = json.loads((self.bundle / 'batch.json').read_text())
        data['pipeline'] = pipeline
        write_json(self.bundle / 'pipeline.json', pipeline)
        write_json(self.bundle / 'batch.json', data)
        with self.assertRaisesRegex(ValueError, 'fixed official'):
            self.run_intake()
        self.assertFalse(self.output.exists())

    def test_source_snapshot_and_worker_code_hash_mismatch_are_rejected(self):
        self.make_current()
        path = self.bundle / intake.SOURCE_PATHS['memory_adapter']
        original = path.read_bytes()
        path.write_bytes(b'changed adapter')
        with self.assertRaisesRegex(ValueError, 'Source snapshot'):
            self.run_intake()
        path.write_bytes(original)
        path = self.bundle / 'cases/case_0001/request.json'
        request = json.loads(path.read_text())
        request['code_sha256']['memory_adapter'] = 'f' * 64
        write_json(path, request)
        with self.assertRaisesRegex(ValueError, 'Request code/source receipt'):
            self.run_intake()

    def test_request_frames_and_memory_release_audit_are_checked(self):
        self.make_current()
        path = self.bundle / 'cases/case_0001/request.json'
        request = json.loads(path.read_text())
        request['settings']['num_frames'] = 129
        write_json(path, request)
        with self.assertRaisesRegex(ValueError, 'bound to the frozen case'):
            self.run_intake()
        request['settings']['num_frames'] = 513
        write_json(path, request)
        path = self.bundle / 'cases/case_0001/memory_audit.json'
        audit = json.loads(path.read_text())
        audit['events'].pop()
        write_json(path, audit)
        self.records[0]['memory_audit_sha256'] = digest(path)
        write_json(self.bundle / 'cases/case_0001/result.json', self.records[0])
        self.save_csv()
        with self.assertRaisesRegex(ValueError, 'both passes and final release'):
            self.run_intake()

    def test_recorded_video_cannot_escape_its_case_or_batch(self):
        self.make_current()
        for remote in ('/unavailable/original/cases/case_0002/native/original-01.mp4',
                       '/elsewhere/cases/case_0001/native/original-01.mp4'):
            self.records[0]['native_video'] = remote
            write_json(self.bundle / 'cases/case_0001/result.json', self.records[0])
            self.save_csv()
            with self.assertRaisesRegex(ValueError, 'Video escapes'):
                self.run_intake()

    def test_source_snapshot_symlink_outside_bundle_is_rejected(self):
        self.make_current()
        path = self.bundle / intake.SOURCE_PATHS['common']
        outside = self.root / 'external_common.py'
        outside.write_bytes(path.read_bytes())
        path.unlink()
        path.symlink_to(outside)
        with self.assertRaisesRegex(ValueError, 'escaping retained'):
            self.run_intake()

    def test_input_changed_during_decode_prevents_handoff_publication(self):
        self.make_current()
        def decode(path):
            (self.root / '01_reference.jpg').write_bytes(b'changed during verification')
            return self.infos[path.name]
        with patch.object(intake, 'inspect_video', side_effect=decode):
            with self.assertRaisesRegex(ValueError, 'Canonical input changed during intake'):
                intake.intake(self.cases, self.bundle, self.output)
        self.assertFalse(self.output.exists())


if __name__ == '__main__':
    unittest.main()
