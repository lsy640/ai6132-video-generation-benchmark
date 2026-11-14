"""Completed-delivery intake fixtures; no experimental results or model loading."""
import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import import_completed as intake
from common import MODEL_ID, MODEL_REVISION, TEXT_REVISION, UPSTREAM_COMMIT, digest, object_digest, write_json
from ai6132_benchmark.dataset import COUNTS, FIELDS
from ai6132_benchmark.io import write_csv, read_csv


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
        frozen = dict(schema_version=1, protocol='shared24', phase='formal', model_id=MODEL_ID,
                      upstream_commit=UPSTREAM_COMMIT, cases_csv_sha256='a' * 64, cases=old,
                      pipeline={'synthetic': 'pipeline'}, settings=settings)
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
                          native_frames=case['requested_frame_num'], native_fps=24)
            request = dict(case=case, settings=dict(settings, num_frames=case['requested_frame_num']),
                           record=dict(record, status='running'))
            write_json(directory / 'request.json', request)
            write_json(directory / 'result.json', record)
            self.records.append(record)
            self.infos[video.name] = dict(width=768, height=432, frames=case['requested_frame_num'], fps=24,
                                         duration=case['requested_frame_num'] / 24, codec='h264')
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
        self.assertTrue(all(Path(r['native_video']).is_relative_to(self.bundle) for r in rows))
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


if __name__ == '__main__':
    unittest.main()
