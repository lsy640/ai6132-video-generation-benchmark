"""Synthetic intake/media fixtures. These are not experimental quality results."""
import copy
import csv
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image
import import_completed as intake
from ai6132_benchmark import dataset, media
from ai6132_benchmark.io import read_csv, sha256, write_csv


class IntakeTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='longcat-intake-')
        self.root = Path(self.temporary.name).resolve()
        self.addCleanup(self.temporary.cleanup)
        self.raw, self.output, self.reference = [self.root/x for x in ('raw', 'delivery', 'reference')]
        self.raw.mkdir(); self.reference.mkdir()
        source = self.root/'source.csv'
        categories = [k for k, n in sorted(dataset.COUNTS.items()) for _ in range(n)]
        originals = []
        for i, category in enumerate(categories, 1):
            image = self.root/f'{i:02d}_reference.jpg'
            Image.new('RGB', (64, 36), (i*5, 30, 70)).save(image)
            originals.append(dict(image_type=category, image_path=image.name, prompt=f' Original prompt {i} ',
                                  seed='42', frame_num='512' if i <= 6 else '128'))
        write_csv(source, list(originals[0]), originals)
        self.cases = self.root/'cases.canonical.csv'
        dataset.canonicalize(source, self.cases)
        cases, receipt = dataset.load_canonical(self.cases)
        config = json.loads((intake.REFERENCE/'batch.json').read_text())
        self.profile = config['profile']
        config.update(canonical_sha256=sha256(self.cases), receipt_sha256=sha256(self.cases.with_suffix('.receipt.json')))
        shutil.copy2(self.cases, self.raw/'cases.canonical.csv')
        shutil.copy2(self.cases.with_suffix('.receipt.json'), self.raw/'cases.canonical.receipt.json')
        config['code_sha256'] = {}
        for name in ('engine.py', 'worker.py', 'wave.slurm'):
            (self.raw/'code').mkdir(exist_ok=True)
            (self.raw/'code'/name).write_text('synthetic source '+name)
            shutil.copy2(self.raw/'code'/name, self.reference/name)
            config['code_sha256'][name] = sha256(self.reference/name)
        self.save(self.raw/'config.json', config)
        shutil.copy2(self.raw/'config.json', self.reference/'batch.json')
        (self.raw/'provenance').mkdir()
        for name in ('environment-freeze.txt', 'weights-manifest.json'):
            (self.raw/'provenance'/name).write_text('synthetic receipt '+name)
        runtime = {'environment_sha256': sha256(self.raw/'provenance/environment-freeze.txt'),
                   'weights_manifest_sha256': sha256(self.raw/'provenance/weights-manifest.json')}
        self.save(self.raw/'provenance/runtime.json', runtime)
        self.save(self.reference/'runtime.json', runtime)
        for name in ('trial_engine.py', 'trial_config.json'):
            (self.reference/name).write_text('synthetic trial source '+name)
        self.save(self.raw/'COMPLETION.json', {'all_succeeded': True, 'counts': {'succeeded': 24}})
        spec = importlib.util.spec_from_file_location('frozen_longcat_worker', intake.REFERENCE/'worker.py')
        worker = importlib.util.module_from_spec(spec); spec.loader.exec_module(worker)
        rows = []
        self.infos = {}
        for case in cases:
            cid, frames = case['case_id'], int(case['target_frames'])
            folder = self.raw/'cases'/cid
            folder.mkdir(parents=True)
            shutil.copy2(self.root/case['image_path'], folder/'input.jpg')
            video = folder/'native.mp4'; video.write_bytes(('synthetic video '+cid).encode())
            blocks = 40 if frames == 513 else 0
            request = dict(self.profile, source_case=cid, prompt=case['prompt'], seed=42,
                           num_frames=frames, image_sha256=receipt['images'][cid]['sha256'], cpu_offload_block_count=blocks)
            report = dict(request, status='succeeded', elapsed_seconds=5.0,
                          native=dict(frames=frames, width=832, height=480, fps=15.0,
                                      duration_seconds=frames/15, sha256=sha256(video)))
            self.save(folder/'request.json', request); self.save(folder/'run.json', report)
            origin = 'imported_validated_trial' if cid == 'case_0001' else 'new_generation'
            result = dict(status='succeeded', origin=origin, attempt_count=1, source_run_sha256=sha256(folder/'run.json'))
            self.save(folder/'result.json', result)
            if cid == 'case_0001':
                proof = [('run_trial.py', sha256(self.reference/'trial_engine.py')),
                         ('config.json', sha256(self.reference/'trial_config.json')),
                         ('weights-manifest.json', runtime['weights_manifest_sha256']),
                         ('environment-freeze.txt', runtime['environment_sha256'])]
                (folder/'code-and-environment.sha256').write_text('\n'.join(digest+'  /synthetic/'+name for name,digest in proof))
            self.save(folder/'generation-shape.json', dict(frames=frames, offloaded_blocks=blocks, gpu='NVIDIA L40S'))
            self.save(folder/'denoise-result.json', dict(seconds=2.0, peak_gpu_gib=37.0, completed_steps=16))
            (folder/'steps.jsonl').write_text('\n'.join(json.dumps(dict(step=i, total_steps=16, seconds=.1)) for i in range(1,17)))
            row = dict(case_id=cid, category=case['category'], model_id=intake.WEIGHT_ID, phase='full24', seed='42',
                source_frame_num=case['source_frame_num'], target_frames=case['target_frames'], status='succeeded', origin=origin,
                weight_revision=intake.WEIGHT_REVISION, steps=16, resolution='480p', elapsed_seconds=5.0, generation_seconds=2.0,
                peak_memory_gib=37.0, job_id='synthetic', attempt_count=1, native_video=f'cases/{cid}/native.mp4',
                native_frames=frames, native_width=832, native_height=480, native_fps=15.0, native_duration_seconds=frames/15,
                video_sha256=sha256(video), evaluation_video='', evaluation_sha256='', failure_reason='')
            rows.append(row)
            self.infos[cid] = dict(frames=frames, width=832, height=480, fps=15.0, duration=frames/15, codec='h264', timestamps=[0.0, (frames-1)/15])
        write_csv(self.raw/'runs.csv', worker.FIELDS, rows)
        self.source_sha = sha256(self.raw/'runs.csv')
        self.save(self.reference/'verification.json', {'source_runs_sha256': self.source_sha})
        self.addCleanup(patch.stopall)
        patch.object(intake, 'REFERENCE', self.reference).start()
        patch.object(intake, 'PUBLISHED_RECORDS', self.reference).start()
        patch.object(intake, 'inspect_native', side_effect=self.inspect).start()

    def save(self, path, value):
        path.write_text(json.dumps(value, indent=2), encoding='utf-8')

    def inspect(self, path, decoder='ffmpeg'):
        path = Path(path)
        cid = path.stem if path.stem.startswith('case_') else path.parent.name
        return copy.deepcopy(self.infos[cid])

    def export(self):
        return intake.export(self.raw, self.cases, self.output)

    def test_portable_table_preserves_original_status_bytes_and_reused_origin(self):
        self.export()
        _, rows, _ = read_csv(self.output/'runs.csv')
        self.assertEqual(len(rows), 24)
        self.assertTrue(all(r['model_id'] == intake.MODEL_ID and r['status'] == 'success' and r['source_status'] == 'succeeded' for r in rows))
        self.assertTrue(all(not Path(r['native_video']).is_absolute() and not Path(r['config']).is_absolute() for r in rows))
        self.assertEqual(rows[0]['origin'], 'imported_validated_trial')
        self.assertEqual(rows[0]['source_formal_evaluation'], 'false')
        self.assertEqual(self.source_sha, sha256(self.raw/'runs.csv'))
        relocated = self.root/'relocated'
        shutil.copytree(self.output, relocated)
        self.assertEqual(intake.verify(self.raw, self.cases, relocated)['cases'], 24)
        with self.assertRaisesRegex(ValueError, 'new delivery'):
            self.export()

    def test_changed_video_rejected_before_creating_delivery(self):
        (self.raw/'cases/case_0002/native.mp4').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'Native bytes'):
            self.export()
        self.assertFalse(self.output.exists())

    def test_coherently_replaced_video_and_source_records_are_rejected(self):
        self.export()
        folder = self.raw/'cases/case_0002'
        (folder/'native.mp4').write_bytes(b'synthetic replacement native')
        digest = sha256(folder/'native.mp4')
        report = json.loads((folder/'run.json').read_text())
        report['native']['sha256'] = digest
        self.save(folder/'run.json', report)
        fields, rows, _ = read_csv(self.raw/'runs.csv')
        rows[1]['video_sha256'] = digest
        replacement = self.raw/'replacement.csv'
        write_csv(replacement, fields, rows)
        replacement.replace(self.raw/'runs.csv')
        rejected = self.root/'rejected-delivery'
        with patch.object(intake, 'inspect_native') as decode:
            with self.assertRaisesRegex(ValueError, 'published output records'):
                intake.export(self.raw, self.cases, rejected)
            with self.assertRaisesRegex(ValueError, 'published output records'):
                intake.verify(self.raw, self.cases, self.output)
        decode.assert_not_called()
        self.assertFalse(rejected.exists())

    def test_missing_published_anchor_is_rejected_before_export(self):
        (self.reference/'verification.json').unlink()
        with self.assertRaisesRegex(ValueError, 'Missing or unsafe retained path'):
            self.export()
        self.assertFalse(self.output.exists())

    def test_request_seed_and_missing_rows_rejected(self):
        p = self.raw/'cases/case_0002/request.json'
        request = json.loads(p.read_text()); request['seed'] = 43; self.save(p, request)
        with self.assertRaisesRegex(ValueError, 'Request/result'):
            self.export()
        fields, rows, _ = read_csv(self.raw/'runs.csv')
        (self.raw/'runs.csv').unlink(); write_csv(self.raw/'runs.csv', fields, rows[:-1])
        with self.assertRaisesRegex(ValueError, '24 canonical'):
            self.export()

    def test_modified_frozen_code_rejected(self):
        (self.raw/'code/engine.py').write_text('changed')
        with self.assertRaisesRegex(ValueError, 'Source code'):
            self.export()

    def test_wrong_decoded_frame_count_rejected(self):
        self.infos['case_0002']['frames'] = 128
        with self.assertRaisesRegex(ValueError, 'Decoded native'):
            self.export()

    def test_imported_trial_code_proof_is_checked(self):
        (self.raw/'cases/case_0001/code-and-environment.sha256').write_text('changed  /synthetic/run_trial.py')
        with self.assertRaisesRegex(ValueError, 'Imported trial source proof'):
            self.export()

    def test_changed_delivered_config_rejected(self):
        self.export()
        (self.output/'configs/case_0002.json').write_text('{}')
        with self.assertRaisesRegex(ValueError, 'Delivery file'):
            intake.verify(self.raw, self.cases, self.output)

    def test_export_is_accepted_by_existing_four_model_media_entry(self):
        self.export()
        _, rows, _ = read_csv(self.output/'runs.csv')
        runs = {intake.MODEL_ID: self.output/'runs.csv'}
        for i in range(3):
            model = f'synthetic_model_{i}'
            filename = self.output/(model+'.csv')
            write_csv(filename, intake.FIELDS, [dict(r, model_id=model) for r in rows])
            runs[model] = filename
        def normalize(source, target):
            shutil.copyfile(source, target)
            before = self.inspect(source)
            return before, dict(width=768, height=432, fps=12, duration=before['duration'])
        with patch.object(media, 'inspect_video', side_effect=self.inspect), patch.object(media, 'normalize', side_effect=normalize), \
             patch.object(media, 'binary', return_value='synthetic_ffmpeg'), patch.object(media.subprocess, 'check_output', return_value='synthetic FFmpeg version'):
            result = media.prepare(self.cases, runs, self.root/'synthetic-four-model-handoff', frozen_outputs=True)
        self.assertEqual(result, {'items': 96, 'success': 96})


class PublishedAnchorTests(unittest.TestCase):
    def test_published_anchor_matches_retained_original_csv_and_batch(self):
        published = intake.document(intake.PUBLISHED_RECORDS/'verification.json')
        self.assertEqual(sha256(intake.PUBLISHED_RECORDS/'source/runs.csv'), published['source_runs_sha256'])
        self.assertEqual(sha256(intake.REFERENCE/'batch.json'), published['source_batch_sha256'])


class TemporaryRootTests(unittest.TestCase):
    def test_fixture_uses_system_temp_directory_independently_of_checkout_depth(self):
        fixture = IntakeTests('test_portable_table_preserves_original_status_bytes_and_reused_origin')
        self.addCleanup(fixture.doCleanups)
        # A shallow fictitious path catches parent indexing without touching it.
        with patch(__name__ + '.__file__', '/fixture.py'):
            fixture.setUp()
        self.assertEqual(fixture.root.parent, Path(tempfile.gettempdir()).resolve())


class RealDecoderTest(unittest.TestCase):
    def test_pyav_decodes_frames_and_monotonic_timestamps(self):
        import av
        with tempfile.TemporaryDirectory() as tmp:
            filename = Path(tmp)/'fixture.mp4'
            with av.open(str(filename), 'w') as output:
                stream = output.add_stream('libx264', rate=15)
                stream.width, stream.height, stream.pix_fmt = 64, 36, 'yuv420p'
                for i in range(5):
                    frame = av.VideoFrame.from_image(Image.new('RGB', (64, 36), (i*30, 0, 0)))
                    for packet in stream.encode(frame): output.mux(packet)
                for packet in stream.encode(): output.mux(packet)
            info = intake.inspect_native(filename, 'pyav')
            self.assertEqual((info['frames'], info['fps'], info['width'], info['height']), (5,15.0,64,36))
            self.assertTrue(all(b>a for a,b in zip(info['timestamps'], info['timestamps'][1:])))


if __name__ == '__main__':
    unittest.main()
