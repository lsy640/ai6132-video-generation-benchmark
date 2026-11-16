"""CPU-only tests of path rebinding, frozen code and no-overwrite behavior."""
import json
from pathlib import Path
import unittest

import test_import_completed as fixtures
import run_case
import rebuild_batch
from ai6132_benchmark.io import sha256


class ReplayPreparationTests(unittest.TestCase):
    def setUp(self):
        self.fixture=fixtures.IntakeTests('test_portable_table_preserves_original_status_bytes_and_reused_origin')
        self.fixture.setUp(); self.addCleanup(self.fixture.doCleanups)
        self.root=self.fixture.root
        self.runtime=self.root/'runtime'; self.runtime.mkdir()
        profile=json.loads((run_case.HERE/'frozen/batch.json').read_text())['profile']
        (self.runtime/'WEIGHTS_READY').write_text(profile['model_revision'])

    def test_replay_uses_absolute_shared_image_and_proven_memory_profiles(self):
        for cid, blocks in [('case_0001',40),('case_0024',0)]:
            out=self.root/cid
            request=run_case.prepare_request(self.fixture.cases,cid,self.runtime,out)
            data=json.loads(request.read_text())
            self.assertTrue(Path(data['image_relative_path']).is_absolute())
            self.assertEqual(sha256(Path(data['image_relative_path'])),data['image_sha256'])
            self.assertEqual(data['cpu_offload_block_count'],blocks)
            self.assertEqual((data['seed'],data['num_inference_steps']), (42,16))
            with self.assertRaisesRegex(ValueError,'new case output'):
                run_case.prepare_request(self.fixture.cases,cid,self.runtime,out)

    def test_wrong_runtime_revision_is_rejected_before_output_created(self):
        (self.runtime/'WEIGHTS_READY').write_text('different')
        out=self.root/'rejected'
        with self.assertRaisesRegex(ValueError,'revision'):
            run_case.prepare_request(self.fixture.cases,'case_0001',self.runtime,out)
        self.assertFalse(out.exists())

    def test_rebuild_binds_new_paths_and_preserves_generation_engine(self):
        source=self.fixture.raw/'cases/case_0001'
        (source/'evaluation.mp4').write_bytes(b'synthetic processing copy')
        report=json.loads((source/'run.json').read_text())
        report['evaluation']={'sha256':sha256(source/'evaluation.mp4')}
        (source/'run.json').write_text(json.dumps(report))
        out=self.root/'new-batch'
        config=rebuild_batch.prepare(self.fixture.cases,self.runtime,source,out)
        self.assertEqual(len(config['short_order']),18)
        self.assertEqual(len(config['long_order'])-len(config['reuse']),5)
        self.assertEqual(config['input_root'],str(self.fixture.cases.parent))
        self.assertEqual(sha256(out/'code/engine.py'),sha256(run_case.HERE/'frozen/engine.py'))
        self.assertEqual(config['reuse'],{'case_0001':str(source)})
        self.assertIn('Path(config[\'input_root\'])',(out/'code/worker.py').read_text())
        self.assertEqual(sha256(out/'code/worker.py'), sha256(run_case.HERE/'worker.py'))
        self.assertEqual(config['code_sha256']['worker.py'], sha256(run_case.HERE/'worker.py'))
        with self.assertRaisesRegex(ValueError,'new batch'):
            rebuild_batch.prepare(self.fixture.cases,self.runtime,source,out)

    def test_frozen_files_match_the_original_published_receipts(self):
        frozen=run_case.HERE/'frozen'
        config=json.loads((frozen/'batch.json').read_text())
        for name, expected in config['code_sha256'].items():
            self.assertEqual(sha256(frozen/name),expected)


if __name__=='__main__': unittest.main()
