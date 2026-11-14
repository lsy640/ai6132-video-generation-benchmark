"""CPU-only protocol checks. These do not claim to validate GPU inference."""
import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import run_cases
from common import read_cases, select_cases, write_json


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        (self.root / "image.png").write_bytes(b"input fixture; no image decoding in CPU protocol tests")
        self.csv = self.root / "cases.csv"

    def manifest(self, count=1, alias="image_path", prompt="  Original, English prompt  "):
        with self.csv.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=["case_id", "category", alias, "prompt", "seed"])
            writer.writeheader()
            for i in range(count):
                writer.writerow({"case_id": f"case_{i}", "category": f"category_{i % 6}", alias: "image.png",
                                 "prompt": prompt, "seed": 100 + i})

    def run_fixture(self, *, phase="smoke", failed=(), interrupted=(), mutate_during_asset_check=None):
        args = ["--cases", str(self.csv), "--phase", phase, "--upstream", str(self.root / "source"),
                "--assets", str(self.root / "weights"), "--output", str(self.root / "result")]
        calls = []

        def fake_worker(command, **kwargs):
            request = Path(command[-1])
            payload = json.loads(request.read_text())
            calls.append(payload)
            result = dict(payload["record"])
            if result["case_id"] in interrupted:
                raise KeyboardInterrupt()
            if result["case_id"] in failed:
                result.update(status="technical_failure", failure_reason="Synthetic CUDA failure")
            else:
                clip = request.parent / "native" / "original.mp4"
                clip.parent.mkdir()
                clip.write_bytes(b"retained native fixture")
                result.update(status="success", native_video=str(clip), video_sha256=run_cases.digest(clip))
            write_json(request.parent / "result.json", result)
            return 0 if result["status"] == "success" else 1

        def fake_asset_check(_root):
            if mutate_during_asset_check:
                mutate_during_asset_check()
            return {}

        with patch.object(run_cases, "verify_upstream"), patch.object(run_cases, "verify_assets", side_effect=fake_asset_check), \
             patch.object(run_cases.platform, "platform", return_value="CPU protocol fixture"), \
             patch.object(run_cases.subprocess, "check_output", return_value="fixture-package==0\n"), \
             patch.object(run_cases.subprocess, "call", side_effect=fake_worker):
            status = run_cases.main(args)
        return status, calls

    def test_alias_prompt_seed_and_batch_order_are_preserved(self):
        self.manifest(24, alias="first_frame_path")
        rows = read_cases(self.csv)
        self.assertEqual(rows[0]["prompt"], "  Original, English prompt  ")
        self.assertEqual(rows[0]["seed"], 100)
        self.assertEqual([r["case_id"] for r in select_cases(rows, "formal", 2)], [f"case_{i}" for i in range(12)])
        self.assertEqual(len(select_cases(rows, "formal", 4)), 24)

    def test_conflicting_image_aliases_and_duplicate_ids_fail(self):
        self.csv.write_text("case_id,category,image_path,image,prompt,seed\na,x,image.png,other.png,prompt,1\n")
        with self.assertRaisesRegex(ValueError, "conflicting"):
            read_cases(self.csv)
        self.csv.write_text("case_id,category,image_path,prompt,seed\na,x,image.png,prompt,1\na,x,image.png,prompt,1\n")
        with self.assertRaisesRegex(ValueError, "duplicate"):
            read_cases(self.csv)

    def test_resume_never_rerolls_successful_or_failed_cases(self):
        self.manifest(3)
        status, first = self.run_fixture(phase="pilot", failed={"case_1"})
        self.assertEqual((status, len(first)), (1, 3))
        status, second = self.run_fixture(phase="pilot")
        self.assertEqual((status, len(second)), (1, 0))
        with (self.root / "result/runs.csv").open() as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 3)
        self.assertEqual(rows[1]["failure_reason"], "Synthetic CUDA failure")
        self.assertEqual(first[0]["case"]["seed"], 100)

    def test_changed_prompt_or_missing_original_blocks_rerun(self):
        self.manifest()
        self.assertEqual(self.run_fixture()[0], 0)
        self.manifest(prompt="A different prompt")
        with self.assertRaisesRegex(ValueError, "Frozen batch changed"):
            self.run_fixture()
        self.manifest()
        next((self.root / "result/cases/case_0/native").glob("*.mp4")).unlink()
        with self.assertRaisesRegex(ValueError, "refusing regeneration"):
            self.run_fixture()

    def test_csv_changed_during_asset_verification_cannot_mix_rows_and_hash(self):
        original_prompt = "Original snapshot prompt"
        self.manifest(prompt=original_prompt)
        original_sha = run_cases.digest(self.csv)
        status, calls = self.run_fixture(mutate_during_asset_check=lambda: self.manifest(prompt="Updated shared prompt"))
        self.assertEqual(status, 0)
        frozen = json.loads((self.root / "result/batch.json").read_text())
        self.assertEqual(frozen["cases_csv_sha256"], original_sha)
        self.assertNotEqual(frozen["cases_csv_sha256"], run_cases.digest(self.csv))
        self.assertEqual(frozen["cases"][0]["prompt"], original_prompt)
        self.assertEqual(calls[0]["case"]["prompt"], original_prompt)
        with self.assertRaisesRegex(ValueError, "Frozen batch changed"):
            self.run_fixture()

    def test_incomplete_attempt_is_preserved_while_remaining_cases_continue(self):
        self.manifest(3)
        self.run_fixture(phase="pilot", interrupted={"case_1"})
        status, calls = self.run_fixture(phase="pilot")
        self.assertEqual((status, len(calls)), (1, 1))
        self.assertEqual(calls[0]["case"]["case_id"], "case_2")
        with (self.root / "result/runs.csv").open() as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(rows[1]["status"], "incomplete")


if __name__ == "__main__":
    unittest.main()
