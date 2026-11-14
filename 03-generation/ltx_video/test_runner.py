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

    def run_fixture(self, *, phase="smoke", failed=(), interrupted=(),
                    per_category=2, mutate_during_asset_check=None, before_worker=None):
        args = ["--cases", str(self.csv), "--phase", phase, "--upstream", str(self.root / "source"),
                "--assets", str(self.root / "weights"), "--output", str(self.root / "result"),
                "--per-category", str(per_category)]
        calls = []

        def fake_worker(command, **kwargs):
            request = Path(command[-1])
            payload = json.loads(request.read_text())
            if before_worker:
                before_worker(payload)
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

    def test_interruption_exports_every_selected_case_without_starting_pending_ones(self):
        self.manifest(3)
        status, calls = self.run_fixture(phase="pilot", interrupted={"case_1"})
        self.assertEqual((status, len(calls)), (1, 2))
        output = self.root / "result"
        with (output / "runs.csv").open() as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual([row["case_id"] for row in rows], ["case_0", "case_1", "case_2"])
        self.assertEqual([row["status"] for row in rows], ["success", "incomplete", "not_started"])
        self.assertEqual(rows[2]["seed"], "102")
        self.assertEqual(rows[2]["elapsed_seconds"], "")
        self.assertEqual(rows[2]["gpu_model"], "")
        self.assertFalse((output / "cases/case_2").exists())

    def test_full_denominator_is_exported_before_first_worker(self):
        self.manifest(3)
        snapshots = []

        def inspect_export(_payload):
            with (self.root / "result/runs.csv").open() as stream:
                snapshots.append(list(csv.DictReader(stream)))

        self.run_fixture(phase="pilot", before_worker=inspect_export)
        self.assertEqual(len(snapshots[0]), 3)
        self.assertEqual([row["status"] for row in snapshots[0]],
                         ["incomplete", "not_started", "not_started"])

    def test_expansion_keeps_attempts_and_cannot_shrink_handoff(self):
        self.manifest(24)
        status, first = self.run_fixture(phase="formal")
        self.assertEqual((status, len(first)), (0, 12))
        output = self.root / "result"
        selection = json.loads((output / "selection.json").read_text())
        self.assertEqual(len(selection["case_ids"]), 12)
        status, new = self.run_fixture(phase="formal", per_category=4, interrupted={"case_13"})
        self.assertEqual((status, len(new)), (1, 2))
        rows = run_cases.records(output)
        self.assertEqual(len(rows), 24)
        self.assertEqual(sum(row["status"] == "not_started" for row in rows), 10)
        before = (output / "runs.csv").read_bytes()
        with self.assertRaisesRegex(ValueError, "Cannot shrink"):
            self.run_fixture(phase="formal", per_category=2)
        self.assertEqual((output / "runs.csv").read_bytes(), before)
        status, resumed = self.run_fixture(phase="formal", per_category=4)
        self.assertEqual((status, len(resumed)), (1, 10))
        self.assertEqual(len(run_cases.records(output)), 24)


if __name__ == "__main__":
    unittest.main()
