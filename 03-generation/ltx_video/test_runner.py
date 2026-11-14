"""CPU-only protocol checks. These do not claim to validate GPU inference."""
import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import run_cases
from common import nearest_ltx_frames, read_cases, select_cases, write_json


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

    def shared_manifest(self, count=24, frame_updates=None, include_legacy_fields=False):
        categories = (["single-human"] * 3 + ["multiple-human"] * 3 +
                      [label for label in ("animal", "transportation", "food", "plant", "other",
                                          "architecture", "scenery", "indoor", "abstract") for _ in range(2)])
        long_ids = {1, 4, 7, 9, 19, 23}
        fields = ["image_type", "image_path", "prompt", "seed", "frame_num"]
        if include_legacy_fields:
            fields += ["case_id", "category"]
        with self.csv.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for i in range(1, count + 1):
                filename = f"{i:02d}_shared image.png"
                (self.root / filename).write_bytes((self.root / "image.png").read_bytes())
                row = {"image_type": categories[i - 1], "image_path": filename,
                       "prompt": f"  Original, shared prompt {i}  ", "seed": 42,
                       "frame_num": (frame_updates or {}).get(i, 512 if i in long_ids else 128)}
                if include_legacy_fields:
                    row.update(case_id=f"{i:02d}", category=categories[i - 1])
                writer.writerow(row)

    def run_fixture(self, *, phase="smoke", failed=(), interrupted=(),
                    per_category=2, protocol="legacy", extra_args=(),
                    mutate_during_asset_check=None, before_worker=None):
        args = ["--cases", str(self.csv), "--phase", phase, "--upstream", str(self.root / "source"),
                "--assets", str(self.root / "weights"), "--output", str(self.root / "result"),
                "--per-category", str(per_category), "--protocol", protocol, *extra_args]
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

    def test_shared24_selects_all_rows_and_maps_frame_counts_without_rewriting_inputs(self):
        self.shared_manifest()
        original_csv = self.csv.read_bytes()
        original_image = (self.root / "01_shared image.png").read_bytes()
        rows = read_cases(self.csv, protocol="shared24")
        selected = select_cases(rows, "formal", 2, protocol="shared24")
        self.assertEqual([row["case_id"] for row in selected], [f"{i:02d}" for i in range(1, 25)])
        self.assertEqual(len({row["category"] for row in selected}), 11)
        self.assertEqual(selected[0]["prompt"], "  Original, shared prompt 1  ")
        self.assertTrue(all(row["seed"] == 42 for row in selected))
        self.assertEqual([nearest_ltx_frames(n) for n in (128, 512)], [129, 513])
        self.assertEqual(sum(row["requested_frame_num"] == 129 for row in rows), 18)
        self.assertEqual(sum(row["requested_frame_num"] == 513 for row in rows), 6)
        self.assertEqual(select_cases(rows, "formal", 4, protocol="shared24"), rows)
        self.assertEqual(self.csv.read_bytes(), original_csv)
        self.assertEqual((self.root / "01_shared image.png").read_bytes(), original_image)

    def test_shared24_passes_each_case_frame_count_to_worker_and_exports_targets(self):
        self.shared_manifest()
        status, calls = self.run_fixture(phase="formal", protocol="shared24")
        self.assertEqual((status, len(calls)), (0, 24))
        for payload in calls:
            expected = 513 if payload["case"]["case_id"] in {"01", "04", "07", "09", "19", "23"} else 129
            self.assertEqual(payload["settings"]["num_frames"], expected)
            self.assertEqual(payload["record"]["requested_frame_num"], expected)
            self.assertEqual(payload["record"]["target_frame_num"], expected - 1)
            self.assertEqual((payload["settings"]["width"], payload["settings"]["height"],
                              payload["settings"]["frame_rate"]), (768, 432, 24))
        with (self.root / "result/runs.csv").open() as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 24)
        self.assertEqual(rows[0]["protocol"], "shared24")
        self.assertEqual((rows[0]["target_frame_num"], rows[0]["requested_frame_num"]), ("512", "513"))

    def test_shared24_rejects_num_frames_override_and_wrong_distribution(self):
        self.shared_manifest()
        with self.assertRaisesRegex(ValueError, "conflicts"):
            self.run_fixture(phase="formal", protocol="shared24", extra_args=["--num-frames", "129"])
        self.shared_manifest(frame_updates={2: 512})
        with self.assertRaisesRegex(ValueError, "18 rows of 128"):
            self.run_fixture(phase="formal", protocol="shared24")
        self.shared_manifest(count=23)
        with self.assertRaisesRegex(ValueError, "IDs 01..24"):
            self.run_fixture(phase="formal", protocol="shared24")

    def test_shared24_freezes_protocol_and_per_case_frames(self):
        self.shared_manifest(count=1, include_legacy_fields=True)
        self.assertEqual(self.run_fixture(protocol="shared24")[0], 0)
        with self.assertRaisesRegex(ValueError, "Frozen batch changed"):
            self.run_fixture(protocol="legacy")
        self.shared_manifest(count=1, frame_updates={1: 128}, include_legacy_fields=True)
        with self.assertRaisesRegex(ValueError, "Frozen batch changed"):
            self.run_fixture(protocol="shared24")


if __name__ == "__main__":
    unittest.main()
