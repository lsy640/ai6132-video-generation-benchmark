"""Synthetic fixtures only; these tests do not contain experimental ratings."""

import contextlib
import io
import hashlib
import json
import tempfile
import unittest
from unittest import mock
from pathlib import Path

import human_eval as h


class HumanEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.manifest = self.root / "anonymous.csv"
        (self.root / "images").mkdir()
        (self.root / "videos").mkdir()
        self.image = self.root / "images/clip_0001.png"
        self.rows = [{
            "clip_id": f"clip_{number:04}", "video_path": f"videos/clip_{number:04}.mp4",
            "image_path": f"images/clip_{number:04}.png", "prompt": "A person walks, then turns.",
            "status": "success", "failure_reason": "",
        } for number in range(1, 49)]
        # Preparation only copies bytes. B must supply actually decoded media.
        for row in self.rows:
            (self.root / row["image_path"]).write_bytes(b"synthetic reference fixture")
            (self.root / row["video_path"]).write_bytes(b"synthetic video fixture")
        self.rows[-1].update(status="generation_failed", video_path="",
                             failure_reason="out_of_memory")
        h.write_csv(self.manifest, h.MANIFEST_FIELDS, self.rows)
        self.raters = ["A", "B", "C", "D"]

    def tearDown(self):
        self.temp.cleanup()

    def prepare(self, name="pack"):
        out = self.root / name
        h.prepare(self.manifest, out, self.raters, 6132, 48, "formal")
        return out

    def submissions(self, out):
        return {rater: out / "raters" / rater / "ratings.csv" for rater in self.raters}

    def fill_synthetic_test_scores(self, files):
        for path in files.values():
            rows = h.read_csv(path, h.RATING_FIELDS)
            for row in rows:
                if row["clip_id"] != "clip_0048":
                    row.update(dict.fromkeys(h.SCORES, "3"))
            h.write_csv(path, h.RATING_FIELDS, rows)

    def test_reproducible_independent_order_and_no_mapping_leak(self):
        first, second = self.prepare("first"), self.prepare("second")
        orders = []
        for rater in self.raters:
            relative = Path("raters") / rater / "order.csv"
            self.assertEqual((first / relative).read_bytes(), (second / relative).read_bytes())
            order = h.read_csv(first / relative, h.ORDER_FIELDS)
            orders.append([row["clip_id"] for row in order])
            self.assertEqual(set(orders[-1]), {row["clip_id"] for row in self.rows})
            text = (first / relative).read_text(encoding="utf-8-sig")
            self.assertNotIn("MODEL_NAME", text)
            self.assertNotIn("source_model_name", text)
            self.assertNotIn("model_id", text)
            self.assertNotIn("case_id", text)
            self.assertTrue((first / "assets/clip_0001.mp4").is_file())
            for row in h.read_csv(first / "raters" / rater / "ratings.csv", h.RATING_FIELDS):
                self.assertTrue(all(row[score] == "" for score in h.SCORES))
        self.assertEqual(len({tuple(order) for order in orders}), 4)

    def test_original_manifest_order_does_not_change_permutations(self):
        first = self.prepare("first")
        h.write_csv(self.manifest, h.MANIFEST_FIELDS, list(reversed(self.rows)))
        second = self.prepare("second")
        for rater in self.raters:
            relative = Path("raters") / rater / "order.csv"
            self.assertEqual((first / relative).read_bytes(), (second / relative).read_bytes())

    def test_valid_submissions_and_preserve_input_bytes(self):
        files = self.submissions(self.prepare())
        self.fill_synthetic_test_scores(files)
        before = {rater: path.read_bytes() for rater, path in files.items()}
        report = h.validate_submissions(self.manifest, files, self.raters)
        self.assertTrue(report["valid"])
        self.assertEqual(report["expected_total_records"], 192)
        self.assertEqual(before, {rater: path.read_bytes() for rater, path in files.items()})

    def test_blank_success_scores_are_missing(self):
        files = self.submissions(self.prepare())
        report = h.validate_submissions(self.manifest, files, self.raters)
        self.assertFalse(report["valid"])
        self.assertEqual(len(report["errors"]), 47 * 4 * 3)
        self.assertEqual({error["code"] for error in report["errors"]}, {"missing_score"})

    def test_duplicate_unknown_missing_wrong_rater_and_invalid_scores(self):
        files = self.submissions(self.prepare())
        self.fill_synthetic_test_scores(files)
        rows = h.read_csv(files["A"], h.RATING_FIELDS)
        successful = [row for row in rows if row["clip_id"] != "clip_0048"]
        successful[0]["rater_id"] = "B"
        successful[0][h.SCORES[0]] = "3.0"
        successful[0][h.SCORES[1]] = "0"
        successful[0][h.SCORES[2]] = "6"
        remove_id = successful[1]["clip_id"]
        rows = [row for row in rows if row["clip_id"] != remove_id]
        rows.append(dict(successful[0]))
        rows.append({**successful[2], "clip_id": "clip_9999"})
        h.write_csv(files["A"], h.RATING_FIELDS, rows)
        report = h.validate_submissions(self.manifest, files, self.raters)
        self.assertFalse(report["valid"])
        self.assertEqual({error["code"] for error in report["errors"]},
                         {"wrong_rater", "invalid_score", "duplicate_clip", "unknown_clip", "missing_clip"})

    def test_failed_clip_requires_blank_scores_and_reason(self):
        files = self.submissions(self.prepare())
        self.fill_synthetic_test_scores(files)
        rows = h.read_csv(files["D"], h.RATING_FIELDS)
        failure = next(row for row in rows if row["clip_id"] == "clip_0048")
        failure.update(reference_fidelity="1", comment="")
        h.write_csv(files["D"], h.RATING_FIELDS, rows)
        report = h.validate_submissions(self.manifest, files, self.raters)
        self.assertEqual({error["code"] for error in report["errors"]},
                         {"failed_clip_has_score", "missing_failure_reason"})

    def test_reject_duplicate_manifest_and_mapping_column(self):
        h.write_csv(self.manifest, h.MANIFEST_FIELDS, self.rows + [self.rows[0]])
        with self.assertRaises(h.InputError):
            self.prepare()
        h.write_csv(self.manifest, h.MANIFEST_FIELDS + ["model_id"],
                    [{**self.rows[0], "model_id": "hidden"}])
        with self.assertRaises(h.InputError):
            self.prepare()

    def test_reject_model_identifying_source_paths_and_raw_failure_logs(self):
        self.rows[0]["video_path"] = "LTX/source_video.mp4"
        h.write_csv(self.manifest, h.MANIFEST_FIELDS, self.rows)
        with self.assertRaises(h.InputError):
            self.prepare()
        self.rows[0]["video_path"] = "videos/clip_0001.mp4"
        self.rows[-1]["failure_reason"] = "MODEL_NAME traceback"
        h.write_csv(self.manifest, h.MANIFEST_FIELDS, self.rows)
        with self.assertRaises(h.InputError):
            self.prepare()

    def test_reject_overwrite_and_wrong_batch(self):
        self.prepare()
        with self.assertRaises(h.InputError):
            self.prepare()
        with self.assertRaises(h.InputError):
            h.prepare(self.manifest, self.root / "new", self.raters, 6132, 96, "formal")

    def test_calibration_is_separate(self):
        h.write_csv(self.manifest, h.MANIFEST_FIELDS, self.rows[:3])
        out = self.root / "calibration"
        h.prepare(self.manifest, out, self.raters, 6132, 3, "calibration")
        self.assertEqual(json.loads((out / "preparation.json").read_text())["phase"], "calibration")
        with self.assertRaises(h.InputError):
            h.prepare(self.manifest, self.root / "formal", self.raters, 6132, 3, "formal")

    def test_reject_missing_media_and_missing_failure_reason(self):
        self.image.unlink()
        with self.assertRaises(h.InputError):
            self.prepare()
        self.rows[-1]["failure_reason"] = ""
        h.write_csv(self.manifest, h.MANIFEST_FIELDS, self.rows)
        with self.assertRaises(h.InputError):
            h.load_manifest(self.manifest)

    def test_cli_rejects_changed_manifest_before_reading_submissions(self):
        out = self.prepare()
        self.rows[0]["prompt"] = "Changed after preparation"
        h.write_csv(self.manifest, h.MANIFEST_FIELDS, self.rows)
        args = ["validate", "--manifest", str(self.manifest),
                "--preparation", str(out / "preparation.json"), "--ratings"]
        args += [f"{rater}={path}" for rater, path in self.submissions(out).items()]
        with contextlib.redirect_stderr(io.StringIO()) as stderr:
            self.assertEqual(h.main(args), 2)
        self.assertIn("不同", stderr.getvalue())

    def test_cli_passes_complete_scores_and_does_not_overwrite_report(self):
        out = self.prepare()
        files = self.submissions(out)
        self.fill_synthetic_test_scores(files)
        report = self.root / "check.json"
        args = ["validate", "--manifest", str(self.manifest),
                "--preparation", str(out / "preparation.json"),
                "--report", str(report), "--ratings"]
        args += [f"{rater}={path}" for rater, path in files.items()]
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(h.main(args), 0)
        before = report.read_bytes()
        with contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(h.main(args), 2)
        self.assertEqual(report.read_bytes(), before)

    def test_preparation_hash_and_rows_use_same_snapshot(self):
        original_hash = hashlib.sha256(self.manifest.read_bytes()).hexdigest()
        original_copy = h.shutil.copyfile

        def copy_while_source_manifest_changes(source, destination):
            self.rows[0]["prompt"] = "Modified while copying"
            h.write_csv(self.manifest, h.MANIFEST_FIELDS, self.rows)
            return original_copy(source, destination)

        with mock.patch.object(h.shutil, "copyfile", side_effect=copy_while_source_manifest_changes):
            out = self.prepare()
        metadata = json.loads((out / "preparation.json").read_text())
        self.assertEqual(metadata["manifest_sha256"], original_hash)
        orders = h.read_csv(out / "raters/A/order.csv", h.ORDER_FIELDS)
        self.assertEqual(next(row["prompt"] for row in orders if row["clip_id"] == "clip_0001"),
                         "A person walks, then turns.")


if __name__ == "__main__":
    unittest.main()
