"""Shared, dependency-free validation for member C's LTX-Video runs."""
from __future__ import annotations

import csv
import hashlib
import io
import json
import re
from pathlib import Path

UPSTREAM_COMMIT = "50f0aef1297574a4b29acbdb874e3a363a6d5f2e"
MODEL_ID = "ltxv-2b-0.9.8-distilled"
MODEL_REPO = "Lightricks/LTX-Video"
MODEL_REVISION = "8984fa25007f376c1a299016d0957a37a2f797bb"
TEXT_REPO = "PixArt-alpha/PixArt-XL-2-1024-MS"
TEXT_REVISION = "b89adadeccd9ead2adcb9fa2825d3fabec48d404"
WEIGHT_FILES = (
    "ltxv-2b-0.9.8-distilled.safetensors",
    "ltxv-spatial-upscaler-0.9.8.safetensors",
)
TEXT_FILES = (
    "text_encoder/config.json",
    "text_encoder/model-00001-of-00002.safetensors",
    "text_encoder/model-00002-of-00002.safetensors",
    "text_encoder/model.safetensors.index.json",
    "tokenizer/added_tokens.json",
    "tokenizer/special_tokens_map.json",
    "tokenizer/spiece.model",
    "tokenizer/tokenizer_config.json",
)
IMAGE_ALIASES = ("image_path", "image", "first_frame", "first_frame_path", "reference_image")


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def object_digest(obj: object) -> str:
    return hashlib.sha256(json.dumps(obj, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def nearest_ltx_frames(target: int) -> int:
    """Nearest positive 8*n+1 frame count; a tie chooses the smaller value."""
    if target < 1:
        raise ValueError("Target frame count must be positive")
    lower = ((target - 1) // 8) * 8 + 1
    return min((lower, lower + 8), key=lambda value: (abs(value - target), value))


def read_cases(csv_path: Path, image_root: Path | None = None, *, csv_bytes: bytes | None = None,
               protocol: str = "legacy") -> list[dict]:
    """Preserve CSV order and prompt text; paths default to the CSV directory."""
    root = (image_root or csv_path.parent).resolve()
    snapshot = csv_path.read_bytes() if csv_bytes is None else csv_bytes
    with io.StringIO(snapshot.decode("utf-8-sig"), newline="") as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames or []
        if len(fields) != len(set(fields)):
            raise ValueError("Duplicate CSV column names")
        if protocol not in ("legacy", "shared24"):
            raise ValueError(f"Unknown protocol: {protocol}")
        required = ({"image_type", "image_path", "prompt", "seed", "frame_num"} if protocol == "shared24"
                    else {"case_id", "category", "prompt", "seed"})
        missing = required - set(fields)
        aliases = (["image_path"] if "image_path" in fields else []) if protocol == "shared24" else [key for key in IMAGE_ALIASES if key in fields]
        if missing or not aliases:
            raise ValueError(f"Required {protocol} fields: {sorted(required)} plus an image path; missing={sorted(missing)}, image columns={aliases}")
        rows, seen = [], set()
        for number, row in enumerate(reader, 2):
            if None in row or any(row.get(key) is None for key in fields):
                raise ValueError(f"Malformed CSV row {number}")
            if protocol == "shared24":
                match = re.match(r"^(0[1-9]|1[0-9]|2[0-4])_", Path(row["image_path"].strip()).name)
                if not match:
                    raise ValueError(f"shared24 image filename must start with 01_ through 24_ at row {number}")
                case_id = match.group(1)
            else:
                case_id = row["case_id"].strip()
            if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,99}", case_id) or case_id in seen:
                raise ValueError(f"Invalid or duplicate case_id at row {number}: {case_id!r}")
            seen.add(case_id)
            category, prompt = (row["image_type"] if protocol == "shared24" else row["category"].strip()), row["prompt"]
            if not category.strip() or not prompt.strip():
                raise ValueError(f"Empty category/prompt for {case_id}")
            raw_seed = row["seed"].strip()
            if not raw_seed.isascii() or not raw_seed.isdecimal() or not 0 <= int(raw_seed) < 2**32:
                raise ValueError(f"seed for {case_id} must be an explicit integer in [0, 2**32)")
            values = [row[key].strip() for key in aliases if row[key].strip()]
            if not values or len(set(values)) != 1:
                raise ValueError(f"Missing or conflicting image path aliases for {case_id}")
            path = Path(values[0]).expanduser()
            path = (path if path.is_absolute() else root / path).resolve()
            if not path.is_file():
                raise ValueError(f"Missing shared first image for {case_id}: {path}")
            case = {"case_id": case_id, "category": category, "image_path": str(path),
                    "image_sha256": digest(path), "prompt": prompt, "seed": int(raw_seed)}
            if protocol == "shared24":
                raw_frames = row["frame_num"].strip()
                if raw_frames not in ("128", "512"):
                    raise ValueError(f"shared24 frame_num for {case_id} must be 128 or 512")
                case["target_frame_num"] = int(raw_frames)
                case["requested_frame_num"] = nearest_ltx_frames(int(raw_frames))
            rows.append(case)
    if not rows:
        raise ValueError("cases.csv is empty")
    return rows


def read_manifest_snapshot(csv_path: Path, image_root: Path | None = None, *,
                           protocol: str = "legacy") -> tuple[list[dict], str]:
    """Parse and hash the same bytes, even if the shared CSV changes afterwards."""
    snapshot = csv_path.read_bytes()
    return read_cases(csv_path, image_root, csv_bytes=snapshot, protocol=protocol), hashlib.sha256(snapshot).hexdigest()


def select_cases(rows: list[dict], phase: str, per_category: int, protocol: str = "legacy") -> list[dict]:
    if phase == "smoke":
        if len(rows) != 1:
            raise ValueError("Environment smoke CSV must contain exactly 1 non-evaluation case")
        return rows
    if phase == "pilot":
        if len(rows) != 3:
            qualifier = "technical-check" if protocol == "shared24" else "independent calibration"
            raise ValueError(f"Pilot CSV must contain exactly 3 {qualifier} cases")
        return rows
    if protocol == "shared24":
        targets = [row["target_frame_num"] for row in rows]
        if (len(rows) != 24 or targets.count(128) != 18 or targets.count(512) != 6 or
                {row["case_id"] for row in rows} != {f"{i:02d}" for i in range(1, 25)}):
            raise ValueError("Formal shared24 requires IDs 01..24 exactly once, with 18 rows of 128 frames and 6 rows of 512 frames")
        return rows
    counts = {}
    for row in rows:
        counts[row["category"]] = counts.get(row["category"], 0) + 1
    if len(counts) != 6 or any(count < per_category for count in counts.values()):
        raise ValueError(f"Formal batch needs 6 categories with >= {per_category} cases each: {counts}")
    taken, selected = {}, []
    for row in rows:
        category = row["category"]
        taken[category] = taken.get(category, 0) + 1
        if taken[category] <= per_category:
            selected.append(row)
    return selected


def expected_assets() -> list[str]:
    return ["ltx/" + name for name in WEIGHT_FILES] + ["text/" + name for name in TEXT_FILES]


def verify_assets(root: Path) -> dict:
    manifest = json.loads((root / "assets.json").read_text(encoding="utf-8"))
    if manifest.get("model_revision") != MODEL_REVISION or manifest.get("text_revision") != TEXT_REVISION:
        raise ValueError("Asset revisions differ from the audited revisions")
    files = manifest.get("files", {})
    if set(files) != set(expected_assets()):
        raise ValueError("Asset manifest does not contain exactly the required model/upscaler/T5/tokenizer files")
    for name, sha in files.items():
        if digest(root / name) != sha:
            raise ValueError(f"Asset checksum mismatch: {name}")
    return manifest
