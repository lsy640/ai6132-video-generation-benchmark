"""Shared, dependency-free validation for member C's LTX-Video runs."""
from __future__ import annotations

import csv
import hashlib
import io
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ai6132_benchmark.dataset import FIELDS, validate_profile

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


def read_cases(csv_path: Path, image_root: Path | None = None, *, csv_bytes: bytes | None = None) -> list[dict]:
    """Preserve CSV order and prompt text; paths default to the CSV directory."""
    root = (image_root or csv_path.parent).resolve()
    snapshot = csv_path.read_bytes() if csv_bytes is None else csv_bytes
    with io.StringIO(snapshot.decode("utf-8-sig"), newline="") as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames or []
        if len(fields) != len(set(fields)):
            raise ValueError("Duplicate CSV column names")
        if fields != FIELDS:
            raise ValueError(f"Canonical manifest requires fields in order: {','.join(FIELDS)}")
        rows, seen = [], set()
        for number, row in enumerate(reader, 2):
            if None in row or any(row.get(key) is None for key in fields):
                raise ValueError(f"Malformed CSV row {number}")
            case_id = row["case_id"].strip()
            if not re.fullmatch(r"case_[0-9]{4}", case_id) or case_id in seen:
                raise ValueError(f"Invalid or duplicate case_id at row {number}: {case_id!r}")
            seen.add(case_id)
            category, prompt = row["category"].strip(), row["prompt"]
            if not category or not prompt.strip():
                raise ValueError(f"Empty category/prompt for {case_id}")
            raw_seed = row["seed"].strip()
            if not raw_seed.isascii() or not raw_seed.isdecimal() or not 0 <= int(raw_seed) < 2**32:
                raise ValueError(f"seed for {case_id} must be an explicit integer in [0, 2**32)")
            if not row['image_path'].strip():
                raise ValueError(f'Missing image_path for {case_id}')
            path = Path(row['image_path']).expanduser()
            path = (path if path.is_absolute() else root / path).resolve()
            if not path.is_file():
                raise ValueError(f"Missing shared first image for {case_id}: {path}")
            item = {"case_id": case_id, "category": category, "image_path": str(path),
                    "image_sha256": digest(path), "prompt": prompt, "seed": int(raw_seed)}
            item.update(target_frames=int(row['target_frames']), source_frame_num=int(row['source_frame_num']))
            validate_profile([item], formal=False)
            rows.append(item)
    if not rows:
        raise ValueError("cases.csv is empty")
    return rows


def read_manifest_snapshot(csv_path: Path, image_root: Path | None = None) -> tuple[list[dict], str]:
    """Parse and hash the same bytes, even if the shared CSV changes afterwards."""
    snapshot = csv_path.read_bytes()
    return read_cases(csv_path, image_root, csv_bytes=snapshot), hashlib.sha256(snapshot).hexdigest()


def select_cases(rows: list[dict], phase: str) -> list[dict]:
    """Use the entire canonical manifest; no category-based subset selection."""
    validate_profile(rows, formal=phase == 'formal')
    if phase == 'formal':
        return rows
    if phase == "smoke":
        if len(rows) != 1:
            raise ValueError("Environment smoke CSV must contain exactly 1 non-evaluation case")
        return rows
    if phase == "pilot":
        if len(rows) != 3:
            raise ValueError("Pilot CSV must contain exactly 3 independent calibration cases")
        if {row['target_frames'] for row in rows} != {129, 513}:
            raise ValueError('Pilot must include both 129-frame and 513-frame cases')
        return rows
    raise ValueError(f'Unknown phase: {phase}')


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
