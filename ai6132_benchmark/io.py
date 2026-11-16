"""Small dependency-free helpers; no model imports or automatic downloads."""
import csv
import hashlib
import io
import json
from pathlib import Path


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read_csv(path):
    data = Path(path).read_bytes()
    reader = csv.DictReader(io.StringIO(data.decode('utf-8-sig'), newline=''), strict=True)
    fields = reader.fieldnames or []
    if not fields or len(set(fields)) != len(fields):
        raise ValueError('Missing or duplicate CSV headers')
    rows = list(reader)
    if any(None in row or any(v is None for v in row.values()) for row in rows):
        raise ValueError('Malformed CSV row')
    return fields, rows, hashlib.sha256(data).hexdigest()


def write_csv(path, fields, rows):
    with Path(path).open('x', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, data):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('x', encoding='utf-8') as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    temporary.replace(path)


def fingerprint(data):
    return hashlib.sha256(json.dumps(data, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def human_module():
    import importlib.util
    filename = Path(__file__).resolve().parents[1] / '05-human-evaluation/human_eval.py'
    spec = importlib.util.spec_from_file_location('ai6132_human_eval', filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def evaluation_inputs(manifest, mapping, phase='formal'):
    if phase not in {'formal', 'pilot', 'diagnostic'}:
        raise ValueError('Unknown evaluation phase')
    from .dataset import CATEGORIES
    rows = human_module().load_manifest(manifest, check_files=True)
    fields, links, _ = read_csv(mapping)
    if not {'clip_id', 'case_id', 'model_id', 'category', 'target_frames'} <= set(fields):
        raise ValueError('Mapping must retain case/model/category/target_frames')
    if len({r['clip_id'] for r in links}) != len(links) or {r['clip_id'] for r in links} != {r['clip_id'] for r in rows}:
        raise ValueError('Mapping must match the anonymous manifest exactly')
    if any(r['category'] not in CATEGORIES or int(r['target_frames']) not in (129, 513) for r in links):
        raise ValueError('Unknown category or frame profile')
    keys = [(r['case_id'], r['model_id']) for r in links]
    if len(set(keys)) != len(keys):
        raise ValueError('Duplicate case/model mapping')
    if phase in {'formal', 'diagnostic'}:
        from collections import Counter
        from .dataset import validate_profile
        models = {r['model_id'] for r in links}
        cases = {r['case_id'] for r in links}
        if phase == 'formal' and (len(models) != 4 or len(cases) != 24 or len(rows) != 96):
            raise ValueError('Formal evaluation requires 24 cases x 4 models')
        if phase == 'diagnostic' and (not 1 <= len(models) <= 4 or len(cases) != 24 or len(rows) != 24 * len(models)):
            raise ValueError('Diagnostic evaluation requires every model to retain all 24 cases')
        for case_id in cases:
            group = [r for r in links if r['case_id'] == case_id]
            if {r['model_id'] for r in group} != models or len({(r['category'], r['target_frames']) for r in group}) != 1:
                raise ValueError('Different models must share the same case profile')
        distinct = [next(r for r in links if r['case_id'] == key) for key in cases]
        validate_profile([{**r, 'source_frame_num': int(r['target_frames']) - 1} for r in distinct])
    lookup = {r['clip_id']: r for r in links}
    receipt = {'manifest_sha256': sha256(manifest), 'mapping_sha256': sha256(mapping), 'files': {}}
    for row in rows:
        for field in ('image_path', 'video_path'):
            if row[field]:
                receipt['files'][row[field]] = sha256(Path(manifest).resolve().parent / row[field])
    return rows, lookup, receipt
