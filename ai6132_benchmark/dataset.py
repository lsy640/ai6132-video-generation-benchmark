"""Preserve A's source CSV; derive and verify the mixed-length canonical manifest."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import re

from .io import read_csv, sha256, write_csv, write_json

SUBJECT = {'single-human', 'multiple-human', 'animal', 'transportation', 'food', 'plant', 'other'}
BACKGROUND = {'architecture', 'scenery', 'indoor', 'abstract'}
CATEGORIES = SUBJECT | BACKGROUND
FIELDS = ['case_id', 'category', 'image_path', 'prompt', 'seed', 'source_frame_num', 'target_frames']
COUNTS = {name: 3 if name in {'single-human', 'multiple-human'} else 2 for name in CATEGORIES}


def validate_profile(rows, formal=True):
    for row in rows:
        if not {'source_frame_num', 'target_frames'} <= row.keys():
            raise ValueError('Canonical manifest requires source_frame_num and target_frames on every case')
        if row['category'] not in CATEGORIES:
            raise ValueError(f"Unknown category: {row['category']}")
        if int(row['source_frame_num']) not in (128, 512) or int(row['target_frames']) != int(row['source_frame_num']) + 1:
            raise ValueError('Expected source/target frame mapping 128/129 or 512/513')
    if formal and (Counter(row['category'] for row in rows) != COUNTS or
                   Counter(int(row['target_frames']) for row in rows) != {129: 18, 513: 6}):
        raise ValueError('Formal manifest must preserve A: 24 cases / 11 categories / 18 short + 6 long')


def load_canonical(path, decode=False, formal=True):
    path = Path(path).resolve()
    fields, rows, csv_sha = read_csv(path)
    if fields != FIELDS:
        raise ValueError(f'Canonical CSV must have fields {FIELDS}')
    seen = set()
    images = {}
    for row in rows:
        case = row['case_id']
        if not re.fullmatch(r'case_[0-9]{4}', case) or case in seen:
            raise ValueError(f'Invalid/duplicate case_id: {case}')
        seen.add(case)
        if not row['prompt'].strip() or not row['seed'].isascii() or not row['seed'].isdecimal() or not 0 <= int(row['seed']) < 2**32:
            raise ValueError(f'Invalid prompt/seed: {case}')
        image = (path.parent / row['image_path']).resolve()
        if not image.is_file():
            raise ValueError(f'Missing reference: {case}')
        info = {'sha256': sha256(image)}
        if decode:
            from PIL import Image
            with Image.open(image) as pic:
                pic.load()  # Full decode, not only JPEG headers.
                if pic.width * 9 != pic.height * 16:
                    raise ValueError(f'Expected a shared 16:9 crop: {case}')
                info.update(width=pic.width, height=pic.height)
        images[case] = info
    validate_profile(rows, formal=formal)
    receipt_path = path.with_suffix('.receipt.json')
    if receipt_path.exists():
        frozen = json.loads(receipt_path.read_text())
        if frozen['canonical_sha256'] != csv_sha or any(
                images[k]['sha256'] != frozen['images'][k]['sha256'] for k in images):
            raise ValueError('Canonical input differs from its frozen receipt')
    return rows, {'canonical_sha256': csv_sha, 'images': images}


def canonicalize(source, output):
    source, output = Path(source).resolve(), Path(output).resolve()
    if source == output or output.exists() or output.with_suffix('.receipt.json').exists():
        raise ValueError('Use a new output; never overwrite source or a frozen manifest')
    fields, original, source_sha = read_csv(source)
    if fields != ['image_type', 'image_path', 'prompt', 'seed', 'frame_num']:
        raise ValueError('Unexpected A source schema')
    rows = []
    for row in original:
        if not row['prompt'].strip() or not row['seed'].isascii() or not row['seed'].isdecimal() or not 0 <= int(row['seed']) < 2**32:
            raise ValueError('Invalid source prompt/seed')
        match = re.match(r'([0-9]{2})_', Path(row['image_path']).name)
        if not match:
            raise ValueError('Image names must retain A\'s two-digit identity prefix')
        image = (source.parent / row['image_path']).resolve()
        rows.append(dict(zip(FIELDS, [f'case_{int(match[1]):04d}', row['image_type'],
                                     os.path.relpath(image, output.parent), row['prompt'], row['seed'],
                                     row['frame_num'], str(int(row['frame_num']) + 1)])))
    validate_profile(rows)
    output.parent.mkdir(parents=True, exist_ok=True)
    # Validate all images before publishing the CSV. Output remains distinct from source.
    from PIL import Image
    images = {}
    if len({r['case_id'] for r in rows}) != len(rows):
        raise ValueError('Duplicate image identity prefix')
    for row in rows:
        image = (output.parent / row['image_path']).resolve()
        with Image.open(image) as pic:
            pic.load()
            if pic.width * 9 != pic.height * 16:
                raise ValueError(f"Not a shared 16:9 image: {row['case_id']}")
            images[row['case_id']] = dict(sha256=sha256(image), width=pic.width, height=pic.height)
    if sha256(source) != source_sha:
        raise ValueError('A source changed during adaptation')
    write_csv(output, FIELDS, rows)
    receipt = {'schema_version': 1, 'source_sha256': source_sha, 'canonical_sha256': sha256(output),
               'mapping': {'128': 129, '512': 513}, 'images': images,
               'source_sampling_method': 'not supplied; do not claim verified random sampling'}
    write_json(output.with_suffix('.receipt.json'), receipt)
    return receipt


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path)
    p.add_argument('--out', type=Path)
    p.add_argument('--verify', type=Path)
    args = p.parse_args()
    if args.verify:
        rows, actual = load_canonical(args.verify, decode=True)
        frozen = json.loads(args.verify.with_suffix('.receipt.json').read_text())
        if actual['canonical_sha256'] != frozen['canonical_sha256'] or actual['images'] != frozen['images']:
            raise ValueError('Canonical manifest or images differ from the frozen receipt')
        print(json.dumps({'verified': True, 'cases': len(rows), 'short': 18, 'long': 6}))
    elif args.source and args.out:
        canonicalize(args.source, args.out)
        print(json.dumps({'created': str(args.out), 'cases': 24, 'short': 18, 'long': 6}))
    else:
        p.error('Use --source CSV --out NEW_CSV or --verify CANONICAL_CSV')


if __name__ == '__main__':
    main()
