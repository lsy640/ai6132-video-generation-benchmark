"""Receipt already downloaded, revision-pinned local assets; never download weights."""
import argparse
from pathlib import Path
import re

from .io import sha256, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--revision', nargs='+', required=True, metavar='REPO=COMMIT')
    args = parser.parse_args()
    root = args.root.resolve()
    target = root / 'assets.json'
    if target.exists():
        parser.error('Never overwrite a prepared asset receipt')
    revisions = {}
    for item in args.revision:
        repo, sep, revision = item.partition('=')
        if not sep or not re.fullmatch(r'[0-9a-f]{40}', revision) or repo in revisions:
            parser.error('Each source needs a unique repository and a full revision SHA')
        revisions[repo] = revision
    files = {str(p.relative_to(root)): sha256(p) for p in sorted(root.rglob('*'))
             if p.is_file() and not any(s.startswith('.') for s in p.relative_to(root).parts)}
    if not files:
        parser.error('No local assets; download fixed revisions in a separate batch job first')
    write_json(target, {'repositories': revisions, 'files': files})
    print(f'Receipted {len(files)} local files; no download performed')


if __name__ == '__main__':
    main()
