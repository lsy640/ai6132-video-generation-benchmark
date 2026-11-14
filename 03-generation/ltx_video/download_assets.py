#!/usr/bin/env python3
"""Explicit, revision-pinned download; never called automatically by the runner."""
import argparse
from pathlib import Path

from common import (MODEL_REPO, MODEL_REVISION, TEXT_REPO, TEXT_REVISION,
                    WEIGHT_FILES, TEXT_FILES, digest, expected_assets, write_json)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, required=True, help="Large persistent storage, not a small home quota")
    args = parser.parse_args()
    from huggingface_hub import hf_hub_download

    root = args.assets.expanduser().resolve()
    for repo, revision, names, folder in (
        (MODEL_REPO, MODEL_REVISION, WEIGHT_FILES, "ltx"),
        (TEXT_REPO, TEXT_REVISION, TEXT_FILES, "text"),
    ):
        for name in names:
            hf_hub_download(repo_id=repo, revision=revision, filename=name, local_dir=root / folder)
    manifest = {"model_repo": MODEL_REPO, "model_revision": MODEL_REVISION,
                "text_repo": TEXT_REPO, "text_revision": TEXT_REVISION,
                "files": {name: digest(root / name) for name in expected_assets()}}
    write_json(root / "assets.json", manifest)
    print(f"Downloaded and checksummed {len(manifest['files'])} required files: {root}")


if __name__ == "__main__":
    main()
