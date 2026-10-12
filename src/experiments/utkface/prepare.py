from __future__ import annotations

import argparse
import hashlib
from collections import Counter
from pathlib import Path

from PIL import Image

from src.experiments.utkface.canonical import get_canonical_config
from src.experiments.utkface.data import find_samples, make_manifest, make_splits
from src.experiments.utkface.train import atomic_json


def audit_dataset(*, directory: Path, output: Path) -> dict:
    config = get_canonical_config(
        method="ce", data_directory=directory, output_directory=output.parent, seed=42
    )
    samples = find_samples(directory=directory)
    digest = hashlib.sha256()
    shapes = Counter()
    for sample in samples:
        image_hash = hashlib.sha256(sample.path.read_bytes()).digest()
        digest.update(sample.path.name.encode())
        digest.update(image_hash)
        with Image.open(sample.path) as image:
            shapes[image.size] += 1
            image.verify()
    manifest = make_manifest(
        samples=samples, splits=make_splits(samples=samples, config=config.data)
    )
    report = {
        "directory": str(directory.resolve()),
        "images": len(samples),
        "content_sha256": digest.hexdigest(),
        "age_counts": dict(sorted(Counter(sample.age for sample in samples).items())),
        "dimensions": {
            f"{width}x{height}": count for (width, height), count in shapes.items()
        },
        "manifest": manifest,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(path=output, value=report)
    print(
        {
            key: value
            for key, value in report.items()
            if key not in ("manifest", "age_counts")
        },
        flush=True,
    )
    print(manifest["counts"], flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify UTKFace images, hash dataset contents, and record original splits"
    )
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    audit_dataset(directory=args.data_dir, output=args.output)


if __name__ == "__main__":
    main()
