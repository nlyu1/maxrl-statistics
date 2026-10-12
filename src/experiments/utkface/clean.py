import argparse
import json
from pathlib import Path

from src.experiments.utkface.data import (
    create_manifest,
    read_manifest,
    training_indices,
)
from src.experiments.utkface.run import write_json


def clean(*, manifest: Path, destination: Path) -> dict:
    original = read_manifest(path=manifest)
    image_dirs = {
        (manifest.parent / r["relative_path"]).parent for r in original["records"]
    }
    if len(image_dirs) != 1:
        raise ValueError("UTKFace preparation expects a single image directory")
    destination.mkdir(parents=True, exist_ok=True)
    payload = create_manifest(
        image_dir=image_dirs.pop(),
        destination=destination / "manifest.json",
        split_seed=42,
        deduplicate=True,
    )
    payload["source"] = original["source"]
    write_json(path=destination / "manifest.json", payload=payload)
    subsets = {
        str(size): training_indices(payload=payload, limit=size, subset_seed=2026)
        for size in (1024, 4096)
    }
    write_json(
        path=destination / "training_subsets.json",
        payload={
            "data_fingerprint": payload["fingerprint"],
            "subset_seed": 2026,
            "indices": subsets,
        },
    )
    return {
        "fingerprint": payload["fingerprint"],
        "counts": {k: len(v) for k, v in payload["splits"].items()},
        "preprocessing": {
            k: v
            for k, v in payload["preprocessing"].items()
            if k != "conflicting_groups"
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            clean(manifest=args.manifest, destination=args.destination), indent=2
        )
    )


if __name__ == "__main__":
    main()
