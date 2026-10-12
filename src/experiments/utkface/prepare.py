import argparse
import hashlib
import json
import tarfile
import urllib.request
from pathlib import Path

from src.experiments.utkface.data import create_manifest

DATASET_REVISION = "e80f21fb21fa3631380c3cb634b79be3ac2f4cba"
DATASET_URL = f"https://huggingface.co/datasets/py97/UTKFace-Cropped/resolve/{DATASET_REVISION}/UTKFace.tar.gz"
DATASET_SHA256 = "2c0655397b498e81b82d685ee74bee8c2e4364a5897caf3afec6ea48929024a7"


def prepare(*, destination: Path, split_seed: int) -> dict:
    destination.mkdir(parents=True, exist_ok=True)
    archive = destination / "UTKFace.tar.gz"
    if not archive.exists():
        urllib.request.urlretrieve(DATASET_URL, archive)
    with archive.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if digest != DATASET_SHA256:
        raise ValueError(f"Archive checksum mismatch: {digest}")
    with tarfile.open(archive) as stream:
        stream.extractall(destination, filter="data")
    payload = create_manifest(
        image_dir=destination / "UTKFace",
        destination=destination / "manifest.json",
        split_seed=split_seed,
    )
    payload["source"] = {
        "url": DATASET_URL,
        "sha256": digest,
        "revision": DATASET_REVISION,
    }
    (destination / "manifest.json").write_text(json.dumps(payload, indent=2) + "\n")
    return {
        "fingerprint": payload["fingerprint"],
        "splits": {name: len(indices) for name, indices in payload["splits"].items()},
        "excluded": payload["excluded"],
        "source": payload["source"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--split-seed", type=int, default=42)
    args = parser.parse_args()
    print(
        json.dumps(
            prepare(destination=args.destination, split_seed=args.split_seed), indent=2
        )
    )


if __name__ == "__main__":
    main()
