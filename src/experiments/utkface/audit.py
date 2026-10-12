import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from src.experiments.utkface.data import read_manifest, rgb_fingerprint
from src.experiments.utkface.run import write_json


def duplicate_summary(*, groups: dict[str, list[dict]]) -> dict:
    duplicates = [group for group in groups.values() if len(group) > 1]
    crossed = [
        group for group in duplicates if len({record["split"] for record in group}) > 1
    ]
    return {
        "duplicate_groups": len(duplicates),
        "duplicate_images": sum(len(group) for group in duplicates),
        "cross_split_groups": len(crossed),
        "cross_split_images": sum(len(group) for group in crossed),
        "cross_split_images_by_split": dict(
            Counter(record["split"] for group in crossed for record in group)
        ),
        "conflicting_age_groups": sum(
            len({record["age"] for record in group}) > 1 for group in duplicates
        ),
        "cross_split_examples": crossed[:10],
    }


def audit(*, manifest: Path, output_dir: Path) -> dict:
    payload = read_manifest(path=manifest)
    membership = {
        index: split
        for split, indices in payload["splits"].items()
        for index in indices
    }
    byte_groups = defaultdict(list)
    pixel_groups = defaultdict(list)
    for index, record in enumerate(payload["records"]):
        path = manifest.parent / record["relative_path"]
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        entry = {**record, "split": membership[index]}
        byte_groups[digest].append(entry)
        pixels = rgb_fingerprint(path=path)
        pixel_groups[pixels].append(entry)
    age_arrays = {
        split: np.array([payload["records"][i]["age"] for i in indices])
        for split, indices in payload["splits"].items()
    }
    mean_age, median_age = age_arrays["train"].mean(), np.median(age_arrays["train"])
    singleton_test = [
        group[0]["relative_path"]
        for group in pixel_groups.values()
        if len(group) == 1 and group[0]["split"] == "test"
    ]
    result = {
        "fingerprint": payload["fingerprint"],
        "source": payload.get("source", {}),
        "excluded": payload["excluded"],
        "counts": {split: len(ages) for split, ages in age_arrays.items()},
        "decades": {
            split: dict(sorted(Counter((ages // 10 * 10).tolist()).items()))
            for split, ages in age_arrays.items()
        },
        "constant_mean_age": float(mean_age),
        "constant_median_age": float(median_age),
        "constant_baselines": {
            split: {
                "mean_mae": float(np.abs(ages - mean_age).mean()),
                "mean_rmse": float(np.square(ages - mean_age).mean() ** 0.5),
                "median_mae": float(np.abs(ages - median_age).mean()),
            }
            for split, ages in age_arrays.items()
        },
        "identical_bytes": duplicate_summary(groups=byte_groups),
        "identical_rgb_pixels": duplicate_summary(groups=pixel_groups),
        "singleton_test_count": len(singleton_test),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(path=output_dir / "data_audit.json", payload=result)
    write_json(
        path=output_dir / "test_singletons.json",
        payload={
            "data_fingerprint": payload["fingerprint"],
            "rule": "Test images with a unique decoded RGB pixel hash across the entire dataset",
            "relative_paths": singleton_test,
        },
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(audit(manifest=args.manifest, output_dir=args.output_dir), indent=2)
    )


if __name__ == "__main__":
    main()
