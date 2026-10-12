import json
from pathlib import Path

import pytest
from PIL import Image

from src.experiments.utkface.data import FaceDataset, create_manifest, read_manifest


def test_original_split_is_reproducible_disjoint_and_evaluation_is_unpadded(
    tmp_path: Path,
) -> None:
    image_dir = tmp_path / "UTKFace"
    image_dir.mkdir()
    for decade in range(10):
        for index in range(25):
            Image.new("RGB", (12, 12)).save(
                image_dir / f"{decade * 10}_0_0_{index:04d}.jpg.chip.jpg"
            )
    Image.new("RGB", (12, 12)).save(image_dir / "105_0_0_9999.jpg.chip.jpg")
    manifest = tmp_path / "manifest.json"
    first = create_manifest(image_dir=image_dir, destination=manifest, split_seed=42)
    second = create_manifest(image_dir=image_dir, destination=manifest, split_seed=42)
    assert first == second
    assert {name: len(indices) for name, indices in first["splits"].items()} == {
        "train": 200,
        "val": 25,
        "test": 25,
    }
    assert first["excluded"]["outside_age_range"] == 1
    payload = read_manifest(path=manifest)
    shards = [
        FaceDataset(
            manifest_path=manifest,
            payload=payload,
            split="val",
            limit=0,
            rank=rank,
            world_size=2,
        )
        for rank in range(2)
    ]
    assert [len(shard) for shard in shards] == [13, 12]
    assert not {record.relative_path for record in shards[0].records} & {
        record.relative_path for record in shards[1].records
    }
    assert shards[0][0][0].shape == (3, 224, 224)
    payload["records"][0]["age"] += 1
    manifest.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="fingerprint"):
        read_manifest(path=manifest)
