import hashlib
import json
import os
import random
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from pydantic import ConfigDict
from pydantic.dataclasses import dataclass
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torchvision import transforms

from src.experiments.utkface.losses import MAX_AGE


@dataclass(kw_only=True, frozen=True, config=ConfigDict(arbitrary_types_allowed=True))
class FaceRecord:
    relative_path: str
    age: int


def manifest_fingerprint(*, records: list[dict], splits: dict[str, list[int]]) -> str:
    payload = json.dumps(
        {"records": records, "splits": splits}, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def rgb_fingerprint(*, path: Path) -> str:
    with Image.open(path) as image:
        rgb = image.convert("RGB")
        return hashlib.sha256(str(rgb.size).encode() + rgb.tobytes()).hexdigest()


def create_manifest(
    *, image_dir: Path, destination: Path, split_seed: int, deduplicate: bool
) -> dict:
    pattern = re.compile(r"^(\d+)_\d+_\d+_\d+\.jpg(?:\.chip\.jpg)?$")
    records = []
    excluded = {"outside_age_range": 0, "malformed_filename": 0}
    for path in sorted(image_dir.glob("*.jpg")):
        match = pattern.fullmatch(path.name)
        if match is None:
            excluded["malformed_filename"] += 1
            continue
        age = int(match.group(1))
        if age > MAX_AGE:
            excluded["outside_age_range"] += 1
            continue
        with Image.open(path) as image:
            image.verify()
        records.append({
            "relative_path": Path(os.path.relpath(path, destination.parent)).as_posix(),
            "age": age,
        })
    if not records:
        raise ValueError(f"No valid UTKFace images in {image_dir}")
    preprocessing = {"deduplicated_before_split": deduplicate}
    if deduplicate:
        groups = defaultdict(list)
        for record in records:
            groups[
                rgb_fingerprint(path=destination.parent / record["relative_path"])
            ].append(record)
        conflicts = [
            group for group in groups.values() if len({r["age"] for r in group}) > 1
        ]
        consistent = [
            group for group in groups.values() if len({r["age"] for r in group}) == 1
        ]
        preprocessing.update({
            "valid_before_deduplication": len(records),
            "conflicting_groups_dropped": len(conflicts),
            "conflicting_images_dropped": sum(map(len, conflicts)),
            "consistent_duplicate_copies_dropped": sum(
                len(group) - 1 for group in consistent
            ),
            "conflicting_groups": conflicts,
        })
        records = sorted(
            (group[0] for group in consistent), key=lambda r: r["relative_path"]
        )
    indices = list(range(len(records)))
    decades = [record["age"] // 10 for record in records]
    train, temporary = train_test_split(
        indices, test_size=0.2, random_state=split_seed, stratify=decades
    )
    val, test = train_test_split(
        temporary,
        test_size=0.5,
        random_state=split_seed,
        stratify=[decades[i] for i in temporary],
    )
    splits = {"train": train, "val": val, "test": test}
    payload = {
        "records": records,
        "splits": splits,
        "split_seed": split_seed,
        "excluded": excluded,
        "preprocessing": preprocessing,
        "fingerprint": manifest_fingerprint(records=records, splits=splits),
    }
    destination.write_text(json.dumps(payload, indent=2) + "\n")
    return payload


def training_indices(*, payload: dict, limit: int, subset_seed: int) -> list[int]:
    indices = sorted(payload["splits"]["train"])
    if limit > len(indices):
        raise ValueError(f"Requested {limit} training images from {len(indices)}")
    if not limit or limit == len(indices):
        return indices
    if limit <= 4096 < len(indices):
        indices, _ = train_test_split(
            indices,
            train_size=4096,
            random_state=subset_seed,
            stratify=[payload["records"][i]["age"] // 10 for i in indices],
        )
    if limit < len(indices):
        indices, _ = train_test_split(
            indices,
            train_size=limit,
            random_state=subset_seed,
            stratify=[payload["records"][i]["age"] // 10 for i in indices],
        )
    return sorted(indices)


def read_manifest(*, path: Path) -> dict:
    payload = json.loads(path.read_text())
    records, splits = payload["records"], payload["splits"]
    if manifest_fingerprint(records=records, splits=splits) != payload["fingerprint"]:
        raise ValueError("Manifest fingerprint mismatch")
    indices = [index for split in ("train", "val", "test") for index in splits[split]]
    if sorted(indices) != list(range(len(records))):
        raise ValueError("Splits must partition the dataset without overlaps")
    if any(not 0 <= record["age"] <= MAX_AGE for record in records):
        raise ValueError("Ages must be integers in [0, 99]")
    return payload


def image_transform(*, training: bool) -> transforms.Compose:
    spatial = (
        [
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        ]
        if training
        else [transforms.Resize(256), transforms.CenterCrop(224)]
    )
    return transforms.Compose([
        *spatial,
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])


class FaceDataset(Dataset):
    def __init__(
        self,
        *,
        manifest_path: Path,
        payload: dict,
        split: str,
        limit: int,
        rank: int,
        world_size: int,
        subset_seed: int,
    ) -> None:
        indices = payload["splits"][split]
        if limit > len(indices):
            raise ValueError(
                f"Requested {limit} examples from {len(indices)} in {split}"
            )
        indices = (
            training_indices(payload=payload, limit=limit, subset_seed=subset_seed)
            if split == "train"
            else indices[:limit]
            if limit
            else indices
        )
        if split != "train":
            indices = indices[rank::world_size]
        self.records = [FaceRecord(**payload["records"][i]) for i in indices]
        self.indices = indices
        self.root = manifest_path.parent
        self.transform = image_transform(training=split == "train")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, int]:
        record = self.records[index]
        with Image.open(self.root / record.relative_path) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, record.age, self.indices[index]


def seed_worker(worker_id: int) -> None:
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)
