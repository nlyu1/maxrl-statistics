from __future__ import annotations

import hashlib
import json
import random
import re
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from pydantic import ConfigDict
from pydantic.dataclasses import dataclass
from sklearn.model_selection import train_test_split
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, DistributedSampler, Sampler
from torchvision import transforms

from src.experiments.utkface.config import DataConfig


@dataclass(kw_only=True, frozen=True)
class FaceSample:
    path: Path
    age: int


def find_samples(*, directory: Path) -> list[FaceSample]:
    image_directory = directory
    for name in ("UTKFace", "utkface"):
        if (directory / name).is_dir():
            image_directory = directory / name
            break
    paths = sorted(image_directory.iterdir())
    pattern = re.compile(r"^(\d+)_\d+_\d+_\d+\.jpg(?:\.chip\.jpg)?$")
    matched = [path for path in paths if pattern.match(path.name)]
    samples = []
    for path in matched or paths:
        if not path.is_file() or path.suffix.lower() != ".jpg":
            continue
        age = path.name.split("_", maxsplit=1)[0]
        if age.isdigit() and 0 <= int(age) <= 99:
            samples.append(FaceSample(path=path.resolve(), age=int(age)))
    if not samples:
        raise ValueError(
            f"No UTKFace images with ages 0–99 found in {image_directory}."
        )
    return samples


def _limit_indices(
    *,
    indices: list[int],
    samples: list[FaceSample],
    count: int,
    seed: int,
) -> list[int]:
    if count == 0 or count >= len(indices):
        return indices
    selected, _ = train_test_split(
        indices,
        train_size=count,
        random_state=seed,
        stratify=[min(samples[index].age // 10, 9) for index in indices],
    )
    return selected


def make_splits(
    *,
    samples: list[FaceSample],
    config: DataConfig,
) -> dict[str, list[int]]:
    indices = list(range(len(samples)))
    train, remaining = train_test_split(
        indices,
        test_size=0.2,
        random_state=config.split_seed,
        stratify=[min(sample.age // 10, 9) for sample in samples],
    )
    validation, test = train_test_split(
        remaining,
        test_size=0.5,
        random_state=config.split_seed,
        stratify=[min(samples[index].age // 10, 9) for index in remaining],
    )
    return {
        split: _limit_indices(
            indices=selected,
            samples=samples,
            count=limit,
            seed=config.split_seed,
        )
        for split, selected, limit in (
            ("train", train, config.max_train_samples),
            ("validation", validation, config.max_eval_samples),
            ("test", test, config.max_eval_samples),
        )
    }


def make_manifest(
    *,
    samples: list[FaceSample],
    splits: dict[str, list[int]],
) -> dict:
    rows = {
        split: [
            {
                "file": samples[index].path.name,
                "age": samples[index].age,
                "bytes": samples[index].path.stat().st_size,
            }
            for index in indices
        ]
        for split, indices in splits.items()
    }
    digest = hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()
    return {
        "sha256": digest,
        "counts": {key: len(value) for key, value in rows.items()},
        "splits": rows,
    }


class FaceDataset(Dataset):
    def __init__(
        self,
        *,
        samples: list[FaceSample],
        indices: list[int],
        input_size: int,
        augment: bool,
    ) -> None:
        self.samples = [samples[index] for index in indices]
        normalization = transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        )
        if augment:
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(input_size, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.ToTensor(),
                normalization,
            ])
        else:
            resize = round(input_size * 256 / 224)
            self.transform = transforms.Compose([
                transforms.Resize(resize),
                transforms.CenterCrop(input_size),
                transforms.ToTensor(),
                normalization,
            ])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[Tensor, int]:
        sample = self.samples[index]
        with Image.open(sample.path) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, sample.age


class EvaluationSampler(Sampler[int]):
    def __init__(self, *, size: int, rank: int, world_size: int) -> None:
        self.indices = range(rank, size, world_size)

    def __iter__(self) -> Iterator[int]:
        return iter(self.indices)

    def __len__(self) -> int:
        return len(self.indices)


def seed_worker(worker_id: int) -> None:
    del worker_id
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)


@dataclass(kw_only=True, config=ConfigDict(arbitrary_types_allowed=True))
class DataBundle:
    train: DataLoader
    validation: DataLoader
    test: DataLoader
    train_sampler: DistributedSampler
    manifest: dict
    generator: torch.Generator


def build_data(
    *,
    config: DataConfig,
    global_batch_size: int,
    eval_batch_size: int,
    seed: int,
    rank: int,
    world_size: int,
) -> DataBundle:
    if global_batch_size % world_size:
        raise ValueError("global_batch_size must be divisible by world_size.")
    if global_batch_size // world_size < 2:
        raise ValueError("ResNet BatchNorm requires at least two images per rank.")
    samples = find_samples(directory=config.directory)
    splits = make_splits(samples=samples, config=config)
    datasets = {
        split: FaceDataset(
            samples=samples,
            indices=indices,
            input_size=config.input_size,
            augment=split == "train",
        )
        for split, indices in splits.items()
    }
    sampler = DistributedSampler(
        datasets["train"],
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=seed,
        drop_last=True,
    )
    generator = torch.Generator(device="cpu").manual_seed(seed + rank)
    loaders = {}
    for split, dataset in datasets.items():
        train = split == "train"
        split_sampler = (
            sampler
            if train
            else EvaluationSampler(
                size=len(dataset),
                rank=rank,
                world_size=world_size,
            )
        )
        loaders[split] = DataLoader(
            dataset,
            batch_size=global_batch_size // world_size if train else eval_batch_size,
            sampler=split_sampler,
            num_workers=config.num_workers,
            pin_memory=True,
            drop_last=train,
            worker_init_fn=seed_worker,
            generator=generator,
            persistent_workers=False,
        )
    if len(loaders["train"]) == 0:
        raise ValueError("The training split is smaller than one global batch.")
    return DataBundle(
        train=loaders["train"],
        validation=loaders["validation"],
        test=loaders["test"],
        train_sampler=sampler,
        manifest=make_manifest(samples=samples, splits=splits),
        generator=generator,
    )
