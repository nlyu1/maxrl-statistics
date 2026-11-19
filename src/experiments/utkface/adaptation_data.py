import hashlib
import os
import random
import struct
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Literal

import numpy as np
import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, Sampler
from torchvision import transforms

from src.experiments.utkface.config import RunConfig
from src.experiments.utkface.data import (
    FaceDataset,
    read_manifest,
    seed_worker,
)

Pipeline = Literal["original", "fixed"]
Split = Literal["train", "val", "test"]

_NORMALIZE = transforms.Normalize(
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225),
)
_ORIGINAL_TRANSFORM = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    _NORMALIZE,
])
_FIXED_TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    _NORMALIZE,
])


def _derived_seed(*components: int) -> int:
    digest = hashlib.blake2b(digest_size=8)
    for component in components:
        if component < 0:
            raise ValueError("Seed components must be nonnegative")
        digest.update(struct.pack("<Q", component))
    return int.from_bytes(digest.digest(), "little")


@contextmanager
def _isolated_seed(seed: int) -> Iterator[None]:
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.random.get_rng_state()
    try:
        random.seed(seed)
        np.random.seed(seed % 2**32)
        torch.random.default_generator.manual_seed(seed)
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.random.set_rng_state(torch_state)


def epoch_order(*, count: int, epoch: int, order_seed: int) -> Tensor:
    if count < 0:
        raise ValueError("count must be nonnegative")
    if epoch < 1:
        raise ValueError("Global training epoch must be positive")
    if order_seed < 0:
        raise ValueError("order_seed must be nonnegative")
    generator = torch.Generator(device="cpu").manual_seed(
        _derived_seed(order_seed, epoch)
    )
    return torch.randperm(count, generator=generator, dtype=torch.int64)


class _EpochOrderSampler(Sampler[int]):
    def __init__(
        self,
        *,
        count: int,
        epoch: int,
        order_seed: int,
        rank: int,
        world_size: int,
    ) -> None:
        if world_size < 1 or not 0 <= rank < world_size:
            raise ValueError("Invalid distributed rank or world size")
        self.order = epoch_order(count=count, epoch=epoch, order_seed=order_seed)
        self.rank = rank
        self.world_size = world_size

    def __iter__(self) -> Iterator[int]:
        return iter(self.order[self.rank :: self.world_size].tolist())

    def __len__(self) -> int:
        return (len(self.order) + self.world_size - 1 - self.rank) // self.world_size


class _AdaptationDataset(Dataset):
    def __init__(
        self,
        *,
        source: FaceDataset,
        pipeline: Pipeline,
        epoch: int,
        view_seed: int,
        split: Split,
    ) -> None:
        self.source = source
        self.pipeline = pipeline
        self.epoch = epoch
        self.view_seed = view_seed
        self.split = split

    def __len__(self) -> int:
        return len(self.source)

    def __getitem__(self, index: int) -> tuple[Tensor, int, int]:
        record = self.source.records[index]
        global_index = self.source.indices[index]
        with Image.open(self.source.root / record.relative_path) as image:
            rgb = image.convert("RGB")
            if self.split == "train" and self.pipeline == "original":
                seed = _derived_seed(self.view_seed, global_index, self.epoch)
                with _isolated_seed(seed):
                    tensor = _ORIGINAL_TRANSFORM(rgb)
            else:
                tensor = _FIXED_TRANSFORM(rgb)
        return tensor, record.age, global_index


def build_image_loader(
    *,
    run: RunConfig,
    pipeline: Pipeline,
    split: Split,
    epoch: int,
    view_seed: int,
    order_seed: int,
) -> DataLoader:
    if split == "train" and epoch < 1:
        raise ValueError("Training epoch must be positive")
    if split != "train" and epoch != 0:
        raise ValueError("Validation and test loaders use epoch=0")
    if view_seed < 0 or order_seed < 0:
        raise ValueError("view_seed and order_seed must be nonnegative")

    payload = read_manifest(path=run.manifest)
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    source = FaceDataset(
        manifest_path=run.manifest,
        payload=payload,
        split=split,
        limit=run.train_limit if split == "train" else run.eval_limit,
        rank=rank,
        world_size=world_size,
        subset_seed=run.subset_seed,
    )
    dataset = _AdaptationDataset(
        source=source,
        pipeline=pipeline if split == "train" else "fixed",
        epoch=epoch,
        view_seed=view_seed,
        split=split,
    )
    if split == "train":
        sampler = _EpochOrderSampler(
            count=len(dataset),
            epoch=epoch,
            order_seed=order_seed,
            rank=rank,
            world_size=world_size,
        )
        batch_size = run.batch_size
        drop_last = True
    else:
        sampler = range(len(dataset))
        batch_size = run.eval_batch_size
        drop_last = False
    generator = torch.Generator(device="cpu").manual_seed(
        _derived_seed(order_seed, epoch, 0x574F524B)
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=False,
        num_workers=run.num_workers,
        drop_last=drop_last,
        pin_memory=run.device.startswith("cuda"),
        persistent_workers=run.num_workers > 0,
        worker_init_fn=seed_worker,
        generator=generator,
    )
