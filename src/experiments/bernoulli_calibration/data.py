import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from pydantic import ConfigDict
from pydantic.dataclasses import dataclass
from torch import Tensor

from src.experiments.bernoulli_calibration.config import DatasetConfig


@dataclass(kw_only=True, config=ConfigDict(arbitrary_types_allowed=True))
class Dataset:
    tokens: dict[str, Tensor]
    probabilities: dict[str, Tensor]
    fingerprint: str


def prepare_dataset(*, root: Path, config: DatasetConfig) -> Path:
    destination = root / "dataset.pt"
    metadata = root / "dataset.json"
    if destination.exists():
        if DatasetConfig.model_validate_json(metadata.read_text()) != config:
            raise ValueError("Existing dataset has a different configuration")
        return destination
    root.mkdir(parents=True, exist_ok=True)
    generator = np.random.default_rng(config.seed)
    seen: set[bytes] = set()
    tokens = {}
    probabilities = {}
    values = np.asarray(config.word_values)
    for split, size in (
        ("train", config.train_capacity),
        ("validation", config.validation_size),
        ("test", config.test_size),
    ):
        rows = []
        while len(rows) < size:
            candidates = generator.integers(
                len(values), size=(size - len(rows), config.length), dtype=np.int64
            )
            for row in candidates:
                key = np.bincount(row, minlength=len(values)).tobytes()
                if key not in seen:
                    seen.add(key)
                    rows.append(row)
        array = np.stack(rows)
        score = values[array].sum(axis=-1) / np.sqrt(config.length)
        tokens[split] = torch.from_numpy(array)
        probabilities[split] = torch.from_numpy(1 / (1 + np.exp(-score))).float()
    digest = hashlib.sha256(config.model_dump_json().encode())
    for value in tokens.values():
        digest.update(value.numpy().tobytes())
    torch.save(
        {
            "tokens": tokens,
            "probabilities": probabilities,
            "fingerprint": digest.hexdigest(),
        },
        destination,
    )
    metadata.write_text(config.model_dump_json(indent=2) + "\n")
    summary = {
        split: {
            "size": len(probability),
            "mean": probability.mean().item(),
            "std": probability.std().item(),
            "minimum": probability.min().item(),
            "maximum": probability.max().item(),
            "half_fraction": (probability == 0.5).float().mean().item(),
        }
        for split, probability in probabilities.items()
    }
    (root / "dataset-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return destination


def load_dataset(*, path: Path, device: torch.device, train_size: int) -> Dataset:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if train_size > len(payload["tokens"]["train"]):
        raise ValueError("Requested training prefix exceeds dataset capacity")
    return Dataset(
        tokens={
            split: (value[:train_size] if split == "train" else value).to(device)
            for split, value in payload["tokens"].items()
        },
        probabilities={
            split: (value[:train_size] if split == "train" else value).to(device)
            for split, value in payload["probabilities"].items()
        },
        fingerprint=payload["fingerprint"],
    )
