from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch import nn

from src.experiments.utkface import train
from src.experiments.utkface.canonical import get_canonical_config
from src.experiments.utkface.config import UTKFaceConfig
from src.experiments.utkface.data import (
    EvaluationSampler,
    find_samples,
    make_manifest,
    make_splits,
)
from src.experiments.utkface.evaluate import batch_metrics
from src.experiments.utkface.model import create_policy


@pytest.fixture
def image_directory(tmp_path: Path) -> Path:
    directory = tmp_path / "UTKFace"
    directory.mkdir()
    generator = np.random.default_rng(31)
    for age in range(100):
        for instance in range(2):
            pixels = generator.integers(0, 256, (32, 32, 3), dtype=np.uint8)
            Image.fromarray(pixels).save(
                directory / f"{age}_0_0_{instance:05d}.jpg.chip.jpg"
            )
    Image.new("RGB", (32, 32)).save(directory / "101_0_0_00000.jpg.chip.jpg")
    return directory


def test_original_decade_stratified_split_is_preserved(
    image_directory: Path, tmp_path: Path
) -> None:
    config = get_canonical_config(
        method="ce", data_directory=image_directory, output_directory=tmp_path, seed=19
    )
    samples = find_samples(directory=image_directory.parent)
    assert len(samples) == 200
    train_indices, remaining = train_test_split(
        list(range(200)),
        test_size=0.2,
        random_state=42,
        stratify=[sample.age // 10 for sample in samples],
    )
    validation, test = train_test_split(
        remaining,
        test_size=0.5,
        random_state=42,
        stratify=[samples[index].age // 10 for index in remaining],
    )
    splits = make_splits(samples=samples, config=config.data)
    assert splits == {"train": train_indices, "validation": validation, "test": test}
    assert set(train_indices).isdisjoint(validation)
    assert set(train_indices).isdisjoint(test)
    assert set(validation).isdisjoint(test)
    manifest = make_manifest(samples=samples, splits=splits)
    assert manifest["counts"] == {"train": 160, "validation": 20, "test": 20}
    assert make_manifest(samples=samples, splits=splits) == manifest


@pytest.mark.parametrize("size,world_size", ((23, 2), (5, 8), (100, 3)))
def test_evaluation_shards_neither_pad_nor_drop_examples(
    size: int, world_size: int
) -> None:
    shards = [
        list(EvaluationSampler(size=size, rank=rank, world_size=world_size))
        for rank in range(world_size)
    ]
    flattened = [index for shard in shards for index in shard]
    assert len(flattened) == len(set(flattened)) == size
    assert sorted(flattened) == list(range(size))


def test_resnet18_has_a_100_bin_categorical_head() -> None:
    model = create_policy(pretrained=False, seed=42).eval()
    assert model.fc.in_features == 512
    assert model.fc.out_features == 100
    with torch.no_grad():
        assert model(torch.zeros(2, 3, 32, 32)).shape == (2, 100)


def test_best_of_one_equals_expected_reward_and_larger_budgets_improve() -> None:
    torch.manual_seed(9)
    metrics = batch_metrics(
        logits=torch.randn(4, 100),
        targets=torch.tensor([0, 30, 70, 99]),
        temperature_years=5,
    )
    torch.testing.assert_close(
        metrics["best_of_1_reward"], metrics["expected_reward"], atol=1e-6, rtol=1e-6
    )
    assert (metrics["best_of_64_reward"] >= metrics["best_of_8_reward"]).all()
    assert (metrics["best_of_8_reward"] >= metrics["best_of_1_reward"]).all()
    assert (metrics["best_of_64_reward"] <= 1 + 1e-6).all()


def tiny_policy(*, pretrained: bool, seed: int) -> nn.Module:
    assert not pretrained
    torch.manual_seed(seed)
    return nn.Sequential(
        nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.1), nn.Linear(3, 100)
    )


def test_interrupted_training_resumes_with_identical_parameters_and_metrics(
    image_directory: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(train, "create_policy", tiny_policy)
    config = get_canonical_config(
        method="tailrl",
        data_directory=image_directory,
        output_directory=tmp_path / "full",
        seed=23,
    )
    payload = config.model_dump()
    payload["pretrained"] = False
    payload["data"].update(input_size=32, num_workers=0)
    payload["training"].update(
        epochs=2,
        global_batch_size=32,
        eval_batch_size=32,
        device="cpu",
        precision="fp32",
        checkpoint_epochs=(1, 2),
    )
    payload["objective"].update(num_rollouts=8, degree=4)
    full_config = UTKFaceConfig.model_validate(payload)
    full = train.run_training(config=full_config, resume=False)
    payload["output_directory"] = tmp_path / "resumed"
    resumed_config = UTKFaceConfig.model_validate(payload)
    original_epoch = train.train_epoch
    calls = 0

    def interrupted_epoch(**kwargs: object) -> dict:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise InterruptedError("Simulated interruption between epochs")
        return original_epoch(**kwargs)

    monkeypatch.setattr(train, "train_epoch", interrupted_epoch)
    with pytest.raises(InterruptedError):
        train.run_training(config=resumed_config, resume=False)
    monkeypatch.setattr(train, "train_epoch", original_epoch)
    resumed = train.run_training(config=resumed_config, resume=True)
    assert full["test"] == resumed["test"]
    assert full["best_epoch"] == resumed["best_epoch"]
    assert full["initial_model_sha256"] == resumed["initial_model_sha256"]
    checkpoints = [
        torch.load(
            config.output_directory / "checkpoints" / "last.pt",
            weights_only=False,
        )
        for config in (full_config, resumed_config)
    ]
    for name, value in checkpoints[0]["model"].items():
        torch.testing.assert_close(value, checkpoints[1]["model"][name], rtol=0, atol=0)
    assert (
        json.loads((resumed_config.output_directory / "summary.json").read_text())[
            "test"
        ]
        == full["test"]
    )
