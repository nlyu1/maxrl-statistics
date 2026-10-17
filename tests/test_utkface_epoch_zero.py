import json
import random
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, DistributedSampler, TensorDataset

from src.experiments.utkface import run
from src.experiments.utkface.config import canonical_config
from src.experiments.utkface.distributed import DistributedContext
from src.experiments.utkface.figure_bundle import series
from src.experiments.utkface.model import DigitHead, ScalarHead


def test_plot_omits_initial_stats_and_uses_training_only_moving_means() -> None:
    runs = [
        {
            "epochs": 6,
            "curves": [[epoch, epoch + offset] for epoch in range(1, 7)],
            "initial": {"validation": {"mae_mean": initial}},
        }
        for offset, initial in ((0, 20), (2, 30))
    ]
    smoothed = series(runs=runs, metric="mae_mean", window=3)
    raw = series(runs=runs, metric="mae_mean", window=1)
    assert smoothed["x"] == [2, 3, 4, 5]
    assert raw["x"] == list(range(1, 7))
    assert smoothed["mean"][0] == 3 and raw["mean"][0] == 2
    assert smoothed["sd"][0] == raw["sd"][0] == pytest.approx(2**0.5)


def test_initial_validation_preserves_weights_mode_and_random_streams(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = canonical_config(manifest=Path("unused"), output_dir=Path("unused"))
    config = config.model_copy(update={"device": "cpu", "precision": "fp32"})
    context = DistributedContext(rank=0, world_size=1, device=torch.device("cpu"))
    model = nn.Sequential(nn.Flatten(), nn.Linear(48, 100))
    loader = DataLoader(
        TensorDataset(torch.randn(8, 3, 4, 4), torch.arange(8) * 10, torch.arange(8)),
        batch_size=4,
        generator=torch.Generator().manual_seed(31),
    )
    original_evaluate = run.evaluate

    def consuming_evaluate(**kwargs: object) -> dict:
        random.random()
        np.random.random()
        torch.rand(5)
        return original_evaluate(**kwargs)

    monkeypatch.setattr(run, "evaluate", consuming_evaluate)
    numpy_state, python_state = np.random.get_state(), random.getstate()
    torch_state, loader_state = torch.get_rng_state(), loader.generator.get_state()
    weights = {name: value.clone() for name, value in model.state_dict().items()}
    initial = run.initial_validation(
        model=model,
        loader=loader,
        config=config,
        context=context,
        data_fingerprint="clean-validation",
        origin="recorded",
    )
    assert initial["epoch"] == 0 and initial["validation"]["count"] == 8
    assert initial["origin"] == "recorded" and len(initial["model_sha256"]) == 64
    assert model.training and random.getstate() == python_state
    np.testing.assert_array_equal(np.random.get_state()[1], numpy_state[1])
    assert np.random.get_state()[2:] == numpy_state[2:]
    assert torch.equal(torch.get_rng_state(), torch_state)
    assert torch.equal(loader.generator.get_state(), loader_state)
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, weights[name], rtol=0, atol=0)


@pytest.mark.parametrize("head", ["categorical", "scalar", "digits"])
def test_training_records_initial_validation_without_an_extra_trained_epoch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, head: str
) -> None:
    payload = {
        "fingerprint": "clean-split",
        "records": [{"age": i * 10} for i in range(8)],
    }
    config = canonical_config(manifest=tmp_path / "unused", output_dir=tmp_path / head)
    config = config.model_copy(
        update={
            "head": head,
            "loss": config.loss.model_copy(update={"method": "mae"}),
            "device": "cpu",
            "precision": "fp32",
            "cpu_threads": 2,
            "epochs": 1,
            "batch_size": 4,
            "eval_batch_size": 4,
            "num_workers": 0,
            "save_checkpoint": False,
            "pretrained": False,
        }
    )
    images = torch.randn(8, 3, 4, 4)
    dataset = TensorDataset(images, torch.arange(8) * 10, torch.arange(8))
    dataset.indices = list(range(8))

    def loader(
        *, config: object, context: object, payload: dict, split: str
    ) -> DataLoader:
        training = split == "train"
        return DataLoader(
            dataset,
            batch_size=4,
            sampler=DistributedSampler(dataset, num_replicas=1, rank=0)
            if training
            else None,
            generator=torch.Generator().manual_seed(42),
            drop_last=training,
        )

    def model(*, pretrained: bool, head: str) -> nn.Module:
        final = (
            ScalarHead(features=48)
            if head == "scalar"
            else DigitHead(features=48)
            if head == "digits"
            else nn.Linear(48, 100)
        )
        return nn.Sequential(nn.Flatten(), final)

    monkeypatch.setattr(run, "read_manifest", lambda **kwargs: payload)
    monkeypatch.setattr(run, "make_loader", loader)
    monkeypatch.setattr(run, "build_model", model)
    result = run.train(
        config=config,
        context=DistributedContext(rank=0, world_size=1, device=torch.device("cpu")),
    )
    initial = json.loads((config.output_dir / "initial_validation.json").read_text())
    history = json.loads((config.output_dir / "history.json").read_text())
    assert initial == result["initial_validation"]
    assert initial["epoch"] == 0 and initial["validation"]["count"] == 8
    assert [row["epoch"] for row in history] == [1]
    assert history[0]["train_count"] == 8 and result["best_epoch"] == 1
    assert (initial["validation"]["label_nll"] is None) == (head == "scalar")
