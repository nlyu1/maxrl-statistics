from itertools import product
from pathlib import Path

import pytest
import torch

from src.experiments.bernoulli_calibration.config import (
    canonical_dataset,
    canonical_run,
)
from src.experiments.bernoulli_calibration.data import load_dataset, prepare_dataset
from src.experiments.bernoulli_calibration.model import BagProbabilityModel
from src.experiments.bernoulli_calibration.objectives import (
    grouped_advantages,
    native_losses,
)
from src.experiments.bernoulli_calibration.run import evaluate, train
from src.experiments.bernoulli_calibration.study import families, select_rates
from src.experiments.utils import set_seeds


@pytest.mark.parametrize("count", [2, 3, 4])
@pytest.mark.parametrize("method", ["rloo", "maxrl"])
@pytest.mark.parametrize("protocol", ["shared", "split"])
@pytest.mark.parametrize("probability,target", [(0.2, 0.3), (0.8, 0.65), (0.5, 0.5)])
def test_exact_expected_gradients(
    count: int, method: str, protocol: str, probability: float, target: float
) -> None:
    config = canonical_run(root=Path("unused")).model_copy(
        update={
            "method": method,
            "protocol": protocol,
            "num_rollouts": count,
        }
    )
    actions = torch.tensor(list(product((0.0, 1.0), repeat=count)), dtype=torch.float64)
    label_rows = torch.tensor(
        list(product((0.0, 1.0), repeat=count if protocol == "split" else 1)),
        dtype=torch.float64,
    )
    expected = torch.zeros((), dtype=torch.float64)
    for labels in label_rows:
        rewards = (actions == labels).double()
        advantages = grouped_advantages(
            rewards=rewards, labels=labels.expand_as(actions), config=config
        )
        action_mass = probability ** actions.sum(-1) * (1 - probability) ** (
            count - actions.sum(-1)
        )
        label_mass = target ** labels.sum() * (1 - target) ** (
            len(labels) - labels.sum()
        )
        expected += (
            label_mass
            * (action_mass * (advantages * (actions - probability)).mean(-1)).sum()
        )
    logit = torch.tensor(
        [torch.logit(torch.tensor(probability, dtype=torch.float64))],
        requires_grad=True,
    )
    loss = native_losses(
        logits=logit, targets=torch.tensor([target], dtype=torch.float64), config=config
    ).sum()
    gradient = -torch.autograd.grad(loss, logit)[0].squeeze()
    torch.testing.assert_close(expected, gradient, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("method", ["rloo", "grpo", "maxrl"])
def test_single_class_partition_matches_shared(method: str) -> None:
    config = canonical_run(root=Path("unused")).model_copy(update={"method": method})
    rewards = torch.tensor([[0.0, 1.0, 0.0, 1.0], [1.0, 1.0, 1.0, 1.0]])
    labels = torch.ones_like(rewards)
    shared = grouped_advantages(rewards=rewards, labels=labels, config=config)
    split = grouped_advantages(
        rewards=rewards,
        labels=labels,
        config=config.model_copy(update={"protocol": "split"}),
    )
    torch.testing.assert_close(shared, split)


@pytest.mark.parametrize("method", ["ce", "mse", "mae"])
def test_scalar_probability_targets(method: str) -> None:
    config = canonical_run(root=Path("unused")).model_copy(update={"method": method})
    logits = torch.logit(torch.tensor([0.7], dtype=torch.float64)).requires_grad_()
    loss = native_losses(
        logits=logits, targets=torch.tensor([0.7], dtype=torch.float64), config=config
    )
    gradient = torch.autograd.grad(loss.sum(), logits)[0]
    if method == "mae":
        assert gradient.item() < 0
    else:
        torch.testing.assert_close(
            gradient, torch.zeros_like(gradient), atol=1e-14, rtol=0
        )


def test_dataset_splits_and_training_prefix(tmp_path: Path) -> None:
    config = canonical_dataset().model_copy(
        update={
            "train_capacity": 64,
            "validation_size": 16,
            "test_size": 16,
        }
    )
    path = prepare_dataset(root=tmp_path, config=config)
    small = load_dataset(path=path, device=torch.device("cpu"), train_size=32)
    large = load_dataset(path=path, device=torch.device("cpu"), train_size=64)
    torch.testing.assert_close(small.tokens["train"], large.tokens["train"][:32])
    torch.testing.assert_close(small.tokens["validation"], large.tokens["validation"])
    bags = [
        tuple(sorted(row.tolist())) for split in large.tokens.values() for row in split
    ]
    assert len(bags) == len(set(bags))
    expected = (
        torch.tensor(config.word_values)[large.tokens["train"]].sum(-1).div(4).sigmoid()
    )
    torch.testing.assert_close(large.probabilities["train"], expected)


def test_architecture_and_permutation_invariance() -> None:
    torch.manual_seed(42)
    model = BagProbabilityModel(
        data=canonical_dataset(), config=canonical_run(root=Path("unused"))
    )
    tokens = torch.randint(32, (3, 16))
    assert sum(parameter.numel() for parameter in model.parameters()) == 8257
    torch.testing.assert_close(model(tokens=tokens), model(tokens=tokens.flip(-1)))


def test_supervised_shared_baselines_are_reused() -> None:
    configurations = families(root=Path("unused"), train_size=8192, device="cuda:0")
    assert (
        len(configurations) == len({config.family for config in configurations}) == 57
    )
    shared_ce = [
        config
        for config in configurations
        if config.method == "ce" and config.protocol == "shared"
    ]
    assert len(shared_ce) == 1


def test_rate_selection_uses_native_loss(tmp_path: Path) -> None:
    candidates = [
        {
            "family": "rloo-shared-n16",
            "config": {"learning_rate": rate, "identifier": str(rate)},
            "final": {"validation": {"native_loss": loss, "rmse": rmse}},
        }
        for rate, loss, rmse in ((1e-3, 0.2, 0.9), (1e-2, 0.4, 0.1))
    ]
    assert select_rates(results=candidates, root=tmp_path, train_size=8192) == {
        "rloo-shared-n16": 1e-3
    }


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA trajectory check")
def test_gpu_validation_snapshots_and_head_updates(tmp_path: Path) -> None:
    prepare_dataset(
        root=tmp_path,
        config=canonical_dataset().model_copy(
            update={
                "train_capacity": 64,
                "validation_size": 16,
                "test_size": 16,
            }
        ),
    )
    config = canonical_run(root=tmp_path).model_copy(
        update={
            "identifier": "capture-check",
            "stage": "main",
            "steps": 4,
            "train_size": 32,
            "batch_size": 16,
            "eval_interval": 4,
        }
    )
    with torch.cuda.device(0), torch.device("cuda:0"):
        set_seeds(config.seed)
        model = BagProbabilityModel(data=canonical_dataset(), config=config)
        initial = {
            name: value.detach().cpu().clone()
            for name, value in model.state_dict().items()
        }
    result = train(config=config, record_validation=True)
    trajectory = torch.load(
        config.output_dir / "validation-trajectory.pt",
        map_location="cpu",
        weights_only=True,
    )
    assert trajectory["step"].tolist() == [0, 1, 2, 4]
    assert trajectory["q"].shape == (4, 16)
    assert (
        trajectory["initial_model_fingerprint"] == result["initial_model_fingerprint"]
    )
    rmse = (trajectory["q"][-1] - trajectory["f"]).square().mean().sqrt().item()
    assert rmse == pytest.approx(result["final"]["validation"]["rmse"], abs=1e-6)
    final = torch.load(
        config.output_dir / "final.pt", map_location="cpu", weights_only=True
    )
    assert all((final[name] - initial[name]).abs().max().item() > 0 for name in initial)
    with torch.cuda.device(0), torch.device("cuda:0"):
        restored = BagProbabilityModel(data=canonical_dataset(), config=config)
        restored.load_state_dict(final)
        dataset = load_dataset(
            path=tmp_path / "dataset.pt", device=torch.device("cuda:0"), train_size=32
        )
        metrics, predictions = evaluate(
            model=restored, dataset=dataset, split="validation", config=config
        )
    torch.testing.assert_close(predictions, trajectory["q"][-1], rtol=0, atol=1e-7)
    assert metrics["rmse"] == pytest.approx(
        result["final"]["validation"]["rmse"], abs=1e-7
    )
