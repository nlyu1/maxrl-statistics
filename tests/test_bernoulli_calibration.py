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


@pytest.mark.parametrize("count", [2, 3, 4])
@pytest.mark.parametrize("method", ["rloo", "maxrl"])
@pytest.mark.parametrize("protocol", ["shared", "split"])
def test_exact_expected_gradients(count: int, method: str, protocol: str) -> None:
    config = canonical_run(root=Path("unused")).model_copy(
        update={
            "method": method,
            "protocol": protocol,
            "num_rollouts": count,
        }
    )
    probability, target = 0.2, 0.3
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
