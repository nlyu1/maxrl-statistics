import itertools
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from src.experiments.utkface.advantages import batched_advantages
from src.experiments.utkface.config import canonical_config
from src.experiments.utkface.losses import (
    log_policy,
    log_window_probabilities,
    policy_loss,
)
from src.experiments.utkface.metrics import METRIC_NAMES, batch_metrics
from src.experiments.utkface.model import build_model


def loss_config(*, method: str, scale: float = 5.0):
    config = canonical_config(manifest=Path("unused"), output_dir=Path("unused")).loss
    return type(config).model_validate({
        **config.model_dump(),
        "method": method,
        "kernel_scale_years": scale,
    })


def gradient(
    *, logits: torch.Tensor, targets: torch.Tensor, method: str
) -> torch.Tensor:
    return torch.autograd.grad(
        policy_loss(logits=logits, targets=targets, config=loss_config(method=method)),
        logits,
        retain_graph=True,
    )[0]


@pytest.mark.parametrize("method", ["ce", "binary_maxrl_population"])
def test_binary_population_is_exact_cross_entropy(method: str) -> None:
    torch.manual_seed(4)
    logits = torch.randn(3, 100, dtype=torch.float64, requires_grad=True)
    targets = torch.tensor([0, 49, 99])
    got = policy_loss(logits=logits, targets=targets, config=loss_config(method=method))
    expected = F.cross_entropy(logits, targets)
    torch.testing.assert_close(got, expected)
    torch.testing.assert_close(
        gradient(logits=logits, targets=targets, method=method),
        torch.autograd.grad(expected, logits)[0],
    )


@pytest.mark.parametrize("reward_kind", ["mae", "mse"])
def test_population_rloo_and_grpo_score_function(reward_kind: str) -> None:
    torch.manual_seed(6)
    logits = torch.randn(3, 100, dtype=torch.float64, requires_grad=True)
    targets = torch.tensor([0, 34, 99])
    probs = logits.softmax(-1).detach()
    error = (torch.arange(100) - targets[:, None]).double() / 99
    reward = 1 - (error.abs() if reward_kind == "mae" else error.square())
    centered = reward - (probs * reward).sum(-1, keepdim=True)
    rloo_expected = -probs * centered / len(targets)
    std = (probs * centered.square()).sum(-1, keepdim=True).sqrt()
    torch.testing.assert_close(
        gradient(
            logits=logits, targets=targets, method=f"rloo_{reward_kind}_population"
        ),
        rloo_expected,
    )
    torch.testing.assert_close(
        gradient(
            logits=logits, targets=targets, method=f"grpo_{reward_kind}_population"
        ),
        rloo_expected / std,
    )


def test_mse_is_squared_mean_error_not_expected_squared_error() -> None:
    logits = torch.zeros(1, 100, dtype=torch.float64, requires_grad=True)
    target = torch.tensor([50])
    got = policy_loss(logits=logits, targets=target, config=loss_config(method="mse"))
    torch.testing.assert_close(got, torch.tensor((0.5 / 99) ** 2, dtype=torch.float64))
    assert float(got.detach()) < 1e-4


@pytest.mark.parametrize("kind", ["mae", "mse"])
@pytest.mark.parametrize("scale", [2.0, 5.0, 10.0])
def test_statistical_maxrl_gradient_is_policy_minus_kernel_posterior(
    kind: str, scale: float
) -> None:
    torch.manual_seed(8)
    logits = torch.randn(2, 100, dtype=torch.float64, requires_grad=True)
    targets = torch.tensor([12, 82])
    errors = (torch.arange(100) - targets[:, None]).double() / scale
    log_kernel = -errors.abs() if kind == "mae" else -0.5 * errors.square()
    posterior = (logits + log_kernel).softmax(-1)
    loss = policy_loss(
        logits=logits,
        targets=targets,
        config=loss_config(method=f"stat_maxrl_{kind}_population", scale=scale),
    )
    torch.testing.assert_close(
        torch.autograd.grad(loss, logits)[0], (logits.softmax(-1) - posterior) / 2
    )


def test_tail_population_matches_independent_threshold_loop_and_gradcheck() -> None:
    torch.manual_seed(10)
    logits = torch.randn(3, 100, dtype=torch.float64, requires_grad=True)
    targets = torch.tensor([0, 51, 99])
    probs = logits.softmax(-1)
    expected = []
    for row, target in zip(probs, targets, strict=True):
        for distance in range(99):
            low, high = (
                max(0, int(target) - distance),
                min(100, int(target) + distance + 1),
            )
            expected.append(-row[low:high].sum().log())
    reference = torch.stack(expected).mean()
    got = policy_loss(
        logits=logits, targets=targets, config=loss_config(method="tailrl_population")
    )
    torch.testing.assert_close(got, reference)
    torch.testing.assert_close(
        torch.autograd.grad(got, logits, retain_graph=True)[0],
        torch.autograd.grad(reference, logits)[0],
    )
    assert torch.autograd.gradcheck(
        lambda x: policy_loss(
            logits=x, targets=targets, config=loss_config(method="tailrl_population")
        ),
        (logits,),
        fast_mode=True,
    )


@pytest.mark.parametrize(
    "method",
    [
        "tailrl_population",
        "stat_maxrl_mae_population",
        "stat_maxrl_mse_population",
        "ce",
    ],
)
def test_extreme_logits_remain_finite(method: str) -> None:
    logits = torch.full((2, 100), -10000.0, requires_grad=True)
    with torch.no_grad():
        logits[:, 99] = 10000
    loss = policy_loss(
        logits=logits, targets=torch.tensor([0, 99]), config=loss_config(method=method)
    )
    assert torch.isfinite(loss)
    assert torch.isfinite(torch.autograd.grad(loss, logits)[0]).all()


def test_tail_floor_removes_low_survival_gradients() -> None:
    logits = torch.zeros(1, 100, dtype=torch.float64, requires_grad=True)
    config = loss_config(method="tailrl_population").model_copy(
        update={"survival_floor": 0.5}
    )
    log_windows = log_window_probabilities(
        log_probs=log_policy(logits=logits), targets=torch.tensor([0])
    )
    expected = -log_windows.clamp_min(torch.tensor(0.5).log().double()).mean()
    torch.testing.assert_close(
        policy_loss(logits=logits, targets=torch.tensor([0]), config=config), expected
    )


@pytest.mark.parametrize("estimator", ["grpo", "rloo", "binary_maxrl", "tailrl"])
def test_estimators_batch_last_dimension_and_ties(estimator: str) -> None:
    rewards = torch.tensor(
        [
            [[0.0, 0.5, 0.5, 1.0], [1.0, 1.0, 1.0, 1.0]],
            [[0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 1.0]],
        ],
        dtype=torch.float64,
    )
    got = batched_advantages(
        rewards=rewards, estimator=estimator, survival_floor=0.0, std_floor=1e-4
    )
    rows = [
        batched_advantages(
            rewards=row, estimator=estimator, survival_floor=0.0, std_floor=1e-4
        )
        for row in rewards.reshape(-1, 4)
    ]
    torch.testing.assert_close(got, torch.stack(rows).reshape_as(got))
    torch.testing.assert_close(got.mean(-1), torch.zeros(2, 2, dtype=torch.float64))
    assert got[0, 0, 1] == got[0, 0, 2]


def test_sampled_rloo_expectation_is_unbiased_by_enumeration() -> None:
    probs = torch.tensor([0.2, 0.3, 0.5], dtype=torch.float64)
    rewards = torch.tensor([0.0, 0.6, 1.0], dtype=torch.float64)
    expectation = torch.zeros(3, dtype=torch.float64)
    for combination in itertools.product(range(3), repeat=3):
        actions = torch.tensor(combination)
        adv = batched_advantages(
            rewards=rewards[actions],
            estimator="rloo",
            survival_floor=0.0,
            std_floor=1e-4,
        )
        scores = F.one_hot(actions, num_classes=3).double() - probs
        expectation += probs[actions].prod() * (adv[:, None] * scores).mean(0)
    torch.testing.assert_close(expectation, probs * (rewards - (probs * rewards).sum()))


def test_finite_binary_maxrl_has_known_attenuation() -> None:
    probs = torch.tensor([0.2, 0.3, 0.5], dtype=torch.float64)
    expectation = torch.zeros(3, dtype=torch.float64)
    for combination in itertools.product(range(3), repeat=3):
        actions = torch.tensor(combination)
        rewards = (actions == 2).double()
        advantages = batched_advantages(
            rewards=rewards,
            estimator="binary_maxrl",
            survival_floor=0.0,
            std_floor=1e-4,
        )
        scores = F.one_hot(actions, num_classes=3).double() - probs
        expectation += probs[actions].prod() * (advantages[:, None] * scores).mean(0)
    population = F.one_hot(torch.tensor(2), num_classes=3).double() - probs
    torch.testing.assert_close(expectation, (1 - (1 - probs[2]) ** 2) * population)


def test_tail_binary_matches_binary_maxrl_and_hand_computed_gaps() -> None:
    reward = torch.tensor(
        [[0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0]], dtype=torch.float64
    )
    tail = batched_advantages(
        rewards=reward, estimator="tailrl", survival_floor=0.0, std_floor=1e-4
    )
    binary = batched_advantages(
        rewards=reward, estimator="binary_maxrl", survival_floor=0.0, std_floor=1e-4
    )
    torch.testing.assert_close(tail, binary)
    continuous = batched_advantages(
        rewards=torch.tensor([0.2, 0.5, 0.5, 1.0], dtype=torch.float64),
        estimator="tailrl",
        survival_floor=0.0,
        std_floor=1e-4,
    )
    weights = torch.tensor([0.2, 0.6, 0.6, 2.6], dtype=torch.float64)
    torch.testing.assert_close(continuous, weights - weights.mean())


def test_single_rollout_is_rejected_and_bf16_is_promoted() -> None:
    with pytest.raises(ValueError, match="two rollouts"):
        batched_advantages(
            rewards=torch.ones(2, 1),
            estimator="rloo",
            survival_floor=0.0,
            std_floor=1e-4,
        )
    got = batched_advantages(
        rewards=torch.tensor([0.0, 0.5, 1.0], dtype=torch.bfloat16),
        estimator="tailrl",
        survival_floor=0.0,
        std_floor=1e-4,
    )
    assert got.dtype == torch.float32


def test_architecture_and_uniform_policy_metrics() -> None:
    model = build_model(pretrained=False)
    assert model.fc.in_features == 512 and model.fc.out_features == 100
    assert (
        len(model.layer1)
        == len(model.layer2)
        == len(model.layer3)
        == len(model.layer4)
        == 2
    )
    metrics = batch_metrics(logits=torch.zeros(1, 100), targets=torch.tensor([0]))[0]
    values = dict(zip(METRIC_NAMES, metrics.tolist(), strict=True))
    assert values["mae_mean"] == pytest.approx(49.5)
    assert values["expected_mae"] == pytest.approx(49.5)
    assert (
        values["oracle_best_of_64_mae"]
        < values["oracle_best_of_8_mae"]
        < values["expected_mae"]
    )


@pytest.mark.parametrize(
    "method", ["grpo_mae", "grpo_mse", "rloo_mae", "rloo_mse", "binary_maxrl", "tailrl"]
)
def test_sampled_losses_have_finite_gradients(method: str) -> None:
    torch.manual_seed(77)
    logits = torch.randn(3, 100, requires_grad=True)
    loss = policy_loss(
        logits=logits,
        targets=torch.tensor([0, 50, 99]),
        config=loss_config(method=method),
    )
    assert torch.isfinite(loss)
    assert torch.isfinite(torch.autograd.grad(loss, logits)[0]).all()
