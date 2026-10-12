from __future__ import annotations

from itertools import pairwise, product
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from src.experiments.utkface.canonical import get_canonical_config
from src.experiments.utkface.config import (
    POPULATION_METHODS,
    SAMPLED_METHODS,
    Method,
    ObjectiveConfig,
)
from src.experiments.utkface.estimators import (
    binary_maxrl_advantages,
    grpo_advantages,
    rloo_advantages,
    tailrl_advantages,
)
from src.experiments.utkface.objectives import (
    age_errors,
    age_rewards,
    compute_loss,
    statistical_log_kernel,
    tailrl_population_loss,
)


def objective(*, method: Method) -> ObjectiveConfig:
    config = get_canonical_config(
        method=method,
        data_directory=Path("/tmp/unused"),
        output_directory=Path("/tmp/unused"),
        seed=42,
    ).objective
    return ObjectiveConfig.model_validate({
        **config.model_dump(),
        "num_rollouts": 8,
        "degree": 4,
    })


@pytest.mark.parametrize("method", (*POPULATION_METHODS, *SAMPLED_METHODS))
def test_every_objective_has_finite_nonzero_gradients(method: Method) -> None:
    torch.manual_seed(1729)
    logits = torch.randn(4, 100, dtype=torch.float64, requires_grad=True)
    loss = compute_loss(
        logits=logits,
        targets=torch.tensor([0, 27, 60, 99]),
        config=objective(method=method),
    )
    (gradient,) = torch.autograd.grad(loss, logits)
    assert torch.isfinite(loss)
    assert torch.isfinite(gradient).all()
    if method != "maxrl_binary":
        assert gradient.norm() > 0
    torch.testing.assert_close(
        gradient.sum(dim=-1), torch.zeros(4, dtype=torch.float64), atol=1e-7, rtol=0
    )


def test_population_binary_maxrl_is_exactly_cross_entropy() -> None:
    torch.manual_seed(3)
    logits = torch.randn(3, 100, dtype=torch.float64, requires_grad=True)
    targets = torch.tensor([0, 49, 99])
    ce = compute_loss(logits=logits, targets=targets, config=objective(method="ce"))
    binary = compute_loss(
        logits=logits,
        targets=targets,
        config=objective(method="maxrl_binary_population"),
    )
    torch.testing.assert_close(binary, ce, rtol=0, atol=0)
    torch.testing.assert_close(
        torch.autograd.grad(binary, logits)[0],
        torch.autograd.grad(ce, logits)[0],
        rtol=0,
        atol=0,
    )


@pytest.mark.parametrize("method", ("mse", "rloo_population", "grpo_population"))
def test_population_gradients_match_their_definitions(method: Method) -> None:
    torch.manual_seed(13)
    logits = torch.randn(3, 100, dtype=torch.float64, requires_grad=True)
    targets = torch.tensor([0, 47, 99])
    config = objective(method=method)
    probabilities = logits.softmax(dim=-1).detach()
    if method == "mse":
        ages = torch.arange(100, dtype=torch.float64)
        mean = (probabilities * ages).sum(dim=-1, keepdim=True)
        expected = (
            2 * (mean - targets[:, None]) / 99**2 * probabilities * (ages - mean) / 3
        )
    else:
        rewards = age_rewards(
            errors=age_errors(targets=targets).double(), reward=config.reward
        )
        centered = rewards - (probabilities * rewards).sum(dim=-1, keepdim=True)
        expected = -probabilities * centered / 3
        if method == "grpo_population":
            deviation = (
                (probabilities * centered.square()).sum(dim=-1, keepdim=True).sqrt()
            )
            expected /= deviation + config.standard_deviation_epsilon
            expected *= deviation >= config.standard_deviation_floor
    loss = compute_loss(logits=logits, targets=targets, config=config)
    torch.testing.assert_close(
        torch.autograd.grad(loss, logits)[0], expected, atol=1e-12, rtol=1e-12
    )


def test_population_tailrl_matches_original_ordinal_age_windows() -> None:
    torch.manual_seed(4)
    logits = torch.randn(4, 100, dtype=torch.float64, requires_grad=True)
    targets = torch.tensor([0, 34, 50, 99])
    log_probs = logits.log_softmax(dim=-1)
    errors = age_errors(targets=targets).double()
    reference = -torch.stack([
        log_probs.masked_fill(errors.abs() > distance, -torch.inf).logsumexp(dim=-1)
        for distance in range(99)
    ]).mean()
    result = compute_loss(
        logits=logits, targets=targets, config=objective(method="tailrl_population")
    )
    torch.testing.assert_close(result, reference, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(
        torch.autograd.grad(result, logits)[0],
        torch.autograd.grad(reference, logits)[0],
        rtol=1e-11,
        atol=1e-12,
    )


@pytest.mark.parametrize("reward", ("mae", "mse", "binary"))
def test_population_tailrl_matches_threshold_integral(reward: str) -> None:
    logits = (
        torch.linspace(-3, 3, 100, dtype=torch.float64).unsqueeze(0).requires_grad_()
    )
    log_probs = logits.log_softmax(dim=-1)
    rewards = age_rewards(
        errors=age_errors(targets=torch.tensor([37])).double(), reward=reward
    )
    levels = torch.cat((
        torch.zeros(1, dtype=torch.float64),
        rewards.unique(sorted=True),
    ))
    reference = torch.zeros((), dtype=torch.float64)
    for lower, upper in pairwise(levels):
        survival = log_probs.masked_fill(rewards < upper, -torch.inf).logsumexp(dim=-1)
        reference -= (upper - lower) * survival.mean()
    result = tailrl_population_loss(
        log_probs=log_probs, rewards=rewards, survival_floor=0.0
    )
    torch.testing.assert_close(result, reference, atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize(
    "method", ("stat_maxrl_mse_population", "stat_maxrl_mae_population")
)
def test_statistical_population_gradient_is_prior_minus_posterior(
    method: Method,
) -> None:
    torch.manual_seed(6)
    logits = torch.randn(2, 100, dtype=torch.float64, requires_grad=True)
    targets = torch.tensor([12, 83])
    config = objective(method=method)
    kernel = statistical_log_kernel(
        errors=age_errors(targets=targets).double(),
        squared=method == "stat_maxrl_mse_population",
        temperature_years=config.temperature_years,
    )
    expected = (logits.softmax(dim=-1) - (logits + kernel).softmax(dim=-1)) / 2
    loss = compute_loss(logits=logits, targets=targets, config=config)
    torch.testing.assert_close(
        torch.autograd.grad(loss, logits)[0], expected, atol=1e-12, rtol=1e-12
    )


@pytest.mark.parametrize("method", POPULATION_METHODS)
def test_population_objectives_remain_finite_at_extreme_logits(method: Method) -> None:
    logits = torch.full((2, 100), -10000.0, requires_grad=True)
    with torch.no_grad():
        logits[:, 99] = 10000.0
    loss = compute_loss(
        logits=logits, targets=torch.tensor([0, 80]), config=objective(method=method)
    )
    (gradient,) = torch.autograd.grad(loss, logits)
    assert torch.isfinite(loss)
    assert torch.isfinite(gradient).all()


def reference_tailrl(*, row: torch.Tensor, floor: float) -> torch.Tensor:
    result = torch.zeros_like(row)
    for index, reward in enumerate(row):
        previous = 0.0
        for level in row.unique(sorted=True).tolist():
            survival = (row >= level).float().mean().item()
            if reward >= level and survival >= floor:
                result[index] += (level - previous) / survival
            previous = level
    return result - result.mean()


@pytest.mark.parametrize("floor", (0.0, 0.3))
def test_batched_tailrl_matches_threshold_definition_with_ties(floor: float) -> None:
    rewards = torch.tensor([[0, 0.4, 0.4, 1, 0.8], [0.3, 0.3, 0.3, 0.3, 0.3]])
    expected = torch.stack([reference_tailrl(row=row, floor=floor) for row in rewards])
    result = tailrl_advantages(rewards=rewards, survival_floor=floor)
    torch.testing.assert_close(result, expected, atol=1e-6, rtol=1e-6)
    permutation = torch.tensor([4, 0, 2, 1, 3])
    shuffled = tailrl_advantages(rewards=rewards[:, permutation], survival_floor=floor)
    torch.testing.assert_close(shuffled, result[:, permutation], atol=1e-6, rtol=1e-6)


def test_sampled_binary_tailrl_recovers_binary_maxrl_including_all_failures() -> None:
    successes = torch.tensor([
        [0, 1, 0, 1, 0],
        [0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1],
    ]).float()
    torch.testing.assert_close(
        tailrl_advantages(rewards=successes, survival_floor=0.0),
        binary_maxrl_advantages(successes=successes),
        atol=1e-6,
        rtol=1e-6,
    )
    assert not grpo_advantages(
        rewards=torch.ones(2, 4),
        epsilon=1e-8,
        standard_deviation_floor=1e-4,
    ).any()


def test_rloo_is_unbiased_by_exhaustive_rollout_enumeration() -> None:
    probabilities = torch.tensor([0.2, 0.3, 0.5], dtype=torch.float64)
    rewards = torch.tensor([0.1, 0.4, 1.0], dtype=torch.float64)
    groups = torch.tensor(list(product(range(3), repeat=3)))
    group_probabilities = probabilities[groups].prod(dim=-1)
    advantages = rloo_advantages(rewards=rewards[groups])
    scores = F.one_hot(groups, num_classes=3).double() - probabilities
    estimate = (advantages[:, :, None] * scores).mean(dim=1)
    expected = probabilities * (rewards - (probabilities * rewards).sum())
    torch.testing.assert_close(
        (group_probabilities[:, None] * estimate).sum(dim=0),
        expected,
        atol=1e-14,
        rtol=1e-14,
    )
