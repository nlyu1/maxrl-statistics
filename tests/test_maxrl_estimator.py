from __future__ import annotations

import math
from itertools import combinations

import torch

from src.maxrl_estimator import MaxRLEstimatorConfig


def _leave_one_out_omega_from_definition(
    *, complements: list[float], held_out_index: int, degree: int
) -> float:
    peer_complements = [
        value for index, value in enumerate(complements) if index != held_out_index
    ]

    omega = 0.0
    for k in range(degree):
        products = [math.prod(choice) for choice in combinations(peer_complements, k)]
        omega += sum(products) / math.comb(len(peer_complements), k)

    return omega


def _expected_score_weights_from_definition(
    *, likelihoods: torch.Tensor, sup_likelihood: float, degree: int
) -> torch.Tensor:
    normalized_likelihoods = likelihoods / sup_likelihood
    complements = 1.0 - normalized_likelihoods
    expected = torch.empty_like(normalized_likelihoods, dtype=torch.float64)

    for batch_index in range(likelihoods.shape[0]):
        row_complements = complements[batch_index].tolist()
        for rollout_index in range(likelihoods.shape[1]):
            omega = _leave_one_out_omega_from_definition(
                complements=row_complements,
                held_out_index=rollout_index,
                degree=degree,
            )
            expected[batch_index, rollout_index] = (
                normalized_likelihoods[batch_index, rollout_index] * omega
            )

    return expected


def _score_weights(
    *,
    likelihoods: torch.Tensor,
    sup_likelihood: float,
    degree: int,
    subtract_baseline: bool = False,
) -> torch.Tensor:
    config = MaxRLEstimatorConfig.initialize(
        degree=degree,
        sup_likelihood=sup_likelihood,
        subtract_baseline=subtract_baseline,
    )
    score_weights = config.compute_score_weights(log_likelihoods=likelihoods.log())
    return score_weights.to(torch.float64)


def test_exact_small_cases_match_the_leave_one_out_definition() -> None:
    sup_likelihood = 2.0

    for num_rollouts in [2, 3, 5, 8]:
        likelihoods = torch.linspace(
            0.15,
            1.85,
            steps=2 * num_rollouts,
            dtype=torch.float64,
        ).reshape(2, num_rollouts)

        for degree in sorted({1, max(1, num_rollouts // 2), num_rollouts}):
            got = _score_weights(
                likelihoods=likelihoods,
                sup_likelihood=sup_likelihood,
                degree=degree,
            )
            expected = _expected_score_weights_from_definition(
                likelihoods=likelihoods,
                sup_likelihood=sup_likelihood,
                degree=degree,
            )

            assert torch.allclose(got, expected, rtol=1.0e-12, atol=1.0e-12)


def test_tiny_case_can_be_checked_by_hand() -> None:
    complements = torch.tensor([[0.2, 0.4, 0.6, 0.8]], dtype=torch.float64)
    likelihoods = 1.0 - complements

    expected_omega = torch.tensor(
        [
            [
                1.0
                + (0.4 + 0.6 + 0.8) / 3.0
                + (0.4 * 0.6 + 0.4 * 0.8 + 0.6 * 0.8) / 3.0,
                1.0
                + (0.2 + 0.6 + 0.8) / 3.0
                + (0.2 * 0.6 + 0.2 * 0.8 + 0.6 * 0.8) / 3.0,
                1.0
                + (0.2 + 0.4 + 0.8) / 3.0
                + (0.2 * 0.4 + 0.2 * 0.8 + 0.4 * 0.8) / 3.0,
                1.0
                + (0.2 + 0.4 + 0.6) / 3.0
                + (0.2 * 0.4 + 0.2 * 0.6 + 0.4 * 0.6) / 3.0,
            ],
        ],
        dtype=torch.float64,
    )
    expected_score_weights = likelihoods * expected_omega

    got = _score_weights(
        likelihoods=likelihoods,
        sup_likelihood=1.0,
        degree=3,
    )

    assert torch.allclose(
        got,
        expected_score_weights,
        rtol=1.0e-12,
        atol=1.0e-12,
    )


def test_degree_one_returns_normalized_likelihoods() -> None:
    likelihoods = torch.tensor(
        [
            [0.2, 0.7, 1.1, 1.6],
            [1.9, 1.3, 0.5, 0.1],
        ],
        dtype=torch.float64,
    )
    sup_likelihood = 2.0

    got = _score_weights(
        likelihoods=likelihoods,
        sup_likelihood=sup_likelihood,
        degree=1,
    )

    assert torch.allclose(
        got,
        likelihoods / sup_likelihood,
        rtol=1.0e-12,
        atol=1.0e-12,
    )


def test_single_rollout_returns_normalized_likelihoods() -> None:
    likelihoods = torch.tensor([[0.7], [1.9]], dtype=torch.float64)
    sup_likelihood = 2.5
    config = MaxRLEstimatorConfig.initialize(
        degree=1,
        sup_likelihood=sup_likelihood,
        subtract_baseline=False,
    )

    got = config.compute_score_weights(log_likelihoods=likelihoods.log())

    assert torch.allclose(
        got,
        likelihoods / sup_likelihood,
        rtol=1.0e-12,
        atol=1.0e-12,
    )


def test_rollout_permutation_only_permutes_the_outputs() -> None:
    likelihoods = torch.tensor(
        [
            [0.15, 0.35, 0.55, 0.75, 0.95],
            [0.20, 0.90, 0.40, 0.80, 0.60],
        ],
        dtype=torch.float64,
    )
    permutation = torch.tensor([3, 0, 4, 1, 2])
    config = MaxRLEstimatorConfig.initialize(
        degree=5, sup_likelihood=1.0, subtract_baseline=False
    )

    original = config.compute_score_weights(log_likelihoods=likelihoods.log())
    permuted = config.compute_score_weights(
        log_likelihoods=likelihoods[:, permutation].log()
    )

    assert torch.allclose(
        permuted,
        original[:, permutation],
        rtol=1.0e-12,
        atol=1.0e-12,
    )


def test_constant_complements_have_a_geometric_sum_closed_form() -> None:
    num_rollouts = 1024
    degree = 1024
    complements = torch.tensor([0.0, 0.25, 0.75, 0.999], dtype=torch.float64)
    likelihoods = (1.0 - complements[:, None]).expand(-1, num_rollouts)

    got = _score_weights(
        likelihoods=likelihoods,
        sup_likelihood=1.0,
        degree=degree,
    )

    expected_omega = torch.tensor(
        [
            sum(float(complement) ** k for k in range(degree))
            for complement in complements
        ],
        dtype=torch.float64,
    )
    expected_score_weights = (1.0 - complements[:, None]) * expected_omega[:, None]

    assert torch.allclose(
        got,
        expected_score_weights,
        rtol=2.0e-10,
        atol=2.0e-10,
    )


def test_score_weights_are_detached_from_autograd() -> None:
    likelihoods = torch.tensor(
        [[0.2, 0.4, 0.6, 0.8]],
        dtype=torch.float64,
        requires_grad=True,
    )
    config = MaxRLEstimatorConfig.initialize(
        degree=4, sup_likelihood=1.0, subtract_baseline=False
    )

    score_weights = config.compute_score_weights(log_likelihoods=likelihoods.log())

    assert not score_weights.requires_grad


def test_private_core_handles_zero_and_one_complements() -> None:
    complements = torch.tensor(
        [
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 1.0],
            [0.0, 1.0, 0.0, 1.0],
        ],
        dtype=torch.float64,
    )
    degree = 4
    config = MaxRLEstimatorConfig.initialize(
        degree=degree, sup_likelihood=1.0, subtract_baseline=False
    )

    got = config._log_leave_one_out_weight(complement_nl=complements).exp()
    expected = torch.empty_like(got)
    for batch_index in range(complements.shape[0]):
        row_complements = complements[batch_index].tolist()
        for rollout_index in range(complements.shape[1]):
            expected[batch_index, rollout_index] = _leave_one_out_omega_from_definition(
                complements=row_complements,
                held_out_index=rollout_index,
                degree=degree,
            )

    assert torch.allclose(got, expected, rtol=1.0e-12, atol=1.0e-12)


def test_leave_one_out_dp_stays_finite_for_tiny_likelihoods() -> None:
    # The log-space DP is what must survive extreme likelihoods; the final
    # linear-space exit in compute_score_weights underflows as expected.
    num_rollouts = 1024
    degree = 1024
    config = MaxRLEstimatorConfig.initialize(
        degree=degree, sup_likelihood=1.0, subtract_baseline=False
    )
    # sigma_j = exp(-1000) => complement a_j = 1 - sigma_j ~= 1; log a_j ~= 0.
    log_likelihoods = torch.full((1, num_rollouts), -1000.0, dtype=torch.float64)
    complements = -torch.expm1(log_likelihoods)

    log_omega = config._log_leave_one_out_weight(complement_nl=complements)
    expected = torch.full_like(log_omega, math.log(degree))

    assert torch.isfinite(log_omega).all()
    assert torch.allclose(log_omega, expected, rtol=0.0, atol=1.0e-10)


def test_baseline_degree_one_centers_sigma() -> None:
    likelihoods = torch.tensor(
        [
            [0.2, 0.7, 1.1, 1.6],
            [1.9, 1.3, 0.5, 0.1],
        ],
        dtype=torch.float64,
    )
    sup_likelihood = 2.0
    num_rollouts = likelihoods.shape[1]

    got = _score_weights(
        likelihoods=likelihoods,
        sup_likelihood=sup_likelihood,
        degree=1,
        subtract_baseline=True,
    )

    sigma = likelihoods / sup_likelihood
    sigma_bar = sigma.mean(dim=-1, keepdim=True)
    expected = (num_rollouts / (num_rollouts - 1)) * (sigma - sigma_bar)

    assert torch.allclose(got, expected, rtol=1.0e-12, atol=1.0e-12)


def test_baseline_matches_peer_only_form() -> None:
    # With subtract_baseline=True, c_j = omega_j * (sigma_j - sigma_bar_{-j}).
    # Compare to the no-baseline output c_j^0 = omega_j * sigma_j and check
    # c_j^0 - c_j = omega_j * sigma_bar_{-j}.
    likelihoods = torch.tensor(
        [
            [0.10, 0.40, 0.85, 1.25, 1.70],
            [1.95, 0.30, 0.60, 1.10, 0.75],
        ],
        dtype=torch.float64,
    )
    sup_likelihood = 2.0
    degree = 3
    num_rollouts = likelihoods.shape[1]

    without_baseline = _score_weights(
        likelihoods=likelihoods,
        sup_likelihood=sup_likelihood,
        degree=degree,
        subtract_baseline=False,
    )
    with_baseline = _score_weights(
        likelihoods=likelihoods,
        sup_likelihood=sup_likelihood,
        degree=degree,
        subtract_baseline=True,
    )

    sigma = likelihoods / sup_likelihood
    sigma_sum = sigma.sum(dim=-1, keepdim=True)
    peer_mean = (sigma_sum - sigma) / (num_rollouts - 1)
    # Recover omega_j from the no-baseline output: c_j^0 / sigma_j.
    omega = without_baseline / sigma
    expected_baseline_subtracted = omega * peer_mean

    assert torch.allclose(
        without_baseline - with_baseline,
        expected_baseline_subtracted,
        rtol=1.0e-12,
        atol=1.0e-12,
    )


def test_baseline_requires_multiple_rollouts() -> None:
    import pytest

    config = MaxRLEstimatorConfig.initialize(
        degree=1, sup_likelihood=1.0, subtract_baseline=True
    )
    log_likelihoods = torch.tensor([[-0.5]], dtype=torch.float64)

    with pytest.raises(AssertionError):
        config.compute_score_weights(log_likelihoods=log_likelihoods)
