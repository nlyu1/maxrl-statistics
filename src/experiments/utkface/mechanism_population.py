import argparse
import hashlib
import json
import platform
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import scipy
import torch
from numpy.typing import NDArray
from pydantic import ConfigDict
from pydantic.dataclasses import dataclass
from scipy.optimize import Bounds, LinearConstraint, minimize
from scipy.special import logsumexp, ndtr
from torch import Tensor

from src.experiments.utkface.config import LossConfig, Method
from src.experiments.utkface.losses import (
    MAX_AGE,
    NUM_BINS,
    log_policy,
    policy_loss,
    scalar_loss,
)
from src.experiments.utkface.run import write_json

AGES = np.arange(NUM_BINS, dtype=np.float64)
SCORES = (
    -80.0,
    -40.0,
    -20.0,
    -12.0,
    -8.0,
    -4.0,
    -1.0,
    0.0,
    1.0,
    4.0,
    8.0,
    12.0,
    20.0,
    40.0,
    80.0,
)
REFERENCE_TARGETS = (0, 1, 49, 50, 98, 99)
SOURCE_NAMES = (
    "src/experiments/utkface/mechanism_population.py",
    "src/experiments/utkface/config.py",
    "src/experiments/utkface/losses.py",
    "src/experiments/utkface/model.py",
    "src/experiments/utkface/noise.py",
    "src/experiments/utkface/metrics.py",
    "pyproject.toml",
)


@dataclass(kw_only=True, frozen=True, config=ConfigDict(arbitrary_types_allowed=True))
class PopulationProblem:
    q: NDArray[np.float64]
    windows: NDArray[np.float64]
    weights: NDArray[np.float64]

    def value_gradient(
        self, *, probabilities: NDArray[np.float64]
    ) -> tuple[float, NDArray[np.float64]]:
        masses = self.windows @ probabilities
        if np.any(masses <= 0):
            raise ValueError("A positive-weight target window has zero probability")
        value = -float(self.weights @ np.log(masses))
        gradient = -(self.windows.T @ (self.weights / masses))
        return value, gradient

    def hessian(self, *, probabilities: NDArray[np.float64]) -> NDArray[np.float64]:
        masses = self.windows @ probabilities
        return self.windows.T @ ((self.weights / masses**2)[:, None] * self.windows)


def loss_config(*, method: Method) -> LossConfig:
    return LossConfig(
        method=method,
        kernel_scale_years=2.0,
        num_rollouts=64,
        survival_floor=0.0,
        std_floor=1e-4,
    )


def target_distribution(
    *, ages: tuple[int, ...], weights: tuple[float, ...]
) -> NDArray[np.float64]:
    if len(ages) != len(weights) or len(set(ages)) != len(ages):
        raise ValueError("Target ages must be unique and match the weights")
    q = np.zeros(NUM_BINS, dtype=np.float64)
    if any(age < 0 or age > MAX_AGE for age in ages):
        raise ValueError("Target ages must lie in [0, 99]")
    q[list(ages)] = weights
    if not np.isfinite(q).all() or np.any(q < 0) or not np.isclose(q.sum(), 1):
        raise ValueError("Target weights must be nonnegative and sum to one")
    return q / q.sum()


def population_problem(*, q: NDArray[np.float64]) -> PopulationProblem:
    if q.shape != (NUM_BINS,) or not np.isfinite(q).all() or np.any(q < 0):
        raise ValueError(
            "A population law must be a finite 100-action probability vector"
        )
    if abs(float(q.sum()) - 1.0) > 1e-12:
        raise ValueError("A population law must sum to one")
    support = np.flatnonzero(q > 0)
    targets = np.repeat(support, MAX_AGE)
    radii = np.tile(np.arange(MAX_AGE), len(support))
    windows = (np.abs(AGES[None] - targets[:, None]) <= radii[:, None]).astype(
        np.float64
    )
    weights = np.repeat(q[support] / MAX_AGE, MAX_AGE)
    return PopulationProblem(q=q, windows=windows, weights=weights)


def repository_expected_loss(
    *, logits: Tensor, q: NDArray[np.float64], method: Method
) -> Tensor:
    config = loss_config(method=method)
    terms = [
        float(q[target])
        * policy_loss(
            logits=logits[None],
            targets=torch.tensor([target], dtype=torch.int64),
            config=config,
        )
        for target in np.flatnonzero(q > 0).tolist()
    ]
    return torch.stack(terms).sum()


def binomial_equivalence() -> dict:
    reports = []
    for dtype in (torch.float64, torch.float32):
        actions = torch.arange(NUM_BINS, dtype=dtype)
        coefficients = (
            torch.lgamma(torch.tensor(float(NUM_BINS), dtype=dtype))
            - torch.lgamma(actions + 1)
            - torch.lgamma(NUM_BINS - actions)
        )
        rows = []
        for score in SCORES:
            for target in REFERENCE_TARGETS:
                for method in ("mse", "mae"):
                    parameter = torch.tensor(score, dtype=dtype, requires_grad=True)
                    scalar_prediction = MAX_AGE * parameter.sigmoid()
                    logits = coefficients + actions * parameter
                    categorical_prediction = (
                        log_policy(logits=logits[None]).exp()[0] * actions
                    ).sum()
                    labels = torch.tensor([target], dtype=torch.int64)
                    config = loss_config(method=method)
                    scalar_value = scalar_loss(
                        predictions=scalar_prediction.reshape(1),
                        targets=labels,
                        config=config,
                    )
                    categorical_value = policy_loss(
                        logits=logits[None], targets=labels, config=config
                    )
                    scalar_gradient = torch.autograd.grad(
                        scalar_value, parameter, retain_graph=True
                    )[0]
                    categorical_gradient = torch.autograd.grad(
                        categorical_value, parameter, retain_graph=True
                    )[0]
                    scalar_mean_gradient = torch.autograd.grad(
                        scalar_prediction, parameter, retain_graph=True
                    )[0]
                    categorical_mean_gradient = torch.autograd.grad(
                        categorical_prediction, parameter
                    )[0]
                    distance = min(
                        abs(float(scalar_prediction.detach()) - target),
                        abs(float(categorical_prediction.detach()) - target),
                    )
                    rows.append({
                        "score": score,
                        "target": target,
                        "method": method,
                        "scalar_prediction": float(scalar_prediction.detach()),
                        "categorical_prediction": float(
                            categorical_prediction.detach()
                        ),
                        "mean_absolute_error": float(
                            (categorical_prediction - scalar_prediction).detach().abs()
                        ),
                        "loss_absolute_error": float(
                            (categorical_value - scalar_value).detach().abs()
                        ),
                        "scalar_parameter_gradient": float(scalar_gradient),
                        "categorical_parameter_gradient": float(categorical_gradient),
                        "gradient_absolute_error": float(
                            (categorical_gradient - scalar_gradient).abs()
                        ),
                        "mean_gradient_absolute_error": float(
                            (categorical_mean_gradient - scalar_mean_gradient).abs()
                        ),
                        "near_mae_kink": method == "mae" and distance <= 1e-5,
                        "saturated_score": abs(score) >= 20,
                    })
        mean_error = max(row["mean_absolute_error"] for row in rows)
        gradient_error = max(row["gradient_absolute_error"] for row in rows)
        mean_tolerance = 1e-10 if dtype == torch.float64 else 2e-3
        gradient_tolerance = 1e-10 if dtype == torch.float64 else 2e-4
        reports.append({
            "dtype": str(dtype),
            "rows": rows,
            "maximum_mean_absolute_error": mean_error,
            "maximum_loss_absolute_error": max(
                row["loss_absolute_error"] for row in rows
            ),
            "maximum_gradient_absolute_error": gradient_error,
            "maximum_mean_gradient_absolute_error": max(
                row["mean_gradient_absolute_error"] for row in rows
            ),
            "mean_tolerance": mean_tolerance,
            "gradient_tolerance": gradient_tolerance,
            "passed": mean_error <= mean_tolerance
            and gradient_error <= gradient_tolerance,
        })
    return {
        "identity": "a_k=log binom(99,k)+k*s implies E_q[k]=99*sigmoid(s)",
        "parameter_gradient_scope": "The shared parameter is s; arbitrary upstream shared parameters follow by the chain rule.",
        "precision_limit": "Floating-point saturation and MAE kinks are retained in the rows. Float32 tolerances measure numerical agreement, not an exact arithmetic identity.",
        "reports": reports,
    }


def feasible_probabilities(
    *, values: NDArray[np.float64], floor: float
) -> NDArray[np.float64]:
    excess = np.maximum(values - floor, 0.0)
    if float(excess.sum()) <= 0:
        raise ValueError("A candidate has no probability above the numerical floor")
    return floor + (1 - NUM_BINS * floor) * excess / excess.sum()


def optimality_certificate(
    *, problem: PopulationProblem, probabilities: NDArray[np.float64], tolerance: float
) -> dict:
    value, gradient = problem.value_gradient(probabilities=probabilities)
    multiplier = float(probabilities @ gradient)
    reduced = gradient - multiplier
    active = probabilities > 1e-8
    gap = max(0.0, multiplier - float(gradient.min()))
    sum_error = abs(float(probabilities.sum()) - 1)
    nonnegative_violation = max(0.0, -float(probabilities.min()))
    return {
        "tail_loss": value,
        "simplex_sum_error": sum_error,
        "nonnegative_violation": nonnegative_violation,
        "minimum_probability": float(probabilities.min()),
        "gradient_dot_probability": multiplier,
        "gradient_dot_probability_identity_error": abs(multiplier + 1),
        "full_simplex_frank_wolfe_gap": gap,
        "active_stationarity_maximum": float(np.abs(reduced[active]).max()),
        "active_probability_threshold": 1e-8,
        "dual_feasibility_violation": max(0.0, -float(reduced.min())),
        "complementarity_maximum": float(np.max(probabilities * np.abs(reduced))),
        "tolerance": tolerance,
        "certified": sum_error <= 1e-10
        and nonnegative_violation <= 1e-12
        and gap <= tolerance,
        "certificate_scope": "For this convex objective, the full-simplex Frank-Wolfe gap upper-bounds the objective suboptimality. It does not by itself bound policy-mean error when the minimizer is ill-conditioned or nonunique.",
    }


def polish_solution(
    *,
    problem: PopulationProblem,
    probabilities: NDArray[np.float64],
    floor: float,
    tolerance: float,
) -> tuple[NDArray[np.float64], int]:
    current = feasible_probabilities(values=probabilities, floor=floor)
    for iteration in range(40):
        value, gradient = problem.value_gradient(probabilities=current)
        multiplier = float(current @ gradient)
        if multiplier - float(gradient.min()) <= tolerance:
            return current, iteration
        free = (current > 1e-10) | (gradient < multiplier - tolerance)
        indices = np.flatnonzero(free)
        if len(indices) < 2:
            return current, iteration
        hessian = problem.hessian(probabilities=current)[np.ix_(indices, indices)]
        system = np.zeros((len(indices) + 1, len(indices) + 1))
        system[:-1, :-1] = hessian
        system[-1, :-1] = 1
        system[:-1, -1] = 1
        right = np.r_[-gradient[indices], 0.0]
        direction = np.zeros(NUM_BINS)
        direction[indices] = np.linalg.lstsq(system, right, rcond=1e-12)[0][:-1]
        directional_derivative = float(gradient @ direction)
        if directional_derivative >= -1e-18:
            return current, iteration
        negative = direction < 0
        if not np.any(negative):
            return current, iteration
        maximum_step = float(np.min((current[negative] - floor) / -direction[negative]))
        step = min(1.0, 0.99 * maximum_step)
        accepted = False
        for _ in range(50):
            candidate = feasible_probabilities(
                values=current + step * direction, floor=floor
            )
            candidate_value, _ = problem.value_gradient(probabilities=candidate)
            if candidate_value <= value + 1e-4 * step * directional_derivative:
                current = candidate
                accepted = True
                break
            step *= 0.5
        if not accepted:
            return current, iteration
    return current, 40


def solve_population(
    *,
    problem: PopulationProblem,
    starts: int,
    floor: float,
    tolerance: float,
    seed: int,
) -> tuple[NDArray[np.float64], dict]:
    generator = np.random.default_rng(seed)
    initial = [np.full(NUM_BINS, 1 / NUM_BINS), problem.q]
    initial.extend(generator.dirichlet(np.ones(NUM_BINS), size=starts - 2))
    candidates = []
    reports = []
    for index, values in enumerate(initial):
        starting = feasible_probabilities(values=values, floor=floor)

        def objective(probabilities: NDArray[np.float64]):
            value, gradient = problem.value_gradient(probabilities=probabilities)
            return MAX_AGE * value, MAX_AGE * gradient

        result = minimize(
            objective,
            starting,
            method="SLSQP",
            jac=True,
            bounds=Bounds(np.full(NUM_BINS, floor), np.ones(NUM_BINS)),
            constraints=LinearConstraint(np.ones((1, NUM_BINS)), [1.0], [1.0]),
            options={"maxiter": 2000, "ftol": 1e-12},
        )
        if not np.isfinite(result.x).all():
            raise RuntimeError(
                f"Population solver returned nonfinite probabilities: {result.message}"
            )
        probabilities, polishing_iterations = polish_solution(
            problem=problem, probabilities=result.x, floor=floor, tolerance=tolerance
        )
        certificate = optimality_certificate(
            problem=problem, probabilities=probabilities, tolerance=tolerance
        )
        candidates.append(probabilities)
        reports.append({
            "start": index,
            "initialization": "uniform"
            if index == 0
            else "q"
            if index == 1
            else "dirichlet",
            "scipy_success": bool(result.success),
            "scipy_status": int(result.status),
            "scipy_message": str(result.message),
            "scipy_iterations": int(result.nit),
            "polishing_iterations": polishing_iterations,
            "mean": float(probabilities @ AGES),
            "certificate": certificate,
        })
    certified = [
        index for index, row in enumerate(reports) if row["certificate"]["certified"]
    ]
    pool = certified if certified else list(range(len(reports)))
    chosen = min(pool, key=lambda index: reports[index]["certificate"]["tail_loss"])
    means = [row["mean"] for row in reports]
    values = [row["certificate"]["tail_loss"] for row in reports]
    return candidates[chosen], {
        "algorithm": "Convex simplex objective; scipy SLSQP with exact Jacobian, followed by constrained Newton polishing",
        "actions": NUM_BINS,
        "numerical_probability_floor": floor,
        "floor_scope": "The numerical solve uses p_k>=floor; the reported duality gap is for the full simplex p_k>=0, so certification checks the unrestricted problem.",
        "starts": reports,
        "chosen_start": chosen,
        "certified_starts": len(certified),
        "all_starts_certified": len(certified) == len(reports),
        "objective_range_across_starts": max(values) - min(values),
        "mean_range_across_starts": max(means) - min(means),
        "maximum_l1_distance_to_chosen": max(
            float(np.abs(candidate - candidates[chosen]).sum())
            for candidate in candidates
        ),
        "chosen_certificate": reports[chosen]["certificate"],
    }


def policy_summary(
    *, probabilities: NDArray[np.float64], problem: PopulationProblem
) -> dict:
    q = problem.q
    mean = float(probabilities @ AGES)
    median = min(MAX_AGE, int(np.searchsorted(np.cumsum(probabilities), 0.5)))
    q_mean = float(q @ AGES)
    positive = q > 0
    nll = -float(q[positive] @ np.log(probabilities[positive]))
    tail, _ = problem.value_gradient(probabilities=probabilities)
    return {
        "probabilities": probabilities.tolist(),
        "mean": mean,
        "median": median,
        "mode": int(probabilities.argmax()),
        "mean_shift_from_target_mean": mean - q_mean,
        "mean_prediction_mse_years2": float(q @ (mean - AGES) ** 2),
        "mean_prediction_mae_years": float(q @ np.abs(mean - AGES)),
        "median_prediction_mse_years2": float(q @ (median - AGES) ** 2),
        "median_prediction_mae_years": float(q @ np.abs(median - AGES)),
        "mean_mse_excess_over_bayes_years2": (mean - q_mean) ** 2,
        "sampled_action_mse_years2": float(
            probabilities @ (((AGES[:, None] - AGES[None]) ** 2) @ q)
        ),
        "label_nll": nll,
        "tail_loss": tail,
    }


def objective_autograd_audit(*, problem: PopulationProblem) -> dict:
    probabilities = 0.95 * problem.q + 0.05 / NUM_BINS
    logits = torch.tensor(
        np.log(probabilities), dtype=torch.float64, requires_grad=True
    )
    mean = float(probabilities @ AGES)
    q_mean = float(problem.q @ AGES)
    tail_value, tail_gradient = problem.value_gradient(probabilities=probabilities)
    expected = {
        "tailrl_population": (
            tail_value,
            probabilities * (tail_gradient - probabilities @ tail_gradient),
        ),
        "ce": (-float(problem.q @ np.log(probabilities)), probabilities - problem.q),
        "mse": (
            float(problem.q @ (mean - AGES) ** 2) / MAX_AGE**2,
            2 * (mean - q_mean) / MAX_AGE**2 * probabilities * (AGES - mean),
        ),
    }
    rows = []
    for method, (value, gradient) in expected.items():
        actual = repository_expected_loss(logits=logits, q=problem.q, method=method)
        actual_gradient = torch.autograd.grad(actual, logits)[0].numpy()
        rows.append({
            "method": method,
            "analytical_loss": value,
            "repository_loss": float(actual.detach()),
            "loss_absolute_error": abs(float(actual.detach()) - value),
            "logit_gradient_maximum_absolute_error": float(
                np.abs(actual_gradient - gradient).max()
            ),
        })
    generator = np.random.default_rng(20261029)
    direction = generator.normal(size=NUM_BINS)
    direction -= direction.mean()
    direction /= np.linalg.norm(direction)
    step = 1e-6
    plus, _ = problem.value_gradient(probabilities=probabilities + step * direction)
    minus, _ = problem.value_gradient(probabilities=probabilities - step * direction)
    finite_difference_error = abs(
        (plus - minus) / (2 * step) - tail_gradient @ direction
    )
    return {
        "dtype": "float64",
        "repository_checks": rows,
        "tail_simplex_directional_gradient_error": float(finite_difference_error),
        "finite_difference_step": step,
        "passed": all(
            row["loss_absolute_error"] <= 1e-10
            and row["logit_gradient_maximum_absolute_error"] <= 1e-10
            for row in rows
        )
        and finite_difference_error <= 1e-8,
    }


def analyze_population(
    *,
    name: str,
    q: NDArray[np.float64],
    starts: int,
    floor: float,
    tolerance: float,
    seed: int,
) -> dict:
    problem = population_problem(q=q)
    probabilities, solver = solve_population(
        problem=problem, starts=starts, floor=floor, tolerance=tolerance, seed=seed
    )
    ce = policy_summary(probabilities=q, problem=problem)
    tail = policy_summary(probabilities=probabilities, problem=problem)
    return {
        "name": name,
        "target": ce,
        "ce_population_optimum": ce,
        "scalar_mse_population_optimum": {
            "mean": ce["mean"],
            "mse_years2": ce["mean_prediction_mse_years2"],
        },
        "categorical_mean_mse_population_optimum": {
            "mean": ce["mean"],
            "mse_years2": ce["mean_prediction_mse_years2"],
            "policy_identified": False,
            "example_optimum": "The CE target policy q is one member of the mean-MSE optimum set.",
        },
        "tail_population_candidate": tail,
        "tail_minus_ce_mean_years": tail["mean"] - ce["mean"],
        "tail_minus_ce_tail_loss": tail["tail_loss"] - ce["tail_loss"],
        "solver": solver,
        "autograd_audit": objective_autograd_audit(problem=problem),
    }


def population_cases() -> list[tuple[str, NDArray[np.float64]]]:
    cases = (
        ("irregular_boundary", (0, 1, 3), (0.6, 0.3, 0.1)),
        ("irregular_interior", (48, 49, 51), (0.6, 0.3, 0.1)),
        ("irregular_reflected_boundary", (96, 98, 99), (0.1, 0.3, 0.6)),
        ("irregular_reflected_interior", (48, 50, 51), (0.1, 0.3, 0.6)),
        ("adjacent_mean_preserving_boundary", (0, 1, 2), (0.5, 1 / 3, 1 / 6)),
        ("adjacent_mean_preserving_interior", (48, 49, 50), (0.5, 1 / 3, 1 / 6)),
        ("four_age_skew_boundary", (0, 1, 3, 7), (0.45, 0.25, 0.2, 0.1)),
        ("four_age_skew_interior", (45, 46, 48, 52), (0.45, 0.25, 0.2, 0.1)),
        ("four_age_symmetric_boundary", (0, 33, 66, 99), (0.1, 0.4, 0.4, 0.1)),
        ("four_age_equal_mean_boundary", (0, 33, 66, 99), (0.3, 0.1, 0.4, 0.2)),
        ("four_age_symmetric_interior", (20, 40, 60, 80), (0.1, 0.4, 0.4, 0.1)),
        ("four_age_equal_mean_interior", (20, 40, 60, 80), (0.3, 0.1, 0.4, 0.2)),
        ("asymmetric_multimodal", (10, 25, 60, 88), (0.45, 0.05, 0.1, 0.4)),
    )
    return [
        (name, target_distribution(ages=ages, weights=weights))
        for name, ages, weights in cases
    ]


def adjacent_analytic_audit(*, reports: list[dict]) -> list[dict]:
    root = np.sqrt(10.0)
    weights = ((10 - root) / 15, 2 * root / 15, (5 - root) / 15)
    rows = []
    for offset, name in (
        (0, "adjacent_mean_preserving_boundary"),
        (48, "adjacent_mean_preserving_interior"),
    ):
        if name not in {row["name"] for row in reports}:
            continue
        report = next(row for row in reports if row["name"] == name)
        reference = target_distribution(
            ages=(offset, offset + 1, offset + 2), weights=weights
        )
        problem = population_problem(q=np.asarray(report["target"]["probabilities"]))
        certificate = optimality_certificate(
            problem=problem, probabilities=reference, tolerance=1e-8
        )
        rows.append({
            "name": name,
            "closed_form_policy": reference.tolist(),
            "closed_form_mean": float(reference @ AGES),
            "target_mean": float(problem.q @ AGES),
            "solver_policy_l1_error": float(
                np.abs(
                    reference - report["tail_population_candidate"]["probabilities"]
                ).sum()
            ),
            "closed_form_certificate": certificate,
        })
    return rows


def wrong_way_tail_audit() -> dict:
    probabilities = np.full(NUM_BINS, 1e-15)
    probabilities[:3] = (0.01, 0.8, 0.19)
    probabilities /= probabilities.sum()
    q = target_distribution(ages=(0,), weights=(1.0,))
    logits = torch.tensor(
        np.log(probabilities), dtype=torch.float64, requires_grad=True
    )
    policy_mean = (logits.softmax(-1) * torch.arange(NUM_BINS)).sum()
    mean_gradient = torch.autograd.grad(policy_mean, logits, retain_graph=True)[0]
    value = repository_expected_loss(logits=logits, q=q, method="tailrl_population")
    tail_gradient = torch.autograd.grad(value, logits)[0]
    step = 1e-3
    after = logits.detach() - step * tail_gradient
    after_value = repository_expected_loss(
        logits=after, q=q, method="tailrl_population"
    )
    after_mean = (after.softmax(-1) * torch.arange(NUM_BINS)).sum()
    alignment = float(mean_gradient @ tail_gradient)
    return {
        "target": 0,
        "probabilities": probabilities.tolist(),
        "mean": float(policy_mean.detach()),
        "mean_gradient_dot_tail_gradient": alignment,
        "mean_directional_derivative_under_tail_descent": -alignment,
        "tail_loss_before": float(value.detach()),
        "tail_loss_after": float(after_value),
        "mean_after": float(after_mean),
        "gradient_step": step,
        "verified_wrong_way_step": float(after_value) < float(value.detach())
        and float(after_mean) > float(policy_mean.detach()),
        "scope": "A local optimization counterexample; it is not a population optimum or a generalization result.",
    }


def rounded_clipped_law(*, age: int, std: float) -> NDArray[np.float64]:
    if not 0 <= age <= MAX_AGE or std <= 0 or not np.isfinite(std):
        raise ValueError(
            "Rounded/clipped Gaussian noise needs an age in [0,99] and positive SD"
        )
    lower = (AGES[1:-1] - 0.5 - age) / std
    upper = (AGES[1:-1] + 0.5 - age) / std
    interior = np.where(
        lower >= 0, ndtr(-lower) - ndtr(-upper), ndtr(upper) - ndtr(lower)
    )
    q = np.r_[ndtr((0.5 - age) / std), interior, ndtr((age - 98.5) / std)]
    if np.any(q < 0) or not np.isfinite(q).all() or abs(float(q.sum()) - 1) > 1e-12:
        raise ValueError("Rounded/clipped Gaussian probabilities failed normalization")
    return q / q.sum()


def noise_law_report(*, std: float) -> dict:
    laws = np.stack([rounded_clipped_law(age=age, std=std) for age in range(NUM_BINS)])
    means = laws @ AGES
    bias = means - AGES
    groups = {}
    for name, mask in (
        ("lower_boundary_0_19", AGES < 20),
        ("interior_20_79", (AGES >= 20) & (AGES <= 79)),
        ("upper_boundary_80_99", AGES >= 80),
    ):
        groups[name] = {
            "original_age_count": int(mask.sum()),
            "mean_bias_years": float(bias[mask].mean()),
            "mean_absolute_bias_years": float(np.abs(bias[mask]).mean()),
            "maximum_absolute_bias_years": float(np.abs(bias[mask]).max()),
        }
    return {
        "std_years": std,
        "laws_by_original_age": laws.tolist(),
        "conditional_target_means": means.tolist(),
        "conditional_bias_years": bias.tolist(),
        "scalar_mse_optimal_prediction": means.tolist(),
        "categorical_mean_mse_optimal_prediction": means.tolist(),
        "ce_mean_prediction_at_population_optimum": means.tolist(),
        "groups_uniform_over_original_age": groups,
        "maximum_normalization_error": float(np.abs(laws.sum(1) - 1).max()),
        "reflection_symmetry_error_years": float(
            np.abs(means + means[::-1] - MAX_AGE).max()
        ),
        "scope": "Exact expectation over fresh independent rounded/clipped labels conditional on each original integer annotation. These are not the realized targets of a fixed corruption field, and groups weight original ages uniformly rather than by UTKFace frequency.",
    }


def logit_objective_values(
    *, log_probabilities: NDArray[np.float64], q: NDArray[np.float64]
) -> dict:
    probabilities = np.exp(log_probabilities)
    mean = float(probabilities @ AGES)
    targets = np.repeat(np.arange(NUM_BINS), MAX_AGE)
    radii = np.tile(np.arange(MAX_AGE), NUM_BINS)
    windows = np.abs(AGES[None] - targets[:, None]) <= radii[:, None]
    masked = np.where(windows, log_probabilities[None], -np.inf)
    log_masses = logsumexp(masked, axis=1)
    weights = np.repeat(q / MAX_AGE, MAX_AGE)
    posterior = np.exp(masked - log_masses[:, None])
    tail_gradient = probabilities - weights @ posterior
    residual = mean - float(q @ AGES)
    scalar_gradient = 2 * residual / MAX_AGE**2
    return {
        "mean": mean,
        "scalar_mse_loss": float(q @ (mean - AGES) ** 2) / MAX_AGE**2,
        "scalar_mse_prediction_gradient": scalar_gradient,
        "categorical_mse_loss": float(q @ (mean - AGES) ** 2) / MAX_AGE**2,
        "categorical_mse_logit_gradient": scalar_gradient
        * probabilities
        * (AGES - mean),
        "ce_loss": -float(q @ log_probabilities),
        "ce_logit_gradient": probabilities - q,
        "tail_loss": -float(weights @ log_masses),
        "tail_logit_gradient": tail_gradient,
    }


def noise_gradient_audit(
    *, logits: NDArray[np.float64], age: int, std: float, verify: bool
) -> dict:
    log_probabilities = logits - logsumexp(logits)
    q = rounded_clipped_law(age=age, std=std)
    clean_q = target_distribution(ages=(age,), weights=(1.0,))
    clean = logit_objective_values(log_probabilities=log_probabilities, q=clean_q)
    noisy = logit_objective_values(log_probabilities=log_probabilities, q=q)
    methods = {}
    for label, prefix in (
        ("scalar_mse", "scalar_mse"),
        ("categorical_mean_mse", "categorical_mse"),
        ("ce", "ce"),
        ("tail", "tail"),
    ):
        if label == "scalar_mse":
            gradient = np.asarray([noisy["scalar_mse_prediction_gradient"]])
            clean_gradient = np.asarray([clean["scalar_mse_prediction_gradient"]])
        else:
            gradient = noisy[f"{prefix}_logit_gradient"]
            clean_gradient = clean[f"{prefix}_logit_gradient"]
        methods[label] = {
            "exact_expected_loss": noisy[f"{prefix}_loss"],
            "clean_label_loss": clean[f"{prefix}_loss"],
            "expected_gradient_norm": float(np.linalg.norm(gradient)),
            "clean_gradient_norm": float(np.linalg.norm(clean_gradient)),
            "gradient_bias_norm": float(np.linalg.norm(gradient - clean_gradient)),
            "expected_gradient": gradient.tolist(),
            "gradient_bias": (gradient - clean_gradient).tolist(),
            "gradient_coordinates": "prediction in years"
            if label == "scalar_mse"
            else "100 categorical logits",
        }
    checks = []
    if verify:
        tensor = torch.tensor(logits, dtype=torch.float64, requires_grad=True)
        for method, prefix in (
            ("mse", "categorical_mse"),
            ("ce", "ce"),
            ("tailrl_population", "tail"),
        ):
            value = repository_expected_loss(logits=tensor, q=q, method=method)
            gradient = torch.autograd.grad(value, tensor)[0].numpy()
            checks.append({
                "method": method,
                "loss_absolute_error": abs(
                    float(value.detach()) - noisy[f"{prefix}_loss"]
                ),
                "gradient_maximum_absolute_error": float(
                    np.abs(gradient - noisy[f"{prefix}_logit_gradient"]).max()
                ),
            })
        prediction = torch.tensor(
            noisy["mean"], dtype=torch.float64, requires_grad=True
        )
        scalar_value = torch.stack([
            float(q[target])
            * scalar_loss(
                predictions=prediction.reshape(1),
                targets=torch.tensor([target]),
                config=loss_config(method="mse"),
            )
            for target in np.flatnonzero(q > 0).tolist()
        ]).sum()
        scalar_gradient = torch.autograd.grad(scalar_value, prediction)[0]
        checks.append({
            "method": "scalar_mse",
            "loss_absolute_error": abs(
                float(scalar_value.detach()) - noisy["scalar_mse_loss"]
            ),
            "gradient_maximum_absolute_error": abs(
                float(scalar_gradient) - noisy["scalar_mse_prediction_gradient"]
            ),
        })
    return {
        "original_age": age,
        "std_years": std,
        "conditional_target_mean": float(q @ AGES),
        "conditional_target_bias_years": float(q @ AGES) - age,
        "policy_mean": noisy["mean"],
        "methods": methods,
        "repository_autograd_checks": checks,
        "repository_checks_performed": verify,
        "repository_checks_passed": all(
            row["loss_absolute_error"] <= 1e-10
            and row["gradient_maximum_absolute_error"] <= 1e-10
            for row in checks
        )
        if verify
        else None,
    }


def load_logit_rows(
    *, path: Path, maximum_rows: int
) -> tuple[NDArray[np.float64], NDArray[np.int64], dict]:
    if path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as saved:
            logits = np.asarray(saved["logits"], dtype=np.float64)
            targets = np.asarray(saved["targets"])
    elif path.suffix == ".pt":
        saved = torch.load(path, map_location="cpu", weights_only=True)
        logits = saved["logits"].double().numpy()
        targets = saved["targets"].numpy()
    elif path.suffix == ".json":
        saved = json.loads(path.read_text())
        logits = np.asarray(saved["logits"], dtype=np.float64)
        targets = np.asarray(saved["targets"])
    else:
        raise ValueError(
            "Saved outputs must be .npz, .pt or .json with logits and targets"
        )
    if (
        logits.ndim != 2
        or logits.shape[1] != NUM_BINS
        or targets.shape != (len(logits),)
    ):
        raise ValueError(
            "Saved logits must be [N,100] and original-age targets must be [N]"
        )
    if (
        len(logits) == 0
        or not np.isfinite(logits).all()
        or not np.isfinite(targets).all()
    ):
        raise ValueError("Saved logits and targets must be nonempty and finite")
    if not np.equal(targets, np.rint(targets)).all() or np.any(
        (targets < 0) | (targets > MAX_AGE)
    ):
        raise ValueError("Saved targets must be original integer ages in [0,99]")
    selected = np.linspace(
        0, len(logits) - 1, min(len(logits), maximum_rows), dtype=np.int64
    )
    return (
        logits[selected],
        targets[selected].astype(np.int64),
        {
            "path": str(path.resolve()),
            "sha256": file_sha256(path=path),
            "available_rows": len(logits),
            "selected_rows": selected.tolist(),
            "selection": "All rows up to maximum_rows, otherwise evenly spaced input rows; no outcome selection.",
        },
    )


def file_sha256(*, path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def provenance() -> dict:
    root = Path(__file__).resolve().parents[3]
    return {
        "generated_utc": datetime.now(UTC).isoformat(),
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
        "source_sha256": {name: file_sha256(path=root / name) for name in SOURCE_NAMES},
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "hostname": platform.node(),
        "device": "cpu",
        "precision": "float64 analytical/convex probes; float32 binomial numerical comparison",
        "execution_scope": "No neural-network training or GPU benchmark; CPU float64 is deliberate for numerical certificates.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--starts", type=int, default=3)
    parser.add_argument("--probability-floor", type=float, default=1e-14)
    parser.add_argument("--certificate-tolerance", type=float, default=1e-8)
    parser.add_argument(
        "--noise-stds", nargs="+", type=float, default=(1.0, 3.0, 5.0, 10.0)
    )
    parser.add_argument("--noise-solve-ages", nargs="+", type=int, default=(0, 49, 99))
    parser.add_argument(
        "--cases",
        nargs="+",
        choices=[name for name, _ in population_cases()],
        default=[name for name, _ in population_cases()],
    )
    parser.add_argument("--skip-noise-solves", action="store_true")
    parser.add_argument("--logits", nargs="*", type=Path, default=())
    parser.add_argument("--maximum-logit-rows", type=int, default=16)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.starts < 3 or not 0 < args.probability_floor < 1 / NUM_BINS:
        raise ValueError(
            "Use at least three starts and a probability floor in (0,0.01)"
        )
    if args.certificate_tolerance <= 0 or args.maximum_logit_rows < 1:
        raise ValueError(
            "Certificate tolerance and maximum logit rows must be positive"
        )
    if any(std <= 0 or not np.isfinite(std) for std in args.noise_stds):
        raise ValueError("Noise SDs must be finite and positive")
    if any(age < 0 or age > MAX_AGE for age in args.noise_solve_ages):
        raise ValueError("Noise solve ages must lie in [0,99]")
    torch.set_num_threads(1)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "status": "running",
        "provenance": provenance(),
        "binomial_equivalence": binomial_equivalence(),
        "wrong_way_tail": wrong_way_tail_audit(),
        "population_cases": [],
        "noise_laws": [],
        "noise_population_cases": [],
        "saved_logit_audits": [],
        "limits": [
            "Population results concern unrestricted 100-action conditional policies, not shared neural features, finite samples, optimizer selection or generalization.",
            "Only certified numerical candidates have the reported convex-objective error bound; optimizer success alone is insufficient.",
            "An objective-gap certificate alone does not quantify error in a policy mean; multiple-start mean agreement is a diagnostic, not a rigorous mean interval.",
            "Rounded/clipped Gaussian expectations model fresh independent labels conditional on an original annotation, not verified true age or a fixed realized corruption field.",
            "Optional saved-logit probes require logits and original-age targets; checkpoint inference and checkpoint selection are outside this script.",
        ],
    }
    write_json(path=args.output, payload=report)
    cases = [(name, q) for name, q in population_cases() if name in args.cases]
    for index, (name, q) in enumerate(cases):
        row = analyze_population(
            name=name,
            q=q,
            starts=args.starts,
            floor=args.probability_floor,
            tolerance=args.certificate_tolerance,
            seed=2026102900 + index,
        )
        report["population_cases"].append(row)
        write_json(path=args.output, payload=report)
        print(
            json.dumps({
                "population_case": name,
                "mean_shift": row["tail_minus_ce_mean_years"],
                "duality_gap": row["solver"]["chosen_certificate"][
                    "full_simplex_frank_wolfe_gap"
                ],
                "certified": row["solver"]["chosen_certificate"]["certified"],
            }),
            flush=True,
        )
    report["adjacent_closed_form_audits"] = adjacent_analytic_audit(
        reports=report["population_cases"]
    )
    for std_index, std in enumerate(args.noise_stds):
        report["noise_laws"].append(noise_law_report(std=std))
        for age in () if args.skip_noise_solves else args.noise_solve_ages:
            row = analyze_population(
                name=f"rounded_clipped_sd{std:g}_original{age}",
                q=rounded_clipped_law(age=age, std=std),
                starts=args.starts,
                floor=args.probability_floor,
                tolerance=args.certificate_tolerance,
                seed=2026103000 + std_index * NUM_BINS + age,
            )
            row["original_age"] = age
            row["std_years"] = std
            row["tail_mean_bias_from_original_years"] = (
                row["tail_population_candidate"]["mean"] - age
            )
            row["scalar_ce_mean_bias_from_original_years"] = row["target"]["mean"] - age
            report["noise_population_cases"].append(row)
            write_json(path=args.output, payload=report)
            print(
                json.dumps({
                    "noise_case": row["name"],
                    "tail_mean": row["tail_population_candidate"]["mean"],
                    "target_mean": row["target"]["mean"],
                    "certified": row["solver"]["chosen_certificate"]["certified"],
                }),
                flush=True,
            )
    for path in args.logits:
        logits, targets, metadata = load_logit_rows(
            path=path, maximum_rows=args.maximum_logit_rows
        )
        rows = [
            noise_gradient_audit(
                logits=values, age=int(age), std=std, verify=index == 0
            )
            for index, (values, age) in enumerate(zip(logits, targets, strict=True))
            for std in args.noise_stds
        ]
        report["saved_logit_audits"].append({"source": metadata, "rows": rows})
        write_json(path=args.output, payload=report)
    all_cases = report["population_cases"] + report["noise_population_cases"]
    report["all_population_candidates_certified"] = all(
        row["solver"]["chosen_certificate"]["certified"] for row in all_cases
    )
    report["all_repository_autograd_checks_passed"] = all(
        row["autograd_audit"]["passed"] for row in all_cases
    )
    report["status"] = "complete"
    write_json(path=args.output, payload=report)
    print(
        json.dumps({
            "output": str(args.output),
            "population_cases": len(all_cases),
            "all_candidates_certified": report["all_population_candidates_certified"],
            "all_autograd_checks_passed": report[
                "all_repository_autograd_checks_passed"
            ],
            "binomial_checks_passed": all(
                row["passed"] for row in report["binomial_equivalence"]["reports"]
            ),
        }),
        flush=True,
    )


if __name__ == "__main__":
    main()
