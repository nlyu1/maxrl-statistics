from __future__ import annotations

import math
from itertools import combinations

import torch
from jaxtyping import Float
from torch import Tensor

_GAUSS_LEGENDRE_CACHE: dict[tuple[int, torch.device, torch.dtype], tuple[Tensor, Tensor]] = {}


def compute_maxrl_score_weights(
    *,
    log_likelihoods: Float[Tensor, "batch rollout"],
    likelihood_ceiling: float,
    truncation_order: int | None = None,
    computation_dtype: torch.dtype | None = None,
    validate_inputs: bool = True,
) -> Float[Tensor, "batch rollout"]:
    """
    Compute detached score-function weights for generalized MaxRL.

    For each batch item, rollouts z_j have data likelihoods

        q_j = q(y | z_j),       0 <= q_j <= beta,

    where beta is ``likelihood_ceiling``. We normalize immediately,

        l_j = q_j / beta in [0, 1],
        p = E[l_j].

    Since log E[q_j] = log(beta) + log(p), the beta term is constant in
    theta. The relevant maximum-likelihood term is log(p), with Taylor
    expansion around p=1:

        log(p) = -sum_{k >= 1} (1 - p)^k / k,       0 < p <= 1.

    The order-T truncated objective is

        J_T(p) = -sum_{k=1}^T (1 - p)^k / k,

    whose gradient is

        grad J_T(p) = w_T(p) grad p,
        w_T(p) = sum_{k=0}^{T-1} (1 - p)^k.

    With R rollouts, the default is T=R: the largest objective order whose
    derivative polynomial can be estimated from R-1 leave-one-out samples.
    For rollout j, define failures a_i = 1 - l_i and let e_k(a_-j) be the
    degree-k elementary symmetric polynomial over all failures except j.
    The leave-one-out derivative-weight estimator is

        omega_j = sum_{k=0}^{T-1} e_k(a_-j) / binom(R - 1, k).

    Each term e_k/binom(R-1,k) is a U-statistic with expectation
    (1-p)^k, so E[omega_j] = w_T(p). Since omega_j is built without sample
    j, omega_j is independent of l_j * grad log m_theta(z_j | x). Therefore

        (1/R) sum_j omega_j l_j grad log m_theta(z_j | x)

    is an unbiased estimator of grad J_T(p).

    This function returns the per-rollout equivalent-loss weights

        score_weight_j = omega_j * l_j.

    Use them as ``loss = -(policy_logprobs * score_weights.detach()).mean()``.
    Batch rows are independent; each row may have a different underlying p.
    """
    scaled_likelihoods = normalize_likelihoods_by_ceiling(
        log_likelihoods=log_likelihoods,
        likelihood_ceiling=likelihood_ceiling,
        computation_dtype=computation_dtype,
        validate_inputs=validate_inputs,
    )
    derivative_weights = estimate_maxrl_derivative_weights(
        scaled_likelihoods=scaled_likelihoods,
        truncation_order=truncation_order,
    )
    score_weights = scaled_likelihoods * derivative_weights

    rollout_count = scaled_likelihoods.shape[1]
    order = rollout_count if truncation_order is None else truncation_order
    if order == rollout_count:
        score_weights = _replace_binary_full_order_score_weights(
            scaled_likelihoods=scaled_likelihoods,
            score_weights=score_weights,
        )

    return score_weights.detach()


def normalize_likelihoods_by_ceiling(
    *,
    log_likelihoods: Float[Tensor, "batch rollout"],
    likelihood_ceiling: float,
    computation_dtype: torch.dtype | None = None,
    validate_inputs: bool = True,
) -> Float[Tensor, "batch rollout"]:
    """
    Convert log q(y|z) values to normalized likelihoods l=q/beta in [0, 1].

    The Maclaurin expansion used by MaxRL is valid for the normalized
    probability p=E[l] in (0,1]. If ``likelihood_ceiling`` is not really a
    ceiling, the expansion is being used outside its intended radius, so the
    default behavior is to raise instead of silently clipping real errors.
    """
    if log_likelihoods.ndim != 2:
        raise ValueError(
            "log_likelihoods must have shape [batch, rollout]; "
            f"got shape {tuple(log_likelihoods.shape)}."
        )
    if log_likelihoods.shape[1] < 1:
        raise ValueError("log_likelihoods must contain at least one rollout.")
    if likelihood_ceiling <= 0.0 or not math.isfinite(likelihood_ceiling):
        raise ValueError(
            "likelihood_ceiling must be a positive finite scalar; "
            f"got {likelihood_ceiling}."
        )

    work_dtype = _resolve_computation_dtype(
        input_dtype=log_likelihoods.dtype,
        computation_dtype=computation_dtype,
    )
    detached_log_likelihoods = log_likelihoods.detach().to(dtype=work_dtype)

    if validate_inputs:
        if bool(torch.isnan(detached_log_likelihoods).any().item()):
            raise ValueError("log_likelihoods must not contain NaN.")
        if bool(torch.isposinf(detached_log_likelihoods).any().item()):
            raise ValueError("log_likelihoods must not contain +inf.")

    log_ceiling = math.log(likelihood_ceiling)

    if validate_inputs:
        tolerance = 1e-6 if work_dtype == torch.float64 else 1e-4
        max_log_overshoot = detached_log_likelihoods.max() - log_ceiling
        if bool((max_log_overshoot > tolerance).item()):
            raise ValueError(
                "likelihood_ceiling must upper-bound all likelihoods. "
                f"largest log-likelihood exceeds log(ceiling) by "
                f"{float(max_log_overshoot.detach().cpu()):.6g}."
            )

    scaled_likelihoods = torch.exp(detached_log_likelihoods - log_ceiling)
    return scaled_likelihoods.clamp(max=1.0)


def estimate_maxrl_derivative_weights(
    *,
    scaled_likelihoods: Float[Tensor, "batch rollout"],
    truncation_order: int | None = None,
) -> Float[Tensor, "batch rollout"]:
    """
    Estimate w_T(p)=sum_{k=0}^{T-1}(1-p)^k for every leave-one-out rollout.

    Args:
        scaled_likelihoods: Detached l=q/beta values in [0,1].
        truncation_order: Objective truncation order T. Defaults to the rollout
            count R, giving the largest unbiased order available from R samples.

    Returns:
        omega_j values with shape [batch, rollout]. Multiplying by l_j gives
        score-function weights for an unbiased gradient estimator of J_T.
    """
    if scaled_likelihoods.ndim != 2:
        raise ValueError(
            "scaled_likelihoods must have shape [batch, rollout]; "
            f"got shape {tuple(scaled_likelihoods.shape)}."
        )
    rollout_count = scaled_likelihoods.shape[1]
    order = rollout_count if truncation_order is None else truncation_order
    _validate_truncation_order(
        truncation_order=order,
        rollout_count=rollout_count,
    )

    max_degree = order - 1
    if max_degree == 0:
        return torch.ones_like(scaled_likelihoods)
    if order == rollout_count:
        return estimate_full_order_derivative_weights_quadrature(
            scaled_likelihoods=scaled_likelihoods,
        )

    failures = 1.0 - scaled_likelihoods
    full_symmetric_means = normalized_elementary_symmetric_polynomials(
        values=failures,
        max_degree=max_degree,
    )

    omega = torch.ones_like(failures)
    loo_symmetric_mean_previous = torch.ones_like(failures)
    rollout_count_float = float(rollout_count)

    for degree in range(1, max_degree + 1):
        denominator = rollout_count_float - float(degree)
        loo_symmetric_mean = (
            rollout_count_float * full_symmetric_means[:, degree, None]
            - float(degree) * failures * loo_symmetric_mean_previous
        ) / denominator
        omega = omega + loo_symmetric_mean
        loo_symmetric_mean_previous = loo_symmetric_mean

    return omega


def estimate_full_order_derivative_weights_quadrature(
    *,
    scaled_likelihoods: Float[Tensor, "batch rollout"],
    quadrature_chunk_size: int = 128,
) -> Float[Tensor, "batch rollout"]:
    """
    Float32-stable full-order leave-one-out estimator.

    This handles the default T=R case. Let M=R-1 and a_i=1-l_i. For any
    leave-one-out set,

        sum_{k=0}^M e_k(a_-j) / binom(M, k)
        = R * integral_0^1 prod_{i != j} (1 - t l_i) dt.

    The integrand is a degree-M polynomial. Gauss-Legendre quadrature with
    ceil(R/2) nodes is exact in real arithmetic, and this implementation
    evaluates the positive products with logsumexp instead of subtracting
    neighboring elementary-symmetric coefficients.
    """
    if scaled_likelihoods.ndim != 2:
        raise ValueError(
            "scaled_likelihoods must have shape [batch, rollout]; "
            f"got shape {tuple(scaled_likelihoods.shape)}."
        )
    if quadrature_chunk_size < 1:
        raise ValueError(
            "quadrature_chunk_size must be positive; "
            f"got {quadrature_chunk_size}."
        )

    batch_size, rollout_count = scaled_likelihoods.shape
    node_count = (rollout_count + 1) // 2
    nodes, weights = _gauss_legendre_unit_interval(
        node_count=node_count,
        device=scaled_likelihoods.device,
        dtype=scaled_likelihoods.dtype,
    )
    log_quadrature_sum = scaled_likelihoods.new_full(
        (batch_size, rollout_count),
        -torch.inf,
    )

    for start in range(0, node_count, quadrature_chunk_size):
        stop = min(start + quadrature_chunk_size, node_count)
        nodes_chunk = nodes[start:stop]
        weights_chunk = weights[start:stop]
        log_factors = torch.log1p(
            -scaled_likelihoods[:, None, :] * nodes_chunk[None, :, None]
        )
        log_all_rollout_products = log_factors.sum(dim=-1)
        log_leave_one_out_products = log_all_rollout_products[:, :, None] - log_factors
        log_weighted_terms = (
            weights_chunk.log()[None, :, None] + log_leave_one_out_products
        )
        log_chunk_sum = torch.logsumexp(log_weighted_terms, dim=1)
        log_quadrature_sum = torch.logaddexp(log_quadrature_sum, log_chunk_sum)

    return float(rollout_count) * log_quadrature_sum.exp()


def normalized_elementary_symmetric_polynomials(
    *,
    values: Float[Tensor, "batch sample"],
    max_degree: int,
) -> Float[Tensor, "batch degree_plus_one"]:
    """
    Compute normalized elementary symmetric polynomials.

    For n input values a_1,...,a_n this returns U_k for k=0..max_degree:

        U_k = e_k(a_1,...,a_n) / binom(n, k),      U_0 = 1.

    The normalization is essential for R around 1024: raw e_k terms are
    binomial-scale and can overflow, while U_k remains in the convex hull of
    products of k values. For values in [0,1], every U_k is also in [0,1].

    The update after observing the n-th value a is

        U'_k = ((n-k)/n) U_k + (k/n) a U_{k-1}.
    """
    if values.ndim != 2:
        raise ValueError(
            "values must have shape [batch, sample]; "
            f"got shape {tuple(values.shape)}."
        )
    if max_degree < 0:
        raise ValueError(f"max_degree must be non-negative; got {max_degree}.")
    sample_count = values.shape[1]
    if max_degree > sample_count:
        raise ValueError(
            "max_degree cannot exceed the number of samples; "
            f"got max_degree={max_degree}, sample_count={sample_count}."
        )

    coefficients = values.new_zeros(values.shape[0], max_degree + 1)
    coefficients[:, 0] = 1.0
    if max_degree == 0:
        return coefficients

    degrees = torch.arange(
        max_degree + 1,
        device=values.device,
        dtype=values.dtype,
    )

    for sample_index in range(sample_count):
        observed_count = sample_index + 1
        upper_degree = min(observed_count, max_degree)
        degree_slice = degrees[1 : upper_degree + 1]
        old_same_degree = coefficients[:, 1 : upper_degree + 1].clone()
        old_previous_degree = coefficients[:, :upper_degree].clone()
        keep_same = (float(observed_count) - degree_slice) / float(observed_count)
        add_previous = degree_slice / float(observed_count)
        coefficients[:, 1 : upper_degree + 1] = (
            keep_same[None, :] * old_same_degree
            + add_previous[None, :]
            * values[:, sample_index, None]
            * old_previous_degree
        )

    return coefficients


def compute_maxrl_score_weights_reference(
    *,
    log_likelihoods: Float[Tensor, "batch rollout"],
    likelihood_ceiling: float,
    truncation_order: int | None = None,
    computation_dtype: torch.dtype | None = torch.float32,
    validate_inputs: bool = True,
) -> Float[Tensor, "batch rollout"]:
    """
    Slow reference implementation using explicit subset averages.

    This is intended for small-rollout tests and derivation checks. Runtime is
    exponential in ``truncation_order`` because it enumerates combinations.
    """
    scaled_likelihoods = normalize_likelihoods_by_ceiling(
        log_likelihoods=log_likelihoods,
        likelihood_ceiling=likelihood_ceiling,
        computation_dtype=computation_dtype,
        validate_inputs=validate_inputs,
    )
    derivative_weights = estimate_maxrl_derivative_weights_reference(
        scaled_likelihoods=scaled_likelihoods,
        truncation_order=truncation_order,
    )
    return (scaled_likelihoods * derivative_weights).detach()


def estimate_maxrl_derivative_weights_reference(
    *,
    scaled_likelihoods: Float[Tensor, "batch rollout"],
    truncation_order: int | None = None,
) -> Float[Tensor, "batch rollout"]:
    """Explicit leave-one-out U-statistic estimator for small test cases."""
    if scaled_likelihoods.ndim != 2:
        raise ValueError(
            "scaled_likelihoods must have shape [batch, rollout]; "
            f"got shape {tuple(scaled_likelihoods.shape)}."
        )
    rollout_count = scaled_likelihoods.shape[1]
    order = rollout_count if truncation_order is None else truncation_order
    _validate_truncation_order(
        truncation_order=order,
        rollout_count=rollout_count,
    )

    batch_weights: list[Tensor] = []
    for batch_index in range(scaled_likelihoods.shape[0]):
        rollout_weights: list[Tensor] = []
        for rollout_index in range(rollout_count):
            keep_mask = torch.ones(
                rollout_count,
                device=scaled_likelihoods.device,
                dtype=torch.bool,
            )
            keep_mask[rollout_index] = False
            loo_failures = 1.0 - scaled_likelihoods[batch_index, keep_mask]
            rollout_weights.append(
                _estimate_single_derivative_weight_reference(
                    loo_failures=loo_failures,
                    truncation_order=order,
                )
            )
        batch_weights.append(torch.stack(rollout_weights))

    return torch.stack(batch_weights)


def _estimate_single_derivative_weight_reference(
    *,
    loo_failures: Float[Tensor, "loo_rollout"],
    truncation_order: int,
) -> Float[Tensor, ""]:
    loo_count = loo_failures.shape[0]
    max_degree = truncation_order - 1
    if max_degree > loo_count:
        raise ValueError(
            "truncation_order is too high for the leave-one-out sample count; "
            f"got truncation_order={truncation_order}, loo_count={loo_count}."
        )

    total = loo_failures.new_tensor(1.0)
    for degree in range(1, max_degree + 1):
        subset_sum = loo_failures.new_tensor(0.0)
        for subset in combinations(range(loo_count), degree):
            subset_indices = torch.tensor(
                subset,
                device=loo_failures.device,
                dtype=torch.long,
            )
            subset_sum = subset_sum + loo_failures[subset_indices].prod()
        total = total + subset_sum / float(math.comb(loo_count, degree))

    return total


def _validate_truncation_order(
    *,
    truncation_order: int,
    rollout_count: int,
) -> None:
    if truncation_order < 1:
        raise ValueError(
            f"truncation_order must be at least 1; got {truncation_order}."
        )
    if truncation_order > rollout_count:
        raise ValueError(
            "truncation_order cannot exceed rollout_count for the leave-one-out "
            f"estimator; got truncation_order={truncation_order}, "
            f"rollout_count={rollout_count}."
        )


def _resolve_computation_dtype(
    *,
    input_dtype: torch.dtype,
    computation_dtype: torch.dtype | None,
) -> torch.dtype:
    if computation_dtype is not None:
        if not computation_dtype.is_floating_point:
            raise ValueError(
                f"computation_dtype must be floating point; got {computation_dtype}."
            )
        if computation_dtype not in (torch.float32, torch.float64):
            raise ValueError(
                "computation_dtype must be torch.float32 or torch.float64 for "
                f"stable MaxRL polynomial estimates; got {computation_dtype}."
            )
        return computation_dtype

    return torch.float32


def _replace_binary_full_order_score_weights(
    *,
    scaled_likelihoods: Float[Tensor, "batch rollout"],
    score_weights: Float[Tensor, "batch rollout"],
) -> Float[Tensor, "batch rollout"]:
    binary_entries = (scaled_likelihoods == 0.0) | (scaled_likelihoods == 1.0)
    binary_rows = binary_entries.all(dim=1)
    if not bool(binary_rows.any().item()):
        return score_weights

    rollout_count = scaled_likelihoods.shape[1]
    successes = scaled_likelihoods == 1.0
    success_count = successes.sum(dim=1, keepdim=True)
    exact_binary_weights = torch.where(
        success_count > 0,
        successes.to(dtype=score_weights.dtype)
        * (float(rollout_count) / success_count.clamp_min(1).to(score_weights.dtype)),
        torch.zeros_like(score_weights),
    )
    return torch.where(binary_rows[:, None], exact_binary_weights, score_weights)


def _gauss_legendre_unit_interval(
    *,
    node_count: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[Tensor, Tensor]:
    cache_key = (node_count, device, dtype)
    cached = _GAUSS_LEGENDRE_CACHE.get(cache_key)
    if cached is not None:
        return cached

    roots, weights = _gauss_legendre_minus_one_one(
        node_count=node_count,
        device=device,
        dtype=dtype,
    )

    nodes_tensor = 0.5 * (roots + 1.0)
    weights_tensor = 0.5 * weights
    max_node = 1.0 - torch.finfo(dtype).eps
    nodes_tensor = nodes_tensor.clamp(min=0.0, max=max_node)

    _GAUSS_LEGENDRE_CACHE[cache_key] = (nodes_tensor, weights_tensor)
    return nodes_tensor, weights_tensor


def _gauss_legendre_minus_one_one(
    *,
    node_count: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[Tensor, Tensor]:
    if node_count < 1:
        raise ValueError(f"node_count must be positive; got {node_count}.")

    positive_root_count = (node_count + 1) // 2
    root_numbers = torch.arange(
        positive_root_count,
        device=device,
        dtype=dtype,
    )
    pi = torch.tensor(math.pi, device=device, dtype=dtype)
    roots_positive = torch.cos(
        pi * (root_numbers + 0.75) / (float(node_count) + 0.5)
    )

    derivative = roots_positive.new_empty(roots_positive.shape)
    tolerance = 8.0 * torch.finfo(dtype).eps
    for _ in range(32):
        legendre_value, derivative = _legendre_value_and_derivative(
            degree=node_count,
            x=roots_positive,
        )
        step = legendre_value / derivative
        roots_positive = roots_positive - step
        if bool((step.abs().max() <= tolerance).item()):
            break

    _, derivative = _legendre_value_and_derivative(
        degree=node_count,
        x=roots_positive,
    )
    weights_positive = 2.0 / (
        (1.0 - roots_positive.square()) * derivative.square()
    )

    roots = roots_positive.new_empty(node_count)
    weights = weights_positive.new_empty(node_count)
    roots[:positive_root_count] = -roots_positive
    weights[:positive_root_count] = weights_positive
    roots[node_count - positive_root_count :] = roots_positive.flip(0)
    weights[node_count - positive_root_count :] = weights_positive.flip(0)
    return roots, weights


def _legendre_value_and_derivative(
    *,
    degree: int,
    x: Float[Tensor, "node"],
) -> tuple[Float[Tensor, "node"], Float[Tensor, "node"]]:
    previous = torch.zeros_like(x)
    current = torch.ones_like(x)

    for order in range(1, degree + 1):
        next_value = (
            (float(2 * order - 1) * x * current)
            - (float(order - 1) * previous)
        ) / float(order)
        previous = current
        current = next_value

    derivative = float(degree) * (x * current - previous) / (x.square() - 1.0)
    return current, derivative
