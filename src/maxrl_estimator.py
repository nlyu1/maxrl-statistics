from __future__ import annotations

import math

import torch
from jaxtyping import Float
from torch import Tensor

"""
Generalized MaxRL score-weight estimators.

Theory (see theory/maxrl-background.md)
---------------------------------------
Fix one prompt/target (x, y). Let z_j ~ m_theta(.|x) be N i.i.d. rollouts with
per-rollout likelihoods l_j := l(y, z_j) in (0, beta], score vectors
S_j := grad_theta log m_theta(z_j | x), and marginal likelihood
p := E_z[l(y, z)].

The compute-indexed truncated Maximum Likelihood objective of order T is the
degree-T Maclaurin expansion of log(p) about p = 1:

    J_T(p) := - sum_{k=1}^{T} (1 - p)^k / k,        p in (0, 1].

It satisfies
    grad J_T(p) = w_T(p) * grad p,
    w_T(p)     = sum_{k=0}^{T-1} (1 - p)^k.

Using grad p = E_z[l * S] gives the leave-one-out estimator

    g_hat_T := (1/N) * sum_j  omega_j * l_j * S_j,

where omega_j is any estimator of w_T(p) built only from {l_k : k != j}; by
independence of omega_j and (l_j, S_j), E[g_hat_T] = w_T(p) * grad p = grad J_T.

This module returns the per-rollout *score weights*

    weight_j := omega_j * l_tilde_j,        l_tilde_j := l_j / beta,

where beta is any theta-independent ceiling with l_j <= beta. The 1/beta
factor is a theta-independent constant, so it only rescales the gradient:
  - it is required so l_tilde in (0, 1] lies inside the Maclaurin radius of
    convergence for log(.) about 1,
  - it does not change the direction of grad J_T.

A training loop uses the weights as
    loss = -(log m_theta(z|x) * weight).mean()
with `weight.detach()`; gradient of this loss matches grad J_T in expectation.

Unbiased estimation of (1 - p)^k
--------------------------------
Let a_i := 1 - l_tilde_i so (1 - p)^k = E[prod_{i in S} a_i] for any fixed
subset S of size k (by i.i.d.). The minimum-variance unbiased estimator
(MVUE) with M samples is the U-statistic

    U_k(a_1, ..., a_M) = e_k(a_1, ..., a_M) / C(M, k),

where e_k is the k-th elementary symmetric polynomial. In our problem, M =
N - 1 (leave-one-out) and k ranges over 0, ..., T - 1.

We compute the MVUE efficiently by:
  1. a forward DP on all N samples producing p_k^{(N)} := e_k(a)/C(N, k) for
     k = 0, ..., N, using a convex-combination recurrence that stays in [0,1]
     throughout (so it is well-behaved in low precision);
  2. a leave-one-out recurrence over k that peels off sample j using
         E_k(a) = e_k(a_{-j}) + a_j * e_{k-1}(a_{-j}),
     normalized to
         p_k^{loo, j} = N/(N-k) * p_k^{(N)} - k/(N-k) * a_j * p_{k-1}^{loo, j}.

Both stages are O(B * N * T) time and O(B * N) memory and vectorize over
batch and leave-one-out index.
"""


def _leave_one_out_weight(
    *,
    a: Float[Tensor, "batch rollout"],
) -> Float[Tensor, "batch rollout"]:
    """
    Accumulate

        omega_j := sum_{k=0}^{N-2} p_k^{loo, j},

    where p_k^{loo, j} := e_k(a_{-j}) / C(N-1, k) is the leave-one-out MVUE of
    (1 - p)^k built from the N - 1 samples excluding j. This is the truncation
    degree T = N - 1 variant of w_T(p) = sum_{k=0}^{T-1} (1 - p)^k.

    Stage 1 -- all-samples MVUE p_k^{(N)} := e_k(a) / C(N, k) via a normalized
    forward DP (derived by substituting e_k^{(n)} = e_k^{(n-1)} + a_n
    e_{k-1}^{(n-1)} and dividing by C(n, k)):

        p_k^{(n)} = (n - k)/n * p_k^{(n-1)} + k/n * a_n * p_{k-1}^{(n-1)},
        p_0^{(n)} = 1,     p_k^{(0)} = 1[k = 0].

    Coefficients sum to 1 and a_i in [0, 1], so every iterate stays in [0, 1]
    -- no overflow, no catastrophic cancellation.

    Stage 2 -- peel off sample j (derived by normalizing
    e_k(a) = e_k(a_{-j}) + a_j * e_{k-1}(a_{-j})):

        p_k^{loo, j} = [N / (N - k)] * p_k^{(N)}
                      - [k / (N - k)] * a_j * p_{k-1}^{loo, j}.

    Base case p_0^{loo, j} = 1. The loop stops at k = N - 2, so N - k >= 2 and
    the recurrence is stable. Not a convex combination, but each term lies in
    [0, 1] by construction, so it stays well conditioned in float32.
    """
    batch, num_rollouts = a.shape
    n = num_rollouts
    device, dtype = a.device, a.dtype

    # Stage 1: forward DP -> p_full[:, k] = e_k(a) / C(N, k).
    p_full = torch.zeros(batch, n + 1, device=device, dtype=dtype)
    p_full[:, 0] = 1.0
    k_range = torch.arange(n + 1, device=device, dtype=dtype)
    for step in range(1, n + 1):
        alpha = (step - k_range) / step  # shape [N+1]
        beta_ = k_range / step  # shape [N+1]
        shifted = torch.nn.functional.pad(p_full[:, :-1], (1, 0))  # p_{k-1}^{(step-1)}
        a_step = a[:, step - 1].unsqueeze(-1)  # [B, 1]
        p_full = alpha * p_full + beta_ * a_step * shifted

    # Stage 2: leave-one-out accumulation. k = 0 contribution is 1 for every j.
    omega = torch.ones_like(a)
    q_prev = torch.ones_like(a)
    for k in range(1, n - 1):
        # p_full[:, k] has shape [B]; broadcast across the leave-one-out axis.
        q_curr = (n / (n - k)) * p_full[:, k : k + 1] - (k / (n - k)) * a * q_prev
        omega = omega + q_curr
        q_prev = q_curr
    return omega


def _scaled_likelihoods_and_complements(
    *,
    log_likelihoods: Float[Tensor, "batch rollout"],
    likelihood_ceiling: float,
) -> tuple[Float[Tensor, "batch rollout"], Float[Tensor, "batch rollout"]]:
    """
    Return (l_tilde, a) = (l / beta, 1 - l / beta).

    Uses -expm1(log l_tilde) so a stays precise when l_tilde is very close
    to 1 (i.e. when a rollout's likelihood is near the ceiling).
    """
    log_beta = math.log(likelihood_ceiling)
    log_tilde = log_likelihoods - log_beta
    tilde_l = log_tilde.exp()
    a = -torch.expm1(log_tilde)
    return tilde_l, a


def compute_maxrl_score_weights(
    *,
    log_likelihoods: Float[Tensor, "batch rollout"],
    likelihood_ceiling: float,
) -> Float[Tensor, "batch rollout"]:
    """
    Analytic (MVUE) generalized MaxRL score weights.

    Returns `weight_j = omega_j * l_tilde_j` such that

        (1/N) * sum_j  weight_j * S_j

    is an unbiased estimator of grad_theta J_T(p_theta), the gradient of the
    degree-T Maclaurin expansion of log p_theta about p = 1, where
    p_theta = E_z[l(y, z)].

    Spec (used to drive tests):
      For any fixed (x, y), any theta-independent l_j in (0, beta] i.i.d.
      across j, and any T in [1, N],

          E[ omega_j(l_tilde_{-j}) ] = w_T(p) = sum_{k=0}^{T-1} (1 - p)^k,

      with p = E[l_tilde] = E[l] / beta. The returned weight satisfies
      `weight.shape == log_likelihoods.shape` and is detached.

    Args:
        log_likelihoods: [B, N] detached log l(y, z_j), per batch sample and
            rollout. Must satisfy log_likelihoods <= log(likelihood_ceiling).
            This invariant is not checked.
        likelihood_ceiling: beta > 0, any theta-independent upper bound on the
            per-rollout likelihood. Used to rescale l -> l/beta into (0, 1] so
            the log Maclaurin expansion about 1 is valid.

    Returns:
        Score weights of shape [B, N] in the same dtype as `log_likelihoods`,
        detached. `p` (hence w_T(p)) may differ across batch samples: every
        reduction along `rollout` is per-batch-row.

    Notes:
        Internally runs in float32 for stability; the output is cast back to
        the input dtype.
    """
    _, num_rollouts = log_likelihoods.shape
    if num_rollouts < 2:
        raise ValueError(
            f"need num_rollouts >= 2 for leave-one-out; got {num_rollouts}"
        )
    with torch.no_grad():
        # l/beta, 1.0 - l/beta
        tilde_l, a = _scaled_likelihoods_and_complements(
            log_likelihoods=log_likelihoods.float(),
            likelihood_ceiling=likelihood_ceiling,
        )
        # Bulk of the calculation
        omega = _leave_one_out_weight(a=a)
        return (omega * tilde_l).type_as(log_likelihoods)
