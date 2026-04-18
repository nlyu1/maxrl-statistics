from __future__ import annotations

import math

import torch
from jaxtyping import Float
from torch import Tensor

from config.base import BaseConfig


class MaxRLEstimatorConfig(BaseConfig):
    shrinkage_factor: float
    degree: int

    @classmethod
    def initialize(
        cls, *, shrinkage_factor: float, degree: int
    ) -> "MaxRLEstimatorConfig":
        assert 1 <= degree, f"Degree must be nontrivial, got {degree}"
        assert 0.0 < shrinkage_factor <= 2.0, "Understand maclaurin!"
        return cls(shrinkage_factor=shrinkage_factor, degree=degree)


def compute_maxrl_score_weights(
    *,
    log_likelihoods: Float[Tensor, "batch rollout"],
    sup_likelihood: float,
    shrinkage_factor: float,  # Between (0, 2]
    degree: int,
) -> Float[Tensor, "batch rollout"]:
    """
    Mathematical contract
    --------------------
    Suppress batch indexing and fix one prompt / target pair (x, y)

    Let
        R := num_rollouts
        S := sup_likelihood
        D := degree
        c := shrinkage_factor
        z_j ~ m_theta(- | x) i.i.d. for j = 1 ... R
        l_j := l(y, z_j) in (0, S)
            **Detached** rollout-conditional likelihood
        rho_theta := E_z[l(y, z)]
            Rollout-marginal likelihood
        sigma_theta := c * rho_theta / S
            Within (0, 2) under assumptions.
        score_j := grad_theta log m_theta(z_j | x).
            Note that this is the gradient of **policy score**,
                not of the provided log-likelihoods

    Key assumptions: these are not checked
    - c in (0, 2]
    - 0 < sigma_theta < 2
    - 1 <= D <= R
        - To estimate to degree-D, we only need the (D-1)-th power
        - Leaving one out yields R-1 samples, so D-1 <= R-1 ==> D <= R

    Population-objective
    --------------------
    The returned weights target the degree-D Maclaurin truncation
        of log(sigma_theta) about sigma_theta = 1:

    J_D(theta) := -sum_{k=1}^D (1 - sigma_theta)^k / k

    Its sigma-derivative is
        omega_D(sigma) := sum_{d=0}^{D-1} (1-sigma)^d

    The exact log-likelihood gradient is recovered in the limit D -> infty.
        For finite order, the discrepancy is the geometric factor
        1 - (1 - sigma_theta)^D

    Estimator returned by this function
    -----------------------------------
    This function returns detached per-rollout score weights `w_j` such that

        g_hat(theta) := (1 / R) * sum_j (w_j * score_j)

    is an unbiased estimator of grad_theta J_D(theta)
    """
    with torch.no_grad():
        normalized_ll = log_likelihoods + math.log(shrinkage_factor / sup_likelihood)
        # 1 - sigma_theta samples
        complement_normalized_likelihood = -torch.expm1(normalized_ll)
        # For each rollout, estimate omega_D(sigma) using leave-one-out complements
        omega = _leave_one_out_weight(complement_normalized_likelihood, degree=degree)
        return (omega * normalized_ll.exp()).type_as(log_likelihoods)
