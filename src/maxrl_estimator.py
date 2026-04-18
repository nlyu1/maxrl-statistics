from __future__ import annotations

import math

import torch
from jaxtyping import Float
from torch import Tensor

from src.config.base import BaseConfig


class MaxRLEstimatorConfig(BaseConfig):
    degree: int
    sup_likelihood: float

    @classmethod
    def initialize(
        cls, *, degree: int, sup_likelihood: float
    ) -> "MaxRLEstimatorConfig":
        assert 1 <= degree, f"Degree must be nontrivial, got {degree}"
        return cls(
            degree=degree,
            sup_likelihood=sup_likelihood,
        )

    def compute_log_score_weights(
        self,
        *,
        log_likelihoods: Float[Tensor, "batch rollout"],
    ) -> Float[Tensor, "batch rollout"]:
        """
        Mathematical contract
        --------------------
        Suppress batch indexing and fix one prompt / target pair (x, y)

        Let
            R := num_rollouts
            S := sup_likelihood
            D := degree
            z_j ~ m_theta(- | x) i.i.d. for j = 1 ... R
            l_j := l(y, z_j) in (0, S)
                **Detached** rollout-conditional likelihood
            rho_theta := E_z[l(y, z)]
                Rollout-marginal likelihood
            sigma_theta := rho_theta / S
                Within (0, 1] under assumptions.
            score_j := grad_theta log m_theta(z_j | x).
                Note that this is the gradient of **policy score**,
                    not of the provided log-likelihoods

        Key assumptions: these are not checked
        - 0 < sigma_theta <= 1
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
        This function returns log of detached per-rollout score weights `w_j` such that

            g_hat(theta) := (1 / R) * sum_j (w_j * score_j)

        is an unbiased estimator of grad_theta J_D(theta)
        """
        num_rollouts = log_likelihoods.shape[1]
        assert self.degree <= num_rollouts

        with torch.no_grad():
            normalized_ll = log_likelihoods - math.log(self.sup_likelihood)
            # 1 - sigma_theta samples
            complement_normalized_likelihood = -torch.expm1(normalized_ll)

            if num_rollouts == 1:
                # zeroth-degree expansion omega_1(sigma) is just 1
                return normalized_ll
            else:
                # For each rollout, estimate omega_D(sigma) using leave-one-out complements
                log_omega = self._log_leave_one_out_weight(
                    complement_normalized_likelihood
                )
                return (log_omega + normalized_ll).type_as(log_likelihoods)

    def _log_leave_one_out_weight(
        self, complement_nl: Float[Tensor, "batch rollout"]
    ) -> Float[Tensor, "batch rollout"]:
        """
        Given samples of (1 - sigma), estimates log omega_D(sigma)
        for each sample using leave-one-out's

        log_llo_weight_j = log - sum_{d=0}^{D-1} ()

        All estimates happen arithmetically in non-log space.
            We just store in log-space for stability.

        -------------
        Mathematically, we compute u-statistics matrix
        u_j^k = e_k(sigma_{-j}) / Binom(R - 1, k)

        Here, e_k(sigma_{-j}) is the symmetric polynomial
            w.r.t. R-1 rollouts ignoring j.

        e_k(sigma_{-j}) = sum_{|J|=k} prod_{j in J} a_j

        Intuitively, this just estimates the product by averaging across all subsets.

        See `README.md` for implementation semantics.
        """
        batch_size, num_rollouts = complement_nl.shape
        degree = self.degree
        device = complement_nl.device

        # States stay in [0, 1] with no cancellation, so fp32 is sufficient.
        # Promote bf16/fp16 inputs (bf16's ~3 digits lose accuracy past R ~ 64);
        # keep fp32/fp64 inputs as-is.
        compute_dtype = torch.promote_types(complement_nl.dtype, torch.float32)
        a = complement_nl.to(compute_dtype)

        # Subdiagonal scale of M(a): ratio[k-1] = k / (R - k) for k = 1..D-1.
        k_vec = torch.arange(1, degree, device=device, dtype=compute_dtype)
        ratio = k_vec / (num_rollouts - k_vec)  # [D-1]

        # Forward scan: alpha[:, j] = alpha_j for j = 0..R-1 (alpha_R unused).
        # Invariant alpha_{j, 0} = 1 pre-filled; only update coordinates 1..D-1.
        alpha = torch.zeros(
            batch_size, num_rollouts, degree, device=device, dtype=compute_dtype
        )
        alpha[:, :, 0] = 1.0
        for j in range(1, num_rollouts):
            prev = alpha[:, j - 1]
            a_j = a[:, j - 1].unsqueeze(-1)
            alpha[:, j, 1:] = prev[:, 1:] + ratio * a_j * prev[:, :-1]

        # Backward scan: beta as rolling [B, D], starting at beta_R = 1.
        # Invariant beta_{j, D-1} = 1; only update coordinates 0..D-2.
        # Pair beta_j with alpha_{j-1} before updating to beta_{j-1}.
        beta = torch.ones(batch_size, degree, device=device, dtype=compute_dtype)
        omega = torch.empty(
            batch_size, num_rollouts, device=device, dtype=compute_dtype
        )
        for j in range(num_rollouts, 0, -1):
            omega[:, j - 1] = (beta * alpha[:, j - 1]).sum(dim=-1)
            a_j = a[:, j - 1].unsqueeze(-1)
            beta[:, :-1] = beta[:, :-1] + ratio * a_j * beta[:, 1:]

        return omega.log().type_as(complement_nl)
