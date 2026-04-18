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

        The DP is carried out entirely in log-space (logaddexp / logsumexp);
            every per-step linear-space identity is annotated in-line below.

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

        # Log-space throughout. All DP states are sums of positives (no
        # subtractions), so: linear + becomes logaddexp, linear * becomes
        # add. Both are free of cancellation; log-space also lifts fp64's
        # ~1e308 overflow ceiling that bit linear-space beta at D ~ R with
        # R >= 1024. Promote bf16/fp16 to fp32 for the DP.
        compute_dtype = torch.promote_types(complement_nl.dtype, torch.float32)
        # log(1 - sigma_j). a_j = 0 (rollout at the ceiling) gives -inf,
        # which propagates correctly as "no contribution to e_k for k >= 1".
        log_a = complement_nl.to(compute_dtype).log()

        # Subdiagonal scale of M(a): log ratio[k-1] = log(k / (R - k))
        # for k = 1..D-1.
        k_vec = torch.arange(1, degree, device=device, dtype=compute_dtype)
        log_ratio = k_vec.log() - (num_rollouts - k_vec).log()  # [D-1]

        # Forward scan over rollouts.
        #   Linear:   alpha[j, k] = e_k(a_0..a_{j-1}) / C(R-1, k),  [B, R, D]
        #             alpha[j, 0] = 1;  alpha[0, k>=1] = 0
        #             alpha[j, k] = alpha[j-1, k]
        #                         + (k/(R-k)) * a_{j-1} * alpha[j-1, k-1]
        # Log form: store log_alpha; the + above becomes logaddexp, the
        # triple product becomes the sum log_ratio + log_a_j + log_prev.
        log_alpha = torch.full(
            (batch_size, num_rollouts, degree),
            float("-inf"),
            device=device,
            dtype=compute_dtype,
        )
        log_alpha[:, :, 0] = 0.0  # alpha[j, 0] = 1 for every j
        for j in range(1, num_rollouts):
            prev = log_alpha[:, j - 1]
            log_a_j = log_a[:, j - 1].unsqueeze(-1)
            log_alpha[:, j, 1:] = torch.logaddexp(
                prev[:, 1:],
                log_ratio + log_a_j + prev[:, :-1],
            )

        # Backward scan: rolling log_beta [B, D].
        #   Linear:   beta[k] represents sum_l e_l(a_j..a_{R-1})
        #                                  * C(R-1, k) / C(R-1, k+l)
        #             beta_R = 1 (log 0). beta[D-1] stays at 1 throughout.
        #             beta_{j-1}[k] = beta_j[k]
        #                           + (k+1)/(R-k-1) * a_{j-1} * beta_j[k+1]
        # Pair log_beta_j with log_alpha_{j-1} to emit log_omega[j-1],
        # then step with logaddexp.
        log_beta = torch.zeros(
            batch_size, degree, device=device, dtype=compute_dtype
        )
        log_omega = torch.empty(
            batch_size, num_rollouts, device=device, dtype=compute_dtype
        )
        for j in range(num_rollouts, 0, -1):
            # Linear: omega[j-1] = sum_k beta[k] * alpha[j-1, k].
            log_omega[:, j - 1] = torch.logsumexp(
                log_beta + log_alpha[:, j - 1], dim=-1
            )
            log_a_j = log_a[:, j - 1].unsqueeze(-1)
            log_beta[:, :-1] = torch.logaddexp(
                log_beta[:, :-1],
                log_ratio + log_a_j + log_beta[:, 1:],
            )

        return log_omega.type_as(complement_nl)
