from __future__ import annotations

from typing import Self

import torch
from jaxtyping import Float
from torch import Tensor

from src.config.base import BaseConfig


class MaxRLEstimatorConfig(BaseConfig):
    degree: int
    log_sup_likelihood: float
    subtract_baseline: bool

    @classmethod
    def initialize(
        cls, *, degree: int, log_sup_likelihood: float, subtract_baseline: bool
    ) -> Self:
        assert 1 <= degree, f"Degree must be nontrivial, got {degree}"
        return cls(
            degree=degree,
            log_sup_likelihood=log_sup_likelihood,
            subtract_baseline=subtract_baseline,
        )

    def compute_score_weights(
        self,
        *,
        log_likelihoods: Float[Tensor, "batch rollout"],
    ) -> Float[Tensor, "batch rollout"]:
        """
        Mathematical contract
        --------------------
        Suppress batch indexing and fix one prompt / target pair (x, y).
        Notation matches writeup/gradient-estimator.qmd.

        Let
            R := num_rollouts
            L := exp(log_sup_likelihood)              (bound on l)
            D := degree
            z_j ~ m_theta(- | x) i.i.d. for j = 1 ... R
            l_j := l(y, z_j) in [0, L]
                **Detached** rollout-conditional likelihood
            sigma_j := l_j / L in [0, 1]
                Normalized per-rollout likelihood
            sigma_theta := E_z[sigma_j] = E_z[l_j] / L
                Marginal normalized likelihood, in (0, 1] under assumptions
            S_j := grad_theta log m_theta(z_j | x)
                Per-rollout policy score vector (gradient of log-policy,
                not of the provided log-likelihoods)

        Key assumptions: these are not checked
        - 0 < sigma_theta <= 1
        - 1 <= D <= R
            - To estimate to degree-D, we only need the (D-1)-th power
            - Leaving one out yields R-1 samples, so D-1 <= R-1 ==> D <= R

        Population objective
        --------------------
        The returned weights target the degree-D Maclaurin truncation of
        log(sigma_theta) about sigma_theta = 1:

            J_D(theta) := -sum_{k=1}^D (1 - sigma_theta)^k / k

        Its sigma-derivative is the weight function
            w_D(sigma) := sum_{d=0}^{D-1} (1 - sigma)^d,

        so grad_theta J_D = w_D(sigma_theta) * E[sigma_j * S_j]. As D -> infty
        this recovers the exact log-likelihood gradient of log sigma_theta;
        at finite D the gradient residual is the geometric factor
        1 - (1 - sigma_theta)^D.

        Estimator returned by this function
        -----------------------------------
        Returns the detached signed per-rollout coefficient c_j such that

            g_hat(theta) := (1 / R) * sum_j c_j * S_j

        is an unbiased estimator of grad_theta J_D(theta). Each omega_j is a
        leave-one-out U-statistic built from the peer complements
        a_{-j} = (1 - sigma_i)_{i != j}, satisfying E[omega_j] = w_D(sigma_theta).

        If subtract_baseline is False,
            c_j = omega_j * sigma_j                      (always >= 0).
        If subtract_baseline is True (requires R >= 2),
            c_j = omega_j * (sigma_j - sigma_bar_{-j})
                = omega_j * (R / (R - 1)) * (sigma_j - sigma_bar),
        where sigma_bar = mean_i sigma_i and sigma_bar_{-j} is the peer-only
        mean (sum_{i != j} sigma_i) / (R - 1). The baseline b_j := omega_j *
        sigma_bar_{-j} is a function of z_{-j} alone; since omega_j is already
        leave-one-out and sigma_j is fresh, E[b_j | z_{-j}] = omega_j *
        sigma_theta and E[b_j * S_j] = 0, so subtracting it preserves
        unbiasedness and typically reduces variance.

        The leave-one-out DP for log(omega_j) is run entirely in log-space;
        we only exit to linear space at the very end, when forming the
        signed product omega_j * sigma_effective_j.
        """
        num_rollouts = log_likelihoods.shape[1]
        assert self.degree <= num_rollouts
        if self.subtract_baseline:
            assert num_rollouts >= 2, (
                "subtract_baseline=True requires at least 2 rollouts; "
                f"got num_rollouts={num_rollouts}"
            )

        with torch.no_grad():
            # log(sigma_j) = log(l_j) - log(L)
            normalized_ll = log_likelihoods - self.log_sup_likelihood
            sigma_effective = self._maybe_subtract_baseline_from_normalized_ll(
                normalized_ll=normalized_ll
            )

            if num_rollouts == 1:
                # Degree-1 weight function w_1(sigma) = 1, so c_j = sigma_j.
                return sigma_effective.type_as(log_likelihoods)

            # Per-rollout complements a_j = 1 - sigma_j, in [0, 1]
            complement_normalized_likelihood = -torch.expm1(normalized_ll)
            # For each rollout j, estimate w_D(sigma_theta) via the
            # leave-one-out U-statistic omega_j over peer complements a_{-j}.
            log_omega = self._log_leave_one_out_weight(complement_normalized_likelihood)
            # Exit log-space here: omega_j >= 0, sigma_effective_j is signed
            # when subtract_baseline=True.
            return (log_omega.exp() * sigma_effective).type_as(log_likelihoods)

    def _maybe_subtract_baseline_from_normalized_ll(
        self,
        *,
        normalized_ll: Float[Tensor, "batch rollout"],
    ) -> Float[Tensor, "batch rollout"]:
        """
        Convert normalized_ll_j = log(sigma_j) into the sigma-factor that
        multiplies omega_j in the final score coefficient c_j.

        subtract_baseline=False:
            sigma_effective_j = sigma_j = exp(normalized_ll_j),  in [0, 1].
        subtract_baseline=True (R >= 2):
            sigma_effective_j = sigma_j - sigma_bar_{-j}
                              = (R / (R - 1)) * (sigma_j - sigma_bar),
            which is signed. See `compute_score_weights` for unbiasedness.

        The DP that produces log(omega_j) stays in log-space; this helper is
        where (and the only place where) the pipeline leaves log-space.
        """
        compute_dtype = torch.promote_types(normalized_ll.dtype, torch.float32)
        sigma = normalized_ll.to(compute_dtype).exp()
        if not self.subtract_baseline:
            return sigma
        num_rollouts = sigma.shape[-1]
        sigma_bar = sigma.mean(dim=-1, keepdim=True)
        scale = num_rollouts / (num_rollouts - 1)
        return scale * (sigma - sigma_bar)

    def _log_leave_one_out_weight(
        self, complement_nl: Float[Tensor, "batch rollout"]
    ) -> Float[Tensor, "batch rollout"]:
        """
        Given per-rollout complements a_j = 1 - sigma_j (batched as
        `complement_nl`), compute log(omega_j) for each rollout j, where

            omega_j := sum_{d=0}^{D-1} e_d(a_{-j}) / C(R - 1, d)

        is the leave-one-out U-statistic satisfying E[omega_j] = w_D(sigma_theta).
        Here e_d(a_{-j}) is the d-th elementary symmetric polynomial in the
        R-1 peer complements a_{-j} = (a_i)_{i != j}, so
        e_d(a_{-j}) / C(R-1, d) is the U-statistic estimator of (1-sigma_theta)^d.

        Implementation follows the state-vector / prefix-suffix-scan formulation
        in writeup/gradient-estimator.qmd. For any subset I of rollouts, let

            x_k(I) := e_k((a_i)_{i in I}) / C(R - 1, k),   k = 0, ..., D-1

        be the normalized elementary-symmetric state. Appending one complement
        a to I updates the state by the lower-bidiagonal matrix M(a) in R^{DxD}:

            M(a)_{k, k} = 1,   M(a)_{k, k-1} = (k / (R - k)) * a,

        so x(I u {a}) = M(a) * x(I). Forward prefix states alpha_j = x({1..j})
        are built by a forward scan; suffix covectors beta_j (with beta_R = 1^T)
        are built by a backward scan; and the leave-one-out weight is

            omega_j = beta_j^T * alpha_{j-1}.

        The DP is carried out entirely in log-space (logaddexp / logsumexp);
        every per-step linear-space identity is annotated in-line below.
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
        log_beta = torch.zeros(batch_size, degree, device=device, dtype=compute_dtype)
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
