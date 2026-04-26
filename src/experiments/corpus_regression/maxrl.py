from __future__ import annotations

import math
from contextlib import nullcontext
from typing import Self

import torch
from einops import rearrange, repeat
from jaxtyping import Float, Int
from pydantic import ConfigDict
from pydantic.dataclasses import dataclass
from torch import Tensor
from tqdm.autonotebook import tqdm

from src.experiments.corpus_regression.config import CorpusRegressionStudyBaseConfig
from src.experiments.corpus_regression.state import CorpusRegressionStudyBaseState
from src.maxrl_estimator import MaxRLEstimatorConfig


class CorpusRegressionMaxRLConfig(CorpusRegressionStudyBaseConfig):
    """Zero-step rollout, fully on-policy MaxRL config.

    Policy: isotropic Gaussian m_theta(z | x) = N(f_theta(x), sigma^2 I_D).

    use_factorized_likelihoods=True (canonical): per-coordinate scoring.
    Each output dim d is treated as an independent 1-D regression with
    likelihood l_d(y_d, z_d) = N(y_d; z_d, sigma^2). sigma_theta_d is
    bounded above by 1/sqrt(2) regardless of D.

    use_factorized_likelihoods=False: joint scoring with l(y, z) =
    N(y; z, sigma^2 I_D). sigma_theta <= 2^(-D/2), exponentially small in D.
    """

    num_rollouts_per_sample: int
    degree: int
    gaussian_stdev: float
    subtract_baseline: bool
    use_factorized_likelihoods: bool

    def get_state_cls(self) -> type["CorpusRegressionMaxRLState"]:
        return CorpusRegressionMaxRLState

    def _build_estimator_config(self) -> MaxRLEstimatorConfig:
        sigma = self.gaussian_stdev
        d = self.data.embedding_dim
        assert sigma > 0.0
        log_sup_one_dim = -math.log(math.sqrt(2.0 * math.pi) * sigma)
        log_sup_likelihood = (
            log_sup_one_dim if self.use_factorized_likelihoods else d * log_sup_one_dim
        )
        return MaxRLEstimatorConfig.initialize(
            degree=self.degree,
            log_sup_likelihood=log_sup_likelihood,
            subtract_baseline=self.subtract_baseline,
        )

    def _extra_state_kwargs(self) -> dict:
        return {"estimator_config": self._build_estimator_config()}

    @classmethod
    def get_canonical(
        cls,
        *,
        num_rollouts_per_sample: int,
        gaussian_stdev: float,
        subtract_baseline: bool,
        use_factorized_likelihoods: bool,
        **kwargs: object,
    ) -> Self:
        assert num_rollouts_per_sample >= 1
        assert gaussian_stdev > 0.0
        config = cls(
            **cls.canonical_kwargs(**kwargs),
            num_rollouts_per_sample=num_rollouts_per_sample,
            degree=num_rollouts_per_sample,
            gaussian_stdev=gaussian_stdev,
            subtract_baseline=subtract_baseline,
            use_factorized_likelihoods=use_factorized_likelihoods,
        )
        config.save_config_json()
        return config


@dataclass(kw_only=True, config=ConfigDict(arbitrary_types_allowed=True))
class CorpusRegressionMaxRLState(CorpusRegressionStudyBaseState):
    estimator_config: MaxRLEstimatorConfig

    def compute_last_step_projections(
        self,
        *,
        context: Int[Tensor, "batch seq"],
    ) -> Float[Tensor, "batch D"]:
        return self.model(tokens=context)

    def _compute_score_weights(
        self,
        *,
        rollouts: Float[Tensor, "batch rollout D"],
        target: Float[Tensor, "batch D"],
    ) -> Float[Tensor, "batch rollout D"]:
        """Per-dim score coefficients. In joint mode the same scalar is broadcast
        across the D dimension so the caller can multiply per-dim log-policy uniformly.
        """
        sigma = self.config.gaussian_stdev
        log_L = self.estimator_config.log_sup_likelihood
        with torch.no_grad():
            rollouts_f = rollouts.float()
            target_f = target.float()
            sq: Float[Tensor, "batch rollout D"] = (
                (target_f.unsqueeze(1) - rollouts_f) / sigma
            ).pow(2)
            d = sq.shape[-1]
            if self.config.use_factorized_likelihoods:
                log_lik: Float[Tensor, "batch rollout D"] = -0.5 * sq + log_L
                sw_flat = self.estimator_config.compute_score_weights(
                    log_likelihoods=rearrange(log_lik, "b r d -> (b d) r"),
                )
                return rearrange(sw_flat, "(b d) r -> b r d", d=d)
            log_lik_joint: Float[Tensor, "batch rollout"] = -0.5 * sq.sum(-1) + log_L
            sw_joint = self.estimator_config.compute_score_weights(
                log_likelihoods=log_lik_joint,
            )
            return repeat(sw_joint, "b r -> b r d", d=d)

    def train_step(
        self,
        *,
        batch: tuple[
            Int[Tensor, "batch seq"],
            Float[Tensor, "batch D"],
        ],
    ) -> None:
        tokens, target = batch
        tokens = tokens.to(device=self.device, dtype=torch.long)
        target: Float[Tensor, "batch D"] = target.to(
            device=self.device, dtype=torch.bfloat16
        )

        num_rollouts = self.config.num_rollouts_per_sample
        sigma = self.config.gaussian_stdev
        assert 1 <= self.config.degree <= num_rollouts
        assert sigma > 0.0

        device_context = (
            torch.cuda.device(self.device)
            if self.device.type == "cuda"
            else nullcontext()
        )
        with (
            device_context,
            torch.autocast(device_type=self.device.type, dtype=torch.bfloat16),
        ):
            prediction: Float[Tensor, "batch D"] = self.compute_last_step_projections(
                context=tokens
            )

            # Rollouts are isotropic Gaussian samples around the deterministic
            # prediction. They are treated as fixed samples from m_theta(.|x);
            # no gradient flows through z itself -- the policy-gradient signal
            # comes from logp.
            with torch.no_grad():
                noise: Float[Tensor, "batch rollout D"] = torch.randn(
                    prediction.shape[0],
                    num_rollouts,
                    prediction.shape[1],
                    device=prediction.device,
                    dtype=prediction.dtype,
                )
                rollouts: Float[Tensor, "batch rollout D"] = (
                    prediction.unsqueeze(1) + sigma * noise
                )

            logp_per_dim: Float[Tensor, "batch rollout D"] = -0.5 * (
                (rollouts - prediction.unsqueeze(1)) / sigma
            ).pow(2)
            score_weights: Float[Tensor, "batch rollout D"] = (
                self._compute_score_weights(rollouts=rollouts, target=target)
            )
            loss = -(logp_per_dim * score_weights.detach()).sum(-1).mean()

            with torch.no_grad():
                self.train_target_counter.tick(
                    x=prediction.detach().float().cpu(),
                    y=target.detach().float().cpu(),
                )

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.step_and_zero_grad()

    def run_train_epoch(self, *, epoch: int) -> None:
        self.current_epoch = epoch
        self.train_target_counter.empty_()
        self.model.train()

        pbar = tqdm(self.train_dl, desc=f"maxrl epoch {epoch}")
        for batch in pbar:
            self.train_step(batch=batch)
            stats = self.train_target_counter.get_stats()
            pbar.set_postfix(
                train_corr=f"{float(stats.corr.mean()):.4f}",
                train_mse=f"{float(stats.mse.mean()):.4f}",
            )

    def run_training(self) -> None:
        for epoch in range(self.config.train_epochs):
            self.run_train_epoch(epoch=epoch)
            validation = self.compute_validation()
            self.serialize_at_end_of_epoch(validation=validation)
