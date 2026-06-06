from __future__ import annotations

from contextlib import nullcontext
from typing import Self

import torch
import torch.nn.functional as F
from einops import repeat
from jaxtyping import Float, Int
from pydantic import ConfigDict
from pydantic.dataclasses import dataclass
from torch import Tensor

from src.experiments.corpus_regression.config import CorpusRegressionStudyBaseConfig
from src.experiments.corpus_regression.state import (
    CorpusRegressionStudyBaseState,
    TrainStepOutput,
    _batch_sufficient_stats,
)


class CorpusRegressionRLOOConfig(CorpusRegressionStudyBaseConfig):
    """Zero-step rollout, fully on-policy RLOO config.

    Policy: isotropic Gaussian m_theta(z | x) = N(f_theta(x), sigma^2 I_D).
    Rewards are negative squared error against the target. The leave-one-out
    baseline subtracts the per-rollout peer mean (no std normalization, in
    contrast to GRPO). With R rollouts:

        A_j = r_j - mean_{i != j} r_i = (R / (R - 1)) * (r_j - r_bar).

    factorized=True: per-coordinate reward r_{j,d} = -(y_d - z_{j,d})^2 with
        a per-dim leave-one-out baseline; loss pairs per-dim logp_d with A_{j,d}.
    factorized=False: joint reward r_j = -|y - z_j|^2 (sum over D) with a single
        baseline; the same scalar A_j is broadcast across the D dims.
    """

    num_rollouts_per_sample: int
    gaussian_stdev: float
    factorized: bool

    def get_state_cls(self) -> type["CorpusRegressionRLOOState"]:
        return CorpusRegressionRLOOState

    @classmethod
    def get_canonical(
        cls,
        *,
        num_rollouts_per_sample: int,
        gaussian_stdev: float,
        factorized: bool,
        **kwargs: object,
    ) -> Self:
        assert num_rollouts_per_sample >= 2, (
            "RLOO requires at least 2 rollouts for the leave-one-out baseline; "
            f"got num_rollouts_per_sample={num_rollouts_per_sample}"
        )
        assert gaussian_stdev > 0.0
        config = cls(
            **cls.canonical_kwargs(**kwargs),
            num_rollouts_per_sample=num_rollouts_per_sample,
            gaussian_stdev=gaussian_stdev,
            factorized=factorized,
        )
        config.save_config_json()
        return config


@dataclass(kw_only=True, config=ConfigDict(arbitrary_types_allowed=True))
class CorpusRegressionRLOOState(CorpusRegressionStudyBaseState):
    def compute_last_step_projections(
        self,
        *,
        context: Int[Tensor, "batch seq"],
    ) -> Float[Tensor, "batch D"]:
        return self.model(tokens=context)

    def _compute_advantages(
        self,
        *,
        rollouts: Float[Tensor, "batch rollout D"],
        target: Float[Tensor, "batch D"],
    ) -> Float[Tensor, "batch rollout D"]:
        """Per-dim leave-one-out advantages. In joint mode the same scalar is
        broadcast across the D dimension so the caller can multiply per-dim
        log-policy uniformly."""
        num_rollouts = rollouts.shape[1]
        d = rollouts.shape[-1]
        scale = num_rollouts / (num_rollouts - 1)
        with torch.no_grad():
            squared_err: Float[Tensor, "batch rollout D"] = (
                rollouts.float() - target.unsqueeze(1).float()
            ).pow(2)
            if self.config.factorized:
                rewards_per_dim: Float[Tensor, "batch rollout D"] = -squared_err
                r_bar = rewards_per_dim.mean(dim=1, keepdim=True)
                return scale * (rewards_per_dim - r_bar)
            rewards: Float[Tensor, "batch rollout"] = -squared_err.sum(-1)
            r_bar = rewards.mean(dim=-1, keepdim=True)
            advantages_scalar: Float[Tensor, "batch rollout"] = scale * (
                rewards - r_bar
            )
            return repeat(advantages_scalar, "b r -> b r d", d=d)

    def train_step(
        self,
        *,
        batch: tuple[
            Int[Tensor, "batch seq"],
            Float[Tensor, "batch D"],
            Int[Tensor, "batch"],
        ],
    ) -> TrainStepOutput:
        tokens, target, _lookahead_token_ids = batch
        tokens = tokens.to(device=self.device, dtype=torch.long)
        target: Float[Tensor, "batch D"] = target.to(
            device=self.device, dtype=torch.bfloat16
        )

        num_rollouts = self.config.num_rollouts_per_sample
        sigma = self.config.gaussian_stdev
        assert num_rollouts >= 2
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
            advantages: Float[Tensor, "batch rollout D"] = self._compute_advantages(
                rollouts=rollouts, target=target,
            )
            loss = -(logp_per_dim * advantages.detach()).sum(-1).mean()

            with torch.no_grad():
                mse = F.mse_loss(prediction.float(), target.float()).item()
                xx, xy, yy, pred_sum, target_sum, n = _batch_sufficient_stats(
                    prediction=prediction, target=target,
                )

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.step_and_zero_grad()
        return TrainStepOutput(
            loss=loss.item(), mse=mse, xx=xx, xy=xy, yy=yy,
            pred_sum=pred_sum, target_sum=target_sum, n=n,
        )
