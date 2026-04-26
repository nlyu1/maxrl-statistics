from __future__ import annotations

from contextlib import nullcontext
from typing import Self

import torch
from jaxtyping import Float, Int
from pydantic import ConfigDict
from pydantic.dataclasses import dataclass
from torch import Tensor
from tqdm.autonotebook import tqdm

from src.experiments.corpus_regression.config import CorpusRegressionStudyBaseConfig
from src.experiments.corpus_regression.state import CorpusRegressionStudyBaseState


class CorpusRegressionGRPOConfig(CorpusRegressionStudyBaseConfig):
    """Zero-step rollout, fully on-policy GRPO config.

    The policy is an isotropic Gaussian on the D-dim output,
        m_theta(z | x) = N(f_theta(x), sigma^2 I_D),
    so "rollouts" are Gaussian samples around the model prediction.
    Rewards are joint negative squared error -|z - target|^2 (sum over D);
    advantages are group-standardized across the rollout group.
    """

    num_rollouts_per_sample: int
    gaussian_stdev: float

    def get_state_cls(self) -> type["CorpusRegressionGRPOState"]:
        return CorpusRegressionGRPOState

    @classmethod
    def get_canonical(
        cls,
        *,
        num_rollouts_per_sample: int,
        gaussian_stdev: float,
        **kwargs: object,
    ) -> Self:
        config = cls(
            **cls.canonical_kwargs(**kwargs),
            num_rollouts_per_sample=num_rollouts_per_sample,
            gaussian_stdev=gaussian_stdev,
        )
        config.save_config_json()
        return config


@dataclass(kw_only=True, config=ConfigDict(arbitrary_types_allowed=True))
class CorpusRegressionGRPOState(CorpusRegressionStudyBaseState):
    def compute_last_step_projections(
        self,
        *,
        context: Int[Tensor, "batch seq"],
    ) -> Float[Tensor, "batch D"]:
        return self.model(tokens=context)

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
                noise: Float[Tensor, "batch rollouts D"] = torch.randn(
                    prediction.shape[0],
                    num_rollouts,
                    prediction.shape[1],
                    device=prediction.device,
                    dtype=prediction.dtype,
                )
                rollouts: Float[Tensor, "batch rollouts D"] = (
                    prediction.unsqueeze(1) + sigma * noise
                )

            # log m_theta(z|x) up to theta-independent constants. Joint logp =
            # sum over D of per-coord logp; constants drop out because GRPO
            # advantages sum to zero within each rollout group.
            logp_rollouts: Float[Tensor, "batch rollouts"] = -0.5 * (
                (rollouts - prediction.unsqueeze(1)) / sigma
            ).pow(2).sum(-1)

            # Joint squared-error reward, summed across the D output coords.
            rewards: Float[Tensor, "batch rollouts"] = -(
                rollouts - target.unsqueeze(1)
            ).pow(2).sum(-1)
            advantages: Float[Tensor, "batch rollouts"] = (
                rewards - rewards.mean(-1, keepdim=True)
            ) / (rewards.std(-1, keepdim=True) + 1e-8)

            loss = -(logp_rollouts * advantages.detach()).mean()

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

        pbar = tqdm(self.train_dl, desc=f"grpo epoch {epoch}")
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
