from __future__ import annotations

import math
from contextlib import nullcontext
from typing import Self

import torch
from jaxtyping import Float, Int
from pydantic import ConfigDict
from pydantic.dataclasses import dataclass
from torch import Tensor
from tqdm.autonotebook import tqdm

from src.experiments.bag_of_words.config import BagOfWordsStudyBaseConfig, DatasetKind
from src.experiments.bag_of_words.state import BagOfWordsStudyBaseState
from src.maxrl_estimator import MaxRLEstimatorConfig


class BagOfWordsMaxRLConfig(BagOfWordsStudyBaseConfig):
    """
    Zero-step rollout, fully on-policy MaxRL config.

    The policy is a Gaussian on the scalar output,
        m_theta(z | x) = Normal(f_theta(x), gaussian_stdev^2).
    MaxRL weights use the Gaussian likelihood of the noisy target under each
    rollout as l(y, z), with maximum likelihood backed out from gaussian_stdev.
    """

    num_rollouts_per_sample: int
    degree: int
    # Please note that this Gaussian stdev denotes the mode's implied
    # distribution's Gaussian stdev, **not the data's Gaussian stdev**.
    gaussian_stdev: float
    subtract_baseline: bool

    def get_state_cls(self) -> type["BagOfWordsMaxRLState"]:
        return BagOfWordsMaxRLState

    def _build_estimator_config(self) -> MaxRLEstimatorConfig:
        sigma = self.gaussian_stdev
        assert sigma > 0.0
        log_sup_likelihood = -math.log(math.sqrt(2.0 * math.pi) * sigma)
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
        dataset: DatasetKind,
        num_rollouts_per_sample: int,
        gaussian_stdev: float,
        subtract_baseline: bool,
        **kwargs: object,
    ) -> Self:
        assert num_rollouts_per_sample >= 1
        assert gaussian_stdev > 0.0
        config = cls(
            **cls.canonical_kwargs(dataset=dataset, **kwargs),
            num_rollouts_per_sample=num_rollouts_per_sample,
            degree=num_rollouts_per_sample,
            gaussian_stdev=gaussian_stdev,
            subtract_baseline=subtract_baseline,
        )
        config.save_config_json()
        return config


@dataclass(kw_only=True, config=ConfigDict(arbitrary_types_allowed=True))
class BagOfWordsMaxRLState(BagOfWordsStudyBaseState):
    estimator_config: MaxRLEstimatorConfig

    def compute_last_step_projections(
        self,
        *,
        context: Int[Tensor, "batch seq"],
    ) -> Float[Tensor, "batch"]:
        # Fixed-width EOS padding is intentionally part of the causal context here:
        # this projects from the final padded position, not the last non-pad token.
        return self.model(tokens=context).squeeze(-1)

    def _compute_score_weights(
        self,
        *,
        rollouts: Float[Tensor, "batch rollout"],
        target: Float[Tensor, "batch"],
    ) -> Float[Tensor, "batch rollout"]:
        sigma = self.config.gaussian_stdev
        with torch.no_grad():
            rollouts_f = rollouts.float()
            target_f = target.float()
            log_target_likelihoods: Float[Tensor, "batch rollout"] = -0.5 * (
                (target_f.unsqueeze(-1) - rollouts_f) / sigma
            ).pow(2) + self.estimator_config.log_sup_likelihood
        return self.estimator_config.compute_score_weights(
            log_likelihoods=log_target_likelihoods,
        )

    def train_step(
        self,
        *,
        batch: tuple[
            Int[Tensor, "batch seq"],
            Float[Tensor, "batch"],
            Float[Tensor, "batch"],
        ],
    ) -> None:
        tokens, target, ground_truth = batch
        tokens = tokens.to(device=self.device, dtype=torch.long)
        target: Float[Tensor, "batch"] = target.to(
            device=self.device, dtype=torch.bfloat16
        )
        ground_truth = ground_truth.to(device=self.device, dtype=torch.bfloat16)

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
            prediction: Float[Tensor, "batch"] = self.compute_last_step_projections(
                context=tokens
            )

            # Rollouts are fixed samples from m_theta(.|x); no gradient flows
            # through z itself. The policy-gradient signal comes from logp.
            with torch.no_grad():
                noise: Float[Tensor, "batch rollout"] = torch.randn(
                    prediction.shape[0],
                    num_rollouts,
                    device=prediction.device,
                    dtype=prediction.dtype,
                )
                rollouts: Float[Tensor, "batch rollout"] = (
                    prediction.unsqueeze(-1) + sigma * noise
                )

            logp_rollouts: Float[Tensor, "batch rollout"] = -0.5 * (
                (rollouts - prediction.unsqueeze(-1)) / sigma
            ).pow(2)
            score_weights = self._compute_score_weights(
                rollouts=rollouts,
                target=target,
            )

            loss = -(logp_rollouts * score_weights.detach()).mean()

            with torch.no_grad():
                pred_cpu = prediction.detach().float().cpu()[:, None]
                self.train_corr_target_counter.tick(
                    x=pred_cpu,
                    y=target.detach().float().cpu()[:, None],
                )
                self.train_corr_ground_truth_counter.tick(
                    x=pred_cpu,
                    y=ground_truth.detach().float().cpu()[:, None],
                )

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.step_and_zero_grad()

    def run_train_epoch(self, *, epoch: int) -> None:
        self.current_epoch = epoch
        self.train_corr_target_counter.empty_()
        self.train_corr_ground_truth_counter.empty_()
        self.model.train()

        pbar = tqdm(self.train_dl, desc=f"maxrl epoch {epoch}")
        for batch in pbar:
            self.train_step(batch=batch)
            train_corr_target = float(
                self.train_corr_target_counter.get_stats().corr.squeeze(0)
            )
            train_corr_ground_truth = float(
                self.train_corr_ground_truth_counter.get_stats().corr.squeeze(0)
            )
            pbar.set_postfix(
                train_corr_target=f"{train_corr_target:.4f}",
                train_corr_ground_truth=f"{train_corr_ground_truth:.4f}",
            )

    def run_training(self) -> None:
        for epoch in range(self.config.train_epochs):
            self.run_train_epoch(epoch=epoch)
            validation = self.compute_validation()
            self.serialize_at_end_of_epoch(validation=validation)
