from __future__ import annotations

from src.experiments.bag_of_words.config import BagOfWordsStudyBaseConfig


class BagOfWordsMaxRLonfig(BagOfWordsStudyBaseConfig):
    """
    Zero-step rollout, fully on-policy MaxRL config

    The policy is a Gaussian on the scalar output,
        m_theta(z | x) = Normal(f_theta(x), gaussian_stdev^2),
    so "rollouts" are Gaussian samples around the model prediction and
    the score function is (z - f_theta(x)) / sigma^2 * grad f_theta(x).
    Advantages are standard GRPO group-standardized negative MSE rewards
    against the (noisy) target.
    """

    # Estimate order = num_rollouts_per_sample
    num_rollouts_per_sample: int
    gaussian_stdev: float
    # We need an additional term to bound the log-expansion.
    # In the canonical bag-of-words dataset, this is [-1, 1] -> 2.0
    reward_range: float

    def get_state_cls(self) -> type["BagOfWordsGRPOState"]:
        return BagOfWordsGRPOState

    @classmethod
    def get_canonical(
        cls,
        *,
        num_rollouts_per_sample: int,
        gaussian_stdev: float,
        **kwargs: object,
    ) -> "BagOfWordsGRPOConfig":
        config = cls(
            **cls._canonical_kwargs(**kwargs),
            num_rollouts_per_sample=num_rollouts_per_sample,
            gaussian_stdev=gaussian_stdev,
        )
        config.prepare_study_folder()
        return config


@dataclass(kw_only=True, config=ConfigDict(arbitrary_types_allowed=True))
class BagOfWordsMaxRLState(BagOfWordsStudyBaseState):
    last_pred_norm: float = 0.0

    def compute_last_step_projections(
        self,
        *,
        context: Int[Tensor, "batch seq"],
    ) -> Float[Tensor, "batch"]:
        # Fixed-width EOS padding is intentionally part of the causal context here:
        # this projects from the final padded position, not the last non-pad token.
        return self.model(tokens=context).squeeze(-1)

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

            # Rollouts are Gaussian samples around the deterministic prediction.
            # They are treated as fixed samples from m_theta(.|x); no gradient flows
            # through z itself -- the policy-gradient signal must come from logp.
            with torch.no_grad():
                noise: Float[Tensor, "batch rollouts"] = torch.randn(
                    prediction.shape[0],
                    num_rollouts,
                    device=prediction.device,
                    dtype=prediction.dtype,
                )
                rollouts: Float[Tensor, "batch rollouts"] = (
                    prediction.unsqueeze(-1) + sigma * noise
                )

            # log m_theta(z|x) up to theta-independent constants. Constants drop
            # out because GRPO advantages sum to zero within each rollout group.
            logp_rollouts: Float[Tensor, "batch rollouts"] = -0.5 * (
                (rollouts - prediction.unsqueeze(-1)) / sigma
            ).pow(2)

            # Compute score weights
            score_weights: Float[Tensor, "batch rollouts"] = (
                self._compute_score_weights(rollouts=rollouts, target=target)
            )

            loss = -(logp_rollouts * score_weights.detach()).mean()

            with torch.no_grad():
                self.last_pred_norm = float(prediction.detach().float().norm().cpu())
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

        pbar = tqdm(self.train_dl, desc=f"grpo epoch {epoch}")
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
                pred_norm=f"{self.last_pred_norm:.4f}",
            )

    def run_training(self) -> None:
        for epoch in range(self.config.train_epochs):
            self.run_train_epoch(epoch=epoch)
            validation = self.compute_validation()
            self.serialize_at_end_of_epoch(validation=validation)
