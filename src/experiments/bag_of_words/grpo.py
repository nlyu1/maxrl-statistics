from src.experiments.bag_of_words.config import BagOfWordsStudyBaseConfig


class BagOfWordsGRPOConfig(BagOfWordsStudyBaseConfig):
    """
    Zero-step rollout, fully on-policy GRPO config.

    Fields: number of samples, and the Guassian implied noise model
    """

    num_rollouts_per_sample: int
    gaussian_stdev: float

    def get_state_cls(self) -> type["BagOfWordsGRPOConfig"]:
        return BagOfWordsGRPOConfig


@dataclass(kw_only=True, config=ConfigDict(arbitrary_types_allowed=True))
class BagOfWordsGRPOState(BagOfWordsStudyBaseState):
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
            # Sample rollouts samples per prediction as Gaussian around center
            with torch.no_grad():
                rollouts: Float[Tensor, "batch rollouts"] = 0
            # Compute Gaussian likelihood
            logp_rollouts: Float[Tensor, "batch rollouts"] = ...
            # Rewards are just negative-mse
            rewards: Float[Tensor, "batch rollouts"] = -(
                rollouts - target.unsqueeze(-1)
            ).pow(2)
            advantages = (rewards - rewards.mean(-1, keepdim=True)) / rewards.std(
                -1, keepdim=True
            )
            # Policy loss
            policy_loss = einsum(logp_rollouts, advantages.detach(), "b r, b r -> ...")

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
