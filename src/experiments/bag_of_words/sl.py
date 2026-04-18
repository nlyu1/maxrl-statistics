from __future__ import annotations

from contextlib import nullcontext

import torch
import torch.nn.functional as F
from jaxtyping import Float, Int
from pydantic import ConfigDict
from pydantic.dataclasses import dataclass
from torch import Tensor
from tqdm.autonotebook import tqdm

from src.experiments.bag_of_words.config import BagOfWordsStudyBaseConfig
from src.experiments.bag_of_words.state import BagOfWordsStudyBaseState


class BagOfWordsSLConfig(BagOfWordsStudyBaseConfig):
    def get_state_cls(self) -> type["BagOfWordsSLState"]:
        return BagOfWordsSLState


@dataclass(kw_only=True, config=ConfigDict(arbitrary_types_allowed=True))
class BagOfWordsSLState(BagOfWordsStudyBaseState):
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
    ) -> float:
        tokens, target, ground_truth = batch
        tokens = tokens.to(device=self.device, dtype=torch.long)
        target = target.to(device=self.device, dtype=torch.bfloat16)
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
            prediction = self.compute_last_step_projections(context=tokens)
            loss = F.mse_loss(prediction, target)
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
        return float(loss.detach().cpu())

    def run_train_epoch(self, *, epoch: int) -> None:
        self.current_epoch = epoch
        self.train_corr_target_counter.empty_()
        self.train_corr_ground_truth_counter.empty_()
        self.model.train()

        pbar = tqdm(self.train_dl, desc=f"sl epoch {epoch}")
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
