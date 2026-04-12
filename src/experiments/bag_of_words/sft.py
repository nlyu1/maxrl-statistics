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


class BagOfWordsSFTConfig(BagOfWordsStudyBaseConfig):
    def get_state_cls(self) -> type["BagOfWordsSFTState"]:
        return BagOfWordsSFTState


@dataclass(kw_only=True)
class SFTTrainStepOutput:
    loss: float


@dataclass(kw_only=True)
class SFTTrainEpochOutput:
    loss: float
    pred_corr: float


@dataclass(kw_only=True, config=ConfigDict(arbitrary_types_allowed=True))
class BagOfWordsSFTState(BagOfWordsStudyBaseState):
    def compute_last_step_projections(
        self,
        *,
        context: Int[Tensor, "batch seq"],
    ) -> Float[Tensor, "batch"]:
        return self.model(tokens=context).squeeze(-1)

    def train_step(
        self,
        *,
        batch: tuple[
            Int[Tensor, "batch seq"],
            Float[Tensor, "batch"],
            Float[Tensor, "batch"],
        ],
    ) -> SFTTrainStepOutput:
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
                self.train_corr_counter.tick(
                    x=prediction.detach().float().cpu()[:, None],
                    y=ground_truth.detach().float().cpu()[:, None],
                )

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.step_and_zero_grad()
        return SFTTrainStepOutput(loss=float(loss.detach().cpu()))

    def run_train_epoch(self, *, epoch: int) -> SFTTrainEpochOutput:
        self.current_epoch = epoch
        self.train_corr_counter.empty_()
        self.model.train()

        losses: list[float] = []
        pbar = tqdm(self.train_dl, desc=f"sft epoch {epoch}")
        for batch in pbar:
            output = self.train_step(batch=batch)
            losses.append(output.loss)
            pred_corr = float(self.train_corr_counter.get_stats().squeeze(0))
            pbar.set_postfix(loss=f"{output.loss:.4f}", pred_corr=f"{pred_corr:.4f}")

        return SFTTrainEpochOutput(
            loss=sum(losses) / len(losses),
            pred_corr=float(self.train_corr_counter.get_stats().squeeze(0)),
        )

    def run_training(self) -> None:
        for epoch in range(self.config.train_epochs):
            train_output = self.run_train_epoch(epoch=epoch)
            validation = self.compute_validation()
            self.serialize_at_end_of_epoch(validation=validation)
            tqdm.write(
                f"epoch {epoch:2d}  "
                f"loss={train_output.loss:.4f}  "
                f"pred_corr={train_output.pred_corr:.4f}  "
                f"val_corr={validation.compute_corr().item():.4f}"
            )
