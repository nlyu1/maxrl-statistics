from __future__ import annotations

from contextlib import nullcontext

import torch
import torch.nn.functional as F
from jaxtyping import Float, Int
from pydantic import ConfigDict
from pydantic.dataclasses import dataclass
from torch import Tensor
from tqdm.autonotebook import tqdm

from src.experiments.corpus_regression.config import CorpusRegressionStudyBaseConfig
from src.experiments.corpus_regression.state import CorpusRegressionStudyBaseState


class CorpusRegressionSLConfig(CorpusRegressionStudyBaseConfig):
    def get_state_cls(self) -> type["CorpusRegressionSLState"]:
        return CorpusRegressionSLState


@dataclass(kw_only=True, config=ConfigDict(arbitrary_types_allowed=True))
class CorpusRegressionSLState(CorpusRegressionStudyBaseState):
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
        target = target.to(device=self.device, dtype=torch.bfloat16)

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

        pbar = tqdm(self.train_dl, desc=f"sl epoch {epoch}")
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
