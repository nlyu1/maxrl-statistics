from __future__ import annotations

from contextlib import nullcontext

import torch
import torch.nn.functional as F
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
            Int[Tensor, "batch"],
        ],
    ) -> TrainStepOutput:
        tokens, target, _lookahead_token_ids = batch
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
