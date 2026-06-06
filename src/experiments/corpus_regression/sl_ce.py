"""
SL with NTP cross-entropy loss for corpus-regression.

Variant of `sl.py` that fine-tunes the pretrained `AutoModelForCausalLM`
end-to-end (no swapped-in regression head) using cross-entropy on the
lookahead token id (K=1 only). Validation reuses the projection from
`ntp_baseline.py`:

    expected_label = softmax(logits[:, -1, :]) @ label_projector

so the projected regression MSE is directly comparable to SL's MSE under the
same Rademacher (or token_id) target. Training loss is CE; the `mse` field in
TrainStepOutput is the projected regression MSE diagnostic, mirroring the
existing trainers' schema.
"""

from __future__ import annotations

import json
import shutil
import warnings
from contextlib import nullcontext
from pathlib import Path
from typing import Literal, Self

import polars as pl
import torch
import torch.nn.functional as F
from jaxtyping import Float, Int
from pydantic import ConfigDict, ValidationError
from pydantic.dataclasses import dataclass
from torch import Tensor, nn
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from tqdm.autonotebook import tqdm
from transformers import AutoConfig, AutoModelForCausalLM

from src.config.base import BaseConfig
from src.data.corpus_regression import (
    CorpusRegressionDataloadingConfig,
    CorpusRegressionDataset,
    CorpusRegressionDatasetConfig,
)
from src.experiments.corpus_regression.state import (
    CorpusRegressionValidationOutput,
    TrainStepOutput,
    _align_val_schema,
    _batch_sufficient_stats,
    _cycle_dataloader,
    _per_dim_stats,
    _per_dim_stats_from_tensors,
    _scalar_corr_from_stats,
    _scalar_var_from_stats,
)
from src.model.optimizer import CausalLMFullParamMuonOptimizerConfig


class CorpusRegressionSLCEConfig(BaseConfig):
    """Config for the NTP cross-entropy SL variant.

    Mirrors `CorpusRegressionStudyBaseConfig` but with:
      - `pretrained_model: str` instead of `model: CausalLMConfig` (no
        swapped-in regression head — use the model's native LM head)
      - `optimizer: CausalLMFullParamMuonOptimizerConfig` (Muon for backbone
        2D matrices, AdamW for embed+lm_head+1D, both groups on a single LR)
    """

    data: CorpusRegressionDatasetConfig
    dataset_folder: Path
    dataloading: CorpusRegressionDataloadingConfig

    pretrained_model: str
    optimizer: CausalLMFullParamMuonOptimizerConfig

    train_steps: int
    val_every_n_steps: int
    study_folder: Path
    compile_model: bool = True
    compile_mode: str = "reduce-overhead"

    # When True, instantiate the architecture of `pretrained_model` with random
    # weights instead of loading the pretrained checkpoint. Mirrors
    # `CausalLMConfig.train_from_scratch`.
    train_from_scratch: bool = False

    @classmethod
    def canonical_kwargs(
        cls,
        *,
        dataset_base_folder: Path,
        study_base_folder: Path,
        num_lookforward_tokens: int,
        embedding_dim: int = 32,
        label_type: Literal["rademacher", "token_id"] = "rademacher",
        normalize_labels: bool = False,
        label_range: tuple[float, float] = (0.0, 1.0),
        prefix_length: int = 128,
        num_samples: int = 100_000,
        model_name: str = "HuggingFaceTB/SmolLM2-135M",
        batch_size: int = 64,
        eval_batch_size_multiple: int = 2,
        weight_decay: float = 0.0,
        lr_per_sample: float = 1e-5,
        backbone_lr_divisor: float = 6.66,
        train_steps: int = 10_000,
        val_every_n_steps: int = 2000,
        clip_grad_norm: float = 1.0,
        compile_model: bool = True,
        train_from_scratch: bool = False,
    ) -> dict:
        data_config = CorpusRegressionDatasetConfig(
            prefix_length=prefix_length,
            num_samples=num_samples,
            pretrained_tokenizer_model_name=model_name,
            num_lookforward_tokens=num_lookforward_tokens,
            embedding_dim=embedding_dim,
            label_type=label_type,
            normalize_labels=normalize_labels,
            label_range=label_range,
        )
        dataset_folder = data_config.get_canonical_folder(dataset_base_folder)
        # Build (or reuse) the on-disk dataset.
        data_config.init_or_load_from(folder=dataset_folder, **data_config.model_dump())

        dataloading = CorpusRegressionDataloadingConfig(
            train_batch_size=batch_size,
            eval_batch_size=batch_size * eval_batch_size_multiple,
            drop_last=True,
        )

        # Mirror SL's per-sample learning-rate convention so SL and SL-CE
        # respond identically to canonical knobs (batch size). Per-token
        # scaling (× prefix_length) is dropped — CE here fires only at the
        # last position, so per-token doesn't make sense. The "head_lr /
        # divisor" path in SL becomes a single full-param lr here — both
        # Muon and AdamW groups train on the same backbone LR; no separate
        # head/backbone group, since `lm_head` is the native LM head rather
        # than a freshly-init'd regression head.
        head_lr = lr_per_sample * batch_size
        effective_backbone_lr_divisor = (
            1.0 if train_from_scratch else backbone_lr_divisor
        )
        return dict(
            data=data_config,
            dataset_folder=dataset_folder,
            dataloading=dataloading,
            pretrained_model=model_name,
            optimizer=CausalLMFullParamMuonOptimizerConfig(
                lr=head_lr / effective_backbone_lr_divisor,
                weight_decay=weight_decay,
                clip_grad_norm=clip_grad_norm,
            ),
            train_steps=train_steps,
            val_every_n_steps=val_every_n_steps,
            study_folder=(
                study_base_folder
                / dataset_folder.name
                / f"lr_{lr_per_sample:.2e}"
            ),
            compile_model=compile_model,
            train_from_scratch=train_from_scratch,
        )

    @classmethod
    def get_canonical(cls, **kwargs: object) -> Self:
        config = cls(**cls.canonical_kwargs(**kwargs))
        config.save_config_json()
        return config

    @classmethod
    def has_study_started(cls, study_folder: Path) -> bool:
        return (study_folder / "train_metrics.parquet").exists() and (
            study_folder / "config.json"
        ).exists()

    @classmethod
    def is_study_complete(cls, study_folder: Path) -> bool:
        if not cls.has_study_started(study_folder):
            return False
        train_steps = json.loads((study_folder / "config.json").read_text())[
            "train_steps"
        ]
        max_step = (
            pl.read_parquet(study_folder / "train_metrics.parquet")
            .select(pl.col("step").max())
            .item()
        )
        return max_step >= train_steps - 1

    @classmethod
    def prepare_study_folder(cls, *, study_folder: Path, tag: str) -> bool:
        """Return True if the study should run; False if it's already complete."""
        if cls.is_study_complete(study_folder):
            print(f"!!! {tag} already complete, skipping ({study_folder})")
            return False
        if cls.has_study_started(study_folder):
            print(f"=== {tag} partial → wiping {study_folder} ===")
            shutil.rmtree(study_folder)
        return True

    def save_config_json(self) -> None:
        self.study_folder.mkdir(parents=True, exist_ok=True)
        config_path = self.study_folder / "config.json"
        config_json = self.model_dump_json(indent=2)
        if not config_path.exists():
            config_path.write_text(config_json)
            return

        try:
            saved = type(self).model_validate_json(config_path.read_text())
        except ValidationError as e:
            warnings.warn(
                f"{config_path} fails validation under the current schema; "
                f"overwriting.\n{e}"
            )
            config_path.write_text(config_json)
            return
        if saved != self:
            warnings.warn(
                f"{config_path} already exists with different config contents"
            )

    def _build_label_projector(
        self, *, vocab_size: int, device: torch.device
    ) -> Float[Tensor, "vocab D"]:
        """Mirror of `ntp_baseline.run_ntp_baseline` lines 137–151. The
        projector turns next-token probabilities into the regression target
        space, so the projected MSE is comparable to SL/RL methods."""
        if self.data.label_type == "rademacher":
            return (
                self.data._rademacher_matrix(vocab_size).float().to(device=device)
            )
        if self.data.label_type == "token_id":
            projector = torch.arange(
                vocab_size, device=device, dtype=torch.float32
            ).unsqueeze(1)
            if self.data.normalize_labels:
                lo, hi = self.data.label_range
                projector = lo + (hi - lo) * (projector / vocab_size)
            return projector
        raise ValueError(f"Unknown label_type: {self.data.label_type!r}")

    def initialize(
        self,
        *,
        device: torch.device,
    ) -> "CorpusRegressionSLCEState":
        torch.set_float32_matmul_precision("medium")
        self.save_config_json()
        dataset = CorpusRegressionDataset.load_from(self.dataset_folder)
        train_dl = self.dataloading.get_train_dataloader(dataset)
        val_dl = self.dataloading.get_val_dataloader(dataset)

        device_context = (
            torch.cuda.device(device) if device.type == "cuda" else nullcontext()
        )
        with device_context:
            if self.train_from_scratch:
                hf_config = AutoConfig.from_pretrained(self.pretrained_model)
                model = AutoModelForCausalLM.from_config(hf_config).to(
                    device=device, dtype=torch.bfloat16
                )
            else:
                model = AutoModelForCausalLM.from_pretrained(
                    self.pretrained_model, torch_dtype=torch.bfloat16
                ).to(device=device)
            optimizer = self.optimizer.get_optimizer(model)
            label_projector = self._build_label_projector(
                vocab_size=model.config.vocab_size, device=device
            )
            if self.compile_model and device.type == "cuda":
                model = torch.compile(model, mode=self.compile_mode)

        return CorpusRegressionSLCEState(
            config=self,
            model=model,
            optimizer=optimizer,
            dataset=dataset,
            train_dl=train_dl,
            val_dl=val_dl,
            device=device,
            label_projector=label_projector,
        )


@dataclass(kw_only=True, config=ConfigDict(arbitrary_types_allowed=True))
class CorpusRegressionSLCEState:
    """State for the NTP-CE supervised trainer.

    Mirrors the public surface of `CorpusRegressionStudyBaseState`
    (`run_training` / `compute_validation` / metrics serialization /
    `step_and_zero_grad`) but holds its own `label_projector` and types
    `config` as `CorpusRegressionSLCEConfig`. The four helpers are duplicated
    verbatim from the base state — kept here intentionally to avoid forcing
    the base class to accept either config type.
    """

    config: CorpusRegressionSLCEConfig
    model: nn.Module
    optimizer: torch.optim.Optimizer
    dataset: CorpusRegressionDataset
    train_dl: DataLoader
    val_dl: DataLoader
    device: torch.device
    label_projector: Float[Tensor, "vocab D"]

    def compute_last_step_projections(
        self,
        *,
        context: Int[Tensor, "batch seq"],
    ) -> Float[Tensor, "batch D"]:
        """Project last-position next-token logits through the label projector.

        Identical to the formula in `ntp_baseline.py` lines 219–224, exposed
        as a method so `compute_validation` produces predictions in the same
        target space as SL/GRPO/RLOO/MaxRL.
        """
        output = self.model(input_ids=context, use_cache=False)
        logits: Float[Tensor, "batch vocab"] = output.logits[:, -1, :]
        probs: Float[Tensor, "batch vocab"] = F.softmax(logits.float(), dim=-1)
        projected: Float[Tensor, "batch D"] = probs @ self.label_projector
        return projected.to(logits.dtype)

    def train_step(
        self,
        *,
        batch: tuple[
            Int[Tensor, "batch seq"],
            Float[Tensor, "batch D"],
            Int[Tensor, "batch"],
        ],
    ) -> TrainStepOutput:
        tokens, target, lookahead_token_ids = batch
        tokens = tokens.to(device=self.device, dtype=torch.long)
        target = target.to(device=self.device, dtype=torch.bfloat16)
        lookahead_token_ids = lookahead_token_ids.to(
            device=self.device, dtype=torch.long
        )

        device_context = (
            torch.cuda.device(self.device)
            if self.device.type == "cuda"
            else nullcontext()
        )
        with (
            device_context,
            torch.autocast(device_type=self.device.type, dtype=torch.bfloat16),
        ):
            output = self.model(input_ids=tokens, use_cache=False)
            logits: Float[Tensor, "batch vocab"] = output.logits[:, -1, :]
            # CE always in float32 for numerical stability — matches the
            # `softmax(logits.float())` pattern in ntp_baseline.py.
            loss = F.cross_entropy(logits.float(), lookahead_token_ids)
            with torch.no_grad():
                probs: Float[Tensor, "batch vocab"] = F.softmax(
                    logits.float(), dim=-1
                )
                projected: Float[Tensor, "batch D"] = probs @ self.label_projector
                mse = F.mse_loss(projected.float(), target.float()).item()
                xx, xy, yy, pred_sum, target_sum, n = _batch_sufficient_stats(
                    prediction=projected, target=target,
                )

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.step_and_zero_grad()
        return TrainStepOutput(
            loss=loss.item(), mse=mse, xx=xx, xy=xy, yy=yy,
            pred_sum=pred_sum, target_sum=target_sum, n=n,
        )

    def compute_validation(self) -> CorpusRegressionValidationOutput:
        """Verbatim copy of `CorpusRegressionStudyBaseState.compute_validation`
        — kept here so the base class doesn't need to accept either config
        type."""
        model_was_training = self.model.training
        self.model.eval()

        predictions: list[Float[Tensor, "batch D"]] = []
        targets: list[Float[Tensor, "batch D"]] = []
        device_context = (
            torch.cuda.device(self.device)
            if self.device.type == "cuda"
            else nullcontext()
        )
        with (
            torch.no_grad(),
            device_context,
            torch.autocast(device_type=self.device.type, dtype=torch.bfloat16),
        ):
            for tokens, target, _lookahead_token_ids in tqdm(
                self.val_dl,
                desc="validation",
            ):
                tokens: Int[Tensor, "batch seq"] = tokens.to(
                    device=self.device,
                    dtype=torch.long,
                )
                target: Float[Tensor, "batch D"] = target.to(
                    device=self.device,
                    dtype=torch.bfloat16,
                )
                prediction = self.compute_last_step_projections(context=tokens)
                predictions.append(prediction.float().cpu())
                targets.append(target.float().cpu())

        if model_was_training:
            self.model.train()

        return CorpusRegressionValidationOutput(
            model_preds=torch.cat(predictions),
            target=torch.cat(targets),
        )

    def _flush_train_metrics(self, buffer: list[dict[str, float]]) -> None:
        # Verbatim of state.CorpusRegressionStudyBaseState._flush_train_metrics —
        # see the docstring there. Kept as a copy because sl_ce inherits from
        # BaseConfig rather than the corpus-regression base state, so the
        # plumbing has to live alongside its own train_step.
        if not buffer:
            return
        target_schema = {
            "step": pl.Int64,
            "loss": pl.Float64,
            "mse": pl.Float64,
            "corr": pl.Float64,
            "pred_var": pl.Float64,
            "target_var": pl.Float64,
        }
        new_rows = pl.DataFrame(buffer, schema=target_schema)
        metrics_path = self.config.study_folder / "train_metrics.parquet"
        if metrics_path.exists():
            existing = pl.read_parquet(metrics_path)
            for col, dtype in target_schema.items():
                if col not in existing.columns:
                    existing = existing.with_columns(
                        pl.lit(None, dtype=dtype).alias(col)
                    )
            existing = existing.select(list(target_schema.keys()))
            min_new_step = new_rows["step"].min()
            existing = existing.filter(pl.col("step") < min_new_step)
            new_rows = pl.concat([existing, new_rows]).sort("step")
        new_rows.write_parquet(metrics_path)

    def _serialize_val(
        self,
        *,
        step: int,
        validation: CorpusRegressionValidationOutput,
        train_window_xx: Tensor,
        train_window_xy: Tensor,
        train_window_yy: Tensor,
        train_window_pred_sum: Tensor,
        train_window_target_sum: Tensor,
        train_window_n: int,
    ) -> None:
        # Verbatim of state.CorpusRegressionStudyBaseState._serialize_val —
        # see the docstring there for the window-aggregate semantics of
        # `train_target_*`.
        validation.save_to(self.config.study_folder / f"val-step-{step}")

        val_counter = validation.compute_counter()
        sufficient_stats: dict[str, object] = {}
        sufficient_stats.update(_per_dim_stats(val_counter, prefix="val_target"))
        sufficient_stats.update(
            _per_dim_stats_from_tensors(
                xx=train_window_xx,
                xy=train_window_xy,
                yy=train_window_yy,
                pred_sum=train_window_pred_sum,
                target_sum=train_window_target_sum,
                n=train_window_n,
                prefix="train_target",
            )
        )

        metrics_path = self.config.study_folder / "val_metrics.parquet"
        new_row = pl.DataFrame({
            "step": [step],
            **{key: [value] for key, value in sufficient_stats.items()},
        })
        if metrics_path.exists():
            existing = pl.read_parquet(metrics_path)
            existing = _align_val_schema(existing, target_columns=new_row.columns)
            existing = existing.filter(pl.col("step") != step)
            new_row = pl.concat([existing, new_row]).sort("step")
        new_row.write_parquet(metrics_path)

    def run_training(self) -> None:
        """Flat step-based training loop. Validates every val_every_n_steps.

        Verbatim of `state.CorpusRegressionStudyBaseState.run_training`; see
        that docstring for the buffering/aggregation semantics."""
        self.model.train()
        batch_iter = _cycle_dataloader(self.train_dl)
        train_buffer: list[dict[str, float]] = []

        train_window_xx: Tensor | None = None
        train_window_xy: Tensor | None = None
        train_window_yy: Tensor | None = None
        train_window_pred_sum: Tensor | None = None
        train_window_target_sum: Tensor | None = None
        train_window_n: int = 0

        pbar = tqdm(range(self.config.train_steps), desc="training")
        for step in pbar:
            output = self.train_step(batch=next(batch_iter))
            corr = _scalar_corr_from_stats(xx=output.xx, xy=output.xy, yy=output.yy)
            pred_var = _scalar_var_from_stats(
                sq_sum=output.xx, val_sum=output.pred_sum, n=output.n,
            )
            target_var = _scalar_var_from_stats(
                sq_sum=output.yy, val_sum=output.target_sum, n=output.n,
            )
            train_buffer.append({
                "step": step,
                "loss": output.loss,
                "mse": output.mse,
                "corr": corr,
                "pred_var": pred_var,
                "target_var": target_var,
            })
            pbar.set_postfix(
                loss=f"{output.loss:.4f}",
                mse=f"{output.mse:.4f}",
                corr=f"{corr:.4f}",
                pred_var=f"{pred_var:.4f}",
            )

            if train_window_xx is None:
                train_window_xx = torch.zeros_like(output.xx)
                train_window_xy = torch.zeros_like(output.xy)
                train_window_yy = torch.zeros_like(output.yy)
                train_window_pred_sum = torch.zeros_like(output.pred_sum)
                train_window_target_sum = torch.zeros_like(output.target_sum)
            train_window_xx += output.xx
            train_window_xy += output.xy
            train_window_yy += output.yy
            train_window_pred_sum += output.pred_sum
            train_window_target_sum += output.target_sum
            train_window_n += output.n

            is_val_step = (
                (step + 1) % self.config.val_every_n_steps == 0
                or step == self.config.train_steps - 1
            )
            if is_val_step:
                validation = self.compute_validation()
                self._flush_train_metrics(train_buffer)
                assert train_window_xx is not None
                assert train_window_pred_sum is not None
                assert train_window_target_sum is not None
                self._serialize_val(
                    step=step,
                    validation=validation,
                    train_window_xx=train_window_xx,
                    train_window_xy=train_window_xy,
                    train_window_yy=train_window_yy,
                    train_window_pred_sum=train_window_pred_sum,
                    train_window_target_sum=train_window_target_sum,
                    train_window_n=train_window_n,
                )
                train_buffer = []
                train_window_xx.zero_()
                train_window_xy.zero_()
                train_window_yy.zero_()
                train_window_pred_sum.zero_()
                train_window_target_sum.zero_()
                train_window_n = 0
                self.model.train()

    def step_and_zero_grad(self) -> None:
        clip_grad_norm_(
            self.model.parameters(),
            max_norm=self.config.optimizer.clip_grad_norm,
        )
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
