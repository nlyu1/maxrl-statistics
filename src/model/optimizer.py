from __future__ import annotations

import math
from typing import Literal

from muon import SingleDeviceMuonWithAuxAdam
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR

from src.config.base import BaseConfig
from src.model.minimal import CausalLMWithLinearHead


class LRScheduleConfig(BaseConfig):
    """Linear-warmup + cosine-decay LR schedule descriptor.

    Lockstep multiplicative factor `f(t) ∈ [0, 1]` applied to every param
    group's nominal LR each optimizer step. The factor is **the same** across
    backbone-Muon, backbone-AdamW, and (when present) head-AdamW groups, so
    the head/backbone LR ratio established by the canonical config is
    preserved throughout training.

    `lr_schedule == "flat"` disables scheduling entirely — `build_lr_scheduler`
    returns `None`, no scheduler is ever instantiated, and `warmup_ratio` /
    `lr_min_ratio` are ignored. This is the default and reproduces today's
    flat-LR behaviour byte-for-byte.

    `lr_schedule == "cosine"`:
      - linear ramp from `1/W` → `1` over the first `W = round(warmup_ratio
        * train_steps)` steps,
      - then half-cosine decay from `1` → `lr_min_ratio` over the remaining
        `train_steps - W` steps.
    """

    lr_schedule: Literal["flat", "cosine"] = "flat"
    warmup_ratio: float = 0.05
    lr_min_ratio: float = 0.1


def build_lr_scheduler(
    optimizer: Optimizer,
    *,
    schedule_config: LRScheduleConfig,
    train_steps: int,
) -> LambdaLR | None:
    """Build a `LambdaLR` for non-flat schedules; return `None` for `"flat"`.

    The returned scheduler should be stepped **after** every `optimizer.step()`
    call (PyTorch convention). The lambda function captures `train_steps` and
    `schedule_config` by closure — concretely:

        f(t) = (t + 1) / W                                            for t < W
        f(t) = lr_min_ratio + 0.5 * (1 − lr_min_ratio)
               * (1 + cos(π * (t − W) / (T − W)))                     for t ≥ W

    with `T = train_steps` and `W = round(warmup_ratio * train_steps)`. The
    `(t + 1) / W` warmup form (vs. the HuggingFace `t / W` convention) is
    chosen so step 0 has a small but non-zero LR — avoids a wasted first step
    while still letting Muon's momentum buffer and AdamW's preconditioner
    state warm up before the full LR lands.
    """
    if schedule_config.lr_schedule == "flat":
        return None
    if schedule_config.lr_schedule != "cosine":
        raise ValueError(
            f"Unsupported lr_schedule={schedule_config.lr_schedule!r}; "
            "expected 'flat' or 'cosine'.",
        )
    if train_steps < 1:
        raise ValueError(f"train_steps must be ≥ 1, got {train_steps}.")
    warmup_ratio = schedule_config.warmup_ratio
    lr_min_ratio = schedule_config.lr_min_ratio
    if not 0.0 <= warmup_ratio < 1.0:
        raise ValueError(
            f"warmup_ratio must be in [0.0, 1.0), got {warmup_ratio}.",
        )
    if not 0.0 <= lr_min_ratio <= 1.0:
        raise ValueError(
            f"lr_min_ratio must be in [0.0, 1.0], got {lr_min_ratio}.",
        )

    warmup_steps = round(warmup_ratio * train_steps)
    if warmup_steps >= train_steps:
        raise ValueError(
            f"warmup_steps ({warmup_steps}) must be < train_steps "
            f"({train_steps}); reduce warmup_ratio.",
        )
    decay_steps = train_steps - warmup_steps

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(decay_steps)
        progress = min(1.0, max(0.0, progress))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return lr_min_ratio + (1.0 - lr_min_ratio) * cosine

    return LambdaLR(optimizer, lr_lambda=lr_lambda)


class CausalLMWithLinearHeadOptimizerConfig(BaseConfig):
    """
    Single-optimizer config using SingleDeviceMuonWithAuxAdam.

    Backbone hidden matrices (2D, not in muon_skip_names) use Muon.
    Everything else — embeds, lm_head, 1D params, and the regression head — uses AdamW.
    Betas (0.9, 0.95) and Muon momentum 0.95 are canonical and hardcoded.
    """

    lr: float
    head_lr: float
    weight_decay: float
    muon_skip_names: tuple[str, ...] = ("embed", "lm_head")
    clip_grad_norm: float
    lr_schedule: LRScheduleConfig = LRScheduleConfig()

    def get_optimizer(
        self, model: CausalLMWithLinearHead
    ) -> SingleDeviceMuonWithAuxAdam:
        muon_params = []
        adam_backbone_params = []
        for name, param in model.backbone.named_parameters():
            if not param.requires_grad:
                continue
            if param.ndim >= 2 and not any(s in name for s in self.muon_skip_names):
                muon_params.append(param)
            else:
                adam_backbone_params.append(param)

        # linear_head is always AdamW — last projection, best practice is to skip Muon
        head_params = list(model.linear_head.parameters())

        param_groups = [
            dict(
                params=muon_params,
                lr=self.lr,
                momentum=0.95,
                weight_decay=self.weight_decay,
                use_muon=True,
            ),
            dict(
                params=adam_backbone_params,
                lr=self.lr,
                betas=(0.9, 0.95),
                eps=1e-10,
                weight_decay=self.weight_decay,
                use_muon=False,
            ),
            dict(
                params=head_params,
                lr=self.head_lr,
                betas=(0.9, 0.95),
                eps=1e-10,
                weight_decay=self.weight_decay,
                use_muon=False,
            ),
        ]
        return SingleDeviceMuonWithAuxAdam(param_groups)


class CausalLMFullParamMuonOptimizerConfig(BaseConfig):
    """
    Muon-based optimizer for an `AutoModelForCausalLM` (no swapped regression head).

    Backbone hidden matrices (2D, not in muon_skip_names) use Muon.
    Everything else — embeds, lm_head, 1D params — uses AdamW. Single LR for
    both groups (no separate head LR), preserving the previous SL_CE
    "one LR for everything" semantics; the 2D matrices are simply re-routed
    from AdamW to Muon. Betas (0.9, 0.95), eps 1e-10, Muon momentum 0.95 are
    canonical and hardcoded — matches `CausalLMWithLinearHeadOptimizerConfig`
    so the two configs are directly comparable when running side-by-side.

    Used by the `sl_ce` (NTP cross-entropy) trainer where the model is a vanilla
    HuggingFace causal LM with the pretrained `lm_head` retained.
    """

    lr: float
    weight_decay: float
    muon_skip_names: tuple[str, ...] = ("embed", "lm_head")
    clip_grad_norm: float
    lr_schedule: LRScheduleConfig = LRScheduleConfig()

    def get_optimizer(self, model: nn.Module) -> SingleDeviceMuonWithAuxAdam:
        muon_params = []
        adam_params = []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if param.ndim >= 2 and not any(s in name for s in self.muon_skip_names):
                muon_params.append(param)
            else:
                adam_params.append(param)

        param_groups = [
            dict(
                params=muon_params,
                lr=self.lr,
                momentum=0.95,
                weight_decay=self.weight_decay,
                use_muon=True,
            ),
            dict(
                params=adam_params,
                lr=self.lr,
                betas=(0.9, 0.95),
                eps=1e-10,
                weight_decay=self.weight_decay,
                use_muon=False,
            ),
        ]
        return SingleDeviceMuonWithAuxAdam(param_groups)
