from __future__ import annotations

from muon import SingleDeviceMuonWithAuxAdam
from torch import nn

from src.config.base import BaseConfig
from src.model.minimal import CausalLMWithLinearHead


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

