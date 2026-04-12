from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import torch
import torch.nn.functional as F
from jaxtyping import Float, Int
from torch import Tensor, nn
from transformers.modeling_outputs import CausalLMOutputWithPast

from src.config.base import BaseConfig


class CausalLMConfig(BaseConfig):
    pretrained_model: str
    initial_output_norms: list[float]  # one norm per output dimension

    def get_model(self) -> CausalLMWithLinearHead:
        from transformers import AutoModelForCausalLM

        backbone = AutoModelForCausalLM.from_pretrained(self.pretrained_model)
        return CausalLMWithLinearHead(
            backbone=backbone,
            initial_output_norms=torch.tensor(self.initial_output_norms),
        )


class _CausalLMConfigLike(Protocol):
    hidden_size: int


class CausalLMBackbone(Protocol):
    config: _CausalLMConfigLike

    def __call__(
        self,
        *,
        input_ids: Int[Tensor, "batch seq"],
        use_cache: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        **kwargs: Any,
    ) -> CausalLMOutputWithPast: ...

    def generate(
        self,
        input_ids: Int[Tensor, "batch seq"],
        **kwargs: Any,
    ) -> Any: ...

    def compute_transition_scores(
        self,
        sequences: Int[Tensor, "batch seq"],
        scores: tuple[Float[Tensor, "batch vocab"], ...],
        normalize_logits: bool = ...,
        beam_indices: Tensor | None = ...,
    ) -> Float[Tensor, "batch step"]: ...


class CausalLMWithLinearHead(nn.Module):
    @dataclass
    class RolloutOutput:
        tokens: Int[Tensor, "batch rollout step"]
        logprobs: Float[Tensor, "batch rollout step"]  # frozen

    @dataclass
    class GradientOutput:
        logprobs: Float[Tensor, "batch rollout step"]  # grad-attached
        projections: Float[Tensor, "batch rollout odim"]  # final-step only

    def __init__(
        self,
        *,
        backbone: CausalLMBackbone,
        initial_output_norms: Float[Tensor, "odim"],
    ):
        super().__init__()
        self.backbone = backbone
        self.output_dim = len(initial_output_norms)
        hidden_size = self.backbone.config.hidden_size
        backbone_dtype = next(backbone.parameters()).dtype
        self.linear_head = nn.Linear(
            hidden_size, self.output_dim, bias=False, dtype=backbone_dtype
        )
        self._init_linear_head_weights(initial_output_norms)

    @torch.no_grad()
    def _init_linear_head_weights(self, initial_output_norms: Float[Tensor, "odim"]):
        nn.init.normal_(self.linear_head.weight)
        head_norms: Float[Tensor, "odim"] = self.linear_head.weight.norm(dim=1)
        for j in range(self.output_dim):
            self.linear_head.weight[j] = (
                self.linear_head.weight[j] / head_norms[j] * initial_output_norms[j]
            )

    def forward(
        self, *, tokens: Int[Tensor, "batch seq"]
    ) -> Float[Tensor, "batch odim"]:
        """Supervised batched forward. Returns last-position projection. Gradients attached."""
        output: CausalLMOutputWithPast = self.backbone(
            input_ids=tokens,
            use_cache=False,
            output_hidden_states=True,
            return_dict=True,
        )
        last_hidden: Float[Tensor, "batch hdim"] = output.hidden_states[-1][:, -1, :]
        return self.linear_head(last_hidden)

    @torch.no_grad()
    def rollout(
        self,
        *,
        context: Int[Tensor, "batch seq"],
        num_rollout_samples: int,
        num_rollout_steps: int,
    ) -> RolloutOutput:
        batch_size, context_len = context.shape
        flat_context = self._repeat_context(
            context=context, num_rollouts=num_rollout_samples
        )

        generation_output = self.backbone.generate(
            input_ids=flat_context,
            do_sample=True,
            temperature=1,
            top_k=None,
            top_p=None,
            min_new_tokens=num_rollout_steps,
            max_new_tokens=num_rollout_steps,
            return_dict_in_generate=True,
            output_scores=True,
            use_cache=True,
        )

        full_sequences: Int[Tensor, "flat_batch full_seq"] = generation_output.sequences
        flat_rollout_tokens = full_sequences[:, context_len:]
        rollout_tokens: Int[Tensor, "batch rollout step"] = flat_rollout_tokens.reshape(
            batch_size, num_rollout_samples, num_rollout_steps
        )

        flat_logprobs: Float[Tensor, "flat_batch step"] = (
            self.backbone.compute_transition_scores(
                sequences=generation_output.sequences,
                scores=generation_output.scores,
                normalize_logits=True,
            )
        )
        rollout_logprobs: Float[Tensor, "batch rollout step"] = flat_logprobs.reshape(
            batch_size, num_rollout_samples, num_rollout_steps
        )

        return CausalLMWithLinearHead.RolloutOutput(
            tokens=rollout_tokens,
            logprobs=rollout_logprobs,
        )

    def logprobs_and_projs(
        self,
        *,
        context: Int[Tensor, "batch seq"],
        rollouts: Int[Tensor, "batch rollout step"],
    ) -> GradientOutput:
        """
        Forward pass with gradients attached.
        Logprobs: per-step, from last context token (inclusive) to last rollout
        token (exclusive) — exactly the positions that affected sampling.
        Projections: linear head applied to the final rollout step hidden state.
        """
        batch_size, context_len = context.shape
        num_rollouts, rollout_steps = rollouts.shape[1], rollouts.shape[2]

        flat_context = self._repeat_context(context=context, num_rollouts=num_rollouts)
        flat_rollouts: Int[Tensor, "flat_batch step"] = rollouts.reshape(
            batch_size * num_rollouts, rollout_steps
        )
        full_sequences: Int[Tensor, "flat_batch full_seq"] = torch.cat(
            [flat_context, flat_rollouts], dim=1
        )

        output: CausalLMOutputWithPast = self.backbone(
            input_ids=full_sequences,
            use_cache=False,
            output_hidden_states=True,
            return_dict=True,
        )

        # logits[:, t, :] predicts token at position t+1.
        # Rollout token at position context_len+k is predicted by logits[:, context_len-1+k, :].
        logits: Float[Tensor, "flat_batch full_seq vocab"] = output.logits
        rollout_logits = logits[:, context_len - 1 : context_len - 1 + rollout_steps, :]
        per_step_logprobs: Float[Tensor, "flat_batch step"] = (
            F.log_softmax(rollout_logits.float(), dim=-1)
            .gather(dim=-1, index=flat_rollouts.unsqueeze(-1))
            .squeeze(-1)
        )
        logprobs: Float[Tensor, "batch rollout step"] = per_step_logprobs.reshape(
            batch_size, num_rollouts, rollout_steps
        )

        # Projection from the final rollout step hidden state
        last_layer_hidden: Float[Tensor, "flat_batch full_seq hdim"] = (
            output.hidden_states[-1]
        )
        final_hidden: Float[Tensor, "flat_batch hdim"] = last_layer_hidden[:, -1, :]
        flat_projections: Float[Tensor, "flat_batch odim"] = self.linear_head(
            final_hidden
        )
        projections: Float[Tensor, "batch rollout odim"] = flat_projections.reshape(
            batch_size, num_rollouts, self.output_dim
        )

        return CausalLMWithLinearHead.GradientOutput(
            logprobs=logprobs,
            projections=projections,
        )

    @staticmethod
    def _repeat_context(
        *,
        context: Int[Tensor, "batch seq"],
        num_rollouts: int,
    ) -> Int[Tensor, "flat_batch seq"]:
        batch_size, context_len = context.shape
        return (
            context[:, None, :]
            .expand(batch_size, num_rollouts, context_len)
            .reshape(batch_size * num_rollouts, context_len)
        )
