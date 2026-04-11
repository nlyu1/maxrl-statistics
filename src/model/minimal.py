from __future__ import annotations

from dataclasses import dataclass

from jaxtyping import Float, Int
from torch import Tensor, nn

from __future__ import annotations

from typing import Any, Protocol

import torch
from jaxtyping import Float, Int
from torch import Tensor
from transformers.modeling_outputs import CausalLMOutputWithPast

class CausalLMConfig(BaseConfig):
    pretrained_model: str
    initial_output_norms: Float[list, "odim"]

    def get_model(self) -> "CausalLMWithLinearHead":
        ...


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
    ) -> CausalLMOutputWithPast:
        ...

    def generate(
        self,
        input_ids: Int[Tensor, "batch seq"],
        **kwargs: Any,
    ) -> Any:
        ...

    def compute_transition_scores(
        self,
        sequences: Int[Tensor, "batch seq"],
        scores: tuple[Float[Tensor, "batch vocab"], ...],
        normalize_logits: bool = ...,
        beam_indices: Tensor | None = ...,
    ) -> Float[Tensor, "batch step"]:
        ...

class CausalLMWithLinearHead(nn.Module):
    @dataclass
    class RolloutOutput:
        tokens: Int[Tensor, "batch rollout step"]
        logprobs: Float[Tensor, "batch rollout step"]

    @dataclass
    class GradientOutput:
        """
        Gradients-attached return results.
        - Logprobs is summed over the whole rollout step trajectory
        - Projections is taken from the final rollout step
        """
        logprobs: Float[Tensor, "batch seq rollout"]
        projections: Float[Tensor, "batch seq rollout odim"]

    # Todo: tighten
    def __init__(
        self, *, backbone: nn.Module, initial_output_norms: Float[list, "odim"]
    ):
        super().__init__()
        self.backbone = backbone

        assert len(initial_output_norm.shape) == 1
        self.output_dim = len(initial_output_norms)

        self.backbone = backbone
        hidden_size = self.backbone.hidden_size
        self.linear_head = nn.Linear(
            hidden_size,
            output_dim,
            bias=False,
        )

        self._init_linear_head_weights(initial_output_norms)

    @torch.no_grad()
    def _init_linear_head_weights(self, initial_output_norms: Float[Tensor, "odim"]):
        nn.init.normal_(self.linear_head.weight)
        # Double-triple check the dimension math here
        head_norms: Float[Tensor, "odim"] = self.linear_head.weight.norm(1)  # right?
        for j, value in enumerate(initial_output_norms):
            self.linear_head.weight[j] = (
                self.linear_head.weight[j] / head_norms[j] * initial_output_norms[j]
            )

    def forward(
        self, *, tokens: Int[Tensor, "batch seq"]
    ) -> Float[Tensor, "batch seq odim"]:
        """
        Supervised, batched forward. Gradients attached
        """
        # What's the proper typing here? Find out
        hidden: Float[Tensor, "batch seq hdim"] = self.backbone(
            input_ids=tokens,
            use_cache=False,
            output_hidden_states=True,
            return_dict=True,
        )
        last_hidden: Float[Tensor, "batch seq hidden"] = hidden_states[-1]
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
            context=context, num_rollout_steps=num_rollout_steps
        )

        # Typing!
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

        rollout_tokens: Int[Tensor, "batch rollouts rseq"] = flat_rollout_tokens.reshape(
            batch_size,
            num_rollout_samples,
            num_rollout_steps
        )

        rollout_logprobs = flat_rollout_logprobs.reshape(
            ...
        )
        return RolloutOutput(
            tokens=rollout_tokens,
            logprobs=rollout_logprobs
        )

    def logprobs_and_projs(
        self, *,
        context: Int[Tensor, "batch seq"],
        rollouts: Int[Tensor, "batch rollout step"]
    ) -> tu

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