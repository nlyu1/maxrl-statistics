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
    def get_model(self) -> "CausalLMWithLinearHead":
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

    def parameters(self, recurse: bool = True) -> Any: ...


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
        """
        Shared-prefill rollout path.

        We prefill the prompt once per batch item, repeat the resulting cache across
        rollout samples, and then sample fixed rollout steps manually.
        """
        batch_size = context.shape[0]
        flat_batch_size = batch_size * num_rollout_samples

        if num_rollout_steps == 0:
            empty_tokens: Int[Tensor, "flat_batch step"] = context.new_empty(
                flat_batch_size, 0
            )
            empty_logprobs: Float[Tensor, "flat_batch step"] = torch.empty(
                flat_batch_size, 0, device=context.device, dtype=torch.float32
            )
            return CausalLMWithLinearHead.RolloutOutput(
                tokens=empty_tokens.reshape(batch_size, num_rollout_samples, 0),
                logprobs=empty_logprobs.reshape(batch_size, num_rollout_samples, 0),
            )

        prefill_output: CausalLMOutputWithPast = self.backbone(
            input_ids=context,
            use_cache=True,
            return_dict=True,
        )
        past_key_values = prefill_output.past_key_values
        if past_key_values is None:
            raise ValueError(
                "Backbone returned no past_key_values with use_cache=True."
            )

        next_logits: Float[Tensor, "flat_batch vocab"] = self._repeat_rows(
            prefill_output.logits[:, -1, :], num_rollout_samples
        )
        past_key_values = self._repeat_past_key_values(
            past_key_values, num_rollout_samples
        )

        sampled_tokens: list[Int[Tensor, "flat_batch"]] = []
        sampled_logprobs: list[Float[Tensor, "flat_batch"]] = []

        for step_idx in range(num_rollout_steps):
            next_tokens, token_logprobs = self._sample_from_logits(next_logits)
            sampled_tokens.append(next_tokens)
            sampled_logprobs.append(token_logprobs)

            if step_idx + 1 == num_rollout_steps:
                break

            step_output: CausalLMOutputWithPast = self.backbone(
                input_ids=next_tokens[:, None],
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
            )
            past_key_values = step_output.past_key_values
            if past_key_values is None:
                raise ValueError(
                    "Backbone returned no past_key_values during cached rollout."
                )
            next_logits = step_output.logits[:, -1, :]

        flat_tokens: Int[Tensor, "flat_batch step"] = torch.stack(sampled_tokens, dim=1)
        flat_logprobs: Float[Tensor, "flat_batch step"] = torch.stack(
            sampled_logprobs, dim=1
        )

        rollout_tokens: Int[Tensor, "batch rollout step"] = flat_tokens.reshape(
            batch_size, num_rollout_samples, num_rollout_steps
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

        flat_context = self._repeat_rows(context, num_rollouts)
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
    def _repeat_rows(
        tensor: Tensor,
        repeats: int,
    ) -> Tensor:
        if repeats == 1:
            return tensor
        return torch.repeat_interleave(
            tensor,
            repeats=repeats,
            dim=0,
            output_size=tensor.shape[0] * repeats,
        )

    @staticmethod
    def _repeat_past_key_values(
        past_key_values: Any,
        repeats: int,
    ) -> Any:
        if repeats == 1:
            return past_key_values

        # Newer HF cache objects expose batch_repeat_interleave().
        if hasattr(past_key_values, "batch_repeat_interleave"):
            repeated = past_key_values.batch_repeat_interleave(repeats)
            return past_key_values if repeated is None else repeated

        # Fallback for tuple-of-tuples past_key_values.
        return tuple(
            tuple(
                torch.repeat_interleave(
                    tensor,
                    repeats=repeats,
                    dim=0,
                    output_size=tensor.shape[0] * repeats,
                )
                for tensor in layer
            )
            for layer in past_key_values
        )

    @staticmethod
    def _sample_from_logits(
        logits: Float[Tensor, "flat_batch vocab"],
    ) -> tuple[Int[Tensor, "flat_batch"], Float[Tensor, "flat_batch"]]:
        logits_float = logits.float()
        logprobs = F.log_softmax(logits_float, dim=-1)
        tokens = torch.distributions.Categorical(logits=logits_float).sample()
        token_logprobs = logprobs.gather(dim=-1, index=tokens[:, None]).squeeze(-1)
        return tokens, token_logprobs
