"""
GRPO training on the bag-of-words regression task.

The model generates T "thinking tokens" via rollout after reading the context,
then a projection head reads off the final hidden state to predict the target.
GRPO optimizes the generation policy so rollouts produce hidden states that
yield better predictions.

Usage:
    uv run python notebooks/bow-grpo/run_grpo.py
"""

from __future__ import annotations

import os
from contextlib import nullcontext
from pathlib import Path

import polars as pl
import torch
import torch.nn.functional as F
from jaxtyping import Float, Int
from torch import Tensor
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm

repo_root = Path(__file__).resolve().parents[2]
os.chdir(repo_root)

from src.data.bag_of_words import BagOfWordsDatasetConfig, canonical_bags  # noqa: E402
from src.data.parquet import TokenizedParquetDatasetConfig  # noqa: E402
from src.data.dataloading import DataloadingConfig  # noqa: E402
from src.metrics import CorrelationCounter  # noqa: E402
from src.model.minimal import CausalLMConfig, CausalLMWithLinearHead  # noqa: E402
from src.model.optimizer import CausalLMWithLinearHeadOptimizerConfig  # noqa: E402

# ── hyperparameters ──────────────────────────────────────────────────────────

DEVICE = torch.device("cuda:1")

# Dataset (matches SFT baseline)
SNR = 100.0
NUM_WORDS = 15
NUM_SAMPLES = 50_000
AUX_WORDS_RATIO = 0.5
PROMPT_LENGTH = 128
WORD_DECAY_POWER = 1.0
MODEL_NAME = "HuggingFaceTB/SmolLM2-360M"

# GRPO-specific
NUM_ROLLOUT_SAMPLES = 8   # G: rollouts per context for group statistics
NUM_ROLLOUT_STEPS = 1     # T: "thinking tokens" generated per rollout
GRPO_BATCH_SIZE = 8       # micro-batch; effective batch = GRPO_BATCH_SIZE * GRAD_ACCUM_STEPS
GRAD_ACCUM_STEPS = 4      # accumulate to effective batch of 32
PROJECTION_LOSS_WEIGHT = 1.0  # weight on projection MSE relative to policy gradient

# Training schedule
NUM_SFT_WARMUP_EPOCHS = 1   # 1 epoch of SFT warmup before GRPO
NUM_GRPO_EPOCHS = 10

# Optimizer (same structure as SFT baseline)
LR_PER_TOKEN = 1.25e-7
BACKBONE_LR_DIVISOR = 5.0
WEIGHT_DECAY = 0.0
CLIP_GRAD_NORM = 1.0

# Paths
DATA_BASE = repo_root / "artifacts" / "bow-data"
STUDY_FOLDER = (
    repo_root / "artifacts" / "bow-grpo"
    / f"grpo_G{NUM_ROLLOUT_SAMPLES}_T{NUM_ROLLOUT_STEPS}_frozen_snr{int(SNR)}"
)

# ── dataset setup ────────────────────────────────────────────────────────────


def setup_data():
    dataset_folder = DATA_BASE / (
        f"{NUM_WORDS}-words_snr-{SNR}_len-{PROMPT_LENGTH}"
        f"_pow-{WORD_DECAY_POWER}_ar-{AUX_WORDS_RATIO}"
    )
    data_config = BagOfWordsDatasetConfig.init_or_load_from(
        folder=dataset_folder,
        snr=SNR,
        num_train_samples=NUM_SAMPLES,
        num_val_samples=NUM_SAMPLES,
        prompt_length=PROMPT_LENGTH,
        word_assignments=list(canonical_bags[NUM_WORDS]),
        aux_words_ratio=AUX_WORDS_RATIO,
        word_decay_power=WORD_DECAY_POWER,
    )

    tokenization = TokenizedParquetDatasetConfig(
        tokenizer_model_name=MODEL_NAME,
        folder=dataset_folder,
        filter_samples_above_n_tokens=384,
        pad_to_multiple=8,
    )
    dataset = tokenization.init_or_load_dataset()

    # SFT warmup uses larger batches (matches SFT baseline)
    sft_dataloading = DataloadingConfig(
        train_batch_size=64,
        eval_batch_size=256,
        drop_last=True,
        world_size=1,
        rank=0,
    )
    sft_train_dl = sft_dataloading.get_train_dataloader(dataset)

    # GRPO uses smaller micro-batches (fans out G rollouts per item)
    grpo_dataloading = DataloadingConfig(
        train_batch_size=GRPO_BATCH_SIZE,
        eval_batch_size=256,
        drop_last=True,
        world_size=1,
        rank=0,
    )
    grpo_train_dl = grpo_dataloading.get_train_dataloader(dataset)
    val_dl = grpo_dataloading.get_val_dataloader(dataset)
    return data_config, dataset, sft_train_dl, grpo_train_dl, val_dl


def setup_model():
    torch.set_float32_matmul_precision("medium")
    model_config = CausalLMConfig(
        pretrained_model=MODEL_NAME,
        initial_output_norms=[SNR / (1 + SNR**2) ** 0.5],  # match target_signal_std
    )
    with torch.cuda.device(DEVICE):
        model = model_config.get_model().to(device=DEVICE, dtype=torch.bfloat16)

    # Scale LR by effective batch size (micro_batch * grad_accum)
    effective_batch = GRPO_BATCH_SIZE * GRAD_ACCUM_STEPS
    head_lr = LR_PER_TOKEN * effective_batch * PROMPT_LENGTH
    opt_config = CausalLMWithLinearHeadOptimizerConfig(
        lr=head_lr / BACKBONE_LR_DIVISOR,
        head_lr=head_lr,
        weight_decay=WEIGHT_DECAY,
        clip_grad_norm=CLIP_GRAD_NORM,
    )
    optimizer = opt_config.get_optimizer(model)

    # Skip torch.compile — CUDA graphs + rollout memory is too tight
    return model, optimizer


# ── Rollout SFT step ─────────────────────────────────────────────────────────


def rollout_sft_step(
    model: CausalLMWithLinearHead,
    optimizer: torch.optim.Optimizer,
    tokens: Int[Tensor, "batch seq"],
    target: Float[Tensor, "batch"],
) -> float:
    """
    Rollout SFT: generate T tokens under current policy (no grad through
    sampling), then supervise the projection from the final hidden state.
    This matches the computational path that GRPO will later optimize.
    """
    with torch.autocast(device_type=DEVICE.type, dtype=torch.bfloat16):
        # Single rollout per context (G=1) — clean SFT analog
        rollout_output = model.rollout(
            context=tokens,
            num_rollout_samples=1,
            num_rollout_steps=NUM_ROLLOUT_STEPS,
        )
        # Forward with grad to get projection from rollout endpoint
        grad_output = model.logprobs_and_projs(
            context=tokens,
            rollouts=rollout_output.tokens,
        )
        # [B, 1, 1] → [B]
        prediction = grad_output.projections.squeeze(-1).squeeze(-1)
        loss = F.mse_loss(prediction, target)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    clip_grad_norm_(model.parameters(), max_norm=CLIP_GRAD_NORM)
    optimizer.step()
    return float(loss.detach().cpu())


# ── GRPO training step ──────────────────────────────────────────────────────


def grpo_microbatch(
    model: CausalLMWithLinearHead,
    tokens: Int[Tensor, "batch seq"],
    target: Float[Tensor, "batch"],
) -> dict[str, float]:
    """
    One GRPO micro-batch: computes loss and calls backward (accumulates grads).
    Does NOT step the optimizer.
    """
    # ── rollout (no grad) ────────────────────────────────────────────────
    with torch.autocast(device_type=DEVICE.type, dtype=torch.bfloat16):
        rollout_output = model.rollout(
            context=tokens,
            num_rollout_samples=NUM_ROLLOUT_SAMPLES,
            num_rollout_steps=NUM_ROLLOUT_STEPS,
        )
    rollout_tokens: Int[Tensor, "B G T"] = rollout_output.tokens

    # ── forward with grad ────────────────────────────────────────────────
    with torch.autocast(device_type=DEVICE.type, dtype=torch.bfloat16):
        grad_output = model.logprobs_and_projs(
            context=tokens,
            rollouts=rollout_tokens,
        )
    logprobs: Float[Tensor, "B G T"] = grad_output.logprobs
    projections: Float[Tensor, "B G"] = grad_output.projections.squeeze(-1)

    # ── rewards & advantages ─────────────────────────────────────────────
    target_expanded: Float[Tensor, "B G"] = target[:, None].expand_as(projections)
    with torch.no_grad():
        projs_f = projections.detach().float()
        rewards: Float[Tensor, "B G"] = -(projs_f - target_expanded.float()) ** 2
        reward_mean = rewards.mean(dim=1, keepdim=True)
        reward_std = rewards.std(dim=1, keepdim=True).clamp(min=1e-6)
        advantages: Float[Tensor, "B G"] = (rewards - reward_mean) / reward_std

        # Diagnostics: how much signal does the policy gradient have?
        proj_std = projs_f.std(dim=1).mean().item()        # spread across rollouts
        proj_mean_abs = projs_f.mean(dim=1).abs().mean().item()  # scale of predictions
        adv_abs_mean = advantages.abs().mean().item()       # effective advantage magnitude

    # ── losses (scaled by 1/GRAD_ACCUM_STEPS for correct averaging) ─────
    per_rollout_logprob: Float[Tensor, "B G"] = logprobs.sum(dim=-1)
    policy_loss = -(advantages * per_rollout_logprob).mean()

    # No projection loss — head is frozen during GRPO. Pure policy gradient.
    projection_loss = torch.zeros(1, device=policy_loss.device)

    total_loss = policy_loss / GRAD_ACCUM_STEPS

    total_loss.backward()

    # Backbone grad norm (from policy loss — the only backbone gradient source)
    backbone_grad_norm = 0.0
    for p in model.backbone.parameters():
        if p.grad is not None:
            backbone_grad_norm += p.grad.detach().float().norm().item() ** 2
    backbone_grad_norm = backbone_grad_norm ** 0.5

    head_grad_norm = 0.0
    for p in model.linear_head.parameters():
        if p.grad is not None:
            head_grad_norm += p.grad.detach().float().norm().item() ** 2
    head_grad_norm = head_grad_norm ** 0.5

    return {
        "policy_loss": float(policy_loss.detach().cpu()),
        "projection_loss": float(projection_loss.detach().cpu()),
        "reward_mean": float(rewards.mean().cpu()),
        "reward_std": float(reward_std.mean().cpu()),
        "proj_std": proj_std,
        "proj_mean_abs": proj_mean_abs,
        "adv_abs_mean": adv_abs_mean,
        "backbone_grad_norm": backbone_grad_norm,
        "head_grad_norm": head_grad_norm,
    }


# ── validation ───────────────────────────────────────────────────────────────


@torch.no_grad()
def run_validation(
    model: CausalLMWithLinearHead,
    val_dl,
    epoch: int,
) -> dict[str, float]:
    """
    Validation using direct forward (same as SFT) for apple-to-apple comparison.
    Direct forward only (apple-to-apple with SFT). Rollout validation is too
    expensive per epoch — can be done as a one-off after training.
    """
    model.eval()

    direct_counter = CorrelationCounter.initialize(dim=1, device=DEVICE)
    direct_gt_counter = CorrelationCounter.initialize(dim=1, device=DEVICE)

    with (
        torch.cuda.device(DEVICE),
        torch.autocast(device_type=DEVICE.type, dtype=torch.bfloat16),
    ):
        for tokens, target, ground_truth in tqdm(val_dl, desc=f"val epoch {epoch}"):
            tokens = tokens.to(device=DEVICE, dtype=torch.long)
            target = target.to(device=DEVICE, dtype=torch.bfloat16)
            ground_truth = ground_truth.to(device=DEVICE, dtype=torch.bfloat16)

            direct_pred = model(tokens=tokens).squeeze(-1).float()
            direct_counter.tick(
                x=direct_pred[:, None], y=target.float()[:, None]
            )
            direct_gt_counter.tick(
                x=direct_pred[:, None], y=ground_truth.float()[:, None]
            )

    model.train()

    direct_stats = direct_counter.get_stats()
    direct_gt_stats = direct_gt_counter.get_stats()
    return {
        "val_direct_corr_target": float(direct_stats.corr.squeeze(0).cpu()),
        "val_direct_corr_gt": float(direct_gt_stats.corr.squeeze(0).cpu()),
    }


# ── main ─────────────────────────────────────────────────────────────────────


def main():
    STUDY_FOLDER.mkdir(parents=True, exist_ok=True)
    print(f"Study folder: {STUDY_FOLDER}")
    print(f"SNR={SNR}, R²={SNR**2 / (1 + SNR**2):.4f}")
    print(f"GRPO: G={NUM_ROLLOUT_SAMPLES}, T={NUM_ROLLOUT_STEPS}")
    print(f"Micro-batch: {GRPO_BATCH_SIZE}, grad_accum: {GRAD_ACCUM_STEPS}, effective: {GRPO_BATCH_SIZE * GRAD_ACCUM_STEPS}")
    print(f"SFT warmup epochs: {NUM_SFT_WARMUP_EPOCHS}")
    print()

    data_config, dataset, sft_train_dl, grpo_train_dl, val_dl = setup_data()
    model, optimizer = setup_model()

    all_metrics: list[dict] = []

    # ── Phase 1: Rollout SFT warmup ────────────────────────────────────────
    # Generate T tokens under current policy, then supervise the projection
    # from the final hidden state. This matches GRPO's computational path.
    print(f"=== Phase 1: Rollout SFT warmup ({NUM_SFT_WARMUP_EPOCHS} epoch, T={NUM_ROLLOUT_STEPS}) ===")
    model.train()

    for sft_epoch in range(NUM_SFT_WARMUP_EPOCHS):
        warmup_corr = CorrelationCounter.initialize(dim=1, device=DEVICE)
        warmup_gt_corr = CorrelationCounter.initialize(dim=1, device=DEVICE)
        pbar = tqdm(sft_train_dl, desc=f"rollout-sft epoch {sft_epoch}")
        for tokens, target, ground_truth in pbar:
            tokens = tokens.to(device=DEVICE, dtype=torch.long)
            target = target.to(device=DEVICE, dtype=torch.bfloat16)

            with torch.cuda.device(DEVICE):
                loss = rollout_sft_step(model, optimizer, tokens, target)

            # Track correlation using direct forward (for comparison with SFT baseline)
            with torch.no_grad(), torch.autocast(device_type=DEVICE.type, dtype=torch.bfloat16):
                direct_pred = model(tokens=tokens).squeeze(-1).float()
                warmup_corr.tick(x=direct_pred[:, None], y=target.float()[:, None])
                warmup_gt_corr.tick(x=direct_pred[:, None], y=ground_truth.float().to(DEVICE)[:, None])

            tc = float(warmup_corr.get_stats().corr.squeeze(0).cpu())
            gc = float(warmup_gt_corr.get_stats().corr.squeeze(0).cpu())
            pbar.set_postfix(corr_t=f"{tc:.4f}", corr_gt=f"{gc:.4f}", loss=f"{loss:.4f}")

        tqdm.write(f"SFT warmup epoch {sft_epoch}: corr_t={tc:.4f}, corr_gt={gc:.4f}")

    # Validate after warmup
    val_metrics = run_validation(model, val_dl, epoch=-1)
    print(f"Post-warmup validation: {val_metrics}")
    all_metrics.append({"epoch": -1, "phase": "sft_warmup", **val_metrics})

    # ── Phase 2: GRPO training ───────────────────────────────────────────
    # Freeze projection head — fixed reward landscape for GRPO
    for p in model.linear_head.parameters():
        p.requires_grad_(False)
    print(f"\n=== Phase 2: GRPO training (grad_accum={GRAD_ACCUM_STEPS}, head FROZEN) ===")
    model.train()

    DIAG_EVERY = 200  # print diagnostic summary every N micro-batches

    for epoch in range(NUM_GRPO_EPOCHS):
        train_corr = CorrelationCounter.initialize(dim=1, device=DEVICE)
        train_gt_corr = CorrelationCounter.initialize(dim=1, device=DEVICE)
        epoch_policy_loss = 0.0
        epoch_proj_loss = 0.0
        num_optimizer_steps = 0
        accum_idx = 0
        step_idx = 0

        # Running averages for diagnostics
        diag_accum = {
            "proj_std": 0.0, "proj_mean_abs": 0.0, "adv_abs_mean": 0.0,
            "backbone_grad_norm": 0.0, "head_grad_norm": 0.0,
            "reward_std": 0.0, "policy_loss": 0.0,
        }
        diag_count = 0

        optimizer.zero_grad(set_to_none=True)
        pbar = tqdm(grpo_train_dl, desc=f"grpo epoch {epoch}")
        for tokens, target, ground_truth in pbar:
            tokens = tokens.to(device=DEVICE, dtype=torch.long)
            target = target.to(device=DEVICE, dtype=torch.bfloat16)
            ground_truth = ground_truth.to(device=DEVICE, dtype=torch.bfloat16)

            with torch.cuda.device(DEVICE):
                mb_metrics = grpo_microbatch(model, tokens, target)

            epoch_policy_loss += mb_metrics["policy_loss"]
            epoch_proj_loss += mb_metrics["projection_loss"]
            accum_idx += 1
            step_idx += 1

            # Accumulate diagnostics
            for k in diag_accum:
                diag_accum[k] += mb_metrics[k]
            diag_count += 1

            if accum_idx == GRAD_ACCUM_STEPS:
                clip_grad_norm_(model.parameters(), max_norm=CLIP_GRAD_NORM)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                num_optimizer_steps += 1
                accum_idx = 0

            # Track direct-forward correlation for comparison with SFT
            with torch.no_grad():
                pred = model(tokens=tokens).squeeze(-1).float()
                train_corr.tick(x=pred[:, None], y=target.float()[:, None])
                train_gt_corr.tick(
                    x=pred[:, None], y=ground_truth.float()[:, None]
                )

            tc = float(train_corr.get_stats().corr.squeeze(0).cpu())
            gc = float(train_gt_corr.get_stats().corr.squeeze(0).cpu())
            pbar.set_postfix(
                corr_gt=f"{gc:.4f}",
                proj_std=f"{mb_metrics['proj_std']:.4f}",
                rew_std=f"{mb_metrics['reward_std']:.4f}",
                adv=f"{mb_metrics['adv_abs_mean']:.3f}",
                bb_gn=f"{mb_metrics['backbone_grad_norm']:.2e}",
            )

            # Periodic diagnostic dump
            if step_idx % DIAG_EVERY == 0 and diag_count > 0:
                avg = {k: v / diag_count for k, v in diag_accum.items()}
                proj_snr = avg["proj_mean_abs"] / max(avg["proj_std"], 1e-8)
                tqdm.write(
                    f"  [step {step_idx}] DIAGNOSTICS (avg over {diag_count} mb):\n"
                    f"    proj_std={avg['proj_std']:.4f}  proj_mean_abs={avg['proj_mean_abs']:.4f}  proj_snr={proj_snr:.1f}\n"
                    f"    reward_std={avg['reward_std']:.4f}  |advantage|={avg['adv_abs_mean']:.4f}\n"
                    f"    backbone_grad_norm={avg['backbone_grad_norm']:.2e}  head_grad_norm={avg['head_grad_norm']:.2e}\n"
                    f"    policy_loss={avg['policy_loss']:.4f}  corr_gt={gc:.4f}"
                )
                diag_accum = {k: 0.0 for k in diag_accum}
                diag_count = 0

        # Flush any remaining accumulated grads
        if accum_idx > 0:
            clip_grad_norm_(model.parameters(), max_norm=CLIP_GRAD_NORM)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            num_optimizer_steps += 1

        total_mb = len(grpo_train_dl)

        # End-of-epoch validation
        val_metrics = run_validation(model, val_dl, epoch=epoch)
        train_target_stats = train_corr.get_stats()
        train_gt_stats = train_gt_corr.get_stats()

        epoch_metrics = {
            "epoch": epoch,
            "phase": "grpo",
            "train_corr_target": float(train_target_stats.corr.squeeze(0).cpu()),
            "train_corr_gt": float(train_gt_stats.corr.squeeze(0).cpu()),
            "avg_policy_loss": epoch_policy_loss / max(total_mb, 1),
            "avg_proj_loss": epoch_proj_loss / max(total_mb, 1),
            "optimizer_steps": num_optimizer_steps,
            **val_metrics,
        }
        all_metrics.append(epoch_metrics)

        tqdm.write(
            f"epoch {epoch:2d}  "
            f"train_corr_t={epoch_metrics['train_corr_target']:.4f}  "
            f"train_corr_gt={epoch_metrics['train_corr_gt']:.4f}  "
            f"val_corr_t={val_metrics['val_direct_corr_target']:.4f}  "
            f"val_corr_gt={val_metrics['val_direct_corr_gt']:.4f}  "
            f"pol_loss={epoch_metrics['avg_policy_loss']:.4f}  "
            f"proj_loss={epoch_metrics['avg_proj_loss']:.4f}"
        )

        # Save metrics
        pl.DataFrame(all_metrics).write_parquet(STUDY_FOLDER / "metrics.parquet")

    print("\nGRPO training done.")
    print(f"Metrics saved to {STUDY_FOLDER / 'metrics.parquet'}")


if __name__ == "__main__":
    main()
