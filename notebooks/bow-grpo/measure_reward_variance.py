"""
Diagnostic: measure reward variance across rollouts at different T values.
This tells us whether the policy gradient has any signal to work with.

Usage:
    uv run python notebooks/bow-grpo/measure_reward_variance.py
"""

from __future__ import annotations

import os
from pathlib import Path

import torch
import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor
from tqdm import tqdm

repo_root = Path(__file__).resolve().parents[2]
os.chdir(repo_root)

from src.data.bag_of_words import BagOfWordsDatasetConfig, canonical_bags  # noqa: E402
from src.data.dataloading import DataloadingConfig  # noqa: E402
from src.data.parquet import TokenizedParquetDatasetConfig  # noqa: E402
from src.model.minimal import CausalLMConfig, CausalLMWithLinearHead  # noqa: E402
from src.model.optimizer import CausalLMWithLinearHeadOptimizerConfig  # noqa: E402

DEVICE = torch.device("cuda:1")
NUM_ROLLOUT_SAMPLES = 16  # enough for statistics, fits in memory
T_VALUES = [0, 1, 2, 4, 8, 16]
NUM_BATCHES = 100  # more batches to compensate for fewer rollouts
BATCH_SIZE = 8

# Dataset setup (same as other experiments)
DATA_BASE = repo_root / "artifacts" / "bow-data"
SNR = 1.0
NUM_WORDS = 15
PROMPT_LENGTH = 128
MODEL_NAME = "HuggingFaceTB/SmolLM2-360M"


def main():
    torch.set_float32_matmul_precision("medium")

    dataset_folder = DATA_BASE / (
        f"{NUM_WORDS}-words_snr-{SNR}_len-{PROMPT_LENGTH}_pow-1.0_ar-0.5"
    )
    data_config = BagOfWordsDatasetConfig.init_or_load_from(
        folder=dataset_folder,
        snr=SNR,
        num_train_samples=50_000,
        num_val_samples=50_000,
        prompt_length=PROMPT_LENGTH,
        word_assignments=list(canonical_bags[NUM_WORDS]),
        aux_words_ratio=0.5,
        word_decay_power=1.0,
    )
    tokenization = TokenizedParquetDatasetConfig(
        tokenizer_model_name=MODEL_NAME,
        folder=dataset_folder,
        filter_samples_above_n_tokens=384,
        pad_to_multiple=8,
    )
    dataset = tokenization.init_or_load_dataset()
    dl = DataloadingConfig(
        train_batch_size=BATCH_SIZE,
        eval_batch_size=BATCH_SIZE,
        drop_last=True,
        world_size=1,
        rank=0,
    ).get_val_dataloader(dataset)

    # --- Fresh model (pretrained, no finetuning) ---
    model_config = CausalLMConfig(
        pretrained_model=MODEL_NAME,
        initial_output_norms=[SNR],
    )
    with torch.cuda.device(DEVICE):
        model = model_config.get_model().to(device=DEVICE, dtype=torch.bfloat16)
    model.eval()

    # --- Also do 1 epoch of SFT warmup for a "warm" model ---
    # We'll measure both fresh and warm
    for model_label in ["fresh_pretrained", "after_1epoch_sft"]:
        if model_label == "after_1epoch_sft":
            # Quick SFT warmup
            print("\n--- Running 1 epoch SFT warmup ---")
            opt_config = CausalLMWithLinearHeadOptimizerConfig(
                lr=2.048e-4 / 5.0,
                head_lr=2.048e-4,
                weight_decay=0.0,
                clip_grad_norm=1.0,
            )
            optimizer = opt_config.get_optimizer(model)
            model.train()
            sft_dl = DataloadingConfig(
                train_batch_size=64,
                eval_batch_size=64,
                drop_last=True,
                world_size=1,
                rank=0,
            ).get_train_dataloader(dataset)
            for tokens, target, _ in tqdm(sft_dl, desc="sft warmup"):
                tokens = tokens.to(device=DEVICE, dtype=torch.long)
                target = target.to(device=DEVICE, dtype=torch.bfloat16)
                with torch.autocast(device_type=DEVICE.type, dtype=torch.bfloat16):
                    pred = model(tokens=tokens).squeeze(-1)
                    loss = F.mse_loss(pred, target)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            model.eval()

        print(f"\n{'='*60}")
        print(f"Model: {model_label}")
        print(f"{'='*60}")

        for T in T_VALUES:
            if T == 0:
                # Zero-step: just direct forward, measure variance of direct projection
                # (baseline — should be zero variance since it's deterministic)
                all_proj_stds: list[float] = []
                all_reward_stds: list[float] = []
                batch_count = 0
                for tokens, target, _ in dl:
                    if batch_count >= NUM_BATCHES:
                        break
                    tokens = tokens.to(device=DEVICE, dtype=torch.long)
                    target = target.to(device=DEVICE, dtype=torch.bfloat16)
                    with (
                        torch.no_grad(),
                        torch.autocast(device_type=DEVICE.type, dtype=torch.bfloat16),
                    ):
                        pred = model(tokens=tokens).squeeze(-1)  # [B]
                        # "Fake" G dimension by repeating
                        pred_expanded = pred[:, None].expand(-1, NUM_ROLLOUT_SAMPLES)  # [B, G]
                        target_expanded = target[:, None].expand_as(pred_expanded)
                        rewards = -(pred_expanded - target_expanded) ** 2
                        all_proj_stds.append(0.0)  # deterministic
                        all_reward_stds.append(0.0)  # deterministic
                    batch_count += 1
                print(f"  T={T:3d}: proj_std=0.0000 (deterministic)  reward_std=0.0000")
                continue

            all_proj_stds = []
            all_reward_stds = []
            all_proj_means = []
            all_advantage_abs_means = []
            batch_count = 0

            for tokens, target, _ in dl:
                if batch_count >= NUM_BATCHES:
                    break
                tokens = tokens.to(device=DEVICE, dtype=torch.long)
                target = target.to(device=DEVICE, dtype=torch.bfloat16)

                with (
                    torch.no_grad(),
                    torch.autocast(device_type=DEVICE.type, dtype=torch.bfloat16),
                ):
                    ro = model.rollout(
                        context=tokens,
                        num_rollout_samples=NUM_ROLLOUT_SAMPLES,
                        num_rollout_steps=T,
                    )
                    go = model.logprobs_and_projs(
                        context=tokens,
                        rollouts=ro.tokens,
                    )
                    projs: Float[Tensor, "B G"] = go.projections.squeeze(-1).float()
                    target_exp = target[:, None].expand_as(projs).float()
                    rewards = -(projs - target_exp) ** 2

                    # Per-context statistics
                    proj_std = projs.std(dim=1).mean().item()  # mean over batch
                    proj_mean = projs.mean(dim=1).abs().mean().item()
                    reward_std = rewards.std(dim=1).mean().item()

                    # Advantages
                    adv = (rewards - rewards.mean(dim=1, keepdim=True)) / rewards.std(
                        dim=1, keepdim=True
                    ).clamp(min=1e-6)
                    adv_abs_mean = adv.abs().mean().item()

                    all_proj_stds.append(proj_std)
                    all_proj_means.append(proj_mean)
                    all_reward_stds.append(reward_std)
                    all_advantage_abs_means.append(adv_abs_mean)
                batch_count += 1

            avg_proj_std = sum(all_proj_stds) / len(all_proj_stds)
            avg_proj_mean = sum(all_proj_means) / len(all_proj_means)
            avg_reward_std = sum(all_reward_stds) / len(all_reward_stds)
            avg_adv_abs = sum(all_advantage_abs_means) / len(all_advantage_abs_means)
            # Signal-to-noise: how much of the projection is signal vs rollout noise
            proj_snr = avg_proj_mean / max(avg_proj_std, 1e-8)

            print(
                f"  T={T:3d}: "
                f"proj_std={avg_proj_std:.4f}  "
                f"proj_mean={avg_proj_mean:.4f}  "
                f"proj_snr={proj_snr:.1f}  "
                f"reward_std={avg_reward_std:.4f}  "
                f"|advantage|={avg_adv_abs:.4f}"
            )


if __name__ == "__main__":
    main()
