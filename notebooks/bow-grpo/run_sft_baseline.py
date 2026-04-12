"""
SFT baseline at SNR=1.0 for comparison with GRPO.

Usage:
    uv run python notebooks/bow-grpo/run_sft_baseline.py
"""

import os
from pathlib import Path

import torch

repo_root = Path(__file__).resolve().parents[2]
os.chdir(repo_root)

from src.experiments.bag_of_words.sft import BagOfWordsSFTConfig  # noqa: E402

DEVICE = torch.device("cuda:1")
DATA_BASE = repo_root / "artifacts" / "bow-data"
STUDY_BASE = repo_root / "artifacts" / "bow-grpo"

config = BagOfWordsSFTConfig.get_canonical(
    dataset_base_folder=DATA_BASE,
    study_base_folder=STUDY_BASE / "sft-baseline",
    snr=1.0,
    num_words=15,
    num_samples=50_000,
    aux_words_ratio=0.5,
    prompt_length=128,
    filter_samples_above_n_tokens=384,
    word_decay_power=1.0,
    batch_size=64,
    eval_batch_size_multiple=4,
    lr_per_token=1.25e-7,
    backbone_lr_divisor=5.0,
    pad_to_multiple=8,
    train_epochs=10,
)

print(f"Study folder: {config.study_folder}")
print(f"Dataset R^2 target: {config.data.rsq:.4f}")
print(f"Backbone lr: {config.optimizer.lr:.3e}")
print(f"Head lr:     {config.optimizer.head_lr:.3e}")

state = config.initialize(device=DEVICE)
state.run_training()
print("SFT baseline done.")
