"""
Single MaxRL training run on the bag-of-words task.

Usage:
    uv run python experiments/bow/maxrl-corr/single_run.py \
        --corr 0.22 --num-rollouts 64 --seed 51 --device cuda:0 \
        --subtract-baseline

Artifacts -> artifacts/bow-maxrl-sweep/seed-{S}/rollouts-{N}/{baseline_mode}/{dataset_name}/
"""

import gc
import random
import shutil
import sys

import click
import numpy as np
import torch

from src import chdir_repo_base, get_repo_base

chdir_repo_base()
repo_root = get_repo_base()

from src.experiments.bag_of_words.analysis import BagOfWordsAnalysisConfig  # noqa: E402
from src.experiments.bag_of_words.maxrl import BagOfWordsMaxRLConfig  # noqa: E402

STUDY_BASE = repo_root / "artifacts" / "bow-maxrl-sweep"
DATA_BASE = repo_root / "artifacts" / "bow-data"
AUX_WORDS_RATIO = 0.5
GAUSSIAN_STDEV = 1.0


def baseline_mode_folder(*, subtract_baseline: bool) -> str:
    if subtract_baseline:
        return "subtract-baseline"
    return "no-subtract-baseline"


def cleanup_cuda(device: torch.device) -> None:
    gc.collect()
    if device.type != "cuda":
        return
    with torch.cuda.device(device):
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    torch.compiler.reset()


@click.command()
@click.option("--corr", type=float, required=True)
@click.option("--num-rollouts", type=int, required=True)
@click.option("--seed", type=int, required=True)
@click.option("--device", type=str, required=True)
@click.option("--train-epochs", type=int, default=20, show_default=True)
@click.option("--subtract-baseline/--no-subtract-baseline", required=True)
def main(
    corr: float,
    num_rollouts: int,
    seed: int,
    device: str,
    train_epochs: int,
    subtract_baseline: bool,
) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    baseline_mode = baseline_mode_folder(subtract_baseline=subtract_baseline)
    study_base = (
        STUDY_BASE / f"seed-{seed}" / f"rollouts-{num_rollouts}" / baseline_mode
    )
    torch_device = torch.device(device)

    config = BagOfWordsMaxRLConfig.get_canonical(
        dataset_base_folder=DATA_BASE,
        study_base_folder=study_base,
        corr=corr,
        aux_words_ratio=AUX_WORDS_RATIO,
        train_epochs=train_epochs,
        num_rollouts_per_sample=num_rollouts,
        gaussian_stdev=GAUSSIAN_STDEV,
        subtract_baseline=subtract_baseline,
    )
    tag = (
        f"seed={seed} corr={corr:.4f} rollouts={num_rollouts} "
        f"baseline={baseline_mode}"
    )
    study_folder = config.study_folder

    if BagOfWordsAnalysisConfig.is_study_complete(study_folder):
        print(f"!!! {tag} already complete, skipping ({study_folder})")
        sys.exit(0)
    if BagOfWordsAnalysisConfig.has_study_started(study_folder):
        print(f"=== {tag} partial -> wiping {study_folder} ===")
        shutil.rmtree(study_folder)

    print(f"=== {tag}  folder={study_folder} ===")
    state = None
    try:
        state = config.initialize(device=torch_device)
        state.run_training()
    finally:
        del state
        cleanup_cuda(torch_device)


if __name__ == "__main__":
    main()
