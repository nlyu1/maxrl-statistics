"""
Single SL training run on the bag-of-words task.

Usage:
    uv run python experiments/bow/sl-corr/single_run.py \\
        --corr 0.22 --seed 51 --device cuda:0

Artifacts → artifacts/bow-sl-sweep/seed-{S}/{dataset_name}/
"""

import sys

import click
import torch

from src import chdir_repo_base, get_repo_base

chdir_repo_base()
repo_root = get_repo_base()

from src.experiments.bag_of_words.sl import BagOfWordsSLConfig  # noqa: E402
from src.experiments.utils import cleanup_cuda, prepare_study_folder, set_seeds  # noqa: E402

STUDY_BASE = repo_root / "artifacts" / "bow-sl-sweep"
DATA_BASE = repo_root / "artifacts" / "bow-data"
AUX_WORDS_RATIO = 0.5


@click.command()
@click.option("--corr", type=float, required=True)
@click.option("--seed", type=int, required=True)
@click.option("--device", type=str, required=True)
@click.option("--train-epochs", type=int, default=20, show_default=True)
def main(
    corr: float,
    seed: int,
    device: str,
    train_epochs: int,
) -> None:
    set_seeds(seed)

    study_base = STUDY_BASE / f"seed-{seed}"
    torch_device = torch.device(device)

    config = BagOfWordsSLConfig.get_canonical(
        dataset_base_folder=DATA_BASE,
        study_base_folder=study_base,
        corr=corr,
        aux_words_ratio=AUX_WORDS_RATIO,
        train_epochs=train_epochs,
    )
    tag = f"seed={seed} corr={corr:.4f}"
    study_folder = config.study_folder

    if not prepare_study_folder(study_folder=study_folder, tag=tag):
        sys.exit(0)

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
