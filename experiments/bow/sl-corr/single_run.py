"""
Single SL training run on the bag-of-words task.

Usage:
    uv run python experiments/bow/sl-corr/single_run.py \\
        --dataset homoskedastic --corr 0.22 --seed 51 --device cuda:0

Artifacts → artifacts/bow-sl-{hom,row-het,word-het}-sweep/seed-{S}/{dataset_name}/
"""

import sys

import click
import torch

from src import chdir_repo_base, get_repo_base

chdir_repo_base()
repo_root = get_repo_base()

from src.experiments.bag_of_words.config import sweep_root_name  # noqa: E402
from src.experiments.bag_of_words.sl import BagOfWordsSLConfig  # noqa: E402
from src.experiments.utils import cleanup_cuda, prepare_study_folder, set_seeds  # noqa: E402

ARTIFACTS_ROOT = repo_root / "artifacts"
DATA_BASE = ARTIFACTS_ROOT / "bow-data"
METHOD = "sl"
AUX_WORDS_RATIO = 0.5
DATASET_CHOICES = ("homoskedastic", "row_heteroskedastic", "word_heteroskedastic")


@click.command()
@click.option("--dataset", type=click.Choice(DATASET_CHOICES), required=True)
@click.option("--corr", type=float, required=True)
@click.option("--seed", type=int, required=True)
@click.option("--device", type=str, required=True)
@click.option("--train-epochs", type=int, default=20, show_default=True)
def main(
    dataset: str,
    corr: float,
    seed: int,
    device: str,
    train_epochs: int,
) -> None:
    set_seeds(seed)

    study_base = (
        ARTIFACTS_ROOT
        / sweep_root_name(method=METHOD, dataset=dataset)
        / f"seed-{seed}"
    )
    torch_device = torch.device(device)

    config = BagOfWordsSLConfig.get_canonical(
        dataset=dataset,
        dataset_base_folder=DATA_BASE,
        study_base_folder=study_base,
        corr=corr,
        aux_words_ratio=AUX_WORDS_RATIO,
        train_epochs=train_epochs,
    )
    tag = f"dataset={dataset} seed={seed} corr={corr:.4f}"
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
