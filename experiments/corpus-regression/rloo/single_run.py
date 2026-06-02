"""
Single RLOO training run on the corpus-regression task.

Usage:
    uv run python experiments/corpus-regression/rloo/single_run.py \\
        --num-lookforward-tokens 4 --num-rollouts 64 --seed 51 --device cuda:0 \\
        --factorized True

Artifacts ->
    artifacts/corpus-regression/artifacts/rloo/seed-{S}/rollouts-{N}/
    {factorized_mode}/{dataset_name}/
where {dataset_name} encodes num_lookforward_tokens (e.g. ..._look4_dim32).
"""

import sys

import click
import torch

from src import chdir_repo_base

chdir_repo_base()

from src.experiments.corpus_regression.config import (  # noqa: E402
    artifacts_dir,
    data_dir,
    factorized_mode_folder,
    sigma_folder,
)
from src.experiments.corpus_regression.rloo import CorpusRegressionRLOOConfig  # noqa: E402
from src.experiments.utils import cleanup_cuda, set_seeds  # noqa: E402

METHOD = "rloo"


@click.command()
@click.option("--num-lookforward-tokens", type=int, required=True)
@click.option("--num-rollouts", type=int, required=True)
@click.option("--seed", type=int, required=True)
@click.option("--device", type=str, required=True)
@click.option("--factorized", type=bool, required=True)
@click.option("--train-epochs", type=int, default=5, show_default=True)
@click.option("--num-samples", type=int, default=100_000, show_default=True)
@click.option("--gaussian-stdev", type=float, default=1.0, show_default=True)
@click.option("--label-type", type=click.Choice(["rademacher", "token_id"]), default="rademacher", show_default=True)
@click.option("--normalize-labels", is_flag=True, default=False, show_default=True)
@click.option("--label-range", type=(float, float), default=(0.0, 1.0), show_default=True, help="Target (lo, hi) range for label normalization.")
def main(
    num_lookforward_tokens: int,
    num_rollouts: int,
    seed: int,
    device: str,
    factorized: bool,
    train_epochs: int,
    num_samples: int,
    gaussian_stdev: float,
    label_type: str,
    normalize_labels: bool,
    label_range: tuple[float, float],
) -> None:
    set_seeds(seed)

    factorized_mode = factorized_mode_folder(factorized=factorized)
    study_base = (
        artifacts_dir()
        / METHOD
        / f"seed-{seed}"
        / f"rollouts-{num_rollouts}"
        / sigma_folder(gaussian_stdev=gaussian_stdev)
        / factorized_mode
    )
    torch_device = torch.device(device)

    config = CorpusRegressionRLOOConfig.get_canonical(
        dataset_base_folder=data_dir(),
        study_base_folder=study_base,
        num_lookforward_tokens=num_lookforward_tokens,
        train_epochs=train_epochs,
        num_rollouts_per_sample=num_rollouts,
        gaussian_stdev=gaussian_stdev,
        factorized=factorized,
        num_samples=num_samples,
        label_type=label_type,
        normalize_labels=normalize_labels,
        label_range=label_range,
    )
    tag = (
        f"seed={seed} look={num_lookforward_tokens} rollouts={num_rollouts} "
        f"factorized={factorized_mode}"
    )
    study_folder = config.study_folder

    if not CorpusRegressionRLOOConfig.prepare_study_folder(
        study_folder=study_folder, tag=tag,
    ):
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
