"""
Single GRPO training run on the corpus-regression task.

Usage:
    uv run python experiments/corpus-regression/grpo/single_run.py \\
        --num-lookforward-tokens 4 --num-rollouts 64 --seed 51 --device cuda:0

Artifacts ->
    artifacts/corpus-regression/artifacts/grpo/seed-{S}/rollouts-{N}/{dataset_name}/
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
)
from src.experiments.corpus_regression.grpo import CorpusRegressionGRPOConfig  # noqa: E402
from src.experiments.utils import cleanup_cuda, set_seeds  # noqa: E402

METHOD = "grpo"
GAUSSIAN_STDEV = 1.0


@click.command()
@click.option("--num-lookforward-tokens", type=int, required=True)
@click.option("--num-rollouts", type=int, required=True)
@click.option("--seed", type=int, required=True)
@click.option("--device", type=str, required=True)
@click.option("--train-epochs", type=int, default=5, show_default=True)
def main(
    num_lookforward_tokens: int,
    num_rollouts: int,
    seed: int,
    device: str,
    train_epochs: int,
) -> None:
    set_seeds(seed)

    study_base = (
        artifacts_dir() / METHOD / f"seed-{seed}" / f"rollouts-{num_rollouts}"
    )
    torch_device = torch.device(device)

    config = CorpusRegressionGRPOConfig.get_canonical(
        dataset_base_folder=data_dir(),
        study_base_folder=study_base,
        num_lookforward_tokens=num_lookforward_tokens,
        train_epochs=train_epochs,
        num_rollouts_per_sample=num_rollouts,
        gaussian_stdev=GAUSSIAN_STDEV,
    )
    tag = f"seed={seed} look={num_lookforward_tokens} rollouts={num_rollouts}"
    study_folder = config.study_folder

    if not CorpusRegressionGRPOConfig.prepare_study_folder(
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
