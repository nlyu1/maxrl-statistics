"""
Single MaxRL training run on the corpus-regression task.

Usage:
    uv run python experiments/corpus-regression/maxrl/single_run.py \\
        --num-lookforward-tokens 4 --num-rollouts 64 --seed 51 --device cuda:0 \\
        --subtract-baseline True --use-factorized-likelihoods True

Artifacts ->
    artifacts/corpus-regression/artifacts/maxrl/seed-{S}/rollouts-{N}/
    {baseline_mode}/{likelihood_mode}/{dataset_name}/
where {dataset_name} encodes num_lookforward_tokens (e.g. ..._look4_dim32).
"""

import sys

import click
import torch

from src import chdir_repo_base

chdir_repo_base()

from src.experiments.corpus_regression.config import (  # noqa: E402
    artifacts_dir,
    baseline_mode_folder,
    data_dir,
    likelihood_mode_folder,
    sigma_folder,
)
from src.experiments.corpus_regression.maxrl import CorpusRegressionMaxRLConfig  # noqa: E402
from src.experiments.utils import cleanup_cuda, set_seeds  # noqa: E402

METHOD = "maxrl"


@click.command()
@click.option("--num-lookforward-tokens", type=int, required=True)
@click.option("--num-rollouts", type=int, required=True)
@click.option("--seed", type=int, required=True)
@click.option("--device", type=str, required=True)
@click.option("--subtract-baseline", type=bool, required=True)
@click.option("--use-factorized-likelihoods", type=bool, required=True)
@click.option("--train-epochs", type=int, default=5, show_default=True)
@click.option("--num-samples", type=int, default=100_000, show_default=True)
@click.option("--gaussian-stdev", type=float, default=1.0, show_default=True)
def main(
    num_lookforward_tokens: int,
    num_rollouts: int,
    seed: int,
    device: str,
    subtract_baseline: bool,
    use_factorized_likelihoods: bool,
    train_epochs: int,
    num_samples: int,
    gaussian_stdev: float,
) -> None:
    set_seeds(seed)

    baseline_mode = baseline_mode_folder(subtract_baseline=subtract_baseline)
    likelihood_mode = likelihood_mode_folder(
        use_factorized_likelihoods=use_factorized_likelihoods,
    )
    study_base = (
        artifacts_dir()
        / METHOD
        / f"seed-{seed}"
        / f"rollouts-{num_rollouts}"
        / sigma_folder(gaussian_stdev=gaussian_stdev)
        / baseline_mode
        / likelihood_mode
    )
    torch_device = torch.device(device)

    config = CorpusRegressionMaxRLConfig.get_canonical(
        dataset_base_folder=data_dir(),
        study_base_folder=study_base,
        num_lookforward_tokens=num_lookforward_tokens,
        train_epochs=train_epochs,
        num_rollouts_per_sample=num_rollouts,
        gaussian_stdev=gaussian_stdev,
        subtract_baseline=subtract_baseline,
        use_factorized_likelihoods=use_factorized_likelihoods,
        num_samples=num_samples,
    )
    tag = (
        f"seed={seed} look={num_lookforward_tokens} rollouts={num_rollouts} "
        f"baseline={baseline_mode} likelihood={likelihood_mode}"
    )
    study_folder = config.study_folder

    if not CorpusRegressionMaxRLConfig.prepare_study_folder(
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
