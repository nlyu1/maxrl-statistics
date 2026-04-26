"""
Single SL training run on the corpus-regression task.

Usage:
    uv run python experiments/corpus-regression/sl/single_run.py \\
        --num-lookforward-tokens 4 --seed 51 --device cuda:0

Artifacts -> artifacts/corpus-regression/artifacts/sl/seed-{S}/{dataset_name}/
where {dataset_name} encodes num_lookforward_tokens (e.g. ..._look4_dim32).
"""

import sys

import click
import torch

from src import chdir_repo_base, get_repo_base

chdir_repo_base()
repo_root = get_repo_base()

from src.experiments.corpus_regression.sl import CorpusRegressionSLConfig  # noqa: E402
from src.experiments.utils import cleanup_cuda, set_seeds  # noqa: E402

PROJECT_ROOT = repo_root / "artifacts" / "corpus-regression"
DATA_BASE = PROJECT_ROOT
ARTIFACTS_BASE = PROJECT_ROOT / "artifacts"
METHOD = "sl"


@click.command()
@click.option("--num-lookforward-tokens", type=int, required=True)
@click.option("--seed", type=int, required=True)
@click.option("--device", type=str, required=True)
@click.option("--train-epochs", type=int, default=20, show_default=True)
def main(
    num_lookforward_tokens: int,
    seed: int,
    device: str,
    train_epochs: int,
) -> None:
    set_seeds(seed)

    study_base = ARTIFACTS_BASE / METHOD / f"seed-{seed}"
    torch_device = torch.device(device)

    config = CorpusRegressionSLConfig.get_canonical(
        dataset_base_folder=DATA_BASE,
        study_base_folder=study_base,
        num_lookforward_tokens=num_lookforward_tokens,
        train_epochs=train_epochs,
    )
    tag = f"seed={seed} look={num_lookforward_tokens}"
    study_folder = config.study_folder

    if not CorpusRegressionSLConfig.prepare_study_folder(
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
