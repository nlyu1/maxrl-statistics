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
from pathlib import Path

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
@click.option("--train-steps", type=int, default=10_000, show_default=True)
@click.option("--val-every-n-steps", type=int, default=2000, show_default=True)
@click.option("--num-samples", type=int, default=100_000, show_default=True)
@click.option("--gaussian-stdev", type=float, default=1.0, show_default=True)
@click.option("--lr-per-sample", type=float, default=1e-5, show_default=True, help="Per-sample learning rate; final head_lr = lr_per_sample × batch_size.")
@click.option("--lr-schedule", type=click.Choice(["flat", "cosine"]), default="flat", show_default=True, help="LR schedule shape. 'flat' = no scheduler (today's behaviour). 'cosine' = linear warmup then half-cosine decay to lr_min_ratio × peak.")
@click.option("--warmup-ratio", type=float, default=0.05, show_default=True, help="Warmup duration as a fraction of train_steps. Ignored when --lr-schedule flat.")
@click.option("--lr-min-ratio", type=float, default=0.1, show_default=True, help="Cosine end LR as a fraction of peak LR. Ignored when --lr-schedule flat.")
@click.option("--label-type", type=click.Choice(["rademacher", "token_id"]), default="rademacher", show_default=True)
@click.option("--normalize-labels", is_flag=True, default=False, show_default=True)
@click.option("--label-range", type=(float, float), default=(0.0, 1.0), show_default=True, help="Target (lo, hi) range for label normalization.")
@click.option("--train-from-scratch", is_flag=True, default=False, show_default=True, help="Random-init the backbone instead of loading pretrained weights.")
def main(
    num_lookforward_tokens: int,
    num_rollouts: int,
    seed: int,
    device: str,
    factorized: bool,
    train_steps: int,
    val_every_n_steps: int,
    num_samples: int,
    gaussian_stdev: float,
    lr_per_sample: float,
    lr_schedule: str,
    warmup_ratio: float,
    lr_min_ratio: float,
    label_type: str,
    normalize_labels: bool,
    label_range: tuple[float, float],
    train_from_scratch: bool,
) -> None:
    set_seeds(seed)

    factorized_mode = factorized_mode_folder(factorized=factorized)
    method_dir = Path(METHOD) / ("from_scratch" if train_from_scratch else "from_pretrain")
    study_base = (
        artifacts_dir()
        / method_dir
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
        train_steps=train_steps,
        val_every_n_steps=val_every_n_steps,
        num_rollouts_per_sample=num_rollouts,
        gaussian_stdev=gaussian_stdev,
        factorized=factorized,
        num_samples=num_samples,
        lr_per_sample=lr_per_sample,
        lr_schedule=lr_schedule,
        warmup_ratio=warmup_ratio,
        lr_min_ratio=lr_min_ratio,
        label_type=label_type,
        normalize_labels=normalize_labels,
        label_range=label_range,
        train_from_scratch=train_from_scratch,
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
