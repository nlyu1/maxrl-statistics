"""
Single SL+NTP-CE training run on the corpus-regression task.

Variant of `experiments/corpus-regression/sl/single_run.py` that uses
cross-entropy on the lookahead token id (K=1) instead of MSE on the
Rademacher embedding. Validation projects `softmax(logits) @ label_projector`
to produce predictions in the same target space, so the resulting `mse`
metric is directly comparable to SL.

Usage:
    uv run python experiments/corpus-regression/sl_ce/single_run.py \\
        --num-lookforward-tokens 1 --seed 51 --device cuda:0

Artifacts -> artifacts/corpus-regression/artifacts/sl_ce/from_pretrain/seed-{S}/{dataset_name}/
(or .../sl_ce/from_scratch/seed-{S}/... when --train-from-scratch is set).
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
)
from src.experiments.corpus_regression.sl_ce import (  # noqa: E402
    CorpusRegressionSLCEConfig,
)
from src.experiments.utils import cleanup_cuda, set_seeds  # noqa: E402

METHOD = "sl_ce"


@click.command()
@click.option("--num-lookforward-tokens", type=int, required=True)
@click.option("--seed", type=int, required=True)
@click.option("--device", type=str, required=True)
@click.option("--train-steps", type=int, default=10_000, show_default=True)
@click.option("--val-every-n-steps", type=int, default=2000, show_default=True)
@click.option("--num-samples", type=int, default=100_000, show_default=True)
@click.option("--lr-per-sample", type=float, default=1e-5, show_default=True, help="Per-sample learning rate; final lr = lr_per_sample × batch_size / divisor.")
@click.option(
    "--lr-schedule",
    type=click.Choice(["flat", "cosine"]),
    default="flat",
    show_default=True,
    help=(
        "LR schedule shape. 'flat' = no scheduler (today's behaviour). "
        "'cosine' = linear warmup then half-cosine decay to "
        "lr_min_ratio × peak."
    ),
)
@click.option(
    "--warmup-ratio",
    type=float,
    default=0.05,
    show_default=True,
    help="Warmup duration as a fraction of train_steps. Ignored when --lr-schedule flat.",
)
@click.option(
    "--lr-min-ratio",
    type=float,
    default=0.1,
    show_default=True,
    help="Cosine end LR as a fraction of peak LR. Ignored when --lr-schedule flat.",
)
@click.option(
    "--label-type",
    type=click.Choice(["rademacher", "token_id"]),
    default="rademacher",
    show_default=True,
)
@click.option("--normalize-labels", is_flag=True, default=False, show_default=True)
@click.option(
    "--label-range",
    type=(float, float),
    default=(0.0, 1.0),
    show_default=True,
    help="Target (lo, hi) range for label normalization.",
)
@click.option(
    "--train-from-scratch",
    is_flag=True,
    default=False,
    show_default=True,
    help="Random-init the backbone instead of loading pretrained weights.",
)
def main(
    num_lookforward_tokens: int,
    seed: int,
    device: str,
    train_steps: int,
    val_every_n_steps: int,
    num_samples: int,
    lr_per_sample: float,
    lr_schedule: str,
    warmup_ratio: float,
    lr_min_ratio: float,
    label_type: str,
    normalize_labels: bool,
    label_range: tuple[float, float],
    train_from_scratch: bool,
) -> None:
    if num_lookforward_tokens != 1:
        # CE on a token id only makes sense at the next-token horizon. For
        # K>1 the dataset stores only the K-th-ahead token id, but the CE
        # objective at the model's last position predicts position
        # prefix_length, not prefix_length+K-1, so the two would silently
        # diverge. Hard-fail here rather than producing misleading metrics.
        raise click.BadParameter(
            f"sl_ce only supports num-lookforward-tokens=1 (got {num_lookforward_tokens}).",
            param_hint="--num-lookforward-tokens",
        )

    set_seeds(seed)

    method_dir = Path(METHOD) / ("from_scratch" if train_from_scratch else "from_pretrain")
    study_base = artifacts_dir() / method_dir / f"seed-{seed}"
    torch_device = torch.device(device)

    config = CorpusRegressionSLCEConfig.get_canonical(
        dataset_base_folder=data_dir(),
        study_base_folder=study_base,
        num_lookforward_tokens=num_lookforward_tokens,
        train_steps=train_steps,
        val_every_n_steps=val_every_n_steps,
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
    tag = f"seed={seed} look={num_lookforward_tokens}"
    study_folder = config.study_folder

    if not CorpusRegressionSLCEConfig.prepare_study_folder(
        study_folder=study_folder,
        tag=tag,
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
