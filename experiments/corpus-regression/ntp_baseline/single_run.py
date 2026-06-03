"""
NTP baseline: intrinsic variance proxy for corpus-regression.

Usage:
    uv run python experiments/corpus-regression/ntp_baseline/single_run.py \
        --num-lookforward-tokens 1 --device cuda:0

Artifacts -> artifacts/corpus-regression/artifacts/ntp_baseline/{dataset_name}/
This is inference-only (no training). Runs in under 1 minute on GPU.
"""

import sys

import click
import torch

from src import chdir_repo_base

chdir_repo_base()

from src.experiments.corpus_regression.config import (  # noqa: E402
    CorpusRegressionStudyBaseConfig,
    artifacts_dir,
    data_dir,
)
from src.experiments.corpus_regression.ntp_baseline import (  # noqa: E402
    NTPBaselineConfig,
    run_ntp_baseline,
)

METHOD = "ntp_baseline"


@click.command()
@click.option("--num-lookforward-tokens", type=int, default=1, show_default=True)
@click.option("--device", type=str, required=True)
@click.option("--num-samples", type=int, default=100_000, show_default=True)
@click.option("--eval-batch-size", type=int, default=128, show_default=True)
@click.option("--label-type", type=click.Choice(["rademacher", "token_id"]), default="rademacher", show_default=True)
@click.option("--normalize-labels", is_flag=True, default=False, show_default=True)
@click.option("--label-range", type=(float, float), default=(0.0, 1.0), show_default=True, help="Target (lo, hi) range for label normalization.")
def main(
    num_lookforward_tokens: int,
    device: str,
    num_samples: int,
    eval_batch_size: int,
    label_type: str,
    normalize_labels: bool,
    label_range: tuple[float, float],
) -> None:
    torch_device = torch.device(device)
    study_base = artifacts_dir() / METHOD

    config = NTPBaselineConfig.get_canonical(
        dataset_base_folder=data_dir(),
        study_base_folder=study_base,
        num_lookforward_tokens=num_lookforward_tokens,
        num_samples=num_samples,
        eval_batch_size=eval_batch_size,
        label_type=label_type,
        normalize_labels=normalize_labels,
        label_range=label_range,
    )

    tag = f"ntp_baseline look={num_lookforward_tokens}"
    study_folder = config.study_folder

    # Skip if already complete.
    if (study_folder / "val_metrics.parquet").exists():
        print(f"!!! {tag} already complete, skipping ({study_folder})")
        sys.exit(0)

    print(f"=== {tag}  folder={study_folder} ===")
    run_ntp_baseline(config, device=torch_device)


if __name__ == "__main__":
    main()
