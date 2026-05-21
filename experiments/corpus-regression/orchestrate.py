"""
Unified orchestrator for corpus-regression experiments.

Replaces the per-method orchestrate.py scripts with a single entry point
that supports configurable sweep grids and dynamic GPU scheduling.

Usage:
    uv run python experiments/corpus-regression/orchestrate.py \
        --method sl --seeds 51,61 --train-epochs 5

    uv run python experiments/corpus-regression/orchestrate.py \
        --method maxrl --seeds 51 \
        --lookforward-tokens 1,4,8 \
        --rollout-steps 16,1024 \
        --num-samples 50000 \
        --train-epochs 5 \
        --subtract-baseline --use-factorized-likelihoods

    uv run python experiments/corpus-regression/orchestrate.py \
        --method grpo --seeds 51 --gpu-ids 0,1,2,3 --dry-run

    uv run python experiments/corpus-regression/orchestrate.py \
        --method maxrl --seeds 51,61 --gaussian-stdev 0.5
"""

from __future__ import annotations

import itertools
import sys
from pathlib import Path
from typing import Literal

import click

from src import chdir_repo_base, get_repo_base

chdir_repo_base()
repo_root = get_repo_base()

from src.experiments.gpu_pool import GPUPool, Job  # noqa: E402

Method = Literal["sl", "grpo", "rloo", "maxrl"]

DEFAULT_LOOKFORWARD_TOKENS = (1, 2, 3, 4, 5, 6, 7, 8)
DEFAULT_ROLLOUT_STEPS = (4, 16, 128, 1024)
DEFAULT_NUM_SAMPLES = 100_000
DEFAULT_TRAIN_EPOCHS = 5
DEFAULT_GAUSSIAN_STDEV = 1.0


# ─── Custom Click parameter types ─────────────────────────────────────────────


class IntList(click.ParamType):
    """Parse comma-separated integers: '1,4,8' -> (1, 4, 8)."""

    name = "int_list"

    def convert(
        self,
        value: str | tuple[int, ...],
        param: click.Parameter | None,
        ctx: click.Context | None,
    ) -> tuple[int, ...]:
        if isinstance(value, tuple):
            return value
        try:
            return tuple(int(x.strip()) for x in value.split(","))
        except ValueError:
            self.fail(
                f"'{value}' is not a valid comma-separated list of integers",
                param,
                ctx,
            )


class FloatList(click.ParamType):
    """Parse comma-separated floats: '0.5,1.0,2.0' -> (0.5, 1.0, 2.0)."""

    name = "float_list"

    def convert(
        self,
        value: str | tuple[float, ...],
        param: click.Parameter | None,
        ctx: click.Context | None,
    ) -> tuple[float, ...]:
        if isinstance(value, tuple):
            return value
        try:
            return tuple(float(x.strip()) for x in value.split(","))
        except ValueError:
            self.fail(
                f"'{value}' is not a valid comma-separated list of floats",
                param,
                ctx,
            )


INT_LIST = IntList()
FLOAT_LIST = FloatList()


# ─── Job generation ───────────────────────────────────────────────────────────


def _script_path(method: str) -> Path:
    return (
        repo_root / "experiments" / "corpus-regression" / method / "single_run.py"
    )


def _build_sl_jobs(
    *,
    seeds: tuple[int, ...],
    lookforward_tokens: tuple[int, ...],
    num_samples: int,
    train_epochs: int,
    gaussian_stdev: float,
) -> list[Job]:
    script = str(_script_path("sl"))
    jobs: list[Job] = []
    for seed, lft in itertools.product(seeds, lookforward_tokens):
        cmd = [
            sys.executable, script,
            "--num-lookforward-tokens", str(lft),
            "--seed", str(seed),
            "--device", "{device}",
            "--train-epochs", str(train_epochs),
            "--num-samples", str(num_samples),
            "--gaussian-stdev", str(gaussian_stdev),
        ]
        jobs.append(Job(
            cmd_template=cmd,
            label=f"sl_seed-{seed}_look-{lft}_sigma-{gaussian_stdev}",
        ))
    return jobs


def _build_grpo_jobs(
    *,
    seeds: tuple[int, ...],
    lookforward_tokens: tuple[int, ...],
    rollout_steps: tuple[int, ...],
    num_samples: int,
    train_epochs: int,
    gaussian_stdev: float,
) -> list[Job]:
    script = str(_script_path("grpo"))
    jobs: list[Job] = []
    for seed, lft, rollouts in itertools.product(
        seeds, lookforward_tokens, rollout_steps,
    ):
        cmd = [
            sys.executable, script,
            "--num-lookforward-tokens", str(lft),
            "--num-rollouts", str(rollouts),
            "--seed", str(seed),
            "--device", "{device}",
            "--train-epochs", str(train_epochs),
            "--num-samples", str(num_samples),
            "--gaussian-stdev", str(gaussian_stdev),
        ]
        jobs.append(Job(
            cmd_template=cmd,
            label=f"grpo_seed-{seed}_look-{lft}_roll-{rollouts}_sigma-{gaussian_stdev}",
        ))
    return jobs


def _build_rloo_jobs(
    *,
    seeds: tuple[int, ...],
    lookforward_tokens: tuple[int, ...],
    rollout_steps: tuple[int, ...],
    num_samples: int,
    train_epochs: int,
    factorized: bool,
    gaussian_stdev: float,
) -> list[Job]:
    script = str(_script_path("rloo"))
    jobs: list[Job] = []
    for seed, lft, rollouts in itertools.product(
        seeds, lookforward_tokens, rollout_steps,
    ):
        cmd = [
            sys.executable, script,
            "--num-lookforward-tokens", str(lft),
            "--num-rollouts", str(rollouts),
            "--seed", str(seed),
            "--device", "{device}",
            "--train-epochs", str(train_epochs),
            "--num-samples", str(num_samples),
            "--factorized", str(factorized),
            "--gaussian-stdev", str(gaussian_stdev),
        ]
        jobs.append(Job(
            cmd_template=cmd,
            label=f"rloo_seed-{seed}_look-{lft}_roll-{rollouts}_fact-{factorized}_sigma-{gaussian_stdev}",
        ))
    return jobs


def _build_maxrl_jobs(
    *,
    seeds: tuple[int, ...],
    lookforward_tokens: tuple[int, ...],
    rollout_steps: tuple[int, ...],
    num_samples: int,
    train_epochs: int,
    subtract_baseline: bool,
    use_factorized_likelihoods: bool,
    gaussian_stdev: float,
) -> list[Job]:
    script = str(_script_path("maxrl"))
    jobs: list[Job] = []
    for seed, lft, rollouts in itertools.product(
        seeds, lookforward_tokens, rollout_steps,
    ):
        cmd = [
            sys.executable, script,
            "--num-lookforward-tokens", str(lft),
            "--num-rollouts", str(rollouts),
            "--seed", str(seed),
            "--device", "{device}",
            "--train-epochs", str(train_epochs),
            "--num-samples", str(num_samples),
            "--subtract-baseline", str(subtract_baseline),
            "--use-factorized-likelihoods", str(use_factorized_likelihoods),
            "--gaussian-stdev", str(gaussian_stdev),
        ]
        jobs.append(Job(
            cmd_template=cmd,
            label=(
                f"maxrl_seed-{seed}_look-{lft}_roll-{rollouts}"
                f"_bl-{subtract_baseline}_fact-{use_factorized_likelihoods}_sigma-{gaussian_stdev}"
            ),
        ))
    return jobs


# ─── CLI ──────────────────────────────────────────────────────────────────────


@click.command()
@click.option(
    "--method",
    type=click.Choice(["sl", "grpo", "rloo", "maxrl"]),
    required=True,
    help="Training method to run.",
)
@click.option(
    "--seeds",
    type=INT_LIST,
    required=True,
    help="Comma-separated random seeds (e.g. '51,61,121').",
)
@click.option(
    "--lookforward-tokens",
    type=INT_LIST,
    default=",".join(str(x) for x in DEFAULT_LOOKFORWARD_TOKENS),
    show_default=True,
    help="Comma-separated lookforward token counts to sweep.",
)
@click.option(
    "--rollout-steps",
    type=INT_LIST,
    default=",".join(str(x) for x in DEFAULT_ROLLOUT_STEPS),
    show_default=True,
    help="Comma-separated rollout counts (ignored for SL).",
)
@click.option(
    "--num-samples",
    type=int,
    default=DEFAULT_NUM_SAMPLES,
    show_default=True,
    help="Total training samples for the dataset.",
)
@click.option(
    "--train-epochs",
    type=int,
    default=DEFAULT_TRAIN_EPOCHS,
    show_default=True,
    help="Number of training epochs per run.",
)
@click.option(
    "--gaussian-stdev",
    type=float,
    default=DEFAULT_GAUSSIAN_STDEV,
    show_default=True,
    help="Standard deviation for Gaussian policy rollouts.",
)
@click.option(
    "--gpu-ids",
    type=INT_LIST,
    default=None,
    help="Comma-separated GPU device IDs (default: all visible).",
)
@click.option(
    "--subtract-baseline/--no-subtract-baseline",
    default=True,
    show_default=True,
    help="MaxRL: subtract baseline from score weights.",
)
@click.option(
    "--use-factorized-likelihoods/--no-use-factorized-likelihoods",
    default=True,
    show_default=True,
    help="MaxRL: use per-coordinate factorized likelihoods.",
)
@click.option(
    "--factorized/--no-factorized",
    default=True,
    show_default=True,
    help="RLOO: use per-coordinate factorized advantages.",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Print the job list and exit without running.",
)
@click.option(
    "--fail-fast",
    is_flag=True,
    help="Abort remaining jobs on first failure (default: continue all).",
)
def main(
    method: str,
    seeds: tuple[int, ...],
    lookforward_tokens: tuple[int, ...],
    rollout_steps: tuple[int, ...],
    num_samples: int,
    train_epochs: int,
    gaussian_stdev: float,
    gpu_ids: tuple[int, ...] | None,
    subtract_baseline: bool,
    use_factorized_likelihoods: bool,
    factorized: bool,
    dry_run: bool,
    fail_fast: bool,
) -> None:
    # Validate gaussian_stdev
    if gaussian_stdev <= 0.0:
        raise click.BadParameter(
            f"gaussian_stdev must be positive, got {gaussian_stdev}",
            param_hint="--gaussian-stdev",
        )

    # Build job list based on method.
    common = dict(
        seeds=seeds,
        lookforward_tokens=lookforward_tokens,
        num_samples=num_samples,
        train_epochs=train_epochs,
        gaussian_stdev=gaussian_stdev,
    )

    if method == "sl":
        jobs = _build_sl_jobs(**common)
    elif method == "grpo":
        jobs = _build_grpo_jobs(**common, rollout_steps=rollout_steps)
    elif method == "rloo":
        jobs = _build_rloo_jobs(
            **common, rollout_steps=rollout_steps, factorized=factorized,
        )
    elif method == "maxrl":
        jobs = _build_maxrl_jobs(
            **common,
            rollout_steps=rollout_steps,
            subtract_baseline=subtract_baseline,
            use_factorized_likelihoods=use_factorized_likelihoods,
        )
    else:
        raise click.BadParameter(f"Unknown method: {method}")

    # Summary.
    pool = GPUPool(device_ids=list(gpu_ids) if gpu_ids else None)
    click.echo(f"method={method}  seeds={seeds}  lookforward_tokens={lookforward_tokens}")
    if method != "sl":
        click.echo(f"rollout_steps={rollout_steps}")
    click.echo(
        f"num_samples={num_samples}  train_epochs={train_epochs}  "
        f"gaussian_stdev={gaussian_stdev}  gpus={pool.num_devices}  total_jobs={len(jobs)}"
    )

    if dry_run:
        click.echo("\nJob list:")
        for i, job in enumerate(jobs):
            click.echo(f"  [{i:3d}] {job.label}")
        return

    # Dispatch.
    from src.experiments.corpus_regression.config import artifacts_dir  # noqa: E402

    log_dir = artifacts_dir() / method / "logs"
    results = pool.run(jobs, log_dir=log_dir, fail_fast=fail_fast)

    # Report.
    failures = [r for r in results if r.returncode != 0]
    succeeded = len(results) - len(failures)
    click.echo(f"\n===== {method} complete: {succeeded}/{len(jobs)} succeeded =====")
    if failures:
        click.echo("Failures:")
        for r in failures:
            click.echo(f"  [{r.device}] {r.job.label} — exit {r.returncode}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
