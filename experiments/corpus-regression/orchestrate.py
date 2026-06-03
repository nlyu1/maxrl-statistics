"""
Unified orchestrator for corpus-regression experiments.

Replaces the per-method orchestrate.py scripts with a single entry point
that supports configurable sweep grids and dynamic GPU scheduling.

Usage:
    uv run python experiments/corpus-regression/orchestrate.py \
        --method sl --seeds 51,61 --train-steps 5

    uv run python experiments/corpus-regression/orchestrate.py \
        --method maxrl --seeds 51 \
        --lookforward-tokens 1,4,8 \
        --rollout-steps 16,1024 \
        --num-samples 50000,100000 \
        --train-steps 5 \
        --subtract-baseline --use-factorized-likelihoods

    uv run python experiments/corpus-regression/orchestrate.py \
        --method grpo --seeds 51 --gpu-ids 0,1,2,3 --dry-run

    uv run python experiments/corpus-regression/orchestrate.py \
        --method sl --method maxrl --seeds 51,61 \
        --gaussian-stdev 0.5,1.0 --num-samples 50000,100000
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

from src.experiments.corpus_regression.config import (  # noqa: E402
    artifacts_dir,
    baseline_mode_folder,
    factorized_mode_folder,
    likelihood_mode_folder,
    sigma_folder,
)
from src.experiments.gpu_pool import GPUPool, Job  # noqa: E402

Method = Literal["sl", "sl_ce", "grpo", "rloo", "maxrl", "ntp_baseline"]

DEFAULT_LOOKFORWARD_TOKENS = (1, 2, 3, 4, 5, 6, 7, 8)
DEFAULT_ROLLOUT_STEPS = (4, 16, 128, 1024)
DEFAULT_NUM_SAMPLES = 100_000
DEFAULT_TRAIN_STEPS = 10_000
DEFAULT_VAL_EVERY_N_STEPS = 2000
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


def _method_dir(method: str, *, train_from_scratch: bool) -> str:
    """Method-level artifact subfolder. From-scratch sweeps land in a parallel
    `<method>_scratch` tree alongside the pretrained-weight `<method>` tree."""
    return f"{method}_scratch" if train_from_scratch else method


def _build_sl_jobs(
    *,
    seeds: tuple[int, ...],
    lookforward_tokens: tuple[int, ...],
    num_samples_values: tuple[int, ...],
    train_steps: int,
    val_every_n_steps: int,
    label_type: str,
    normalize_labels: bool,
    label_range: tuple[float, float],
    train_from_scratch: bool,
) -> list[Job]:
    script = str(_script_path("sl"))
    method_dir = _method_dir("sl", train_from_scratch=train_from_scratch)
    jobs: list[Job] = []
    for seed, lft, ns in itertools.product(seeds, lookforward_tokens, num_samples_values):
        label = f"sl_seed-{seed}_look-{lft}_ns-{ns}_lbl-{label_type}"
        log_path = (
            artifacts_dir() / method_dir / "logs"
            / f"seed-{seed}"
            / f"look-{lft}_ns-{ns}.log"
        )
        if label_type != "rademacher":
            log_path = log_path.parent / label_type / log_path.name
        if normalize_labels:
            log_path = log_path.parent / "normalized" / log_path.name
        cmd = [
            sys.executable, script,
            "--num-lookforward-tokens", str(lft),
            "--seed", str(seed),
            "--device", "{device}",
            "--train-steps", str(train_steps),
            "--val-every-n-steps", str(val_every_n_steps),
            "--num-samples", str(ns),
            "--label-type", label_type,
        ]
        if normalize_labels:
            cmd.append("--normalize-labels")
            cmd.extend(["--label-range", str(label_range[0]), str(label_range[1])])
        if train_from_scratch:
            cmd.append("--train-from-scratch")
        jobs.append(Job(cmd_template=cmd, label=label, log_path=log_path))
    return jobs


def _build_sl_ce_jobs(
    *,
    seeds: tuple[int, ...],
    lookforward_tokens: tuple[int, ...],
    num_samples_values: tuple[int, ...],
    train_steps: int,
    val_every_n_steps: int,
    label_type: str,
    normalize_labels: bool,
    label_range: tuple[float, float],
    train_from_scratch: bool,
) -> list[Job]:
    """Mirror of `_build_sl_jobs` for the NTP-CE supervised variant.

    `sl_ce` only supports K=1 — `single_run.py` rejects other values — but
    we keep the same CLI surface as SL so the orchestrator's `--lookforward-
    tokens` flag behaves predictably; non-1 values will surface as an error
    inside the spawned process.
    """
    script = str(_script_path("sl_ce"))
    method_dir = _method_dir("sl_ce", train_from_scratch=train_from_scratch)
    jobs: list[Job] = []
    for seed, lft, ns in itertools.product(seeds, lookforward_tokens, num_samples_values):
        label = f"sl_ce_seed-{seed}_look-{lft}_ns-{ns}_lbl-{label_type}"
        log_path = (
            artifacts_dir() / method_dir / "logs"
            / f"seed-{seed}"
            / f"look-{lft}_ns-{ns}.log"
        )
        if label_type != "rademacher":
            log_path = log_path.parent / label_type / log_path.name
        if normalize_labels:
            log_path = log_path.parent / "normalized" / log_path.name
        cmd = [
            sys.executable, script,
            "--num-lookforward-tokens", str(lft),
            "--seed", str(seed),
            "--device", "{device}",
            "--train-steps", str(train_steps),
            "--val-every-n-steps", str(val_every_n_steps),
            "--num-samples", str(ns),
            "--label-type", label_type,
        ]
        if normalize_labels:
            cmd.append("--normalize-labels")
            cmd.extend(["--label-range", str(label_range[0]), str(label_range[1])])
        if train_from_scratch:
            cmd.append("--train-from-scratch")
        jobs.append(Job(cmd_template=cmd, label=label, log_path=log_path))
    return jobs


def _build_grpo_jobs(
    *,
    seeds: tuple[int, ...],
    lookforward_tokens: tuple[int, ...],
    rollout_steps: tuple[int, ...],
    num_samples_values: tuple[int, ...],
    train_steps: int,
    val_every_n_steps: int,
    gaussian_stdev_values: tuple[float, ...],
    label_type: str,
    normalize_labels: bool,
    label_range: tuple[float, float],
    train_from_scratch: bool,
) -> list[Job]:
    script = str(_script_path("grpo"))
    method_dir = _method_dir("grpo", train_from_scratch=train_from_scratch)
    jobs: list[Job] = []
    for seed, lft, rollouts, ns, stdev in itertools.product(
        seeds, lookforward_tokens, rollout_steps, num_samples_values, gaussian_stdev_values,
    ):
        label = f"grpo_seed-{seed}_look-{lft}_roll-{rollouts}_ns-{ns}_sigma-{stdev}_lbl-{label_type}"
        log_path = (
            artifacts_dir() / method_dir / "logs"
            / f"seed-{seed}"
            / f"rollouts-{rollouts}"
            / sigma_folder(gaussian_stdev=stdev)
            / f"look-{lft}_ns-{ns}.log"
        )
        if label_type != "rademacher":
            log_path = log_path.parent / label_type / log_path.name
        if normalize_labels:
            log_path = log_path.parent / "normalized" / log_path.name
        cmd = [
            sys.executable, script,
            "--num-lookforward-tokens", str(lft),
            "--num-rollouts", str(rollouts),
            "--seed", str(seed),
            "--device", "{device}",
            "--train-steps", str(train_steps),
            "--val-every-n-steps", str(val_every_n_steps),
            "--num-samples", str(ns),
            "--gaussian-stdev", str(stdev),
            "--label-type", label_type,
        ]
        if normalize_labels:
            cmd.append("--normalize-labels")
            cmd.extend(["--label-range", str(label_range[0]), str(label_range[1])])
        if train_from_scratch:
            cmd.append("--train-from-scratch")
        jobs.append(Job(cmd_template=cmd, label=label, log_path=log_path))
    return jobs


def _build_rloo_jobs(
    *,
    seeds: tuple[int, ...],
    lookforward_tokens: tuple[int, ...],
    rollout_steps: tuple[int, ...],
    num_samples_values: tuple[int, ...],
    train_steps: int,
    val_every_n_steps: int,
    factorized: bool,
    gaussian_stdev_values: tuple[float, ...],
    label_type: str,
    normalize_labels: bool,
    label_range: tuple[float, float],
    train_from_scratch: bool,
) -> list[Job]:
    script = str(_script_path("rloo"))
    method_dir = _method_dir("rloo", train_from_scratch=train_from_scratch)
    jobs: list[Job] = []
    for seed, lft, rollouts, ns, stdev in itertools.product(
        seeds, lookforward_tokens, rollout_steps, num_samples_values, gaussian_stdev_values,
    ):
        label = (
            f"rloo_seed-{seed}_look-{lft}_roll-{rollouts}"
            f"_ns-{ns}_fact-{factorized}_sigma-{stdev}_lbl-{label_type}"
        )
        log_path = (
            artifacts_dir() / method_dir / "logs"
            / f"seed-{seed}"
            / f"rollouts-{rollouts}"
            / sigma_folder(gaussian_stdev=stdev)
            / factorized_mode_folder(factorized=factorized)
            / f"look-{lft}_ns-{ns}.log"
        )
        if label_type != "rademacher":
            log_path = log_path.parent / label_type / log_path.name
        if normalize_labels:
            log_path = log_path.parent / "normalized" / log_path.name
        cmd = [
            sys.executable, script,
            "--num-lookforward-tokens", str(lft),
            "--num-rollouts", str(rollouts),
            "--seed", str(seed),
            "--device", "{device}",
            "--train-steps", str(train_steps),
            "--val-every-n-steps", str(val_every_n_steps),
            "--num-samples", str(ns),
            "--factorized", str(factorized),
            "--gaussian-stdev", str(stdev),
            "--label-type", label_type,
        ]
        if normalize_labels:
            cmd.append("--normalize-labels")
            cmd.extend(["--label-range", str(label_range[0]), str(label_range[1])])
        if train_from_scratch:
            cmd.append("--train-from-scratch")
        jobs.append(Job(cmd_template=cmd, label=label, log_path=log_path))
    return jobs


def _build_maxrl_jobs(
    *,
    seeds: tuple[int, ...],
    lookforward_tokens: tuple[int, ...],
    rollout_steps: tuple[int, ...],
    num_samples_values: tuple[int, ...],
    train_steps: int,
    val_every_n_steps: int,
    subtract_baseline: bool,
    use_factorized_likelihoods: bool,
    gaussian_stdev_values: tuple[float, ...],
    label_type: str,
    normalize_labels: bool,
    label_range: tuple[float, float],
    train_from_scratch: bool,
) -> list[Job]:
    script = str(_script_path("maxrl"))
    method_dir = _method_dir("maxrl", train_from_scratch=train_from_scratch)
    jobs: list[Job] = []
    for seed, lft, rollouts, ns, stdev in itertools.product(
        seeds, lookforward_tokens, rollout_steps, num_samples_values, gaussian_stdev_values,
    ):
        label = (
            f"maxrl_seed-{seed}_look-{lft}_roll-{rollouts}"
            f"_ns-{ns}_bl-{subtract_baseline}_fact-{use_factorized_likelihoods}_sigma-{stdev}_lbl-{label_type}"
        )
        log_path = (
            artifacts_dir() / method_dir / "logs"
            / f"seed-{seed}"
            / f"rollouts-{rollouts}"
            / sigma_folder(gaussian_stdev=stdev)
            / baseline_mode_folder(subtract_baseline=subtract_baseline)
            / likelihood_mode_folder(use_factorized_likelihoods=use_factorized_likelihoods)
            / f"look-{lft}_ns-{ns}.log"
        )
        if label_type != "rademacher":
            log_path = log_path.parent / label_type / log_path.name
        if normalize_labels:
            log_path = log_path.parent / "normalized" / log_path.name
        cmd = [
            sys.executable, script,
            "--num-lookforward-tokens", str(lft),
            "--num-rollouts", str(rollouts),
            "--seed", str(seed),
            "--device", "{device}",
            "--train-steps", str(train_steps),
            "--val-every-n-steps", str(val_every_n_steps),
            "--num-samples", str(ns),
            "--subtract-baseline", str(subtract_baseline),
            "--use-factorized-likelihoods", str(use_factorized_likelihoods),
            "--gaussian-stdev", str(stdev),
            "--label-type", label_type,
        ]
        if normalize_labels:
            cmd.append("--normalize-labels")
            cmd.extend(["--label-range", str(label_range[0]), str(label_range[1])])
        if train_from_scratch:
            cmd.append("--train-from-scratch")
        jobs.append(Job(cmd_template=cmd, label=label, log_path=log_path))
    return jobs


def _build_ntp_jobs(
    *,
    lookforward_tokens: tuple[int, ...],
    num_samples_values: tuple[int, ...],
    label_type: str,
    normalize_labels: bool,
    label_range: tuple[float, float],
) -> list[Job]:
    """NTP baseline is inference-only and deterministic — no seeds, rollouts,
    stdev, train_steps, baseline, or factorized flags. The cross-product is
    only over (lookforward_tokens × num_samples_values)."""
    script = str(_script_path("ntp_baseline"))
    jobs: list[Job] = []
    for lft, ns in itertools.product(lookforward_tokens, num_samples_values):
        label = f"ntp_baseline_look-{lft}_ns-{ns}_lbl-{label_type}"
        log_path = (
            artifacts_dir() / "ntp_baseline" / "logs"
            / f"look-{lft}_ns-{ns}.log"
        )
        if label_type != "rademacher":
            log_path = log_path.parent / label_type / log_path.name
        if normalize_labels:
            log_path = log_path.parent / "normalized" / log_path.name
        cmd = [
            sys.executable, script,
            "--num-lookforward-tokens", str(lft),
            "--num-samples", str(ns),
            "--label-type", label_type,
            "--device", "{device}",
        ]
        if normalize_labels:
            cmd.append("--normalize-labels")
            cmd.extend(["--label-range", str(label_range[0]), str(label_range[1])])
        jobs.append(Job(cmd_template=cmd, label=label, log_path=log_path))
    return jobs


# ─── CLI ──────────────────────────────────────────────────────────────────────


@click.command()
@click.option(
    "--method",
    type=click.Choice(["sl", "sl_ce", "grpo", "rloo", "maxrl", "ntp_baseline"]),
    multiple=True,
    required=True,
    help="Training method(s) to run. Repeat for cross-algorithm batching.",
)
@click.option(
    "--seeds",
    type=INT_LIST,
    default=None,
    help=(
        "Comma-separated random seeds (e.g. '51,61,121'). "
        "Required when any non-NTP method is selected; ignored by ntp_baseline."
    ),
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
    type=INT_LIST,
    default=str(DEFAULT_NUM_SAMPLES),
    show_default=True,
    help="Comma-separated training sample counts to sweep.",
)
@click.option(
    "--train-steps",
    type=int,
    default=DEFAULT_TRAIN_STEPS,
    show_default=True,
    help="Total number of gradient steps per run.",
)
@click.option(
    "--val-every-n-steps",
    type=int,
    default=DEFAULT_VAL_EVERY_N_STEPS,
    show_default=True,
    help="Validate every N gradient steps.",
)
@click.option(
    "--gaussian-stdev",
    type=FLOAT_LIST,
    default=str(DEFAULT_GAUSSIAN_STDEV),
    show_default=True,
    help="Comma-separated stdevs for Gaussian policy rollouts (ignored for SL).",
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
    "--label-type",
    type=click.Choice(["rademacher", "token_id"]),
    default="rademacher",
    show_default=True,
    help="Label type for the regression target.",
)
@click.option(
    "--normalize-labels",
    is_flag=True,
    default=False,
    show_default=True,
    help="Normalize labels before training.",
)
@click.option(
    "--label-range",
    type=(float, float),
    default=(0.0, 1.0),
    show_default=True,
    help="Target (lo, hi) range for label normalization (only effective with --normalize-labels).",
)
@click.option(
    "--train-from-scratch/--no-train-from-scratch",
    default=False,
    show_default=True,
    help=(
        "Random-init the backbone instead of loading pretrained weights. "
        "Routes artifacts to <method>_scratch/ subtrees. Incompatible with "
        "ntp_baseline (inference-only)."
    ),
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
    method: tuple[str, ...],
    seeds: tuple[int, ...] | None,
    lookforward_tokens: tuple[int, ...],
    rollout_steps: tuple[int, ...],
    num_samples: tuple[int, ...],
    train_steps: int,
    val_every_n_steps: int,
    gaussian_stdev: tuple[float, ...],
    gpu_ids: tuple[int, ...] | None,
    subtract_baseline: bool,
    use_factorized_likelihoods: bool,
    factorized: bool,
    label_type: str,
    normalize_labels: bool,
    label_range: tuple[float, float],
    train_from_scratch: bool,
    dry_run: bool,
    fail_fast: bool,
) -> None:
    # Validate gaussian_stdev values.
    for stdev in gaussian_stdev:
        if stdev <= 0.0:
            raise click.BadParameter(
                f"gaussian_stdev must be positive, got {stdev}",
                param_hint="--gaussian-stdev",
            )

    # Deduplicate methods, preserving order.
    methods = list(dict.fromkeys(method))

    # NTP baseline is inference-only on the pretrained model — there's no
    # meaningful "from scratch" version of an intrinsic-variance baseline.
    if train_from_scratch and "ntp_baseline" in methods:
        raise click.BadParameter(
            "ntp_baseline cannot be combined with --train-from-scratch "
            "(NTP baseline is inference-only on pretrained weights).",
            param_hint="--train-from-scratch",
        )

    # NTP baseline is deterministic and ignores --seeds. Other methods require it.
    non_ntp_methods = [m for m in methods if m != "ntp_baseline"]
    if non_ntp_methods and not seeds:
        raise click.BadParameter(
            f"--seeds is required when any non-NTP method is selected (got methods={methods})",
            param_hint="--seeds",
        )
    seeds_tuple: tuple[int, ...] = seeds if seeds is not None else ()

    # Build job list per method.
    jobs_by_method: dict[str, list[Job]] = {}
    for m in methods:
        if m == "sl":
            jobs_by_method[m] = _build_sl_jobs(
                seeds=seeds_tuple,
                lookforward_tokens=lookforward_tokens,
                num_samples_values=num_samples,
                train_steps=train_steps,
                val_every_n_steps=val_every_n_steps,
                label_type=label_type,
                normalize_labels=normalize_labels,
                label_range=label_range,
                train_from_scratch=train_from_scratch,
            )
        elif m == "sl_ce":
            jobs_by_method[m] = _build_sl_ce_jobs(
                seeds=seeds_tuple,
                lookforward_tokens=lookforward_tokens,
                num_samples_values=num_samples,
                train_steps=train_steps,
                val_every_n_steps=val_every_n_steps,
                label_type=label_type,
                normalize_labels=normalize_labels,
                label_range=label_range,
                train_from_scratch=train_from_scratch,
            )
        elif m == "grpo":
            jobs_by_method[m] = _build_grpo_jobs(
                seeds=seeds_tuple,
                lookforward_tokens=lookforward_tokens,
                rollout_steps=rollout_steps,
                num_samples_values=num_samples,
                train_steps=train_steps,
                val_every_n_steps=val_every_n_steps,
                gaussian_stdev_values=gaussian_stdev,
                label_type=label_type,
                normalize_labels=normalize_labels,
                label_range=label_range,
                train_from_scratch=train_from_scratch,
            )
        elif m == "rloo":
            jobs_by_method[m] = _build_rloo_jobs(
                seeds=seeds_tuple,
                lookforward_tokens=lookforward_tokens,
                rollout_steps=rollout_steps,
                num_samples_values=num_samples,
                train_steps=train_steps,
                val_every_n_steps=val_every_n_steps,
                factorized=factorized,
                gaussian_stdev_values=gaussian_stdev,
                label_type=label_type,
                normalize_labels=normalize_labels,
                label_range=label_range,
                train_from_scratch=train_from_scratch,
            )
        elif m == "maxrl":
            jobs_by_method[m] = _build_maxrl_jobs(
                seeds=seeds_tuple,
                lookforward_tokens=lookforward_tokens,
                rollout_steps=rollout_steps,
                num_samples_values=num_samples,
                train_steps=train_steps,
                val_every_n_steps=val_every_n_steps,
                subtract_baseline=subtract_baseline,
                use_factorized_likelihoods=use_factorized_likelihoods,
                gaussian_stdev_values=gaussian_stdev,
                label_type=label_type,
                normalize_labels=normalize_labels,
                label_range=label_range,
                train_from_scratch=train_from_scratch,
            )
        elif m == "ntp_baseline":
            jobs_by_method[m] = _build_ntp_jobs(
                lookforward_tokens=lookforward_tokens,
                num_samples_values=num_samples,
                label_type=label_type,
                normalize_labels=normalize_labels,
                label_range=label_range,
            )
        else:
            raise click.BadParameter(f"Unknown method: {m}")

    all_jobs = [job for m in methods for job in jobs_by_method[m]]

    # Summary.
    pool = GPUPool(device_ids=list(gpu_ids) if gpu_ids else None)
    click.echo(f"methods={methods}  seeds={seeds_tuple}  lookforward_tokens={lookforward_tokens}")
    has_rollouts = any(m not in ("sl", "sl_ce", "ntp_baseline") for m in methods)
    if has_rollouts:
        click.echo(f"rollout_steps={rollout_steps}")
    click.echo(
        f"num_samples={num_samples}  gaussian_stdev={gaussian_stdev}  "
        f"train_steps={train_steps}  gpus={pool.num_devices}  "
        f"train_from_scratch={train_from_scratch}  total_jobs={len(all_jobs)}"
    )

    if dry_run:
        click.echo("\nJob list:")
        global_idx = 0
        for m in methods:
            method_jobs = jobs_by_method[m]
            click.echo(f"\n  ── {m} ({len(method_jobs)} jobs) ──")
            for job in method_jobs:
                click.echo(f"    [{global_idx:3d}] {job.label}")
                global_idx += 1
        return

    # Dispatch — no pool-level log_dir; each job carries its own log_path.
    results = pool.run(all_jobs, fail_fast=fail_fast)

    # Report.
    failures = [r for r in results if r.returncode != 0]
    succeeded = len(results) - len(failures)
    click.echo(
        f"\n===== complete: {succeeded}/{len(all_jobs)} succeeded "
        f"(methods={methods}) ====="
    )
    if failures:
        click.echo("Failures:")
        for r in failures:
            click.echo(f"  [{r.device}] {r.job.label} — exit {r.returncode}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
