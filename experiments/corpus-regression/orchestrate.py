"""
Unified orchestrator for corpus-regression experiments.

Replaces the per-method orchestrate.py scripts with a single entry point
that supports configurable sweep grids and dynamic GPU scheduling.

Usage:
    uv run python experiments/corpus-regression/orchestrate.py \
        --method sl_mse --seeds 51,61 --train-steps 5

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
        --method sl_mse --method maxrl --seeds 51,61 \
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
    DEFAULT_BATCH_SIZE,
    artifacts_dir,
    baseline_mode_folder,
    batch_size_segment,
    factorized_mode_folder,
    likelihood_mode_folder,
    lr_schedule_segment,
    sigma_folder,
)
from src.experiments.gpu_pool import GPUPool, Job  # noqa: E402

Method = Literal["sl_mse", "sl_ce", "grpo", "rloo", "maxrl", "pretrained_baseline"]
LRSchedule = Literal["flat", "cosine"]

DEFAULT_LOOKFORWARD_TOKENS = (1, 2, 3, 4, 5, 6, 7, 8)
DEFAULT_ROLLOUT_STEPS = (4, 16, 128, 1024)
DEFAULT_NUM_SAMPLES = 100_000
DEFAULT_TRAIN_STEPS = 10_000
DEFAULT_VAL_EVERY_N_STEPS = 2000
DEFAULT_GAUSSIAN_STDEV = 1.0
DEFAULT_LR_PER_SAMPLE = 1e-5
DEFAULT_LR_SCHEDULE: LRSchedule = "flat"
DEFAULT_WARMUP_RATIO = 0.05
DEFAULT_LR_MIN_RATIO = 0.1


def _lr_log_segment(lr_per_sample: float) -> str:
    """Path segment that disambiguates concurrent LR sweeps in
    `<artifacts_dir>/<method>/logs/...`. Mirrors `lr_<value>` levels added by
    each method's `study_folder` so logs and parquets stay aligned."""
    return f"lr_{lr_per_sample:.2e}"


def _bs_log_segment(batch_size: int) -> str | None:
    """Path segment that disambiguates concurrent batch-size sweeps in
    `<artifacts_dir>/<method>/logs/...`. Returns ``None`` for the canonical
    default (`DEFAULT_BATCH_SIZE = 64`) so existing log paths are byte-for-
    byte preserved — exactly mirroring `batch_size_segment` from config.py
    on the artifact side."""
    return batch_size_segment(batch_size=batch_size)


def _steps_log_segment(train_steps: int) -> str:
    """Path segment that disambiguates concurrent train-steps sweeps. Mirrors
    `steps-{N:06d}` added by `canonical_kwargs`'s `study_folder` so logs and
    parquets stay aligned. Zero-pad to 6 digits so lexicographic sort matches
    numeric order up to 999_999 steps."""
    return f"steps-{train_steps:06d}"


def _schedule_label_suffix(
    *,
    lr_schedule: str,
    warmup_ratio: float,
    lr_min_ratio: float,
) -> str:
    """Job-label suffix for non-flat schedules; empty string for flat. Mirrors
    `lr_schedule_segment` but encoded for the inline label, not the path."""
    if lr_schedule == "flat":
        return ""
    return (
        f"_sched-{lr_schedule}_warm-{warmup_ratio:.3f}_min-{lr_min_ratio:.2f}"
    )


def _append_schedule_segment(
    log_path: Path,
    *,
    lr_schedule: str,
    warmup_ratio: float,
    lr_min_ratio: float,
) -> Path:
    """Insert the optional schedule segment into a log path **between** the
    `steps-{N:06d}` directory and the `<label_type>/<filename>.log` tail.

    The incoming `log_path` ends in
    ``.../<lr_seg>/<steps_seg>/<label_type>/<look-{K}_ns-{N}.log>`` (or with
    a ``normalized/`` prefix on the basename's parent). For non-flat
    schedules we slot the schedule segment between `<steps_seg>` and
    `<label_type>` so the log tree mirrors the corresponding study_folder
    layout (where `lr_schedule_segment` is also appended after
    `steps-{N:06d}`). For flat schedules this is a no-op.
    """
    sched = lr_schedule_segment(
        lr_schedule=lr_schedule,
        warmup_ratio=warmup_ratio,
        lr_min_ratio=lr_min_ratio,
    )
    if sched is None:
        return log_path
    # `log_path` is .../<lr_seg>/<steps_seg>/<...tail>` — peel off the tail
    # (label_type subdir + filename, possibly with a `normalized/` parent),
    # insert the schedule segment after steps_seg, and re-append the tail.
    # We assert the `steps-` prefix matches so misuse is caught loudly.
    ancestor_parts = log_path.parts
    steps_idx: int | None = None
    for idx in range(len(ancestor_parts) - 1, -1, -1):
        if ancestor_parts[idx].startswith("steps-"):
            steps_idx = idx
            break
    if steps_idx is None:
        raise ValueError(
            f"Expected log_path to contain a `steps-*` segment; got {log_path}",
        )
    head = Path(*ancestor_parts[: steps_idx + 1])
    tail_parts = ancestor_parts[steps_idx + 1 :]
    return head.joinpath(sched, *tail_parts)


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


def _method_dir(method: str, *, train_from_scratch: bool) -> Path:
    """Method-level artifact subfolder. From-scratch sweeps land in a
    `<method>/from_scratch` sibling alongside `<method>/from_pretrain` so the
    initialization regime is an explicit path level rather than an ad-hoc
    `_scratch` suffix."""
    from_dir = "from_scratch" if train_from_scratch else "from_pretrain"
    return Path(method) / from_dir


def _build_sl_mse_jobs(
    *,
    seeds: tuple[int, ...],
    lookforward_tokens: tuple[int, ...],
    num_samples_values: tuple[int, ...],
    batch_size_values: tuple[int, ...],
    train_steps: int,
    val_every_n_steps: int,
    lr_per_sample_values: tuple[float, ...],
    lr_schedule: str,
    warmup_ratio: float,
    lr_min_ratio: float,
    label_type: str,
    normalize_labels: bool,
    label_range: tuple[float, float],
    train_from_scratch: bool,
) -> list[Job]:
    script = str(_script_path("sl_mse"))
    method_dir = _method_dir("sl_mse", train_from_scratch=train_from_scratch)
    steps_seg = _steps_log_segment(train_steps)
    sched_suffix = _schedule_label_suffix(
        lr_schedule=lr_schedule,
        warmup_ratio=warmup_ratio,
        lr_min_ratio=lr_min_ratio,
    )
    jobs: list[Job] = []
    for seed, lft, ns, bs, lr_per_sample in itertools.product(
        seeds, lookforward_tokens, num_samples_values, batch_size_values,
        lr_per_sample_values,
    ):
        lr_seg = _lr_log_segment(lr_per_sample)
        bs_seg = _bs_log_segment(bs)
        label = (
            f"sl_mse_seed-{seed}_look-{lft}_ns-{ns}_bs-{bs}"
            f"_lr-{lr_per_sample:.2e}_steps-{train_steps:06d}"
            f"_lbl-{label_type}{sched_suffix}"
        )
        log_path = artifacts_dir() / method_dir / "logs" / f"seed-{seed}"
        if bs_seg is not None:
            log_path = log_path / bs_seg
        log_path = (
            log_path
            / lr_seg
            / steps_seg
            / label_type
            / f"look-{lft}_ns-{ns}.log"
        )
        if normalize_labels:
            log_path = log_path.parent / "normalized" / log_path.name
        log_path = _append_schedule_segment(
            log_path,
            lr_schedule=lr_schedule,
            warmup_ratio=warmup_ratio,
            lr_min_ratio=lr_min_ratio,
        )
        cmd = [
            sys.executable, script,
            "--num-lookforward-tokens", str(lft),
            "--seed", str(seed),
            "--device", "{device}",
            "--train-steps", str(train_steps),
            "--val-every-n-steps", str(val_every_n_steps),
            "--num-samples", str(ns),
            "--batch-size", str(bs),
            "--lr-per-sample", str(lr_per_sample),
            "--lr-schedule", lr_schedule,
            "--warmup-ratio", str(warmup_ratio),
            "--lr-min-ratio", str(lr_min_ratio),
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
    batch_size_values: tuple[int, ...],
    train_steps: int,
    val_every_n_steps: int,
    lr_per_sample_values: tuple[float, ...],
    lr_schedule: str,
    warmup_ratio: float,
    lr_min_ratio: float,
    label_type: str,
    normalize_labels: bool,
    label_range: tuple[float, float],
    train_from_scratch: bool,
) -> list[Job]:
    """Mirror of `_build_sl_mse_jobs` for the NTP-CE supervised variant.

    `sl_ce` only supports K=1 — `single_run.py` rejects other values — but
    we keep the same CLI surface as SL so the orchestrator's `--lookforward-
    tokens` flag behaves predictably; non-1 values will surface as an error
    inside the spawned process.
    """
    script = str(_script_path("sl_ce"))
    method_dir = _method_dir("sl_ce", train_from_scratch=train_from_scratch)
    steps_seg = _steps_log_segment(train_steps)
    sched_suffix = _schedule_label_suffix(
        lr_schedule=lr_schedule,
        warmup_ratio=warmup_ratio,
        lr_min_ratio=lr_min_ratio,
    )
    jobs: list[Job] = []
    for seed, lft, ns, bs, lr_per_sample in itertools.product(
        seeds, lookforward_tokens, num_samples_values, batch_size_values,
        lr_per_sample_values,
    ):
        lr_seg = _lr_log_segment(lr_per_sample)
        bs_seg = _bs_log_segment(bs)
        label = (
            f"sl_ce_seed-{seed}_look-{lft}_ns-{ns}_bs-{bs}"
            f"_lr-{lr_per_sample:.2e}_steps-{train_steps:06d}"
            f"_lbl-{label_type}{sched_suffix}"
        )
        log_path = artifacts_dir() / method_dir / "logs" / f"seed-{seed}"
        if bs_seg is not None:
            log_path = log_path / bs_seg
        log_path = (
            log_path
            / lr_seg
            / steps_seg
            / label_type
            / f"look-{lft}_ns-{ns}.log"
        )
        if normalize_labels:
            log_path = log_path.parent / "normalized" / log_path.name
        log_path = _append_schedule_segment(
            log_path,
            lr_schedule=lr_schedule,
            warmup_ratio=warmup_ratio,
            lr_min_ratio=lr_min_ratio,
        )
        cmd = [
            sys.executable, script,
            "--num-lookforward-tokens", str(lft),
            "--seed", str(seed),
            "--device", "{device}",
            "--train-steps", str(train_steps),
            "--val-every-n-steps", str(val_every_n_steps),
            "--num-samples", str(ns),
            "--batch-size", str(bs),
            "--lr-per-sample", str(lr_per_sample),
            "--lr-schedule", lr_schedule,
            "--warmup-ratio", str(warmup_ratio),
            "--lr-min-ratio", str(lr_min_ratio),
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
    batch_size_values: tuple[int, ...],
    train_steps: int,
    val_every_n_steps: int,
    gaussian_stdev_values: tuple[float, ...],
    lr_per_sample_values: tuple[float, ...],
    lr_schedule: str,
    warmup_ratio: float,
    lr_min_ratio: float,
    label_type: str,
    normalize_labels: bool,
    label_range: tuple[float, float],
    train_from_scratch: bool,
) -> list[Job]:
    script = str(_script_path("grpo"))
    method_dir = _method_dir("grpo", train_from_scratch=train_from_scratch)
    steps_seg = _steps_log_segment(train_steps)
    sched_suffix = _schedule_label_suffix(
        lr_schedule=lr_schedule,
        warmup_ratio=warmup_ratio,
        lr_min_ratio=lr_min_ratio,
    )
    jobs: list[Job] = []
    for seed, lft, rollouts, ns, bs, stdev, lr_per_sample in itertools.product(
        seeds, lookforward_tokens, rollout_steps, num_samples_values,
        batch_size_values, gaussian_stdev_values, lr_per_sample_values,
    ):
        lr_seg = _lr_log_segment(lr_per_sample)
        bs_seg = _bs_log_segment(bs)
        label = (
            f"grpo_seed-{seed}_look-{lft}_roll-{rollouts}_ns-{ns}_bs-{bs}"
            f"_sigma-{stdev:.1f}_lr-{lr_per_sample:.2e}_steps-{train_steps:06d}"
            f"_lbl-{label_type}{sched_suffix}"
        )
        log_path = (
            artifacts_dir() / method_dir / "logs"
            / f"seed-{seed}"
            / f"rollouts-{rollouts}"
            / sigma_folder(gaussian_stdev=stdev)
        )
        if bs_seg is not None:
            log_path = log_path / bs_seg
        log_path = (
            log_path
            / lr_seg
            / steps_seg
            / label_type
            / f"look-{lft}_ns-{ns}.log"
        )
        if normalize_labels:
            log_path = log_path.parent / "normalized" / log_path.name
        log_path = _append_schedule_segment(
            log_path,
            lr_schedule=lr_schedule,
            warmup_ratio=warmup_ratio,
            lr_min_ratio=lr_min_ratio,
        )
        cmd = [
            sys.executable, script,
            "--num-lookforward-tokens", str(lft),
            "--num-rollouts", str(rollouts),
            "--seed", str(seed),
            "--device", "{device}",
            "--train-steps", str(train_steps),
            "--val-every-n-steps", str(val_every_n_steps),
            "--num-samples", str(ns),
            "--batch-size", str(bs),
            "--gaussian-stdev", str(stdev),
            "--lr-per-sample", str(lr_per_sample),
            "--lr-schedule", lr_schedule,
            "--warmup-ratio", str(warmup_ratio),
            "--lr-min-ratio", str(lr_min_ratio),
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
    batch_size_values: tuple[int, ...],
    train_steps: int,
    val_every_n_steps: int,
    factorized: bool,
    gaussian_stdev_values: tuple[float, ...],
    lr_per_sample_values: tuple[float, ...],
    lr_schedule: str,
    warmup_ratio: float,
    lr_min_ratio: float,
    label_type: str,
    normalize_labels: bool,
    label_range: tuple[float, float],
    train_from_scratch: bool,
) -> list[Job]:
    script = str(_script_path("rloo"))
    method_dir = _method_dir("rloo", train_from_scratch=train_from_scratch)
    steps_seg = _steps_log_segment(train_steps)
    sched_suffix = _schedule_label_suffix(
        lr_schedule=lr_schedule,
        warmup_ratio=warmup_ratio,
        lr_min_ratio=lr_min_ratio,
    )
    jobs: list[Job] = []
    for seed, lft, rollouts, ns, bs, stdev, lr_per_sample in itertools.product(
        seeds, lookforward_tokens, rollout_steps, num_samples_values,
        batch_size_values, gaussian_stdev_values, lr_per_sample_values,
    ):
        lr_seg = _lr_log_segment(lr_per_sample)
        bs_seg = _bs_log_segment(bs)
        label = (
            f"rloo_seed-{seed}_look-{lft}_roll-{rollouts}"
            f"_ns-{ns}_bs-{bs}_fact-{factorized}_sigma-{stdev:.1f}"
            f"_lr-{lr_per_sample:.2e}_steps-{train_steps:06d}"
            f"_lbl-{label_type}{sched_suffix}"
        )
        log_path = (
            artifacts_dir() / method_dir / "logs"
            / f"seed-{seed}"
            / f"rollouts-{rollouts}"
            / sigma_folder(gaussian_stdev=stdev)
            / factorized_mode_folder(factorized=factorized)
        )
        if bs_seg is not None:
            log_path = log_path / bs_seg
        log_path = (
            log_path
            / lr_seg
            / steps_seg
            / label_type
            / f"look-{lft}_ns-{ns}.log"
        )
        if normalize_labels:
            log_path = log_path.parent / "normalized" / log_path.name
        log_path = _append_schedule_segment(
            log_path,
            lr_schedule=lr_schedule,
            warmup_ratio=warmup_ratio,
            lr_min_ratio=lr_min_ratio,
        )
        cmd = [
            sys.executable, script,
            "--num-lookforward-tokens", str(lft),
            "--num-rollouts", str(rollouts),
            "--seed", str(seed),
            "--device", "{device}",
            "--train-steps", str(train_steps),
            "--val-every-n-steps", str(val_every_n_steps),
            "--num-samples", str(ns),
            "--batch-size", str(bs),
            "--factorized", str(factorized),
            "--gaussian-stdev", str(stdev),
            "--lr-per-sample", str(lr_per_sample),
            "--lr-schedule", lr_schedule,
            "--warmup-ratio", str(warmup_ratio),
            "--lr-min-ratio", str(lr_min_ratio),
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
    batch_size_values: tuple[int, ...],
    train_steps: int,
    val_every_n_steps: int,
    subtract_baseline: bool,
    use_factorized_likelihoods: bool,
    gaussian_stdev_values: tuple[float, ...],
    lr_per_sample_values: tuple[float, ...],
    lr_schedule: str,
    warmup_ratio: float,
    lr_min_ratio: float,
    label_type: str,
    normalize_labels: bool,
    label_range: tuple[float, float],
    train_from_scratch: bool,
) -> list[Job]:
    script = str(_script_path("maxrl"))
    method_dir = _method_dir("maxrl", train_from_scratch=train_from_scratch)
    steps_seg = _steps_log_segment(train_steps)
    sched_suffix = _schedule_label_suffix(
        lr_schedule=lr_schedule,
        warmup_ratio=warmup_ratio,
        lr_min_ratio=lr_min_ratio,
    )
    jobs: list[Job] = []
    for seed, lft, rollouts, ns, bs, stdev, lr_per_sample in itertools.product(
        seeds, lookforward_tokens, rollout_steps, num_samples_values,
        batch_size_values, gaussian_stdev_values, lr_per_sample_values,
    ):
        lr_seg = _lr_log_segment(lr_per_sample)
        bs_seg = _bs_log_segment(bs)
        label = (
            f"maxrl_seed-{seed}_look-{lft}_roll-{rollouts}"
            f"_ns-{ns}_bs-{bs}_bl-{subtract_baseline}_fact-{use_factorized_likelihoods}"
            f"_sigma-{stdev:.1f}_lr-{lr_per_sample:.2e}_steps-{train_steps:06d}"
            f"_lbl-{label_type}{sched_suffix}"
        )
        log_path = (
            artifacts_dir() / method_dir / "logs"
            / f"seed-{seed}"
            / f"rollouts-{rollouts}"
            / sigma_folder(gaussian_stdev=stdev)
            / baseline_mode_folder(subtract_baseline=subtract_baseline)
            / likelihood_mode_folder(use_factorized_likelihoods=use_factorized_likelihoods)
        )
        if bs_seg is not None:
            log_path = log_path / bs_seg
        log_path = (
            log_path
            / lr_seg
            / steps_seg
            / label_type
            / f"look-{lft}_ns-{ns}.log"
        )
        if normalize_labels:
            log_path = log_path.parent / "normalized" / log_path.name
        log_path = _append_schedule_segment(
            log_path,
            lr_schedule=lr_schedule,
            warmup_ratio=warmup_ratio,
            lr_min_ratio=lr_min_ratio,
        )
        cmd = [
            sys.executable, script,
            "--num-lookforward-tokens", str(lft),
            "--num-rollouts", str(rollouts),
            "--seed", str(seed),
            "--device", "{device}",
            "--train-steps", str(train_steps),
            "--val-every-n-steps", str(val_every_n_steps),
            "--num-samples", str(ns),
            "--batch-size", str(bs),
            "--subtract-baseline", str(subtract_baseline),
            "--use-factorized-likelihoods", str(use_factorized_likelihoods),
            "--gaussian-stdev", str(stdev),
            "--lr-per-sample", str(lr_per_sample),
            "--lr-schedule", lr_schedule,
            "--warmup-ratio", str(warmup_ratio),
            "--lr-min-ratio", str(lr_min_ratio),
            "--label-type", label_type,
        ]
        if normalize_labels:
            cmd.append("--normalize-labels")
            cmd.extend(["--label-range", str(label_range[0]), str(label_range[1])])
        if train_from_scratch:
            cmd.append("--train-from-scratch")
        jobs.append(Job(cmd_template=cmd, label=label, log_path=log_path))
    return jobs


def _build_pretrained_baseline_jobs(
    *,
    lookforward_tokens: tuple[int, ...],
    num_samples_values: tuple[int, ...],
    label_type: str,
    normalize_labels: bool,
    label_range: tuple[float, float],
) -> list[Job]:
    """Pretrained-model intrinsic-variance baseline is inference-only and
    deterministic — no seeds, rollouts, stdev, train_steps, baseline, or
    factorized flags. The cross-product is only over
    (lookforward_tokens × num_samples_values).

    Path layout mirrors the trained methods' `<method>/<from_*>/...` shape:
    pretrained_baseline always lives under `from_pretrain` (there is no
    from_scratch counterpart — it's inference-only on pretrained weights)."""
    script = str(_script_path("pretrained_baseline"))
    jobs: list[Job] = []
    for lft, ns in itertools.product(lookforward_tokens, num_samples_values):
        label = f"pretrained_baseline_look-{lft}_ns-{ns}_lbl-{label_type}"
        log_path = (
            artifacts_dir() / "pretrained_baseline" / "from_pretrain" / "logs" / label_type
            / f"look-{lft}_ns-{ns}.log"
        )
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
    type=click.Choice(["sl_mse", "sl_ce", "grpo", "rloo", "maxrl", "pretrained_baseline"]),
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
        "Required when any non-baseline method is selected; ignored by pretrained_baseline."
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
    "--batch-size",
    type=INT_LIST,
    default=str(DEFAULT_BATCH_SIZE),
    show_default=True,
    help=(
        "Comma-separated training batch sizes to sweep. "
        f"Non-default values (default = {DEFAULT_BATCH_SIZE}) route artifacts "
        "and logs under a `bs-{N}` path segment between `<dataset>` and "
        "`lr_<value>`; the default value is omitted from the path so existing "
        "artifacts retain byte-for-byte layout. Ignored by pretrained_baseline."
    ),
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
    "--lr-per-sample",
    type=FLOAT_LIST,
    default=str(DEFAULT_LR_PER_SAMPLE),
    show_default=True,
    help=(
        "Comma-separated per-sample learning rates to sweep. "
        "Final head_lr = lr_per_sample × batch_size; "
        "backbone gets head_lr / divisor (divisor=6.66 for fine-tune, 1.0 for "
        "from-scratch). Affects every method except pretrained_baseline (inference-only)."
    ),
)
@click.option(
    "--lr-schedule",
    type=click.Choice(["flat", "cosine"]),
    default=DEFAULT_LR_SCHEDULE,
    show_default=True,
    help=(
        "LR schedule shape. 'flat' = no scheduler, byte-for-byte today's "
        "behaviour and today's artifact paths. 'cosine' = linear warmup to "
        "peak LR, then half-cosine decay to lr_min_ratio × peak; runs are "
        "routed to a `sched-warmup-X.XXX-cosine-X.XX` sibling subtree under "
        "the existing `lr_*` segment."
    ),
)
@click.option(
    "--warmup-ratio",
    type=float,
    default=DEFAULT_WARMUP_RATIO,
    show_default=True,
    help="Warmup duration as a fraction of train_steps. Ignored when --lr-schedule flat.",
)
@click.option(
    "--lr-min-ratio",
    type=float,
    default=DEFAULT_LR_MIN_RATIO,
    show_default=True,
    help="Cosine end LR as a fraction of peak LR. Ignored when --lr-schedule flat.",
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
        "Routes artifacts to <method>/from_scratch/ subtrees. Incompatible with "
        "pretrained_baseline (inference-only)."
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
    batch_size: tuple[int, ...],
    train_steps: int,
    val_every_n_steps: int,
    gaussian_stdev: tuple[float, ...],
    lr_per_sample: tuple[float, ...],
    lr_schedule: str,
    warmup_ratio: float,
    lr_min_ratio: float,
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
    for lr in lr_per_sample:
        if lr <= 0.0:
            raise click.BadParameter(
                f"lr_per_sample must be positive, got {lr}",
                param_hint="--lr-per-sample",
            )
    for bs in batch_size:
        if bs <= 0:
            raise click.BadParameter(
                f"batch_size must be positive, got {bs}",
                param_hint="--batch-size",
            )
    # LR-schedule arg validation. Detailed range checks live in
    # `build_lr_scheduler`; here we just catch obvious nonsense early so
    # `--dry-run` surfaces the error before any subprocess is spawned.
    if lr_schedule not in ("flat", "cosine"):
        raise click.BadParameter(
            f"lr_schedule must be one of {{flat, cosine}}, got {lr_schedule!r}",
            param_hint="--lr-schedule",
        )
    if not 0.0 <= warmup_ratio < 1.0:
        raise click.BadParameter(
            f"warmup_ratio must be in [0.0, 1.0), got {warmup_ratio}",
            param_hint="--warmup-ratio",
        )
    if not 0.0 <= lr_min_ratio <= 1.0:
        raise click.BadParameter(
            f"lr_min_ratio must be in [0.0, 1.0], got {lr_min_ratio}",
            param_hint="--lr-min-ratio",
        )

    # Deduplicate methods, preserving order.
    methods = list(dict.fromkeys(method))

    # Pretrained-model baseline is inference-only on the pretrained model —
    # there's no meaningful "from scratch" version of an intrinsic-variance
    # baseline.
    if train_from_scratch and "pretrained_baseline" in methods:
        raise click.BadParameter(
            "pretrained_baseline cannot be combined with --train-from-scratch "
            "(pretrained_baseline is inference-only on pretrained weights).",
            param_hint="--train-from-scratch",
        )

    # Pretrained-model baseline is deterministic and ignores --seeds. Other
    # methods require it.
    non_baseline_methods = [m for m in methods if m != "pretrained_baseline"]
    if non_baseline_methods and not seeds:
        raise click.BadParameter(
            f"--seeds is required when any non-baseline method is selected (got methods={methods})",
            param_hint="--seeds",
        )
    seeds_tuple: tuple[int, ...] = seeds if seeds is not None else ()

    # Build job list per method.
    jobs_by_method: dict[str, list[Job]] = {}
    for m in methods:
        if m == "sl_mse":
            jobs_by_method[m] = _build_sl_mse_jobs(
                seeds=seeds_tuple,
                lookforward_tokens=lookforward_tokens,
                num_samples_values=num_samples,
                batch_size_values=batch_size,
                train_steps=train_steps,
                val_every_n_steps=val_every_n_steps,
                lr_per_sample_values=lr_per_sample,
                lr_schedule=lr_schedule,
                warmup_ratio=warmup_ratio,
                lr_min_ratio=lr_min_ratio,
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
                batch_size_values=batch_size,
                train_steps=train_steps,
                val_every_n_steps=val_every_n_steps,
                lr_per_sample_values=lr_per_sample,
                lr_schedule=lr_schedule,
                warmup_ratio=warmup_ratio,
                lr_min_ratio=lr_min_ratio,
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
                batch_size_values=batch_size,
                train_steps=train_steps,
                val_every_n_steps=val_every_n_steps,
                gaussian_stdev_values=gaussian_stdev,
                lr_per_sample_values=lr_per_sample,
                lr_schedule=lr_schedule,
                warmup_ratio=warmup_ratio,
                lr_min_ratio=lr_min_ratio,
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
                batch_size_values=batch_size,
                train_steps=train_steps,
                val_every_n_steps=val_every_n_steps,
                factorized=factorized,
                gaussian_stdev_values=gaussian_stdev,
                lr_per_sample_values=lr_per_sample,
                lr_schedule=lr_schedule,
                warmup_ratio=warmup_ratio,
                lr_min_ratio=lr_min_ratio,
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
                batch_size_values=batch_size,
                train_steps=train_steps,
                val_every_n_steps=val_every_n_steps,
                subtract_baseline=subtract_baseline,
                use_factorized_likelihoods=use_factorized_likelihoods,
                gaussian_stdev_values=gaussian_stdev,
                lr_per_sample_values=lr_per_sample,
                lr_schedule=lr_schedule,
                warmup_ratio=warmup_ratio,
                lr_min_ratio=lr_min_ratio,
                label_type=label_type,
                normalize_labels=normalize_labels,
                label_range=label_range,
                train_from_scratch=train_from_scratch,
            )
        elif m == "pretrained_baseline":
            jobs_by_method[m] = _build_pretrained_baseline_jobs(
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
    has_rollouts = any(m not in ("sl_mse", "sl_ce", "pretrained_baseline") for m in methods)
    if has_rollouts:
        click.echo(f"rollout_steps={rollout_steps}")
    click.echo(
        f"num_samples={num_samples}  batch_size={batch_size}  "
        f"gaussian_stdev={gaussian_stdev}  "
        f"lr_per_sample={lr_per_sample}  "
        f"lr_schedule={lr_schedule}  warmup_ratio={warmup_ratio}  "
        f"lr_min_ratio={lr_min_ratio}  "
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
