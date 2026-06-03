"""Tests for the corpus-regression orchestrator's `--train-from-scratch` switch.

These tests load `experiments/corpus-regression/orchestrate.py` directly by
path (the dash in the directory name forbids an ordinary import) and
exercise:
  * The job-builder helpers (`_build_sl_jobs`, `_build_grpo_jobs`,
    `_build_rloo_jobs`, `_build_maxrl_jobs`) — verifying that
    `train_from_scratch=True` routes log paths under `<method>_scratch/`
    and appends `--train-from-scratch` to the job's cmd template.
  * The Click CLI — verifying `--train-from-scratch` rejects
    `--method ntp_baseline` and that dry-run output mentions `_scratch`.

The tests do not load any HuggingFace weights or run any GPU code.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
from click.testing import CliRunner

from src import get_repo_base


def _load_orchestrate_module():
    """Load orchestrate.py by path because its parent directory is dashed."""
    orchestrate_path = (
        get_repo_base()
        / "experiments"
        / "corpus-regression"
        / "orchestrate.py"
    )
    spec = importlib.util.spec_from_file_location(
        "corpus_regression_orchestrate", orchestrate_path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {orchestrate_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


orch = _load_orchestrate_module()


def _build_one_sl_cmd(*, train_from_scratch: bool) -> tuple[list[str], Path]:
    jobs = orch._build_sl_jobs(
        seeds=(51,),
        lookforward_tokens=(1,),
        num_samples_values=(100,),
        train_steps=2,
        val_every_n_steps=1,
        label_type="rademacher",
        normalize_labels=False,
        label_range=(0.0, 1.0),
        train_from_scratch=train_from_scratch,
    )
    assert len(jobs) == 1
    return list(jobs[0].cmd_template), jobs[0].log_path


def test_sl_job_builder_off_keeps_pretrained_layout() -> None:
    cmd, log_path = _build_one_sl_cmd(train_from_scratch=False)
    assert "--train-from-scratch" not in cmd
    # Log path lives under sl/, not sl_scratch/
    parts = log_path.parts
    assert "sl" in parts
    assert "sl_scratch" not in parts


def test_sl_job_builder_on_routes_to_scratch_subtree() -> None:
    cmd, log_path = _build_one_sl_cmd(train_from_scratch=True)
    assert cmd.count("--train-from-scratch") == 1
    parts = log_path.parts
    assert "sl_scratch" in parts
    assert "sl" not in [p for p in parts if p == "sl"]


def test_grpo_job_builder_on_routes_to_scratch_subtree() -> None:
    jobs = orch._build_grpo_jobs(
        seeds=(51,),
        lookforward_tokens=(1,),
        rollout_steps=(8,),
        num_samples_values=(100,),
        train_steps=2,
        val_every_n_steps=1,
        gaussian_stdev_values=(1.0,),
        label_type="rademacher",
        normalize_labels=False,
        label_range=(0.0, 1.0),
        train_from_scratch=True,
    )
    assert len(jobs) == 1
    assert "--train-from-scratch" in jobs[0].cmd_template
    assert "grpo_scratch" in jobs[0].log_path.parts


def test_rloo_job_builder_on_routes_to_scratch_subtree() -> None:
    jobs = orch._build_rloo_jobs(
        seeds=(51,),
        lookforward_tokens=(1,),
        rollout_steps=(8,),
        num_samples_values=(100,),
        train_steps=2,
        val_every_n_steps=1,
        factorized=True,
        gaussian_stdev_values=(1.0,),
        label_type="rademacher",
        normalize_labels=False,
        label_range=(0.0, 1.0),
        train_from_scratch=True,
    )
    assert len(jobs) == 1
    assert "--train-from-scratch" in jobs[0].cmd_template
    assert "rloo_scratch" in jobs[0].log_path.parts


def test_maxrl_job_builder_on_routes_to_scratch_subtree() -> None:
    jobs = orch._build_maxrl_jobs(
        seeds=(51,),
        lookforward_tokens=(1,),
        rollout_steps=(8,),
        num_samples_values=(100,),
        train_steps=2,
        val_every_n_steps=1,
        subtract_baseline=True,
        use_factorized_likelihoods=True,
        gaussian_stdev_values=(1.0,),
        label_type="rademacher",
        normalize_labels=False,
        label_range=(0.0, 1.0),
        train_from_scratch=True,
    )
    assert len(jobs) == 1
    assert "--train-from-scratch" in jobs[0].cmd_template
    assert "maxrl_scratch" in jobs[0].log_path.parts


def test_method_dir_helper() -> None:
    assert orch._method_dir("sl", train_from_scratch=False) == "sl"
    assert orch._method_dir("sl", train_from_scratch=True) == "sl_scratch"
    assert orch._method_dir("maxrl", train_from_scratch=True) == "maxrl_scratch"


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


def test_cli_rejects_train_from_scratch_with_ntp_baseline(
    runner: CliRunner,
) -> None:
    result = runner.invoke(
        orch.main,
        [
            "--method", "ntp_baseline",
            "--method", "sl",
            "--seeds", "51",
            "--lookforward-tokens", "1",
            "--num-samples", "100",
            "--train-steps", "2",
            "--val-every-n-steps", "1",
            "--rollout-steps", "8",
            "--train-from-scratch",
            "--dry-run",
        ],
    )
    assert result.exit_code != 0
    assert "ntp_baseline cannot be combined with --train-from-scratch" in (
        result.output
    )


def test_cli_dry_run_with_scratch_lists_jobs_and_announces_flag(
    runner: CliRunner,
) -> None:
    result = runner.invoke(
        orch.main,
        [
            "--method", "sl",
            "--seeds", "51",
            "--lookforward-tokens", "1",
            "--num-samples", "100",
            "--train-steps", "2",
            "--val-every-n-steps", "1",
            "--rollout-steps", "8",
            "--train-from-scratch",
            "--dry-run",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "train_from_scratch=True" in result.output
    assert "sl_seed-51_look-1_ns-100_lbl-rademacher" in result.output


def test_cli_dry_run_without_scratch_announces_flag_off(
    runner: CliRunner,
) -> None:
    result = runner.invoke(
        orch.main,
        [
            "--method", "sl",
            "--seeds", "51",
            "--lookforward-tokens", "1",
            "--num-samples", "100",
            "--train-steps", "2",
            "--val-every-n-steps", "1",
            "--rollout-steps", "8",
            "--dry-run",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "train_from_scratch=False" in result.output
