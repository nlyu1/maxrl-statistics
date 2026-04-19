"""
Orchestrator for the canonical MaxRL bag-of-words sweep.

Hardcodes cuda:0 and cuda:1. Seeds are the outermost loop with a cross-device
barrier between seeds. The rollout axis is partitioned across devices, and each
device traverses the full corr list.

Usage:
    uv run python experiments/bow/maxrl-corr/orchestrate.py --subtract-baseline
    uv run python experiments/bow/maxrl-corr/orchestrate.py --subtract-baseline --dry-run
"""

import subprocess
from pathlib import Path
from threading import Thread

import click

from src import chdir_repo_base, get_repo_base

chdir_repo_base()
repo_root = get_repo_base()

from src.data.bag_of_words import (  # noqa: E402
    candidate_corrs,
    candidate_rollout_steps,
    candidate_seeds,
)

SINGLE_RUN = repo_root / "experiments" / "bow" / "maxrl-corr" / "single_run.py"
LOG_BASE = repo_root / "artifacts" / "bow-maxrl-sweep" / "logs"


def baseline_mode_folder(*, subtract_baseline: bool) -> str:
    if subtract_baseline:
        return "subtract-baseline"
    return "no-subtract-baseline"


def stride_4_top_down(n: int) -> list[int]:
    return [i for offset in range(4) for i in range(offset, n, 4)]


def stride_4_bottom_up(n: int) -> list[int]:
    return list(reversed(stride_4_top_down(n)))


Job = tuple[float, int]  # (corr, num_rollouts)


def build_jobs(*, device_id: int) -> list[Job]:
    n = len(candidate_corrs)
    half = len(candidate_rollout_steps) // 2
    if device_id == 0:
        corr_order = [candidate_corrs[i] for i in stride_4_top_down(n)]
        rollouts = candidate_rollout_steps[:half]
    else:
        corr_order = [candidate_corrs[i] for i in stride_4_bottom_up(n)]
        rollouts = list(reversed(candidate_rollout_steps[half:]))
    return [(corr, r) for corr in corr_order for r in rollouts]


def run_jobs(
    *,
    jobs: list[Job],
    device_id: int,
    seed: int,
    log_path: Path,
    subtract_baseline: bool,
) -> None:
    device = f"cuda:{device_id}"
    baseline_flag = "--subtract-baseline"
    if not subtract_baseline:
        baseline_flag = "--no-subtract-baseline"
    with log_path.open("ab") as log_fh:
        baseline_mode = baseline_mode_folder(subtract_baseline=subtract_baseline)
        header = (
            f"\n===== seed={seed} device={device} baseline={baseline_mode} "
            f"({len(jobs)} jobs) =====\n"
        )
        log_fh.write(header.encode())
        log_fh.flush()
        for corr, num_rollouts in jobs:
            cmd = [
                "uv",
                "run",
                "python",
                str(SINGLE_RUN),
                "--corr",
                str(corr),
                "--num-rollouts",
                str(num_rollouts),
                "--seed",
                str(seed),
                "--device",
                device,
                baseline_flag,
            ]
            tag = (
                f"[{device} seed={seed} corr={corr} rollouts={num_rollouts} "
                f"baseline={baseline_mode}]"
            )
            print(f"{tag} launching", flush=True)
            returncode = subprocess.call(cmd, stdout=log_fh, stderr=subprocess.STDOUT)
            status = "done" if returncode == 0 else f"FAILED rc={returncode}"
            print(f"{tag} {status}", flush=True)


@click.command()
@click.option("--dry-run", is_flag=True, help="Print the planned job lists and exit.")
@click.option("--subtract-baseline/--no-subtract-baseline", required=True)
def main(dry_run: bool, subtract_baseline: bool) -> None:
    jobs_by_device = {dev: build_jobs(device_id=dev) for dev in (0, 1)}
    baseline_mode = baseline_mode_folder(subtract_baseline=subtract_baseline)

    print(f"baseline mode: {baseline_mode}")
    for dev, jobs in jobs_by_device.items():
        print(f"device {dev}: {len(jobs)} jobs per seed")
        for corr, r in jobs:
            print(f"  corr={corr:.3f}  rollouts={r}")
    total_seeds = len(candidate_seeds)
    total_jobs = sum(len(j) for j in jobs_by_device.values()) * total_seeds
    print(f"\nseeds={candidate_seeds}  total jobs across sweep: {total_jobs}")
    if dry_run:
        return

    log_base = LOG_BASE / baseline_mode
    log_base.mkdir(parents=True, exist_ok=True)

    for seed in candidate_seeds:
        threads = [
            Thread(
                target=run_jobs,
                kwargs=dict(
                    jobs=jobs_by_device[dev],
                    device_id=dev,
                    seed=seed,
                    log_path=log_base / f"device_{dev}.log",
                    subtract_baseline=subtract_baseline,
                ),
                daemon=True,
                name=f"device-{dev}-seed-{seed}",
            )
            for dev in (0, 1)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        print(f"\n===== seed {seed} complete on both devices =====\n")

    print("All seeds done.")


if __name__ == "__main__":
    main()
