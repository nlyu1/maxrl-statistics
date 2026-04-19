"""
Orchestrator for the canonical GRPO bag-of-words sweep.

Hardcodes cuda:0 and cuda:1. Seeds are the outermost loop (a cross-device
barrier separates seeds). Within a seed, GPU0 walks candidate_corrs
top-to-bottom in stride-4 order * candidate_rollout_steps[:len//2] natural,
and GPU1 walks bottom-to-top * candidate_rollout_steps[len//2:] reversed.
Both GPUs traverse the full corr list without collision because the rollout
axis is partitioned.

Usage:
    uv run python experiments/bow/grpo-corr/orchestrate.py
    uv run python experiments/bow/grpo-corr/orchestrate.py --dry-run
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

SINGLE_RUN = repo_root / "experiments" / "bow" / "grpo-corr" / "single_run.py"
LOG_BASE = repo_root / "artifacts" / "bow-grpo-sweep" / "logs"


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


def run_jobs(*, jobs: list[Job], device_id: int, seed: int, log_path: Path) -> None:
    device = f"cuda:{device_id}"
    with log_path.open("ab") as log_fh:
        header = f"\n===== seed={seed} device={device} ({len(jobs)} jobs) =====\n"
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
            ]
            tag = f"[{device} seed={seed} corr={corr} rollouts={num_rollouts}]"
            print(f"{tag} launching", flush=True)
            returncode = subprocess.call(cmd, stdout=log_fh, stderr=subprocess.STDOUT)
            status = "done" if returncode == 0 else f"FAILED rc={returncode}"
            print(f"{tag} {status}", flush=True)


@click.command()
@click.option("--dry-run", is_flag=True, help="Print the planned job lists and exit.")
def main(dry_run: bool) -> None:
    jobs_by_device = {dev: build_jobs(device_id=dev) for dev in (0, 1)}

    for dev, jobs in jobs_by_device.items():
        print(f"device {dev}: {len(jobs)} jobs per seed")
        for corr, r in jobs:
            print(f"  corr={corr:.3f}  rollouts={r}")
    total_seeds = len(candidate_seeds)
    total_jobs = sum(len(j) for j in jobs_by_device.values()) * total_seeds
    print(f"\nseeds={candidate_seeds}  total jobs across sweep: {total_jobs}")
    if dry_run:
        return

    LOG_BASE.mkdir(parents=True, exist_ok=True)

    # Ensure children die if orchestrator is killed (default process group).
    for seed in candidate_seeds:
        threads = [
            Thread(
                target=run_jobs,
                kwargs=dict(
                    jobs=jobs_by_device[dev],
                    device_id=dev,
                    seed=seed,
                    log_path=LOG_BASE / f"device_{dev}.log",
                ),
                daemon=True,
                name=f"device-{dev}-seed-{seed}",
            )
            for dev in (0, 1)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        print(f"\n===== seed {seed} complete on both devices =====\n")

    print("All seeds done.")


if __name__ == "__main__":
    main()
