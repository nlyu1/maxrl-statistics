"""
Orchestrator for one GRPO corpus-regression sweep (single seed).

Hardcodes cuda:0 and cuda:1. Both devices traverse the full
`candidate_lookforward_tokens` grid (GPU0 top-down, GPU1 bottom-up) and the
rollout axis is partitioned by extremes-vs-middles:
    GPU0 -> [1024, 4]   (max + min)
    GPU1 -> [16, 128]   (middle two)
giving each device a balanced mix of cheap and expensive runs. The cartesian
product over (lookforward, rollouts) is collision-free because the rollout
sets are disjoint. Seed iteration is the caller's responsibility.

Usage:
    uv run python experiments/corpus-regression/grpo/orchestrate.py --seed 51
    uv run python experiments/corpus-regression/grpo/orchestrate.py --seed 51 --dry-run
"""

import subprocess
from pathlib import Path
from threading import Thread

import click

from src import chdir_repo_base, get_repo_base

chdir_repo_base()
repo_root = get_repo_base()

from src.data.corpus_regression import candidate_lookforward_tokens  # noqa: E402
from src.experiments.corpus_regression.config import artifacts_dir  # noqa: E402

SINGLE_RUN = repo_root / "experiments" / "corpus-regression" / "grpo" / "single_run.py"
METHOD = "grpo"

# Disjoint rollout assignments per device.
device_rollouts: dict[int, list[int]] = {0: [1024, 4], 1: [16, 128]}


Job = tuple[int, int]  # (num_lookforward_tokens, num_rollouts)


def build_jobs(*, device_id: int) -> list[Job]:
    if device_id == 0:
        looks = list(candidate_lookforward_tokens)
    else:
        looks = list(reversed(candidate_lookforward_tokens))
    rollouts = device_rollouts[device_id]
    return [(look, r) for look in looks for r in rollouts]


def run_jobs(
    *, jobs: list[Job], device_id: int, seed: int, log_path: Path,
) -> None:
    device = f"cuda:{device_id}"
    with log_path.open("ab") as log_fh:
        header = (
            f"\n===== seed={seed} device={device} ({len(jobs)} jobs) =====\n"
        )
        log_fh.write(header.encode())
        log_fh.flush()
        for look, num_rollouts in jobs:
            cmd = [
                "uv", "run", "python", str(SINGLE_RUN),
                "--num-lookforward-tokens", str(look),
                "--num-rollouts", str(num_rollouts),
                "--seed", str(seed),
                "--device", device,
            ]
            tag = f"[{device} seed={seed} look={look} rollouts={num_rollouts}]"
            print(f"{tag} launching", flush=True)
            returncode = subprocess.call(cmd, stdout=log_fh, stderr=subprocess.STDOUT)
            status = "done" if returncode == 0 else f"FAILED rc={returncode}"
            print(f"{tag} {status}", flush=True)


@click.command()
@click.option("--seed", type=int, required=True)
@click.option("--dry-run", is_flag=True, help="Print the planned job lists and exit.")
def main(seed: int, dry_run: bool) -> None:
    jobs_by_device = {dev: build_jobs(device_id=dev) for dev in (0, 1)}

    print(f"seed={seed}")
    for dev, jobs in jobs_by_device.items():
        print(f"device {dev}: {len(jobs)} jobs")
        for look, r in jobs:
            print(f"  look={look}  rollouts={r}")
    total = sum(len(j) for j in jobs_by_device.values())
    print(f"\ntotal jobs: {total}")
    if dry_run:
        return

    log_dir = artifacts_dir() / METHOD / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    threads = [
        Thread(
            target=run_jobs,
            kwargs=dict(
                jobs=jobs_by_device[dev],
                device_id=dev,
                seed=seed,
                log_path=log_dir / f"device_{dev}.log",
            ),
            daemon=True,
            name=f"device-{dev}",
        )
        for dev in (0, 1)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    print(f"\n===== seed={seed} complete =====")


if __name__ == "__main__":
    main()
