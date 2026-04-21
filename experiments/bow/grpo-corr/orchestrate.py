"""
Orchestrator for one GRPO bag-of-words sweep (single dataset, single seed).

Hardcodes cuda:0 and cuda:1. Both devices traverse the full corr grid in
opposing stride-4 orders (GPU0 top-down, GPU1 bottom-up) without collision
because the rollout axis is partitioned by round-robin on index:
    GPU0 = candidate_rollout_steps[0::2]
    GPU1 = candidate_rollout_steps[1::2]
Each device thus gets a mix of small/medium/large rollouts rather than one
device being stuck with only the expensive ones. Seed iteration is the
caller's responsibility (bash driver).

Usage:
    uv run python experiments/bow/grpo-corr/orchestrate.py \\
        --dataset homoskedastic --seed 51
    uv run python experiments/bow/grpo-corr/orchestrate.py \\
        --dataset homoskedastic --seed 51 --dry-run
"""

import subprocess
from pathlib import Path
from threading import Thread

import click

from src import chdir_repo_base, get_repo_base

chdir_repo_base()
repo_root = get_repo_base()

from src.data.bag_of_words import candidate_corrs, candidate_rollout_steps  # noqa: E402
from src.experiments.bag_of_words.config import sweep_root_name  # noqa: E402

SINGLE_RUN = repo_root / "experiments" / "bow" / "grpo-corr" / "single_run.py"
ARTIFACTS_ROOT = repo_root / "artifacts"
METHOD = "grpo"
DATASET_CHOICES = ("homoskedastic", "row_heteroskedastic", "word_heteroskedastic")


def stride_4_top_down(n: int) -> list[int]:
    return [i for offset in range(4) for i in range(offset, n, 4)]


def stride_4_bottom_up(n: int) -> list[int]:
    return list(reversed(stride_4_top_down(n)))


Job = tuple[float, int]  # (corr, num_rollouts)


def build_jobs(*, device_id: int) -> list[Job]:
    n = len(candidate_corrs)
    if device_id == 0:
        corr_order = [candidate_corrs[i] for i in stride_4_top_down(n)]
    else:
        corr_order = [candidate_corrs[i] for i in stride_4_bottom_up(n)]
    rollouts = candidate_rollout_steps[device_id::2]
    return [(corr, r) for corr in corr_order for r in rollouts]


def run_jobs(
    *, jobs: list[Job], device_id: int, dataset: str, seed: int, log_path: Path,
) -> None:
    device = f"cuda:{device_id}"
    with log_path.open("ab") as log_fh:
        header = (
            f"\n===== dataset={dataset} seed={seed} device={device} "
            f"({len(jobs)} jobs) =====\n"
        )
        log_fh.write(header.encode())
        log_fh.flush()
        for corr, num_rollouts in jobs:
            cmd = [
                "uv", "run", "python", str(SINGLE_RUN),
                "--dataset", dataset,
                "--corr", str(corr),
                "--num-rollouts", str(num_rollouts),
                "--seed", str(seed),
                "--device", device,
            ]
            tag = (
                f"[{device} dataset={dataset} seed={seed} corr={corr} "
                f"rollouts={num_rollouts}]"
            )
            print(f"{tag} launching", flush=True)
            returncode = subprocess.call(cmd, stdout=log_fh, stderr=subprocess.STDOUT)
            status = "done" if returncode == 0 else f"FAILED rc={returncode}"
            print(f"{tag} {status}", flush=True)


@click.command()
@click.option("--dataset", type=click.Choice(DATASET_CHOICES), required=True)
@click.option("--seed", type=int, required=True)
@click.option("--dry-run", is_flag=True, help="Print the planned job lists and exit.")
def main(dataset: str, seed: int, dry_run: bool) -> None:
    jobs_by_device = {dev: build_jobs(device_id=dev) for dev in (0, 1)}

    print(f"dataset={dataset} seed={seed}")
    for dev, jobs in jobs_by_device.items():
        print(f"device {dev}: {len(jobs)} jobs")
        for corr, r in jobs:
            print(f"  corr={corr:.3f}  rollouts={r}")
    total = sum(len(j) for j in jobs_by_device.values())
    print(f"\ntotal jobs: {total}")
    if dry_run:
        return

    log_dir = ARTIFACTS_ROOT / sweep_root_name(method=METHOD, dataset=dataset) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    threads = [
        Thread(
            target=run_jobs,
            kwargs=dict(
                jobs=jobs_by_device[dev],
                device_id=dev,
                dataset=dataset,
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

    print(f"\n===== dataset={dataset} seed={seed} complete =====")


if __name__ == "__main__":
    main()
