"""
Orchestrator for one SL bag-of-words sweep (single dataset, single seed).

Hardcodes cuda:0 and cuda:1. Corrs are partitioned between devices: GPU0 gets
candidate_corrs[:len//2] in stride-4 top-to-bottom order, GPU1 gets
candidate_corrs[len//2:] in stride-4 bottom-to-top order. Seed iteration is
the caller's responsibility (bash driver).

Usage:
    uv run python experiments/bow/sl-corr/orchestrate.py \\
        --dataset homoskedastic --seed 51
    uv run python experiments/bow/sl-corr/orchestrate.py \\
        --dataset homoskedastic --seed 51 --dry-run
"""

import subprocess
from pathlib import Path
from threading import Thread

import click

from src import chdir_repo_base, get_repo_base

chdir_repo_base()
repo_root = get_repo_base()

from src.data.bag_of_words import candidate_corrs  # noqa: E402
from src.experiments.bag_of_words.config import (  # noqa: E402
    DATASET_KINDS,
    stride_4_bottom_up,
    stride_4_top_down,
    sweep_root_name,
)

SINGLE_RUN = repo_root / "experiments" / "bow" / "sl-corr" / "single_run.py"
ARTIFACTS_ROOT = repo_root / "artifacts"
METHOD = "sl"


def build_jobs(*, device_id: int) -> list[float]:
    half = len(candidate_corrs) // 2
    if device_id == 0:
        low = candidate_corrs[:half]
        return [low[i] for i in stride_4_top_down(len(low))]
    high = candidate_corrs[half:]
    return [high[i] for i in stride_4_bottom_up(len(high))]


def run_jobs(
    *, jobs: list[float], device_id: int, dataset: str, seed: int, log_path: Path,
) -> None:
    device = f"cuda:{device_id}"
    with log_path.open("ab") as log_fh:
        header = (
            f"\n===== dataset={dataset} seed={seed} device={device} "
            f"({len(jobs)} jobs) =====\n"
        )
        log_fh.write(header.encode())
        log_fh.flush()
        for corr in jobs:
            cmd = [
                "uv", "run", "python", str(SINGLE_RUN),
                "--dataset", dataset,
                "--corr", str(corr),
                "--seed", str(seed),
                "--device", device,
            ]
            tag = f"[{device} dataset={dataset} seed={seed} corr={corr}]"
            print(f"{tag} launching", flush=True)
            returncode = subprocess.call(cmd, stdout=log_fh, stderr=subprocess.STDOUT)
            status = "done" if returncode == 0 else f"FAILED rc={returncode}"
            print(f"{tag} {status}", flush=True)


@click.command()
@click.option("--dataset", type=click.Choice(DATASET_KINDS), required=True)
@click.option("--seed", type=int, required=True)
@click.option("--dry-run", is_flag=True, help="Print the planned job lists and exit.")
def main(dataset: str, seed: int, dry_run: bool) -> None:
    jobs_by_device = {dev: build_jobs(device_id=dev) for dev in (0, 1)}

    print(f"dataset={dataset} seed={seed}")
    for dev, jobs in jobs_by_device.items():
        print(f"device {dev}: {len(jobs)} jobs")
        for corr in jobs:
            print(f"  corr={corr:.3f}")
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
