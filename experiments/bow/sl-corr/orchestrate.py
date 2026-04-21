"""
Orchestrator for the canonical SL bag-of-words sweep.

Hardcodes cuda:0 and cuda:1. Each device thread iterates over the full
(seed, job) product independently — no cross-device barrier. Corrs are
partitioned between devices: GPU0 gets candidate_corrs[:len//2] in stride-4
top-to-bottom order, GPU1 gets candidate_corrs[len//2:] in stride-4
bottom-to-top order.

Usage:
    uv run python experiments/bow/sl-corr/orchestrate.py
    uv run python experiments/bow/sl-corr/orchestrate.py --dry-run
"""

import subprocess
from pathlib import Path
from threading import Thread

import click

from src import chdir_repo_base, get_repo_base

chdir_repo_base()
repo_root = get_repo_base()

from src.data.bag_of_words import candidate_corrs, candidate_seeds  # noqa: E402

SINGLE_RUN = repo_root / "experiments" / "bow" / "sl-corr" / "single_run.py"
LOG_BASE = repo_root / "artifacts" / "bow-sl-sweep" / "logs"


def stride_4_top_down(n: int) -> list[int]:
    return [i for offset in range(4) for i in range(offset, n, 4)]


def stride_4_bottom_up(n: int) -> list[int]:
    return list(reversed(stride_4_top_down(n)))


def build_jobs(*, device_id: int) -> list[float]:
    half = len(candidate_corrs) // 2
    if device_id == 0:
        low = candidate_corrs[:half]
        return [low[i] for i in stride_4_top_down(len(low))]
    high = candidate_corrs[half:]
    return [high[i] for i in stride_4_bottom_up(len(high))]


def run_jobs(
    *, jobs: list[float], device_id: int, seeds: list[int], log_path: Path
) -> None:
    device = f"cuda:{device_id}"
    with log_path.open("ab") as log_fh:
        for seed in seeds:
            header = f"\n===== seed={seed} device={device} ({len(jobs)} jobs) =====\n"
            log_fh.write(header.encode())
            log_fh.flush()
            for corr in jobs:
                cmd = [
                    "uv", "run", "python", str(SINGLE_RUN),
                    "--corr", str(corr),
                    "--seed", str(seed),
                    "--device", device,
                ]
                tag = f"[{device} seed={seed} corr={corr}]"
                print(f"{tag} launching", flush=True)
                returncode = subprocess.call(
                    cmd, stdout=log_fh, stderr=subprocess.STDOUT
                )
                status = "done" if returncode == 0 else f"FAILED rc={returncode}"
                print(f"{tag} {status}", flush=True)


@click.command()
@click.option("--dry-run", is_flag=True, help="Print the planned job lists and exit.")
def main(dry_run: bool) -> None:
    jobs_by_device = {dev: build_jobs(device_id=dev) for dev in (0, 1)}

    for dev, jobs in jobs_by_device.items():
        print(f"device {dev}: {len(jobs)} jobs per seed")
        for corr in jobs:
            print(f"  corr={corr:.3f}")
    total_seeds = len(candidate_seeds)
    total_jobs = sum(len(j) for j in jobs_by_device.values()) * total_seeds
    print(f"\nseeds={candidate_seeds}  total jobs across sweep: {total_jobs}")
    if dry_run:
        return

    LOG_BASE.mkdir(parents=True, exist_ok=True)

    threads = [
        Thread(
            target=run_jobs,
            kwargs=dict(
                jobs=jobs_by_device[dev],
                device_id=dev,
                seeds=list(candidate_seeds),
                log_path=LOG_BASE / f"device_{dev}.log",
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

    print("All seeds done.")


if __name__ == "__main__":
    main()
