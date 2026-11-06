import argparse
import fcntl
import json
import multiprocessing
import os
import subprocess
import time
from collections import deque
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path

import numpy as np

from src.experiments.bernoulli_calibration.config import (
    LEARNING_RATES,
    LR_EXTENSION_FACTOR,
    LR_EXTENSION_ROUNDS,
    METHODS,
    ROLLOUTS,
    SEEDS,
    SUPERVISED,
    TRAIN_SIZES,
    RunConfig,
    canonical_dataset,
    canonical_run,
)
from src.experiments.bernoulli_calibration.data import prepare_dataset
from src.experiments.bernoulli_calibration.run import train
from src.experiments.utkface.run import write_json
from src.experiments.utkface.scheduler import gpu_inventory, gpu_is_free


def families(*, root: Path, train_size: int, device: str) -> list[RunConfig]:
    base = canonical_run(root=root)
    result = []
    for method in METHODS:
        for protocol in ("shared", "split"):
            counts = (
                (16,) if method in SUPERVISED and protocol == "shared" else ROLLOUTS
            )
            for count in counts:
                result.append(
                    base.model_copy(
                        update={
                            "method": method,
                            "protocol": protocol,
                            "num_rollouts": count,
                            "train_size": train_size,
                            "device": device,
                        }
                    )
                )
    return result


def execute(
    *, jobs: list[RunConfig], executor: ProcessPoolExecutor, slots: int, root: Path
) -> list[dict]:
    remaining = deque(jobs)
    active = {}
    results = []
    for _ in range(min(slots, len(jobs))):
        config = remaining.popleft()
        active[executor.submit(train, config=config)] = config.identifier
    while active:
        completed, _ = wait(active, return_when=FIRST_COMPLETED)
        for future in completed:
            identifier = active.pop(future)
            result = future.result()
            results.append(result)
            print(
                json.dumps({
                    "completed": len(results),
                    "total": len(jobs),
                    "identifier": identifier,
                    "seconds": result["seconds"],
                    "validation_native_loss": result["final"]["validation"][
                        "native_loss"
                    ],
                    "validation_rmse": result["final"]["validation"]["rmse"],
                }),
                flush=True,
            )
            write_json(
                path=root / "progress.json",
                payload={
                    "completed": len(results),
                    "total": len(jobs),
                    "last_completed": identifier,
                    "active": list(active.values()),
                },
            )
            if remaining:
                config = remaining.popleft()
                active[executor.submit(train, config=config)] = config.identifier
    return results


def select_rates(
    *, results: list[dict], root: Path, train_size: int
) -> dict[str, float]:
    grouped: dict[str, list[dict]] = {}
    for result in results:
        grouped.setdefault(result["family"], []).append(result)
    selection = {}
    for family, candidates in grouped.items():
        candidates.sort(
            key=lambda result: (
                result["final"]["validation"]["native_loss"],
                result["config"]["learning_rate"],
            )
        )
        winner = candidates[0]
        selection[family] = {
            "learning_rate": winner["config"]["learning_rate"],
            "criterion": "final_validation_native_loss",
            "native_loss": winner["final"]["validation"]["native_loss"],
            "identifier": winner["config"]["identifier"],
            "grid_boundary": winner["config"]["learning_rate"]
            in (
                min(item["config"]["learning_rate"] for item in candidates),
                max(item["config"]["learning_rate"] for item in candidates),
            ),
            "candidates": [
                {
                    "learning_rate": item["config"]["learning_rate"],
                    "native_loss": item["final"]["validation"]["native_loss"],
                }
                for item in candidates
            ],
        }
    write_json(path=root / f"selection-train{train_size}.json", payload=selection)
    return {family: row["learning_rate"] for family, row in selection.items()}


def resizing_diagnostic(*, results: list[dict]) -> dict:
    checks = []
    for method in ("ce", "mse"):
        rows = [
            result
            for result in results
            if result["config"]["method"] == method
            and result["config"]["protocol"] == "shared"
        ]
        gaps = [
            row["final"]["validation"]["rmse"] > row["final"]["train"]["rmse"] + 0.02
            and row["final"]["validation"]["rmse"] > 1.5 * row["final"]["train"]["rmse"]
            for row in rows
        ]
        deviation = float(
            np.std([row["final"]["validation"]["rmse"] for row in rows], ddof=1)
        )
        checks.append({
            "method": method,
            "heavy_overfit_seeds": sum(gaps),
            "validation_rmse_seed_sd": deviation,
            "triggered": sum(gaps) >= 2 or deviation > 0.02,
        })
    return {
        "checks": checks,
        "increase_training_set": any(row["triggered"] for row in checks),
    }


def run_study(
    *, root: Path, gpu: int, jobs_per_gpu: int, pilot_steps: int, main_steps: int
) -> None:
    root.mkdir(parents=True, exist_ok=True)
    inventory = gpu_inventory()
    target = next(row for row in inventory if row["index"] == gpu)
    if not gpu_is_free(gpu=target):
        raise RuntimeError(f"Requested GPU is occupied: {target}")
    claim_dir = root.parent / "gpu-claims"
    claim_dir.mkdir(exist_ok=True)
    with (claim_dir / f"{target['uuid']}.lock").open("a") as claim:
        fcntl.flock(claim, fcntl.LOCK_EX | fcntl.LOCK_NB)
        prepare_dataset(root=root, config=canonical_dataset())
        provenance = {
            "source_revision": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip(),
            "gpu": target,
            "jobs_per_gpu": jobs_per_gpu,
            "pilot_steps": pilot_steps,
            "main_steps": main_steps,
            "learning_rates": LEARNING_RATES,
            "rollouts": ROLLOUTS,
            "seeds": SEEDS,
            "started_at": time.time(),
            "pid": os.getpid(),
        }
        write_json(path=root / "study.json", payload=provenance)
        with ProcessPoolExecutor(
            max_workers=jobs_per_gpu, mp_context=multiprocessing.get_context("spawn")
        ) as executor:
            for train_size in TRAIN_SIZES:
                configurations = families(
                    root=root, train_size=train_size, device=f"cuda:{gpu}"
                )
                pilot = []
                for base in configurations:
                    for rate in LEARNING_RATES:
                        pilot.append(
                            base.model_copy(
                                update={
                                    "stage": "pilot",
                                    "learning_rate": rate,
                                    "steps": pilot_steps,
                                    "identifier": f"pilot-t{train_size}-{base.family}-lr{rate:g}",
                                }
                            )
                        )
                pilot_results = execute(
                    jobs=pilot, executor=executor, slots=jobs_per_gpu, root=root
                )
                selected = select_rates(
                    results=pilot_results, root=root, train_size=train_size
                )
                for _ in range(LR_EXTENSION_ROUNDS):
                    extension_jobs = []
                    for base in configurations:
                        rates = sorted(
                            result["config"]["learning_rate"]
                            for result in pilot_results
                            if result["family"] == base.family
                        )
                        winner = selected[base.family]
                        if winner not in (rates[0], rates[-1]):
                            continue
                        rate = (
                            winner / LR_EXTENSION_FACTOR
                            if winner == rates[0]
                            else winner * LR_EXTENSION_FACTOR
                        )
                        extension_jobs.append(
                            base.model_copy(
                                update={
                                    "stage": "pilot",
                                    "learning_rate": rate,
                                    "steps": pilot_steps,
                                    "identifier": f"pilot-t{train_size}-{base.family}-lr{rate:g}",
                                }
                            )
                        )
                    if not extension_jobs:
                        break
                    pilot_results.extend(
                        execute(
                            jobs=extension_jobs,
                            executor=executor,
                            slots=jobs_per_gpu,
                            root=root,
                        )
                    )
                    selected = select_rates(
                        results=pilot_results, root=root, train_size=train_size
                    )
                main_jobs = [
                    base.model_copy(
                        update={
                            "stage": "main",
                            "steps": main_steps,
                            "seed": seed,
                            "learning_rate": selected[base.family],
                            "identifier": f"main-t{train_size}-{base.family}-s{seed}",
                        }
                    )
                    for base in configurations
                    for seed in SEEDS
                ]
                main_results = execute(
                    jobs=main_jobs, executor=executor, slots=jobs_per_gpu, root=root
                )
                diagnostic = resizing_diagnostic(results=main_results)
                write_json(
                    path=root / f"resizing-train{train_size}.json", payload=diagnostic
                )
                if not diagnostic["increase_training_set"]:
                    break
        write_json(
            path=root / "completion.json",
            payload={
                "completed_at": time.time(),
                "train_size": train_size,
                "resizing_diagnostic": diagnostic,
            },
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--jobs-per-gpu", type=int, required=True)
    parser.add_argument("--pilot-steps", type=int, default=256)
    parser.add_argument("--main-steps", type=int, default=2048)
    arguments = parser.parse_args()
    run_study(
        root=arguments.root,
        gpu=arguments.gpu,
        jobs_per_gpu=arguments.jobs_per_gpu,
        pilot_steps=arguments.pilot_steps,
        main_steps=arguments.main_steps,
    )


if __name__ == "__main__":
    main()
