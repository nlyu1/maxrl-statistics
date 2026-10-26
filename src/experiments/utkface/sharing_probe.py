import argparse
import fcntl
import json
import os
import signal
import subprocess
import time
from pathlib import Path
from statistics import mean

import torch

from src.experiments.utkface.config import NoiseConfig, RunConfig, canonical_config
from src.experiments.utkface.distributed import distributed_context
from src.experiments.utkface.run import train, write_json
from src.experiments.utkface.scheduler import gpu_inventory, gpu_is_free


def worker(*, config_path: Path, start_path: Path) -> None:
    config = RunConfig.model_validate_json(config_path.read_text())
    directory = config_path.parent
    with (
        distributed_context(device=config.device) as context,
        torch.cuda.device(context.device),
    ):
        torch.empty(1, device=context.device)
        write_json(path=directory / "ready.json", payload={"pid": os.getpid()})
        deadline = time.monotonic() + 180
        while not start_path.exists():
            if time.monotonic() > deadline:
                raise TimeoutError("Cohort did not start")
            time.sleep(0.1)
        torch.cuda.reset_peak_memory_stats(context.device)
        started = time.monotonic()
        result = train(config=config, context=context)
        torch.cuda.synchronize(context.device)
        history = json.loads((config.output_dir / "history.json").read_text())
        write_json(
            path=directory / "probe-result.json",
            payload={
                "pid": os.getpid(),
                "wall_seconds": time.monotonic() - started,
                "epoch_seconds": [row["seconds"] for row in history],
                "steady_epoch_seconds": mean(row["seconds"] for row in history[1:]),
                "peak_allocated_mib": torch.cuda.max_memory_allocated(context.device)
                / 2**20,
                "peak_reserved_mib": torch.cuda.max_memory_reserved(context.device)
                / 2**20,
                "validation": result["validation"],
                "test_evaluated": False,
            },
        )


def cohort(
    *,
    root: Path,
    manifest: Path,
    gpu_index: int,
    method: str,
    concurrency: int,
    epochs: int,
    index: int,
) -> dict:
    directory = root / f"{index:02d}-{method}-jobs{concurrency}"
    directory.mkdir()
    start_path = directory / "start"
    selected = next(gpu for gpu in gpu_inventory() if gpu["index"] == gpu_index)
    release_deadline = time.monotonic() + 15
    while not selected["pids"] and not gpu_is_free(gpu=selected):
        if time.monotonic() > release_deadline:
            break
        time.sleep(1)
        selected = next(gpu for gpu in gpu_inventory() if gpu["index"] == gpu_index)
    if not gpu_is_free(gpu=selected):
        raise RuntimeError(f"GPU became occupied before cohort launch: {selected}")
    processes = []
    handles = []
    peak_used_mib = selected["memory_mib"]
    utilizations = []
    started = time.monotonic()
    try:
        for replica in range(concurrency):
            run_root = directory / f"replica{replica}"
            run_root.mkdir()
            base = canonical_config(manifest=manifest, output_dir=run_root / "run")
            config = base.model_copy(
                update={
                    "device": f"cuda:{gpu_index}",
                    "epochs": epochs,
                    "train_limit": 4096,
                    "learning_rate": 1e-3,
                    "loss": base.loss.model_copy(update={"method": method}),
                    "noise": NoiseConfig(mode="fixed", std_years=10, seed=2026),
                    "evaluate_test": False,
                    "save_checkpoint": False,
                }
            )
            config_path = run_root / "worker-config.json"
            write_json(path=config_path, payload=config.model_dump(mode="json"))
            handle = (run_root / "worker.log").open("w")
            handles.append(handle)
            processes.append(
                subprocess.Popen(
                    [
                        "uv",
                        "run",
                        "--no-sync",
                        "python",
                        "-u",
                        "-m",
                        "src.experiments.utkface.sharing_probe",
                        "--worker-config",
                        str(config_path),
                        "--start-path",
                        str(start_path),
                    ],
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
            )
        deadline = time.monotonic() + 150
        while not all(
            (directory / f"replica{i}" / "ready.json").exists()
            for i in range(concurrency)
        ):
            if any(process.poll() is not None for process in processes):
                raise RuntimeError(
                    f"Worker exited before readiness; inspect {directory}"
                )
            if time.monotonic() > deadline:
                raise TimeoutError(f"Worker readiness timed out: {directory}")
            time.sleep(0.2)
        owned = {
            json.loads((directory / f"replica{i}" / "ready.json").read_text())["pid"]
            for i in range(concurrency)
        }
        selected = next(gpu for gpu in gpu_inventory() if gpu["index"] == gpu_index)
        if set(selected["pids"]) - owned:
            raise RuntimeError(f"Another workload appeared on GPU: {selected}")
        cohort_started = time.monotonic()
        start_path.touch()
        while any(process.poll() is None for process in processes):
            selected = next(gpu for gpu in gpu_inventory() if gpu["index"] == gpu_index)
            if set(selected["pids"]) - owned:
                raise RuntimeError(f"Another workload appeared on GPU: {selected}")
            peak_used_mib = max(peak_used_mib, selected["memory_mib"])
            utilizations.append(selected["utilization"])
            if time.monotonic() - cohort_started > 600:
                raise TimeoutError(f"Cohort timed out: {directory}")
            time.sleep(0.5)
        codes = [process.wait() for process in processes]
        if any(code != 0 for code in codes):
            raise RuntimeError(f"Worker failure {codes}; inspect {directory}")
        results = [
            json.loads((directory / f"replica{i}" / "probe-result.json").read_text())
            for i in range(concurrency)
        ]
        slowest_epoch = max(row["steady_epoch_seconds"] for row in results)
        summary = {
            "method": method,
            "concurrency": concurrency,
            "epochs": epochs,
            "excluded_warmup_epochs": 1,
            "replicas": results,
            "mean_job_epoch_seconds": mean(
                row["steady_epoch_seconds"] for row in results
            ),
            "slowest_job_epoch_seconds": slowest_epoch,
            "aggregate_images_per_second": concurrency * 4096 / slowest_epoch,
            "peak_device_used_mib": peak_used_mib,
            "mean_sampled_gpu_utilization": mean(utilizations),
            "cohort_seconds_after_barrier": time.monotonic() - cohort_started,
            "cohort_seconds_including_startup": time.monotonic() - started,
        }
        write_json(path=directory / "summary.json", payload=summary)
        return summary
    finally:
        for process in processes:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                process.wait(timeout=30)
        for handle in handles:
            handle.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker-config", type=Path)
    parser.add_argument("--start-path", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--gpu", type=int)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--max-concurrency", type=int, choices=(4, 8), default=4)
    parser.add_argument("--dispatch-lock", type=Path)
    args = parser.parse_args()
    if args.worker_config is not None:
        if args.start_path is None:
            raise ValueError("Worker requires --start-path")
        worker(config_path=args.worker_config, start_path=args.start_path)
        return
    if any(
        value is None
        for value in (
            args.manifest,
            args.output_dir,
            args.gpu,
            args.dispatch_lock,
        )
    ):
        raise ValueError("Probe requires manifest, output-dir, gpu and dispatch-lock")
    if args.epochs < 3:
        raise ValueError("At least three epochs are required")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    args.dispatch_lock.parent.mkdir(parents=True, exist_ok=True)
    with args.dispatch_lock.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        first = next(gpu for gpu in gpu_inventory() if gpu["index"] == args.gpu)
        if not gpu_is_free(gpu=first):
            raise RuntimeError(f"Selected GPU is occupied: {first}")
        time.sleep(10)
        second = next(gpu for gpu in gpu_inventory() if gpu["index"] == args.gpu)
        if not gpu_is_free(gpu=second):
            raise RuntimeError(f"Selected GPU became occupied: {second}")
        sequence = (
            ("mse", 1),
            ("mse", 2),
            ("mse", 4),
            ("tailrl_population", 4),
            ("tailrl_population", 2),
            ("tailrl_population", 1),
            ("mse", 1),
        )
        if args.max_concurrency == 8:
            sequence = (
                ("mse", 4),
                ("mse", 8),
                ("tailrl_population", 8),
                ("tailrl_population", 4),
            )
        summaries = []
        write_json(
            path=args.output_dir / "design.json",
            payload={
                "gpu": second,
                "epochs": args.epochs,
                "sequence": sequence,
                "scope": "Throughput and memory only; no test evaluation or checkpoints",
                "git_commit": subprocess.check_output(
                    ["git", "rev-parse", "HEAD"],
                    text=True,
                ).strip(),
            },
        )
        for index, (method, concurrency) in enumerate(sequence):
            print(
                json.dumps({
                    "event": "cohort_started",
                    "method": method,
                    "concurrency": concurrency,
                }),
                flush=True,
            )
            result = cohort(
                root=args.output_dir,
                manifest=args.manifest,
                gpu_index=args.gpu,
                method=method,
                concurrency=concurrency,
                epochs=args.epochs,
                index=index,
            )
            summaries.append(result)
            write_json(
                path=args.output_dir / "summary.json", payload={"cohorts": summaries}
            )
            print(
                json.dumps({
                    key: value for key, value in result.items() if key != "replicas"
                }),
                flush=True,
            )


if __name__ == "__main__":
    main()
