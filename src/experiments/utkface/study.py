from __future__ import annotations

import argparse
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from src.experiments.utkface.canonical import get_canonical_config
from src.experiments.utkface.config import (
    POPULATION_METHODS,
    SAMPLED_METHODS,
    UTKFaceConfig,
)
from src.experiments.utkface.train import atomic_json


def make_study(*, args: argparse.Namespace) -> list[UTKFaceConfig]:
    configs = []
    for method in args.methods:
        for seed in args.seeds:
            canonical = get_canonical_config(
                method=method,
                data_directory=args.data_dir.resolve(),
                output_directory=args.output_dir.resolve(),
                seed=seed,
            )
            rates = args.learning_rates or [canonical.training.learning_rate]
            temperatures = args.temperatures or [canonical.objective.temperature_years]
            if not method.startswith("stat_maxrl"):
                temperatures = [canonical.objective.temperature_years]
            for rate in rates:
                for temperature in temperatures:
                    payload = canonical.model_dump(mode="json")
                    payload["objective"].update(temperature_years=temperature)
                    payload["training"].update(learning_rate=rate)
                    updates = {
                        "training": {
                            "epochs": args.epochs,
                            "global_batch_size": args.batch_size,
                        },
                        "data": {
                            "num_workers": args.num_workers,
                            "input_size": args.input_size,
                            "max_train_samples": args.max_train_samples,
                            "max_eval_samples": args.max_eval_samples,
                        },
                        "objective": {
                            "num_rollouts": args.num_rollouts,
                            "degree": args.degree,
                        },
                    }
                    for section, values in updates.items():
                        payload[section].update({
                            key: value
                            for key, value in values.items()
                            if value is not None
                        })
                    identifier = f"{method}/lr_{rate:g}_tau_{temperature:g}/seed_{seed}"
                    payload["output_directory"] = str(
                        args.output_dir.resolve() / "runs" / identifier
                    )
                    configs.append(UTKFaceConfig.model_validate(payload))
    if len({config.output_directory for config in configs}) != len(configs):
        raise ValueError("Duplicate methods, seeds, or hyperparameters in study.")
    return configs


def run_group(
    *, configs: list[UTKFaceConfig], gpu_group: str, study_directory: Path
) -> None:
    devices = gpu_group.split(",")
    if not devices or any(not device.isdigit() for device in devices):
        raise ValueError(
            f"GPU groups must contain comma-separated device indices: {gpu_group}"
        )
    for config in configs:
        identifier = config.output_directory.relative_to(study_directory / "runs")
        config_path = study_directory / "configs" / identifier.with_suffix(".json")
        log_path = study_directory / "logs" / identifier.with_suffix(".log")
        config_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        atomic_json(path=config_path, value=config.model_dump(mode="json"))
        command = ["uv", "run", "--no-sync", "python"]
        if len(devices) > 1:
            command += [
                "-m",
                "torch.distributed.run",
                "--standalone",
                "--nnodes=1",
                f"--nproc-per-node={len(devices)}",
            ]
        command += [
            "-m",
            "src.experiments.utkface",
            "--config",
            str(config_path),
            "--resume",
        ]
        print(f"Starting {identifier} on GPU(s) {gpu_group}", flush=True)
        with log_path.open("a") as log:
            subprocess.run(
                command,
                env={**os.environ, "CUDA_VISIBLE_DEVICES": gpu_group},
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )
        print(f"Finished {identifier}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Matched UTKFace study with optional validation hyperparameter grid"
    )
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--gpu-groups",
        nargs="+",
        required=True,
        help="Independent workers, e.g. 0,1 2,3",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=(*POPULATION_METHODS, *SAMPLED_METHODS),
        default=POPULATION_METHODS,
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=(42, 43, 44))
    parser.add_argument("--learning-rates", type=float, nargs="+")
    parser.add_argument("--temperatures", type=float, nargs="+")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--input-size", type=int)
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-eval-samples", type=int)
    parser.add_argument("--num-rollouts", type=int)
    parser.add_argument("--degree", type=int)
    args = parser.parse_args()
    groups = [group.split(",") for group in args.gpu_groups]
    device_indices = [index for group in groups for index in group]
    if any(not index.isdigit() for index in device_indices):
        raise ValueError("GPU groups require comma-separated device indices.")
    if len(set(device_indices)) != len(device_indices):
        raise ValueError("Independent GPU groups must not overlap.")
    if len({len(group) for group in groups}) != 1:
        raise ValueError("Matched study workers must use the same number of GPUs.")
    configs = make_study(args=args)
    for config in configs:
        for group in groups:
            if config.training.global_batch_size % len(group):
                raise ValueError(
                    "Global batch size must divide evenly over each GPU group."
                )
    directory = args.output_dir.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    atomic_json(
        path=directory / "study.json",
        value={
            "runs": [config.model_dump(mode="json") for config in configs],
            "gpu_groups": args.gpu_groups,
        },
    )
    with ThreadPoolExecutor(max_workers=len(groups)) as pool:
        futures = [
            pool.submit(
                run_group,
                configs=configs[index :: len(groups)],
                gpu_group=group,
                study_directory=directory,
            )
            for index, group in enumerate(args.gpu_groups)
        ]
        for future in futures:
            future.result()


if __name__ == "__main__":
    main()
