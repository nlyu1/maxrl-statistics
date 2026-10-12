from __future__ import annotations

import argparse
from pathlib import Path

from src.experiments.utkface.canonical import get_canonical_config
from src.experiments.utkface.config import (
    POPULATION_METHODS,
    SAMPLED_METHODS,
    UTKFaceConfig,
)
from src.experiments.utkface.train import run_training


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="100-bin ResNet-18 UTKFace experiment")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--config", type=Path)
    source.add_argument("--data-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--method", choices=(*POPULATION_METHODS, *SAMPLED_METHODS))
    parser.add_argument("--seed", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument(
        "--batch-size", type=int, help="Global image batch across all GPUs"
    )
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--temperature-years", type=float)
    parser.add_argument("--num-rollouts", type=int)
    parser.add_argument("--degree", type=int)
    parser.add_argument("--reward", choices=("mae", "mse", "binary"))
    parser.add_argument("--survival-floor", type=float)
    parser.add_argument("--precision", choices=("bf16", "fp32"))
    parser.add_argument("--device", choices=("cuda", "cpu"))
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-eval-samples", type=int)
    parser.add_argument("--input-size", type=int)
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def configuration_from_args(*, args: argparse.Namespace) -> UTKFaceConfig:
    if args.config is not None:
        config = UTKFaceConfig.model_validate_json(args.config.read_text())
    else:
        if args.output_dir is None or args.method is None:
            raise ValueError("--data-dir requires --output-dir and --method.")
        config = get_canonical_config(
            method=args.method,
            data_directory=args.data_dir.resolve(),
            output_directory=args.output_dir.resolve(),
            seed=42 if args.seed is None else args.seed,
        )
    payload = config.model_dump(mode="json")
    updates = {
        "objective": {
            "method": args.method,
            "temperature_years": args.temperature_years,
            "num_rollouts": args.num_rollouts,
            "degree": args.degree,
            "reward": args.reward,
            "survival_floor": args.survival_floor,
        },
        "training": {
            "epochs": args.epochs,
            "global_batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "precision": args.precision,
            "device": args.device,
        },
        "data": {
            "num_workers": args.num_workers,
            "max_train_samples": args.max_train_samples,
            "max_eval_samples": args.max_eval_samples,
            "input_size": args.input_size,
        },
    }
    for section, values in updates.items():
        payload[section].update({
            key: value for key, value in values.items() if value is not None
        })
    if args.seed is not None:
        payload["seed"] = args.seed
    if args.output_dir is not None:
        payload["output_directory"] = str(args.output_dir.resolve())
    if args.no_pretrained:
        payload["pretrained"] = False
    return UTKFaceConfig.model_validate(payload)


def main() -> None:
    args = parse_args()
    run_training(config=configuration_from_args(args=args), resume=args.resume)


if __name__ == "__main__":
    main()
