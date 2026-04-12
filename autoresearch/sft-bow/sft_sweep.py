from __future__ import annotations

import argparse
from itertools import product
from pathlib import Path

from base_sweep import (
    DEFAULT_DATASET_BASE_FOLDER,
    DEFAULT_STUDY_ROOT,
    SFTExperimentSpec,
    float_label,
    read_json,
    run_specs_serial,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch a small LR sweep for the SNR=1 bag-of-words SFT baseline."
    )
    parser.add_argument("--stage-name", default="lr-coarse")
    parser.add_argument("--study-root", type=Path, default=DEFAULT_STUDY_ROOT)
    parser.add_argument("--dataset-base-folder", type=Path, default=DEFAULT_DATASET_BASE_FOLDER)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument(
        "--lr-per-token",
        type=float,
        nargs="+",
        default=[1.0e-8, 2.3e-8, 5.0e-8, 1.0e-7],
    )
    parser.add_argument("--backbone-lr-divisor", type=float, nargs="+", default=[5.0])
    parser.add_argument(
        "--from-recommendation",
        type=Path,
        help="Seed a fine sweep from diagnose_lr.py output.",
    )
    parser.add_argument(
        "--lr-multipliers",
        type=float,
        nargs="+",
        default=[0.5, 0.75, 1.25, 1.5],
    )
    parser.add_argument(
        "--backbone-divisor-multipliers",
        type=float,
        nargs="+",
        default=[1.0],
    )
    parser.add_argument("--snr", type=float, default=1.0)
    parser.add_argument("--aux-words-ratio", type=float, default=0.0)
    parser.add_argument("--word-decay-power", type=float, default=1.0)
    parser.add_argument("--train-epochs", type=int, default=10)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def lr_pairs(args: argparse.Namespace) -> list[tuple[float, float]]:
    if args.from_recommendation is not None:
        recommendation = read_json(args.from_recommendation)
        center_lr = float(recommendation["lr_per_token"])
        center_divisor = float(recommendation["backbone_lr_divisor"])
        return [
            (center_lr * lr_mul, center_divisor * divisor_mul)
            for lr_mul, divisor_mul in product(
                args.lr_multipliers,
                args.backbone_divisor_multipliers,
            )
        ]

    lrs = list(args.lr_per_token)
    divisors = list(args.backbone_lr_divisor)
    if len(lrs) == len(divisors):
        return list(zip(lrs, divisors, strict=True))
    if len(divisors) == 1:
        return [(lr, divisors[0]) for lr in lrs]
    if len(lrs) == 1:
        return [(lrs[0], divisor) for divisor in divisors]
    return [(lr, divisor) for lr, divisor in product(lrs, divisors)]


def main() -> None:
    args = parse_args()
    specs = [
        SFTExperimentSpec(
            run_id=(
                f"{args.stage_name}_{idx:02d}"
                f"_lpt-{float_label(lr)}_bdiv-{float_label(divisor)}"
            ),
            stage=args.stage_name,
            snr=args.snr,
            aux_words_ratio=args.aux_words_ratio,
            word_decay_power=args.word_decay_power,
            lr_per_token=lr,
            backbone_lr_divisor=divisor,
            train_epochs=args.train_epochs,
        )
        for idx, (lr, divisor) in enumerate(lr_pairs(args))
    ]
    run_specs_serial(
        specs=specs,
        device=args.device,
        dataset_base_folder=args.dataset_base_folder,
        study_root=args.study_root,
        dry_run=args.dry_run,
        num_shards=args.num_shards,
        shard_index=args.shard_index,
    )


if __name__ == "__main__":
    main()
