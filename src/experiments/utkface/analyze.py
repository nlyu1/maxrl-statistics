from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import polars as pl

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.experiments.utkface.config import UTKFaceConfig
from src.experiments.utkface.train import atomic_json

GROUP_COLUMNS = [
    "method",
    "learning_rate",
    "temperature_years",
    "num_rollouts",
    "degree",
    "reward",
]
REPORT_METRICS = [
    "mean_mae",
    "greedy_mae",
    "expected_mae",
    "best_of_64_reward",
    "entropy",
]


def read_runs(
    *, directory: Path, allow_incomplete: bool
) -> tuple[pl.DataFrame, list[dict], dict]:
    study_path = directory / "study.json"
    if study_path.exists():
        study = json.loads(study_path.read_text())
        expected = [Path(run["output_directory"]) for run in study["runs"]]
    else:
        expected = sorted(path.parent for path in directory.rglob("config.json"))
    if not expected:
        raise ValueError("No UTKFace runs found.")
    missing = [str(path) for path in expected if not (path / "summary.json").exists()]
    if missing and not allow_incomplete:
        raise ValueError(
            f"Study is incomplete: {len(missing)} runs missing summary.json."
        )
    rows = []
    curves = []
    comparison_settings = set()
    for path in expected:
        if not (path / "summary.json").exists():
            continue
        config = UTKFaceConfig.model_validate_json((path / "config.json").read_text())
        summary = json.loads((path / "summary.json").read_text())
        settings = config.scientific_settings()
        settings.pop("objective")
        settings.pop("seed")
        settings["training"].pop("learning_rate")
        settings["world_size"] = summary["world_size"]
        comparison_settings.add(json.dumps(settings, sort_keys=True))
        group = {
            "method": config.objective.method,
            "learning_rate": config.training.learning_rate,
            "temperature_years": config.objective.temperature_years,
            "num_rollouts": config.objective.num_rollouts,
            "degree": config.objective.degree,
            "reward": config.objective.reward,
        }
        rows.append({
            **group,
            "seed": config.seed,
            "run_directory": str(path),
            "best_epoch": summary["best_epoch"],
            "validation_mean_mae": summary["best_validation_mean_mae"],
            "split_sha256": summary["split_sha256"],
            "initial_model_sha256": summary["initial_model_sha256"],
            **{f"test_{key}": value for key, value in summary["test"].items()},
        })
        curves.extend(
            {**group, "seed": config.seed, **row}
            for row in json.loads((path / "metrics.json").read_text())
        )
    if not rows:
        raise ValueError("No completed UTKFace runs found.")
    if len(comparison_settings) != 1:
        raise ValueError(
            "Compared runs differ in training budget, data, model, precision, or world size."
        )
    frame = pl.DataFrame(rows)
    if frame["split_sha256"].n_unique() != 1:
        raise ValueError("Cannot compare runs with different dataset splits.")
    initializations = frame.group_by("seed").agg(
        pl.col("initial_model_sha256").n_unique().alias("count")
    )
    if initializations.filter(pl.col("count") != 1).height:
        raise ValueError(
            "Methods with the same seed did not start from identical parameters."
        )
    if frame.group_by([*GROUP_COLUMNS, "seed"]).len().filter(pl.col("len") > 1).height:
        raise ValueError("Duplicate method/hyperparameter/seed runs.")
    if not allow_incomplete:
        seed_groups = frame.group_by(GROUP_COLUMNS).agg(
            pl.col("seed").sort().alias("seeds")
        )
        if len({tuple(seeds) for seeds in seed_groups["seeds"].to_list()}) != 1:
            raise ValueError("All comparison arms must use the same paired seeds.")
    return (
        frame,
        curves,
        {
            "planned_runs": len(expected),
            "completed_runs": len(rows),
            "missing_runs": missing,
        },
    )


def aggregate(*, runs: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    metrics = ["validation_mean_mae", *(f"test_{name}" for name in REPORT_METRICS)]
    grouped = (
        runs
        .group_by(GROUP_COLUMNS)
        .agg(
            pl.len().alias("seeds"),
            *[pl.col(name).mean().alias(f"{name}_mean") for name in metrics],
            *[pl.col(name).std(ddof=1).alias(f"{name}_sd") for name in metrics],
        )
        .sort([
            "method",
            "validation_mean_mae_mean",
            "learning_rate",
            "temperature_years",
        ])
    )
    selected = grouped.unique(subset=["method"], keep="first", maintain_order=True)
    return grouped, selected


def plot_curves(*, curves: list[dict], selected: pl.DataFrame, output: Path) -> None:
    frame = pl.DataFrame(curves, infer_schema_length=None).join(
        selected.select(GROUP_COLUMNS), on=GROUP_COLUMNS, how="inner"
    )
    metrics = [
        ("validation/mean_mae", "Mean prediction MAE (years)"),
        ("validation/expected_mae", "Expected sampled MAE (years)"),
        ("validation/best_of_64_reward", "Expected best of 64 reward"),
        ("validation/entropy", "Policy entropy (nats)"),
    ]
    figure, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    for axis, (metric, label) in zip(axes.flat, metrics, strict=True):
        statistics = (
            frame
            .group_by(["method", "epoch"])
            .agg(
                pl.col(metric).mean().alias("mean"),
                pl.col(metric).std(ddof=1).fill_null(0).alias("sd"),
            )
            .sort(["method", "epoch"])
        )
        for method in selected["method"].to_list():
            subset = statistics.filter(pl.col("method") == method)
            epochs, mean, deviation = (
                subset[name].to_numpy() for name in ("epoch", "mean", "sd")
            )
            axis.plot(epochs, mean, label=method)
            axis.fill_between(epochs, mean - deviation, mean + deviation, alpha=0.12)
        axis.set(xlabel="Epoch", ylabel=label)
        axis.grid(alpha=0.2)
    axes.flat[0].legend(fontsize=8)
    figure.savefig(output, dpi=170)
    plt.close(figure)


def write_report(*, selected: pl.DataFrame, status: dict, output: Path) -> None:
    lines = [
        "# UTKFace comparison",
        "",
        f"Completed {status['completed_runs']} of {status['planned_runs']} planned runs.",
        "",
        (
            "Hyperparameters and checkpoints were selected using validation mean-prediction MAE. "
            "Test results below are mean ± sample standard deviation across seeds; one-seed runs have no SD estimate."
        ),
        "",
        "| Method | Seeds | LR | τ (years) | Mean MAE ↓ | Greedy MAE ↓ | Expected MAE ↓ | Best-of-64 reward ↑ | Entropy |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in selected.iter_rows(named=True):
        values = []
        for metric in REPORT_METRICS:
            mean, deviation = row[f"test_{metric}_mean"], row[f"test_{metric}_sd"]
            values.append(
                f"{mean:.4f}" if deviation is None else f"{mean:.4f} ± {deviation:.4f}"
            )
        lines.append(
            f"| {row['method']} | {row['seeds']} | {row['learning_rate']:g} | {row['temperature_years']:g} | "
            + " | ".join(values)
            + " |"
        )
    lines.extend([
        "",
        (
            "CE and population zero-one MaxRL are mathematically identical. "
            "MSE trains the categorical mean and does not constrain the sampled policy's variance. "
            "Best-of-64 reward assumes an oracle chooses the sampled age closest to the label; it is not a deployable prediction rule."
        ),
        "",
        (
            "A common learning rate is a controlled comparison, but objective gradient scales differ. "
            "A validation-selected learning-rate grid is needed before claiming one objective is better."
        ),
        "",
        "![Validation curves](validation_curves.png)",
        "",
    ])
    output.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate UTKFace seeds and select hyperparameters using validation only"
    )
    parser.add_argument("--study-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    runs, curves, status = read_runs(
        directory=args.study_dir.resolve(), allow_incomplete=args.allow_incomplete
    )
    grouped, selected = aggregate(runs=runs)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    runs.write_csv(args.output_dir / "runs.csv")
    grouped.write_csv(args.output_dir / "all_hyperparameters.csv")
    selected.write_csv(args.output_dir / "selected.csv")
    atomic_json(path=args.output_dir / "status.json", value=status)
    plot_curves(
        curves=curves,
        selected=selected,
        output=args.output_dir / "validation_curves.png",
    )
    write_report(selected=selected, status=status, output=args.output_dir / "report.md")
    print(selected, flush=True)


if __name__ == "__main__":
    main()
