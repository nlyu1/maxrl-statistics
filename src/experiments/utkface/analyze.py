import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import polars as pl

from src.experiments.utkface.run import write_json

REPORT_METRICS = (
    "mae_mean",
    "rmse_mean",
    "mae_median",
    "mae_mode",
    "expected_mae",
    "entropy",
    "ce",
    "tailrl",
    "oracle_best_of_64_mae",
)


def analyze(*, runs_dir: Path, output_dir: Path) -> None:
    runs = [
        json.loads(path.read_text()) for path in sorted(runs_dir.glob("*/result.json"))
    ]
    if not runs or any("test" not in run for run in runs):
        raise ValueError("Analysis requires completed runs with test evaluation")
    if len({run["data_fingerprint"] for run in runs}) != 1:
        raise ValueError("Cannot compare runs with different data splits")
    records = []
    histories = []
    for run in runs:
        config = run["config"]
        method, seed = config["loss"]["method"], config["seed"]
        if method.startswith("stat_maxrl"):
            method += f"_s{config['loss']['kernel_scale_years']:g}"
        record = {
            "method": method,
            "seed": seed,
            "kernel_scale_years": config["loss"]["kernel_scale_years"],
            "learning_rate": config["learning_rate"],
            "best_epoch": run["best_epoch"],
            "seconds": run["seconds"],
            "val_mae": run["validation"]["mae_mean"],
        }
        record.update({metric: run["test"][metric] for metric in REPORT_METRICS})
        records.append(record)
        history = json.loads(
            (runs_dir / Path(config["output_dir"]).name / "history.json").read_text()
        )
        histories.extend(
            {
                "method": method,
                "seed": seed,
                "epoch": row["epoch"],
                "val_mae": row["validation"]["mae_mean"],
                "seconds": row["seconds"],
            }
            for row in history
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = pl.DataFrame(records).sort("method", "seed")
    summary = (
        frame
        .group_by("method")
        .agg(
            pl.len().alias("seeds"),
            *[
                expression
                for metric in REPORT_METRICS
                for expression in (
                    pl.col(metric).mean().alias(metric + "_mean"),
                    pl.col(metric).std().alias(metric + "_sd"),
                )
            ],
        )
        .sort("mae_mean_mean")
    )
    frame.write_csv(output_dir / "runs.csv")
    summary.write_csv(output_dir / "summary.csv")
    pl.DataFrame(histories).write_csv(output_dir / "learning_curves.csv")
    write_json(path=output_dir / "results.json", payload=runs)
    write_json(path=output_dir / "summary.json", payload=summary.to_dicts())
    labels = summary["method"].to_list()
    figure, axes = plt.subplots(
        1, 3, figsize=(15, 7), sharey=True, layout="constrained"
    )
    for axis, metric, title in zip(
        axes,
        ("mae_mean", "rmse_mean", "expected_mae"),
        (
            "Mean prediction MAE (years)",
            "Mean prediction RMSE (years)",
            "Expected sampled age MAE (years)",
        ),
        strict=True,
    ):
        axis.barh(
            range(len(labels)),
            summary[metric + "_mean"].to_list(),
            xerr=summary[metric + "_sd"].fill_null(0).to_list(),
            capsize=3,
        )
        axis.set_xlabel(title)
        axis.grid(axis="x", alpha=0.2)
    axes[0].set_yticks(range(len(labels)), labels)
    axes[0].invert_yaxis()
    figure.suptitle(
        "UTKFace, ResNet-18 / 100-bin policy; bars show mean ± sample SD across seeds"
    )
    figure.savefig(output_dir / "comparison.png", dpi=180)
    figure.savefig(output_dir / "comparison.pdf")
    plt.close(figure)
    curves = (
        pl
        .DataFrame(histories)
        .group_by("method", "epoch")
        .agg(pl.col("val_mae").mean())
        .sort("epoch")
    )
    figure, axis = plt.subplots(figsize=(10, 6), layout="constrained")
    for method in labels:
        curve = curves.filter(pl.col("method") == method)
        axis.plot(curve["epoch"].to_list(), curve["val_mae"].to_list(), label=method)
    axis.set(xlabel="Epoch", ylabel="Validation mean prediction MAE (years)")
    axis.legend(fontsize=8, loc="upper right")
    axis.grid(alpha=0.2)
    figure.savefig(output_dir / "learning_curves.png", dpi=180)
    figure.savefig(output_dir / "learning_curves.pdf")
    plt.close(figure)
    print(summary)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    analyze(runs_dir=args.runs_dir, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
