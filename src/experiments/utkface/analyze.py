import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import polars as pl

from src.experiments.utkface.run import write_json

TEST_METRICS = (
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
REPORT_METRICS = (*TEST_METRICS, "singleton_mae_mean", "singleton_rmse_mean")


def method_label(*, method: str) -> str:
    labels = {
        "mse": "MSE of policy mean",
        "ce": "CE / population binary MaxRL",
        "grpo_mae_population": "Population GRPO, MAE",
        "grpo_mse_population": "Population GRPO, MSE",
        "rloo_mae_population": "Population RLOO, MAE",
        "rloo_mse_population": "Population RLOO, MSE",
        "tailrl_population": "Population TailRL",
        "tailrl": "TailRL, 64 samples",
        "binary_maxrl": "Binary MaxRL, 64 samples",
    }
    if method.startswith("stat_maxrl"):
        kind = "MAE" if "_mae_" in method else "MSE"
        return f"Statistical MaxRL, {kind}, {method.rsplit('_s', 1)[1]} years"
    return labels[method]


def analyze(*, runs_dir: Path, scores_dir: Path, output_dir: Path) -> None:
    runs = [
        json.loads(path.read_text()) for path in sorted(runs_dir.glob("*/result.json"))
    ]
    if not runs or any("test" not in run for run in runs):
        raise ValueError("Analysis requires completed runs with test evaluation")
    if len({run["data_fingerprint"] for run in runs}) != 1:
        raise ValueError("Cannot compare runs with different data splits")
    records = []
    histories = []
    decades = []
    for run in runs:
        config = run["config"]
        score = json.loads(
            (scores_dir / Path(config["output_dir"]).name / "result.json").read_text()
        )
        if (
            score["data_fingerprint"] != run["data_fingerprint"]
            or score["epoch"] != run["best_epoch"]
        ):
            raise ValueError("Scored checkpoint does not match the training result")
        run["test_singletons"] = score["test_singletons"]
        run["restoration_mae_difference"] = score["restoration_mae_difference"]
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
        record.update({metric: run["test"][metric] for metric in TEST_METRICS})
        record.update({
            "singleton_mae_mean": run["test_singletons"]["mae_mean"],
            "singleton_rmse_mean": run["test_singletons"]["rmse_mean"],
        })
        records.append(record)
        for split in ("test", "test_singletons"):
            decades.extend(
                {
                    "method": method,
                    "seed": seed,
                    "split": split,
                    "decade": int(decade),
                    **metrics,
                }
                for decade, metrics in run[split]["decades"].items()
            )
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
    pl.DataFrame(decades).write_csv(output_dir / "decade_metrics.csv")
    write_json(path=output_dir / "results.json", payload=runs)
    write_json(path=output_dir / "summary.json", payload=summary.to_dicts())
    labels = summary["method"].to_list()
    figure, axes = plt.subplots(
        1, 4, figsize=(19, 8), sharey=True, layout="constrained"
    )
    for axis, metric, title in zip(
        axes,
        ("mae_mean", "rmse_mean", "expected_mae", "singleton_mae_mean"),
        (
            "Mean prediction MAE (years)",
            "Mean prediction RMSE (years)",
            "Expected sampled age MAE (years)",
            "Singleton-image test MAE (years)",
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
    axes[0].set_yticks(
        range(len(labels)), [method_label(method=method) for method in labels]
    )
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
        .agg(pl.col("val_mae").mean(), pl.col("val_mae").std().alias("sd"))
        .sort("epoch")
    )
    figure, axes = plt.subplots(
        2, 2, figsize=(13, 9), sharex=True, sharey=True, layout="constrained"
    )
    for method in labels:
        curve = curves.filter(pl.col("method") == method)
        panel = (
            2
            if method.startswith("stat_maxrl_mae")
            else 3
            if method.startswith("stat_maxrl_mse")
            else 1
            if method.startswith(("grpo", "rloo"))
            else 0
        )
        axis = axes.flat[panel]
        epochs, means = curve["epoch"].to_numpy(), curve["val_mae"].to_numpy()
        sd = curve["sd"].fill_null(0).to_numpy()
        (line,) = axis.plot(epochs, means, label=method_label(method=method))
        axis.fill_between(
            epochs, means - sd, means + sd, color=line.get_color(), alpha=0.12
        )
    for axis, title in zip(
        axes.flat,
        (
            "Supervised / TailRL / binary MaxRL",
            "Population GRPO / RLOO",
            "Statistical MaxRL: Laplace kernels",
            "Statistical MaxRL: Gaussian kernels",
        ),
        strict=True,
    ):
        axis.set(xlabel="Epoch", ylabel="Validation MAE (years)", title=title)
        axis.legend(fontsize=7, loc="upper right")
        axis.grid(alpha=0.2)
    figure.savefig(output_dir / "learning_curves.png", dpi=180)
    figure.savefig(output_dir / "learning_curves.pdf")
    plt.close(figure)
    print(summary)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", type=Path, required=True)
    parser.add_argument("--scores-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    analyze(
        runs_dir=args.runs_dir, scores_dir=args.scores_dir, output_dir=args.output_dir
    )


if __name__ == "__main__":
    main()
