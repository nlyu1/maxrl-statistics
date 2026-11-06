import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import polars as pl
from plotly.subplots import make_subplots

from src.experiments.bernoulli_calibration.config import METHODS, ROLLOUTS, SUPERVISED

COLORS = dict(
    zip(
        METHODS,
        ("#333333", "#16806a", "#ab65ad", "#c66a24", "#b64747", "#327fc0"),
        strict=True,
    )
)


def representative(*, result: dict, protocol: str, rollouts: int) -> bool:
    config = result["config"]
    return config["protocol"] == protocol and (
        config["num_rollouts"] == rollouts
        or protocol == "shared"
        and config["method"] in SUPERVISED
    )


def report(*, root: Path) -> None:
    completion = json.loads((root / "completion.json").read_text())
    train_size = completion["train_size"]
    paths = sorted((root / "runs").glob(f"main-t{train_size}-*/result.json"))
    results = [json.loads(path.read_text()) for path in paths]
    if not results:
        raise RuntimeError("No completed main runs")
    destination = root / "report"
    destination.mkdir(exist_ok=True)
    histories = {
        result["config"]["identifier"]: json.loads(
            (path.parent / "history.json").read_text()
        )
        for path, result in zip(paths, results, strict=True)
    }
    rows = []
    curve_rows = []
    for result in results:
        config = result["config"]
        identifiers = {
            key: config[key]
            for key in (
                "method",
                "protocol",
                "num_rollouts",
                "seed",
                "learning_rate",
                "train_size",
            )
        }
        rows.append({
            **identifiers,
            "best_native_step": result["best_native_step"],
            "seconds": result["seconds"],
            **{f"final_{key}": value for key, value in result["test_final"].items()},
            **{
                f"best_native_{key}": value
                for key, value in result["test_best_native"].items()
            },
        })
        for row in histories[config["identifier"]]:
            curve_rows.append({
                **identifiers,
                "step": row["step"],
                "epochs": row["epochs"],
                **{
                    f"{split}_{key}": value
                    for split in ("train", "validation", "diagnostics")
                    for key, value in row[split].items()
                },
            })
    table = pl.DataFrame(rows)
    table.write_csv(destination / "results.csv")
    table.write_parquet(destination / "results.parquet")
    pl.DataFrame(curve_rows, infer_schema_length=None).write_parquet(
        destination / "curves.parquet"
    )
    summary = (
        table
        .group_by("method", "protocol", "num_rollouts")
        .agg(
            pl.len().alias("seeds"),
            *[
                expression
                for metric in (
                    "final_mae",
                    "final_rmse",
                    "final_pearson",
                    "final_entropy",
                    "best_native_rmse",
                    "final_expected_reward",
                )
                for expression in (
                    pl.col(metric).mean().alias(f"{metric}_mean"),
                    pl.col(metric).std().alias(f"{metric}_sd"),
                )
            ],
        )
        .sort("protocol", "method", "num_rollouts")
    )
    summary.write_csv(destination / "summary.csv")
    plt.rcParams.update({
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
    for protocol in ("shared", "split"):
        selected = [
            row
            for row in results
            if representative(result=row, protocol=protocol, rollouts=64)
        ]
        figure, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
        interactive = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=(
                "Probability MAE",
                "Probability RMSE",
                "Pearson correlation",
                "Bernoulli entropy",
            ),
        )
        for panel, (axis, metric) in enumerate(
            zip(axes.flat, ("mae", "rmse", "pearson", "entropy"), strict=True)
        ):
            for method in METHODS:
                candidates = [
                    row for row in selected if row["config"]["method"] == method
                ]
                histories_for_method = [
                    histories[row["config"]["identifier"]] for row in candidates
                ]
                steps = [entry["step"] for entry in histories_for_method[0]]
                values = np.array([
                    [entry["validation"][metric] for entry in history]
                    for history in histories_for_method
                ])
                average, deviation = values.mean(0), values.std(0, ddof=1)
                axis.plot(steps, average, label=method, color=COLORS[method])
                axis.fill_between(
                    steps,
                    average - deviation,
                    average + deviation,
                    color=COLORS[method],
                    alpha=0.12,
                )
                interactive.add_trace(
                    go.Scatter(
                        x=steps,
                        y=average,
                        name=method,
                        legendgroup=method,
                        showlegend=panel == 0,
                        line={"color": COLORS[method]},
                    ),
                    row=panel // 2 + 1,
                    col=panel % 2 + 1,
                )
            if metric == "entropy":
                truth = selected[0]["final"]["validation"]["true_entropy"]
                axis.axhline(
                    truth, color="#777777", linestyle="--", label="true entropy"
                )
            axis.set(xlabel="Optimizer update", ylabel=metric)
            axis.grid(alpha=0.2)
        axes[0, 0].legend()
        figure.suptitle(
            f"Validation curves: {protocol} labels, N=64, train={train_size}; mean +/- seed SD"
        )
        figure.savefig(destination / f"curves-{protocol}.png", dpi=160)
        figure.savefig(destination / f"curves-{protocol}.svg")
        plt.close(figure)
        interactive.update_layout(
            title=f"{protocol} labels, N=64; validation seed means", height=750
        )
        interactive.write_html(
            destination / f"curves-{protocol}.html", include_plotlyjs=True
        )
        figure, axes = plt.subplots(2, 3, figsize=(12, 8), constrained_layout=True)
        for axis, method in zip(axes.flat, METHODS, strict=True):
            candidates = [row for row in selected if row["config"]["method"] == method]
            for result in candidates:
                predictions = pl.read_parquet(
                    root
                    / "runs"
                    / result["config"]["identifier"]
                    / "test-predictions.parquet"
                )
                axis.scatter(
                    predictions["f"],
                    predictions["q_final"],
                    s=2,
                    alpha=0.025,
                    color=COLORS[method],
                    rasterized=True,
                )
                grouped = (
                    predictions.group_by("f").agg(pl.col("q_final").mean()).sort("f")
                )
                axis.plot(
                    grouped["f"],
                    grouped["q_final"],
                    ".",
                    color=COLORS[method],
                    alpha=0.45,
                    markersize=3,
                )
            axis.plot([0, 1], [0, 1], "--", color="#888888", linewidth=1)
            axis.set(
                title=method,
                xlabel="Ground truth f(x)",
                ylabel="Model probability",
                xlim=(0, 1),
                ylim=(0, 1),
            )
        figure.suptitle(
            f"Test calibration: {protocol} labels, N=64; final checkpoints, all seeds"
        )
        figure.savefig(destination / f"scatter-{protocol}.png", dpi=160)
        plt.close(figure)
    figure, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    for axis, protocol in zip(axes, ("shared", "split"), strict=True):
        for method in ("rloo", "grpo", "maxrl"):
            candidates = summary.filter(
                (pl.col("method") == method) & (pl.col("protocol") == protocol)
            ).sort("num_rollouts")
            axis.errorbar(
                candidates["num_rollouts"],
                candidates["final_rmse_mean"],
                yerr=candidates["final_rmse_sd"],
                marker="o",
                label=method,
                color=COLORS[method],
            )
        ce = summary.filter(
            (pl.col("method") == "ce") & (pl.col("protocol") == protocol)
        )
        axis.plot(
            ROLLOUTS if protocol == "shared" else ce["num_rollouts"],
            np.repeat(ce["final_rmse_mean"][0], len(ROLLOUTS))
            if protocol == "shared"
            else ce["final_rmse_mean"],
            "--",
            color=COLORS["ce"],
            label="ce",
        )
        axis.set(
            title=protocol,
            xlabel="Rollouts per context",
            ylabel="Final test probability RMSE",
            xscale="log",
            xticks=ROLLOUTS,
        )
        axis.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        axis.legend()
        axis.grid(alpha=0.2)
    figure.savefig(destination / "rollout-comparison.png", dpi=160)
    plt.close(figure)
    lines = [
        "# Bernoulli bag-of-words calibration",
        "",
        f"Training bags: {train_size}; validation: 8192; test: 8192. Three matched seeds.",
        "",
        "Learning rates use final pilot validation native loss. GRPO/RLOO use expected mismatch; MaxRL uses its finite-compute objective. Calibration metrics never select rates or checkpoints.",
        "",
        "Curves include step zero. Shading is seed standard deviation. Scatters show raw predictions with within-probability means separately per seed. Raw predictions are retained for every main run.",
        "",
        "| Protocol | Method | N | Final test RMSE | Best native checkpoint RMSE | Final test Pearson |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in summary.iter_rows(named=True):
        if (
            row["num_rollouts"] == 64
            or row["protocol"] == "shared"
            and row["method"] in SUPERVISED
        ):
            lines.append(
                f"| {row['protocol']} | {row['method']} | {row['num_rollouts']} | {row['final_rmse_mean']:.4f} +/- {row['final_rmse_sd']:.4f} | {row['best_native_rmse_mean']:.4f} | {row['final_pearson_mean']:.4f} |"
            )
    for name in (
        "curves-shared",
        "curves-split",
        "rollout-comparison",
        "scatter-shared",
        "scatter-split",
    ):
        lines.extend(("", f"![{name}]({name}.png)"))
    (destination / "REPORT.md").write_text("\n".join(lines) + "\n")
    print(destination)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    report(root=parser.parse_args().root)


if __name__ == "__main__":
    main()
