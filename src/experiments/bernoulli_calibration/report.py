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

from src.experiments.bernoulli_calibration.config import (
    METHODS,
    ROLLOUTS,
    SUPERVISED,
    DatasetConfig,
)

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


def curve_statistics(
    *, histories: list[list[dict]], split: str, metric: str
) -> tuple[np.ndarray, np.ndarray]:
    values = np.array([
        [
            np.nan
            if metric == "pearson" and not entry[split]["pearson_defined"]
            else entry[split][metric]
            for entry in history
        ]
        for history in histories
    ])
    masked = np.ma.masked_invalid(values)
    return masked.mean(0).filled(np.nan), masked.std(0, ddof=1).filled(np.nan)


def report(*, root: Path) -> None:
    if (root / "SUPERSEDED.md").exists():
        raise ValueError(f"Refusing to report a superseded cohort: {root}")
    completion = json.loads((root / "completion.json").read_text())
    train_size = completion["train_size"]
    data_config = DatasetConfig.model_validate_json((root / "dataset.json").read_text())
    selection = json.loads((root / f"selection-train{train_size}.json").read_text())
    boundary_winners = sum(row["grid_boundary"] for row in selection.values())
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
            "labels_per_context": 1
            if config["protocol"] == "shared"
            else config["num_rollouts"],
            "policy_rollouts_per_context": 0
            if config["method"] in SUPERVISED
            else config["num_rollouts"],
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
    table = pl.DataFrame(rows).with_columns(
        pl
        .when(pl.col("final_pearson_defined"))
        .then(pl.col("final_pearson"))
        .otherwise(None)
        .alias("final_pearson"),
        pl
        .when(pl.col("best_native_pearson_defined"))
        .then(pl.col("best_native_pearson"))
        .otherwise(None)
        .alias("best_native_pearson"),
    )
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
            pl.col("learning_rate").first(),
            pl.col("final_pearson_defined").sum().alias("pearson_defined_seeds"),
            *[
                expression
                for metric in (
                    "final_mae",
                    "final_rmse",
                    "final_pearson",
                    "final_entropy",
                    "final_mse_skill",
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
    pilot_rows = []
    for path in sorted((root / "runs").glob(f"pilot-t{train_size}-*/result.json")):
        candidate = json.loads(path.read_text())
        pilot_rows.append({
            **{
                key: candidate["config"][key]
                for key in ("method", "protocol", "num_rollouts", "learning_rate")
            },
            "validation_native_loss": candidate["final"]["validation"]["native_loss"],
            "validation_rmse": candidate["final"]["validation"]["rmse"],
            "seconds": candidate["seconds"],
        })
    pl.DataFrame(pilot_rows).write_csv(destination / "pilot-losses.csv")
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
                average, deviation = curve_statistics(
                    histories=histories_for_method, split="validation", metric=metric
                )
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
                axis.annotate(
                    "true entropy",
                    (steps[0], truth),
                    xytext=(5, 5),
                    textcoords="offset points",
                    color="#666666",
                )
                interactive.add_hline(
                    y=truth,
                    line_dash="dash",
                    line_color="#777777",
                    row=2,
                    col=2,
                    annotation_text="True entropy",
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
        trace_groups = {64: list(range(len(interactive.data)))}
        for count in ROLLOUTS:
            if count == 64:
                continue
            start = len(interactive.data)
            candidates = [
                row
                for row in results
                if representative(result=row, protocol=protocol, rollouts=count)
            ]
            for panel, metric in enumerate(("mae", "rmse", "pearson", "entropy")):
                for method in METHODS:
                    entries = [
                        histories[row["config"]["identifier"]]
                        for row in candidates
                        if row["config"]["method"] == method
                    ]
                    steps = [entry["step"] for entry in entries[0]]
                    average, _ = curve_statistics(
                        histories=entries, split="validation", metric=metric
                    )
                    interactive.add_trace(
                        go.Scatter(
                            x=steps,
                            y=average,
                            name=method,
                            legendgroup=method,
                            showlegend=panel == 0,
                            line={"color": COLORS[method]},
                            visible=False,
                        ),
                        row=panel // 2 + 1,
                        col=panel % 2 + 1,
                    )
            trace_groups[count] = list(range(start, len(interactive.data)))
        interactive.update_layout(
            title=f"{protocol} labels, N=64; validation seed means",
            height=750,
            updatemenus=[
                {
                    "buttons": [
                        {
                            "label": f"N={count}",
                            "method": "update",
                            "args": [
                                {
                                    "visible": [
                                        index in trace_groups[count]
                                        for index in range(len(interactive.data))
                                    ]
                                },
                                {
                                    "title": f"{protocol} labels, N={count}; validation seed means"
                                },
                            ],
                        }
                        for count in (64, 2, 4, 8, 16, 256)
                    ],
                    "x": 1.0,
                    "y": 1.15,
                }
            ],
        )
        interactive.write_html(
            destination / f"curves-{protocol}.html", include_plotlyjs=True
        )
        figure, axes = plt.subplots(2, 3, figsize=(12, 8), constrained_layout=True)
        for axis, method in zip(axes.flat, METHODS, strict=True):
            candidates = [row for row in selected if row["config"]["method"] == method]
            for split, style in (("train", "--"), ("validation", "-")):
                entries = [histories[row["config"]["identifier"]] for row in candidates]
                steps = [entry["step"] for entry in entries[0]]
                mean, sd = curve_statistics(
                    histories=entries, split=split, metric="native_loss"
                )
                axis.plot(steps, mean, style, color=COLORS[method], label=split)
                axis.fill_between(
                    steps, mean - sd, mean + sd, color=COLORS[method], alpha=0.1
                )
            axis.set(title=method, xlabel="Optimizer update", ylabel="Native loss")
            axis.legend()
            axis.grid(alpha=0.2)
        figure.suptitle(f"Method losses: {protocol} labels, N=64; mean +/- seed SD")
        figure.savefig(destination / f"native-loss-{protocol}.png", dpi=160)
        plt.close(figure)
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
        f"The teacher is f(x)=sigmoid(sum(word_values)/4 + {data_config.logit_bias:.8f}) for length-16 contexts. Its population mean is {data_config.target_mean:g}; the offset is solved from the exact word-sampling distribution. The model is trained from scratch with 128-dimensional embeddings, normalized sum pooling, a 32-unit GELU hidden layer, and one output logit (8257 parameters). Labels are resampled on every visit. Seeds vary optimization and observations on one fixed dataset split.",
        "",
        "Learning rates use final pilot validation native loss. GRPO/RLOO use expected mismatch; MaxRL uses its finite-compute objective. Calibration metrics never select rates or checkpoints.",
        "",
        "Curves include step zero. Shading is seed standard deviation. Scatters show raw predictions with within-probability means separately per seed. Raw predictions are retained for every main run.",
        "",
        "MSE skill is 1 - MSE(q,f)/Var(f), using the constant empirical prior as its zero-skill reference. It is a diagnostic, not a selection criterion. Undefined Pearson values are null in the tables.",
        "",
        f"Completed {len(pilot_rows)} pilot trials and {len(results)} main runs. Each main run uses 2048 optimizer updates. See pilot-losses.csv for native-loss LR comparisons and curves.parquet for every method/protocol/rollout/seed trajectory.",
        "",
        f"The bounded pilot search leaves {boundary_winners}/{len(selection)} learning-rate winners at a searched boundary after up to two extensions. These preliminary sweeps do not establish globally optimal learning rates.",
        "",
        f"Training-size diagnostic: {json.dumps(completion['resizing_diagnostic'], sort_keys=True)}",
        "",
        "| Protocol | Method | N | Learning rate | Final test RMSE | Best native checkpoint RMSE | Final test Pearson |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary.iter_rows(named=True):
        if (
            row["num_rollouts"] == 64
            or row["protocol"] == "shared"
            and row["method"] in SUPERVISED
        ):
            count = (
                "all"
                if row["protocol"] == "shared" and row["method"] in SUPERVISED
                else str(row["num_rollouts"])
            )
            pearson = (
                "undefined"
                if row["final_pearson_mean"] is None
                else f"{row['final_pearson_mean']:.4f}"
            )
            lines.append(
                f"| {row['protocol']} | {row['method']} | {count} | {row['learning_rate']:g} | {row['final_rmse_mean']:.4f} +/- {row['final_rmse_sd']:.4f} | {row['best_native_rmse_mean']:.4f} | {pearson} |"
            )
    for name in (
        "curves-shared",
        "curves-split",
        "rollout-comparison",
        "native-loss-shared",
        "native-loss-split",
        "scatter-shared",
        "scatter-split",
    ):
        lines.extend(("", f"![{name}]({name}.png)"))
    if (destination / "comparisons" / "GALLERY.md").exists():
        lines.extend((
            "",
            "[Rollout comparisons and correlation animations](comparisons/GALLERY.md)",
        ))
    (destination / "REPORT.md").write_text("\n".join(lines) + "\n")
    print(destination)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    report(root=parser.parse_args().root)


if __name__ == "__main__":
    main()
