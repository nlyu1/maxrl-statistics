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
REPORT_METRICS = (
    *TEST_METRICS,
    "singleton_mae_mean",
    "singleton_rmse_mean",
    "macro_decade_mae",
)


def method_label(*, method: str) -> str:
    if "_n" in method:
        base, count = method.rsplit("_n", 1)
        sampled_names = {
            "grpo_mae": "GRPO, MAE",
            "grpo_mse": "GRPO, MSE",
            "rloo_mae": "RLOO, MAE",
            "rloo_mse": "RLOO, MSE",
            "tailrl": "TailRL",
            "binary_maxrl": "Binary MaxRL",
        }
        return f"{sampled_names[base]}, {count} samples"
    labels = {
        "mse": "MSE of policy mean",
        "ce": "CE / population binary MaxRL",
        "binary_maxrl_population": "Population binary MaxRL / CE",
        "grpo_mae_population": "Population GRPO, MAE",
        "grpo_mse_population": "Population GRPO, MSE",
        "rloo_mae_population": "Population RLOO, MAE",
        "rloo_mse_population": "Population RLOO, MSE",
        "tailrl_population": "Population TailRL",
    }
    if method.startswith("stat_maxrl"):
        kind = "MAE" if "_mae_" in method else "MSE"
        return f"Statistical MaxRL, {kind}, {method.rsplit('_s', 1)[1]} years"
    return labels[method]


def formatted_statistic(*, row: dict, metric: str) -> str:
    mean, sd = row[metric + "_mean"], row[metric + "_sd"]
    return f"{mean:.2f}" if sd is None else f"{mean:.2f} ± {sd:.2f}"


def write_metric_tables(
    *, summary: pl.DataFrame, runs: list[dict], output_dir: Path
) -> None:
    lines = [
        "# UTKFace measured results",
        "",
        f"{len(runs)} runs, {len(summary)} configurations. Epoch budgets: {sorted({run['config']['epochs'] for run in runs})}. Seeds: {sorted({run['config']['seed'] for run in runs})}.",
        "",
        "Values are means ± sample standard deviations across seeds. Single-seed rows show the measured value alone. These are not confidence intervals. Checkpoints minimize validation MAE of the policy mean.",
        "",
        f"Test examples: {sorted({run['test']['count'] for run in runs})}; singleton-image test examples: {sorted({run['test_singletons']['count'] for run in runs})}.",
        "",
        "| Objective | Seeds | Mean age MAE, years | Mean age RMSE, years | Singleton test MAE, years | Macro decade MAE, years |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary.to_dicts():
        values = [
            formatted_statistic(row=row, metric=metric)
            for metric in (
                "mae_mean",
                "rmse_mean",
                "singleton_mae_mean",
                "macro_decade_mae",
            )
        ]
        lines.append(
            f"| {method_label(method=row['method'])} | {row['seeds']} | "
            + " | ".join(values)
            + " |"
        )
    lines += [
        "",
        "Macro decade MAE gives equal weight to each observed age decade. The original test split is imbalanced. The singleton slice retains only images with unique decoded pixels in the entire dataset; training and checkpoint selection use the original split.",
        "",
        "| Objective | Expected policy draw MAE, years | Entropy, nats | Categorical NLL, nats | Oracle best-of-64 MAE, years |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in summary.to_dicts():
        values = [
            formatted_statistic(row=row, metric=metric)
            for metric in ("expected_mae", "entropy", "ce", "oracle_best_of_64_mae")
        ]
        lines.append(
            f"| {method_label(method=row['method'])} | " + " | ".join(values) + " |"
        )
    lines += [
        "",
        "Policy draw errors and oracle errors are evaluated exactly over all 100 actions. Oracle best-of-64 uses the target to select the closest draw; it is a distribution diagnostic.",
        "",
        "![Point and policy errors](comparison.png)",
        "",
        "![Validation learning curves](learning_curves.png)",
        "",
        "Per-seed values: [runs.csv](runs.csv). Aggregates: [summary.csv](summary.csv). Full metrics, configs and provenance: [results.json](results.json). Age-decade metrics: [decade_metrics.csv](decade_metrics.csv).",
        "",
    ]
    (output_dir / "metrics.md").write_text("\n".join(lines))


def analyze(*, runs_dir: Path, scores_dir: Path, output_dir: Path) -> None:
    runs = [
        json.loads(path.read_text()) for path in sorted(runs_dir.glob("*/result.json"))
    ]
    if not runs:
        raise ValueError("Analysis requires completed training runs")
    if len({run["data_fingerprint"] for run in runs}) != 1:
        raise ValueError("Cannot compare runs with different data splits")
    records = []
    histories = []
    decades = []
    protocols: dict[str, set[str]] = {}
    observed_seeds: dict[str, set[int]] = {}
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
        if "test" not in run:
            run["test"] = score["test"]
        run["restoration_mae_difference"] = score["restoration_mae_difference"]
        method, seed = config["loss"]["method"], config["seed"]
        if method.startswith("stat_maxrl"):
            method += f"_s{config['loss']['kernel_scale_years']:g}"
        elif method not in {"mse", "ce"} and not method.endswith("_population"):
            method += f"_n{config['loss']['num_rollouts']}"
        protocol = {
            key: value
            for key, value in config.items()
            if key not in {"seed", "device", "output_dir", "manifest"}
        }
        protocol["world_size"] = run["world_size"]
        protocols.setdefault(method, set()).add(json.dumps(protocol, sort_keys=True))
        seeds = observed_seeds.setdefault(method, set())
        if len(protocols[method]) != 1 or seed in seeds:
            raise ValueError(
                f"{method}: aggregation requires distinct seeds with one shared training configuration"
            )
        seeds.add(seed)
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
            "macro_decade_mae": sum(
                metrics["mae_mean"] for metrics in run["test"]["decades"].values()
            )
            / len(run["test"]["decades"]),
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
    write_metric_tables(summary=summary, runs=runs, output_dir=output_dir)
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
