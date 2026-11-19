import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import torch
from matplotlib.ticker import ScalarFormatter

from src.experiments.bernoulli_calibration.config import (
    METHODS,
    ROLLOUTS,
    SEEDS,
    SUPERVISED,
    Protocol,
)
from src.experiments.bernoulli_calibration.report import COLORS
from src.experiments.bernoulli_calibration.visualizations import (
    ScatterTrajectory,
    scatter_animation,
)
from src.experiments.utkface.run import write_json


def pointwise_maxrl_probability(
    *, truth: np.ndarray, protocol: Protocol, num_rollouts: int
) -> np.ndarray:
    if num_rollouts < 2:
        raise ValueError("Centered MaxRL requires at least two rollouts")
    probability = truth.astype(np.float64)
    one = probability if protocol == "split" else np.ones_like(probability)
    zero = 1 - probability if protocol == "split" else np.ones_like(probability)
    powers = np.arange(num_rollouts - 1)

    def derivative(prediction: np.ndarray) -> np.ndarray:
        positive = ((1 - one * prediction)[:, None] ** powers).sum(-1)
        negative = ((1 - zero * (1 - prediction))[:, None] ** powers).sum(-1)
        return -probability * one * positive + (1 - probability) * zero * negative

    low, high = np.zeros_like(probability), np.ones_like(probability)
    for _ in range(80):
        middle = (low + high) / 2
        decrease = derivative(middle) < 0
        low = np.where(decrease, middle, low)
        high = np.where(decrease, high, middle)
    result = (low + high) / 2
    at_zero = derivative(np.zeros_like(result))
    at_one = derivative(np.ones_like(result))
    result = np.where(at_zero >= 0, 0, result)
    result = np.where(at_one <= 0, 1, result)
    return np.where((at_zero == 0) & (at_one == 0), probability, result)


def compare_distributions(
    *, reference_root: Path, shifted_root: Path, seed: int
) -> None:
    shifted_config = json.loads((shifted_root / "dataset.json").read_text())
    roots = {"reference": reference_root, "shifted": shifted_root}
    datasets = {
        name: torch.load(root / "dataset.pt", map_location="cpu", weights_only=True)
        for name, root in roots.items()
    }
    for split in ("train", "validation", "test"):
        torch.testing.assert_close(
            datasets["reference"]["tokens"][split],
            datasets["shifted"]["tokens"][split],
            rtol=0,
            atol=0,
        )
    destination = shifted_root / "report" / "distribution-shift"
    destination.mkdir(exist_ok=True)
    plt.rcParams.update({
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
    statistics = {}
    tables = []
    references = []
    finite_rows = []
    for cohort, root in roots.items():
        probability = (
            datasets[cohort]["probabilities"]["test"].numpy().astype(np.float64)
        )
        variance = float(probability.var())
        statistics[cohort] = {
            "training_bags": json.loads((root / "completion.json").read_text())[
                "train_size"
            ],
            "test_mean": float(probability.mean()),
            "test_std": float(probability.std()),
            "test_median": float(np.median(probability)),
            "test_quantile10": float(np.quantile(probability, 0.1)),
            "test_quantile90": float(np.quantile(probability, 0.9)),
            "test_majority_positive_fraction": float((probability > 0.5).mean()),
        }
        unique, counts = np.unique(probability, return_counts=True)
        for protocol in ("shared", "split"):
            for count in ROLLOUTS:
                optimum = pointwise_maxrl_probability(
                    truth=unique, protocol=protocol, num_rollouts=count
                )
                finite_rows.append({
                    "cohort": cohort,
                    "protocol": protocol,
                    "num_rollouts": count,
                    "pointwise_native_optimum_rmse": float(
                        np.sqrt(np.average((optimum - unique) ** 2, weights=counts))
                    ),
                    "optimum_zero_context_fraction": float(
                        counts[optimum == 0].sum() / counts.sum()
                    ),
                })
        table = pl.read_csv(root / "report" / "results.csv").with_columns(
            pl.lit(cohort).alias("cohort"),
            (1 - pl.col("final_rmse").pow(2) / variance).alias("mse_skill"),
            pl
            .when(pl.col("final_pearson_defined"))
            .then(pl.col("final_pearson"))
            .otherwise(None)
            .alias("final_pearson"),
        )
        tables.append(
            table.select(
                "cohort",
                "method",
                "protocol",
                "num_rollouts",
                "seed",
                "learning_rate",
                "train_size",
                "final_mae",
                "final_rmse",
                "final_pearson",
                "final_pearson_defined",
                "final_entropy",
                "final_expected_reward",
                "mse_skill",
            )
        )
        for name, prediction in (
            ("constant zero", np.zeros_like(probability)),
            (
                f"constant {shifted_config['target_mean']:g}",
                np.full_like(probability, shifted_config["target_mean"]),
            ),
            ("constant empirical prior", np.full_like(probability, probability.mean())),
            ("oracle probabilities", probability),
            ("oracle majority decision", (probability > 0.5).astype(float)),
        ):
            mse = float(np.mean((prediction - probability) ** 2))
            references.append({
                "cohort": cohort,
                "reference": name,
                "mae": float(np.mean(np.abs(prediction - probability))),
                "rmse": mse**0.5,
                "mse_skill": 1 - mse / variance,
                "expected_reward": float(
                    np.mean(
                        probability * prediction + (1 - probability) * (1 - prediction)
                    )
                ),
            })
    combined = pl.concat(tables)
    combined.write_csv(destination / "paired-results.csv")
    summary = (
        combined
        .group_by("cohort", "method", "protocol", "num_rollouts")
        .agg(
            pl.len().alias("seeds"),
            pl.col("final_pearson_defined").sum().alias("pearson_defined_seeds"),
            pl.col("learning_rate").first(),
            pl.col("train_size").first(),
            *[
                expression
                for metric in (
                    "final_mae",
                    "final_rmse",
                    "final_pearson",
                    "final_entropy",
                    "final_expected_reward",
                    "mse_skill",
                )
                for expression in (
                    pl.col(metric).mean().alias(f"{metric}_mean"),
                    pl.col(metric).std().alias(f"{metric}_sd"),
                )
            ],
        )
        .sort("cohort", "protocol", "method", "num_rollouts")
    )
    summary.write_csv(destination / "paired-summary.csv")
    pl.DataFrame(references).write_csv(
        destination / "constant-and-oracle-references.csv"
    )
    finite_table = pl.DataFrame(finite_rows)
    finite_table.write_csv(destination / "finite-budget-pointwise-optima.csv")
    figure, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    for cohort, color in (("reference", "#777777"), ("shifted", "#347fb8")):
        probability = datasets[cohort]["probabilities"]["validation"].numpy()
        label = f"{cohort}: mean={probability.mean():.3f}"
        axes[0].hist(
            probability,
            bins=np.linspace(0, 1, 21),
            weights=np.full(len(probability), 1 / len(probability)),
            alpha=0.4,
            color=color,
            label=label,
        )
        unique, counts = np.unique(probability, return_counts=True)
        axes[1].step(
            unique,
            np.cumsum(counts) / len(probability),
            where="post",
            color=color,
            label=label,
        )
        axes[0].axvline(probability.mean(), color=color, linestyle="--", linewidth=1)
    axes[0].set(
        xlabel="Ground-truth probability f(x)",
        ylabel="Fraction of validation contexts",
        xlim=(0, 1),
    )
    axes[1].set(
        xlabel="Ground-truth probability f(x)",
        ylabel="Cumulative fraction",
        xlim=(0, 1),
        ylim=(0, 1),
    )
    axes[0].legend()
    axes[1].legend()
    figure.suptitle("Teacher distribution shift on identical word bags")
    figure.savefig(destination / "teacher-distribution.png", dpi=170)
    figure.savefig(destination / "teacher-distribution.svg")
    plt.close(figure)
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for row, protocol in enumerate(("shared", "split")):
        for column, (metric, label) in enumerate((
            ("final_rmse", "Final test probability RMSE"),
            ("mse_skill", "MSE skill relative to constant prior"),
        )):
            axis = axes[row, column]
            for method in ("ce", "mse", "maxrl"):
                for cohort, style in (("reference", "--"), ("shifted", "-")):
                    rows = summary.filter(
                        (pl.col("cohort") == cohort)
                        & (pl.col("protocol") == protocol)
                        & (pl.col("method") == method)
                    ).sort("num_rollouts")
                    mean, sd = (
                        rows[f"{metric}_mean"].to_numpy(),
                        rows[f"{metric}_sd"].to_numpy(),
                    )
                    if protocol == "shared" and method in SUPERVISED:
                        mean, sd = (
                            np.repeat(mean, len(ROLLOUTS)),
                            np.repeat(sd, len(ROLLOUTS)),
                        )
                    axis.errorbar(
                        ROLLOUTS,
                        mean,
                        yerr=sd,
                        linestyle=style,
                        marker="o",
                        markersize=3,
                        color=COLORS[method],
                        label=f"{method.upper()} {cohort}",
                        alpha=0.9,
                    )
            if metric == "mse_skill":
                axis.axhline(0, color="#999999", linewidth=1)
            axis.set(
                title=protocol,
                xlabel="Rollouts per context N",
                ylabel=label,
                xscale="log",
                xticks=ROLLOUTS,
            )
            axis.xaxis.set_major_formatter(ScalarFormatter())
            axis.grid(alpha=0.2)
    axes[0, 0].legend(fontsize=8)
    figure.suptitle(
        "Distribution shift: matched contexts and model; rates retuned by native loss\nMeans +/- seed SD; raw errors and variance-normalized skill"
    )
    figure.savefig(destination / "rmse-and-skill.png", dpi=160)
    plt.close(figure)
    panels = []
    frame_steps = []
    initializations = []
    for cohort, root in roots.items():
        size = json.loads((root / "completion.json").read_text())["train_size"]
        for count in (8, 64, 256):
            run = root / "runs" / f"main-t{size}-maxrl-split-n{count}-s{seed}"
            trajectory = torch.load(
                run / "validation-trajectory.pt", map_location="cpu", weights_only=True
            )
            steps = trajectory["step"].numpy()
            chosen = np.flatnonzero(
                (steps <= 256) | (steps % 128 == 0) | (steps == steps[-1])
            )
            if frame_steps:
                np.testing.assert_array_equal(steps[chosen], np.array(frame_steps))
            else:
                frame_steps = steps[chosen].tolist()
            initializations.append(trajectory["initial_model_fingerprint"])
            panels.append(
                ScatterTrajectory(
                    method="maxrl",
                    title=f"{cohort.capitalize()}: N={count}",
                    truth=trajectory["f"].numpy(),
                    predictions=trajectory["q"].numpy()[chosen],
                )
            )
    if len(set(initializations)) != 1:
        raise ValueError("Cross-distribution model initializations do not match")
    scatter_animation(
        panels=panels,
        captions=[
            f"MaxRL | split labels | seed {seed} | update {step}"
            for step in frame_steps
        ],
        durations=[
            800
            if step == 0
            else 1800
            if step == frame_steps[-1]
            else 250
            if step < 32
            else 160
            if step <= 256
            else 120
            for step in frame_steps
        ],
        destination=destination / "maxrl-training-distribution-shift",
        footer="Top: balanced reference. Bottom: shifted teacher. Same word bags and model initialization.\nActual recorded validation predictions; each cohort uses native-loss-selected learning rates.",
    )
    config = shifted_config
    write_json(
        path=destination / "manifest.json",
        payload={
            "reference_root": str(reference_root),
            "shifted_root": str(shifted_root),
            "target_mean": config["target_mean"],
            "logit_bias": config["logit_bias"],
            "identical_word_bags": True,
            "identical_model_initialization": True,
            "statistics": statistics,
            "animation_seed": seed,
        },
    )
    lines = [
        "# Teacher distribution shift",
        "",
        f"Shifted teacher: f(x)=sigmoid(sum(word_values)/4 + {config['logit_bias']:.8f}); exact population mean {config['target_mean']:g}. Vocabulary, architecture and stored word bags match across all splits. Each cohort uses its reported training prefix; rates are retuned independently using native validation losses.",
        "",
        "Raw RMSE is sensitive to the variance of f. MSE skill is 1-MSE(q,f)/Var(f): constant-prior prediction has skill zero and oracle probabilities have skill one. These diagnostics do not select learning rates. Constant-zero and oracle-majority references expose high reward from majority-class prediction.",
        "",
        "| Cohort | Training bags | Mean f | Median f | SD f | Fraction f > 0.5 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for cohort, row in statistics.items():
        lines.append(
            f"| {cohort} | {row['training_bags']} | {row['test_mean']:.4f} | {row['test_median']:.4f} | {row['test_std']:.4f} | {row['test_majority_positive_fraction']:.4f} |"
        )
    lines.extend((
        "",
        "## Final test results at N=256",
        "",
        "RMSE entries show the mean +/- sample standard deviation across three seeds. Shared-label supervised baselines use one label per context and are reused across rollout counts.",
        "",
        "| Labels | Method | Reference RMSE | Shifted RMSE | Shifted MSE skill |",
        "|---|---|---:|---:|---:|",
    ))
    for protocol in ("shared", "split"):
        for method in METHODS:
            rows = summary.filter(
                (pl.col("protocol") == protocol)
                & (pl.col("method") == method)
                & (
                    (pl.col("num_rollouts") == 256)
                    | pl.lit(protocol == "shared" and method in SUPERVISED)
                )
            )
            values = {
                cohort: rows.filter(pl.col("cohort") == cohort).row(0, named=True)
                for cohort in roots
            }
            errors = [
                f"{values[cohort]['final_rmse_mean']:.5f} +/- {values[cohort]['final_rmse_sd']:.5f}"
                for cohort in roots
            ]
            lines.append(
                f"| {protocol} | {method.upper()} | {errors[0]} | {errors[1]} | {values['shifted']['mse_skill_mean']:.4f} |"
            )
    lines.extend((
        "",
        "[All rollout counts and metrics](paired-summary.csv) | [Individual seeds](paired-results.csv) | [Constant and oracle references](constant-and-oracle-references.csv)",
        "",
        "## Finite-rollout objective bias",
        "",
        "The exact centered MaxRL objective has degree N-1. Its pointwise optimum is computed separately for each true probability, including exact boundary checks, with no model or sampling error. These are calibration errors at the native-objective optimum, not lower bounds on achievable calibration error. Comparing them with trained runs helps distinguish objective bias from optimization and estimation effects.",
        "",
        "| N | Shifted shared: trained RMSE | Shared: pointwise native optimum RMSE | Shifted split: trained RMSE | Split: pointwise native optimum RMSE |",
        "|---:|---:|---:|---:|---:|",
    ))
    for count in ROLLOUTS:
        values = []
        for protocol in ("shared", "split"):
            trained = summary.filter(
                (pl.col("cohort") == "shifted")
                & (pl.col("method") == "maxrl")
                & (pl.col("protocol") == protocol)
                & (pl.col("num_rollouts") == count)
            )["final_rmse_mean"].item()
            optimum = finite_table.filter(
                (pl.col("cohort") == "shifted")
                & (pl.col("protocol") == protocol)
                & (pl.col("num_rollouts") == count)
            )["pointwise_native_optimum_rmse"].item()
            values.extend((f"{trained:.5f}", f"{optimum:.5f}"))
        lines.append(f"| {count} | " + " | ".join(values) + " |")
    lines.extend((
        "",
        "[Pointwise native optima for both teacher distributions](finite-budget-pointwise-optima.csv)",
    ))
    for name, suffix in (
        ("teacher-distribution", "png"),
        ("rmse-and-skill", "png"),
        ("maxrl-training-distribution-shift", "gif"),
    ):
        lines.extend(("", f"![{name}]({name}.{suffix})"))
    (destination / "REPORT.md").write_text("\n".join(lines) + "\n")
    report = shifted_root / "report/REPORT.md"
    content = report.read_text()
    if "distribution-shift/REPORT.md" not in content:
        report.write_text(
            content
            + "\n[Matched distribution-shift comparison](distribution-shift/REPORT.md)\n"
        )
    print(destination, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--shifted-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=SEEDS[0])
    arguments = parser.parse_args()
    compare_distributions(
        reference_root=arguments.reference_root,
        shifted_root=arguments.shifted_root,
        seed=arguments.seed,
    )


if __name__ == "__main__":
    main()
