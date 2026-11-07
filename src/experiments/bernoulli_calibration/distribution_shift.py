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

from src.experiments.bernoulli_calibration.config import ROLLOUTS, SEEDS, SUPERVISED
from src.experiments.bernoulli_calibration.report import COLORS
from src.experiments.bernoulli_calibration.visualizations import (
    ScatterTrajectory,
    scatter_animation,
)
from src.experiments.utkface.run import write_json


def compare_distributions(
    *, reference_root: Path, shifted_root: Path, seed: int
) -> None:
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
                "final_entropy",
                "final_expected_reward",
                "mse_skill",
            )
        )
        for name, prediction in (
            ("constant zero", np.zeros_like(probability)),
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
    config = json.loads((shifted_root / "dataset.json").read_text())
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
