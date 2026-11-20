import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from src.experiments.bernoulli_calibration.distribution_shift import (
    pointwise_maxrl_probability,
)
from src.experiments.pass_rate_calibration.config import (
    MAXRL_ROLLOUTS,
    SERIES_COLORS,
    SOLVER,
    SOURCES,
)
from src.experiments.pass_rate_calibration.data import read_problems
from src.experiments.pass_rate_calibration.metrics import prediction_metrics
from src.experiments.utkface.run import write_json


def build_labels(*, root: Path) -> pl.DataFrame:
    rollouts = pl.read_parquet(
        root / "rollouts.parquet",
        columns=[
            "problem_id",
            "completion_tokens",
            "finish_reason",
            "correct",
            "grader_error",
        ],
    )
    per_problem = rollouts.group_by("problem_id").agg(
        samples=pl.len(),
        successes=pl.col("correct").sum(),
        p_hat=pl.col("correct").mean(),
        truncation_rate=(pl.col("finish_reason") == "length").mean(),
        mean_completion_tokens=pl.col("completion_tokens").mean(),
        grader_errors=pl.col("grader_error").sum(),
    )
    labels = (
        read_problems(root=root)
        .select("problem_id", "dataset", "subject", "split")
        .join(per_problem, on="problem_id", how="left")
    )
    if labels["samples"].null_count() or (labels["samples"] != SOLVER.samples).any():
        raise ValueError("Every problem needs exactly the configured rollout count")
    labels.write_parquet(root / "labels.parquet")
    return labels


def optimum_rows(*, truth: np.ndarray) -> list[dict]:
    threshold = np.where(truth > 0.5, 1.0, np.where(truth < 0.5, 0.0, 0.5))
    interior = (truth > 0) & (truth < 1)
    rows = []
    for rollouts in MAXRL_ROLLOUTS:
        optima = {
            "maxrl-shared": pointwise_maxrl_probability(
                truth=truth, protocol="shared", num_rollouts=rollouts
            ),
            "maxrl-split": pointwise_maxrl_probability(
                truth=truth, protocol="split", num_rollouts=rollouts
            ),
            "rloo/grpo": threshold,
        }
        for method, optimum in optima.items():
            boundary = (optimum == 0) | (optimum == 1)
            rows.append({
                "method": method,
                "rollouts": rollouts,
                "rmse": float(np.sqrt(((optimum - truth) ** 2).mean())),
                "mae": float(np.abs(optimum - truth).mean()),
                "at_zero": float((optimum == 0).mean()),
                "at_one": float((optimum == 1).mean()),
                "interior_collapsed": float(
                    (boundary & interior).sum() / interior.sum()
                ),
            })
    return rows


def histogram_panel(*, axis, values: np.ndarray, title: str, color: str) -> None:
    axis.hist(
        values, bins=np.linspace(0, 1, 21), color=color, edgecolor="white", linewidth=1
    )
    axis.set_title(
        f"{title} (n={len(values)})\np̂=0: {int((values == 0).sum())}, "
        f"p̂=1: {int((values == 1).sum())}, mean {values.mean():.3f}",
        fontsize=9,
    )
    axis.set_xlabel("pass rate p̂ (mean@256)")
    axis.grid(axis="y", alpha=0.3, linewidth=0.5)
    axis.spines[["top", "right"]].set_visible(False)


def plot_pass_rates(*, labels: pl.DataFrame, path: Path) -> None:
    panels = [
        ("Minerva train", labels.filter(pl.col("split") == "train")),
        ("Minerva val", labels.filter(pl.col("split") == "val")),
        ("JEEBench (all subjects)", labels.filter(pl.col("split") == "test")),
        *[
            (f"JEEBench {subject}", labels.filter(pl.col("subject") == subject))
            for subject in sorted(
                labels.filter(pl.col("dataset") == "jeebench")["subject"].unique()
            )
        ],
    ]
    figure, axes = plt.subplots(2, 3, figsize=(12, 6.5), constrained_layout=True)
    for axis, (title, frame) in zip(axes.flat, panels, strict=True):
        histogram_panel(
            axis=axis,
            values=frame["p_hat"].to_numpy(),
            title=title,
            color=SERIES_COLORS[0] if title.startswith("Minerva") else SERIES_COLORS[1],
        )
    axes[0, 0].set_ylabel("problems")
    axes[1, 0].set_ylabel("problems")
    figure.savefig(path, dpi=150)
    plt.close(figure)


def plot_lengths(*, root: Path, path: Path) -> None:
    rollouts = pl.read_parquet(
        root / "rollouts.parquet",
        columns=["problem_id", "completion_tokens", "correct"],
    ).join(read_problems(root=root).select("problem_id", "dataset"), on="problem_id")
    bins = np.linspace(0, SOLVER.max_tokens, 65)
    figure, axes = plt.subplots(
        1, 2, figsize=(12, 4), constrained_layout=True, sharey=True
    )
    for axis, source in zip(axes, SOURCES, strict=True):
        frame = rollouts.filter(pl.col("dataset") == source.name)
        for color, (label, outcome) in zip(
            SERIES_COLORS, (("correct", 1.0), ("incorrect", 0.0)), strict=False
        ):
            values = frame.filter(pl.col("correct") == outcome)[
                "completion_tokens"
            ].to_numpy()
            axis.hist(
                values,
                bins=bins,
                histtype="step",
                linewidth=2,
                color=color,
                label=f"{label} ({len(values):,})",
            )
        truncated = float((frame["completion_tokens"] >= SOLVER.max_tokens).mean())
        axis.axvline(SOLVER.max_tokens, color="#52514e", linestyle="--", linewidth=1)
        axis.set_title(
            f"{source.name}: solver response length (truncated at cap: {truncated:.1%})",
            fontsize=10,
        )
        axis.set_xlabel("completion tokens")
        axis.set_yscale("log")
        axis.grid(axis="y", alpha=0.3, linewidth=0.5)
        axis.spines[["top", "right"]].set_visible(False)
        axis.legend(frameon=False, fontsize=9)
    axes[0].set_ylabel("rollouts (log scale)")
    figure.savefig(path, dpi=150)
    plt.close(figure)


def markdown_table(*, frame: pl.DataFrame) -> str:
    def cell(value) -> str:
        return f"{value:.4f}" if isinstance(value, float) else str(value)

    header = "| " + " | ".join(frame.columns) + " |"
    rule = "|" + "---|" * len(frame.columns)
    rows = [
        "| " + " | ".join(cell(value) for value in row) + " |"
        for row in frame.iter_rows()
    ]
    return "\n".join([header, rule, *rows])


def report(*, root: Path) -> dict:
    labels = build_labels(root=root)
    gate = []
    for source in SOURCES:
        observed = float(
            labels.filter(pl.col("dataset") == source.name)["p_hat"].mean()
        )
        gate.append({
            "dataset": source.name,
            "pass_at_1": observed,
            "reference": source.reference_pass_at_1,
            "within_tolerance": abs(observed - source.reference_pass_at_1)
            <= SOLVER.gate_tolerance,
        })
    train = labels.filter(pl.col("split") == "train")["p_hat"].to_numpy()
    prior = float(train.mean())
    baselines = {
        split: prediction_metrics(
            q=np.full(frame.height, prior), p=frame["p_hat"].to_numpy()
        )
        for (split,), frame in labels.group_by("split", maintain_order=True)
    }
    optima = pl.DataFrame(optimum_rows(truth=train))
    plot_pass_rates(labels=labels, path=root / "pass-rate-histograms.png")
    plot_lengths(root=root, path=root / "solver-length-histograms.png")
    summary = {
        "gate": gate,
        "gate_passed": all(row["within_tolerance"] for row in gate),
        "train_prior": prior,
        "constant_prior_metrics": baselines,
        "grader_errors": int(labels["grader_errors"].sum()),
        "truncation_rate": {
            name: float(frame["truncation_rate"].mean())
            for (name,), frame in labels.group_by("dataset")
        },
        "split_sizes": {
            name: frame.height for (name,), frame in labels.group_by("split")
        },
        "pointwise_optima": optima.to_dicts(),
    }
    write_json(path=root / "ground-truth.json", payload=summary)
    lines = [
        "# Phase 0 ground truth",
        "",
        markdown_table(frame=pl.DataFrame(gate)),
        "",
        f"Gate passed: {summary['gate_passed']}. Grader errors: {summary['grader_errors']}.",
        f"Training prior (mean p̂ over 240 train problems): {prior:.4f}.",
        "",
        "## Constant training-prior predictor",
        "",
        markdown_table(
            frame=pl.DataFrame([
                {"split": split, **metrics} for split, metrics in baselines.items()
            ]).select("split", "mae", "rmse", "expected_reward", "bias")
        ),
        "",
        "## Pointwise native optima on the training p̂",
        "",
        markdown_table(frame=optima),
        "",
        "![pass rates](pass-rate-histograms.png)",
        "",
        "![solver lengths](solver-length-histograms.png)",
        "",
    ]
    (root / "ground-truth.md").write_text("\n".join(lines))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    report(root=args.root)


if __name__ == "__main__":
    main()
