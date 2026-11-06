import argparse
import json
from functools import cache
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import torch
from matplotlib.ticker import ScalarFormatter
from PIL import Image
from pydantic import ConfigDict
from pydantic.dataclasses import dataclass

from src.experiments.bernoulli_calibration.config import (
    METHODS,
    ROLLOUTS,
    SEEDS,
    SUPERVISED,
)
from src.experiments.bernoulli_calibration.report import COLORS, representative
from src.experiments.utkface.run import write_json


@dataclass(kw_only=True, config=ConfigDict(arbitrary_types_allowed=True))
class ScatterTrajectory:
    method: str
    title: str
    truth: np.ndarray
    predictions: np.ndarray


def scatter_animation(
    *,
    panels: list[ScatterTrajectory],
    captions: list[str],
    durations: list[int],
    destination: Path,
    footer: str,
) -> None:
    if len(panels) != 6 or len(captions) != len(durations):
        raise ValueError(
            "Scatter animations require six panels and one duration per frame"
        )
    if any(
        panel.predictions.shape != (len(captions), len(panel.truth)) for panel in panels
    ):
        raise ValueError("Trajectory frame shapes disagree")
    figure, axes = plt.subplots(2, 3, figsize=(12, 8), dpi=100)
    figure.subplots_adjust(
        left=0.065, right=0.98, bottom=0.11, top=0.88, wspace=0.26, hspace=0.32
    )
    heading = figure.suptitle(captions[0], fontsize=15, y=0.97)
    figure.text(0.065, 0.025, footer, fontsize=9, color="#555555")
    artists = []
    for axis, panel in zip(axes.flat, panels, strict=True):
        sample = np.linspace(
            0, len(panel.truth) - 1, min(1024, len(panel.truth)), dtype=int
        )
        probability, groups = np.unique(panel.truth, return_inverse=True)
        counts = np.bincount(groups)
        points = axis.scatter([], [], s=5, alpha=0.15, color=COLORS[panel.method])
        (mean,) = axis.plot(
            [], [], color=COLORS[panel.method], linewidth=1.4, marker=".", markersize=3
        )
        axis.plot([0, 1], [0, 1], "--", color="#999999", linewidth=1)
        text = axis.text(
            0.035,
            0.96,
            "",
            transform=axis.transAxes,
            va="top",
            fontsize=9,
            bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none", "pad": 2},
        )
        axis.set(
            title=panel.title,
            xlabel="Ground truth f(x)",
            ylabel="Model probability q(x)",
            xlim=(-0.015, 1.015),
            ylim=(-0.015, 1.015),
        )
        axis.set_xticks((0, 0.5, 1))
        axis.set_yticks((0, 0.5, 1))
        artists.append((points, mean, text, sample, probability, groups, counts))
    frames = []
    for index, caption in enumerate(captions):
        heading.set_text(caption)
        for panel, artist in zip(panels, artists, strict=True):
            points, mean, text, sample, probability, groups, counts = artist
            predictions = panel.predictions[index]
            points.set_offsets(
                np.column_stack((panel.truth[sample], predictions[sample]))
            )
            mean.set_data(
                probability, np.bincount(groups, weights=predictions) / counts
            )
            centered_p = panel.truth - panel.truth.mean()
            centered_q = predictions - predictions.mean()
            denominator = np.linalg.norm(centered_p) * np.linalg.norm(centered_q)
            pearson = (
                f"{np.dot(centered_p, centered_q) / denominator:.3f}"
                if denominator
                else "undefined"
            )
            rmse = np.sqrt(np.mean(np.square(predictions - panel.truth)))
            text.set_text(f"RMSE {rmse:.4f}   r {pearson}")
        figure.canvas.draw()
        frame = Image.fromarray(
            np.asarray(figure.canvas.buffer_rgba())[:, :, :3].copy()
        )
        if not frames:
            palette = frame.quantize(colors=128)
        frames.append(frame.quantize(palette=palette, dither=Image.Dither.NONE))
    frames[0].save(
        destination.with_suffix(".gif"),
        save_all=True,
        append_images=frames[1:],
        duration=durations,
        loop=0,
        optimize=True,
        disposal=2,
    )
    figure.savefig(destination.with_suffix(".png"), dpi=160)
    plt.close(figure)
    with Image.open(destination.with_suffix(".gif")) as animation:
        if animation.n_frames != len(captions):
            raise RuntimeError(f"GIF frame count mismatch: {destination}")


def comparison_plots(*, root: Path, destination: Path, results: list[dict]) -> None:
    summary = pl.read_csv(root / "report" / "summary.csv")
    for protocol in ("shared", "split"):
        figure, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
        for axis, metric in zip(
            axes.flat, ("mae", "rmse", "pearson", "entropy"), strict=True
        ):
            for method in METHODS:
                rows = summary.filter(
                    (pl.col("method") == method) & (pl.col("protocol") == protocol)
                ).sort("num_rollouts")
                means = rows[f"final_{metric}_mean"].to_numpy()
                deviations = rows[f"final_{metric}_sd"].to_numpy()
                if protocol == "shared" and method in SUPERVISED:
                    means, deviations = (
                        np.repeat(means, len(ROLLOUTS)),
                        np.repeat(deviations, len(ROLLOUTS)),
                    )
                axis.errorbar(
                    ROLLOUTS,
                    means,
                    yerr=deviations,
                    marker="o",
                    markersize=4,
                    capsize=2,
                    label=method.upper(),
                    color=COLORS[method],
                )
            if metric == "entropy":
                axis.axhline(
                    results[0]["test_final"]["true_entropy"],
                    linestyle="--",
                    color="#888888",
                )
            axis.set(
                xscale="log",
                xticks=ROLLOUTS,
                xlabel="Rollouts per context N",
                ylabel=f"Final test {metric}",
            )
            axis.xaxis.set_major_formatter(ScalarFormatter())
            axis.grid(alpha=0.2)
        axes[0, 0].legend(fontsize=8)
        figure.suptitle(
            f"{protocol.capitalize()} labels: rollout comparison; mean +/- SD, three seeds"
        )
        figure.savefig(destination / f"rollout-metrics-{protocol}.png", dpi=160)
        figure.savefig(destination / f"rollout-metrics-{protocol}.svg")
        plt.close(figure)
        figure, axes = plt.subplots(3, 3, figsize=(14, 10), constrained_layout=True)
        colors = plt.get_cmap("viridis")(np.linspace(0.05, 0.9, len(ROLLOUTS)))
        for row, method in enumerate(("rloo", "grpo", "maxrl")):
            for count, color in zip(ROLLOUTS, colors, strict=True):
                candidates = [
                    result
                    for result in results
                    if result["config"]["method"] == method
                    and result["config"]["protocol"] == protocol
                    and result["config"]["num_rollouts"] == count
                ]
                histories = [
                    json.loads(
                        (
                            root
                            / "runs"
                            / result["config"]["identifier"]
                            / "history.json"
                        ).read_text()
                    )
                    for result in candidates
                ]
                steps = [entry["step"] for entry in histories[0]]
                for column, metric in enumerate(("rmse", "pearson", "entropy")):
                    values = np.array([
                        [entry["validation"][metric] for entry in history]
                        for history in histories
                    ])
                    mean, sd = values.mean(0), values.std(0, ddof=1)
                    axis = axes[row, column]
                    axis.plot(steps, mean, color=color, label=f"N={count}")
                    axis.fill_between(
                        steps, mean - sd, mean + sd, color=color, alpha=0.1
                    )
                    axis.set(
                        title=f"{method.upper()}: {metric}",
                        xlabel="Optimizer update",
                        ylabel=metric,
                    )
                    axis.grid(alpha=0.15)
        axes[0, 0].legend(ncol=2, fontsize=8)
        figure.suptitle(
            f"{protocol.capitalize()} labels: validation trajectories across N; three seeds\nEach N uses its own loss-selected learning rate"
        )
        figure.savefig(
            destination / f"learning-curves-rollouts-{protocol}.png", dpi=150
        )
        plt.close(figure)


def visualizations(*, root: Path, seed: int) -> None:
    if (root / "SUPERSEDED.md").exists():
        raise ValueError(f"Refusing to visualize a superseded cohort: {root}")
    train_size = json.loads((root / "completion.json").read_text())["train_size"]
    results = [
        json.loads(path.read_text())
        for path in sorted((root / "runs").glob(f"main-t{train_size}-*/result.json"))
    ]
    if seed not in {row["config"]["seed"] for row in results}:
        raise ValueError("Animation seed is not present in the completed study")
    destination = root / "report" / "comparisons"
    destination.mkdir(exist_ok=True)
    plt.rcParams.update({
        "font.size": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
    comparison_plots(root=root, destination=destination, results=results)

    @cache
    def final_predictions(
        *, method: str, protocol: str, count: int
    ) -> tuple[np.ndarray, np.ndarray]:
        candidates = [
            result
            for result in results
            if result["config"]["method"] == method
            and representative(result=result, protocol=protocol, rollouts=count)
        ]
        tables = [
            pl.read_parquet(
                root / "runs" / row["config"]["identifier"] / "test-predictions.parquet"
            )
            for row in candidates
        ]
        return np.concatenate([
            table["f"].to_numpy() for table in tables
        ]), np.concatenate([table["q_final"].to_numpy() for table in tables])

    for protocol in ("shared", "split"):
        panels = []
        for method in METHODS:
            truth, _ = final_predictions(
                method=method, protocol=protocol, count=ROLLOUTS[0]
            )
            values = []
            for count in ROLLOUTS:
                target, probability = final_predictions(
                    method=method, protocol=protocol, count=count
                )
                np.testing.assert_array_equal(target, truth)
                values.append(probability)
            panels.append(
                ScatterTrajectory(
                    method=method,
                    title=method.upper(),
                    truth=truth,
                    predictions=np.stack(values),
                )
            )
        scatter_animation(
            panels=panels,
            captions=[
                f"{protocol.capitalize()} labels | final test predictions | N={count}"
                for count in ROLLOUTS
            ],
            durations=[1100] * (len(ROLLOUTS) - 1) + [2200],
            destination=destination / f"scatter-by-rollouts-{protocol}",
            footer="Three original seeds pooled; fixed sampled points and full-data conditional means.\nStatistics use all test predictions. Dashed diagonal: perfect probability matching.",
        )

    selected = [
        result
        for result in results
        if result["config"]["seed"] == seed
        and (
            result["config"]["method"] == "maxrl"
            or representative(
                result=result, protocol=result["config"]["protocol"], rollouts=64
            )
        )
    ]
    trajectory_audit = []
    trajectories = {}
    for result in selected:
        config = result["config"]
        trajectory = torch.load(
            root / "runs" / config["identifier"] / "validation-trajectory.pt",
            map_location="cpu",
            weights_only=True,
        )
        if (
            result["initial_model_fingerprint"]
            != trajectory["initial_model_fingerprint"]
            or result["dataset_fingerprint"] != trajectory["dataset_fingerprint"]
        ):
            raise ValueError(
                f"Trajectory initialization or dataset mismatch: {config['identifier']}"
            )
        rmse = (trajectory["q"][-1] - trajectory["f"]).square().mean().sqrt().item()
        difference = abs(rmse - result["final"]["validation"]["rmse"])
        if difference > 1e-6:
            raise ValueError(
                f"Trajectory endpoint does not match metrics: {config['identifier']}"
            )
        trajectory_audit.append({
            "run": config["identifier"],
            "endpoint_rmse_difference": difference,
        })
        trajectories[(config["protocol"], config["method"], config["num_rollouts"])] = (
            trajectory
        )
    write_json(path=destination / "trajectory-audit.json", payload=trajectory_audit)
    for protocol in ("shared", "split"):
        reference = trajectories[(protocol, "maxrl", 64)]
        steps = reference["step"].numpy()
        chosen = np.flatnonzero(
            (steps <= 256) | (steps % 128 == 0) | (steps == steps[-1])
        )
        animation_steps = steps[chosen]
        durations = [
            800
            if step == 0
            else 1800
            if step == steps[-1]
            else 300
            if step < 32
            else 160
            if step <= 256
            else 120
            for step in animation_steps
        ]
        truth = reference["f"].numpy()
        for kind in ("methods-n64", "maxrl-all-rollouts"):
            panels = []
            keys = (
                [
                    (
                        method,
                        16 if protocol == "shared" and method in SUPERVISED else 64,
                    )
                    for method in METHODS
                ]
                if kind == "methods-n64"
                else [("maxrl", count) for count in ROLLOUTS]
            )
            for method, count in keys:
                trajectory = trajectories[(protocol, method, count)]
                np.testing.assert_array_equal(trajectory["step"].numpy(), steps)
                np.testing.assert_array_equal(trajectory["f"].numpy(), truth)
                panels.append(
                    ScatterTrajectory(
                        method=method,
                        title=method.upper()
                        if kind == "methods-n64"
                        else f"MaxRL: N={count}",
                        truth=truth,
                        predictions=trajectory["q"].numpy()[chosen],
                    )
                )
            scatter_animation(
                panels=panels,
                captions=[
                    f"{protocol.capitalize()} labels | validation | seed {seed} | update {step}"
                    for step in animation_steps
                ],
                durations=durations,
                destination=destination / f"scatter-training-{protocol}-{kind}",
                footer="Predictions recorded during the original training run; loss-selected learning rates.\nFixed validation points; full-data conditional means and statistics; no interpolation.",
            )
    manifest = {
        "seed": seed,
        "recorded_runs": len(selected),
        "training_gif_steps": animation_steps.tolist(),
        "maximum_endpoint_rmse_difference": max(
            row["endpoint_rmse_difference"] for row in trajectory_audit
        ),
        "original_seed_count": len({row["config"]["seed"] for row in results}),
        "rollouts": list(ROLLOUTS),
    }
    write_json(path=destination / "manifest.json", payload=manifest)
    lines = [
        "# Rollout comparisons and correlation animations",
        "",
        "Static metric and learning-curve comparisons use all three original seeds. Each configuration retains its original loss-selected learning rate.",
        "",
        "Shared-label supervised baselines use one observed label per context and are reused across N. Split-label methods all receive N independent labels per context; increasing N changes both the label budget and the rollout budget.",
        "",
        f"Rollout GIFs show final test predictions as N changes. Training GIFs use validation predictions recorded during the original seed-{seed} runs. Frames use actual model outputs, including early updates, without interpolation. Each scatter uses fixed sampled points; conditional means and statistics use the full split.",
        "",
        f"Verified {len(selected)} recorded trajectories against run metrics. Maximum endpoint RMSE discrepancy: {manifest['maximum_endpoint_rmse_difference']:.8g}. See trajectory-audit.json for every endpoint check.",
        "",
    ]
    for protocol in ("shared", "split"):
        lines.extend((f"## {protocol.capitalize()} labels", ""))
        for name in (
            f"rollout-metrics-{protocol}",
            f"learning-curves-rollouts-{protocol}",
        ):
            lines.extend((f"![{name}]({name}.png)", ""))
        for name in (
            f"scatter-by-rollouts-{protocol}",
            f"scatter-training-{protocol}-methods-n64",
            f"scatter-training-{protocol}-maxrl-all-rollouts",
        ):
            lines.extend((f"![{name}]({name}.gif)", ""))
    (destination / "GALLERY.md").write_text("\n".join(lines))
    print(json.dumps(manifest), flush=True)
    print(destination, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=SEEDS[0])
    arguments = parser.parse_args()
    visualizations(
        root=arguments.root,
        seed=arguments.seed,
    )


if __name__ == "__main__":
    main()
