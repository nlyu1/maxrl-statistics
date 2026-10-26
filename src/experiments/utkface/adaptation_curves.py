import argparse
import base64
import html
import json
import statistics
from pathlib import Path

import numpy as np

from src.experiments.utkface.adaptation_report import (
    COLORS,
    EXTENDED_NOISES,
    LOSS_LABELS,
    LOSSES,
    NOISES,
    POINT_METRICS,
    plt,
    read_json,
    save,
    unique_record,
    write_json,
)

METRICS = {
    "mae_mean": "MAE (years)",
    "rmse_mean": "RMSE (years)",
    "spearman_mean": "Spearman correlation",
}


def load_curves(*, root: Path, records: list[dict]) -> list[dict]:
    curves = []
    for record in records:
        result = read_json(root / "runs" / record["identifier"] / "result.json")
        start = record["config"]["global_start_epoch"]
        history = result["history"]
        expected = list(range(start + 1, record["final_global_epoch"] + 1))
        if [row["epoch"] for row in history] != expected:
            raise ValueError(f"Incomplete epoch history: {record['identifier']}")
        initial = result["initialization"]["validation"]
        points = [{"epoch": start, "validation": initial}] + history
        selected = min(points, key=lambda row: row["validation"]["mae_mean"])
        if selected["epoch"] != record["best_epoch"]:
            raise ValueError(f"Selection mismatch: {record['identifier']}")
        minimum = selected["validation"]["mae_mean"]
        curves.append({
            **record,
            "points": [
                {
                    "epoch": row["epoch"],
                    "validation": {
                        key: row["validation"][key] for key, _ in POINT_METRICS
                    },
                    **{
                        key: row[key]
                        for key in (
                            "train_loss",
                            "grad_norm",
                            "epoch_seconds",
                            "validation_seconds",
                        )
                        if key in row
                    },
                }
                for row in points
            ],
            "curve_summary": {
                "best_validation_mae": minimum,
                "last_validation_mae": points[-1]["validation"]["mae_mean"],
                "last_five_validation_mae": statistics.mean(
                    row["validation"]["mae_mean"] for row in points[-5:]
                ),
                "epochs_within_0_1_of_best": sum(
                    row["validation"]["mae_mean"] <= minimum + 0.1 for row in history
                ),
                "epochs_within_0_25_of_best": sum(
                    row["validation"]["mae_mean"] <= minimum + 0.25 for row in history
                ),
                "train_epoch_seconds_median": statistics.median(
                    row["epoch_seconds"] for row in history
                ),
                "validation_seconds_median": statistics.median(
                    row["validation_seconds"] for row in history
                ),
            },
        })
    return curves


def item(
    record: dict, *, label: str = "", color: str = "", linestyle: str = "-"
) -> dict:
    return {
        "record": record,
        "label": label or record["loss_label"],
        "color": color or COLORS[LOSSES.index(record["loss"])],
        "linestyle": linestyle,
    }


def grid(
    *,
    panels: list[dict],
    title: str,
    metric: str,
    output_dir: Path,
    name: str,
    span: float,
) -> str:
    count = len(panels)
    fig, axes = plt.subplots(2, count, figsize=(4.5 * count, 7.7), squeeze=False)
    handles = {}
    for column, panel in enumerate(panels):
        values = []
        epochs = []
        for entry in panel["items"]:
            record = entry["record"]
            x = np.array([p["epoch"] for p in record["points"]])
            y = np.array(
                [
                    p["validation"][metric]
                    if p["validation"][metric] is not None
                    else np.nan
                    for p in record["points"]
                ],
                dtype=float,
            )
            values.extend(y[np.isfinite(y)].tolist())
            epochs.extend(x.tolist())
            selected = record["best_epoch"]
            position = np.flatnonzero(x == selected)[0]
            for axis in axes[:, column]:
                (line,) = axis.plot(
                    x,
                    y,
                    label=entry["label"],
                    color=entry["color"],
                    ls=entry["linestyle"],
                    lw=1.7,
                    alpha=0.92,
                )
                handles[entry["label"]] = line
                if np.isfinite(y[position]):
                    axis.scatter(
                        [selected],
                        [y[position]],
                        marker="*",
                        s=95,
                        color=entry["color"],
                        edgecolor="white",
                        lw=0.6,
                        zorder=5,
                    )
            if "fork" in panel:
                for axis in axes[:, column]:
                    axis.axvline(panel["fork"], color="#888888", ls=":", lw=0.8)
        upper, lower = axes[:, column]
        upper.set_title(panel["title"], fontweight="bold")
        lower.set_title("Near-best region", fontsize=10)
        best = max(values) if metric == "spearman_mean" else min(values)
        if metric == "spearman_mean":
            upper.set_ylim(min(-0.05, min(values) - 0.02), 1.01)
            lower.set_ylim(max(-1.0, best - span), min(1.0, best + 0.008))
        else:
            upper.set_ylim(max(0, min(values) - 0.35), max(values) + 0.4)
            lower.set_ylim(best - 0.08, best + span)
        close_epochs = [
            p["epoch"]
            for entry in panel["items"]
            for p in entry["record"]["points"]
            if p["validation"][metric] is not None
            and (
                p["validation"][metric] >= best - span
                if metric == "spearman_mean"
                else p["validation"][metric] <= best + span
            )
        ]
        lower.set_xlim(max(min(epochs), min(close_epochs) - 2), max(epochs) + 0.5)
        upper.set_xlim(min(epochs), max(epochs) + 0.5)
        for axis in (upper, lower):
            axis.grid(alpha=0.2)
            axis.set_xlabel("Global epoch")
            axis.spines[["right", "top"]].set_visible(False)
        if column == 0:
            upper.set_ylabel("Clean validation " + METRICS[metric])
            lower.set_ylabel("Clean validation " + METRICS[metric])
    fig.suptitle(title, fontsize=15, fontweight="bold", y=1.015)
    fig.legend(
        handles.values(),
        handles.keys(),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.99),
        ncol=min(6, len(handles)),
        frameon=False,
    )
    fig.text(
        0.5,
        -0.015,
        "Raw epoch curves; no smoothing. Stars mark checkpoints selected by "
        "validation mean MAE. Each lower panel zooms into its best region.",
        ha="center",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0.02, 1, 0.93))
    return save(fig, output_dir, name)


def figures(*, curves: list[dict], output_dir: Path) -> list[dict]:
    output = []

    def render(
        *,
        panels: list[dict],
        title: str,
        name: str,
        metric: str = "mae_mean",
        span: float = 1.3,
    ) -> None:
        path = grid(
            panels=panels,
            title=title,
            metric=metric,
            output_dir=output_dir,
            name=name,
            span=span,
        )
        output.append({"path": path, "title": title})

    for pipeline in ("original", "fixed"):
        panels = [
            {
                "title": "Clean" if noise == 0 else f"Fixed SD {noise:g}",
                "items": [
                    item(
                        unique_record(
                            records=curves,
                            membership="augmentation",
                            pipeline=pipeline,
                            backbone="ImageNet end-to-end",
                            noise_sd=noise,
                            loss=loss,
                        )
                    )
                    for loss in LOSSES
                ],
            }
            for noise in NOISES
        ]
        for metric, span in (
            ("mae_mean", 1.3),
            ("rmse_mean", 2.0),
            ("spearman_mean", 0.08),
        ):
            render(
                panels=panels,
                title=f"End-to-end ImageNet adaptation · {pipeline} inputs",
                name=f"curves_augmentation_{pipeline}_{metric}.png",
                metric=metric,
                span=span,
            )
    for suite in ("scratch_cosine", "scratch_warmup_constant"):
        panels = [
            {
                "title": "Clean" if noise == 0 else f"Fixed SD {noise:g}",
                "items": [
                    item(
                        unique_record(
                            records=curves, suite=suite, noise_sd=noise, loss=loss
                        )
                    )
                    for loss in LOSSES
                ],
            }
            for noise in NOISES
        ]
        render(
            panels=panels,
            title="From scratch · "
            + (
                "LR 1e-3, cosine decay"
                if suite == "scratch_cosine"
                else "LR 3e-4, warmup then constant"
            ),
            name=f"curves_{suite}_mae.png",
            span=2.0,
        )
    panels = [
        {
            "title": "Clean" if noise == 0 else f"Fixed SD {noise:g}",
            "items": [
                item(
                    unique_record(
                        records=curves,
                        membership="frozen_heads",
                        backbone_source=source,
                        loss="mse",
                        noise_sd=noise,
                    ),
                    label=label,
                    color=color,
                )
                for source, label, color in (
                    ("imagenet", "ImageNet features", "#7b4ab5"),
                    ("cat_mse", "SL features", "#276fbf"),
                    ("ce", "CE features", "#d1495b"),
                )
            ],
        }
        for noise in EXTENDED_NOISES
    ]
    render(
        panels=panels,
        title="Frozen feature sources · common fresh categorical MSE head",
        name="curves_frozen_sources_mae.png",
        span=1.5,
    )
    panels = [
        {
            "title": "Clean" if noise == 0 else f"Fixed SD {noise:g}",
            "items": [
                item(
                    unique_record(
                        records=curves,
                        membership="frozen_heads",
                        backbone_source="cat_mse",
                        noise_sd=noise,
                        loss=loss,
                    )
                )
                for loss in LOSSES
            ],
        }
        for noise in EXTENDED_NOISES
    ]
    render(
        panels=panels,
        title="Frozen SL features · five fresh head objectives",
        name="curves_sl_feature_heads_mae.png",
        span=0.7,
    )
    for backbone in ("SL@25 trainable", "SL@25 frozen"):
        panels = []
        for noise in NOISES:
            original = unique_record(
                records=curves,
                membership="augmentation",
                pipeline="original",
                backbone="ImageNet end-to-end",
                noise_sd=noise,
                loss="mse",
            )
            entries = [
                item(
                    original,
                    label="Original SL (0–50)",
                    color="#333333",
                    linestyle="--",
                )
            ]
            entries.extend(
                item(
                    unique_record(
                        records=curves,
                        suite="post_training",
                        backbone=backbone,
                        noise_sd=noise,
                        loss=loss,
                    ),
                    label="MSE reset control" if loss == "mse" else LOSS_LABELS[loss],
                )
                for loss in LOSSES[:4]
            )
            panels.append({
                "title": "Clean" if noise == 0 else f"Fixed SD {noise:g}",
                "items": entries,
                "fork": 25,
            })
        render(
            panels=panels,
            title="SL epoch-25 continuation · " + backbone,
            name=f"curves_post_{'frozen' if backbone.endswith('frozen') else 'trainable'}_mae.png",
            span=1.1,
        )
    return output


def comparison_summary(curves: list[dict]) -> dict:
    pairs = []
    for pipeline in ("original", "fixed"):
        for noise in NOISES:
            runs = [
                unique_record(
                    records=curves,
                    membership="augmentation",
                    pipeline=pipeline,
                    backbone="ImageNet end-to-end",
                    noise_sd=noise,
                    loss=loss,
                )
                for loss in ("mse", "tailrl_population")
            ]
            pairs.append({
                "pipeline": pipeline,
                "noise_sd": noise,
                "runs": [
                    {
                        "job": r["identifier"],
                        "best_epoch": r["best_epoch"],
                        **r["curve_summary"],
                        "best_test": r["best_test"],
                        "final_test": r["final_test"],
                    }
                    for r in runs
                ],
            })
    timing = {}
    for frozen in (False, True):
        rows = [r for r in curves if r["config"]["use_feature_bank"] == frozen]
        timing["cached_head" if frozen else "image_training"] = {
            "n_runs": len(rows),
            "median_run_median_training_epoch_seconds": statistics.median(
                r["curve_summary"]["train_epoch_seconds_median"] for r in rows
            ),
            "median_run_median_validation_seconds": statistics.median(
                r["curve_summary"]["validation_seconds_median"] for r in rows
            ),
        }
    return {"augmentation_mse_tail": pairs, "observed_timing": timing}


def gallery(*, figures: list[dict]) -> str:
    images = []
    for figure in figures:
        encoded = base64.b64encode(Path(figure["path"]).read_bytes()).decode()
        images.append(
            f"<figure><h2>{html.escape(figure['title'])}</h2>"
            f'<img src="data:image/png;base64,{encoded}"></figure>'
        )
    return (
        "<!doctype html><html><head><meta charset=utf-8>"
        "<meta name=viewport content='width=device-width'><title>Training curves</title>"
        "<style>body{font:16px/1.5 system-ui;margin:30px auto;max-width:1800px;"
        "padding:0 24px;background:#f4f6f9;color:#182235}figure{margin:36px 0;"
        "background:white;padding:16px;border:1px solid #d7dfe8}img{width:100%;"
        "height:auto}h2{font-size:20px}</style></head><body>"
        "<h1>Training trajectories and near-best regions</h1>"
        "<p><a href='report.html'>All test results and frozen-head tables</a></p>"
        "<p>178 full training histories were checked against their selected checkpoint. "
        "The plots show clean validation metrics at every epoch, including initialization. "
        "Stars use validation mean-MAE selection, also on RMSE and Spearman plots. "
        "Each figure shows both the full range and a labeled zoom; lines are unsmoothed. "
        "Test data were evaluated at the selected and final checkpoints only. "
        "All runs use one training seed and one fixed corruption field.</p>"
        + "".join(images)
        + "</body></html>"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = read_json(args.output_dir / "report-data.json")
    curves = load_curves(root=args.root, records=report["records"])
    output = figures(curves=curves, output_dir=args.output_dir)
    summary = comparison_summary(curves)
    write_json(
        args.output_dir / "training-curves-data.json",
        {"curves": curves, "figures": output, "summary": summary},
    )
    write_json(args.output_dir / "training-curves-summary.json", summary)
    (args.output_dir / "training-curves.html").write_text(gallery(figures=output))
    print(json.dumps({"runs": len(curves), "figures": len(output), "summary": summary}))


if __name__ == "__main__":
    main()
