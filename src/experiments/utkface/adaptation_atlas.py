import argparse
import base64
import hashlib
import html
import json
import shutil
import statistics
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
from PIL import Image
from plotly.offline import get_plotlyjs

from src.experiments.utkface.adaptation_curves import grid, item
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

QUESTIONS = (
    ("q1", "Q1 · Original versus fixed inputs"),
    ("q2_scratch", "Q2 · Pretraining and scratch recipes"),
    ("q2_freeze", "Q2 · Trainable versus frozen ImageNet backbone"),
    ("q3_source", "Q3 · Feature sources, all five head losses"),
    ("q3_heads", "Q3 · Head losses, all three feature sources"),
    ("q3_training", "Q3 · Training the SL and CE source backbones"),
    ("q4", "Q4 · SL epoch-25 continuations"),
    ("previous", "Earlier audit · SL versus TailRL features"),
)


def old_curves(*, root: Path) -> tuple[list[dict], dict, list[dict]]:
    summary = read_json(root / "summary.json")
    records, contrasts = [], []
    for condition, payload in summary["conditions"].items():
        sources = (
            ("clean_mse", "clean_tail")
            if condition == "clean"
            else ("fixed_mse", "fixed_tail")
        )
        for key, fit in payload["head_refits"].items():
            history = read_json(Path(fit["history"]))
            points = [{"epoch": 0, "validation": fit["initial_validation"]}] + history
            source, objective, _ = key.split(":")
            loss = "mse" if objective == "mse" else "tailrl_population"
            records.append({
                "identifier": f"old_{condition}_{source}_{objective}",
                "loss": loss,
                "loss_label": LOSS_LABELS[loss],
                "condition": condition,
                "source": source,
                "noise_sd": 0 if condition == "clean" else 10,
                "best_epoch": fit["best_epoch"],
                "points": points,
                "best_test": fit["test"],
                "final_test": {},
                "pipeline": "fixed",
                "backbone": source + " frozen",
                "config": {"run": {"learning_rate": 0.001}},
            })
        for source in sources:
            original = payload["original_heads"][source]
            history_path = Path(original["checkpoint"]).parent / "history.json"
            history = read_json(history_path)
            loss = "mse" if source.endswith("mse") else "tailrl_population"
            records.append({
                "identifier": f"old_source_{source}",
                "loss": loss,
                "loss_label": LOSS_LABELS[loss],
                "condition": condition,
                "source": source,
                "noise_sd": 0 if condition == "clean" else 10,
                "best_epoch": original["checkpoint_epoch"],
                "points": history,
                "best_test": original["test"],
                "final_test": {},
                "pipeline": "original",
                "backbone": source + " end-to-end",
                "config": {"run": {"learning_rate": 0.001}},
            })
        for key, contrast in payload["paired_test_mae_differences"].items():
            first, second = "", ""
            for source in sources:
                if key == f"{source}:tail_random_minus_mse_random":
                    first, second = (
                        f"old_{condition}_{source}_tail",
                        f"old_{condition}_{source}_mse",
                    )
                for objective in ("mse", "tail"):
                    if key == f"{source}:{objective}_minus_original":
                        first, second = (
                            f"old_{condition}_{source}_{objective}",
                            f"old_source_{source}",
                        )
            for objective in ("mse", "tail"):
                if key == f"tail_backbone_minus_mse_backbone:{objective}_random":
                    first, second = (
                        f"old_{condition}_{sources[1]}_{objective}",
                        f"old_{condition}_{sources[0]}_{objective}",
                    )
            if first:
                contrasts.append({
                    "first": first,
                    "second": second,
                    "delta": contrast["first_minus_second_mae"],
                    "ci": contrast["bootstrap_95_percent"],
                })
    if len(records) != 12:
        raise ValueError(
            "Earlier audit must supply eight head and four source histories"
        )
    return records, summary, contrasts


def catalog(*, current: list[dict], old: list[dict]) -> list[dict]:
    comparisons = []

    def add(
        question: str,
        title: str,
        entries: list[tuple[dict, str]],
        note: str,
        *,
        fork: int = -1,
    ) -> None:
        row = {
            "id": f"{question}_{len(comparisons):03d}",
            "question": question,
            "title": title,
            "note": note,
            "entries": [
                {"id": r["identifier"], "label": label} for r, label in entries
            ],
        }
        if fork >= 0:
            row["fork"] = fork
        comparisons.append(row)

    def new(**query: object) -> dict:
        return unique_record(records=current, **query)

    for noise in NOISES:
        for backbone in ("ImageNet end-to-end", "ImageNet frozen"):
            for loss in LOSSES:
                entries = [
                    (
                        new(
                            membership="augmentation",
                            pipeline=pipeline,
                            backbone=backbone,
                            noise_sd=noise,
                            loss=loss,
                        ),
                        pipeline.title(),
                    )
                    for pipeline in ("original", "fixed")
                ]
                add(
                    "q1",
                    f"SD {noise:g} · {backbone} · {LOSS_LABELS[loss]}",
                    entries,
                    "Only the training input pipeline changes. Original includes crop, flip and color jitter; fixed uses center crops. Evaluation inputs are fixed in both arms.",
                )
        for pipeline in ("original", "fixed"):
            for loss in LOSSES:
                entries = [
                    (
                        new(
                            membership="augmentation",
                            pipeline=pipeline,
                            backbone=backbone,
                            noise_sd=noise,
                            loss=loss,
                        ),
                        label,
                    )
                    for backbone, label in (
                        ("ImageNet end-to-end", "End-to-end"),
                        ("ImageNet frozen", "Frozen ImageNet"),
                    )
                ]
                add(
                    "q2_freeze",
                    f"SD {noise:g} · {pipeline} · {LOSS_LABELS[loss]}",
                    entries,
                    "Both arms start from ImageNet. Frozen mode fits a fresh head on cached features; end-to-end mode adapts the backbone and head.",
                )
        for loss in LOSSES:
            entries = [
                (
                    new(
                        membership="augmentation",
                        pipeline="original",
                        backbone="ImageNet end-to-end",
                        noise_sd=noise,
                        loss=loss,
                    ),
                    "ImageNet · 1e-3 cosine",
                )
            ]
            entries += [
                (new(suite=suite, noise_sd=noise, loss=loss), label)
                for suite, label in (
                    ("scratch_cosine", "Scratch · 1e-3 cosine"),
                    ("scratch_warmup_constant", "Scratch · 3e-4 constant"),
                )
            ]
            add(
                "q2_scratch",
                f"SD {noise:g} · {LOSS_LABELS[loss]} · three recipes",
                entries,
                "All runs use original augmentation and 50 epochs. The scratch recipes change both peak learning rate and decay shape; warmup is 5%.",
            )
    for noise in EXTENDED_NOISES:
        for loss in LOSSES:
            entries = [
                (
                    new(
                        membership="frozen_heads",
                        noise_sd=noise,
                        backbone_source=source,
                        loss=loss,
                    ),
                    label,
                )
                for source, label in (
                    ("imagenet", "ImageNet features"),
                    ("cat_mse", "SL features"),
                    ("ce", "CE features"),
                )
            ]
            add(
                "q3_source",
                f"SD {noise:g} · {LOSS_LABELS[loss]} · feature sources",
                entries,
                "Fresh heads use matching corrupted labels and the same keyed augmented views. Learned source backbones come from their validation-selected checkpoints at this noise level.",
            )
        for source, label in (
            ("imagenet", "ImageNet"),
            ("cat_mse", "SL"),
            ("ce", "CE"),
        ):
            entries = [
                (
                    new(
                        membership="frozen_heads",
                        noise_sd=noise,
                        backbone_source=source,
                        loss=loss,
                    ),
                    LOSS_LABELS[loss],
                )
                for loss in LOSSES
            ]
            add(
                "q3_heads",
                f"SD {noise:g} · {label} features · five head objectives",
                entries,
                "All 512 feature dimensions are fixed. These are fresh-head fits, with 50 optimization epochs; the feature source is identical within this comparison.",
            )
        membership = "augmentation" if noise in NOISES else "source_extensions"
        entries = [
            (
                new(
                    membership=membership,
                    pipeline="original",
                    noise_sd=noise,
                    backbone="ImageNet end-to-end",
                    loss=loss,
                ),
                LOSS_LABELS[loss],
            )
            for loss in ("mse", "ce")
        ]
        add(
            "q3_training",
            f"SD {noise:g} · SL and CE source training",
            entries,
            "These end-to-end source runs supply the frozen SL/CE features. They are not additional head refits.",
        )
    for noise in NOISES:
        parent = new(
            membership="augmentation",
            pipeline="original",
            backbone="ImageNet end-to-end",
            noise_sd=noise,
            loss="mse",
        )
        for backbone in ("SL@25 trainable", "SL@25 frozen"):
            control = new(
                suite="post_training", backbone=backbone, noise_sd=noise, loss="mse"
            )
            all_entries = [(parent, "Original SL (0–50)"), (control, "MSE restart")]
            for loss in LOSSES[1:4]:
                candidate = new(
                    suite="post_training", backbone=backbone, noise_sd=noise, loss=loss
                )
                add(
                    "q4",
                    f"SD {noise:g} · {backbone} · {LOSS_LABELS[loss]} versus MSE",
                    all_entries + [(candidate, LOSS_LABELS[loss])],
                    "All continuation arms retain the same SL epoch-25 model and reset optimizer/schedule for 25 epochs. The original SL baseline can select any epoch 0–50. Only SL parents were used in the final continuation plan.",
                    fork=25,
                )
            entries = [(parent, "Original SL (0–50)")] + [
                (
                    new(
                        suite="post_training",
                        backbone=backbone,
                        noise_sd=noise,
                        loss=loss,
                    ),
                    "MSE restart" if loss == "mse" else LOSS_LABELS[loss],
                )
                for loss in LOSSES[:4]
            ]
            add(
                "q4",
                f"SD {noise:g} · {backbone} · all continuations",
                entries,
                "The same epoch-25 SL parent is used for all four categorical losses. The frozen branch also freezes BatchNorm. CE-source fresh-head fits appear in Q3; CE-parent end-to-end continuations were not part of this final plan.",
                fork=25,
            )
    for condition, sources in (
        ("clean", ("clean_mse", "clean_tail")),
        ("fixed_s10", ("fixed_mse", "fixed_tail")),
    ):
        for source in sources:
            entries = [
                (
                    next(
                        r
                        for r in old
                        if r["identifier"] == f"old_{condition}_{source}_{loss}"
                    ),
                    label,
                )
                for loss, label in (
                    ("mse", "Fresh MSE head"),
                    ("tail", "Fresh TailRL head"),
                )
            ]
            add(
                "previous",
                f"Earlier · {condition} · {source} features · head losses",
                entries,
                "Earlier audit: deterministic center-crop features, common head seed 20261025, common order seed 20261025, 50 head epochs. Only mean MAE/RMSE were recorded for these head histories.",
            )
        for loss in ("mse", "tail"):
            entries = [
                (
                    next(
                        r
                        for r in old
                        if r["identifier"] == f"old_{condition}_{source}_{loss}"
                    ),
                    label,
                )
                for source, label in zip(
                    sources, ("SL features", "TailRL features"), strict=True
                )
            ]
            add(
                "previous",
                f"Earlier · {condition} · common {loss} head objective",
                entries,
                "The same head architecture, initial weights, objective and fitting procedure are used on each feature source. Final fitted head weights differ.",
            )
        add(
            "previous",
            f"Earlier · {condition} · original end-to-end training",
            [
                (
                    next(r for r in old if r["identifier"] == f"old_source_{source}"),
                    label,
                )
                for source, label in zip(
                    sources, ("Original SL", "Original TailRL"), strict=True
                )
            ],
            "These are the original source training histories behind the earlier frozen-head finding, rather than the newly trained source models.",
        )
    covered = {e["id"] for c in comparisons for e in c["entries"]}
    expected = {r["identifier"] for r in current + old}
    if covered != expected:
        raise ValueError(
            f"Catalog coverage differs: missing={expected - covered}, extra={covered - expected}"
        )
    return comparisons


def compact(record: dict) -> dict:
    fields = [field for field, _ in POINT_METRICS]
    return {
        "id": record["identifier"],
        "best_epoch": record["best_epoch"],
        "best": {k: record["best_test"].get(k) for k in fields},
        "final": {k: record["final_test"].get(k) for k in fields},
        "points": [
            {"epoch": p["epoch"], "v": [p["validation"].get(k) for k in fields]}
            for p in record["points"]
        ],
        "backbone": record["backbone"],
        "pipeline": record["pipeline"],
        "lr": record["config"]["run"]["learning_rate"],
    }


def frozen_matrix(*, records: list[dict], output_dir: Path) -> str:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8), sharey=True)
    for axis, source, label in zip(
        axes,
        ("imagenet", "cat_mse", "ce"),
        ("ImageNet features", "SL features", "CE features"),
        strict=True,
    ):
        matrix = np.array([
            [
                unique_record(
                    records=records,
                    membership="frozen_heads",
                    backbone_source=source,
                    noise_sd=noise,
                    loss=loss,
                )["best_test"]["mae_mean"]
                for loss in LOSSES
            ]
            for noise in EXTENDED_NOISES
        ])
        shown = axis.imshow(matrix, vmin=5, vmax=14.1, cmap="YlOrRd", aspect="auto")
        for (y, x), value in np.ndenumerate(matrix):
            axis.text(
                x,
                y,
                f"{value:.3f}",
                ha="center",
                va="center",
                color="white" if value > 11 else "#18283b",
                fontsize=10,
            )
        axis.set_title(label, fontweight="bold")
        axis.set_xticks(
            range(5), [LOSS_LABELS[k] for k in LOSSES], rotation=30, ha="right"
        )
        axis.set_yticks(range(5), [f"SD {n:g}" for n in EXTENDED_NOISES])
    fig.suptitle(
        "All 75 frozen-head fits · clean-test policy-mean MAE", fontweight="bold"
    )
    fig.colorbar(
        shown, ax=axes, label="MAE (years; lower is better)", fraction=0.02, pad=0.02
    )
    fig.subplots_adjust(left=0.055, right=0.89, bottom=0.22, top=0.84, wspace=0.12)
    return save(fig, output_dir, "q3_all_frozen_heads.png")


def freeze_effect(*, records: list[dict], output_dir: Path) -> str:
    fig, axes = plt.subplots(1, 5, figsize=(18, 4.6))
    for axis, loss in zip(axes, LOSSES, strict=True):
        for pipeline, backbone, label, color, style in (
            (
                "original",
                "ImageNet end-to-end",
                "Original · end-to-end",
                COLORS[0],
                "-",
            ),
            ("fixed", "ImageNet end-to-end", "Fixed · end-to-end", COLORS[1], "-"),
            ("original", "ImageNet frozen", "Original · frozen", COLORS[0], "--"),
            ("fixed", "ImageNet frozen", "Fixed · frozen", COLORS[1], "--"),
        ):
            y = [
                unique_record(
                    records=records,
                    membership="augmentation",
                    pipeline=pipeline,
                    backbone=backbone,
                    noise_sd=n,
                    loss=loss,
                )["best_test"]["mae_mean"]
                for n in NOISES
            ]
            axis.plot(NOISES, y, "o", color=color, ls=style, label=label)
        axis.set_title(LOSS_LABELS[loss], fontweight="bold")
        axis.set_xlabel("Fixed label-noise SD")
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("Selected clean-test mean MAE")
    fig.legend(
        *axes[0].get_legend_handles_labels(),
        loc="upper center",
        ncol=4,
        bbox_to_anchor=(0.5, 1.02),
    )
    fig.tight_layout(rect=(0, 0, 1, 0.89))
    return save(fig, output_dir, "q2_frozen_vs_trainable.png")


def previous_figures(*, old: list[dict], summary: dict, output_dir: Path) -> list[str]:
    paths = []
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for axis, (condition, payload) in zip(
        axes, summary["conditions"].items(), strict=True
    ):
        sources = (
            ("clean_mse", "clean_tail")
            if condition == "clean"
            else ("fixed_mse", "fixed_tail")
        )
        for source_index, source in enumerate(sources):
            for objective, offset, color in (
                ("mse", -0.16, COLORS[0]),
                ("tail", 0.16, COLORS[2]),
            ):
                result = payload["head_refits"][f"{source}:{objective}:random"]
                value = result["test"]["mae_mean"]
                axis.bar(
                    source_index + offset,
                    value,
                    width=0.3,
                    color=color,
                    label=f"Fresh {objective} head" if source_index == 0 else "",
                )
                axis.text(
                    source_index + offset,
                    value + 0.018,
                    f"{value:.3f}",
                    ha="center",
                    fontsize=10,
                )
            original = payload["original_heads"][source]["test"]["mae_mean"]
            axis.scatter(
                [source_index],
                [original],
                marker="D",
                color="#444444",
                s=50,
                label="Stored original head" if source_index == 0 else "",
                zorder=4,
            )
            axis.annotate(
                f"Original {original:.3f}",
                (source_index, original),
                xytext=(0, 13),
                textcoords="offset points",
                ha="center",
                fontsize=9,
            )
        axis.set_xticks([0, 1], ["Frozen SL backbone", "Frozen TailRL backbone"])
        axis.set_ylim((5.0, 5.55) if condition == "clean" else (6.0, 7.45))
        axis.set_title(
            "Clean" if condition == "clean" else "Fixed SD10", fontweight="bold"
        )
        axis.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("Selected clean-test policy-mean MAE (years)")
    axes[1].legend(loc="upper left", fontsize=9)
    fig.suptitle(
        "Earlier frozen-feature audit · eight matched head refits", fontweight="bold"
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    paths.append(save(fig, output_dir, "previous_feature_audit.png"))
    panels = []
    for condition in ("clean", "fixed_s10"):
        for ending in ("mse", "tail"):
            records = [
                r
                for r in old
                if not r["identifier"].startswith("old_source_")
                and r["condition"] == condition
                and r["source"].endswith(ending)
            ]
            panels.append({
                "title": f"{condition} · {ending} features",
                "items": [item(r) for r in records],
            })
    paths.append(
        grid(
            panels=panels,
            title="Earlier head refits · full curves and near-best regions",
            metric="mae_mean",
            output_dir=output_dir,
            name="previous_head_curves.png",
            span=0.6,
        )
    )
    fig, axis = plt.subplots(figsize=(11, 4.8))
    points = []
    for condition, payload in summary["conditions"].items():
        for objective in ("mse", "tail"):
            row = payload["paired_test_mae_differences"][
                f"tail_backbone_minus_mse_backbone:{objective}_random"
            ]
            points.append((f"{condition} · common {objective} head", row))
        key = "clean_mse" if condition == "clean" else "fixed_mse"
        row = payload["paired_test_mae_differences"][
            f"{key}:tail_random_minus_mse_random"
        ]
        points.append((f"{condition} · Tail head − MSE head on SL features", row))
    for y, (label, row) in enumerate(points):
        value, (lower, upper) = (
            row["first_minus_second_mae"],
            row["bootstrap_95_percent"],
        )
        axis.errorbar(
            value,
            y,
            xerr=[[value - lower], [upper - value]],
            fmt="o",
            capsize=4,
            color=COLORS[0],
        )
    axis.set_yticks(range(len(points)), [p[0] for p in points])
    axis.axvline(0, color="#777777", ls="--", lw=1)
    axis.set_xlabel(
        "First − second test MAE (years); paired test-image bootstrap 95% interval"
    )
    axis.set_title("Earlier representation and readout contrasts", fontweight="bold")
    axis.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    paths.append(save(fig, output_dir, "previous_paired_contrasts.png"))
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
    groups = (
        (
            "clean",
            "clean_mse",
            "clean_tail",
            "ridge_condition",
            "Clean",
            "ridge_condition",
        ),
        (
            "fixed_s10",
            "fixed_mse",
            "fixed_tail",
            "ridge_condition",
            "SD10 · noisy labels",
            "ridge_condition",
        ),
        (
            "fixed_s10",
            "fixed_mse",
            "fixed_tail",
            "ridge_clean",
            "SD10 · oracle clean labels",
            "ridge_clean",
        ),
    )
    for i, (condition, sl, tail, probe, label, contrast) in enumerate(groups):
        payload = summary["conditions"][condition]
        for source, offset, color in ((sl, -0.16, COLORS[0]), (tail, 0.16, COLORS[2])):
            value = payload["backbones"][source][probe]["test"]["mae_mean"]
            axes[0].bar(
                i + offset,
                value,
                width=0.3,
                color=color,
                label=("SL features" if source == sl else "TailRL features")
                if i == 0
                else "",
            )
            axes[0].text(
                i + offset, value + 0.025, f"{value:.3f}", ha="center", fontsize=9
            )
        row = payload["paired_test_mae_differences"][
            f"tail_backbone_minus_mse_backbone:{contrast}"
        ]
        value, (low, high) = row["first_minus_second_mae"], row["bootstrap_95_percent"]
        axes[1].errorbar(
            value,
            i,
            xerr=[[value - low], [high - value]],
            fmt="o",
            capsize=4,
            color=COLORS[0],
        )
    axes[0].set_xticks(range(3), [g[4] for g in groups], rotation=15, ha="right")
    axes[0].set_ylim(5.0, 7.05)
    axes[0].set_ylabel("Selected test mean MAE")
    axes[0].legend(fontsize=9)
    axes[1].set_yticks(range(3), [g[4] for g in groups])
    axes[1].axvline(0, color="#777777", ls="--", lw=1)
    axes[1].set_xlabel("Tail features − SL features · paired 95% interval")
    for axis in axes:
        axis.grid(alpha=0.2)
    fig.suptitle(
        "Earlier scalar ridge probes · oracle labels are a representation diagnostic",
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    paths.append(save(fig, output_dir, "previous_ridge_probes.png"))
    return paths


def benchmark_figures(*, reports: Path, output_dir: Path) -> tuple[list[str], dict]:
    cohorts = read_json(reports / "utkface-sharing-eight-20261026" / "summary.json")[
        "cohorts"
    ]
    rows = []
    for index, c in enumerate(cohorts):
        name = f"{index:02d}-{c['method']}-jobs{c['concurrency']}"
        configs = [
            read_json(
                reports
                / "utkface-sharing-eight-20261026"
                / name
                / f"replica{i}"
                / "worker-config.json"
            )
            for i in range(c["concurrency"])
        ]
        normalized = [
            {k: v for k, v in config.items() if k != "output_dir"} for config in configs
        ]
        if any(x != normalized[0] for x in normalized):
            raise ValueError("Sharing replicas do not use identical configurations")
        mae = [r["validation"]["mae_mean"] for r in c["replicas"]]
        rows.append({
            "method": c["method"],
            "concurrency": c["concurrency"],
            "seed": configs[0]["seed"],
            "epochs": c["epochs"],
            "mae": mae,
            "standard_deviation": statistics.stdev(mae),
            "range": max(mae) - min(mae),
            **{
                k: c[k]
                for k in (
                    "aggregate_images_per_second",
                    "peak_device_used_mib",
                    "mean_job_epoch_seconds",
                )
            },
        })
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3))
    for loss, color, label in (
        ("mse", COLORS[0], "MSE"),
        ("tailrl_population", COLORS[2], "TailRL"),
    ):
        points = sorted(
            (r for r in rows if r["method"] == loss), key=lambda r: r["concurrency"]
        )
        axes[0].plot(
            [r["concurrency"] for r in points],
            [r["aggregate_images_per_second"] for r in points],
            "o-",
            label=label,
            color=color,
        )
        axes[1].plot(
            [r["concurrency"] for r in points],
            [r["peak_device_used_mib"] / 1024 for r in points],
            "o-",
            label=label,
            color=color,
        )
    axes[0].set_ylabel("Aggregate training images / second")
    axes[1].set_ylabel("Peak total GPU memory (GiB)")
    for axis in axes:
        axis.set_xticks([4, 8])
        axis.set_xlabel("Concurrent jobs per GPU")
        axis.grid(alpha=0.2)
        axis.legend()
    axes[0].set_title("Throughput increases about 19–20%", fontweight="bold")
    axes[1].set_title("Memory roughly doubles", fontweight="bold")
    fig.suptitle(
        "Archived four-epoch sharing probes · steady epochs 2–4", fontweight="bold"
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    paths = [save(fig, output_dir, "gpu_sharing.png")]
    fig, axis = plt.subplots(figsize=(11, 4.7))
    for i, row in enumerate(rows):
        n = len(row["mae"])
        x = i + np.linspace(-0.16, 0.16, n)
        axis.scatter(
            x,
            row["mae"],
            color=COLORS[0] if row["method"] == "mse" else COLORS[2],
            s=45,
        )
        axis.plot(
            [i - 0.22, i + 0.22],
            [statistics.mean(row["mae"])] * 2,
            color="#444444",
            lw=2,
        )
    axis.set_xticks(
        range(len(rows)),
        [
            f"{LOSS_LABELS[r['method']]} · {r['concurrency']} jobs\nsame configured seed {r['seed']}"
            for r in rows
        ],
    )
    axis.set_ylabel("Selected validation MAE after four training epochs")
    axis.set_title(
        "Short-run training repeatability · one dot per independently trained worker",
        fontweight="bold",
    )
    axis.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    paths.append(save(fig, output_dir, "training_repeatability.png"))
    return paths, {
        "short_probe_repeats": rows,
        "scope": "Four-epoch original-protocol training repeats; not repeated evaluation of one checkpoint or a seed interval for the new 50-epoch study.",
    }


def image_html(path: str) -> str:
    svg = Path(path).with_suffix(".svg")
    encoded = base64.b64encode(svg.read_bytes()).decode()
    return f'<figure><img alt="{html.escape(svg.stem)}" src="data:image/svg+xml;base64,{encoded}"></figure>'


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--previous-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    current = read_json(args.report_dir / "training-curves-data.json")["curves"]
    old, earlier, old_contrasts = old_curves(root=args.previous_root)
    comparisons = catalog(current=current, old=old)
    contrasts = [
        {
            "first": r["first"],
            "second": r["second"],
            "delta": r["mae_difference"],
            "ci": r["bootstrap_95"],
        }
        for r in read_json(args.report_dir / "contrasts.json")["contrasts"]
    ]
    data = {
        "runs": [compact(r) for r in current + old],
        "comparisons": comparisons,
        "contrasts": contrasts + old_contrasts,
        "fields": [field for field, _ in POINT_METRICS],
        "questions": [{"id": key, "label": label} for key, label in QUESTIONS],
        "coverage": f"Coverage checked: all {len(current)} current fits and {len(old)} earlier histories appear in {len(comparisons)} comparison views. Each current comparison supports all nine recorded point metrics.",
    }
    write_json(output / "atlas-data.json", data)
    previous = previous_figures(old=old, summary=earlier, output_dir=output)
    matrix = frozen_matrix(records=current, output_dir=output)
    freeze = freeze_effect(records=current, output_dir=output)
    operations, repeatability = benchmark_figures(
        reports=args.report_dir.parent, output_dir=output
    )
    write_json(output / "randomness-evidence.json", repeatability)
    for pipeline in ("original", "fixed"):
        panels = [
            {
                "title": f"SD {noise:g}",
                "items": [
                    item(
                        unique_record(
                            records=current,
                            membership="augmentation",
                            pipeline=pipeline,
                            backbone="ImageNet frozen",
                            noise_sd=noise,
                            loss=loss,
                        )
                    )
                    for loss in LOSSES
                ],
            }
            for noise in NOISES
        ]
        grid(
            panels=panels,
            title=f"Frozen ImageNet head training · {pipeline} views",
            metric="mae_mean",
            output_dir=output,
            name=f"curves_frozen_imagenet_{pipeline}.png",
            span=1.3,
        )
    copied = {}
    for name in (
        "augmentation_mae",
        "scratch_initialization_mae",
        "post_training_mae",
        "curves_post_trainable_mae",
        "curves_post_frozen_mae",
        "curves_augmentation_original_mae_mean",
        "curves_augmentation_fixed_mae_mean",
    ):
        for suffix in (".png", ".svg"):
            shutil.copy2(args.report_dir / (name + suffix), output / (name + suffix))
        copied[name] = str(output / (name + ".png"))
    question_html = (
        '<div class="box finding"><p><strong>Q1 · Input pipeline:</strong> the end-to-end SD10 TailRL–MSE gap changes from −0.265 years with augmentation to +0.968 years with fixed input. Original augmentation wins 10/15 end-to-end method/noise pairs. The comparison changes crop, flip and jitter together.</p></div>'
        + image_html(copied["augmentation_mae"])
        + '<div class="box finding"><p><strong>Q2 · Initialization and freezing:</strong> ImageNet end-to-end training beats scratch 1e-3 cosine in all 15 method/noise pairs. TailRL beats MSE in 3/3 scratch cosine conditions; MSE beats TailRL in 3/3 scratch 3e-4 constant conditions. End-to-end adaptation beats frozen ImageNet features in all 30 input/method/noise pairs.</p></div>'
        + image_html(copied["scratch_initialization_mae"])
        + image_html(freeze)
        + '<div class="box finding"><p><strong>Q3 · Matched features:</strong> SL features beat CE features in all 25 head-loss/noise cells. On SL features, fresh MSE heads beat TailRL heads in all five noise conditions by 0.008–0.060 years. The new source matrix contains ImageNet, SL and CE features.</p></div>'
        + image_html(matrix)
        + '<div class="box finding"><p><strong>Q4 · Post-training:</strong> the selected original SL model remains ahead of all 24 continuations on test MAE. MSE restart controls beat matched TailRL continuations in all six backbone/noise cells. SL parent optima are at epochs 42/21/12 for clean/SD5/SD10; all continuations start at 25. The original baseline can select earlier epochs, so the matched MSE restart comparison is essential.</p></div>'
        + image_html(copied["post_training_mae"])
        + image_html(copied["curves_post_trainable_mae"])
        + image_html(copied["curves_post_frozen_mae"])
    )
    text = Path(__file__).with_suffix(".html").read_text()
    replacements = {
        "__PLOTLY__": get_plotlyjs(),
        "__DATA__": json.dumps(data, separators=(",", ":"), allow_nan=False).replace(
            "</", "<\\/"
        ),
        "__QUESTIONS__": question_html,
        "__PREVIOUS__": "".join(image_html(p) for p in previous),
        "__RANDOMNESS__": image_html(operations[1]),
        "__OPERATIONS__": '<div class="box"><p>The observed median image-training epoch costs 11.03 seconds of training plus 4.22 seconds of validation. Cached head fitting costs 0.447 plus 0.043 seconds, excluding feature-bank construction. These timings come from mixed concurrency. The controlled sharing probes below compare four versus eight jobs/GPU: aggregate throughput rises about 19–20%, while each job takes longer and total memory roughly doubles from 9,000 to 18,000 MiB. Concurrency is limited per GPU and optionally by GPU count.</p></div>'
        + image_html(operations[0]),
    }
    for key, value in replacements.items():
        text = text.replace(key, value)
    (output / "atlas.html").write_text(text)
    selected_previews = [
        matrix,
        freeze,
        *previous,
        *operations,
        copied["augmentation_mae"],
        copied["scratch_initialization_mae"],
        copied["post_training_mae"],
    ]
    for path in selected_previews:
        image = Image.open(path)
        image.thumbnail((2200, 1500))
        image.save(Path(path).with_suffix(".webp"), format="WEBP", quality=86)
    summary = {
        "current_curves": len(current),
        "previous_curves": len(old),
        "comparisons": len(comparisons),
        "point_metrics": len(POINT_METRICS),
        "question_comparisons": {
            key: sum(c["question"] == key for c in comparisons) for key, _ in QUESTIONS
        },
        "new_training_launched": False,
        "repeatability": repeatability,
    }
    write_json(output / "summary.json", summary)
    with ZipFile(
        output / "atlas.zip", "w", compression=ZIP_DEFLATED, compresslevel=6
    ) as archive:
        for p in sorted(output.iterdir()):
            if p.suffix in (".html", ".json", ".svg", ".webp"):
                archive.write(p, arcname=p.name)
    package = output / "atlas.zip"
    print(
        json.dumps({
            **summary,
            "archive_bytes": package.stat().st_size,
            "archive_sha256": hashlib.sha256(package.read_bytes()).hexdigest(),
        })
    )


if __name__ == "__main__":
    main()
