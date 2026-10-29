import argparse
import hashlib
import json
import textwrap
from collections import defaultdict
from pathlib import Path

import matplotlib
import numpy as np
import polars as pl

matplotlib.use("Agg")
import matplotlib.pyplot as plt

LABELS = {
    "categorical_mse": "Categorical MSE",
    "categorical_tailrl": "Categorical TailRL",
    "categorical_ce": "Categorical CE",
    "scalar_mse": "Scalar MSE",
    "scalar_mae": "Scalar MAE",
    "binomial_ce": "Binomial CE (one score)",
    "binomial_tailrl": "Binomial TailRL (one score)",
    "head_tailrl_backbone_mse": "TailRL head / MSE backbone",
    "head_mse_backbone_tailrl": "MSE head / TailRL backbone",
    "binomial_ce_free": "CE: shared score + free residual",
    "binomial_ce_affine": "CE: shared score + age-affine residual",
    "binomial_tailrl_free": "TailRL: shared score + free residual",
    "binomial_tailrl_affine": "TailRL: shared score + age-affine residual",
    "categorical_ce_binomial_init_fp32": "CE: ordinary head, ordinal start (FP32)",
    "categorical_tailrl_binomial_init_fp32": "TailRL: ordinary head, ordinal start (FP32)",
    "categorical_ce_random_fp32": "CE: ordinary head, random start (FP32)",
    "categorical_tailrl_random_fp32": "TailRL: ordinary head, random start (FP32)",
    "binomial_ce_free_fp32": "CE: score + free residual (FP32)",
    "binomial_tailrl_free_fp32": "TailRL: score + free residual (FP32)",
}
COLORS = dict(
    zip(
        LABELS,
        (
            "#c83b45",
            "#0077a1",
            "#444444",
            "#b05fab",
            "#7b3294",
            "#8c6d31",
            "#009a85",
            "#e69f00",
            "#56b4e9",
            "#d55e00",
            "#cc79a7",
            "#009e73",
            "#882255",
            "#332288",
            "#88ccee",
            "#44aa99",
            "#117733",
            "#ddcc77",
            "#aa4499",
        ),
        strict=True,
    )
)
METRICS = {
    "mae_mean": "Mean-age MAE (years)",
    "rmse_mean": "Mean-age RMSE (years)",
    "spearman_mean": "Mean-age Spearman correlation",
}
GROUPS = {
    "main": (
        "categorical_mse",
        "categorical_tailrl",
        "categorical_ce",
        "scalar_mse",
        "scalar_mae",
    ),
    "score-controls": (
        "categorical_ce",
        "binomial_ce",
        "binomial_ce_free",
        "binomial_ce_affine",
        "scalar_mse",
        "scalar_mae",
    ),
    "tail-score-controls": (
        "categorical_tailrl",
        "binomial_tailrl",
        "binomial_tailrl_free",
        "binomial_tailrl_affine",
        "scalar_mse",
        "scalar_mae",
    ),
    "routes": (
        "categorical_mse",
        "head_mse_backbone_tailrl",
        "head_tailrl_backbone_mse",
        "categorical_tailrl",
    ),
    "initialization-controls-ce": (
        "categorical_ce_binomial_init_fp32",
        "categorical_ce_random_fp32",
        "binomial_ce_free_fp32",
    ),
    "initialization-controls-tail": (
        "categorical_tailrl_binomial_init_fp32",
        "categorical_tailrl_random_fp32",
        "binomial_tailrl_free_fp32",
    ),
}
CONTRASTS = (
    ("head_mse_backbone_tailrl", "categorical_mse"),
    ("categorical_tailrl", "head_tailrl_backbone_mse"),
    ("binomial_ce", "categorical_ce"),
    ("binomial_tailrl", "categorical_tailrl"),
    ("scalar_mse", "categorical_mse"),
    ("scalar_mae", "scalar_mse"),
    ("binomial_ce_free", "binomial_ce"),
    ("binomial_tailrl_free", "binomial_tailrl"),
    ("binomial_ce_free", "binomial_ce_affine"),
    ("binomial_tailrl_free", "binomial_tailrl_affine"),
    ("binomial_ce_affine", "binomial_ce"),
    ("binomial_tailrl_affine", "binomial_tailrl"),
    ("categorical_ce_binomial_init_fp32", "categorical_ce_random_fp32"),
    ("categorical_tailrl_binomial_init_fp32", "categorical_tailrl_random_fp32"),
    ("categorical_ce_binomial_init_fp32", "binomial_ce_free_fp32"),
    ("categorical_tailrl_binomial_init_fp32", "binomial_tailrl_free_fp32"),
)


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def numeric(metrics: dict) -> dict:
    return {
        key: value
        for key, value in metrics.items()
        if key != "count"
        and not isinstance(value, bool)
        and (value is None or isinstance(value, (int, float)))
    }


def load(*, root: Path, allow_incomplete: bool) -> tuple[dict, list[dict], list[str]]:
    plan = read(root / "plan.json")
    jobs, runs, missing = plan["jobs"], [], []
    if len({job["identifier"] for job in jobs}) != len(jobs):
        raise ValueError("Duplicate planned identifiers")
    if plan["phase"] != "scientific" and not allow_incomplete:
        raise ValueError(
            "Smoke results are not a scientific cohort; use --allow-incomplete for labeled progress"
        )
    for job in jobs:
        directory = root / "runs" / job["identifier"]
        path = directory / "result.json"
        config = read(root / "configs" / f"{job['identifier']}.json")
        if (
            config["variant"] != job["variant"]
            or config["adaptation"]["run"]["noise"]["std_years"] != job["noise_sd"]
        ):
            raise ValueError(
                f"Job labels differ from configuration: {job['identifier']}"
            )
        result = read(path) if path.is_file() else {}
        complete = result.get("status") == "complete"
        if result and result["config"] != config:
            raise ValueError(f"Result and planned configuration differ: {path}")
        if not complete:
            missing.append(job["identifier"])
            if not allow_incomplete:
                continue
            result = {
                "config": config,
                "history": read(directory / "history.json")
                if (directory / "history.json").is_file()
                else [],
                "initialization": read(directory / "initialization.json")
                if (directory / "initialization.json").is_file()
                else {},
                "provenance": read(directory / "provenance.json")
                if (directory / "provenance.json").is_file()
                else {},
            }
        expected = list(range(1, config["adaptation"]["run"]["epochs"] + 1))
        epochs = [row["epoch"] for row in result["history"]]
        if epochs != expected[: len(epochs)] or (complete and epochs != expected):
            raise ValueError(
                f"Incomplete or nonconsecutive history: {job['identifier']}"
            )
        if complete:
            scores = [result["initialization"]["validation"]["mae_mean"]] + [
                row["validation"]["mae_mean"] for row in result["history"]
            ]
            if result["best_epoch"] != min(range(len(scores)), key=scores.__getitem__):
                raise ValueError(
                    f"Selected checkpoint differs from earliest validation minimum: {job['identifier']}"
                )
        runs.append({**job, "result": result, "complete": complete})
    if missing and not allow_incomplete:
        raise ValueError(f"Missing {len(missing)} planned results: {missing}")
    cells = defaultdict(set)
    for job in jobs:
        cells[(job["noise_sd"], job["variant"])].add(job["block"])
    blocks = {job["block"] for job in jobs}
    variants = {job["variant"] for job in jobs}
    plan["cohort_valid"] = len(blocks) == 3 and all(
        cells[(noise, variant)] == blocks for noise in (0, 10) for variant in variants
    )
    if not plan["cohort_valid"] and not allow_incomplete:
        raise ValueError("Scientific cells must contain three paired training blocks")
    return plan, runs, missing


def tables(*, runs: list[dict]) -> tuple[list[dict], list[dict]]:
    endpoints, curves = [], []
    for run in runs:
        result = run["result"]
        base = {key: run[key] for key in ("identifier", "block", "noise_sd", "variant")}
        for checkpoint, prefix in (("selected", "best"), ("final", "final")):
            for split, suffix in (("val", "validation"), ("test", "test")):
                for metric, value in numeric(
                    result.get(f"{prefix}_{suffix}") or {}
                ).items():
                    endpoints.append({
                        **base,
                        "checkpoint": checkpoint,
                        "split": split,
                        "metric": metric,
                        "value": value,
                    })
        initial = result.get("initialization", {})
        history = (
            [{"epoch": 0, "validation": initial["validation"]}] if initial else []
        ) + result["history"]
        for row in history:
            values = [
                ("val", key, value) for key, value in numeric(row["validation"]).items()
            ]
            values += [
                ("train", key, row[key])
                for key in ("train_loss", "head_loss", "backbone_loss")
                if key in row
            ]
            values += [
                ("train", f"{group}.{key}", value)
                for group, metrics in row.get("gradient_diagnostics", {}).items()
                for key, value in metrics.items()
            ]
            curves.extend(
                {
                    **base,
                    "epoch": row["epoch"],
                    "split": split,
                    "metric": metric,
                    "value": value,
                }
                for split, metric, value in values
            )
        if run["complete"]:
            endpoints.append({
                **base,
                "checkpoint": "trajectory",
                "split": "train",
                "metric": "selected_epoch",
                "value": result["best_epoch"],
            })
            for key in ("train_loss", "head_loss", "backbone_loss"):
                endpoints.append({
                    **base,
                    "checkpoint": "final",
                    "split": "train",
                    "metric": key,
                    "value": result["history"][-1][key],
                })
            diagnostics = defaultdict(list)
            for row in curves:
                if (
                    row["identifier"] == run["identifier"]
                    and row["split"] == "train"
                    and row["value"] is not None
                ):
                    diagnostics[row["metric"]].append(row["value"])
            endpoints.extend(
                {
                    **base,
                    "checkpoint": "trajectory",
                    "split": "train",
                    "metric": f"epoch_mean.{key}",
                    "value": float(np.mean(values)),
                }
                for key, values in diagnostics.items()
            )
    return endpoints, curves


def pairing_audit(*, runs: list[dict]) -> dict:
    records, checks = [], []
    for run in runs:
        result, config = run["result"], run["result"]["config"]
        provenance = result.get("provenance", {})
        normalized = json.loads(json.dumps(config))
        normalized.pop("variant")
        for key in ("output_dir", "head"):
            normalized["adaptation"]["run"].pop(key)
        normalized["adaptation"]["run"]["loss"].pop("method")
        record = {
            key: run[key]
            for key in ("identifier", "block", "noise_sd", "variant", "complete")
        }
        record.update({
            "config_sha256": digest(config),
            "paired_config_sha256": digest(normalized),
            "source_sha256": digest(provenance.get("source_sha256", {})),
            "head_family": "scalar"
            if run["variant"].startswith("scalar_")
            else "categorical_binomial_init"
            if "_binomial_init_" in run["variant"]
            else "binomial"
            if run["variant"].startswith("binomial_")
            else "categorical",
        })
        record.update({
            key: provenance.get(key)
            for key in (
                "manifest_fingerprint",
                "manifest_sha256",
                "training_indices_sha256",
                "clean_labels_sha256",
                "encoder_initial_sha256",
                "head_initial_sha256",
                "score_initial_sha256",
                "residual_initial_sha256",
                "model_seed",
                "head_seed",
                "view_seed",
                "order_seed",
                "noise_seed",
            )
        })
        for field in (
            "consumed_indices_sha256",
            "global_labels_sha256",
            "training_labels_sha256",
            "consumed_labels_sha256",
        ):
            record[field] = digest([
                (row["epoch"], row[field]) for row in result["history"]
            ])
        records.append(record)
    for block, noise in sorted({(row["block"], row["noise_sd"]) for row in records}):
        group = [
            row
            for row in records
            if row["block"] == block and row["noise_sd"] == noise and row["complete"]
        ]
        fields = (
            [
                key
                for key in group[0]
                if key
                not in {
                    "identifier",
                    "block",
                    "noise_sd",
                    "variant",
                    "complete",
                    "config_sha256",
                    "head_family",
                    "head_initial_sha256",
                    "score_initial_sha256",
                    "residual_initial_sha256",
                }
            ]
            if group
            else []
        )
        for field in fields:
            checks.append({
                "block": block,
                "noise_sd": noise,
                "field": field,
                "family": "all",
                "jobs": len(group),
                "passed": len({row[field] for row in group}) == 1
                and all(row[field] is not None for row in group),
            })
        for family in (
            "categorical",
            "categorical_binomial_init",
            "scalar",
            "binomial",
        ):
            heads = [
                row["head_initial_sha256"]
                for row in group
                if row["head_family"] == family
            ]
            if heads:
                checks.append({
                    "block": block,
                    "noise_sd": noise,
                    "field": "head_initial_sha256",
                    "family": family,
                    "jobs": len(heads),
                    "passed": len(set(heads)) == 1 and heads[0] is not None,
                })
        for field, families in (
            (
                "score_initial_sha256",
                {"scalar", "binomial", "categorical_binomial_init"},
            ),
            ("residual_initial_sha256", {"binomial"}),
        ):
            values = [row[field] for row in group if row["head_family"] in families]
            if values:
                checks.append({
                    "block": block,
                    "noise_sd": noise,
                    "field": field,
                    "family": "+".join(sorted(families)),
                    "jobs": len(values),
                    "passed": len(set(values)) == 1 and values[0] is not None,
                })
    if any(not check["passed"] for check in checks):
        raise ValueError(
            f"Paired-block audit failed: {[check for check in checks if not check['passed']]}"
        )
    complete = [row for row in records if row["complete"]]
    for field in (
        "source_sha256",
        "manifest_fingerprint",
        "manifest_sha256",
        "training_indices_sha256",
        "clean_labels_sha256",
    ):
        if complete and len({row[field] for row in complete}) != 1:
            raise ValueError(f"Scientific cohort differs in {field}")
    return {
        "runs": records,
        "checks": checks,
        "scope": "Completed jobs within each block/noise cell; head equality only within compatible parameter families. Partial jobs are listed but not compared.",
    }


def aggregate(rows: list[dict], keys: list[str]) -> pl.DataFrame:
    return (
        pl
        .DataFrame(rows)
        .with_columns(pl.col("value").cast(pl.Float64))
        .group_by(keys)
        .agg(
            pl.len().alias("n_rows"),
            pl.col("value").count().alias("n_defined"),
            pl.col("value").mean().alias("mean"),
            pl.col("value").std(ddof=1).alias("sample_sd"),
        )
        .sort(keys)
    )


def contrasts(endpoints: list[dict]) -> list[dict]:
    lookup = {
        (
            row["block"],
            row["noise_sd"],
            row["variant"],
            row["checkpoint"],
            row["split"],
            row["metric"],
        ): row
        for row in endpoints
    }
    rows = []
    for first, second in CONTRASTS:
        for key, left in lookup.items():
            if (
                key[2] != first
                or key[3] not in {"selected", "final"}
                or key[4] not in {"val", "test"}
            ):
                continue
            right = lookup.get((*key[:2], second, *key[3:]))
            if right is not None:
                value = (
                    left["value"] - right["value"]
                    if left["value"] is not None and right["value"] is not None
                    else None
                )
                rows.append({
                    "block": key[0],
                    "noise_sd": key[1],
                    "contrast": f"{first} - {second}",
                    "first": first,
                    "second": second,
                    "checkpoint": key[3],
                    "split": key[4],
                    "metric": key[5],
                    "first_value": left["value"],
                    "second_value": right["value"],
                    "value": value,
                })
    return rows


def save(
    *, figure, output: Path, name: str, formats: tuple[str, ...], note: str
) -> list[str]:
    figure.text(
        0.5, 0.015, textwrap.fill(note, width=160), ha="center", va="bottom", fontsize=8
    )
    figure.tight_layout(rect=(0, 0.12, 1, 0.82 if figure.legends else 0.94))
    paths = []
    for extension in formats:
        path = output / f"{name}.{extension}"
        figure.savefig(path, dpi=190, bbox_inches="tight")
        paths.append(str(path))
    plt.close(figure)
    return paths


def curve_figure(
    *,
    rows: list[dict],
    runs: list[dict],
    variants: tuple[str, ...],
    metric: str,
    split: str,
    zoom: bool,
    title: str,
):
    noises = sorted({row["noise_sd"] for row in rows})
    figure, axes = plt.subplots(
        1, len(noises), figsize=(7 * len(noises), 4.8), squeeze=False
    )
    for axis, noise in zip(axes[0], noises, strict=True):
        selected = []
        for variant in variants:
            points = [
                row
                for row in rows
                if row["noise_sd"] == noise
                and row["variant"] == variant
                and row["metric"] == metric
                and row["split"] == split
            ]
            if not points:
                continue
            grouped = aggregate(points, ["epoch"]).to_dicts()
            epochs, means = (
                np.array([p["epoch"] for p in grouped]),
                np.array([p["mean"] for p in grouped], dtype=float),
            )
            deviations = np.array([
                p["sample_sd"] if p["sample_sd"] is not None else np.nan
                for p in grouped
            ])
            for block in sorted({p["block"] for p in points}):
                block_points = sorted(
                    (p for p in points if p["block"] == block and p["epoch"] > 0),
                    key=lambda p: p["epoch"],
                )
                axis.plot(
                    [p["epoch"] for p in block_points],
                    [p["value"] for p in block_points],
                    color=COLORS[variant],
                    alpha=0.2,
                    linewidth=0.8,
                )
            trained = epochs > 0
            axis.plot(
                epochs[trained],
                means[trained],
                color=COLORS[variant],
                label=LABELS[variant],
                linewidth=1.8,
            )
            axis.fill_between(
                epochs[trained],
                (means - deviations)[trained],
                (means + deviations)[trained],
                color=COLORS[variant],
                alpha=0.13,
            )
            axis.scatter(
                epochs[~trained],
                means[~trained],
                color=COLORS[variant],
                marker="o",
                s=28,
                facecolors="none",
            )
            best = [
                run["result"]
                for run in runs
                if run["noise_sd"] == noise
                and run["variant"] == variant
                and run["complete"]
                and run["result"]["best_validation"].get(metric) is not None
            ]
            if best and split == "val":
                x, y = (
                    np.mean([r["best_epoch"] for r in best]),
                    np.mean([r["best_validation"][metric] for r in best]),
                )
                axis.scatter(x, y, color=COLORS[variant], marker="*", s=90, zorder=5)
                selected.append(y)
        if zoom and selected:
            low, high = min(selected), max(selected)
            margin = max((high - low) * 0.2, 0.03 if "spearman" in metric else 0.25)
            axis.set_ylim(low - margin, high + margin)
        axis.set(
            title="Clean labels"
            if noise == 0
            else f"Fixed label noise SD {noise:g} years",
            xlabel="Training epoch",
            ylabel=METRICS.get(metric, metric),
        )
        axis.grid(alpha=0.2)
    legend = {
        label: handle
        for axis in axes[0]
        for handle, label in zip(*axis.get_legend_handles_labels(), strict=True)
    }
    figure.legend(
        legend.values(),
        legend.keys(),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.99),
        ncol=2,
        fontsize=8,
    )
    figure.suptitle(title, y=1.06)
    return figure


def point_figure(*, rows: list[dict], metric: str, split: str, paired: bool):
    selected = [
        row
        for row in rows
        if row["metric"] == metric
        and row["split"] == split
        and row["checkpoint"] in {"selected", "final"}
    ]
    names = list(
        dict.fromkeys(row["contrast" if paired else "variant"] for row in selected)
    )
    noises = sorted({row["noise_sd"] for row in selected})
    figure, axes = plt.subplots(
        1,
        len(noises),
        figsize=(8 * len(noises), max(4, len(names) * 0.55)),
        squeeze=False,
        sharey=True,
    )
    for axis, noise in zip(axes[0], noises, strict=True):
        for index, name in enumerate(names):
            for checkpoint, offset, marker in (
                ("selected", -0.12, "o"),
                ("final", 0.12, "s"),
            ):
                values = [
                    r["value"]
                    for r in selected
                    if r["noise_sd"] == noise
                    and r["checkpoint"] == checkpoint
                    and r["contrast" if paired else "variant"] == name
                    and r["value"] is not None
                ]
                if values:
                    color = "#0077a1" if checkpoint == "selected" else "#c83b45"
                    axis.scatter(
                        values,
                        (np.arange(len(values)) - (len(values) - 1) / 2) * 0.055
                        + index
                        + offset,
                        color=color,
                        alpha=0.45,
                        s=18,
                    )
                    axis.errorbar(
                        np.mean(values),
                        index + offset,
                        xerr=np.std(values, ddof=1) if len(values) > 1 else None,
                        fmt=marker,
                        color=color,
                        capsize=3,
                        label=checkpoint if index == 0 else None,
                    )
        axis.set_yticks(
            range(len(names)),
            [
                " − ".join(LABELS[v] for v in name.split(" - "))
                if paired
                else LABELS[name]
                for name in names
            ],
            fontsize=8,
        )
        axis.set_ylim(len(names) - 0.5, -0.5)
        axis.set(
            title=f"Label-noise SD {noise:g} years",
            xlabel=("Δ " if paired else "") + METRICS[metric],
        )
        axis.grid(axis="x", alpha=0.2)
        if paired:
            axis.axvline(0, color="black", linewidth=0.8)
        axis.legend(fontsize=8)
    figure.suptitle(
        f"{'Within-block contrasts' if paired else 'All methods'} · clean {split} · validation-selected versus equal final epoch"
    )
    return figure


def report(
    *,
    root: Path,
    output: Path,
    allow_incomplete: bool,
    formats: tuple[str, ...],
    extras: dict,
) -> dict:
    plan, runs, missing = load(root=root, allow_incomplete=allow_incomplete)
    endpoints, curves = tables(runs=runs)
    audit = pairing_audit(runs=runs)
    differences = contrasts(endpoints)
    progress = (
        "PROGRESS ONLY: incomplete or smoke cohort. "
        if missing or plan["phase"] != "scientific" or not plan["cohort_valid"]
        else ""
    )
    output.mkdir(parents=True, exist_ok=True)
    files, frames = [], {}
    for name, rows, keys in (
        (
            "endpoints",
            endpoints,
            ["noise_sd", "variant", "checkpoint", "split", "metric"],
        ),
        ("trajectories", curves, ["noise_sd", "variant", "epoch", "split", "metric"]),
        (
            "contrasts",
            differences,
            [
                "noise_sd",
                "contrast",
                "first",
                "second",
                "checkpoint",
                "split",
                "metric",
            ],
        ),
    ):
        if rows:
            frames[name] = aggregate(rows, keys)
            for suffix, frame in (
                ("individual", pl.DataFrame(rows)),
                ("summary", frames[name]),
            ):
                path = output / f"{name}-{suffix}.tsv"
                frame.write_csv(path, separator="\t")
                files.append(str(path))
    if audit["runs"]:
        pl.DataFrame(audit["runs"]).write_csv(
            output / "pairing-audit-runs.tsv", separator="\t"
        )
    for group, requested in GROUPS.items():
        variants = tuple(v for v in requested if any(r["variant"] == v for r in runs))
        if not variants or (group == "routes" and len(variants) != 4):
            continue
        for metric in METRICS:
            for zoom in (False, True):
                if any(row["metric"] == metric for row in curves):
                    figure = curve_figure(
                        rows=curves,
                        runs=runs,
                        variants=variants,
                        metric=metric,
                        split="val",
                        zoom=zoom,
                        title=f"{group} · {'near-selected y range' if zoom else 'full range'}",
                    )
                    files += save(
                        figure=figure,
                        output=output,
                        name=f"curves-{group}-{metric}-{'near-best' if zoom else 'full'}",
                        formats=formats,
                        note=progress
                        + "Faint lines show individual blocks; lines and bands show mean ± sample SD. Open circles show epoch 0. Stars show the mean of each block's selected epoch and score, a summary point distinct from the mean curve or any individual checkpoint.",
                    )
    if curves:
        for metric in ("backbone.update_rms_mean", "backbone.clip_frequency"):
            figure = curve_figure(
                rows=curves,
                runs=runs,
                variants=tuple(LABELS),
                metric=metric,
                split="train",
                zoom=False,
                title=f"Actual encoder {'Adam update RMS' if 'rms' in metric else 'gradient-clipping frequency'}",
            )
            files += save(
                figure=figure,
                output=output,
                name=f"encoder-{metric.split('.')[-1]}",
                formats=formats,
                note=progress
                + "Update RMS: actual parameter deltas sampled every five steps; clipping: fraction of steps above the separate encoder norm bound.",
            )
    for paired, rows in ((False, endpoints), (True, differences)):
        for split in ("val", "test"):
            for metric in METRICS:
                if any(r["split"] == split and r["metric"] == metric for r in rows):
                    figure = point_figure(
                        rows=rows, metric=metric, split=split, paired=paired
                    )
                    files += save(
                        figure=figure,
                        output=output,
                        name=f"{'paired-contrasts' if paired else 'endpoints'}-{split}-{metric}",
                        formats=formats,
                        note=progress
                        + "Points show individual training blocks. Large markers show means; bars show sample SD across independent paired training/corruption blocks.",
                    )
    metadata = {}
    for name, path in extras.items():
        paths = [path] if path.is_file() else sorted(path.rglob("*.json"))
        metadata[name] = [
            {"path": str(p), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
            for p in paths
        ]
    summary = {
        "schema": "utkface-mechanism-report-v1",
        "root": str(root),
        "phase": plan["phase"],
        "scientific_complete": not progress,
        "planned_cohort_valid": plan["cohort_valid"],
        "initialization": plan["initialization"],
        "epochs": plan["epochs"],
        "planned_jobs": len(plan["jobs"]),
        "completed_jobs": sum(r["complete"] for r in runs),
        "missing_jobs": missing,
        "uncertainty": "Sample SD (ddof=1) across paired joint training/corruption blocks; n_defined is explicit, null SD for fewer than two blocks. No seed CI or image bootstrap.",
        "metric_scope": "All numeric archived metrics; Spearman nulls remain undefined. Train losses retain native objective scales and should not be compared across losses.",
        "contrast_direction": "first minus second; negative MAE/RMSE favors first, positive Spearman favors first",
        "individual_endpoints": endpoints,
        "individual_contrasts": differences,
        "summaries": {name: frame.to_dicts() for name, frame in frames.items()},
        "pairing_audit": audit,
        "optional_artifacts": metadata,
        "files": files,
    }
    (output / "report.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument("--population", type=Path)
    parser.add_argument("--probes", type=Path)
    parser.add_argument(
        "--formats", nargs="+", choices=("png", "svg", "pdf"), default=("png", "svg")
    )
    args = parser.parse_args()
    extras = {
        name: path
        for name in ("population", "probes")
        if (path := getattr(args, name)) is not None
    }
    if any(not path.exists() for path in extras.values()):
        raise FileNotFoundError("An optional artifact path does not exist")
    result = report(
        root=args.root,
        output=args.output_dir,
        allow_incomplete=args.allow_incomplete,
        formats=tuple(args.formats),
        extras=extras,
    )
    print(
        json.dumps({
            "report": str(args.output_dir / "report.json"),
            "completed_jobs": result["completed_jobs"],
            "scientific_complete": result["scientific_complete"],
            "artifacts": len(result["files"]),
        })
    )


if __name__ == "__main__":
    main()
