import argparse
import base64
import hashlib
import html
import json
import statistics
import subprocess
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt

SCHEMA = "utkface-adaptation-results-report"
VERSION = 1
POINT_METRICS = (
    ("mae_mean", "MAE mean"),
    ("mae_median", "MAE median"),
    ("mae_mode", "MAE mode"),
    ("rmse_mean", "RMSE mean"),
    ("rmse_median", "RMSE median"),
    ("rmse_mode", "RMSE mode"),
    ("spearman_mean", "ρ mean"),
    ("spearman_median", "ρ median"),
    ("spearman_mode", "ρ mode"),
)
METRIC_FIELDS = {"mae": "mae_mean", "rmse": "rmse_mean", "spearman": "spearman_mean"}
LOSSES = ("mse", "ce", "tailrl_population", "grpo_mse_population", "mae")
NOISES = (0.0, 5.0, 10.0)
EXTENDED_NOISES = (0.0, 1.0, 3.0, 5.0, 10.0)
EXPECTED_AUGMENTATION_MAE = {
    ("mse", 0.0): 5.188055,
    ("mse", 5.0): 5.691205,
    ("mse", 10.0): 6.598493,
    ("tailrl_population", 0.0): 5.310527,
    ("tailrl_population", 5.0): 5.872117,
    ("tailrl_population", 10.0): 6.333085,
}
EXPECTED_SCRATCH_MAE = {
    ("scratch_cosine", "mse", 0.0): 6.316574,
    ("scratch_cosine", "mse", 5.0): 6.131762,
    ("scratch_cosine", "mse", 10.0): 7.572000,
    ("scratch_cosine", "tailrl_population", 0.0): 5.746922,
    ("scratch_cosine", "tailrl_population", 5.0): 6.091205,
    ("scratch_cosine", "tailrl_population", 10.0): 7.093164,
    ("scratch_warmup_constant", "mse", 0.0): 5.904485,
    ("scratch_warmup_constant", "mse", 5.0): 6.157398,
    ("scratch_warmup_constant", "mse", 10.0): 7.175081,
    ("scratch_warmup_constant", "tailrl_population", 0.0): 6.159701,
    ("scratch_warmup_constant", "tailrl_population", 5.0): 6.274686,
    ("scratch_warmup_constant", "tailrl_population", 10.0): 7.258301,
}
EXPECTED_PARENT_BEST_EPOCHS = {0.0: 42, 5.0: 21, 10.0: 12}
LOSS_LABELS = {
    "mse": "Cat. MSE",
    "ce": "CE",
    "tailrl_population": "TailRL",
    "grpo_mse_population": "GRPO-MSE",
    "mae": "Scalar MAE",
}
SUITE_COUNTS = {
    "augmentation": 60,
    "scratch_cosine": 15,
    "scratch_warmup_constant": 15,
    "frozen_heads": 75,
    "post_training": 24,
    "source_extensions": 4,
}
SUITE_PRIORITY = (
    "post_training",
    "scratch_warmup_constant",
    "scratch_cosine",
    "source_extensions",
    "frozen_heads",
    "augmentation",
)
COLORS = ("#276fbf", "#d1495b", "#198754", "#7b4ab5", "#ef8354")


def read_json(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        return json.load(handle)


def write_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    temporary.replace(path)


def sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def manifest_fingerprint(manifest: dict) -> str:
    payload = {"records": manifest["records"], "splits": manifest["splits"]}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def memberships(design: dict) -> dict[str, list[dict]]:
    result: dict[str, list[dict]] = defaultdict(list)
    for suite, rows in design["suites"].items():
        for row in rows:
            result[row["job"]].append({"suite": suite, **row})
    return dict(result)


def archive_source(*, root: Path, recorded: str) -> Path:
    path = Path(recorded)
    return path if path.is_absolute() else root / "source" / path


def git_commit(*, result: dict, root: Path) -> str:
    provenance = result.get("provenance") or result["source"]
    recorded = provenance["environment"].get("git_commit")
    if recorded is not None:
        return str(recorded)
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=root / "source",
        text=True,
    ).strip()


def collect(*, root: Path) -> tuple[list[dict], list[dict], list[dict], dict]:
    plan_path, design_path, state_path = (
        root / "plan.json",
        root / "design.json",
        root / "state.json",
    )
    plan, design, state = map(read_json, (plan_path, design_path, state_path))
    manifest_path = Path(plan["manifest"])
    manifest = read_json(manifest_path)
    plan_jobs = {job["identifier"]: job for job in plan["jobs"]}
    if set(plan_jobs) != set(state["jobs"]):
        raise ValueError("Plan and state job identifiers differ")
    if state["status"] != "completed" or len(plan_jobs) != 193:
        raise ValueError("Archive is not the completed 193-job snapshot")
    fingerprint = manifest_fingerprint(manifest)
    if manifest["fingerprint"] != fingerprint:
        raise ValueError("Manifest fingerprint mismatch")
    split_counts = {name: len(rows) for name, rows in manifest["splits"].items()}
    if split_counts != {"train": 18462, "val": 2308, "test": 2308}:
        raise ValueError(f"Unexpected manifest split counts: {split_counts}")
    if {name: len(rows) for name, rows in design["suites"].items()} != SUITE_COUNTS:
        raise ValueError("Design suite counts differ")
    design_rows = memberships(design)
    train, banks, audits = [], [], []

    for identifier in sorted(plan_jobs):
        job, job_state = plan_jobs[identifier], state["jobs"][identifier]
        result_path = Path(
            job_state.get("result_path") or Path(job["output_dir"]) / "result.json"
        )
        if job_state["status"] != "completed" or not result_path.is_file():
            raise ValueError(f"Completed result is missing: {identifier}")
        result = read_json(result_path)
        code_hashes = (
            result["provenance"]["source_sha256"]
            if job["kind"] == "train"
            else result["source"]["code_sha256"]
        )
        hashes_ok = bool(code_hashes)
        for recorded, expected in code_hashes.items():
            path = archive_source(root=root, recorded=recorded)
            hashes_ok &= path.is_file() and sha256(path) == expected
        commit = git_commit(result=result, root=root)
        if result.get("config") != read_json(Path(job["config_path"])):
            raise ValueError(f"Result configuration differs from plan: {identifier}")
        config_present, provenance_present = (
            bool(result.get("config")),
            bool(result.get("provenance") or result.get("source")),
        )
        audit = {
            "identifier": identifier,
            "kind": job["kind"],
            "status": job_state["status"],
            "result_path": str(result_path),
            "result_sha256": sha256(result_path),
            "design_suites": [row["suite"] for row in design_rows.get(identifier, [])],
            "config_present": config_present,
            "provenance_present": provenance_present,
            "source_file_count": len(code_hashes),
            "source_hashes_verified": hashes_ok,
            "git_commit": commit,
            "git_commit_origin": "recorded"
            if job["kind"] == "train"
            else "archived_source_checkout",
            "passed": config_present
            and provenance_present
            and hashes_ok
            and bool(commit),
        }
        if job["kind"] == "bank":
            if result.get("complete") is not True:
                raise ValueError(f"Feature bank is incomplete: {identifier}")
            banks.append({
                "identifier": identifier,
                "config": result["config"],
                "source": result["source"],
                "bank_sha256": result["bank_sha256"],
                "created_at": result.get("created_at"),
                "audit": audit,
            })
        else:
            if result.get("status") != "complete":
                raise ValueError(f"Training result is incomplete: {identifier}")
            for endpoint in ("best_test", "final_test"):
                if result[endpoint]["count"] != 2308:
                    raise ValueError(f"{identifier} {endpoint} does not have n=2308")
            if set(result["best_test"]) != set(result["final_test"]):
                raise ValueError(f"{identifier} test endpoint fields differ")
            if not {field for field, _ in POINT_METRICS}.issubset(result["best_test"]):
                raise ValueError(f"{identifier} lacks a required point metric")
            if result["provenance"]["manifest_fingerprint"] != fingerprint:
                raise ValueError(f"{identifier} uses another manifest")
            rows = design_rows.get(identifier, [])
            suites = {row["suite"] for row in rows}
            if not suites:
                raise ValueError(f"Training result is not in the design: {identifier}")
            config = result["config"]
            source_row = next((row for row in rows if "backbone_source" in row), {})
            if "frozen_heads" in suites:
                backbone = {
                    "imagenet": "ImageNet frozen",
                    "cat_mse": "SL frozen",
                    "ce": "CE frozen",
                }[source_row["backbone_source"]]
            elif config["initialization"] == "scratch":
                backbone = "Scratch"
            elif config["initialization"] == "checkpoint":
                backbone = (
                    "SL@25 frozen" if config["backbone_frozen"] else "SL@25 trainable"
                )
            else:
                backbone = (
                    "ImageNet frozen"
                    if config["backbone_frozen"]
                    else "ImageNet end-to-end"
                )
            suite = next(name for name in SUITE_PRIORITY if name in suites)
            loss = config["run"]["loss"]["method"]
            train.append({
                "identifier": identifier,
                "design_suites": [row["suite"] for row in rows],
                "design_aliases": rows,
                "suite": suite,
                "pipeline": config["pipeline"],
                "backbone": backbone,
                "backbone_source": source_row.get("backbone_source"),
                "loss": loss,
                "loss_label": LOSS_LABELS.get(loss, loss),
                "noise_mode": config["run"]["noise"]["mode"],
                "noise_sd": float(config["run"]["noise"]["std_years"]),
                "schedule": config["schedule_kind"],
                "best_epoch": result["best_epoch"],
                "final_global_epoch": result["final_global_epoch"],
                "best_test": result["best_test"],
                "final_test": result["final_test"],
                "initialization": {
                    key: value
                    for key, value in result["initialization"].items()
                    if key != "rows"
                },
                "initialization_source": result["initialization_source"],
                "config": config,
                "provenance": result["provenance"],
            })
        audits.append(audit)

    if len(train) != 178 or len(banks) != 15:
        raise ValueError(
            f"Unexpected inventory: {len(train)} train, {len(banks)} banks"
        )
    if set(design_rows) != {record["identifier"] for record in train}:
        raise ValueError("Design jobs and training results differ")
    failed = [audit["identifier"] for audit in audits if not audit["passed"]]
    if failed:
        raise ValueError(f"Archive audit failed for {failed}")
    raw = {
        "plan": plan,
        "plan_path": plan_path,
        "design": design,
        "design_path": design_path,
        "state": state,
        "state_path": state_path,
        "manifest": manifest,
        "manifest_path": manifest_path,
        "manifest_fingerprint": fingerprint,
    }
    return train, banks, audits, raw


def unique_record(
    *, records: list[dict], membership: str = "", **query: object
) -> dict:
    matches = [
        record
        for record in records
        if (not membership or membership in record["design_suites"])
        and all(record.get(key) == value for key, value in query.items())
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected one run for {membership} {query}; found {len(matches)}"
        )
    return matches[0]


def run_metadata(record: dict) -> dict:
    config = record["config"]
    return {
        "pipeline": record["pipeline"],
        "initialization": config["initialization"],
        "backbone_frozen": config["backbone_frozen"],
        "schedule": record["schedule"],
        "schedule_detail": config["run"]["schedule"],
        "learning_rate": config["run"]["learning_rate"],
        "backbone_learning_rate": config["backbone_learning_rate"],
        "global_start_epoch": config["global_start_epoch"],
        "local_epochs": config["run"]["epochs"],
        "best_epoch": record["best_epoch"],
        "final_global_epoch": record["final_global_epoch"],
    }


def point(*, records: list[dict], metric: str, **query: object) -> dict:
    membership = query.pop("membership", "")
    record = unique_record(records=records, membership=membership, **query)
    value = record["best_test"][METRIC_FIELDS[metric]]
    if value is None:
        raise ValueError(f"Metric is undefined for {record['identifier']}")
    return {
        "job_id": record["identifier"],
        "value": float(value),
        **run_metadata(record),
    }


def series(label: str, points: list[dict]) -> dict:
    return {
        "label": label,
        "job_ids": [point["job_id"] for point in points],
        "values": [point["value"] for point in points],
        "points": points,
    }


def loss_series(
    *, records: list[dict], metric: str, label: str, **query: object
) -> dict:
    return series(
        label,
        [point(records=records, metric=metric, loss=loss, **query) for loss in LOSSES],
    )


def line_series(
    *,
    records: list[dict],
    metric: str,
    label: str,
    xs: tuple[float, ...],
    **query: object,
) -> dict:
    points = [point(records=records, metric=metric, noise_sd=x, **query) for x in xs]
    return {
        "label": label,
        "x": list(xs),
        "job_ids": [point["job_id"] for point in points],
        "values": [point["value"] for point in points],
        "points": points,
    }


def augmentation_data(records: list[dict], metric: str) -> dict:
    facets = []
    xs = NOISES
    for backbone in ("ImageNet end-to-end", "ImageNet frozen"):
        for loss in LOSSES:
            facets.append({
                "title": f"{LOSS_LABELS[loss]} · {backbone}",
                "series": [
                    line_series(
                        records=records,
                        metric=metric,
                        label=pipeline.title(),
                        xs=xs,
                        membership="augmentation",
                        pipeline=pipeline,
                        backbone=backbone,
                        loss=loss,
                    )
                    for pipeline in ("original", "fixed")
                ],
            })
    return {
        "x": list(xs),
        "facets": facets,
    }


def scratch_data(records: list[dict], metric: str) -> dict:
    facets = []
    for noise in NOISES:
        imagenet = loss_series(
            records=records,
            metric=metric,
            label="ImageNet · 1e-3 cosine",
            membership="augmentation",
            pipeline="original",
            backbone="ImageNet end-to-end",
            noise_sd=noise,
        )
        scratch = [
            loss_series(
                records=records, metric=metric, label=label, suite=suite, noise_sd=noise
            )
            for label, suite in (
                ("Scratch · 1e-3 cosine", "scratch_cosine"),
                ("Scratch · 3e-4 warm+constant", "scratch_warmup_constant"),
            )
        ]
        facets.append({
            "title": f"Label-noise SD {noise:g}",
            "series": [imagenet, *scratch],
        })
    return {
        "x_labels": [LOSS_LABELS[loss] for loss in LOSSES],
        "facets": facets,
    }


def frozen_data(records: list[dict], metric: str) -> dict:
    xs = EXTENDED_NOISES
    common = [
        line_series(
            records=records,
            metric=metric,
            label=backbone,
            xs=xs,
            membership="frozen_heads",
            backbone=backbone,
            loss="mse",
        )
        for backbone in ("ImageNet frozen", "SL frozen", "CE frozen")
    ]
    heads = [
        line_series(
            records=records,
            metric=metric,
            label=LOSS_LABELS[loss],
            xs=xs,
            membership="frozen_heads",
            backbone="SL frozen",
            loss=loss,
        )
        for loss in LOSSES
    ]
    if len(common) != 3 or len(heads) != 5:
        raise ValueError("Frozen figure must contain three sources and five SL heads")
    return {
        "panels": [
            {
                "title": "Common Cat. MSE readout · three feature sources",
                "series": common,
            },
            {"title": "SL features · five head losses", "series": heads},
        ]
    }


def frozen_table(records: list[dict]) -> list[dict]:
    source_order = {
        name: number
        for number, name in enumerate(("ImageNet frozen", "SL frozen", "CE frozen"))
    }
    loss_order = {name: number for number, name in enumerate(LOSSES)}
    rows = []
    for record in records:
        if "frozen_heads" not in record["design_suites"]:
            continue
        rows.append({
            "job": record["identifier"],
            "feature_source": record["backbone"],
            "head_loss": record["loss"],
            "head_loss_label": record["loss_label"],
            "noise_sd": record["noise_sd"],
            "best_epoch": record["best_epoch"],
            "best_test": {
                field: record["best_test"][field] for field, _ in POINT_METRICS
            },
        })
    rows.sort(
        key=lambda row: (
            source_order[row["feature_source"]],
            loss_order[row["head_loss"]],
            row["noise_sd"],
        )
    )
    if len(rows) != 75:
        raise ValueError(f"Expected 75 frozen-head rows, found {len(rows)}")
    return rows


def post_training_data(records: list[dict], metric: str) -> dict:
    facets = []
    parent_metadata = post_training_metadata(records)["parents_by_noise_sd"]
    for noise in NOISES:
        reference = point(
            records=records,
            metric=metric,
            membership="augmentation",
            pipeline="original",
            backbone="ImageNet end-to-end",
            loss="mse",
            noise_sd=noise,
        )
        for loss in LOSSES[:4]:
            bars = [
                {
                    "label": label,
                    **point(
                        records=records,
                        metric=metric,
                        suite="post_training",
                        backbone=backbone,
                        loss=loss,
                        noise_sd=noise,
                    ),
                }
                for label, backbone in (
                    ("Trainable", "SL@25 trainable"),
                    ("Frozen", "SL@25 frozen"),
                )
            ]
            facets.append({
                "title": f"{LOSS_LABELS[loss]} · SD {noise:g}",
                "bars": bars,
                "parent_best_epoch": parent_metadata[f"{noise:g}"]["parent_best_epoch"],
                "post_start_epoch": parent_metadata[f"{noise:g}"]["post_start_epoch"],
                "reference": {
                    "label": "Original SL50",
                    **reference,
                },
            })
    return {"facets": facets}


def overview_data(records: list[dict], metric: str) -> dict:
    short = NOISES
    e2e = [
        line_series(
            records=records,
            metric=metric,
            label=f"{pipeline.title()} · {LOSS_LABELS[loss]}",
            xs=short,
            membership="augmentation",
            pipeline=pipeline,
            backbone="ImageNet end-to-end",
            loss=loss,
        )
        for pipeline in ("original", "fixed")
        for loss in ("mse", "tailrl_population")
    ]
    scratch = [
        line_series(
            records=records,
            metric=metric,
            label=f"{label} · {LOSS_LABELS[loss]}",
            xs=short,
            suite=suite,
            loss=loss,
        )
        for suite, label in (
            ("scratch_cosine", "Scratch cosine"),
            ("scratch_warmup_constant", "Scratch warm+constant"),
        )
        for loss in ("mse", "tailrl_population")
    ]
    frozen = [
        line_series(
            records=records,
            metric=metric,
            label=f"{backbone} · Cat. MSE",
            xs=EXTENDED_NOISES,
            membership="frozen_heads",
            backbone=backbone,
            loss="mse",
        )
        for backbone in ("ImageNet frozen", "SL frozen", "CE frozen")
    ]
    post_pairing = post_training_metadata(records)["parents_by_noise_sd"]
    post = []
    for backbone in ("SL@25 trainable", "SL@25 frozen"):
        points = []
        for noise in short:
            candidate = point(
                records=records,
                metric=metric,
                suite="post_training",
                backbone=backbone,
                loss="tailrl_population",
                noise_sd=noise,
            )
            baseline = point(
                records=records,
                metric=metric,
                suite="post_training",
                backbone=backbone,
                loss="mse",
                noise_sd=noise,
            )
            points.append({
                "job_id": candidate["job_id"],
                "value": candidate["value"] - baseline["value"],
                "pipeline": candidate["pipeline"],
                "initialization": candidate["initialization"],
                "backbone_frozen": candidate["backbone_frozen"],
                "schedule": candidate["schedule"],
                "schedule_detail": candidate["schedule_detail"],
                "global_start_epoch": candidate["global_start_epoch"],
                "local_epochs": candidate["local_epochs"],
                "best_epoch": candidate["best_epoch"],
                "final_global_epoch": candidate["final_global_epoch"],
                "candidate": candidate,
                "reset_mse_baseline": baseline,
            })
        post.append({
            "label": backbone,
            "x": list(short),
            "values": [item["value"] for item in points],
            "points": points,
        })
    return {
        "panels": [
            {
                "title": "ImageNet end-to-end · Cat. MSE versus TailRL",
                "series": e2e,
            },
            {
                "title": "Scratch recipes · Cat. MSE versus TailRL",
                "series": scratch,
            },
            {
                "title": "Frozen heads · common Cat. MSE readout",
                "series": frozen,
            },
            {
                "title": "Post-TailRL minus reset Cat. MSE · zero reference",
                "series": post,
            },
        ],
        "post_parent_best_epochs": post_pairing,
    }


def axis_label(metric: str, *, delta: bool = False) -> str:
    labels = {
        "mae": "Best-test mean-point MAE (years; lower is better)",
        "rmse": "Best-test mean-point RMSE (years; lower is better)",
        "spearman": "Best-test mean-point ρ (higher is better)",
    }
    return ("Δ " if delta else "") + labels[metric]


def save(fig: plt.Figure, output_dir: Path, name: str) -> str:
    path, temporary = output_dir / name, (output_dir / name).with_suffix(".tmp.png")
    fig.savefig(temporary, dpi=220, bbox_inches="tight", facecolor="white")
    temporary.replace(path)
    plt.close(fig)
    return str(path)


def plot_augmentation(*, data: dict, metric: str, output_dir: Path, name: str) -> str:
    fig, axes = plt.subplots(
        2,
        5,
        figsize=(18, 7.5),
        squeeze=False,
        sharex=True,
        sharey="row",
        constrained_layout=True,
    )
    for axis, facet in zip(axes.flat, data["facets"], strict=True):
        for index, item in enumerate(facet["series"]):
            axis.plot(
                item["x"],
                item["values"],
                color=COLORS[index],
                marker="o" if index == 0 else "s",
                linewidth=1.7,
                label=item["label"],
            )
        axis.set_xticks(data["x"], [f"SD {x:g}" for x in data["x"]])
        axis.set_title(facet["title"], fontsize=9, fontweight="bold")
        axis.grid(axis="y", color="#d8dee9", linewidth=0.7)
        axis.set_axisbelow(True)
        axis.set_ylabel(axis_label(metric), fontsize=8)
    axes.flat[0].legend(frameon=False, fontsize=8)
    return save(fig, output_dir, name)


def plot_frozen(*, data: dict, metric: str, output_dir: Path, name: str) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.3), constrained_layout=True)
    for axis, panel in zip(axes, data["panels"], strict=True):
        for index, item in enumerate(panel["series"]):
            axis.plot(
                item["x"],
                item["values"],
                color=COLORS[index],
                marker=("o", "s", "^", "D", "v")[index],
                linewidth=1.8,
                label=item["label"],
            )
        axis.set_xticks(EXTENDED_NOISES, [f"SD {noise:g}" for noise in EXTENDED_NOISES])
        axis.set_title(panel["title"], fontsize=10, fontweight="bold")
        axis.set_ylabel(axis_label(metric), fontsize=8)
        axis.grid(axis="y", color="#d8dee9", linewidth=0.7)
        axis.set_axisbelow(True)
        axis.legend(frameon=False, fontsize=8)
    if metric == "mae":
        axes[0].set_ylim(5.0, 9.0)
        axes[1].set_ylim(5.0, 7.0)
    return save(fig, output_dir, name)


def plot_bars(
    *,
    data: dict,
    metric: str,
    output_dir: Path,
    name: str,
    shape: tuple[int, int],
    figsize: tuple[float, float],
    rotate: int,
) -> str:
    fig, axes = plt.subplots(
        *shape, figsize=figsize, squeeze=False, sharey=True, constrained_layout=True
    )
    for axis, facet in zip(axes.flat, data["facets"], strict=True):
        positions = list(range(len(data["x_labels"])))
        width = 0.8 / len(facet["series"])
        for index, item in enumerate(facet["series"]):
            offset = (index - (len(facet["series"]) - 1) / 2) * width
            axis.bar(
                [x + offset for x in positions],
                item["values"],
                width,
                color=COLORS[index],
                label=item["label"],
            )
        axis.set_xticks(
            positions,
            data["x_labels"],
            rotation=rotate,
            ha="right" if rotate else "center",
        )
        axis.set_title(facet["title"], fontsize=9, fontweight="bold")
        axis.grid(axis="y", color="#d8dee9", linewidth=0.7)
        axis.set_axisbelow(True)
        axis.set_ylabel(axis_label(metric), fontsize=8)
    if data.get("legend", True):
        axes.flat[0].legend(frameon=False, fontsize=8)
    return save(fig, output_dir, name)


def plot_post(*, data: dict, metric: str, output_dir: Path, name: str) -> str:
    fig, axes = plt.subplots(
        3, 4, figsize=(14, 8), squeeze=False, sharey=True, constrained_layout=True
    )
    for axis, facet in zip(axes.flat, data["facets"], strict=True):
        axis.bar(
            (0, 1),
            [item["value"] for item in facet["bars"]],
            color=COLORS[:2],
            width=0.65,
        )
        axis.axhline(
            facet["reference"]["value"], color="#222", linestyle="--", linewidth=1.2
        )
        axis.set_xticks((0, 1), [item["label"] for item in facet["bars"]])
        axis.set_title(
            f"{facet['title']}\nparent best e{facet['parent_best_epoch']} · "
            f"post starts e{facet['post_start_epoch']}",
            fontsize=8,
            fontweight="bold",
        )
        axis.grid(axis="y", color="#d8dee9", linewidth=0.7)
        axis.set_axisbelow(True)
        axis.set_ylabel(axis_label(metric), fontsize=8)
    axes.flat[0].legend(
        handles=[
            plt.Rectangle((0, 0), 1, 1, color=COLORS[0]),
            plt.Rectangle((0, 0), 1, 1, color=COLORS[1]),
            plt.Line2D((0, 1), (0, 0), color="#222", linestyle="--"),
        ],
        labels=["Trainable", "Frozen", "Original SL50"],
        frameon=False,
        fontsize=8,
    )
    return save(fig, output_dir, name)


def plot_overview(*, data: dict, metric: str, output_dir: Path, name: str) -> str:
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    for index, (axis, panel) in enumerate(zip(axes.flat, data["panels"], strict=True)):
        for item in panel["series"]:
            axis.plot(item["x"], item["values"], marker="o", label=item["label"])
        axis.set_title(panel["title"], fontsize=10, fontweight="bold")
        axis.set_ylabel(axis_label(metric, delta=index == 3), fontsize=8)
        axis.grid(axis="y", color="#d8dee9", linewidth=0.7)
        axis.legend(frameon=False, fontsize=7)
    axes.flat[3].axhline(0, color="#555", linewidth=0.8)
    for noise, item in data["post_parent_best_epochs"].items():
        axes.flat[3].annotate(
            f"parent e{item['parent_best_epoch']}",
            (float(noise), 0),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            fontsize=7,
        )
    return save(fig, output_dir, name)


def point_matrix(records: list[dict]) -> dict:
    index = {}
    for record in records:
        key = (
            record["suite"],
            record["pipeline"],
            record["backbone"],
            record["noise_sd"],
            record["loss"],
        )
        if key in index:
            raise ValueError(f"Duplicate logical run key: {key}")
        index[key] = record
    rows = []
    for key, categorical in index.items():
        tail = index.get((*key[:-1], "tailrl_population"))
        if key[-1] != "mse" or tail is None:
            continue
        values = {}
        for field, _ in POINT_METRICS:
            left, right = categorical["best_test"][field], tail["best_test"][field]
            values[field] = (
                float(right) - float(left)
                if left is not None and right is not None
                else None
            )
        rows.append({
            "suite": key[0],
            "pipeline": key[1],
            "backbone": key[2],
            "noise_sd": key[3],
            "label": f"{key[0]} · {key[2]} · SD {key[3]:g}",
            "categorical_job": categorical["identifier"],
            "tail_job": tail["identifier"],
            "categorical_values": {
                field: categorical["best_test"][field] for field, _ in POINT_METRICS
            },
            "tail_values": {
                field: tail["best_test"][field] for field, _ in POINT_METRICS
            },
            "delta": values,
        })
    suite_order = {name: number for number, name in enumerate(SUITE_PRIORITY)}
    rows.sort(
        key=lambda row: (
            suite_order[row["suite"]],
            row["pipeline"],
            row["backbone"],
            row["noise_sd"],
        )
    )
    if not rows:
        raise ValueError("No matched TailRL/categorical-MSE point-metric pairs")
    scales = {
        field: max(
            (
                abs(row["delta"][field])
                for row in rows
                if row["delta"][field] is not None
            ),
            default=1.0,
        )
        or 1.0
        for field, _ in POINT_METRICS
    }
    return {
        "metric_fields": [field for field, _ in POINT_METRICS],
        "metrics": {field: label for field, label in POINT_METRICS},
        "direction": {
            "mae": "lower is better",
            "rmse": "lower is better",
            "spearman": "higher is better",
        },
        "rows": rows,
        "scales": scales,
        "normalized": [
            [
                row["delta"][field] / scales[field]
                if row["delta"][field] is not None
                else None
                for field, _ in POINT_METRICS
            ]
            for row in rows
        ],
    }


def plot_point_matrix(*, data: dict, output_dir: Path, name: str) -> str:
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(16, max(7, len(data["rows"]) * 0.28)),
        sharey=True,
        constrained_layout=True,
    )
    image = None
    for panel, (start, stop) in enumerate(((0, 3), (3, 6), (6, 9))):
        image = axes[panel].imshow(
            np.asarray([row[start:stop] for row in data["normalized"]], dtype=float),
            aspect="auto",
            cmap="coolwarm",
            vmin=-1,
            vmax=1,
            interpolation="nearest",
        )
        axes[panel].set_xticks(
            range(stop - start),
            [POINT_METRICS[number][1] for number in range(start, stop)],
            rotation=35,
            ha="right",
        )
        axes[panel].set_title(("MAE", "RMSE", "Spearman")[panel], fontweight="bold")
        for row_number, row in enumerate(data["rows"]):
            for column, (field, _) in enumerate(POINT_METRICS[start:stop]):
                axes[panel].text(
                    column,
                    row_number,
                    f"{row['delta'][field]:.2f}"
                    if row["delta"][field] is not None
                    else "—",
                    ha="center",
                    va="center",
                    fontsize=5.5,
                )
    axes[0].set_yticks(
        range(len(data["rows"])), [row["label"] for row in data["rows"]], fontsize=6
    )
    fig.colorbar(
        image,
        ax=axes.tolist(),
        shrink=0.55,
        label="Signed value normalized within each metric",
    )
    fig.suptitle(
        "TailRL − categorical MSE · nine best-test point metrics\n"
        "Negative MAE/RMSE favors TailRL; positive Spearman favors TailRL",
        fontweight="bold",
    )
    return save(fig, output_dir, name)


def metric_figures(
    *, records: list[dict], metric: str, output_dir: Path
) -> tuple[list[dict], dict]:
    figures, values = [], {}
    builders = (
        (
            "augmentation",
            augmentation_data(records, metric),
            plot_augmentation,
            "Original versus fixed augmentation, with numeric noise ordering",
        ),
        (
            "scratch_initialization",
            scratch_data(records, metric),
            None,
            "Initialization and schedule recipes",
        ),
        (
            "frozen_heads",
            frozen_data(records, metric),
            plot_frozen,
            "Three-source common readout and five SL-head readout curves",
        ),
    )
    for key, data, plotter, caption in builders:
        name = f"{key}_{metric}.png"
        if plotter is not None:
            path = plotter(data=data, metric=metric, output_dir=output_dir, name=name)
        else:
            path = plot_bars(
                data=data,
                metric=metric,
                output_dir=output_dir,
                name=name,
                shape=(1, 3),
                figsize=(15, 4.8),
                rotate=25,
            )
        figures.append({
            "name": key,
            "metric": metric,
            "path": path,
            "caption": caption,
        })
        values[name] = {"metric": metric, "axis_label": axis_label(metric), **data}
    post = post_training_data(records, metric)
    name = f"post_training_{metric}.png"
    path = plot_post(data=post, metric=metric, output_dir=output_dir, name=name)
    figures.append({
        "name": "post_training",
        "metric": metric,
        "path": path,
        "caption": "Epoch-25 post-training vs original epoch-50 SL",
    })
    values[name] = {
        "metric": metric,
        "axis_label": axis_label(metric),
        **post,
    }
    overview = overview_data(records, metric)
    name = f"overview_{metric}.png"
    path = plot_overview(data=overview, metric=metric, output_dir=output_dir, name=name)
    figures.append({
        "name": "overview",
        "metric": metric,
        "path": path,
        "caption": "Condensed source values; exact components are in report-data.json",
    })
    values[name] = {
        "metric": metric,
        "axis_label": axis_label(metric),
        **overview,
    }
    return figures, values


def descriptive_summary(records: list[dict], metric: str) -> list[dict]:
    rows = []
    for suite in SUITE_COUNTS:
        members = [record for record in records if suite in record["design_suites"]]
        values = [
            float(record["best_test"][METRIC_FIELDS[metric]]) for record in members
        ]
        choose = max if metric == "spearman" else min
        best = choose(
            members, key=lambda record: record["best_test"][METRIC_FIELDS[metric]]
        )
        rows.append({
            "suite": suite,
            "n": len(values),
            "min": min(values),
            "median": statistics.median(values),
            "max": max(values),
            "selected_endpoint": metric,
            "extreme_job": best["identifier"],
            "extreme_value": float(best["best_test"][METRIC_FIELDS[metric]]),
            "direction": "higher" if metric == "spearman" else "lower",
        })
    return rows


def audit_summary(banks: list[dict], audits: list[dict], design: dict) -> dict:
    kinds = Counter(audit["kind"] for audit in audits)
    membership_count = sum(len(rows) for rows in design["suites"].values())
    unique_train = len({
        row["job"] for rows in design["suites"].values() for row in rows
    })
    return {
        "state_status": "completed",
        "completed_jobs": len(audits),
        "training_results": kinds["train"],
        "completed_banks": kinds["bank"],
        "design_memberships": membership_count,
        "unique_training_jobs": unique_train,
        "reused_training_memberships": membership_count - unique_train,
        "all_193_results_present": len(audits) == 193
        and kinds == {"bank": 15, "train": 178},
        "all_configs_present": all(audit["config_present"] for audit in audits),
        "all_provenance_present": all(audit["provenance_present"] for audit in audits),
        "all_source_hashes_verified": all(
            audit["source_hashes_verified"] for audit in audits
        ),
        "unique_git_commits": sorted({audit["git_commit"] for audit in audits}),
        "bank_sha256_unique": len({bank["bank_sha256"] for bank in banks}),
    }


def post_training_metadata(records: list[dict]) -> dict:
    by_id = {record["identifier"]: record for record in records}
    parents = {}
    for noise in NOISES:
        children = [
            record
            for record in records
            if record["suite"] == "post_training" and record["noise_sd"] == noise
        ]
        parent_ids = {
            alias["parent"]
            for record in children
            for alias in record["design_aliases"]
            if alias["suite"] == "post_training"
        }
        if len(parent_ids) != 1:
            raise ValueError(
                f"Expected one SL parent at noise {noise}, found {parent_ids}"
            )
        parent_id = parent_ids.pop()
        parent = by_id[parent_id]
        parents[f"{noise:g}"] = {
            "parent": parent_id,
            "parent_best_epoch": parent["best_epoch"],
            "post_start_epoch": 25,
            "post_starts_after_parent_best": 25 > parent["best_epoch"],
        }
    observed = {
        float(key): value["parent_best_epoch"] for key, value in parents.items()
    }
    if observed != EXPECTED_PARENT_BEST_EPOCHS:
        raise ValueError(
            f"Unexpected SL parent best epochs: {observed}, "
            f"expected {EXPECTED_PARENT_BEST_EPOCHS}"
        )
    return {
        "selection": "Clean validation mean MAE, including epoch zero",
        "source_checkpoint": "Original-augmentation categorical MSE, fixed epoch 25",
        "post_global_epoch_interval": [25, 50],
        "local_epochs": 25,
        "parents_by_noise_sd": parents,
        "warning": (
            "Post-training starts at epoch 25. The original SL run selects epoch 42 "
            "on clean data, epoch 21 at SD 5, and epoch 12 at SD 10."
        ),
    }


def recipe_metadata(records: list[dict]) -> dict:
    cosine = unique_record(
        records=records, suite="scratch_cosine", loss="mse", noise_sd=0.0
    )
    constant = unique_record(
        records=records,
        suite="scratch_warmup_constant",
        loss="mse",
        noise_sd=0.0,
    )
    post = unique_record(
        records=records,
        suite="post_training",
        backbone="SL@25 trainable",
        loss="mse",
        noise_sd=0.0,
    )
    return {
        "augmentation": {
            "original": (
                "RandomResizedCrop(224, scale=(0.8, 1.0)); RandomHorizontalFlip; "
                "ColorJitter(0.2, 0.2, 0.2); normalize"
            ),
            "fixed": "Resize(256); CenterCrop(224); normalize",
        },
        "scratch_cosine": {
            "job": cosine["identifier"],
            "initialization": "scratch",
            "peak_learning_rate": cosine["config"]["run"]["learning_rate"],
            "schedule": cosine["schedule"],
            "schedule_detail": cosine["config"]["run"]["schedule"],
            "epochs": cosine["config"]["run"]["epochs"],
            "best_epoch": cosine["best_epoch"],
        },
        "scratch_warmup_constant": {
            "job": constant["identifier"],
            "initialization": "scratch",
            "peak_learning_rate": constant["config"]["run"]["learning_rate"],
            "schedule": constant["schedule"],
            "schedule_detail": constant["config"]["run"]["schedule"],
            "epochs": constant["config"]["run"]["epochs"],
            "best_epoch": constant["best_epoch"],
        },
        "post_training": {
            "example_job": post["identifier"],
            **run_metadata(post),
            **post_training_metadata(records),
        },
    }


def snapshot_checks(records: list[dict]) -> dict:
    checks = []
    for (loss, noise), expected in EXPECTED_AUGMENTATION_MAE.items():
        observed = point(
            records=records,
            metric="mae",
            membership="augmentation",
            pipeline="original",
            backbone="ImageNet end-to-end",
            loss=loss,
            noise_sd=noise,
        )
        checks.append({
            "name": f"original augmentation {loss} SD {noise:g}",
            "job_id": observed["job_id"],
            "best_epoch": observed["best_epoch"],
            "expected_mae_mean": expected,
            "observed_mae_mean": observed["value"],
            "passed": abs(observed["value"] - expected) <= 1e-6,
        })
    for (suite, loss, noise), expected in EXPECTED_SCRATCH_MAE.items():
        observed = point(
            records=records,
            metric="mae",
            suite=suite,
            loss=loss,
            noise_sd=noise,
        )
        checks.append({
            "name": f"{suite} {loss} SD {noise:g}",
            "job_id": observed["job_id"],
            "best_epoch": observed["best_epoch"],
            "expected_mae_mean": expected,
            "observed_mae_mean": observed["value"],
            "passed": abs(observed["value"] - expected) <= 1e-6,
        })
    frozen_deltas = []
    for loss in LOSSES:
        for noise in EXTENDED_NOISES:
            sl = point(
                records=records,
                metric="mae",
                membership="frozen_heads",
                backbone="SL frozen",
                loss=loss,
                noise_sd=noise,
            )
            ce = point(
                records=records,
                metric="mae",
                membership="frozen_heads",
                backbone="CE frozen",
                loss=loss,
                noise_sd=noise,
            )
            frozen_deltas.append({
                "loss": loss,
                "noise_sd": noise,
                "sl_job": sl["job_id"],
                "ce_job": ce["job_id"],
                "sl_minus_ce_mae_mean": sl["value"] - ce["value"],
            })
    mean_delta = statistics.fmean(
        item["sl_minus_ce_mae_mean"] for item in frozen_deltas
    )
    checks.append({
        "name": "SL frozen minus CE frozen mean MAE across 25 matched cases",
        "expected_mean_delta": -0.718,
        "observed_mean_delta": mean_delta,
        "cases": frozen_deltas,
        "passed": abs(mean_delta + 0.718) <= 5e-4
        and all(item["sl_minus_ce_mae_mean"] < 0 for item in frozen_deltas),
    })
    result = {
        "all_passed": all(check["passed"] for check in checks),
        "checks": checks,
    }
    if not result["all_passed"]:
        raise ValueError("Root-supplied snapshot value checks failed")
    return result


def protected_file_state(output_dir: Path) -> dict:
    path = output_dir / "contrasts.json"
    if not path.is_file():
        return {"path": str(path), "present": False}
    return {
        "path": str(path),
        "present": True,
        "sha256": sha256(path),
        "size_bytes": path.stat().st_size,
    }


def render_html(*, report: dict, figures: list[dict], summary: list[dict]) -> str:
    audit = report["audit"]
    cards = "".join(
        f"<div class=card><span>{html.escape(label)}</span><strong>{value}</strong></div>"
        for label, value in (
            ("Completed jobs", audit["completed_jobs"]),
            ("Training runs", audit["training_results"]),
            ("Feature banks", audit["completed_banks"]),
            ("Design memberships", audit["design_memberships"]),
            ("Reused memberships", audit["reused_training_memberships"]),
            ("Distinct commits", len(audit["unique_git_commits"])),
        )
    )
    figure_html, current = [], ""
    for figure in figures:
        label = figure["metric"] or "Matched comparison"
        if label != current:
            figure_html.append(f"<h2>{html.escape(label.upper())} figures</h2>")
            current = label
        encoded = base64.b64encode(Path(figure["path"]).read_bytes()).decode()
        figure_html.append(
            f'<figure><img src="data:image/png;base64,{encoded}" alt="{html.escape(Path(figure["path"]).name)}">'
            f"<figcaption>{html.escape(figure['caption'])}</figcaption></figure>"
        )
    suite_rows = "".join(
        "<tr>"
        + "".join(
            f"<td>{html.escape(str(row[column]))}</td>"
            for column in (
                "suite",
                "n",
                "min",
                "median",
                "max",
                "extreme_job",
                "extreme_value",
                "direction",
            )
        )
        + "</tr>"
        for row in summary
    )
    audit_rows = "".join(
        "<tr>"
        + "".join(
            f"<td>{html.escape(str(row[column]))}</td>"
            for column in (
                "identifier",
                "kind",
                "status",
                "source_file_count",
                "source_hashes_verified",
                "config_present",
                "provenance_present",
                "git_commit",
                "result_sha256",
            )
        )
        + "</tr>"
        for row in report["job_audits"]
    )
    frozen_rows = "".join(
        "<tr>"
        + "".join(
            f"<td>{html.escape(str(value))}</td>"
            for value in (
                row["job"],
                row["feature_source"],
                row["head_loss_label"],
                row["noise_sd"],
                row["best_epoch"],
            )
        )
        + "".join(
            (
                f"<td>{row['best_test'][field]:.6f}</td>"
                if row["best_test"][field] is not None
                else "<td>—</td>"
            )
            for field, _ in POINT_METRICS
        )
        + "</tr>"
        for row in report["frozen_heads_table"]
    )
    recipes = report["recipe_metadata"]
    parent_text = "; ".join(
        f"SD {noise}: parent best epoch {item['parent_best_epoch']}, "
        f"post interval {item['post_start_epoch']}–50"
        for noise, item in recipes["post_training"]["parents_by_noise_sd"].items()
    )
    metadata_html = (
        f"<p><strong>Augmentation original:</strong> "
        f"{html.escape(recipes['augmentation']['original'])}</p>"
        f"<p><strong>Augmentation fixed:</strong> "
        f"{html.escape(recipes['augmentation']['fixed'])}</p>"
        f"<p><strong>Scratch cosine:</strong> peak LR "
        f"{recipes['scratch_cosine']['peak_learning_rate']:g}; schedule "
        f"{html.escape(recipes['scratch_cosine']['schedule'])}; "
        f"{html.escape(json.dumps(recipes['scratch_cosine']['schedule_detail'], sort_keys=True))}.</p>"
        f"<p><strong>Scratch warmup+constant:</strong> peak LR "
        f"{recipes['scratch_warmup_constant']['peak_learning_rate']:g}; schedule "
        f"{html.escape(recipes['scratch_warmup_constant']['schedule'])}; "
        f"{html.escape(json.dumps(recipes['scratch_warmup_constant']['schedule_detail'], sort_keys=True))}.</p>"
        f"<p><strong>Post-training:</strong> {html.escape(parent_text)}. "
        f"{html.escape(recipes['post_training']['warning'])} "
        "All post branches retain the matching SL epoch-25 model and use a fresh optimizer "
        "and 25-epoch schedule. The MSE branch is the matched restart control. "
        "This report does not bootstrap or load contrasts.json.</p>"
    )
    data_json = json.dumps(report, separators=(",", ":"), allow_nan=False).replace(
        "</", "<\\/"
    )
    document = r"""<!doctype html><html lang=en><head><meta charset=utf-8>
<meta name=viewport content="width=device-width"><title>UTKFace adaptation results · 2026-10-26</title>
<style>*{box-sizing:border-box}body{margin:0;background:#f3f6fa;color:#172033;font:15px/1.45 system-ui,sans-serif}
header{background:linear-gradient(135deg,#102a43,#276fbf);color:white;padding:34px max(24px,6vw)}header p{margin:0;color:#e1edf8}
h1{margin:0 0 8px;font-size:clamp(28px,4vw,48px)}main{max-width:1600px;margin:auto;padding:26px}h2{margin-top:34px}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin-top:-50px}
.card{background:white;padding:16px;box-shadow:0 4px 18px #102a4325}.card span{display:block;color:#5d687a;font-size:12px;text-transform:uppercase}.card strong{font-size:28px}
figure{margin:22px 0;background:white;padding:14px;border:1px solid #d9e0ea}figure img{width:100%;height:auto;display:block}figcaption,.muted{color:#5d687a}
.table-wrap{overflow:auto;max-height:75vh;background:white;border:1px solid #d9e0ea}table{border-collapse:collapse;width:100%;font-size:12px;white-space:nowrap}
th,td{padding:7px 9px;border-bottom:1px solid #d9e0ea;text-align:right}th{position:sticky;top:0;background:#eaf1f8}td:nth-child(-n+7),th:nth-child(-n+7){text-align:left}
.controls{display:flex;gap:10px;flex-wrap:wrap;margin:12px 0}select,input,button{padding:8px 10px;border:1px solid #aeb9c8;background:white}details,.metadata{background:white;border:1px solid #d9e0ea;padding:14px;margin:12px 0}.metadata p{margin:.35em 0}</style></head><body>
<header><h1>UTKFace adaptation results</h1><p>Immutable archive snapshot: 193 completed jobs, 178 distinct training results and 15 feature banks. This HTML reports source facts and exact values.</p></header>
<main><section class=cards>__CARDS__</section><h2>Archive integrity</h2>
<p>All 193 result files, configurations and recorded source hashes were verified. Training results record the source commit; feature-bank code is verified against the archived source checkout. The manifest has 18,462 train, 2,308 validation and 2,308 test rows; every training result reports n=2,308 for both test endpoints.</p>
<h2>Recipe and timing metadata</h2><div class=metadata>__METADATA__</div>
<h2>Suite inventory · __PRIMARY__</h2><div class=table-wrap><table><thead><tr><th>Suite</th><th>N memberships</th><th>Min</th><th>Median</th><th>Max</th><th>Extreme job</th><th>Value</th><th>Direction</th></tr></thead><tbody>__SUITES__</tbody></table></div>
<h2>Figures</h2>__FIGURES__<h2>Frozen-head source comparison · 75 runs</h2>
<p class=muted>All feature-source, head-loss and noise combinations are retained. Noise levels are ordered numerically as 0, 1, 3, 5 and 10.</p>
<div class=table-wrap><table><thead><tr><th>Job</th><th>Feature source</th><th>Head loss</th><th>Noise SD</th><th>Best epoch</th>__FROZEN_HEADERS__</tr></thead><tbody>__FROZEN_ROWS__</tbody></table></div>
<h2>All 178 training results</h2>
<div class=controls><input id=search placeholder="Search job, suite, backbone…" size=30><select id=suite><option value="">All suites</option></select><select id=pipeline><option value="">All pipelines</option></select><select id=backbone><option value="">All backbones</option></select><select id=loss><option value="">All losses</option></select><select id=noise><option value="">All noise SD</option></select><select id=endpoint><option value=best_test>Best test</option><option value=final_test>Final test</option></select><label><input id=allmetrics type=checkbox> Show all test metrics</label><span id=count></span></div>
<div class=table-wrap><table id=runs><thead></thead><tbody></tbody></table></div>
<details><summary>All 193 job and provenance audits</summary><div class=table-wrap><table><thead><tr><th>Job</th><th>Kind</th><th>Status</th><th>Source files</th><th>Hashes verified</th><th>Config</th><th>Provenance</th><th>Git commit</th><th>Result SHA-256</th></tr></thead><tbody>__AUDITS__</tbody></table></div></details>
<p class=muted>All metric fields, configs, initialization records, source hashes and exact plotted values are retained in report-data.json. No confidence intervals are shown because contrasts.json is not loaded by this report.</p></main>
<script>const DATA=__DATA__,POINTS=__POINTS__, $=id=>document.getElementById(id);
for(const [id,key] of [['suite','suite'],['pipeline','pipeline'],['backbone','backbone'],['loss','loss'],['noise','noise_sd']])[...new Set(DATA.records.map(r=>r[key]))].sort((a,b)=>typeof a==='number'?a-b:String(a).localeCompare(String(b))).forEach(v=>{const o=document.createElement('option');o.value=v;o.textContent=v;$(id).append(o)});
function fmt(v){return v===null||v===undefined?'—':Number(v).toFixed(3)}
function draw(){const endpoint=$('endpoint').value,q=$('search').value.toLowerCase(),all=$('allmetrics').checked;
const rows=DATA.records.filter(r=>(!$('suite').value||r.design_suites.includes($('suite').value))&&(!$('pipeline').value||r.pipeline===$('pipeline').value)&&(!$('backbone').value||r.backbone===$('backbone').value)&&(!$('loss').value||r.loss===$('loss').value)&&(!$('noise').value||String(r.noise_sd)===$('noise').value)&&(!q||[r.identifier,r.design_suites.join(' '),r.backbone,r.loss_label].join(' ').toLowerCase().includes(q)));
const keys=all?Object.keys(rows[0]?.[endpoint]||{}).filter(k=>k!=='count'&&!k.endsWith('_defined')):POINTS;
const fixed=['Job','Design suites','Pipeline','Backbone','Loss','Noise SD',endpoint==='best_test'?'Best epoch':'Final epoch'];
$('runs').tHead.innerHTML='<tr>'+fixed.concat(keys).map(k=>`<th>${k}</th>`).join('')+'</tr>';
$('runs').tBodies[0].innerHTML=rows.map(r=>'<tr>'+[r.identifier,r.design_suites.join(', '),r.pipeline,r.backbone,r.loss_label,r.noise_sd,endpoint==='best_test'?r.best_epoch:r.final_global_epoch,...keys.map(k=>fmt(r[endpoint][k]))].map(v=>`<td>${v}</td>`).join('')+'</tr>').join('');
$('count').textContent=`${rows.length} / ${DATA.records.length} runs`}
document.querySelectorAll('.controls select,.controls input').forEach(x=>x.addEventListener('input',draw));draw();</script></body></html>"""
    replacements = {
        "__CARDS__": cards,
        "__PRIMARY__": html.escape(report["primary_metric"].upper()),
        "__SUITES__": suite_rows,
        "__METADATA__": metadata_html,
        "__FIGURES__": "".join(figure_html),
        "__AUDITS__": audit_rows,
        "__FROZEN_HEADERS__": "".join(
            f"<th>{html.escape(label)}</th>" for _, label in POINT_METRICS
        ),
        "__FROZEN_ROWS__": frozen_rows,
        "__DATA__": data_json,
        "__POINTS__": json.dumps([field for field, _ in POINT_METRICS]),
    }
    for key, value in replacements.items():
        document = document.replace(key, value)
    return document


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--metrics", nargs="+", choices=tuple(METRIC_FIELDS), default=("mae",)
    )
    args = parser.parse_args()
    root, output_dir = args.root.resolve(), args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    protected_before = protected_file_state(output_dir)
    train, banks, audits, raw = collect(root=root)
    checks = snapshot_checks(train)
    recipes = recipe_metadata(train)
    figures, figure_values = [], {}
    for metric in dict.fromkeys(args.metrics):
        metric_output, values = metric_figures(
            records=train, metric=metric, output_dir=output_dir
        )
        figures.extend(metric_output)
        figure_values.update(values)
    matrix = point_matrix(train)
    matrix_name = "point_metrics_sl_vs_tail.png"
    matrix_path = plot_point_matrix(
        data=matrix, output_dir=output_dir, name=matrix_name
    )
    figures.append({
        "name": "point_metrics_sl_vs_tail",
        "metric": "",
        "path": matrix_path,
        "caption": "Matched TailRL minus categorical-MSE deltas; colors normalize separately within each metric.",
    })
    figure_values[matrix_name] = matrix
    frozen_rows = frozen_table(train)
    audit = audit_summary(banks, audits, raw["design"])
    input_paths = {
        name: raw[name + "_path"] for name in ("plan", "design", "state", "manifest")
    }
    generated_at = datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()
    report = {
        "schema": SCHEMA,
        "version": VERSION,
        "generated_at": generated_at,
        "source_root": str(root),
        "primary_metric": args.metrics[0],
        "metrics_generated": list(dict.fromkeys(args.metrics)),
        "input_sha256": {
            name: {"path": str(path), "sha256": sha256(path)}
            for name, path in input_paths.items()
        },
        "manifest_fingerprint": raw["manifest_fingerprint"],
        "audit": audit,
        "recipe_metadata": recipes,
        "snapshot_checks": checks,
        "statistical_inference": {
            "bootstrap": False,
            "confidence_intervals": False,
            "contrasts_json_loaded": False,
        },
        "protected_files": [protected_before],
        "design": raw["design"],
        "records": train,
        "banks": banks,
        "job_audits": audits,
        "figures": figures,
        "figure_values": figure_values,
        "point_metric_pairing": matrix["rows"],
        "frozen_heads_table": frozen_rows,
    }
    summary_rows = descriptive_summary(train, args.metrics[0])
    summary = {
        "schema": SCHEMA,
        "version": VERSION,
        "generated_at": generated_at,
        "source_root": str(root),
        "primary_metric": args.metrics[0],
        "audit": audit,
        "recipe_metadata": recipes,
        "snapshot_checks_passed": checks["all_passed"],
        "statistical_inference": {
            "bootstrap": False,
            "confidence_intervals": False,
            "contrasts_json_loaded": False,
        },
        "protected_files": [protected_before],
        "manifest": {
            "path": str(raw["manifest_path"]),
            "sha256": sha256(raw["manifest_path"]),
            "fingerprint": raw["manifest_fingerprint"],
            "counts": {
                name: len(rows) for name, rows in raw["manifest"]["splits"].items()
            },
        },
        "suite_summaries": summary_rows,
        "figures": [Path(figure["path"]).name for figure in figures],
        "figure_value_keys": sorted(figure_values),
        "point_metric_pairs": len(matrix["rows"]),
    }
    write_json(output_dir / "report-data.json", report)
    write_json(output_dir / "summary.json", summary)
    (output_dir / "report.html").write_text(
        render_html(report=report, figures=figures, summary=summary_rows)
    )
    protected_after = protected_file_state(output_dir)
    if protected_after != protected_before:
        raise ValueError("Protected contrasts.json changed during report generation")
    missing = [
        Path(figure["path"]).name
        for figure in figures
        if not Path(figure["path"]).is_file()
    ]
    if missing:
        raise ValueError(f"Missing generated figures: {missing}")
    print(
        json.dumps(
            {
                "audit": audit,
                "snapshot_checks_passed": checks["all_passed"],
                "primary_metric": args.metrics[0],
                "suite_summaries": summary_rows,
                "figures": [Path(figure["path"]).name for figure in figures],
                "point_metric_pairs": len(matrix["rows"]),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
