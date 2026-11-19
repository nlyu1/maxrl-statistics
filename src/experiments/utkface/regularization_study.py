import argparse
import fcntl
import hashlib
import json
import math
import os
import statistics
import subprocess
from pathlib import Path

from src.experiments.utkface.adaptation_config import AdaptationConfig, AdaptationPlan
from src.experiments.utkface.adaptation_dispatch import run as dispatch
from src.experiments.utkface.adaptation_plan import VARIANTS
from src.experiments.utkface.run import write_json
from src.experiments.utkface.scratch_longrun import (
    checksum,
    configured,
    job_for,
    persist,
    plan_for,
    read,
)

WEIGHT_DECAYS = (0.0, 1e-5, 1e-4, 1e-3)
NOISES = (0, 3, 5, 10)
METHODS = tuple(v for v in VARIANTS if v[0] != "grpo_mse")
INITIALIZATIONS = (("imagenet", 50), ("scratch", 500))
POINT_METRICS = tuple(
    f"{metric}_{point}"
    for point in ("mean", "median", "mode")
    for metric in ("mae", "rmse", "spearman")
)


def scientific_config(*, config: AdaptationConfig) -> dict:
    if config.initialization not in {"imagenet", "scratch"} or config.backbone_frozen:
        raise ValueError("This study requires fresh end-to-end initialization")
    payload = config.model_dump(mode="json")
    for key in ("initial_checkpoint", "feature_bank", "checkpoint_every", "control"):
        del payload[key]
    for key in (
        "output_dir",
        "device",
        "num_workers",
        "cpu_threads",
        "save_checkpoint",
    ):
        del payload["run"][key]
    payload["schedule_epochs"] = config.control.schedule_epochs
    return payload


def fingerprint(*, config: AdaptationConfig) -> str:
    encoded = json.dumps(scientific_config(config=config), sort_keys=True).encode()
    return hashlib.sha256(encoded).hexdigest()


def study_config(
    *,
    source: dict,
    output: Path,
    initialization: str,
    epochs: int,
    noise: int,
    decay: float,
) -> AdaptationConfig:
    payload = {
        **source,
        "run": {**source["run"]},
        "initialization": initialization,
        "schedule_kind": "warmup_constant",
    }
    payload["run"].update({
        "weight_decay": decay,
        "pretrained": initialization == "imagenet",
        "noise": {
            "mode": "fixed" if noise else "none",
            "std_years": noise,
            "seed": 2026,
        },
        "schedule": {"lr_schedule": "flat", "warmup_ratio": 0.05, "lr_min_ratio": 0.1},
    })
    return configured(
        source=payload,
        output=output,
        epochs=epochs,
        lr=3e-4,
        schedule_epochs=50,
        extend=output / "unused",
        extend_epoch=0,
        evaluate_test=True,
    )


def verify_reuse(*, path: Path, desired: AdaptationConfig) -> dict:
    result = read(path)
    actual = AdaptationConfig.model_validate(result["config"])
    if scientific_config(config=actual) != scientific_config(config=desired):
        raise ValueError(f"Reuse changes scientific settings: {path}")
    if (
        result["status"] != "complete"
        or result["final_global_epoch"] != desired.run.epochs
    ):
        raise ValueError(f"Reuse is incomplete: {path}")
    if [p["epoch"] for p in result["history"]] != list(
        range(1, desired.run.epochs + 1)
    ):
        raise ValueError(f"Reuse lacks its complete history: {path}")
    for name in ("best.pt", "final.pt", "history.json"):
        if not (path.parent / name).is_file():
            raise FileNotFoundError(path.parent / name)
    source_root = Path(__file__).resolve().parents[3]
    hashes = result["provenance"]["source_sha256"]
    if not hashes:
        raise ValueError("Reuse has no recorded learning-code hashes")
    for name, expected in hashes.items():
        if checksum(source_root / name) != expected:
            raise ValueError(f"Reuse learning code differs: {name}")
    return {"result_sha256": checksum(path), "verified_learning_files": len(hashes)}


def create(
    *, root: Path, source: Path, long_source: Path, max_gpus: int, per_gpu_jobs: int
) -> None:
    if root.exists():
        raise FileExistsError(root)
    if max_gpus < 0 or not 1 <= per_gpu_jobs <= 8:
        raise ValueError("Invalid GPU or per-GPU slot bound")
    root.mkdir(parents=True)
    manifest = Path(read(source / "plan.json")["manifest"])
    jobs, rows, protected = [], [], {}
    queue = root / "queue"
    for initialization, epochs in INITIALIZATIONS:
        for variant in METHODS:
            name, method, head = variant
            template = (
                source
                / "runs"
                / (
                    f"scratch_warmup_constant_{name}_sd0"
                    if initialization == "scratch"
                    else f"aug_original_full_{name}_sd0"
                )
                / "result.json"
            )
            baseline = read(template)
            protected[str(template)] = checksum(template)
            for noise_index, noise in enumerate(NOISES):
                for decay_index, decay in enumerate(WEIGHT_DECAYS):
                    identifier = f"{initialization}_{name}_sd{noise}_wd{decay:g}"
                    directory = queue / "runs" / identifier
                    config = study_config(
                        source=baseline["config"],
                        output=directory,
                        initialization=initialization,
                        epochs=epochs,
                        noise=noise,
                        decay=decay,
                    )
                    reused = (
                        initialization == "scratch"
                        and decay == 1e-4
                        and noise in (0, 5, 10)
                    )
                    result_path = directory / "result.json"
                    reuse = {}
                    if reused:
                        result_path = (
                            long_source
                            / "long"
                            / "runs"
                            / f"long_warmup_constant_{name}_sd{noise}_lr3e-04"
                            / "result.json"
                        )
                        reuse = verify_reuse(path=result_path, desired=config)
                        protected[str(result_path)] = reuse["result_sha256"]
                    else:
                        jobs.append(
                            job_for(
                                root=queue,
                                identifier=identifier,
                                config=config,
                                priority=(1000 if initialization == "scratch" else 0)
                                + noise_index * 10
                                + decay_index,
                            )
                        )
                    rows.append({
                        "job": identifier,
                        "initialization": initialization,
                        "epochs": epochs,
                        "variant": name,
                        "method": method,
                        "head": head,
                        "noise_sd": noise,
                        "weight_decay": decay,
                        "reused": reused,
                        "result": str(result_path),
                        "scientific_fingerprint": fingerprint(config=config),
                        "reuse_audit": reuse,
                    })
    if len(rows) != 128 or len(jobs) != 116 or sum(r["reused"] for r in rows) != 12:
        raise ValueError(
            "Weight-decay matrix differs from the requested 128 configurations"
        )
    plan_for(
        root=queue,
        manifest=manifest,
        jobs=jobs,
        max_gpus=max_gpus,
        per_gpu_jobs=per_gpu_jobs,
        min_free_gib=40,
    )
    persist(root / "registry.json", {"rows": rows})
    persist(
        root / "study.json",
        {
            "source_study": str(source),
            "source_long_study": str(long_source),
            "source_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip(),
            "manifest": str(manifest),
            "manifest_sha256": checksum(manifest),
            "weight_decays": list(WEIGHT_DECAYS),
            "noise_levels": list(NOISES),
            "variants": [list(v) for v in METHODS],
            "initializations": dict(INITIALIZATIONS),
            "learning_rate": 3e-4,
            "warmup_epochs": 2.5,
            "schedule": "Linear warmup for 160 updates, then constant through the final epoch.",
            "train_limit": 4096,
            "model_seed": 42,
            "noise_seed": 2026,
            "noise_mode": "fixed",
            "input_pipeline": "original",
            "max_gpus": max_gpus,
            "per_gpu_jobs": per_gpu_jobs,
            "configurations": 128,
            "new_jobs": len(jobs),
            "reused_jobs": 12,
            "checkpoint_every": 5,
            "retained_full_checkpoints": 1,
            "model_milestones": [25, 50, 100, 200, 300, 400, 500],
            "checkpoint_selection": "Minimum clean validation mean-prediction MAE, including epoch zero.",
            "recorded_point_metrics": list(POINT_METRICS),
            "protected_source_results": protected,
        },
    )
    write_json(
        path=root / "workflow.json",
        payload={
            "phase": "ready",
            "status": "ready",
            "configurations": 128,
            "new_jobs": 116,
            "reused_jobs": 12,
        },
    )


def select_decays(*, rows: list[dict]) -> list[dict]:
    selections = []
    for initialization, _ in INITIALIZATIONS:
        for name, _, _ in METHODS:
            candidates = []
            for decay in WEIGHT_DECAYS:
                fits = [
                    r
                    for r in rows
                    if r["initialization"] == initialization
                    and r["variant"] == name
                    and r["weight_decay"] == decay
                ]
                if {r["noise_sd"] for r in fits} != set(NOISES) or len(fits) != 4:
                    raise ValueError("WD selection requires all four noise conditions")
                candidates.append({
                    "weight_decay": decay,
                    "mean_validation_mae": statistics.mean(
                        r["best_validation"]["mae_mean"] for r in fits
                    ),
                })
            best = min(
                candidates,
                key=lambda r: (
                    r["mean_validation_mae"],
                    r["weight_decay"] != 1e-4,
                    r["weight_decay"],
                ),
            )
            selections.append({
                "initialization": initialization,
                "variant": name,
                "selected_weight_decay": best["weight_decay"],
                "candidates": candidates,
            })
    return selections


def report(*, root: Path) -> None:
    import html

    import polars as pl

    from src.experiments.utkface.adaptation_curves import grid, item
    from src.experiments.utkface.adaptation_report import LOSS_LABELS

    rows, metric_rows = [], []
    for entry in read(root / "registry.json")["rows"]:
        result = read(Path(entry["result"]))
        config = AdaptationConfig.model_validate(result["config"])
        if fingerprint(config=config) != entry["scientific_fingerprint"]:
            raise ValueError(f"Completed configuration differs: {entry['job']}")
        if (
            result["status"] != "complete"
            or result["final_global_epoch"] != entry["epochs"]
        ):
            raise ValueError(f"Incomplete report result: {entry['job']}")
        history = result["history"]
        if [p["epoch"] for p in history] != list(range(1, entry["epochs"] + 1)):
            raise ValueError(f"Incomplete report history: {entry['job']}")
        points = [
            {"epoch": 0, "validation": result["initialization"]["validation"]}
        ] + history
        record = {
            **entry,
            "identifier": entry["job"],
            "loss": entry["method"],
            "loss_label": LOSS_LABELS[entry["method"]],
            "best_epoch": result["best_epoch"],
            "best_validation": result["best_validation"],
            "best_test": result["best_test"],
            "final_test": result["final_test"],
            "points": points,
        }
        rows.append(record)
        metric = {
            key: entry[key]
            for key in (
                "job",
                "initialization",
                "epochs",
                "variant",
                "noise_sd",
                "weight_decay",
                "reused",
            )
        }
        metric.update({
            "best_epoch": result["best_epoch"],
            "validation_mean_mae": result["best_validation"]["mae_mean"],
        })
        for endpoint in ("best_test", "final_test"):
            metric.update({
                f"{endpoint}_{key}": result[endpoint][key] for key in POINT_METRICS
            })
        metric_rows.append(metric)
    if len(rows) != 128:
        raise ValueError("Incomplete result matrix")
    output = root / "report"
    output.mkdir(exist_ok=True)
    selections = select_decays(rows=rows)
    write_json(
        path=output / "results.json",
        payload={"runs": rows, "validation_selected_weight_decays": selections},
    )
    pl.DataFrame(metric_rows).write_csv(output / "metrics.tsv", separator="\t")
    figures = []
    for initialization, epochs in INITIALIZATIONS:
        for metric, span in (
            ("mae_mean", 1.4),
            ("rmse_mean", 2.1),
            ("spearman_mean", 0.09),
        ):
            for name, method, _ in METHODS:
                panels = []
                for noise in NOISES:
                    fits = [
                        r
                        for r in rows
                        if r["initialization"] == initialization
                        and r["variant"] == name
                        and r["noise_sd"] == noise
                    ]
                    panels.append({
                        "title": "Clean" if not noise else f"Fixed SD {noise}",
                        "items": [
                            item(
                                r,
                                label=f"WD {r['weight_decay']:g}",
                                color=("#276fbf", "#ef8354", "#198754", "#7b4ab5")[
                                    WEIGHT_DECAYS.index(r["weight_decay"])
                                ],
                            )
                            for r in fits
                        ],
                    })
                filename = f"decay_{initialization}_{name}_{metric}.png"
                title = f"{initialization}, {epochs} epochs / {LOSS_LABELS[method]} / weight decay"
                grid(
                    panels=panels,
                    title=title,
                    metric=metric,
                    output_dir=output,
                    name=filename,
                    span=span,
                )
                figures.append({"file": filename, "title": title + " / " + metric})
            for decay in WEIGHT_DECAYS:
                panels = []
                for noise in NOISES:
                    fits = [
                        r
                        for r in rows
                        if r["initialization"] == initialization
                        and r["weight_decay"] == decay
                        and r["noise_sd"] == noise
                    ]
                    panels.append({
                        "title": "Clean" if not noise else f"Fixed SD {noise}",
                        "items": [item(r) for r in fits],
                    })
                filename = f"methods_{initialization}_wd{decay:g}_{metric}.png"
                title = (
                    f"{initialization}, {epochs} epochs / WD {decay:g} / four methods"
                )
                grid(
                    panels=panels,
                    title=title,
                    metric=metric,
                    output_dir=output,
                    name=filename,
                    span=span,
                )
                figures.append({"file": filename, "title": title + " / " + metric})
    columns = (
        "initialization",
        "epochs",
        "variant",
        "noise_sd",
        "weight_decay",
        "best_epoch",
        "validation_mean_mae",
        "best_test_mae_mean",
        "best_test_rmse_mean",
        "best_test_spearman_mean",
        "final_test_mae_mean",
        "reused",
    )
    table_rows = "".join(
        "<tr>"
        + "".join(f"<td>{html.escape(str(row[key]))}</td>" for key in columns)
        + "</tr>"
        for row in metric_rows
    )
    gallery = "".join(
        f"<details><summary>{html.escape(f['title'])}</summary><img loading=lazy src='{f['file']}'></details>"
        for f in figures
    )
    (output / "report.html").write_text(
        "<!doctype html><html><head><meta charset=utf-8><title>UTKFace weight decay</title><style>body{font:15px/1.5 system-ui;max-width:1800px;margin:28px auto;padding:0 22px;color:#172438}table{border-collapse:collapse;font-size:12px}td,th{border:1px solid #ccd5df;padding:6px}img{width:100%}details{margin:16px 0}summary{font-weight:600}</style></head><body><h1>Weight decay: 128 configurations</h1><p>Constant LR 3e-4 after 2.5 warmup epochs. Four weight decays, four methods, clean/fixed SD3/SD5/SD10. End-to-end ImageNet 50 epochs and scratch 500 epochs. Model seed 42; 4,096 training images. Twelve exact scratch controls are reused. Checkpoints and global weight-decay selections use clean validation mean MAE. All plotted curves are raw; lower panels enlarge the near-best region. Stars always mark mean-MAE checkpoint selection.</p><p><a href='metrics.tsv'>All nine mean/median/mode metrics</a> / <a href='results.json'>All curves and validation-only WD selection</a></p><table><thead><tr>"
        + "".join(f"<th>{key}</th>" for key in columns)
        + "</tr></thead><tbody>"
        + table_rows
        + "</tbody></table>"
        + gallery
        + "</body></html>"
    )
    write_json(
        path=output / "summary.json",
        payload={
            "configurations": len(rows),
            "new_fits": 116,
            "reused_fits": 12,
            "figures": len(figures),
            "selection_uses_test": False,
            "selected_weight_decays": selections,
        },
    )


def run_study(*, root: Path) -> None:
    with (root / "study.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        (root / "coordinator.pid").write_text(str(os.getpid()) + "\n")
        study = read(root / "study.json")
        plan = AdaptationPlan.model_validate(read(root / "queue" / "plan.json"))
        write_json(
            path=root / "workflow.json",
            payload={
                "phase": "training",
                "status": "running",
                "configurations": 128,
                "new_jobs": 116,
                "reused_jobs": 12,
                "max_gpus": plan.max_gpus,
                "per_gpu_jobs": plan.per_gpu_jobs,
            },
        )
        dispatch(plan=plan)
        status = read(plan.output_dir / "state.json")["status"]
        if status != "completed":
            write_json(
                path=root / "workflow.json",
                payload={"phase": "training", "status": status},
            )
            raise RuntimeError(f"Weight-decay queue ended with status {status}")
        for path, expected in study["protected_source_results"].items():
            if checksum(Path(path)) != expected:
                raise ValueError(f"Protected source result changed: {path}")
        report(root=root)
        write_json(
            path=root / "workflow.json",
            payload={
                "phase": "complete",
                "status": "completed",
                "configurations": 128,
                "report": str(root / "report"),
            },
        )


def preflight(*, root: Path, source: Path) -> None:
    root.mkdir(parents=True, exist_ok=False)
    jobs = []
    manifest = Path(read(source / "plan.json")["manifest"])
    for initialization, _ in INITIALIZATIONS:
        for name, _, _ in METHODS:
            template = (
                source
                / "runs"
                / (
                    f"scratch_warmup_constant_{name}_sd0"
                    if initialization == "scratch"
                    else f"aug_original_full_{name}_sd0"
                )
                / "result.json"
            )
            base = study_config(
                source=read(template)["config"],
                output=root / "runs" / f"{initialization}_{name}",
                initialization=initialization,
                epochs=50,
                noise=3,
                decay=0 if initialization == "scratch" else 1e-3,
            )
            config = base.model_copy(
                update={
                    "run": base.run.model_copy(
                        update={
                            "epochs": 5,
                            "train_limit": 512,
                            "eval_limit": 128,
                            "num_workers": 2,
                            "cpu_threads": 2,
                            "evaluate_test": False,
                            "schedule": base.run.schedule.model_copy(
                                update={"warmup_ratio": 0.5}
                            ),
                        }
                    ),
                    "control": base.control.model_copy(
                        update={"schedule_epochs": 5, "model_milestones": (5,)}
                    ),
                    "checkpoint_every": 5,
                }
            )
            config = AdaptationConfig.model_validate(config.model_dump(mode="json"))
            jobs.append(
                job_for(root=root, identifier=f"{initialization}_{name}", config=config)
            )
    plan = plan_for(
        root=root,
        manifest=manifest,
        jobs=jobs,
        max_gpus=1,
        per_gpu_jobs=2,
        min_free_gib=1,
    )
    dispatch(plan=plan)
    common_categorical = {
        initialization: set() for initialization, _ in INITIALIZATIONS
    }
    for job in jobs:
        result = read(job.output_dir / "result.json")
        if result["final_global_epoch"] != 5 or len(result["history"]) != 5:
            raise AssertionError(f"Preflight did not finish: {job.identifier}")
        rates = result["history"]
        if any(
            abs(lr - 3e-4) > 1e-12 for row in rates[2:] for lr in row["learning_rates"]
        ):
            raise AssertionError("Preflight LR is not constant after 2.5 warmup epochs")
        if any(not math.isfinite(row["train_loss"]) for row in rates):
            raise AssertionError("Preflight loss is nonfinite")
        if not set(POINT_METRICS).issubset(result["best_validation"]):
            raise AssertionError("Preflight misses a requested point metric")
        if result["peak_allocated_mib"] <= 0:
            raise AssertionError("Preflight did not allocate CUDA memory")
        if result["config"]["run"]["head"] == "categorical":
            common_categorical[result["config"]["initialization"]].add(
                result["initialization"]["model_sha256"]
            )
    if any(len(values) != 1 for values in common_categorical.values()):
        raise AssertionError(
            "Categorical objectives do not share the same initial model"
        )
    write_json(
        path=root / "verification.json",
        payload={
            "passed": True,
            "jobs": len(jobs),
            "warmup_epochs": 2.5,
            "initializations": [x[0] for x in INITIALIZATIONS],
            "methods": [x[0] for x in METHODS],
            "tested_weight_decays": [0, 1e-3],
        },
    )
    print(json.dumps(read(root / "verification.json")))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-study", type=Path)
    parser.add_argument("--source-long-study", type=Path)
    parser.add_argument("--create", action="store_true")
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--report", action="store_true")
    parser.add_argument("--max-gpus", type=int, default=4)
    parser.add_argument("--per-gpu-jobs", type=int, default=8)
    args = parser.parse_args()
    if args.preflight:
        if args.source_study is None:
            raise ValueError("--source-study is required")
        preflight(root=args.output_dir, source=args.source_study)
    elif args.create:
        if args.source_study is None or args.source_long_study is None:
            raise ValueError("Both source studies are required")
        create(
            root=args.output_dir,
            source=args.source_study,
            long_source=args.source_long_study,
            max_gpus=args.max_gpus,
            per_gpu_jobs=args.per_gpu_jobs,
        )
        print(json.dumps(read(args.output_dir / "study.json"), indent=2))
    elif args.report:
        report(root=args.output_dir)
    else:
        run_study(root=args.output_dir)


if __name__ == "__main__":
    main()
