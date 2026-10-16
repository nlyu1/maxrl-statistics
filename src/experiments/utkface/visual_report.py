import argparse
import fcntl
import hashlib
import json
import math
import time
from collections import Counter
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import polars as pl

from src.experiments.utkface.coordination import CoordinationConfig
from src.experiments.utkface.metrics import METRIC_NAMES
from src.experiments.utkface.scheduler import dispatch_summary, status_summary
from src.experiments.utkface.study import StudyPlan
from src.experiments.utkface.study_report import FINAL_STAGES

CURVE_METRICS = ("mae_mean", "rmse_mean", "median_spearman", "label_nll", "entropy")
METRICS = (*METRIC_NAMES, "rmse_mean", "median_spearman")
MAXIMIZED_METRICS = {"median_spearman", "mode_accuracy", "mass_within_5"}


def read_json(*, path: Path) -> object:
    for attempt in range(4):
        try:
            return json.loads(path.read_text())
        except json.JSONDecodeError:
            if attempt == 3:
                raise
            time.sleep(0.05)
    raise RuntimeError(path)


def metrics(*, values: dict) -> dict:
    return {name: values.get(name) for name in METRICS}


def collect(*, plan_path: Path, state: dict) -> tuple[dict, list[dict]]:
    plan = StudyPlan.model_validate_json(plan_path.read_text())
    cohort = hashlib.sha256(
        json.dumps(plan.training_source_sha256, sort_keys=True).encode()
    ).hexdigest()
    rows = []
    for job in plan.jobs:
        record = state["jobs"][job.identifier]
        config = job.config.model_dump(mode="json")
        history, result = [], {}
        if record["attempts"]:
            last = record["attempts"][-1]
            config = read_json(path=Path(last["config_path"]))
            history_path = Path(last["output_dir"]) / "history.json"
            if history_path.exists():
                history = read_json(path=history_path)
        if record["status"] == "completed":
            result = read_json(path=Path(record["result_path"]))
            if len(history) != config["epochs"]:
                raise ValueError(f"Incomplete completed history: {job.identifier}")
            if result["data_fingerprint"] != plan.data_fingerprint:
                raise ValueError("Mixed data splits")
            if [row["epoch"] for row in history] != list(
                range(1, config["epochs"] + 1)
            ):
                raise ValueError(f"Nonsequential completed epochs: {job.identifier}")
        best = (
            min(history, key=lambda row: row["validation"]["mae_mean"])
            if history
            else {}
        )
        if result and result["best_epoch"] != best["epoch"]:
            raise ValueError(f"Checkpoint selection mismatch: {job.identifier}")
        extrema = {}
        for name in METRICS:
            defined = [
                row for row in history if row["validation"].get(name) is not None
            ]
            if defined:
                optimum = (max if name in MAXIMIZED_METRICS else min)(
                    defined, key=lambda row: row["validation"][name]
                )
                extrema[name] = {
                    "epoch": optimum["epoch"],
                    "value": optimum["validation"][name],
                }
        curves, cumulative = [], 0
        for row in history:
            cumulative += row["train_count"]
            curves.append([
                row["epoch"],
                *[row["validation"].get(name) for name in CURVE_METRICS],
                row["learning_rate"],
                cumulative,
                cumulative / result.get("global_batch_size", config["batch_size"]),
            ])
        noise_mse = (
            sum(row["label_noise"]["noise_mse_years2"] for row in history)
            / len(history)
            if history
            else 0
        )
        signal = history[0]["label_noise"]["signal_variance_years2"] if history else 0
        rows.append({
            "id": job.identifier,
            "batch": plan.output_dir.name,
            "stage": job.stage,
            "variant": job.variant,
            "method": config["loss"]["method"],
            "head": config["head"],
            "width": config["loss"]["kernel_scale_years"],
            "size": config["train_limit"] or 18462,
            "epochs": config["epochs"],
            "seed": config["seed"],
            "subsetSeed": config["subset_seed"],
            "noiseSeed": config["noise"]["seed"],
            "noise": config["noise"]["mode"],
            "std": config["noise"]["std_years"],
            "schedule": config["schedule"]["lr_schedule"],
            "lr": config["learning_rate"]
            if record["attempts"] or not job.tune_from
            else None,
            "status": record["status"],
            "reason": record.get("reason", ""),
            "completedEpochs": len(history),
            "bestEpoch": best.get("epoch"),
            "best": metrics(values=best.get("validation", {})),
            "final": metrics(values=history[-1]["validation"] if history else {}),
            "test": metrics(values=result.get("test", {})),
            "optima": extrema,
            "snr": 10 * math.log10(signal / noise_mse) if noise_mse else None,
            "noiseRmse": math.sqrt(noise_mse) if noise_mse else 0,
            "fingerprint": plan.data_fingerprint,
            "cohort": cohort,
            "trainingFingerprint": result.get("training_fingerprint"),
            "commit": result.get("environment", {}).get("git_commit", ""),
            "attempts": len(record["attempts"]),
            "curves": curves,
        })
    return {
        "path": str(plan_path),
        "root": str(plan.output_dir),
        "progress": status_summary(plan=plan, state=state),
        "planSha256": hashlib.sha256(plan_path.read_bytes()).hexdigest(),
        "cohort": cohort,
        "dataFingerprint": plan.data_fingerprint,
    }, rows


def build(*, plans: tuple[Path, ...], output_dir: Path, coordination_path: str) -> None:
    control = (
        CoordinationConfig.model_validate_json(Path(coordination_path).read_text())
        if coordination_path
        else None
    )
    lock = control.lock_path.open("a") if control else nullcontext()
    states = []
    with lock as handle:
        if control:
            fcntl.flock(handle, fcntl.LOCK_EX)
        for path in plans:
            plan = StudyPlan.model_validate_json(path.read_text())
            state_path = plan.output_dir / "state.json"
            states.append(
                read_json(path=state_path)
                if state_path.exists()
                else {
                    "status": "not_started",
                    "jobs": {
                        job.identifier: {"status": "pending", "attempts": []}
                        for job in plan.jobs
                    },
                }
            )
    studies, runs = [], []
    for path, state in zip(plans, states, strict=True):
        study, rows = collect(plan_path=path, state=state)
        if control:
            study["progress"] = dispatch_summary(
                plan=StudyPlan.model_validate_json(path.read_text()),
                state=state,
                coordination=control,
            )
            for row in rows:
                if row["id"] in control.held_jobs and row["status"] == "pending":
                    row["status"] = "held"
                    row["reason"] = "training_seed_repetition_not_selected"
        studies.append(study)
        runs.extend(rows)
    completed = [row for row in runs if row["status"] == "completed"]
    stages = Counter(row["stage"] for row in completed)
    evaluations = sum(
        count
        for stage, count in stages.items()
        if stage in FINAL_STAGES | {"subset_replication", "matched_steps"}
    )
    payload = {
        "generated": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "historicalReportDeadline": "2026-10-14T08:30:00+08:00",
        "historicalExtensionCutoff": "2026-10-14T07:30:00+08:00",
        "studies": studies,
        "runs": runs,
        "summary": {
            "completed": len(completed),
            "evaluations": evaluations,
            "preflight": stages["smoke"],
            "tuning": len(completed) - evaluations - stages["smoke"],
            "held": sum(row["status"] == "held" for row in runs),
            "queued": sum(row["status"] == "pending" for row in runs),
            "gpuHours": sum(study["progress"]["gpu_hours"] for study in studies),
            "selectedComplete": all(
                study["progress"]["status"] == "completed" for study in studies
            ),
        },
        "curveColumns": [
            "epoch",
            *CURVE_METRICS,
            "learning_rate",
            "examples",
            "updates",
        ],
    }
    serialized = json.dumps(payload, separators=(",", ":"), allow_nan=False).replace(
        "<", "\\u003c"
    )
    template = Path(__file__).with_name("visual_report.html").read_text()
    rendered = template.replace("__REPORT_DATA__", serialized)
    output_dir.mkdir(parents=True, exist_ok=True)
    pending = output_dir / "dashboard.next.html"
    pending.write_text(rendered)
    pending.replace(output_dir / "dashboard.html")
    (output_dir / "dashboard-data.json").write_text(
        json.dumps(payload, allow_nan=False)
    )
    flat = []
    curves = []
    for row in runs:
        flat.append({
            **{
                key: value
                for key, value in row.items()
                if key not in {"curves", "best", "final", "test", "optima"}
            },
            **{
                f"{phase}_{key}": value
                for phase in ("best", "final", "test")
                for key, value in row[phase].items()
            },
        })
        curves.extend(
            {"id": row["id"], **dict(zip(payload["curveColumns"], values, strict=True))}
            for values in row["curves"]
        )
    pl.DataFrame(flat, infer_schema_length=None).write_csv(
        output_dir / "dashboard-runs.csv"
    )
    if curves:
        pl.DataFrame(curves, infer_schema_length=None).write_csv(
            output_dir / "dashboard-curves.csv"
        )
    print(
        json.dumps({
            "output": str(output_dir / "dashboard.html"),
            "generated": payload["generated"],
            "planned_runs": len(runs),
            "completed": sum(row["status"] == "completed" for row in runs),
        })
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plans", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--coordination", default="")
    parser.add_argument("--watch-until", default="")
    parser.add_argument("--refresh-seconds", type=float, default=300)
    args = parser.parse_args()
    if args.refresh_seconds < 60:
        parser.error("Refresh interval must be at least 60 seconds")
    deadline = datetime.fromisoformat(args.watch_until) if args.watch_until else None
    if deadline and deadline.tzinfo is None:
        parser.error("Watch deadline must include a timezone")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    lock = (
        (args.output_dir / "report-watch.lock").open("a") if deadline else nullcontext()
    )
    with lock as handle:
        if deadline:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        while True:
            build(
                plans=tuple(args.plans),
                output_dir=args.output_dir,
                coordination_path=args.coordination,
            )
            if not deadline or time.time() >= deadline.timestamp():
                break
            time.sleep(
                max(0, min(args.refresh_seconds, deadline.timestamp() - time.time()))
            )


if __name__ == "__main__":
    main()
