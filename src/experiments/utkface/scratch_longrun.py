import argparse
import hashlib
import json
import math
import os
import statistics
import subprocess
from pathlib import Path

from src.experiments.utkface.adaptation_config import (
    AdaptationConfig,
    AdaptationJob,
    AdaptationPlan,
    TrainingControl,
)
from src.experiments.utkface.adaptation_dispatch import run as dispatch
from src.experiments.utkface.adaptation_plan import VARIANTS
from src.experiments.utkface.run import write_json

RATES = (3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2)
NOISES = (0, 5, 10)
SCHEDULES = (
    ("warmup_constant", "scratch_warmup_constant", 3e-4),
    ("cosine", "scratch_cosine", 1e-3),
)
DISPATCH_LOCK = Path("/tmp/utkface-extension-20261014/global-dispatch.lock")


def read(path: Path):
    return json.loads(path.read_text())


def checksum(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def persist(path: Path, payload: dict) -> None:
    if path.exists():
        if read(path) != payload:
            raise ValueError(f"Immutable study artifact differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(path=path, payload=payload)


def configured(
    *,
    source: dict,
    output: Path,
    epochs: int,
    lr: float,
    schedule_epochs: int,
    extend: Path,
    extend_epoch: int,
    evaluate_test: bool,
) -> AdaptationConfig:
    payload = {**source, "run": {**source["run"]}}
    payload["run"].update({
        "output_dir": str(output),
        "epochs": epochs,
        "learning_rate": lr,
        "device": "cuda:0",
        "evaluate_test": evaluate_test,
    })
    payload.update({
        "backbone_learning_rate": lr,
        "initial_checkpoint": str(output / "unused"),
        "feature_bank": str(output / "unused"),
        "checkpoint_every": 5,
        "control": TrainingControl(
            schedule_epochs=schedule_epochs,
            extend_checkpoint=extend,
            extend_epoch=extend_epoch,
            keep_latest_checkpoints=1,
            model_milestones=tuple(
                e for e in (25, 50, 100, 200, 300, 400, 500) if e <= epochs
            ),
        ).model_dump(mode="json"),
    })
    return AdaptationConfig.model_validate(payload)


def job_for(
    *,
    root: Path,
    identifier: str,
    config: AdaptationConfig,
    priority: int = 0,
    dependencies: tuple[str, ...] = (),
) -> AdaptationJob:
    path = root / "configs" / f"{identifier}.json"
    persist(path, config.model_dump(mode="json"))
    return AdaptationJob(
        identifier=identifier,
        kind="train",
        config_path=path,
        output_dir=config.run.output_dir,
        dependencies=dependencies,
        priority=priority,
    )


def plan_for(
    *,
    root: Path,
    manifest: Path,
    jobs: list[AdaptationJob],
    max_gpus: int,
    per_gpu_jobs: int,
    min_free_gib: float,
) -> AdaptationPlan:
    plan = AdaptationPlan(
        output_dir=root,
        manifest=manifest,
        jobs=tuple(jobs),
        max_gpus=max_gpus,
        per_gpu_jobs=per_gpu_jobs,
        min_free_gib=min_free_gib,
        poll_seconds=5,
        dispatch_lock=DISPATCH_LOCK,
    )
    persist(root / "plan.json", plan.model_dump(mode="json"))
    return plan


def create(*, root: Path, source: Path, max_gpus: int, per_gpu_jobs: int) -> None:
    if not 1 <= max_gpus <= 4 or not 1 <= per_gpu_jobs <= 8:
        raise ValueError("Study limits are at most four GPUs and eight jobs/GPU")
    if root.exists():
        raise FileExistsError(root)
    root.mkdir(parents=True)
    manifest = Path(read(source / "plan.json")["manifest"])
    jobs, registry = [], []
    screen = root / "screen"
    for schedule, source_suite, default in SCHEDULES:
        for variant in VARIANTS:
            for noise in NOISES:
                old_directory = (
                    source / "runs" / f"{source_suite}_{variant[0]}_sd{noise}"
                )
                baseline = read(old_directory / "result.json")
                if (
                    baseline["status"] != "complete"
                    or baseline["final_global_epoch"] != 50
                ):
                    raise ValueError(f"Incomplete source baseline: {old_directory}")
                if not (old_directory / "checkpoints" / "epoch_050.pt").is_file():
                    raise FileNotFoundError(
                        old_directory / "checkpoints" / "epoch_050.pt"
                    )
                for lr in RATES:
                    identifier = f"screen_{schedule}_{variant[0]}_sd{noise}_lr{lr:.0e}"
                    reused = lr == default
                    directory = (
                        old_directory if reused else screen / "runs" / identifier
                    )
                    if not reused:
                        config = configured(
                            source=baseline["config"],
                            output=directory,
                            epochs=50,
                            lr=lr,
                            schedule_epochs=50,
                            extend=directory / "unused",
                            extend_epoch=0,
                            evaluate_test=False,
                        )
                        jobs.append(
                            job_for(root=screen, identifier=identifier, config=config)
                        )
                    registry.append({
                        "schedule": schedule,
                        "variant": variant[0],
                        "method": variant[1],
                        "noise_sd": noise,
                        "lr": lr,
                        "default_lr": default,
                        "reused": reused,
                        "job": identifier,
                        "result": str(directory / "result.json"),
                        "checkpoint": str(directory / "checkpoints" / "epoch_050.pt"),
                        "source_result_sha256": checksum(directory / "result.json")
                        if reused
                        else None,
                    })
    if len(jobs) != 150 or len(registry) != 180:
        raise ValueError("Unexpected LR screening matrix")
    plan_for(
        root=screen,
        manifest=manifest,
        jobs=jobs,
        max_gpus=max_gpus,
        per_gpu_jobs=per_gpu_jobs,
        min_free_gib=40,
    )
    persist(root / "screen-registry.json", {"rows": registry})
    persist(
        root / "study.json",
        {
            "source_study": str(source),
            "manifest": str(manifest),
            "source_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip(),
            "learning_rates": list(RATES),
            "noise_levels": list(NOISES),
            "max_gpus": max_gpus,
            "per_gpu_jobs": per_gpu_jobs,
            "selection": "One LR per method and schedule, minimizing the mean of validation-selected mean MAE over clean/SD5/SD10; no test metrics used.",
            "constant_warmup": "5% of the 50-epoch screen (2.5 epochs), retained during extension.",
            "cosine_warmup": "5% of the fresh 500-epoch trajectory (25 epochs).",
            "screen_jobs": 150,
            "reused_screen_results": 30,
            "long_runs_maximum": 60,
            "default_long_runs": 30,
            "regular_checkpoint_every": 5,
            "latest_full_checkpoints_retained": 1,
            "model_milestones": [25, 50, 100, 200, 300, 400, 500],
        },
    )


def select_rates(*, root: Path) -> dict:
    registry_path = root / "screen-registry.json"
    rows = read(registry_path)["rows"]
    groups = []
    for schedule, _, default in SCHEDULES:
        for variant in VARIANTS:
            candidates = []
            for lr in RATES:
                entries = [
                    r
                    for r in rows
                    if r["schedule"] == schedule
                    and r["variant"] == variant[0]
                    and r["lr"] == lr
                ]
                if {r["noise_sd"] for r in entries} != set(NOISES):
                    raise ValueError("LR selection lacks a noise condition")
                values, details, invalid = [], [], []
                for entry in entries:
                    path = Path(entry["result"])
                    if not path.is_file():
                        invalid.append(entry["noise_sd"])
                        continue
                    result = read(path)
                    value = result["best_validation"]["mae_mean"]
                    if (
                        result["status"] != "complete"
                        or result["final_global_epoch"] != 50
                        or not math.isfinite(value)
                    ):
                        raise ValueError(f"Invalid screening result: {path}")
                    if result["config"]["run"]["learning_rate"] != lr:
                        raise ValueError(f"Screening LR differs: {path}")
                    values.append(value)
                    details.append({
                        "noise_sd": entry["noise_sd"],
                        "validation_mae": value,
                        "selected_epoch": result["best_epoch"],
                        "result": str(path),
                        "sha256": checksum(path),
                    })
                candidates.append({
                    "lr": lr,
                    "mean_validation_mae": statistics.mean(values)
                    if not invalid
                    else None,
                    "invalid_noise_conditions": invalid,
                    "conditions": details,
                })
            eligible = [c for c in candidates if c["mean_validation_mae"] is not None]
            if not eligible:
                raise ValueError(f"No valid LR for {schedule}/{variant[0]}")
            best = min(
                eligible,
                key=lambda c: (c["mean_validation_mae"], c["lr"] != default, c["lr"]),
            )
            groups.append({
                "schedule": schedule,
                "variant": variant[0],
                "method": variant[1],
                "best_lr": best["lr"],
                "default_lr": default,
                "candidates": candidates,
            })
    selection = {
        "screen_registry_sha256": checksum(registry_path),
        "groups": groups,
        "criterion": "Average validation-selected mean MAE across clean/SD5/SD10.",
        "test_metrics_used": False,
    }
    persist(root / "lr-selection.json", selection)
    return selection


def long_plan(*, root: Path, study: dict, selection: dict) -> AdaptationPlan:
    stage = root / "long"
    registry = read(root / "screen-registry.json")["rows"]
    jobs, rows = [], []
    for group in selection["groups"]:
        for lr in sorted({group["default_lr"], group["best_lr"]}):
            for noise in NOISES:
                entry = next(
                    r
                    for r in registry
                    if r["schedule"] == group["schedule"]
                    and r["variant"] == group["variant"]
                    and r["lr"] == lr
                    and r["noise_sd"] == noise
                )
                screen = read(Path(entry["result"]))
                identifier = (
                    f"long_{group['schedule']}_{group['variant']}_sd{noise}_lr{lr:.0e}"
                )
                constant = group["schedule"] == "warmup_constant"
                directory = stage / "runs" / identifier
                config = configured(
                    source=screen["config"],
                    output=directory,
                    epochs=500,
                    lr=lr,
                    schedule_epochs=50 if constant else 500,
                    extend=Path(entry["checkpoint"])
                    if constant
                    else directory / "unused",
                    extend_epoch=50 if constant else 0,
                    evaluate_test=True,
                )
                jobs.append(
                    job_for(
                        root=stage,
                        identifier=identifier,
                        config=config,
                        priority=0 if lr == group["default_lr"] else 1,
                    )
                )
                rows.append({
                    "job": identifier,
                    "schedule": group["schedule"],
                    "variant": group["variant"],
                    "method": group["method"],
                    "noise_sd": noise,
                    "lr": lr,
                    "roles": [
                        role
                        for role, rate in (
                            ("default", group["default_lr"]),
                            ("selected", group["best_lr"]),
                        )
                        if lr == rate
                    ],
                    "continued_from_50": constant,
                    "result": str(directory / "result.json"),
                })
    persist(root / "long-registry.json", {"rows": rows})
    return plan_for(
        root=stage,
        manifest=Path(study["manifest"]),
        jobs=jobs,
        max_gpus=study["max_gpus"],
        per_gpu_jobs=study["per_gpu_jobs"],
        min_free_gib=40,
    )


def validate_screen_failures(*, plan: AdaptationPlan) -> None:
    state = read(plan.output_dir / "state.json")
    for identifier, record in state["jobs"].items():
        if record["status"] == "completed":
            continue
        if record["status"] != "failed" or not record["attempts"]:
            raise RuntimeError(
                f"Screening did not finish: {identifier}/{record['status']}"
            )
        log = Path(record["attempts"][-1]["log_path"]).read_text()
        if not any(
            token in log.lower()
            for token in ("non-finite", "nonfinite", "must be finite")
        ):
            raise RuntimeError(
                f"Unexpected screening failure: {identifier}; inspect {record['attempts'][-1]['log_path']}"
            )


def report_results(*, root: Path) -> None:
    from src.experiments.utkface.adaptation_curves import grid, item
    from src.experiments.utkface.adaptation_report import LOSS_LABELS

    records = []
    for row in read(root / "long-registry.json")["rows"]:
        result = read(Path(row["result"]))
        points = [
            {"epoch": 0, "validation": result["initialization"]["validation"]}
        ] + result["history"]
        records.append({
            **row,
            "identifier": row["job"],
            "loss": row["method"],
            "loss_label": LOSS_LABELS[row["method"]],
            "best_epoch": result["best_epoch"],
            "points": points,
            "best_test": result["best_test"],
            "final_test": result["final_test"],
            "best_validation": result["best_validation"],
        })
    output = root / "report"
    output.mkdir(exist_ok=True)
    for schedule, _, _ in SCHEDULES:
        for variant in VARIANTS:
            panels = []
            for noise in NOISES:
                fits = [
                    r
                    for r in records
                    if r["schedule"] == schedule
                    and r["variant"] == variant[0]
                    and r["noise_sd"] == noise
                ]
                panels.append({
                    "title": f"SD {noise}",
                    "items": [
                        item(
                            r,
                            label=f"{'+'.join(r['roles'])} LR {r['lr']:g}",
                            color="#276fbf" if "default" in r["roles"] else "#198754",
                        )
                        for r in fits
                    ],
                })
            grid(
                panels=panels,
                title=f"500-epoch scratch · {schedule} · {LOSS_LABELS[variant[1]]}",
                metric="mae_mean",
                output_dir=output,
                name=f"{schedule}_{variant[0]}.png",
                span=1.3,
            )
    write_json(
        path=output / "results.json",
        payload={"runs": records, "selection": read(root / "lr-selection.json")},
    )


def run_study(*, root: Path) -> None:
    import fcntl

    study = read(root / "study.json")
    with (root / "study.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        (root / "coordinator.pid").write_text(str(os.getpid()) + "\n")
        screen = AdaptationPlan.model_validate(read(root / "screen" / "plan.json"))
        write_json(
            path=root / "workflow.json",
            payload={
                "phase": "screen_50",
                "status": "running",
                "max_gpus": study["max_gpus"],
                "per_gpu_jobs": study["per_gpu_jobs"],
            },
        )
        dispatch(plan=screen)
        validate_screen_failures(plan=screen)
        selection = select_rates(root=root)
        long = long_plan(root=root, study=study, selection=selection)
        write_json(
            path=root / "workflow.json",
            payload={
                "phase": "long_500",
                "status": "running",
                "long_jobs": len(long.jobs),
                "selection": str(root / "lr-selection.json"),
            },
        )
        dispatch(plan=long)
        if read(long.output_dir / "state.json")["status"] != "completed":
            raise RuntimeError("500-epoch stage has incomplete or failed runs")
        report_results(root=root)
        write_json(
            path=root / "workflow.json",
            payload={
                "phase": "complete",
                "status": "completed",
                "long_jobs": len(long.jobs),
                "report": str(root / "report"),
            },
        )


def preflight(*, root: Path, source: Path) -> None:
    import torch

    root.mkdir(parents=True, exist_ok=False)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    jobs = []
    for variant in (VARIANTS[0], VARIANTS[4]):
        base = read(
            source
            / "runs"
            / f"scratch_warmup_constant_{variant[0]}_sd0"
            / "result.json"
        )["config"]
        for role, epochs, extend_epoch in (
            ("prefix", 2, 0),
            ("continuous", 4, 0),
            ("extended", 4, 2),
        ):
            identifier = f"{variant[0]}_{role}"
            directory = root / "runs" / identifier
            config = configured(
                source=base,
                output=directory,
                epochs=epochs,
                lr=3e-4,
                schedule_epochs=2,
                extend=root
                / "runs"
                / f"{variant[0]}_prefix"
                / "checkpoints"
                / "epoch_002.pt",
                extend_epoch=extend_epoch,
                evaluate_test=False,
            )
            config = config.model_copy(
                update={
                    "run": config.run.model_copy(
                        update={
                            "train_limit": 1024,
                            "eval_limit": 128,
                            "num_workers": 2,
                            "cpu_threads": 2,
                            "deterministic": True,
                        }
                    ),
                    "checkpoint_every": 1,
                    "control": config.control.model_copy(
                        update={
                            "model_milestones": tuple(e for e in (2, 4) if e <= epochs)
                        }
                    ),
                }
            )
            jobs.append(
                job_for(
                    root=root,
                    identifier=identifier,
                    config=config,
                    dependencies=(f"{variant[0]}_prefix",) if extend_epoch else (),
                )
            )
    for variant in VARIANTS[:4]:
        base = read(
            source / "runs" / f"scratch_cosine_{variant[0]}_sd0" / "result.json"
        )["config"]
        identifier = f"cosine_{variant[0]}"
        directory = root / "runs" / identifier
        config = configured(
            source=base,
            output=directory,
            epochs=4,
            lr=1e-3,
            schedule_epochs=4,
            extend=directory / "unused",
            extend_epoch=0,
            evaluate_test=False,
        )
        config = config.model_copy(
            update={
                "run": config.run.model_copy(
                    update={
                        "train_limit": 1024,
                        "eval_limit": 128,
                        "num_workers": 2,
                        "cpu_threads": 2,
                        "deterministic": True,
                    }
                ),
                "checkpoint_every": 1,
            }
        )
        jobs.append(job_for(root=root, identifier=identifier, config=config))
    plan = plan_for(
        root=root,
        manifest=Path(base["run"]["manifest"]),
        jobs=jobs,
        max_gpus=1,
        per_gpu_jobs=2,
        min_free_gib=2,
    )
    dispatch(plan=plan)
    if read(root / "state.json")["status"] != "completed":
        raise RuntimeError("Preflight jobs did not all complete")
    assertions = []
    for variant in (VARIANTS[0], VARIANTS[4]):
        outcomes = {}
        for role in ("continuous", "extended"):
            directory = root / "runs" / f"{variant[0]}_{role}"
            result = read(directory / "result.json")
            saved = torch.load(
                directory / "checkpoints" / "epoch_004.pt",
                map_location="cpu",
                weights_only=True,
            )
            if [h["epoch"] for h in result["history"]] != [1, 2, 3, 4]:
                raise AssertionError("Extended history lost its prefix")
            if len(list((directory / "checkpoints").glob("epoch_*.pt"))) != 1:
                raise AssertionError("Rolling checkpoint retention failed")
            outcomes[role] = (result, saved)
        a, b = outcomes["continuous"], outcomes["extended"]
        if a[0]["final_model_sha256"] != b[0]["final_model_sha256"]:
            raise AssertionError(
                f"Deterministic resume differs from uninterrupted training: {variant[0]}"
            )
        for parameter, state in a[1]["optimizer"]["state"].items():
            for key, expected in state.items():
                actual = b[1]["optimizer"]["state"][parameter][key]
                if isinstance(expected, torch.Tensor) and not torch.equal(
                    expected, actual
                ):
                    raise AssertionError(
                        f"Resumed optimizer differs: {variant[0]}/{key}"
                    )
        assertions.append({
            "variant": variant[0],
            "model_bitwise_equal": True,
            "optimizer_bitwise_equal": True,
            "history_preserved": True,
        })
    write_json(
        path=root / "verification.json",
        payload={"passed": True, "jobs": len(jobs), "resume": assertions},
    )
    print(json.dumps(read(root / "verification.json")))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-study", type=Path)
    parser.add_argument("--create", action="store_true")
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--max-gpus", type=int, default=4)
    parser.add_argument("--per-gpu-jobs", type=int, default=8)
    args = parser.parse_args()
    if args.preflight:
        if args.source_study is None:
            raise ValueError("--source-study is required for preflight")
        preflight(root=args.output_dir, source=args.source_study)
    elif args.create:
        if args.source_study is None:
            raise ValueError("--source-study is required for creation")
        create(
            root=args.output_dir,
            source=args.source_study,
            max_gpus=args.max_gpus,
            per_gpu_jobs=args.per_gpu_jobs,
        )
        print(json.dumps(read(args.output_dir / "study.json"), indent=2))
    else:
        run_study(root=args.output_dir)


if __name__ == "__main__":
    main()
