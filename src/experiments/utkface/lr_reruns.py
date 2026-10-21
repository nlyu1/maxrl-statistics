import argparse
import ast
import hashlib
import json
import os
import subprocess
from collections import Counter
from datetime import datetime
from functools import partial
from pathlib import Path
from zoneinfo import ZoneInfo

from pydantic import Field, model_validator

from src.config.base import BaseConfig
from src.experiments.utkface import scheduler
from src.experiments.utkface.config import RunConfig
from src.experiments.utkface.coordination import (
    CoordinationConfig,
    canonical_coordination,
)
from src.experiments.utkface.run import write_json
from src.experiments.utkface.study import TRAINING_SOURCES, StudyJob, StudyPlan
from src.experiments.utkface.vlm_eval import GPUProcessException
from src.experiments.utkface.vlm_queue import process_matches_exception

PILOT_STAGES = {
    "wide",
    "refine",
    "subset_tuning",
    "flat_tuning",
    "digits_tuning",
    "smoke",
}
ORCHESTRATION_SOURCES = (
    "src/experiments/utkface/lr_reruns.py",
    "src/experiments/utkface/scheduler.py",
    "src/experiments/utkface/vlm_eval.py",
    "src/experiments/utkface/vlm_queue.py",
)
PLACEHOLDER_COMMAND = (
    "import os,time,torch; assert torch.cuda.device_count()==1; "
    'x=torch.empty((1,),device="cuda"); '
    'print("READY",os.getpid(),os.environ["CUDA_VISIBLE_DEVICES"],'
    "torch.cuda.get_device_name(0),flush=True); time.sleep(31536000)"
)


class GPUAdmission(BaseConfig):
    gpu_uuids: tuple[str, ...] = Field(min_length=1, max_length=8)
    ignored_gpu_processes: tuple[GPUProcessException, ...]

    @model_validator(mode="after")
    def validate_scope(self) -> "GPUAdmission":
        if len(set(self.gpu_uuids)) != len(self.gpu_uuids):
            raise ValueError("GPU UUIDs must be unique")
        identities = [(item.gpu_uuid, item.pid) for item in self.ignored_gpu_processes]
        if len(set(identities)) != len(identities) or any(
            uuid not in self.gpu_uuids for uuid, _ in identities
        ):
            raise ValueError("Process exceptions must be unique and on allowed GPUs")
        return self


def shared_learning_rate(*, config: RunConfig) -> float:
    if config.schedule.lr_schedule != "cosine" or config.head == "digits":
        return config.learning_rate
    method = config.loss.method
    if method in {
        "tailrl",
        "tailrl_population",
        "ce",
        "binary_maxrl",
        "binary_maxrl_population",
        "mae",
        "mse",
    } or method.startswith("stat_maxrl_mae"):
        return 1e-3
    if method.startswith("stat_maxrl_mse"):
        return 3e-4 if config.train_limit == 0 else 1e-3
    if method.startswith("grpo_mae"):
        return 1e-3 if config.train_limit == 4096 else 3e-4
    if method.startswith("grpo_mse"):
        return 3e-4
    if method.startswith("rloo_mae"):
        return 1e-4
    if method.startswith("rloo_mse"):
        return 1e-3
    raise ValueError(f"No shared learning rate for {method}")


def source_hashes(*, paths: tuple[str, ...]) -> dict[str, str]:
    return {name: hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in paths}


def build_plan(
    *, source_plans: tuple[Path, ...], root: Path, max_concurrent: int
) -> tuple[StudyPlan, dict]:
    plans = [StudyPlan.model_validate_json(path.read_text()) for path in source_plans]
    if not plans or len({plan.data_fingerprint for plan in plans}) != 1:
        raise ValueError("Source plans must share one cleaned split")
    if any(
        plan.training_source_sha256 != plans[0].training_source_sha256 for plan in plans
    ):
        raise ValueError("Historical training implementations differ")
    jobs, changes, reused = [], [], []
    for path, plan in zip(source_plans, plans, strict=True):
        state = json.loads((plan.output_dir / "state.json").read_text())
        if state["status"] != "completed":
            raise ValueError(f"Historical queue is not complete: {path}")
        for job in plan.jobs:
            record = state["jobs"][job.identifier]
            if record["status"] != "completed" or job.stage in PILOT_STAGES:
                continue
            result_path = Path(record["result_path"])
            result = json.loads(result_path.read_text())
            config = RunConfig.model_validate(result["config"])
            if result["data_fingerprint"] != plan.data_fingerprint:
                raise ValueError("Historical run belongs to another data split")
            target = shared_learning_rate(config=config)
            source = {"identifier": job.identifier, "result_path": str(result_path)}
            if target == config.learning_rate:
                reused.append(source)
                continue
            if (
                config.precision != "bf16"
                or not config.evaluate_test
                or not config.save_checkpoint
            ):
                raise ValueError("Expected complete bfloat16 scientific evaluations")
            identifier = "shared_lr_" + job.identifier
            updated = RunConfig.model_validate({
                **config.model_dump(),
                "learning_rate": target,
                "output_dir": root / "runs" / identifier,
                "device": "cuda:0",
            })
            jobs.append(
                StudyJob.model_validate({
                    **job.model_dump(),
                    "identifier": identifier,
                    "config": updated,
                    "optional": False,
                    "priority": 0
                    if job.stage == "main"
                    else 20
                    if job.stage == "noise"
                    else 10,
                    "dependencies": (),
                    "tune_from": (),
                    "tuning_rank": 0,
                    "gate_candidate": (),
                    "gate_reference": (),
                })
            )
            changes.append({
                **source,
                "new_identifier": identifier,
                "stage": job.stage,
                "variant": job.variant,
                "seed": config.seed,
                "train_limit": config.train_limit,
                "epochs": config.epochs,
                "old_learning_rate": config.learning_rate,
                "new_learning_rate": target,
                "training_fingerprint": result["training_fingerprint"],
                "historical_training_seconds": result["seconds"],
            })
    if len({job.identifier for job in jobs}) != len(jobs):
        raise ValueError("Rerun identifiers are not unique")
    plan = StudyPlan(
        output_dir=root,
        data_fingerprint=plans[0].data_fingerprint,
        training_source_sha256=source_hashes(paths=TRAINING_SOURCES),
        budget_gpu_hours=12,
        max_concurrent=max_concurrent,
        jobs=tuple(jobs),
    )
    return plan, {
        "source_plans": {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in source_plans
        },
        "historical_training_source_sha256": plans[0].training_source_sha256,
        "changes": changes,
        "reused": reused,
        "rerun_count": len(jobs),
        "reused_count": len(reused),
        "stages": dict(Counter(job.stage for job in jobs)),
        "historical_training_gpu_hours": sum(
            item["historical_training_seconds"] for item in changes
        )
        / 3600,
        "orchestration_source_sha256": source_hashes(paths=ORCHESTRATION_SOURCES),
    }


def capture_placeholder(*, pid: int, gpu_uuid: str) -> GPUProcessException:
    process = Path(f"/proc/{pid}")
    command = (process / "cmdline").read_bytes()
    arguments = command.rstrip(b"\0").decode().split("\0")
    if arguments[1:4] != ["-B", "-u", "-c"] or len(arguments) != 5:
        raise ValueError(f"PID {pid} is not the inspected inline placeholder")
    if ast.dump(ast.parse(arguments[4])) != ast.dump(ast.parse(PLACEHOLDER_COMMAND)):
        raise ValueError(f"PID {pid} performs work beyond the inspected placeholder")
    exception = GPUProcessException(
        pid=pid,
        uid=process.stat().st_uid,
        start_time_ticks=int(
            (process / "stat").read_text().rsplit(")", 1)[1].split()[19]
        ),
        command_sha256=hashlib.sha256(command).hexdigest(),
        gpu_uuid=gpu_uuid,
        boot_id=Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
    )
    if not process_matches_exception(exception=exception):
        raise RuntimeError("Placeholder identity changed during capture")
    return exception


def admitted_inventory(*, admission: GPUAdmission) -> list[dict]:
    verified = {
        (item.gpu_uuid, item.pid)
        for item in admission.ignored_gpu_processes
        if process_matches_exception(exception=item)
    }
    return [
        {
            **gpu,
            "pids": [pid for pid in gpu["pids"] if (gpu["uuid"], pid) not in verified],
            "ignored_pids": [
                pid for pid in gpu["pids"] if (gpu["uuid"], pid) in verified
            ],
        }
        for gpu in scheduler.gpu_inventory()
        if gpu["uuid"] in admission.gpu_uuids
    ]


def prepare(*, args: argparse.Namespace) -> None:
    if args.root.exists():
        raise FileExistsError("Refusing to replace an existing rerun ledger")
    inventory = scheduler.gpu_inventory()
    selected = [gpu for gpu in inventory if gpu["index"] in args.gpus]
    if len(selected) != len(args.gpus) or len(set(args.gpus)) != len(args.gpus):
        raise ValueError("Requested GPU indices must exist and be unique")
    exceptions = []
    for pid in args.ignore_pids:
        matches = [gpu for gpu in selected if pid in gpu["pids"]]
        if len(matches) != 1:
            raise ValueError(f"PID {pid} is not on exactly one allowed GPU")
        exceptions.append(capture_placeholder(pid=pid, gpu_uuid=matches[0]["uuid"]))
    admission = GPUAdmission(
        gpu_uuids=tuple(gpu["uuid"] for gpu in selected),
        ignored_gpu_processes=tuple(exceptions),
    )
    plan, audit = build_plan(
        source_plans=tuple(args.source_plans),
        root=args.root,
        max_concurrent=len(selected),
    )
    if audit["rerun_count"] != 59 or audit["reused_count"] != 163:
        raise ValueError(
            "The authorized regime requires exactly 59 reruns and 163 reused evaluations"
        )
    coordination = canonical_coordination(root=args.root).model_copy(
        update={
            "lock_path": args.dispatch_lock,
            "max_concurrent": len(selected),
        }
    )
    audit.update({
        "created_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "source_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "gpu_indices": args.gpus,
    })
    args.root.mkdir(parents=True)
    for filename, payload in (
        ("plan.json", plan.model_dump(mode="json")),
        ("admission.json", admission.model_dump(mode="json")),
        ("coordination.json", coordination.model_dump(mode="json")),
        ("migration.json", audit),
    ):
        write_json(path=args.root / filename, payload=payload)
    print(
        json.dumps(
            {
                key: audit[key]
                for key in (
                    "rerun_count",
                    "reused_count",
                    "stages",
                    "gpu_indices",
                    "historical_training_gpu_hours",
                )
            },
            indent=2,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    setup = commands.add_parser("prepare")
    setup.add_argument("--root", type=Path, required=True)
    setup.add_argument("--source-plans", type=Path, nargs="+", required=True)
    setup.add_argument("--gpus", type=int, nargs="+", required=True)
    setup.add_argument("--ignore-pids", type=int, nargs="*", default=[])
    setup.add_argument("--dispatch-lock", type=Path, required=True)
    for name in ("run", "status"):
        command = commands.add_parser(name)
        command.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args=args)
        return
    plan = StudyPlan.model_validate_json((args.root / "plan.json").read_text())
    coordination = CoordinationConfig.model_validate_json(
        (args.root / "coordination.json").read_text()
    )
    if args.command == "status":
        state = json.loads((args.root / "state.json").read_text())
        print(
            json.dumps(
                scheduler.dispatch_summary(
                    plan=plan, state=state, coordination=coordination
                ),
                indent=2,
            )
        )
        return
    audit = json.loads((args.root / "migration.json").read_text())
    if (
        source_hashes(paths=ORCHESTRATION_SOURCES)
        != audit["orchestration_source_sha256"]
    ):
        raise ValueError("Rerun orchestration differs from the frozen queue")
    admission = GPUAdmission.model_validate_json(
        (args.root / "admission.json").read_text()
    )
    os.environ["UTKFACE_DISPATCH_REVISION"] = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True
    ).strip()
    scheduler.run(
        plan=plan,
        poll_seconds=10,
        coordination=coordination,
        gpu_inventory_source=partial(admitted_inventory, admission=admission),
    )


if __name__ == "__main__":
    main()
