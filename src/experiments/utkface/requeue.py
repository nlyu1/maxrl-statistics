import argparse
import fcntl
import hashlib
import json
import os
import select
import signal
import subprocess
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from src.experiments.utkface.coordination import CoordinationConfig
from src.experiments.utkface.run import write_json
from src.experiments.utkface.scheduler import charged_seconds
from src.experiments.utkface.study import StudyJob, StudyPlan


def continuation(
    *,
    original_path: Path,
    archived_path: Path,
    previous_control: dict,
    output_dir: Path,
) -> tuple[StudyPlan, CoordinationConfig]:
    original = StudyPlan.model_validate_json(original_path.read_text())
    archived = StudyPlan.model_validate_json(archived_path.read_text())
    if original.training_source_sha256 != archived.training_source_sha256:
        raise ValueError("Continuation must preserve the training implementation")
    if original.data_fingerprint != archived.data_fingerprint:
        raise ValueError("Continuation must preserve the cleaned split")
    jobs = tuple(
        StudyJob.model_validate({
            **job.model_dump(),
            "config": {
                **job.config.model_dump(),
                "output_dir": output_dir / "runs" / job.identifier,
            },
        })
        for job in archived.jobs
        if job.stage in {"subset_replication", "matched_steps"}
    )
    if Counter(job.stage for job in jobs) != {
        "subset_replication": 16,
        "matched_steps": 8,
    }:
        raise ValueError(
            "Expected sixteen subset replications and eight matched-update runs"
        )
    plan = StudyPlan(
        output_dir=output_dir,
        data_fingerprint=original.data_fingerprint,
        training_source_sha256=original.training_source_sha256,
        budget_gpu_hours=archived.budget_gpu_hours,
        max_concurrent=8,
        jobs=jobs,
    )
    identifiers = {job.identifier for job in (*original.jobs, *plan.jobs)}
    if len(identifiers) != len(original.jobs) + len(plan.jobs):
        raise ValueError("Continuation duplicates an original queue identifier")
    if any(set(job.dependencies) - identifiers for job in plan.jobs):
        raise ValueError("Continuation has missing prerequisites")
    held = tuple(
        job.identifier for job in original.jobs if job.stage == "noise_replication"
    )
    if len(held) != 40 or any(
        not job.optional for job in original.jobs if job.identifier in held
    ):
        raise ValueError(
            "Only the forty optional noise training-seed repetitions are held"
        )
    control = CoordinationConfig(
        lock_path=Path(previous_control["lock_path"]),
        peer_plans=(original_path, output_dir / "plan.json"),
        max_concurrent=8,
        enforce_gpu_hour_budget=False,
        held_jobs=held,
        deadlines={},
        priorities={
            **previous_control["priorities"],
            **{job.identifier: job.priority for job in jobs},
        },
        promoted=tuple(previous_control["promoted"]),
    )
    return plan, control


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--original-plan", type=Path, required=True)
    parser.add_argument("--archived-extension-plan", type=Path, required=True)
    parser.add_argument("--previous-coordination", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--previous-supervisor-pid", type=int, required=True)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    previous_control = json.loads(args.previous_coordination.read_text())
    plan, control = continuation(
        original_path=args.original_plan,
        archived_path=args.archived_extension_plan,
        previous_control=previous_control,
        output_dir=args.output_dir,
    )
    if args.output_dir.exists():
        raise FileExistsError("Refusing to replace an existing continuation ledger")
    with control.lock_path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        original_root = args.original_plan.parent
        state = json.loads((original_root / "state.json").read_text())
        archived = json.loads(
            (args.archived_extension_plan.parent / "state.json").read_text()
        )
        if archived["status"] != "deadline_reached" or any(
            r["attempts"] for r in archived["jobs"].values()
        ):
            raise ValueError("Expected the untouched, unstarted historical extension")
        for path, digest in plan.training_source_sha256.items():
            if hashlib.sha256(Path(path).read_bytes()).hexdigest() != digest:
                raise ValueError(f"Frozen training source changed: {path}")
        pid = args.previous_supervisor_pid
        if int((original_root / "supervisor.pid").read_text()) != pid:
            raise ValueError("Supervisor ownership changed")
        command = (
            Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode()
        )
        if (
            "src.experiments.utkface.scheduler" not in command
            or str(args.original_plan) not in command
        ):
            raise ValueError("Refusing to stop an unrelated supervisor")
        original = StudyPlan.model_validate_json(args.original_plan.read_text())
        pending = [
            j
            for j in original.jobs
            if state["jobs"][j.identifier]["status"] == "pending"
        ]
        held = sum(j.identifier in control.held_jobs for j in pending)
        audit = {
            "authorized_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
            "source_revision": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip(),
            "original_plan": str(args.original_plan),
            "original_plan_sha256": hashlib.sha256(
                args.original_plan.read_bytes()
            ).hexdigest(),
            "archived_extension_plan": str(args.archived_extension_plan),
            "archived_extension_plan_sha256": hashlib.sha256(
                args.archived_extension_plan.read_bytes()
            ).hexdigest(),
            "original_attempts_preserved": sum(
                len(r["attempts"]) for r in state["jobs"].values()
            ),
            "original_gpu_hours_preserved": charged_seconds(
                state=state, now=time.time()
            )
            / 3600,
            "original_required_pending": sum(not j.optional for j in pending),
            "original_new_setting_optional_pending": sum(
                j.optional and j.identifier not in control.held_jobs for j in pending
            ),
            "held_training_seed_repetitions": held,
            "reinstated_runs": len(plan.jobs),
            "removed_independent_noise_runs": 32,
            "queued_pending": len(pending) - held + len(plan.jobs),
            "previous_supervisor_pid": pid,
            "previous_supervisor_command": command,
            "frozen_training_sources_verified": True,
            "gpu_hour_cap_enforced": False,
            "max_concurrent": 8,
            "applied": args.apply,
        }
        if args.apply:
            args.output_dir.mkdir()
            write_json(
                path=args.output_dir / "original_state_before_update.json",
                payload=state,
            )
            write_json(
                path=args.output_dir / "plan.json", payload=plan.model_dump(mode="json")
            )
            write_json(
                path=args.output_dir / "coordination.json",
                payload=control.model_dump(mode="json"),
            )
            write_json(
                path=args.output_dir / "state.json",
                payload={
                    "status": "running",
                    "jobs": {
                        j.identifier: {"status": "pending", "attempts": []}
                        for j in plan.jobs
                    },
                },
            )
            with os.fdopen(os.pidfd_open(pid), "rb") as descriptor:
                os.kill(pid, signal.SIGTERM)
                if not select.select([descriptor], [], [], 10)[0]:
                    raise RuntimeError(
                        "Owned supervisor did not exit; no replacement was launched"
                    )
            with (original_root / "supervisor.lock").open("a") as supervisor_lock:
                fcntl.flock(supervisor_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if json.loads((original_root / "state.json").read_text()) != state:
                raise RuntimeError("Saved original state changed during the handoff")
            audit["previous_supervisor_exited"] = True
            write_json(path=args.output_dir / "requeue-audit.json", payload=audit)
        print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
