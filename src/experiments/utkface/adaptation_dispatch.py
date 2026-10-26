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
from pathlib import Path

from src.experiments.utkface.adaptation_config import (
    AdaptationConfig,
    AdaptationJob,
    AdaptationPlan,
    FeatureBankConfig,
)
from src.experiments.utkface.run import write_json
from src.experiments.utkface.scheduler import gpu_inventory, gpu_is_free, ready_gpus

TERMINAL = {"completed", "failed", "skipped", "stopped"}
MAX_ATTEMPTS = 3
STARTUP_TIMEOUT_SECONDS = 180.0
TERMINATION_GRACE_SECONDS = 30.0
GPU_COOLDOWN_SECONDS = 900.0
INTERRUPTION_RETURNCODES = {-15, -9, 137, 143}
DISTRIBUTED_ENVIRONMENT = {
    "GROUP_RANK",
    "LOCAL_RANK",
    "LOCAL_WORLD_SIZE",
    "MASTER_ADDR",
    "MASTER_PORT",
    "NODE_RANK",
    "RANK",
    "ROLE_RANK",
    "ROLE_WORLD_SIZE",
    "TORCHELASTIC_MAX_RESTARTS",
    "TORCHELASTIC_RESTART_COUNT",
    "TORCHELASTIC_RUN_ID",
    "WORLD_SIZE",
}
DISTRIBUTED_ENVIRONMENT_PREFIXES = ("NCCL_", "PET_", "TORCHELASTIC_")


class GpuAdmissionRejected(RuntimeError):
    pass


class JobAlreadyComplete(RuntimeError):
    pass


def now() -> float:
    return time.time()


def plan_digest(*, plan: AdaptationPlan) -> str:
    digest = hashlib.sha256()
    digest.update(plan.model_dump_json().encode())
    digest.update(plan.manifest.read_bytes())
    for job in sorted(plan.jobs, key=lambda item: item.identifier):
        digest.update(job.identifier.encode())
        digest.update(job.config_path.read_bytes())
    return digest.hexdigest()


def validate_plan(*, plan: AdaptationPlan) -> None:
    if not plan.jobs:
        raise ValueError("Adaptation plan must contain at least one job")
    if not plan.manifest.is_file():
        raise FileNotFoundError(f"Missing adaptation manifest: {plan.manifest}")
    identifiers = [job.identifier for job in plan.jobs]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("Adaptation job identifiers must be unique")
    if any(
        not identifier
        or identifier in {".", ".."}
        or "/" in identifier
        or "\\" in identifier
        for identifier in identifiers
    ):
        raise ValueError("Adaptation job identifiers must be safe path components")
    known = set(identifiers)
    outputs = [job.output_dir.resolve() for job in plan.jobs]
    if len(outputs) != len(set(outputs)):
        raise ValueError("Adaptation jobs must have distinct output directories")
    if plan.output_dir.resolve() in outputs:
        raise ValueError("A job output directory cannot equal the plan output directory")
    for job in plan.jobs:
        if not job.config_path.is_file():
            raise FileNotFoundError(f"Missing job config: {job.config_path}")
        model = job_configuration(job=job)
        if model.run.manifest.resolve() != plan.manifest.resolve():
            raise ValueError(
                f"Job {job.identifier} config manifest differs from the plan"
            )
        if model.run.output_dir.resolve() != job.output_dir.resolve():
            raise ValueError(
                f"Job {job.identifier} config output_dir differs from the plan"
            )
        if job.identifier in job.dependencies:
            raise ValueError(f"Job {job.identifier} depends on itself")
        missing = set(job.dependencies) - known
        if missing:
            raise ValueError(
                f"Job {job.identifier} has unknown dependencies: {sorted(missing)}"
            )
    visiting, visited = set(), set()

    def visit(identifier: str) -> None:
        if identifier in visiting:
            raise ValueError(f"Adaptation dependency cycle includes {identifier}")
        if identifier in visited:
            return
        visiting.add(identifier)
        job = next(item for item in plan.jobs if item.identifier == identifier)
        for dependency in job.dependencies:
            visit(dependency)
        visiting.remove(identifier)
        visited.add(identifier)

    for identifier in identifiers:
        visit(identifier)


def read_json(*, path: Path) -> object:
    return json.loads(path.read_text())


def job_configuration(*, job: AdaptationJob) -> AdaptationConfig | FeatureBankConfig:
    payload = read_json(path=job.config_path)
    model = AdaptationConfig if job.kind == "train" else FeatureBankConfig
    return model.model_validate(payload)


def complete_artifact(*, job: AdaptationJob) -> Path | None:
    path = job.output_dir / "result.json"
    if not path.is_file():
        return None
    try:
        payload = read_json(path=path)
    except (OSError, ValueError):
        return None
    return path if isinstance(payload, dict) else None


def event(*, root: Path, event_type: str, **fields) -> None:
    record = {"time": now(), "event": event_type, **fields}
    with (root / "supervisor.log").open("a") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def boot_id() -> str:
    return Path("/proc/sys/kernel/random/boot_id").read_text().strip()


BOOT_ID = boot_id()


def process_identity(*, pid: int) -> dict | None:
    process = Path(f"/proc/{pid}")
    try:
        stat = (process / "stat").read_text()
        command = (process / "cmdline").read_bytes()
        uid = process.stat().st_uid
        fields = stat.rsplit(")", 1)[1].split()
        identity = {
            "pid": pid,
            "uid": uid,
            "process_group": int(fields[2]),
            "start_time_ticks": int(fields[19]),
            "command_sha256": hashlib.sha256(command).hexdigest(),
            "boot_id": BOOT_ID,
        }
    except (
        FileNotFoundError,
        IndexError,
        PermissionError,
        ProcessLookupError,
        ValueError,
    ):
        return None
    return identity


def identity_matches(*, identity: dict | None, current: dict | None) -> bool:
    return bool(
        identity
        and current
        and identity["pid"] == current["pid"]
        and identity["uid"] == current["uid"]
        and identity["process_group"] == current["process_group"]
        and identity["start_time_ticks"] == current["start_time_ticks"]
        and identity["command_sha256"] == current["command_sha256"]
        and identity["boot_id"] == current["boot_id"]
    )


def attempt_identity(*, attempt: dict, name: str) -> dict | None:
    identity = attempt.get(name)
    if identity is None:
        return None
    current = process_identity(pid=identity["pid"])
    return (
        identity
        if identity_matches(identity=identity, current=current)
        and identity["uid"] == os.getuid()
        else None
    )


def attempt_group_identity(*, attempt: dict) -> dict | None:
    for name in ("worker_identity", "launcher_identity"):
        identity = attempt_identity(attempt=attempt, name=name)
        if identity is not None:
            return identity
    return None


def owned_processes_alive(*, attempt: dict) -> tuple[bool, bool]:
    return (
        attempt_identity(attempt=attempt, name="launcher_identity") is not None,
        attempt_identity(attempt=attempt, name="worker_identity") is not None,
    )


def terminate_owned_group(
    *, attempt: dict, force: bool, reason: str, root: Path
) -> bool:
    identity = attempt_group_identity(attempt=attempt)
    if identity is None:
        return False
    try:
        if os.getpgid(identity["pid"]) != attempt["process_group"]:
            return False
        os.killpg(
            attempt["process_group"], signal.SIGKILL if force else signal.SIGTERM
        )
    except ProcessLookupError:
        return False
    except PermissionError:
        event(
            root=root,
            event_type="termination_denied",
            identifier=attempt["identifier"],
            pid=identity["pid"],
            reason=reason,
        )
        return False
    event(
        root=root,
        event_type="termination_signal",
        identifier=attempt["identifier"],
        pid=identity["pid"],
        process_group=attempt["process_group"],
        signal=signal.SIGKILL if force else signal.SIGTERM,
        reason=reason,
    )
    return True


def request_termination(
    *, attempt: dict, reason: str, force: bool, root: Path
) -> None:
    attempt.setdefault("termination_requested_at", now())
    attempt.setdefault("termination_reason", reason)
    terminate_owned_group(
        attempt=attempt, force=force, reason=reason, root=root
    )
    if force:
        attempt.setdefault("forced_termination_at", now())


def command_matches_worker(*, pid: int, attempt: dict) -> bool:
    try:
        arguments = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        return False
    module = (
        "src.experiments.utkface.adaptation_train"
        if attempt["kind"] == "train"
        else "src.experiments.utkface.feature_bank"
    )
    return (
        module.encode() in arguments
        and attempt["config_path"].encode() in arguments
    )


def register_worker(*, attempt: dict, root: Path) -> bool:
    if attempt.get("worker_identity") is not None:
        return True
    path = Path(attempt["output_dir"]) / "worker-pid.json"
    try:
        if path.stat().st_mtime_ns <= attempt["started_at_ns"]:
            return False
        payload = read_json(path=path)
    except (FileNotFoundError, OSError, ValueError):
        return False
    if not isinstance(payload, dict) or type(payload.get("pid")) is not int:
        return False
    pid = payload["pid"]
    identity = process_identity(pid=pid)
    if (
        identity is None
        or identity["uid"] != os.getuid()
        or identity["process_group"] != attempt["process_group"]
        or pid == os.getpid()
        or not command_matches_worker(pid=pid, attempt=attempt)
    ):
        return False
    attempt["worker_identity"] = identity
    attempt["worker_registered_at"] = now()
    event(
        root=root,
        event_type="worker_registered",
        identifier=attempt["identifier"],
        launcher_pid=(attempt.get("launcher_identity") or {}).get("pid"),
        worker_pid=pid,
        process_group=attempt["process_group"],
    )
    return True


def recorded_identity(*, record: dict, pid: int) -> dict | None:
    for attempt in reversed(record["attempts"]):
        for name in ("worker_identity", "launcher_identity"):
            identity = attempt.get(name)
            if identity is not None and identity["pid"] == pid:
                return identity
    return None


def prepare_stale_worker_pid(*, record: dict, job: AdaptationJob) -> None:
    path = job.output_dir / "worker-pid.json"
    if not path.exists():
        return
    try:
        payload = read_json(path=path)
    except (OSError, ValueError):
        return
    if not isinstance(payload, dict) or type(payload.get("pid")) is not int:
        return
    pid = payload["pid"]
    current = process_identity(pid=pid)
    prior = recorded_identity(record=record, pid=pid)
    if current is not None:
        if identity_matches(identity=prior, current=current):
            raise RuntimeError(
                f"Previous worker is still running for {job.identifier}: pid {pid}"
            )
        raise RuntimeError(
            f"worker-pid.json refers to a live unowned process for "
            f"{job.identifier}: pid {pid}"
        )
    if prior is not None:
        path.unlink()


def sanitized_environment(*, gpu_uuid: str) -> dict[str, str]:
    environment = {
        name: value
        for name, value in os.environ.items()
        if name not in DISTRIBUTED_ENVIRONMENT
        and not name.startswith(DISTRIBUTED_ENVIRONMENT_PREFIXES)
    }
    environment.update({
        "CUDA_VISIBLE_DEVICES": gpu_uuid,
        "PYTHONUNBUFFERED": "1",
    })
    return environment


def attempt_configuration(
    *, job: AdaptationJob, plan: AdaptationPlan, attempt_number: int
) -> Path:
    model = job_configuration(job=job)
    if model.run.manifest.resolve() != plan.manifest.resolve():
        raise ValueError(
            f"Job {job.identifier} config manifest differs from the plan"
        )
    if model.run.output_dir.resolve() != job.output_dir.resolve():
        raise ValueError(
            f"Job {job.identifier} config output_dir differs from the plan"
        )
    run = model.run.model_copy(update={
        "device": "cuda:0",
        "output_dir": job.output_dir,
    })
    configured = model.model_copy(update={"run": run})
    attempt_directory = (
        plan.output_dir
        / "attempts"
        / job.identifier
        / f"attempt{attempt_number}"
    )
    attempt_directory.mkdir(parents=True, exist_ok=True)
    path = attempt_directory / "config.json"
    write_json(path=path, payload=configured.model_dump(mode="json"))
    return path


def command_for(*, job: AdaptationJob, config_path: Path) -> list[str]:
    module = (
        "src.experiments.utkface.adaptation_train"
        if job.kind == "train"
        else "src.experiments.utkface.feature_bank"
    )
    return [
        "uv",
        "run",
        "--no-sync",
        "python",
        "-m",
        module,
        "--config",
        str(config_path),
    ]


def launch_job(
    *,
    job: AdaptationJob,
    plan: AdaptationPlan,
    gpu: dict,
    state: dict,
    children: dict[str, subprocess.Popen],
    root: Path,
) -> None:
    record = state["jobs"][job.identifier]
    if len(record["attempts"]) >= MAX_ATTEMPTS:
        raise RuntimeError("Attempt limit reached before launch")
    artifact = complete_artifact(job=job)
    if artifact is not None:
        raise JobAlreadyComplete(str(artifact))
    prepare_stale_worker_pid(record=record, job=job)
    job.output_dir.mkdir(parents=True, exist_ok=True)
    attempt_number = len(record["attempts"]) + 1
    config_path = attempt_configuration(
        job=job, plan=plan, attempt_number=attempt_number
    )
    log_path = config_path.parent / "worker.log"
    command = command_for(job=job, config_path=config_path)
    started_at_ns = time.time_ns()
    started_at = started_at_ns / 1_000_000_000
    environment = sanitized_environment(gpu_uuid=gpu["uuid"])
    environment["OMP_NUM_THREADS"] = str(
        job_configuration(job=job).run.cpu_threads
    )
    previous = record["attempts"][-1] if record["attempts"] else None
    with log_path.open("w") as log:
        child = subprocess.Popen(
            command,
            cwd=Path(__file__).resolve().parents[3],
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    launcher = process_identity(pid=child.pid)
    attempt = {
        "attempt": attempt_number,
        "identifier": job.identifier,
        "kind": job.kind,
        "pid": child.pid,
        "launcher_identity": launcher,
        "process_group": child.pid,
        "gpu_index": gpu["index"],
        "gpu_uuid": gpu["uuid"],
        "started_at": started_at,
        "started_at_ns": started_at_ns,
        "config_path": str(config_path),
        "log_path": str(log_path),
        "output_dir": str(job.output_dir),
        "command": command,
        "run_device": "cuda:0",
        "cuda_visible_devices": gpu["uuid"],
    }
    if previous is not None:
        attempt["resume"] = {
            "from_attempt": previous["attempt"],
            "previous_gpu_uuid": previous.get("gpu_uuid"),
            "previous_ended_at": previous.get("ended_at"),
            "previous_reason": previous.get("retry_reason"),
        }
    attempt["cpu_threads"] = environment["OMP_NUM_THREADS"]
    record["attempts"].append(attempt)
    record["status"] = "running"
    record.pop("reason", None)
    children[job.identifier] = child
    try:
        persist_state(state=state, plan=plan)
    except (OSError, TypeError, ValueError):
        if launcher is not None:
            try:
                os.killpg(attempt["process_group"], signal.SIGKILL)
            except ProcessLookupError:
                pass
        raise
    event(
        root=root,
        event_type="launched",
        identifier=job.identifier,
        gpu_index=gpu["index"],
        gpu_uuid=gpu["uuid"],
        pid=child.pid,
    )


def success_state(*, state: dict, plan: AdaptationPlan) -> dict[str, bool]:
    result = {}
    for job in plan.jobs:
        if state["jobs"][job.identifier]["status"] == "completed":
            result[job.identifier] = complete_artifact(job=job) is not None
    return result


def propagate_dependencies(*, state: dict, plan: AdaptationPlan) -> None:
    jobs = {job.identifier: job for job in plan.jobs}
    changed = True
    while changed:
        changed = False
        for identifier, job in jobs.items():
            record = state["jobs"][identifier]
            if record["status"] != "pending":
                continue
            failed = [
                dependency
                for dependency in job.dependencies
                if state["jobs"][dependency]["status"] in {"failed", "skipped"}
            ]
            if failed:
                record.update({
                    "status": "skipped",
                    "reason": f"dependency_not_complete:{failed[0]}",
                })
                changed = True


def ready_jobs(*, state: dict, plan: AdaptationPlan) -> list[AdaptationJob]:
    successes = success_state(state=state, plan=plan)
    candidates = []
    for job in plan.jobs:
        record = state["jobs"][job.identifier]
        if (
            record["status"] == "pending"
            and all(successes.get(dependency, False) for dependency in job.dependencies)
        ):
            candidates.append(job)
    return sorted(candidates, key=lambda job: (job.priority, job.identifier))


def foreign_gpu_pids(
    *, gpu: dict, attempts: list[dict]
) -> list[int]:
    if not attempts or not any(
        attempt.get("worker_identity") is not None for attempt in attempts
    ):
        return []
    owned_groups = {attempt["process_group"] for attempt in attempts}
    foreign = []
    for pid in gpu["pids"]:
        identity = process_identity(pid=pid)
        if identity is None or identity["process_group"] not in owned_groups:
            foreign.append(pid)
    return foreign


def active_state(*, state: dict) -> list[tuple[str, dict]]:
    return [
        (identifier, record)
        for identifier, record in state["jobs"].items()
        if record["status"] == "running"
    ]


def reap_jobs(
    *,
    state: dict,
    plan: AdaptationPlan,
    children: dict[str, subprocess.Popen],
    reusable: set[str],
    root: Path,
) -> None:
    current = now()
    for identifier, record in active_state(state=state):
        job = next(item for item in plan.jobs if item.identifier == identifier)
        attempt = record["attempts"][-1]
        register_worker(attempt=attempt, root=root)
        child = children.get(identifier)
        returncode = child.poll() if child is not None else None
        launcher_alive, worker_alive = owned_processes_alive(attempt=attempt)
        if launcher_alive or worker_alive:
            if (
                attempt.get("worker_identity") is None
                and current - attempt["started_at"] >= STARTUP_TIMEOUT_SECONDS
                and "termination_requested_at" not in attempt
            ):
                attempt["startup_timed_out"] = True
                request_termination(
                    attempt=attempt,
                    reason="worker_registration_timeout",
                    force=False,
                    root=root,
                )
            elif (
                "termination_requested_at" in attempt
                and current - attempt["termination_requested_at"]
                >= TERMINATION_GRACE_SECONDS
                and "forced_termination_at" not in attempt
            ):
                request_termination(
                    attempt=attempt,
                    reason=attempt.get("termination_reason", "grace_expired"),
                    force=True,
                    root=root,
                )
            continue
        if child is not None and returncode is None:
            returncode = child.poll()
            if returncode is None:
                continue
        if child is not None:
            returncode = child.wait()
            children.pop(identifier, None)
        artifact = complete_artifact(job=job)
        attempt["ended_at"] = current
        attempt["seconds"] = current - attempt["started_at"]
        attempt["returncode"] = returncode
        attempt["worker_registered"] = attempt.get("worker_identity") is not None
        if artifact is not None:
            record.update({"status": "completed", "result_path": str(artifact)})
            reusable.add(attempt["gpu_uuid"])
            event(
                root=root,
                event_type="completed",
                identifier=identifier,
                seconds=attempt["seconds"],
            )
        else:
            retry_reason = interruption_reason(
                attempt=attempt,
                returncode=returncode,
                resumed_without_owner=child is None,
            )
            if retry_reason is not None and len(record["attempts"]) < MAX_ATTEMPTS:
                attempt["retry_reason"] = retry_reason
                record.update({
                    "status": "pending",
                    "reason": f"retry_queued:{retry_reason}",
                })
                event(
                    root=root,
                    event_type="retry_queued",
                    identifier=identifier,
                    attempt=attempt["attempt"],
                    reason=retry_reason,
                    gpu_uuid=attempt["gpu_uuid"],
                    returncode=returncode,
                )
            else:
                reason = (
                    f"retry_limit_reached:{retry_reason}"
                    if retry_reason is not None
                    else "worker_exited_without_result"
                )
                record.update({"status": "failed", "reason": reason})
                event(
                    root=root,
                    event_type="failed",
                    identifier=identifier,
                    reason=reason,
                    returncode=returncode,
                )


def interruption_reason(
    *, attempt: dict, returncode: int | None, resumed_without_owner: bool
) -> str | None:
    if attempt.get("yielded_to_foreign_process"):
        return "foreign_gpu_process"
    if attempt.get("startup_timed_out"):
        return "worker_registration_timeout"
    if attempt.get("termination_reason"):
        return attempt["termination_reason"]
    if returncode in INTERRUPTION_RETURNCODES:
        return f"process_signal:{returncode}"
    if resumed_without_owner and returncode is None:
        return "process_disappeared_after_supervisor_restart"
    return None


def monitor_foreign(
    *,
    state: dict,
    inventory: list[dict],
    reusable: set[str],
    root: Path,
) -> None:
    by_uuid = {gpu["uuid"]: gpu for gpu in inventory}
    current = now()
    attempts_by_gpu: dict[str, list[dict]] = {}
    for _, record in active_state(state=state):
        attempt = record["attempts"][-1]
        attempts_by_gpu.setdefault(attempt["gpu_uuid"], []).append(attempt)
    for uuid, attempts in attempts_by_gpu.items():
        gpu = by_uuid.get(uuid)
        if gpu is None:
            continue
        foreign = foreign_gpu_pids(gpu=gpu, attempts=attempts)
        if not foreign or any(
            attempt.get("yielded_to_foreign_process") for attempt in attempts
        ):
            continue
        cooldowns = state.setdefault("gpu_cooldowns", {})
        count = cooldowns.get(uuid, {}).get("count", 0) + 1
        cooldowns[uuid] = {
            "count": count,
            "cooldown_until": current
            + min(3600.0, GPU_COOLDOWN_SECONDS * count),
        }
        reusable.discard(uuid)
        event(
            root=root,
            event_type="foreign_gpu_process",
            gpu_uuid=uuid,
            gpu_index=gpu["index"],
            foreign_pids=foreign,
            cooldown_until=cooldowns[uuid]["cooldown_until"],
        )
        for attempt in attempts:
            attempt["yielded_to_foreign_process"] = True
            attempt["foreign_gpu_pids"] = foreign
            if "termination_requested_at" not in attempt:
                request_termination(
                    attempt=attempt,
                    reason="foreign_gpu_process",
                    force=False,
                    root=root,
                )


def eligible_gpus(
    *,
    inventory: list[dict],
    state: dict,
    plan: AdaptationPlan,
    first_idle: dict[str, float],
    reusable: set[str],
) -> list[dict]:
    current = now()
    active_by_uuid: dict[str, list[dict]] = {}
    for _, record in active_state(state=state):
        attempt = record["attempts"][-1]
        active_by_uuid.setdefault(attempt["gpu_uuid"], []).append(attempt)
    candidates = []
    leased = {
        uuid
        for uuid, attempts in active_by_uuid.items()
        if len(attempts) >= plan.per_gpu_jobs
    }
    interruptions = {
        uuid: {"cooldown_until": value["cooldown_until"]}
        for uuid, value in state.get("gpu_cooldowns", {}).items()
    }
    idle = ready_gpus(
        inventory=inventory,
        leased=leased,
        interruptions=interruptions,
        first_idle=first_idle,
        reusable=reusable,
        now=current,
    )
    for gpu in idle:
        if gpu["uuid"] not in active_by_uuid:
            candidates.append({**gpu, "admission": "idle"})
    for gpu in inventory:
        uuid = gpu["uuid"]
        attempts = active_by_uuid.get(uuid, [])
        if (
            attempts
            and len(attempts) < plan.per_gpu_jobs
            and current
            >= state.get("gpu_cooldowns", {})
            .get(uuid, {})
            .get("cooldown_until", 0)
            and not foreign_gpu_pids(gpu=gpu, attempts=attempts)
        ):
            candidates.append({**gpu, "admission": "owned_slot"})
    return sorted(
        candidates,
        key=lambda gpu: (
            gpu.get("admission") != "owned_slot",
            gpu["uuid"] not in reusable,
            len(active_by_uuid.get(gpu["uuid"], [])),
            gpu["index"],
        ),
    )


def final_gpu_admission(
    *, candidate: dict, state: dict, plan: AdaptationPlan
) -> dict:
    current = next(
        (gpu for gpu in gpu_inventory() if gpu["uuid"] == candidate["uuid"]),
        None,
    )
    if current is None:
        raise GpuAdmissionRejected(
            f"GPU disappeared during admission: {candidate['uuid']}"
        )
    active = [
        record["attempts"][-1]
        for _, record in active_state(state=state)
        if record["attempts"][-1]["gpu_uuid"] == current["uuid"]
    ]
    if active:
        if len(active) >= plan.per_gpu_jobs:
            raise GpuAdmissionRejected(
                f"GPU slot limit reached during admission: {current['uuid']}"
            )
        foreign = foreign_gpu_pids(gpu=current, attempts=active)
        if foreign:
            raise GpuAdmissionRejected(
                f"GPU gained foreign processes during admission: {foreign}"
            )
        cooldown = (
            state.get("gpu_cooldowns", {})
            .get(current["uuid"], {})
            .get("cooldown_until", 0)
        )
        if now() < cooldown:
            raise GpuAdmissionRejected(
                f"GPU is cooling down after preemption: {current['uuid']}"
            )
    elif not gpu_is_free(gpu=current):
        raise GpuAdmissionRejected(
            f"New GPU is no longer idle during admission: {current['uuid']}"
        )
    return {**current, "admission": candidate["admission"]}


def schedule(
    *,
    state: dict,
    plan: AdaptationPlan,
    inventory: list[dict],
    first_idle: dict[str, float],
    reusable: set[str],
    children: dict[str, subprocess.Popen],
    root: Path,
) -> None:
    propagate_dependencies(state=state, plan=plan)
    active_count = len(active_state(state=state))
    if active_count >= plan.max_jobs:
        return
    jobs = ready_jobs(state=state, plan=plan)
    candidates = eligible_gpus(
        inventory=inventory,
        state=state,
        plan=plan,
        first_idle=first_idle,
        reusable=reusable,
    )
    for job in jobs:
        if active_count >= plan.max_jobs or not candidates:
            break
        selected = None
        for gpu in list(candidates):
            active = [
                record["attempts"][-1]
                for _, record in active_state(state=state)
                if record["attempts"][-1]["gpu_uuid"] == gpu["uuid"]
            ]
            if len(active) >= plan.per_gpu_jobs:
                candidates.remove(gpu)
                continue
            if active and foreign_gpu_pids(gpu=gpu, attempts=active):
                candidates.remove(gpu)
                continue
            if not active and not gpu_is_free(gpu=gpu):
                first_idle.pop(gpu["uuid"], None)
                candidates.remove(gpu)
                continue
            try:
                selected = final_gpu_admission(
                    candidate=gpu, state=state, plan=plan
                )
            except GpuAdmissionRejected:
                candidates.remove(gpu)
                if gpu["admission"] == "idle":
                    first_idle.pop(gpu["uuid"], None)
                continue
            break
        if selected is None:
            break
        try:
            launch_job(
                job=job,
                plan=plan,
                gpu=selected,
                state=state,
                children=children,
                root=root,
            )
        except JobAlreadyComplete as error:
            record = state["jobs"][job.identifier]
            record.update({
                "status": "completed",
                "result_path": str(error),
            })
            event(root=root, event_type="completed_existing", identifier=job.identifier)
            continue
        except (OSError, RuntimeError, subprocess.SubprocessError, ValueError) as error:
            record = state["jobs"][job.identifier]
            record.update({"status": "failed", "reason": f"launch_failed:{error}"})
            event(
                root=root,
                event_type="launch_failed",
                identifier=job.identifier,
                error=str(error),
            )
            continue
        active_count = len(active_state(state=state))
        reusable.discard(selected["uuid"])
        first_idle.pop(selected["uuid"], None)


def wait_for_progress(
    *, active: list[tuple[str, dict]], timeout: float
) -> None:
    descriptors = []
    try:
        for _, record in active:
            attempt = record["attempts"][-1]
            pids = {
                identity["pid"]
                for name in ("launcher_identity", "worker_identity")
                if (identity := attempt.get(name)) is not None
            }
            for pid in pids:
                try:
                    descriptors.append(os.pidfd_open(pid))
                except ProcessLookupError:
                    continue
        if descriptors:
            select.select(descriptors, [], [], timeout)
        else:
            time.sleep(timeout)
    finally:
        for descriptor in descriptors:
            os.close(descriptor)


def compact_status(*, state: dict, plan: AdaptationPlan) -> dict:
    current = now()
    counts = Counter(record["status"] for record in state["jobs"].values())
    active = [
        {
            "job": identifier,
            "attempt": record["attempts"][-1]["attempt"],
            "gpu_index": record["attempts"][-1]["gpu_index"],
            "gpu_uuid": record["attempts"][-1]["gpu_uuid"],
            "run_device": "cuda:0",
            "pid": record["attempts"][-1]["pid"],
            "worker_pid": (
                record["attempts"][-1].get("worker_identity") or {}
            ).get("pid"),
            "minutes": (current - record["attempts"][-1]["started_at"]) / 60,
        }
        for identifier, record in active_state(state=state)
    ]
    return {
        "status": state["status"],
        "counts": dict(counts),
        "active": active,
        "active_jobs": len(active),
        "max_jobs": plan.max_jobs,
        "per_gpu_jobs": plan.per_gpu_jobs,
        "attempt_limit": MAX_ATTEMPTS,
        "gpu_cooldowns": state.get("gpu_cooldowns", {}),
        "updated_at": state.get("updated_at"),
    }


def initialize_state(*, plan: AdaptationPlan) -> dict:
    jobs = {}
    for job in plan.jobs:
        artifact = complete_artifact(job=job)
        jobs[job.identifier] = (
            {
                "status": "completed",
                "attempts": [],
                "result_path": str(artifact),
            }
            if artifact is not None
            else {"status": "pending", "attempts": []}
        )
    return {
        "version": 2,
        "status": "running",
        "created_at": now(),
        "updated_at": now(),
        "plan_digest": plan_digest(plan=plan),
        "jobs": jobs,
        "gpu_cooldowns": {},
    }


def persist_state(*, state: dict, plan: AdaptationPlan) -> None:
    if any(
        record["status"] == "running" and not record["attempts"]
        for record in state["jobs"].values()
    ):
        raise ValueError("Running adaptation job has no attempt")
    state["updated_at"] = now()
    write_json(path=plan.output_dir / "state.json", payload=state)
    write_json(
        path=plan.output_dir / "status.json",
        payload=compact_status(state=state, plan=plan),
    )


def load_state(*, plan: AdaptationPlan, root: Path) -> dict:
    path = root / "state.json"
    if not path.exists():
        return initialize_state(plan=plan)
    state = read_json(path=path)
    expected = {job.identifier for job in plan.jobs}
    if not isinstance(state, dict) or set(state.get("jobs", {})) != expected:
        raise ValueError("Saved adaptation state does not match the plan")
    if state.get("version") != 2:
        raise ValueError("Saved adaptation state has an unsupported version")
    if state.get("plan_digest") != plan_digest(plan=plan):
        raise ValueError("Plan or job configs changed after queue initialization")
    jobs = {job.identifier: job for job in plan.jobs}
    for identifier, record in state["jobs"].items():
        if len(record.get("attempts", [])) > MAX_ATTEMPTS:
            raise ValueError(f"Job {identifier} exceeds the retry bound")
        status = record.get("status")
        if status not in TERMINAL | {"pending", "running", "launching"}:
            raise ValueError(f"Job {identifier} has an invalid status: {status}")
        if status == "completed" and complete_artifact(job=jobs[identifier]) is None:
            raise ValueError(f"Job {identifier} lost its result artifact")
        if status in {"running", "launching"} and not record.get("attempts"):
            raise ValueError(f"Job {identifier} claims active state without an attempt")
        if status == "stopped":
            if len(record["attempts"]) < MAX_ATTEMPTS:
                record["status"] = "pending"
            else:
                record.update({
                    "status": "failed",
                    "reason": "retry_limit_reached:queue_stopped",
                })
        if status == "launching":
            record["status"] = "running"
        if (
            record["status"] == "running"
            and record.get("attempts")
            and record["attempts"][-1].get("launcher_identity") is None
        ):
            record["attempts"][-1]["resumed_after_supervisor_restart"] = True
    state.setdefault("gpu_cooldowns", {})
    if state["status"] in {"completed", "failed"}:
        if any(
            record["status"] not in TERMINAL for record in state["jobs"].values()
        ):
            raise ValueError("Terminal adaptation queue contains a nonterminal job")
        if state["status"] == "completed" and any(
            record["status"] != "completed" for record in state["jobs"].values()
        ):
            raise ValueError("Completed adaptation queue contains an unsuccessful job")
        return state
    if state["status"] not in {"running", "stopping", "stopped"}:
        raise ValueError(f"Saved adaptation queue has invalid status: {state['status']}")
    if state["status"] != "running":
        state["status"] = "running"
    return state


def finish_if_terminal(*, state: dict, plan: AdaptationPlan) -> bool:
    statuses = {record["status"] for record in state["jobs"].values()}
    if statuses <= TERMINAL:
        state["status"] = (
            "completed"
            if statuses == {"completed"}
            else "failed"
        )
        return True
    return False


def run(*, plan: AdaptationPlan) -> None:
    if os.name != "posix":
        raise RuntimeError("adaptation_dispatch requires Linux process groups")
    validate_plan(plan=plan)
    root = plan.output_dir
    root.mkdir(parents=True, exist_ok=True)
    (root / "attempts").mkdir(exist_ok=True)
    plan.dispatch_lock.parent.mkdir(parents=True, exist_ok=True)
    with (
        (root / "queue.lock").open("a") as queue_lock,
        plan.dispatch_lock.open("a") as dispatch_lock,
    ):
        fcntl.flock(queue_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        (root / "supervisor.pid").write_text(str(os.getpid()) + "\n")
        state = load_state(plan=plan, root=root)
        if state["status"] in {"completed", "failed"}:
            print(json.dumps(compact_status(state=state, plan=plan), indent=2))
            return
        children: dict[str, subprocess.Popen] = {}
        first_idle: dict[str, float] = {}
        reusable: set[str] = set()
        shutdown = False
        force_shutdown = False

        def interrupt(signum, frame) -> None:
            nonlocal shutdown, force_shutdown
            if shutdown:
                force_shutdown = True
            shutdown = True

        signal.signal(signal.SIGTERM, interrupt)
        signal.signal(signal.SIGINT, interrupt)
        event(root=root, event_type="supervisor_started", pid=os.getpid())
        while True:
            inventory = gpu_inventory()
            monitor_foreign(
                state=state, inventory=inventory, reusable=reusable, root=root
            )
            reap_jobs(
                state=state,
                plan=plan,
                children=children,
                reusable=reusable,
                root=root,
            )
            if shutdown:
                state["status"] = "stopping"
                for _, record in active_state(state=state):
                    attempt = record["attempts"][-1]
                    if "termination_requested_at" not in attempt:
                        request_termination(
                            attempt=attempt,
                            reason="supervisor_shutdown",
                            force=force_shutdown,
                            root=root,
                        )
                    elif force_shutdown and "forced_termination_at" not in attempt:
                        request_termination(
                            attempt=attempt,
                            reason="supervisor_shutdown",
                            force=True,
                            root=root,
                        )
            else:
                fcntl.flock(dispatch_lock, fcntl.LOCK_EX)
                try:
                    inventory = gpu_inventory()
                    schedule(
                        state=state,
                        plan=plan,
                        inventory=inventory,
                        first_idle=first_idle,
                        reusable=reusable,
                        children=children,
                        root=root,
                    )
                finally:
                    fcntl.flock(dispatch_lock, fcntl.LOCK_UN)
            propagate_dependencies(state=state, plan=plan)
            terminal = finish_if_terminal(state=state, plan=plan)
            if terminal:
                persist_state(state=state, plan=plan)
                event(
                    root=root,
                    event_type="supervisor_finished",
                    status=state["status"],
                )
                break
            if shutdown and not active_state(state=state):
                state["status"] = "stopped"
                persist_state(state=state, plan=plan)
                break
            persist_state(state=state, plan=plan)
            wait_for_progress(
                active=active_state(state=state), timeout=plan.poll_seconds
            )
        print(json.dumps(compact_status(state=state, plan=plan), indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--status", action="store_true")
    args = parser.parse_args()
    plan = AdaptationPlan.model_validate_json(args.plan.read_text())
    validate_plan(plan=plan)
    if args.status:
        state_path = plan.output_dir / "state.json"
        if not state_path.exists():
            print(json.dumps({"status": "not_started"}, indent=2))
            return
        print(
            json.dumps(
                compact_status(state=read_json(path=state_path), plan=plan),
                indent=2,
            )
        )
        return
    run(plan=plan)


if __name__ == "__main__":
    main()
