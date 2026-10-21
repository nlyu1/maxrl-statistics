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
from collections.abc import Callable
from pathlib import Path

from src.experiments.utkface.config import RunConfig
from src.experiments.utkface.coordination import (
    CoordinationConfig,
    canonical_coordination,
)
from src.experiments.utkface.run import write_json
from src.experiments.utkface.study import StudyJob, StudyPlan

TERMINAL = {"completed", "skipped", "failed"}
IDLE_CHECK_INTERVAL_SECONDS = 10.0


def gpu_inventory() -> list[dict]:
    query = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,utilization.gpu,memory.used",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    processes = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    occupied = {}
    for row in processes.splitlines():
        uuid, pid = (token.strip() for token in row.split(","))
        occupied.setdefault(uuid, []).append(int(pid))
    result = []
    for row in query.splitlines():
        index, uuid, utilization, memory = (token.strip() for token in row.split(","))
        result.append({
            "index": int(index),
            "uuid": uuid,
            "utilization": int(utilization),
            "memory_mib": int(memory),
            "pids": occupied.get(uuid, []),
        })
    return result


def gpu_is_free(*, gpu: dict) -> bool:
    return not gpu["pids"] and gpu["memory_mib"] <= 1024 and gpu["utilization"] <= 5


def ready_gpus(
    *,
    inventory: list[dict],
    leased: set[str],
    interruptions: dict,
    first_idle: dict[str, float],
    reusable: set[str],
    now: float,
) -> list[dict]:
    ready = []
    for gpu in inventory:
        uuid = gpu["uuid"]
        if (
            not gpu_is_free(gpu=gpu)
            or uuid in leased
            or now < interruptions.get(uuid, {}).get("cooldown_until", 0)
        ):
            first_idle.pop(uuid, None)
            reusable.discard(uuid)
            continue
        if uuid in reusable:
            ready.append(gpu)
        elif uuid not in first_idle:
            first_idle[uuid] = now
        elif now - first_idle[uuid] >= IDLE_CHECK_INTERVAL_SECONDS:
            ready.append(gpu)
    return sorted(ready, key=lambda gpu: (gpu["uuid"] not in reusable, gpu["index"]))


def wait_for_progress(*, active: list[dict], timeout: float) -> None:
    descriptors = []
    try:
        for record in active:
            try:
                descriptors.append(os.pidfd_open(record["attempts"][-1]["pid"]))
            except ProcessLookupError:
                continue
        if descriptors:
            select.select(descriptors, [], [], timeout)
        else:
            time.sleep(timeout)
    finally:
        for descriptor in descriptors:
            os.close(descriptor)


def owned_process(*, attempt: dict) -> bool:
    path = Path(f"/proc/{attempt['pid']}/cmdline")
    try:
        command = path.read_bytes().replace(b"\0", b" ").decode()
    except FileNotFoundError:
        return False
    return (
        "src.experiments.utkface.run" in command and attempt["config_path"] in command
    )


def process_group(*, pid: int) -> int:
    try:
        return os.getpgid(pid)
    except ProcessLookupError:
        return -1
    except PermissionError:
        return -2


def terminate_attempt(*, attempt: dict) -> None:
    if "termination_requested_at" not in attempt and owned_process(attempt=attempt):
        if process_group(pid=attempt["pid"]) != attempt["pid"]:
            raise RuntimeError(
                "Refusing to signal a process outside its dedicated group"
            )
        try:
            os.killpg(attempt["pid"], signal.SIGTERM)
        except ProcessLookupError:
            return
        attempt["termination_requested_at"] = time.time()


def charged_seconds(*, state: dict, now: float) -> float:
    return sum(
        attempt.get("seconds", now - attempt["started_at"])
        for record in state["jobs"].values()
        for attempt in record["attempts"]
    )


def retryable_preemption(*, attempt: dict) -> bool:
    return bool(
        attempt.get("yielded_to_foreign_process")
        and not attempt.get("timed_out")
        and (
            attempt.get("returncode") in {-15, -9, 143, 137}
            or attempt.get("adopted_preemption")
        )
    )


def register_preemption(*, state: dict, attempt: dict, now: float) -> None:
    entry = state["gpu_interruptions"].setdefault(
        attempt["gpu_uuid"], {"count": 0, "cooldown_until": 0}
    )
    entry["count"] += 1
    entry["cooldown_until"] = now + min(3600, 900 * entry["count"])


def restore_preempted_queue(*, state: dict, now: float) -> list[str]:
    if state["status"] != "failed":
        return []
    failed = [
        identifier
        for identifier, record in state["jobs"].items()
        if record["status"] == "failed"
    ]
    if not failed or any(
        not retryable_preemption(attempt=state["jobs"][identifier]["attempts"][-1])
        for identifier in failed
    ):
        return []
    for identifier in failed:
        state["jobs"][identifier]["status"] = "pending"
    for record in state["jobs"].values():
        if record["status"] == "skipped" and record.get("reason") == "failed":
            record["status"] = "pending"
            del record["reason"]
    state["status"] = "running"
    state.setdefault("recoveries", []).append({
        "time": now,
        "reason": "external_gpu_preemption",
        "retried_jobs": failed,
        "gpu_hours_preserved": charged_seconds(state=state, now=now) / 3600,
    })
    return failed


def runtime_limit(*, job: StudyJob, state: dict, plan: StudyPlan) -> float:
    exposures = job.config.train_limit or 18432
    normalized = []
    for candidate in plan.jobs:
        record = state["jobs"][candidate.identifier]
        if record["status"] == "completed" and candidate.stage != "smoke":
            result = json.loads(Path(record["result_path"]).read_text())
            size = result["train_unique_images"]
            normalized.append(
                record["attempts"][-1]["seconds"]
                / (candidate.config.epochs * (size + 2308))
            )
    per_example = (
        sorted(normalized)[int(len(normalized) * 0.8)] if normalized else 0.0016
    )
    estimate = job.config.epochs * (exposures + 2308) * per_example + 30
    return max(180, estimate * 2)


def selected_result(*, identifiers: tuple[str, ...], state: dict, rank: int) -> dict:
    candidates = [
        json.loads(Path(state["jobs"][identifier]["result_path"]).read_text())
        for identifier in identifiers
        if state["jobs"][identifier]["status"] == "completed"
    ]
    candidates.sort(
        key=lambda result: (
            result["validation"]["mae_mean"],
            result["config"]["learning_rate"],
        )
    )
    if rank >= len(candidates):
        raise RuntimeError("Too few successful validation candidates for LR selection")
    return candidates[rank]


def gate_passes(*, job: StudyJob, state: dict) -> bool:
    if not job.gate_candidate:
        return True
    candidate = selected_result(identifiers=job.gate_candidate, state=state, rank=0)
    reference = selected_result(identifiers=job.gate_reference, state=state, rank=0)
    return (
        candidate["validation"]["mae_mean"]
        <= job.competitive_ratio * reference["validation"]["mae_mean"]
    )


def status_summary(*, plan: StudyPlan, state: dict) -> dict:
    now = time.time()
    counts = Counter(record["status"] for record in state["jobs"].values())
    stages = {
        stage: dict(
            Counter(
                state["jobs"][job.identifier]["status"]
                for job in plan.jobs
                if job.stage == stage
            )
        )
        for stage in sorted({job.stage for job in plan.jobs})
    }

    active = [
        {
            "job": identifier,
            "gpu": record["attempts"][-1]["gpu_index"],
            "minutes": (now - record["attempts"][-1]["started_at"]) / 60,
        }
        for identifier, record in state["jobs"].items()
        if record["status"] == "running"
    ]
    return {
        "status": state["status"],
        "gpu_hours": charged_seconds(state=state, now=now) / 3600,
        "budget_gpu_hours": plan.budget_gpu_hours,
        "counts": dict(counts),
        "stages": stages,
        "active": active,
        "updated_at": state.get("updated_at"),
    }


def dispatch_summary(
    *, plan: StudyPlan, state: dict, coordination: CoordinationConfig
) -> dict:
    held = sum(
        record["status"] == "pending" and identifier in coordination.held_jobs
        for identifier, record in state["jobs"].items()
    )
    pending = sum(r["status"] == "pending" for r in state["jobs"].values())
    return {
        **status_summary(plan=plan, state=state),
        "budget_gpu_hours": plan.budget_gpu_hours
        if coordination.enforce_gpu_hour_budget
        else None,
        "planning_gpu_hours": plan.budget_gpu_hours,
        "gpu_hour_cap_enforced": coordination.enforce_gpu_hour_budget,
        "max_concurrent": coordination.max_concurrent,
        "queued_pending": pending - held,
        "held_pending": held,
        "new_gpu_idle_checks": 2,
        "new_gpu_idle_interval_seconds": IDLE_CHECK_INTERVAL_SECONDS,
        "immediate_successful_gpu_reuse": True,
    }


def run(
    *,
    plan: StudyPlan,
    poll_seconds: float,
    coordination: CoordinationConfig,
    gpu_inventory_source: Callable[[], list[dict]],
) -> None:
    root = plan.output_dir
    root.mkdir(parents=True, exist_ok=True)
    coordination.lock_path.parent.mkdir(parents=True, exist_ok=True)
    with (
        (root / "supervisor.lock").open("a") as lock,
        coordination.lock_path.open("a") as dispatch_lock,
    ):
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        (root / "supervisor.pid").write_text(str(os.getpid()) + "\n")
        state_path = root / "state.json"
        state = (
            json.loads(state_path.read_text())
            if state_path.exists()
            else {
                "status": "running",
                "jobs": {
                    job.identifier: {"status": "pending", "attempts": []}
                    for job in plan.jobs
                },
            }
        )
        if set(state["jobs"]) != {job.identifier for job in plan.jobs}:
            raise ValueError("Saved queue state does not match the study plan")
        previous_state = json.loads(json.dumps(state))
        state.setdefault("gpu_interruptions", {})
        historical_interruptions = {}
        for record in state["jobs"].values():
            for attempt in record["attempts"]:
                if attempt.get("yielded_to_foreign_process"):
                    historical_interruptions.setdefault(attempt["gpu_uuid"], []).append(
                        attempt
                    )
        for uuid, attempts in historical_interruptions.items():
            if uuid not in state["gpu_interruptions"]:
                last = max(attempt["termination_requested_at"] for attempt in attempts)
                state["gpu_interruptions"][uuid] = {
                    "count": len(attempts),
                    "cooldown_until": last + min(3600, 900 * len(attempts)),
                }
        recovered = restore_preempted_queue(state=state, now=time.time())
        if recovered:
            write_json(
                path=root / f"state_before_preemption_recovery_{int(time.time())}.json",
                payload=previous_state,
            )
            print(json.dumps({"recovered_preempted_jobs": recovered}), flush=True)
        children = {}
        logs = {}
        first_idle = {}
        (root / "configs").mkdir(exist_ok=True)
        (root / "logs").mkdir(exist_ok=True)
        budget = plan.budget_gpu_hours * 3600
        while True:
            fcntl.flock(dispatch_lock, fcntl.LOCK_EX)
            now = time.time()
            peers = []
            for peer_path in coordination.peer_plans:
                peer = StudyPlan.model_validate_json(peer_path.read_text())
                peer_state_path = peer.output_dir / "state.json"
                if peer.output_dir == root or not peer_state_path.exists():
                    continue
                peers.append((peer, json.loads(peer_state_path.read_text())))
            inventory = gpu_inventory_source()
            by_uuid = {gpu["uuid"]: gpu for gpu in inventory}
            reusable = set()
            for job in plan.jobs:
                record = state["jobs"][job.identifier]
                if record["status"] != "running":
                    continue
                attempt = record["attempts"][-1]
                child = children.get(job.identifier)
                returncode = child.poll() if child else None
                if returncode is None and not owned_process(attempt=attempt):
                    if (
                        attempt.get("yielded_to_foreign_process")
                        and attempt.get("termination_requested_at")
                        and not attempt.get("timed_out")
                        and not (Path(attempt["output_dir"]) / "result.json").exists()
                    ):
                        attempt["adopted_preemption"] = True
                        attempt["exit_status_inferred"] = True
                    returncode = (
                        0
                        if (Path(attempt["output_dir"]) / "result.json").exists()
                        else 1
                    )
                if returncode is not None:
                    result_path = Path(attempt["output_dir"]) / "result.json"
                    ended_at = (
                        min(
                            now, max(attempt["started_at"], result_path.stat().st_mtime)
                        )
                        if child is None and result_path.exists()
                        else now
                    )
                    attempt.update({
                        "seconds": ended_at - attempt["started_at"],
                        "returncode": returncode,
                        "ended_at": ended_at,
                    })
                    if job.identifier in logs:
                        logs.pop(job.identifier).close()
                    if returncode == 0 and result_path.exists():
                        record.update({
                            "status": "completed",
                            "result_path": str(result_path),
                        })
                        if (
                            "gpu_uuid" in attempt
                            and "termination_requested_at" not in attempt
                            and not attempt.get("yielded_to_foreign_process")
                        ):
                            reusable.add(attempt["gpu_uuid"])
                    elif attempt.get("deadline_reached"):
                        record.update({
                            "status": "skipped",
                            "reason": "deadline_reached",
                        })
                    elif retryable_preemption(attempt=attempt):
                        record["status"] = "pending"
                    else:
                        record["status"] = "failed"
                        state["status"] = "failed"
                    continue
                gpu = by_uuid[attempt["gpu_uuid"]]
                foreign = [
                    pid
                    for pid in gpu["pids"]
                    if process_group(pid=pid) not in {-1, attempt["pid"]}
                ]
                if foreign and "termination_requested_at" not in attempt:
                    attempt["yielded_to_foreign_process"] = True
                    attempt["foreign_gpu_pids"] = foreign
                    register_preemption(state=state, attempt=attempt, now=now)
                    terminate_attempt(attempt=attempt)
                    print(
                        json.dumps({
                            "yielded": job.identifier,
                            "gpu": attempt["gpu_index"],
                            "foreign_gpu_pids": foreign,
                            "cooldown_until": state["gpu_interruptions"][
                                attempt["gpu_uuid"]
                            ]["cooldown_until"],
                        }),
                        flush=True,
                    )
                elif (
                    now - attempt["started_at"] >= attempt["runtime_limit_seconds"]
                    and "termination_requested_at" not in attempt
                ):
                    attempt["timed_out"] = True
                    terminate_attempt(attempt=attempt)
                elif (
                    "termination_requested_at" in attempt
                    and now - attempt["termination_requested_at"] > 30
                    and owned_process(attempt=attempt)
                ):
                    try:
                        os.killpg(attempt["pid"], signal.SIGKILL)
                    except ProcessLookupError:
                        pass

            active = [
                record
                for record in state["jobs"].values()
                if record["status"] == "running"
            ]
            spent = charged_seconds(state=state, now=now)
            deadline = coordination.deadlines.get(str(root), 0)
            if deadline and now >= deadline:
                state["status"] = "deadline_reached"
                for record in active:
                    record["attempts"][-1]["deadline_reached"] = True
                    terminate_attempt(attempt=record["attempts"][-1])
            if coordination.enforce_gpu_hour_budget and spent >= budget:
                state["status"] = "budget_exhausted"
                for record in active:
                    terminate_attempt(attempt=record["attempts"][-1])
            if state["status"] in {"failed", "budget_exhausted", "deadline_reached"}:
                for record in state["jobs"].values():
                    if record["status"] == "pending":
                        record.update({"status": "skipped", "reason": state["status"]})
                if not active:
                    state["updated_at"] = now
                    write_json(path=state_path, payload=state)
                    write_json(
                        path=root / "status.json",
                        payload=dispatch_summary(
                            plan=plan, state=state, coordination=coordination
                        ),
                    )
                    break
            else:
                peer_active = [
                    record
                    for _, peer_state in peers
                    for record in peer_state["jobs"].values()
                    if record["status"] == "running"
                ]
                combined = {"jobs": dict(state["jobs"])}
                for _, peer_state in peers:
                    if set(combined["jobs"]) & set(peer_state["jobs"]):
                        raise ValueError(
                            "Coordinated queues have duplicate job identifiers"
                        )
                    combined["jobs"].update(peer_state["jobs"])
                leased = {
                    record["attempts"][-1]["gpu_uuid"]
                    for record in (*active, *peer_active)
                }
                free = ready_gpus(
                    inventory=inventory,
                    leased=leased,
                    interruptions=state["gpu_interruptions"],
                    first_idle=first_idle,
                    reusable=reusable,
                    now=now,
                )
                mandatory_done = all(
                    state["jobs"][job.identifier]["status"] in TERMINAL
                    for job in plan.jobs
                    if not job.optional
                )
                ready = []
                for job in sorted(
                    plan.jobs,
                    key=lambda item: (
                        coordination.priorities.get(item.identifier, item.priority),
                        item.identifier,
                    ),
                ):
                    record = state["jobs"][job.identifier]
                    if (
                        job.identifier in coordination.held_jobs
                        or record["status"] != "pending"
                        or (
                            job.optional
                            and not mandatory_done
                            and job.identifier not in coordination.promoted
                        )
                    ):
                        continue
                    if not all(
                        combined["jobs"][dependency]["status"] in TERMINAL
                        for dependency in job.dependencies
                    ):
                        continue
                    dependency_skipped = any(
                        combined["jobs"][dependency]["status"] == "skipped"
                        for dependency in job.dependencies
                    )
                    if dependency_skipped or not gate_passes(job=job, state=combined):
                        record.update({
                            "status": "skipped",
                            "reason": "dependency_skipped"
                            if dependency_skipped
                            else "scalar_head_not_competitive",
                        })
                        continue
                    ready.append(job)
                closed = sum(
                    attempt["seconds"]
                    for record in state["jobs"].values()
                    for attempt in record["attempts"]
                    if "seconds" in attempt
                )
                reserved = sum(
                    record["attempts"][-1]["runtime_limit_seconds"] for record in active
                )
                peer_ready = any(
                    peer_state["status"] == "running"
                    and (
                        not coordination.deadlines.get(str(peer.output_dir), 0)
                        or now + 180 < coordination.deadlines[str(peer.output_dir)]
                    )
                    and record["status"] == "pending"
                    and item.identifier not in coordination.held_jobs
                    and not item.optional
                    and all(
                        combined["jobs"][dependency]["status"] == "completed"
                        for dependency in item.dependencies
                    )
                    for peer, peer_state in peers
                    for item in peer.jobs
                    for record in (peer_state["jobs"][item.identifier],)
                )
                turn_path = coordination.lock_path.with_suffix(".turn")
                previous_owner = turn_path.read_text() if turn_path.exists() else ""
                for job in ready:
                    if (
                        not free
                        or len(active) + len(peer_active) >= coordination.max_concurrent
                        or (
                            peer_ready
                            and (
                                len(active) > len(peer_active)
                                or (
                                    len(active) == len(peer_active)
                                    and previous_owner == str(root)
                                )
                            )
                        )
                    ):
                        break
                    limit = runtime_limit(job=job, state=state, plan=plan)
                    if deadline and now + limit > deadline:
                        continue
                    if (
                        coordination.enforce_gpu_hour_budget
                        and closed + reserved + limit > budget
                    ):
                        continue
                    gpu = free.pop(0)
                    if any(
                        hashlib.sha256(Path(name).read_bytes()).hexdigest() != digest
                        for name, digest in plan.training_source_sha256.items()
                    ):
                        raise RuntimeError(
                            "Training source changed after the plan was frozen"
                        )
                    record = state["jobs"][job.identifier]
                    attempt_number = len(record["attempts"])
                    output = (
                        root / "runs" / job.identifier / f"attempt_{attempt_number}"
                    )
                    overrides = {"output_dir": output, "device": "cuda:0"}
                    if job.tune_from:
                        best = selected_result(
                            identifiers=job.tune_from,
                            state=combined,
                            rank=job.tuning_rank,
                        )
                        overrides["learning_rate"] = best["config"]["learning_rate"]
                    configuration = RunConfig.model_validate({
                        **job.config.model_dump(),
                        **overrides,
                    })
                    config_path = (
                        root
                        / "configs"
                        / f"{job.identifier}_attempt{attempt_number}.json"
                    )
                    write_json(
                        path=config_path, payload=configuration.model_dump(mode="json")
                    )
                    environment = {
                        **os.environ,
                        "CUDA_VISIBLE_DEVICES": gpu["uuid"],
                        "OMP_NUM_THREADS": str(configuration.cpu_threads),
                    }
                    command = [
                        "uv",
                        "run",
                        "--python",
                        "3.12",
                        "python",
                        "-m",
                        "src.experiments.utkface.run",
                        "--config",
                        str(config_path),
                        "--manifest",
                        str(configuration.manifest),
                        "--output-dir",
                        str(output),
                    ]
                    current = next(
                        item
                        for item in gpu_inventory_source()
                        if item["uuid"] == gpu["uuid"]
                    )
                    if not gpu_is_free(gpu=current):
                        first_idle.pop(gpu["uuid"], None)
                        continue
                    log = (
                        root / "logs" / f"{job.identifier}_attempt{attempt_number}.log"
                    ).open("w")
                    child = subprocess.Popen(
                        command,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        env=environment,
                        start_new_session=True,
                    )
                    attempt = {
                        "pid": child.pid,
                        "gpu_index": gpu["index"],
                        "gpu_uuid": gpu["uuid"],
                        "ignored_gpu_processes": current.get("ignored_pids", []),
                        "started_at": time.time(),
                        "runtime_limit_seconds": limit,
                        "config_path": str(config_path),
                        "output_dir": str(output),
                        "gpu_admission": "reuse"
                        if gpu["uuid"] in reusable
                        else "two_idle_checks",
                        "idle_observed_seconds": time.time() - first_idle[gpu["uuid"]]
                        if gpu["uuid"] in first_idle
                        else 0,
                        "dispatch_source_revision": os.environ.get(
                            "UTKFACE_DISPATCH_REVISION", ""
                        ),
                    }
                    record["attempts"].append(attempt)
                    record["status"] = "running"
                    children[job.identifier] = child
                    logs[job.identifier] = log
                    active.append(record)
                    turn_path.write_text(str(root))
                    previous_owner = str(root)
                    reserved += limit
                    first_idle.pop(gpu["uuid"], None)
                    print(
                        json.dumps({
                            "launched": job.identifier,
                            "gpu": gpu["index"],
                            "learning_rate": configuration.learning_rate,
                            "gpu_admission": "reuse"
                            if gpu["uuid"] in reusable
                            else "two_idle_checks",
                            "previous_completion_gap_seconds": min(
                                (
                                    attempt["started_at"] - earlier["ended_at"]
                                    for item in state["jobs"].values()
                                    for earlier in item["attempts"]
                                    if earlier.get("gpu_uuid") == gpu["uuid"]
                                    and "ended_at" in earlier
                                ),
                                default=None,
                            ),
                        }),
                        flush=True,
                    )
                pending = [
                    record
                    for identifier, record in state["jobs"].items()
                    if record["status"] == "pending"
                    and identifier not in coordination.held_jobs
                ]
                if not pending and not active:
                    state["status"] = "completed"
                elif (
                    not active
                    and coordination.enforce_gpu_hour_budget
                    and ready
                    and all(
                        closed + runtime_limit(job=job, state=state, plan=plan) > budget
                        for job in ready
                    )
                ):
                    state["status"] = "budget_exhausted"
            state["updated_at"] = time.time()
            write_json(path=state_path, payload=state)
            write_json(
                path=root / "status.json",
                payload=dispatch_summary(
                    plan=plan, state=state, coordination=coordination
                ),
            )
            if state["status"] == "completed":
                break
            fcntl.flock(dispatch_lock, fcntl.LOCK_UN)
            wait_for_progress(active=active, timeout=poll_seconds)
        print(
            json.dumps(
                dispatch_summary(plan=plan, state=state, coordination=coordination),
                indent=2,
            ),
            flush=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=float, default=10)
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--coordination", type=Path)
    args = parser.parse_args()
    plan = StudyPlan.model_validate_json(args.plan.read_text())
    coordination = (
        CoordinationConfig.model_validate_json(args.coordination.read_text())
        if args.coordination
        else canonical_coordination(root=plan.output_dir)
    )
    if args.status:
        state_path = plan.output_dir / "state.json"
        print(
            json.dumps(
                dispatch_summary(
                    plan=plan,
                    state=json.loads(state_path.read_text()),
                    coordination=coordination,
                ),
                indent=2,
            )
            if state_path.exists()
            else "Study has not started"
        )
    else:
        os.environ["UTKFACE_DISPATCH_REVISION"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip()
        run(
            plan=plan,
            poll_seconds=args.poll_seconds,
            coordination=coordination,
            gpu_inventory_source=gpu_inventory,
        )


if __name__ == "__main__":
    main()
