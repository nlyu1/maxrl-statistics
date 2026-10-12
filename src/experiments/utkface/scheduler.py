import argparse
import fcntl
import hashlib
import json
import os
import signal
import subprocess
import time
from collections import Counter
from pathlib import Path

from src.experiments.utkface.config import RunConfig
from src.experiments.utkface.run import write_json
from src.experiments.utkface.study import StudyJob, StudyPlan

TERMINAL = {"completed", "skipped", "failed"}


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


def owned_process(*, attempt: dict) -> bool:
    path = Path(f"/proc/{attempt['pid']}/cmdline")
    if not path.exists():
        return False
    command = path.read_bytes().replace(b"\0", b" ").decode()
    return (
        "src.experiments.utkface.run" in command and attempt["config_path"] in command
    )


def process_group(*, pid: int) -> int:
    try:
        return os.getpgid(pid)
    except ProcessLookupError:
        return -1


def terminate_attempt(*, attempt: dict) -> None:
    if "termination_requested_at" not in attempt and owned_process(attempt=attempt):
        if process_group(pid=attempt["pid"]) != attempt["pid"]:
            raise RuntimeError(
                "Refusing to signal a process outside its dedicated group"
            )
        os.killpg(attempt["pid"], signal.SIGTERM)
        attempt["termination_requested_at"] = time.time()


def charged_seconds(*, state: dict, now: float) -> float:
    return sum(
        attempt.get("seconds", now - attempt["started_at"])
        for record in state["jobs"].values()
        for attempt in record["attempts"]
    )


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


def run(*, plan: StudyPlan, poll_seconds: float) -> None:
    root = plan.output_dir
    root.mkdir(parents=True, exist_ok=True)
    with (root / "supervisor.lock").open("a") as lock:
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
        children = {}
        logs = {}
        streaks = {}
        (root / "configs").mkdir(exist_ok=True)
        (root / "logs").mkdir(exist_ok=True)
        budget = plan.budget_gpu_hours * 3600
        while True:
            now = time.time()
            inventory = gpu_inventory()
            by_uuid = {gpu["uuid"]: gpu for gpu in inventory}
            for job in plan.jobs:
                record = state["jobs"][job.identifier]
                if record["status"] != "running":
                    continue
                attempt = record["attempts"][-1]
                child = children.get(job.identifier)
                returncode = child.poll() if child else None
                if returncode is None and not owned_process(attempt=attempt):
                    returncode = (
                        0
                        if (Path(attempt["output_dir"]) / "result.json").exists()
                        else 1
                    )
                if returncode is not None:
                    attempt.update({
                        "seconds": now - attempt["started_at"],
                        "returncode": returncode,
                        "ended_at": now,
                    })
                    if job.identifier in logs:
                        logs.pop(job.identifier).close()
                    result_path = Path(attempt["output_dir"]) / "result.json"
                    if returncode == 0 and result_path.exists():
                        record.update({
                            "status": "completed",
                            "result_path": str(result_path),
                        })
                    elif (
                        attempt.get("yielded_to_foreign_process")
                        and len(record["attempts"]) < 3
                    ):
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
                    terminate_attempt(attempt=attempt)
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
                    os.killpg(attempt["pid"], signal.SIGKILL)

            active = [
                record
                for record in state["jobs"].values()
                if record["status"] == "running"
            ]
            spent = charged_seconds(state=state, now=now)
            if spent >= budget:
                state["status"] = "budget_exhausted"
                for record in active:
                    terminate_attempt(attempt=record["attempts"][-1])
            if state["status"] in {"failed", "budget_exhausted"}:
                for record in state["jobs"].values():
                    if record["status"] == "pending":
                        record.update({"status": "skipped", "reason": state["status"]})
                if not active:
                    state["updated_at"] = now
                    write_json(path=state_path, payload=state)
                    write_json(
                        path=root / "status.json",
                        payload=status_summary(plan=plan, state=state),
                    )
                    break
            else:
                leased = {record["attempts"][-1]["gpu_uuid"] for record in active}
                for gpu in inventory:
                    streaks[gpu["uuid"]] = (
                        streaks.get(gpu["uuid"], 0) + 1
                        if gpu_is_free(gpu=gpu) and gpu["uuid"] not in leased
                        else 0
                    )
                free = [gpu for gpu in inventory if streaks[gpu["uuid"]] >= 3]
                mandatory_done = all(
                    state["jobs"][job.identifier]["status"] in TERMINAL
                    for job in plan.jobs
                    if not job.optional
                )
                ready = []
                for job in sorted(
                    plan.jobs, key=lambda item: (item.priority, item.identifier)
                ):
                    record = state["jobs"][job.identifier]
                    if record["status"] != "pending" or (
                        job.optional and not mandatory_done
                    ):
                        continue
                    if not all(
                        state["jobs"][dependency]["status"] in TERMINAL
                        for dependency in job.dependencies
                    ):
                        continue
                    if any(
                        state["jobs"][dependency]["status"] == "skipped"
                        for dependency in job.dependencies
                    ) or not gate_passes(job=job, state=state):
                        record.update({
                            "status": "skipped",
                            "reason": "scalar_head_not_competitive",
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
                for job in ready:
                    if not free or len(active) >= plan.max_concurrent:
                        break
                    limit = runtime_limit(job=job, state=state, plan=plan)
                    if closed + reserved + limit > budget:
                        continue
                    gpu = free.pop(0)
                    current = next(
                        item for item in gpu_inventory() if item["uuid"] == gpu["uuid"]
                    )
                    if not gpu_is_free(gpu=current):
                        streaks[gpu["uuid"]] = 0
                        continue
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
                            identifiers=job.tune_from, state=state, rank=job.tuning_rank
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
                    log = (
                        root / "logs" / f"{job.identifier}_attempt{attempt_number}.log"
                    ).open("w")
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
                        "started_at": time.time(),
                        "runtime_limit_seconds": limit,
                        "config_path": str(config_path),
                        "output_dir": str(output),
                    }
                    record["attempts"].append(attempt)
                    record["status"] = "running"
                    children[job.identifier] = child
                    logs[job.identifier] = log
                    active.append(record)
                    reserved += limit
                    streaks[gpu["uuid"]] = 0
                    print(
                        json.dumps({
                            "launched": job.identifier,
                            "gpu": gpu["index"],
                            "learning_rate": configuration.learning_rate,
                        }),
                        flush=True,
                    )
                pending = [
                    record
                    for record in state["jobs"].values()
                    if record["status"] == "pending"
                ]
                if not pending and not active:
                    state["status"] = "completed"
                elif (
                    not active
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
                payload=status_summary(plan=plan, state=state),
            )
            if state["status"] == "completed":
                break
            time.sleep(poll_seconds)
        print(json.dumps(status_summary(plan=plan, state=state), indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=float, default=20)
    parser.add_argument("--status", action="store_true")
    args = parser.parse_args()
    plan = StudyPlan.model_validate_json(args.plan.read_text())
    if args.status:
        state_path = plan.output_dir / "state.json"
        print(
            json.dumps(
                status_summary(plan=plan, state=json.loads(state_path.read_text())),
                indent=2,
            )
            if state_path.exists()
            else "Study has not started"
        )
    else:
        run(plan=plan, poll_seconds=args.poll_seconds)


if __name__ == "__main__":
    main()
