import argparse
import fcntl
import hashlib
import json
import os
import shutil
import signal
import subprocess
import time
from pathlib import Path

from src.experiments.utkface.adaptation_dispatch import BOOT_ID, sanitized_environment
from src.experiments.utkface.adaptation_train import json_sha256
from src.experiments.utkface.scheduler import gpu_inventory

PYTHON = ("uv", "run", "--no-sync", "python")


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def write(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(f".tmp.{os.getpid()}")
    with temporary.open("w") as stream:
        stream.write(json.dumps(payload, indent=2, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def process(pid: int) -> dict:
    directory = Path(f"/proc/{pid}")
    try:
        fields = (directory / "stat").read_text().rsplit(")", 1)[1].split()
        return {
            "group": int(fields[2]),
            "start": fields[19],
            "state": fields[0],
            "uid": directory.stat().st_uid,
            "argv": (directory / "cmdline").read_bytes().decode().split("\0"),
        }
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        return {}


def alive(worker: dict) -> bool:
    current = process(worker["pid"])
    return bool(
        current
        and current["state"] != "Z"
        and current["uid"] == os.getuid()
        and current["start"] == worker["start"]
        and current["group"] == worker["pid"]
        and worker["boot"] == BOOT_ID
        and all(token in current["argv"] for token in worker["tokens"])
    )


def inventory() -> dict:
    return {gpu["uuid"]: gpu for gpu in gpu_inventory()}


def ready(*, config: dict, first_idle: dict, devices: dict) -> list[str]:
    available = []
    for uuid, gpu in devices.items():
        idle = (
            not gpu["pids"]
            and gpu["memory_mib"] <= config["idle_memory_mib"]
            and gpu["utilization"] <= config["idle_utilization"]
        )
        if not idle:
            first_idle.pop(uuid, None)
        elif time.monotonic() - first_idle.setdefault(uuid, time.monotonic()) >= 10:
            available.append(uuid)
    return available


def fit_complete(job: dict, *, commit: str) -> bool:
    result = Path(job["output"]) / "result.json"
    if not result.is_file():
        return False
    payload = read(result)
    expected = read(Path(job["config"]))
    provenance = payload["provenance"]
    fingerprint = json_sha256({
        key: value for key, value in provenance.items() if key != "fingerprint"
    })
    if (
        payload.get("status") != "complete"
        or payload.get("config") != expected
        or provenance["mechanism_config"] != expected
        or provenance["fingerprint"] != fingerprint
        or provenance["environment"]["git_commit"] != commit
    ):
        raise RuntimeError(f"Invalid or unsuccessful fit: {result}")
    return True


def study_status(*, root: Path) -> dict:
    plan = read(root / "plan.json")
    state = read(root / "state.json") if (root / "state.json").exists() else {}
    if state.get("failures") or state.get("status") == "failed":
        raise RuntimeError(f"Study contains failed fits: {root}")
    configs = {job["config"] for job in plan["jobs"]}
    active = [
        int(path.name)
        for path in Path("/proc").iterdir()
        if path.name.isdigit()
        and (current := process(int(path.name)))
        and current["state"] != "Z"
        and "src.experiments.utkface.mechanism_train" in current["argv"]
        and configs.intersection(current["argv"])
    ]
    completed = sum(
        fit_complete(job, commit=plan["git_commit"]) for job in plan["jobs"]
    )
    return {
        "completed": completed,
        "total": len(plan["jobs"]),
        "active": active,
        "complete": completed == len(plan["jobs"])
        and state.get("status") == "complete"
        and not active
        and not state.get("active")
        and state.get("pending") == 0,
    }


def pinned(stage: dict) -> str:
    checkout = stage["checkout"]
    revision = subprocess.check_output(
        ["git", "-C", checkout, "rev-parse", "HEAD"], text=True
    ).strip()
    if stage["commit"] != "mutable":
        dirty = subprocess.check_output(
            ["git", "-C", checkout, "status", "--porcelain"], text=True
        ).strip()
        if revision != stage["commit"] or dirty:
            raise ValueError(
                f"Pinned checkout differs from its clean source: {checkout}"
            )
    return revision


def probe_complete(*, stage: dict, variants: list[str], plan: dict, key: str) -> bool:
    directory = Path(stage["output_dir"]) / f"g{key}"
    report = (
        read(directory / "report.json") if (directory / "report.json").exists() else {}
    )
    requested = {
        "root": str(Path(stage["root"]).resolve()),
        "plan_sha256": sha256(Path(stage["root"]) / "plan.json"),
        "variants": variants,
        "checkpoints": stage["checkpoints"],
    }
    request = report.get("requested", {})
    jobs = {
        job["identifier"]: job for job in plan["jobs"] if job["variant"] in variants
    }
    expected = {
        (identifier, checkpoint)
        for identifier in jobs
        for checkpoint in stage["checkpoints"]
    }
    actual = {
        (row["source"]["identifier"], row["source"]["checkpoint"])
        for row in report.get("runs", [])
    }
    if (
        actual != expected
        or len(report.get("runs", [])) != len(expected)
        or report.get("git_commit") != stage["commit"]
        or any(request.get(key) != value for key, value in requested.items())
    ):
        return False
    sources = request["source_sha256"]
    if not sources or any(
        sha256(Path(stage["checkout"]) / name) != digest
        for name, digest in sources.items()
    ):
        raise ValueError(f"Probe report source changed: {directory}")
    for row in report["runs"]:
        source = row["source"]
        files = {
            source[f"{name}_path"]: source[f"{name}_sha256"]
            for name in ("config", "checkpoint")
        }
        files[str(Path(jobs[source["identifier"]]["output"]) / "result.json")] = source[
            "result_sha256"
        ]
        if any(sha256(Path(path)) != digest for path, digest in files.items()):
            raise ValueError(f"Probe source artifacts changed: {directory}")
    return True


def load(*, path: Path) -> dict:
    config = read(path)
    if config["schema"] != "utkface-mechanism-workflow-v1" or not config["stages"]:
        raise ValueError("Unsupported or empty workflow")
    if not 1 <= config["max_gpus"] <= 4 or not 1 <= config["jobs_per_gpu"] <= 8:
        raise ValueError("Use at most four GPUs and eight jobs per GPU")
    if (
        not 1 <= config["poll_seconds"] <= 60
        or min(config["idle_memory_mib"], config["idle_utilization"]) < 0
    ):
        raise ValueError("Invalid polling or idle thresholds")
    names = [stage["name"] for stage in config["stages"]]
    if len(names) != len(set(names)) or any(
        name in {"", ".", ".."} or Path(name).name != name for name in names
    ):
        raise ValueError("Stage names must be distinct path components")
    for stage in config["stages"]:
        if stage["kind"] not in {"study", "probes", "report", "command"}:
            raise ValueError(f"Unsupported stage: {stage}")
        commit = stage["commit"]
        if not (len(commit) == 40 and set(commit) <= set("0123456789abcdef")) and (
            commit != "mutable" or stage["kind"] in {"study", "probes"}
        ):
            raise ValueError("Learning/probe stages require a full pinned commit")
        if stage["kind"] == "study" and type(stage["launch"]) is not bool:
            raise ValueError("Study launch must be a boolean")
        if stage["kind"] == "probes" and (
            not stage["checkpoints"]
            or len(set(stage["checkpoints"])) != len(stage["checkpoints"])
        ):
            raise ValueError("Probe checkpoints must be nonempty and distinct")
    artifact_root = Path(config["artifact_root"]).resolve()
    directories = [
        config["output_dir"],
        *(
            s[key]
            for s in config["stages"]
            for key in ("root", "output_dir")
            if key in s
        ),
        *(item for s in config["stages"] for item in s.get("artifacts", [])),
    ]
    if any(
        not Path(item).is_absolute()
        or not Path(item).resolve().is_relative_to(artifact_root)
        for item in directories
    ):
        raise ValueError("Artifact path is outside artifact_root")
    if (
        not Path(config["environment"]["UV_PROJECT_ENVIRONMENT"]).is_absolute()
        or not Path(config["dispatch_lock"]).is_absolute()
    ):
        raise ValueError("Environment and dispatch lock paths must be absolute")
    return config


def status_summary(*, config: Path) -> dict:
    settings = load(path=config)
    path = Path(settings["output_dir"]) / "workflow.json"
    return {
        "workflow": read(path) if path.exists() else {},
        "studies": {
            stage["name"]: study_status(root=Path(stage["root"]))
            for stage in settings["stages"]
            if stage["kind"] == "study"
        },
    }


def run(*, config_path: Path) -> None:
    config = load(path=config_path)
    output = Path(config["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    with (output / "supervisor.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)

        def cancel(signum, frame):
            raise KeyboardInterrupt(f"Received signal {signum}")

        previous = signal.signal(signal.SIGTERM, cancel)
        try:
            execute(config=config, output=output)
        finally:
            signal.signal(signal.SIGTERM, previous)


def execute(*, config: dict, output: Path) -> None:
    digest = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    path = output / "workflow.json"
    state = (
        read(path)
        if path.exists()
        else {"config_sha256": digest, "config": config, "stages": {}}
    )
    if state["config_sha256"] != digest:
        raise ValueError("Workflow configuration changed; use a new output_dir")
    write(output / "config.json", config)
    children = {}

    def event(name: str, **fields) -> None:
        state.update(time=time.time(), status=name)
        write(path, state)
        with (output / "events.jsonl").open("a") as events:
            events.write(
                json.dumps({"time": time.time(), "event": name, **fields}) + "\n"
            )
        print(json.dumps({"event": name, **fields}), flush=True)

    def launch(stage: dict, argv: list[str], suffix: str, gpu: str = "") -> dict:
        revision = pinned(stage)
        free_bytes = {
            path: shutil.disk_usage(path).free
            for path in (config["artifact_root"], "/tmp")
        }
        if not all(free_bytes.values()):
            raise OSError(f"Artifact or temporary filesystem is full: {free_bytes}")
        environment = sanitized_environment(gpu_uuid=gpu)
        environment.update(config["environment"])
        environment.update(
            CUDA_VISIBLE_DEVICES=gpu, OMP_NUM_THREADS="1", PYTHONUNBUFFERED="1"
        )
        started_at = time.time()
        with (output / f"{stage['name']}-{suffix}.log").open("a") as log:
            child = subprocess.Popen(
                argv,
                cwd=stage["checkout"],
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        children[child.pid] = child
        identity = process(child.pid)
        if not identity:
            raise RuntimeError(f"Worker exited during launch: {argv}")
        return {
            "pid": child.pid,
            "start": identity["start"],
            "tokens": argv[4:],
            "started_at": started_at,
            "boot": BOOT_ID,
            "gpu": gpu,
            "revision": revision,
            "argv": argv,
            "disk_free_bytes": free_bytes,
        }

    def command(module: str, arguments: list[str]) -> list[str]:
        return [*PYTHON, "-u", "-m", f"src.experiments.utkface.{module}", *arguments]

    def exited(worker: dict) -> None:
        child = children.pop(worker["pid"], None)
        if child is not None:
            worker["returncode"] = child.wait()
        if worker.get("returncode", 0) != 0:
            raise RuntimeError(f"Worker failed: {worker['argv']}")

    def terminate(worker: dict, since: float) -> None:
        if alive(worker):
            try:
                os.killpg(
                    worker["pid"],
                    signal.SIGKILL if time.time() - since >= 30 else signal.SIGTERM,
                )
            except ProcessLookupError:
                pass

    def gpu_stage(stage: dict, record: dict, plan: dict) -> None:
        training = stage["kind"] == "study"
        variants = list(dict.fromkeys(job["variant"] for job in plan["jobs"]))
        groups = stage.get("groups") or [
            variants[index :: config["max_gpus"]] for index in range(config["max_gpus"])
        ]
        groups = [group for group in groups if group]
        flattened = [variant for group in groups for variant in group]
        if not training and (
            len(groups) > config["max_gpus"]
            or len(flattened) != len(set(flattened))
            or set(flattened) != set(variants)
        ):
            raise ValueError(
                "Probe groups must cover every variant once on at most max_gpus GPUs"
            )
        tasks = (
            {job["identifier"]: job for job in plan["jobs"]}
            if training
            else {str(index): group for index, group in enumerate(groups)}
        )
        pending, first_idle = set(tasks), {}
        while pending or record["workers"]:
            devices = inventory()
            owned = {
                worker["pid"] for worker in record["workers"].values() if alive(worker)
            }
            for key in list(pending):
                worker = record["workers"].get(key, {})
                if worker and alive(worker):
                    foreign = [
                        pid
                        for pid in devices[worker["gpu"]]["pids"]
                        if process(pid).get("group") not in owned
                    ]
                    if foreign or "preempted_at" in worker:
                        worker.setdefault("preempted_at", time.time())
                        terminate(worker, worker["preempted_at"])
                        event(
                            "yielding_gpu",
                            stage=stage["name"],
                            gpu=worker["gpu"],
                            foreign=foreign,
                        )
                    continue
                if worker:
                    if "preempted_at" not in worker:
                        exited(worker)
                    else:
                        child = children.pop(worker["pid"], None)
                        if child is not None:
                            child.wait()
                    del record["workers"][key]
                if training:
                    complete = fit_complete(tasks[key], commit=stage["commit"])
                else:
                    complete = probe_complete(
                        stage=stage, variants=tasks[key], plan=plan, key=key
                    )
                if complete:
                    pending.remove(key)
                elif worker and "preempted_at" not in worker:
                    raise RuntimeError(f"Worker stopped before completing task: {key}")
            with Path(config["dispatch_lock"]).open("a") as dispatch:
                fcntl.flock(dispatch, fcntl.LOCK_EX)
                devices = inventory()
                free = ready(config=config, first_idle=first_idle, devices=devices)
                occupied = {worker["gpu"] for worker in record["workers"].values()}
                for key, task in tasks.items():
                    if key not in pending or key in record["workers"]:
                        continue
                    lanes = [
                        uuid
                        for uuid in occupied
                        if training
                        and sum(
                            worker["gpu"] == uuid
                            for worker in record["workers"].values()
                        )
                        < config["jobs_per_gpu"]
                        and not any(
                            worker["gpu"] == uuid and "preempted_at" in worker
                            for worker in record["workers"].values()
                        )
                        and all(
                            process(pid).get("group") in owned
                            for pid in devices[uuid]["pids"]
                        )
                    ]
                    if lanes:
                        gpu = lanes[0]
                    elif free and len(occupied) < config["max_gpus"]:
                        gpu = free.pop(0)
                        occupied.add(gpu)
                        first_idle.pop(gpu, None)
                    else:
                        continue
                    if training:
                        argv = command("mechanism_train", ["--config", task["config"]])
                    else:
                        arguments = ["--root", stage["root"], "--device", "cuda:0"]
                        arguments.extend([
                            "--output-dir",
                            str(Path(stage["output_dir"]) / f"g{key}"),
                            "--variants",
                            *task,
                        ])
                        argv = command(
                            "mechanism_probe",
                            [*arguments, "--checkpoints", *stage["checkpoints"]],
                        )
                    record["workers"][key] = launch(stage, argv, key, gpu)
                    owned.add(record["workers"][key]["pid"])
                    event("launched_worker", stage=stage["name"], task=key, gpu=gpu)
            if training:
                write(
                    Path(stage["root"]) / "state.json",
                    {
                        "time": time.time(),
                        "status": "running" if pending else "complete",
                        "pending": len(pending - record["workers"].keys()),
                        "active": {
                            key: {"pid": worker["pid"], "gpu": worker["gpu"]}
                            for key, worker in record["workers"].items()
                        },
                        "completed": len(tasks) - len(pending),
                        "failures": [],
                    },
                )
            event("waiting_workers", stage=stage["name"], pending=len(pending))
            if pending:
                time.sleep(config["poll_seconds"])

    try:
        for stage in config["stages"]:
            record = state["stages"].setdefault(
                stage["name"], {"workers": {}, "config": stage}
            )
            state["current_stage"] = stage["name"]
            record.setdefault("revision", pinned(stage))
            event("running", stage=stage["name"])
            kind = stage["kind"]
            if kind in {"study", "probes"}:
                root = Path(stage["root"])
                plan = read(root / "plan.json")
                inputs = {
                    "plan": sha256(root / "plan.json"),
                    **{
                        job["identifier"]: sha256(Path(job["config"]))
                        for job in plan["jobs"]
                    },
                }
                if (
                    plan["git_commit"] != stage["commit"]
                    or record.setdefault("inputs", inputs) != inputs
                ):
                    raise ValueError(f"Study source/configuration changed: {root}")
                if (
                    len({job["identifier"] for job in plan["jobs"]})
                    != len(plan["jobs"])
                    or not plan["jobs"]
                ):
                    raise ValueError("Study jobs must be nonempty and distinct")
                for job in plan["jobs"]:
                    model = read(Path(job["config"]))
                    run = model["adaptation"]["run"]
                    if (
                        Path(job["identifier"]).name != job["identifier"]
                        or job["identifier"] in {"", ".", ".."}
                        or model["variant"] != job["variant"]
                        or Path(run["output_dir"]).resolve()
                        != Path(job["output"]).resolve()
                        or not Path(job["output"])
                        .resolve()
                        .is_relative_to(root.resolve())
                    ):
                        raise ValueError(
                            f"Study plan/config mismatch: {job['identifier']}"
                        )
                    result = Path(job["output"]) / "result.json"
                    if result.is_file():
                        hashes = read(result)["provenance"]["source_sha256"]
                        if not hashes or any(
                            sha256(Path(stage["checkout"]) / name) != digest
                            for name, digest in hashes.items()
                        ):
                            raise ValueError(
                                f"Training source differs from pinned checkout: {result}"
                            )
                with (root / "supervisor.lock").open("a") as study_lock:
                    if kind == "study":
                        while True:
                            snapshot = study_status(root=root)
                            try:
                                fcntl.flock(study_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                                locked = True
                            except BlockingIOError:
                                locked = False
                            record["study"] = snapshot
                            if locked and snapshot["complete"]:
                                break
                            owned = {
                                worker["pid"]
                                for worker in record["workers"].values()
                                if alive(worker)
                            }
                            active_groups = {
                                process(pid).get("group") for pid in snapshot["active"]
                            }
                            if stage["launch"] and locked and not active_groups - owned:
                                gpu_stage(stage, record, plan)
                                record["study"] = study_status(root=root)
                                break
                            if locked:
                                fcntl.flock(study_lock, fcntl.LOCK_UN)
                            event("waiting_study", stage=stage["name"], **snapshot)
                            time.sleep(config["poll_seconds"])
                    else:
                        fcntl.flock(study_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        if not study_status(root=root)["complete"]:
                            raise RuntimeError(f"Probe sources are incomplete: {root}")
                        gpu_stage(stage, record, plan)
            else:
                artifacts = (
                    stage["artifacts"]
                    if kind == "command"
                    else [str(Path(stage["output_dir"]) / "report.json")]
                )
                if not artifacts:
                    raise ValueError("Commands require completion artifacts")
                if record.get("complete") and record.get("artifact_sha256") == {
                    item: sha256(Path(item))
                    for item in artifacts
                    if Path(item).is_file()
                }:
                    continue
                worker = record["workers"].get("command", {})
                if worker:
                    while alive(worker):
                        time.sleep(config["poll_seconds"])
                    exited(worker)
                argv = (
                    [*PYTHON, *stage["argv"]]
                    if kind == "command"
                    else command(
                        "mechanism_report",
                        ["--root", stage["root"], "--output-dir", stage["output_dir"]],
                    )
                )
                worker = record["workers"]["command"] = launch(stage, argv, "command")
                record["revision"] = worker["revision"]
                event("launched_command", stage=stage["name"])
                while alive(worker):
                    time.sleep(config["poll_seconds"])
                exited(worker)
                if any(
                    not Path(item).is_file()
                    or Path(item).stat().st_mtime < worker["started_at"]
                    for item in artifacts
                ):
                    raise RuntimeError(
                        f"Command did not refresh completion artifacts: {stage['name']}"
                    )
                report = read(Path(artifacts[0])) if kind == "report" else {}
                if kind == "report" and (
                    not report["scientific_complete"]
                    or Path(report["root"]).resolve() != Path(stage["root"]).resolve()
                    or report["completed_jobs"] != report["planned_jobs"]
                ):
                    raise RuntimeError(
                        f"Report contains incomplete fits: {stage['name']}"
                    )
                record["artifact_sha256"] = {
                    item: sha256(Path(item)) for item in artifacts
                }
                record["workers"].clear()
            record["complete"] = True
            event("completed_stage", stage=stage["name"])
        event("complete")
    except BaseException as error:
        workers = [
            worker
            for record in state["stages"].values()
            for worker in record["workers"].values()
        ]
        for worker in workers:
            if alive(worker):
                worker.setdefault("preempted_at", time.time())
        deadline = time.time() + 31
        while time.time() < deadline and any(alive(worker) for worker in workers):
            for worker in workers:
                if "preempted_at" in worker:
                    terminate(worker, worker["preempted_at"])
            time.sleep(1)
        event("failed", error=repr(error))
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--status", action="store_true")
    args = parser.parse_args()
    if args.status:
        print(json.dumps(status_summary(config=args.config)))
    else:
        run(config_path=args.config)


if __name__ == "__main__":
    main()
