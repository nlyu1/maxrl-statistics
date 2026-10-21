import argparse
import fcntl
import json
import os
import signal
import socket
import subprocess
import time
from pathlib import Path

from src.experiments.utkface.run import write_json
from src.experiments.utkface.scheduler import gpu_inventory, gpu_is_free, ready_gpus
from src.experiments.utkface.vlm_eval import load_plan, report, stop_container

DOWNLOAD_SCRIPT = """
import json
import os
import time
from pathlib import Path
from huggingface_hub import snapshot_download
root = Path('/study')
plan = json.loads((root / 'plan.json').read_text())
target = root / 'downloads.json'
state = json.loads(target.read_text()) if target.exists() else {}
for model in plan['models']:
    if model['identifier'] in state:
        continue
    path = snapshot_download(repo_id=model['repository'], revision=model['revision'],
        ignore_patterns=['*.md', 'LICENSE', '.gitattributes'], max_workers=4)
    state[model['identifier']] = {'snapshot': path, 'ready_at': time.time()}
    temporary = root / 'downloads.json.tmp'
    temporary.write_text(json.dumps(state, indent=2) + '\\n')
    os.replace(temporary, target)
    print(model['identifier'], 'ready', flush=True)
"""


def owned_worker(*, pid: int, root: Path) -> bool:
    try:
        command = (
            Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode()
        )
    except FileNotFoundError:
        return False
    return "src.experiments.utkface.vlm_eval worker" in command and str(root) in command


def terminate_worker(*, pid: int, root: Path) -> None:
    if owned_worker(pid=pid, root=root):
        if os.getpgid(pid) != pid:
            raise RuntimeError(
                "Refusing to signal a worker outside its dedicated process group"
            )
        os.killpg(pid, signal.SIGTERM)


def container_pids(*, container: str) -> set[int]:
    result = subprocess.run(
        ["docker", "top", container, "-eo", "pid"],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode:
        return set()
    return {
        int(line.strip()) for line in result.stdout.splitlines()[1:] if line.strip()
    }


def available_port() -> int:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return listener.getsockname()[1]


def start_downloads(*, root: Path, plan) -> str:
    name = "utkface-vlm-download-" + root.name.rsplit("-", 1)[-1]
    inspected = subprocess.run(
        ["docker", "inspect", name],
        check=False,
        capture_output=True,
        text=True,
    )
    if inspected.returncode == 0:
        metadata = json.loads(inspected.stdout)[0]
        if metadata["Config"]["Labels"].get("utkface.vlm.root") != str(root):
            raise RuntimeError(
                "The existing download container belongs to another study"
            )
        if metadata["State"]["Running"]:
            return name
        code = metadata["State"]["ExitCode"]
        if code == 0:
            return name
        raise RuntimeError(
            f"Model prefetch container exited with code {code}; inspect {name}"
        )
    subprocess.run(
        ["docker", "volume", "create", plan.cache_volume],
        check=True,
        capture_output=True,
    )
    subprocess.run(
        [
            "docker",
            "run",
            "-d",
            "--name",
            name,
            "--label",
            f"utkface.vlm.root={root}",
            "--entrypoint",
            "uv",
            "-v",
            f"{plan.cache_volume}:/root/.cache/huggingface",
            "-v",
            f"{root}:/study",
            plan.image,
            "run",
            "--no-project",
            "--no-sync",
            "python",
            "-c",
            DOWNLOAD_SCRIPT,
        ],
        check=True,
    )
    return name


def run_queue(*, root: Path) -> None:
    plan = load_plan(root=root)
    with (root / "queue.lock").open("a") as supervisor:
        fcntl.flock(supervisor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        (root / "supervisor.pid").write_text(str(os.getpid()) + "\n")
        state_path = root / "state.json"
        state = (
            json.loads(state_path.read_text())
            if state_path.exists()
            else {
                "status": "running",
                "created_at": time.time(),
                "jobs": {
                    model.identifier: {"status": "pending", "attempts": []}
                    for model in plan.models
                },
                "gpu_interruptions": {},
            }
        )
        if state["status"] == "completed":
            report(root=root)
            return
        for record in state["jobs"].values():
            if record["status"] == "failed":
                record["status"] = "pending"
        state["status"] = "running"
        state.pop("error", None)
        download_container = start_downloads(root=root, plan=plan)
        state["download_container"] = download_container
        first_idle, reusable, children, downloads = {}, set(), {}, {}
        plan.dispatch_lock.parent.mkdir(parents=True, exist_ok=True)
        shutdown = False

        def interrupt(signum, frame):
            nonlocal shutdown
            shutdown = True

        signal.signal(signal.SIGTERM, interrupt)
        signal.signal(signal.SIGINT, interrupt)
        while True:
            now = time.time()
            inventory = gpu_inventory()
            by_uuid = {gpu["uuid"]: gpu for gpu in inventory}
            for identifier, record in state["jobs"].items():
                if record["status"] != "running":
                    continue
                attempt = record["attempts"][-1]
                pid, container = attempt["pid"], attempt["container"]
                alive = owned_worker(pid=pid, root=root)
                if alive:
                    gpu = by_uuid[attempt["gpu_uuid"]]
                    foreign = set(gpu["pids"]) - container_pids(container=container)
                    if foreign and not attempt.get("preempted"):
                        attempt.update({
                            "preempted": True,
                            "foreign_pids": sorted(foreign),
                            "preempted_at": now,
                        })
                        entry = state["gpu_interruptions"].setdefault(
                            attempt["gpu_uuid"], {"count": 0, "cooldown_until": 0}
                        )
                        entry["count"] += 1
                        entry["cooldown_until"] = now + min(3600, 900 * entry["count"])
                        terminate_worker(pid=pid, root=root)
                    elif shutdown:
                        terminate_worker(pid=pid, root=root)
                    continue
                child = children.pop(identifier, None)
                if child is not None:
                    attempt["returncode"] = child.wait()
                attempt["finished_at"] = now
                attempt["seconds"] = now - attempt["started_at"]
                stop_container(root=root, container=container)
                specs = [
                    spec
                    for spec in plan.evaluations
                    if spec.model_identifier == identifier
                ]
                complete = all(
                    (root / "evaluations" / spec.identifier / "result.json").exists()
                    for spec in specs
                )
                if complete:
                    record["status"] = "completed"
                    reusable.add(attempt["gpu_uuid"])
                    print(f"Completed {identifier}", flush=True)
                elif attempt.get("preempted") or shutdown:
                    record["status"] = "pending"
                else:
                    record["status"] = "failed"
                    state["status"] = "failed"
                    state["error"] = (
                        f"Worker {identifier} exited before completing its evaluations; see workers/{container}"
                    )
            active = [
                record
                for record in state["jobs"].values()
                if record["status"] == "running"
            ]
            download_state = subprocess.check_output(
                [
                    "docker",
                    "inspect",
                    "--format",
                    "{{json .State}}",
                    download_container,
                ],
                text=True,
            )
            download_status = json.loads(download_state)
            if not download_status["Running"] and download_status["ExitCode"] != 0:
                state["status"] = "failed"
                state["error"] = (
                    "CPU model prefetch failed; inspect the download container logs"
                )
            if state["status"] == "failed" or shutdown:
                for record in active:
                    terminate_worker(pid=record["attempts"][-1]["pid"], root=root)
                if not active:
                    if shutdown:
                        state["status"] = "stopped"
                    state["updated_at"] = now
                    write_json(path=state_path, payload=state)
                    report(root=root)
                    break
            else:
                downloads_path = root / "downloads.json"
                downloads = (
                    json.loads(downloads_path.read_text())
                    if downloads_path.exists()
                    else {}
                )
                with plan.dispatch_lock.open("a") as dispatch:
                    fcntl.flock(dispatch, fcntl.LOCK_EX)
                    leased = {record["attempts"][-1]["gpu_uuid"] for record in active}
                    free = ready_gpus(
                        inventory=inventory,
                        leased=leased,
                        interruptions=state["gpu_interruptions"],
                        first_idle=first_idle,
                        reusable=reusable,
                        now=now,
                    )
                    pending = [
                        identifier
                        for identifier, record in state["jobs"].items()
                        if record["status"] == "pending" and identifier in downloads
                    ]
                    slots = max(0, plan.max_parallel_models - len(active))
                    for identifier, gpu in zip(
                        pending[:slots], free[:slots], strict=False
                    ):
                        refreshed = next(
                            item
                            for item in gpu_inventory()
                            if item["uuid"] == gpu["uuid"]
                        )
                        if not gpu_is_free(gpu=refreshed):
                            first_idle.pop(gpu["uuid"], None)
                            continue
                        record = state["jobs"][identifier]
                        attempt_index = len(record["attempts"]) + 1
                        container = (
                            f"utkface-vlm-{identifier}-{int(now)}-{attempt_index}"
                        )
                        command = [
                            "uv",
                            "run",
                            "--no-sync",
                            "python",
                            "-m",
                            "src.experiments.utkface.vlm_eval",
                            "worker",
                            "--root",
                            str(root),
                            "--model",
                            identifier,
                            "--gpu-uuid",
                            gpu["uuid"],
                            "--port",
                            str(available_port()),
                            "--container",
                            container,
                        ]
                        log_path = root / f"{container}.log"
                        with log_path.open("w") as log:
                            child = subprocess.Popen(
                                command,
                                stdout=log,
                                stderr=subprocess.STDOUT,
                                stdin=subprocess.DEVNULL,
                                start_new_session=True,
                                env={
                                    **os.environ,
                                    "OMP_NUM_THREADS": "2",
                                    "OPENBLAS_NUM_THREADS": "2",
                                },
                            )
                        attempt = {
                            "pid": child.pid,
                            "container": container,
                            "gpu_uuid": gpu["uuid"],
                            "gpu_index": gpu["index"],
                            "started_at": time.time(),
                            "command": command,
                            "idle_observed_seconds": now
                            - first_idle.get(gpu["uuid"], now),
                            "log": str(log_path),
                        }
                        record["attempts"].append(attempt)
                        record["status"] = "running"
                        children[identifier] = child
                        first_idle.pop(gpu["uuid"], None)
                        reusable.discard(gpu["uuid"])
                        print(
                            f"Launched {identifier} on GPU {gpu['index']} ({gpu['uuid']})",
                            flush=True,
                        )
            if all(
                record["status"] == "completed" for record in state["jobs"].values()
            ):
                state["status"] = "completed"
            state["updated_at"] = time.time()
            state["completed_evaluations"] = sum(
                (root / "evaluations" / spec.identifier / "result.json").exists()
                for spec in plan.evaluations
            )
            state["ready_models"] = sorted(downloads)
            write_json(path=state_path, payload=state)
            report(root=root)
            if state["status"] == "completed":
                print(
                    "All nine configurations completed; REPORT.md and summary.csv are ready",
                    flush=True,
                )
                break
            time.sleep(plan.poll_seconds)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--detach", action="store_true")
    args = parser.parse_args()
    if args.detach:
        load_plan(root=args.root)
        with (args.root / "supervisor.log").open("a") as log:
            child = subprocess.Popen(
                [
                    "uv",
                    "run",
                    "--no-sync",
                    "python",
                    "-m",
                    "src.experiments.utkface.vlm_queue",
                    "--root",
                    str(args.root),
                ],
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                env={**os.environ, "OMP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2"},
            )
        print(json.dumps({"supervisor_pid": child.pid, "root": str(args.root)}))
    else:
        run_queue(root=args.root)


if __name__ == "__main__":
    main()
