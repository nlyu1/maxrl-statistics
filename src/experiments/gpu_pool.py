"""Thread-based GPU work-queue dispatcher.

Each worker thread owns one CUDA device and pulls jobs from a shared queue,
executing them as subprocesses. Jobs specify a `{device}` placeholder in their
command template; the pool substitutes the assigned device string at dispatch time.

Thread-based (not process-based) because workers spend their time in
`subprocess.Popen()` — no GIL contention, no CUDA fork-safety concerns, no
pickling overhead.
"""

from __future__ import annotations

import queue
import signal
import subprocess
import threading
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import torch


@dataclass(kw_only=True, frozen=True)
class Job:
    """One atomic unit of work: a command template to run on a GPU.

    The `cmd_template` list may contain the literal string ``{device}``
    which the pool replaces with the assigned device (e.g. ``cuda:2``).
    """

    cmd_template: list[str]
    label: str = ""
    env: dict[str, str] = field(default_factory=dict)
    log_path: Path | None = None


@dataclass(kw_only=True, frozen=True)
class JobResult:
    """Outcome of a dispatched job."""

    job: Job
    device: str
    returncode: int


class GPUPool:
    """Distribute jobs across available CUDA devices via a work-stealing queue.

    Usage::

        pool = GPUPool()          # auto-detect all GPUs
        pool = GPUPool(device_ids=[0, 2, 3])  # specific devices
        results = pool.run(jobs)  # blocks until all complete

    Each GPU processes one job at a time. When a job finishes, the worker
    immediately pulls the next job from the shared queue — no static
    partitioning, no idle GPUs while work remains.
    """

    def __init__(self, *, device_ids: Sequence[int] | None = None) -> None:
        if device_ids is not None:
            self._devices = [f"cuda:{i}" for i in device_ids]
        else:
            n = torch.cuda.device_count()
            if n == 0:
                raise RuntimeError(
                    "No CUDA devices available. Set --gpu-ids explicitly or "
                    "ensure CUDA_VISIBLE_DEVICES is configured."
                )
            self._devices = [f"cuda:{i}" for i in range(n)]

    @property
    def num_devices(self) -> int:
        return len(self._devices)

    def run(
        self,
        jobs: Sequence[Job],
        *,
        log_dir: Path | None = None,
        fail_fast: bool = False,
    ) -> list[JobResult]:
        """Dispatch all jobs and block until completion.

        Parameters
        ----------
        jobs
            Sequence of Job objects to execute.
        log_dir
            If provided, per-job stdout/stderr is written to
            ``log_dir/{sanitized_label}.log``.
        fail_fast
            If True, stop dispatching new jobs after the first failure.
            Workers that have already started a job will finish it.

        Returns
        -------
        list[JobResult]
            Results in completion order. Inspect `returncode` to detect failures.
        """
        if log_dir is not None:
            log_dir.mkdir(parents=True, exist_ok=True)

        total_jobs = len(jobs)
        work_queue: queue.Queue[tuple[int, Job] | None] = queue.Queue()
        for idx, job in enumerate(jobs, start=1):
            work_queue.put((idx, job))

        results: list[JobResult] = []
        results_lock = threading.Lock()
        abort_event = threading.Event()

        # Track active subprocesses so the SIGINT handler can kill them.
        active_procs: dict[int, subprocess.Popen] = {}  # thread ident -> Popen
        active_procs_lock = threading.Lock()

        # Install SIGINT handler for immediate Ctrl+C termination.
        original_sigint = signal.getsignal(signal.SIGINT)

        def _sigint_handler(signum: int, frame: object) -> None:
            abort_event.set()
            with active_procs_lock:
                for proc in active_procs.values():
                    proc.terminate()
            raise SystemExit(130)

        signal.signal(signal.SIGINT, _sigint_handler)

        def _worker(device: str) -> None:
            tid = threading.current_thread().ident
            while not abort_event.is_set():
                try:
                    item = work_queue.get_nowait()
                except queue.Empty:
                    return
                if item is None:
                    return

                idx, job = item
                cmd = [tok.replace("{device}", device) for tok in job.cmd_template]

                log_fh = None
                if job.log_path is not None:
                    job.log_path.parent.mkdir(parents=True, exist_ok=True)
                    log_fh = job.log_path.open("w")
                elif log_dir is not None:
                    safe_label = (
                        job.label.replace(" ", "_")
                        .replace("/", "-")
                        .replace("=", "-")
                        or "unnamed"
                    )
                    log_fh = (log_dir / f"{safe_label}.log").open("w")

                print(
                    f"[{idx:>{len(str(total_jobs))}}/{total_jobs}] [{device}] "
                    f"Starting: {job.label}",
                    flush=True,
                )

                proc = subprocess.Popen(
                    cmd,
                    stdout=log_fh if log_fh else subprocess.DEVNULL,
                    stderr=subprocess.STDOUT if log_fh else subprocess.DEVNULL,
                )
                with active_procs_lock:
                    active_procs[tid] = proc

                proc.wait()

                with active_procs_lock:
                    active_procs.pop(tid, None)

                if log_fh is not None:
                    log_fh.close()

                result = JobResult(
                    job=job,
                    device=device,
                    returncode=proc.returncode,
                )

                with results_lock:
                    results.append(result)

                status = "done" if proc.returncode == 0 else f"FAILED rc={proc.returncode}"
                print(
                    f"[{idx:>{len(str(total_jobs))}}/{total_jobs}] [{device}] "
                    f"{job.label} — {status}",
                    flush=True,
                )

                if proc.returncode != 0 and fail_fast:
                    abort_event.set()
                    # Drain the queue to unblock other workers.
                    while not work_queue.empty():
                        try:
                            work_queue.get_nowait()
                        except queue.Empty:
                            break
                    return

        threads = [
            threading.Thread(target=_worker, args=(dev,), daemon=True, name=f"gpu-{dev}")
            for dev in self._devices
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Restore original signal handler.
        signal.signal(signal.SIGINT, original_sigint)

        return results
