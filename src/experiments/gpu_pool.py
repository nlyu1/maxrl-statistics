"""Thread-based GPU work-queue dispatcher.

Each worker thread owns one CUDA device and pulls jobs from a shared queue,
executing them as subprocesses. Jobs specify a `{device}` placeholder in their
command template; the pool substitutes the assigned device string at dispatch time.

Thread-based (not process-based) because workers spend their time in
`subprocess.run()` — no GIL contention, no CUDA fork-safety concerns, no
pickling overhead.
"""

from __future__ import annotations

import queue
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

        work_queue: queue.Queue[Job | None] = queue.Queue()
        for job in jobs:
            work_queue.put(job)

        results: list[JobResult] = []
        results_lock = threading.Lock()
        abort_event = threading.Event()

        def _worker(device: str) -> None:
            while not abort_event.is_set():
                try:
                    item = work_queue.get_nowait()
                except queue.Empty:
                    return
                if item is None:
                    return

                cmd = [tok.replace("{device}", device) for tok in item.cmd_template]

                log_fh = None
                if log_dir is not None:
                    safe_label = (
                        item.label.replace(" ", "_")
                        .replace("/", "-")
                        .replace("=", "-")
                        or "unnamed"
                    )
                    log_fh = (log_dir / f"{safe_label}.log").open("a")

                proc = subprocess.run(
                    cmd,
                    stdout=log_fh if log_fh else subprocess.DEVNULL,
                    stderr=subprocess.STDOUT if log_fh else subprocess.DEVNULL,
                )

                if log_fh is not None:
                    log_fh.close()

                result = JobResult(
                    job=item,
                    device=device,
                    returncode=proc.returncode,
                )

                with results_lock:
                    results.append(result)

                status = "done" if proc.returncode == 0 else f"FAILED rc={proc.returncode}"
                print(f"[{device}] {item.label} — {status}", flush=True)

                if proc.returncode != 0 and fail_fast:
                    abort_event.set()
                    # Drain the queue to unblock other workers
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

        return results
