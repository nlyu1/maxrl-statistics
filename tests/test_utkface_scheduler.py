import json
import os
import time
from pathlib import Path

import pytest

from src.experiments.utkface.config import canonical_config
from src.experiments.utkface.coordination import canonical_coordination
from src.experiments.utkface.data import manifest_fingerprint
from src.experiments.utkface.scheduler import charged_seconds, gpu_is_free, run
from src.experiments.utkface.study import StudyJob, StudyPlan, make_plan


@pytest.fixture(autouse=True)
def simulated_dispatch_clock(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.experiments.utkface import scheduler

    clock = [time.time()]

    def wait(*, active: list[dict], timeout: float) -> None:
        clock[0] += 10

    monkeypatch.setattr(scheduler.time, "time", lambda: clock[0])
    monkeypatch.setattr(scheduler, "wait_for_progress", wait)


def test_adopted_external_preemption_preserves_unknown_exit_status() -> None:
    from src.experiments.utkface.scheduler import retryable_preemption

    attempt = {
        "returncode": 1,
        "yielded_to_foreign_process": True,
        "adopted_preemption": True,
        "exit_status_inferred": True,
    }
    assert retryable_preemption(attempt=attempt)
    assert not retryable_preemption(attempt={**attempt, "timed_out": True})
    assert not retryable_preemption(
        attempt={**attempt, "yielded_to_foreign_process": False}
    )


@pytest.mark.parametrize("running", [False, True])
def test_deadline_preserves_attempts_and_only_stops_owned_queue(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, running: bool
) -> None:
    from src.experiments.utkface import scheduler

    item = job(root=tmp_path, identifier="deadline", lr=1e-4)
    plan = StudyPlan(
        output_dir=tmp_path,
        data_fingerprint="test",
        training_source_sha256={},
        budget_gpu_hours=10,
        max_concurrent=6,
        jobs=(item,),
    )
    attempt = {
        "pid": 123,
        "gpu_uuid": "gpu-0",
        "gpu_index": 0,
        "started_at": time.time() - 2,
        "runtime_limit_seconds": 180,
        "output_dir": str(tmp_path / "missing"),
        "config_path": "owned-config",
    }
    if not running:
        attempt.update({"seconds": 1, "returncode": 143})
    (tmp_path / "state.json").write_text(
        json.dumps({
            "status": "running",
            "jobs": {
                item.identifier: {
                    "status": "running" if running else "pending",
                    "attempts": [attempt],
                }
            },
        })
    )
    control = canonical_coordination(root=tmp_path)
    control = type(control).model_validate({
        **control.model_dump(),
        "deadlines": {str(tmp_path): time.time() - 1},
    })
    alive = [running]
    stopped = []

    def stop(*, attempt: dict) -> None:
        stopped.append(attempt["pid"])
        alive[0] = False
        attempt["termination_requested_at"] = time.time()

    monkeypatch.setattr(scheduler, "gpu_inventory", fake_inventory)
    monkeypatch.setattr(scheduler, "owned_process", lambda **kwargs: alive[0])
    monkeypatch.setattr(scheduler, "terminate_attempt", stop)
    monkeypatch.setattr(scheduler.time, "sleep", lambda seconds: None)
    run(plan=plan, poll_seconds=0, coordination=control)
    state = json.loads((tmp_path / "state.json").read_text())
    record = state["jobs"][item.identifier]
    assert state["status"] == "deadline_reached"
    assert record["status"] == "skipped" and record["reason"] == "deadline_reached"
    assert len(record["attempts"]) == 1
    assert record["attempts"][0]["seconds"] > 0
    assert stopped == ([123] if running else [])


def job(
    *, root: Path, identifier: str, lr: float, dependencies: tuple[str, ...] = ()
) -> StudyJob:
    config = canonical_config(
        manifest=root / "manifest.json", output_dir=root / "runs" / identifier
    )
    config = type(config).model_validate({
        **config.model_dump(),
        "epochs": 1,
        "learning_rate": lr,
    })
    return StudyJob(
        identifier=identifier,
        stage="main" if dependencies else "wide",
        variant="ce",
        priority=1 if dependencies else 0,
        optional=False,
        config=config,
        dependencies=dependencies,
        tune_from=dependencies,
        tuning_rank=0,
        gate_candidate=(),
        gate_reference=(),
        competitive_ratio=1.05,
    )


def fake_inventory() -> list[dict]:
    return [
        {
            "index": index,
            "uuid": f"gpu-{index}",
            "utilization": 0,
            "memory_mib": 0,
            "pids": [],
        }
        for index in range(8)
    ]


@pytest.mark.parametrize(
    "gpu,expected",
    [
        ({"pids": [], "memory_mib": 500, "utilization": 0}, True),
        ({"pids": [123], "memory_mib": 500, "utilization": 0}, False),
        ({"pids": [], "memory_mib": 1500, "utilization": 0}, False),
        ({"pids": [], "memory_mib": 0, "utilization": 50}, False),
    ],
)
def test_free_gpu_requires_no_compute_processes_low_memory_and_utilization(
    gpu: dict, expected: bool
) -> None:
    assert gpu_is_free(gpu=gpu) is expected


def test_budget_counts_failed_and_running_attempts() -> None:
    state = {
        "jobs": {
            "a": {"attempts": [{"started_at": 0, "seconds": 50}, {"started_at": 80}]},
            "b": {"attempts": [{"started_at": 20, "seconds": 40}]},
        }
    }
    assert charged_seconds(state=state, now=100) == 110


def test_study_contains_requested_main_subset_and_noise_matrices(
    tmp_path: Path,
) -> None:
    records = [{"relative_path": f"image{i}.jpg", "age": i % 100} for i in range(10000)]
    splits = {
        "train": list(range(8000)),
        "val": list(range(8000, 9000)),
        "test": list(range(9000, 10000)),
    }
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps({
            "records": records,
            "splits": splits,
            "fingerprint": manifest_fingerprint(records=records, splits=splits),
            "preprocessing": {"deduplicated_before_split": True},
        })
    )
    plan = make_plan(
        manifest=manifest,
        output_dir=tmp_path / "study",
        budget_gpu_hours=64,
        max_concurrent=6,
    )
    main = [item for item in plan.jobs if item.stage == "main"]
    assert len(main) == 66 and len({item.variant for item in main}) == 22
    assert {item.config.seed for item in main} == {42, 43, 44}
    assert all(
        item.config.epochs == 50 and item.config.head == "categorical" for item in main
    )
    assert (
        len([item for item in main if item.config.loss.method.startswith("stat_maxrl")])
        == 24
    )
    assert (
        len([
            item
            for item in plan.jobs
            if item.stage == "subset" and item.config.head == "categorical"
        ])
        == 16
    )
    assert (
        len([
            item
            for item in plan.jobs
            if item.stage == "noise"
            and item.config.head == "categorical"
            and not item.optional
        ])
        == 48
    )
    from src.experiments.utkface.extension import make_coordination, make_extension

    frozen = plan.model_dump_json()
    extra = make_extension(base=plan, output_dir=tmp_path / "extension")
    control = make_coordination(
        base=plan,
        extension=extra,
        base_path=tmp_path / "plan.json",
        extension_path=tmp_path / "extension/plan.json",
        deadline=1000,
    )
    assert plan.model_dump_json() == frozen
    assert len(extra.jobs) == 56 and extra.budget_gpu_hours == 24
    assert extra.training_source_sha256 == plan.training_source_sha256
    assert (
        control.max_concurrent == 6 and control.deadlines[str(extra.output_dir)] == 1000
    )
    assert {
        item.config.noise.seed
        for item in extra.jobs
        if item.stage == "noise_realization"
    } == {2027, 2028}
    assert all(
        item.config.epochs * item.config.train_limit == 50 * 4096
        for item in extra.jobs
        if item.stage == "matched_steps"
    )
    assert all(
        set(item.dependencies) <= {source.identifier for source in plan.jobs}
        for item in extra.jobs
    )
    from src.experiments.utkface.requeue import continuation

    original_path = tmp_path / "plan.json"
    original_path.write_text(plan.model_dump_json())
    extra.output_dir.mkdir()
    archived_path = extra.output_dir / "plan.json"
    archived_path.write_text(extra.model_dump_json())
    resumed, updated = continuation(
        original_path=original_path,
        archived_path=archived_path,
        previous_control=control.model_dump(mode="json"),
        output_dir=tmp_path / "continuation",
    )
    assert len(resumed.jobs) == 24
    assert sum(j.stage == "subset_replication" for j in resumed.jobs) == 16
    assert sum(j.stage == "matched_steps" for j in resumed.jobs) == 8
    assert all(j.config.noise.mode == "none" for j in resumed.jobs)
    assert len(updated.held_jobs) == 40
    assert all(
        j.stage == "noise_replication"
        for j in plan.jobs
        if j.identifier in updated.held_jobs
    )
    assert updated.max_concurrent == 8 and not updated.enforce_gpu_hour_budget
    assert not updated.deadlines
    assert original_path.read_text() == frozen
    assert archived_path.read_text() == extra.model_dump_json()


@pytest.mark.parametrize("peer_count", [0, 2])
def test_scheduler_selects_validation_lr_respects_eight_gpu_cap_and_exports_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, peer_count: int
) -> None:
    from src.experiments.utkface import scheduler

    jobs = tuple(
        job(root=tmp_path, identifier=f"pilot{i}", lr=1e-4 * (i + 1)) for i in range(8)
    )
    final = job(
        root=tmp_path,
        identifier="final",
        lr=1e-4,
        dependencies=tuple(item.identifier for item in jobs),
    )
    plan = StudyPlan(
        output_dir=tmp_path,
        data_fingerprint="test",
        training_source_sha256={},
        budget_gpu_hours=10,
        max_concurrent=6,
        jobs=(*jobs, final),
    )
    selected = []
    peaks = []
    original_write = scheduler.write_json

    class Child:
        pid = 123456

        def __init__(self, command, **kwargs):
            config = json.loads(
                Path(command[command.index("--config") + 1]).read_text()
            )
            output = Path(config["output_dir"])
            output.mkdir(parents=True)
            selected.append(config)
            (output / "result.json").write_text(
                json.dumps({
                    "config": config,
                    "validation": {"mae_mean": 10 - config["learning_rate"] * 1000},
                    "train_unique_images": 18462,
                })
            )

        def poll(self):
            return 0

    def record_write(*, path, payload):
        if path.name == "state.json":
            peaks.append(
                sum(item["status"] == "running" for item in payload["jobs"].values())
            )
        original_write(path=path, payload=payload)

    monkeypatch.setattr(scheduler, "gpu_inventory", fake_inventory)
    monkeypatch.setattr(scheduler.subprocess, "Popen", Child)
    monkeypatch.setattr(scheduler.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(scheduler, "write_json", record_write)
    control = canonical_coordination(root=plan.output_dir)
    if peer_count:
        peer_root = tmp_path / "peer"
        peer_root.mkdir()
        peer = StudyPlan.model_validate({
            **plan.model_dump(),
            "output_dir": peer_root,
            "jobs": tuple(
                job(root=peer_root, identifier=f"peer{i}", lr=1e-4)
                for i in range(peer_count)
            ),
        })
        peer_path = peer_root / "plan.json"
        peer_path.write_text(peer.model_dump_json())
        (peer_root / "state.json").write_text(
            json.dumps({
                "status": "running",
                "jobs": {
                    item.identifier: {
                        "status": "running",
                        "attempts": [{"gpu_uuid": f"gpu-{i}"}],
                    }
                    for i, item in enumerate(peer.jobs)
                },
            })
        )
        control = type(control).model_validate({
            **control.model_dump(),
            "peer_plans": (peer_path,),
        })
    run(plan=plan, poll_seconds=0, coordination=control)
    state = json.loads((tmp_path / "state.json").read_text())
    assert state["status"] == "completed" and max(peaks) == 8 - peer_count
    assert selected[-1]["learning_rate"] == pytest.approx(8e-4)
    assert all(item["status"] == "completed" for item in state["jobs"].values())


@pytest.mark.parametrize("enforce", [False, True])
def test_scheduler_only_enforces_gpu_hour_budget_when_enabled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, enforce: bool
) -> None:
    from src.experiments.utkface import scheduler

    item = job(root=tmp_path, identifier="cannot_fit", lr=1e-4)
    plan = StudyPlan(
        output_dir=tmp_path,
        data_fingerprint="test",
        training_source_sha256={},
        budget_gpu_hours=0.01,
        max_concurrent=6,
        jobs=(item,),
    )
    monkeypatch.setattr(scheduler, "gpu_inventory", fake_inventory)
    monkeypatch.setattr(scheduler.time, "sleep", lambda seconds: None)

    class Child:
        pid = 123456

        def __init__(self, command, **kwargs):
            config = json.loads(
                Path(command[command.index("--config") + 1]).read_text()
            )
            output = Path(config["output_dir"])
            output.mkdir(parents=True)
            (output / "result.json").write_text(
                json.dumps({
                    "config": config,
                    "validation": {"mae_mean": 5},
                    "train_unique_images": 18462,
                })
            )

        def poll(self):
            return 0

    old_attempt = {"started_at": time.time() - 1000, "seconds": 1000, "returncode": 143}
    (tmp_path / "state.json").write_text(
        json.dumps({
            "status": "running",
            "jobs": {item.identifier: {"status": "pending", "attempts": [old_attempt]}},
        })
    )
    monkeypatch.setattr(scheduler.subprocess, "Popen", Child)
    run(
        plan=plan,
        poll_seconds=0,
        coordination=canonical_coordination(root=plan.output_dir).model_copy(
            update={"enforce_gpu_hour_budget": enforce}
        ),
    )
    state = json.loads((tmp_path / "state.json").read_text())
    assert state["status"] == ("budget_exhausted" if enforce else "completed")
    attempts = state["jobs"]["cannot_fit"]["attempts"]
    assert attempts[0] == old_attempt
    assert len(attempts) == (1 if enforce else 2)
    assert charged_seconds(state=state, now=time.time()) >= 1000


def test_immediate_reuse_preserves_final_free_check_and_held_jobs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from src.experiments.utkface import scheduler

    jobs = tuple(
        job(root=tmp_path, identifier=name, lr=1e-4)
        for name in ("first", "second", "held")
    )
    plan = StudyPlan(
        output_dir=tmp_path,
        data_fingerprint="test",
        training_source_sha256={},
        budget_gpu_hours=10,
        max_concurrent=6,
        jobs=jobs,
    )
    launches = []
    checks = []

    def inventory():
        checks.append(time.time())
        return fake_inventory()[:1]

    class Child:
        pid = 123456

        def __init__(self, command, **kwargs):
            config = json.loads(
                Path(command[command.index("--config") + 1]).read_text()
            )
            launches.append(time.time())
            assert checks[-1] == launches[-1]
            output = Path(config["output_dir"])
            output.mkdir(parents=True)
            (output / "result.json").write_text(
                json.dumps({
                    "config": config,
                    "validation": {"mae_mean": 5},
                    "train_unique_images": 18462,
                })
            )

        def poll(self):
            return 0

    monkeypatch.setattr(scheduler, "gpu_inventory", inventory)
    monkeypatch.setattr(scheduler.subprocess, "Popen", Child)
    run(
        plan=plan,
        poll_seconds=10,
        coordination=canonical_coordination(root=tmp_path).model_copy(
            update={"held_jobs": ("held",)}
        ),
    )
    state = json.loads((tmp_path / "state.json").read_text())
    first = state["jobs"]["first"]["attempts"][-1]
    second = state["jobs"]["second"]["attempts"][-1]
    assert second["started_at"] == first["ended_at"]
    assert state["jobs"]["held"] == {"status": "pending", "attempts": []}
    progress = json.loads((tmp_path / "status.json").read_text())
    assert progress["held_pending"] == 1 and progress["queued_pending"] == 0


def test_final_free_check_rejects_gpu_that_became_occupied(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from src.experiments.utkface import scheduler

    item = job(root=tmp_path, identifier="race", lr=1e-4)
    plan = StudyPlan(
        output_dir=tmp_path,
        data_fingerprint="test",
        training_source_sha256={},
        budget_gpu_hours=1,
        max_concurrent=6,
        jobs=(item,),
    )
    calls = [0]

    def inventory():
        calls[0] += 1
        gpus = fake_inventory()[:1]
        if calls[0] == 3:
            gpus[0]["pids"] = [99999]
        return gpus

    class Child:
        pid = 123456

        def __init__(self, command, **kwargs):
            assert calls[0] >= 6
            config = json.loads(
                Path(command[command.index("--config") + 1]).read_text()
            )
            output = Path(config["output_dir"])
            output.mkdir(parents=True)
            (output / "result.json").write_text(
                json.dumps({
                    "config": config,
                    "validation": {"mae_mean": 5},
                    "train_unique_images": 18462,
                })
            )

        def poll(self):
            return 0

    monkeypatch.setattr(scheduler, "gpu_inventory", inventory)
    monkeypatch.setattr(scheduler.subprocess, "Popen", Child)
    run(plan=plan, poll_seconds=10, coordination=canonical_coordination(root=tmp_path))
    assert json.loads((tmp_path / "state.json").read_text())["status"] == "completed"


def test_new_gpu_requires_idle_checks_ten_seconds_apart_and_resets_when_busy() -> None:
    from src.experiments.utkface.scheduler import ready_gpus

    gpu = fake_inventory()[0]
    first_idle = {}

    def ready(*, now, current=gpu, reusable=frozenset(), interruptions=None):
        return ready_gpus(
            inventory=[current],
            leased=set(),
            interruptions=interruptions or {},
            first_idle=first_idle,
            reusable=set(reusable),
            now=now,
        )

    assert ready(now=0) == []
    assert ready(now=9.99) == []
    assert ready(now=10) == [gpu]
    assert ready(now=11, current={**gpu, "pids": [99999]}) == []
    assert ready(now=12) == []
    assert ready(now=22) == [gpu]
    assert (
        ready(
            now=23,
            reusable={gpu["uuid"]},
            interruptions={gpu["uuid"]: {"cooldown_until": 30}},
        )
        == []
    )
    assert ready(now=30, reusable={gpu["uuid"]}) == [gpu]


def test_scheduler_adopts_completed_attempt_without_charging_offline_gap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from src.experiments.utkface import scheduler

    item = job(root=tmp_path, identifier="adopted", lr=1e-4)
    plan = StudyPlan(
        output_dir=tmp_path,
        data_fingerprint="test",
        training_source_sha256={},
        budget_gpu_hours=10,
        max_concurrent=6,
        jobs=(item,),
    )
    output = tmp_path / "run"
    output.mkdir()
    result_path = output / "result.json"
    result_path.write_text("{}")
    now = time.time()
    os.utime(result_path, (now - 600, now - 600))
    (tmp_path / "state.json").write_text(
        json.dumps({
            "status": "running",
            "jobs": {
                "adopted": {
                    "status": "running",
                    "attempts": [
                        {
                            "pid": 123456,
                            "started_at": now - 1200,
                            "output_dir": str(output),
                            "config_path": "unused",
                        }
                    ],
                }
            },
        })
    )
    monkeypatch.setattr(scheduler, "gpu_inventory", fake_inventory)
    monkeypatch.setattr(scheduler, "owned_process", lambda **kwargs: False)
    run(
        plan=plan,
        poll_seconds=0,
        coordination=canonical_coordination(root=plan.output_dir),
    )
    state = json.loads((tmp_path / "state.json").read_text())
    assert state["status"] == "completed"
    assert charged_seconds(state=state, now=now) == pytest.approx(600)


def test_report_excludes_tuning_from_seed_aggregation_and_preserves_scalar_nll(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from src.experiments.utkface import study_report

    jobs, states = [], {}
    for identifier, stage, head, seed, mae in (
        ("first", "main", "categorical", 42, 1.0),
        ("second", "main", "categorical", 43, 3.0),
        ("pilot", "wide", "categorical", 42, 99.0),
        ("scalar", "scalar", "scalar", 42, 5.0),
    ):
        item = job(root=tmp_path, identifier=identifier, lr=1e-4)
        config = type(item.config).model_validate({
            **item.config.model_dump(),
            "head": head,
            "seed": seed,
            "loss": {**item.config.loss.model_dump(), "method": "mse"},
        })
        item = StudyJob.model_validate({
            **item.model_dump(),
            "config": config,
            "stage": stage,
            "variant": "mse_scalar" if head == "scalar" else "mse",
        })
        output = config.output_dir
        output.mkdir(parents=True)
        validation = {"mae_mean": mae, "median_spearman": 0.5, "label_nll": 2.0}
        (output / "history.json").write_text(
            json.dumps([
                {
                    "epoch": 1,
                    "learning_rate": 1e-4,
                    "train_loss": 0.1,
                    "grad_norm": 1.0,
                    "train_count": 64,
                    "validation": validation,
                    "label_noise": {
                        "signal_variance_years2": 10,
                        "noise_mse_years2": 0,
                    },
                }
            ])
        )
        result_path = output / "result.json"
        result_path.write_text(
            json.dumps({
                "config": config.model_dump(mode="json"),
                "data_fingerprint": "test",
                "training_fingerprint": "same-subset",
                "train_unique_images": 18462,
                "best_epoch": 1,
                "validation": validation,
                "test": {
                    metric: None if head == "scalar" and metric == "label_nll" else mae
                    for metric in study_report.METRICS
                },
            })
        )
        jobs.append(item)
        states[identifier] = {
            "status": "completed",
            "result_path": str(result_path),
            "attempts": [{"started_at": 0, "seconds": 10}],
        }
    plan = StudyPlan(
        output_dir=tmp_path,
        data_fingerprint="test",
        training_source_sha256={},
        budget_gpu_hours=64,
        max_concurrent=6,
        jobs=tuple(jobs),
    )
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(plan.model_dump_json())
    (tmp_path / "state.json").write_text(
        json.dumps({"status": "completed", "jobs": states})
    )
    monkeypatch.setattr(study_report, "describe_noise", lambda **kwargs: [])
    study_report.report(
        plan_path=plan_path, output_dir=tmp_path / "report", figures=True
    )
    summary = json.loads((tmp_path / "report" / "summary.json").read_text())
    categorical = next(row for row in summary if row["stage"] == "main")
    scalar = next(row for row in summary if row["stage"] == "scalar")
    assert len(summary) == 2 and categorical["seeds"] == 2
    assert categorical["mae_mean_mean"] == 2
    assert categorical["mae_mean_sd"] == pytest.approx(2**0.5)
    assert scalar["label_nll_mean"] is None
    assert (tmp_path / "report" / "main.png").exists()


@pytest.mark.parametrize("returncode,expected", [(143, True), (-9, True), (1, False)])
def test_preemption_recovery_preserves_attempts_budget_and_unrelated_skips(
    returncode: int, expected: bool
) -> None:
    from src.experiments.utkface.scheduler import restore_preempted_queue

    attempts = [
        {
            "started_at": 10,
            "seconds": 30,
            "yielded_to_foreign_process": True,
            "returncode": returncode,
        }
        for _ in range(3)
    ]
    state = {
        "status": "failed",
        "jobs": {
            "interrupted": {"status": "failed", "attempts": attempts},
            "dependent": {"status": "skipped", "reason": "failed", "attempts": []},
            "scalar": {
                "status": "skipped",
                "reason": "scalar_head_not_competitive",
                "attempts": [],
            },
        },
    }
    restored = restore_preempted_queue(state=state, now=100)
    assert bool(restored) == expected
    assert charged_seconds(state=state, now=100) == 90
    assert state["jobs"]["interrupted"]["attempts"] == attempts
    assert state["jobs"]["scalar"]["status"] == "skipped"
    assert state["status"] == ("running" if expected else "failed")
    assert state["jobs"]["dependent"]["status"] == (
        "pending" if expected else "skipped"
    )


def test_repeated_external_gpu_preemptions_back_off_without_aborting_study(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from src.experiments.utkface import scheduler

    item = job(root=tmp_path, identifier="preempted", lr=1e-4)
    plan = StudyPlan(
        output_dir=tmp_path,
        data_fingerprint="test",
        training_source_sha256={},
        budget_gpu_hours=10,
        max_concurrent=6,
        jobs=(item,),
    )
    children = []

    class Child:
        def __init__(self, command, **kwargs):
            self.pid = 10000 + len(children)
            self.terminated = False
            self.uuid = kwargs["env"]["CUDA_VISIBLE_DEVICES"]
            config = json.loads(
                Path(command[command.index("--config") + 1]).read_text()
            )
            output = Path(config["output_dir"])
            output.mkdir(parents=True)
            if len(children) == 4:
                (output / "result.json").write_text(
                    json.dumps({
                        "config": config,
                        "validation": {"mae_mean": 5},
                        "train_unique_images": 18462,
                    })
                )
            children.append(self)

        def poll(self):
            if self is children[-1] and len(children) == 5:
                return 0
            return 143 if self.terminated else None

    def inventory():
        gpus = fake_inventory()
        for child in children:
            if not child.terminated:
                gpu = next(gpu for gpu in gpus if gpu["uuid"] == child.uuid)
                gpu["pids"] = [child.pid, 99999]
        return gpus

    def terminate(*, attempt):
        child = next(child for child in children if child.pid == attempt["pid"])
        child.terminated = True
        attempt["termination_requested_at"] = time.time()

    monkeypatch.setattr(scheduler, "gpu_inventory", inventory)
    monkeypatch.setattr(scheduler.subprocess, "Popen", Child)
    monkeypatch.setattr(scheduler, "owned_process", lambda **kwargs: True)
    monkeypatch.setattr(scheduler, "process_group", lambda *, pid: pid)
    monkeypatch.setattr(scheduler, "terminate_attempt", terminate)
    monkeypatch.setattr(scheduler.time, "sleep", lambda seconds: None)
    run(
        plan=plan,
        poll_seconds=0,
        coordination=canonical_coordination(root=plan.output_dir),
    )
    state = json.loads((tmp_path / "state.json").read_text())
    assert state["status"] == "completed"
    assert len(state["jobs"]["preempted"]["attempts"]) == 5
    assert len({child.uuid for child in children}) == 5
    assert len(state["gpu_interruptions"]) == 4
    assert all(
        entry["cooldown_until"] > time.time()
        for entry in state["gpu_interruptions"].values()
    )
    assert charged_seconds(state=state, now=time.time()) > 0
