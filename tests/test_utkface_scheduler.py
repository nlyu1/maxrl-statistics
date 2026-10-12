import json
import os
import time
from pathlib import Path

import pytest

from src.experiments.utkface.config import canonical_config
from src.experiments.utkface.data import manifest_fingerprint
from src.experiments.utkface.scheduler import charged_seconds, gpu_is_free, run
from src.experiments.utkface.study import StudyJob, StudyPlan, make_plan


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


def test_scheduler_selects_validation_lr_respects_six_gpu_cap_and_exports_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
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
    run(plan=plan, poll_seconds=0)
    state = json.loads((tmp_path / "state.json").read_text())
    assert state["status"] == "completed" and max(peaks) == 6
    assert selected[-1]["learning_rate"] == pytest.approx(8e-4)
    assert all(item["status"] == "completed" for item in state["jobs"].values())


def test_scheduler_does_not_start_a_job_exceeding_remaining_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
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
    run(plan=plan, poll_seconds=0)
    state = json.loads((tmp_path / "state.json").read_text())
    assert state["status"] == "budget_exhausted"
    assert state["jobs"]["cannot_fit"]["attempts"] == []


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
    run(plan=plan, poll_seconds=0)
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
