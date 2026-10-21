import json
import os
from pathlib import Path

import pytest

from src.experiments.utkface import lr_reruns
from src.experiments.utkface.config import RunConfig, canonical_config
from src.experiments.utkface.lr_reruns import (
    GPUAdmission,
    admitted_inventory,
    build_plan,
    capture_placeholder,
    shared_learning_rate,
)
from src.experiments.utkface.scheduler import gpu_is_free
from src.experiments.utkface.study import StudyJob, StudyPlan
from src.experiments.utkface.vlm_eval import GPUProcessException


@pytest.mark.parametrize(
    "method,size,head,schedule,expected",
    [
        ("tailrl_population", 0, "categorical", "cosine", 1e-3),
        ("tailrl", 0, "categorical", "cosine", 1e-3),
        ("binary_maxrl_population", 0, "categorical", "cosine", 1e-3),
        ("binary_maxrl", 0, "categorical", "cosine", 1e-3),
        ("ce", 4096, "categorical", "cosine", 1e-3),
        ("stat_maxrl_mae_population", 0, "categorical", "cosine", 1e-3),
        ("stat_maxrl_mae", 0, "categorical", "cosine", 1e-3),
        ("stat_maxrl_mse_population", 0, "categorical", "cosine", 3e-4),
        ("stat_maxrl_mse_population", 1024, "categorical", "cosine", 1e-3),
        ("grpo_mae_population", 0, "categorical", "cosine", 3e-4),
        ("grpo_mae_population", 4096, "categorical", "cosine", 1e-3),
        ("grpo_mae_population", 1024, "categorical", "cosine", 3e-4),
        ("grpo_mse", 0, "categorical", "cosine", 3e-4),
        ("rloo_mae", 0, "categorical", "cosine", 1e-4),
        ("rloo_mse", 0, "categorical", "cosine", 1e-3),
        ("mae", 4096, "categorical", "cosine", 1e-3),
        ("mse", 1024, "categorical", "cosine", 1e-3),
        ("mse", 4096, "scalar", "cosine", 1e-3),
        ("ce", 4096, "digits", "cosine", 3e-3),
        ("ce", 4096, "categorical", "flat", 3e-3),
    ],
)
def test_shared_rates_preserve_explicit_regime_exceptions(
    method: str, size: int, head: str, schedule: str, expected: float
) -> None:
    base = canonical_config(manifest=Path("unused"), output_dir=Path("unused"))
    config = RunConfig.model_validate({
        **base.model_dump(),
        "learning_rate": 3e-3,
        "train_limit": size,
        "head": head,
        "loss": {**base.loss.model_dump(), "method": method},
        "schedule": {**base.schedule.model_dump(), "lr_schedule": schedule},
    })
    assert shared_learning_rate(config=config) == expected


def test_plan_uses_executed_configs_and_preserves_historical_records(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "original"
    root.mkdir()
    base = canonical_config(manifest=root / "manifest.json", output_dir=root / "runs")
    jobs, records = [], {}
    for identifier, actual_lr in (("change", 3e-3), ("keep", 1e-3)):
        config = base.model_copy(update={"evaluate_test": True})
        jobs.append(
            StudyJob(
                identifier=identifier,
                stage="main",
                variant="ce",
                priority=30,
                optional=True,
                config=config,
                dependencies=("old_pilot",),
                tune_from=("old_pilot",),
                tuning_rank=0,
                gate_candidate=("old_gate",),
                gate_reference=("old_reference",),
                competitive_ratio=1.05,
            )
        )
        result_path = root / f"{identifier}.json"
        result_path.write_text(
            json.dumps({
                "config": {
                    **config.model_dump(mode="json"),
                    "learning_rate": actual_lr,
                },
                "data_fingerprint": "clean-split",
                "training_fingerprint": "fixed-subset",
                "seconds": 360,
            })
        )
        records[identifier] = {"status": "completed", "result_path": str(result_path)}
    source = StudyPlan(
        output_dir=root,
        data_fingerprint="clean-split",
        training_source_sha256={},
        budget_gpu_hours=12,
        max_concurrent=6,
        jobs=tuple(jobs),
    )
    plan_path = root / "plan.json"
    plan_path.write_text(source.model_dump_json())
    state_path = root / "state.json"
    state_path.write_text(json.dumps({"status": "completed", "jobs": records}))
    before = {path: path.read_bytes() for path in root.iterdir()}
    monkeypatch.setattr(lr_reruns, "source_hashes", lambda **kwargs: {})
    plan, audit = build_plan(
        source_plans=(plan_path,), root=tmp_path / "reruns", max_concurrent=6
    )
    assert len(plan.jobs) == 1 and audit["reused_count"] == 1
    item = plan.jobs[0]
    assert item.identifier == "shared_lr_change" and not item.optional
    assert item.config.learning_rate == 1e-3 and item.config.precision == "bf16"
    assert (
        item.dependencies
        == item.tune_from
        == item.gate_candidate
        == item.gate_reference
        == ()
    )
    assert item.config.output_dir == plan.output_dir / "runs" / item.identifier
    assert audit["changes"][0]["old_learning_rate"] == 3e-3
    assert audit["changes"][0]["training_fingerprint"] == "fixed-subset"
    assert all(path.read_bytes() == contents for path, contents in before.items())


@pytest.mark.parametrize(
    "verified,pids,memory,utilization,free",
    [
        (True, [101], 439, 0, True),
        (False, [101], 439, 0, False),
        (True, [101, 202], 439, 0, False),
        (True, [101], 1500, 0, False),
        (True, [101], 439, 50, False),
    ],
)
def test_gpu_admission_requires_live_identity_and_retains_usage_limits(
    monkeypatch: pytest.MonkeyPatch,
    verified: bool,
    pids: list[int],
    memory: int,
    utilization: int,
    free: bool,
) -> None:
    exception = GPUProcessException(
        pid=101,
        uid=1,
        start_time_ticks=1,
        command_sha256="0" * 64,
        gpu_uuid="gpu-9",
        boot_id="boot",
    )
    admission = GPUAdmission(gpu_uuids=("gpu-9",), ignored_gpu_processes=(exception,))
    gpu = {
        "index": 9,
        "uuid": "gpu-9",
        "pids": pids,
        "memory_mib": memory,
        "utilization": utilization,
    }
    monkeypatch.setattr(
        lr_reruns, "process_matches_exception", lambda **kwargs: verified
    )
    monkeypatch.setattr(
        lr_reruns.scheduler,
        "gpu_inventory",
        lambda: [
            {**gpu, "index": 0, "uuid": "gpu-0"},
            gpu,
        ],
    )
    inventory = admitted_inventory(admission=admission)
    assert len(inventory) == 1 and inventory[0]["index"] == 9
    assert gpu_is_free(gpu=inventory[0]) is free
    assert (
        inventory[0]["memory_mib"] == memory
        and inventory[0]["utilization"] == utilization
    )
    assert gpu["pids"] == pids


def test_capture_rejects_an_unapproved_process_command() -> None:
    with pytest.raises(ValueError, match="inline placeholder"):
        capture_placeholder(pid=os.getpid(), gpu_uuid="unused")
