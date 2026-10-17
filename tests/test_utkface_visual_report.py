import json
import math
from pathlib import Path

import pytest

from src.experiments.utkface.config import canonical_config
from src.experiments.utkface.study import StudyJob, StudyPlan
from src.experiments.utkface.visual_report import build, collect


@pytest.mark.parametrize("head", ["categorical", "scalar"])
def test_report_distinguishes_selected_final_and_metric_optimal_epochs(
    tmp_path: Path, head: str
) -> None:
    output = tmp_path / "attempt"
    output.mkdir()
    config = canonical_config(manifest=tmp_path / "manifest.json", output_dir=output)
    config = type(config).model_validate({
        **config.model_dump(),
        "head": head,
        "epochs": 2,
        "loss": {**config.loss.model_dump(), "method": "mae"},
        "noise": {"mode": "fresh", "std_years": 3, "seed": 2027},
    })
    config_path = tmp_path / "config.json"
    config_path.write_text(config.model_dump_json())
    history = [
        {
            "epoch": epoch,
            "train_count": 64,
            "learning_rate": 1e-4,
            "validation": {
                "mae_mean": mae,
                "median_spearman": rho,
                "label_nll": None if head == "scalar" else nll,
                "mass_within_5": rho,
            },
            "label_noise": {
                "noise_mse_years2": noise_mse,
                "signal_variance_years2": 100,
            },
        }
        for epoch, mae, rho, nll, noise_mse in (
            (1, 4.0, 0.7, 6.0, 1.0),
            (2, 5.0, 0.9, 3.0, 9.0),
        )
    ]
    (output / "history.json").write_text(json.dumps(history))
    initial = {
        "epoch": 0,
        "data_fingerprint": "clean-split",
        "head": head,
        "seed": config.seed,
        "validation": {"mae_mean": 1.0, "label_nll": None if head == "scalar" else 2.0},
    }
    (output / "initial_validation.json").write_text(json.dumps(initial))
    result_path = output / "result.json"
    result_path.write_text(
        json.dumps({
            "best_epoch": 1,
            "data_fingerprint": "clean-split",
            "training_fingerprint": "fixed-subset",
            "test": {"mae_mean": 4.5, "label_nll": None if head == "scalar" else 7.0},
        })
    )
    job = StudyJob(
        identifier="confirmation",
        stage="noise",
        variant="mae",
        priority=0,
        optional=False,
        config=config,
        dependencies=(),
        tune_from=(),
        tuning_rank=0,
        gate_candidate=(),
        gate_reference=(),
        competitive_ratio=1.05,
    )
    plan = StudyPlan(
        output_dir=tmp_path,
        data_fingerprint="clean-split",
        training_source_sha256={"training.py": "frozen"},
        budget_gpu_hours=64,
        max_concurrent=6,
        jobs=(job,),
    )
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(plan.model_dump_json())
    state = {
        "status": "completed",
        "jobs": {
            job.identifier: {
                "status": "completed",
                "result_path": str(result_path),
                "attempts": [
                    {
                        "config_path": str(config_path),
                        "output_dir": str(output),
                        "started_at": 0,
                        "seconds": 10,
                    }
                ],
            }
        },
    }
    (tmp_path / "state.json").write_text(json.dumps(state))
    _, rows = collect(plan_path=plan_path, state=state)
    row = rows[0]
    assert row["initial"] == initial
    assert row["completedEpochs"] == 2 and len(row["curves"]) == 2
    assert row["bestEpoch"] == 1
    assert row["best"]["mae_mean"] == 4
    assert row["final"]["mae_mean"] == 5
    assert row["test"]["mae_mean"] == 4.5
    assert row["optima"]["median_spearman"]["epoch"] == 2
    assert row["optima"]["mass_within_5"]["epoch"] == 2
    assert row["snr"] == pytest.approx(10 * math.log10(100 / 5))
    assert row["noiseSeed"] == 2027
    assert row["curves"][-1][-1] == 2
    if head == "scalar":
        assert row["test"]["label_nll"] is None
        assert "label_nll" not in row["optima"]
    report = tmp_path / "report"
    build(plans=(plan_path,), output_dir=report, coordination_path="")
    assert "__REPORT_DATA__" not in (report / "dashboard.html").read_text()
    assert "final_mae_mean" in (report / "dashboard-runs.csv").read_text()
    assert (report / "dashboard-curves.csv").exists()
