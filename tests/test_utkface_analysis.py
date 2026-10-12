from __future__ import annotations

import polars as pl

from src.experiments.utkface.analyze import REPORT_METRICS, aggregate


def test_hyperparameter_selection_uses_validation_and_preserves_seed_variation() -> (
    None
):
    rows = []
    for method in ("ce", "tailrl_population"):
        for learning_rate, validation, test in ((0.001, 8.0, 20.0), (0.01, 9.0, 1.0)):
            for seed, offset in ((42, -1.0), (43, 1.0)):
                rows.append({
                    "method": method,
                    "learning_rate": learning_rate,
                    "temperature_years": 5.0,
                    "num_rollouts": 64,
                    "degree": 64,
                    "reward": "mae",
                    "seed": seed,
                    "validation_mean_mae": validation + offset,
                    **{f"test_{name}": test + offset for name in REPORT_METRICS},
                })
    all_parameters, selected = aggregate(runs=pl.DataFrame(rows))
    assert all_parameters.height == 4
    assert selected.height == 2
    assert selected["learning_rate"].to_list() == [0.001, 0.001]
    assert selected["test_mean_mae_mean"].to_list() == [20.0, 20.0]
    assert selected["seeds"].to_list() == [2, 2]
    assert selected["test_mean_mae_sd"].to_list() == [2**0.5, 2**0.5]
