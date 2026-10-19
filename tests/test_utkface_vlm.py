import math

import pytest

from src.experiments.utkface.vlm_eval import (
    parse_age,
    prediction_metrics,
    summarize_records,
)


@pytest.mark.parametrize(
    "content, expected",
    [
        (r"The answer is \boxed{30}.", 30.0),
        (r"\boxed{29.5}", 29.5),
        (r"\boxed{100}", None),
        (r"\boxed{30} then \boxed{not sure}", None),
        (r"<think>\boxed{25}</think>No answer", None),
        (r"<think>30</think>\boxed{31}", 31.0),
        ("Age 30", None),
    ],
)
def test_boxed_final_answer(content, expected):
    assert parse_age(content=content) == expected


def record(*, index, target, values):
    return {
        "index": index,
        "target": target,
        "response": {
            "usage": {"completion_tokens": 10},
            "choices": [
                {
                    "index": i,
                    "finish_reason": "stop",
                    "message": {"content": rf"\boxed{{{value}}}"},
                }
                for i, value in enumerate(values)
            ],
        },
    }


def test_sample_error_and_error_of_sample_mean_differ():
    summary = summarize_records(
        records=[
            record(index=0, target=30, values=[20, 40]),
            record(index=1, target=50, values=[40, 60]),
        ],
        samples=2,
    )
    assert summary["pooled_single_draw"]["mae"] == 10
    assert summary["pooled_single_draw"]["rmse"] == 10
    assert summary["sample_mean_complete8"]["mae"] == 0
    assert summary["sample_median_complete8"]["mae"] == 0
    assert summary["sample_mean_complete8"]["spearman"] == pytest.approx(1)


def test_pooled_rmse_is_not_average_replicate_rmse():
    summary = summarize_records(
        records=[record(index=0, target=30, values=[30, 40])], samples=2
    )
    assert summary["pooled_single_draw"]["rmse"] == pytest.approx(math.sqrt(50))
    assert summary["replicate_mean_and_sd"]["rmse"]["mean"] == 5


def test_invalid_predictions_preserve_coverage_and_penalties():
    summary = summarize_records(
        records=[record(index=0, target=30, values=["invalid", 40])], samples=2
    )
    assert summary["valid_predictions"] == 1
    assert summary["invalid_predictions"] == 1
    assert summary["sample_mean_complete8"]["count"] == 0
    assert summary["sample_mean_available"]["count"] == 1
    assert summary["worst_case_invalid_penalty"]["mae"] == (69 + 10) / 2


def test_constant_predictions_have_undefined_correlation():
    assert prediction_metrics(pairs=[(30, 20), (30, 40)])["spearman"] is None


def test_missing_completion_is_an_error():
    with pytest.raises(ValueError, match="completion indices"):
        summarize_records(records=[record(index=0, target=30, values=[30])], samples=2)
