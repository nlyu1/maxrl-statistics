import http.client
import json
import math
import os
import runpy
import urllib.error
from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from src.experiments.utkface.vlm_eval import (
    parse_age,
    prediction_metrics,
    query_image,
    summarize_records,
    wait_for_server,
)
from src.experiments.utkface.vlm_queue import DOWNLOAD_SCRIPT, owned_worker


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


def test_request_sends_pixels_without_age_filename_or_label(tmp_path, monkeypatch):
    source = tmp_path / "37_0_0_123.jpg"
    source.write_bytes(b"image-pixels")
    captured = []

    def fake_request(*, url, payload):
        captured.append(payload)
        return {"model": payload["model"], "choices": [], "usage": {}}

    monkeypatch.setattr("src.experiments.utkface.vlm_eval.request_json", fake_request)
    plan = SimpleNamespace(
        manifest=tmp_path / "manifest.json",
        prompt="Estimate age in years.",
        temperature=0.7,
        top_p=0.8,
        top_k=20,
        presence_penalty=0.0,
        sampling_seed=20261020,
    )
    spec = SimpleNamespace(model_identifier="model", mode="thinking_on")
    result = query_image(
        plan=plan,
        spec=spec,
        payload={"records": [{"relative_path": source.name, "age": 37}]},
        index=0,
        port=8000,
        samples=8,
        max_tokens=256,
    )
    request = captured[0]
    assert request["n"] == 8
    assert request["chat_template_kwargs"] == {"enable_thinking": True}
    assert request["messages"][0]["content"][0]["image_url"]["url"].startswith(
        "data:image/jpeg;base64,"
    )
    assert source.name not in str(request)
    assert "target" not in request
    assert result["target"] == 37


def test_download_helper_is_valid_python_and_handles_json(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "huggingface_hub.snapshot_download", lambda **kwargs: "/cached/snapshot"
    )
    (tmp_path / "plan.json").write_text(
        json.dumps({
            "models": [
                {
                    "identifier": "model",
                    "repository": "Qwen/model",
                    "revision": "pinned",
                }
            ]
        })
    )
    source = DOWNLOAD_SCRIPT.replace("Path('/study')", f"Path({str(tmp_path)!r})")
    helper = tmp_path / "download_helper.py"
    helper.write_text(source)
    runpy.run_path(str(helper))
    assert (
        json.loads((tmp_path / "downloads.json").read_text())["model"]["snapshot"]
        == "/cached/snapshot"
    )


def test_process_ownership_rejects_unrelated_pid(tmp_path):
    assert not owned_worker(pid=os.getpid(), root=tmp_path)


@pytest.mark.parametrize(
    "failure",
    [
        ConnectionResetError("Container port is not ready"),
        http.client.RemoteDisconnected("Server is not ready"),
        urllib.error.URLError("Connection refused"),
        TimeoutError("Health request timed out"),
    ],
)
def test_readiness_retries_transient_connections(monkeypatch, failure):
    responses = iter([failure, nullcontext(SimpleNamespace(status=200))])
    sleeps = []

    def urlopen(*args, **kwargs):
        response = next(responses)
        if isinstance(response, Exception):
            raise response
        return response

    monkeypatch.setattr(
        "src.experiments.utkface.vlm_eval.subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(stdout="true"),
    )
    monkeypatch.setattr("src.experiments.utkface.vlm_eval.urllib.request.urlopen", urlopen)
    monkeypatch.setattr("src.experiments.utkface.vlm_eval.time.sleep", sleeps.append)
    wait_for_server(port=8000, container="study-worker")
    assert sleeps == [3]


def test_readiness_still_rejects_exited_container(monkeypatch):
    monkeypatch.setattr(
        "src.experiments.utkface.vlm_eval.subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(stdout="false"),
    )
    with pytest.raises(RuntimeError, match="exited during startup"):
        wait_for_server(port=8000, container="study-worker")
