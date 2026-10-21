import errno
import hashlib
import http.client
import json
import math
import os
import runpy
import urllib.error
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.experiments.utkface.scheduler import gpu_is_free
from src.experiments.utkface.vlm_eval import (
    GPUProcessException,
    parse_age,
    prediction_metrics,
    query_image,
    summarize_records,
    wait_for_server,
    worker,
)
from src.experiments.utkface.vlm_queue import (
    DOWNLOAD_SCRIPT,
    gpu_inventory_for_plan,
    owned_worker,
    process_matches_exception,
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
    monkeypatch.setattr(
        "src.experiments.utkface.vlm_eval.urllib.request.urlopen", urlopen
    )
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


@pytest.fixture
def gpu_holder_exception():
    process = Path(f"/proc/{os.getpid()}")
    fields = (process / "stat").read_text().rsplit(")", 1)[1].split()
    return GPUProcessException(
        pid=os.getpid(),
        uid=os.getuid(),
        start_time_ticks=int(fields[19]),
        command_sha256=hashlib.sha256((process / "cmdline").read_bytes()).hexdigest(),
        gpu_uuid="approved-gpu",
        boot_id=Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
    )


def test_process_exception_requires_verified_live_identity(gpu_holder_exception):
    assert process_matches_exception(exception=gpu_holder_exception)


@pytest.mark.parametrize(
    "field, value",
    [
        ("uid", -1),
        ("start_time_ticks", 0),
        ("command_sha256", "0" * 64),
        ("boot_id", "another-boot"),
        ("pid", 2**31 - 1),
    ],
)
def test_changed_or_missing_process_is_not_exempt(gpu_holder_exception, field, value):
    exception = gpu_holder_exception.model_copy(update={field: value})
    assert not process_matches_exception(exception=exception)


@pytest.mark.parametrize(
    "uuid, other_pids, memory_mib, utilization, expected_free",
    [
        ("approved-gpu", [], 439, 0, True),
        ("approved-gpu", [9999999], 439, 0, False),
        ("approved-gpu", [], 2048, 0, False),
        ("approved-gpu", [], 439, 10, False),
        ("another-gpu", [], 439, 0, False),
    ],
)
def test_gpu_exceptions_preserve_other_process_and_usage_checks(
    monkeypatch,
    gpu_holder_exception,
    uuid,
    other_pids,
    memory_mib,
    utilization,
    expected_free,
):
    gpu = {
        "uuid": uuid,
        "index": 0,
        "pids": [gpu_holder_exception.pid, *other_pids],
        "memory_mib": memory_mib,
        "utilization": utilization,
    }
    monkeypatch.setattr(
        "src.experiments.utkface.vlm_queue.gpu_inventory", lambda: [gpu]
    )
    inventory = gpu_inventory_for_plan(
        plan=SimpleNamespace(ignored_gpu_processes=(gpu_holder_exception,))
    )
    assert gpu_is_free(gpu=inventory[0]) is expected_free
    assert gpu["pids"] == [gpu_holder_exception.pid, *other_pids]
    if uuid == gpu_holder_exception.gpu_uuid:
        assert inventory[0]["ignored_pids"] == [gpu_holder_exception.pid]


def test_worker_releases_container_when_log_storage_is_full(tmp_path, monkeypatch):
    plan = SimpleNamespace(
        models=(SimpleNamespace(identifier="model"),),
        evaluations=(),
        manifest=tmp_path / "manifest.json",
        fingerprint="frozen",
    )
    stopped = []
    original_write_text = Path.write_text

    def write_text(path, *args, **kwargs):
        if path.name == "vllm.log":
            raise OSError(errno.ENOSPC, "No space left on device")
        return original_write_text(path, *args, **kwargs)

    monkeypatch.setattr("src.experiments.utkface.vlm_eval.load_plan", lambda **kw: plan)
    monkeypatch.setattr(
        "src.experiments.utkface.vlm_eval.read_manifest",
        lambda **kw: {"fingerprint": "frozen"},
    )
    monkeypatch.setattr(
        "src.experiments.utkface.vlm_eval.docker_command", lambda **kw: ["docker"]
    )
    monkeypatch.setattr(
        "src.experiments.utkface.vlm_eval.subprocess.run",
        lambda *args, **kw: SimpleNamespace(stdout="logs", stderr=""),
    )
    monkeypatch.setattr(
        "src.experiments.utkface.vlm_eval.wait_for_server", lambda **kw: None
    )
    monkeypatch.setattr(
        "src.experiments.utkface.vlm_eval.stop_container",
        lambda **kw: stopped.append(kw["container"]),
    )
    monkeypatch.setattr(
        "src.experiments.utkface.vlm_eval.signal.signal", lambda *args: None
    )
    monkeypatch.setattr(Path, "write_text", write_text)
    with pytest.raises(OSError) as caught:
        worker(
            root=tmp_path,
            identifier="model",
            gpu_uuid="test-gpu",
            port=8000,
            container="study-worker",
        )
    assert caught.value.errno == errno.ENOSPC
    assert stopped == ["study-worker"]
