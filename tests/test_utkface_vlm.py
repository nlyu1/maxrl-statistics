import errno
import hashlib
import http.client
import json
import math
import os
import runpy
import urllib.error
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from pathlib import Path
from threading import Event
from types import SimpleNamespace

import pytest

from src.experiments.utkface.scheduler import gpu_is_free
from src.experiments.utkface.vlm_eval import (
    GPUProcessException,
    SharedGPURuntime,
    SharedServingPolicy,
    evaluate_images,
    load_shared_policy,
    parse_age,
    prediction_metrics,
    query_image,
    shared_runtime,
    summarize_records,
    tune,
    wait_for_server,
    worker,
)
from src.experiments.utkface.vlm_queue import (
    DOWNLOAD_SCRIPT,
    gpu_inventory_for_plan,
    owned_worker,
    process_matches_exception,
    shared_gpu_is_ready,
    shared_stop_reason,
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


@pytest.fixture
def shared_grant():
    return SharedGPURuntime(
        model_identifier="0p8b_posttrained",
        gpu_uuid="approved-gpu",
        resident_processes=(),
        baseline_memory_mib=20000,
        total_memory_mib=46068,
        gpu_memory_utilization=0.35,
        max_num_seqs=64,
        client_concurrencies=(8, 16),
    )


@pytest.mark.parametrize(
    "elapsed,accepted,valid",
    [(599, True, False), (600, False, False), (600, True, True)],
)
def test_shared_policy_requires_completed_observation(
    tmp_path, shared_grant, elapsed, accepted, valid
):
    observed = tmp_path / "observation.json"
    observed.write_text(
        json.dumps({
            "accepted": accepted,
            "elapsed_seconds": elapsed,
        "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
            "gpus": [
                {
                    "index": i,
                    "uuid": "approved-gpu" if i == 1 else str(i),
                    "max_utilization": 0,
                }
                for i in (1, 2, 3, 5, 9)
            ],
        })
    )
    policy = SharedServingPolicy(
        observation=observed,
        observation_sha256=hashlib.sha256(observed.read_bytes()).hexdigest(),
        minimum_headroom_mib=6144,
        maximum_resident_cpu_fraction=0.25,
        grants=(shared_grant,),
    )
    (tmp_path / "shared-serving.json").write_text(policy.model_dump_json())
    if not valid:
        with pytest.raises(ValueError, match="ten-minute"):
            load_shared_policy(root=tmp_path)
        return
    loaded = load_shared_policy(root=tmp_path)
    assert loaded == policy
    assert shared_runtime(policy=loaded, identifier="4b_posttrained") is None
    another_boot = json.loads(observed.read_text())
    another_boot["boot_id"] = "different-boot"
    observed.write_text(json.dumps(another_boot))
    wrong_boot = policy.model_copy(update={
        "observation_sha256": hashlib.sha256(observed.read_bytes()).hexdigest()
    })
    (tmp_path / "shared-serving.json").write_text(wrong_boot.model_dump_json())
    with pytest.raises(ValueError, match="another server boot"):
        load_shared_policy(root=tmp_path)
    (tmp_path / "shared-serving.json").write_text(policy.model_dump_json())
    observed.write_text("changed")
    with pytest.raises(ValueError, match="observation changed"):
        load_shared_policy(root=tmp_path)


@pytest.mark.parametrize(
    "uuid,pids,memory,utilization,expected",
    [
        ("approved-gpu", [], 20000, 0, True),
        ("approved-gpu", [123], 20000, 0, False),
        ("approved-gpu", [], 20257, 0, False),
        ("approved-gpu", [], 20000, 6, False),
        ("different-gpu", [], 20000, 0, False),
    ],
)
def test_shared_dispatch_checks_residents_activity_and_memory(
    shared_grant, uuid, pids, memory, utilization, expected
):
    gpu = {"uuid": uuid, "pids": pids, "memory_mib": memory, "utilization": utilization}
    assert (
        shared_gpu_is_ready(
            gpu=gpu,
            grant=shared_grant,
            policy=SimpleNamespace(minimum_headroom_mib=6144),
        )
        is expected
    )


@pytest.mark.parametrize(
    "memory,cpu_seconds,expected",
    [
        (40000, 11, "shared_gpu_memory_headroom"),
        (35000, 16, "resident_simulation_cpu_activity"),
        (35000, 11, ""),
    ],
)
def test_shared_worker_yields_to_memory_or_resident_activity(
    monkeypatch, shared_grant, memory, cpu_seconds, expected
):
    monkeypatch.setattr(
        "src.experiments.utkface.vlm_queue.resident_cpu_seconds",
        lambda **kw: {"1": cpu_seconds},
    )
    attempt = {"resident_cpu_seconds": {"1": 10}, "resident_cpu_sampled_at": 100}
    assert (
        shared_stop_reason(
            gpu={"memory_mib": memory},
            grant=shared_grant,
            policy=SimpleNamespace(
                minimum_headroom_mib=6144, maximum_resident_cpu_fraction=0.25
            ),
            attempt=attempt,
            now=110,
        )
        == expected
    )


@pytest.mark.parametrize(
    "temperature,top_p,presence_penalty", [(0.7, 0.8, 0.0), (1.0, 0.95, 1.5)]
)
def test_request_sends_pixels_without_age_filename_or_label(
    tmp_path, monkeypatch, temperature, top_p, presence_penalty
):
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
        temperature=temperature,
        top_p=top_p,
        top_k=20,
        presence_penalty=presence_penalty,
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
    assert request["temperature"] == temperature
    assert request["top_p"] == top_p
    assert request["top_k"] == 20
    assert request["min_p"] == 0.0
    assert request["presence_penalty"] == presence_penalty
    assert request["repetition_penalty"] == 1.0
    assert result["sampling"]["presence_penalty"] == presence_penalty
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
        gpu_memory_utilization=0.8,
        max_num_seqs=256,
        client_concurrencies=(32, 64),
        max_model_len=12288,
        max_num_batched_tokens=8192,
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


@pytest.mark.parametrize("cap_passes, expected_cap", [(True, 4096), (False, 8192)])
def test_tuning_reports_exhausted_caps_and_fills_http_limits(
    tmp_path, monkeypatch, cap_passes, expected_cap
):
    calls = []

    def evaluate(**kwargs):
        calls.append(kwargs)
        records = [
            record(index=i, target=30, values=[30] * kwargs["samples"])
            for i in kwargs["indices"]
        ]
        if kwargs["samples"] == 2 and not cap_passes:
            for response in records:
                for choice in response["response"]["choices"]:
                    choice["message"]["content"] = ""
                    choice["finish_reason"] = "length"
        seconds = 1 if kwargs["concurrency"] == 64 else 2
        return records, seconds

    monkeypatch.setattr("src.experiments.utkface.vlm_eval.evaluate_images", evaluate)
    monkeypatch.setattr(
        "src.experiments.utkface.vlm_eval.utilization_monitor",
        lambda **kwargs: nullcontext([]),
    )
    result = tune(
        root=tmp_path,
        plan=SimpleNamespace(
            token_caps_direct=(1024,),
            token_caps_thinking=(4096, 8192),
            pilot_indices=tuple(range(64)),
            client_concurrencies=(32, 64),
            samples=8,
        ),
        spec=SimpleNamespace(identifier="thinking", mode="thinking_on"),
        payload={},
        port=8000,
        gpu_uuid="test-gpu",
    )
    assert result["max_tokens"] == expected_cap
    assert result["cap_criterion_met"] is cap_passes
    assert result["token_budget_limited"] is not cap_passes
    assert result["concurrency"] == 64
    throughput_calls = [call for call in calls if "concurrency_" in call["output"].name]
    assert [call["concurrency"] for call in throughput_calls] == [32, 64, 64, 32]
    assert all(len(call["indices"]) >= call["concurrency"] for call in throughput_calls)
    assert all(call["max_tokens"] == expected_cap for call in throughput_calls)


def test_evaluation_error_does_not_wait_for_blocked_http_requests(
    tmp_path, monkeypatch
):
    release = Event()
    blocked = Event()

    def query(**kwargs):
        if kwargs["index"] == 0:
            assert blocked.wait(timeout=5)
            raise OSError(errno.ENOSPC, "No space left on device")
        blocked.set()
        assert release.wait(timeout=10)
        return record(index=1, target=30, values=[30, 30])

    monkeypatch.setattr("src.experiments.utkface.vlm_eval.query_image", query)
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(
            evaluate_images,
            plan=SimpleNamespace(),
            spec=SimpleNamespace(),
            payload={},
            indices=(0, 1),
            port=8000,
            samples=2,
            max_tokens=256,
            concurrency=2,
            output=tmp_path,
        )
        try:
            with pytest.raises(OSError) as caught:
                future.result(timeout=3)
            assert caught.value.errno == errno.ENOSPC
        finally:
            release.set()
