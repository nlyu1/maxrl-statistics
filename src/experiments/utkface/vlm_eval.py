import argparse
import base64
import hashlib
import json
import math
import re
import signal
import statistics
import subprocess
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from pathlib import Path
from typing import Literal

import numpy as np
import polars as pl
from huggingface_hub import HfApi, hf_hub_download
from pydantic import BaseModel, ConfigDict, Field
from scipy.stats import spearmanr
from sklearn.model_selection import train_test_split

from src.experiments.utkface.data import read_manifest
from src.experiments.utkface.run import write_json


class ModelSpec(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    identifier: str
    size: str
    kind: Literal["base", "posttrained"]
    repository: str
    revision: str
    template_file: str
    template_sha256: str


class EvaluationSpec(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    identifier: str
    model_identifier: str
    mode: Literal["base", "thinking_off", "thinking_on"]


class GPUProcessException(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    pid: int = Field(gt=0)
    uid: int = Field(ge=0)
    start_time_ticks: int = Field(ge=0)
    command_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    gpu_uuid: str
    boot_id: str


class SharedGPURuntime(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    model_identifier: Literal["0p8b_posttrained", "2b_posttrained"]
    gpu_uuid: str
    resident_processes: tuple[GPUProcessException, ...]
    baseline_memory_mib: int = Field(ge=0)
    total_memory_mib: int = Field(gt=0)
    gpu_memory_utilization: float = Field(gt=0, le=0.4)
    max_num_seqs: int = Field(gt=0, le=64)
    client_concurrencies: tuple[int, ...]


class SharedServingPolicy(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    observation: Path
    observation_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    minimum_headroom_mib: int = Field(ge=4096)
    maximum_resident_cpu_fraction: float = Field(gt=0, le=0.25)
    grants: tuple[SharedGPURuntime, ...]


class EvaluationPlan(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    manifest: Path
    fingerprint: str
    image: str = Field(pattern=r"^vllm/vllm-openai@sha256:[0-9a-f]{64}$")
    cache_volume: str
    models: tuple[ModelSpec, ...]
    evaluations: tuple[EvaluationSpec, ...]
    indices: tuple[int, ...]
    pilot_indices: tuple[int, ...]
    subset_seed: int
    sampling_seed: int
    samples: int = Field(ge=2)
    prompt: str
    temperature: float
    top_p: float
    top_k: int
    presence_penalty: float
    token_caps_direct: tuple[int, ...]
    token_caps_thinking: tuple[int, ...]
    client_concurrencies: tuple[int, ...]
    max_num_seqs: int
    max_model_len: int
    max_num_batched_tokens: int
    gpu_memory_utilization: float
    max_parallel_models: int
    poll_seconds: float
    dispatch_lock: Path
    ignored_gpu_processes: tuple[GPUProcessException, ...]
    source_hashes: dict[str, str]


SOURCE_FILES = (
    "src/experiments/utkface/vlm_eval.py",
    "src/experiments/utkface/vlm_queue.py",
    "src/experiments/utkface/scheduler.py",
)

FOLLOWUP_PROMPT = (
    "Estimate the person's age in years from their appearance in this photo. "
    "Give your single best estimate, even if uncertain. "
    r"Your final answer must be exactly one integer from 0 to 99 inside \boxed{}, "
    "with no other text or range."
)


def source_hashes() -> dict[str, str]:
    return {
        name: hashlib.sha256(Path(name).read_bytes()).hexdigest()
        for name in SOURCE_FILES
    }


def load_plan(*, root: Path) -> EvaluationPlan:
    plan = EvaluationPlan.model_validate_json((root / "plan.json").read_text())
    if source_hashes() != plan.source_hashes:
        raise ValueError("Evaluation sources differ from the frozen plan")
    for model in plan.models:
        if (
            hashlib.sha256((root / model.template_file).read_bytes()).hexdigest()
            != model.template_sha256
        ):
            raise ValueError("Chat template differs from the pinned checkpoint")
    return plan


def load_shared_policy(*, root: Path) -> SharedServingPolicy | None:
    path = root / "shared-serving.json"
    if not path.exists():
        return None
    policy = SharedServingPolicy.model_validate_json(path.read_text())
    content = policy.observation.read_bytes()
    if hashlib.sha256(content).hexdigest() != policy.observation_sha256:
        raise ValueError("Shared-GPU idle observation changed")
    observed = json.loads(content)
    if not observed["accepted"] or observed["elapsed_seconds"] < 600:
        raise ValueError("Shared GPUs require a completed ten-minute idle observation")
    if {row["index"] for row in observed["gpus"]} != {1, 2, 3, 5, 9}:
        raise ValueError("Idle observation must cover all five authorized candidates")
    if any(row["max_utilization"] > 5 for row in observed["gpus"]):
        raise ValueError("A candidate GPU was active during the observation")
    if len({grant.model_identifier for grant in policy.grants}) != len(
        policy.grants
    ) or len({grant.gpu_uuid for grant in policy.grants}) != len(policy.grants):
        raise ValueError("Each shared model requires a distinct GPU")
    observed_uuids = {row["uuid"] for row in observed["gpus"]}
    if any(grant.gpu_uuid not in observed_uuids for grant in policy.grants):
        raise ValueError("Shared GPU was not observed")
    if any(
        process.gpu_uuid != grant.gpu_uuid
        for grant in policy.grants
        for process in grant.resident_processes
    ):
        raise ValueError("Resident exception differs from its granted GPU")
    return policy


def shared_runtime(
    *, policy: SharedServingPolicy | None, identifier: str
) -> SharedGPURuntime | None:
    if policy is None:
        return None
    return next(
        (grant for grant in policy.grants if grant.model_identifier == identifier), None
    )


def prepare(*, root: Path, manifest: Path, image: str) -> None:
    if root.exists():
        raise FileExistsError(root)
    payload = read_manifest(path=manifest)
    validation = sorted(payload["splits"]["val"])
    if len(validation) != 2308:
        raise ValueError("Expected the cleaned 2308-image validation split")
    seed = 20261020
    indices, remaining = train_test_split(
        validation,
        train_size=1024,
        random_state=seed,
        stratify=[payload["records"][i]["age"] // 10 for i in validation],
    )
    pilot, _ = train_test_split(
        remaining,
        train_size=64,
        random_state=seed,
        stratify=[payload["records"][i]["age"] // 10 for i in remaining],
    )
    root.mkdir(parents=True)
    (root / "templates").mkdir()
    models, evaluations = [], []
    api = HfApi()
    for size in ("0.8B", "2B", "4B"):
        paired_repository = f"Qwen/Qwen3.5-{size}"
        paired_revision = api.model_info(paired_repository).sha
        template_path = hf_hub_download(
            paired_repository, "chat_template.jinja", revision=paired_revision
        )
        template_file = f"templates/{size}.jinja"
        (root / template_file).write_bytes(Path(template_path).read_bytes())
        for kind in ("base", "posttrained"):
            repository = paired_repository + ("-Base" if kind == "base" else "")
            identifier = size.lower().replace(".", "p") + "_" + kind
            models.append(
                ModelSpec(
                    identifier=identifier,
                    size=size,
                    kind=kind,
                    repository=repository,
                    revision=api.model_info(repository).sha,
                    template_file=template_file,
                    template_sha256=hashlib.sha256(
                        Path(template_path).read_bytes()
                    ).hexdigest(),
                )
            )
            modes = ("base",) if kind == "base" else ("thinking_off", "thinking_on")
            for mode in modes:
                evaluations.append(
                    EvaluationSpec(
                        identifier=size.lower().replace(".", "p") + "_" + mode,
                        model_identifier=identifier,
                        mode=mode,
                    )
                )
    plan = EvaluationPlan(
        manifest=manifest,
        fingerprint=payload["fingerprint"],
        image=image,
        cache_volume="utkface-vlm-qwen35-20261020",
        models=tuple(models),
        evaluations=tuple(evaluations),
        indices=tuple(sorted(indices)),
        pilot_indices=tuple(
            int(i) for i in np.random.default_rng(seed).permutation(pilot)
        ),
        subset_seed=seed,
        sampling_seed=seed,
        samples=8,
        prompt=r"What is the person's age in years? Put your final answer in \boxed{}.",
        temperature=0.7,
        top_p=0.8,
        top_k=20,
        presence_penalty=0.0,
        token_caps_direct=(128, 256, 512),
        token_caps_thinking=(256, 512, 1024, 2048),
        client_concurrencies=(8, 32),
        max_num_seqs=128,
        max_model_len=4096,
        max_num_batched_tokens=8192,
        gpu_memory_utilization=0.8,
        max_parallel_models=3,
        poll_seconds=10,
        dispatch_lock=Path("/tmp/utkface-extension-20261014/global-dispatch.lock"),
        ignored_gpu_processes=(),
        source_hashes=source_hashes(),
    )
    write_json(path=root / "plan.json", payload=plan.model_dump(mode="json"))
    write_json(
        path=root / "subset.json",
        payload={
            "manifest": str(manifest),
            "fingerprint": plan.fingerprint,
            "seed": seed,
            "split": "val",
            "indices": plan.indices,
            "pilot_indices": plan.pilot_indices,
            "age_decades": {
                str(decade): sum(
                    payload["records"][i]["age"] // 10 == decade for i in indices
                )
                for decade in range(10)
            },
        },
    )
    write_json(
        path=root / "provenance.json",
        payload={
            "commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip(),
            "base_chat_template": "Paired post-trained checkpoint's official template; enable_thinking=False",
            "sampling": "Identical temperature, top-p, top-k and presence penalty in all nine configurations",
            "pilot_selection": "Token limits and throughput only, using validation images disjoint from the 1024-image evaluation subset",
            "created_at": time.time(),
        },
    )


def prepare_followup(*, root: Path, reference: Path, cache_volume: str) -> None:
    if root.exists():
        raise FileExistsError(root)
    reference_plan = EvaluationPlan.model_validate_json(
        (reference / "plan.json").read_text()
    )
    manifest = reference_plan.manifest.resolve()
    payload = read_manifest(path=manifest)
    if payload["fingerprint"] != reference_plan.fingerprint:
        raise ValueError("Reference dataset fingerprint changed")
    if (
        len(reference_plan.indices) != 1024
        or len(set(reference_plan.indices)) != 1024
        or len(reference_plan.pilot_indices) < 64
        or set(reference_plan.indices) & set(reference_plan.pilot_indices)
        or not set(reference_plan.indices).issubset(payload["splits"]["val"])
    ):
        raise ValueError("Reference evaluation and pilot subsets are not valid")
    plan = EvaluationPlan.model_validate({
        **reference_plan.model_dump(),
        "manifest": manifest,
        "cache_volume": cache_volume,
        "prompt": FOLLOWUP_PROMPT,
        "token_caps_direct": (1024,),
        "token_caps_thinking": (4096, 8192),
        "client_concurrencies": (32, 64),
        "max_num_seqs": 256,
        "max_model_len": 12288,
        "source_hashes": source_hashes(),
    })
    root.mkdir(parents=True)
    (root / "templates").mkdir()
    for model in plan.models:
        content = (reference / model.template_file).read_bytes()
        if hashlib.sha256(content).hexdigest() != model.template_sha256:
            raise ValueError("Reference template differs from the pinned checkpoint")
        (root / model.template_file).write_bytes(content)
    write_json(path=root / "plan.json", payload=plan.model_dump(mode="json"))
    (root / "subset.json").write_bytes((reference / "subset.json").read_bytes())
    write_json(
        path=root / "provenance.json",
        payload={
            "commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip(),
            "reference_root": str(reference.resolve()),
            "reference_plan_sha256": hashlib.sha256(
                (reference / "plan.json").read_bytes()
            ).hexdigest(),
            "reference_subset_sha256": hashlib.sha256(
                (reference / "subset.json").read_bytes()
            ).hexdigest(),
            "unchanged": "Pinned checkpoints, templates, 1024 evaluation images, 64 pilot images, eight samples, seeds and sampling parameters",
            "serving": "BF16, FlashAttention 2, CUDA graphs/compilation enabled, 256 active sequences, 12288-token context",
            "pilot_selection": "At most 3% length-limited answers without a valid final number; an unmet criterion at the maximum authorized cap is explicitly flagged. Throughput uses full decode warm-up and two trials in opposite orders on all 64 disjoint pilot images with n=8, comparing 32/64 HTTP requests; accuracy is not a selection criterion.",
            "created_at": time.time(),
        },
    )


def sampling_parameters(*, plan: EvaluationPlan) -> dict:
    return {
        "temperature": plan.temperature,
        "top_p": plan.top_p,
        "top_k": plan.top_k,
        "min_p": 0.0,
        "presence_penalty": plan.presence_penalty,
        "repetition_penalty": 1.0,
    }


def prepare_thinking_rerun(*, root: Path, reference: Path, cache_volume: str) -> None:
    prepare_followup(root=root, reference=reference, cache_volume=cache_volume)
    plan = EvaluationPlan.model_validate_json((root / "plan.json").read_text())
    evaluations = tuple(spec for spec in plan.evaluations if spec.mode == "thinking_on")
    identifiers = {spec.model_identifier for spec in evaluations}
    if len(evaluations) != 3 or len(identifiers) != 3:
        raise ValueError("Expected three distinct thinking-on model sizes")
    plan = EvaluationPlan.model_validate({
        **plan.model_dump(),
        "evaluations": evaluations,
        "models": tuple(
            model for model in plan.models if model.identifier in identifiers
        ),
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 20,
        "presence_penalty": 1.5,
    })
    write_json(path=root / "plan.json", payload=plan.model_dump(mode="json"))
    provenance_path = root / "provenance.json"
    provenance = json.loads(provenance_path.read_text())
    provenance.update({
        "sampling": sampling_parameters(plan=plan),
        "sampling_source": "https://huggingface.co/Qwen/Qwen3.5-4B#best-practices",
        "unchanged": "Pinned checkpoints, templates, 1024 evaluation images, 64 pilot images, eight samples, seeds, prompt, token-cap and concurrency pilots, serving configuration",
        "superseded_evaluations": [spec.identifier for spec in evaluations],
        "retention": "Reference responses stay in the reference root; no responses or tuning trials are copied into this rerun",
    })
    write_json(path=provenance_path, payload=provenance)


def parse_age(*, content: str) -> float | None:
    final = content.rsplit("</think>", 1)[-1]
    boxes = re.findall(r"\\boxed\s*\{([^{}]*)\}", final)
    if not boxes:
        return None
    answer = boxes[-1].strip()
    if not re.fullmatch(r"\+?(?:\d+(?:\.\d*)?|\.\d+)", answer):
        return None
    value = float(answer)
    return value if math.isfinite(value) and 0 <= value <= 99 else None


def prediction_metrics(*, pairs: list[tuple[float, int]]) -> dict:
    if not pairs:
        return {"count": 0, "mae": None, "rmse": None, "spearman": None}
    predictions, targets = np.asarray(pairs, dtype=np.float64).T
    errors = predictions - targets
    defined = len(set(predictions)) > 1 and len(set(targets)) > 1
    return {
        "count": len(pairs),
        "mae": float(np.abs(errors).mean()),
        "rmse": float(np.square(errors).mean() ** 0.5),
        "spearman": float(spearmanr(predictions, targets).statistic)
        if defined
        else None,
    }


def summarize_records(*, records: list[dict], samples: int) -> dict:
    pooled, means, medians, complete_means, complete_medians = [], [], [], [], []
    replicates = [[] for _ in range(samples)]
    invalid_squared, invalid_absolute, invalid, truncated, tokens = 0.0, 0.0, 0, 0, 0
    histogram = [0] * (samples + 1)
    for record in records:
        choices = sorted(record["response"]["choices"], key=lambda c: c["index"])
        if [c["index"] for c in choices] != list(range(samples)):
            raise ValueError("Every image must retain all requested completion indices")
        target = record["target"]
        values = []
        tokens += record["response"]["usage"]["completion_tokens"]
        for choice in choices:
            content = choice["message"].get("content") or ""
            value = parse_age(content=content)
            truncated += choice["finish_reason"] == "length"
            if value is None:
                invalid += 1
                error = max(target, 99 - target)
            else:
                values.append(value)
                pooled.append((value, target))
                replicates[choice["index"]].append((value, target))
                error = abs(value - target)
            invalid_absolute += error
            invalid_squared += error**2
        histogram[len(values)] += 1
        if values:
            means.append((statistics.mean(values), target))
            medians.append((statistics.median(values), target))
        if len(values) == samples:
            complete_means.append(means[-1])
            complete_medians.append(medians[-1])
    per_replicate = [prediction_metrics(pairs=pairs) for pairs in replicates]
    replicate_summary = {}
    for metric in ("mae", "rmse", "spearman"):
        values = [row[metric] for row in per_replicate if row[metric] is not None]
        replicate_summary[metric] = {
            "mean": statistics.mean(values) if values else None,
            "sample_sd": statistics.stdev(values) if len(values) > 1 else None,
            "defined_replicates": len(values),
        }
    total = len(records) * samples
    return {
        "images": len(records),
        "samples_per_image": samples,
        "requested_predictions": total,
        "valid_predictions": len(pooled),
        "invalid_predictions": invalid,
        "valid_fraction": len(pooled) / total if total else 0,
        "truncated_predictions": truncated,
        "completion_tokens": tokens,
        "valid_samples_histogram": histogram,
        "pooled_single_draw": prediction_metrics(pairs=pooled),
        "sample_mean_complete8": prediction_metrics(pairs=complete_means),
        "sample_median_complete8": prediction_metrics(pairs=complete_medians),
        "sample_mean_available": prediction_metrics(pairs=means),
        "sample_median_available": prediction_metrics(pairs=medians),
        "per_replicate": per_replicate,
        "replicate_mean_and_sd": replicate_summary,
        "worst_case_invalid_penalty": {
            "mae": invalid_absolute / total if total else None,
            "rmse": (invalid_squared / total) ** 0.5 if total else None,
            "definition": "Invalid responses receive the larger error of predicting age 0 or 99; valid predictions retain their observed error",
        },
    }


def request_json(*, url: str, payload: dict) -> dict:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request, timeout=3600) as response:
                return json.load(response)
        except urllib.error.HTTPError as error:
            detail = error.read().decode()
            if error.code not in {429, 502, 503, 504} or attempt == 2:
                raise RuntimeError(f"HTTP {error.code}: {detail}") from error
        except (urllib.error.URLError, TimeoutError):
            if attempt == 2:
                raise
        time.sleep(2 ** (attempt + 1))
    raise RuntimeError("Unreachable HTTP retry state")


def query_image(
    *,
    plan: EvaluationPlan,
    spec: EvaluationSpec,
    payload: dict,
    index: int,
    port: int,
    samples: int,
    max_tokens: int,
) -> dict:
    record = payload["records"][index]
    image_bytes = (plan.manifest.parent / record["relative_path"]).read_bytes()
    started = time.monotonic()
    response = request_json(
        url=f"http://127.0.0.1:{port}/v1/chat/completions",
        payload={
            "model": spec.model_identifier,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": "data:image/jpeg;base64,"
                                + base64.b64encode(image_bytes).decode(),
                            },
                        },
                        {"type": "text", "text": plan.prompt},
                    ],
                }
            ],
            "n": samples,
            "max_tokens": max_tokens,
            **sampling_parameters(plan=plan),
            "seed": plan.sampling_seed + index,
            "chat_template_kwargs": {"enable_thinking": spec.mode == "thinking_on"},
        },
    )
    if response["model"] != spec.model_identifier:
        raise ValueError("Serving model identifier does not match the evaluation")
    return {
        "index": index,
        "target": record["age"],
        "image_sha256": hashlib.sha256(image_bytes).hexdigest(),
        "sampling": sampling_parameters(plan=plan),
        "request_seconds": time.monotonic() - started,
        "response": response,
    }


def evaluate_images(
    *,
    plan: EvaluationPlan,
    spec: EvaluationSpec,
    payload: dict,
    indices: tuple[int, ...],
    port: int,
    samples: int,
    max_tokens: int,
    concurrency: int,
    output: Path,
) -> tuple[list[dict], float]:
    output.mkdir(parents=True, exist_ok=True)
    pending = [i for i in indices if not (output / f"{i:05d}.json").exists()]
    started = time.monotonic()
    executor = ThreadPoolExecutor(max_workers=concurrency)
    try:
        futures = {
            executor.submit(
                query_image,
                plan=plan,
                spec=spec,
                payload=payload,
                index=index,
                port=port,
                samples=samples,
                max_tokens=max_tokens,
            ): index
            for index in pending
        }
        for future in as_completed(futures):
            record = future.result()
            write_json(path=output / f"{record['index']:05d}.json", payload=record)
    finally:
        executor.shutdown(wait=False, cancel_futures=True)
    elapsed = time.monotonic() - started
    records = [json.loads((output / f"{i:05d}.json").read_text()) for i in indices]
    for index, record in zip(indices, records, strict=True):
        if (
            record["index"] != index
            or record["target"] != payload["records"][index]["age"]
        ):
            raise ValueError(
                "Saved response does not match the frozen evaluation subset"
            )
    return records, elapsed


@contextmanager
def utilization_monitor(*, gpu_uuid: str, port: int):
    snapshots = []
    stopped = threading.Event()

    def sample():
        while not stopped.is_set():
            measurement = {"time": time.time()}
            result = subprocess.run(
                [
                    "nvidia-smi",
                    f"--id={gpu_uuid}",
                    "--query-gpu=utilization.gpu,memory.used",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                check=False,
                timeout=10,
            )
            if result.returncode:
                measurement["gpu_error"] = result.stderr
            else:
                utilization, memory = result.stdout.strip().split(",")
                measurement.update({
                    "gpu_utilization_percent": float(utilization),
                    "memory_mib": float(memory),
                })
            try:
                with urllib.request.urlopen(
                    f"http://127.0.0.1:{port}/metrics", timeout=5
                ) as response:
                    metrics = response.read().decode()
                for name in (
                    "num_requests_running",
                    "num_requests_waiting",
                    "kv_cache_usage_perc",
                    "num_preemptions_total",
                ):
                    values = re.findall(
                        rf"^vllm:{name}(?:\{{[^\n]*\}})? ([\d.eE+-]+)$",
                        metrics,
                        re.MULTILINE,
                    )
                    if values:
                        measurement[name] = sum(float(value) for value in values)
            except (urllib.error.URLError, TimeoutError, ConnectionError) as error:
                measurement["metrics_error"] = str(error)
            snapshots.append(measurement)
            stopped.wait(2)

    thread = threading.Thread(target=sample, daemon=True)
    thread.start()
    try:
        yield snapshots
    finally:
        stopped.set()
        thread.join(timeout=20)
        if thread.is_alive():
            raise RuntimeError("GPU utilization monitor did not finish")


def tune(
    *,
    root: Path,
    plan: EvaluationPlan,
    spec: EvaluationSpec,
    payload: dict,
    port: int,
    gpu_uuid: str,
) -> dict:
    directory = root / "evaluations" / spec.identifier
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / "tuning.json"
    if target.exists():
        tuning = json.loads(target.read_text())
        if tuning["pilot_protocol"] != "warm_decode_balanced_v2":
            raise ValueError("Archive the previous tuning before repeating pilots")
        return tuning
    caps = (
        plan.token_caps_thinking
        if spec.mode == "thinking_on"
        else plan.token_caps_direct
    )
    cap_results = []
    for cap in caps:
        records, seconds = evaluate_images(
            plan=plan,
            spec=spec,
            payload=payload,
            indices=plan.pilot_indices[:16],
            port=port,
            samples=2,
            max_tokens=cap,
            concurrency=8,
            output=directory / "pilot" / f"tokens_{cap}",
        )
        metrics = summarize_records(records=records, samples=2)
        unfinished = sum(
            choice["finish_reason"] == "length"
            and parse_age(content=choice["message"].get("content") or "") is None
            for record in records
            for choice in record["response"]["choices"]
        )
        cap_results.append({
            "max_tokens": cap,
            "seconds": seconds,
            "metrics": metrics,
            "unfinished_length_limited": unfinished,
        })
        cap_criterion_met = unfinished / metrics["requested_predictions"] <= 0.03
        if cap_criterion_met:
            break
    concurrency_results = []
    warmup = directory / "pilot" / f"warmup_{time.time_ns()}"
    evaluate_images(
        plan=plan,
        spec=spec,
        payload=payload,
        indices=plan.pilot_indices,
        port=port,
        samples=plan.samples,
        max_tokens=min(cap, 256),
        concurrency=max(plan.client_concurrencies),
        output=warmup,
    )
    order = [
        (round_index, concurrency)
        for round_index, candidates in enumerate((
            plan.client_concurrencies,
            plan.client_concurrencies[::-1],
        ))
        for concurrency in candidates
    ]
    for round_index, concurrency in order:
        if len(plan.pilot_indices) < concurrency:
            raise ValueError("Throughput pilot must fill the HTTP concurrency limit")
        with utilization_monitor(gpu_uuid=gpu_uuid, port=port) as snapshots:
            records, seconds = evaluate_images(
                plan=plan,
                spec=spec,
                payload=payload,
                indices=plan.pilot_indices,
                port=port,
                samples=plan.samples,
                max_tokens=cap,
                concurrency=concurrency,
                output=directory
                / "pilot"
                / f"concurrency_{concurrency}_{time.time_ns()}",
            )
        metrics = summarize_records(records=records, samples=plan.samples)
        concurrency_results.append({
            "concurrency": concurrency,
            "round": round_index,
            "requests": len(records),
            "seconds": seconds,
            "images_per_second": len(records) / max(seconds, 1e-6),
            "output_tokens_per_second": metrics["completion_tokens"]
            / max(seconds, 1e-6),
            "utilization": snapshots,
            "metrics": metrics,
        })
    rates = {
        concurrency: statistics.median(
            row["images_per_second"]
            for row in concurrency_results
            if row["concurrency"] == concurrency
        )
        for concurrency in plan.client_concurrencies
    }
    peak = max(rates.values())
    selected = min(
        concurrency for concurrency, rate in rates.items() if rate >= 0.95 * peak
    )
    result = {
        "pilot_protocol": "warm_decode_balanced_v2",
        "max_tokens": cap,
        "concurrency": selected,
        "median_images_per_second": rates,
        "cap_criterion_met": cap_criterion_met,
        "token_budget_limited": not cap_criterion_met,
        "selection": "Lowest concurrency within 5% of peak median images/s from two trials in opposite orders on the same 64 pilot images, using all eight samples and the selected token cap after decode warm-up",
        "token_trials": cap_results,
        "concurrency_trials": concurrency_results,
        "selected_without_age_accuracy": True,
    }
    write_json(path=target, payload=result)
    return result


def docker_command(
    *,
    root: Path,
    plan: EvaluationPlan,
    model: ModelSpec,
    gpu_uuid: str,
    port: int,
    container: str,
) -> list[str]:
    return [
        "docker",
        "run",
        "-d",
        "--name",
        container,
        "--label",
        f"utkface.vlm.root={root}",
        "--entrypoint",
        "uv",
        "--gpus",
        f"device={gpu_uuid}",
        "--ipc=host",
        "-p",
        f"127.0.0.1:{port}:8000",
        "-v",
        f"{plan.cache_volume}:/root/.cache/huggingface",
        "-v",
        f"{root}:/study:ro",
        "-e",
        "VLLM_USAGE_STATS_DISABLED=1",
        "-e",
        "DO_NOT_TRACK=1",
        plan.image,
        "run",
        "--no-project",
        "--no-sync",
        "vllm",
        "serve",
        "--model",
        model.repository,
        "--revision",
        model.revision,
        "--tokenizer-revision",
        model.revision,
        "--chat-template",
        f"/study/{model.template_file}",
        "--served-model-name",
        model.identifier,
        "--dtype",
        "bfloat16",
        "--generation-config",
        "vllm",
        "--reasoning-parser",
        "qwen3",
        "--attention-backend",
        "FLASH_ATTN",
        "--attention-config",
        '{"flash_attn_version":2}',
        "--max-model-len",
        str(plan.max_model_len),
        "--max-num-seqs",
        str(plan.max_num_seqs),
        "--max-num-batched-tokens",
        str(plan.max_num_batched_tokens),
        "--gpu-memory-utilization",
        str(plan.gpu_memory_utilization),
        "--limit-mm-per-prompt",
        '{"image":1,"video":0}',
    ]


def stop_container(*, root: Path, container: str) -> None:
    result = subprocess.run(
        [
            "docker",
            "inspect",
            "--format",
            '{{index .Config.Labels "utkface.vlm.root"}}',
            container,
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode:
        return
    if result.stdout.strip() != str(root):
        raise RuntimeError("Refusing to stop a container not owned by this study")
    subprocess.run(
        ["docker", "stop", "-t", "15", container], check=True, capture_output=True
    )


def wait_for_server(*, port: int, container: str) -> None:
    deadline = time.monotonic() + 1200
    while time.monotonic() < deadline:
        result = subprocess.run(
            ["docker", "inspect", "--format", "{{.State.Running}}", container],
            capture_output=True,
            text=True,
            check=True,
        )
        if result.stdout.strip() != "true":
            raise RuntimeError(
                "vLLM container exited during startup; inspect its saved logs"
            )
        try:
            with urllib.request.urlopen(
                f"http://127.0.0.1:{port}/v1/models", timeout=3
            ) as response:
                if response.status == 200:
                    return
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            pass
        time.sleep(3)
    raise TimeoutError("vLLM did not become ready within 20 minutes")


def worker(
    *, root: Path, identifier: str, gpu_uuid: str, port: int, container: str
) -> None:
    plan = load_plan(root=root)
    sharing = shared_runtime(
        policy=load_shared_policy(root=root), identifier=identifier
    )
    if sharing is not None:
        if sharing.gpu_uuid != gpu_uuid:
            raise ValueError("Worker GPU differs from the authorized sharing grant")
        plan = EvaluationPlan.model_validate({
            **plan.model_dump(),
            "gpu_memory_utilization": sharing.gpu_memory_utilization,
            "max_num_seqs": sharing.max_num_seqs,
            "client_concurrencies": sharing.client_concurrencies,
        })
    serving = {
        "gpu_memory_utilization": plan.gpu_memory_utilization,
        "max_num_seqs": plan.max_num_seqs,
        "client_concurrencies": plan.client_concurrencies,
        "max_model_len": plan.max_model_len,
        "max_num_batched_tokens": plan.max_num_batched_tokens,
        "sharing": sharing.model_dump(mode="json") if sharing is not None else None,
    }
    model = next(model for model in plan.models if model.identifier == identifier)
    payload = read_manifest(path=plan.manifest)
    if payload["fingerprint"] != plan.fingerprint:
        raise ValueError("Dataset fingerprint changed")
    directory = root / "workers" / container
    directory.mkdir(parents=True)

    def interrupt(signum, frame):
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, interrupt)
    signal.signal(signal.SIGINT, interrupt)
    command = docker_command(
        root=root,
        plan=plan,
        model=model,
        gpu_uuid=gpu_uuid,
        port=port,
        container=container,
    )
    write_json(
        path=directory / "launch.json",
        payload={
            "command": command,
            "started_at": time.time(),
            "gpu_uuid": gpu_uuid,
            "source_hashes": source_hashes(),
            "serving": serving,
        },
    )
    try:
        subprocess.run(command, check=True)
        wait_for_server(port=port, container=container)
        for spec in plan.evaluations:
            if spec.model_identifier != identifier:
                continue
            destination = root / "evaluations" / spec.identifier
            tuning = tune(
                root=root,
                plan=plan,
                spec=spec,
                payload=payload,
                port=port,
                gpu_uuid=gpu_uuid,
            )
            if (destination / "result.json").exists():
                continue
            records, seconds = evaluate_images(
                plan=plan,
                spec=spec,
                payload=payload,
                indices=plan.indices,
                port=port,
                samples=plan.samples,
                max_tokens=tuning["max_tokens"],
                concurrency=tuning["concurrency"],
                output=destination / "records",
            )
            result = {
                "evaluation": spec.model_dump(),
                "model": model.model_dump(),
                "fingerprint": plan.fingerprint,
                "sampling": sampling_parameters(plan=plan),
                "serving": serving,
                "tuning": tuning,
                "metrics": summarize_records(records=records, samples=plan.samples),
                "successful_attempt_seconds": seconds,
                "completed_at": time.time(),
            }
            write_json(path=destination / "result.json", payload=result)
            print(
                json.dumps({
                    "completed": spec.identifier,
                    "metrics": result["metrics"],
                }),
                flush=True,
            )
        write_json(
            path=directory / "completed.json", payload={"completed_at": time.time()}
        )
    finally:
        try:
            logs = subprocess.run(
                ["docker", "logs", container],
                check=False,
                capture_output=True,
                text=True,
            )
            (directory / "vllm.log").write_text(logs.stdout + logs.stderr)
        finally:
            stop_container(root=root, container=container)


def report(*, root: Path) -> dict:
    plan = EvaluationPlan.model_validate_json((root / "plan.json").read_text())
    results = []
    for spec in plan.evaluations:
        path = root / "evaluations" / spec.identifier / "result.json"
        if path.exists():
            result = json.loads(path.read_text())
            if result["metrics"]["images"] != len(plan.indices):
                raise ValueError(
                    "Completed evaluation does not cover the frozen subset"
                )
            results.append(result)
    rows = []
    for result in results:
        metrics = result["metrics"]
        row = {
            "configuration": result["evaluation"]["identifier"],
            "size": result["model"]["size"],
            "mode": result["evaluation"]["mode"],
            "valid_fraction": metrics["valid_fraction"],
            "invalid_predictions": metrics["invalid_predictions"],
            "truncated_predictions": metrics["truncated_predictions"],
            "max_tokens": result["tuning"]["max_tokens"],
            "client_concurrency": result["tuning"]["concurrency"],
        }
        for aggregation in (
            "pooled_single_draw",
            "sample_mean_complete8",
            "sample_median_complete8",
            "sample_mean_available",
            "sample_median_available",
        ):
            for metric, value in metrics[aggregation].items():
                row[f"{aggregation}_{metric}"] = value
        for metric, values in metrics["replicate_mean_and_sd"].items():
            row[f"replicate_{metric}_mean"] = values["mean"]
            row[f"replicate_{metric}_sd"] = values["sample_sd"]
        rows.append(row)
    summary = {
        "completed": len(results),
        "expected": len(plan.evaluations),
        "results": results,
    }
    write_json(path=root / "summary.json", payload=summary)
    if rows:
        pl.DataFrame(rows).write_csv(root / "summary.csv")
    lines = [
        "# Qwen3.5 UTKFace evaluation",
        "",
        f"Completed {len(results)}/{len(plan.evaluations)} configurations. Each configuration requests {plan.samples} samples on the same {len(plan.indices)} validation images.",
        "",
        f"Prompt: `{plan.prompt}`",
        "",
        "Sampling: `"
        + json.dumps(sampling_parameters(plan=plan), sort_keys=True)
        + "`",
        "",
        "| Configuration | Valid % | Pooled MAE | Pooled RMSE | Pooled Spearman | Mean-8 MAE | Median-8 MAE | Complete-8 images | Truncated |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    def cell(value):
        return "undefined" if value is None else f"{value:.4f}"

    for result in results:
        metrics = result["metrics"]
        pooled = metrics["pooled_single_draw"]
        lines.append(
            f"| {result['evaluation']['identifier']} | {100 * metrics['valid_fraction']:.2f} | "
            f"{cell(pooled['mae'])} | {cell(pooled['rmse'])} | {cell(pooled['spearman'])} | "
            f"{cell(metrics['sample_mean_complete8']['mae'])} | {cell(metrics['sample_median_complete8']['mae'])} | "
            f"{metrics['sample_mean_complete8']['count']} | {metrics['truncated_predictions']} |"
        )
    lines.extend([
        "",
        "All errors are in years. Correlation is Spearman with average ranks for ties.",
        "",
        "Pooled metrics concatenate all valid prediction/target pairs, repeating each image's target eight times. Mean-8 and median-8 first aggregate predictions within an image and require all eight answers to parse. The CSV/JSON also retain aggregates using available valid samples, each replicate's metrics, and the mean/sample SD across eight replicates. These sampling SDs are not training-seed uncertainty.",
        "",
        "Invalid answers remain in the raw responses and coverage counts; they are not silently imputed. JSON includes explicitly labeled worst-case invalid penalties. Truncation counts include length-limited completions even when a boxed number is present. Main prompts and sampling parameters are fixed; token caps and client concurrency are selected on disjoint pilot images using completion and throughput, without selecting on age accuracy.",
        "",
        "Base models use their own weights/tokenizers/processors with the paired post-trained model's official chat template and thinking disabled. Post-trained modes use pinned post-trained weights. Sampling parameters are recorded in the plan and new results. Images are sent as bytes; filenames and labels are absent from model requests.",
    ])
    (root / "REPORT.md").write_text("\n".join(lines) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "action", choices=("prepare", "followup", "thinking-rerun", "worker", "report")
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--image")
    parser.add_argument("--model")
    parser.add_argument("--gpu-uuid")
    parser.add_argument("--port", type=int)
    parser.add_argument("--container")
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--cache-volume")
    args = parser.parse_args()
    if args.action == "prepare":
        if args.manifest is None or args.image is None:
            parser.error("prepare requires --manifest and --image")
        prepare(root=args.root, manifest=args.manifest, image=args.image)
    elif args.action == "followup":
        if args.reference is None or args.cache_volume is None:
            parser.error("followup requires --reference and --cache-volume")
        prepare_followup(
            root=args.root, reference=args.reference, cache_volume=args.cache_volume
        )
    elif args.action == "thinking-rerun":
        if args.reference is None or args.cache_volume is None:
            parser.error("thinking-rerun requires --reference and --cache-volume")
        prepare_thinking_rerun(
            root=args.root, reference=args.reference, cache_volume=args.cache_volume
        )
    elif args.action == "worker":
        if any(
            value is None
            for value in (args.model, args.gpu_uuid, args.port, args.container)
        ):
            parser.error("worker requires --model, --gpu-uuid, --port and --container")
        worker(
            root=args.root,
            identifier=args.model,
            gpu_uuid=args.gpu_uuid,
            port=args.port,
            container=args.container,
        )
    else:
        print(
            json.dumps({
                key: value
                for key, value in report(root=args.root).items()
                if key != "results"
            })
        )


if __name__ == "__main__":
    main()
