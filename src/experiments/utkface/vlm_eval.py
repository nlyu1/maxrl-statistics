import argparse
import base64
import hashlib
import json
import math
import re
import signal
import statistics
import subprocess
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
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
    source_hashes: dict[str, str]


SOURCE_FILES = (
    "src/experiments/utkface/vlm_eval.py",
    "src/experiments/utkface/vlm_queue.py",
    "src/experiments/utkface/scheduler.py",
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
            with urllib.request.urlopen(request, timeout=900) as response:
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
            "temperature": plan.temperature,
            "top_p": plan.top_p,
            "top_k": plan.top_k,
            "presence_penalty": plan.presence_penalty,
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
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
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


def tune(
    *,
    root: Path,
    plan: EvaluationPlan,
    spec: EvaluationSpec,
    payload: dict,
    port: int,
) -> dict:
    directory = root / "evaluations" / spec.identifier
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / "tuning.json"
    if target.exists():
        return json.loads(target.read_text())
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
        cap_results.append({"max_tokens": cap, "seconds": seconds, "metrics": metrics})
        if metrics["truncated_predictions"] / metrics["requested_predictions"] <= 0.03:
            break
    concurrency_results = []
    for concurrency in plan.client_concurrencies:
        records, seconds = evaluate_images(
            plan=plan,
            spec=spec,
            payload=payload,
            indices=plan.pilot_indices[16:32],
            port=port,
            samples=plan.samples,
            max_tokens=cap,
            concurrency=concurrency,
            output=directory / "pilot" / f"concurrency_{concurrency}_{time.time_ns()}",
        )
        metrics = summarize_records(records=records, samples=plan.samples)
        concurrency_results.append({
            "concurrency": concurrency,
            "seconds": seconds,
            "output_tokens_per_second": metrics["completion_tokens"]
            / max(seconds, 1e-6),
            "metrics": metrics,
        })
    selected = max(concurrency_results, key=lambda row: row["output_tokens_per_second"])
    result = {
        "max_tokens": cap,
        "concurrency": selected["concurrency"],
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
        "--enforce-eager",
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
        except (urllib.error.URLError, TimeoutError):
            pass
        time.sleep(3)
    raise TimeoutError("vLLM did not become ready within 20 minutes")


def worker(
    *, root: Path, identifier: str, gpu_uuid: str, port: int, container: str
) -> None:
    plan = load_plan(root=root)
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
        },
    )
    try:
        subprocess.run(command, check=True)
        wait_for_server(port=port, container=container)
        for spec in plan.evaluations:
            if spec.model_identifier != identifier:
                continue
            destination = root / "evaluations" / spec.identifier
            if (destination / "result.json").exists():
                continue
            tuning = tune(root=root, plan=plan, spec=spec, payload=payload, port=port)
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
        logs = subprocess.run(
            ["docker", "logs", container], check=False, capture_output=True, text=True
        )
        (directory / "vllm.log").write_text(logs.stdout + logs.stderr)
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
        "Base models use their own weights/tokenizers/processors with the paired post-trained model's official chat template and thinking disabled. Thinking-on/off use identical post-trained weights and sampling parameters. Images are sent as bytes; filenames and labels are absent from model requests.",
    ])
    (root / "REPORT.md").write_text("\n".join(lines) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "worker", "report"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--image")
    parser.add_argument("--model")
    parser.add_argument("--gpu-uuid")
    parser.add_argument("--port", type=int)
    parser.add_argument("--container")
    args = parser.parse_args()
    if args.action == "prepare":
        if args.manifest is None or args.image is None:
            parser.error("prepare requires --manifest and --image")
        prepare(root=args.root, manifest=args.manifest, image=args.image)
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
