import argparse
import base64
import hashlib
import io
import json
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import torch
from PIL import Image
from torch import Tensor

from src.experiments.utkface.config import NoiseConfig, RunConfig
from src.experiments.utkface.data import (
    image_transform,
    read_manifest,
    training_indices,
)
from src.experiments.utkface.losses import NUM_BINS, log_policy
from src.experiments.utkface.metrics import METRIC_NAMES, batch_metrics
from src.experiments.utkface.model import build_model
from src.experiments.utkface.noise import noise_summary, noisy_labels

METHODS = ("tailrl_population", "ce", "grpo_mae_population", "grpo_mse_population")
CONDITIONS = (("none", 0), *(("fixed", std) for std in (1, 3, 5, 10)))


def sha256(*, path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def select_runs(*, snapshot: dict) -> list[dict]:
    selected = []
    for mode, std in CONDITIONS:
        for method in METHODS:
            matches = [
                run
                for run in snapshot["runs"]
                if run["status"] == "completed"
                and run["stage"] == ("subset" if mode == "none" else "noise")
                and run["head"] == "categorical"
                and run["size"] == 4096
                and run["epochs"] == 50
                and run["seed"] == 42
                and run["method"] == method
                and run["noise"] == mode
                and run["std"] == std
            ]
            if len(matches) != 1:
                raise ValueError((mode, std, method, len(matches)))
            selected.append(matches[0])
    return selected


def sample_indices(
    *, payload: dict, candidates: list[int], count: int, seed: int
) -> list[int]:
    if count < 10 or count > len(candidates):
        raise ValueError("Example counts must cover all ten age decades")
    generator = np.random.default_rng(seed)
    buckets = {
        decade: generator.permutation([
            index
            for index in candidates
            if payload["records"][index]["age"] // 10 == decade
        ]).tolist()
        for decade in range(10)
    }
    selected = []
    while len(selected) < count:
        for decade in generator.permutation(10):
            if buckets[decade] and len(selected) < count:
                selected.append(buckets[decade].pop())
    return selected


def examples(
    *, manifest: Path, payload: dict, indices: list[int], train: set[int]
) -> tuple[list[dict], Tensor]:
    records, tensors = [], []
    transform = image_transform(training=False)
    preview_transform = transform.transforms[:2]
    for index in indices:
        record = payload["records"][index]
        path = manifest.parent / record["relative_path"]
        with Image.open(path) as image:
            rgb = image.convert("RGB")
            tensors.append(transform(rgb))
            for operation in preview_transform:
                rgb = operation(rgb)
            buffer = io.BytesIO()
            rgb.save(buffer, format="JPEG", quality=90)
        records.append({
            "index": index,
            "split": "train" if index in train else "test",
            "age": record["age"],
            "relative_path": record["relative_path"],
            "image_sha256": sha256(path=path),
            "image": "data:image/jpeg;base64,"
            + base64.b64encode(buffer.getvalue()).decode(),
        })
    return records, torch.stack(tensors)


def corruption_fields(
    *, payload: dict, indices: list[int], train: set[int], seed: int
) -> dict:
    ages = [record["age"] for record in payload["records"]]
    fields = {}
    for mode, std in CONDITIONS:
        config = NoiseConfig(mode=mode, std_years=std, seed=seed)
        fields[f"{mode}:{std}"] = []
        labels = noisy_labels(ages=ages, config=config, epoch=1)
        fields[f"{mode}:{std}"].append([
            int(labels[index]) if index in train else None for index in indices
        ])
    return fields


def write_viewer(*, data: dict, path: Path) -> None:
    conditions = {f"{mode}:{std}" for mode, std in CONDITIONS}
    runs = [
        run
        for run in data["runs"]
        if (run["noise"]["mode"], run["noise"]["std_years"]) in CONDITIONS
    ]
    expected = {(mode, std, method) for mode, std in CONDITIONS for method in METHODS}
    actual = {
        (run["noise"]["mode"], run["noise"]["std_years"], run["method"]) for run in runs
    }
    if actual != expected or len(runs) != len(expected):
        raise ValueError("Expected one checkpoint per method and fixed-noise setting")
    report = {
        **data,
        "runs": runs,
        "corruptions": {key: data["corruptions"][key] for key in sorted(conditions)},
    }
    template = Path(__file__).with_suffix(".html").read_text()
    if template.count("__POLICY_DATA__") != 1:
        raise ValueError("Expected exactly one report data placeholder")
    encoded = json.dumps(report, separators=(",", ":"), allow_nan=False)
    path.write_text(
        template.replace("__POLICY_DATA__", encoded.replace("<", "\\u003c"))
    )


def infer(
    *,
    checkpoint: Path,
    config: RunConfig,
    images: Tensor,
    targets: Tensor,
    device: torch.device,
    batch_size: int,
) -> tuple[dict, dict]:
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if saved["config"] != config.model_dump(mode="json"):
        raise ValueError(f"Checkpoint/config mismatch: {checkpoint}")
    probabilities, summaries = [], []
    with torch.cuda.device(device), torch.device(device), torch.inference_mode():
        model = build_model(pretrained=False, head="categorical").to(device).eval()
        model.load_state_dict(saved["model"], strict=True)
        bins = torch.arange(NUM_BINS, dtype=torch.float32, device=device)
        for start in range(0, len(images), batch_size):
            batch_targets = targets[start : start + batch_size].to(device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(images[start : start + batch_size].to(device))
            log_probs = log_policy(logits=logits)
            probs = log_probs.exp()
            if not torch.isfinite(log_probs).all() or not torch.allclose(
                probs.sum(-1), torch.ones(len(probs), device=device), atol=1e-6
            ):
                raise ValueError(f"Invalid policy: {checkpoint}")
            metrics = batch_metrics(logits=logits, targets=batch_targets).cpu().tolist()
            means = (probs * bins).sum(-1).cpu().tolist()
            medians = (probs.cumsum(-1) < 0.5).sum(-1).clamp_max(99).cpu().tolist()
            modes = probs.argmax(-1).cpu().tolist()
            for row, mean, median, mode in zip(
                metrics, means, medians, modes, strict=True
            ):
                summaries.append({
                    "mean": mean,
                    "median": median,
                    "mode": mode,
                    **dict(zip(METRIC_NAMES, row, strict=True)),
                })
            probabilities.extend(probs.cpu().tolist())
        del model
    return {
        "probabilities": probabilities,
        "summaries": summaries,
    }, {key: saved[key] for key in ("epoch", "validation", "data_fingerprint")}


def build(
    *,
    snapshot_path: Path,
    output_dir: Path,
    device: str,
    train_count: int,
    test_count: int,
    sample_seed: int,
    batch_size: int,
) -> None:
    cuda = torch.device(device)
    if cuda.type != "cuda" or batch_size < 1:
        raise ValueError("Inference requires a CUDA device and a positive batch size")
    output_dir.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    snapshot = json.loads(snapshot_path.read_text())
    runs = select_runs(snapshot=snapshot)
    first_path = Path(runs[0]["resultPath"])
    first_config = RunConfig.model_validate_json(
        first_path.with_name("config.json").read_text()
    )
    manifest = first_config.manifest
    payload = read_manifest(path=manifest)
    train = training_indices(
        payload=payload, limit=4096, subset_seed=first_config.subset_seed
    )
    train_hash = hashlib.sha256(json.dumps(train).encode()).hexdigest()
    selected = sample_indices(
        payload=payload, candidates=train, count=train_count, seed=sample_seed
    ) + sample_indices(
        payload=payload,
        candidates=payload["splits"]["test"],
        count=test_count,
        seed=sample_seed + 1,
    )
    records, images = examples(
        manifest=manifest, payload=payload, indices=selected, train=set(train)
    )
    clean_ages = [record["age"] for record in payload["records"]]
    targets = torch.tensor([record["age"] for record in records])
    data = {
        "generated": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "data_fingerprint": payload["fingerprint"],
        "training_fingerprint": train_hash,
        "sample_seed": sample_seed,
        "sample_rule": "Balanced random sampling across age decades before any model inference; fixed across all settings.",
        "sample_counts": {"train": train_count, "test": test_count},
        "inference": {
            "device": str(cuda),
            "precision": "bf16 CNN, float32 log-softmax",
            "torch": torch.__version__,
            "transform": "Resize(256), CenterCrop(224), ImageNet normalization; no training augmentation",
        },
        "commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "snapshot_sha256": sha256(path=snapshot_path),
        "examples": records,
        "corruptions": corruption_fields(
            payload=payload,
            indices=selected,
            train=set(train),
            seed=first_config.noise.seed,
        ),
        "runs": [],
    }
    common_fields = (
        "head",
        "seed",
        "epochs",
        "batch_size",
        "weight_decay",
        "clip_grad_norm",
        "schedule",
        "pretrained",
        "precision",
        "train_limit",
        "eval_limit",
        "subset_seed",
    )
    reference = first_config.model_dump(mode="json")
    for position, run in enumerate(runs, start=1):
        result_path = Path(run["resultPath"])
        result = json.loads(result_path.read_text())
        config = RunConfig.model_validate_json(
            result_path.with_name("config.json").read_text()
        )
        values = config.model_dump(mode="json")
        if any(values[field] != reference[field] for field in common_fields):
            raise ValueError(f"Unmatched training settings: {run['id']}")
        if (
            result["config"] != values
            or result["data_fingerprint"] != payload["fingerprint"]
            or result["training_fingerprint"] != train_hash
            or run["fingerprint"] != payload["fingerprint"]
            or run["trainingFingerprint"] != train_hash
            or config.loss.method != run["method"]
            or config.noise.mode != run["noise"]
            or config.noise.std_years != run["std"]
            or config.noise.seed != first_config.noise.seed
            or config.learning_rate != run["lr"]
            or result["best_epoch"] != run["bestEpoch"]
        ):
            raise ValueError(f"Run provenance mismatch: {run['id']}")
        history = json.loads(result_path.with_name("history.json").read_text())
        best = min(history, key=lambda row: row["validation"]["mae_mean"])
        if len(history) != config.epochs or best["epoch"] != result["best_epoch"]:
            raise ValueError(f"Invalid checkpoint selection: {run['id']}")
        labels = noisy_labels(ages=clean_ages, config=config.noise, epoch=best["epoch"])
        statistics = noise_summary(
            clean=torch.tensor(clean_ages)[train], noisy=labels[train]
        )
        for key, value in statistics.items():
            recorded = best["label_noise"][key]
            if value is None:
                if recorded is not None:
                    raise ValueError((run["id"], key))
            elif not np.isclose(value, recorded, atol=1e-10, rtol=1e-10):
                raise ValueError((run["id"], key, value, recorded))
        checkpoint = result_path.with_name("best.pt").resolve()
        predictions, saved = infer(
            checkpoint=checkpoint,
            config=config,
            images=images,
            targets=targets,
            device=cuda,
            batch_size=batch_size,
        )
        if (
            saved["epoch"] != result["best_epoch"]
            or saved["data_fingerprint"] != payload["fingerprint"]
            or saved["validation"] != best["validation"]
        ):
            raise ValueError(f"Checkpoint metadata mismatch: {checkpoint}")
        data["runs"].append({
            "id": run["id"],
            "method": config.loss.method,
            "noise": config.noise.model_dump(mode="json"),
            "epoch": saved["epoch"],
            "learning_rate": config.learning_rate,
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": sha256(path=checkpoint),
            "result_path": str(result_path.resolve()),
            "result_sha256": sha256(path=result_path),
            "training_commit": result["environment"]["git_commit"],
            "validation": best["validation"],
            "test": result["test"],
            "noise_reconstruction_verified": True,
            **predictions,
        })
        print(
            json.dumps({"completed": position, "total": len(runs), "run": run["id"]}),
            flush=True,
        )
    encoded = json.dumps(data, separators=(",", ":"), allow_nan=False)
    (output_dir / "policy-examples.json").write_text(encoded + "\n")
    write_viewer(data=data, path=output_dir / "viewer.html")
    audit = {
        key: value
        for key, value in data.items()
        if key not in {"examples", "corruptions", "runs"}
    }
    audit["examples"] = [
        {key: value for key, value in record.items() if key != "image"}
        for record in records
    ]
    audit["runs"] = [
        {
            key: value
            for key, value in run.items()
            if key not in {"probabilities", "summaries"}
        }
        for run in data["runs"]
    ]
    audit["checks"] = {
        "checkpoint_count": len(runs),
        "example_count": len(records),
        "probability_vectors": len(runs) * len(records),
        "all_100_labels": True,
        "corruptions_match_training_history": True,
        "matched_training_subset": True,
        "validation_mae_checkpoint_selection": True,
    }
    audit["viewer_sha256"] = sha256(path=output_dir / "viewer.html")
    (output_dir / "audit.json").write_text(
        json.dumps(audit, indent=2, allow_nan=False) + "\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--train-count", type=int, default=32)
    parser.add_argument("--test-count", type=int, default=16)
    parser.add_argument("--sample-seed", type=int, default=20261022)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    build(
        snapshot_path=args.snapshot,
        output_dir=args.output_dir,
        device=args.device,
        train_count=args.train_count,
        test_count=args.test_count,
        sample_seed=args.sample_seed,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
