import argparse
import hashlib
import json
import os
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader

from src.experiments.utkface.config import RunConfig
from src.experiments.utkface.data import FaceDataset, image_transform, read_manifest
from src.experiments.utkface.losses import log_policy
from src.experiments.utkface.model import build_model
from src.experiments.utkface.noise import noise_summary, noisy_labels
from src.experiments.utkface.policy_examples import select_runs, sha256

METHODS = ("tailrl_population", "ce")


def write_json(*, path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def collect(*, snapshot_path: Path, output_dir: Path, device: str) -> None:
    if not device.startswith("cuda:"):
        raise ValueError("Checkpoint inference requires an explicit CUDA device")
    output_dir.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    snapshot = json.loads(snapshot_path.read_text())
    runs = [run for run in select_runs(snapshot=snapshot) if run["method"] in METHODS]
    if len(runs) != 10:
        raise ValueError("Expected two methods at five matched fixed-noise settings")
    reference = RunConfig.model_validate_json(
        Path(runs[0]["resultPath"]).with_name("config.json").read_text()
    )
    payload = read_manifest(path=reference.manifest)
    datasets = [
        FaceDataset(
            manifest_path=reference.manifest,
            payload=payload,
            split=split,
            limit=4096 if split == "train" else 0,
            rank=0,
            world_size=1,
            subset_seed=reference.subset_seed,
        )
        for split in ("train", "test")
    ]
    for dataset in datasets:
        dataset.transform = image_transform(training=False)
    indices = np.array([index for dataset in datasets for index in dataset.indices])
    ages = np.array([payload["records"][index]["age"] for index in indices])
    train_indices = datasets[0].indices
    training_hash = hashlib.sha256(json.dumps(train_indices).encode()).hexdigest()
    loader = DataLoader(
        ConcatDataset(datasets),
        batch_size=reference.eval_batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        generator=torch.Generator(device="cpu").manual_seed(20261022),
    )
    audit = {
        "generated": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "snapshot_sha256": sha256(path=snapshot_path),
        "data_fingerprint": payload["fingerprint"],
        "training_fingerprint": training_hash,
        "manifest": str(reference.manifest.resolve()),
        "train_count": len(datasets[0]),
        "test_count": len(datasets[1]),
        "device": device,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "precision": "bfloat16 CNN; float32 log-softmax",
        "transform": "Resize(256), CenterCrop(224), ImageNet normalization",
        "torch": torch.__version__,
        "runs": [],
    }
    clean_ages = [record["age"] for record in payload["records"]]
    common = (
        "head",
        "seed",
        "epochs",
        "batch_size",
        "eval_batch_size",
        "weight_decay",
        "clip_grad_norm",
        "schedule",
        "pretrained",
        "precision",
        "train_limit",
        "eval_limit",
        "subset_seed",
        "learning_rate",
    )
    reference_values = reference.model_dump(mode="json")
    cuda = torch.device(device)
    for ordinal, run in enumerate(runs, start=1):
        result_path = Path(run["resultPath"])
        result = json.loads(result_path.read_text())
        config = RunConfig.model_validate_json(
            result_path.with_name("config.json").read_text()
        )
        config_values = config.model_dump(mode="json")
        if (
            any(config_values[key] != reference_values[key] for key in common)
            or result["config"] != config_values
            or result["data_fingerprint"] != payload["fingerprint"]
            or result["training_fingerprint"] != training_hash
            or config.loss.method != run["method"]
            or config.noise.mode != run["noise"]
            or config.noise.std_years != run["std"]
            or config.noise.seed != 2026
        ):
            raise ValueError(f"Unmatched source run: {run['id']}")
        checkpoint = result_path.with_name("best.pt").resolve()
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        history = json.loads(result_path.with_name("history.json").read_text())
        best = min(history, key=lambda row: row["validation"]["mae_mean"])
        if (
            saved["config"] != config_values
            or saved["data_fingerprint"] != payload["fingerprint"]
            or saved["epoch"] != best["epoch"]
            or saved["epoch"] != result["best_epoch"]
            or saved["validation"] != best["validation"]
            or len(history) != config.epochs
        ):
            raise ValueError(f"Invalid checkpoint selection: {run['id']}")
        labels = noisy_labels(ages=clean_ages, config=config.noise, epoch=best["epoch"])
        statistics = noise_summary(
            clean=torch.tensor(clean_ages)[train_indices], noisy=labels[train_indices]
        )
        for key, value in statistics.items():
            recorded = best["label_noise"][key]
            if (value is None and recorded is not None) or (
                value is not None
                and not np.isclose(value, recorded, atol=1e-10, rtol=1e-10)
            ):
                raise ValueError((run["id"], key, value, recorded))
        model = build_model(pretrained=False, head="categorical").eval()
        model.load_state_dict(saved["model"], strict=True)
        pieces, observed = [], []
        with torch.cuda.device(cuda), torch.device(cuda), torch.inference_mode():
            model = model.to(cuda)
        with torch.cuda.device(cuda), torch.inference_mode():
            for images, _, batch_indices in loader:
                with (
                    torch.device(cuda),
                    torch.autocast(device_type="cuda", dtype=torch.bfloat16),
                ):
                    logits = model(images.to(cuda, non_blocking=True))
                    log_probs = log_policy(logits=logits)
                pieces.append(log_probs.cpu().numpy())
                observed.extend(batch_indices.tolist())
        values = np.concatenate(pieces)
        if not np.array_equal(indices, observed) or not np.isfinite(values).all():
            raise ValueError("Invalid inference order or non-finite log probabilities")
        probs = np.exp(values.astype(np.float64))
        sum_error = float(np.max(np.abs(probs.sum(1) - 1)))
        if sum_error > 1e-6:
            raise ValueError((run["id"], "probability sum", sum_error))
        test = slice(len(datasets[0]), len(indices))
        means = probs[test] @ np.arange(100)
        restored = {
            "mae_mean": float(np.mean(np.abs(means - ages[test]))),
            "label_nll": float(
                -values[test][np.arange(len(datasets[1])), ages[test]].mean()
            ),
            "entropy": float(-(probs[test] * values[test]).sum(1).mean()),
        }
        differences = {key: restored[key] - result["test"][key] for key in restored}
        if any(abs(value) > 0.02 for value in differences.values()):
            raise ValueError((run["id"], "test reconstruction", differences))
        artifact = output_dir / f"{run['method']}-s{int(run['std'])}.npz"
        np.savez_compressed(artifact, log_probs=values, indices=indices, ages=ages)
        audit["runs"].append({
            "id": run["id"],
            "method": run["method"],
            "std": run["std"],
            "epoch": saved["epoch"],
            "config": config_values,
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": sha256(path=checkpoint),
            "result_sha256": sha256(path=result_path),
            "test": result["test"],
            "reconstructed_test": restored,
            "test_reconstruction_delta": differences,
            "noise_reconstruction_verified": True,
            "maximum_probability_sum_error": sum_error,
            "artifact": str(artifact),
            "artifact_sha256": sha256(path=artifact),
        })
        write_json(path=output_dir / "inference-audit.json", data=audit)
        print(
            json.dumps({"completed": ordinal, "total": len(runs), "run": run["id"]}),
            flush=True,
        )
        del model, saved


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", required=True)
    args = parser.parse_args()
    collect(snapshot_path=args.snapshot, output_dir=args.output_dir, device=args.device)


if __name__ == "__main__":
    main()
