import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from src.experiments.gpu_pool import GPUPool, Job
from src.experiments.utkface.config import RunConfig
from src.experiments.utkface.data import FaceDataset, read_manifest
from src.experiments.utkface.distributed import distributed_context
from src.experiments.utkface.model import build_model
from src.experiments.utkface.run import evaluate, make_loader, write_json


def score(
    *,
    checkpoint_path: Path,
    manifest: Path,
    subset: Path,
    output_dir: Path,
    device: str,
) -> dict:
    saved = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    config = RunConfig.model_validate({
        **saved["config"],
        "manifest": manifest,
        "output_dir": output_dir,
        "device": device,
    })
    payload = read_manifest(path=manifest)
    descriptor = json.loads(subset.read_text())
    if (
        saved["data_fingerprint"] != payload["fingerprint"]
        or descriptor["data_fingerprint"] != payload["fingerprint"]
    ):
        raise ValueError("Checkpoint, subset and manifest must use the same dataset")
    requested = set(descriptor["relative_paths"])
    test_paths = {
        payload["records"][index]["relative_path"]
        for index in payload["splits"]["test"]
    }
    if (
        not requested
        or not requested <= test_paths
        or len(requested) != len(descriptor["relative_paths"])
    ):
        raise ValueError("Subset must contain unique paths from the test split")
    with (
        distributed_context(device=device) as context,
        torch.cuda.device(context.device),
    ):
        if context.world_size != 1 or context.device.type != "cuda":
            raise ValueError("Checkpoint scoring uses one CUDA GPU per process")
        torch.set_num_threads(config.cpu_threads)
        torch.backends.cudnn.benchmark = True
        model = build_model(pretrained=False).to(context.device)
        model.load_state_dict(saved["model"])
        full_loader = make_loader(
            config=config, context=context, payload=payload, split="test"
        )
        full = evaluate(model=model, loader=full_loader, config=config, context=context)
        dataset = FaceDataset(
            manifest_path=manifest,
            payload=payload,
            split="test",
            limit=0,
            rank=0,
            world_size=1,
        )
        indices = [
            index
            for index, record in enumerate(dataset.records)
            if record.relative_path in requested
        ]
        loader = DataLoader(
            Subset(dataset, indices),
            batch_size=config.eval_batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=True,
        )
        filtered = evaluate(model=model, loader=loader, config=config, context=context)
    original = json.loads((checkpoint_path.parent / "result.json").read_text())
    if saved["epoch"] != original["best_epoch"]:
        raise AssertionError("Checkpoint epoch does not match the selected epoch")
    difference = None
    if "test" in original:
        difference = abs(full["mae_mean"] - original["test"]["mae_mean"])
        if difference > 0.05 or full["count"] != original["test"]["count"]:
            raise AssertionError(
                f"Checkpoint reconstruction mismatch: MAE difference={difference}"
            )
    result = {
        "config": config.model_dump(mode="json"),
        "checkpoint": str(checkpoint_path),
        "epoch": saved["epoch"],
        "data_fingerprint": payload["fingerprint"],
        "subset_rule": descriptor["rule"],
        "test": full,
        "test_singletons": filtered,
        "restoration_mae_difference": difference,
        "original_test_verified": "test" in original,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(path=output_dir / "result.json", payload=result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--checkpoint", type=Path)
    mode.add_argument("--runs-dir", type=Path)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--subset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--gpu-ids", type=int, nargs="+", default=[0])
    args = parser.parse_args()
    if args.checkpoint is not None:
        print(
            json.dumps(
                score(
                    checkpoint_path=args.checkpoint,
                    manifest=args.manifest,
                    subset=args.subset,
                    output_dir=args.output_dir,
                    device=args.device,
                ),
                indent=2,
            )
        )
        return
    checkpoints = sorted(args.runs_dir.glob("*/best.pt"))
    if not checkpoints:
        raise ValueError("No checkpoints found")
    jobs = [
        Job(
            label=path.parent.name,
            cmd_template=[
                "uv",
                "run",
                "--python",
                "3.12",
                "python",
                "-m",
                "src.experiments.utkface.score",
                "--checkpoint",
                str(path),
                "--manifest",
                str(args.manifest),
                "--subset",
                str(args.subset),
                "--output-dir",
                str(args.output_dir / path.parent.name),
                "--device",
                "{device}",
            ],
        )
        for path in checkpoints
    ]
    results = GPUPool(device_ids=args.gpu_ids).run(
        jobs, log_dir=args.output_dir / "logs", fail_fast=True
    )
    if len(results) != len(jobs) or any(result.returncode != 0 for result in results):
        raise RuntimeError("Checkpoint scoring failed; inspect the per-run logs")


if __name__ == "__main__":
    main()
