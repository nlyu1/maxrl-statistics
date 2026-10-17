import argparse
import hashlib
import json
import os
import platform
import random
import subprocess
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Literal, get_args

import numpy as np
import torch
import torchvision
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.optim import AdamW
from torch.utils.data import DataLoader, DistributedSampler

from src.experiments.utils import set_seeds
from src.experiments.utkface.config import Method, RunConfig, canonical_config
from src.experiments.utkface.data import FaceDataset, read_manifest, seed_worker
from src.experiments.utkface.distributed import DistributedContext, distributed_context
from src.experiments.utkface.losses import policy_loss, scalar_loss
from src.experiments.utkface.metrics import (
    METRIC_NAMES,
    POINT_METRIC_NAMES,
    batch_metrics,
    median_predictions,
    median_spearman,
    metric_summary,
    scalar_metrics,
)
from src.experiments.utkface.model import build_model
from src.experiments.utkface.noise import noise_summary, noisy_labels
from src.model.optimizer import build_lr_scheduler


def write_json(*, path: Path, payload: dict | list) -> None:
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def autocast_context(*, config: RunConfig, context: DistributedContext):
    return torch.autocast(
        device_type=context.device.type,
        dtype=torch.bfloat16,
        enabled=config.precision == "bf16",
    )


def make_loader(
    *, config: RunConfig, context: DistributedContext, payload: dict, split: str
) -> DataLoader:
    training = split == "train"
    dataset = FaceDataset(
        manifest_path=config.manifest,
        payload=payload,
        split=split,
        limit=config.train_limit if training else config.eval_limit,
        rank=context.rank,
        world_size=context.world_size,
        subset_seed=config.subset_seed,
    )
    sampler = (
        DistributedSampler(
            dataset,
            num_replicas=context.world_size,
            rank=context.rank,
            shuffle=True,
            seed=config.seed,
            drop_last=True,
        )
        if training
        else None
    )
    generator = torch.Generator(device="cpu").manual_seed(config.seed + context.rank)
    return DataLoader(
        dataset,
        batch_size=config.batch_size if training else config.eval_batch_size,
        sampler=sampler,
        shuffle=False,
        num_workers=config.num_workers,
        drop_last=training,
        pin_memory=context.device.type == "cuda",
        persistent_workers=config.num_workers > 0,
        worker_init_fn=seed_worker,
        generator=generator,
    )


@torch.no_grad()
def evaluate(
    *,
    model: nn.Module,
    loader: DataLoader,
    config: RunConfig,
    context: DistributedContext,
) -> dict:
    model.eval()
    names = POINT_METRIC_NAMES if config.head == "scalar" else METRIC_NAMES
    totals = torch.zeros(len(names) + 1, device=context.device, dtype=torch.float64)
    pairs = []
    decade_totals = torch.zeros(10, 3, device=context.device, dtype=torch.float64)
    for images, targets, _ in loader:
        with (
            torch.device(context.device),
            autocast_context(config=config, context=context),
        ):
            images = images.to(context.device, non_blocking=True)
            targets = targets.to(context.device, non_blocking=True)
            logits = model(images)
            metrics = (
                scalar_metrics(predictions=logits, targets=targets)
                if config.head == "scalar"
                else batch_metrics(logits=logits, targets=targets)
            )
            medians = median_predictions(logits=logits, scalar=config.head == "scalar")
        pairs.extend(zip(medians.cpu().tolist(), targets.cpu().tolist(), strict=True))
        totals[:-1] += metrics.double().sum(0)
        totals[-1] += len(targets)
        decade = targets // 10
        decade_totals[:, 0].scatter_add_(0, decade, metrics[:, 0].double())
        decade_totals[:, 1].scatter_add_(0, decade, metrics[:, 2].double())
        decade_totals[:, 2].scatter_add_(
            0, decade, torch.ones_like(targets, dtype=torch.float64)
        )
    context.sum(values=totals)
    context.sum(values=decade_totals)
    result = metric_summary(totals=totals, names=names)
    result.update(median_spearman(pairs=context.gather_pairs(pairs=pairs)))
    if config.head == "scalar":
        result["label_nll"] = None
        result["label_nll_reason"] = (
            "Scalar regression defines no categorical label distribution"
        )
    result["decades"] = {
        str(i * 10): {
            "count": int(row[2]),
            "mae_mean": row[0] / row[2],
            "mae_median": row[1] / row[2],
        }
        for i, row in enumerate(decade_totals.cpu().tolist())
        if row[2] > 0
    }
    return result


def initial_validation(
    *,
    model: nn.Module,
    loader: DataLoader,
    config: RunConfig,
    context: DistributedContext,
    data_fingerprint: str,
    origin: Literal["recorded", "reconstructed"],
) -> dict:
    started = time.monotonic()
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    numpy_state, python_state = np.random.get_state(), random.getstate()
    training = model.training
    generator_state = loader.generator.get_state()
    devices = (
        [
            context.device.index
            if context.device.index is not None
            else torch.cuda.current_device()
        ]
        if context.device.type == "cuda"
        else []
    )
    try:
        with torch.random.fork_rng(devices=devices):
            validation = evaluate(
                model=model, loader=loader, config=config, context=context
            )
    finally:
        model.train(training)
        np.random.set_state(numpy_state)
        random.setstate(python_state)
        loader.generator.set_state(generator_state)
    return {
        "epoch": 0,
        "validation": validation,
        "model_sha256": digest.hexdigest(),
        "data_fingerprint": data_fingerprint,
        "head": config.head,
        "seed": config.seed,
        "pretrained": config.pretrained,
        "precision": config.precision,
        "eval_batch_size": config.eval_batch_size,
        "eval_limit": config.eval_limit,
        "world_size": context.world_size,
        "device": str(context.device),
        "origin": origin,
        "seconds": time.monotonic() - started,
    }


def train(*, config: RunConfig, context: DistributedContext) -> dict:
    torch.set_num_threads(config.cpu_threads)
    set_seeds(config.seed)
    torch.use_deterministic_algorithms(config.deterministic)
    torch.backends.cudnn.benchmark = not config.deterministic
    payload = read_manifest(path=config.manifest)
    if context.primary:
        config.output_dir.mkdir(parents=True, exist_ok=True)
        if (config.output_dir / "config.json").exists():
            raise FileExistsError(f"Run already exists: {config.output_dir}")
        write_json(
            path=config.output_dir / "config.json",
            payload=config.model_dump(mode="json"),
        )
    context.barrier()
    train_loader = make_loader(
        config=config, context=context, payload=payload, split="train"
    )
    val_loader = make_loader(
        config=config, context=context, payload=payload, split="val"
    )
    if len(train_loader) == 0:
        raise ValueError(
            "Training subset must contain at least one full batch per rank"
        )
    model = build_model(pretrained=config.pretrained, head=config.head).to(
        context.device
    )
    if context.world_size > 1:
        if context.device.type == "cuda":
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        wrapped = DistributedDataParallel(
            model,
            device_ids=[context.device.index]
            if context.device.type == "cuda"
            else None,
        )
    else:
        wrapped = model
    set_seeds(config.seed + context.rank)
    optimizer = AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scheduler = build_lr_scheduler(
        optimizer,
        schedule_config=config.schedule,
        train_steps=config.epochs * len(train_loader),
    )
    started = time.monotonic()
    initial = initial_validation(
        model=model,
        loader=val_loader,
        config=config,
        context=context,
        data_fingerprint=payload["fingerprint"],
        origin="recorded",
    )
    if context.primary:
        write_json(path=config.output_dir / "initial_validation.json", payload=initial)
        print(
            json.dumps({
                "epoch": 0,
                "val_mae": initial["validation"]["mae_mean"],
                "val_rmse": initial["validation"]["rmse_mean"],
                "seconds": initial["seconds"],
            }),
            flush=True,
        )
    history = []
    best_mae = float("inf")
    best_epoch = 0
    best_state = {}
    clean_ages = torch.tensor([record["age"] for record in payload["records"]])
    subset_indices = torch.tensor(train_loader.dataset.indices)
    training_fingerprint = hashlib.sha256(
        json.dumps(train_loader.dataset.indices).encode()
    ).hexdigest()
    for epoch in range(1, config.epochs + 1):
        train_loader.sampler.set_epoch(epoch - 1)
        wrapped.train()
        totals = torch.zeros(3, device=context.device, dtype=torch.float64)
        epoch_start = time.monotonic()
        labels = noisy_labels(
            ages=clean_ages.tolist(), config=config.noise, epoch=epoch
        )
        label_statistics = noise_summary(
            clean=clean_ages[subset_indices], noisy=labels[subset_indices]
        )
        labels = labels.to(context.device)
        for images, _, indices in train_loader:
            optimizer.zero_grad(set_to_none=True)
            with (
                torch.device(context.device),
                autocast_context(config=config, context=context),
            ):
                images = images.to(context.device, non_blocking=True)
                targets = labels[indices.to(context.device, non_blocking=True)]
                logits = wrapped(images)
                loss = (
                    scalar_loss(predictions=logits, targets=targets, config=config.loss)
                    if config.head == "scalar"
                    else policy_loss(logits=logits, targets=targets, config=config.loss)
                )
            loss.backward()
            norm = nn.utils.clip_grad_norm_(
                model.parameters(), config.clip_grad_norm, error_if_nonfinite=True
            )
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            totals[0] += loss.detach().double() * len(targets)
            totals[1] += norm.detach().double() * len(targets)
            totals[2] += len(targets)
        context.sum(values=totals)
        validation = evaluate(
            model=model, loader=val_loader, config=config, context=context
        )
        row = {
            "epoch": epoch,
            "train_loss": (totals[0] / totals[2]).item(),
            "grad_norm": (totals[1] / totals[2]).item(),
            "train_count": int(totals[2].item()),
            "learning_rate": optimizer.param_groups[0]["lr"],
            "validation": validation,
            "seconds": time.monotonic() - epoch_start,
            "label_noise": label_statistics,
        }
        history.append(row)
        if validation["mae_mean"] < best_mae:
            best_mae = validation["mae_mean"]
            best_epoch = epoch
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
            if context.primary and config.save_checkpoint:
                torch.save(
                    {
                        "model": best_state,
                        "config": config.model_dump(mode="json"),
                        "epoch": epoch,
                        "validation": validation,
                        "data_fingerprint": payload["fingerprint"],
                    },
                    config.output_dir / "best.pt",
                )
        if context.primary:
            write_json(path=config.output_dir / "history.json", payload=history)
            print(
                json.dumps({
                    "epoch": epoch,
                    "val_mae": validation["mae_mean"],
                    "val_rmse": validation["rmse_mean"],
                    "seconds": row["seconds"],
                }),
                flush=True,
            )
    model.load_state_dict(best_state)
    result = {
        "config": config.model_dump(mode="json"),
        "initial_validation": initial,
        "best_epoch": best_epoch,
        "validation": history[best_epoch - 1]["validation"],
        "seconds": time.monotonic() - started,
        "data_fingerprint": payload["fingerprint"],
        "training_fingerprint": training_fingerprint,
        "train_unique_images": len(train_loader.dataset),
        "world_size": context.world_size,
        "global_batch_size": config.batch_size * context.world_size,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
            "hostname": platform.node(),
            "git_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip(),
            "gpu": torch.cuda.get_device_name(context.device)
            if context.device.type == "cuda"
            else "cpu",
        },
    }
    if config.evaluate_test:
        test_loader = make_loader(
            config=config, context=context, payload=payload, split="test"
        )
        result["test"] = evaluate(
            model=model, loader=test_loader, config=config, context=context
        )
    if context.primary:
        write_json(path=config.output_dir / "result.json", payload=result)
    context.barrier()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--method", choices=get_args(Method))
    for name in (
        "seed",
        "epochs",
        "batch_size",
        "eval_batch_size",
        "num_workers",
        "train_limit",
        "eval_limit",
        "cpu_threads",
        "num_rollouts",
    ):
        parser.add_argument("--" + name.replace("_", "-"), type=int)
    for name in (
        "learning_rate",
        "kernel_scale_years",
        "survival_floor",
        "weight_decay",
    ):
        parser.add_argument("--" + name.replace("_", "-"), type=float)
    parser.add_argument("--device")
    parser.add_argument("--precision", choices=("bf16", "fp32"))
    parser.add_argument("--head", choices=("categorical", "scalar", "digits"))
    parser.add_argument("--lr-schedule", choices=("cosine", "flat"))
    for name in ("pretrained", "evaluate_test", "save_checkpoint", "deterministic"):
        parser.add_argument(
            "--" + name.replace("_", "-"), action=argparse.BooleanOptionalAction
        )
    overrides = vars(parser.parse_args())
    config_path = overrides.pop("config", None)
    base = (
        json.loads(config_path.read_text())
        if config_path
        else canonical_config(
            manifest=overrides["manifest"], output_dir=overrides["output_dir"]
        ).model_dump()
    )
    loss_keys = {"method", "kernel_scale_years", "num_rollouts", "survival_floor"}
    base["loss"] = {
        **base["loss"],
        **{key: overrides.pop(key) for key in list(overrides) if key in loss_keys},
    }
    if "lr_schedule" in overrides:
        base["schedule"] = {
            **base["schedule"],
            "lr_schedule": overrides.pop("lr_schedule"),
        }
    config = RunConfig.model_validate({**base, **overrides})
    if config.deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    with distributed_context(device=config.device) as context:
        cuda_context = (
            torch.cuda.device(context.device)
            if context.device.type == "cuda"
            else nullcontext()
        )
        with cuda_context:
            train(config=config, context=context)


if __name__ == "__main__":
    main()
