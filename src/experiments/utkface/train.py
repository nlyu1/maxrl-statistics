from __future__ import annotations

import hashlib
import json
import platform
import random
import shutil
import subprocess
import time
from importlib.metadata import version
from pathlib import Path

import numpy as np
import polars as pl
import torch
import torch.distributed as dist
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader

from src.experiments.utils import set_seeds
from src.experiments.utkface.config import UTKFaceConfig
from src.experiments.utkface.data import DataBundle, build_data
from src.experiments.utkface.distributed import (
    Runtime,
    execution_context,
    finish_runtime,
    initialize_runtime,
    sum_across_ranks,
    synchronize,
    unwrap,
    wrap_distributed,
)
from src.experiments.utkface.evaluate import evaluate
from src.experiments.utkface.model import create_policy
from src.experiments.utkface.objectives import compute_loss
from src.model.optimizer import build_lr_scheduler


def atomic_json(*, path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def model_digest(*, model: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def environment_metadata(*, runtime: Runtime) -> dict:
    return {
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
        ).strip(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {
            name: version(name)
            for name in (
                "torch",
                "torchvision",
                "numpy",
                "scikit-learn",
                "pydantic",
                "polars",
            )
        },
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "device": str(runtime.device),
        "world_size": runtime.world_size,
        "gpu": torch.cuda.get_device_name(runtime.device)
        if runtime.device.type == "cuda"
        else "cpu",
    }


def atomic_copy(*, source: Path, destination: Path) -> None:
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    shutil.copyfile(source, temporary)
    temporary.replace(destination)


def capture_rng(*, data: DataBundle, runtime: Runtime) -> dict:
    state = {
        "torch": torch.get_rng_state(),
        "numpy": np.random.get_state(),
        "python": random.getstate(),
        "loader": data.generator.get_state(),
    }
    if runtime.device.type == "cuda":
        state["cuda"] = torch.cuda.get_rng_state(runtime.device)
    return state


def restore_rng(*, state: dict, data: DataBundle, runtime: Runtime) -> None:
    torch.set_rng_state(state["torch"].cpu())
    np.random.set_state(state["numpy"])
    random.setstate(state["python"])
    data.generator.set_state(state["loader"].cpu())
    if runtime.device.type == "cuda":
        torch.cuda.set_rng_state(state["cuda"].cpu(), runtime.device)


def save_checkpoint(
    *,
    path: Path,
    model: nn.Module,
    optimizer: Optimizer,
    scheduler: LRScheduler,
    config: UTKFaceConfig,
    data: DataBundle,
    runtime: Runtime,
    epoch: int,
    best_epoch: int,
    best_mae: float,
    history: list[dict],
    initial_digest: str,
) -> None:
    rank_rng = capture_rng(data=data, runtime=runtime)
    rng_states = [None] * runtime.world_size
    if runtime.world_size > 1:
        dist.all_gather_object(rng_states, rank_rng)
    else:
        rng_states[0] = rank_rng
    if runtime.primary:
        state = {
            "model": unwrap(model=model).state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "config": config.scientific_settings(),
            "split_sha256": data.manifest["sha256"],
            "world_size": runtime.world_size,
            "epoch": epoch,
            "best_epoch": best_epoch,
            "best_mae": best_mae,
            "history": history,
            "rng_states": rng_states,
            "initial_digest": initial_digest,
        }
        temporary = path.with_suffix(".pt.tmp")
        torch.save(state, temporary)
        temporary.replace(path)


def train_epoch(
    *,
    model: nn.Module,
    loader: DataLoader,
    optimizer: Optimizer,
    scheduler: LRScheduler,
    config: UTKFaceConfig,
    runtime: Runtime,
) -> dict[str, float]:
    model.train()
    totals = torch.zeros(3, device=runtime.device, dtype=torch.float64)
    started = time.perf_counter()
    for images, targets in loader:
        images = images.to(runtime.device, non_blocking=True)
        targets = targets.to(runtime.device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with execution_context(runtime=runtime, precision=config.training.precision):
            logits = model(images)
            with torch.autocast(device_type=runtime.device.type, enabled=False):
                loss = compute_loss(
                    logits=logits, targets=targets, config=config.objective
                )
        if not torch.isfinite(loss):
            raise FloatingPointError(
                f"Nonfinite {config.objective.method} training loss."
            )
        loss.backward()
        norm = nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=config.training.gradient_clip,
            error_if_nonfinite=True,
        )
        optimizer.step()
        scheduler.step()
        count = targets.numel()
        totals[0] += loss.detach().double() * count
        totals[1] += norm.detach().double() * count
        totals[2] += count
    totals = sum_across_ranks(values=totals, runtime=runtime)
    return {
        "train/loss": (totals[0] / totals[2]).item(),
        "train/gradient_norm": (totals[1] / totals[2]).item(),
        "train/examples": totals[2].item(),
        "train/seconds": time.perf_counter() - started,
        "train/learning_rate": scheduler.get_last_lr()[0],
    }


def run_training(*, config: UTKFaceConfig, resume: bool) -> dict:
    runtime = initialize_runtime(device=config.training.device)
    try:
        return _run_training(config=config, resume=resume, runtime=runtime)
    finally:
        finish_runtime(runtime=runtime)


def _run_training(*, config: UTKFaceConfig, resume: bool, runtime: Runtime) -> dict:
    torch.set_num_threads(4)
    set_seeds(seed=config.seed)
    if runtime.device.type == "cuda":
        if config.training.precision == "bf16" and not torch.cuda.is_bf16_supported():
            raise RuntimeError(
                "This GPU does not support bfloat16; request fp32 explicitly."
            )
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    output = config.output_directory
    checkpoint_directory = output / "checkpoints"
    if runtime.primary:
        checkpoint_directory.mkdir(parents=True, exist_ok=True)
    synchronize(runtime=runtime)
    config_path = output / "config.json"
    if config_path.exists():
        previous = UTKFaceConfig.model_validate_json(config_path.read_text())
        if previous.scientific_settings() != config.scientific_settings():
            raise ValueError(
                f"Output directory belongs to a different configuration: {output}"
            )
        if not resume:
            raise FileExistsError(f"Run already exists; use --resume: {output}")
    elif runtime.primary:
        atomic_json(path=config_path, value=config.model_dump(mode="json"))
    synchronize(runtime=runtime)
    summary_path = output / "summary.json"
    data = build_data(
        config=config.data,
        global_batch_size=config.training.global_batch_size,
        eval_batch_size=config.training.eval_batch_size,
        seed=config.seed,
        rank=runtime.rank,
        world_size=runtime.world_size,
    )
    manifest_path = output / "splits.json"
    if manifest_path.exists():
        if json.loads(manifest_path.read_text())["sha256"] != data.manifest["sha256"]:
            raise ValueError(
                "Output directory dataset manifest differs from the current dataset."
            )
    elif runtime.primary:
        atomic_json(path=manifest_path, value=data.manifest)
    if resume and summary_path.exists():
        return json.loads(summary_path.read_text())
    metadata = environment_metadata(runtime=runtime)
    if runtime.primary:
        atomic_json(path=output / "environment.json", value=metadata)
    model = create_policy(pretrained=config.pretrained, seed=config.seed)
    initial_digest = model_digest(model=model)
    model = wrap_distributed(model=model, runtime=runtime)
    set_seeds(seed=config.seed + runtime.rank)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=config.training.learning_rate,
        momentum=config.training.momentum,
        weight_decay=config.training.weight_decay,
    )
    scheduler = build_lr_scheduler(
        optimizer,
        schedule_config=config.training.schedule,
        train_steps=config.training.epochs * len(data.train),
    )
    if scheduler is None:
        raise ValueError("UTKFace training requires an explicit cosine schedule.")
    history = []
    start_epoch = 1
    best_epoch = 0
    best_mae = float("inf")
    last = checkpoint_directory / "last.pt"
    best = checkpoint_directory / "best.pt"
    if resume and last.exists():
        checkpoint = torch.load(last, map_location=runtime.device, weights_only=False)
        if checkpoint["config"] != config.scientific_settings():
            raise ValueError("Checkpoint configuration differs from the requested run.")
        if checkpoint["split_sha256"] != data.manifest["sha256"]:
            raise ValueError(
                "Checkpoint dataset split differs from the current dataset."
            )
        if checkpoint["world_size"] != runtime.world_size:
            raise ValueError(
                "Exact resumption requires the checkpoint's original world_size."
            )
        unwrap(model=model).load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        restore_rng(
            state=checkpoint["rng_states"][runtime.rank], data=data, runtime=runtime
        )
        start_epoch = checkpoint["epoch"] + 1
        best_epoch = checkpoint["best_epoch"]
        best_mae = checkpoint["best_mae"]
        history = checkpoint["history"]
        initial_digest = checkpoint["initial_digest"]
        if runtime.primary and best_epoch == checkpoint["epoch"]:
            atomic_copy(source=last, destination=best)
        synchronize(runtime=runtime)
    else:
        initial = evaluate(
            model=model, loader=data.validation, config=config, runtime=runtime
        )
        history.append({
            "epoch": 0,
            **{f"validation/{key}": value for key, value in initial.items()},
        })
    started = time.perf_counter()
    for epoch in range(start_epoch, config.training.epochs + 1):
        data.train_sampler.set_epoch(epoch)
        train_metrics = train_epoch(
            model=model,
            loader=data.train,
            optimizer=optimizer,
            scheduler=scheduler,
            config=config,
            runtime=runtime,
        )
        validation = evaluate(
            model=model, loader=data.validation, config=config, runtime=runtime
        )
        improved = validation["mean_mae"] < best_mae
        if improved:
            best_mae = validation["mean_mae"]
            best_epoch = epoch
        history.append({
            "epoch": epoch,
            **train_metrics,
            **{f"validation/{key}": value for key, value in validation.items()},
        })
        save_checkpoint(
            path=last,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            config=config,
            data=data,
            runtime=runtime,
            epoch=epoch,
            best_epoch=best_epoch,
            best_mae=best_mae,
            history=history,
            initial_digest=initial_digest,
        )
        if runtime.primary:
            if improved:
                atomic_copy(source=last, destination=best)
            if epoch in config.training.checkpoint_epochs:
                atomic_copy(
                    source=last, destination=checkpoint_directory / f"epoch_{epoch}.pt"
                )
            atomic_json(path=output / "metrics.json", value=history)
            pl.DataFrame(history, infer_schema_length=None).write_parquet(
                output / "metrics.parquet"
            )
            print(
                f"{config.objective.method} seed={config.seed} epoch={epoch}/{config.training.epochs} "
                f"val_mean_mae={validation['mean_mae']:.3f} "
                f"val_greedy_mae={validation['greedy_mae']:.3f} "
                f"val_expected_mae={validation['expected_mae']:.3f} "
                f"entropy={validation['entropy']:.3f} "
                f"train_seconds={train_metrics['train/seconds']:.1f}",
                flush=True,
            )
        synchronize(runtime=runtime)
    best_checkpoint = torch.load(best, map_location=runtime.device, weights_only=False)
    unwrap(model=model).load_state_dict(best_checkpoint["model"])
    test = evaluate(model=model, loader=data.test, config=config, runtime=runtime)
    summary = {
        "method": config.objective.method,
        "seed": config.seed,
        "temperature_years": config.objective.temperature_years,
        "num_rollouts": config.objective.num_rollouts,
        "reward": config.objective.reward,
        "precision": config.training.precision,
        "world_size": runtime.world_size,
        "environment": metadata,
        "initial_model_sha256": initial_digest,
        "split_sha256": data.manifest["sha256"],
        "split_counts": data.manifest["counts"],
        "best_epoch": best_epoch,
        "best_validation_mean_mae": best_mae,
        "validation": best_checkpoint["history"][-1],
        "test": test,
        "elapsed_seconds_this_invocation": time.perf_counter() - started,
    }
    if runtime.primary:
        atomic_json(path=summary_path, value=summary)
        print(json.dumps(summary, indent=2), flush=True)
    return summary
