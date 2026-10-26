import argparse
import hashlib
import json
import os
import platform
import random
import re
import subprocess
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
import torchvision
from torch import Tensor, nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from src.experiments.utils import cleanup_cuda, set_seeds
from src.experiments.utkface.adaptation_config import AdaptationConfig
from src.experiments.utkface.adaptation_data import (
    build_image_loader,
    epoch_order,
)
from src.experiments.utkface.adaptation_metrics import evaluate_outputs
from src.experiments.utkface.data import read_manifest
from src.experiments.utkface.feature_bank import load_feature_bank
from src.experiments.utkface.losses import policy_loss, scalar_loss
from src.experiments.utkface.model import build_model
from src.experiments.utkface.noise import noisy_labels
from src.experiments.utkface.run import write_json

HEAD_SEED = 42
REGULAR_CHECKPOINT = re.compile(r"^epoch_(\d{3})\.pt$")
SOURCE_FILES = (
    "src/experiments/utkface/adaptation_config.py",
    "src/experiments/utkface/adaptation_data.py",
    "src/experiments/utkface/adaptation_metrics.py",
    "src/experiments/utkface/adaptation_train.py",
    "src/experiments/utkface/data.py",
    "src/experiments/utkface/distributed.py",
    "src/experiments/utkface/feature_bank.py",
    "src/experiments/utkface/losses.py",
    "src/experiments/utkface/metrics.py",
    "src/experiments/utkface/model.py",
    "src/experiments/utkface/noise.py",
    "src/model/optimizer.py",
)


def file_sha256(*, path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(2**20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_sha256(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def model_sha256(*, state: dict[str, Tensor]) -> str:
    digest = hashlib.sha256()
    for name, value in state.items():
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def serialized_rng_state() -> dict:
    numpy_state = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": {
            "name": numpy_state[0],
            "keys": numpy_state[1].tolist(),
            "position": numpy_state[2],
            "has_gauss": numpy_state[3],
            "cached_gaussian": numpy_state[4],
        },
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all(),
    }


def restore_rng_state(*, state: dict) -> None:
    numpy_state = state["numpy"]
    random.setstate(state["python"])
    np.random.set_state((
        numpy_state["name"],
        np.asarray(numpy_state["keys"], dtype=np.uint32),
        numpy_state["position"],
        numpy_state["has_gauss"],
        numpy_state["cached_gaussian"],
    ))
    torch.set_rng_state(state["torch_cpu"])
    torch.cuda.set_rng_state_all(state["torch_cuda"])


def config_payload(*, config: AdaptationConfig) -> dict:
    return config.model_dump(mode="json")


def source_root() -> Path:
    return Path(__file__).resolve().parents[3]


def source_sha256() -> dict[str, str]:
    root = source_root()
    return {
        name: file_sha256(path=root / name)
        for name in SOURCE_FILES
        if (root / name).is_file()
    }


def git_head() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=source_root(),
        text=True,
    ).strip()


def fresh_head(*, model: nn.Module, head: str) -> nn.Module:
    if head == "categorical":
        return nn.Linear(model.fc.in_features, model.fc.out_features)
    if head == "scalar":
        return model.fc.__class__(features=model.fc.linear.in_features)
    if head == "digits":
        return model.fc.__class__(features=model.fc.tens.in_features)
    raise ValueError(f"Unsupported head: {head}")


def common_head_state(*, head: str) -> dict[str, Tensor]:
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(HEAD_SEED)
        template = build_model(pretrained=False, head=head)
        candidate = fresh_head(model=template, head=head)
    return {
        name: value.detach().cpu().clone()
        for name, value in candidate.state_dict().items()
    }


def apply_common_head(*, model: nn.Module, head: str) -> None:
    model.fc.load_state_dict(common_head_state(head=head), strict=True)


def load_native_checkpoint(*, path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(saved, dict) or not isinstance(saved.get("model"), dict):
        raise TypeError(f"Checkpoint has no native model state: {path}")
    return saved


def provenance(
    *,
    config: AdaptationConfig,
    manifest: dict,
    bank: dict | None,
    initialization_source: dict,
) -> dict:
    common_head = common_head_state(head=config.run.head)
    source = {
        "adaptation_config": config_payload(config=config),
        "manifest_fingerprint": manifest["fingerprint"],
        "pipeline": config.pipeline,
        "initialization": config.initialization,
        "initial_checkpoint": str(config.initial_checkpoint)
        if config.initialization == "checkpoint"
        else None,
        "initial_checkpoint_sha256": file_sha256(path=config.initial_checkpoint)
        if config.initialization == "checkpoint"
        else None,
        "backbone_frozen": config.backbone_frozen,
        "feature_bank": str(config.feature_bank) if bank is not None else None,
        "feature_bank_sha256": bank["metadata"]["bank_sha256"]
        if bank is not None
        else None,
        "feature_bank_source": bank["metadata"]["source"]["fingerprint"]
        if bank is not None
        else None,
        "head": config.run.head,
        "head_initialization": initialization_source["head_initialization"],
        "common_head_sha256": model_sha256(state=common_head),
        "global_start_epoch": config.global_start_epoch,
        "local_epochs": config.run.epochs,
        "final_global_epoch": config.global_start_epoch + config.run.epochs,
        "view_seed": config.view_seed,
        "order_seed": config.order_seed,
        "schedule_kind": config.schedule_kind,
        "precision": config.run.precision,
        "extension_source": extension_origin(config=config),
        "environment": {
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "torchvision": str(torchvision.__version__),
            "cuda": str(torch.version.cuda),
            "hostname": platform.node(),
            "git_commit": git_head(),
        },
        "source_sha256": source_sha256(),
    }
    return {**source, "fingerprint": json_sha256(source)}


def bank_source_checkpoint(*, bank: dict) -> Path | None:
    source = bank["metadata"]["source"]["initialization"]
    checkpoint = source["checkpoint"]
    if checkpoint["path"] is None:
        return None
    path = Path(checkpoint["path"])
    if file_sha256(path=path) != checkpoint["sha256"]:
        raise ValueError(f"Feature-bank source checkpoint changed: {path}")
    return path


def build_initial_model(
    *, config: AdaptationConfig, bank: dict | None
) -> tuple[nn.Module, dict]:
    if bank is None:
        if config.initialization == "checkpoint":
            saved = load_native_checkpoint(path=config.initial_checkpoint)
            saved_payload = saved.get("adaptation_config", saved.get("config"))
            saved_head = None
            if isinstance(saved_payload, dict):
                if isinstance(saved_payload.get("run"), dict):
                    saved_head = saved_payload["run"].get("head")
                elif isinstance(saved_payload.get("head"), str):
                    saved_head = saved_payload["head"]
            if (
                saved_head is not None
                and saved_head != config.run.head
                and config.backbone_frozen
            ):
                source_state = {
                    name: value
                    for name, value in saved["model"].items()
                    if not name.startswith("fc.")
                }
                model = build_model(pretrained=False, head=config.run.head)
                model.load_state_dict(source_state, strict=False)
                apply_common_head(model=model, head=config.run.head)
            else:
                model = build_model(pretrained=False, head=config.run.head)
                model.load_state_dict(saved["model"], strict=True)
            initialization_source = {
                "kind": "checkpoint",
                "path": str(config.initial_checkpoint),
                "sha256": file_sha256(path=config.initial_checkpoint),
                "retained_head": not (
                    saved_head is not None
                    and saved_head != config.run.head
                    and config.backbone_frozen
                ),
                "head_initialization": (
                    "fresh_common_seed_42"
                    if (
                        saved_head is not None
                        and saved_head != config.run.head
                        and config.backbone_frozen
                    )
                    else "retained_from_checkpoint"
                ),
            }
        elif config.initialization in {"imagenet", "scratch"}:
            model = build_model(
                pretrained=config.initialization == "imagenet",
                head=config.run.head,
            )
            apply_common_head(model=model, head=config.run.head)
            initialization_source = {
                "kind": config.initialization,
                "path": None,
                "sha256": None,
                "head_initialization": "fresh_common_seed_42",
            }
        else:
            raise ValueError(f"Unsupported initialization: {config.initialization}")
    else:
        source_initialization = bank["metadata"]["source"]["initialization"]
        source_kind = source_initialization["initialization"]
        if config.initialization == "checkpoint" and source_kind != "checkpoint":
            raise ValueError(
                "Checkpoint-initialized training requires a checkpoint feature bank"
            )
        if source_kind == "imagenet":
            model = build_model(pretrained=True, head=config.run.head)
            initialization_source = {
                "kind": "feature_bank_imagenet",
                "path": None,
                "sha256": None,
                "head_initialization": "fresh_common_seed_42",
            }
        elif source_kind == "checkpoint":
            path = bank_source_checkpoint(bank=bank)
            assert path is not None
            saved = load_native_checkpoint(path=path)
            source_head = bank["metadata"]["config"]["run"]["head"]
            if source_head == config.run.head:
                model = build_model(pretrained=False, head=config.run.head)
                model.load_state_dict(saved["model"], strict=True)
            else:
                model = build_model(pretrained=False, head=config.run.head)
                source_state = {
                    name: value
                    for name, value in saved["model"].items()
                    if not name.startswith("fc.")
                }
                model.load_state_dict(source_state, strict=False)
                apply_common_head(model=model, head=config.run.head)
            initialization_source = {
                "kind": "feature_bank_checkpoint",
                "path": str(path),
                "sha256": file_sha256(path=path),
                "source_head": source_head,
                "head_initialization": (
                    "retained_from_checkpoint"
                    if config.initialization == "checkpoint"
                    and source_head == config.run.head
                    else "fresh_common_seed_42"
                ),
            }
        else:
            raise ValueError(f"Unsupported feature-bank source: {source_kind}")
        if config.initialization != "checkpoint":
            apply_common_head(model=model, head=config.run.head)

    head_state = model.fc.state_dict()
    return model, {
        **initialization_source,
        "head_sha256": model_sha256(state=head_state),
    }


def validate_bank(*, config: AdaptationConfig, bank: dict, manifest: dict) -> None:
    metadata = bank["metadata"]
    bank_config = metadata["config"]
    first = config.global_start_epoch + 1
    final = config.global_start_epoch + config.run.epochs
    assigned = list(range(first, final + 1))
    if bank_config["pipeline"] != config.pipeline:
        raise ValueError("Feature-bank pipeline differs from training pipeline")
    if config.initialization == "checkpoint":
        source_initialization = metadata["source"]["initialization"]
        source_checkpoint = source_initialization["checkpoint"]
        if (
            source_initialization["initialization"] != "checkpoint"
            or source_checkpoint["path"] is None
            or Path(source_checkpoint["path"]).resolve()
            != config.initial_checkpoint.resolve()
        ):
            raise ValueError(
                "Training checkpoint differs from the feature-bank source checkpoint"
            )
        if file_sha256(path=config.initial_checkpoint) != source_checkpoint["sha256"]:
            raise ValueError("Feature-bank source checkpoint changed")
    if bank_config["run"]["train_limit"] != config.run.train_limit:
        raise ValueError("Feature-bank training limit differs")
    if bank_config["run"]["subset_seed"] != config.run.subset_seed:
        raise ValueError("Feature-bank subset seed differs")
    if metadata["data"]["manifest"]["fingerprint"] != manifest["fingerprint"]:
        raise ValueError("Feature-bank manifest differs")
    if config.run.train_limit and len(bank["train_indices"]) != config.run.train_limit:
        raise ValueError("Feature-bank training row count differs")
    if bank["train_features"].shape[-1] != 512:
        raise ValueError("Feature-bank width differs from ResNet-18 layer4")
    if metadata["epochs"]["assigned"] != assigned:
        raise ValueError("Feature-bank epochs differ from requested global epochs")


def build_optimizer(*, config: AdaptationConfig, model: nn.Module) -> AdamW:
    if config.backbone_frozen:
        for name, parameter in model.named_parameters():
            parameter.requires_grad = name.startswith("fc.")
        return AdamW(
            model.fc.parameters(),
            lr=config.run.learning_rate,
            weight_decay=config.run.weight_decay,
        )

    head_parameters = list(model.fc.parameters())
    head_ids = {id(parameter) for parameter in head_parameters}
    backbone_parameters = [
        parameter for parameter in model.parameters() if id(parameter) not in head_ids
    ]
    return AdamW(
        [
            {
                "params": backbone_parameters,
                "lr": config.backbone_learning_rate,
            },
            {
                "params": head_parameters,
                "lr": config.run.learning_rate,
            },
        ],
        weight_decay=config.run.weight_decay,
    )


def warmup_constant_scheduler(
    *, optimizer: AdamW, total_steps: int, warmup_ratio: float
) -> LambdaLR:
    if total_steps < 1:
        raise ValueError("Training must contain at least one step")
    if not 0 <= warmup_ratio < 1:
        raise ValueError("warmup_ratio must be in [0, 1)")
    warmup_steps = round(total_steps * warmup_ratio)
    if warmup_steps >= total_steps:
        raise ValueError("Warmup must end before training ends")

    def scale(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))
        return 1.0

    return LambdaLR(optimizer, lr_lambda=scale)


def build_scheduler(
    *, config: AdaptationConfig, optimizer: AdamW, steps_per_epoch: int
) -> LambdaLR | None:
    total_steps = steps_per_epoch * config.control.schedule_epochs
    if config.schedule_kind == "warmup_constant":
        return warmup_constant_scheduler(
            optimizer=optimizer,
            total_steps=total_steps,
            warmup_ratio=config.run.schedule.warmup_ratio,
        )
    if config.schedule_kind == "cosine":
        from src.model.optimizer import build_lr_scheduler

        return build_lr_scheduler(
            optimizer,
            schedule_config=config.run.schedule.model_copy(
                update={"lr_schedule": "cosine"}
            ),
            train_steps=total_steps,
        )
    raise ValueError(f"Unsupported schedule: {config.schedule_kind}")


def model_state_cpu(*, model: nn.Module) -> dict[str, Tensor]:
    return {
        name: value.detach().cpu().clone() for name, value in model.state_dict().items()
    }


def filter_head_state(*, state: dict[str, Tensor]) -> dict[str, Tensor]:
    head = {
        name: value.detach().cpu().clone()
        for name, value in state.items()
        if name == "fc" or name.startswith("fc.")
    }
    if not head:
        raise ValueError("Model state has no fc head parameters")
    return head


def expand_head_state(
    *, static_state: dict[str, Tensor], head: dict[str, Tensor]
) -> dict[str, Tensor]:
    expected = {name for name in static_state if name == "fc" or name.startswith("fc.")}
    if set(head) != expected:
        missing = expected - set(head)
        unexpected = set(head) - expected
        raise ValueError(
            f"Head checkpoint keys differ: missing={sorted(missing)}, "
            f"unexpected={sorted(unexpected)}"
        )
    return {
        **{
            name: value.detach().cpu().clone()
            for name, value in static_state.items()
            if name not in expected
        },
        **head,
    }


def save_native_model(
    *,
    path: Path,
    model: nn.Module,
    config: AdaptationConfig,
    epoch: int,
    validation: dict,
    manifest_fingerprint: str,
    run_provenance: dict,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".pt.tmp")
    checkpoint_model = {
        name: value.detach().cpu().clone() for name, value in model.state_dict().items()
    }
    torch.save(
        {
            "model": checkpoint_model,
            "model_sha256": model_sha256(state=checkpoint_model),
            "config": config.run.model_dump(mode="json"),
            "adaptation_config": config_payload(config=config),
            "epoch": epoch,
            "validation": validation,
            "head": config.run.head,
            "pretrained": config.initialization == "imagenet",
            "data_fingerprint": manifest_fingerprint,
            "provenance": run_provenance,
        },
        temporary,
    )
    temporary.replace(path)


def save_full_checkpoint(
    *,
    path: Path,
    model: nn.Module,
    optimizer: AdamW,
    scheduler: LambdaLR | None,
    config: AdaptationConfig,
    epoch: int,
    validation: dict,
    history: list[dict],
    initialization_record: dict,
    best_epoch: int,
    best_validation: dict,
    best_model: dict[str, Tensor],
    initialization_source: dict,
    bank_metadata: dict | None,
    run_provenance: dict,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".pt.tmp")
    checkpoint_scope = "head" if config.backbone_frozen else "full"
    if checkpoint_scope == "head":
        if bank_metadata is None:
            raise ValueError(
                "Frozen training requires feature-bank checkpoint metadata"
            )
        model_payload = filter_head_state(state=model.state_dict())
        best_model_payload = filter_head_state(state=best_model)
    else:
        model_payload = model_state_cpu(model=model)
        best_model_payload = best_model
    torch.save(
        {
            "checkpoint_scope": checkpoint_scope,
            "head_scope": "fc" if checkpoint_scope == "head" else None,
            "model": model_payload,
            "best_model": best_model_payload,
            "model_sha256": model_sha256(state=model_payload),
            "best_model_sha256": model_sha256(state=best_model_payload),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "rng": serialized_rng_state(),
            "config": config_payload(config=config),
            "epoch": epoch,
            "validation": validation,
            "history": history,
            "initialization_record": initialization_record,
            "best_epoch": best_epoch,
            "best_validation": best_validation,
            "initialization_source": initialization_source,
            "bank_metadata": bank_metadata,
            "head": config.run.head,
            "pretrained": config.initialization == "imagenet",
            "provenance": run_provenance,
        },
        temporary,
    )
    temporary.replace(path)


def load_full_checkpoint(
    *,
    path: Path,
    config: AdaptationConfig,
    model: nn.Module,
    optimizer: AdamW,
    scheduler: LambdaLR | None,
    initialization_source: dict,
    bank_metadata: dict | None,
    run_provenance: dict,
) -> dict:
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if saved.get("config") != config_payload(config=config):
        raise ValueError(f"Checkpoint config differs: {path}")
    expected_scope = "head" if config.backbone_frozen else "full"
    if saved.get("checkpoint_scope") != expected_scope or saved.get("head_scope") != (
        "fc" if expected_scope == "head" else None
    ):
        raise ValueError(f"Checkpoint scope differs: {path}")
    if saved.get("initialization_source") != initialization_source:
        raise ValueError(f"Checkpoint initialization source differs: {path}")
    checkpoint_model_sha256 = model_sha256(state=saved["model"])
    checkpoint_best_model_sha256 = model_sha256(state=saved["best_model"])
    if expected_scope == "head":
        if bank_metadata is None:
            raise ValueError(f"Frozen checkpoint has no current bank metadata: {path}")
        if saved.get("bank_metadata") != bank_metadata:
            raise ValueError(f"Checkpoint feature-bank metadata differs: {path}")
        static_state = model_state_cpu(model=model)
        saved["model"] = expand_head_state(
            static_state=static_state,
            head=saved["model"],
        )
        saved["best_model"] = expand_head_state(
            static_state=static_state,
            head=saved["best_model"],
        )
    elif saved.get("bank_metadata") is not None:
        raise ValueError(
            f"Full checkpoint unexpectedly has feature-bank metadata: {path}"
        )
    if checkpoint_model_sha256 != saved.get("model_sha256"):
        raise ValueError(f"Checkpoint model hash differs: {path}")
    if checkpoint_best_model_sha256 != saved.get("best_model_sha256"):
        raise ValueError(f"Checkpoint best-model hash differs: {path}")
    saved_provenance = saved.get("provenance", {})
    for key in (
        "initial_checkpoint_sha256",
        "feature_bank_sha256",
        "manifest_fingerprint",
    ):
        if saved_provenance.get(key) != run_provenance.get(key):
            raise ValueError(f"Checkpoint provenance differs for {key}: {path}")
    model.load_state_dict(saved["model"], strict=True)
    optimizer.load_state_dict(saved["optimizer"])
    if (scheduler is None) != (saved["scheduler"] is None):
        raise ValueError(f"Checkpoint scheduler presence differs: {path}")
    if scheduler is not None:
        scheduler.load_state_dict(saved["scheduler"])
    restore_rng_state(state=saved["rng"])
    return saved


def regular_checkpoints(*, output_dir: Path) -> list[tuple[int, Path]]:
    root = output_dir / "checkpoints"
    if not root.is_dir():
        return []
    result = []
    for path in root.glob("epoch_*.pt"):
        match = REGULAR_CHECKPOINT.fullmatch(path.name)
        if match is not None:
            result.append((int(match.group(1)), path))
    return sorted(result)


def extension_origin(*, config: AdaptationConfig) -> dict | None:
    if not config.control.extend_epoch:
        return None
    path = config.control.extend_checkpoint
    saved = torch.load(path, map_location="cpu", weights_only=True)
    return {
        "checkpoint": str(path),
        "sha256": file_sha256(path=path),
        "epoch": saved["epoch"],
        "config": saved["config"],
        "provenance": saved["provenance"],
        "model_sha256": saved["model_sha256"],
        "optimizer_and_rng_retained": True,
    }


def prepare_extension_checkpoint(
    *, config: AdaptationConfig, run_provenance: dict, initialization_source: dict
) -> None:
    path = config.control.extend_checkpoint
    saved = torch.load(path, map_location="cpu", weights_only=True)
    source_config = saved["config"]
    source = {**source_config, "run": {**source_config["run"]}}
    target = config_payload(config=config)
    target = {**target, "run": {**target["run"]}}
    source_control = source.pop("control", None)
    target.pop("control")
    schedule_epochs = (
        source_control["schedule_epochs"] if source_control else source["run"]["epochs"]
    )
    if schedule_epochs != config.control.schedule_epochs:
        raise ValueError("Extension changes the completed warmup schedule")
    if (
        source["schedule_kind"] != "warmup_constant"
        or source["initialization"] != "scratch"
    ):
        raise ValueError(
            "Only scratch constant-LR checkpoints can extend their horizon"
        )
    for key in ("output_dir", "epochs", "device", "evaluate_test"):
        source["run"][key] = target["run"][key]
    for key in ("initial_checkpoint", "feature_bank"):
        source[key] = target[key]
    if source != target:
        raise ValueError(
            "Extension changes training settings beyond horizon/evaluation/output"
        )
    epoch = config.control.extend_epoch
    if saved["epoch"] != epoch or [row["epoch"] for row in saved["history"]] != list(
        range(1, epoch + 1)
    ):
        raise ValueError("Extension checkpoint has an incomplete prefix history")
    if (
        saved["checkpoint_scope"] != "full"
        or saved["initialization_source"] != initialization_source
    ):
        raise ValueError("Extension checkpoint initialization or scope differs")
    if (
        saved["provenance"]["manifest_fingerprint"]
        != run_provenance["manifest_fingerprint"]
    ):
        raise ValueError("Extension checkpoint uses another manifest")
    if not all(key in saved for key in ("optimizer", "scheduler", "rng", "best_model")):
        raise ValueError(
            "Extension requires optimizer, scheduler, RNG and best-model state"
        )
    checkpoint = config.run.output_dir / "checkpoints" / f"epoch_{epoch:03d}.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    if checkpoint.exists():
        raise FileExistsError(checkpoint)
    migrated = {
        **saved,
        "config": config_payload(config=config),
        "provenance": run_provenance,
        "extension_origin": run_provenance["extension_source"],
    }
    temporary = checkpoint.with_suffix(".pt.tmp")
    torch.save(migrated, temporary)
    temporary.replace(checkpoint)


def prune_regular_checkpoints(*, config: AdaptationConfig) -> None:
    checkpoints = regular_checkpoints(output_dir=config.run.output_dir)
    directory = (config.run.output_dir / "checkpoints").resolve()
    for _, path in checkpoints[: -config.control.keep_latest_checkpoints]:
        if path.resolve().parent != directory or path.is_symlink():
            raise ValueError("Checkpoint retention encountered a path outside its run")
        path.unlink()


def archive_stale_artifacts(*, output_dir: Path) -> None:
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    for name in (
        "history.json",
        "predictions.json",
        "best-validation-predictions.json",
        "best-test-predictions.json",
        "final-validation-predictions.json",
        "final-test-predictions.json",
    ):
        path = output_dir / name
        if path.exists():
            path.replace(output_dir / f"{path.stem}.failed-{stamp}{path.suffix}")


def configure_backbone_state(*, model: nn.Module, frozen: bool) -> None:
    for name, parameter in model.named_parameters():
        parameter.requires_grad = not frozen or name.startswith("fc.")


def set_frozen_batchnorm_eval(*, model: nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.eval()


def prepare_training_mode(*, model: nn.Module, frozen: bool) -> None:
    model.train()
    if frozen:
        set_frozen_batchnorm_eval(model=model)


def collection_device(*, config: AdaptationConfig) -> torch.device:
    device = torch.device(config.run.device)
    if device.type != "cuda":
        raise ValueError("Adaptation training requires a CUDA device")
    if config.run.precision != "bf16":
        raise ValueError("Adaptation training requires bf16 precision")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("The selected GPU does not support bfloat16")
    return device


@contextmanager
def context(*, device: torch.device) -> Iterator[None]:
    with (
        torch.cuda.device(device),
        torch.device(device),
        torch.autocast(device_type="cuda", dtype=torch.bfloat16),
    ):
        yield


def rows_from_evaluation(
    *, evaluation: dict, targets: Tensor, indices: Tensor
) -> list[dict]:
    predictions = evaluation["predictions"].float().cpu()
    target_values = targets.float().cpu()
    rows = []
    for row, (mean, median, mode, target, index) in enumerate(
        zip(
            predictions[:, 0].tolist(),
            predictions[:, 1].tolist(),
            predictions[:, 2].tolist(),
            target_values.tolist(),
            indices.tolist(),
            strict=True,
        )
    ):
        rows.append({
            "row": row,
            "manifest_index": index,
            "target": target,
            "mean": mean,
            "median": median,
            "mode": mode,
        })
    return rows


@torch.inference_mode()
def evaluate_loader(
    *,
    model: nn.Module,
    loader: DataLoader,
    config: AdaptationConfig,
    device: torch.device,
) -> dict:
    outputs = []
    targets = []
    indices = []
    model.eval()
    started = time.monotonic()
    for images, ages, batch_indices in loader:
        with context(device=device):
            batch_images = images.to(device, non_blocking=True)
            logits = model(batch_images)
        outputs.append(logits)
        targets.append(ages.to(device, non_blocking=True))
        indices.append(batch_indices)
    output_tensor = torch.cat(outputs)
    target_tensor = torch.cat(targets)
    index_tensor = torch.cat(indices)
    with context(device=device):
        evaluation = evaluate_outputs(
            outputs=output_tensor,
            targets=target_tensor,
            scalar=config.run.head == "scalar",
        )
    return {
        "metrics": evaluation["metrics"],
        "rows": rows_from_evaluation(
            evaluation=evaluation,
            targets=target_tensor,
            indices=index_tensor,
        ),
        "seconds": time.monotonic() - started,
    }


@torch.inference_mode()
def evaluate_bank(
    *,
    model: nn.Module,
    features: Tensor,
    ages: Tensor,
    indices: Tensor,
    config: AdaptationConfig,
    device: torch.device,
) -> dict:
    outputs = []
    model.eval()
    started = time.monotonic()
    for start in range(0, len(features), config.run.eval_batch_size):
        stop = start + config.run.eval_batch_size
        with context(device=device):
            outputs.append(model.fc(features[start:stop]))
    output_tensor = torch.cat(outputs)
    targets = ages.to(device, non_blocking=True)
    with context(device=device):
        evaluation = evaluate_outputs(
            outputs=output_tensor,
            targets=targets,
            scalar=config.run.head == "scalar",
        )
    return {
        "metrics": evaluation["metrics"],
        "rows": rows_from_evaluation(
            evaluation=evaluation,
            targets=targets,
            indices=indices,
        ),
        "seconds": time.monotonic() - started,
    }


def bank_view(*, bank: dict, epoch: int) -> Tensor:
    assigned = bank["metadata"]["epochs"]["assigned"]
    mapping = bank["metadata"]["epochs"]["epoch_to_view"]
    try:
        offset = assigned.index(epoch)
    except ValueError as error:
        raise ValueError(f"Epoch {epoch} is absent from feature bank") from error
    return bank["train_features"][mapping[offset]]


@torch.no_grad()
def epoch_labels(
    *, clean_ages: list[int], config: AdaptationConfig, epoch: int
) -> Tensor:
    return noisy_labels(ages=clean_ages, config=config.run.noise, epoch=epoch)


def train_image_epoch(
    *,
    model: nn.Module,
    loader: DataLoader,
    labels: Tensor,
    config: AdaptationConfig,
    device: torch.device,
    optimizer: AdamW,
    scheduler: LambdaLR | None,
    epoch: int,
) -> dict:
    prepare_training_mode(model=model, frozen=config.backbone_frozen)
    loss_sum = 0.0
    norm_sum = 0.0
    count = 0
    started = time.monotonic()
    for images, _, indices in loader:
        optimizer.zero_grad(set_to_none=True)
        with context(device=device):
            batch_images = images.to(device, non_blocking=True)
            batch_labels = labels[indices].to(device, non_blocking=True)
            outputs = model(batch_images)
            loss = (
                scalar_loss(
                    predictions=outputs,
                    targets=batch_labels,
                    config=config.run.loss,
                )
                if config.run.head == "scalar"
                else policy_loss(
                    logits=outputs,
                    targets=batch_labels,
                    config=config.run.loss,
                )
            )
        loss.backward()
        norm = nn.utils.clip_grad_norm_(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            config.run.clip_grad_norm,
            error_if_nonfinite=True,
        )
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        loss_sum += loss.detach().double().item() * len(indices)
        norm_sum += norm.detach().double().item() * len(indices)
        count += len(indices)
    if count < 1:
        raise ValueError("Training epoch produced no examples")
    row = {
        "epoch": epoch,
        "train_loss": loss_sum / count,
        "grad_norm": norm_sum / count,
        "train_count": count,
        "learning_rates": [group["lr"] for group in optimizer.param_groups],
        "epoch_seconds": time.monotonic() - started,
    }
    return row


def train_bank_epoch(
    *,
    model: nn.Module,
    bank: dict,
    labels: Tensor,
    config: AdaptationConfig,
    device: torch.device,
    optimizer: AdamW,
    scheduler: LambdaLR | None,
    epoch: int,
) -> dict:
    prepare_training_mode(model=model, frozen=True)
    features = bank_view(bank=bank, epoch=epoch)
    global_indices = bank["train_indices"]
    order = epoch_order(
        count=len(global_indices),
        epoch=epoch,
        order_seed=config.order_seed,
    )
    loss_sum = 0.0
    norm_sum = 0.0
    count = 0
    started = time.monotonic()
    for start in range(0, len(order), config.run.batch_size):
        stop = start + config.run.batch_size
        if stop > len(order):
            break
        positions = order[start:stop]
        clean_positions = global_indices[positions]
        optimizer.zero_grad(set_to_none=True)
        with context(device=device):
            batch_features = features[positions.to(device, non_blocking=True)]
            batch_labels = labels[clean_positions].to(device, non_blocking=True)
            outputs = model.fc(batch_features)
            loss = (
                scalar_loss(
                    predictions=outputs,
                    targets=batch_labels,
                    config=config.run.loss,
                )
                if config.run.head == "scalar"
                else policy_loss(
                    logits=outputs,
                    targets=batch_labels,
                    config=config.run.loss,
                )
            )
        loss.backward()
        norm = nn.utils.clip_grad_norm_(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            config.run.clip_grad_norm,
            error_if_nonfinite=True,
        )
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        loss_sum += loss.detach().double().item() * len(positions)
        norm_sum += norm.detach().double().item() * len(positions)
        count += len(positions)
    if count < 1:
        raise ValueError("Training epoch produced no examples")
    row = {
        "epoch": epoch,
        "train_loss": loss_sum / count,
        "grad_norm": norm_sum / count,
        "train_count": count,
        "learning_rates": [group["lr"] for group in optimizer.param_groups],
        "epoch_seconds": time.monotonic() - started,
    }
    return row


def evaluate(
    *,
    model: nn.Module,
    split: str,
    config: AdaptationConfig,
    device: torch.device,
    bank: dict | None,
    loader: DataLoader | None,
) -> dict:
    if bank is not None:
        return evaluate_bank(
            model=model,
            features=bank[f"{split}_features"],
            ages=bank[f"{split}_ages"],
            indices=bank[f"{split}_indices"],
            config=config,
            device=device,
        )
    if loader is None:
        raise ValueError(f"Missing {split} loader")
    return evaluate_loader(
        model=model,
        loader=loader,
        config=config,
        device=device,
    )


def write_prediction_artifact(
    *,
    path: Path,
    checkpoint: str,
    split: str,
    epoch: int,
    result: dict,
) -> None:
    write_json(
        path=path,
        payload={
            "checkpoint": checkpoint,
            "split": split,
            "epoch": epoch,
            "metrics": result["metrics"],
            "rows": result["rows"],
        },
    )


def train_job(config: AdaptationConfig) -> dict:
    output_dir = config.run.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(path=output_dir / "worker-pid.json", payload={"pid": os.getpid()})
    device = collection_device(config=config)
    torch.set_num_threads(config.run.cpu_threads)
    torch.use_deterministic_algorithms(config.run.deterministic)
    torch.backends.cudnn.benchmark = not config.run.deterministic

    result_path = output_dir / "result.json"
    payload = config_payload(config=config)
    if result_path.is_file():
        completed = json.loads(result_path.read_text())
        if completed.get("config") != payload:
            raise ValueError(f"Completed result config differs: {result_path}")
        return completed

    config_path = output_dir / "config.json"
    if config_path.is_file():
        existing_config = json.loads(config_path.read_text())
        if existing_config != payload:
            raise ValueError(f"Existing run config differs: {config_path}")
    else:
        write_json(path=config_path, payload=payload)

    set_seeds(config.run.seed)
    manifest = read_manifest(path=config.run.manifest)
    bank = (
        load_feature_bank(path=config.feature_bank, device=device)
        if config.use_feature_bank
        else None
    )
    if bank is not None:
        validate_bank(config=config, bank=bank, manifest=manifest)

    with torch.cuda.device(device):
        model, initialization_source = build_initial_model(
            config=config,
            bank=bank,
        )
        run_provenance = provenance(
            config=config,
            manifest=manifest,
            bank=bank,
            initialization_source=initialization_source,
        )
        configure_backbone_state(
            model=model,
            frozen=config.backbone_frozen,
        )
        if bank is None:
            model = model.to(device)
        else:
            model.fc = model.fc.to(device)
        optimizer = build_optimizer(config=config, model=model)

        val_loader = (
            None
            if bank is not None
            else build_image_loader(
                run=config.run,
                pipeline=config.pipeline,
                split="val",
                epoch=0,
                view_seed=config.view_seed,
                order_seed=config.order_seed,
            )
        )
        test_loader = (
            build_image_loader(
                run=config.run,
                pipeline=config.pipeline,
                split="test",
                epoch=0,
                view_seed=config.view_seed,
                order_seed=config.order_seed,
            )
            if bank is None and config.run.evaluate_test
            else None
        )
        if bank is not None:
            examples_per_epoch = len(bank["train_indices"])
        else:
            scheduler_loader = build_image_loader(
                run=config.run,
                pipeline=config.pipeline,
                split="train",
                epoch=config.global_start_epoch + 1,
                view_seed=config.view_seed,
                order_seed=config.order_seed,
            )
            examples_per_epoch = len(scheduler_loader.dataset)
        steps_per_epoch = examples_per_epoch // config.run.batch_size
        if steps_per_epoch < 1:
            raise ValueError("Training split must contain one full batch")
        scheduler = build_scheduler(
            config=config,
            optimizer=optimizer,
            steps_per_epoch=steps_per_epoch,
        )
        torch.cuda.reset_peak_memory_stats(device)

        checkpoints = regular_checkpoints(output_dir=output_dir)
        if not checkpoints and config.control.extend_epoch:
            prepare_extension_checkpoint(
                config=config,
                run_provenance=run_provenance,
                initialization_source=initialization_source,
            )
            checkpoints = regular_checkpoints(output_dir=output_dir)
        if checkpoints:
            latest_epoch, latest_path = checkpoints[-1]
            saved = load_full_checkpoint(
                path=latest_path,
                config=config,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                initialization_source=initialization_source,
                bank_metadata=bank["metadata"] if bank is not None else None,
                run_provenance=run_provenance,
            )
            history = saved["history"]
            initialization_record = saved["initialization_record"]
            best_epoch = saved["best_epoch"]
            best_validation = saved["best_validation"]
            best_model = saved["best_model"]
            start_after = latest_epoch
            model.load_state_dict(best_model, strict=True)
            save_native_model(
                path=output_dir / "best.pt",
                model=model,
                config=config,
                epoch=best_epoch,
                validation=best_validation,
                manifest_fingerprint=manifest["fingerprint"],
                run_provenance=run_provenance,
            )
            model.load_state_dict(saved["model"], strict=True)
            milestone = output_dir / "model-milestones" / f"epoch_{latest_epoch:03d}.pt"
            if (
                latest_epoch in config.control.model_milestones
                and not milestone.exists()
            ):
                save_native_model(
                    path=milestone,
                    model=model,
                    config=config,
                    epoch=latest_epoch,
                    validation=saved["validation"],
                    manifest_fingerprint=manifest["fingerprint"],
                    run_provenance=run_provenance,
                )
        else:
            archive_stale_artifacts(output_dir=output_dir)
            history = []
            initialization = evaluate(
                model=model,
                split="val",
                config=config,
                device=device,
                bank=bank,
                loader=val_loader,
            )
            initialization["validation"] = initialization["metrics"]
            initialization["epoch"] = config.global_start_epoch
            initialization["validation_seconds"] = initialization["seconds"]
            initialization["model_sha256"] = model_sha256(
                state=model_state_cpu(model=model)
            )
            initialization["initialization_source"] = initialization_source
            initialization_record = initialization
            best_epoch = config.global_start_epoch
            best_validation = initialization["validation"]
            best_model = model_state_cpu(model=model)
            save_native_model(
                path=output_dir / "best.pt",
                model=model,
                config=config,
                epoch=best_epoch,
                validation=best_validation,
                manifest_fingerprint=manifest["fingerprint"],
                run_provenance=run_provenance,
            )
            write_json(
                path=output_dir / "best-validation-predictions.json",
                payload={
                    "checkpoint": "best",
                    "split": "val",
                    "epoch": best_epoch,
                    "metrics": best_validation,
                    "rows": initialization["rows"],
                },
            )
            start_after = config.global_start_epoch

        final_global_epoch = config.global_start_epoch + config.run.epochs
        clean_ages = [record["age"] for record in manifest["records"]]
        for epoch in range(start_after + 1, final_global_epoch + 1):
            labels = epoch_labels(
                clean_ages=clean_ages,
                config=config,
                epoch=epoch,
            )
            if bank is None:
                train_loader = build_image_loader(
                    run=config.run,
                    pipeline=config.pipeline,
                    split="train",
                    epoch=epoch,
                    view_seed=config.view_seed,
                    order_seed=config.order_seed,
                )
                row = train_image_epoch(
                    model=model,
                    loader=train_loader,
                    labels=labels,
                    config=config,
                    device=device,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=epoch,
                )
            else:
                row = train_bank_epoch(
                    model=model,
                    bank=bank,
                    labels=labels,
                    config=config,
                    device=device,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=epoch,
                )
            validation = evaluate(
                model=model,
                split="val",
                config=config,
                device=device,
                bank=bank,
                loader=val_loader,
            )
            row["validation"] = validation.pop("metrics")
            row["validation_seconds"] = validation["seconds"]
            row["predictions_sha256"] = json_sha256(validation["rows"])
            history.append(row)
            if row["validation"]["mae_mean"] < best_validation["mae_mean"]:
                best_epoch = epoch
                best_validation = row["validation"]
                best_model = model_state_cpu(model=model)
                save_native_model(
                    path=output_dir / "best.pt",
                    model=model,
                    config=config,
                    epoch=best_epoch,
                    validation=best_validation,
                    manifest_fingerprint=manifest["fingerprint"],
                    run_provenance=run_provenance,
                )
            write_json(path=output_dir / "history.json", payload=history)
            if epoch % config.checkpoint_every == 0:
                save_full_checkpoint(
                    path=output_dir / "checkpoints" / f"epoch_{epoch:03d}.pt",
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    config=config,
                    epoch=epoch,
                    validation=row["validation"],
                    history=history,
                    initialization_record=initialization_record,
                    best_epoch=best_epoch,
                    best_validation=best_validation,
                    best_model=best_model,
                    initialization_source=initialization_source,
                    bank_metadata=bank["metadata"] if bank is not None else None,
                    run_provenance=run_provenance,
                )
                prune_regular_checkpoints(config=config)
            if epoch in config.control.model_milestones:
                save_native_model(
                    path=output_dir / "model-milestones" / f"epoch_{epoch:03d}.pt",
                    model=model,
                    config=config,
                    epoch=epoch,
                    validation=row["validation"],
                    manifest_fingerprint=manifest["fingerprint"],
                    run_provenance=run_provenance,
                )

        final_state = model_state_cpu(model=model)
        final_validation = evaluate(
            model=model,
            split="val",
            config=config,
            device=device,
            bank=bank,
            loader=val_loader,
        )
        save_native_model(
            path=output_dir / "final.pt",
            model=model,
            config=config,
            epoch=final_global_epoch,
            validation=final_validation["metrics"],
            manifest_fingerprint=manifest["fingerprint"],
            run_provenance=run_provenance,
        )
        best_model_checkpoint = load_native_checkpoint(path=output_dir / "best.pt")
        checkpoint_model_hash = best_model_checkpoint["model_sha256"]
        if checkpoint_model_hash != model_sha256(state=best_model):
            raise ValueError("best.pt does not match embedded best model")
        write_prediction_artifact(
            path=output_dir / "final-validation-predictions.json",
            checkpoint="final",
            split="val",
            epoch=final_global_epoch,
            result=final_validation,
        )
        final_test = None
        if config.run.evaluate_test:
            final_test = evaluate(
                model=model,
                split="test",
                config=config,
                device=device,
                bank=bank,
                loader=test_loader,
            )
            write_prediction_artifact(
                path=output_dir / "final-test-predictions.json",
                checkpoint="final",
                split="test",
                epoch=final_global_epoch,
                result=final_test,
            )

        model.load_state_dict(best_model, strict=True)
        best_validation_full = evaluate(
            model=model,
            split="val",
            config=config,
            device=device,
            bank=bank,
            loader=val_loader,
        )
        write_prediction_artifact(
            path=output_dir / "best-validation-predictions.json",
            checkpoint="best",
            split="val",
            epoch=best_epoch,
            result=best_validation_full,
        )
        best_test = None
        if config.run.evaluate_test:
            best_test = evaluate(
                model=model,
                split="test",
                config=config,
                device=device,
                bank=bank,
                loader=test_loader,
            )
            write_prediction_artifact(
                path=output_dir / "best-test-predictions.json",
                checkpoint="best",
                split="test",
                epoch=best_epoch,
                result=best_test,
            )

        result = {
            "config": payload,
            "status": "complete",
            "provenance": run_provenance,
            "initialization_source": initialization_source,
            "initialization": initialization_record,
            "best_epoch": best_epoch,
            "best_validation": best_validation_full["metrics"],
            "final_global_epoch": final_global_epoch,
            "final_validation": final_validation["metrics"],
            "best_test": best_test["metrics"] if best_test is not None else None,
            "final_test": final_test["metrics"] if final_test is not None else None,
            "history": history,
            "peak_allocated_mib": torch.cuda.max_memory_allocated(device) / 2**20,
            "peak_reserved_mib": torch.cuda.max_memory_reserved(device) / 2**20,
            "final_model_sha256": model_sha256(state=final_state),
            "best_model_sha256": model_sha256(state=best_model),
            "predictions": {
                "best": "best-validation-predictions.json",
                "final": "final-validation-predictions.json",
                "best_test": (
                    "best-test-predictions.json" if best_test is not None else None
                ),
                "final_test": (
                    "final-test-predictions.json" if final_test is not None else None
                ),
            },
        }
        write_json(path=result_path, payload=result)

    cleanup_cuda(device=device)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = AdaptationConfig.model_validate_json(args.config.read_text())
    train_job(config=config)


if __name__ == "__main__":
    main()
