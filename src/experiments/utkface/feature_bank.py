from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import platform
import time
from contextlib import nullcontext
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import torch
import torchvision
from torch import Tensor, nn
from torch.utils.data import DataLoader

from src.experiments.utils import set_seeds
from src.experiments.utkface.adaptation_config import FeatureBankConfig
from src.experiments.utkface.adaptation_data import build_image_loader
from src.experiments.utkface.data import (
    FaceDataset,
    read_manifest,
    training_indices,
)
from src.experiments.utkface.model import build_model
from src.experiments.utkface.run import write_json

SCHEMA = "utkface-feature-bank"
VERSION = 1
FEATURE_DIMENSION = 512
FEATURE_DTYPE = torch.bfloat16


def _json_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _file_sha256(*, path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _tensor_sha256(tensor: Tensor) -> str:
    contiguous = tensor.detach().cpu().contiguous()
    if contiguous.dtype == torch.bfloat16:
        return hashlib.sha256(
            contiguous.view(torch.uint8).numpy().tobytes()
        ).hexdigest()
    return hashlib.sha256(contiguous.numpy().tobytes()).hexdigest()


def _encoder_state_sha256(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        if name == "fc" or name.startswith("fc."):
            continue
        contiguous = value.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(tuple(contiguous.shape)).encode())
        digest.update(str(contiguous.dtype).encode())
        digest.update(contiguous.numpy().tobytes())
    return digest.hexdigest()


def _source_file_hashes() -> dict[str, str]:
    root = Path(__file__).resolve().parents[3]
    paths = (
        Path(__file__).resolve(),
        root / "src/experiments/utkface/adaptation_data.py",
        root / "src/experiments/utkface/config.py",
        root / "src/experiments/utkface/data.py",
        root / "src/experiments/utkface/model.py",
    )
    return {str(path): _file_sha256(path=path) for path in paths}


def _environment() -> dict:
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "cuda": torch.version.cuda,
    }


def _transform_provenance(*, pipeline: Literal["original", "fixed"]) -> dict:
    evaluation = {
        "resize": 256,
        "center_crop": 224,
        "tensor": True,
        "normalize_mean": [0.485, 0.456, 0.406],
        "normalize_std": [0.229, 0.224, 0.225],
    }
    training = (
        {
            "random_resized_crop": 224,
            "random_resized_crop_scale": [0.8, 1.0],
            "random_horizontal_flip": 0.5,
            "color_jitter": {
                "brightness": 0.2,
                "contrast": 0.2,
                "saturation": 0.2,
            },
            "seed": "derived(view_seed, manifest_index, global_epoch)",
        }
        if pipeline == "original"
        else evaluation
    )
    descriptor = {
        "pipeline": pipeline,
        "training": training,
        "evaluation": evaluation,
    }
    return {**descriptor, "sha256": _json_sha256(descriptor)}


def _manifest_provenance(*, payload: dict, path: Path) -> dict:
    return {
        "path": str(path.resolve()),
        "sha256": _file_sha256(path=path),
        "fingerprint": payload["fingerprint"],
    }


def _assigned_epochs(config: FeatureBankConfig) -> list[int]:
    start = config.global_start_epoch + 1
    return list(range(start, start + config.run.epochs))


def _source_rows(*, payload: dict, config: FeatureBankConfig) -> dict[str, dict]:
    rows = {}
    for split in ("train", "val", "test"):
        if split == "train":
            indices = training_indices(
                payload=payload,
                limit=config.run.train_limit,
                subset_seed=config.run.subset_seed,
            )
        else:
            indices = payload["splits"][split]
            limit = config.run.eval_limit
            if limit:
                indices = indices[:limit]
        ages = [int(payload["records"][index]["age"]) for index in indices]
        rows[split] = {
            "count": len(indices),
            "indices": indices,
            "ages": ages,
            "indices_sha256": _tensor_sha256(torch.tensor(indices, dtype=torch.int64)),
            "ages_sha256": _tensor_sha256(torch.tensor(ages, dtype=torch.int64)),
        }
    return rows


def _make_source(
    *, config: FeatureBankConfig, payload: dict, split: str
) -> FaceDataset:
    return FaceDataset(
        manifest_path=config.run.manifest,
        payload=payload,
        split=split,
        limit=config.run.train_limit if split == "train" else config.run.eval_limit,
        rank=0,
        world_size=1,
        subset_seed=config.run.subset_seed,
    )


def _extract_features(
    *,
    model: nn.Module,
    loader: DataLoader,
    config: FeatureBankConfig,
) -> tuple[Tensor, Tensor, Tensor]:
    device = torch.device(config.run.device)
    cuda_context = torch.cuda.device(device) if device.type == "cuda" else nullcontext()
    features: list[Tensor] = []
    indices: list[Tensor] = []
    ages: list[Tensor] = []
    autocast_enabled = config.run.precision == "bf16"
    with (
        cuda_context,
        torch.inference_mode(),
    ):
        model.to(device)
        model.eval()
        model.requires_grad_(False)
        for images, label_batch, index_batch in loader:
            with (
                torch.device(device),
                torch.autocast(
                    device_type=device.type,
                    dtype=torch.bfloat16,
                    enabled=autocast_enabled,
                ),
            ):
                images = images.to(device, non_blocking=device.type == "cuda")
                values = model.conv1(images)
                values = model.bn1(values)
                values = model.relu(values)
                values = model.maxpool(values)
                values = model.layer1(values)
                values = model.layer2(values)
                values = model.layer3(values)
                values = model.layer4(values)
                values = model.avgpool(values)
                values = torch.flatten(values, 1)
            if values.shape != (len(images), FEATURE_DIMENSION):
                raise ValueError(f"Unexpected feature shape: {tuple(values.shape)}")
            if not torch.isfinite(values).all():
                raise ValueError("Feature extractor produced non-finite values")
            features.append(
                values.to(dtype=FEATURE_DTYPE, device="cpu", non_blocking=False)
            )
            indices.append(index_batch.to(dtype=torch.int64, device="cpu"))
            ages.append(label_batch.to(dtype=torch.int64, device="cpu"))
    feature_tensor = (
        torch.cat(features, dim=0)
        if features
        else torch.empty((0, FEATURE_DIMENSION), dtype=FEATURE_DTYPE)
    )
    index_tensor = (
        torch.cat(indices, dim=0) if indices else torch.empty((0,), dtype=torch.int64)
    )
    age_tensor = (
        torch.cat(ages, dim=0) if ages else torch.empty((0,), dtype=torch.int64)
    )
    return feature_tensor, index_tensor, age_tensor


def _chunk_payload(
    *,
    source_fingerprint: str,
    source_epoch: int,
    features: Tensor,
    indices: Tensor,
    ages: Tensor,
    extraction_indices_sha256: str,
) -> dict:
    return {
        "schema": SCHEMA,
        "version": VERSION,
        "source_fingerprint": source_fingerprint,
        "source_epoch": source_epoch,
        "feature_shape": list(features.shape),
        "feature_dtype": "torch.bfloat16",
        "feature_sha256": _tensor_sha256(features),
        "extraction_indices_sha256": extraction_indices_sha256,
        "features": features,
        "indices": indices,
        "ages": ages,
    }


def _read_valid_chunk(
    *,
    path: Path,
    source_fingerprint: str,
    source_epoch: int,
    expected_indices: Tensor,
    expected_ages: Tensor,
) -> dict | None:
    if not path.is_file():
        return None
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except (
        EOFError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
        pickle.UnpicklingError,
    ):
        return None
    if not isinstance(payload, dict):
        return None
    if (
        "source_fingerprint" in payload
        and payload["source_fingerprint"] != source_fingerprint
    ):
        raise ValueError("Existing feature-bank chunk has a different source")
    if "source_epoch" in payload and payload["source_epoch"] != source_epoch:
        raise ValueError("Existing feature-bank chunk has a different source epoch")
    required = {
        "schema",
        "version",
        "source_fingerprint",
        "source_epoch",
        "feature_shape",
        "feature_dtype",
        "feature_sha256",
        "extraction_indices_sha256",
        "features",
        "indices",
        "ages",
    }
    if set(payload) != required:
        return None
    if (
        payload["schema"] != SCHEMA
        or payload["version"] != VERSION
        or payload["source_fingerprint"] != source_fingerprint
        or payload["source_epoch"] != source_epoch
        or payload["feature_dtype"] != "torch.bfloat16"
    ):
        return None
    features = payload["features"]
    indices = payload["indices"]
    ages = payload["ages"]
    if not all(isinstance(value, Tensor) for value in (features, indices, ages)):
        return None
    expected_shape = [len(expected_indices), FEATURE_DIMENSION]
    if (
        list(features.shape) != expected_shape
        or payload["feature_shape"] != expected_shape
        or features.dtype != FEATURE_DTYPE
        or indices.dtype != torch.int64
        or ages.dtype != torch.int64
        or not torch.equal(indices, expected_indices)
        or not torch.equal(ages, expected_ages)
        or payload["feature_sha256"] != _tensor_sha256(features)
        or not isinstance(payload["extraction_indices_sha256"], str)
        or len(payload["extraction_indices_sha256"]) != 64
    ):
        return None
    return payload


def _atomic_save_torch(*, path: Path, payload: dict) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as handle:
            torch.save(payload, handle)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _extract_or_reuse(
    *,
    model: nn.Module,
    config: FeatureBankConfig,
    source_fingerprint: str,
    source: FaceDataset,
    split: Literal["train", "val", "test"],
    source_epoch: int,
    path: Path,
) -> tuple[dict, dict]:
    indices = torch.tensor(source.indices, dtype=torch.int64)
    ages = torch.tensor([record.age for record in source.records], dtype=torch.int64)
    cached = _read_valid_chunk(
        path=path,
        source_fingerprint=source_fingerprint,
        source_epoch=source_epoch,
        expected_indices=indices,
        expected_ages=ages,
    )
    if cached is not None:
        return cached, {
            "seconds": 0.0,
            "reused": True,
            "file": path.relative_to(config.output_dir).as_posix(),
        }
    epoch = (
        source_epoch
        if split == "train" and config.pipeline == "original"
        else config.global_start_epoch + 1
        if split == "train"
        else 0
    )
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if rank != 0 or world_size != 1:
        raise ValueError("Feature-bank extraction must run without data parallelism")
    loader = build_image_loader(
        run=config.run,
        pipeline=config.pipeline,
        split=split,
        epoch=epoch,
        view_seed=config.view_seed,
        order_seed=config.order_seed,
    )
    if split == "train" and len(source) % config.run.batch_size:
        raise ValueError(
            "Feature banks require a training subset divisible by batch_size"
        )
    if len(loader.dataset) != len(source):
        raise ValueError("Feature-bank loader omitted canonical source rows")
    started = time.monotonic()
    features, extracted_indices, extracted_ages = _extract_features(
        model=model, loader=loader, config=config
    )
    if len(extracted_indices) != len(indices):
        raise ValueError("Feature-bank loader omitted canonical source rows")
    if len(extracted_indices.unique()) != len(extracted_indices):
        raise ValueError("Feature-bank loader returned duplicate rows")
    positions = {int(value): index for index, value in enumerate(extracted_indices)}
    try:
        canonical_positions = torch.tensor(
            [positions[int(index)] for index in indices], dtype=torch.int64
        )
    except KeyError as error:
        raise ValueError(
            "Feature-bank loader omitted a canonical source row"
        ) from error
    features = features[canonical_positions]
    extracted_ages = extracted_ages[canonical_positions]
    if not torch.equal(extracted_ages, ages):
        raise ValueError(f"{split} ages changed during feature extraction")
    payload = _chunk_payload(
        source_fingerprint=source_fingerprint,
        source_epoch=source_epoch,
        features=features,
        indices=indices,
        ages=ages,
        extraction_indices_sha256=_tensor_sha256(extracted_indices),
    )
    _atomic_save_torch(path=path, payload=payload)
    return payload, {
        "seconds": time.monotonic() - started,
        "reused": False,
        "file": path.relative_to(config.output_dir).as_posix(),
    }


def _checkpoint_metadata(saved: dict) -> dict:
    return {
        key: saved[key]
        for key in ("epoch", "data_fingerprint", "head", "pretrained")
        if key in saved
    }


def _initialize_model(*, config: FeatureBankConfig) -> tuple[nn.Module, dict]:
    if config.initialization == "imagenet":
        model = build_model(pretrained=True, head=config.run.head)
        checkpoint = {
            "path": None,
            "sha256": None,
            "saved_metadata": None,
        }
        imagenet_identity = "torchvision:ResNet18_Weights.IMAGENET1K_V1"
    else:
        checkpoint_path = config.initial_checkpoint.resolve()
        saved = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if not isinstance(saved, dict) or not isinstance(saved.get("model"), dict):
            raise ValueError(f"Invalid model checkpoint: {checkpoint_path}")
        model = build_model(pretrained=False, head=config.run.head)
        model.load_state_dict(saved["model"], strict=True)
        checkpoint = {
            "path": str(checkpoint_path),
            "sha256": _file_sha256(path=checkpoint_path),
            "saved_metadata": _checkpoint_metadata(saved),
        }
        imagenet_identity = None
    model.eval()
    model.requires_grad_(False)
    return model, {
        "initialization": config.initialization,
        "checkpoint": checkpoint,
        "imagenet_identity": imagenet_identity,
        "encoder_state_sha256": _encoder_state_sha256(model),
    }


def _source_provenance(
    *,
    config: FeatureBankConfig,
    manifest: dict,
    transform: dict,
    initialization: dict,
) -> dict:
    source = {
        "config": config.model_dump(mode="json"),
        "config_sha256": _json_sha256(config.model_dump(mode="json")),
        "manifest": manifest,
        "transform": transform,
        "initialization": initialization,
        "feature_extractor": {
            "architecture": "resnet18",
            "stages": [
                "conv1",
                "bn1",
                "relu",
                "maxpool",
                "layer1",
                "layer2",
                "layer3",
                "layer4",
                "avgpool",
                "flatten",
            ],
            "dimension": FEATURE_DIMENSION,
            "output_dtype": "bfloat16",
        },
        "code_sha256": _source_file_hashes(),
        "environment": _environment(),
    }
    return {**source, "fingerprint": _json_sha256(source)}


def _chunk_result(*, view: int, source_epoch: int, payload: dict, timing: dict) -> dict:
    return {
        "view": view,
        "source_epoch": source_epoch,
        "feature_shape": payload["feature_shape"],
        "feature_sha256": payload["feature_sha256"],
        "indices_sha256": _tensor_sha256(payload["indices"]),
        "ages_sha256": _tensor_sha256(payload["ages"]),
        "extraction_indices_sha256": payload["extraction_indices_sha256"],
        "seconds": timing["seconds"],
        "reused": timing["reused"],
    }


def _safe_artifact_path(*, root: Path, relative: object) -> Path:
    if not isinstance(relative, str):
        raise TypeError("Artifact path must be a string")
    artifact = Path(relative)
    if artifact.is_absolute() or ".." in artifact.parts:
        raise ValueError(f"Unsafe artifact path: {relative}")
    resolved_root = root.resolve()
    resolved = (root / artifact).resolve()
    if not resolved.is_relative_to(resolved_root):
        raise ValueError(f"Artifact path escapes the bank: {relative}")
    return root / artifact


def _expected_rows(
    *, payload: dict, config: FeatureBankConfig, split: str
) -> tuple[Tensor, Tensor]:
    if split == "train":
        indices = training_indices(
            payload=payload,
            limit=config.run.train_limit,
            subset_seed=config.run.subset_seed,
        )
    else:
        indices = payload["splits"][split]
        if config.run.eval_limit:
            indices = indices[: config.run.eval_limit]
    ages = [payload["records"][index]["age"] for index in indices]
    return (
        torch.tensor(indices, dtype=torch.int64),
        torch.tensor(ages, dtype=torch.int64),
    )


def _validate_rows(
    *,
    rows: dict,
    payload: dict,
    config: FeatureBankConfig,
    split: str,
    indices: Tensor,
    ages: Tensor,
) -> None:
    expected_indices, expected_ages = _expected_rows(
        payload=payload, config=config, split=split
    )
    if (
        indices.dtype != torch.int64
        or ages.dtype != torch.int64
        or indices.ndim != 1
        or ages.ndim != 1
        or len(indices) != len(ages)
        or not torch.equal(indices.cpu(), expected_indices)
        or not torch.equal(ages.cpu(), expected_ages)
    ):
        raise ValueError(f"{split} indices or ages do not match the manifest")
    stored = rows[split]
    if (
        stored["count"] != len(indices)
        or stored["indices"] != indices.cpu().tolist()
        or stored["ages"] != ages.cpu().tolist()
        or stored["indices_sha256"] != _tensor_sha256(indices)
        or stored["ages_sha256"] != _tensor_sha256(ages)
    ):
        raise ValueError(f"{split} result row provenance mismatch")


def _validate_source(*, result: dict) -> None:
    source = result["source"]
    claimed = source["fingerprint"]
    without_fingerprint = {
        key: value for key, value in source.items() if key != "fingerprint"
    }
    if claimed != _json_sha256(without_fingerprint):
        raise ValueError("Feature-bank source fingerprint mismatch")
    for path_value, expected in source["code_sha256"].items():
        path = Path(path_value)
        if not path.is_file() or _file_sha256(path=path) != expected:
            raise ValueError(f"Feature-bank source changed: {path}")
    checkpoint = source["initialization"]["checkpoint"]
    if checkpoint["path"] is not None:
        path = Path(checkpoint["path"])
        if not path.is_file() or _file_sha256(path=path) != checkpoint["sha256"]:
            raise ValueError(f"Feature-bank checkpoint changed: {path}")


def _validate_result_header(*, result: dict, path: Path) -> FeatureBankConfig:
    required = {
        "schema",
        "version",
        "complete",
        "created_at",
        "config",
        "source",
        "data",
        "epochs",
        "features",
        "timings",
        "chunks",
        "bank_file",
        "bank_sha256",
        "bank_bytes",
    }
    if set(result) != required:
        raise ValueError("Feature-bank result schema mismatch")
    if (
        result["schema"] != SCHEMA
        or result["version"] != VERSION
        or result["complete"] is not True
    ):
        raise ValueError("Feature-bank result is not a completed compatible bank")
    config = FeatureBankConfig.model_validate(result["config"])
    if config.output_dir.resolve() != path.resolve():
        raise ValueError("Feature-bank config output directory mismatch")
    data = result["data"]
    manifest_path = Path(data["manifest"]["path"])
    payload = read_manifest(path=manifest_path)
    if (
        payload["fingerprint"] != data["manifest"]["fingerprint"]
        or _file_sha256(path=manifest_path) != data["manifest"]["sha256"]
    ):
        raise ValueError("Feature-bank manifest changed")
    expected_transform = _transform_provenance(pipeline=config.pipeline)
    if result["source"]["transform"] != expected_transform:
        raise ValueError("Feature-bank transform provenance mismatch")
    _validate_source(result=result)
    return config


def _validate_chunks(
    *,
    result: dict,
    root: Path,
    config: FeatureBankConfig,
    bank: dict,
) -> None:
    assigned = _assigned_epochs(config)
    if config.pipeline == "fixed":
        expected = [
            {
                "path": "views/fixed_view_0000.pt",
                "source_epoch": 0,
                "features": bank["train_features"][0],
                "indices": bank["train_indices"],
                "ages": bank["train_ages"],
            }
        ]
    else:
        expected = [
            {
                "path": f"views/epoch_{epoch:04d}.pt",
                "source_epoch": epoch,
                "features": bank["train_features"][view],
                "indices": bank["train_indices"],
                "ages": bank["train_ages"],
            }
            for view, epoch in enumerate(assigned)
        ]
    expected.extend(
        {
            "path": f"views/{split}.pt",
            "source_epoch": 0,
            "features": bank[f"{split}_features"],
            "indices": bank[f"{split}_indices"],
            "ages": bank[f"{split}_ages"],
        }
        for split in ("val", "test")
    )

    chunks = result["chunks"]
    if not isinstance(chunks, list) or not all(
        isinstance(chunk, dict)
        and set(chunk) == {"path", "sha256"}
        and isinstance(chunk["sha256"], str)
        and len(chunk["sha256"]) == 64
        for chunk in chunks
    ):
        raise ValueError("Feature-bank chunk manifest is malformed")
    by_path = {chunk["path"]: chunk["sha256"] for chunk in chunks}
    if len(by_path) != len(chunks) or set(by_path) != {
        chunk["path"] for chunk in expected
    }:
        raise ValueError("Feature-bank chunk manifest differs from its config")

    required = {
        "schema",
        "version",
        "source_fingerprint",
        "source_epoch",
        "feature_shape",
        "feature_dtype",
        "feature_sha256",
        "extraction_indices_sha256",
        "features",
        "indices",
        "ages",
    }
    for chunk in expected:
        relative = chunk["path"]
        path = _safe_artifact_path(root=root, relative=relative)
        if not path.is_file() or _file_sha256(path=path) != by_path[relative]:
            raise ValueError(f"Feature-bank chunk checksum mismatch: {relative}")
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if (
            not isinstance(payload, dict)
            or set(payload) != required
            or payload["schema"] != SCHEMA
            or payload["version"] != VERSION
            or payload["source_fingerprint"] != result["source"]["fingerprint"]
            or payload["source_epoch"] != chunk["source_epoch"]
            or payload["feature_dtype"] != "torch.bfloat16"
            or not isinstance(payload["extraction_indices_sha256"], str)
            or len(payload["extraction_indices_sha256"]) != 64
        ):
            raise ValueError(f"Feature-bank chunk schema mismatch: {relative}")
        features = payload["features"]
        indices = payload["indices"]
        ages = payload["ages"]
        expected_indices = chunk["indices"].cpu()
        expected_ages = chunk["ages"].cpu()
        expected_features = chunk["features"].cpu()
        if (
            not all(isinstance(value, Tensor) for value in (features, indices, ages))
            or features.dtype != FEATURE_DTYPE
            or indices.dtype != torch.int64
            or ages.dtype != torch.int64
            or list(features.shape) != list(expected_features.shape)
            or payload["feature_shape"] != list(expected_features.shape)
            or not torch.equal(features, expected_features)
            or not torch.equal(indices, expected_indices)
            or not torch.equal(ages, expected_ages)
            or payload["feature_sha256"] != _tensor_sha256(features)
        ):
            raise ValueError(f"Feature-bank chunk tensor mismatch: {relative}")

    views = result["features"]["views"]
    if not isinstance(views, list) or len(views) != (
        1 if config.pipeline == "fixed" else len(assigned)
    ):
        raise ValueError("Feature-bank view metadata count mismatch")
    for view, record in enumerate(views):
        source_epoch = 0 if config.pipeline == "fixed" else assigned[view]
        if (
            not isinstance(record, dict)
            or set(record)
            != {
                "view",
                "source_epoch",
                "feature_shape",
                "feature_sha256",
                "indices_sha256",
                "ages_sha256",
                "extraction_indices_sha256",
                "seconds",
                "reused",
            }
            or record.get("view") != view
            or record.get("source_epoch") != source_epoch
            or record.get("feature_shape")
            != [len(bank["train_indices"]), FEATURE_DIMENSION]
            or record.get("feature_sha256")
            != _tensor_sha256(bank["train_features"][view])
            or record.get("indices_sha256") != _tensor_sha256(bank["train_indices"])
            or record.get("ages_sha256") != _tensor_sha256(bank["train_ages"])
            or type(record.get("seconds")) not in {int, float}
            or record["seconds"] < 0
            or type(record.get("reused")) is not bool
        ):
            raise ValueError("Feature-bank view metadata mismatch")
    timings = result["timings"]
    if (
        not isinstance(timings, dict)
        or set(timings) != {"total_seconds", "train_views", "val", "test"}
        or timings["train_views"] != views
        or type(timings["total_seconds"]) not in {int, float}
        or timings["total_seconds"] < 0
    ):
        raise ValueError("Feature-bank view timing metadata mismatch")
    for split in ("val", "test"):
        timing = timings[split]
        if (
            not isinstance(timing, dict)
            or set(timing) != {"seconds", "reused", "file"}
            or timing["file"] != f"views/{split}.pt"
            or type(timing["seconds"]) not in {int, float}
            or timing["seconds"] < 0
            or type(timing["reused"]) is not bool
        ):
            raise ValueError(f"Feature-bank {split} timing metadata mismatch")


def load_feature_bank(*, path: Path, device: torch.device) -> dict:
    root = Path(path)
    result_path = root / "result.json"
    if not result_path.is_file():
        raise FileNotFoundError(f"Feature bank is incomplete: {root}")
    result = json.loads(result_path.read_text())
    if not isinstance(result, dict):
        raise TypeError("Feature-bank result must be an object")
    config = _validate_result_header(result=result, path=root)
    bank_path = _safe_artifact_path(root=root, relative=result["bank_file"])
    if bank_path.name != "bank.pt" or not bank_path.is_file():
        raise FileNotFoundError(f"Missing feature-bank artifact: {bank_path}")
    if (
        _file_sha256(path=bank_path) != result["bank_sha256"]
        or bank_path.stat().st_size != result["bank_bytes"]
    ):
        raise ValueError("Feature-bank artifact checksum mismatch")

    target_device = torch.device(device)
    bank = torch.load(bank_path, map_location="cpu", weights_only=True)
    required = {
        "schema",
        "version",
        "source_fingerprint",
        "train_features",
        "train_epochs",
        "train_epoch_to_view",
        "train_indices",
        "train_ages",
        "val_features",
        "val_indices",
        "val_ages",
        "test_features",
        "test_indices",
        "test_ages",
    }
    if not isinstance(bank, dict) or set(bank) != required:
        raise ValueError("Feature-bank tensor schema mismatch")
    if (
        bank["schema"] != SCHEMA
        or bank["version"] != VERSION
        or bank["source_fingerprint"] != result["source"]["fingerprint"]
    ):
        raise ValueError("Feature-bank tensor provenance mismatch")

    tensors = {
        key: bank[key]
        for key in required
        if key not in {"schema", "version", "source_fingerprint"}
    }
    if not all(isinstance(value, Tensor) for value in tensors.values()):
        raise ValueError("Feature-bank tensor field has a non-tensor value")
    for key in ("train_features", "val_features", "test_features"):
        if tensors[key].dtype != FEATURE_DTYPE:
            raise ValueError(f"{key} must use bfloat16")
        if not torch.isfinite(tensors[key]).all():
            raise ValueError(f"{key} contains non-finite values")
    for key in required - {
        "schema",
        "version",
        "source_fingerprint",
        "train_features",
        "val_features",
        "test_features",
    }:
        if tensors[key].dtype != torch.int64:
            raise ValueError(f"{key} must use int64")

    manifest = read_manifest(path=Path(result["data"]["manifest"]["path"]))
    _validate_rows(
        rows=result["data"]["rows"],
        payload=manifest,
        config=config,
        split="train",
        indices=tensors["train_indices"],
        ages=tensors["train_ages"],
    )
    _validate_rows(
        rows=result["data"]["rows"],
        payload=manifest,
        config=config,
        split="val",
        indices=tensors["val_indices"],
        ages=tensors["val_ages"],
    )
    _validate_rows(
        rows=result["data"]["rows"],
        payload=manifest,
        config=config,
        split="test",
        indices=tensors["test_indices"],
        ages=tensors["test_ages"],
    )

    assigned = _assigned_epochs(config)
    expected_epochs = [0] if config.pipeline == "fixed" else assigned
    expected_mapping = (
        [0] * len(assigned)
        if config.pipeline == "fixed"
        else list(range(len(assigned)))
    )
    if result["epochs"] != {
        "assigned": assigned,
        "view_epochs": expected_epochs,
        "epoch_to_view": expected_mapping,
    }:
        raise ValueError("Feature-bank epoch metadata mismatch")
    if (
        tensors["train_epochs"].cpu().tolist() != expected_epochs
        or tensors["train_epoch_to_view"].cpu().tolist() != expected_mapping
    ):
        raise ValueError("Feature-bank tensor epoch mapping mismatch")

    view_count = 1 if config.pipeline == "fixed" else len(assigned)
    train_shape = [
        view_count,
        len(tensors["train_indices"]),
        FEATURE_DIMENSION,
    ]
    if list(tensors["train_features"].shape) != train_shape:
        raise ValueError("Feature-bank train feature shape mismatch")
    for split in ("val", "test"):
        expected_shape = [
            len(tensors[f"{split}_indices"]),
            FEATURE_DIMENSION,
        ]
        if list(tensors[f"{split}_features"].shape) != expected_shape:
            raise ValueError(f"Feature-bank {split} shape mismatch")
    if (
        result["features"]["train_shape"] != train_shape
        or result["features"]["val_shape"] != list(tensors["val_features"].shape)
        or result["features"]["test_shape"] != list(tensors["test_features"].shape)
        or result["features"]["dimension"] != FEATURE_DIMENSION
        or result["features"]["dtype"] != "bfloat16"
    ):
        raise ValueError("Feature-bank result feature metadata mismatch")
    expected_view_hashes = [
        _tensor_sha256(tensors["train_features"][index]) for index in range(view_count)
    ]
    if (
        result["features"]["train_view_sha256"] != expected_view_hashes
        or _tensor_sha256(tensors["train_features"])
        != result["features"]["train_sha256"]
        or _tensor_sha256(tensors["val_features"]) != result["features"]["val_sha256"]
        or _tensor_sha256(tensors["test_features"]) != result["features"]["test_sha256"]
    ):
        raise ValueError("Feature-bank feature checksum mismatch")
    _validate_chunks(result=result, root=root, config=config, bank=tensors)
    cuda_context = (
        torch.cuda.device(target_device)
        if target_device.type == "cuda"
        else nullcontext()
    )
    with cuda_context, torch.device(target_device):
        for key in ("train_features", "val_features", "test_features"):
            tensors[key] = tensors[key].to(target_device)
    return {**tensors, "metadata": result}


def create_feature_bank(config: FeatureBankConfig) -> dict:
    worker_path = config.output_dir / "worker-pid.json"
    config.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(path=worker_path, payload={"pid": os.getpid()})
    if config.run.deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.set_num_threads(config.run.cpu_threads)
    set_seeds(config.run.seed)
    torch.use_deterministic_algorithms(config.run.deterministic)
    torch.backends.cudnn.benchmark = not config.run.deterministic

    started = time.monotonic()
    result_path = config.output_dir / "result.json"
    config_payload = config.model_dump(mode="json")
    if result_path.exists():
        existing = json.loads(result_path.read_text())
        if not isinstance(existing, dict) or existing.get("config") != config_payload:
            raise ValueError("Existing feature bank has a different config")

    manifest_path = config.run.manifest.resolve()
    manifest = read_manifest(path=manifest_path)
    transform = _transform_provenance(pipeline=config.pipeline)
    model, initialization = _initialize_model(config=config)
    source = _source_provenance(
        config=config,
        manifest=_manifest_provenance(payload=manifest, path=manifest_path),
        transform=transform,
        initialization=initialization,
    )
    if result_path.exists():
        loaded = load_feature_bank(
            path=config.output_dir,
            device=torch.device(config.run.device),
        )
        if loaded["metadata"]["source"]["fingerprint"] != source["fingerprint"]:
            raise ValueError("Existing feature bank has a different source")
        return loaded["metadata"]

    views_dir = config.output_dir / "views"
    views_dir.mkdir(parents=True, exist_ok=True)
    assigned = _assigned_epochs(config)
    source_fingerprint = source["fingerprint"]
    train_source = _make_source(config=config, payload=manifest, split="train")
    train_chunks: list[dict] = []
    train_timings: list[dict] = []
    if config.pipeline == "fixed":
        chunk_path = views_dir / "fixed_view_0000.pt"
        chunk, timing = _extract_or_reuse(
            model=model,
            config=config,
            source_fingerprint=source_fingerprint,
            source=train_source,
            split="train",
            source_epoch=0,
            path=chunk_path,
        )
        train_chunks.append(chunk)
        train_timings.append(timing)
    else:
        for view, epoch in enumerate(assigned):
            chunk_path = views_dir / f"epoch_{epoch:04d}.pt"
            chunk, timing = _extract_or_reuse(
                model=model,
                config=config,
                source_fingerprint=source_fingerprint,
                source=train_source,
                split="train",
                source_epoch=epoch,
                path=chunk_path,
            )
            train_chunks.append(chunk)
            train_timings.append(timing)

    split_chunks: dict[str, dict] = {}
    split_timings: dict[str, dict] = {}
    for split in ("val", "test"):
        source_split = _make_source(config=config, payload=manifest, split=split)
        chunk, timing = _extract_or_reuse(
            model=model,
            config=config,
            source_fingerprint=source_fingerprint,
            source=source_split,
            split=split,
            source_epoch=0,
            path=views_dir / f"{split}.pt",
        )
        split_chunks[split] = chunk
        split_timings[split] = timing

    if config.pipeline == "fixed":
        train_features = train_chunks[0]["features"].unsqueeze(0).contiguous()
        train_epochs = torch.tensor([0], dtype=torch.int64)
        train_epoch_to_view = torch.zeros(len(assigned), dtype=torch.int64)
    else:
        train_features = torch.empty(
            (len(train_chunks), len(train_source), FEATURE_DIMENSION),
            dtype=FEATURE_DTYPE,
        )
        for view, chunk in enumerate(train_chunks):
            train_features[view].copy_(chunk["features"])
        train_epochs = torch.tensor(assigned, dtype=torch.int64)
        train_epoch_to_view = torch.arange(len(assigned), dtype=torch.int64)
    bank_payload = {
        "schema": SCHEMA,
        "version": VERSION,
        "source_fingerprint": source_fingerprint,
        "train_features": train_features,
        "train_epochs": train_epochs,
        "train_epoch_to_view": train_epoch_to_view,
        "train_indices": train_chunks[0]["indices"],
        "train_ages": train_chunks[0]["ages"],
        "val_features": split_chunks["val"]["features"],
        "val_indices": split_chunks["val"]["indices"],
        "val_ages": split_chunks["val"]["ages"],
        "test_features": split_chunks["test"]["features"],
        "test_indices": split_chunks["test"]["indices"],
        "test_ages": split_chunks["test"]["ages"],
    }
    bank_path = config.output_dir / "bank.pt"
    _atomic_save_torch(path=bank_path, payload=bank_payload)
    bank_sha256 = _file_sha256(path=bank_path)

    rows = _source_rows(payload=manifest, config=config)
    view_epochs = [0] if config.pipeline == "fixed" else assigned
    view_records = [
        _chunk_result(
            view=view,
            source_epoch=view_epochs[view],
            payload=chunk,
            timing=train_timings[view],
        )
        for view, chunk in enumerate(train_chunks)
    ]
    chunk_files = (
        [views_dir / "fixed_view_0000.pt"]
        if config.pipeline == "fixed"
        else [views_dir / f"epoch_{epoch:04d}.pt" for epoch in assigned]
    )
    chunk_files.extend(views_dir / f"{split}.pt" for split in ("val", "test"))
    chunk_paths = [
        {
            "path": path.relative_to(config.output_dir).as_posix(),
            "sha256": _file_sha256(path=path),
        }
        for path in sorted(chunk_files)
    ]
    result = {
        "schema": SCHEMA,
        "version": VERSION,
        "complete": True,
        "created_at": datetime.now(UTC).isoformat(),
        "config": config_payload,
        "source": source,
        "data": {
            "manifest": source["manifest"],
            "rows": rows,
        },
        "epochs": {
            "assigned": assigned,
            "view_epochs": view_epochs,
            "epoch_to_view": train_epoch_to_view.tolist(),
        },
        "features": {
            "dimension": FEATURE_DIMENSION,
            "dtype": "bfloat16",
            "train_shape": list(train_features.shape),
            "val_shape": list(split_chunks["val"]["features"].shape),
            "test_shape": list(split_chunks["test"]["features"].shape),
            "train_sha256": _tensor_sha256(train_features),
            "train_view_sha256": [
                _tensor_sha256(train_features[view])
                for view in range(len(train_features))
            ],
            "val_sha256": _tensor_sha256(split_chunks["val"]["features"]),
            "test_sha256": _tensor_sha256(split_chunks["test"]["features"]),
            "views": view_records,
        },
        "timings": {
            "total_seconds": time.monotonic() - started,
            "train_views": view_records,
            "val": split_timings["val"],
            "test": split_timings["test"],
        },
        "chunks": chunk_paths,
        "bank_file": "bank.pt",
        "bank_sha256": bank_sha256,
        "bank_bytes": bank_path.stat().st_size,
    }
    write_json(path=result_path, payload=result)
    try:
        validated = load_feature_bank(
            path=config.output_dir,
            device=torch.device(config.run.device),
        )
    except (KeyError, OSError, RuntimeError, TypeError, ValueError):
        result_path.unlink(missing_ok=True)
        raise
    return validated["metadata"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = FeatureBankConfig.model_validate_json(args.config.read_text())
    result = create_feature_bank(config)
    print(
        json.dumps(
            {
                "output_dir": str(config.output_dir),
                "pipeline": result["config"]["pipeline"],
                "train_shape": result["features"]["train_shape"],
                "bank_sha256": result["bank_sha256"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
