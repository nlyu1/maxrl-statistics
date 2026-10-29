import argparse
import json
import math
import os
import platform
from pathlib import Path
from typing import Literal

import polars as pl
import torch
from pydantic import ConfigDict
from pydantic.dataclasses import dataclass
from torch import Tensor

from src.experiments.utils import cleanup_cuda
from src.experiments.utkface.adaptation_data import (
    Split,
    build_image_loader,
    epoch_order,
)
from src.experiments.utkface.adaptation_metrics import prediction_values
from src.experiments.utkface.adaptation_train import (
    file_sha256,
    git_head,
    json_sha256,
    model_sha256,
    source_root,
)
from src.experiments.utkface.config import RunConfig
from src.experiments.utkface.data import read_manifest, training_indices
from src.experiments.utkface.losses import NUM_BINS, log_policy
from src.experiments.utkface.mechanism_config import MechanismConfig, MechanismVariant
from src.experiments.utkface.mechanism_train import (
    MECHANISM_SOURCE_FILES,
    build_research_model,
    head_context,
    save_checkpoint,
    tensor_sha256,
)
from src.experiments.utkface.metrics import median_spearman
from src.experiments.utkface.noise import noise_summary, noisy_labels
from src.experiments.utkface.run import write_json

Checkpoint = Literal[
    "best", "final", "epoch_025", "epoch_050", "epoch_100", "epoch_200", "epoch_500"
]
CHECKPOINTS: tuple[Checkpoint, ...] = (
    "best",
    "final",
    "epoch_025",
    "epoch_050",
    "epoch_100",
    "epoch_200",
    "epoch_500",
)
SPLITS: tuple[Split, ...] = ("train", "val", "test")
RIDGE_ALPHAS = tuple(10.0**exponent for exponent in range(-4, 5))
TARGETS = ("supplied_labels", "oracle_clean_labels")
DEFAULT_VARIANTS = (
    MechanismVariant.CATEGORICAL_MSE,
    MechanismVariant.CATEGORICAL_TAILRL,
    MechanismVariant.HEAD_TAILRL_BACKBONE_MSE,
    MechanismVariant.HEAD_MSE_BACKBONE_TAILRL,
)
FEATURE_DIMENSION = 512
PROBE_VIEW_SEED = 20261029
PROBE_ORDER_SEED = 20261029
CPU_THREADS = 4
PROBE_SOURCE_FILES = (
    *MECHANISM_SOURCE_FILES,
    "src/experiments/utkface/mechanism_probe.py",
)


@dataclass(kw_only=True, frozen=True, config=ConfigDict(arbitrary_types_allowed=True))
class FeatureSplit:
    indices: Tensor
    clean_ages: Tensor
    supplied_labels: Tensor
    features: Tensor
    native_outputs: Tensor


def point_metrics(*, predictions: Tensor, targets: Tensor) -> dict:
    predictions = predictions.detach().cpu().double().flatten()
    targets = targets.detach().cpu().long().flatten()
    if predictions.shape != targets.shape or predictions.numel() == 0:
        raise ValueError(
            "Point predictions and targets must have equal nonempty shapes"
        )
    if not torch.isfinite(predictions).all():
        raise ValueError("Point predictions must be finite")
    errors = predictions - targets
    mse = errors.square().mean().item()
    rank = median_spearman(
        pairs=list(zip(predictions.tolist(), targets.tolist(), strict=True))
    )
    return {
        "count": len(targets),
        "mae": errors.abs().mean().item(),
        "mse": mse,
        "rmse": math.sqrt(mse),
        "spearman": rank["median_spearman"],
        "spearman_defined": rank["median_spearman_defined"],
    }


def expected_indices(*, manifest: dict, run: RunConfig) -> dict[Split, Tensor]:
    train = training_indices(
        payload=manifest, limit=run.train_limit, subset_seed=run.subset_seed
    )
    result = {
        "train": train,
        "val": sorted(manifest["splits"]["val"]),
        "test": sorted(manifest["splits"]["test"]),
    }
    if run.eval_limit:
        raise ValueError("Common readouts require complete validation and test splits")
    if min(map(len, result.values())) < 2:
        raise ValueError("Common readouts require at least two images in every split")
    return {
        split: torch.tensor(result[split], dtype=torch.int64, device="cpu")
        for split in SPLITS
    }


def source_hashes() -> dict[str, str]:
    root = source_root()
    return {name: file_sha256(path=root / name) for name in PROBE_SOURCE_FILES}


def load_source(
    *, job: dict, checkpoint: Checkpoint, hashes: dict
) -> tuple[MechanismConfig, dict, dict]:
    identifier = job["identifier"]
    if Path(identifier).name != identifier or identifier in {"", ".", ".."}:
        raise ValueError(f"Invalid job identifier: {identifier}")
    config = MechanismConfig.model_validate_json(Path(job["config"]).read_text())
    run = config.adaptation.run
    if (
        config.variant.value != job["variant"]
        or run.noise.std_years != job["noise_sd"]
        or run.output_dir.resolve() != Path(job["output"]).resolve()
        or run.noise.mode not in {"none", "fixed"}
    ):
        raise ValueError(f"Plan/config mismatch or unsupported noise: {identifier}")
    result_path = run.output_dir / "result.json"
    result = json.loads(result_path.read_text())
    payload = config.model_dump(mode="json")
    if result["status"] != "complete" or result["config"] != payload:
        raise ValueError(
            f"Source run must be complete and config-matched: {identifier}"
        )
    milestone = checkpoint.startswith("epoch_")
    if milestone:
        expected_epoch = int(checkpoint.removeprefix("epoch_"))
        if expected_epoch not in config.adaptation.control.model_milestones:
            raise ValueError(f"Checkpoint milestone was not declared: {checkpoint}")
        path = run.output_dir / "model-milestones" / f"{checkpoint}.pt"
        matching_history = [
            row for row in result["history"] if row["epoch"] == expected_epoch
        ]
        if len(matching_history) != 1:
            raise ValueError(f"Milestone requires one matching history row: {path}")
        expected_validation = matching_history[0]["validation"]
    else:
        expected_epoch = result[f"{checkpoint}_epoch"]
        expected_validation = result[f"{checkpoint}_validation"]
        path = run.output_dir / f"{checkpoint}.pt"
    saved = torch.load(path, map_location="cpu", weights_only=True)
    provenance = saved["provenance"]
    if (
        saved["config"] != payload
        or saved["model_sha256"] != model_sha256(state=saved["model"])
        or saved["epoch"] != expected_epoch
        or saved["validation"] != expected_validation
        or saved["provenance"] != result["provenance"]
        or provenance["mechanism_config"] != payload
        or provenance["fingerprint"]
        != json_sha256({
            key: value for key, value in provenance.items() if key != "fingerprint"
        })
    ):
        raise ValueError(f"Source checkpoint/provenance mismatch: {path}")
    if not milestone and saved["model_sha256"] != result[f"{checkpoint}_model_sha256"]:
        raise ValueError(f"Selected/final model hash differs from its result: {path}")
    if checkpoint == "final" and saved["epoch"] != run.epochs:
        raise ValueError(
            f"Final checkpoint has not reached the configured epoch: {path}"
        )
    for name, expected in provenance["source_sha256"].items():
        if hashes[name] != expected:
            raise ValueError(f"Source code changed since training: {name}")
    metadata = {
        "identifier": identifier,
        "block": job["block"],
        "noise_sd": job["noise_sd"],
        "variant": config.variant.value,
        "checkpoint": checkpoint,
        "epoch": saved["epoch"],
        "config_path": str(Path(job["config"]).resolve()),
        "config_sha256": file_sha256(path=Path(job["config"])),
        "checkpoint_path": str(path.resolve()),
        "checkpoint_sha256": file_sha256(path=path),
        "model_sha256": saved["model_sha256"],
        "result_sha256": file_sha256(path=result_path),
        "training_provenance": provenance,
        "source_training_labels_sha256": result["history"][-1][
            "training_labels_sha256"
        ],
        "source_validation": saved["validation"],
    }
    return config, saved, metadata


def validate_split(
    *,
    values: FeatureSplit,
    indices: Tensor,
    clean_ages: Tensor,
    labels: Tensor,
    scalar: bool,
) -> None:
    count = len(indices)
    output_shape = (count, 1 if scalar else NUM_BINS)
    if (
        not torch.equal(values.indices, indices)
        or not torch.equal(values.clean_ages, clean_ages[indices])
        or not torch.equal(values.supplied_labels, labels[indices])
        or values.features.shape != (count, FEATURE_DIMENSION)
        or values.native_outputs.shape != output_shape
        or values.features.dtype != torch.float32
        or values.native_outputs.dtype != torch.float32
        or values.indices.dtype != torch.int64
        or values.clean_ages.dtype != torch.int64
        or values.supplied_labels.dtype != torch.int64
        or any(
            value.device.type != "cpu"
            for value in (
                values.indices,
                values.clean_ages,
                values.supplied_labels,
                values.features,
                values.native_outputs,
            )
        )
        or not torch.isfinite(values.features).all()
        or not torch.isfinite(values.native_outputs).all()
    ):
        raise ValueError("Feature cache shape, labels, dtype or image order is invalid")


def split_payload(*, values: FeatureSplit) -> dict[str, Tensor]:
    return {
        "indices": values.indices,
        "clean_ages": values.clean_ages,
        "supplied_labels": values.supplied_labels,
        "features": values.features,
        "native_outputs": values.native_outputs,
    }


def collect_features(
    *,
    config: MechanismConfig,
    saved: dict,
    metadata: dict,
    manifest: dict,
    indices: dict[Split, Tensor],
    device: torch.device,
    path: Path,
) -> tuple[dict[Split, FeatureSplit], dict]:
    run = config.adaptation.run
    clean = torch.tensor(
        [row["age"] for row in manifest["records"]], dtype=torch.int64, device="cpu"
    )
    labels = noisy_labels(ages=clean.tolist(), config=run.noise, epoch=1)
    for split in ("val", "test"):
        labels[indices[split]] = clean[indices[split]]
    extraction = {
        "source": metadata,
        "manifest_sha256": file_sha256(path=run.manifest),
        "manifest_fingerprint": manifest["fingerprint"],
        "transform": "Resize(256); CenterCrop(224); ImageNet normalization",
        "view_seed": PROBE_VIEW_SEED,
        "order_seed": PROBE_ORDER_SEED,
        "training_loader_epoch": 1,
        "canonical_order": "Increasing global manifest index within each split",
        "supplied_labels": "Fixed corrupted labels for training; clean labels for validation/test",
        "indices_sha256": {
            split: tensor_sha256(values=indices[split]) for split in SPLITS
        },
        "precision": "bf16 encoder and native head; FP32 CPU feature/output storage",
        "probe_source_sha256": source_hashes(),
    }
    if (
        saved["provenance"]["manifest_fingerprint"] != manifest["fingerprint"]
        or saved["provenance"]["manifest_sha256"] != extraction["manifest_sha256"]
        or saved["provenance"]["training_indices_sha256"]
        != json_sha256(indices["train"].tolist())
        or saved["provenance"]["clean_training_labels_sha256"]
        != tensor_sha256(values=clean[indices["train"]])
        or metadata["source_training_labels_sha256"]
        != tensor_sha256(values=labels[indices["train"]])
    ):
        raise ValueError("Source training manifest, subset or clean labels differ")
    reused = path.is_file()
    if reused:
        cached = torch.load(path, map_location="cpu", weights_only=True)
        if (
            cached["schema"] != "utkface-mechanism-features-v1"
            or cached["provenance"] != extraction
        ):
            raise ValueError(f"Existing feature cache has different provenance: {path}")
        splits = {split: FeatureSplit(**cached["splits"][split]) for split in SPLITS}
        audit = cached["audit"]
    else:
        divisors = [
            size
            for size in range(2, min(run.eval_batch_size, len(indices["train"])) + 1)
            if len(indices["train"]) % size == 0
        ]
        if not divisors:
            raise ValueError(
                "No extraction batch size covers the training subset exactly"
            )
        extraction_run = run.model_copy(
            update={
                "device": str(device),
                "precision": "bf16",
                "batch_size": max(divisors),
            }
        )
        model = build_research_model(config=config).eval().requires_grad_(False)
        model.load_state_dict(saved["model"], strict=True)
        splits = {}
        loader_orders = {}
        with torch.cuda.device(device), torch.inference_mode():
            model = model.to(device)
            for split in SPLITS:
                with torch.device("cpu"):
                    loader = build_image_loader(
                        run=extraction_run,
                        pipeline="fixed",
                        split=split,
                        epoch=1 if split == "train" else 0,
                        view_seed=PROBE_VIEW_SEED,
                        order_seed=PROBE_ORDER_SEED,
                    )
                    expected_order = (
                        indices["train"][
                            epoch_order(
                                count=len(indices["train"]),
                                epoch=1,
                                order_seed=PROBE_ORDER_SEED,
                            )
                        ]
                        if split == "train"
                        else torch.tensor(
                            manifest["splits"][split], dtype=torch.int64, device="cpu"
                        )
                    )
                    observed, features, outputs, ages = [], [], [], []
                    for images, batch_ages, batch_indices in loader:
                        with (
                            torch.device(device),
                            torch.autocast(device_type="cuda", dtype=torch.bfloat16),
                        ):
                            batch_features = model.encoder(
                                images.to(device, non_blocking=True)
                            )
                            with head_context(
                                features=batch_features, precision=config.head_precision
                            ):
                                batch_outputs = model.fc(
                                    batch_features.float()
                                    if config.head_precision == "fp32"
                                    else batch_features
                                )
                        observed.append(batch_indices.cpu())
                        ages.append(batch_ages.cpu())
                        features.append(batch_features.float().cpu())
                        outputs.append(batch_outputs.float().cpu())
                    del loader
                    observed_indices = torch.cat(observed)
                    if not torch.equal(observed_indices, expected_order):
                        raise ValueError(
                            f"Extraction omitted or reordered images in {split}"
                        )
                    order = observed_indices.argsort()
                    splits[split] = FeatureSplit(
                        indices=observed_indices[order],
                        clean_ages=torch.cat(ages)[order].long(),
                        supplied_labels=labels[observed_indices[order]],
                        features=torch.cat(features)[order],
                        native_outputs=torch.cat(outputs)[order],
                    )
                    loader_orders[split] = tensor_sha256(values=observed_indices)
        del model
        cleanup_cuda(device=device)
        audit = {
            "loader_order_sha256": loader_orders,
            "extraction_batch_size": extraction_run.batch_size,
            "evaluation_batch_size": extraction_run.eval_batch_size,
            "device": str(device),
            "gpu_name": torch.cuda.get_device_name(device),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            "torch": str(torch.__version__),
            "cuda": str(torch.version.cuda),
        }
    for split in SPLITS:
        validate_split(
            values=splits[split],
            indices=indices[split],
            clean_ages=clean,
            labels=labels,
            scalar=config.scalar,
        )
    actual_hashes = {
        split: {
            name: tensor_sha256(values=values)
            for name, values in split_payload(values=splits[split]).items()
        }
        for split in SPLITS
    }
    if reused:
        if cached["tensor_sha256"] != actual_hashes:
            raise ValueError(f"Feature cache tensor hash mismatch: {path}")
    else:
        save_checkpoint(
            path=path,
            payload={
                "schema": "utkface-mechanism-features-v1",
                "provenance": extraction,
                "audit": audit,
                "tensor_sha256": actual_hashes,
                "splits": {
                    split: split_payload(values=splits[split]) for split in SPLITS
                },
            },
        )
    return splits, {
        "path": str(path.resolve()),
        "sha256": file_sha256(path=path),
        "provenance": extraction,
        "audit": audit,
        "tensor_sha256": actual_hashes,
    }


def native_diagnostics(*, splits: dict[Split, FeatureSplit], scalar: bool) -> dict:
    result = {}
    for split, values in splits.items():
        predictions = prediction_values(outputs=values.native_outputs, scalar=scalar)[
            :, 0
        ]
        entropy = None
        if not scalar:
            log_probs = log_policy(logits=values.native_outputs.double())
            entropy = -(log_probs.exp() * log_probs).sum(-1).mean().item()
        result[split] = {
            "clean": point_metrics(predictions=predictions, targets=values.clean_ages),
            "entropy": entropy,
            "entropy_defined": not scalar,
        }
        if split == "train":
            noise = (values.supplied_labels - values.clean_ages).double()
            error = predictions.double() - values.clean_ages
            centered_noise = noise - noise.mean()
            variance = centered_noise.square().mean().item()
            covariance = ((error - error.mean()) * centered_noise).mean().item()
            result[split].update({
                "supplied_labels": point_metrics(
                    predictions=predictions, targets=values.supplied_labels
                ),
                "label_noise": noise_summary(
                    clean=values.clean_ages, noisy=values.supplied_labels
                ),
                "noise_alignment_slope": covariance / variance
                if variance > 0
                else None,
                "noise_alignment_slope_defined": variance > 0,
                "noise_error_covariance": covariance,
                "noise_variance": variance,
            })
    return result


def fit_readouts(*, splits: dict[Split, FeatureSplit], path: Path) -> dict:
    with torch.device("cpu"), torch.inference_mode():
        train = splits["train"].features.double()
        feature_mean = train.mean(0)
        feature_scale = (train - feature_mean).square().mean(0).sqrt()
        constant = feature_scale == 0
        feature_scale = feature_scale.masked_fill(constant, 1)
        standardized = {
            split: (splits[split].features.double() - feature_mean) / feature_scale
            for split in SPLITS
        }
        gram = standardized["train"].T @ standardized["train"]
        eigenvalues, eigenvectors = torch.linalg.eigh(gram)
        tolerance = (
            torch.finfo(torch.float64).eps
            * FEATURE_DIMENSION
            * max(eigenvalues.abs().max().item(), 1.0)
        )
        if eigenvalues.min().item() < -tolerance:
            raise ValueError(
                "Standardized feature Gram matrix is not positive semidefinite"
            )
        eigenvalues = eigenvalues.clamp_min(0)
        targets = torch.stack(
            (splits["train"].supplied_labels, splits["train"].clean_ages), dim=1
        ).double()
        target_mean = targets.mean(0)
        projected = eigenvectors.T @ (standardized["train"].T @ (targets - target_mean))
        candidates = {target: [] for target in TARGETS}
        selected_states = {}
        for alpha in RIDGE_ALPHAS:
            coefficients = eigenvectors @ (projected / (eigenvalues[:, None] + alpha))
            predictions = {
                split: standardized[split] @ coefficients + target_mean
                for split in SPLITS
            }
            for column, target in enumerate(TARGETS):
                candidate = {
                    "alpha": alpha,
                    "validation": point_metrics(
                        predictions=predictions["val"][:, column],
                        targets=splits["val"].clean_ages,
                    ),
                    "test": point_metrics(
                        predictions=predictions["test"][:, column],
                        targets=splits["test"].clean_ages,
                    ),
                }
                previous = candidates[target]
                if not previous or (candidate["validation"]["mse"], -alpha) < min(
                    (row["validation"]["mse"], -row["alpha"]) for row in previous
                ):
                    selected_states[target] = {
                        "alpha": alpha,
                        "coefficients": coefficients[:, column].clone(),
                        "intercept": target_mean[column].clone(),
                        "predictions": {
                            split: predictions[split][:, column].clone()
                            for split in SPLITS
                        },
                    }
                previous.append(candidate)
        selected = {
            target: min(
                candidates[target],
                key=lambda row: (row["validation"]["mse"], -row["alpha"]),
            )
            for target in TARGETS
        }
        total = eigenvalues.sum().item()
        mass = eigenvalues / total if total > 0 else torch.zeros_like(eigenvalues)
        positive_mass = mass[mass > 0]
        geometry = {
            "definition": "Training covariance after coordinate-wise standardization",
            "constant_feature_columns": int(constant.sum().item()),
            "covariance_trace": total / len(train),
            "participation_ratio": total**2 / eigenvalues.square().sum().item()
            if total > 0
            else 0.0,
            "entropy_effective_rank": (-positive_mass.mul(positive_mass.log()).sum())
            .exp()
            .item()
            if total > 0
            else 0.0,
        }
        save_checkpoint(
            path=path,
            payload={
                "schema": "utkface-mechanism-ridge-v1",
                "feature_mean": feature_mean,
                "feature_scale": feature_scale,
                "indices": {split: splits[split].indices for split in SPLITS},
                "selected": selected_states,
            },
        )
    return {
        "solver": "CPU float64 eigendecomposition of standardized feature Gram matrix",
        "objective": "sum((prediction-target)^2) + alpha*sum(coefficients^2); unpenalized intercept",
        "prediction_bounds": "Unclipped scalar ridge predictions",
        "standardization": "Training mean and population SD; zero SD columns use scale 1",
        "alpha_grid": list(RIDGE_ALPHAS),
        "selection": "Minimum clean-validation MSE; exact ties favor larger alpha; no test selection",
        "targets": {
            target: {
                "oracle_clean_labels_representation_diagnostic": target
                == "oracle_clean_labels",
                "primary": target == "supplied_labels",
                "candidates": candidates[target],
                "selected": selected[target],
            }
            for target in TARGETS
        },
        "feature_geometry": geometry,
        "artifact": {"path": str(path.resolve()), "sha256": file_sha256(path=path)},
    }


def write_tables(*, report: dict, output_dir: Path) -> None:
    selected_rows, candidate_rows, native_rows = [], [], []
    for row in report["runs"]:
        identity = {
            key: row["source"][key]
            for key in (
                "identifier",
                "block",
                "noise_sd",
                "variant",
                "checkpoint",
                "epoch",
            )
        }
        for target, readout in row["ridge"]["targets"].items():
            selected_rows.append({
                **identity,
                "target": target,
                "oracle_clean_labels_representation_diagnostic": readout[
                    "oracle_clean_labels_representation_diagnostic"
                ],
                "alpha": readout["selected"]["alpha"],
                **{
                    f"{split}_{name}": value
                    for split in ("validation", "test")
                    for name, value in readout["selected"][split].items()
                },
            })
            for candidate in readout["candidates"]:
                for split in ("validation", "test"):
                    candidate_rows.append({
                        **identity,
                        "target": target,
                        "alpha": candidate["alpha"],
                        "split": split,
                        **candidate[split],
                    })
        for split, native in row["native"].items():
            native_rows.append({
                **identity,
                "split": split,
                **{f"clean_{key}": value for key, value in native["clean"].items()},
                "entropy": native["entropy"],
                "noise_alignment_slope": native.get("noise_alignment_slope"),
                **{
                    f"supplied_{name}": native["supplied_labels"][name]
                    if split == "train"
                    else None
                    for name in ("mae", "mse", "rmse", "spearman")
                },
            })
    for name, rows in (
        ("selected-readouts.tsv", selected_rows),
        ("ridge-grid.tsv", candidate_rows),
        ("native-diagnostics.tsv", native_rows),
    ):
        pl.DataFrame(rows, infer_schema_length=None).write_csv(
            output_dir / name, separator="\t"
        )


def probe(
    *,
    root: Path,
    device: str,
    output_dir: Path,
    variants: tuple[MechanismVariant, ...],
    checkpoints: tuple[Checkpoint, ...],
) -> dict:
    if (
        int(os.environ.get("WORLD_SIZE", "1")) != 1
        or int(os.environ.get("RANK", "0")) != 0
    ):
        raise ValueError("The common readout probe requires one process")
    cuda = torch.device(device)
    if cuda.type != "cuda" or cuda.index is None:
        raise ValueError("Feature extraction requires an explicit CUDA device")
    if not variants or len(set(variants)) != len(variants):
        raise ValueError("Select distinct, nonempty variants")
    if not checkpoints or len(set(checkpoints)) != len(checkpoints):
        raise ValueError("Select distinct, nonempty checkpoints")
    if any(checkpoint not in CHECKPOINTS for checkpoint in checkpoints):
        raise ValueError(f"Checkpoint choices must be among {CHECKPOINTS}")
    plan_path = root / "plan.json"
    plan = json.loads(plan_path.read_text())
    if plan["schema"] != "utkface-mechanism-study-v1":
        raise ValueError("Unsupported mechanism study plan")
    jobs = [job for job in plan["jobs"] if MechanismVariant(job["variant"]) in variants]
    if not jobs or len({job["identifier"] for job in jobs}) != len(jobs):
        raise ValueError("No selected jobs or duplicate identifiers")
    if set(variants) != {MechanismVariant(job["variant"]) for job in jobs}:
        raise ValueError("Some selected variants are absent from the plan")
    torch.set_num_threads(CPU_THREADS)
    hashes = source_hashes()
    requested = {
        "root": str(root.resolve()),
        "plan_sha256": file_sha256(path=plan_path),
        "variants": [variant.value for variant in variants],
        "checkpoints": list(checkpoints),
        "alpha_grid": list(RIDGE_ALPHAS),
        "view_seed": PROBE_VIEW_SEED,
        "order_seed": PROBE_ORDER_SEED,
        "source_sha256": hashes,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "report.json"
    if report_path.is_file():
        previous = json.loads(report_path.read_text())
        if previous["requested"] != requested:
            raise ValueError("Output directory belongs to a different probe request")
    report = {
        "schema": "utkface-mechanism-common-readout-v1",
        "requested": requested,
        "git_commit": git_head(),
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "cpu_threads": CPU_THREADS,
        "primary_target": "Supplied training labels, including the fixed corruption field",
        "oracle_target": "Oracle clean-label representation diagnostic; not an available-label training claim",
        "noise_alignment_definition": "Cov(native_mean-clean_age, supplied_label-clean_age) / Var(supplied_label-clean_age), population covariance over fixed-center-crop training images",
        "selection_scope": "Ridge alpha uses clean validation MSE only; source best encoders were selected by clean validation mean MAE",
        "planned_sources": len(jobs) * len(checkpoints),
        "runs": [],
    }
    shared_indices = {}
    shared_manifest = ""
    group_labels = {}
    group_sources = {}
    for job in jobs:
        for checkpoint in checkpoints:
            config, saved, metadata = load_source(
                job=job, checkpoint=checkpoint, hashes=hashes
            )
            manifest = read_manifest(path=config.adaptation.run.manifest)
            indices = expected_indices(manifest=manifest, run=config.adaptation.run)
            if not shared_indices:
                shared_indices = indices
                shared_manifest = manifest["fingerprint"]
                report["manifest_fingerprint"] = shared_manifest
                report["split_counts"] = {
                    split: len(indices[split]) for split in SPLITS
                }
                report["shared_indices_sha256"] = {
                    split: tensor_sha256(values=indices[split]) for split in SPLITS
                }
            if shared_manifest != manifest["fingerprint"] or any(
                not torch.equal(indices[split], shared_indices[split])
                for split in SPLITS
            ):
                raise ValueError(
                    "Source encoders do not use the same faces and split order"
                )
            stem = f"{job['identifier']}--{checkpoint}"
            splits, feature_audit = collect_features(
                config=config,
                saved=saved,
                metadata=metadata,
                manifest=manifest,
                indices=indices,
                device=cuda,
                path=output_dir / "features" / f"{stem}.pt",
            )
            group = (job["block"], job["noise_sd"])
            paired_source = {
                name: saved["provenance"][name]
                for name in (
                    "encoder_initial_sha256",
                    "model_seed",
                    "head_seed",
                    "view_seed",
                    "order_seed",
                    "noise_seed",
                )
            }
            paired_source["training_epochs"] = config.adaptation.run.epochs
            if group in group_sources and group_sources[group] != paired_source:
                raise ValueError(
                    "Paired source encoders have unmatched initialization or seeds"
                )
            group_sources[group] = paired_source
            label_hash = tensor_sha256(values=splits["train"].supplied_labels)
            if group in group_labels and group_labels[group] != label_hash:
                raise ValueError(
                    "Paired source encoders have different corruption fields"
                )
            group_labels[group] = label_hash
            row = {
                "source": metadata,
                "features": feature_audit,
                "native": native_diagnostics(splits=splits, scalar=config.scalar),
                "ridge": fit_readouts(
                    splits=splits, path=output_dir / "readouts" / f"{stem}.pt"
                ),
            }
            row["native_validation_mae_delta"] = (
                row["native"]["val"]["clean"]["mae"]
                - metadata["source_validation"]["mae_mean"]
            )
            report["runs"].append(row)
            report["completed_sources"] = len(report["runs"])
            (output_dir / "runs").mkdir(exist_ok=True)
            write_json(path=output_dir / "runs" / f"{stem}.json", payload=row)
            write_json(path=report_path, payload=report)
            write_tables(report=report, output_dir=output_dir)
            print(
                json.dumps({
                    "completed_sources": report["completed_sources"],
                    "planned_sources": report["planned_sources"],
                    "source": stem,
                }),
                flush=True,
            )
            del saved, splits
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=[variant.value for variant in MechanismVariant],
        default=[variant.value for variant in DEFAULT_VARIANTS],
    )
    parser.add_argument(
        "--checkpoints", nargs="+", choices=CHECKPOINTS, default=["final"]
    )
    args = parser.parse_args()
    probe(
        root=args.root,
        device=args.device,
        output_dir=args.output_dir,
        variants=tuple(MechanismVariant(value) for value in args.variants),
        checkpoints=tuple(args.checkpoints),
    )


if __name__ == "__main__":
    main()
