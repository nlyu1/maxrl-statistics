import argparse
import copy
import json
import os
import platform
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import torchvision
from torch import Tensor, nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from src.experiments.utils import cleanup_cuda, set_seeds
from src.experiments.utkface.adaptation_data import build_image_loader, epoch_order
from src.experiments.utkface.adaptation_train import (
    SOURCE_FILES,
    build_optimizer,
    build_scheduler,
    collection_device,
    context,
    evaluate_loader,
    file_sha256,
    git_head,
    json_sha256,
    model_sha256,
    model_state_cpu,
    restore_rng_state,
    serialized_rng_state,
    source_root,
    write_prediction_artifact,
)
from src.experiments.utkface.config import LossConfig
from src.experiments.utkface.data import read_manifest, training_indices
from src.experiments.utkface.losses import MAX_AGE, NUM_BINS, policy_loss, scalar_loss
from src.experiments.utkface.mechanism_config import MechanismConfig, MechanismLoss
from src.experiments.utkface.model import ScalarHead, build_model
from src.experiments.utkface.noise import noisy_labels
from src.experiments.utkface.run import write_json

UPDATE_SAMPLE_EVERY = 5
MECHANISM_SOURCE_FILES = (
    *SOURCE_FILES,
    "src/experiments/utkface/mechanism_config.py",
    "src/experiments/utkface/mechanism_train.py",
)


class BinomialHead(ScalarHead):
    def __init__(self, *, features: int) -> None:
        super().__init__(features=features)
        bins = torch.arange(NUM_BINS, dtype=torch.float64)
        log_coefficients = (
            torch.lgamma(torch.tensor(float(NUM_BINS), dtype=torch.float64))
            - torch.lgamma(bins + 1)
            - torch.lgamma(NUM_BINS - bins)
        )
        self.register_buffer("bins", bins.float(), persistent=False)
        self.register_buffer(
            "log_coefficients", log_coefficients.float(), persistent=False
        )

    def forward(self, features: Tensor) -> Tensor:
        scores = self.linear(features).float()
        return self.log_coefficients + self.bins * scores


class ResearchModel(nn.Module):
    def __init__(self, *, encoder: nn.Module, fc: nn.Module) -> None:
        super().__init__()
        self.encoder = encoder
        self.fc = fc

    def forward(self, images: Tensor) -> Tensor:
        return self.fc(self.encoder(images))


def build_research_model(*, config: MechanismConfig) -> ResearchModel:
    with torch.device("cpu"), torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(config.adaptation.run.seed)
        encoder = build_model(
            pretrained=config.adaptation.initialization == "imagenet",
            head="categorical",
        )
        features = encoder.fc.in_features
        encoder.fc = nn.Identity()
    with torch.device("cpu"), torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(config.head_seed)
        if config.scalar:
            fc = ScalarHead(features=features)
        elif config.binomial:
            fc = BinomialHead(features=features)
        else:
            fc = nn.Linear(features, NUM_BINS)
    return ResearchModel(encoder=encoder, fc=fc)


def routed_losses(
    *,
    features: Tensor,
    head: nn.Linear,
    targets: Tensor,
    head_loss: LossConfig,
    backbone_loss: LossConfig,
) -> tuple[Tensor, Tensor]:
    head_logits = F.linear(features.detach(), head.weight, head.bias)
    backbone_logits = F.linear(features, head.weight.detach(), head.bias.detach())
    return (
        policy_loss(logits=head_logits, targets=targets, config=head_loss),
        policy_loss(logits=backbone_logits, targets=targets, config=backbone_loss),
    )


def loss_config(*, template: LossConfig, method: MechanismLoss) -> LossConfig:
    return template.model_copy(update={"method": method})


def tensor_sha256(*, values: Tensor) -> str:
    return model_sha256(state={"values": values})


def provenance(
    *,
    config: MechanismConfig,
    model: ResearchModel,
    manifest: dict,
    train_indices: list[int],
) -> dict:
    root = source_root()
    clean_labels = torch.tensor(
        [record["age"] for record in manifest["records"]], dtype=torch.int64
    )
    source = {
        "mechanism_config": config.model_dump(mode="json"),
        "manifest_fingerprint": manifest["fingerprint"],
        "manifest_sha256": file_sha256(path=config.adaptation.run.manifest),
        "training_indices_sha256": json_sha256(train_indices),
        "training_count": len(train_indices),
        "clean_labels_sha256": tensor_sha256(values=clean_labels),
        "clean_training_labels_sha256": tensor_sha256(
            values=clean_labels[train_indices]
        ),
        "encoder_initial_sha256": model_sha256(state=model.encoder.state_dict()),
        "head_initial_sha256": model_sha256(state=model.fc.state_dict()),
        "model_seed": config.adaptation.run.seed,
        "head_seed": config.head_seed,
        "view_seed": config.adaptation.view_seed,
        "order_seed": config.adaptation.order_seed,
        "noise_seed": config.adaptation.run.noise.seed,
        "head_method": config.head_method,
        "backbone_method": config.backbone_method,
        "loss_scaling": "native_canonical_losses",
        "reported_train_loss": "head_objective",
        "gradient_clipping": {
            "groups": ["backbone", "head"],
            "norm_bound": config.adaptation.run.clip_grad_norm,
        },
        "update_diagnostics": {
            "sample_every_steps": UPDATE_SAMPLE_EVERY,
            "sample_offset": 0,
            "rms": "actual_parameter_delta_including_weight_decay",
            "cosine": "actual_delta_versus_negative_clipped_gradient",
        },
        "checkpoint_policy": "best_and_final_models_plus_one_latest_per_epoch",
        "selection": {"split": "val", "metric": "mae_mean", "ties": "earliest"},
        "environment": {
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "torchvision": str(torchvision.__version__),
            "cuda": str(torch.version.cuda),
            "hostname": platform.node(),
            "git_commit": git_head(),
        },
        "source_sha256": {
            name: file_sha256(path=root / name) for name in MECHANISM_SOURCE_FILES
        },
    }
    return {**source, "fingerprint": json_sha256(source)}


def save_checkpoint(*, path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".pt.tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def model_checkpoint(
    *,
    state: dict[str, Tensor],
    config: MechanismConfig,
    epoch: int,
    validation: dict,
    run_provenance: dict,
) -> dict:
    return {
        "model": state,
        "model_sha256": model_sha256(state=state),
        "config": config.model_dump(mode="json"),
        "epoch": epoch,
        "validation": validation,
        "variant": config.variant.value,
        "head": config.adaptation.run.head,
        "provenance": run_provenance,
    }


def save_latest(
    *,
    path: Path,
    model: ResearchModel,
    optimizer: AdamW,
    scheduler: LambdaLR | None,
    config: MechanismConfig,
    epoch: int,
    history: list[dict],
    initialization: dict,
    best_epoch: int,
    best_validation: dict,
    best_model: dict[str, Tensor],
    run_provenance: dict,
    elapsed_seconds: float,
) -> None:
    state = model_state_cpu(model=model)
    save_checkpoint(
        path=path,
        payload={
            "model": state,
            "model_sha256": model_sha256(state=state),
            "best_model": best_model,
            "best_model_sha256": model_sha256(state=best_model),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "rng": serialized_rng_state(),
            "config": config.model_dump(mode="json"),
            "epoch": epoch,
            "history": history,
            "initialization": initialization,
            "best_epoch": best_epoch,
            "best_validation": best_validation,
            "provenance": run_provenance,
            "elapsed_seconds": elapsed_seconds,
        },
    )


def load_latest(
    *,
    path: Path,
    config: MechanismConfig,
    model: ResearchModel,
    optimizer: AdamW,
    scheduler: LambdaLR | None,
    run_provenance: dict,
) -> dict:
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if saved["config"] != config.model_dump(mode="json"):
        raise ValueError(f"Checkpoint config differs: {path}")
    if saved["provenance"]["fingerprint"] != run_provenance["fingerprint"]:
        raise ValueError(f"Checkpoint provenance differs: {path}")
    if saved["model_sha256"] != model_sha256(state=saved["model"]):
        raise ValueError(f"Checkpoint model hash differs: {path}")
    if saved["best_model_sha256"] != model_sha256(state=saved["best_model"]):
        raise ValueError(f"Checkpoint selected-model hash differs: {path}")
    epoch = saved["epoch"]
    if not 0 <= epoch <= config.adaptation.run.epochs:
        raise ValueError(f"Checkpoint epoch is outside the run horizon: {path}")
    if [row["epoch"] for row in saved["history"]] != list(range(1, epoch + 1)):
        raise ValueError(f"Checkpoint history is incomplete: {path}")
    if not 0 <= saved["best_epoch"] <= epoch:
        raise ValueError(f"Checkpoint selected epoch is invalid: {path}")
    model.load_state_dict(saved["model"], strict=True)
    optimizer.load_state_dict(saved["optimizer"])
    if (scheduler is None) != (saved["scheduler"] is None):
        raise ValueError(f"Checkpoint scheduler presence differs: {path}")
    if scheduler is not None:
        scheduler.load_state_dict(saved["scheduler"])
    restore_rng_state(state=saved["rng"])
    return saved


@torch.no_grad()
def update_statistics(
    *, parameters: tuple[nn.Parameter, ...], before: tuple[Tensor, ...]
) -> Tensor:
    delta_squares = []
    grad_squares = []
    dot_products = []
    for parameter, previous in zip(parameters, before, strict=True):
        if parameter.grad is None:
            raise ValueError("A trainable parameter has no gradient")
        delta = parameter.detach().float() - previous
        gradient = parameter.grad.detach().float()
        delta_squares.append(delta.square().sum())
        grad_squares.append(gradient.square().sum())
        dot_products.append(-(delta * gradient).sum())
    delta_squared = torch.stack(delta_squares).sum()
    gradient_squared = torch.stack(grad_squares).sum()
    denominator = (delta_squared * gradient_squared).sqrt()
    defined = denominator > 0
    cosine = torch.where(
        defined,
        torch.stack(dot_products).sum() / denominator.clamp_min(1e-30),
        0.0,
    )
    count = sum(parameter.numel() for parameter in parameters)
    return torch.stack((
        (delta_squared / count).sqrt(),
        cosine.clamp(-1, 1),
        defined.to(delta_squared.dtype),
    ))


def train_epoch(
    *,
    model: ResearchModel,
    loader: DataLoader,
    labels: Tensor,
    config: MechanismConfig,
    device: torch.device,
    optimizer: AdamW,
    scheduler: LambdaLR | None,
    epoch: int,
) -> dict:
    model.train()
    groups = (tuple(model.encoder.parameters()), tuple(model.fc.parameters()))
    head_config = loss_config(
        template=config.adaptation.run.loss, method=config.head_method
    )
    backbone_config = loss_config(
        template=config.adaptation.run.loss, method=config.backbone_method
    )
    loss_totals = torch.zeros(2, device=device, dtype=torch.float64)
    norm_totals = torch.zeros(2, device=device, dtype=torch.float64)
    norm_maxima = torch.zeros(2, device=device, dtype=torch.float64)
    clipped_steps = torch.zeros(2, device=device, dtype=torch.float64)
    update_totals = torch.zeros((2, 3), device=device, dtype=torch.float64)
    sampled_steps = []
    count = 0
    steps = 0
    started = time.monotonic()
    for step, (images, _, indices) in enumerate(loader):
        optimizer.zero_grad(set_to_none=True)
        with context(device=device):
            batch_images = images.to(device, non_blocking=True)
            batch_labels = labels[indices].to(device, non_blocking=True)
            features = model.encoder(batch_images)
            if config.scalar:
                head_value = scalar_loss(
                    predictions=model.fc(features),
                    targets=batch_labels,
                    config=head_config,
                )
                backbone_value = head_value
                objective = head_value
            elif config.binomial:
                head_value = policy_loss(
                    logits=model.fc(features), targets=batch_labels, config=head_config
                )
                backbone_value = head_value
                objective = head_value
            else:
                if not isinstance(model.fc, nn.Linear):
                    raise TypeError("Categorical routes require a linear head")
                head_value, backbone_value = routed_losses(
                    features=features,
                    head=model.fc,
                    targets=batch_labels,
                    head_loss=head_config,
                    backbone_loss=backbone_config,
                )
                objective = head_value + backbone_value
        if not torch.isfinite(objective):
            raise FloatingPointError(
                f"Nonfinite training loss at epoch={epoch}, step={step}"
            )
        objective.backward()
        norms = torch.stack([
            nn.utils.clip_grad_norm_(
                parameters,
                config.adaptation.run.clip_grad_norm,
                error_if_nonfinite=True,
            )
            for parameters in groups
        ]).double()
        sample = step % UPDATE_SAMPLE_EVERY == 0
        before = (
            tuple(
                tuple(parameter.detach().float().clone() for parameter in parameters)
                for parameters in groups
            )
            if sample
            else ()
        )
        optimizer.step()
        if sample:
            for group_index, parameters in enumerate(groups):
                update_totals[group_index] += update_statistics(
                    parameters=parameters, before=before[group_index]
                ).double()
            sampled_steps.append(step + 1)
            del before
        if scheduler is not None:
            scheduler.step()
        batch_count = len(indices)
        loss_totals += (
            torch.stack((head_value.detach(), backbone_value.detach())).double()
            * batch_count
        )
        norm_totals += norms
        norm_maxima = torch.maximum(norm_maxima, norms)
        clipped_steps += norms > config.adaptation.run.clip_grad_norm
        count += batch_count
        steps += 1
    if not count:
        raise ValueError("Training epoch produced no examples")
    losses = (loss_totals / count).cpu().tolist()
    norms_mean = (norm_totals / steps).cpu().tolist()
    norms_max = norm_maxima.cpu().tolist()
    clipped = clipped_steps.cpu().tolist()
    updates = update_totals.cpu().tolist()
    diagnostics = {}
    for index, name in enumerate(("backbone", "head")):
        rms_sum, cosine_sum, defined_count = updates[index]
        diagnostics[name] = {
            "raw_grad_norm_mean": norms_mean[index],
            "raw_grad_norm_max": norms_max[index],
            "clipped_steps": int(clipped[index]),
            "clip_frequency": clipped[index] / steps,
            "update_rms_mean": rms_sum / len(sampled_steps),
            "update_negative_gradient_cosine_mean": (
                cosine_sum / defined_count if defined_count else None
            ),
            "update_cosine_defined_samples": int(defined_count),
        }
    return {
        "epoch": epoch,
        "train_loss": losses[0],
        "head_loss": losses[0],
        "backbone_loss": losses[1],
        "train_count": count,
        "steps": steps,
        "gradient_diagnostics": diagnostics,
        "update_sample_steps": sampled_steps,
        "learning_rates": [group["lr"] for group in optimizer.param_groups],
        "epoch_seconds": time.monotonic() - started,
    }


def image_loader(*, config: MechanismConfig, split: str, epoch: int) -> DataLoader:
    return build_image_loader(
        run=config.adaptation.run,
        pipeline=config.adaptation.pipeline,
        split=split,
        epoch=epoch,
        view_seed=config.adaptation.view_seed,
        order_seed=config.adaptation.order_seed,
    )


def train_job(*, config: MechanismConfig) -> dict:
    if (
        int(os.environ.get("WORLD_SIZE", "1")) != 1
        or int(os.environ.get("RANK", "0")) != 0
    ):
        raise ValueError("Mechanism training supports one process on one GPU")
    device = torch.device(config.adaptation.run.device)
    if device.type != "cuda":
        raise ValueError("Training requires CUDA; use --verify for CPU verification")
    with torch.cuda.device(device):
        device = collection_device(config=config.adaptation)
    torch.set_num_threads(config.adaptation.run.cpu_threads)
    torch.use_deterministic_algorithms(config.adaptation.run.deterministic)
    torch.backends.cudnn.benchmark = not config.adaptation.run.deterministic
    output_dir = config.adaptation.run.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = config.model_dump(mode="json")
    result_path = output_dir / "result.json"
    if result_path.is_file():
        completed = json.loads(result_path.read_text())
        if completed.get("config") != payload:
            raise ValueError(f"Completed result config differs: {result_path}")
        return completed
    config_path = output_dir / "config.json"
    if config_path.is_file() and json.loads(config_path.read_text()) != payload:
        raise ValueError(f"Existing run config differs: {config_path}")
    write_json(path=config_path, payload=payload)
    write_json(path=output_dir / "worker-pid.json", payload={"pid": os.getpid()})
    started = time.monotonic()
    elapsed_before = 0.0
    set_seeds(config.adaptation.run.seed)
    manifest = read_manifest(path=config.adaptation.run.manifest)
    train_indices = training_indices(
        payload=manifest,
        limit=config.adaptation.run.train_limit,
        subset_seed=config.adaptation.run.subset_seed,
    )
    steps_per_epoch = len(train_indices) // config.adaptation.run.batch_size
    if not steps_per_epoch:
        raise ValueError("Training split must contain one full batch")
    model = build_research_model(config=config)
    run_provenance = provenance(
        config=config, model=model, manifest=manifest, train_indices=train_indices
    )
    provenance_path = output_dir / "provenance.json"
    if (
        provenance_path.is_file()
        and json.loads(provenance_path.read_text()) != run_provenance
    ):
        raise ValueError(f"Existing run provenance differs: {provenance_path}")
    write_json(path=provenance_path, payload=run_provenance)

    with torch.cuda.device(device):
        model = model.to(device)
        optimizer = build_optimizer(config=config.adaptation, model=model)
        scheduler = build_scheduler(
            config=config.adaptation,
            optimizer=optimizer,
            steps_per_epoch=steps_per_epoch,
        )
        val_loader = image_loader(config=config, split="val", epoch=0)
        latest_path = output_dir / "latest.pt"
        torch.cuda.reset_peak_memory_stats(device)
        if latest_path.is_file():
            saved = load_latest(
                path=latest_path,
                config=config,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                run_provenance=run_provenance,
            )
            history = saved["history"]
            initialization = saved["initialization"]
            best_epoch = saved["best_epoch"]
            best_validation = saved["best_validation"]
            best_model = saved["best_model"]
            start_after = saved["epoch"]
            elapsed_before = saved["elapsed_seconds"]
            del saved
        else:
            initial_evaluation = evaluate_loader(
                model=model,
                loader=val_loader,
                config=config.adaptation,
                device=device,
            )
            initialization = {
                "epoch": 0,
                "validation": initial_evaluation["metrics"],
                "validation_seconds": initial_evaluation["seconds"],
                "predictions_sha256": json_sha256(initial_evaluation["rows"]),
                "encoder_sha256": run_provenance["encoder_initial_sha256"],
                "head_sha256": run_provenance["head_initial_sha256"],
            }
            write_json(path=output_dir / "initialization.json", payload=initialization)
            write_prediction_artifact(
                path=output_dir / "epoch-zero-validation-predictions.json",
                checkpoint="epoch_zero",
                split="val",
                epoch=0,
                result=initial_evaluation,
            )
            history = []
            best_epoch = 0
            best_validation = initial_evaluation["metrics"]
            best_model = model_state_cpu(model=model)
            start_after = 0
            save_latest(
                path=latest_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                config=config,
                epoch=0,
                history=history,
                initialization=initialization,
                best_epoch=best_epoch,
                best_validation=best_validation,
                best_model=best_model,
                run_provenance=run_provenance,
                elapsed_seconds=time.monotonic() - started,
            )
        save_checkpoint(
            path=output_dir / "best.pt",
            payload=model_checkpoint(
                state=best_model,
                config=config,
                epoch=best_epoch,
                validation=best_validation,
                run_provenance=run_provenance,
            ),
        )
        clean_ages = [record["age"] for record in manifest["records"]]
        train_index_tensor = torch.tensor(
            train_indices, device="cpu", dtype=torch.int64
        )
        for epoch in range(start_after + 1, config.adaptation.run.epochs + 1):
            labels = noisy_labels(
                ages=clean_ages, config=config.adaptation.run.noise, epoch=epoch
            )
            train_loader = image_loader(config=config, split="train", epoch=epoch)
            row = train_epoch(
                model=model,
                loader=train_loader,
                labels=labels,
                config=config,
                device=device,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
            )
            del train_loader
            validation = evaluate_loader(
                model=model,
                loader=val_loader,
                config=config.adaptation,
                device=device,
            )
            row["validation"] = validation["metrics"]
            row["validation_seconds"] = validation["seconds"]
            row["predictions_sha256"] = json_sha256(validation["rows"])
            order = epoch_order(
                count=len(train_indices),
                epoch=epoch,
                order_seed=config.adaptation.order_seed,
            )
            consumed = train_index_tensor[order[: row["train_count"]]]
            row["global_labels_sha256"] = tensor_sha256(values=labels)
            row["training_labels_sha256"] = tensor_sha256(
                values=labels[train_index_tensor]
            )
            row["consumed_indices_sha256"] = tensor_sha256(values=consumed)
            row["consumed_labels_sha256"] = tensor_sha256(values=labels[consumed])
            history.append(row)
            if row["validation"]["mae_mean"] < best_validation["mae_mean"]:
                best_epoch = epoch
                best_validation = row["validation"]
                best_model = model_state_cpu(model=model)
                save_checkpoint(
                    path=output_dir / "best.pt",
                    payload=model_checkpoint(
                        state=best_model,
                        config=config,
                        epoch=best_epoch,
                        validation=best_validation,
                        run_provenance=run_provenance,
                    ),
                )
            write_json(path=output_dir / "history.json", payload=history)
            if epoch in config.adaptation.control.model_milestones:
                save_checkpoint(
                    path=output_dir / "model-milestones" / f"epoch_{epoch:03d}.pt",
                    payload=model_checkpoint(
                        state=model_state_cpu(model=model),
                        config=config,
                        epoch=epoch,
                        validation=row["validation"],
                        run_provenance=run_provenance,
                    ),
                )
            save_latest(
                path=latest_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                config=config,
                epoch=epoch,
                history=history,
                initialization=initialization,
                best_epoch=best_epoch,
                best_validation=best_validation,
                best_model=best_model,
                run_provenance=run_provenance,
                elapsed_seconds=elapsed_before + time.monotonic() - started,
            )
            print(
                json.dumps({
                    "epoch": epoch,
                    "validation": row["validation"],
                    "best_epoch": best_epoch,
                }),
                flush=True,
            )

        final_epoch = config.adaptation.run.epochs
        final_state = model_state_cpu(model=model)
        final_hash = model_sha256(state=final_state)
        best_hash = model_sha256(state=best_model)
        final_validation = evaluate_loader(
            model=model, loader=val_loader, config=config.adaptation, device=device
        )
        save_checkpoint(
            path=output_dir / "final.pt",
            payload=model_checkpoint(
                state=final_state,
                config=config,
                epoch=final_epoch,
                validation=final_validation["metrics"],
                run_provenance=run_provenance,
            ),
        )
        write_prediction_artifact(
            path=output_dir / "final-validation-predictions.json",
            checkpoint="final",
            split="val",
            epoch=final_epoch,
            result=final_validation,
        )
        model.load_state_dict(best_model, strict=True)
        selected_validation = evaluate_loader(
            model=model, loader=val_loader, config=config.adaptation, device=device
        )
        write_prediction_artifact(
            path=output_dir / "best-validation-predictions.json",
            checkpoint="best",
            split="val",
            epoch=best_epoch,
            result=selected_validation,
        )
        selection = {
            "frozen": True,
            "selection_split": "val",
            "selection_metric": "mae_mean",
            "best_epoch": best_epoch,
            "best_validation": best_validation,
            "best_model_sha256": best_hash,
            "final_epoch": final_epoch,
            "final_validation": final_validation["metrics"],
            "final_model_sha256": final_hash,
        }
        write_json(path=output_dir / "selection.json", payload=selection)
        best_test = None
        final_test = None
        if config.adaptation.run.evaluate_test:
            test_loader = image_loader(config=config, split="test", epoch=0)
            best_test = evaluate_loader(
                model=model, loader=test_loader, config=config.adaptation, device=device
            )
            write_prediction_artifact(
                path=output_dir / "best-test-predictions.json",
                checkpoint="best",
                split="test",
                epoch=best_epoch,
                result=best_test,
            )
            model.load_state_dict(final_state, strict=True)
            final_test = evaluate_loader(
                model=model, loader=test_loader, config=config.adaptation, device=device
            )
            write_prediction_artifact(
                path=output_dir / "final-test-predictions.json",
                checkpoint="final",
                split="test",
                epoch=final_epoch,
                result=final_test,
            )
        result = {
            "config": payload,
            "status": "complete",
            "provenance": run_provenance,
            "initialization": initialization,
            "history": history,
            **selection,
            "best_validation": selected_validation["metrics"],
            "best_test": best_test["metrics"] if best_test is not None else None,
            "final_test": final_test["metrics"] if final_test is not None else None,
            "elapsed_seconds": elapsed_before + time.monotonic() - started,
            "peak_allocated_mib": torch.cuda.max_memory_allocated(device) / 2**20,
            "peak_reserved_mib": torch.cuda.max_memory_reserved(device) / 2**20,
            "predictions": {
                "epoch_zero": "epoch-zero-validation-predictions.json",
                "best_validation": "best-validation-predictions.json",
                "final_validation": "final-validation-predictions.json",
                "best_test": "best-test-predictions.json"
                if best_test is not None
                else None,
                "final_test": "final-test-predictions.json"
                if final_test is not None
                else None,
            },
        }
        write_json(path=output_dir / "history.json", payload=history)
        write_json(path=result_path, payload=result)
    cleanup_cuda(device=device)
    return result


def verify_binomial_identity() -> dict:
    with torch.device("cpu"), torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(1701)
        scalar = ScalarHead(features=7)
        torch.random.default_generator.manual_seed(1701)
        binomial = BinomialHead(features=7)
        if model_sha256(state=scalar.state_dict()) != model_sha256(
            state=binomial.state_dict()
        ):
            raise AssertionError("Scalar and binomial seeds produced different heads")
        features = torch.randn(16, 7)
        scalar_predictions = scalar(features).flatten()
        probabilities = binomial(features).softmax(-1)
        mean = (probabilities * torch.arange(NUM_BINS)).sum(-1)
        torch.testing.assert_close(mean, scalar_predictions, rtol=2e-6, atol=2e-4)
        scalar_gradient = torch.autograd.grad(
            scalar_predictions.sum(), scalar.parameters()
        )
        binomial_gradient = torch.autograd.grad(mean.sum(), binomial.parameters())
        for expected, actual in zip(scalar_gradient, binomial_gradient, strict=True):
            torch.testing.assert_close(actual, expected, rtol=3e-4, atol=3e-4)
        return {
            "mean_max_absolute_error": (mean - scalar_predictions).abs().max().item(),
            "state_keys_identical": list(scalar.state_dict())
            == list(binomial.state_dict()),
        }


def verify_routed_gradients() -> dict:
    differences = {}
    with torch.device("cpu"), torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(1702)
        template = ResearchModel(
            encoder=nn.Sequential(nn.Linear(7, 11), nn.Tanh()),
            fc=nn.Linear(11, NUM_BINS),
        ).double()
        features = torch.randn(8, 7, dtype=torch.float64)
        targets = torch.tensor([0, 2, 17, 33, 49, 64, 88, MAX_AGE])
        for method in ("mse", "ce", "tailrl_population"):
            config = LossConfig(
                method=method,
                kernel_scale_years=5,
                num_rollouts=64,
                survival_floor=0,
                std_floor=1e-4,
            )
            native = copy.deepcopy(template)
            routed = copy.deepcopy(template)
            policy_loss(
                logits=native(features), targets=targets, config=config
            ).backward()
            head_value, backbone_value = routed_losses(
                features=routed.encoder(features),
                head=routed.fc,
                targets=targets,
                head_loss=config,
                backbone_loss=config,
            )
            (head_value + backbone_value).backward()
            maximum = 0.0
            for native_parameter, routed_parameter in zip(
                native.parameters(), routed.parameters(), strict=True
            ):
                torch.testing.assert_close(
                    routed_parameter.grad, native_parameter.grad, rtol=1e-11, atol=1e-12
                )
                maximum = max(
                    maximum,
                    (routed_parameter.grad - native_parameter.grad).abs().max().item(),
                )
            differences[method] = maximum
    return differences


def main() -> None:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--config", type=Path)
    mode.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.verify:
        print(
            json.dumps(
                {
                    "binomial_identity": verify_binomial_identity(),
                    "routed_gradient_max_absolute_error": verify_routed_gradients(),
                },
                indent=2,
            )
        )
        return
    config = MechanismConfig.model_validate_json(args.config.read_text())
    train_job(config=config)


if __name__ == "__main__":
    main()
