import argparse
import json
from pathlib import Path

import torch
from torch import nn
from torch.optim import AdamW

from src.experiments.utkface.adaptation_config import AdaptationJob, AdaptationPlan
from src.experiments.utkface.adaptation_train import warmup_constant_scheduler
from src.experiments.utkface.run import write_json


def relocate(*, value, source: Path, destination: Path):
    if isinstance(value, dict):
        return {
            key: relocate(value=item, source=source, destination=destination)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [
            relocate(value=item, source=source, destination=destination)
            for item in value
        ]
    if isinstance(value, str) and value.startswith(str(source)):
        return value.replace(str(source), str(destination), 1).replace(
            "/epoch_025.pt",
            "/epoch_001.pt",
        )
    return value


def create_probe(*, source: AdaptationPlan, output_dir: Path) -> AdaptationPlan:
    identifiers = {
        "aug_original_full_cat_mse_sd0",
        "scratch_warmup_constant_grpo_mse_sd5",
        "aug_fixed_full_scalar_mae_sd0",
        "aug_original_frozen_cat_mse_sd0",
        "aug_original_frozen_ce_sd0",
        "aug_original_frozen_tail_sd10",
        "aug_original_frozen_grpo_mse_sd0",
        "aug_original_frozen_scalar_mae_sd0",
        "head_cat_mse_sd0_scalar_mae",
        "post_sl25_sd0_full_tail",
        "post_sl25_sd0_frozen_ce",
    }
    by_id = {job.identifier: job for job in source.jobs}
    pending = list(identifiers)
    while pending:
        identifier = pending.pop()
        for dependency in by_id[identifier].dependencies:
            if dependency not in identifiers:
                identifiers.add(dependency)
                pending.append(dependency)
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "configs").mkdir()
    jobs = []
    for original in source.jobs:
        if original.identifier not in identifiers:
            continue
        config = relocate(
            value=json.loads(original.config_path.read_text()),
            source=source.output_dir,
            destination=output_dir,
        )
        config["run"].update({"train_limit": 128, "eval_limit": 64, "num_workers": 2})
        config["run"]["epochs"] = 1 if config["global_start_epoch"] else 2
        config["global_start_epoch"] = 1 if config["global_start_epoch"] else 0
        if original.kind == "train":
            config["checkpoint_every"] = 1
        config_path = output_dir / "configs" / f"{original.identifier}.json"
        write_json(path=config_path, payload=config)
        jobs.append(
            AdaptationJob(
                identifier=original.identifier,
                kind=original.kind,
                config_path=config_path,
                output_dir=Path(config["run"]["output_dir"]),
                dependencies=original.dependencies,
                priority=original.priority,
            )
        )
    plan = source.model_copy(
        update={
            "output_dir": output_dir,
            "jobs": tuple(jobs),
            "per_gpu_jobs": 4,
            "max_gpus": 1,
        }
    )
    write_json(path=output_dir / "plan.json", payload=plan.model_dump(mode="json"))
    return plan


def verify_schedule() -> dict:
    parameter = nn.Parameter(torch.ones(1, device="cpu"))
    optimizer = AdamW([parameter], lr=3e-4)
    scheduler = warmup_constant_scheduler(
        optimizer=optimizer,
        total_steps=3200,
        warmup_ratio=0.05,
    )
    rates = []
    for _ in range(3200):
        rates.append(optimizer.param_groups[0]["lr"])
        optimizer.step()
        scheduler.step()
    if abs(rates[0] - 3e-4 / 160) > 1e-12:
        raise AssertionError("Warmup did not start at the prescribed fraction")
    if any(abs(value - 3e-4) > 1e-12 for value in rates[159:]):
        raise AssertionError("Constant learning rate decays after warmup")
    return {"first_lr": rates[0], "warmup_steps": 160, "last_lr": rates[-1]}


def verify(*, plan: AdaptationPlan) -> dict:
    point_metrics = tuple(
        f"{metric}_{statistic}"
        for statistic in ("mean", "median", "mode")
        for metric in ("mae", "rmse", "spearman")
    )
    rows = []
    for job in plan.jobs:
        result = json.loads((job.output_dir / "result.json").read_text())
        if job.kind == "bank":
            if not (job.output_dir / "bank.pt").is_file():
                raise AssertionError(f"Bank payload missing: {job.identifier}")
        else:
            for name in ("best.pt", "final.pt"):
                saved = torch.load(
                    job.output_dir / name, map_location="cpu", weights_only=True
                )
                if not isinstance(saved.get("model"), dict):
                    raise TypeError(f"Non-native checkpoint: {job.identifier}/{name}")
            checkpoints = sorted((job.output_dir / "checkpoints").glob("epoch_*.pt"))
            if not checkpoints:
                raise AssertionError(f"Regular checkpoints missing: {job.identifier}")
            for path in checkpoints:
                saved = torch.load(path, map_location="cpu", weights_only=True)
                if "optimizer" not in saved or "rng" not in saved:
                    raise AssertionError(f"Resume state missing: {path}")
            history = json.loads((job.output_dir / "history.json").read_text())
            for row in history:
                metrics = row["validation"]
                if not all(name in metrics for name in point_metrics):
                    raise AssertionError(f"Point metrics missing: {job.identifier}")
        rows.append({
            "job": job.identifier,
            "kind": job.kind,
            "completed": bool(result),
        })
    summary = {"jobs": rows, "schedule": verify_schedule()}
    write_json(path=plan.output_dir / "verification.json", payload=summary)
    return summary


def prepare_resume(*, plan: AdaptationPlan, output_dir: Path) -> AdaptationPlan:
    job = next(
        item for item in plan.jobs if item.identifier == "aug_original_full_cat_mse_sd0"
    )
    if not (job.output_dir / "result.json").is_file():
        raise FileNotFoundError("Completed source probe is required")
    output_dir.mkdir(parents=True, exist_ok=False)
    for relative in ("result.json", "final.pt", "checkpoints/epoch_002.pt"):
        path = job.output_dir / relative
        if not path.resolve().is_relative_to(plan.output_dir.resolve()):
            raise ValueError("Resume probe artifact escaped its workspace")
        destination = path.with_name(f"{path.stem}-before-resume{path.suffix}")
        if destination.exists():
            raise FileExistsError(destination)
        path.replace(destination)
    resumed = plan.model_copy(
        update={
            "output_dir": output_dir,
            "jobs": (job.model_copy(update={"dependencies": ()}),),
            "per_gpu_jobs": 1,
            "max_gpus": 1,
        }
    )
    write_json(path=output_dir / "plan.json", payload=resumed.model_dump(mode="json"))
    return resumed


def verify_resume(*, plan: AdaptationPlan) -> dict:
    job = plan.jobs[0]
    before = torch.load(
        job.output_dir / "final-before-resume.pt", map_location="cpu", weights_only=True
    )
    after = torch.load(
        job.output_dir / "final.pt", map_location="cpu", weights_only=True
    )
    maxima = []
    for name, expected in before["model"].items():
        actual = after["model"][name]
        if actual.shape != expected.shape or not torch.isfinite(actual).all():
            raise AssertionError(f"Invalid resumed parameter: {name}")
        maxima.append(float((actual.double() - expected.double()).abs().max()))
    old_result = json.loads((job.output_dir / "result-before-resume.json").read_text())
    new_result = json.loads((job.output_dir / "result.json").read_text())
    if [row["epoch"] for row in new_result["history"]] != [1, 2]:
        raise AssertionError("Resumed history did not retain the correct epochs")
    for old, new in zip(old_result["history"], new_result["history"], strict=True):
        if old["learning_rates"] != new["learning_rates"]:
            raise AssertionError("Resumed learning rates differ")
    previous = torch.load(
        job.output_dir / "checkpoints" / "epoch_002-before-resume.pt",
        map_location="cpu",
        weights_only=True,
    )
    resumed = torch.load(
        job.output_dir / "checkpoints" / "epoch_002.pt",
        map_location="cpu",
        weights_only=True,
    )
    if previous["scheduler"] != resumed["scheduler"]:
        raise AssertionError("Resumed scheduler state differs")
    for key, state in previous["optimizer"]["state"].items():
        if not torch.equal(state["step"], resumed["optimizer"]["state"][key]["step"]):
            raise AssertionError("Resumed optimizer step differs")
    result = {
        "max_parameter_difference": max(maxima),
        "history_epochs": [1, 2],
        "optimizer_steps_match": True,
        "scheduler_state_matches": True,
        "selected_test_mae_difference": new_result["best_test"]["mae_mean"]
        - old_result["best_test"]["mae_mean"],
        "deterministic": False,
    }
    write_json(path=plan.output_dir / "verification.json", payload=result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--prepare-resume", action="store_true")
    parser.add_argument("--verify-resume", action="store_true")
    args = parser.parse_args()
    source = AdaptationPlan.model_validate_json(args.plan.read_text())
    if args.verify_resume:
        print(json.dumps(verify_resume(plan=source), indent=2))
    elif args.prepare_resume:
        if args.output_dir is None:
            raise ValueError("Resume probe requires --output-dir")
        prepared = prepare_resume(plan=source, output_dir=args.output_dir)
        print(f"Prepared resume probe at {prepared.output_dir}")
    elif args.verify:
        print(json.dumps(verify(plan=source), indent=2))
    else:
        if args.output_dir is None:
            raise ValueError("Probe creation requires --output-dir")
        probe = create_probe(source=source, output_dir=args.output_dir)
        print(f"Created {len(probe.jobs)} preflight jobs at {probe.output_dir}")


if __name__ == "__main__":
    main()
