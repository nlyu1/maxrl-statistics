import argparse
import json
from pathlib import Path

from src.experiments.utkface.adaptation_config import (
    AdaptationConfig,
    AdaptationJob,
    AdaptationPlan,
    FeatureBankConfig,
    TrainingControl,
)
from src.experiments.utkface.config import NoiseConfig, canonical_config
from src.experiments.utkface.run import write_json

VARIANTS = (
    ("cat_mse", "mse", "categorical"),
    ("ce", "ce", "categorical"),
    ("tail", "tailrl_population", "categorical"),
    ("grpo_mse", "grpo_mse_population", "categorical"),
    ("scalar_mae", "mae", "scalar"),
)
VIEW_SEED = 20261026
ORDER_SEED = 2026


def create_plan(
    *,
    manifest: Path,
    output_dir: Path,
    per_gpu_jobs: int,
    max_gpus: int = 0,
) -> AdaptationPlan:
    if not 1 <= per_gpu_jobs <= 8:
        raise ValueError("per_gpu_jobs must be between 1 and 8")
    if max_gpus < 0:
        raise ValueError("max_gpus must be nonnegative")
    output_dir.mkdir(parents=True, exist_ok=False)
    configs = output_dir / "configs"
    configs.mkdir()
    jobs: dict[str, AdaptationJob] = {}
    suites: dict[str, list[dict]] = {
        "augmentation": [],
        "scratch_cosine": [],
        "scratch_warmup_constant": [],
        "frozen_heads": [],
        "post_training": [],
        "source_extensions": [],
    }
    unused = output_dir / "unused"

    def run_config(*, identifier: str, variant: tuple, std: int, epochs: int):
        _, method, head = variant
        base = canonical_config(
            manifest=manifest, output_dir=output_dir / "runs" / identifier
        )
        return base.model_copy(
            update={
                "head": head,
                "loss": base.loss.model_copy(
                    update={"method": method, "kernel_scale_years": 2.0}
                ),
                "noise": NoiseConfig(
                    mode="fixed" if std else "none", std_years=std, seed=2026
                ),
                "epochs": epochs,
                "learning_rate": 1e-3,
                "train_limit": 4096,
                "evaluate_test": True,
                "save_checkpoint": True,
                "device": "cuda:0",
            }
        )

    def register(
        *, identifier: str, kind: str, config, dependencies: tuple, priority: int
    ):
        if identifier in jobs:
            recorded = json.loads(jobs[identifier].config_path.read_text())
            if recorded != config.model_dump(mode="json"):
                raise ValueError(
                    f"Reused job has conflicting configuration: {identifier}"
                )
            return jobs[identifier]
        path = configs / f"{identifier}.json"
        write_json(path=path, payload=config.model_dump(mode="json"))
        directory = config.run.output_dir
        job = AdaptationJob(
            identifier=identifier,
            kind=kind,
            config_path=path,
            output_dir=directory,
            dependencies=dependencies,
            priority=priority,
        )
        jobs[identifier] = job
        return job

    def bank(*, identifier: str, pipeline: str, source: str, start: int = 0):
        bank_id = f"bank_{identifier}"
        base = run_config(
            identifier=bank_id, variant=VARIANTS[0], std=0, epochs=50 - start
        )
        base = base.model_copy(update={"output_dir": output_dir / "banks" / bank_id})
        checkpoint = (
            (
                jobs[source].output_dir / "checkpoints" / "epoch_025.pt"
                if start
                else jobs[source].output_dir / "best.pt"
            )
            if source
            else unused
        )
        config = FeatureBankConfig(
            run=base,
            pipeline=pipeline,
            initialization="checkpoint" if source else "imagenet",
            initial_checkpoint=checkpoint,
            global_start_epoch=start,
            view_seed=VIEW_SEED,
            order_seed=ORDER_SEED,
        )
        return register(
            identifier=bank_id,
            kind="bank",
            config=config,
            dependencies=(source,) if source else (),
            priority=1,
        )

    def fit(
        *,
        identifier: str,
        variant: tuple,
        std: int,
        pipeline: str,
        frozen: bool,
        initialization: str = "imagenet",
        source: str = "",
        bank_job: str = "",
        start: int = 0,
        constant: bool = False,
        priority: int = 10,
    ):
        run = run_config(
            identifier=identifier, variant=variant, std=std, epochs=50 - start
        )
        run = run.model_copy(
            update={
                "pretrained": initialization == "imagenet",
                "learning_rate": 3e-4 if constant else 1e-3,
                "schedule": run.schedule.model_copy(
                    update={
                        "lr_schedule": "flat" if constant else "cosine",
                        "warmup_ratio": 0.05,
                    }
                ),
            }
        )
        checkpoint = (
            jobs[source].output_dir / "checkpoints" / "epoch_025.pt"
            if source
            else unused
        )
        config = AdaptationConfig(
            run=run,
            pipeline=pipeline,
            initialization=initialization,
            initial_checkpoint=checkpoint,
            backbone_frozen=frozen,
            use_feature_bank=frozen,
            feature_bank=jobs[bank_job].output_dir if bank_job else unused,
            global_start_epoch=start,
            view_seed=VIEW_SEED,
            order_seed=ORDER_SEED,
            checkpoint_every=5,
            backbone_learning_rate=1e-4 if start else run.learning_rate,
            schedule_kind="warmup_constant" if constant else "cosine",
            control=TrainingControl(
                schedule_epochs=run.epochs,
                extend_checkpoint=unused,
                extend_epoch=0,
                keep_latest_checkpoints=1000,
                model_milestones=(),
            ),
        )
        dependencies = tuple(dict.fromkeys(filter(None, (source, bank_job))))
        return register(
            identifier=identifier,
            kind="train",
            config=config,
            dependencies=dependencies,
            priority=priority,
        )

    imagenet_banks = {
        pipeline: bank(identifier=f"imagenet_{pipeline}", pipeline=pipeline, source="")
        for pipeline in ("original", "fixed")
    }
    for pipeline in ("original", "fixed"):
        for std in (0, 5, 10):
            for variant in VARIANTS:
                name = variant[0]
                for frozen in (False, True):
                    identifier = f"aug_{pipeline}_{'frozen' if frozen else 'full'}_{name}_sd{std}"
                    job = fit(
                        identifier=identifier,
                        variant=variant,
                        std=std,
                        pipeline=pipeline,
                        frozen=frozen,
                        bank_job=imagenet_banks[pipeline].identifier if frozen else "",
                        priority=0
                        if pipeline == "original"
                        and name in {"cat_mse", "ce"}
                        and not frozen
                        else 10,
                    )
                    suites["augmentation"].append({"job": job.identifier})

    for constant in (False, True):
        suite = "scratch_warmup_constant" if constant else "scratch_cosine"
        for std in (0, 5, 10):
            for variant in VARIANTS:
                identifier = f"{suite}_{variant[0]}_sd{std}"
                job = fit(
                    identifier=identifier,
                    variant=variant,
                    std=std,
                    pipeline="original",
                    frozen=False,
                    initialization="scratch",
                    constant=constant,
                    priority=20,
                )
                suites[suite].append({"job": job.identifier})

    sources: dict[tuple[str, int], str] = {}
    for std in (0, 1, 3, 5, 10):
        for variant in VARIANTS[:2]:
            name = variant[0]
            identifier = f"aug_original_full_{name}_sd{std}"
            if std in (1, 3):
                fit(
                    identifier=identifier,
                    variant=variant,
                    std=std,
                    pipeline="original",
                    frozen=False,
                    priority=0,
                )
                suites["source_extensions"].append({"job": identifier})
            sources[name, std] = identifier
        for source_name in ("imagenet", "cat_mse", "ce"):
            source_id = "" if source_name == "imagenet" else sources[source_name, std]
            bank_job = (
                imagenet_banks["original"]
                if not source_id
                else bank(
                    identifier=f"{source_name}_sd{std}",
                    pipeline="original",
                    source=source_id,
                )
            )
            for variant in VARIANTS:
                identifier = (
                    f"aug_original_frozen_{variant[0]}_sd{std}"
                    if source_name == "imagenet"
                    else f"head_{source_name}_sd{std}_{variant[0]}"
                )
                job = fit(
                    identifier=identifier,
                    variant=variant,
                    std=std,
                    pipeline="original",
                    frozen=True,
                    bank_job=bank_job.identifier,
                    priority=30,
                )
                suites["frozen_heads"].append({
                    "job": job.identifier,
                    "backbone_source": source_name,
                    "std": std,
                })

    for std in (0, 5, 10):
        source_id = sources["cat_mse", std]
        bank_job = bank(
            identifier=f"cat_mse_sd{std}_epoch25",
            pipeline="original",
            source=source_id,
            start=25,
        )
        for variant in VARIANTS[:4]:
            for frozen in (False, True):
                identifier = (
                    f"post_sl25_sd{std}_{'frozen' if frozen else 'full'}_{variant[0]}"
                )
                job = fit(
                    identifier=identifier,
                    variant=variant,
                    std=std,
                    pipeline="original",
                    frozen=frozen,
                    initialization="checkpoint",
                    source=source_id,
                    bank_job=bank_job.identifier if frozen else "",
                    start=25,
                    priority=5,
                )
                suites["post_training"].append({
                    "job": job.identifier,
                    "parent": source_id,
                })

    plan = AdaptationPlan(
        output_dir=output_dir,
        manifest=manifest,
        jobs=tuple(jobs.values()),
        per_gpu_jobs=per_gpu_jobs,
        max_gpus=max_gpus,
        min_free_gib=40,
        poll_seconds=2,
        dispatch_lock=Path("/tmp/utkface-extension-20261014/global-dispatch.lock"),
    )
    expected_counts = {
        "augmentation": 60,
        "scratch_cosine": 15,
        "scratch_warmup_constant": 15,
        "frozen_heads": 75,
        "post_training": 24,
        "source_extensions": 4,
    }
    if {name: len(rows) for name, rows in suites.items()} != expected_counts:
        raise ValueError("Experiment matrix does not match the authorized counts")
    for job in jobs.values():
        if any(
            dependency not in jobs or dependency == job.identifier
            for dependency in job.dependencies
        ):
            raise ValueError(f"Invalid dependencies for {job.identifier}")
    write_json(path=output_dir / "plan.json", payload=plan.model_dump(mode="json"))
    write_json(
        path=output_dir / "design.json",
        payload={
            "suites": suites,
            "counts": {name: len(rows) for name, rows in suites.items()},
            "distinct_train_jobs": sum(job.kind == "train" for job in jobs.values()),
            "feature_bank_jobs": sum(job.kind == "bank" for job in jobs.values()),
            "source_checkpoint": "Original-augmentation categorical MSE, fixed epoch 25",
            "selection": "Clean validation mean MAE, including epoch zero",
            "noise": "One fixed rounded/clipped Gaussian field per SD, seed 2026",
            "seed": 42,
            "view_seed": VIEW_SEED,
            "order_seed": ORDER_SEED,
            "scratch_constant_schedule": "Peak 3e-4; linear warmup first 5% steps; constant thereafter",
        },
    )
    return plan


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--per-gpu-jobs", type=int, choices=(4, 8), required=True)
    parser.add_argument("--max-gpus", type=int, default=0)
    args = parser.parse_args()
    plan = create_plan(
        manifest=args.manifest,
        output_dir=args.output_dir,
        per_gpu_jobs=args.per_gpu_jobs,
        max_gpus=args.max_gpus,
    )
    print(f"Created {len(plan.jobs)} jobs at {plan.output_dir}")


if __name__ == "__main__":
    main()
