import argparse
import hashlib
from collections import Counter
from pathlib import Path

from pydantic import Field

from src.config.base import BaseConfig
from src.experiments.utkface.config import RunConfig, canonical_config
from src.experiments.utkface.data import read_manifest
from src.experiments.utkface.run import write_json


class StudyJob(BaseConfig):
    identifier: str
    stage: str
    variant: str
    priority: int
    optional: bool
    config: RunConfig
    dependencies: tuple[str, ...]
    tune_from: tuple[str, ...]
    tuning_rank: int = Field(ge=0)
    gate_candidate: tuple[str, ...]
    gate_reference: tuple[str, ...]
    competitive_ratio: float = Field(gt=0)


class StudyPlan(BaseConfig):
    output_dir: Path
    data_fingerprint: str
    training_source_sha256: dict[str, str]
    budget_gpu_hours: float = Field(gt=0)
    max_concurrent: int = Field(ge=1, le=8)
    jobs: tuple[StudyJob, ...]


FULL_VARIANTS = (
    ("tailrl_population", 2.0),
    ("tailrl", 2.0),
    ("binary_maxrl_population", 2.0),
    ("binary_maxrl", 2.0),
    *(
        (f"stat_maxrl_{kind}{suffix}", scale)
        for kind in ("mae", "mse")
        for suffix in ("_population", "")
        for scale in (2.0, 5.0)
    ),
    *(
        (f"{algorithm}_{kind}{suffix}", 2.0)
        for algorithm in ("grpo", "rloo")
        for kind in ("mae", "mse")
        for suffix in ("_population", "")
    ),
    ("mse", 2.0),
    ("mae", 2.0),
)
REDUCED_VARIANTS = (
    ("tailrl_population", 2.0),
    ("mse", 2.0),
    ("mae", 2.0),
    ("ce", 2.0),
    ("stat_maxrl_mae_population", 2.0),
    ("stat_maxrl_mse_population", 2.0),
    ("grpo_mae_population", 2.0),
    ("grpo_mse_population", 2.0),
)
LEARNING_RATES = (1e-4, 3e-4, 1e-3, 3e-3)
TRAINING_SOURCES = tuple(
    f"src/experiments/utkface/{name}.py"
    for name in (
        "config",
        "data",
        "model",
        "losses",
        "advantages",
        "metrics",
        "noise",
        "distributed",
        "run",
    )
) + ("src/model/optimizer.py", "src/experiments/utils.py")


def variant_name(*, method: str, scale: float, head: str) -> str:
    name = method + (f"_s{scale:g}" if method.startswith("stat_maxrl") else "")
    return name + (f"_{head}" if head != "categorical" else "")


def make_plan(
    *, manifest: Path, output_dir: Path, budget_gpu_hours: float, max_concurrent: int
) -> StudyPlan:
    payload = read_manifest(path=manifest)
    if not payload.get("preprocessing", {}).get("deduplicated_before_split"):
        raise ValueError("The scaled study requires deduplication before splitting")
    base = canonical_config(manifest=manifest, output_dir=output_dir)
    jobs = []

    def add(
        *,
        identifier: str,
        stage: str,
        method: str,
        scale: float = 2.0,
        head: str = "categorical",
        epochs: int,
        size: int = 0,
        seed: int = 42,
        lr: float = 3e-4,
        schedule: str = "cosine",
        dependencies: tuple[str, ...] = (),
        tune_from: tuple[str, ...] = (),
        rank: int = 0,
        priority: int,
        optional: bool = False,
        noise_mode: str = "none",
        noise_std: float = 0,
        gate_candidate: tuple[str, ...] = (),
        gate_reference: tuple[str, ...] = (),
    ) -> str:
        configuration = {
            **base.model_dump(),
            "output_dir": output_dir / "runs" / identifier,
            "head": head,
            "epochs": epochs,
            "train_limit": size,
            "seed": seed,
            "learning_rate": lr,
            "schedule": {**base.schedule.model_dump(), "lr_schedule": schedule},
            "noise": {"mode": noise_mode, "std_years": noise_std, "seed": 2026},
            "loss": {
                **base.loss.model_dump(),
                "method": method,
                "kernel_scale_years": scale,
            },
            "evaluate_test": stage
            not in {"wide", "refine", "subset_tuning", "flat_tuning", "digits_tuning"},
            "save_checkpoint": stage
            not in {"wide", "refine", "subset_tuning", "flat_tuning", "digits_tuning"},
        }
        if stage == "smoke":
            configuration.update({
                "train_limit": 256,
                "eval_limit": 65,
                "save_checkpoint": False,
            })
        dependencies = tuple(
            dict.fromkeys((*dependencies, *tune_from, *gate_candidate, *gate_reference))
        )
        jobs.append(
            StudyJob(
                identifier=identifier,
                stage=stage,
                variant=variant_name(method=method, scale=scale, head=head),
                priority=priority,
                optional=optional,
                config=RunConfig.model_validate(configuration),
                dependencies=dependencies,
                tune_from=tune_from,
                tuning_rank=rank,
                gate_candidate=gate_candidate,
                gate_reference=gate_reference,
                competitive_ratio=1.05,
            )
        )
        return identifier

    smokes = tuple(
        add(
            identifier=f"smoke_{head}_{method}",
            stage="smoke",
            method=method,
            head=head,
            epochs=1,
            priority=0,
            noise_mode="fresh",
            noise_std=3,
        )
        for head, method in (
            ("categorical", "stat_maxrl_mse"),
            ("categorical", "stat_maxrl_mae"),
            ("scalar", "mae"),
            ("digits", "ce"),
        )
    )
    refined = {}
    scalar_refined = {}
    for method, scale, head in (
        *((m, s, "categorical") for m, s in FULL_VARIANTS),
        ("mse", 2.0, "scalar"),
        ("mae", 2.0, "scalar"),
    ):
        variant = variant_name(method=method, scale=scale, head=head)
        wide = tuple(
            add(
                identifier=f"wide_{variant}_lr{lr:g}",
                stage="wide",
                method=method,
                scale=scale,
                head=head,
                epochs=10,
                lr=lr,
                dependencies=smokes,
                priority=10,
            )
            for lr in LEARNING_RATES
        )
        confirm = tuple(
            add(
                identifier=f"refine_{variant}_rank{rank}",
                stage="refine",
                method=method,
                scale=scale,
                head=head,
                epochs=30,
                tune_from=wide,
                rank=rank,
                priority=20,
            )
            for rank in range(2)
        )
        (scalar_refined if head == "scalar" else refined)[method, scale] = confirm
        for seed in (42, 43, 44):
            add(
                identifier=f"main_{variant}_seed{seed}",
                stage="scalar" if head == "scalar" else "main",
                method=method,
                scale=scale,
                head=head,
                epochs=50,
                seed=seed,
                tune_from=confirm,
                priority=35 if head == "scalar" else 30,
                gate_candidate=confirm if head == "scalar" else (),
                gate_reference=refined[method, scale] if head == "scalar" else (),
            )

    subset_tuning = {}
    for size in (4096, 1024):
        for method, scale, head in (
            *((m, s, "categorical") for m, s in REDUCED_VARIANTS),
            ("mse", 2.0, "scalar"),
            ("mae", 2.0, "scalar"),
        ):
            variant = variant_name(method=method, scale=scale, head=head)
            gate = (
                {
                    "gate_candidate": scalar_refined[method, scale],
                    "gate_reference": refined[method, scale],
                }
                if head == "scalar"
                else {}
            )
            tuning = tuple(
                add(
                    identifier=f"subset_tune_{size}_{variant}_lr{lr:g}",
                    stage="subset_tuning",
                    method=method,
                    scale=scale,
                    head=head,
                    epochs=20,
                    size=size,
                    lr=lr,
                    dependencies=smokes,
                    priority=40,
                    **gate,
                )
                for lr in LEARNING_RATES
            )
            subset_tuning[method, scale, head, size] = tuning
            add(
                identifier=f"subset_{size}_{variant}_seed42",
                stage="subset",
                method=method,
                scale=scale,
                head=head,
                epochs=50,
                size=size,
                tune_from=tuning,
                priority=50,
                **gate,
            )

    for method, scale, head in (
        *((m, s, "categorical") for m, s in REDUCED_VARIANTS),
        ("mse", 2.0, "scalar"),
        ("mae", 2.0, "scalar"),
    ):
        variant = variant_name(method=method, scale=scale, head=head)
        gate = (
            {
                "gate_candidate": scalar_refined[method, scale],
                "gate_reference": refined[method, scale],
            }
            if head == "scalar"
            else {}
        )
        tuning = tuple(
            add(
                identifier=f"flat_tune_{variant}_lr{lr:g}",
                stage="flat_tuning",
                method=method,
                scale=scale,
                head=head,
                epochs=20,
                size=4096,
                lr=lr,
                schedule="flat",
                dependencies=smokes,
                priority=60,
                **gate,
            )
            for lr in LEARNING_RATES
        )
        add(
            identifier=f"flat_{variant}_seed42",
            stage="flat",
            method=method,
            scale=scale,
            head=head,
            epochs=50,
            size=4096,
            schedule="flat",
            tune_from=tuning,
            priority=65,
            **gate,
        )
        clean_tuning = subset_tuning[method, scale, head, 4096]
        for mode in ("fixed", "fresh"):
            for std in (1, 3, 5, 10):
                add(
                    identifier=f"noise_{mode}_s{std}_{variant}_seed42",
                    stage="noise",
                    method=method,
                    scale=scale,
                    head=head,
                    epochs=50,
                    size=4096,
                    tune_from=clean_tuning,
                    noise_mode=mode,
                    noise_std=std,
                    priority=80 if std == 10 else 70,
                    optional=std == 10,
                    **gate,
                )
            for std in (3, 5):
                add(
                    identifier=f"noise_{mode}_s{std}_{variant}_seed43",
                    stage="noise_replication",
                    method=method,
                    scale=scale,
                    head=head,
                    epochs=50,
                    size=4096,
                    seed=43,
                    tune_from=clean_tuning,
                    noise_mode=mode,
                    noise_std=std,
                    priority=110,
                    optional=True,
                    **gate,
                )
        if head == "scalar":
            continue
        digit_tuning = tuple(
            add(
                identifier=f"digits_tune_{variant}_lr{lr:g}",
                stage="digits_tuning",
                method=method,
                scale=scale,
                head="digits",
                epochs=20,
                size=4096,
                lr=lr,
                dependencies=smokes,
                priority=90,
                optional=True,
            )
            for lr in LEARNING_RATES[:3]
        )
        add(
            identifier=f"digits_subset_{variant}",
            stage="digits",
            method=method,
            scale=scale,
            head="digits",
            epochs=50,
            size=4096,
            tune_from=digit_tuning,
            priority=95,
            optional=True,
        )
        add(
            identifier=f"digits_full_{variant}",
            stage="digits",
            method=method,
            scale=scale,
            head="digits",
            epochs=50,
            tune_from=digit_tuning,
            priority=100,
            optional=True,
        )

    plan = StudyPlan(
        output_dir=output_dir,
        data_fingerprint=payload["fingerprint"],
        training_source_sha256={
            name: hashlib.sha256(Path(name).read_bytes()).hexdigest()
            for name in TRAINING_SOURCES
        },
        budget_gpu_hours=budget_gpu_hours,
        max_concurrent=max_concurrent,
        jobs=tuple(jobs),
    )
    identifiers = [job.identifier for job in jobs]
    if len(set(identifiers)) != len(jobs) or any(
        dep not in identifiers for job in jobs for dep in job.dependencies
    ):
        raise ValueError("Study identifiers and dependencies must be valid")
    if sum(job.stage == "main" for job in jobs) != 66:
        raise AssertionError("The primary matrix must contain 66 runs")
    return plan


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--budget-gpu-hours", type=float, default=64)
    parser.add_argument("--max-concurrent", type=int, default=6)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if (args.output_dir / "plan.json").exists():
        raise FileExistsError("A study plan already exists")
    plan = make_plan(
        manifest=args.manifest,
        output_dir=args.output_dir,
        budget_gpu_hours=args.budget_gpu_hours,
        max_concurrent=args.max_concurrent,
    )
    write_json(path=args.output_dir / "plan.json", payload=plan.model_dump(mode="json"))
    print(dict(Counter(job.stage for job in plan.jobs)))


if __name__ == "__main__":
    main()
