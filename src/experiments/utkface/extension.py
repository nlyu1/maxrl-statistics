import argparse
import json
from datetime import datetime
from pathlib import Path

from src.experiments.utkface.config import RunConfig
from src.experiments.utkface.coordination import CoordinationConfig
from src.experiments.utkface.run import write_json
from src.experiments.utkface.study import REDUCED_VARIANTS, StudyJob, StudyPlan


def make_extension(*, base: StudyPlan, output_dir: Path) -> StudyPlan:
    originals = {job.identifier: job for job in base.jobs}
    jobs = []

    def add(
        *, source: StudyJob, identifier: str, stage: str, priority: int, changes: dict
    ) -> None:
        config = RunConfig.model_validate({
            **source.config.model_dump(),
            **changes,
            "output_dir": output_dir / "runs" / identifier,
        })
        jobs.append(
            StudyJob(**{
                **source.model_dump(),
                "identifier": identifier,
                "stage": stage,
                "priority": priority,
                "optional": False,
                "config": config,
                "dependencies": source.tune_from,
            })
        )

    for method, scale in REDUCED_VARIANTS:
        variant = method + (f"_s{scale:g}" if method.startswith("stat_maxrl") else "")
        clean = originals[f"subset_4096_{variant}_seed42"]
        for seed in (43, 44):
            add(
                source=clean,
                identifier=f"extra_subset4096_{variant}_seed{seed}",
                stage="subset_replication",
                priority=25,
                changes={"seed": seed},
            )
        small = originals[f"subset_1024_{variant}_seed42"]
        add(
            source=small,
            identifier=f"extra_steps1024_{variant}_seed42",
            stage="matched_steps",
            priority=35,
            changes={"epochs": 200},
        )
        for mode in ("fixed", "fresh"):
            source = originals[f"noise_{mode}_s5_{variant}_seed42"]
            for noise_seed in (2027, 2028):
                add(
                    source=source,
                    identifier=f"extra_noise_{mode}_s5_{variant}_noise{noise_seed}",
                    stage="noise_realization",
                    priority=15,
                    changes={
                        "noise": {
                            **source.config.noise.model_dump(),
                            "seed": noise_seed,
                        }
                    },
                )
    return StudyPlan(
        output_dir=output_dir,
        data_fingerprint=base.data_fingerprint,
        training_source_sha256=base.training_source_sha256,
        budget_gpu_hours=24,
        max_concurrent=6,
        jobs=tuple(jobs),
    )


def make_coordination(
    *,
    base: StudyPlan,
    extension: StudyPlan,
    base_path: Path,
    extension_path: Path,
    deadline: float,
) -> CoordinationConfig:
    priorities = {}
    promoted = []
    baseline = {"ce", "mae", "mse", "tailrl_population"}
    for job in base.jobs:
        method = job.config.loss.method
        if job.stage == "main":
            priorities[job.identifier] = 0
        elif job.stage == "scalar":
            priorities[job.identifier] = 1 if job.config.seed == 42 else 40
        elif job.stage == "subset_tuning":
            priorities[job.identifier] = (
                20
                if job.config.train_limit == 1024
                else 10
                if method in baseline
                else 30
            )
        elif job.stage == "subset":
            priorities[job.identifier] = 9
        elif job.stage == "flat_tuning":
            priorities[job.identifier] = 12 if method in baseline else 32
        elif job.stage == "flat":
            priorities[job.identifier] = 11
        elif job.stage == "noise" and not job.optional:
            priorities[job.identifier] = (
                11 if job.config.noise.std_years == 5 else 14
            ) + (0 if method in baseline else 20)
        elif job.stage == "digits_tuning" and method in {
            "ce",
            "mse",
            "tailrl_population",
        }:
            promoted.append(job.identifier)
            priorities[job.identifier] = 13
        elif (
            job.stage == "digits"
            and job.config.train_limit == 4096
            and method in {"ce", "mse", "tailrl_population"}
        ):
            promoted.append(job.identifier)
            priorities[job.identifier] = 12
    return CoordinationConfig(
        lock_path=extension.output_dir / "global-dispatch.lock",
        peer_plans=(base_path, extension_path),
        max_concurrent=6,
        deadlines={str(extension.output_dir): deadline},
        priorities=priorities,
        promoted=tuple(promoted),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-plan", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--deadline", required=True)
    args = parser.parse_args()
    base = StudyPlan.model_validate_json(args.base_plan.read_text())
    plan = make_extension(base=base, output_dir=args.output_dir)
    path = args.output_dir / "plan.json"
    if path.exists():
        raise FileExistsError(path)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(path=path, payload=plan.model_dump(mode="json"))
    control = make_coordination(
        base=base,
        extension=plan,
        base_path=args.base_plan,
        extension_path=path,
        deadline=datetime.fromisoformat(args.deadline).timestamp(),
    )
    write_json(
        path=args.output_dir / "coordination.json",
        payload=control.model_dump(mode="json"),
    )
    print(
        json.dumps({
            "jobs": len(plan.jobs),
            "budget_gpu_hours": 24,
            "deadline": args.deadline,
            "global_gpu_cap": 6,
        })
    )


if __name__ == "__main__":
    main()
