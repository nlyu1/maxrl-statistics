import argparse
import json
from pathlib import Path

from src.experiments.gpu_pool import GPUPool, Job
from src.experiments.utkface.config import RunConfig, canonical_config
from src.experiments.utkface.run import write_json

CORE_METHODS = (
    "mse",
    "ce",
    "grpo_mae_population",
    "grpo_mse_population",
    "rloo_mae_population",
    "rloo_mse_population",
    "tailrl_population",
    "stat_maxrl_mae_population",
    "stat_maxrl_mse_population",
    "binary_maxrl",
    "tailrl",
)


def run_sweep(
    *, configs: list[RunConfig], gpu_ids: list[int], output_dir: Path
) -> None:
    plan_dir = output_dir / "plans"
    plan_dir.mkdir(parents=True, exist_ok=True)
    jobs = []
    for config in configs:
        label = config.output_dir.name
        plan = plan_dir / f"{label}.json"
        write_json(path=plan, payload=config.model_dump(mode="json"))
        jobs.append(
            Job(
                label=label,
                cmd_template=[
                    "uv",
                    "run",
                    "--python",
                    "3.12",
                    "python",
                    "-m",
                    "src.experiments.utkface.run",
                    "--config",
                    str(plan),
                    "--manifest",
                    str(config.manifest),
                    "--output-dir",
                    str(config.output_dir),
                    "--device",
                    "{device}",
                ],
            )
        )
    results = GPUPool(device_ids=gpu_ids).run(
        jobs, log_dir=output_dir / "logs", fail_fast=True
    )
    failures = [result.job.label for result in results if result.returncode != 0]
    if failures or len(results) != len(jobs):
        raise RuntimeError(
            f"Incomplete sweep: {len(results)}/{len(jobs)}; failures={failures}"
        )


def select_pilot(*, pilot_dir: Path, destination: Path) -> list[dict]:
    candidates = [
        json.loads(path.read_text()) for path in sorted(pilot_dir.glob("*/result.json"))
    ]
    selected = []
    for method in CORE_METHODS:
        runs = [run for run in candidates if run["config"]["loss"]["method"] == method]
        if not runs:
            raise ValueError(f"No completed pilot runs for {method}")
        best = min(runs, key=lambda run: run["validation"]["mae_mean"])
        selected.append({
            "method": method,
            "learning_rate": best["config"]["learning_rate"],
            "kernel_scale_years": best["config"]["loss"]["kernel_scale_years"],
            "pilot_val_mae": best["validation"]["mae_mean"],
            "pilot_output": best["config"]["output_dir"],
        })
    write_json(path=destination, payload=selected)
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("pilot", "main", "select"), required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gpu-ids", type=int, nargs="+", required=True)
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.stage == "select":
        if args.selection is None:
            raise ValueError("--selection is required")
        print(
            json.dumps(
                select_pilot(pilot_dir=args.output_dir, destination=args.selection),
                indent=2,
            )
        )
        return
    base = canonical_config(manifest=args.manifest, output_dir=args.output_dir)
    configs = []
    if args.stage == "pilot":
        for method in CORE_METHODS:
            for scale in (
                (2.0, 5.0, 10.0) if method.startswith("stat_maxrl") else (5.0,)
            ):
                for lr in (1e-4, 3e-4, 1e-3):
                    label = f"{method}_s{scale:g}_lr{lr:g}"
                    payload = {
                        **base.model_dump(),
                        "epochs": 3,
                        "train_limit": 4096,
                        "eval_limit": 512,
                        "save_checkpoint": False,
                        "output_dir": args.output_dir / label,
                        "learning_rate": lr,
                        "loss": {
                            **base.loss.model_dump(),
                            "method": method,
                            "kernel_scale_years": scale,
                        },
                    }
                    configs.append(RunConfig.model_validate(payload))
    else:
        if args.selection is None:
            raise ValueError("--selection is required for the main sweep")
        choices = json.loads(args.selection.read_text())
        if {choice["method"] for choice in choices} != set(CORE_METHODS):
            raise ValueError("Selection must contain every comparison method")
        for choice in choices:
            for seed in args.seeds:
                label = f"{choice['method']}_seed{seed}"
                payload = {
                    **base.model_dump(),
                    "epochs": args.epochs,
                    "seed": seed,
                    "evaluate_test": True,
                    "output_dir": args.output_dir / label,
                    "learning_rate": choice["learning_rate"],
                    "loss": {
                        **base.loss.model_dump(),
                        "method": choice["method"],
                        "kernel_scale_years": choice["kernel_scale_years"],
                    },
                }
                configs.append(RunConfig.model_validate(payload))
    run_sweep(configs=configs, gpu_ids=args.gpu_ids, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
