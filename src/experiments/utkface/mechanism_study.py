import argparse
import fcntl
import json
import os
import signal
import subprocess
import time
from collections import Counter
from contextlib import contextmanager
from pathlib import Path

from src.experiments.utkface.adaptation_config import AdaptationConfig, TrainingControl
from src.experiments.utkface.adaptation_dispatch import (
    process_identity,
    sanitized_environment,
)
from src.experiments.utkface.config import NoiseConfig, canonical_config
from src.experiments.utkface.mechanism_config import MechanismConfig, MechanismVariant
from src.experiments.utkface.run import write_json
from src.experiments.utkface.scheduler import gpu_inventory, gpu_is_free

BLOCKS = (
    {"model": 731, "head": 1731, "order": 2731, "view": 3731, "noise": 4731},
    {"model": 947, "head": 1947, "order": 2947, "view": 3947, "noise": 4947},
    {"model": 1163, "head": 2163, "order": 3163, "view": 4163, "noise": 5163},
)
VARIANTS = tuple(variant.value for variant in MechanismVariant)
HEAD_METHODS = {
    "categorical_mse": ("categorical", "mse"),
    "categorical_tailrl": ("categorical", "tailrl_population"),
    "categorical_ce": ("categorical", "ce"),
    "scalar_mse": ("scalar", "mse"),
    "scalar_mae": ("scalar", "mae"),
    "binomial_tailrl": ("categorical", "tailrl_population"),
    "binomial_ce": ("categorical", "ce"),
    "binomial_ce_free": ("categorical", "ce"),
    "binomial_ce_affine": ("categorical", "ce"),
    "binomial_tailrl_free": ("categorical", "tailrl_population"),
    "binomial_tailrl_affine": ("categorical", "tailrl_population"),
    "categorical_ce_binomial_init_fp32": ("categorical", "ce"),
    "categorical_tailrl_binomial_init_fp32": ("categorical", "tailrl_population"),
    "categorical_ce_random_fp32": ("categorical", "ce"),
    "categorical_tailrl_random_fp32": ("categorical", "tailrl_population"),
    "binomial_ce_free_fp32": ("categorical", "ce"),
    "binomial_tailrl_free_fp32": ("categorical", "tailrl_population"),
    "head_tailrl_backbone_mse": ("categorical", "tailrl_population"),
    "head_mse_backbone_tailrl": ("categorical", "mse"),
}


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def create(
    *,
    root: Path,
    manifest: Path,
    epochs: int,
    train_limit: int,
    smoke: bool,
    initialization: str,
    variants: tuple[str, ...],
) -> None:
    if (
        initialization not in {"imagenet", "scratch"}
        or not variants
        or set(variants) - set(VARIANTS)
    ):
        raise ValueError("Specify a supported initialization and mechanism variants")
    root.mkdir(parents=True, exist_ok=False)
    (root / "configs").mkdir()
    jobs = []
    for block, seeds in enumerate(BLOCKS[:1] if smoke else BLOCKS):
        for noise in (10,) if smoke else (0, 10):
            for variant in variants:
                identifier = f"b{block}_sd{noise}_{variant}"
                output = root / "runs" / identifier
                head, method = HEAD_METHODS[variant]
                base = canonical_config(manifest=manifest, output_dir=output)
                run = base.model_copy(
                    update={
                        "head": head,
                        "loss": base.loss.model_copy(update={"method": method}),
                        "noise": NoiseConfig(
                            mode="fixed" if noise else "none",
                            std_years=noise,
                            seed=seeds["noise"],
                        ),
                        "seed": seeds["model"],
                        "epochs": epochs,
                        "pretrained": initialization == "imagenet",
                        "train_limit": train_limit,
                        "learning_rate": 3e-4,
                        "weight_decay": 0.0,
                        "cpu_threads": 1,
                        "num_workers": 2,
                        "evaluate_test": True,
                        "schedule": base.schedule.model_copy(
                            update={"lr_schedule": "flat", "warmup_ratio": 0.05}
                        ),
                    }
                )
                unused = root / "unused"
                adaptation = AdaptationConfig(
                    run=run,
                    pipeline="original",
                    initialization=initialization,
                    initial_checkpoint=unused,
                    backbone_frozen=False,
                    use_feature_bank=False,
                    feature_bank=unused,
                    global_start_epoch=0,
                    view_seed=seeds["view"],
                    order_seed=seeds["order"],
                    checkpoint_every=5,
                    backbone_learning_rate=run.learning_rate,
                    schedule_kind="warmup_constant",
                    control=TrainingControl(
                        schedule_epochs=min(epochs, 50),
                        extend_checkpoint=unused,
                        extend_epoch=0,
                        keep_latest_checkpoints=1,
                        model_milestones=tuple(
                            e for e in (25, 50, 100, 200, 500) if e <= epochs
                        ),
                    ),
                )
                config = MechanismConfig(
                    adaptation=adaptation, variant=variant, head_seed=seeds["head"]
                )
                config_path = root / "configs" / f"{identifier}.json"
                write_json(path=config_path, payload=config.model_dump(mode="json"))
                jobs.append({
                    "identifier": identifier,
                    "block": block,
                    "noise_sd": noise,
                    "variant": variant,
                    "config": str(config_path),
                    "output": str(output),
                })
    write_json(
        path=root / "plan.json",
        payload={
            "schema": "utkface-mechanism-study-v1",
            "phase": "smoke" if smoke else "scientific",
            "initialization": initialization,
            "epochs": epochs,
            "warmup_epochs": min(epochs, 50) * 0.05,
            "created_at": time.time(),
            "git_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip(),
            "blocks": BLOCKS,
            "jobs": jobs,
            "primary": {
                "selection": f"Minimum clean validation mean MAE over epochs 0..{epochs}",
                "endpoint": "Clean test mean MAE at that selected checkpoint",
                "route_contrasts": [
                    "head_mse_backbone_tailrl - categorical_mse",
                    "categorical_tailrl - head_tailrl_backbone_mse",
                ],
                "score_contrasts": [
                    "categorical_ce_binomial_init_fp32 - categorical_ce_random_fp32",
                    "categorical_tailrl_binomial_init_fp32 - categorical_tailrl_random_fp32",
                    "categorical_ce_binomial_init_fp32 - binomial_ce_free_fp32",
                    "categorical_tailrl_binomial_init_fp32 - binomial_tailrl_free_fp32",
                    "binomial_ce_free - binomial_ce",
                    "binomial_tailrl_free - binomial_tailrl",
                    "binomial_ce_free - binomial_ce_affine",
                    "binomial_tailrl_free - binomial_tailrl_affine",
                    "binomial_ce_affine - binomial_ce",
                    "binomial_tailrl_affine - binomial_tailrl",
                    "binomial_ce - categorical_ce",
                    "binomial_tailrl - categorical_tailrl",
                    "scalar_mse - categorical_mse",
                    "scalar_mae - scalar_mse",
                ],
                "secondary": "Fixed final epoch MAE/RMSE/Spearman and validation trajectories",
                "uncertainty": "Three independent paired joint training/corruption blocks; common image split",
                "heldout_status": "Existing validation and test faces reused for a mechanistic follow-up",
                "direction_claim_requirement": "A routing contrast establishes a route+optimizer intervention. Feature-space direction/sample-weight and update-norm controls are required for a pure directional mechanism.",
                "common_readout_requirement": "Probe matched final-epoch encoders with one refitted readout before assigning a representation mechanism.",
                "candidate_predictions": {
                    "backbone_route": "Both within-head Tail-backbone minus MSE-backbone contrasts are positive, with common-readout support, if MSE feature learning explains the advantage.",
                    "shared_score": "Free-residual CE and/or Tail has greater error than its identically initialized gated shared-score control if the score constraint supplies useful inductive bias.",
                    "conditioning_control": "Compare a full residual against an age-affine residual: the latter adds a redundant location-update path while preserving the one-score prediction family.",
                    "scalar_loss": "Scalar MAE versus scalar MSE determines the additional effect of robust point-loss weighting inside the same score family.",
                },
                "late_learning": "For the scratch500 cohort, compare fixed-epoch validation and common-readout performance at 50 versus 500, with 100 and 200 locating the change; selected-checkpoint outcomes remain separate.",
                "prespecified_direction_followup": {
                    "space": "Backbone pooled features h; q=gradient_h(mean), v=gradient_h(Tail), c=2(mean-y)/99^2",
                    "magnitude_direction_factorial": "Cross per-example magnitudes ||c*q|| and k*||v|| with directions sign(c)*q/||q|| and v/||v||, k chosen to match batch RMS, with Tail head gradients fixed.",
                    "controls": "Common initial state, data/order/noise, separate group clipping, actual Adam update RMS, and common refitted readouts.",
                    "interpretation": "A direction claim requires survival of magnitude/sample-weight controls; route results determine which narrower explanation receives further tests.",
                },
            },
        },
    )


def completed(job: dict) -> bool:
    path = Path(job["output"]) / "result.json"
    if not path.is_file():
        return False
    result = read(path)
    expected = read(Path(job["config"]))
    if result.get("config") != expected:
        raise ValueError(f"Completed result has a different config: {path}")
    return result.get("status") == "complete"


@contextmanager
def worker_cleanup(*, active: dict):
    try:
        yield
    finally:
        for worker in active.values():
            process = worker["process"]
            if process.poll() is None and os.getpgid(process.pid) == process.pid:
                os.killpg(process.pid, signal.SIGTERM)
        for worker in active.values():
            worker["process"].wait(timeout=30)
            worker["log"].close()


def run(*, root: Path, max_gpus: int, jobs_per_gpu: int) -> None:
    if not 1 <= max_gpus <= 4 or not 1 <= jobs_per_gpu <= 8:
        raise ValueError("This study uses at most four GPUs and eight jobs per GPU")
    plan = read(root / "plan.json")
    study_lock = (root / "supervisor.lock").open("a")
    fcntl.flock(study_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    active = {}
    failures = []
    lanes = {}
    pending = [job for job in plan["jobs"] if not completed(job)]
    global_lock_path = Path("/tmp/utkface-extension-20261014/global-dispatch.lock")
    global_lock_path.parent.mkdir(parents=True, exist_ok=True)
    with global_lock_path.open("a") as dispatch_lock:
        fcntl.flock(dispatch_lock, fcntl.LOCK_EX)
        first = [g for g in gpu_inventory() if gpu_is_free(gpu=g)]
        time.sleep(10)
        stable = {g["uuid"] for g in first}
        available = [
            g for g in gpu_inventory() if g["uuid"] in stable and gpu_is_free(gpu=g)
        ]
        for gpu in available[:max_gpus]:
            lanes[gpu["uuid"]] = gpu
    if not lanes:
        raise RuntimeError("No idle GPU available for this study")
    with (
        (root / "events.jsonl").open("a", buffering=1) as events,
        worker_cleanup(active=active),
    ):
        while pending or active:
            for identifier, worker in list(active.items()):
                code = worker["process"].poll()
                if code is None:
                    continue
                worker["log"].close()
                job = worker["job"]
                successful = code == 0 and completed(job)
                record = {
                    "time": time.time(),
                    "identifier": identifier,
                    "event": "completed" if successful else "failed",
                    "returncode": code,
                    "gpu": worker["gpu"],
                }
                events.write(json.dumps(record) + "\n")
                print(json.dumps(record), flush=True)
                if not successful:
                    failures.append(record)
                del active[identifier]
            inventory = {g["uuid"]: g for g in gpu_inventory()}
            for uuid in list(lanes):
                owned_groups = {
                    worker["process"].pid
                    for worker in active.values()
                    if worker["gpu"] == uuid
                }
                foreign = [
                    pid
                    for pid in inventory[uuid]["pids"]
                    if (identity := process_identity(pid=pid)) is not None
                    and identity["process_group"] not in owned_groups
                ]
                if foreign:
                    raise RuntimeError(
                        f"Foreign processes appeared on claimed GPU: {foreign}"
                    )
                capacity = jobs_per_gpu - sum(
                    worker["gpu"] == uuid for worker in active.values()
                )
                for _ in range(min(capacity, len(pending))):
                    job = pending.pop(0)
                    output = Path(job["output"])
                    output.mkdir(parents=True, exist_ok=True)
                    log = (output / "worker.log").open("a")
                    environment = sanitized_environment(gpu_uuid=uuid)
                    environment["OMP_NUM_THREADS"] = "1"
                    process = subprocess.Popen(
                        [
                            "uv",
                            "run",
                            "--no-sync",
                            "python",
                            "-u",
                            "-m",
                            "src.experiments.utkface.mechanism_train",
                            "--config",
                            job["config"],
                        ],
                        stdin=subprocess.DEVNULL,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        env=environment,
                        start_new_session=True,
                    )
                    active[job["identifier"]] = {
                        "process": process,
                        "log": log,
                        "gpu": uuid,
                        "job": job,
                    }
                    events.write(
                        json.dumps({
                            "time": time.time(),
                            "event": "launched",
                            "identifier": job["identifier"],
                            "gpu": uuid,
                            "pid": process.pid,
                        })
                        + "\n"
                    )
            write_json(
                path=root / "state.json",
                payload={
                    "time": time.time(),
                    "status": "running" if pending or active else "complete",
                    "pending": len(pending),
                    "active": {
                        key: {"pid": row["process"].pid, "gpu": row["gpu"]}
                        for key, row in active.items()
                    },
                    "completed": sum(completed(job) for job in plan["jobs"]),
                    "failures": failures,
                },
            )
            if pending or active:
                time.sleep(5)
    if failures:
        raise RuntimeError(f"Study has {len(failures)} unsuccessful jobs")


def status(*, root: Path) -> None:
    plan = read(root / "plan.json")
    rows = []
    for job in plan["jobs"]:
        history_path = Path(job["output"]) / "history.json"
        history = json.loads(history_path.read_text()) if history_path.is_file() else []
        rows.append({
            "job": job["identifier"],
            "complete": completed(job),
            "epoch": history[-1]["epoch"] if history else 0,
        })
    print(
        json.dumps({
            "completed": sum(row["complete"] for row in rows),
            "total": len(rows),
            "epoch_counts": dict(Counter(row["epoch"] for row in rows)),
            "state": read(root / "state.json")
            if (root / "state.json").is_file()
            else {},
        })
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--create", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--train-limit", type=int, default=4096)
    parser.add_argument(
        "--initialization", choices=("imagenet", "scratch"), default="imagenet"
    )
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=VARIANTS)
    parser.add_argument("--max-gpus", type=int, default=4)
    parser.add_argument("--jobs-per-gpu", type=int, default=8)
    args = parser.parse_args()
    if args.create:
        if args.manifest is None:
            raise ValueError("Creation requires --manifest")
        create(
            root=args.root,
            manifest=args.manifest,
            epochs=args.epochs,
            train_limit=args.train_limit,
            smoke=args.smoke,
            initialization=args.initialization,
            variants=tuple(args.variants),
        )
    elif args.status:
        status(root=args.root)
    else:
        run(root=args.root, max_gpus=args.max_gpus, jobs_per_gpu=args.jobs_per_gpu)


if __name__ == "__main__":
    main()
