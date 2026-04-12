from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
RUN_SFT_SCRIPT = SCRIPT_DIR / "run_sft.py"

DEFAULT_DATASET_BASE_FOLDER = REPO_ROOT / "artifacts" / "bow-data"
DEFAULT_STUDY_ROOT = REPO_ROOT / "artifacts" / "bow-sft-overnight"


@dataclass(frozen=True, kw_only=True)
class SFTExperimentSpec:
    run_id: str
    stage: str
    snr: float
    aux_words_ratio: float
    word_decay_power: float
    lr_per_token: float
    backbone_lr_divisor: float
    num_words: int = 15
    num_samples: int = 50_000
    prompt_length: int = 128
    filter_samples_above_n_tokens: int = 384
    batch_size: int = 64
    eval_batch_size_multiple: int = 4
    train_epochs: int = 10
    model_name: str = "HuggingFaceTB/SmolLM2-360M"
    compile_model: bool = True


def float_label(value: float) -> str:
    return f"{value:.3g}".replace("-", "m").replace(".", "p").replace("+", "")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def spec_command(
    *,
    spec: SFTExperimentSpec,
    device: str,
    dataset_base_folder: Path,
    study_root: Path,
) -> list[str]:
    cmd = [
        sys.executable,
        str(RUN_SFT_SCRIPT),
        "--run-id",
        spec.run_id,
        "--stage",
        spec.stage,
        "--device",
        device,
        "--dataset-base-folder",
        str(dataset_base_folder),
        "--study-root",
        str(study_root),
        "--snr",
        str(spec.snr),
        "--aux-words-ratio",
        str(spec.aux_words_ratio),
        "--word-decay-power",
        str(spec.word_decay_power),
        "--lr-per-token",
        str(spec.lr_per_token),
        "--backbone-lr-divisor",
        str(spec.backbone_lr_divisor),
        "--num-words",
        str(spec.num_words),
        "--num-samples",
        str(spec.num_samples),
        "--prompt-length",
        str(spec.prompt_length),
        "--filter-samples-above-n-tokens",
        str(spec.filter_samples_above_n_tokens),
        "--batch-size",
        str(spec.batch_size),
        "--eval-batch-size-multiple",
        str(spec.eval_batch_size_multiple),
        "--train-epochs",
        str(spec.train_epochs),
        "--model-name",
        spec.model_name,
    ]
    if not spec.compile_model:
        cmd.append("--no-compile-model")
    return cmd


def write_plan(*, specs: list[SFTExperimentSpec], study_root: Path, stage: str) -> Path:
    path = study_root / "plans" / f"{stage}.json"
    write_json(path, [asdict(spec) for spec in specs])
    return path


def assert_unique_run_ids(specs: list[SFTExperimentSpec]) -> None:
    run_ids = [spec.run_id for spec in specs]
    if len(run_ids) != len(set(run_ids)):
        raise ValueError("Duplicate run_id in sweep plan")


def shard_specs(
    *,
    specs: list[SFTExperimentSpec],
    num_shards: int,
    shard_index: int,
) -> list[SFTExperimentSpec]:
    if num_shards <= 0:
        raise ValueError("num_shards must be positive")
    if not 0 <= shard_index < num_shards:
        raise ValueError("shard_index must be in [0, num_shards)")
    return [spec for idx, spec in enumerate(specs) if idx % num_shards == shard_index]


def run_specs_serial(
    *,
    specs: list[SFTExperimentSpec],
    device: str,
    dataset_base_folder: Path,
    study_root: Path,
    dry_run: bool,
    num_shards: int,
    shard_index: int,
) -> None:
    if not specs:
        raise ValueError("No experiments requested")
    assert_unique_run_ids(specs)

    plan_path = write_plan(specs=specs, study_root=study_root, stage=specs[0].stage)
    print(f"Wrote plan: {plan_path}")

    selected_specs = shard_specs(
        specs=specs,
        num_shards=num_shards,
        shard_index=shard_index,
    )
    print(
        f"Device {device}: {len(selected_specs)} of {len(specs)} runs "
        f"(shard {shard_index}/{num_shards})"
    )

    for spec in selected_specs:
        cmd = spec_command(
            spec=spec,
            device=device,
            dataset_base_folder=dataset_base_folder,
            study_root=study_root,
        )
        print("  " + " ".join(cmd))

    if dry_run:
        return

    log_root = study_root / "logs" / specs[0].stage
    log_root.mkdir(parents=True, exist_ok=True)

    for spec in selected_specs:
        cmd = spec_command(
            spec=spec,
            device=device,
            dataset_base_folder=dataset_base_folder,
            study_root=study_root,
        )
        log_path = log_root / f"{spec.run_id}.log"
        print(f"[{device}] starting {spec.run_id}; log={log_path}")
        env = os.environ.copy()
        existing_pythonpath = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = (
            str(REPO_ROOT) + (":" + existing_pythonpath if existing_pythonpath else "")
        )
        with log_path.open("w") as log_file:
            proc = subprocess.Popen(
                cmd,
                cwd=REPO_ROOT,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                text=True,
                env=env,
            )
            returncode = proc.wait()
        if returncode != 0:
            raise SystemExit(f"{spec.run_id} failed with return code {returncode}")
        print(f"[{device}] finished {spec.run_id}")
