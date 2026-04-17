"""
SNR sweep over the bag-of-words SL task.

Usage:
    uv run python experiments/bow/sl-snr/sweep.py DEVICE_ID

device_id 0 runs the first half of the shuffled SNR list;
device_id 1 runs the second half (fixed seed=42, no overlap).
Artifacts → artifacts/bow-sl-snr-sweep/
Logs      → artifacts/bow-sl-snr-sweep/logs/device_<id>.log
"""

import gc
import os
import random
import shutil
import subprocess
import sys
from pathlib import Path

import click
import torch
from tqdm import tqdm

from src import chdir_repo_base, get_repo_base

chdir_repo_base()
repo_root = get_repo_base()

from src.experiments.bag_of_words.analysis import BagOfWordsAnalysisConfig
from src.experiments.bag_of_words.sft import BagOfWordsSFTConfig

SNR_LIST = [
    0.010, 0.011, 0.013, 0.014, 0.016, 0.018, 0.020, 0.023, 0.026, 0.029,
    0.033, 0.037, 0.041, 0.046, 0.052, 0.059, 0.066, 0.074, 0.084, 0.094,
    0.106, 0.119, 0.134, 0.151, 0.170, 0.191, 0.215, 0.242, 0.273, 0.307,
    0.346, 0.389, 0.438, 0.492, 0.554, 0.624, 0.702, 0.790, 0.889, 1.000,
]


MODEL_VARIANTS: dict[str, tuple[str, str]] = {
    # flag name → (model_name, folder suffix)
    "qwen":  ("Qwen/Qwen2.5-0.5B",  "-qwen"),
    "qwen3": ("Qwen/Qwen3-0.6B", "-qwen3"),
}

TRAIN_EPOCHS = 15
AUX_WORDS_RATIO = 0.5


def get_study_folder(
    *,
    study_base: Path,
    snr: float,
    num_words: int = 7,
    prompt_length: int = 128,
    word_decay_power: float = 0.0,
    aux_words_ratio: float = AUX_WORDS_RATIO,
) -> Path:
    return study_base / (
        f"{num_words}-words_snr-{snr}_len-{prompt_length}"
        f"_pow-{word_decay_power}_ar-{aux_words_ratio}"
    )


def cleanup_cuda_between_studies(*, device: torch.device) -> None:
    gc.collect()
    if device.type != "cuda":
        return
    with torch.cuda.device(device):
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    torch.compiler.reset()


@click.command()
@click.argument("device_id", type=int)
@click.option("--qwen", is_flag=True, help="Use Qwen/Qwen2.5-0.5B instead of SmolLM2-360M")
@click.option("--qwen3", is_flag=True, help="Use Qwen/Qwen3-0.6B instead of SmolLM2-360M")
@click.option("--train-epochs", type=int, default=TRAIN_EPOCHS, show_default=True)
@click.option(
    "--overwrite/--resume",
    default=True,
    show_default=True,
    help="Replace existing study artifacts before rerunning each SNR.",
)
def main(
    device_id: int,
    qwen: bool,
    qwen3: bool,
    train_epochs: int,
    overwrite: bool,
) -> None:
    # ── resolve folders and model ─────────────────────────────────────────────
    selected = [k for k, v in [("qwen", qwen), ("qwen3", qwen3)] if v]
    if len(selected) > 1:
        raise click.UsageError("At most one model flag may be specified")
    if selected:
        model_name, suffix = MODEL_VARIANTS[selected[0]]
    else:
        model_name, suffix = "HuggingFaceTB/SmolLM2-360M", ""
    study_base = repo_root / "artifacts" / f"bow-sl-snr-sweep{suffix}"
    data_base = repo_root / "artifacts" / f"bow-data{suffix}"

    # ── tee stdout+stderr into a log file ──────────────────────────────────────
    log_path = study_base / "logs" / f"device_{device_id}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    tee_args = ["tee", str(log_path)] if overwrite else ["tee", "-a", str(log_path)]
    tee = subprocess.Popen(tee_args, stdin=subprocess.PIPE)
    os.dup2(tee.stdin.fileno(), sys.stdout.fileno())
    os.dup2(tee.stdin.fileno(), sys.stderr.fileno())
    tee.stdin.close()

    # ── assign half the shuffled list to this device ───────────────────────────
    snrs = SNR_LIST.copy()
    random.seed(42)
    random.shuffle(snrs)
    half    = len(snrs) // 2
    my_snrs = snrs[:half] if device_id % 2 == 0 else snrs[half:]

    device = torch.device(f"cuda:{device_id}")
    print(f"device={device}  {len(my_snrs)} experiments: {[f'{s:.3f}' for s in my_snrs]}\n")

    for snr in tqdm(my_snrs, desc=f"SNR sweep (device {device_id})", position=0):
        study_folder = get_study_folder(study_base=study_base, snr=snr)
        if overwrite and study_folder.exists():
            tqdm.write(f"=== SNR={snr:.4f} removing old artifacts ===")
            shutil.rmtree(study_folder)

        config = BagOfWordsSFTConfig.get_canonical(
            dataset_base_folder=data_base,
            study_base_folder=study_base,
            snr=snr,
            aux_words_ratio=AUX_WORDS_RATIO,
            model_name=model_name,
            train_epochs=train_epochs,
        )
        if BagOfWordsAnalysisConfig.is_study_complete(config.study_folder):
            tqdm.write(f"=== SNR={snr:.4f} already complete, skipping ===")
            continue
        tqdm.write(f"\n=== SNR={snr:.4f}  folder={config.study_folder.name} ===")
        state = None
        try:
            state = config.initialize(device=device)
            state.run_training()
        finally:
            del state
            cleanup_cuda_between_studies(device=device)

    print("\nDone.")


if __name__ == "__main__":
    main()
