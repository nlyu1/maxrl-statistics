"""
Coarse GRPO sweep over (corr, num_rollouts_per_sample) on the bag-of-words task.

Usage:
    uv run python experiments/bow/grpo-corr/coarse_sweep.py DEVICE_ID

device_id 0 runs the first half of the shuffled combo list; device_id 1 runs
the second half. The shuffle seed is chosen so that
(corr=0.01, num_rollouts=4096) is index 0 — stress-tests the largest rollout
group at the smallest correlation first.

Artifacts -> artifacts/bow-grpo-coarse-sweep/rollouts-<N>/<dataset-name>/
Logs      -> artifacts/bow-grpo-coarse-sweep/logs/device_<id>.log
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

from src.experiments.bag_of_words.analysis import BagOfWordsAnalysisConfig  # noqa: E402
from src.experiments.bag_of_words.grpo import BagOfWordsGRPOConfig  # noqa: E402

CORR_LIST = [0.01, 0.1, 1.0]
NUM_ROLLOUTS_LIST = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
COMBOS = [(c, r) for c in CORR_LIST for r in NUM_ROLLOUTS_LIST]

SHUFFLE_SEED = 8  # puts (corr=0.01, num_rollouts=4096) at index 0

TRAIN_EPOCHS = 20
AUX_WORDS_RATIO = 0.5
GAUSSIAN_STDEV = 1.0

STUDY_BASE = repo_root / "artifacts" / "bow-grpo-coarse-sweep"
DATA_BASE = repo_root / "artifacts" / "bow-data"


def get_study_base_for_rollouts(num_rollouts: int) -> Path:
    """Sub-namespace under STUDY_BASE so different rollout counts don't collide."""
    return STUDY_BASE / f"rollouts-{num_rollouts}"


def get_study_folder(
    *,
    corr: float,
    num_rollouts: int,
    num_words: int = 7,
    prompt_length: int = 128,
    word_decay_power: float = 1.0,
    aux_words_ratio: float = AUX_WORDS_RATIO,
) -> Path:
    dataset_name = (
        f"{num_words}-words_corr-{corr}_len-{prompt_length}"
        f"_pow-{word_decay_power}_ar-{aux_words_ratio}"
    )
    return get_study_base_for_rollouts(num_rollouts) / dataset_name


def shuffled_combos() -> list[tuple[float, int]]:
    combos = COMBOS.copy()
    random.Random(SHUFFLE_SEED).shuffle(combos)
    return combos


def cleanup_cuda(device: torch.device) -> None:
    gc.collect()
    if device.type != "cuda":
        return
    with torch.cuda.device(device):
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    torch.compiler.reset()


@click.command()
@click.argument("device_id", type=int)
@click.option("--train-epochs", type=int, default=TRAIN_EPOCHS, show_default=True)
@click.option(
    "--overwrite/--resume",
    default=False,
    show_default=True,
    help="If --overwrite, replace existing study artifacts even when complete. "
    "Default --resume skips completed studies; partial runs are wiped and rerun.",
)
def main(device_id: int, train_epochs: int, overwrite: bool) -> None:
    log_path = STUDY_BASE / "logs" / f"device_{device_id}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    tee_args = ["tee", str(log_path)] if overwrite else ["tee", "-a", str(log_path)]
    tee = subprocess.Popen(tee_args, stdin=subprocess.PIPE)
    os.dup2(tee.stdin.fileno(), sys.stdout.fileno())
    os.dup2(tee.stdin.fileno(), sys.stderr.fileno())
    tee.stdin.close()

    combos = shuffled_combos()
    half = len(combos) // 2
    my_combos = combos[:half] if device_id % 2 == 0 else combos[half:]

    device = torch.device(f"cuda:{device_id}")
    print(
        f"device={device}  {len(my_combos)} experiments: "
        f"{[(f'{c:.3f}', r) for c, r in my_combos]}\n"
    )

    for corr, num_rollouts in tqdm(
        my_combos, desc=f"grpo coarse sweep (device {device_id})", position=0
    ):
        tag = f"corr={corr:.4f} rollouts={num_rollouts}"
        study_folder = get_study_folder(corr=corr, num_rollouts=num_rollouts)

        if study_folder.exists():
            if (
                not overwrite
                and BagOfWordsAnalysisConfig.is_study_complete(study_folder)
            ):
                tqdm.write(f"=== {tag} already complete, skipping ===")
                continue
            tqdm.write(f"=== {tag} removing old artifacts ===")
            shutil.rmtree(study_folder)

        config = BagOfWordsGRPOConfig.get_canonical(
            dataset_base_folder=DATA_BASE,
            study_base_folder=get_study_base_for_rollouts(num_rollouts),
            corr=corr,
            aux_words_ratio=AUX_WORDS_RATIO,
            train_epochs=train_epochs,
            num_rollouts_per_sample=num_rollouts,
            gaussian_stdev=GAUSSIAN_STDEV,
        )
        assert config.study_folder == study_folder, (
            f"study_folder mismatch: {config.study_folder} vs {study_folder}"
        )
        tqdm.write(f"\n=== {tag}  folder={study_folder} ===")

        state = None
        try:
            state = config.initialize(device=device)
            state.run_training()
        finally:
            del state
            cleanup_cuda(device)

    print("\nDone.")


if __name__ == "__main__":
    main()
