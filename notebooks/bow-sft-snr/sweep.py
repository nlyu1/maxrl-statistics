"""
SNR sweep over the bag-of-words SFT task.

Usage:
    python notebooks/scripts/bow-sft-snr-sweep.py DEVICE_ID

device_id 0 runs the first half of the shuffled SNR list;
device_id 1 runs the second half (fixed seed=42, no overlap).
Artifacts → artifacts/bow-sft-snr-sweep/
Logs      → artifacts/bow-sft-snr-sweep/logs/device_<id>.log
"""

import os
import random
import subprocess
import sys
from pathlib import Path

import click
import torch
from tqdm import tqdm

repo_root = Path(__file__).resolve().parents[2]
os.chdir(repo_root)
sys.path.insert(0, str(repo_root))

from src.experiments.bag_of_words.analysis import BagOfWordsAnalysisConfig  # noqa: E402
from src.experiments.bag_of_words.sft import BagOfWordsSFTConfig  # noqa: E402

STUDY_BASE = repo_root / "artifacts" / "bow-sft-snr-sweep"
DATA_BASE  = repo_root / "artifacts" / "bow-data"

SNR_LIST = [
    0.010, 0.011, 0.013, 0.014, 0.016, 0.018, 0.020, 0.023, 0.026, 0.029,
    0.033, 0.037, 0.041, 0.046, 0.052, 0.059, 0.066, 0.074, 0.084, 0.094,
    0.106, 0.119, 0.134, 0.151, 0.170, 0.191, 0.215, 0.242, 0.273, 0.307,
    0.346, 0.389, 0.438, 0.492, 0.554, 0.624, 0.702, 0.790, 0.889, 1.000,
]


@click.command()
@click.argument("device_id", type=int)
def main(device_id: int) -> None:
    # ── tee stdout+stderr into a log file ──────────────────────────────────────
    log_path = STUDY_BASE / "logs" / f"device_{device_id}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    tee = subprocess.Popen(["tee", "-a", str(log_path)], stdin=subprocess.PIPE)
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
        config = BagOfWordsSFTConfig.get_canonical(
            dataset_base_folder=DATA_BASE,
            study_base_folder=STUDY_BASE,
            snr=snr,
            aux_words_ratio=0.5,
        )
        if BagOfWordsAnalysisConfig.is_study_complete(config.study_folder):
            tqdm.write(f"=== SNR={snr:.4f} already complete, skipping ===")
            continue
        tqdm.write(f"\n=== SNR={snr:.4f}  folder={config.study_folder.name} ===")
        state = config.initialize(device=device)
        state.run_training()

    print("\nDone.")


if __name__ == "__main__":
    main()
