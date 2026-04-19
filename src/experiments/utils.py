"""Shared helpers for experiment single-run scripts and sweep orchestrators."""

import gc
import random
import shutil
from pathlib import Path

import numpy as np
import torch

from src.experiments.bag_of_words.analysis import BagOfWordsAnalysisConfig


def set_seeds(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def cleanup_cuda(device: torch.device) -> None:
    gc.collect()
    if device.type != "cuda":
        return
    with torch.cuda.device(device):
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    torch.compiler.reset()


def prepare_study_folder(*, study_folder: Path, tag: str) -> bool:
    """Return True if the study should run; False if it's already complete.

    Wipes any partial artifacts so a retry starts clean.
    """
    if BagOfWordsAnalysisConfig.is_study_complete(study_folder):
        print(f"!!! {tag} already complete, skipping ({study_folder})")
        return False
    if BagOfWordsAnalysisConfig.has_study_started(study_folder):
        print(f"=== {tag} partial → wiping {study_folder} ===")
        shutil.rmtree(study_folder)
    return True
