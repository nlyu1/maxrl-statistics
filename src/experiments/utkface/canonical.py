from pathlib import Path

from src.experiments.utkface.config import (
    DataConfig,
    Method,
    ObjectiveConfig,
    TrainingConfig,
    UTKFaceConfig,
)
from src.model.optimizer import LRScheduleConfig


def get_canonical_config(
    *,
    method: Method,
    data_directory: Path,
    output_directory: Path,
    seed: int,
) -> UTKFaceConfig:
    return UTKFaceConfig(
        objective=ObjectiveConfig(
            method=method,
            reward="binary"
            if method in ("maxrl_binary", "maxrl_binary_population")
            else "mae",
            temperature_years=5.0,
            num_rollouts=64,
            degree=64,
            survival_floor=0.0,
            standard_deviation_epsilon=1e-8,
            standard_deviation_floor=1e-4,
        ),
        data=DataConfig(
            directory=data_directory,
            input_size=224,
            split_seed=42,
            max_train_samples=0,
            max_eval_samples=0,
            num_workers=4,
        ),
        training=TrainingConfig(
            epochs=20,
            global_batch_size=128,
            eval_batch_size=256,
            learning_rate=1e-3,
            weight_decay=1e-4,
            momentum=0.9,
            gradient_clip=10.0,
            schedule=LRScheduleConfig(
                lr_schedule="cosine",
                warmup_ratio=0.05,
                lr_min_ratio=0.0,
            ),
            precision="bf16",
            device="cuda",
            checkpoint_epochs=(1, 10, 20),
        ),
        seed=seed,
        pretrained=True,
        output_directory=output_directory,
    )
