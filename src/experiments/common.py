from src.config.base import BaseConfig


class GRPOConfig(BaseConfig):
    """
    Online-GRPO
    """

    num_rollout_samples: int
    num_rollout_steps: int


class MaxRLConfig(BaseConfig):
    """
    Placeholder for now
    """

    num_rollout_samples: int
    num_rollout_steps: int
    convergence_radius: float
