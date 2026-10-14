from pathlib import Path

from pydantic import Field

from src.config.base import BaseConfig


class CoordinationConfig(BaseConfig):
    lock_path: Path
    peer_plans: tuple[Path, ...]
    max_concurrent: int = Field(ge=1, le=8)
    enforce_gpu_hour_budget: bool
    held_jobs: tuple[str, ...]
    deadlines: dict[str, float]
    priorities: dict[str, int]
    promoted: tuple[str, ...]


def canonical_coordination(*, root: Path) -> CoordinationConfig:
    return CoordinationConfig(
        lock_path=root / "dispatch.lock",
        peer_plans=(),
        max_concurrent=8,
        enforce_gpu_hour_budget=False,
        held_jobs=(),
        deadlines={},
        priorities={},
        promoted=(),
    )
