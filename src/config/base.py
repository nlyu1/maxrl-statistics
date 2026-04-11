"""Common config helpers for visualization and immutable replacement."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict

from src.config.replace import replace_in_nested_config
from src.config.visualization import visualize_nested_config


class ConfigMethodsMixin:
    """Shared helper methods for config-like objects."""

    def visualize(self, **kwargs: Any):
        """Return a notebook-displayable Solara tree for this config object."""
        return visualize_nested_config(self, **kwargs)

    def replace(self, *, path: str, replacement: Any):
        """Return an immutable copy with ``path`` replaced."""
        return replace_in_nested_config(self, path=path, replacement=replacement)


class BaseConfig(BaseModel, ConfigMethodsMixin):
    """Common Pydantic config base with visualization and replacement helpers."""

    model_config = ConfigDict(
        arbitrary_types_allowed=True, populate_by_name=True, extra="forbid", frozen=True
    )


__all__ = ["BaseConfig", "ConfigMethodsMixin"]
