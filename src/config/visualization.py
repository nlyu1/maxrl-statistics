"""Notebook and Solara tree visualization helpers for nested configs."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from dataclasses import fields
from dataclasses import is_dataclass
from datetime import date
from datetime import datetime
from datetime import time
from html import escape
from typing import Any
from typing import Optional

from pydantic import BaseModel

from src.config.replace import child_path

_FIELD_COLOR = "#000000"
_VALUE_COLOR = "#0B2F6B"
_CLASS_COLOR = "#6B1D1D"
_COUNT_COLOR = "#1E5B2A"
_PIPE_COLOR = "#6A6A6A"

_NON_COMPLEX_TYPE_NAMES = {
    "bool",
    "bytes",
    "complex",
    "date",
    "datetime",
    "dict",
    "float",
    "frozenset",
    "int",
    "list",
    "NoneType",
    "OrderedDict",
    "set",
    "str",
    "time",
    "tuple",
}


@dataclass(slots=True)
class _TreeNode:
    name: str
    path: str
    type_name: str
    value_repr: Optional[str]
    children: list["_TreeNode"]
    total_children: int = 0

    @property
    def is_leaf(self) -> bool:
        return len(self.children) == 0


def _iter_children(value: Any) -> list[tuple[str, Any]]:
    if isinstance(value, BaseModel):
        return [
            (field_name, getattr(value, field_name))
            for field_name in type(value).model_fields
        ]

    if is_dataclass(value) and not isinstance(value, type):
        return [(field.name, getattr(value, field.name)) for field in fields(value)]

    if isinstance(value, Mapping):
        return [(str(key), child) for key, child in value.items()]

    if isinstance(value, list | tuple):
        return [(f"[{index}]", child) for index, child in enumerate(value)]

    return []


def _truncate_repr(text: str, *, max_repr_len: int) -> str:
    normalized = text.replace("\n", "\\n")
    if len(normalized) <= max_repr_len:
        return normalized
    return f"{normalized[: max_repr_len - 3]}..."


def _format_leaf_value(value: Any, *, max_repr_len: int) -> str:
    if isinstance(value, str):
        text = repr(value)
    elif isinstance(value, date | datetime | time):
        text = value.isoformat()
    else:
        text = repr(value)
    return _truncate_repr(text, max_repr_len=max_repr_len)


def _build_tree(
    *,
    name: str,
    path: str,
    value: Any,
    max_repr_len: int,
    max_items_per_container: int,
    max_depth: Optional[int],
    depth: int = 0,
    seen: Optional[set[int]] = None,
) -> _TreeNode:
    seen = seen if seen is not None else set()
    children = _iter_children(value)

    if len(children) == 0:
        return _TreeNode(
            name=name,
            path=path,
            type_name=type(value).__name__,
            value_repr=_format_leaf_value(value, max_repr_len=max_repr_len),
            children=[],
        )

    if max_depth is not None and depth >= max_depth:
        return _TreeNode(
            name=name,
            path=path,
            type_name=type(value).__name__,
            value_repr=f"<{type(value).__name__} depth limit reached>",
            children=[],
        )

    value_id = id(value)
    if value_id in seen:
        return _TreeNode(
            name=name,
            path=path,
            type_name=type(value).__name__,
            value_repr=f"<{type(value).__name__} recursive reference>",
            children=[],
        )

    seen.add(value_id)
    total_children = len(children)
    kept_children = children[:max_items_per_container]

    child_nodes = [
        _build_tree(
            name=child_name,
            path=child_path(path, child_name),
            value=child_value,
            max_repr_len=max_repr_len,
            max_items_per_container=max_items_per_container,
            max_depth=max_depth,
            depth=depth + 1,
            seen=seen,
        )
        for child_name, child_value in kept_children
    ]

    if total_children > len(kept_children):
        omitted = total_children - len(kept_children)
        child_nodes.append(
            _TreeNode(
                name="...",
                path=child_path(path, "..."),
                type_name="ellipsis",
                value_repr=f"{omitted} additional item(s) hidden",
                children=[],
            )
        )

    seen.remove(value_id)

    return _TreeNode(
        name=name,
        path=path,
        type_name=type(value).__name__,
        value_repr=None,
        children=child_nodes,
        total_children=total_children,
    )


def _is_complicated_type_name(type_name: str) -> bool:
    return type_name not in _NON_COMPLEX_TYPE_NAMES


def _pipe_html() -> str:
    return f"<span style='color:{_PIPE_COLOR};'> | </span>"


def _segment_html(text: str, *, color: str, bold: bool = False) -> str:
    weight = "700" if bold else "500"
    return f"<span style='color:{color};font-weight:{weight};'>{escape(text)}</span>"


def _summary_html(node: _TreeNode) -> str:
    segments = [_segment_html(node.name, color=_FIELD_COLOR, bold=True)]

    if _is_complicated_type_name(node.type_name):
        segments.append(_segment_html(node.type_name, color=_CLASS_COLOR))

    suffix = "item" if node.total_children == 1 else "items"
    segments.append(
        _segment_html(f"{node.total_children:,} {suffix}", color=_COUNT_COLOR)
    )
    return _pipe_html().join(segments)


def _leaf_html(node: _TreeNode) -> str:
    segments = [_segment_html(node.name, color=_FIELD_COLOR, bold=True)]

    if _is_complicated_type_name(node.type_name):
        segments.append(_segment_html(node.type_name, color=_CLASS_COLOR))

    segments.append(_segment_html(node.value_repr or "", color=_VALUE_COLOR))
    return _pipe_html().join(segments)


def _tooltip_path(node: _TreeNode) -> str:
    return node.path or "(root)"


def _leaf_card_style() -> dict[str, str]:
    return {
        "background": "#fcfcfc",
        "border": "1px solid #d8d8d8",
        "border-radius": "6px",
        "margin": "1px 0",
        "padding": "3px 8px",
    }


def _container_card_style() -> dict[str, str]:
    return {
        "background": "#fcfcfc",
        "border": "1px solid #d8d8d8",
        "border-radius": "6px",
        "margin": "1px 0",
        "overflow": "hidden",
        "padding": "0px",
    }


def _render_leaf_solara(node: _TreeNode) -> None:
    import solara

    with solara.Div(
        style=_leaf_card_style(),
        attributes={"title": _tooltip_path(node)},
    ):
        solara.HTML(
            tag="div",
            unsafe_innerHTML=_leaf_html(node),
            style="font-family:monospace;font-size:12px;line-height:1.2;",
        )


def _render_node_solara(
    *,
    node: _TreeNode,
    depth: int,
    expand_top_level: bool,
    large_container_threshold: int,
    max_container_height_px: int,
) -> None:
    import solara

    if node.is_leaf:
        _render_leaf_solara(node)
        return

    with solara.Div(
        style=_container_card_style(),
        attributes={"title": _tooltip_path(node)},
    ):
        summary = solara.HTML(
            tag="div",
            unsafe_innerHTML=_summary_html(node),
            style="cursor:pointer;font-family:monospace;font-size:12px;line-height:1.2;",
            attributes={"title": _tooltip_path(node)},
        )
        with solara.Details(summary=summary, expand=(expand_top_level and depth == 0)):
            content_style: dict[str, str] = {"padding": "0 6px 4px 6px"}
            if node.total_children >= large_container_threshold:
                content_style["max-height"] = f"{max_container_height_px}px"
                content_style["overflow-x"] = "hidden"
                content_style["overflow-y"] = "auto"

            with solara.Div(style=content_style):
                with solara.Column(gap="1px", style={"padding-left": "8px"}):
                    for child in node.children:
                        _render_node_solara(
                            node=child,
                            depth=depth + 1,
                            expand_top_level=expand_top_level,
                            large_container_threshold=large_container_threshold,
                            max_container_height_px=max_container_height_px,
                        )


def _validate_tree_args(
    *,
    max_repr_len: int,
    max_items_per_container: int,
    max_depth: Optional[int],
    large_container_threshold: int,
    max_container_height_px: int,
) -> None:
    if max_repr_len < 8:
        raise ValueError("max_repr_len must be >= 8")
    if max_items_per_container <= 0:
        raise ValueError("max_items_per_container must be > 0")
    if max_depth is not None and max_depth < 0:
        raise ValueError("max_depth must be >= 0 when provided")
    if large_container_threshold <= 0:
        raise ValueError("large_container_threshold must be > 0")
    if max_container_height_px <= 0:
        raise ValueError("max_container_height_px must be > 0")


def _root_tree(
    *,
    config: Any,
    root_name: Optional[str],
    max_repr_len: int,
    max_items_per_container: int,
    max_depth: Optional[int],
) -> _TreeNode:
    return _build_tree(
        name=root_name or type(config).__name__,
        path="",
        value=config,
        max_repr_len=max_repr_len,
        max_items_per_container=max_items_per_container,
        max_depth=max_depth,
    )


def _tree_css() -> str:
    return """
        .dlg-config-tree .v-expansion-panel::before { box-shadow: none !important; }
        .dlg-config-tree .v-expansion-panel-header { min-height: 26px !important; padding: 4px 8px !important; }
        .dlg-config-tree .v-expansion-panel-content__wrap { padding: 2px 8px 4px 8px !important; }
        .dlg-config-tree .v-ripple__container { display: none !important; }
        .dlg-config-tree .v-expansion-panel,
        .dlg-config-tree .v-expansion-panel-header,
        .dlg-config-tree .v-expansion-panel-content,
        .dlg-config-tree .v-expansion-panel-content__wrap,
        .dlg-config-tree .v-expansion-panel-header__icon .v-icon {
            animation: none !important;
            transition: none !important;
        }
        .dlg-config-tree .v-expansion-panel-header:before,
        .dlg-config-tree .v-expansion-panel-header:hover:before {
            opacity: 0 !important;
        }
        .dlg-config-tree .v-expansion-panel-header:focus,
        .dlg-config-tree .v-expansion-panel-header:focus-visible {
            outline: none !important;
        }
    """


def _render_tree_root(
    *,
    root: _TreeNode,
    expand_top_level: bool,
    large_container_threshold: int,
    max_container_height_px: int,
) -> None:
    import solara

    solara.Style(_tree_css())

    with solara.Column(
        classes=["dlg-config-tree"],
        gap="2px",
        style={"width": "100%"},
    ):
        if root.is_leaf:
            _render_leaf_solara(root)
            return

        _render_node_solara(
            node=root,
            depth=0,
            expand_top_level=expand_top_level,
            large_container_threshold=large_container_threshold,
            max_container_height_px=max_container_height_px,
        )


def visualize_nested_config(
    config: Any,
    *,
    root_name: Optional[str] = None,
    expand_top_level: bool = False,
    max_repr_len: int = 180,
    max_items_per_container: int = 200,
    max_depth: Optional[int] = None,
    large_container_threshold: int = 64,
    max_container_height_px: int = 240,
):
    """Create a notebook-displayable tree view for nested config objects."""
    import solara

    _validate_tree_args(
        max_repr_len=max_repr_len,
        max_items_per_container=max_items_per_container,
        max_depth=max_depth,
        large_container_threshold=large_container_threshold,
        max_container_height_px=max_container_height_px,
    )

    root = _root_tree(
        config=config,
        root_name=root_name,
        max_repr_len=max_repr_len,
        max_items_per_container=max_items_per_container,
        max_depth=max_depth,
    )

    @solara.component
    def _panel():
        _render_tree_root(
            root=root,
            expand_top_level=expand_top_level,
            large_container_threshold=large_container_threshold,
            max_container_height_px=max_container_height_px,
        )

    container, render_context = solara.render(_panel())
    setattr(container, "_solara_render_context", render_context)
    return container


def _tree_root_component(
    *,
    config: Any,
    root_name: Optional[str],
    expand_top_level: bool,
    max_repr_len: int,
    max_items_per_container: int,
    max_depth: Optional[int],
    large_container_threshold: int,
    max_container_height_px: int,
) -> None:
    _validate_tree_args(
        max_repr_len=max_repr_len,
        max_items_per_container=max_items_per_container,
        max_depth=max_depth,
        large_container_threshold=large_container_threshold,
        max_container_height_px=max_container_height_px,
    )
    root = _root_tree(
        config=config,
        root_name=root_name,
        max_repr_len=max_repr_len,
        max_items_per_container=max_items_per_container,
        max_depth=max_depth,
    )
    _render_tree_root(
        root=root,
        expand_top_level=expand_top_level,
        large_container_threshold=large_container_threshold,
        max_container_height_px=max_container_height_px,
    )


__all__ = ["visualize_nested_config"]
