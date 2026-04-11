"""Immutable nested replacement helpers for config-like objects."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import fields, is_dataclass, replace as dataclass_replace
import keyword
from typing import Any

from pydantic import BaseModel, ValidationError

PathToken = str | int


def _path_segment(name: str) -> str:
    if name.startswith("[") and name.endswith("]"):
        return name

    if name.isidentifier() and not keyword.iskeyword(name):
        return f".{name}"

    escaped = name.replace("\\", "\\\\").replace("'", "\\'")
    return f"['{escaped}']"


def child_path(parent_path: str, child_name: str) -> str:
    """Build a child path segment compatible with ``parse_path``."""
    segment = _path_segment(child_name)
    if parent_path:
        return f"{parent_path}{segment}"
    if segment.startswith("."):
        return segment[1:]
    return segment


def _parse_identifier(path: str, index: int) -> tuple[str, int]:
    n = len(path)
    if index >= n or not (path[index].isalpha() or path[index] == "_"):
        raise ValueError(
            f"Invalid path at position {index}: expected identifier in {path!r}"
        )

    end = index + 1
    while end < n and (path[end].isalnum() or path[end] == "_"):
        end += 1
    return path[index:end], end


def _parse_bracket_token(path: str, index: int) -> tuple[PathToken, int]:
    n = len(path)
    if index >= n or path[index] != "[":
        raise ValueError(f"Invalid path at position {index}: expected '[' in {path!r}")

    i = index + 1
    if i >= n:
        raise ValueError(f"Invalid path: unterminated '[' in {path!r}")

    if path[i] in {"'", '"'}:
        quote = path[i]
        i += 1
        chars: list[str] = []
        while i < n:
            ch = path[i]
            if ch == "\\":
                i += 1
                if i >= n:
                    raise ValueError(f"Invalid path escape in {path!r}")
                chars.append(path[i])
                i += 1
                continue
            if ch == quote:
                i += 1
                break
            chars.append(ch)
            i += 1
        else:
            raise ValueError(f"Invalid path: unterminated quoted key in {path!r}")

        if i >= n or path[i] != "]":
            raise ValueError(f"Invalid path: expected closing ']' in {path!r}")
        return "".join(chars), i + 1

    sign = 1
    if path[i] == "-":
        sign = -1
        i += 1

    start_digits = i
    while i < n and path[i].isdigit():
        i += 1
    if start_digits == i:
        raise ValueError(
            f"Invalid path at position {index}: bracket token must be quoted key or integer index in {path!r}"
        )

    if i >= n or path[i] != "]":
        raise ValueError(f"Invalid path: expected closing ']' in {path!r}")
    return sign * int(path[start_digits:i]), i + 1


def parse_path(path: str) -> list[PathToken]:
    """Parse a path from ``visualize_nested_config`` tooltips/copy text."""
    path = path.strip()
    if path in {"", "(root)"}:
        return []

    tokens: list[PathToken] = []
    i = 0
    n = len(path)
    while i < n:
        ch = path[i]
        if ch == ".":
            i += 1
            if i >= n:
                raise ValueError(f"Invalid path: trailing '.' in {path!r}")
            if path[i] == "[":
                token, i = _parse_bracket_token(path, i)
            else:
                token, i = _parse_identifier(path, i)
            tokens.append(token)
            continue
        if ch == "[":
            token, i = _parse_bracket_token(path, i)
            tokens.append(token)
            continue
        token, i = _parse_identifier(path, i)
        tokens.append(token)

    return tokens


def _normalize_sequence_index(length: int, index: int, *, path: str) -> int:
    normalized = index if index >= 0 else length + index
    if normalized < 0 or normalized >= length:
        raise IndexError(
            f"Sequence index {index} out of range for length {length} in path {path!r}"
        )
    return normalized


def _get_child_value(value: Any, token: PathToken, *, path: str) -> Any:
    if isinstance(value, BaseModel):
        if not isinstance(token, str):
            raise TypeError(
                f"Model field access requires string token, got {token!r} in path {path!r}"
            )
        if token not in type(value).model_fields:
            raise KeyError(
                f"Field {token!r} not found in {type(value).__name__} for path {path!r}"
            )
        return getattr(value, token)

    if is_dataclass(value) and not isinstance(value, type):
        if not isinstance(token, str):
            raise TypeError(
                f"Dataclass field access requires string token, got {token!r} in path {path!r}"
            )
        valid_fields = {field.name for field in fields(value)}
        if token not in valid_fields:
            raise KeyError(
                f"Field {token!r} not found in dataclass {type(value).__name__} for path {path!r}"
            )
        return getattr(value, token)

    if isinstance(value, Mapping):
        if token not in value:
            raise KeyError(
                f"Key {token!r} not found in mapping {type(value).__name__} for path {path!r}"
            )
        return value[token]

    if isinstance(value, list | tuple):
        if not isinstance(token, int):
            raise TypeError(
                f"Sequence index must be int, got {token!r} in path {path!r}"
            )
        idx = _normalize_sequence_index(len(value), token, path=path)
        return value[idx]

    raise TypeError(
        f"Cannot traverse into type {type(value).__name__} with token {token!r} for path {path!r}"
    )


def _set_child_copy(value: Any, token: PathToken, child_value: Any, *, path: str) -> Any:
    if isinstance(value, BaseModel):
        if not isinstance(token, str):
            raise TypeError(
                f"Model field update requires string token, got {token!r} in path {path!r}"
            )
        if token not in type(value).model_fields:
            raise KeyError(
                f"Field {token!r} not found in {type(value).__name__} for path {path!r}"
            )
        try:
            return value.model_copy(update={token: child_value})
        except ValidationError as e:
            raise ValueError(f"Invalid replacement at {path!r}: {e}") from e

    if is_dataclass(value) and not isinstance(value, type):
        if not isinstance(token, str):
            raise TypeError(
                f"Dataclass field update requires string token, got {token!r} in path {path!r}"
            )
        valid_fields = {field.name for field in fields(value)}
        if token not in valid_fields:
            raise KeyError(
                f"Field {token!r} not found in dataclass {type(value).__name__} for path {path!r}"
            )
        return dataclass_replace(value, **{token: child_value})

    if isinstance(value, list):
        if not isinstance(token, int):
            raise TypeError(
                f"List update index must be int, got {token!r} in path {path!r}"
            )
        idx = _normalize_sequence_index(len(value), token, path=path)
        copied = list(value)
        copied[idx] = child_value
        return copied

    if isinstance(value, tuple):
        if not isinstance(token, int):
            raise TypeError(
                f"Tuple update index must be int, got {token!r} in path {path!r}"
            )
        idx = _normalize_sequence_index(len(value), token, path=path)
        items = list(value)
        items[idx] = child_value
        if hasattr(value, "_fields"):
            return type(value)(*items)
        return tuple(items)

    if isinstance(value, Mapping):
        if token not in value:
            raise KeyError(
                f"Key {token!r} not found in mapping {type(value).__name__} for path {path!r}"
            )
        if isinstance(value, dict):
            copied_dict = value.copy()
            copied_dict[token] = child_value
            return copied_dict

        items = list(value.items())
        updated_items = [(k, child_value if k == token else v) for k, v in items]
        try:
            return type(value)(updated_items)
        except Exception:
            return dict(updated_items)

    raise TypeError(
        f"Cannot update type {type(value).__name__} with token {token!r} for path {path!r}"
    )


def _replace_tokens(
    value: Any,
    *,
    tokens: list[PathToken],
    replacement: Any,
    path: str,
) -> Any:
    if len(tokens) == 0:
        return replacement

    head = tokens[0]
    if len(tokens) == 1:
        return _set_child_copy(value, head, replacement, path=path)

    next_value = _get_child_value(value, head, path=path)
    replaced_child = _replace_tokens(
        next_value,
        tokens=tokens[1:],
        replacement=replacement,
        path=path,
    )
    return _set_child_copy(value, head, replaced_child, path=path)


def replace_in_nested_config(config: Any, *, path: str, replacement: Any) -> Any:
    """Return an immutable copy of ``config`` with ``path`` replaced."""
    tokens = parse_path(path)
    if len(tokens) == 0:
        return replacement

    return _replace_tokens(config, tokens=tokens, replacement=replacement, path=path)


__all__ = ["PathToken", "child_path", "parse_path", "replace_in_nested_config"]
