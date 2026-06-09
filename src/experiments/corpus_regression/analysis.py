import json
import re
import warnings
from pathlib import Path
from typing import Any, Literal, Self

import numpy as np
import plotly.graph_objects as go
import polars as pl
from plotly.colors import qualitative
from plotly.subplots import make_subplots

from src.config.base import BaseConfig
from src.data.corpus_regression import (
    CorpusRegressionDatasetConfig,
    candidate_lookforward_tokens,
    candidate_rollout_steps,
    candidate_seeds,
)
from src.experiments.corpus_regression.config import (
    baseline_mode_folder,
    batch_size_segment,
    factorized_mode_folder,
    likelihood_mode_folder,
    lr_schedule_segment,
    sigma_folder,
)

_ROLLOUTS_RE = re.compile(r" r=(\d+)")

# Group-axis machinery. Every `from_*_sweep` accepts these axes as either a
# scalar (fixed across the sweep) or a `list` (the chosen *group-by* axis).
# At most one may be a list per call; that one becomes `group_by` and is the
# axis the per-method plots use for legend grouping.
_AXIS_SHORT: dict[str, str] = {
    "num_lookforward_tokens": "look",
    "num_samples": "N",
    "batch_size": "bs",
    "lr_per_sample": "lr",
    "train_from_scratch": "scratch",
    "gaussian_stdev": "sigma",
    "train_steps": "steps",
}


def _resolve_group_axis(
    *,
    candidates: dict[str, Any],
) -> tuple[str | None, list[Any]]:
    """Validate that at most one entry of `candidates` is a `list`. Return
    `(axis_name, values_to_loop)`. When everything is scalar, returns
    `(None, [None])` — caller treats that single-iteration None as
    "no group axis active". Raises `ValueError` if two or more are lists."""
    list_axes = {k: v for k, v in candidates.items() if isinstance(v, list)}
    if len(list_axes) > 1:
        raise ValueError(
            f"At most one of {sorted(candidates.keys())} may be a list "
            f"(the chosen group-by axis); got lists for: {sorted(list_axes)}."
        )
    if not list_axes:
        return None, [None]
    name, values = next(iter(list_axes.items()))
    return name, list(values)


def _format_group_value(*, axis: str, value: Any) -> str:
    """Render a group-axis value for display in the legend / study-name string.
    Mirrors the on-disk folder convention so legend reads identically to path:
      - num_samples → str(int)
      - batch_size → str(int)  (matches `bs-{N}` from `batch_size_segment`)
      - lr_per_sample → f"{v:.2e}"  (matches `f\"lr_{lr:.2e}\"` in config.py)
      - train_from_scratch → str(bool)
      - gaussian_stdev → f"{v:.1f}"  (matches `sigma_folder` pin in config.py)
      - num_lookforward_tokens → str(int)
      - train_steps → f"{int(v):06d}"  (matches `steps-{N:06d}` in config.py)
    """
    if axis == "num_samples":
        return str(int(value))
    if axis == "batch_size":
        return str(int(value))
    if axis == "lr_per_sample":
        return f"{float(value):.2e}"
    if axis == "train_from_scratch":
        return str(bool(value))
    if axis == "gaussian_stdev":
        return f"{float(value):.1f}"
    if axis == "num_lookforward_tokens":
        return str(int(value))
    if axis == "train_steps":
        return f"{int(value):06d}"
    raise ValueError(f"Unknown group axis: {axis!r}")


def _study_name(
    *,
    num_lookforward: int,
    group_by: str | None,
    group_value: Any,
    rollouts: int | None,
) -> str:
    """Build the study-name string. Single source of truth for the format
    documented in the design plan.

    When `group_by ∈ {None, "num_lookforward_tokens"}` the format is
    byte-identical to today's (`"look=N"` / `"look=N r=R"`) so cross-method
    plots that look up `f"look={N}"`-formatted keys keep working."""
    name = f"look={num_lookforward}"
    if group_by is not None and group_by != "num_lookforward_tokens":
        short = _AXIS_SHORT[group_by]
        name = f"{name} {short}={_format_group_value(axis=group_by, value=group_value)}"
    if rollouts is not None:
        name = f"{name} r={rollouts}"
    return name


def _as_list(value: Any) -> list[Any]:
    """Wrap scalar inputs to length-1 lists; pass lists through unchanged.
    Used so each `from_*_sweep` can iterate uniformly over its sweep axes."""
    return value if isinstance(value, list) else [value]


def _select_group_value(
    *,
    group_by: str | None,
    num_lookforward_tokens: Any,
    num_samples: Any,
    lr_per_sample: Any,
    train_from_scratch: Any,
    gaussian_stdev: Any,
    train_steps: Any = None,
    batch_size: Any = None,
) -> Any:
    """Return the current iteration's value for whichever axis is `group_by`.
    Sweep classmethods pass every axis position so this helper can be reused
    across SL / NTP / RL without conditional plumbing at the call site.
    `None` for axes that don't exist in the calling sweep is fine; we only
    look up the axis named by `group_by`."""
    if group_by is None:
        return None
    if group_by == "num_lookforward_tokens":
        return num_lookforward_tokens
    if group_by == "num_samples":
        return num_samples
    if group_by == "batch_size":
        return batch_size
    if group_by == "lr_per_sample":
        return lr_per_sample
    if group_by == "train_from_scratch":
        return train_from_scratch
    if group_by == "gaussian_stdev":
        return gaussian_stdev
    if group_by == "train_steps":
        return train_steps
    raise ValueError(f"Unknown group axis: {group_by!r}")


def _filter_studies_by_lookforward(
    *,
    studies: dict[str, list[Path]],
    study_lookforwards: dict[str, int],
    num_lookforward_tokens: int | None,
) -> list[str]:
    """Return the ordered list of study names matching `num_lookforward_tokens`.
    When the filter is `None`, returns every study key in declaration order
    (today's behaviour). Raises if the filter is set but no study matches —
    silent-empty-figure is worse than a clear ValueError."""
    if num_lookforward_tokens is None:
        return list(studies.keys())
    matches = [
        s for s in studies if study_lookforwards[s] == num_lookforward_tokens
    ]
    if not matches:
        seen = sorted(set(study_lookforwards.values()))
        raise ValueError(
            f"No studies with num_lookforward_tokens={num_lookforward_tokens} "
            f"in this config (have: {seen})"
        )
    return matches


def _seed_folder_name(seed: int) -> str:
    return f"seed-{seed}"


def _method_dir(method: str, *, train_from_scratch: bool) -> Path:
    """Method-level artifact subfolder. From-scratch sweeps land in a
    `<method>/from_scratch` sibling alongside `<method>/from_pretrain` so the
    initialization regime is an explicit path level rather than an ad-hoc
    `_scratch` suffix."""
    from_dir = "from_scratch" if train_from_scratch else "from_pretrain"
    return Path(method) / from_dir


def _attach_bs_segment(*, batch_size: int) -> Path | None:
    """Translate a batch_size into the optional `bs-{N}` path segment.

    Returns ``None`` for the canonical default (`DEFAULT_BATCH_SIZE = 64`)
    so existing artifacts (which never wrote a `bs-*` level) are byte-for-
    byte discoverable. Otherwise returns ``Path(f"bs-{N}")`` for the caller
    to splice into the path between ``{dataset_folder}`` and ``lr_<value>``.

    Mirrors `batch_size_segment` from config.py but typed as `Path | None`
    for direct use in path construction.
    """
    seg = batch_size_segment(batch_size=batch_size)
    return Path(seg) if seg is not None else None


def _splice_bs(*, path: Path, bs_seg: Path | None) -> Path:
    """Append the `bs-{N}` segment when present, else identity. Centralizes
    the omit-at-default convention so all five sweep loaders share the
    same path-construction code."""
    return path if bs_seg is None else path / bs_seg


def _make_attach_sched(
    *,
    lr_schedule: Literal["flat", "cosine"],
    warmup_ratio: float,
    lr_min_ratio: float,
):
    """Closure that appends the optional schedule segment to a path.

    For `lr_schedule == "flat"` returns the identity (default; preserves
    today's exact paths). For `"cosine"` returns ``p / sched_seg``. The
    segment string is computed once via the shared `lr_schedule_segment`
    helper so write-side (config.py) and read-side (analysis.py) cannot
    drift apart.
    """
    sched_seg = lr_schedule_segment(
        lr_schedule=lr_schedule,
        warmup_ratio=warmup_ratio,
        lr_min_ratio=lr_min_ratio,
    )
    if sched_seg is None:
        return lambda p: p
    return lambda p: p / sched_seg


def canonical_dataset_folder_name(
    *,
    num_lookforward_tokens: int,
    num_samples: int = 100_000,
    label_type: Literal["rademacher", "token_id"] = "rademacher",
    normalize_labels: bool = False,
    label_range: tuple[float, float] = (0.0, 1.0),
) -> str:
    """Slug used by single_run scripts as the leaf study folder. Computed from
    `CorpusRegressionDatasetConfig.get_canonical_folder` so it stays in sync
    with run-time artifacts."""
    cfg = CorpusRegressionDatasetConfig(
        **{
            **CorpusRegressionDatasetConfig.canonical_kwargs(
                label_type=label_type,
                normalize_labels=normalize_labels,
                label_range=label_range,
            ),
            "num_lookforward_tokens": num_lookforward_tokens,
            "num_samples": num_samples,
        }
    )
    return cfg.get_canonical_folder(Path("/")).name


def _decode_dim_averaged(df: pl.DataFrame) -> pl.DataFrame:
    """Append scalar `{split}_corr`, `{split}_mse`, `{split}_pred_std`,
    `{split}_target_std`, and `{split}_pred_std_ratio` columns by averaging
    per-dim metrics across the D output dimensions of the list-column
    sufficient stats. The pred/target stds are computed from the new
    `*_pred_sum` and `*_target_sum` columns when present (Var = E[X²] − E[X]²,
    averaged across dims, then sqrt). Old parquets without those columns
    silently get null pred/target stds rather than raising. NTP baseline writes
    both train and val stats; trained methods write val plus an optional
    train-window aggregate."""
    out = df
    for split in ("train", "val"):
        xx_col = f"{split}_target_xx"
        if xx_col not in df.columns:
            continue
        xx = np.asarray(df[f"{split}_target_xx"].to_list(), dtype=np.float64)
        xy = np.asarray(df[f"{split}_target_xy"].to_list(), dtype=np.float64)
        yy = np.asarray(df[f"{split}_target_yy"].to_list(), dtype=np.float64)
        n = df[f"{split}_target_n"].to_numpy().astype(np.float64)
        per_dim_corr = xy / np.sqrt(xx * yy)
        per_dim_mse = (xx - 2.0 * xy + yy) / n[:, None]
        out = out.with_columns(
            pl.Series(f"{split}_corr", per_dim_corr.mean(axis=1)),
            pl.Series(f"{split}_mse", per_dim_mse.mean(axis=1)),
        )

        pred_sum_col = f"{split}_target_pred_sum"
        target_sum_col = f"{split}_target_target_sum"
        if (
            pred_sum_col in df.columns
            and target_sum_col in df.columns
        ):
            pred_sum_lst = df[pred_sum_col].to_list()
            target_sum_lst = df[target_sum_col].to_list()
            # `_align_val_schema` backfills missing rows with None; numpy can't
            # build a 2-D array from a mixed [None, [...], ...] list, so we
            # decode row-by-row and emit nulls where the list-column is null.
            pred_var_rows: list[float | None] = []
            target_var_rows: list[float | None] = []
            ratio_rows: list[float | None] = []
            for i, (pred_row, target_row) in enumerate(
                zip(pred_sum_lst, target_sum_lst, strict=True)
            ):
                if pred_row is None or target_row is None or n[i] <= 0:
                    pred_var_rows.append(None)
                    target_var_rows.append(None)
                    ratio_rows.append(None)
                    continue
                pred_arr = np.asarray(pred_row, dtype=np.float64)
                target_arr = np.asarray(target_row, dtype=np.float64)
                pred_mean = pred_arr / n[i]
                target_mean = target_arr / n[i]
                pred_var_per_dim = xx[i] / n[i] - pred_mean * pred_mean
                target_var_per_dim = yy[i] / n[i] - target_mean * target_mean
                pred_var = float(np.mean(pred_var_per_dim))
                target_var = float(np.mean(target_var_per_dim))
                pred_var_rows.append(pred_var)
                target_var_rows.append(target_var)
                pred_std = float(np.sqrt(max(pred_var, 0.0)))
                target_std = float(np.sqrt(max(target_var, 0.0)))
                ratio_rows.append(
                    pred_std / target_std if target_std > 0.0 else None
                )
            pred_std_rows = [
                float(np.sqrt(max(v, 0.0))) if v is not None else None
                for v in pred_var_rows
            ]
            target_std_rows = [
                float(np.sqrt(max(v, 0.0))) if v is not None else None
                for v in target_var_rows
            ]
            out = out.with_columns(
                pl.Series(f"{split}_pred_var", pred_var_rows, dtype=pl.Float64),
                pl.Series(f"{split}_target_var", target_var_rows, dtype=pl.Float64),
                pl.Series(f"{split}_pred_std", pred_std_rows, dtype=pl.Float64),
                pl.Series(f"{split}_target_std", target_std_rows, dtype=pl.Float64),
                pl.Series(f"{split}_pred_std_ratio", ratio_rows, dtype=pl.Float64),
            )
            # Variance-normalized MSE: MSE / Var(target) on the same split.
            # Reads as "fraction of target variance unexplained" — 1 is the
            # constant-mean baseline, 0 is perfect prediction. `null` when
            # target_var is missing or non-positive (degenerate eval set).
            out = out.with_columns(
                pl.when(pl.col(f"{split}_target_var") > 0.0)
                .then(pl.col(f"{split}_mse") / pl.col(f"{split}_target_var"))
                .otherwise(None)
                .alias(f"{split}_normalized_mse")
            )
    return out


def _has_val_metrics(path: Path) -> bool:
    """True if the study folder contains a `val_metrics.parquet` (the only
    on-disk format the analysis layer accepts)."""
    return (path / "val_metrics.parquet").exists()


class CorpusRegressionAnalysisConfig(BaseConfig):
    """Grouped view over completed/started study folders. Each `studies` key
    is a group name (`"look=4"` for SL, `"look=4 r=128"` for GRPO/MaxRL); when
    a non-lookforward group axis is selected the key is extended with the axis
    short tag (e.g. `"look=4 N=100000 r=128"`). The value is the list of
    per-seed study folders backing that key."""

    studies: dict[str, list[Path]]
    study_seeds: dict[str, list[int]]
    study_lookforwards: dict[str, int]
    # Per-study training batch size, read from `config.json["dataloading"]
    # ["train_batch_size"]`. Old artifacts that pre-date the `bs-{N}` path
    # segment all used 64; backcompat is therefore "whatever config.json
    # says". Surfaced via `get_metric_dataframe` / `get_train_dataframe`
    # as a `batch_size` column so analyses can group/filter on it.
    study_batch_sizes: dict[str, int] = {}
    # The axis whose values vary across `studies` (or None if all axes were
    # scalar at construction time). Drives the per-method plot's legend
    # grouping. `"num_lookforward_tokens"` reproduces today's behaviour.
    group_by: str | None = None
    # Per-study formatted value of the group axis. Empty when `group_by` is
    # None or `"num_lookforward_tokens"` (the lookforward value is already in
    # `study_lookforwards`).
    study_group_values: dict[str, str] = {}

    @classmethod
    def from_grouped(
        cls,
        grouped: dict[str, list[tuple[int, Path]]],
        *,
        group_by: str | None = None,
        study_group_values: dict[str, str] | None = None,
    ) -> Self | None:
        """Drop paths missing `val_metrics.parquet` or `config.json`; drop empty
        groups; return None if nothing survives."""
        studies: dict[str, list[Path]] = {}
        study_seeds: dict[str, list[int]] = {}
        study_lookforwards: dict[str, int] = {}
        study_batch_sizes: dict[str, int] = {}
        surviving_group_values: dict[str, str] = {}
        for name, pairs in grouped.items():
            kept = [
                (s, p)
                for (s, p) in pairs
                if _has_val_metrics(p) and (p / "config.json").exists()
            ]
            if not kept:
                continue
            studies[name] = [p for (_, p) in kept]
            study_seeds[name] = [s for (s, _) in kept]
            first_path = kept[0][1]
            cfg = json.loads((first_path / "config.json").read_text())
            study_lookforwards[name] = int(cfg["data"]["num_lookforward_tokens"])
            # `dataloading.train_batch_size` is present in every config.json
            # this analyzer can read; missing the key would mean a config
            # written by code older than `CorpusRegressionDataloadingConfig`,
            # which never landed in this repo. Read straight rather than
            # imputing 64 to surface schema drift loudly.
            study_batch_sizes[name] = int(
                cfg["dataloading"]["train_batch_size"],
            )
            if study_group_values is not None and name in study_group_values:
                surviving_group_values[name] = study_group_values[name]
        if not studies:
            return None
        return cls(
            studies=studies,
            study_seeds=study_seeds,
            study_lookforwards=study_lookforwards,
            study_batch_sizes=study_batch_sizes,
            group_by=group_by,
            study_group_values=surviving_group_values,
        )

    @classmethod
    def from_pretrained_baseline(
        cls,
        *,
        artifacts_root: Path,
        num_lookforward_tokens: int | list[int] = 1,
        num_samples: int | list[int] = 100_000,
        label_type: Literal["rademacher", "token_id"] = "rademacher",
        normalize_labels: bool = False,
        label_range: tuple[float, float] = (0.0, 1.0),
    ) -> Self | None:
        """`<artifacts_root>/pretrained_baseline/from_pretrain/{dataset_folder}/`.

        Pretrained-model baseline has no seeds (deterministic given dataset).
        We wrap the single result as seed=0 for compatibility with the
        grouped API. The `from_pretrain` level is mandatory — there is no
        from_scratch counterpart for an inference-only intrinsic-variance
        baseline.

        Pass either `num_lookforward_tokens` or `num_samples` as a `list` to
        sweep that axis (becomes `group_by`); the other stays scalar.
        """
        study_base = artifacts_root / "pretrained_baseline" / "from_pretrain"
        if not study_base.exists():
            return None
        group_by, _ = _resolve_group_axis(
            candidates={
                "num_lookforward_tokens": num_lookforward_tokens,
                "num_samples": num_samples,
            }
        )
        looks = _as_list(num_lookforward_tokens)
        samples = _as_list(num_samples)
        grouped: dict[str, list[tuple[int, Path]]] = {}
        study_group_values: dict[str, str] = {}
        for n in looks:
            for ns in samples:
                gv = _select_group_value(
                    group_by=group_by,
                    num_lookforward_tokens=n, num_samples=ns,
                    lr_per_sample=None, train_from_scratch=None,
                    gaussian_stdev=None,
                )
                name = _study_name(
                    num_lookforward=n, group_by=group_by,
                    group_value=gv, rollouts=None,
                )
                folder_name = canonical_dataset_folder_name(
                    num_lookforward_tokens=n,
                    num_samples=ns,
                    label_type=label_type,
                    normalize_labels=normalize_labels,
                    label_range=label_range,
                )
                grouped[name] = [(0, study_base / folder_name)]
                if group_by is not None and group_by != "num_lookforward_tokens":
                    study_group_values[name] = _format_group_value(
                        axis=group_by, value=gv,
                    )
        return cls.from_grouped(
            grouped, group_by=group_by, study_group_values=study_group_values,
        )

    @classmethod
    def from_sl_mse_sweep(
        cls,
        *,
        artifacts_root: Path,
        num_lookforward_tokens: int | list[int] = 1,
        num_samples: int | list[int] = 100_000,
        batch_size: int | list[int] = 64,
        label_type: Literal["rademacher", "token_id"] = "rademacher",
        normalize_labels: bool = False,
        label_range: tuple[float, float] = (0.0, 1.0),
        lr_per_sample: float | list[float] = 1e-5,
        lr_schedule: Literal["flat", "cosine"] = "flat",
        warmup_ratio: float = 0.05,
        lr_min_ratio: float = 0.1,
        train_from_scratch: bool | list[bool] = False,
        train_steps: int | list[int] = 10_000,
    ) -> Self | None:
        """`<artifacts_root>/sl_mse/<from_*>/seed-{S}/{dataset_folder}/[bs-{B}/]lr_{lr}/steps-{N:06d}/[sched-...]/`.

        At most ONE of {`num_lookforward_tokens`, `num_samples`, `batch_size`,
        `lr_per_sample`, `train_from_scratch`, `train_steps`} may be a `list`;
        that becomes the legend `group_by` axis for per-method plots.
        All-scalar (default) → `group_by=None` and a single study (`look=1`).

        The `bs-{N}` segment is omitted at the canonical default (64) so the
        sweep also discovers pre-batch-size artifacts. The
        `sched-warmup-X.XXX-cosine-X.XX` segment is appended only for
        non-flat schedules; defaults reproduce today's flat-LR layout."""
        group_by, _ = _resolve_group_axis(
            candidates={
                "num_lookforward_tokens": num_lookforward_tokens,
                "num_samples": num_samples,
                "batch_size": batch_size,
                "lr_per_sample": lr_per_sample,
                "train_from_scratch": train_from_scratch,
                "train_steps": train_steps,
            }
        )
        attach_sched = _make_attach_sched(
            lr_schedule=lr_schedule,
            warmup_ratio=warmup_ratio,
            lr_min_ratio=lr_min_ratio,
        )
        grouped: dict[str, list[tuple[int, Path]]] = {}
        study_group_values: dict[str, str] = {}
        for n in _as_list(num_lookforward_tokens):
            for ns in _as_list(num_samples):
                for bs in _as_list(batch_size):
                    for lr in _as_list(lr_per_sample):
                        for scratch in _as_list(train_from_scratch):
                            for ts in _as_list(train_steps):
                                study_base = artifacts_root / _method_dir(
                                    "sl_mse", train_from_scratch=scratch,
                                )
                                if not study_base.exists():
                                    continue
                                lr_seg = f"lr_{lr:.2e}"
                                steps_seg = f"steps-{int(ts):06d}"
                                bs_seg = _attach_bs_segment(batch_size=int(bs))
                                dataset_seg = canonical_dataset_folder_name(
                                    num_lookforward_tokens=n, num_samples=ns,
                                    label_type=label_type,
                                    normalize_labels=normalize_labels,
                                    label_range=label_range,
                                )
                                gv = _select_group_value(
                                    group_by=group_by,
                                    num_lookforward_tokens=n, num_samples=ns,
                                    batch_size=bs,
                                    lr_per_sample=lr, train_from_scratch=scratch,
                                    gaussian_stdev=None, train_steps=ts,
                                )
                                name = _study_name(
                                    num_lookforward=n, group_by=group_by,
                                    group_value=gv, rollouts=None,
                                )
                                # The `bs-{N}` segment sits between
                                # `{dataset_seg}` and `lr_seg` exactly like
                                # in `study_folder` written by
                                # `canonical_kwargs`.
                                grouped[name] = [
                                    (
                                        s,
                                        attach_sched(
                                            _splice_bs(
                                                path=study_base
                                                / _seed_folder_name(s)
                                                / dataset_seg,
                                                bs_seg=bs_seg,
                                            )
                                            / lr_seg
                                            / steps_seg
                                        ),
                                    )
                                    for s in candidate_seeds
                                ]
                                if group_by is not None and group_by != "num_lookforward_tokens":
                                    study_group_values[name] = _format_group_value(
                                        axis=group_by, value=gv,
                                    )
        if not grouped:
            return None
        return cls.from_grouped(
            grouped, group_by=group_by, study_group_values=study_group_values,
        )

    @classmethod
    def from_sl_ce_sweep(
        cls,
        *,
        artifacts_root: Path,
        num_lookforward_tokens: int | list[int] = 1,
        num_samples: int | list[int] = 100_000,
        batch_size: int | list[int] = 64,
        label_type: Literal["rademacher", "token_id"] = "rademacher",
        normalize_labels: bool = False,
        label_range: tuple[float, float] = (0.0, 1.0),
        lr_per_sample: float | list[float] = 1e-5,
        lr_schedule: Literal["flat", "cosine"] = "flat",
        warmup_ratio: float = 0.05,
        lr_min_ratio: float = 0.1,
        train_from_scratch: bool | list[bool] = False,
        train_steps: int | list[int] = 10_000,
    ) -> Self | None:
        """`<artifacts_root>/sl_ce/<from_*>/seed-{S}/{dataset_folder}/[bs-{B}/]lr_{lr}/steps-{N:06d}/[sched-...]/`.

        Identical layout and group-axis semantics to `from_sl_mse_sweep`.
        Only K=1 is meaningful for the NTP-CE objective, but we keep the
        lookforward sweep loop here so partially-populated trees still
        render cleanly.
        """
        group_by, _ = _resolve_group_axis(
            candidates={
                "num_lookforward_tokens": num_lookforward_tokens,
                "num_samples": num_samples,
                "batch_size": batch_size,
                "lr_per_sample": lr_per_sample,
                "train_from_scratch": train_from_scratch,
                "train_steps": train_steps,
            }
        )
        attach_sched = _make_attach_sched(
            lr_schedule=lr_schedule,
            warmup_ratio=warmup_ratio,
            lr_min_ratio=lr_min_ratio,
        )
        grouped: dict[str, list[tuple[int, Path]]] = {}
        study_group_values: dict[str, str] = {}
        for n in _as_list(num_lookforward_tokens):
            for ns in _as_list(num_samples):
                for bs in _as_list(batch_size):
                    for lr in _as_list(lr_per_sample):
                        for scratch in _as_list(train_from_scratch):
                            for ts in _as_list(train_steps):
                                study_base = artifacts_root / _method_dir(
                                    "sl_ce", train_from_scratch=scratch,
                                )
                                if not study_base.exists():
                                    continue
                                lr_seg = f"lr_{lr:.2e}"
                                steps_seg = f"steps-{int(ts):06d}"
                                bs_seg = _attach_bs_segment(batch_size=int(bs))
                                dataset_seg = canonical_dataset_folder_name(
                                    num_lookforward_tokens=n, num_samples=ns,
                                    label_type=label_type,
                                    normalize_labels=normalize_labels,
                                    label_range=label_range,
                                )
                                gv = _select_group_value(
                                    group_by=group_by,
                                    num_lookforward_tokens=n, num_samples=ns,
                                    batch_size=bs,
                                    lr_per_sample=lr, train_from_scratch=scratch,
                                    gaussian_stdev=None, train_steps=ts,
                                )
                                name = _study_name(
                                    num_lookforward=n, group_by=group_by,
                                    group_value=gv, rollouts=None,
                                )
                                grouped[name] = [
                                    (
                                        s,
                                        attach_sched(
                                            _splice_bs(
                                                path=study_base
                                                / _seed_folder_name(s)
                                                / dataset_seg,
                                                bs_seg=bs_seg,
                                            )
                                            / lr_seg
                                            / steps_seg
                                        ),
                                    )
                                    for s in candidate_seeds
                                ]
                                if group_by is not None and group_by != "num_lookforward_tokens":
                                    study_group_values[name] = _format_group_value(
                                        axis=group_by, value=gv,
                                    )
        if not grouped:
            return None
        return cls.from_grouped(
            grouped, group_by=group_by, study_group_values=study_group_values,
        )

    @classmethod
    def from_grpo_sweep(
        cls,
        *,
        artifacts_root: Path,
        num_lookforward_tokens: int | list[int] = 1,
        num_samples: int | list[int] = 100_000,
        batch_size: int | list[int] = 64,
        gaussian_stdev: float | list[float] = 1.0,
        label_type: Literal["rademacher", "token_id"] = "rademacher",
        normalize_labels: bool = False,
        label_range: tuple[float, float] = (0.0, 1.0),
        lr_per_sample: float | list[float] = 1e-5,
        lr_schedule: Literal["flat", "cosine"] = "flat",
        warmup_ratio: float = 0.05,
        lr_min_ratio: float = 0.1,
        train_from_scratch: bool | list[bool] = False,
        train_steps: int | list[int] = 10_000,
    ) -> Self | None:
        """`<artifacts_root>/grpo/<from_*>/seed-{S}/rollouts-{N}/sigma-{σ}/{dataset_folder}/[bs-{B}/]lr_{lr}/steps-{N:06d}/[sched-...]/`.

        At most ONE of {`num_lookforward_tokens`, `num_samples`, `batch_size`,
        `lr_per_sample`, `train_from_scratch`, `gaussian_stdev`, `train_steps`}
        may be a `list` (the chosen `group_by` axis). The rollouts dimension
        is always swept across `candidate_rollout_steps`."""
        group_by, _ = _resolve_group_axis(
            candidates={
                "num_lookforward_tokens": num_lookforward_tokens,
                "num_samples": num_samples,
                "batch_size": batch_size,
                "lr_per_sample": lr_per_sample,
                "train_from_scratch": train_from_scratch,
                "gaussian_stdev": gaussian_stdev,
                "train_steps": train_steps,
            }
        )
        attach_sched = _make_attach_sched(
            lr_schedule=lr_schedule,
            warmup_ratio=warmup_ratio,
            lr_min_ratio=lr_min_ratio,
        )
        grouped: dict[str, list[tuple[int, Path]]] = {}
        study_group_values: dict[str, str] = {}
        for n in _as_list(num_lookforward_tokens):
            for ns in _as_list(num_samples):
                for bs in _as_list(batch_size):
                    for lr in _as_list(lr_per_sample):
                        for scratch in _as_list(train_from_scratch):
                            for sigma_v in _as_list(gaussian_stdev):
                                for ts in _as_list(train_steps):
                                    study_base = artifacts_root / _method_dir(
                                        "grpo", train_from_scratch=scratch,
                                    )
                                    if not study_base.exists():
                                        continue
                                    sigma = sigma_folder(gaussian_stdev=sigma_v)
                                    lr_seg = f"lr_{lr:.2e}"
                                    steps_seg = f"steps-{int(ts):06d}"
                                    bs_seg = _attach_bs_segment(batch_size=int(bs))
                                    dataset_seg = canonical_dataset_folder_name(
                                        num_lookforward_tokens=n,
                                        num_samples=ns,
                                        label_type=label_type,
                                        normalize_labels=normalize_labels,
                                        label_range=label_range,
                                    )
                                    gv = _select_group_value(
                                        group_by=group_by,
                                        num_lookforward_tokens=n, num_samples=ns,
                                        batch_size=bs,
                                        lr_per_sample=lr, train_from_scratch=scratch,
                                        gaussian_stdev=sigma_v, train_steps=ts,
                                    )
                                    for r in candidate_rollout_steps:
                                        name = _study_name(
                                            num_lookforward=n, group_by=group_by,
                                            group_value=gv, rollouts=r,
                                        )
                                        grouped[name] = [
                                            (
                                                s,
                                                attach_sched(
                                                    _splice_bs(
                                                        path=study_base
                                                        / _seed_folder_name(s)
                                                        / f"rollouts-{r}"
                                                        / sigma
                                                        / dataset_seg,
                                                        bs_seg=bs_seg,
                                                    )
                                                    / lr_seg
                                                    / steps_seg
                                                ),
                                            )
                                            for s in candidate_seeds
                                        ]
                                        if (
                                            group_by is not None
                                            and group_by != "num_lookforward_tokens"
                                        ):
                                            study_group_values[name] = _format_group_value(
                                                axis=group_by, value=gv,
                                            )
        if not grouped:
            return None
        return cls.from_grouped(
            grouped, group_by=group_by, study_group_values=study_group_values,
        )

    @classmethod
    def from_rloo_sweep(
        cls,
        *,
        artifacts_root: Path,
        factorized: bool,
        num_lookforward_tokens: int | list[int] = 1,
        num_samples: int | list[int] = 100_000,
        batch_size: int | list[int] = 64,
        gaussian_stdev: float | list[float] = 1.0,
        label_type: Literal["rademacher", "token_id"] = "rademacher",
        normalize_labels: bool = False,
        label_range: tuple[float, float] = (0.0, 1.0),
        lr_per_sample: float | list[float] = 1e-5,
        lr_schedule: Literal["flat", "cosine"] = "flat",
        warmup_ratio: float = 0.05,
        lr_min_ratio: float = 0.1,
        train_from_scratch: bool | list[bool] = False,
        train_steps: int | list[int] = 10_000,
    ) -> Self | None:
        """`<artifacts_root>/rloo/<from_*>/seed-{S}/rollouts-{N}/sigma-{σ}/{factorized_mode}/{dataset_folder}/[bs-{B}/]lr_{lr}/steps-{N:06d}/[sched-...]/`.
        `factorized` is fixed per call — surface it in the figure title.

        Group-axis semantics match `from_grpo_sweep`."""
        group_by, _ = _resolve_group_axis(
            candidates={
                "num_lookforward_tokens": num_lookforward_tokens,
                "num_samples": num_samples,
                "batch_size": batch_size,
                "lr_per_sample": lr_per_sample,
                "train_from_scratch": train_from_scratch,
                "gaussian_stdev": gaussian_stdev,
                "train_steps": train_steps,
            }
        )
        factorized_mode = factorized_mode_folder(factorized=factorized)
        attach_sched = _make_attach_sched(
            lr_schedule=lr_schedule,
            warmup_ratio=warmup_ratio,
            lr_min_ratio=lr_min_ratio,
        )
        grouped: dict[str, list[tuple[int, Path]]] = {}
        study_group_values: dict[str, str] = {}
        for n in _as_list(num_lookforward_tokens):
            for ns in _as_list(num_samples):
                for bs in _as_list(batch_size):
                    for lr in _as_list(lr_per_sample):
                        for scratch in _as_list(train_from_scratch):
                            for sigma_v in _as_list(gaussian_stdev):
                                for ts in _as_list(train_steps):
                                    study_base = artifacts_root / _method_dir(
                                        "rloo", train_from_scratch=scratch,
                                    )
                                    if not study_base.exists():
                                        continue
                                    sigma = sigma_folder(gaussian_stdev=sigma_v)
                                    lr_seg = f"lr_{lr:.2e}"
                                    steps_seg = f"steps-{int(ts):06d}"
                                    bs_seg = _attach_bs_segment(batch_size=int(bs))
                                    dataset_seg = canonical_dataset_folder_name(
                                        num_lookforward_tokens=n,
                                        num_samples=ns,
                                        label_type=label_type,
                                        normalize_labels=normalize_labels,
                                        label_range=label_range,
                                    )
                                    gv = _select_group_value(
                                        group_by=group_by,
                                        num_lookforward_tokens=n, num_samples=ns,
                                        batch_size=bs,
                                        lr_per_sample=lr, train_from_scratch=scratch,
                                        gaussian_stdev=sigma_v, train_steps=ts,
                                    )
                                    for r in candidate_rollout_steps:
                                        name = _study_name(
                                            num_lookforward=n, group_by=group_by,
                                            group_value=gv, rollouts=r,
                                        )
                                        grouped[name] = [
                                            (
                                                s,
                                                attach_sched(
                                                    _splice_bs(
                                                        path=study_base
                                                        / _seed_folder_name(s)
                                                        / f"rollouts-{r}"
                                                        / sigma
                                                        / factorized_mode
                                                        / dataset_seg,
                                                        bs_seg=bs_seg,
                                                    )
                                                    / lr_seg
                                                    / steps_seg
                                                ),
                                            )
                                            for s in candidate_seeds
                                        ]
                                        if (
                                            group_by is not None
                                            and group_by != "num_lookforward_tokens"
                                        ):
                                            study_group_values[name] = _format_group_value(
                                                axis=group_by, value=gv,
                                            )
        if not grouped:
            return None
        return cls.from_grouped(
            grouped, group_by=group_by, study_group_values=study_group_values,
        )

    @classmethod
    def from_maxrl_sweep(
        cls,
        *,
        artifacts_root: Path,
        subtract_baseline: bool,
        use_factorized_likelihoods: bool,
        num_lookforward_tokens: int | list[int] = 1,
        num_samples: int | list[int] = 100_000,
        batch_size: int | list[int] = 64,
        gaussian_stdev: float | list[float] = 1.0,
        label_type: Literal["rademacher", "token_id"] = "rademacher",
        normalize_labels: bool = False,
        label_range: tuple[float, float] = (0.0, 1.0),
        lr_per_sample: float | list[float] = 1e-5,
        lr_schedule: Literal["flat", "cosine"] = "flat",
        warmup_ratio: float = 0.05,
        lr_min_ratio: float = 0.1,
        train_from_scratch: bool | list[bool] = False,
        train_steps: int | list[int] = 10_000,
    ) -> Self | None:
        """`<artifacts_root>/maxrl/<from_*>/seed-{S}/rollouts-{N}/sigma-{σ}/{baseline_mode}/{likelihood_mode}/{dataset_folder}/[bs-{B}/]lr_{lr}/steps-{N:06d}/[sched-...]/`.
        Both `subtract_baseline` and `use_factorized_likelihoods` are fixed per
        call — surface them in the figure title.

        Group-axis semantics match `from_grpo_sweep`."""
        group_by, _ = _resolve_group_axis(
            candidates={
                "num_lookforward_tokens": num_lookforward_tokens,
                "num_samples": num_samples,
                "batch_size": batch_size,
                "lr_per_sample": lr_per_sample,
                "train_from_scratch": train_from_scratch,
                "gaussian_stdev": gaussian_stdev,
                "train_steps": train_steps,
            }
        )
        baseline = baseline_mode_folder(subtract_baseline=subtract_baseline)
        likelihood = likelihood_mode_folder(
            use_factorized_likelihoods=use_factorized_likelihoods,
        )
        attach_sched = _make_attach_sched(
            lr_schedule=lr_schedule,
            warmup_ratio=warmup_ratio,
            lr_min_ratio=lr_min_ratio,
        )
        grouped: dict[str, list[tuple[int, Path]]] = {}
        study_group_values: dict[str, str] = {}
        for n in _as_list(num_lookforward_tokens):
            for ns in _as_list(num_samples):
                for bs in _as_list(batch_size):
                    for lr in _as_list(lr_per_sample):
                        for scratch in _as_list(train_from_scratch):
                            for sigma_v in _as_list(gaussian_stdev):
                                for ts in _as_list(train_steps):
                                    study_base = artifacts_root / _method_dir(
                                        "maxrl", train_from_scratch=scratch,
                                    )
                                    if not study_base.exists():
                                        continue
                                    sigma = sigma_folder(gaussian_stdev=sigma_v)
                                    lr_seg = f"lr_{lr:.2e}"
                                    steps_seg = f"steps-{int(ts):06d}"
                                    bs_seg = _attach_bs_segment(batch_size=int(bs))
                                    dataset_seg = canonical_dataset_folder_name(
                                        num_lookforward_tokens=n,
                                        num_samples=ns,
                                        label_type=label_type,
                                        normalize_labels=normalize_labels,
                                        label_range=label_range,
                                    )
                                    gv = _select_group_value(
                                        group_by=group_by,
                                        num_lookforward_tokens=n, num_samples=ns,
                                        batch_size=bs,
                                        lr_per_sample=lr, train_from_scratch=scratch,
                                        gaussian_stdev=sigma_v, train_steps=ts,
                                    )
                                    for r in candidate_rollout_steps:
                                        name = _study_name(
                                            num_lookforward=n, group_by=group_by,
                                            group_value=gv, rollouts=r,
                                        )
                                        grouped[name] = [
                                            (
                                                s,
                                                attach_sched(
                                                    _splice_bs(
                                                        path=study_base
                                                        / _seed_folder_name(s)
                                                        / f"rollouts-{r}"
                                                        / sigma
                                                        / baseline
                                                        / likelihood
                                                        / dataset_seg,
                                                        bs_seg=bs_seg,
                                                    )
                                                    / lr_seg
                                                    / steps_seg
                                                ),
                                            )
                                            for s in candidate_seeds
                                        ]
                                        if (
                                            group_by is not None
                                            and group_by != "num_lookforward_tokens"
                                        ):
                                            study_group_values[name] = _format_group_value(
                                                axis=group_by, value=gv,
                                            )
        if not grouped:
            return None
        return cls.from_grouped(
            grouped, group_by=group_by, study_group_values=study_group_values,
        )

    def describe(self, label: str) -> None:
        n_runs = sum(len(v) for v in self.studies.values())
        n_groups = len(self.studies)
        max_seeds = max(len(v) for v in self.studies.values())
        print(
            f"{label:<32s}  {n_runs:>4d} runs   {n_groups:>3d} groups   "
            f"up to {max_seeds} seeds/group"
        )

    def summarize(self, *, metric: Literal["corr", "mse"]) -> pl.DataFrame:
        """Per-study best-step summary across seeds, sorted by lookforward.

        "Best" is picked by `val_{metric}` (argmax for corr, argmin for mse) —
        same convention as `plot_vs_lookforward(metric=...)`. Train statistics
        are reported at that same best-by-val step (not separately optimized),
        so train and val rows describe the same model checkpoint. NTP baseline
        has only one step (=0); rows then trivially correspond to that step.

        Per-study `pred_std`, `target_std`, and `pred_std_ratio` are reported
        alongside the primary metric whenever the underlying parquet carries
        the `*_pred_sum` / `*_target_sum` sufficient-stat columns. The ratio
        is the regression-collapse signature: ratio → 0 means the model is
        outputting a near-constant prediction (variance shrinking toward zero
        relative to the target). Old parquets without those columns silently
        omit the std/ratio fields rather than raising.

        If the loaded DataFrame has no `train_{metric}` column (trained methods
        whose `val_metrics.parquet` doesn't carry train-window sufficient
        stats), the train column is omitted from the output rather than
        raising. NTP baseline's `val_metrics.parquet` row carries both prefixes,
        so it continues to report a `train_{metric}_mean`."""
        df = self.get_metric_dataframe()
        higher_is_better = metric == "corr"
        val_y = f"val_{metric}"
        train_y = f"train_{metric}"
        has_train = train_y in df.columns

        val_agg = self._aggregate_by_step(df=df, y_name=val_y)
        train_agg = (
            self._aggregate_by_step(df=df, y_name=train_y) if has_train else None
        )

        # Optional collapse-diagnostic columns. Each emits seed-mean per
        # (study, step); we then look them up at the same best-by-val step
        # the primary metric was selected at.
        std_aggs: dict[str, pl.DataFrame] = {}
        for split in ("train", "val"):
            for suffix in (
                # "pred_std", 
                # "target_std", 
                "pred_std_ratio",
                ):
                col = f"{split}_{suffix}"
                if col in df.columns:
                    std_aggs[col] = self._aggregate_by_step(df=df, y_name=col)

        rows: list[dict] = []
        for study in self.studies:
            v = val_agg.filter(pl.col("study") == study)
            if v.is_empty():
                continue
            best = v.sort("mean_y", descending=higher_is_better).head(1)
            best_step = int(best["step"].item())
            row: dict = {
                "study": study,
                "num_lookforward": self.study_lookforwards[study],
                "best_step": best_step,
                "n_seeds": int(best["n_seeds"].item()),
            }
            if train_agg is not None:
                t = train_agg.filter(
                    (pl.col("study") == study) & (pl.col("step") == best_step)
                )
                if t.is_empty():
                    raise ValueError(
                        f"train aggregation missing for study {study!r} step={best_step}; "
                        "val_metrics.parquet is internally inconsistent"
                    )
                row[f"train_{metric}_mean"] = float(t["mean_y"].item())
            row[f"val_{metric}_mean"] = float(best["mean_y"].item())
            for col, agg in std_aggs.items():
                hit = agg.filter(
                    (pl.col("study") == study) & (pl.col("step") == best_step)
                )
                if hit.is_empty():
                    row[f"{col}_mean"] = None
                else:
                    val = hit["mean_y"].item()
                    row[f"{col}_mean"] = (
                        float(val) if val is not None else None
                    )
            rows.append(row)
        return pl.DataFrame(rows).sort(["num_lookforward", "study"])

    def get_metric_dataframe(self) -> pl.DataFrame:
        """Concat per-(study, seed) `val_metrics.parquet` frames, then append
        scalar `val_corr` / `val_mse` (and `train_corr` / `train_mse` when
        train-window sufficient stats are present) columns. Returned frame
        always has a `step` column, plus a `batch_size` column read from
        each study's `config.json`."""
        frames: list[pl.DataFrame] = []
        for name, paths in self.studies.items():
            seeds = self.study_seeds[name]
            bs = self.study_batch_sizes[name]
            for seed, path in zip(seeds, paths, strict=True):
                df = pl.read_parquet(path / "val_metrics.parquet")
                frames.append(
                    df.with_columns(
                        pl.lit(name).alias("study"),
                        pl.lit(seed).alias("seed"),
                        pl.lit(bs, dtype=pl.Int64).alias("batch_size"),
                    )
                )
        return _decode_dim_averaged(pl.concat(frames))

    def get_train_dataframe(self) -> pl.DataFrame:
        """Concat per-(study, seed) `train_metrics.parquet` files.

        Returns frame with columns: study, seed, batch_size, step, loss, mse,
        corr, pred_var, target_var, lr. Parquets that predate any of these
        columns get them synthesized as null so concat schemas line up:
        `corr` (added in the per-step-logging migration), `pred_var` /
        `target_var` (added with the prediction/target sufficient-stat
        columns), and `lr` (added with the linear-warmup + cosine LR
        scheduler). Without these backfills, mixing old + new parquets in
        a single sweep raises `polars.exceptions.ShapeError`."""
        backfill_cols = ("corr", "pred_var", "target_var", "lr")
        frames: list[pl.DataFrame] = []
        for name, paths in self.studies.items():
            seeds = self.study_seeds[name]
            bs = self.study_batch_sizes[name]
            for seed, path in zip(seeds, paths, strict=True):
                train_path = path / "train_metrics.parquet"
                if not train_path.exists():
                    continue
                df = pl.read_parquet(train_path)
                missing = [c for c in backfill_cols if c not in df.columns]
                if missing:
                    df = df.with_columns(
                        [pl.lit(None, dtype=pl.Float64).alias(c) for c in missing]
                    )
                frames.append(
                    df.with_columns(
                        pl.lit(name).alias("study"),
                        pl.lit(seed).alias("seed"),
                        pl.lit(bs, dtype=pl.Int64).alias("batch_size"),
                    )
                )
        if not frames:
            warnings.warn(
                "No train_metrics.parquet found across studies; "
                "returning an empty schema-only frame.",
                stacklevel=2,
            )
            return pl.DataFrame(
                schema={
                    "study": pl.Utf8,
                    "seed": pl.Int64,
                    "batch_size": pl.Int64,
                    "step": pl.Int64,
                    "loss": pl.Float64,
                    "mse": pl.Float64,
                    "corr": pl.Float64,
                    "pred_var": pl.Float64,
                    "target_var": pl.Float64,
                    "lr": pl.Float64,
                }
            )
        return pl.concat(frames)

    def _rollouts_groups(self) -> dict[int, list[str]] | None:
        """Split studies by ` r=(\\d+)` suffix; None if any study lacks it
        (i.e. SL sweeps)."""
        groups: dict[int, list[str]] = {}
        for name in self.studies:
            m = _ROLLOUTS_RE.search(name)
            if m is None:
                return None
            groups.setdefault(int(m.group(1)), []).append(name)
        return dict(sorted(groups.items()))

    def _aggregate_by_step(self, *, df: pl.DataFrame, y_name: str) -> pl.DataFrame:
        """Aggregate across seeds by (study, step). Input frame must already
        have a `step` column (`get_metric_dataframe()` guarantees this)."""
        return (
            df
            .group_by(["study", "step"])
            .agg(
                pl.col(y_name).mean().alias("mean_y"),
                pl.col(y_name).min().alias("min_y"),
                pl.col(y_name).max().alias("max_y"),
                pl.len().alias("n_seeds"),
            )
            .sort(["study", "step"])
        )

    @staticmethod
    def _seed_bar_error_kwargs(
        *, ys: list[float], mins: list[float], maxs: list[float]
    ) -> dict:
        return dict(
            error_y=dict(
                type="data",
                symmetric=False,
                array=[mx - my for mx, my in zip(maxs, ys, strict=True)],
                arrayminus=[my - mn for my, mn in zip(ys, mins, strict=True)],
                thickness=1,
                width=3,
            )
        )

    _DEFAULT_VISIBLE_ROLLOUTS: frozenset[int] = frozenset({4, 128, 1024})

    @staticmethod
    def _apply_compact_layout(fig: go.Figure, *, has_title: bool) -> None:
        # Right margin is explicit so the legend doesn't get clipped when the
        # figure is first rendered hidden inside a collapsed callout.
        fig.update_layout(
            height=320,
            margin=dict(l=50, r=105, t=40 if has_title else 15, b=40),
            title=dict(font=dict(size=13), x=0.02, xanchor="left", y=0.98, yanchor="top")
            if has_title
            else None,
            font=dict(size=11),
            legend=dict(
                font=dict(size=10),
                x=1.01,
                xanchor="left",
                y=1.0,
                yanchor="top",
            ),
        )
        fig.update_xaxes(title_font=dict(size=11), tickfont=dict(size=10))
        fig.update_yaxes(title_font=dict(size=11), tickfont=dict(size=10))

    @staticmethod
    def _save_html(fig: go.Figure, save_path: Path) -> None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(
            save_path,
            include_plotlyjs="cdn",
            full_html=True,
            config={"responsive": True},
        )

    def _study_styling(self) -> tuple[
        dict[str, str], dict[str, str], dict[str, str], bool, set[str] | None,
    ]:
        """Visual style derivation. Returns
        `(study_colors, study_legendgroups, study_legend_names,
          show_group_title, default_visible_groups)`.

        `default_visible_groups`: a set of legendgroup names that should be
        visible-by-default at first render, or `None` to mean "no group-level
        filter" (i.e. only the rollouts filter applies, when present).

        Three regimes:

        1. `group_by ∈ {None, "num_lookforward_tokens"}` reproduces today's
           behaviour exactly:
             - SL: each study gets a unique colour, no group title, all visible.
             - RL: colour by rollouts, legendgroup `look=N`, default-visible =
               just the min-look group (so partial sweeps still show something).

        2. Custom group axis on a non-rollouts (SL/SL-CE/NTP) config: colour by
           the group value, no nested group title.

        3. Custom group axis on a rollouts config: colour by rollouts (preserved
           from today), legendgroup is the formatted group value. All group
           values are visible-by-default; the rollouts filter
           (`_DEFAULT_VISIBLE_ROLLOUTS`) still narrows what's drawn within each
           group.
        """
        rollouts = self._rollouts_groups()
        palette = qualitative.Plotly
        custom_axis = (
            self.group_by is not None
            and self.group_by != "num_lookforward_tokens"
        )

        if not custom_axis:
            # Today's behaviour preserved verbatim.
            if rollouts is None:
                study_colors = {
                    s: palette[i % len(palette)] for i, s in enumerate(self.studies)
                }
                study_legendgroups = {s: s for s in self.studies}
                study_legend_names = {s: s for s in self.studies}
                return study_colors, study_legendgroups, study_legend_names, False, None
            rollouts_colors = {
                r: palette[i % len(palette)] for i, r in enumerate(rollouts.keys())
            }
            study_colors_rl: dict[str, str] = {}
            study_legendgroups_rl: dict[str, str] = {}
            study_legend_names_rl: dict[str, str] = {}
            for r, names in rollouts.items():
                for n in names:
                    study_colors_rl[n] = rollouts_colors[r]
                    study_legendgroups_rl[n] = f"look={self.study_lookforwards[n]}"
                    study_legend_names_rl[n] = f"r={r}"
            min_look = min(self.study_lookforwards.values())
            return (
                study_colors_rl,
                study_legendgroups_rl,
                study_legend_names_rl,
                True,
                {f"look={min_look}"},
            )

        # Custom group axis. Studies are tagged with their formatted group
        # value via `study_group_values`. Format the legendgroup as
        # "{axis_short}={value}" so the legend reads identically to the
        # study-name suffix.
        axis_short = _AXIS_SHORT[self.group_by]
        legendgroup_label: dict[str, str] = {
            s: f"{axis_short}={self.study_group_values[s]}"
            for s in self.studies
        }

        if rollouts is None:
            # SL with custom axis: one trace per group value. Colour by
            # group value, no nested group title.
            unique_groups = list(dict.fromkeys(legendgroup_label[s] for s in self.studies))
            group_colors = {
                g: palette[i % len(palette)] for i, g in enumerate(unique_groups)
            }
            study_colors_sl = {s: group_colors[legendgroup_label[s]] for s in self.studies}
            study_legendgroups_sl = {s: legendgroup_label[s] for s in self.studies}
            study_legend_names_sl = {s: legendgroup_label[s] for s in self.studies}
            return study_colors_sl, study_legendgroups_sl, study_legend_names_sl, False, None

        # RL with custom axis: colour by rollouts (today's behaviour),
        # legendgroup by group value, default-visible = all group values
        # crossed with `_DEFAULT_VISIBLE_ROLLOUTS` (handled in is_default_visible).
        rollouts_colors = {
            r: palette[i % len(palette)] for i, r in enumerate(rollouts.keys())
        }
        study_colors_rl: dict[str, str] = {}
        study_legendgroups_rl: dict[str, str] = {}
        study_legend_names_rl: dict[str, str] = {}
        for r, names in rollouts.items():
            for n in names:
                study_colors_rl[n] = rollouts_colors[r]
                study_legendgroups_rl[n] = legendgroup_label[n]
                study_legend_names_rl[n] = f"r={r}"
        return (
            study_colors_rl,
            study_legendgroups_rl,
            study_legend_names_rl,
            True,
            None,  # all group values default-visible; rollouts filter handles the rest
        )

    def plot_vs_eval(
        self,
        metric: Literal["corr", "mse"],
        *,
        num_lookforward_tokens: int | None = None,
        title: str | None = None,
        show_seed_bar: bool = False,
        save_path: Path | None = None,
    ) -> go.Figure:
        """Two panels (train left, val right) of `{train,val}_{metric}` per
        validation event, x-axis = training step. One line per study
        (seed-mean). Rollout sweeps default-show the min-look group with
        rollouts in `_DEFAULT_VISIBLE_ROLLOUTS`.

        When `num_lookforward_tokens` is provided, only studies with that
        lookforward value are plotted (single-look view). When `None`, all
        studies in `self.studies` are plotted (today's behaviour).

        When `train_{metric}` is missing from the loaded DataFrame (trained
        methods that don't write the train-window aggregate), the figure
        collapses to a single val panel.

        For `metric="mse"` the y-axis plots the variance-normalized MSE
        (`{split}_mse / {split}_target_var`, the fraction of target variance
        unexplained — 1 is the constant-mean baseline, 0 is perfect) and the
        y-range is hard-capped at `[0, 1]` so early-training spikes don't
        dominate the display."""
        df = self.get_metric_dataframe()
        # Normalized-MSE eval rendering: divide each row's MSE by the
        # per-eval target variance so the y-axis reads as a unitless ratio,
        # then cap at 1. We swap the column name in place so the rest of the
        # plotting plumbing (aggregation, hover labels, axis titles) flows
        # through unchanged.
        is_normalized_mse = metric == "mse"
        metric_col = "normalized_mse" if is_normalized_mse else metric
        if is_normalized_mse and f"val_{metric_col}" not in df.columns:
            raise ValueError(
                "plot_vs_eval(metric='mse') requires the val_normalized_mse "
                "column (MSE / target_var). The loaded parquets are missing "
                "the *_target_sum sufficient stats; re-run the eval to emit them."
            )
        rollouts = self._rollouts_groups()
        (
            study_colors,
            study_legendgroups,
            study_legend_names,
            show_group_title,
            default_visible_groups,
        ) = self._study_styling()

        studies_to_plot = _filter_studies_by_lookforward(
            studies=self.studies,
            study_lookforwards=self.study_lookforwards,
            num_lookforward_tokens=num_lookforward_tokens,
        )

        def is_default_visible(study: str) -> bool:
            # SL / no-rollouts: nothing to filter rollouts-wise.
            if rollouts is None:
                if default_visible_groups is None:
                    return True
                return study_legendgroups[study] in default_visible_groups
            # RL: rollouts filter always applies.
            r = int(_ROLLOUTS_RE.search(study).group(1))
            if r not in self._DEFAULT_VISIBLE_ROLLOUTS:
                return False
            if default_visible_groups is None:
                return True
            return study_legendgroups[study] in default_visible_groups

        splits: tuple[str, ...] = (
            ("train", "val") if f"train_{metric_col}" in df.columns else ("val",)
        )
        fig = make_subplots(rows=1, cols=len(splits), horizontal_spacing=0.08)
        for col, split in enumerate(splits, start=1):
            y_name = f"{split}_{metric_col}"
            agg_df = self._aggregate_by_step(df=df, y_name=y_name)
            for study in studies_to_plot:
                sub = agg_df.filter(pl.col("study") == study).sort("step")
                if sub.is_empty():
                    continue
                look = self.study_lookforwards[study]
                steps = sub["step"].to_list()
                ys = sub["mean_y"].to_list()
                mins = sub["min_y"].to_list()
                maxs = sub["max_y"].to_list()
                n_seeds = sub["n_seeds"].to_list()
                customdata = [[look, n] for n in n_seeds]
                error_kwargs = (
                    self._seed_bar_error_kwargs(ys=ys, mins=mins, maxs=maxs)
                    if show_seed_bar
                    else {}
                )
                group_title_kwargs = (
                    dict(legendgrouptitle_text=study_legendgroups[study])
                    if show_group_title
                    else {}
                )
                visible_kwargs = (
                    {} if is_default_visible(study) else dict(visible="legendonly")
                )
                fig.add_trace(
                    go.Scatter(
                        x=steps,
                        y=ys,
                        mode="lines+markers",
                        name=study_legend_names[study],
                        legendgroup=study_legendgroups[study],
                        showlegend=(col == 1),
                        **group_title_kwargs,
                        **visible_kwargs,
                        line=dict(color=study_colors[study]),
                        marker=dict(color=study_colors[study]),
                        customdata=customdata,
                        hovertemplate=(
                            "step: %{x}<br>"
                            f"{y_name}: %{{y}}<br>"
                            "num_lookforward_tokens: %{customdata[0]}<br>"
                            "n_seeds: %{customdata[1]}"
                            f"<extra>{study}</extra>"
                        ),
                        **error_kwargs,
                    ),
                    row=1,
                    col=col,
                )
            fig.update_xaxes(title_text="step", row=1, col=col)
            fig.update_yaxes(title_text=y_name, row=1, col=col)
            # if is_normalized_mse:
            #     # Hard-cap the y-axis at 1; users can autoscale to see beyond.
            #     fig.update_yaxes(range=[0, 1], row=1, col=col)
        if rollouts is not None:
            fig.update_layout(legend=dict(groupclick="togglegroup"))
        if title is not None:
            fig.update_layout(title=title)
        self._apply_compact_layout(fig, has_title=title is not None)
        if save_path is not None:
            self._save_html(fig, save_path)
        return fig

    def plot_vs_step(
        self,
        metric: Literal["loss", "mse"],
        *,
        num_lookforward_tokens: int | None = None,
        title: str | None = None,
        show_seed_bar: bool = False,
        y_clip_quantile: float | None = 0.99,
        save_path: Path | None = None,
    ) -> go.Figure:
        """Per-step train metrics (loss or mse) as continuous lines, one per
        study (seed-mean). Requires new-format study folders with
        train_metrics.parquet.

        When `num_lookforward_tokens` is provided, only studies with that
        lookforward value are plotted (single-look view). When `None`, all
        studies are plotted (today's behaviour).

        Early-training spikes routinely blow up the y-axis. When
        `y_clip_quantile` is not None (default 0.99), the y-axis autorange
        is clipped on both sides via Plotly `autorangeoptions.clipmin`/
        `clipmax`: the lower bound is capped at the `(1 - y_clip_quantile)`
        quantile of the seed-mean ys pooled across all plotted studies and
        the upper bound at the `y_clip_quantile` quantile, with 10%
        headroom on each side scaled by the inter-quantile span. User
        zoom and the autoscale button still work. Pass `None` to disable."""
        df = self.get_train_dataframe()
        if df.is_empty():
            raise ValueError("No train_metrics.parquet found; cannot plot_vs_step")
        rollouts = self._rollouts_groups()
        (
            study_colors,
            study_legendgroups,
            study_legend_names,
            show_group_title,
            default_visible_groups,
        ) = self._study_styling()

        studies_to_plot = _filter_studies_by_lookforward(
            studies=self.studies,
            study_lookforwards=self.study_lookforwards,
            num_lookforward_tokens=num_lookforward_tokens,
        )

        def is_default_visible(study: str) -> bool:
            if rollouts is None:
                if default_visible_groups is None:
                    return True
                return study_legendgroups[study] in default_visible_groups
            r = int(_ROLLOUTS_RE.search(study).group(1))
            if r not in self._DEFAULT_VISIBLE_ROLLOUTS:
                return False
            if default_visible_groups is None:
                return True
            return study_legendgroups[study] in default_visible_groups

        fig = go.Figure()
        agg_df = (
            df
            .group_by(["study", "step"])
            .agg(
                pl.col(metric).mean().alias("mean_y"),
                pl.col(metric).min().alias("min_y"),
                pl.col(metric).max().alias("max_y"),
                pl.len().alias("n_seeds"),
            )
            .sort(["study", "step"])
        )
        for study in studies_to_plot:
            sub = agg_df.filter(pl.col("study") == study).sort("step")
            if sub.is_empty():
                continue
            look = self.study_lookforwards[study]
            steps = sub["step"].to_list()
            ys = sub["mean_y"].to_list()
            mins = sub["min_y"].to_list()
            maxs = sub["max_y"].to_list()
            n_seeds = sub["n_seeds"].to_list()
            customdata = [[look, n] for n in n_seeds]
            error_kwargs = (
                self._seed_bar_error_kwargs(ys=ys, mins=mins, maxs=maxs)
                if show_seed_bar
                else {}
            )
            group_title_kwargs = (
                dict(legendgrouptitle_text=study_legendgroups[study])
                if show_group_title
                else {}
            )
            visible_kwargs = (
                {} if is_default_visible(study) else dict(visible="legendonly")
            )
            fig.add_trace(
                go.Scatter(
                    x=steps,
                    y=ys,
                    mode="lines",
                    name=study_legend_names[study],
                    legendgroup=study_legendgroups[study],
                    **group_title_kwargs,
                    **visible_kwargs,
                    line=dict(color=study_colors[study]),
                    customdata=customdata,
                    hovertemplate=(
                        "step: %{x}<br>"
                        f"{metric}: %{{y}}<br>"
                        "num_lookforward_tokens: %{customdata[0]}<br>"
                        "n_seeds: %{customdata[1]}"
                        f"<extra>{study}</extra>"
                    ),
                    **error_kwargs,
                ),
            )
        fig.update_xaxes(title_text="step")
        fig.update_yaxes(title_text=metric)
        if y_clip_quantile is not None and not agg_df.is_empty():
            means = agg_df["mean_y"]
            q_hi = means.quantile(y_clip_quantile)
            q_lo = means.quantile(1.0 - y_clip_quantile)
            if q_hi is not None and q_lo is not None and q_hi > q_lo:
                pad = 0.10 * (q_hi - q_lo)
                fig.update_yaxes(
                    autorangeoptions=dict(
                        clipmin=q_lo - pad,
                        clipmax=q_hi + pad,
                    )
                )
        if rollouts is not None:
            fig.update_layout(legend=dict(groupclick="togglegroup"))
        if title is not None:
            fig.update_layout(title=title)
        self._apply_compact_layout(fig, has_title=title is not None)
        if save_path is not None:
            self._save_html(fig, save_path)
        return fig

    def plot_vs_lookforward(
        self,
        *,
        metric: Literal["corr", "mse"] = "corr",
        title: str | None = None,
        x_scale: Literal["log", "uniform"] = "uniform",
        show_seed_bar: bool = False,
        save_path: Path | None = None,
    ) -> go.Figure:
        """Best-step `train_{metric}` / `val_{metric}` vs
        `num_lookforward_tokens`, two panels. "Best" is argmax over seed-mean
        for `metric="corr"` and argmin for `metric="mse"`. Rollout sweeps draw
        one curve per rollouts value; only `_DEFAULT_VISIBLE_ROLLOUTS` start
        visible.

        Requires `group_by ∈ {None, "num_lookforward_tokens"}` — i.e. the
        config must have lookforward as the only swept axis (or no swept axis
        at all). Custom group axes are not yet supported here; build a
        separate config to use them.

        When `train_{metric}` is missing from the loaded DataFrame the figure
        collapses to a single val panel."""
        if self.group_by not in (None, "num_lookforward_tokens"):
            raise NotImplementedError(
                f"plot_vs_lookforward is not supported with group_by="
                f"{self.group_by!r}; build a separate config without that "
                "group axis (or sweep num_lookforward_tokens instead)."
            )
        df = self.get_metric_dataframe()
        rollouts = self._rollouts_groups()
        palette = qualitative.Plotly
        higher_is_better = metric == "corr"
        if rollouts is None:
            curve_groups: list[tuple[str, int | None, list[str]]] = [
                ("all", None, list(self.studies.keys()))
            ]
        else:
            curve_groups = [(f"r={r}", r, names) for r, names in rollouts.items()]

        splits: tuple[str, ...] = (
            ("train", "val") if f"train_{metric}" in df.columns else ("val",)
        )
        fig = make_subplots(rows=1, cols=len(splits), horizontal_spacing=0.08)
        for col, split in enumerate(splits, start=1):
            y_name = f"{split}_{metric}"
            agg_df = self._aggregate_by_step(df=df, y_name=y_name)
            for i, (curve_name, r_value, names) in enumerate(curve_groups):
                color = palette[i % len(palette)]
                visible_kwargs = (
                    dict(visible="legendonly")
                    if (
                        r_value is not None
                        and r_value not in self._DEFAULT_VISIBLE_ROLLOUTS
                    )
                    else {}
                )
                rows = _best_step_rows_for_studies(
                    cfg=self, agg_df=agg_df, studies=names,
                    higher_is_better=higher_is_better,
                )
                if not rows:
                    continue
                looks = [r[0] for r in rows]
                xs = (
                    [candidate_lookforward_tokens.index(v) for v in looks]
                    if x_scale == "uniform"
                    else looks
                )
                ys = [r[1] for r in rows]
                mins = [r[2] for r in rows]
                maxs = [r[3] for r in rows]
                customdata = [(r[0], r[4], r[5], r[6]) for r in rows]
                error_kwargs = (
                    self._seed_bar_error_kwargs(ys=ys, mins=mins, maxs=maxs)
                    if show_seed_bar
                    else {}
                )
                fig.add_trace(
                    go.Scatter(
                        x=xs,
                        y=ys,
                        mode="lines+markers",
                        name=curve_name,
                        legendgroup=curve_name,
                        showlegend=(rollouts is not None) and col == 1,
                        **visible_kwargs,
                        line=dict(color=color),
                        marker=dict(color=color),
                        customdata=customdata,
                        hovertemplate=(
                            "num_lookforward_tokens: %{customdata[0]}<br>"
                            f"{y_name}: %{{y}}<br>"
                            "step: %{customdata[1]}<br>"
                            "n_seeds: %{customdata[2]}"
                            "<extra>%{customdata[3]}</extra>"
                        ),
                        **error_kwargs,
                    ),
                    row=1,
                    col=col,
                )
            fig.update_xaxes(title_text="num_lookforward_tokens", row=1, col=col)
            fig.update_yaxes(title_text=y_name, row=1, col=col)
            if x_scale == "log":
                fig.update_xaxes(type="log", row=1, col=col)
            else:
                fig.update_xaxes(
                    tickmode="array",
                    tickvals=list(range(len(candidate_lookforward_tokens))),
                    ticktext=[str(v) for v in candidate_lookforward_tokens],
                    row=1,
                    col=col,
                )
        if title is not None:
            fig.update_layout(title=title)
        self._apply_compact_layout(fig, has_title=title is not None)
        if save_path is not None:
            self._save_html(fig, save_path)
        return fig


# Cross-method comparison helpers.

# On-disk / programmatic method identifiers are lowercase with underscores so
# they're CLI-friendly and idiomatic Python; the displayed legend strings are
# uppercase via this map.
_METHOD_DISPLAY: dict[str, str] = {
    "sl_mse": "SL_MSE",
    "sl_ce": "SL_CE",
    "grpo": "GRPO",
    "maxrl": "MAXRL",
    "rloo": "RLOO",
    "pretrained_baseline": "PRETRAINED_BASELINE",
}

_METHOD_DASH: dict[str, str] = {
    "sl_mse": "solid",
    "sl_ce": "solid",
    "grpo": "dash",
    "maxrl": "dot",
    "rloo": "longdash",
    "pretrained_baseline": "dashdot",
}

_METHOD_COLORS: dict[str, str] = {
    "sl_mse": qualitative.Plotly[0],
    "grpo": qualitative.Plotly[1],
    "maxrl": qualitative.Plotly[2],
    "rloo": qualitative.Plotly[3],
    "pretrained_baseline": qualitative.Plotly[4],
    "sl_ce": qualitative.Plotly[5],
}

# Dash cycle ordered solid → most-broken so rising rollouts read as visually
# denser styles. Modular indexing handles the rare case of more than six
# rollouts values appearing in a single sweep.
_ROLLOUTS_DASH_CYCLE: tuple[str, ...] = (
    "solid", "dot", "dash", "longdash", "dashdot", "longdashdot",
)


def _has_train_col(df: pl.DataFrame | None, metric: str) -> bool:
    """True iff `df` contains the synthesized scalar `train_{metric}` column.
    `_decode_dim_averaged` only emits it when `train_target_*` sufficient
    stats were present in the underlying parquet — i.e. NTP-baseline
    old-format runs and new-format runs that include the train-window
    aggregate written by `state.py::_serialize_val`."""
    return df is not None and f"train_{metric}" in df.columns


def _best_step_rows_for_studies(
    *,
    cfg: CorpusRegressionAnalysisConfig,
    agg_df: pl.DataFrame,
    studies: list[str],
    higher_is_better: bool = True,
) -> list[tuple[int, float, float, float, int, int, str]]:
    """Per-study best-step row, sorted by lookforward. Uses argmax when
    `higher_is_better=True` (corr) and argmin when False (mse).
    Skips studies with no rows so partial sweeps render cleanly."""
    out: list[tuple[int, float, float, float, int, int, str]] = []
    for study in studies:
        sub = agg_df.filter(pl.col("study") == study)
        if sub.is_empty():
            continue
        best = sub.sort("mean_y", descending=higher_is_better).head(1)
        out.append((
            cfg.study_lookforwards[study],
            best["mean_y"].item(),
            best["min_y"].item(),
            best["max_y"].item(),
            best["step"].item(),
            best["n_seeds"].item(),
            study,
        ))
    out.sort(key=lambda r: r[0])
    return out


def plot_methods_vs_lookforward(
    *,
    sl_mse: CorpusRegressionAnalysisConfig | None = None,
    sl_ce: CorpusRegressionAnalysisConfig | None = None,
    grpo: CorpusRegressionAnalysisConfig | None = None,
    maxrl: CorpusRegressionAnalysisConfig | None = None,
    rloo: CorpusRegressionAnalysisConfig | None = None,
    pretrained_baseline: CorpusRegressionAnalysisConfig | None = None,
    metric: Literal["corr", "mse"] = "corr",
    title: str | None = None,
    x_scale: Literal["log", "uniform"] = "uniform",
    save_path: Path | None = None,
) -> go.Figure:
    """Cross-method best-step metric vs `num_lookforward_tokens`. Two panels
    (train, val). SL_MSE (when present) is a single standalone curve; GRPO,
    MaxRL and RLOO traces are grouped by rollouts — one legend group per
    `r=N`, containing one curve per RL method that ran that rollouts value.
    Color = method; dash = rollouts. Default-visible: SL_MSE plus the
    highest-rollouts group.

    `pretrained_baseline` (when present) is drawn as a standalone curve
    representing the intrinsic variance floor. `sl_ce` (when present) is
    drawn as a standalone curve representing the NTP cross-entropy
    supervised baseline."""
    methods_rl: list[tuple[str, CorpusRegressionAnalysisConfig]] = [
        (name, cfg)
        for name, cfg in (("grpo", grpo), ("maxrl", maxrl), ("rloo", rloo))
        if cfg is not None
    ]
    if (
        sl_mse is None
        and sl_ce is None
        and not methods_rl
        and pretrained_baseline is None
    ):
        raise ValueError(
            "at least one of sl_mse/sl_ce/grpo/maxrl/rloo/pretrained_baseline must be provided"
        )

    rollouts_seen: list[int] = []
    for _, cfg in methods_rl:
        rg = cfg._rollouts_groups()
        if rg is None:
            continue
        for r in rg:
            if r not in rollouts_seen:
                rollouts_seen.append(r)
    rollouts_seen.sort()
    dash_by_rollouts = {
        r: _ROLLOUTS_DASH_CYCLE[i % len(_ROLLOUTS_DASH_CYCLE)]
        for i, r in enumerate(rollouts_seen)
    }
    max_rollouts = rollouts_seen[-1] if rollouts_seen else None

    sl_mse_df = sl_mse.get_metric_dataframe() if sl_mse is not None else None
    sl_ce_df = sl_ce.get_metric_dataframe() if sl_ce is not None else None
    pretrained_df = pretrained_baseline.get_metric_dataframe() if pretrained_baseline is not None else None
    rl_dfs: dict[str, pl.DataFrame] = {
        name: cfg.get_metric_dataframe() for name, cfg in methods_rl
    }
    rl_groups: dict[str, dict[int, list[str]]] = {
        name: cfg._rollouts_groups() or {} for name, cfg in methods_rl
    }
    rl_cfgs: dict[str, CorpusRegressionAnalysisConfig] = dict(methods_rl)

    higher_is_better = metric == "corr"

    # Per the call's mix of old- and new-format dataframes, decide whether
    # the figure has a train panel at all, and which configs contribute on
    # the train side. New-format runs without `train_target_*` are silently
    # excluded from the train panel rather than raising.
    any_has_train = any(
        _has_train_col(d, metric)
        for d in (sl_mse_df, sl_ce_df, pretrained_df, *rl_dfs.values())
    )
    splits: tuple[str, ...] = ("train", "val") if any_has_train else ("val",)

    fig = make_subplots(rows=1, cols=len(splits), horizontal_spacing=0.08)
    for col, split in enumerate(splits, start=1):
        y_name = f"{split}_{metric}"

        if sl_mse is not None and (split == "val" or _has_train_col(sl_mse_df, metric)):
            sl_mse_agg = sl_mse._aggregate_by_step(df=sl_mse_df, y_name=y_name)
            sl_mse_rows = _best_step_rows_for_studies(
                cfg=sl_mse, agg_df=sl_mse_agg, studies=list(sl_mse.studies.keys()),
                higher_is_better=higher_is_better,
            )
            if sl_mse_rows:
                _add_methods_curve(
                    fig=fig,
                    rows=sl_mse_rows,
                    col=col,
                    trace_name=_METHOD_DISPLAY["sl_mse"],
                    legendgroup=_METHOD_DISPLAY["sl_mse"],
                    legendgrouptitle_text=None,
                    color=_METHOD_COLORS["sl_mse"],
                    dash="solid",
                    y_name=y_name,
                    x_scale=x_scale,
                    show_legend=(split == "val"),
                    visible_default=True,
                )

        if sl_ce is not None and (split == "val" or _has_train_col(sl_ce_df, metric)):
            sl_ce_agg = sl_ce._aggregate_by_step(df=sl_ce_df, y_name=y_name)
            sl_ce_rows = _best_step_rows_for_studies(
                cfg=sl_ce, agg_df=sl_ce_agg, studies=list(sl_ce.studies.keys()),
                higher_is_better=higher_is_better,
            )
            if sl_ce_rows:
                _add_methods_curve(
                    fig=fig,
                    rows=sl_ce_rows,
                    col=col,
                    trace_name=_METHOD_DISPLAY["sl_ce"],
                    legendgroup=_METHOD_DISPLAY["sl_ce"],
                    legendgrouptitle_text=None,
                    color=_METHOD_COLORS["sl_ce"],
                    dash=_METHOD_DASH["sl_ce"],
                    y_name=y_name,
                    x_scale=x_scale,
                    show_legend=(split == "val"),
                    visible_default=True,
                )

        if pretrained_baseline is not None and (
            split == "val" or _has_train_col(pretrained_df, metric)
        ):
            pretrained_agg = pretrained_baseline._aggregate_by_step(df=pretrained_df, y_name=y_name)
            pretrained_rows = _best_step_rows_for_studies(
                cfg=pretrained_baseline, agg_df=pretrained_agg, studies=list(pretrained_baseline.studies.keys()),
                higher_is_better=higher_is_better,
            )
            if pretrained_rows:
                _add_methods_curve(
                    fig=fig,
                    rows=pretrained_rows,
                    col=col,
                    trace_name=_METHOD_DISPLAY["pretrained_baseline"],
                    legendgroup=_METHOD_DISPLAY["pretrained_baseline"],
                    legendgrouptitle_text=None,
                    color=_METHOD_COLORS["pretrained_baseline"],
                    dash="solid",
                    y_name=y_name,
                    x_scale=x_scale,
                    show_legend=(split == "val"),
                    visible_default=True,
                )

        for r in rollouts_seen:
            for method_name, _ in methods_rl:
                if split == "train" and not _has_train_col(
                    rl_dfs[method_name], metric
                ):
                    continue
                names_for_r = rl_groups[method_name].get(r, [])
                if not names_for_r:
                    continue
                cfg = rl_cfgs[method_name]
                agg_df = cfg._aggregate_by_step(
                    df=rl_dfs[method_name], y_name=y_name,
                )
                rows = _best_step_rows_for_studies(
                    cfg=cfg, agg_df=agg_df, studies=names_for_r,
                    higher_is_better=higher_is_better,
                )
                if not rows:
                    continue
                _add_methods_curve(
                    fig=fig,
                    rows=rows,
                    col=col,
                    trace_name=_METHOD_DISPLAY[method_name],
                    legendgroup=f"r={r}",
                    legendgrouptitle_text=f"r={r}",
                    color=_METHOD_COLORS[method_name],
                    dash=dash_by_rollouts[r],
                    y_name=y_name,
                    x_scale=x_scale,
                    show_legend=(split == "val"),
                    visible_default=(r == max_rollouts),
                )

        fig.update_xaxes(title_text="num_lookforward_tokens", row=1, col=col)
        fig.update_yaxes(title_text=y_name, row=1, col=col)
        if x_scale == "log":
            fig.update_xaxes(type="log", row=1, col=col)
        else:
            fig.update_xaxes(
                tickmode="array",
                tickvals=list(range(len(candidate_lookforward_tokens))),
                ticktext=[str(v) for v in candidate_lookforward_tokens],
                row=1,
                col=col,
            )
    fig.update_layout(legend=dict(groupclick="togglegroup"))
    if title is not None:
        fig.update_layout(title=title)
    CorpusRegressionAnalysisConfig._apply_compact_layout(
        fig, has_title=title is not None
    )
    if save_path is not None:
        CorpusRegressionAnalysisConfig._save_html(fig, save_path)
    return fig


def _add_methods_curve(
    *,
    fig: go.Figure,
    rows: list[tuple[int, float, float, float, int, int, str]],
    col: int,
    trace_name: str,
    legendgroup: str,
    legendgrouptitle_text: str | None,
    color: str,
    dash: str,
    y_name: str,
    x_scale: Literal["log", "uniform"],
    show_legend: bool,
    visible_default: bool,
) -> None:
    looks = [r[0] for r in rows]
    xs = (
        [candidate_lookforward_tokens.index(v) for v in looks]
        if x_scale == "uniform"
        else looks
    )
    ys = [r[1] for r in rows]
    mins = [r[2] for r in rows]
    maxs = [r[3] for r in rows]
    customdata = [(r[0], r[4], r[5], r[6]) for r in rows]
    error_kwargs = CorpusRegressionAnalysisConfig._seed_bar_error_kwargs(
        ys=ys, mins=mins, maxs=maxs,
    )
    group_title_kwargs = (
        dict(legendgrouptitle_text=legendgrouptitle_text)
        if legendgrouptitle_text is not None
        else {}
    )
    visible_kwargs = {} if visible_default else dict(visible="legendonly")
    fig.add_trace(
        go.Scatter(
            x=xs,
            y=ys,
            mode="lines+markers",
            name=trace_name,
            legendgroup=legendgroup,
            **group_title_kwargs,
            showlegend=show_legend,
            line=dict(color=color, dash=dash),
            marker=dict(color=color),
            customdata=customdata,
            hovertemplate=(
                "num_lookforward_tokens: %{customdata[0]}<br>"
                f"{y_name}: %{{y}}<br>"
                "step: %{customdata[1]}<br>"
                "n_seeds: %{customdata[2]}"
                f"<extra>{trace_name} %{{customdata[3]}}</extra>"
            ),
            **visible_kwargs,
            **error_kwargs,
        ),
        row=1,
        col=col,
    )


def _add_eval_curve(
    *,
    fig: go.Figure,
    agg_df: pl.DataFrame,
    col: int,
    trace_name: str,
    legendgroup: str,
    legendgrouptitle_text: str | None,
    color: str,
    dash: str,
    y_name: str,
    show_legend: bool,
    visible_default: bool,
    show_seed_bar: bool,
) -> None:
    """Add a single per-eval Scatter trace to `fig`. `agg_df` must already be
    filtered to one study and sorted by step."""
    steps = agg_df["step"].to_list()
    ys = agg_df["mean_y"].to_list()
    mins = agg_df["min_y"].to_list()
    maxs = agg_df["max_y"].to_list()
    n_seeds = agg_df["n_seeds"].to_list()
    error_kwargs = (
        CorpusRegressionAnalysisConfig._seed_bar_error_kwargs(
            ys=ys, mins=mins, maxs=maxs
        )
        if show_seed_bar
        else {}
    )
    group_title_kwargs = (
        dict(legendgrouptitle_text=legendgrouptitle_text)
        if legendgrouptitle_text is not None
        else {}
    )
    visible_kwargs = {} if visible_default else dict(visible="legendonly")
    fig.add_trace(
        go.Scatter(
            x=steps,
            y=ys,
            mode="lines+markers",
            name=trace_name,
            legendgroup=legendgroup,
            **group_title_kwargs,
            showlegend=show_legend,
            line=dict(color=color, dash=dash),
            marker=dict(color=color),
            customdata=[[n] for n in n_seeds],
            hovertemplate=(
                "step: %{x}<br>"
                f"{y_name}: %{{y}}<br>"
                "n_seeds: %{customdata[0]}"
                f"<extra>{trace_name}</extra>"
            ),
            **visible_kwargs,
            **error_kwargs,
        ),
        row=1,
        col=col,
    )


def plot_methods_vs_eval(
    *,
    sl_mse: CorpusRegressionAnalysisConfig | None = None,
    sl_ce: CorpusRegressionAnalysisConfig | None = None,
    grpo: CorpusRegressionAnalysisConfig | None = None,
    maxrl: CorpusRegressionAnalysisConfig | None = None,
    rloo: CorpusRegressionAnalysisConfig | None = None,
    pretrained_baseline: CorpusRegressionAnalysisConfig | None = None,
    num_lookforward_tokens: int = 1,
    metric: Literal["corr", "mse"] = "corr",
    show_seed_bar: bool = False,
    title: str | None = None,
    save_path: Path | None = None,
) -> go.Figure:
    """Cross-method per-eval metric curves for a single `num_lookforward_tokens`
    value. Two panels (train, val); x-axis is training step at validation
    events. Color = method; dash = rollouts.

    For `metric="mse"` the y-axis plots the variance-normalized MSE
    (`{split}_mse / {split}_target_var`, the fraction of target variance
    unexplained — 1 is the constant-mean baseline, 0 is perfect) and the
    y-range is hard-capped at `[0, 1]`."""
    methods_rl: list[tuple[str, CorpusRegressionAnalysisConfig]] = [
        (name, cfg)
        for name, cfg in (("grpo", grpo), ("maxrl", maxrl), ("rloo", rloo))
        if cfg is not None
    ]
    if (
        sl_mse is None
        and sl_ce is None
        and not methods_rl
        and pretrained_baseline is None
    ):
        raise ValueError(
            "at least one of sl_mse/sl_ce/grpo/maxrl/rloo/pretrained_baseline must be provided"
        )

    # See `plot_vs_eval` — `metric="mse"` swaps to the variance-normalized
    # column so the y-axis reads as a unitless ratio capped at 1.
    is_normalized_mse = metric == "mse"
    metric_col = "normalized_mse" if is_normalized_mse else metric

    # Collect rollout values across RL methods.
    rollouts_seen: list[int] = []
    for _, cfg in methods_rl:
        rg = cfg._rollouts_groups()
        if rg is None:
            continue
        for r in rg:
            if r not in rollouts_seen:
                rollouts_seen.append(r)
    rollouts_seen.sort()
    dash_by_rollouts = {
        r: _ROLLOUTS_DASH_CYCLE[i % len(_ROLLOUTS_DASH_CYCLE)]
        for i, r in enumerate(rollouts_seen)
    }
    max_rollouts = rollouts_seen[-1] if rollouts_seen else None

    # Pre-load metric DataFrames.
    sl_mse_df = sl_mse.get_metric_dataframe() if sl_mse is not None else None
    sl_ce_df = sl_ce.get_metric_dataframe() if sl_ce is not None else None
    pretrained_df = pretrained_baseline.get_metric_dataframe() if pretrained_baseline is not None else None
    rl_dfs: dict[str, pl.DataFrame] = {
        name: cfg.get_metric_dataframe() for name, cfg in methods_rl
    }

    if is_normalized_mse:
        # All present DataFrames must carry the val_normalized_mse column;
        # otherwise the cross-method comparison would silently mix old and
        # new schemas.
        provided = {
            "sl_mse": sl_mse_df,
            "sl_ce": sl_ce_df,
            "pretrained_baseline": pretrained_df,
            **rl_dfs,
        }
        missing = [
            name
            for name, d in provided.items()
            if d is not None and f"val_{metric_col}" not in d.columns
        ]
        if missing:
            raise ValueError(
                "plot_methods_vs_eval(metric='mse') requires the "
                "val_normalized_mse column on every supplied method; missing "
                f"on: {sorted(missing)}. Re-run those evals to emit the "
                "*_target_sum sufficient stats."
            )

    any_has_train = any(
        _has_train_col(d, metric_col)
        for d in (sl_mse_df, sl_ce_df, pretrained_df, *rl_dfs.values())
    )
    splits: tuple[str, ...] = ("train", "val") if any_has_train else ("val",)
    fig = make_subplots(rows=1, cols=len(splits), horizontal_spacing=0.08)
    for col, split in enumerate(splits, start=1):
        y_name = f"{split}_{metric_col}"

        # SL_MSE (no rollouts): single study "look=N".
        if (
            sl_mse is not None
            and sl_mse_df is not None
            and (split == "val" or _has_train_col(sl_mse_df, metric_col))
        ):
            study_name = f"look={num_lookforward_tokens}"
            if study_name in sl_mse.studies:
                agg = sl_mse._aggregate_by_step(
                    df=sl_mse_df.filter(pl.col("study") == study_name), y_name=y_name
                )
                if not agg.is_empty():
                    _add_eval_curve(
                        fig=fig,
                        agg_df=agg,
                        col=col,
                        trace_name=_METHOD_DISPLAY["sl_mse"],
                        legendgroup=_METHOD_DISPLAY["sl_mse"],
                        legendgrouptitle_text=None,
                        color=_METHOD_COLORS["sl_mse"],
                        dash="solid",
                        y_name=y_name,
                        show_legend=(split == "val"),
                        visible_default=True,
                        show_seed_bar=show_seed_bar,
                    )

        # SL+NTP-CE (no rollouts): single study "look=N".
        if (
            sl_ce is not None
            and sl_ce_df is not None
            and (split == "val" or _has_train_col(sl_ce_df, metric_col))
        ):
            study_name = f"look={num_lookforward_tokens}"
            if study_name in sl_ce.studies:
                agg = sl_ce._aggregate_by_step(
                    df=sl_ce_df.filter(pl.col("study") == study_name),
                    y_name=y_name,
                )
                if not agg.is_empty():
                    _add_eval_curve(
                        fig=fig,
                        agg_df=agg,
                        col=col,
                        trace_name=_METHOD_DISPLAY["sl_ce"],
                        legendgroup=_METHOD_DISPLAY["sl_ce"],
                        legendgrouptitle_text=None,
                        color=_METHOD_COLORS["sl_ce"],
                        dash=_METHOD_DASH["sl_ce"],
                        y_name=y_name,
                        show_legend=(split == "val"),
                        visible_default=True,
                        show_seed_bar=show_seed_bar,
                    )

        # Pretrained-model intrinsic-variance baseline: single study "look=N".
        if (
            pretrained_baseline is not None
            and pretrained_df is not None
            and (split == "val" or _has_train_col(pretrained_df, metric_col))
        ):
            study_name = f"look={num_lookforward_tokens}"
            if study_name in pretrained_baseline.studies:
                agg = pretrained_baseline._aggregate_by_step(
                    df=pretrained_df.filter(pl.col("study") == study_name), y_name=y_name
                )
                if not agg.is_empty():
                    _add_eval_curve(
                        fig=fig,
                        agg_df=agg,
                        col=col,
                        trace_name=_METHOD_DISPLAY["pretrained_baseline"],
                        legendgroup=_METHOD_DISPLAY["pretrained_baseline"],
                        legendgrouptitle_text=None,
                        color=_METHOD_COLORS["pretrained_baseline"],
                        dash="solid",
                        y_name=y_name,
                        show_legend=(split == "val"),
                        visible_default=True,
                        show_seed_bar=show_seed_bar,
                    )

        # RL methods: one trace per (rollouts, method) pair.
        for r in rollouts_seen:
            for method_name, cfg in methods_rl:
                if split == "train" and not _has_train_col(
                    rl_dfs[method_name], metric_col
                ):
                    continue
                study_name = f"look={num_lookforward_tokens} r={r}"
                if study_name not in cfg.studies:
                    continue
                agg = cfg._aggregate_by_step(
                    df=rl_dfs[method_name].filter(pl.col("study") == study_name),
                    y_name=y_name,
                )
                if agg.is_empty():
                    continue
                _add_eval_curve(
                    fig=fig,
                    agg_df=agg,
                    col=col,
                    trace_name=_METHOD_DISPLAY[method_name],
                    legendgroup=f"r={r}",
                    legendgrouptitle_text=f"r={r}",
                    color=_METHOD_COLORS[method_name],
                    dash=dash_by_rollouts[r],
                    y_name=y_name,
                    show_legend=(split == "val"),
                    visible_default=(r == max_rollouts),
                    show_seed_bar=show_seed_bar,
                )

        fig.update_xaxes(title_text="step", row=1, col=col)
        fig.update_yaxes(title_text=y_name, row=1, col=col)
        # if is_normalized_mse:
        #     # Hard-cap the y-axis at 1; users can autoscale to see beyond.
        #     fig.update_yaxes(range=[0, 1], row=1, col=col)

    fig.update_layout(legend=dict(groupclick="togglegroup"))
    if title is not None:
        fig.update_layout(title=title)
    CorpusRegressionAnalysisConfig._apply_compact_layout(
        fig, has_title=title is not None
    )
    if save_path is not None:
        CorpusRegressionAnalysisConfig._save_html(fig, save_path)
    return fig


def plot_methods_pred_std_vs_eval(
    *,
    sl_mse: CorpusRegressionAnalysisConfig | None = None,
    sl_ce: CorpusRegressionAnalysisConfig | None = None,
    grpo: CorpusRegressionAnalysisConfig | None = None,
    maxrl: CorpusRegressionAnalysisConfig | None = None,
    rloo: CorpusRegressionAnalysisConfig | None = None,
    pretrained_baseline: CorpusRegressionAnalysisConfig | None = None,
    num_lookforward_tokens: int = 1,
    quantity: Literal["pred_std", "pred_std_ratio"] = "pred_std_ratio",
    show_seed_bar: bool = False,
    title: str | None = None,
    save_path: Path | None = None,
) -> go.Figure:
    """Cross-method collapse diagnostic per eval step. Plots either
    `pred_std` (raw prediction std) or `pred_std_ratio = pred_std/target_std`
    (collapse signature: ratio → 0 means model output collapsing toward a
    near-constant prediction).

    Mirrors `plot_methods_vs_eval`'s layout and style — two panels (train,
    val) with x-axis = training step, color = method, dash = rollouts. Old
    parquets without the `*_pred_sum` / `*_target_sum` columns silently lack
    the y-axis values and are excluded from the plot rather than raising.
    """
    methods_rl: list[tuple[str, CorpusRegressionAnalysisConfig]] = [
        (name, cfg)
        for name, cfg in (("grpo", grpo), ("maxrl", maxrl), ("rloo", rloo))
        if cfg is not None
    ]
    if (
        sl_mse is None
        and sl_ce is None
        and not methods_rl
        and pretrained_baseline is None
    ):
        raise ValueError(
            "at least one of sl_mse/sl_ce/grpo/maxrl/rloo/pretrained_baseline must be provided"
        )

    rollouts_seen: list[int] = []
    for _, cfg in methods_rl:
        rg = cfg._rollouts_groups()
        if rg is None:
            continue
        for r in rg:
            if r not in rollouts_seen:
                rollouts_seen.append(r)
    rollouts_seen.sort()
    dash_by_rollouts = {
        r: _ROLLOUTS_DASH_CYCLE[i % len(_ROLLOUTS_DASH_CYCLE)]
        for i, r in enumerate(rollouts_seen)
    }
    max_rollouts = rollouts_seen[-1] if rollouts_seen else None

    sl_mse_df = sl_mse.get_metric_dataframe() if sl_mse is not None else None
    sl_ce_df = sl_ce.get_metric_dataframe() if sl_ce is not None else None
    pretrained_df = pretrained_baseline.get_metric_dataframe() if pretrained_baseline is not None else None
    rl_dfs: dict[str, pl.DataFrame] = {
        name: cfg.get_metric_dataframe() for name, cfg in methods_rl
    }

    def has_quantity(df: pl.DataFrame | None, split: str) -> bool:
        return df is not None and f"{split}_{quantity}" in df.columns

    any_has_train = any(
        has_quantity(d, "train")
        for d in (sl_mse_df, sl_ce_df, pretrained_df, *rl_dfs.values())
    )
    splits: tuple[str, ...] = ("train", "val") if any_has_train else ("val",)
    fig = make_subplots(rows=1, cols=len(splits), horizontal_spacing=0.08)
    for col, split in enumerate(splits, start=1):
        y_name = f"{split}_{quantity}"

        if (
            sl_mse is not None
            and sl_mse_df is not None
            and has_quantity(sl_mse_df, split)
        ):
            study_name = f"look={num_lookforward_tokens}"
            if study_name in sl_mse.studies:
                agg = sl_mse._aggregate_by_step(
                    df=sl_mse_df.filter(pl.col("study") == study_name), y_name=y_name
                )
                if not agg.is_empty():
                    _add_eval_curve(
                        fig=fig,
                        agg_df=agg,
                        col=col,
                        trace_name=_METHOD_DISPLAY["sl_mse"],
                        legendgroup=_METHOD_DISPLAY["sl_mse"],
                        legendgrouptitle_text=None,
                        color=_METHOD_COLORS["sl_mse"],
                        dash="solid",
                        y_name=y_name,
                        show_legend=(split == "val"),
                        visible_default=True,
                        show_seed_bar=show_seed_bar,
                    )

        if (
            sl_ce is not None
            and sl_ce_df is not None
            and has_quantity(sl_ce_df, split)
        ):
            study_name = f"look={num_lookforward_tokens}"
            if study_name in sl_ce.studies:
                agg = sl_ce._aggregate_by_step(
                    df=sl_ce_df.filter(pl.col("study") == study_name),
                    y_name=y_name,
                )
                if not agg.is_empty():
                    _add_eval_curve(
                        fig=fig,
                        agg_df=agg,
                        col=col,
                        trace_name=_METHOD_DISPLAY["sl_ce"],
                        legendgroup=_METHOD_DISPLAY["sl_ce"],
                        legendgrouptitle_text=None,
                        color=_METHOD_COLORS["sl_ce"],
                        dash=_METHOD_DASH["sl_ce"],
                        y_name=y_name,
                        show_legend=(split == "val"),
                        visible_default=True,
                        show_seed_bar=show_seed_bar,
                    )

        if (
            pretrained_baseline is not None
            and pretrained_df is not None
            and has_quantity(pretrained_df, split)
        ):
            study_name = f"look={num_lookforward_tokens}"
            if study_name in pretrained_baseline.studies:
                agg = pretrained_baseline._aggregate_by_step(
                    df=pretrained_df.filter(pl.col("study") == study_name),
                    y_name=y_name,
                )
                if not agg.is_empty():
                    _add_eval_curve(
                        fig=fig,
                        agg_df=agg,
                        col=col,
                        trace_name=_METHOD_DISPLAY["pretrained_baseline"],
                        legendgroup=_METHOD_DISPLAY["pretrained_baseline"],
                        legendgrouptitle_text=None,
                        color=_METHOD_COLORS["pretrained_baseline"],
                        dash="solid",
                        y_name=y_name,
                        show_legend=(split == "val"),
                        visible_default=True,
                        show_seed_bar=show_seed_bar,
                    )

        for r in rollouts_seen:
            for method_name, cfg in methods_rl:
                if not has_quantity(rl_dfs[method_name], split):
                    continue
                study_name = f"look={num_lookforward_tokens} r={r}"
                if study_name not in cfg.studies:
                    continue
                agg = cfg._aggregate_by_step(
                    df=rl_dfs[method_name].filter(pl.col("study") == study_name),
                    y_name=y_name,
                )
                if agg.is_empty():
                    continue
                _add_eval_curve(
                    fig=fig,
                    agg_df=agg,
                    col=col,
                    trace_name=_METHOD_DISPLAY[method_name],
                    legendgroup=f"r={r}",
                    legendgrouptitle_text=f"r={r}",
                    color=_METHOD_COLORS[method_name],
                    dash=dash_by_rollouts[r],
                    y_name=y_name,
                    show_legend=(split == "val"),
                    visible_default=(r == max_rollouts),
                    show_seed_bar=show_seed_bar,
                )

        fig.update_xaxes(title_text="step", row=1, col=col)
        fig.update_yaxes(title_text=y_name, row=1, col=col)

    fig.update_layout(legend=dict(groupclick="togglegroup"))
    if title is not None:
        fig.update_layout(title=title)
    CorpusRegressionAnalysisConfig._apply_compact_layout(
        fig, has_title=title is not None
    )
    if save_path is not None:
        CorpusRegressionAnalysisConfig._save_html(fig, save_path)
    return fig
