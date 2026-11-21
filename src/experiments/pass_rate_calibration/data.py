import json
from pathlib import Path

import numpy as np
import polars as pl
from datasets import load_dataset

from src.experiments.pass_rate_calibration.config import (
    SOURCES,
    SPLIT_SEED,
    VALIDATION_PROBLEMS,
    DatasetSource,
)


def source_frame(*, source: DatasetSource) -> pl.DataFrame:
    rows = load_dataset(
        source.repository, revision=source.revision, split=source.split
    ).to_list()
    return pl.DataFrame({
        "problem_id": [f"{source.name}-{index:04d}" for index in range(len(rows))],
        "dataset": [source.name] * len(rows),
        "subject": [
            row["subject"] if source.name == "jeebench" else "minerva" for row in rows
        ],
        "problem": [row[source.question_field] for row in rows],
        "answer": [str(row["answer"]) for row in rows],
    })


def build_problems() -> pl.DataFrame:
    frame = pl.concat([source_frame(source=source) for source in SOURCES])
    minerva = frame.filter(pl.col("dataset") == "minerva")["problem_id"].to_list()
    order = np.random.default_rng(SPLIT_SEED).permutation(len(minerva))
    validation = [minerva[index] for index in order[:VALIDATION_PROBLEMS]]
    return frame.with_columns(
        split=pl
        .when(pl.col("dataset") == "jeebench")
        .then(pl.lit("test"))
        .when(pl.col("problem_id").is_in(validation))
        .then(pl.lit("val"))
        .otherwise(pl.lit("train"))
    )


def prepare_problems(*, root: Path) -> pl.DataFrame:
    root.mkdir(parents=True, exist_ok=True)
    frame = build_problems()
    path = root / "problems.parquet"
    if path.exists():
        if not pl.read_parquet(path).equals(frame):
            raise ValueError(f"{path} differs from the canonical problem table")
        return frame
    frame.write_parquet(path)
    return frame


def read_problems(*, root: Path) -> pl.DataFrame:
    return pl.read_parquet(root / "problems.parquet")


def read_labels(*, root: Path) -> pl.DataFrame:
    return pl.read_parquet(root / "labels.parquet")


def require_gate(*, root: Path) -> None:
    if not json.loads((root / "ground-truth.json").read_text())["gate_passed"]:
        raise RuntimeError(
            "Phase 0 pass@1 gate failed; reconcile the solver settings first"
        )
