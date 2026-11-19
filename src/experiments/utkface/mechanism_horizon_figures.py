import argparse
import itertools
import json
import math
from pathlib import Path

import numpy as np
import polars as pl
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from src.experiments.utkface.adaptation_train import file_sha256, git_head, source_root
from src.experiments.utkface.mechanism_dossier import DECISION_MATRIX_SHA256, require
from src.experiments.utkface.mechanism_horizon_report import (
    COMMIT,
    TIMES,
    VARIANTS,
    unique,
)
from src.experiments.utkface.mechanism_probe_report import READOUTS, checked_file
from src.experiments.utkface.mechanism_report import COLORS, LABELS, plt, read, save
from src.experiments.utkface.run import write_json

HORIZON_METHODS = ("scalar_mae", "categorical_tailrl", "binomial_ce", "binomial_tailrl")
TAIL_METHODS = ("binomial_tailrl_free", "binomial_tailrl_affine")
STYLES = {
    "scalar_mae": ("o", "-"),
    "categorical_tailrl": ("^", "--"),
    "binomial_ce": ("s", "-."),
    "binomial_tailrl": ("D", ":"),
    "binomial_tailrl_free": ("o", "-"),
    "binomial_tailrl_affine": ("s", "--"),
}
TABLES = ("individual", "summary", "paired_individual", "paired_summary")
FORMATS = ("png", "svg", "pdf")
SOURCE_FILES = (
    "src/experiments/utkface/mechanism_horizon_figures.py",
    "src/experiments/utkface/mechanism_horizon_report.py",
    "src/experiments/utkface/mechanism_report.py",
    "src/experiments/utkface/mechanism_dossier.py",
    "src/experiments/utkface/mechanism_probe_report.py",
    "src/experiments/utkface/adaptation_train.py",
    "src/experiments/utkface/run.py",
)
NOTE = (
    "Markers: mean of three paired joint blocks; bars: sample SD (ddof=1). "
    "Faint lines connect each block across fixed epochs. Ridge regularization uses clean-validation MSE. "
    "Epochs and readouts reuse the same fits."
)
CAPTIONS = {
    "learning-horizon": (
        "Clean-reference test age MAE in years at fixed epochs 50, 200 and 500 after scratch initialization. "
        "Columns show clean or fixed SD-10 training labels; rows show the native readout or standardized "
        "ridge fitted to supplied training labels. The four methods are scalar MAE, categorical TailRL, "
        "one-score binomial CE and one-score binomial TailRL. Markers and error bars show the mean and "
        "sample standard deviation across three paired joint training/corruption blocks; faint lines "
        "connect the same block across epochs. Each row shares its vertical scale across label conditions. "
        "Ridge regularization is selected by clean-validation MSE for each frozen encoder. "
        "The clean oracle equals supplied-label ridge and is represented once."
    ),
    "tail-free-affine": (
        "Clean-reference test age MAE in years under fixed SD-10 training-label corruption, "
        "at epochs 50, 200 and 500 after scratch initialization. Panels show the native readout, "
        "ridge fitted to supplied training labels and oracle ridge fitted to additional clean reference "
        "training labels for the free and age-affine TailRL residual heads. Markers and error bars show "
        "the mean and sample standard deviation across three paired joint blocks; faint lines connect "
        "the same block across epochs. Panels share their vertical scale. Each ridge fit selects its "
        "regularization by clean-validation MSE. Epochs and readouts reuse the same trained models."
    ),
}


def load_inputs(*, horizon_dir: Path) -> tuple[dict, dict, dict[str, str]]:
    horizon_dir = horizon_dir.resolve()
    report_path = horizon_dir / "report.json"
    report_hash = file_sha256(path=report_path)
    producer = read(report_path)
    require(
        condition=producer["schema"] == "utkface-mechanism-horizon-report-v1"
        and producer["scientific_complete"] is True
        and producer["completed_fits"] == 66
        and producer["completed_probe_sources"] == producer["noise_audits"] == 198
        and producer["n_blocks"] == 3
        and producer["source_commit"] == COMMIT,
        path=report_path,
        expectation="a scientifically complete E8 horizon producer with 66 fits, 198 probes and 198 noise audits",
    )
    require(
        condition=set(producer["tables"]) == set(TABLES)
        and bool(producer["input_sha256"]),
        path=report_path,
        expectation="all four producer tables and recorded upstream input hashes",
    )
    for name, expected in producer["input_sha256"].items():
        checked_file(path=Path(name), expected=expected)
    matrices = [
        name
        for name in producer["input_sha256"]
        if Path(name).name == "long-horizon-decision-matrix.txt"
    ]
    require(
        condition=len(matrices) == 1
        and producer["input_sha256"][matrices[0]] == DECISION_MATRIX_SHA256,
        path=report_path,
        expectation="the frozen long-horizon decision-matrix hash",
    )
    protected = {str(report_path): report_hash}
    frames = {}
    for name in TABLES:
        table = horizon_dir / f"{name.replace('_', '-')}.tsv"
        artifact = producer["tables"][name]
        require(
            condition=Path(artifact["path"]).resolve() == table
            and isinstance(artifact["rows"], int)
            and artifact["rows"] > 0,
            path=table,
            expectation="the producer's original table path and positive row count",
        )
        checked_file(path=table, expected=artifact["sha256"])
        frame = pl.read_csv(table, separator="\t", infer_schema_length=None)
        require(
            condition=frame.height == artifact["rows"],
            path=table,
            expectation="the producer's complete table row count",
        )
        if name in {"summary", "paired_summary"}:
            require(
                condition=all(
                    row["n_rows"] == 3
                    and 0 <= row["n_defined"] <= 3
                    and row["n_undefined"] == 3 - row["n_defined"]
                    for row in frame.select(
                        "n_rows", "n_defined", "n_undefined"
                    ).to_dicts()
                ),
                path=table,
                expectation="three retained blocks and explicit undefined counts in every summary cell",
            )
        frames[name] = frame
        protected[str(table)] = artifact["sha256"]
    plan_path = (Path(producer["training_root"]) / "plan.json").resolve()
    require(
        condition=producer["input_sha256"].get(str(plan_path))
        == file_sha256(path=plan_path),
        path=plan_path,
        expectation="the producer's hashed original training plan",
    )
    plan = read(plan_path)
    jobs = plan["jobs"]
    planned = {job["identifier"]: job for job in jobs}
    cells = {(job["variant"], job["block"], job["noise_sd"]) for job in jobs}
    require(
        condition=plan["git_commit"] == COMMIT
        and len(jobs) == len(planned) == 66
        and cells == set(itertools.product(VARIANTS, range(3), (0, 10))),
        path=plan_path,
        expectation="all 11 methods, two label conditions and three blocks from frozen E8",
    )
    rows = frames["individual"].to_dicts()
    unique(rows=rows, keys=("identifier", "statistic", "readout", "time", "kind"))
    for row in rows:
        job = planned[row["identifier"]]
        require(
            condition=all(
                row[key] == job[key] for key in ("variant", "block", "noise_sd")
            ),
            path=horizon_dir / "individual.tsv",
            expectation=f"the original plan identity for {row['identifier']}",
        )
    require(
        condition=not any(
            row["noise_sd"] == 0 and row["readout"] == "oracle_clean_labels"
            for row in rows
        ),
        path=horizon_dir / "individual.tsv",
        expectation="the producer's checked collapse of clean oracle rows into supplied-label ridge",
    )
    levels = [
        row
        for row in rows
        if row["statistic"] == "probe_test_mae" and row["kind"] == "level"
    ]
    values = unique(
        rows=levels, keys=("variant", "block", "noise_sd", "readout", "time")
    )
    expected = {
        (variant, block, noise, readout, f"e{epoch}")
        for variant, block, noise, readout, epoch in itertools.product(
            VARIANTS, range(3), (0, 10), READOUTS, TIMES
        )
        if noise != 0 or readout != "oracle_clean_labels"
    }
    require(
        condition=set(values) == expected
        and all(
            row["value"] is not None
            and math.isfinite(row["value"])
            and row["value"] >= 0
            for row in levels
        ),
        path=horizon_dir / "individual.tsv",
        expectation="all 495 finite fixed-epoch native/supplied/oracle MAE levels with clean oracle collapsed",
    )
    for name, expected_hash in protected.items():
        checked_file(path=Path(name), expected=expected_hash)
    return producer, values, protected


def panel(
    *, axis: Axes, values: dict, variants: tuple[str, ...], noise: int, readout: str
) -> None:
    for variant in variants:
        scores = np.array([
            [
                values[(variant, block, noise, readout, f"e{epoch}")]["value"]
                for epoch in TIMES
            ]
            for block in range(3)
        ])
        marker, style = STYLES[variant]
        for block_scores in scores:
            axis.plot(
                TIMES,
                block_scores,
                color=COLORS[variant],
                alpha=0.22,
                linewidth=1,
                marker=marker,
                markersize=3,
                zorder=1,
            )
        axis.errorbar(
            TIMES,
            scores.mean(0),
            yerr=scores.std(0, ddof=1),
            color=COLORS[variant],
            linestyle=style,
            marker=marker,
            markersize=6,
            linewidth=2,
            elinewidth=1.2,
            capsize=4,
            label=LABELS[variant],
            zorder=3,
        )
    axis.set(
        xlabel="Training epoch",
        ylabel="Test age MAE (years)",
        xticks=TIMES,
        xlim=(25, 525),
    )
    axis.grid(alpha=0.2)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)


def learning_horizon(*, values: dict) -> tuple[Figure, list[dict]]:
    figure, axes = plt.subplots(2, 2, figsize=(13, 9.5), sharex=True, sharey="row")
    panels = []
    for index, readout in enumerate(("native", "supplied_labels")):
        for column, noise in enumerate((0, 10)):
            axis = axes[index, column]
            panel(
                axis=axis,
                values=values,
                variants=HORIZON_METHODS,
                noise=noise,
                readout=readout,
            )
            condition = (
                "Clean training labels" if noise == 0 else "Fixed SD-10 training labels"
            )
            axis.set_title(f"{condition}\n{READOUTS[readout]}", fontsize=11, pad=10)
            panels.append({
                "noise_sd": noise,
                "readout": readout,
                "variants": HORIZON_METHODS,
            })
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.96),
        ncol=2,
        fontsize=10,
    )
    figure.suptitle("Scratch training at fixed epochs", y=1.01, fontsize=15)
    return figure, panels


def tail_free_affine(*, values: dict) -> tuple[Figure, list[dict]]:
    figure, axes = plt.subplots(1, 3, figsize=(16.5, 5.5), sharex=True, sharey=True)
    panels = []
    for axis, readout in zip(axes, READOUTS, strict=True):
        panel(
            axis=axis, values=values, variants=TAIL_METHODS, noise=10, readout=readout
        )
        axis.set_title(READOUTS[readout], fontsize=11, pad=10)
        panels.append({"noise_sd": 10, "readout": readout, "variants": TAIL_METHODS})
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.96),
        ncol=2,
        fontsize=10,
    )
    figure.suptitle(
        "TailRL residual heads under fixed SD-10 training labels", y=1.01, fontsize=15
    )
    return figure, panels


def report(*, horizon_dir: Path, output_dir: Path) -> dict:
    horizon_dir, output_dir = horizon_dir.resolve(), output_dir.resolve()
    producer, values, protected = load_inputs(horizon_dir=horizon_dir)
    input_hashes = {**producer["input_sha256"], **protected}
    inputs = {Path(name).resolve() for name in input_hashes}
    destinations = [output_dir / "figure-report.json"] + [
        output_dir / f"{name}.{extension}"
        for name, extension in itertools.product(CAPTIONS, FORMATS)
    ]
    for path in destinations:
        require(
            condition=path.resolve() not in inputs,
            path=path,
            expectation="a figure output distinct from every protected producer/input artifact",
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    figures = []
    for name, builder in (
        ("learning-horizon", learning_horizon),
        ("tail-free-affine", tail_free_affine),
    ):
        figure, panels = builder(values=values)
        note = NOTE + (
            " Oracle ridge uses additional clean reference training labels."
            if name == "tail-free-affine"
            else ""
        )
        paths = save(
            figure=figure, output=output_dir, name=name, formats=FORMATS, note=note
        )
        figures.append({
            "name": name,
            "caption": CAPTIONS[name],
            "panels": panels,
            "artifacts": [
                {
                    "path": str(Path(path).resolve()),
                    "format": Path(path).suffix[1:],
                    "sha256": file_sha256(path=Path(path)),
                }
                for path in paths
            ],
        })
    for name, expected in protected.items():
        checked_file(path=Path(name), expected=expected)
    root = source_root()
    result = {
        "schema": "utkface-mechanism-horizon-figures-v1",
        "scientific_complete": True,
        "source_commit": producer["source_commit"],
        "analysis_git_commit": git_head(),
        "analysis_source_sha256": {
            name: file_sha256(path=root / name) for name in SOURCE_FILES
        },
        "producer_report": {
            "path": str(horizon_dir / "report.json"),
            "sha256": protected[str(horizon_dir / "report.json")],
            "analysis_git_commit": producer["analysis_git_commit"],
            "analysis_source_sha256": producer["analysis_source_sha256"],
        },
        "completed_fits": 66,
        "completed_probe_sources": 198,
        "noise_audits": 198,
        "n_blocks": 3,
        "epochs": TIMES,
        "metric": "Clean-reference test age MAE in years",
        "statistic": "probe_test_mae",
        "kind": "level",
        "fixed_epoch_levels_validated": len(values),
        "uncertainty": "Mean ± sample SD (ddof=1) across three paired joint training/corruption blocks; epochs and readouts reuse fits.",
        "native_scope": "Recomputed fixed-center-crop native test MAE at the three fixed epochs, from the producer's probe_test_mae levels.",
        "ridge_scope": producer["probe_scope"],
        "clean_oracle_scope": producer["clean_oracle_scope"],
        "oracle_scope": "SD-10 oracle ridge is a diagnostic using additional clean reference training labels.",
        "figure_scope": "Descriptive fixed-epoch point errors for the named methods and readouts; all other producer tables and secondary raw data remain retained.",
        "input_sha256": input_hashes,
        "protected_horizon_files_sha256": protected,
        "original_horizon_files_unchanged": True,
        "output_shares_horizon_directory": output_dir == horizon_dir,
        "figures": figures,
    }
    write_json(path=output_dir / "figure-report.json", payload=result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--horizon-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = report(horizon_dir=args.horizon_dir, output_dir=args.output_dir)
    print(
        json.dumps({
            "scientific_complete": result["scientific_complete"],
            "report": str(args.output_dir / "figure-report.json"),
        })
    )


if __name__ == "__main__":
    main()
