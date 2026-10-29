import argparse
import itertools
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import polars as pl

from src.experiments.utkface.adaptation_train import file_sha256
from src.experiments.utkface.mechanism_report import (
    COLORS,
    LABELS,
    aggregate,
    plt,
    point_figure,
    read,
    save,
)
from src.experiments.utkface.run import write_json

METHODS = {
    "F": "categorical_ce_binomial_init_fp32",
    "R": "categorical_ce_random_fp32",
    "U": "binomial_ce_free_fp32",
    "P": "categorical_ce_symmetric_swap_init_fp32",
}
COHORTS = {
    "baseline": (36, 72, "55ea16131f05a0d60e191e3083a216db19bb3e65"),
    "assignment": (6, 12, "e71bebcfcc990ab35daa8e1122a6f5eb4e6abfb8"),
}
CONDITIONS = (
    ("native", "selected"),
    ("native", "final"),
    *itertools.product(("ridge_supplied", "ridge_oracle"), ("e25", "e50")),
)
KEYS = ["method", "noise_sd", "readout", "endpoint"]
NOTE = (
    "Dots: three paired joint blocks; bars: mean ± sample SD (ddof=1), descriptive. "
    "P used 2 jobs/GPU, F/R/U used 8; CUDA training was nondeterministic. "
    "Oracle ridge uses extra clean training labels. Epochs and readouts reuse the same fits."
)


def inputs(*, training: dict[str, Path], probes: dict[str, Path]) -> tuple[list[dict], dict]:
    rows, hashes, provenance = [], {}, {}
    for cohort, (fit_count, probe_count, commit) in COHORTS.items():
        paths = [training[cohort] / name for name in ("report.json", "endpoints-individual.tsv")]
        paths += [probes[cohort] / name for name in ("report.json", "selected-readouts.tsv", "native-diagnostics.tsv")]
        hashes.update({str(path.resolve()): file_sha256(path=path) for path in paths})
        fit_report, probe_report = read(paths[0]), read(paths[2])
        if (
            not fit_report["scientific_complete"]
            or not probe_report["scientific_complete"]
            or fit_report["epochs"] != 50
            or fit_report["initialization"] != "imagenet"
            or fit_report["completed_jobs"] != fit_count
            or fit_report["planned_jobs"] != fit_count
            or probe_report["coverage"]["completed_sources"] != probe_count
            or probe_report["coverage"]["planned_sources"] != probe_count
            or Path(fit_report["root"]).resolve() != Path(probe_report["training_root"]).resolve()
        ):
            raise ValueError(f"Incomplete or unmatched source reports: {cohort}")
        plan_path = Path(fit_report["root"]) / "plan.json"
        hashes[str(plan_path.resolve())] = file_sha256(path=plan_path)
        if read(plan_path)["git_commit"] != commit:
            raise ValueError(f"Unexpected training source: {cohort}")
        wanted = {method: variant for method, variant in METHODS.items() if (method == "P") == (cohort == "assignment")}
        selected_records = [
            record for record in fit_report["pairing_audit"]["runs"] if record["variant"] in wanted.values()
        ]
        records = {(record["variant"], record["block"], record["noise_sd"]): record for record in selected_records}
        if (
            len(records) != len(selected_records)
            or len(records) != len(wanted) * 6
            or not all(record["complete"] for record in selected_records)
        ):
            raise ValueError(f"Incomplete method/block/noise audit records: {cohort}")
        path = Path(fit_report["root"]) / "runs" / selected_records[0]["identifier"] / "provenance.json"
        saved = read(path)
        hashes[str(path.resolve())] = file_sha256(path=path)
        precision = {key: saved[key] for key in ("head_precision", "encoder_precision", "float32_matmul_precision")}
        if saved["environment"]["git_commit"] != commit or tuple(precision.values()) != ("fp32", "bf16", "highest"):
            raise ValueError(f"Unexpected representative source/precision: {path}")
        provenance[cohort] = {"path": str(path), "precision": precision, "environment": saved["environment"]}
        tables = [pl.read_csv(path, separator="\t").to_dicts() for path in (paths[1], paths[3])]
        for method, variant in wanted.items():
            values = []
            for row in tables[0]:
                if (
                    row["variant"] == variant
                    and row["split"] == "test"
                    and row["metric"] == "mae_mean"
                    and row["checkpoint"] in {"selected", "final"}
                ):
                    values.append((row, "native", row["checkpoint"], row["value"]))
            for row in tables[1]:
                if row["variant"] == variant and row["epoch"] in {25, 50}:
                    if row["target"] not in {"supplied_labels", "oracle_clean_labels"}:
                        raise ValueError(f"Unexpected ridge target: {row['target']}")
                    values.append((
                        row,
                        "ridge_supplied" if row["target"] == "supplied_labels" else "ridge_oracle",
                        f"e{row['epoch']}",
                        row["test_mae"],
                    ))
            for row, readout, endpoint, value in values:
                record = records[(variant, row["block"], row["noise_sd"])]
                if row["identifier"] != record["identifier"] or not np.isfinite(value):
                    raise ValueError(f"Invalid source identity or MAE: {row}")
                rows.append(
                    dict(
                        {key: row[key] for key in ("block", "noise_sd", "identifier", "variant")},
                        method=method,
                        cohort=cohort,
                        source_commit=commit,
                        readout=readout,
                        endpoint=endpoint,
                        value=value,
                    )
                )
    expected = {(m, b, n, r, e) for m, b, n, (r, e) in itertools.product(METHODS, range(3), (0, 10), CONDITIONS)}
    actual = [(r["method"], r["block"], r["noise_sd"], r["readout"], r["endpoint"]) for r in rows]
    if len(actual) != len(set(actual)) or set(actual) != expected:
        raise ValueError(f"Duplicate/incomplete comparison cells; missing={sorted(expected - set(actual))}")
    verification = training["assignment"].parent / "symmetric-swap-verification.json"
    hashes[str(verification.resolve())] = file_sha256(path=verification)
    return rows, {
        "input_sha256": hashes,
        "representative_training_provenance": provenance,
        "cpu_validation_paths": [str(verification)],
    }


def differences(*, rows: list[dict]) -> list[dict]:
    lookup = {(row["method"], row["block"], row["noise_sd"], row["readout"], row["endpoint"]): row for row in rows}
    paired = []
    for first, second in (("F", "P"), ("F", "R"), ("F", "U"), ("P", "R"), ("P", "U"), ("U", "R")):
        for key, left in lookup.items():
            if key[0] == first:
                right = lookup[(second, *key[1:])]
                paired.append(
                    dict(
                        left,
                        contrast=f"{first} - {second}",
                        first_value=left["value"],
                        second_value=right["value"],
                        value=left["value"] - right["value"],
                    )
                )
    return paired


def figures(*, rows: list[dict], output: Path) -> list[str]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["method"], row["noise_sd"], row["readout"], row["endpoint"])].append(row["value"])
    native = [
        dict(row, checkpoint=row["endpoint"], metric="mae_mean", split="test")
        for row in rows
        if row["readout"] == "native"
    ]
    figure = point_figure(rows=native, metric="mae_mean", split="test", paired=False)
    limits = (min(axis.get_xlim()[0] for axis in figure.axes), max(axis.get_xlim()[1] for axis in figure.axes))
    for axis in figure.axes:
        axis.set_yticks(range(4), [f"{method} · {LABELS[variant]}" for method, variant in METHODS.items()], fontsize=8)
        axis.set_xlim(limits)
    figure.suptitle("FP32 CE initialization: native selected and final predictors")
    paths = save(figure=figure, output=output, name="native-ce", formats=("png", "svg", "pdf"), note=NOTE)
    figure, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=True, sharey=True)
    for axis, noise in zip(axes, (0, 10), strict=True):
        for method, variant in METHODS.items():
            for readout, style, marker in (("ridge_supplied", "-", "o"), ("ridge_oracle", "--", "s")):
                values = np.array([grouped[(method, noise, readout, f"e{epoch}")] for epoch in (25, 50)])
                axis.scatter(np.repeat((25, 50), 3), values.flatten(), color=COLORS[variant], s=15, alpha=0.35)
                axis.errorbar(
                    (25, 50),
                    values.mean(1),
                    yerr=values.std(1, ddof=1),
                    color=COLORS[variant],
                    linestyle=style,
                    marker=marker,
                    capsize=3,
                    label=f"{method}: {'Supplied' if readout == 'ridge_supplied' else 'Oracle'}",
                )
        axis.set(
            title=f"Training-label noise SD {noise} years",
            xlabel="Encoder epoch",
            ylabel="Clean test MAE (years)",
            xticks=(25, 50),
            xlim=(22, 53),
        )
        axis.grid(alpha=0.2)
    handles, labels = axes[-1].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.66, 0.98), ncol=2, fontsize=8)
    figure.suptitle("FP32 CE: common ridge on encoders frozen at the shown epochs", y=1.04)
    return paths + save(
        figure=figure,
        output=output,
        name="common-ridge",
        formats=("png", "svg", "pdf"),
        note=NOTE + " F=ordered ordinary, R=random ordinary, U=score+residual, P=symmetric swaps. Supplied and oracle coincide for clean labels.",
    )


def report(*, training: dict[str, Path], probes: dict[str, Path], output: Path) -> dict:
    if any(output.resolve().is_relative_to(path.resolve()) for path in (*training.values(), *probes.values())):
        raise ValueError("Comparison output must be outside input report directories")
    rows, metadata = inputs(training=training, probes=probes)
    paired = differences(rows=rows)
    frames = {
        "long": pl.DataFrame(rows),
        "summary": aggregate(rows, KEYS),
        "paired": pl.DataFrame(paired),
        "paired_summary": aggregate(paired, ["contrast", *KEYS[1:]]),
    }
    output.mkdir(parents=True, exist_ok=True)
    for name, frame in frames.items():
        frame.write_csv(output / f"{name.replace('_', '-')}.tsv", separator="\t")
    files = figures(rows=rows, output=output)
    manifest = dict(
        schema="utkface-mechanism-initialization-comparison-v1",
        scientific_complete=True,
        methods=METHODS,
        n_blocks=3,
        source_cohorts={
            name: {"training_fits": fits, "probe_sources": probes, "commit": commit}
            for name, (fits, probes, commit) in COHORTS.items()
        },
        selection="Native checkpoints minimize clean-validation mean MAE; ridge alpha minimizes clean-validation MSE.",
        primary="F minus P on clean validation-selected native test MAE; SD10 is secondary.",
        uncertainty=NOTE,
        intervention_scope="Complete initialization/assignment and optimizer path; does not isolate age-gradient direction mediation.",
        execution={"jobs_per_gpu": {"baseline": 8, "assignment": 2}, "deterministic_cuda": False},
        contrast_direction="First minus second; negative MAE favors the first method.",
        coverage={"native_rows": 48, "ridge_rows": 96, "total_rows": len(rows)},
        input_validation="Upstream source/checkpoint/feature validation reused; only report/table/provenance JSON files hashed here.",
        files=files,
        tables=[str(output / f"{name.replace('_', '-')}.tsv") for name in frames],
        **metadata,
    )
    write_json(path=output / "report.json", payload=manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in ("baseline-dir", "assignment-dir", "baseline-probe-dir", "assignment-probe-dir", "output-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    result = report(
        training={"baseline": args.baseline_dir, "assignment": args.assignment_dir},
        probes={"baseline": args.baseline_probe_dir, "assignment": args.assignment_probe_dir},
        output=args.output_dir,
    )
    print(
        json.dumps({
            "scientific_complete": result["scientific_complete"],
            "report": str(args.output_dir / "report.json"),
        })
    )


if __name__ == "__main__":
    main()
