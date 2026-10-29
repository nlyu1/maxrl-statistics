import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import polars as pl

from src.experiments.utkface.adaptation_train import file_sha256, source_root
from src.experiments.utkface.mechanism_noise_audit import (
    ADJUSTMENTS,
    AUDIT_SOURCES,
    BANDS,
)
from src.experiments.utkface.mechanism_probe import PROBE_SOURCE_FILES, write_tables
from src.experiments.utkface.mechanism_report import (
    CONTRASTS,
    LABELS,
    aggregate,
    digest,
    numeric,
    plt,
    read,
    save,
)
from src.experiments.utkface.run import write_json

IDENTITY = ("identifier", "block", "noise_sd", "variant", "checkpoint", "epoch")
KEYS = ["variant", "noise_sd", "epoch", "readout", "split", "target", "metric"]
READOUTS = {
    "native": "Native readout",
    "supplied_labels": "Ridge: supplied labels",
    "oracle_clean_labels": "Oracle ridge: clean labels",
}
REQUIRED_EPOCHS = {"imagenet": (25, 50), "scratch": (50, 200, 500)}
NOTE = (
    "Dots: independent paired training/corruption blocks. Bars: mean ± sample SD "
    "(ddof=1), with up to three blocks; no confidence intervals. Ridge alpha uses "
    "clean validation MSE. Oracle ridge is a clean-label diagnostic. Scalar heads "
    "have no policy entropy; clean-label noise slopes are undefined."
)


def checked_file(path: Path, expected: str) -> None:
    if file_sha256(path=path) != expected:
        raise ValueError(f"Artifact hash changed: {path}")


def load_probes(*, root: Path) -> tuple[dict, list[dict], dict]:
    paths = sorted(root.glob("g*/report.json"))
    if not paths:
        raise FileNotFoundError(f"No sharded probe reports: {root}")
    first = read(paths[0])
    requested = {k: v for k, v in first["requested"].items() if k != "variants"}
    training_root = Path(requested["root"])
    plan = read(training_root / "plan.json")
    checked_file(training_root / "plan.json", requested["plan_sha256"])
    jobs = {job["identifier"]: job for job in plan["jobs"]}
    if len(jobs) != len(plan["jobs"]):
        raise ValueError("Duplicate study identifiers")
    variants = {job["variant"] for job in jobs.values()}
    cells = {(job["block"], job["noise_sd"], job["variant"]) for job in jobs.values()}
    if len(jobs) != len(cells):
        raise ValueError("Duplicate variant/block/noise study cells")
    missing_cells = {(b, n, v) for b in range(3) for n in (0, 10) for v in variants} - cells
    cohort_valid = (
        bool(jobs)
        and len(jobs) == len(cells)
        and plan["phase"] == "scientific"
        and cells == {(b, n, v) for b in range(3) for n in (0, 10) for v in variants}
    )
    epochs = set(REQUIRED_EPOCHS[plan["initialization"]])
    for checkpoint in requested["checkpoints"]:
        if checkpoint != "final" and not checkpoint.startswith("epoch_"):
            raise ValueError("Require fixed-epoch encoders, excluding validation-selected best")
        epochs.add(plan["epochs"] if checkpoint == "final" else int(checkpoint[6:]))
    if set(requested["source_sha256"]) != set(PROBE_SOURCE_FILES):
        raise ValueError("Probe source hash coverage is incomplete")
    for name, expected in requested["source_sha256"].items():
        content = subprocess.check_output(["git", "show", f"{plan['git_commit']}:{name}"], cwd=source_root())
        if hashlib.sha256(content).hexdigest() != expected:
            raise ValueError(f"Probe source differs from pinned Git blob: {name}")
    runs, seen, selected, paired, tensors = [], set(), set(), {}, {}
    split_identity = {key: first[key] for key in ("manifest_fingerprint", "shared_indices_sha256", "split_counts")}
    for path in paths:
        report = read(path)
        request = report["requested"]
        if (
            report["schema"] != "utkface-mechanism-common-readout-v1"
            or report["git_commit"] != plan["git_commit"]
            or {k: v for k, v in request.items() if k != "variants"} != requested
            or any(report[key] != value for key, value in split_identity.items())
            or selected.intersection(request["variants"])
        ):
            raise ValueError(f"Inconsistent or overlapping probe shards: {path}")
        selected.update(request["variants"])
        planned = sum(job["variant"] in request["variants"] for job in jobs.values()) * len(request["checkpoints"])
        if (
            set(request["variants"]) - variants
            or len(set(request["variants"])) != len(request["variants"])
            or report["completed_sources"] != len(report["runs"])
            or report["planned_sources"] != planned
        ):
            raise ValueError(f"Invalid probe coverage metadata: {path}")
        for row in report["runs"]:
            source, features = row["source"], row["features"]
            key = (source["identifier"], source["epoch"])
            job = jobs[source["identifier"]]
            checkpoint = source["checkpoint"]
            epoch = plan["epochs"] if checkpoint == "final" else int(checkpoint[6:])
            if (
                key in seen
                or source["variant"] not in request["variants"]
                or checkpoint not in request["checkpoints"]
                or source["epoch"] != epoch
                or any(source[k] != job[k] for k in ("block", "noise_sd", "variant"))
            ):
                raise ValueError(f"Duplicate or unmatched probe source: {key}")
            seen.add(key)
            checked_file(Path(job["config"]), source["config_sha256"])
            checked_file(Path(job["output"]) / "result.json", source["result_sha256"])
            expected_path = Path(job["output"]) / (
                "final.pt" if checkpoint == "final" else f"model-milestones/{checkpoint}.pt"
            )
            if (
                Path(source["checkpoint_path"]).resolve() != expected_path.resolve()
                or Path(source["config_path"]).resolve() != Path(job["config"]).resolve()
            ):
                raise ValueError(f"Checkpoint/config path differs from plan: {key}")
            checked_file(expected_path, source["checkpoint_sha256"])
            for artifact in (features, row["ridge"]["artifact"]):
                checked_file(Path(artifact["path"]), artifact["sha256"])
            if row["ridge"]["alpha_grid"] != requested["alpha_grid"] or any(
                [candidate["alpha"] for candidate in target["candidates"]] != requested["alpha_grid"]
                for target in row["ridge"]["targets"].values()
            ):
                raise ValueError(f"Ridge candidate grid differs across sources: {key}")
            provenance = source["training_provenance"]
            config = read(Path(job["config"]))
            result = read(Path(job["output"]) / "result.json")
            checked_file(Path(config["adaptation"]["run"]["manifest"]), provenance["manifest_sha256"])
            if (
                provenance["fingerprint"] != digest({k: v for k, v in provenance.items() if k != "fingerprint"})
                or provenance["mechanism_config"] != config
                or result["config"] != config
                or result["status"] != "complete"
                or result["provenance"] != provenance
                or config["adaptation"]["run"]["epochs"] != plan["epochs"]
                or config["adaptation"]["initialization"] != plan["initialization"]
                or config["adaptation"]["run"]["noise"]["std_years"] != job["noise_sd"]
                or provenance["environment"]["git_commit"] != plan["git_commit"]
                or any(requested["source_sha256"].get(k) != v for k, v in provenance["source_sha256"].items())
                or features["provenance"]["source"] != source
                or features["provenance"]["manifest_sha256"] != provenance["manifest_sha256"]
                or features["provenance"]["manifest_fingerprint"] != split_identity["manifest_fingerprint"]
                or features["provenance"]["indices_sha256"] != split_identity["shared_indices_sha256"]
            ):
                raise ValueError(f"Source/feature provenance mismatch: {key}")
            identity = {
                split: {k: values[k] for k in ("indices", "clean_ages")}
                for split, values in features["tensor_sha256"].items()
            }
            if not tensors:
                tensors = identity
            if (
                identity != tensors
                or set(identity) != {"train", "val", "test"}
                or any(
                    values["indices"] != split_identity["shared_indices_sha256"][split]
                    for split, values in identity.items()
                )
            ):
                raise ValueError(f"Feature split/image identities differ: {key}")
            if (
                features["tensor_sha256"]["train"]["supplied_labels"] != source["source_training_labels_sha256"]
                or features["tensor_sha256"]["train"]["clean_ages"] != provenance["clean_training_labels_sha256"]
            ):
                raise ValueError(f"Feature training labels differ from source: {key}")
            pair = {
                k: provenance[k]
                for k in ("encoder_initial_sha256", "model_seed", "head_seed", "view_seed", "order_seed", "noise_seed")
            }
            pair["labels"] = features["tensor_sha256"]["train"]["supplied_labels"]
            cell = (source["block"], source["noise_sd"])
            if paired.setdefault(cell, pair) != pair:
                raise ValueError(f"Unmatched seeds, encoders or supplied labels: {cell}")
            runs.append(row)
    missing = [
        {"identifier": identifier, "epoch": epoch}
        for identifier in jobs
        for epoch in sorted(epochs)
        if (identifier, epoch) not in seen
    ]
    coverage = {
        "planned_sources": len(jobs) * len(epochs),
        "completed_sources": len(runs),
        "missing_sources": missing,
        "cohort_valid": cohort_valid,
        "missing_cohort_cells": [{"block": b, "noise_sd": n, "variant": v} for b, n, v in sorted(missing_cells)],
        "missing_shards": [
            str(root / f"g{i}" / "report.json")
            for i in range(min(4, len(variants)))
            if not (root / f"g{i}" / "report.json").is_file()
        ],
        "unexpected_shards": [str(p) for p in paths if p.parent.name not in {f"g{i}" for i in range(min(4, len(variants)))}],
        "input_reports": [{"path": str(p), "sha256": file_sha256(path=p)} for p in paths],
    }
    return plan, runs, coverage


def long_rows(runs: list[dict]) -> list[dict]:
    rows = []
    for run in runs:
        identity = {key: run["source"][key] for key in IDENTITY}
        if set(run["native"]) != {"train", "val", "test"} or set(run["ridge"]["targets"]) != {
            "supplied_labels",
            "oracle_clean_labels",
        }:
            raise ValueError("Require native train/validation/test and both ridge label targets")
        for split, native in run["native"].items():
            for target in ("clean", "supplied_labels"):
                for metric, value in numeric(native.get(target, {})).items():
                    rows.append(
                        dict(identity, readout="native", split=split, target=target, metric=metric, value=value)
                    )
            rows.append(
                dict(identity, readout="native", split=split, target="none", metric="entropy", value=native["entropy"])
            )
        for target, ridge in run["ridge"]["targets"].items():
            if ridge["primary"] != (target == "supplied_labels") or ridge[
                "oracle_clean_labels_representation_diagnostic"
            ] != (target == "oracle_clean_labels"):
                raise ValueError("Primary and oracle ridge target flags differ")
            selected = min(ridge["candidates"], key=lambda row: (row["validation"]["mse"], -row["alpha"]))
            if selected != ridge["selected"]:
                raise ValueError("Ridge alpha selection differs from clean validation MSE rule")
            for split, name in (("val", "validation"), ("test", "test")):
                for metric, value in numeric(selected[name]).items():
                    rows.append(dict(identity, readout=target, split=split, target="clean", metric=metric, value=value))
    return rows


def paired_rows(rows: list[dict]) -> list[dict]:
    lookup = {(row["variant"], row["block"], *(row[k] for k in KEYS[1:])): row for row in rows}
    pairs = []
    for first, second in CONTRASTS:
        for key, left in lookup.items():
            right = lookup.get((second, *key[1:])) if key[0] == first else None
            if right is not None:
                value = (
                    left["value"] - right["value"] if left["value"] is not None and right["value"] is not None else None
                )
                pairs.append(
                    dict(
                        left,
                        contrast=f"{first} - {second}",
                        first=first,
                        second=second,
                        first_value=left["value"],
                        second_value=right["value"],
                        value=value,
                    )
                )
    return pairs


def load_noise(*, roots: tuple[Path, ...], runs: list[dict]) -> tuple[list[dict], list[dict]]:
    sources = {(run["source"]["identifier"], run["source"]["epoch"]): run for run in runs}
    rows, seen, signatures = [], set(), []
    for root in roots:
        for path in sorted(root.glob("g*/report.json")):
            report = read(path)
            if report["schema"] != "utkface-mechanism-noise-audit-v1":
                raise ValueError(f"Unsupported noise audit: {path}")
            if set(report["audit_source_sha256"]) != set(AUDIT_SOURCES) or not report["reference_seeds"]:
                raise ValueError("Noise audit source hash coverage is incomplete")
            signature = {key: report[key] for key in ("reference_seeds", "audit_source_sha256")}
            if signatures and signature != signatures[0]:
                raise ValueError("Noise audit shards use different reference fields or sources")
            signatures.append(signature)
            for name, expected in report["audit_source_sha256"].items():
                checked_file(source_root() / "src/experiments/utkface" / name, expected)
            for entry in report["sources"]:
                source = entry["provenance"]["source"]
                key = (source["identifier"], source["epoch"])
                probe = sources[key]
                if (
                    key in seen
                    or entry["provenance"] != probe["features"]["provenance"]
                    or entry["tensor_sha256"] != probe["features"]["tensor_sha256"]
                    or Path(entry["archive"]).resolve() != Path(probe["features"]["path"]).resolve()
                    or {(row["band"], row["adjustment"]) for row in entry["rows"]}
                    != {(b, a) for b in BANDS for a in ADJUSTMENTS}
                    or len(entry["rows"]) != len(BANDS) * len(ADJUSTMENTS)
                    or entry["reference_artifact"]["metadata"]["seeds"] != report["reference_seeds"]
                    or entry["reference_artifact"]["metadata"]["std"] != source["noise_sd"]
                    or entry["reference_artifact"]["metadata"]["manifest_fingerprint"]
                    != entry["provenance"]["manifest_fingerprint"]
                    or Path(entry["archive"]).parent.parent.resolve() != Path(report["probe_dir"]).resolve()
                ):
                    raise ValueError(f"Noise audit source mismatch: {key}")
                seen.add(key)
                checked_file(Path(entry["archive"]), entry["archive_sha256"])
                checked_file(Path(entry["reference_artifact"]["path"]), entry["reference_artifact"]["sha256"])
                for row in entry["rows"]:
                    if any(row[k] != source[k] for k in IDENTITY):
                        raise ValueError(f"Noise audit row identity mismatch: {key}")
                    rows.append({k: v for k, v in row.items() if k != "reference_slopes"})
    return rows, [
        {"identifier": identifier, "epoch": epoch} for identifier, epoch in sources if (identifier, epoch) not in seen
    ]


def points(*, rows: list[dict], name: str, title: str, xlabel: str, output: Path, paired: bool) -> list[str]:
    field = "contrast" if paired else "variant"
    present = list(dict.fromkeys(row[field] for row in rows))
    names = present if paired else [name for name in LABELS if name in present]
    epochs = sorted({row["epoch"] for row in rows})
    figure, axes = plt.subplots(1, 2, figsize=(16, max(4.8, len(names) * 0.64)), sharex=True, sharey=True)
    for axis, noise in zip(axes, (0, 10), strict=True):
        for index, label in enumerate(names):
            for offset, epoch in enumerate(epochs):
                values = [
                    row["value"]
                    for row in rows
                    if row[field] == label
                    and row["noise_sd"] == noise
                    and row["epoch"] == epoch
                    and row["value"] is not None
                ]
                if values:
                    y = index + (offset - (len(epochs) - 1) / 2) * 0.24
                    color = plt.get_cmap("viridis")((offset + 0.3) / len(epochs))
                    axis.scatter(values, y + np.linspace(-0.045, 0.045, len(values)), color=color, s=16, alpha=0.5)
                    axis.errorbar(
                        np.mean(values),
                        y,
                        xerr=np.std(values, ddof=1) if len(values) > 1 else None,
                        fmt="o",
                        color=color,
                        capsize=3,
                        label=f"Epoch {epoch}",
                    )
        labels = [" − ".join(LABELS[v] for v in label.split(" - ")) if paired else LABELS[label] for label in names]
        axis.set_yticks(range(len(names)), labels, fontsize=8)
        axis.set_ylim(len(names) - 0.5, -0.5)
        axis.set(title=f"Training-label noise SD {noise} years", xlabel=xlabel)
        axis.grid(axis="x", alpha=0.2)
        if paired:
            axis.axvline(0, color="black", linewidth=0.8)
        handles, labels = axis.get_legend_handles_labels()
        if handles:
            legend = dict(zip(labels, handles, strict=True))
            axis.legend(legend.values(), legend.keys(), fontsize=8)
    figure.suptitle(title)
    return save(figure=figure, output=output, name=name, formats=("png", "svg"), note=NOTE)


def report(*, probe_root: Path, noise_roots: tuple[Path, ...], output: Path) -> dict:
    if any(output.resolve().is_relative_to(root.resolve()) for root in (probe_root, *noise_roots)):
        raise ValueError("Report output must be outside the source probe/audit directories")
    plan, runs, coverage = load_probes(root=probe_root)
    cohort_complete = coverage["cohort_valid"] and not any(
        coverage[k] for k in ("missing_sources", "missing_shards", "unexpected_shards")
    )
    context = f"{'ImageNet' if plan['initialization'] == 'imagenet' else 'Scratch'} fixed encoders"
    if not cohort_complete:
        context += " · PARTIAL COVERAGE"
    rows, noise_missing = long_rows(runs), []
    pairs = paired_rows(rows)
    output.mkdir(parents=True, exist_ok=True)
    frames = {
        "long": pl.DataFrame(rows, infer_schema_length=None),
        "summary": aggregate(rows, KEYS),
        "missing_sources": pl.DataFrame(
            coverage["missing_sources"], schema={"identifier": pl.String, "epoch": pl.Int64}
        ),
    }
    if pairs:
        frames.update(
            paired=pl.DataFrame(pairs, infer_schema_length=None),
            paired_summary=aggregate(pairs, ["contrast", *KEYS[1:]]),
        )
    noise, files = [], []
    if noise_roots:
        noise, noise_missing = load_noise(roots=noise_roots, runs=runs)
        if noise_missing:
            context += " · PARTIAL NOISE AUDIT"
        if noise:
            frames["noise_long"] = pl.DataFrame(noise, infer_schema_length=None)
            slopes = [dict(row, readout="native", target="noise", metric="slope", value=row["slope"]) for row in noise]
            frames["noise_summary"] = aggregate(
                slopes, ["variant", "noise_sd", "epoch", "band", "adjustment", "metric"]
            )
            for adjustment in ADJUSTMENTS:
                for band in BANDS:
                    selected = [row for row in slopes if row["adjustment"] == adjustment and row["band"] == band]
                    files += points(
                        rows=selected,
                        name=f"alignment-{adjustment}-{band}",
                        title=f"{context} · noise alignment · {adjustment} · {band}",
                        xlabel="Cov(error, corruption) / Var(corruption)",
                        output=output,
                        paired=False,
                    )
    for readout, label in READOUTS.items():
        for paired, values in ((False, rows), (True, pairs)):
            selected = [
                r
                for r in values
                if r["readout"] == readout and r["metric"] == "mae" and r["split"] == "test" and r["target"] == "clean"
            ]
            if selected:
                files += points(
                    rows=selected,
                    name=f"{'paired' if paired else 'test-mae'}-{readout}",
                    title=f"{context} · {label} · {'paired method differences' if paired else 'clean test MAE'}",
                    xlabel="Δ MAE (years), first minus second" if paired else "MAE (years)",
                    output=output,
                    paired=paired,
                )
    for metric, target, label in (
        ("entropy", "none", "Native policy entropy"),
        ("mae", "clean", "Native train MAE: clean ages"),
        ("mae", "supplied_labels", "Native train MAE: supplied labels"),
    ):
        selected = [
            r
            for r in rows
            if r["readout"] == "native" and r["split"] == "train" and r["metric"] == metric and r["target"] == target
        ]
        files += points(
            rows=selected,
            name=f"train-{metric}-{target}",
            title=f"{context} · {label}",
            xlabel="Entropy (nats)" if metric == "entropy" else "MAE (years)",
            output=output,
            paired=False,
        )
    for name, frame in frames.items():
        frame.write_csv(output / f"{name.replace('_', '-')}.tsv", separator="\t")
    write_tables(report={"runs": runs}, output_dir=output)
    complete = cohort_complete and not noise_missing
    summary = {
        "schema": "utkface-mechanism-probe-report-v1",
        "scientific_complete": complete,
        "initialization": plan["initialization"],
        "training_root": read(min(probe_root.glob("g*/report.json")))["requested"]["root"],
        "coverage": coverage,
        "noise_audit_requested": bool(noise_roots),
        "noise_missing_sources": noise_missing,
        "noise_completed_sources": len(noise) // (len(BANDS) * len(ADJUSTMENTS)),
        "noise_input_reports": [
            {"path": str(p), "sha256": file_sha256(path=p)}
            for root in noise_roots
            for p in sorted(root.glob("g*/report.json"))
        ],
        "noise_reference_scope": "Conditional randomization references given fixed faces and predictions; not confidence intervals or training-seed uncertainty.",
        "uncertainty": NOTE,
        "files": files,
        "tables": [str(output / f"{name.replace('_', '-')}.tsv") for name in frames]
        + [str(output / name) for name in ("selected-readouts.tsv", "ridge-grid.tsv", "native-diagnostics.tsv")],
    }
    write_json(path=output / "report.json", payload=summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--noise-root", type=Path)
    args = parser.parse_args()
    result = report(
        probe_root=args.probe_root, noise_roots=(args.noise_root,) if args.noise_root else (), output=args.output_dir
    )
    print(
        json.dumps({
            "scientific_complete": result["scientific_complete"],
            "coverage": result["coverage"],
            "report": str(args.output_dir / "report.json"),
        })
    )
    if not result["scientific_complete"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
