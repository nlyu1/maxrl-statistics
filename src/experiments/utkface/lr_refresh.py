import argparse
import copy
import hashlib
import json
import math
import shutil
import statistics
import textwrap
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import polars as pl
from matplotlib.backends.backend_pdf import PdfPages

from src.experiments.utkface.figure_bundle import build, panel_range, series
from src.experiments.utkface.final_report import (
    SCIENTIFIC_STAGES,
    name,
    summarize,
)
from src.experiments.utkface.visual_report import collect

METRICS = ("mae_mean", "mae_mode", "rmse_mean", "median_spearman", "label_nll")
GROUP_FIELDS = (
    "method",
    "head",
    "width",
    "size",
    "epochs",
    "schedule",
    "noise",
    "std",
    "noiseSeed",
    "subsetSeed",
    "comparisonCohort",
    "fingerprint",
    "trainingFingerprint",
)
MATCH_FIELDS = ("stage", "variant", "seed", *GROUP_FIELDS)


def read(*, path: Path) -> dict:
    return json.loads(path.read_text())


def write(*, path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def check(*, condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def family(*, run: dict) -> str:
    stage = run["stage"]
    if stage in {"main", "scalar"}:
        return "full"
    if stage in {"subset", "subset_replication", "matched_steps"}:
        return "subset"
    return stage


def key(*, run: dict) -> tuple:
    return (family(run=run), *(run[field] for field in GROUP_FIELDS))


def stats(*, values: list) -> dict:
    values = [v for v in values if v is not None]
    return {
        "mean": statistics.mean(values) if values else None,
        "sd": statistics.stdev(values) if len(values) > 1 else None,
        "n": len(values),
    }


def grouped(*, runs: list[dict]) -> dict:
    result = defaultdict(list)
    for run in runs:
        result[key(run=run)].append(run)
    for values in result.values():
        values.sort(key=lambda run: run["seed"])
        check(
            condition=len({r["seed"] for r in values}) == len(values)
            and len({r["lr"] for r in values}) == 1,
            message="Duplicate seeds or mixed rates in a comparison group",
        )
    return result


def compare(*, previous: list[dict], current: list[dict]) -> list[dict]:
    before, after = grouped(runs=previous), grouped(runs=current)
    check(
        condition=before.keys() == after.keys(), message="Scientific coverage changed"
    )
    records = []
    for group_key, new in after.items():
        old = before[group_key]
        check(
            condition=[r["seed"] for r in old] == [r["seed"] for r in new],
            message="Paired seeds changed",
        )
        record = {
            "family": group_key[0],
            **{field: new[0][field] for field in GROUP_FIELDS},
            "label": name(run=new[0]),
            "oldSourceCohorts": sorted({r["cohort"] for r in old}),
            "newSourceCohorts": sorted({r["cohort"] for r in new}),
            "seeds": [r["seed"] for r in new],
            "oldIds": [r["id"] for r in old],
            "newIds": [r["id"] for r in new],
            "oldLr": old[0]["lr"],
            "newLr": new[0]["lr"],
            "changed": old[0]["lr"] != new[0]["lr"],
        }
        for phase in ("best", "test"):
            record[phase] = {
                metric: {
                    "old": stats(values=[r[phase][metric] for r in old]),
                    "new": stats(values=[r[phase][metric] for r in new]),
                    "pairedDelta": stats(
                        values=[
                            b[phase][metric] - a[phase][metric]
                            for a, b in zip(old, new, strict=True)
                            if a[phase][metric] is not None
                            and b[phase][metric] is not None
                        ]
                    ),
                    "seedDeltas": [
                        b[phase][metric] - a[phase][metric]
                        if a[phase][metric] is not None and b[phase][metric] is not None
                        else None
                        for a, b in zip(old, new, strict=True)
                    ],
                }
                for metric in METRICS
            }
        records.append(record)
    return records


def assemble(*, original: Path, reruns: Path, old_bundle: Path) -> tuple:
    data = read(path=original)
    previous = [
        r
        for r in data["runs"]
        if r["status"] == "completed"
        and r["stage"] in SCIENTIFIC_STAGES
        and r["test"]["mae_mean"] is not None
    ]
    before = {r["id"]: r for r in previous}
    migration = read(path=reruns / "migration.json")
    plan = read(path=reruns / "plan.json")
    state = read(path=reruns / "state.json")
    status = read(path=reruns / "status.json")
    check(condition=state["status"] == "completed", message="Reruns are incomplete")
    historical_sources = migration["historical_training_source_sha256"]
    source_changes = {
        filename: {"old": historical_sources.get(filename), "new": digest}
        for filename, digest in plan["training_source_sha256"].items()
        if historical_sources.get(filename) != digest
    }
    check(
        condition=source_changes
        == {
            "src/experiments/utkface/run.py": {
                "old": "764ebad17ec97dc8bfc802f86bcdb98360fdfe8408df2d0ee61bee2804ab72ce",
                "new": "8bb27dd5018ee180c5172232b80ff4dec08f37256606d387cbb74e5291e22a3d",
            }
        }
        and historical_sources.keys() == plan["training_source_sha256"].keys(),
        message="Unexpected training source changes beyond audited epoch-zero recording",
    )
    common_cohort = hashlib.sha256(
        json.dumps(historical_sources, sort_keys=True).encode()
    ).hexdigest()
    check(
        condition={r["cohort"] for r in previous} == {common_cohort},
        message="Historical source cohort differs",
    )
    for row in previous:
        row["comparisonCohort"] = common_cohort
    for source, digest in migration["source_plans"].items():
        check(
            condition=hashlib.sha256(Path(source).read_bytes()).hexdigest() == digest,
            message=f"Historical source plan changed: {source}",
        )
    study, rows = collect(plan_path=reruns / "plan.json", state=state)
    for row in rows:
        row["comparisonCohort"] = common_cohort
    by_new_id = {r["id"]: r for r in rows}
    replacements = {c["identifier"]: c for c in migration["changes"]}
    reused = {r["identifier"]: r["result_path"] for r in migration["reused"]}
    check(
        condition=len(before) == len(previous) == 222
        and len(replacements) == len(by_new_id) == len(plan["jobs"]) == 59
        and len(reused) == 163
        and set(replacements).isdisjoint(reused)
        and set(replacements) | set(reused) == set(before),
        message="Expected exactly 59 replacements and 163 reused evaluations",
    )
    check(
        condition={c["new_identifier"] for c in replacements.values()}
        == set(by_new_id),
        message="Rerun identifiers differ from the pre-specified migration",
    )
    historical_bundle = read(path=old_bundle)
    check(
        condition=historical_bundle["sourceSha256"]
        == hashlib.sha256(original.read_bytes()).hexdigest(),
        message="Historical bundle and source snapshot differ",
    )
    current, raw_results = [], []
    for identifier, old in before.items():
        row = copy.deepcopy(old)
        if identifier in replacements:
            change = replacements[identifier]
            row = copy.deepcopy(by_new_id[change["new_identifier"]])
            check(
                condition=all(old[f] == row[f] for f in MATCH_FIELDS)
                and old["lr"] == change["old_learning_rate"]
                and row["lr"] == change["new_learning_rate"],
                message=f"Rerun changes more than the intended rate: {identifier}",
            )
            source = Path(state["jobs"][row["id"]]["result_path"])
            old_config = read(path=Path(change["result_path"]))["config"]
            new_config = read(path=source)["config"]
            check(
                condition={
                    k: v
                    for k, v in old_config.items()
                    if k not in {"learning_rate", "output_dir", "device"}
                }
                == {
                    k: v
                    for k, v in new_config.items()
                    if k not in {"learning_rate", "output_dir", "device"}
                },
                message=f"Executed configuration differs: {identifier}",
            )
            raw_results.append({
                "sourceRunId": identifier,
                "resultPath": str(source),
                "study_job": next(
                    j for j in plan["jobs"] if j["identifier"] == row["id"]
                ),
                "result": read(path=source),
            })
        else:
            source = Path(reused[identifier])
            row["initial"] = (
                historical_bundle["runs"]
                .get(identifier, {})
                .get("initial", row.get("initial", {}))
            )
        result = read(path=source)
        history_path = Path(result["config"]["output_dir"]) / "history.json"
        history = read(path=history_path)
        check(
            condition=[h["epoch"] for h in history]
            == list(range(1, row["epochs"] + 1)),
            message=f"Incomplete history: {row['id']}",
        )
        check(
            condition=result["best_epoch"]
            == row["bestEpoch"]
            == min(history, key=lambda h: h["validation"]["mae_mean"])["epoch"]
            and all(
                row[phase][metric] == values.get(metric)
                for phase, values in (
                    ("best", result["validation"]),
                    ("test", result["test"]),
                )
                for metric in row[phase]
            ),
            message=f"Selected checkpoint metrics differ: {row['id']}",
        )
        check(
            condition=all(
                len(point) == len(data["curveColumns"])
                and all(
                    point[data["curveColumns"].index(metric)]
                    == epoch["validation"][metric]
                    for metric in (
                        "mae_mean",
                        "rmse_mean",
                        "median_spearman",
                        "label_nll",
                    )
                )
                for point, epoch in zip(row["curves"], history, strict=True)
            ),
            message=f"Archived curves differ from raw history: {row['id']}",
        )
        for point, epoch in zip(row["curves"], history, strict=True):
            value = epoch["validation"]["mae_mode"]
            check(condition=math.isfinite(value), message="Nonfinite mode MAE")
            point.append(value)
        row.update({
            "sourceRunId": identifier,
            "resultPath": str(source),
            "historySource": str(history_path),
            "historySha256": hashlib.sha256(history_path.read_bytes()).hexdigest(),
        })
        current.append(row)
    revision = {
        "date": "2026-10-22",
        "reruns": 59,
        "reused": 163,
        "rule": "Pre-specified shared learning rates; no selection using new test scores.",
        "gpuHours": status["gpu_hours"],
        "rerunRoot": str(reruns),
        "planSha256": study["planSha256"],
        "migrationSha256": hashlib.sha256(
            (reruns / "migration.json").read_bytes()
        ).hexdigest(),
        "originalSnapshotSha256": hashlib.sha256(original.read_bytes()).hexdigest(),
    }
    payload = {
        "generated": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "cohortAudit": {
            "comparisonCohort": common_cohort,
            "sourceCohorts": sorted({r["cohort"] for r in current}),
            "sourceChanges": source_changes,
            "basis": (
                "Reviewed diff 928ddd9..b3da562: epoch-zero evaluation/logging "
                "and timing only. Training loop, optimizer, losses and data path "
                "are unchanged. Initial evaluation restores model mode and "
                "Python/NumPy/Torch/loader-generator random state."
            ),
        },
        "learningRateRevision": revision,
        "runs": current,
        "summary": {
            "evaluations": 222,
            "reruns": 59,
            "reused": 163,
            "selectedComplete": True,
        },
        "curveColumns": [*data["curveColumns"], "mae_mode"],
    }
    return payload, compare(previous=previous, current=current), raw_results


def comparison_tables(*, records: list[dict], output: Path) -> None:
    lines = [
        "# UTKFace shared learning rates: previous versus current",
        "",
        (
            "59 pre-specified reruns replace their predecessors; 163 evaluations are reused. "
            "The split, subset fingerprints, seeds, training horizons and all executed config "
            "fields except learning rate/output/device are unchanged. No new test scores "
            "select replacements. This is a follow-up sensitivity study on the same split."
        ),
        "",
        (
            "All numbers below are clean test MAE in years at the validation-mean-MAE-selected "
            "checkpoint. Values with three seeds show sample SD, not confidence intervals. "
            "Noise and 1024-image comparisons have one optimization seed. Rank changes are "
            "descriptive; no statistical significance is claimed."
        ),
        "",
        (
            "The rerun trainer additionally records epoch-zero validation. Its source diff "
            "was reviewed separately; both source hashes are retained in cohortAudit. "
            "Fixed-LR and digit-policy runs retain their historical rates and are not "
            "reinterpreted as interventions isolating only the schedule or architecture."
        ),
        "",
    ]

    def value(record: dict, metric: str, version: str) -> str:
        s = record["test"][metric][version]
        if s["mean"] is None:
            return "n/a"
        return (
            f"{s['mean']:.3f} +/- {s['sd']:.3f}"
            if s["sd"] is not None
            else f"{s['mean']:.3f}"
        )

    sections = [
        (
            "Full data, 50 epochs",
            [r for r in records if r["family"] == "full" and r["epochs"] == 50],
        ),
        (
            "4096 images, 50 epochs",
            [
                r
                for r in records
                if r["family"] == "subset" and r["size"] == 4096 and r["epochs"] == 50
            ],
        ),
        (
            "1024 images, 200 epochs",
            [
                r
                for r in records
                if r["family"] == "subset" and r["size"] == 1024 and r["epochs"] == 200
            ],
        ),
        *[
            (
                f"{mode.title()} label noise, SD {std}",
                [
                    r
                    for r in records
                    if r["family"] == "noise" and r["noise"] == mode and r["std"] == std
                ],
            )
            for mode in ("fixed", "fresh")
            for std in (1, 3, 5, 10)
        ],
    ]
    for title, rows in sections:
        rows.sort(key=lambda r: r["test"]["mae_mean"]["new"]["mean"])
        previous = sorted(rows, key=lambda r: r["test"]["mae_mean"]["old"]["mean"])
        lines += [
            f"## {title}",
            "",
            "| Method | n | Peak LR old -> new | Rank old -> new | Mean MAE old | Mean MAE new | Mode MAE old | Mode MAE new |",
            "|---|---:|---|---|---:|---:|---:|---:|",
        ]
        for rank, row in enumerate(rows, 1):
            old_rank = next(i for i, r in enumerate(previous, 1) if r is row)
            lines.append(
                f"| {row['label']} | {len(row['seeds'])} | {row['oldLr']:g} -> {row['newLr']:g} "
                f"| {old_rank} -> {rank} | {value(row, 'mae_mean', 'old')} "
                f"| {value(row, 'mae_mean', 'new')} | {value(row, 'mae_mode', 'old')} "
                f"| {value(row, 'mae_mode', 'new')} |"
            )
        lines.append("")
    (output / "LR_COMPARISON.md").write_text("\n".join(lines).rstrip() + "\n")
    flat = []
    for record in records:
        flat.append({
            **{
                k: record[k]
                for k in (
                    "family",
                    "label",
                    "size",
                    "epochs",
                    "noise",
                    "std",
                    "oldLr",
                    "newLr",
                    "changed",
                )
            },
            "n": len(record["seeds"]),
            **{
                f"{phase}_{metric}_{version}_{statistic}": value
                for phase in ("best", "test")
                for metric, versions in record[phase].items()
                for version in ("old", "new", "pairedDelta")
                for statistic, value in versions[version].items()
            },
        })
    pl.DataFrame(flat, infer_schema_length=None).write_csv(output / "lr-comparison.csv")


def render_modes(*, bundle: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    column = bundle["curveColumns"].index("mae_mode")
    plt.rcParams.update({
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "savefig.dpi": 150,
    })
    with PdfPages(path) as pdf:
        for page, spec in enumerate(bundle["figures"][:6], 1):
            panels = [p for p in spec["panels"] if p["metric"] == "mae_mean"]
            cols = 2 if len(panels) == 4 else 1
            rows = math.ceil(len(panels) / cols)
            foot = (
                "Shared-LR revision, 2026-10-22. Faint: raw; bold: moving mean "
                "(5 epochs; 10 for 200-epoch runs). Bands: between-seed sample SD. "
                "Axes emphasize later training. Checkpoints use mean-age MAE. "
                + spec["note"]
            )
            caption = "\n".join(textwrap.wrap(foot, width=155))
            caption_height = (len(caption.splitlines()) * 11 + 8) / 72
            legend_count = len({g["legendKey"] for p in panels for g in p["groups"]})
            footer_height = (
                caption_height + (math.ceil(legend_count / 3) * 16 + 12) / 72 + 0.4
            )
            height = 4.1 * rows + footer_height + 0.8
            fig, axes = plt.subplots(rows, cols, figsize=(14, height), squeeze=False)
            legend = {}
            for ax, panel in zip(axes.flat, panels, strict=True):
                curves = []
                for group in panel["groups"]:
                    selected = [
                        {
                            "epochs": bundle["runs"][identifier]["epochs"],
                            "curves": [
                                [p[0], p[column]]
                                for p in bundle["runs"][identifier]["curves"]
                            ],
                        }
                        for identifier in group["ids"]
                    ]
                    curve = series(
                        runs=selected,
                        metric="mae_mean",
                        window=10 if panel["horizon"] == 200 else 5,
                    )
                    raw = series(runs=selected, metric="mae_mean", window=1)
                    ax.plot(
                        raw["x"], raw["mean"], color=group["color"], alpha=0.18, lw=0.8
                    )
                    line = ax.plot(
                        curve["x"],
                        curve["mean"],
                        color=group["color"],
                        ls="--" if group["dash"] == "dash" else "-",
                        lw=2,
                    )[0]
                    if curve["sd"]:
                        ax.fill_between(
                            curve["x"],
                            [
                                v - s
                                for v, s in zip(curve["mean"], curve["sd"], strict=True)
                            ],
                            [
                                v + s
                                for v, s in zip(curve["mean"], curve["sd"], strict=True)
                            ],
                            color=group["color"],
                            alpha=0.09,
                        )
                    curves.append(curve)
                    legend.setdefault(group["legendKey"], (line, group["legendLabel"]))
                ax.set(
                    title=panel["title"],
                    xlabel="Epoch",
                    ylabel="Validation mode MAE (years)",
                    xlim=(20 if panel["horizon"] == 200 else 5, panel["horizon"]),
                    ylim=panel_range(
                        curves=curves, metric="mae_mean", horizon=panel["horizon"]
                    ),
                )
                ax.grid(alpha=0.16)
            fig.suptitle(spec["title"] + " | Greedy decoding", fontsize=16, y=0.98)
            handles, labels = zip(*legend.values(), strict=True)
            fig.legend(
                handles,
                labels,
                loc="lower center",
                bbox_to_anchor=(0.5, (caption_height + 0.18) / height),
                ncol=3,
                fontsize=9,
            )
            fig.text(
                0.5,
                0.1 / height,
                caption,
                ha="center",
                va="bottom",
                fontsize=8,
            )
            fig.text(0.98, 0.015, f"{page} / 6", ha="right", fontsize=8)
            fig.tight_layout(rect=(0, (footer_height + 0.1) / height, 1, 0.94), h_pad=2)
            pdf.savefig(fig)
            fig.savefig(path.parent / f"mode-{page:02d}.png")
            plt.close(fig)


def refresh(*, args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    payload, changes, raw = assemble(
        original=args.original,
        reruns=args.reruns,
        old_bundle=args.historical_bundle,
    )
    snapshot = args.output_dir / "report-data.json"
    write(path=snapshot, value=payload)
    write(path=args.output_dir / "rerun-results.json", value=raw)
    write(path=args.output_dir / "lr-comparison.json", value=changes)
    comparison_tables(records=changes, output=args.output_dir)
    summaries = summarize(runs=payload["runs"])
    write(path=args.output_dir / "final-summaries.json", value=summaries)
    write(path=args.output_dir / "cohort-audit.json", value=payload["cohortAudit"])
    for filename in ("plan.json", "migration.json", "status.json"):
        shutil.copy2(args.reruns / filename, args.output_dir / filename)
    initial = {
        "sourceDataSha256": hashlib.sha256(snapshot.read_bytes()).hexdigest(),
        "runs": {},
        "records": {},
    }
    for run in payload["runs"]:
        if run["initial"]:
            digest = hashlib.sha256(
                json.dumps(run["initial"], sort_keys=True).encode()
            ).hexdigest()
            initial["runs"][run["id"]] = digest
            initial["records"][digest] = run["initial"]
    initial_path = args.output_dir / "epoch-zero.json"
    write(path=initial_path, value=initial)
    build(data=snapshot, output_dir=args.bundle_dir, epoch_zero_path=str(initial_path))
    bundle = read(path=args.bundle_dir / "bundle-data.json")
    render_modes(bundle=bundle, path=args.mode_pdf)
    article = {
        **{
            k: bundle[k]
            for k in (
                "sourceSha256",
                "snapshot",
                "sourceCohort",
                "sourceCohorts",
                "cohortAudit",
                "dataFingerprint",
                "curveColumns",
                "selection",
                "figures",
                "runs",
                "learningRateRevision",
            )
        },
        "source": str(snapshot),
        "modeDefinition": "Global argmax over all 100 ages, including the joint digit policy.",
        "checkpointSelection": "Minimum clean validation mean-age MAE; no separate mode selection.",
    }
    article["figures"] = [f for f in article["figures"] if f["id"] != "04-stat-widths"]
    args.writeup_data.parent.mkdir(parents=True, exist_ok=True)
    args.writeup_data.write_text(
        "window.UTKFACE_DATA="
        + json.dumps(article, separators=(",", ":"), allow_nan=False).replace(
            "<", "\\u003c"
        )
        + ";\n"
    )
    write(
        path=args.output_dir / "refresh-audit.json",
        value={
            **payload["learningRateRevision"],
            "scientificEvaluations": len(payload["runs"]),
            "modePages": 6,
            "figurePages": len(bundle["figures"]),
            "curveColumns": payload["curveColumns"],
            "exactReplacementMappingVerified": True,
            "executedConfigsDifferOnlyInLearningRateOutputDevice": True,
            "rawHistoriesAndCheckpointMetricsVerified": True,
            "historySha256ByRun": {
                r["id"]: r["historySha256"] for r in payload["runs"]
            },
        },
    )
    print(json.dumps(payload["learningRateRevision"], indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    for argument in (
        "original",
        "reruns",
        "historical-bundle",
        "output-dir",
        "bundle-dir",
        "mode-pdf",
        "writeup-data",
    ):
        parser.add_argument("--" + argument, type=Path, required=True)
    refresh(args=parser.parse_args())


if __name__ == "__main__":
    main()
