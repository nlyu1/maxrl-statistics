import argparse
import itertools
import math
from pathlib import Path

import polars as pl

from src.experiments.utkface.adaptation_train import file_sha256, git_head
from src.experiments.utkface.mechanism_dossier import DECISION_MATRIX_SHA256, require
from src.experiments.utkface.mechanism_noise_audit import ADJUSTMENTS, BANDS
from src.experiments.utkface.mechanism_probe_report import checked_file
from src.experiments.utkface.mechanism_report import aggregate, read
from src.experiments.utkface.run import write_json

COMMIT = "e8b7ebe7ef44ae69ca8a6e1b60f50f400fc14c84"
TIMES = (50, 200, 500)
INTERVALS = ((50, 200), (200, 500), (50, 500))
VARIANTS = (
    "categorical_mse",
    "categorical_tailrl",
    "categorical_ce",
    "scalar_mse",
    "scalar_mae",
    "binomial_ce",
    "binomial_tailrl",
    "binomial_ce_free",
    "binomial_ce_affine",
    "binomial_tailrl_free",
    "binomial_tailrl_affine",
)
PAIRS = (
    ("scalar_mae", "categorical_tailrl"),
    ("binomial_tailrl_free", "binomial_tailrl_affine"),
    ("binomial_ce_free", "binomial_ce_affine"),
)
KEYS = ["variant", "noise_sd", "statistic", "readout", "time", "kind", "matrix_status", "prediction_context"]
DESCRIPTIVE_STATS = {
    "archived_test_mae",
    "selected_epoch",
    "selected_to_final_test_change",
    "best_validation_mae",
    "extra_best_validation_improvement",
    "archived_validation_mae",
    "train_objective",
    "probe_test_mae",
    "readout_replacement_gain",
    "train_clean_mae",
    "train_supplied_mae",
    "entropy",
    "policy_variance",
    "own_field_slope",
}
PREDICTIONS = {
    "P1": "Joint longer useful scalar-MAE horizon: later selection, more additional best-validation gain, corroborating fixed-time N/R changes. The matrix labels its individual contrasts D.",
    "P2": "Joint Tail objective/decision divergence; objective decrease and native validation/test worsening must be considered together, without automatic pattern scoring.",
    "P3": "SD10 Tail free-minus-affine probe errors at 500 and differences of 50-to-500 changes are signed positive transfer predictions. CE/clean/selected-error comparisons remain D.",
}


def nullable_difference(*, first, second):
    return None if first is None or second is None else first - second


def row(*, job: dict, statistic: str, readout: str, time: str, kind: str, value) -> dict:
    nullable = statistic in {"entropy", "policy_variance", "own_field_slope"} or (
        statistic == "train_objective" and time in {"selected", "selected_to_500"}
    )
    if value is None and not nullable:
        raise ValueError(f"Missing defined statistic: {job['identifier']}/{statistic}/{time}")
    if value is not None and (isinstance(value, bool) or not math.isfinite(value)):
        raise ValueError(f"Nonfinite statistic: {job['identifier']}/{statistic}/{time}")
    context = (
        "P2"
        if job["variant"] == "categorical_tailrl"
        and statistic in {"train_objective", "archived_validation_mae", "probe_test_mae"}
        else ""
    )
    return dict(
        {key: job[key] for key in ("identifier", "variant", "block", "noise_sd")},
        statistic=statistic,
        readout=readout,
        time=time,
        kind=kind,
        value=value,
        matrix_status="D" if statistic in DESCRIPTIVE_STATS else "X",
        prediction_context=context,
    )


def trajectory(*, job: dict, statistic: str, readout: str, values: dict) -> list[dict]:
    rows = [
        row(job=job, statistic=statistic, readout=readout, time=f"e{epoch}", kind="level", value=values[epoch])
        for epoch in TIMES
    ]
    rows += [
        row(
            job=job,
            statistic=statistic,
            readout=readout,
            time=f"{start}_to_{end}",
            kind="change",
            value=nullable_difference(first=values[end], second=values[start]),
        )
        for start, end in INTERVALS
    ]
    return rows


def unique(*, rows: list[dict], keys: tuple[str, ...]) -> dict:
    result = {tuple(row[key] for key in keys): row for row in rows}
    if len(result) != len(rows):
        raise ValueError(f"Duplicate input/statistic rows for {keys}")
    return result


def complete_inputs(
    *, training_dir: Path, probe_dir: Path, decision_matrix: Path
) -> tuple[list[dict], dict, dict, dict, dict]:
    training_path, probe_path = training_dir / "report.json", probe_dir / "report.json"
    training, probe = read(training_path), read(probe_path)
    require(
        condition=training["schema"] == "utkface-mechanism-report-v1"
        and training["scientific_complete"] is True
        and training["planned_jobs"] == training["completed_jobs"] == 66
        and training["planned_cohort_valid"] is True
        and training["epochs"] == 500
        and training["initialization"] == "scratch"
        and not training["missing_jobs"],
        path=training_path,
        expectation="complete 66-fit scratch500 training report before any table analysis",
    )
    coverage = probe["coverage"]
    require(
        condition=probe["schema"] == "utkface-mechanism-probe-report-v1"
        and probe["scientific_complete"] is True
        and coverage["planned_sources"] == coverage["completed_sources"] == 198
        and coverage["cohort_valid"] is True
        and not any(
            coverage[key] for key in ("missing_sources", "missing_cohort_cells", "missing_shards", "unexpected_shards")
        )
        and probe["noise_audit_requested"] is True
        and probe["noise_completed_sources"] == 198
        and not probe["noise_missing_sources"]
        and Path(probe["training_root"]).resolve() == Path(training["root"]).resolve(),
        path=probe_path,
        expectation="complete 198-source fixed-time probe and noise audits",
    )
    plan_path = Path(training["root"]) / "plan.json"
    plan = read(plan_path)
    jobs = plan["jobs"]
    cells = {(job["variant"], job["block"], job["noise_sd"]) for job in jobs}
    require(
        condition=plan["git_commit"] == COMMIT
        and len(jobs) == 66
        and len({job["identifier"] for job in jobs}) == 66
        and cells == set(itertools.product(VARIANTS, range(3), (0, 10))),
        path=plan_path,
        expectation="all 11 methods, two noise conditions and three blocks from frozen E8 source",
    )
    paths = [
        training_path,
        probe_path,
        plan_path,
        decision_matrix,
        training_dir / "endpoints-individual.tsv",
        training_dir / "trajectories-individual.tsv",
        probe_dir / "long.tsv",
        probe_dir / "noise-long.tsv",
    ]
    require(
        condition=file_sha256(path=decision_matrix) == DECISION_MATRIX_SHA256,
        path=decision_matrix,
        expectation="the exact retained blind decision-matrix SHA-256 before table reads",
    )
    hashes = {str(path.resolve()): file_sha256(path=path) for path in paths}
    matrix = decision_matrix.read_text()
    require(
        condition=all(label in matrix for label in ("P1:", "P2:", "P3:", "D:", "X:")),
        path=decision_matrix,
        expectation="the retained blind P/D/X long-horizon decision matrix",
    )
    for artifact in [*coverage["input_reports"], *probe["noise_input_reports"]]:
        path = Path(artifact["path"])
        checked_file(path=path, expected=artifact["sha256"])
        hashes[str(path.resolve())] = artifact["sha256"]
    for artifact in coverage["input_reports"]:
        source = read(Path(artifact["path"]))
        require(
            condition=source["git_commit"] == COMMIT
            and source["requested"]["plan_sha256"] == hashes[str(plan_path.resolve())]
            and set(source["requested"]["checkpoints"]) == {"epoch_050", "epoch_200", "final"},
            path=Path(artifact["path"]),
            expectation="source-matched E8 probes at fixed epochs 50/200/500",
        )
    endpoints = unique(rows=training["individual_endpoints"], keys=("identifier", "checkpoint", "split", "metric"))
    curves = (
        pl
        .scan_csv(training_dir / "trajectories-individual.tsv", separator="\t")
        .filter(
            ((pl.col("metric") == "mae_mean") & (pl.col("split") == "val"))
            | ((pl.col("metric") == "train_loss") & (pl.col("split") == "train"))
        )
        .collect()
        .to_dicts()
    )
    histories = unique(rows=curves, keys=("identifier", "split", "metric", "epoch"))
    probe_rows = (
        pl
        .scan_csv(probe_dir / "long.tsv", separator="\t")
        .filter((pl.col("split") == "test") & (pl.col("target") == "clean") & (pl.col("metric") == "mae"))
        .collect()
        .to_dicts()
    )
    readouts = unique(rows=probe_rows, keys=("identifier", "readout", "epoch"))
    noise_rows = pl.read_csv(probe_dir / "noise-long.tsv", separator="\t", infer_schema_length=None).to_dicts()
    noise = unique(rows=noise_rows, keys=("identifier", "epoch", "band", "adjustment"))
    identifiers = {job["identifier"] for job in jobs}
    expected_history = {(identifier, "val", "mae_mean", epoch) for identifier in identifiers for epoch in range(501)}
    expected_history |= {
        (identifier, "train", "train_loss", epoch) for identifier in identifiers for epoch in range(1, 501)
    }
    require(
        condition=set(histories) == expected_history,
        path=training_dir / "trajectories-individual.tsv",
        expectation="complete individual validation0..500 and training-loss1..500 histories",
    )
    require(
        condition=set(readouts)
        == set(itertools.product(identifiers, ("native", "supplied_labels", "oracle_clean_labels"), TIMES)),
        path=probe_dir / "long.tsv",
        expectation="all 594 method/block/noise/readout/epoch test-MAE entries",
    )
    require(
        condition=set(noise) == set(itertools.product(identifiers, TIMES, BANDS, ADJUSTMENTS)),
        path=probe_dir / "noise-long.tsv",
        expectation="all 198-source band/adjustment noise records without duplicates",
    )
    if any(
        value["value"] is None or not math.isfinite(value["value"])
        for value in itertools.chain(histories.values(), readouts.values())
    ):
        raise ValueError("Validation histories, training objectives and test MAEs must be finite and defined")
    planned = {job["identifier"]: job for job in jobs}
    for value in itertools.chain(endpoints.values(), histories.values(), readouts.values(), noise.values()):
        job = planned[value["identifier"]]
        if any(value[key] != job[key] for key in ("variant", "block", "noise_sd")):
            raise ValueError(f"Plan/table identity mismatch: {job['identifier']}")
    return (
        jobs,
        endpoints,
        histories,
        readouts,
        {"noise": noise, "input_sha256": hashes, "training_root": training["root"]},
    )


def extract(*, jobs: list[dict], endpoints: dict, histories: dict, readouts: dict, noise: dict) -> list[dict]:
    rows = []
    for job in jobs:
        identifier = job["identifier"]
        validation = {epoch: histories[(identifier, "val", "mae_mean", epoch)]["value"] for epoch in range(501)}
        losses = {epoch: histories[(identifier, "train", "train_loss", epoch)]["value"] for epoch in range(1, 501)}
        selected = endpoints[(identifier, "trajectory", "train", "selected_epoch")]["value"]
        if int(selected) != selected or int(selected) != min(validation, key=validation.__getitem__):
            raise ValueError(f"Selected epoch differs from earliest validation minimum: {identifier}")
        selected = int(selected)
        for checkpoint, epoch in (("selected", selected), ("final", 500)):
            archived = endpoints[(identifier, checkpoint, "val", "mae_mean")]["value"]
            if not math.isclose(archived, validation[epoch], abs_tol=1e-10, rel_tol=1e-10):
                raise ValueError(f"Archived validation differs from history: {identifier}")
            rows.append(
                row(
                    job=job,
                    statistic="archived_test_mae",
                    readout="archived_native",
                    time=checkpoint,
                    kind="level",
                    value=endpoints[(identifier, checkpoint, "test", "mae_mean")]["value"],
                )
            )
        selected_test = endpoints[(identifier, "selected", "test", "mae_mean")]["value"]
        final_test = endpoints[(identifier, "final", "test", "mae_mean")]["value"]
        rows += [
            row(
                job=job,
                statistic="selected_epoch",
                readout="archived_native",
                time="selected",
                kind="level",
                value=selected,
            ),
            row(
                job=job,
                statistic="selected_to_final_test_change",
                readout="archived_native",
                time="selected_to_final",
                kind="change",
                value=final_test - selected_test,
            ),
        ]
        best = {epoch: min(validation[e] for e in range(epoch + 1)) for epoch in TIMES}
        rows += [
            row(
                job=job,
                statistic="best_validation_mae",
                readout="archived_native",
                time=f"through_{epoch}",
                kind="level",
                value=best[epoch],
            )
            for epoch in TIMES
        ]
        rows.append(
            row(
                job=job,
                statistic="extra_best_validation_improvement",
                readout="archived_native",
                time="50_to_500",
                kind="gain",
                value=best[50] - best[500],
            )
        )
        rows += trajectory(job=job, statistic="archived_validation_mae", readout="archived_native", values=validation)
        rows += trajectory(job=job, statistic="train_objective", readout="archived_native", values=losses)
        selected_loss = losses[selected] if selected else None
        rows += [
            row(
                job=job,
                statistic="train_objective",
                readout="archived_native",
                time="selected",
                kind="level",
                value=selected_loss,
            ),
            row(
                job=job,
                statistic="train_objective",
                readout="archived_native",
                time="selected_to_500",
                kind="change",
                value=nullable_difference(first=losses[500], second=selected_loss),
            ),
        ]
        native = {epoch: readouts[(identifier, "native", epoch)]["value"] for epoch in TIMES}
        for readout in ("native", "supplied_labels", "oracle_clean_labels"):
            values = {epoch: readouts[(identifier, readout, epoch)]["value"] for epoch in TIMES}
            if readout == "oracle_clean_labels" and job["noise_sd"] == 0:
                if any(
                    not math.isclose(
                        values[epoch],
                        readouts[(identifier, "supplied_labels", epoch)]["value"],
                        abs_tol=1e-12,
                        rel_tol=1e-12,
                    )
                    for epoch in TIMES
                ):
                    raise ValueError(f"Clean supplied/oracle ridge differs: {identifier}")
                continue
            rows += trajectory(job=job, statistic="probe_test_mae", readout=readout, values=values)
            if readout != "native":
                rows += [
                    row(
                        job=job,
                        statistic="readout_replacement_gain",
                        readout=readout,
                        time=f"e{epoch}",
                        kind="level",
                        value=native[epoch] - values[epoch],
                    )
                    for epoch in TIMES
                ]
        for field, statistic in (
            ("clean_mae", "train_clean_mae"),
            ("supplied_mae", "train_supplied_mae"),
            ("entropy", "entropy"),
            ("policy_variance", "policy_variance"),
        ):
            values = {epoch: noise[(identifier, epoch, "all", "raw")][field] for epoch in TIMES}
            if statistic in {"entropy", "policy_variance"} and any(
                (value is None) != job["variant"].startswith("scalar_") for value in values.values()
            ):
                raise ValueError(f"Policy diagnostic availability differs from head family: {identifier}")
            rows += trajectory(job=job, statistic=statistic, readout="native", values=values)
        slopes = {epoch: noise[(identifier, epoch, "all", "within_exact_age")]["slope"] for epoch in TIMES}
        if job["noise_sd"] == 0 and any(value is not None for value in slopes.values()):
            raise ValueError(f"Clean own-field slopes must remain undefined: {identifier}")
        if job["noise_sd"] == 10 and any(value is None for value in slopes.values()):
            raise ValueError(f"All-age noisy own-field slopes must be defined: {identifier}")
        rows += trajectory(job=job, statistic="own_field_slope", readout="native", values=slopes)
    unique(rows=rows, keys=("identifier", "statistic", "readout", "time", "kind"))
    return rows


def paired_rows(*, rows: list[dict]) -> list[dict]:
    keys = ("variant", "block", "noise_sd", "statistic", "readout", "time", "kind")
    lookup = unique(rows=rows, keys=keys)
    paired = []
    for first, second in PAIRS:
        for key, left in lookup.items():
            if key[0] != first:
                continue
            if left["statistic"] == "train_objective" and (
                first == "scalar_mae" or left["time"] in {"selected", "selected_to_500"}
            ):
                continue
            right = lookup[(second, *key[1:])]
            a, b = (
                (right, left)
                if first == "scalar_mae" and left["statistic"] == "selected_to_final_test_change"
                else (left, right)
            )
            status = (
                "P"
                if first == "binomial_tailrl_free"
                and left["noise_sd"] == 10
                and left["statistic"] == "probe_test_mae"
                and left["time"] in {"e500", "50_to_500"}
                else "D"
            )
            if "X" in {left["matrix_status"], right["matrix_status"]}:
                status = "X"
            context = "P1" if first == "scalar_mae" else "P3" if first == "binomial_tailrl_free" else ""
            if status == "X":
                context = ""
            paired.append(
                dict(
                    left,
                    variant=a["variant"],
                    identifier=a["identifier"],
                    contrast=f"{a['variant']} - {b['variant']}",
                    first_identifier=a["identifier"],
                    second_identifier=b["identifier"],
                    first_value=a["value"],
                    second_value=b["value"],
                    value=nullable_difference(first=a["value"], second=b["value"]),
                    matrix_status=status,
                    prediction_context=context,
                )
            )
    return paired


def report(*, training_dir: Path, probe_dir: Path, decision_matrix: Path, output_dir: Path) -> dict:
    require(
        condition=not any(output_dir.resolve().is_relative_to(path.resolve()) for path in (training_dir, probe_dir)),
        path=output_dir,
        expectation="output outside archived input reports",
    )
    jobs, endpoints, histories, readouts, metadata = complete_inputs(
        training_dir=training_dir, probe_dir=probe_dir, decision_matrix=decision_matrix
    )
    rows = extract(jobs=jobs, endpoints=endpoints, histories=histories, readouts=readouts, noise=metadata.pop("noise"))
    paired = paired_rows(rows=rows)
    frames = {
        "individual": pl.DataFrame(rows, infer_schema_length=None),
        "summary": aggregate(rows, KEYS),
        "paired_individual": pl.DataFrame(paired, infer_schema_length=None),
        "paired_summary": aggregate(paired, ["contrast", *KEYS[1:]]),
    }
    for name in ("summary", "paired_summary"):
        frames[name] = frames[name].with_columns((pl.col("n_rows") - pl.col("n_defined")).alias("n_undefined"))
        if any(value != 3 for value in frames[name]["n_rows"]):
            raise ValueError(f"Every {name} cell must retain three paired joint blocks")
    for name, expected in metadata["input_sha256"].items():
        checked_file(path=Path(name), expected=expected)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in frames.items():
        frame.write_csv(output_dir / f"{name.replace('_', '-')}.tsv", separator="\t")
    result = dict(
        schema="utkface-mechanism-horizon-report-v1",
        scientific_complete=True,
        source_commit=COMMIT,
        analysis_git_commit=git_head(),
        analysis_source_sha256=file_sha256(path=Path(__file__)),
        completed_fits=66,
        completed_probe_sources=198,
        noise_audits=198,
        n_blocks=3,
        predictions=PREDICTIONS,
        status_scope="P3 signed transfer cells are P. The matrix's descriptive scalar-MAE/Tail quantities remain D with P1 context; P2 is a joint interpretation, not an automatic winner rule. No new X statistics generated.",
        changes="Later minus earlier within the same method/block/noise/readout; all 50-to-200,200-to-500,50-to-500 intervals retained.",
        paired_scope="First minus second, formed within block before averaging. Matrix d_G is deliberately Tail minus scalar MAE; cross-objective loss contrasts and different-selected-epoch loss contrasts are omitted.",
        clean_oracle_scope="Clean oracle comparisons are collapsed to supplied ridge after equality checking; noisy oracle remains a diagnostic using extra labels.",
        probe_scope="The common standardized ridge procedure retunes alpha by clean-validation MSE per encoder; fitted weights differ. Archived selected/final errors and recomputed fixed-time errors are never mixed into temporal changes.",
        undefined_scope="Epoch0 has no training objective row; clean field slopes and scalar entropy/policy variance remain null. No 50-epoch validation-selected test outcome is fabricated.",
        best_validation_scope="Windows include epoch0; extra improvement is min(V0..50)-min(V0..500), nonnegative by construction with additional selection opportunities.",
        objective_scope="Native archived train_loss; compare within objective and aligned epochs. Different-selected-epoch objective comparisons remain individual only.",
        secondary_diagnostics="Age bands and raw slopes remain available in the hashed original noise-long.tsv; this extractor uses all-age within-exact-age slopes.",
        uncertainty="Mean and sample SD (ddof=1) across three joint blocks; no p-values, confidence intervals, equivalence claims or winner selection. Epochs/refits are dependent observations.",
        tables={
            name: {
                "path": str((output_dir / f"{name.replace('_', '-')}.tsv").resolve()),
                "rows": frame.height,
                "sha256": file_sha256(path=output_dir / f"{name.replace('_', '-')}.tsv"),
            }
            for name, frame in frames.items()
        },
        **metadata,
    )
    write_json(path=output_dir / "report.json", payload=result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in ("training-dir", "probe-dir", "decision-matrix", "output-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    result = report(
        training_dir=args.training_dir,
        probe_dir=args.probe_dir,
        decision_matrix=args.decision_matrix,
        output_dir=args.output_dir,
    )
    print(result["schema"], result["scientific_complete"], str(args.output_dir / "report.json"))


if __name__ == "__main__":
    main()
