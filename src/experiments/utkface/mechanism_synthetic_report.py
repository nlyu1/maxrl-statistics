import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import polars as pl

from src.experiments.utkface.mechanism_report import aggregate, plt, read, save
from src.experiments.utkface.run import write_json

COLORS = ("#b2182b", "#ef8a62", "#2166ac", "#4d4d4d")


def figure(*, rows: list[dict], methods: list[str], title: str, difference: bool):
    result, axes = plt.subplots(1, 2, figsize=(12, 5.3))
    for axis, law in zip(axes, ("binomial", "endpoint"), strict=True):
        selected = [row for row in rows if row["law"] == law]
        for method, color in zip(methods, COLORS[:len(methods)], strict=True):
            members = [row for row in selected if row["method"] == method]
            sizes = sorted({row["training_size"] for row in members})
            values = [
                np.asarray([row["value"] for row in members if row["training_size"] == n])
                for n in sizes
            ]
            for n, scores in zip(sizes, values, strict=True):
                axis.scatter(np.full(len(scores), n), scores, color=color, s=17, alpha=0.35)
            axis.errorbar(
                sizes,
                [v.mean() for v in values],
                yerr=[v.std(ddof=1) for v in values],
                color=color,
                marker="o",
                markersize=4,
                capsize=3,
                label=method,
            )
        axis.set_xscale("log", base=2)
        axis.set_xticks(sorted({row["training_size"] for row in selected}))
        axis.set_xticklabels(sorted({row["training_size"] for row in selected}))
        axis.set(
            title=f"{law.capitalize()} labels; same conditional mean",
            xlabel="Training examples",
            ylabel=("MSE difference from fixed scalar" if difference else "Conditional-mean MSE")
            + " (years²)",
        )
        if difference:
            axis.axhline(0, color="black", linewidth=0.8)
        axis.grid(alpha=0.2)
    handles, labels = axes[0].get_legend_handles_labels()
    result.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.91), ncol=2)
    result.suptitle(title)
    return result


def report(*, source_dir: Path, audit_dir: Path, output: Path) -> dict:
    source = read(source_dir / "report.json")
    audit = read(audit_dir / "report.json")
    verification = read(audit_dir / "independent-verification.json")
    if (
        source["status"] != "complete"
        or audit["status"] != "complete"
        or verification["status"] != "passed"
        or not source["all_gradients_converged"]
        or not all(row["gradient_tolerance_met"] for row in audit["fits"])
    ):
        raise ValueError("Require complete, converged, independently verified synthetic reports")
    actual_hash = hashlib.sha256((source_dir / "report.json").read_bytes()).hexdigest()
    if actual_hash != audit["protocol"]["source_report_sha256"]:
        raise ValueError("Scalar-ridge audit refers to a different source report")
    output.mkdir(parents=True, exist_ok=True)
    residual_rows = []
    for row in source["paired_differences"]:
        if row["comparison"] == "validation_selected":
            continue
        method = (
            "Full: validation-selected residual ridge"
            if row["comparison"] == "validation_selected_full"
            else f"Full: residual ridge {row['residual_penalty']}"
        )
        residual_rows.append({
            **{key: row[key] for key in ("law", "training_size", "seed")},
            "method": method,
            "value": row["oracle_mse_difference"],
        })
    scalar_rows = []
    for row in audit["paired_differences"]:
        for field, method in (
            ("fixed_tied_mse", "Scalar: fixed score ridge"),
            ("selected_tied_mse", "Scalar: validation-selected score ridge"),
            ("selected_full_mse", "Full: validation-selected residual ridge"),
        ):
            scalar_rows.append({
                **{key: row[key] for key in ("law", "training_size", "seed")},
                "method": method,
                "value": row[field],
            })
    files = []
    for name, rows, title, difference, note in (
        (
            "residual-regularization",
            residual_rows,
            "Extra categorical directions: penalty and sample-size effects",
            True,
            "Oracle input split 3. Negative differences favor the full model. Score ridge is fixed at 0.001.",
        ),
        (
            "scalar-regularization",
            scalar_rows,
            "Matched validation candidate counts: scalar versus full regularization",
            False,
            "Fresh oracle input split 4; same five training blocks as split 3. Each selected family has three candidates.",
        ),
    ):
        pl.DataFrame(rows).write_csv(output / f"{name}-individual.tsv", separator="\t")
        aggregate(rows, ["law", "training_size", "method"]).write_csv(
            output / f"{name}-summary.tsv", separator="\t"
        )
        chart = figure(rows=rows, methods=list(dict.fromkeys(row["method"] for row in rows)),
                       title=title, difference=difference)
        files += save(
            figure=chart,
            output=output,
            name=name,
            formats=("png", "svg"),
            note="Dots: five paired training seeds. Lines and bars: mean ± sample SD. "
            + note + " Metric is excess risk for estimating the known conditional mean.",
        )
    paths = [source_dir / "report.json", audit_dir / "report.json", audit_dir / "independent-verification.json"]
    result = {
        "status": "complete",
        "unique_fits": len(audit["fits"]),
        "new_score_ridge_fits": sum(not row["reused_source_fit"] for row in audit["fits"]),
        "source_artifacts": [{"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()} for path in paths],
        "independent_verification": verification,
        "files": files,
    }
    write_json(path=output / "report.json", payload=result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--audit-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = report(source_dir=args.source_dir, audit_dir=args.audit_dir, output=args.output_dir)
    print(json.dumps({"status": result["status"], "unique_fits": result["unique_fits"], "files": result["files"]}))


if __name__ == "__main__":
    main()
