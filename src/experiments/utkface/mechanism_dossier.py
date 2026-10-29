import argparse
import csv
import hashlib
import io
import json
import re
import shutil
import subprocess
import tarfile
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from src.experiments.utkface.adaptation_train import file_sha256
from src.experiments.utkface.run import write_json

COHORTS = {
    "imagenet": (78, 156, 50, "imagenet", "e8b7ebe7ef44ae69ca8a6e1b60f50f400fc14c84"),
    "initialization": (36, 72, 50, "imagenet", "55ea16131f05a0d60e191e3083a216db19bb3e65"),
    "age-assignment": (6, 12, 50, "imagenet", "e71bebcfcc990ab35daa8e1122a6f5eb4e6abfb8"),
    "scratch500": (66, 198, 500, "scratch", "e8b7ebe7ef44ae69ca8a6e1b60f50f400fc14c84"),
}
SOURCE_NAMES = (
    "report.qmd",
    "_explanation.qmd",
    "_long_horizon.qmd",
    "_prior_evidence.qmd",
    "_design.qmd",
    "_hypotheses.qmd",
    "_theory.qmd",
)
NOTE_NAMES = (
    "evidence-ledger.txt",
    "interpretation-matrix.txt",
    "primary-literature.txt",
    "results-review.txt",
    "probe-review.txt",
    "initialization-review.txt",
    "age-assignment-review.txt",
    "centered-correction-review.txt",
    "synthetic-challenge-review.txt",
    "tail-calibration-proof.txt",
    "long-horizon-decision-matrix.txt",
    "theory-synthesis-v2.txt",
    "final-explanation-challenge.txt",
    "final-long-horizon-review.txt",
)
REPORT_SUFFIXES = {".tsv", ".json", ".png", ".svg", ".pdf"}
MAX_FILE_BYTES = 256 * 1024**2


def require(*, condition: bool, path: Path, expectation: str) -> None:
    if not condition:
        raise ValueError(f"{path}: require {expectation}")


def load(*, path: Path) -> dict:
    return json.loads(path.read_text())


def population_certificate_summary(*, population: dict) -> dict:
    groups = {
        label: [(row["name"], row["solver"]["chosen_certificate"]) for row in population[key]]
        for label, key in (("direct", "population_cases"), ("corruption", "noise_population_cases"))
    }
    counts = {
        label: {"total": len(rows), "certified": sum(certificate["certified"] is True for _, certificate in rows)}
        for label, rows in groups.items()
    }
    rows = [row for group in groups.values() for row in group]
    worst_case, worst = max(rows, key=lambda row: row[1]["full_simplex_frank_wolfe_gap"])
    certified = sum(group["certified"] for group in counts.values())
    return {
        "groups": counts,
        "total": len(rows),
        "certified": certified,
        "uncertified": len(rows) - certified,
        "max_frank_wolfe_gap": worst["full_simplex_frank_wolfe_gap"],
        "max_gap_case": worst_case,
        "certificate_scope": worst["certificate_scope"],
    }


def validate_reports(*, root: Path) -> tuple[list[Path], list[dict]]:
    directories, index = [], []
    for name, (fits, probes, epochs, initialization, commit) in COHORTS.items():
        path = root / name / "report.json"
        training = load(path=path)
        require(
            condition=training["schema"] == "utkface-mechanism-report-v1"
            and training["scientific_complete"] is True
            and training["planned_jobs"] == fits
            and training["completed_jobs"] == fits
            and training["planned_cohort_valid"] is True
            and not training["missing_jobs"]
            and training["epochs"] == epochs
            and training["initialization"] == initialization,
            path=path,
            expectation=f"complete scientific training report, {fits}/{fits} fits at epoch {epochs}",
        )
        raw_root = Path(training["root"])
        plan = load(path=raw_root / "plan.json")
        require(
            condition=plan["git_commit"] == commit and len(plan["jobs"]) == fits,
            path=raw_root / "plan.json",
            expectation=f"frozen source {commit} and {fits} jobs",
        )
        probe_path = root / f"{name}-probes" / "report.json"
        probe = load(path=probe_path)
        coverage = probe["coverage"]
        require(
            condition=probe["schema"] == "utkface-mechanism-probe-report-v1"
            and probe["scientific_complete"] is True
            and coverage["planned_sources"] == probes
            and coverage["completed_sources"] == probes
            and coverage["cohort_valid"] is True
            and not any(
                coverage[key]
                for key in ("missing_sources", "missing_cohort_cells", "missing_shards", "unexpected_shards")
            )
            and Path(probe["training_root"]).resolve() == raw_root.resolve()
            and probe["noise_audit_requested"] is True
            and probe["noise_completed_sources"] == probes
            and not probe["noise_missing_sources"],
            path=probe_path,
            expectation=f"complete scientific probe/noise report, {probes}/{probes} sources",
        )
        for artifact in [*coverage["input_reports"], *probe["noise_input_reports"]]:
            require(
                condition=file_sha256(path=Path(artifact["path"])) == artifact["sha256"],
                path=Path(artifact["path"]),
                expectation="unchanged upstream probe/noise report hash",
            )
        directories.extend((path.parent, probe_path.parent))
        index.append({
            "name": name,
            "fits": fits,
            "probes": probes,
            "source_commit": commit,
            "training_root": str(raw_root),
            "raw_plan": str(raw_root / "plan.json"),
            "raw_probe_reports": coverage["input_reports"],
            "raw_noise_reports": probe["noise_input_reports"],
        })
    combined_path = root / "initialization-combined" / "report.json"
    combined = load(path=combined_path)
    require(
        condition=combined["schema"] == "utkface-mechanism-initialization-comparison-v1"
        and combined["scientific_complete"] is True
        and combined["n_blocks"] == 3
        and combined["coverage"] == {"native_rows": 48, "ridge_rows": 96, "total_rows": 144},
        path=combined_path,
        expectation="complete F/R/U/P comparison with 144 rows and three paired blocks",
    )
    for name, expected in combined["input_sha256"].items():
        require(
            condition=file_sha256(path=Path(name)) == expected,
            path=Path(name),
            expectation="unchanged F/R/U/P input hash",
        )
    source_path, audit_path = root / "synthetic/report.json", root / "synthetic-score-ridge/report.json"
    source, audit = load(path=source_path), load(path=audit_path)
    require(
        condition=source["status"] == "complete"
        and source["all_gradients_converged"] is True
        and len(source["fits"]) == 120
        and len({row["fit_id"] for row in source["fits"]}) == 120
        and len(source["selections"]) == 30,
        path=source_path,
        expectation="complete converged 120-fit source synthetic study",
    )
    require(
        condition=audit["status"] == "complete"
        and len(audit["fits"]) == 180
        and len(audit["selections"]) == 30
        and len({row["fit_id"] for row in audit["fits"]}) == 180
        and all(row["gradient_tolerance_met"] for row in audit["fits"])
        and audit["protocol"]["source_report_sha256"] == file_sha256(path=source_path),
        path=audit_path,
        expectation="complete converged 180-fit scalar-ridge audit matched to its source",
    )
    reused = {row["fit_id"] for row in audit["fits"] if row["reused_source_fit"]}
    require(
        condition=reused == {row["fit_id"] for row in source["fits"]} and len(audit["fits"]) - len(reused) == 60,
        path=audit_path,
        expectation="120 retained source fits and exactly 60 new scalar-ridge fits",
    )
    for path, payload, protocol_key in ((source_path, source, "config"), (audit_path, audit, "protocol")):
        frozen_path = path.parent / "frozen-selection.json"
        frozen = load(path=frozen_path)
        require(
            condition=file_sha256(path=frozen_path) == payload["frozen_selection_sha256"]
            and frozen["selections"] == payload["selections"]
            and load(path=path.parent / "protocol.json") == payload[protocol_key]
            and file_sha256(path=path.parent / "parameters.npz") == frozen["parameters_sha256"],
            path=path.parent,
            expectation="unchanged frozen synthetic protocol, selections and small coefficient archive",
        )
    verification_path = audit_path.parent / "independent-verification.json"
    verification = load(path=verification_path)
    require(
        condition=verification["status"] == "passed"
        and not verification["failures"]
        and set(verification["checks"])
        == {
            "source_report_sha",
            "frozen_sha",
            "parameters_sha",
            "frozen_selections",
            "protocol_sha",
            "audit_source_sha",
            "oracle_split_distinct",
        }
        and all(verification["checks"].values())
        and verification["new_fit_count"] == 60
        and verification["all_strong_convexity_fields_correct"] is True
        and set(verification["absolute_tolerances"])
        == {"objective", "gradient", "validation_mse", "oracle_mse", "mean_readout", "reported_gradient"}
        and all(
            verification["maximum_absolute_errors"][key] <= tolerance
            for key, tolerance in verification["absolute_tolerances"].items()
        ),
        path=verification_path,
        expectation="passed independent verification of all 60 new fits and declared tolerances",
    )
    figures_path = root / "synthetic-figures" / "report.json"
    figures = load(path=figures_path)
    require(
        condition=figures["status"] == "complete"
        and figures["unique_fits"] == 180
        and figures["new_score_ridge_fits"] == 60
        and figures["independent_verification"] == verification
        and {Path(row["path"]).resolve() for row in figures["source_artifacts"]}
        == {source_path.resolve(), audit_path.resolve(), verification_path.resolve()},
        path=figures_path,
        expectation="complete synthetic figure report matched to independent verification",
    )
    for artifact in figures["source_artifacts"]:
        require(
            condition=file_sha256(path=Path(artifact["path"])) == artifact["sha256"],
            path=Path(artifact["path"]),
            expectation="unchanged synthetic report hash",
        )
    centered_path = root / "centered-assignment" / "report.json"
    centered = load(path=centered_path)
    require(
        condition=centered["schema"] == "utkface-centered-assignment-audit-v1"
        and centered["all_checks_passed"] is True
        and len(centered["cases"]) == 4
        and all(case["passed"] for case in centered["cases"])
        and centered["target_step_rows"] == 2000
        and Path(centered["input"]).resolve() == (root / "symmetric-swap-verification.json").resolve()
        and file_sha256(path=Path(centered["input"])) == centered["input_sha256"],
        path=centered_path,
        expectation="passed four-case/2000-row centered-assignment audit with unchanged CPU verification",
    )
    calibration_path = root / "calibration.json"
    calibration = load(path=calibration_path)
    require(
        condition=calibration["checks_passed"] is True
        and len(calibration["cases"]) == 18
        and all(case["passed"] for case in calibration["cases"])
        and len(calibration["inverse_kkt"]) == 5
        and all(case["passed"] for case in calibration["scope_controls"].values()),
        path=calibration_path,
        expectation="passed 18-case/five-inverse calibration report and scope controls",
    )
    population_path = Path(calibration["saved_population"]["source"])
    require(
        condition=population_path.resolve() == (root / "population-v2.json").resolve()
        and file_sha256(path=population_path) == calibration["saved_population"]["source_sha256"],
        path=population_path,
        expectation="unchanged population-v2 input referenced by calibration",
    )
    population = load(path=population_path)
    require(
        condition=population["status"] == "complete" and population["all_repository_autograd_checks_passed"] is True,
        path=population_path,
        expectation="completed population report with passed repository autograd checks",
    )
    directories.extend((
        combined_path.parent,
        source_path.parent,
        audit_path.parent,
        figures_path.parent,
        centered_path.parent,
    ))
    index.extend([
        {
            "name": "synthetic",
            "fits": 120,
            "new_fits": 120,
            "reused_fits": 0,
            "source_commit": source["git_commit"],
            "raw_report": str(source_path),
        },
        {
            "name": "synthetic-score-ridge",
            "fits": 180,
            "new_fits": 60,
            "reused_fits": 120,
            "source_commit": audit["git_commit"],
            "raw_report": str(audit_path),
        },
        {
            "name": "calibration",
            "source_commit": calibration["provenance"]["git_commit"],
            "raw_report": str(calibration_path),
        },
        {
            "name": "population-v2",
            "source_commit": population["provenance"]["git_commit"],
            "raw_report": str(population_path),
            "all_population_candidates_certified": population["all_population_candidates_certified"],
            "certificate_summary": population_certificate_summary(population=population),
        },
        {
            "name": "centered-assignment",
            "source_sha256": centered["source_sha256"],
            "raw_report": str(centered_path),
            "raw_input": centered["input"],
        },
    ])
    return directories + [calibration_path, population_path, Path(centered["input"])], index


def committed_sources(*, directory: Path) -> tuple[list[Path], dict]:
    for name in SOURCE_NAMES:
        if not (directory / name).is_file():
            raise FileNotFoundError(directory / name)
    require(
        condition="will supply this section after completion" not in (directory / "_long_horizon.qmd").read_text(),
        path=directory / "_long_horizon.qmd",
        expectation="final long-horizon prose rather than the retained running-cohort placeholder",
    )
    files = sorted(directory.rglob("*.qmd"))
    require(
        condition=not (directory / "_evidence.qmd").exists(),
        path=directory,
        expectation="no source _evidence.qmd; the dossier builder owns the generated index",
    )
    repository = Path(
        subprocess.check_output(["git", "-C", str(directory), "rev-parse", "--show-toplevel"], text=True).strip()
    )
    relative = [str(path.relative_to(repository)) for path in files]
    tracked = subprocess.check_output(
        ["git", "-C", str(repository), "ls-files", "--", *relative], text=True
    ).splitlines()
    dirty = subprocess.check_output(
        ["git", "-C", str(repository), "status", "--porcelain", "--", *relative], text=True
    ).strip()
    require(
        condition=set(tracked) == {Path(name).as_posix() for name in relative} and not dirty,
        path=directory,
        expectation="all Quarto source files tracked and committed without local edits",
    )
    commit = subprocess.check_output(["git", "-C", str(repository), "rev-parse", "HEAD"], text=True).strip()
    return files, {"repository": str(repository), "directory": str(directory), "git_commit": commit}


def evidence_index(*, index: list[dict], source: dict, notes: list[str]) -> str:
    lines = [
        "## Evidence and reproducibility {#sec-evidence}",
        "",
        f"Quarto source snapshot: `{source['git_commit']}`.",
        "",
        "[SHA-256 file index](SHA256SUMS.tsv) · [Machine-readable dossier manifest](dossier-manifest.json)",
        "",
        "Original report bytes are retained. The dossier records completed artifacts; rendering is a separate step.",
        "",
        "All JSON/TSV evidence and frozen Python source are copied. Only featured figures and their SVG/PDF companions are copied; other visualization paths and hashes remain in the manifest.",
        "",
        "| Evidence | Retained fit coverage / probe sources | Frozen training or analysis commit |",
        "|---|---:|---|",
    ]
    for item in index:
        name = item["name"]
        target = (
            f"evidence/{name}/report.json" if name not in {"calibration", "population-v2"} else f"evidence/{name}.json"
        )
        fits = str(item.get("fits", "—"))
        if item.get("reused_fits", 0):
            fits += f" total ({item['reused_fits']} retained + {item['new_fits']} new)"
        elif name == "synthetic":
            fits += " original (also retained in the scalar-ridge audit)"
        lines.append(
            f"| [{name}]({target}) | {fits} / {item.get('probes', '—')} | `{item.get('source_commit', item.get('source_sha256', ''))}` |"
        )
    convex = next(item for item in index if item["name"] == "synthetic-score-ridge")
    image_fits = sum(item["fits"] for item in index if "training_root" in item)
    lines += [
        "",
        (
            f"There are **{image_fits} unique completed image fits** and **{convex['fits']} unique convex fits**. "
            f"The synthetic source's {convex['reused_fits']} fits are included in the scalar-ridge audit; "
            f"that audit adds {convex['new_fits']} fits. The two synthetic report counts must not be summed."
        ),
        "",
    ]
    lines += ["", "Raw training artifacts remain in persistent storage:", ""]
    lines += [
        f"- **{item['name']}**: `{item['training_root']}`; [retained plan](evidence/protocols/{item['name']}-plan.json)."
        for item in index
        if "training_root" in item
    ]
    lines += [
        "",
        "The manifest retains raw probe/noise paths, report hashes, source commits and the population-certificate flag without changing their scope.",
        "",
    ]
    population = next(item for item in index if item["name"] == "population-v2")
    summary = population["certificate_summary"]
    direct, corruption = summary["groups"]["direct"], summary["groups"]["corruption"]
    lines += [
        (
            f"The retained population candidates have **{summary['certified']}/{summary['total']} certificates** "
            f"(direct: {direct['certified']}/{direct['total']}; corruption: {corruption['certified']}/{corruption['total']}). "
            f"The remaining {summary['uncertified']} candidates are uncertified. "
            f"The largest full-simplex Frank–Wolfe gap is `{summary['max_frank_wolfe_gap']}` "
            f"for `{summary['max_gap_case']}`. The original global flag remains "
            f"`{str(population['all_population_candidates_certified']).lower()}`. "
            "Candidate-solver certification is distinct from the theorem and calibration audit."
        ),
        "",
        "Independent review notes:",
        "",
    ]
    lines += [f"- [{name}](notes/{name})" for name in notes]
    return "\n".join(lines) + "\n"


def frozen_code(*, repository: Path, commits: list[str], output: Path) -> list[dict]:
    inventory = []
    for commit in commits:
        names = subprocess.check_output(
            [
                "git",
                "-C",
                str(repository),
                "ls-tree",
                "-r",
                "--name-only",
                commit,
                "--",
                "src",
                "pyproject.toml",
                "uv.lock",
                "uv.toml",
            ],
            text=True,
        ).splitlines()
        names = [
            name for name in names if Path(name).suffix == ".py" or name in {"pyproject.toml", "uv.lock", "uv.toml"}
        ]
        require(condition=bool(names), path=repository, expectation=f"available frozen Python source at {commit}")
        content = subprocess.check_output([
            "git",
            "-C",
            str(repository),
            "archive",
            "--format=tar",
            commit,
            "--",
            *names,
        ])
        with tarfile.open(fileobj=io.BytesIO(content)) as archive:
            for member in archive:
                if not member.isfile():
                    continue
                destination = output / "source" / commit / member.name
                require(
                    condition=member.name in names
                    and destination.resolve().is_relative_to(output)
                    and member.size <= MAX_FILE_BYTES,
                    path=destination,
                    expectation="a bounded tracked source file",
                )
                stream = archive.extractfile(member)
                if stream is None:
                    raise ValueError(f"Unavailable Git source member: {member.name}")
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(stream.read())
                inventory.append({
                    "source": f"git:{commit}:{member.name}",
                    "path": destination.relative_to(output).as_posix(),
                    "kind": "source",
                    "bytes": destination.stat().st_size,
                    "sha256": file_sha256(path=destination),
                })
    return inventory


def build(*, reports_root: Path, notes_dir: Path, output_dir: Path, source_dir: Path) -> dict:
    reports_root, notes_dir, output_dir, source_dir = (
        path.resolve() for path in (reports_root, notes_dir, output_dir, source_dir)
    )
    if output_dir.exists():
        raise FileExistsError(output_dir)
    reports, index = validate_reports(root=reports_root)
    sources, source = committed_sources(directory=source_dir)
    centered = next(item for item in index if item["name"] == "centered-assignment")
    audit_source = "src/experiments/utkface/mechanism_assignment_audit.py"
    content = subprocess.check_output([
        "git",
        "-C",
        source["repository"],
        "show",
        f"{source['git_commit']}:{audit_source}",
    ])
    require(
        condition=hashlib.sha256(content).hexdigest() == centered["source_sha256"],
        path=Path(source["repository"]) / audit_source,
        expectation="centered-audit source hash retained in the committed source snapshot",
    )
    centered["source_commit"] = source["git_commit"]
    for name in NOTE_NAMES:
        if not (notes_dir / name).is_file():
            raise FileNotFoundError(notes_dir / name)
    notes = sorted(path for path in notes_dir.iterdir() if path.is_file() and path.suffix in {".txt", ".json"})
    require(
        condition=not any(
            output_dir.is_relative_to(path) for path in [notes_dir, source_dir, *(p for p in reports if p.is_dir())]
        ),
        path=output_dir,
        expectation="a separate destination outside copied input directories",
    )
    featured = set()
    for path in sources:
        for target in re.findall(r"!\[[^\]]*\]\(([^\s)]+)", path.read_text()):
            relative = (path.parent / target).relative_to(source_dir)
            if relative.parts[0] == "evidence":
                featured.add(relative.with_suffix(""))
    candidates = [
        (path, Path("evidence") / path.relative_to(reports_root), "report")
        for report in reports
        for path in (sorted(report.rglob("*")) if report.is_dir() else [report])
        if path.is_file() and path.suffix.lower() in REPORT_SUFFIXES
    ]
    copies = [
        (path, destination, kind)
        for path, destination, kind in candidates
        if path.suffix.lower() in {".json", ".tsv"} or destination.with_suffix("") in featured
    ]
    retained = [
        {"source": str(path), "bytes": path.stat().st_size, "sha256": file_sha256(path=path)}
        for path, destination, _ in candidates
        if path.suffix.lower() not in {".json", ".tsv"} and destination.with_suffix("") not in featured
    ]
    copies += [(path, Path("notes") / path.name, "note") for path in notes]
    copies += [
        (reports_root / name / "parameters.npz", Path("evidence") / name / "parameters.npz", "synthetic-parameters")
        for name in ("synthetic", "synthetic-score-ridge")
    ]
    copies += [(path, path.relative_to(source_dir), "quarto") for path in sources]
    copies += [
        (Path(item["raw_plan"]), Path("evidence/protocols") / f"{item['name']}-plan.json", "protocol")
        for item in index
        if "raw_plan" in item
    ]
    for item in index:
        if "raw_plan" in item:
            copies += [
                (
                    Path(job["config"]),
                    Path("evidence/protocols") / item["name"] / "configs" / Path(job["config"]).name,
                    "configuration",
                )
                for job in load(path=Path(item["raw_plan"]))["jobs"]
            ]
            for category in ("probe", "noise"):
                copies += [
                    (
                        Path(row["path"]),
                        Path("evidence/provenance")
                        / item["name"]
                        / category
                        / Path(row["path"]).parent.name
                        / "report.json",
                        "provenance",
                    )
                    for row in item[f"raw_{category}_reports"]
                ]
    inventory = []
    for path, destination, kind in copies:
        require(
            condition=path.stat().st_size <= MAX_FILE_BYTES and not path.is_symlink(),
            path=path,
            expectation="a regular evidence file of at most 256 MiB",
        )
        inventory.append({
            "source": str(path),
            "path": destination.as_posix(),
            "kind": kind,
            "bytes": path.stat().st_size,
            "sha256": file_sha256(path=path),
        })
    require(
        condition=len({item["path"] for item in inventory}) == len(inventory),
        path=output_dir,
        expectation="unique dossier destinations",
    )
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = output_dir.with_name(f".{output_dir.name}.building-{uuid4().hex}")
    staging.mkdir()
    for item in inventory:
        destination = staging / item["path"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(item["source"], destination)
        require(
            condition=file_sha256(path=destination) == item["sha256"],
            path=destination,
            expectation="an unchanged verbatim copy of the validated input",
        )
    inventory += frozen_code(
        repository=Path(source["repository"]),
        commits=sorted({source["git_commit"], *(item["source_commit"] for item in index)}),
        output=staging,
    )
    appendix = staging / "_evidence.qmd"
    appendix.write_text(evidence_index(index=index, source=source, notes=[p.name for p in notes]), encoding="utf-8")
    inventory.append({
        "source": "generated",
        "path": appendix.name,
        "kind": "index",
        "bytes": appendix.stat().st_size,
        "sha256": file_sha256(path=appendix),
    })
    for path in staging.rglob("*.qmd"):
        text = path.read_text()
        targets = re.findall(r"\{\{<\s*include\s+[\"']?([^\s\"'>]+)", text)
        targets += re.findall(r"!\[[^\]]*\]\(([^\s)]+)", text)
        for target in targets:
            if "://" not in target:
                resolved = (path.parent / target).resolve()
                require(
                    condition=resolved.is_relative_to(staging) and resolved.is_file(),
                    path=path,
                    expectation=f"an available local include/figure: {target}",
                )
    for item in inventory:
        if item["source"] != "generated" and not item["source"].startswith("git:"):
            require(
                condition=file_sha256(path=Path(item["source"])) == item["sha256"],
                path=Path(item["source"]),
                expectation="input unchanged throughout dossier assembly",
            )
    manifest = {
        "schema": "utkface-mechanism-dossier-v1",
        "scientific_complete": True,
        "rendered": False,
        "created_utc": datetime.now(UTC).isoformat(),
        "source_snapshot": source,
        "evidence": index,
        "unique_completed_image_fits": sum(item["fits"] for item in index if "training_root" in item),
        "unique_completed_convex_fits": next(item["fits"] for item in index if item["name"] == "synthetic-score-ridge"),
        "files": inventory,
        "figure_scope": "Only figures referenced by committed Quarto, with available SVG/PDF companions, are copied; other figures remain at the hashed server paths.",
        "retained_visualizations": retained,
        "hash_scope": "SHA256SUMS.tsv covers copied files, generated index and manifest; the checksum table excludes itself.",
    }
    write_json(path=staging / "dossier-manifest.json", payload=manifest)
    checksum_rows = inventory + [
        {
            "path": "dossier-manifest.json",
            "bytes": (staging / "dossier-manifest.json").stat().st_size,
            "sha256": file_sha256(path=staging / "dossier-manifest.json"),
        }
    ]
    with (staging / "SHA256SUMS.tsv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=("sha256", "bytes", "path"), delimiter="\t", extrasaction="ignore", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(checksum_rows)
    if output_dir.exists():
        raise FileExistsError(output_dir)
    staging.rename(output_dir)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in ("reports-root", "notes-dir", "output-dir", "source-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    result = build(
        reports_root=args.reports_root, notes_dir=args.notes_dir, output_dir=args.output_dir, source_dir=args.source_dir
    )
    print(
        json.dumps({
            "scientific_complete": result["scientific_complete"],
            "files": len(result["files"]),
            "output_dir": str(args.output_dir),
        })
    )


if __name__ == "__main__":
    main()
