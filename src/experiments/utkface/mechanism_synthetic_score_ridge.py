import argparse
import json
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from src.experiments.utkface.mechanism_population import file_sha256
from src.experiments.utkface.mechanism_synthetic import (
    BASE_SEED,
    DIMENSION,
    LAWS,
    array_hash,
    design_rows,
    fit_model,
    observed_labels,
    payload_hash,
    predictions,
    write_tsv,
)
from src.experiments.utkface.run import write_json

SCORE_RIDGES = (0.001, 1.0, 10.0)
ORACLE_SPLIT = 4


def run_audit(*, source_dir: Path, output_dir: Path) -> None:
    started = time.monotonic()
    source_path = source_dir / "report.json"
    source = json.loads(source_path.read_text())
    config = source["config"]
    root = Path(__file__).resolve().parents[3]
    if source["status"] != "complete" or not source["all_gradients_converged"]:
        raise ValueError("A completed converged synthetic study is required")
    if (config["base_seed"] != BASE_SEED or config["dimension"] != DIMENSION
            or config["laws"] != list(LAWS) or config["score_ridge"] != SCORE_RIDGES[0]
            or len(config["residual_penalties"]) != len(SCORE_RIDGES)
            or config["oracle_split"] == ORACLE_SPLIT):
        raise ValueError("Source protocol does not match this three-candidate audit")
    for name, expected in source["source_sha256"].items():
        if file_sha256(path=root / name) != expected:
            raise ValueError(f"Source implementation changed: {name}")
    parameter_path = source_dir / "parameters.npz"
    freeze_path = source_dir / "frozen-selection.json"
    freeze = json.loads(freeze_path.read_text())
    if (file_sha256(path=freeze_path) != source["frozen_selection_sha256"]
            or freeze["parameters_sha256"] != file_sha256(path=parameter_path)
            or freeze["selections"] != source["selections"]
            or freeze["config_sha256"] != source["config_sha256"]):
        raise ValueError("Source archive or frozen selections changed")
    with np.load(parameter_path) as saved:
        parameters = {name: saved[name] for name in saved.files}
    if any(array_hash(values=parameters[row["fit_id"]]) != row["parameters_sha256"]
           for row in source["fits"]):
        raise ValueError("Source parameter hashes differ")

    output_dir.mkdir(parents=True, exist_ok=False)
    protocol = {
        "hypothesis": "The small endpoint full-head gain can be matched by scalar score shrinkage.",
        "source_report": str(source_path.resolve()),
        "source_report_sha256": file_sha256(path=source_path),
        "score_ridges": SCORE_RIDGES,
        "residual_penalties": config["residual_penalties"],
        "oracle_split": ORACLE_SPLIT,
        "oracle_test_size": config["oracle_test_size"],
        "selection": "Observed-label validation MSE, independently within each family; three candidates per family.",
        "scope": "A scalar regularization competitor on the same fixed-design synthetic problem. Different penalty families and ranges remain different estimators. This is not an image or backbone experiment.",
        "interpretation": "Matching the full head shows that its benefit does not require extra mean flexibility in these candidate families. A remaining full-head gain does not exclude other scalar regularization choices.",
    }
    report = {
        "status": "fitting", "protocol": protocol,
        "protocol_sha256": payload_hash(value=protocol),
        "audit_source_sha256": file_sha256(path=Path(__file__)),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        "environment": {"python": platform.python_version(), "torch": str(torch.__version__),
                        "numpy": np.__version__, "device": "cpu", "precision": "float64", "threads": 1},
        "fits": [], "selections": [], "oracle_test_data": [],
    }
    write_json(path=output_dir / "protocol.json", payload=protocol)
    for seed in config["seeds"]:
        train_x, train_mean = design_rows(count=max(config["training_sizes"]), seed=seed, split=0)
        val_x, val_mean = design_rows(count=config["validation_size"], seed=seed, split=1)
        for law in LAWS:
            train_y = observed_labels(mean=train_mean, seed=seed, split=0, law=law)
            val_y = observed_labels(mean=val_mean, seed=seed, split=1, law=law)
            for size in config["training_sizes"]:
                members = [dict(row) for row in source["fits"]
                           if (row["seed"], row["law"], row["training_size"]) == (seed, law, size)]
                if len(members) != len(SCORE_RIDGES) + 1:
                    raise ValueError("Expected one tied and three full source fits per block")
                identity = {
                    "train_x_sha256": array_hash(values=train_x[:size]),
                    "train_y_sha256": array_hash(values=train_y[:size]),
                    "validation_x_sha256": array_hash(values=val_x),
                    "validation_y_sha256": array_hash(values=val_y),
                }
                for row in members:
                    if any(row[key] != value for key, value in identity.items()):
                        raise ValueError("Source design or labels cannot be reproduced")
                    row["family"] = "tied" if row["tied"] else "full"
                    row["score_ridge"] = config["score_ridge"]
                    row["reused_source_fit"] = True
                    row.pop("oracle_mean_mse")
                    row.pop("oracle_mean_mae")
                    row.pop("validation_selected")
                    pred = predictions(design=val_x, parameters=parameters[row["fit_id"]], tied=row["tied"])
                    if abs(float(np.mean((pred - val_y)**2)) - row["validation_observed_mse"]) > 1e-9:
                        raise ValueError("Source validation prediction cannot be reproduced")
                fixed = next(row for row in members if row["tied"])
                for ridge in SCORE_RIDGES[1:]:
                    fit_id = f"{law}_n{size}_seed{seed}_score_ridge{ridge}"
                    theta, diagnostics = fit_model(design=train_x[:size], targets=train_y[:size],
                        tied=True, penalty=0.0, score_ridge=ridge, max_iterations=config["max_iterations"])
                    if not diagnostics["gradient_tolerance_met"]:
                        raise RuntimeError(f"Score-ridge fit did not converge: {fit_id}")
                    pred = predictions(design=val_x, parameters=theta, tied=True)
                    parameters[fit_id] = theta
                    members.append({
                        "fit_id": fit_id, "law": law, "training_size": size, "seed": seed,
                        "tied": True, "family": "tied", "score_ridge": ridge,
                        "residual_penalty": "tied", "reused_source_fit": False,
                        "validation_observed_mse": float(np.mean((pred - val_y)**2)),
                        "parameters_sha256": array_hash(values=theta), **identity, **diagnostics,
                    })
                selection = {"law": law, "training_size": size, "seed": seed,
                             "fixed_tied_fit": fixed["fit_id"]}
                for family in ("tied", "full"):
                    chosen = min((row for row in members if row["family"] == family),
                                 key=lambda row: (row["validation_observed_mse"], row["fit_id"]))
                    selection[f"selected_{family}_fit"] = chosen["fit_id"]
                source_selection = next(row for row in source["selections"]
                    if (row["seed"], row["law"], row["training_size"]) == (seed, law, size))
                if selection["selected_full_fit"] != source_selection["selected_full_fit"]:
                    raise ValueError("Retained full-head validation selection changed")
                report["fits"].extend(members)
                report["selections"].append(selection)
                write_json(path=output_dir / "report.json", payload=report)

    np.savez_compressed(output_dir / "parameters.npz", **{
        row["fit_id"]: parameters[row["fit_id"]] for row in report["fits"]})
    frozen = {"protocol_sha256": report["protocol_sha256"], "selections": report["selections"],
              "parameters_sha256": file_sha256(path=output_dir / "parameters.npz"),
              "oracle_test_generated": False}
    write_json(path=output_dir / "frozen-selection.json", payload=frozen)
    report["frozen_selection_sha256"] = file_sha256(path=output_dir / "frozen-selection.json")
    report["status"] = "oracle_evaluation"
    write_json(path=output_dir / "report.json", payload=report)
    for seed in config["seeds"]:
        test_x, test_mean = design_rows(count=config["oracle_test_size"], seed=seed, split=ORACLE_SPLIT)
        report["oracle_test_data"].append({"seed": seed,
            "x_sha256": array_hash(values=test_x), "mean_sha256": array_hash(values=test_mean)})
        for row in (row for row in report["fits"] if row["seed"] == seed):
            pred = predictions(design=test_x, parameters=parameters[row["fit_id"]], tied=row["tied"])
            row["oracle_mean_mse"] = float(np.mean((pred - test_mean)**2))
            row["oracle_mean_mae"] = float(np.mean(np.abs(pred - test_mean)))
    by_id = {row["fit_id"]: row for row in report["fits"]}
    pairs = []
    for selection in frozen["selections"]:
        fixed = by_id[selection["fixed_tied_fit"]]
        tied = by_id[selection["selected_tied_fit"]]
        full = by_id[selection["selected_full_fit"]]
        pairs.append({**selection, "selected_score_ridge": tied["score_ridge"],
            "selected_residual_penalty": full["residual_penalty"],
            "fixed_tied_mse": fixed["oracle_mean_mse"], "selected_tied_mse": tied["oracle_mean_mse"],
            "selected_full_mse": full["oracle_mean_mse"],
            "full_minus_fixed_tied_mse": full["oracle_mean_mse"] - fixed["oracle_mean_mse"],
            "full_minus_selected_tied_mse": full["oracle_mean_mse"] - tied["oracle_mean_mse"],
            "selected_tied_minus_fixed_mse": tied["oracle_mean_mse"] - fixed["oracle_mean_mse"]})
    summaries = []
    for law in LAWS:
        for size in config["training_sizes"]:
            members = [row for row in pairs if row["law"] == law and row["training_size"] == size]
            summary = {"law": law, "training_size": size, "count": len(members)}
            for key in ("full_minus_fixed_tied_mse", "full_minus_selected_tied_mse",
                        "selected_tied_minus_fixed_mse"):
                values = np.asarray([row[key] for row in members])
                summary[f"{key}_mean"] = float(values.mean())
                summary[f"{key}_sd"] = float(values.std(ddof=1)) if len(values) > 1 else None
            summaries.append(summary)
    report["paired_differences"], report["paired_summaries"] = pairs, summaries
    report["wall_seconds"], report["status"] = time.monotonic() - started, "complete"
    write_json(path=output_dir / "report.json", payload=report)
    write_tsv(path=output_dir / "paired-differences.tsv", rows=pairs)
    print(json.dumps({"output": str(output_dir), "wall_seconds": report["wall_seconds"],
                      "paired_summaries": summaries}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    with threadpool_limits(limits=1):
        run_audit(source_dir=args.source_dir, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
