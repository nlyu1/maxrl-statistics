import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.linalg import qr
from scipy.special import expit, gammaln, logsumexp
from threadpoolctl import threadpool_limits


def verify(*, source_dir: Path, audit_dir: Path) -> dict:
    report = json.loads((audit_dir / "report.json").read_text())
    source = json.loads((source_dir / "report.json").read_text())
    frozen = json.loads((audit_dir / "frozen-selection.json").read_text())
    config = source["config"]
    with np.load(audit_dir / "parameters.npz") as archive:
        parameters = {name: archive[name] for name in archive.files}

    def sha(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def array_hash(values: np.ndarray) -> str:
        prefix = f"{values.dtype}:{values.shape}:".encode()
        return hashlib.sha256(
            prefix + np.ascontiguousarray(values).tobytes()
        ).hexdigest()

    def design(seed: int, split: int, count: int) -> tuple[np.ndarray, np.ndarray]:
        generator = np.random.default_rng(
            np.random.SeedSequence([config["base_seed"], seed, split, 0])
        )
        x = generator.normal(size=(count, config["dimension"]))
        return np.column_stack((np.ones(count), x)), 99 * expit(
            0.7 * x[:, 0] + 0.4 * x[:, 1] - 0.3 * x[:, 2]
        )

    def labels(mean: np.ndarray, seed: int, split: int, law: str) -> np.ndarray:
        generator = np.random.default_rng(
            np.random.SeedSequence([
                config["base_seed"],
                seed,
                split,
                config["laws"].index(law) + 1,
            ])
        )
        return (
            generator.binomial(99, mean / 99)
            if law == "binomial"
            else 99 * (generator.random(len(mean)) < mean / 99).astype(np.int64)
        )

    ages = np.arange(100, dtype=np.float64)
    coefficients = gammaln(100) - gammaln(ages + 1) - gammaln(100 - ages)
    errors = {
        "objective": 0.0,
        "gradient": 0.0,
        "validation_mse": 0.0,
        "oracle_mse": 0.0,
        "mean_readout": 0.0,
        "reported_gradient": 0.0,
    }
    failures, candidates = [], []
    with threadpool_limits(limits=1):
        basis = qr(np.column_stack((np.ones(100), ages - ages.mean())), mode="full")[0][
            :, 2:
        ].copy()
        if array_hash(basis) != config["residual_basis_sha256"]:
            failures.append("basis")
        for row in report["fits"]:
            seed, law, size = row["seed"], row["law"], row["training_size"]
            theta, width = parameters[row["fit_id"]], 13
            x, mean = design(seed, 0, max(config["training_sizes"]))
            y = labels(mean, seed, 0, law)
            x, y = x[:size], y[:size]
            val_x, val_mean = design(seed, 1, config["validation_size"])
            val_y = labels(val_mean, seed, 1, law)
            oracle_x, oracle_mean = design(seed, 4, config["oracle_test_size"])
            for name, values in (
                ("train_x", x),
                ("train_y", y),
                ("validation_x", val_x),
                ("validation_y", val_y),
                ("parameters", theta),
            ):
                if array_hash(values) != row[name + "_sha256"]:
                    failures.append(row["fit_id"] + ":" + name)
            oracle_hash = next(
                entry["x_sha256"]
                for entry in report["oracle_test_data"]
                if entry["seed"] == seed
            )
            if array_hash(oracle_x) != oracle_hash:
                failures.append(row["fit_id"] + ":oracle_x")

            def probability(
                design_values: np.ndarray,
                *,
                theta: np.ndarray = theta,
                width: int = width,
                tied: bool = row["tied"],
            ) -> tuple[np.ndarray, np.ndarray]:
                logits = coefficients + (design_values @ theta[:width])[:, None] * ages
                if not tied:
                    logits += (
                        design_values @ (basis @ theta[width:].reshape(98, width)).T
                    )
                return np.exp(logits - logsumexp(logits, axis=1, keepdims=True)), logits

            probabilities, logits = probability(x)
            ridge = row["score_ridge"]
            objective = float(
                np.mean(logsumexp(logits, axis=1) - logits[np.arange(size), y])
            ) + 0.5 * ridge * float(theta[:width] @ theta[:width])
            gradient = x.T @ (probabilities @ ages - y) / size + ridge * theta[:width]
            if not row["tied"]:
                penalty = float(row["residual_penalty"])
                objective += 0.5 * penalty * float(theta[width:] @ theta[width:])
                delta = probabilities.copy()
                delta[np.arange(size), y] -= 1
                gradient = np.r_[
                    gradient,
                    (
                        basis.T @ (delta.T @ x) / size
                        + penalty * theta[width:].reshape(98, width)
                    ).ravel(),
                ]
            errors["objective"] = max(
                errors["objective"], abs(objective - row["objective"])
            )
            errors["gradient"] = max(errors["gradient"], float(abs(gradient).max()))
            errors["reported_gradient"] = max(
                errors["reported_gradient"],
                abs(float(abs(gradient).max()) - row["gradient_infinity_norm"]),
            )
            val_probabilities, _ = probability(val_x)
            oracle_probabilities, _ = probability(oracle_x)
            val_prediction = val_probabilities @ ages
            oracle_prediction = oracle_probabilities @ ages
            errors["validation_mse"] = max(
                errors["validation_mse"],
                abs(
                    float(np.mean((val_prediction - val_y) ** 2))
                    - row["validation_observed_mse"]
                ),
            )
            errors["oracle_mse"] = max(
                errors["oracle_mse"],
                abs(
                    float(np.mean((oracle_prediction - oracle_mean) ** 2))
                    - row["oracle_mean_mse"]
                ),
            )
            if row["tied"]:
                errors["mean_readout"] = max(
                    errors["mean_readout"],
                    float(abs(oracle_prediction - 99 * expit(oracle_x @ theta)).max()),
                )
            if law == "endpoint" and size == 128 and row["tied"]:
                candidates.append({
                    "seed": seed,
                    "score_ridge": ridge,
                    "validation_mse": row["validation_observed_mse"],
                    "oracle_mse": row["oracle_mean_mse"],
                    "beta_norm": float(np.linalg.norm(theta)),
                    "nuisance_beta_norm": float(np.linalg.norm(theta[4:])),
                })
        for selection in report["selections"]:
            members = [
                row
                for row in report["fits"]
                if (row["law"], row["training_size"], row["seed"])
                == (selection["law"], selection["training_size"], selection["seed"])
            ]
            for family in ("tied", "full"):
                options = [row for row in members if row["family"] == family]
                if len(options) != 3:
                    failures.append("candidate_count")
                chosen = min(
                    options,
                    key=lambda row: (row["validation_observed_mse"], row["fit_id"]),
                )
                if chosen["fit_id"] != selection["selected_" + family + "_fit"]:
                    failures.append("selection")
    checks = {
        "source_report_sha": sha(source_dir / "report.json")
        == report["protocol"]["source_report_sha256"],
        "frozen_sha": sha(audit_dir / "frozen-selection.json")
        == report["frozen_selection_sha256"],
        "parameters_sha": sha(audit_dir / "parameters.npz")
        == frozen["parameters_sha256"],
        "frozen_selections": frozen["selections"] == report["selections"],
        "protocol_sha": hashlib.sha256(
            json.dumps(report["protocol"], sort_keys=True).encode()
        ).hexdigest()
        == report["protocol_sha256"],
        "audit_source_sha": sha(
            Path("src/experiments/utkface/mechanism_synthetic_score_ridge.py")
        )
        == report["audit_source_sha256"],
        "oracle_split_distinct": report["protocol"]["oracle_split"]
        not in (0, 1, config["oracle_split"]),
    }
    summaries = []
    for law in config["laws"]:
        for size in config["training_sizes"]:
            pairs = [
                row
                for row in report["paired_differences"]
                if row["law"] == law and row["training_size"] == size
            ]
            means = {
                key: float(np.mean([row[key] for row in pairs]))
                for key in ("fixed_tied_mse", "selected_tied_mse", "selected_full_mse")
            }
            summaries.append({
                "law": law,
                "n": size,
                "mean_mse": means,
                "pooled_rmse": {
                    key: float(np.sqrt(value)) for key, value in means.items()
                },
                "tied_better_than_full_seeds": sum(
                    row["full_minus_selected_tied_mse"] > 0 for row in pairs
                ),
                "selected_score_ridges": dict(
                    Counter(str(row["selected_score_ridge"]) for row in pairs)
                ),
                "scalar_relative_risk_reduction_vs_fixed": 1
                - means["selected_tied_mse"] / means["fixed_tied_mse"],
                "scalar_relative_risk_reduction_vs_full": 1
                - means["selected_tied_mse"] / means["selected_full_mse"],
            })

    new = []
    with threadpool_limits(limits=1):
        for row in report["fits"]:
            if row["reused_source_fit"]:
                continue
            generator = np.random.default_rng(
                np.random.SeedSequence([config["base_seed"], row["seed"], 0, 0])
            )
            z = generator.normal(size=(max(config["training_sizes"]), 12))
            x = np.column_stack((np.ones(len(z)), z))
            mean = 99 * expit(0.7 * z[:, 0] + 0.4 * z[:, 1] - 0.3 * z[:, 2])
            generator = np.random.default_rng(
                np.random.SeedSequence([
                    config["base_seed"],
                    row["seed"],
                    0,
                    config["laws"].index(row["law"]) + 1,
                ])
            )
            y = (
                generator.binomial(99, mean / 99)
                if row["law"] == "binomial"
                else 99 * (generator.random(len(mean)) < mean / 99).astype(np.int64)
            )
            x, y = x[: row["training_size"]], y[: row["training_size"]]
            theta, ridge = parameters[row["fit_id"]], row["score_ridge"]
            gradient = x.T @ (99 * expit(x @ theta) - y) / len(y) + ridge * theta
            new.append({
                "fit_id": row["fit_id"],
                "reported_gap": row["objective_gap_upper_bound"],
                "independent_gap": float(gradient @ gradient) / (2 * ridge),
                "independent_grad_inf": float(abs(gradient).max()),
                "ridge_field_matches": row["strong_convexity_lower_bound"] == ridge,
            })
    noises = {law: [] for law in config["laws"]}
    for seed in config["seeds"]:
        generator = np.random.default_rng(
            np.random.SeedSequence([config["base_seed"], seed, 4, 0])
        )
        x = generator.normal(size=(20000, 12))
        q = expit(0.7 * x[:, 0] + 0.4 * x[:, 1] - 0.3 * x[:, 2])
        noises["binomial"].append(float(np.mean(99 * q * (1 - q))))
        noises["endpoint"].append(float(np.mean(99**2 * q * (1 - q))))
    noise = {key: float(np.mean(values)) for key, values in noises.items()}
    pairs = [
        row
        for row in report["paired_differences"]
        if row["law"] == "endpoint" and row["training_size"] == 128
    ]
    means = {
        key: float(np.mean([row[key] for row in pairs]))
        for key in ("fixed_tied_mse", "selected_tied_mse", "selected_full_mse")
    }
    return {
        "independent_implementation": "NumPy/SciPy float64, no imports from experiment code",
        "checks": checks,
        "failures": failures,
        "maximum_absolute_errors": errors,
        "endpoint_n128_tied_candidates": candidates,
        "summary": summaries,
        "new_fit_count": len(new),
        "all_strong_convexity_fields_correct": all(
            row["ridge_field_matches"] for row in new
        ),
        "max_reported_gap": max(row["reported_gap"] for row in new),
        "max_independent_gap": max(row["independent_gap"] for row in new),
        "max_independent_grad_inf": max(row["independent_grad_inf"] for row in new),
        "max_absolute_gap_disagreement": max(
            abs(row["reported_gap"] - row["independent_gap"]) for row in new
        ),
        "irreducible_observed_mse": noise,
        "endpoint_n128_expected_observed_mse": {
            key: value + noise["endpoint"] for key, value in means.items()
        },
        "endpoint_n128_total_risk_reduction_vs_fixed": (
            means["fixed_tied_mse"] - means["selected_tied_mse"]
        )
        / (noise["endpoint"] + means["fixed_tied_mse"]),
        "endpoint_n128_total_risk_reduction_vs_full": (
            means["selected_full_mse"] - means["selected_tied_mse"]
        )
        / (noise["endpoint"] + means["selected_full_mse"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replay the previously executed independent synthetic verification."
    )
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--audit-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = verify(source_dir=args.source_dir, audit_dir=args.audit_dir)
    result["absolute_tolerances"] = {
        "objective": 1e-9,
        "gradient": 1e-8,
        "validation_mse": 1e-8,
        "oracle_mse": 1e-8,
        "mean_readout": 1e-8,
        "reported_gradient": 1e-10,
    }
    passed = (
        not result["failures"]
        and all(result["checks"].values())
        and result["all_strong_convexity_fields_correct"]
        and all(
            result["maximum_absolute_errors"][key] <= tolerance
            for key, tolerance in result["absolute_tolerances"].items()
        )
    )
    result["status"] = "passed" if passed else "failed"
    payload = json.dumps(result, indent=2, allow_nan=False)
    if args.output is not None:
        args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload, flush=True)
    if not passed:
        raise RuntimeError("Independent synthetic verification failed")


if __name__ == "__main__":
    main()
