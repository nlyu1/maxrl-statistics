import argparse
import csv
import hashlib
import json
import math
import platform
from fractions import Fraction
from pathlib import Path

import numpy as np
import scipy
from numpy.typing import NDArray
from scipy.special import logsumexp

MAX_AGE = 99
CENTER = MAX_AGE / 2
AGES = np.arange(MAX_AGE + 1, dtype=np.int64)
COEFFICIENTS = tuple(math.comb(MAX_AGE, int(age)) for age in AGES)
BASE_PROBABILITIES = np.asarray(COEFFICIENTS, dtype=np.float64) / float(2**MAX_AGE)
STEPS = (1e-3, 1e-4, 1e-5, 1e-6, 1e-7)
DERIVATIVE_RTOL = 5e-5
DERIVATIVE_ATOL = 1e-6
PROBABILITY_ATOL = 1e-12


def alignment_counts(*, values: NDArray[np.float64]) -> dict[str, int]:
    return {
        "improving": int(np.count_nonzero(values < 0)),
        "worsening": int(np.count_nonzero(values > 0)),
        "stationary": int(np.count_nonzero(values == 0)),
    }


def exact_kappa(*, mapping: NDArray[np.int64]) -> Fraction:
    numerator = sum(
        coefficient * (2 * age - MAX_AGE) * (2 * int(code) - MAX_AGE)
        for age, (coefficient, code) in enumerate(
            zip(COEFFICIENTS, mapping, strict=True)
        )
    )
    return Fraction(numerator, 2 ** (MAX_AGE + 2))


def retained_maps(*, verification: dict) -> list[dict]:
    blocks = verification["symmetric_swap_initialization"]["blocks"]
    if len(blocks) != 3 or len({block["head_seed"] for block in blocks}) != 3:
        raise ValueError("Require three independently identified retained maps")
    cases = []
    for block in blocks:
        assignment = block["symmetric_age_assignment"]
        values = assignment["mapping"]
        if len(values) != len(AGES) or any(type(value) is not int for value in values):
            raise ValueError("Age codes must be 100 integers")
        mapping = np.asarray(values, dtype=np.int64)
        checks = {
            "permutation": bool(np.array_equal(np.sort(mapping), AGES)),
            "symmetric_pairs": bool(
                np.all((mapping == AGES) | (mapping == MAX_AGE - AGES))
            ),
            "twenty_five_swapped_pairs": bool(
                np.count_nonzero(mapping[:50] != AGES[:50]) == 25
            ),
        }
        if not all(checks.values()):
            raise ValueError(f"Invalid symmetric mapping: {checks}")
        checks["involution"] = bool(np.array_equal(mapping[mapping], AGES))
        checks["exact_binomial_coefficient_symmetry"] = all(
            COEFFICIENTS[age] == COEFFICIENTS[int(code)]
            for age, code in enumerate(mapping)
        )
        checks["exact_centered_probability_assignment"] = bool(
            np.array_equal(BASE_PROBABILITIES[mapping], BASE_PROBABILITIES)
        )
        if not all(checks.values()):
            raise ValueError(f"Mapping does not preserve the centered policy: {checks}")
        cases.append({
            "name": f"swapped_head_{block['head_seed']}",
            "mapping": mapping,
            "metadata": {
                "head_seed": block["head_seed"],
                "assignment": assignment,
                "mapping_checks": checks,
            },
        })
    return cases


def audit_case(
    *, name: str, mapping: NDArray[np.int64], metadata: dict
) -> tuple[dict, list[dict]]:
    kappa_exact = exact_kappa(mapping=mapping)
    kappa = float(kappa_exact)
    ordered = bool(np.array_equal(mapping, AGES))
    centered_codes = mapping.astype(np.float64) - CENTER
    residuals = CENTER - AGES
    ce_gradient = -centered_codes
    analytic_mean = -kappa * ce_gradient
    analytic_point = residuals * analytic_mean
    rows, summaries = [], []
    for step in STEPS:
        scores = -step * ce_gradient
        tilt = scores[:, None] * centered_codes[None, :]
        normalizers = logsumexp(np.log(BASE_PROBABILITIES)[None, :] + tilt, axis=1)
        log_ratios = tilt - normalizers[:, None]
        probabilities = BASE_PROBABILITIES[None, :] * np.exp(log_ratios)
        mean_change = (BASE_PROBABILITIES[None, :] * np.expm1(log_ratios)) @ (
            AGES - CENTER
        )
        ce_change = normalizers - scores * centered_codes
        point_change = mean_change * (residuals + 0.5 * mean_change)
        finite_difference = point_change / step
        error = np.abs(finite_difference - analytic_point)
        mean_error = np.abs(mean_change / step - analytic_mean)
        if not all(
            np.isfinite(values).all()
            for values in (probabilities, ce_change, finite_difference, mean_error)
        ):
            raise FloatingPointError(f"Nonfinite centered audit values: {name}, {step}")
        summaries.append({
            "step": step,
            "ce_decreasing_targets": int(np.count_nonzero(ce_change < 0)),
            "ce_change_min": float(ce_change.min()),
            "ce_change_max": float(ce_change.max()),
            "point_alignment": alignment_counts(values=point_change),
            "max_probability_mass_error": float(
                np.abs(probabilities.sum(axis=1) - 1).max()
            ),
            "max_abs_mean_derivative_error": float(mean_error.max()),
            "max_abs_point_derivative_error": float(error.max()),
            "max_scaled_point_derivative_error": float(
                (
                    error / (DERIVATIVE_ATOL + DERIVATIVE_RTOL * np.abs(analytic_point))
                ).max()
            ),
        })
        rows.extend(
            {
                "case": name,
                "step": step,
                "target": int(target),
                "code": int(mapping[target]),
                "score_after_step": float(scores[target]),
                "mean_after_step": CENTER + float(mean_change[target]),
                "ce_change": float(ce_change[target]),
                "half_squared_point_error_change": float(point_change[target]),
                "analytic_mean_derivative": float(analytic_mean[target]),
                "finite_mean_derivative": float(mean_change[target] / step),
                "abs_mean_derivative_error": float(mean_error[target]),
                "analytic_point_derivative": float(analytic_point[target]),
                "finite_point_derivative": float(finite_difference[target]),
                "abs_point_derivative_error": float(error[target]),
                "point_alignment": (
                    "improving"
                    if point_change[target] < 0
                    else "worsening"
                    if point_change[target] > 0
                    else "stationary"
                ),
            }
            for target in range(len(AGES))
        )
    expected = {
        "improving": 100 if ordered else 50,
        "worsening": 0 if ordered else 50,
        "stationary": 0,
    }
    analytic_counts = alignment_counts(values=analytic_point)
    checks = {
        "nonzero_kappa": kappa_exact != 0,
        "ordered_variance_identity": not ordered or kappa_exact == Fraction(MAX_AGE, 4),
        "analytic_expected_counts": analytic_counts == expected,
        "all_steps_lower_ce_for_all_targets": all(
            row["ce_decreasing_targets"] == len(AGES) for row in summaries
        ),
        "all_steps_preserve_probability_mass": all(
            row["max_probability_mass_error"] <= PROBABILITY_ATOL for row in summaries
        ),
        "smallest_step_alignment_matches_analytic": (
            summaries[-1]["point_alignment"] == analytic_counts
        ),
        "smallest_step_derivative_matches": bool(
            np.allclose(
                finite_difference,
                analytic_point,
                rtol=DERIVATIVE_RTOL,
                atol=DERIVATIVE_ATOL,
            )
        ),
        "finite_difference_error_decreases": (
            summaries[-1]["max_abs_point_derivative_error"]
            < summaries[0]["max_abs_point_derivative_error"]
        ),
    }
    return {
        "case": name,
        **metadata,
        "kappa_exact": str(kappa_exact),
        "kappa_float64": kappa,
        "analytic_point_alignment": analytic_counts,
        "expected_point_alignment_if_nonzero_kappa": expected,
        "checks": checks,
        "passed": all(checks.values()),
        "steps": summaries,
    }, rows


def audit(*, verification: Path, output_dir: Path) -> dict:
    raw = verification.read_bytes()
    cases = retained_maps(verification=json.loads(raw))
    input_hash = hashlib.sha256(raw).hexdigest()
    source_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    results, rows = [], []
    for case in [{"name": "ordered", "mapping": AGES, "metadata": {}}, *cases]:
        result, values = audit_case(**case)
        results.append(result)
        rows.extend(
            {"input_sha256": input_hash, "source_sha256": source_hash, **row}
            for row in values
        )
    report = {
        "schema": "utkface-centered-assignment-audit-v1",
        "scope": "Held head, centered score, independent unit feature w=1 per target; not neural-training mediation or an observed SGD-frequency estimate.",
        "input": str(verification.resolve()),
        "input_sha256": input_hash,
        "source": str(Path(__file__).resolve()),
        "source_sha256": source_hash,
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
        "design": {
            "binomial_trials": MAX_AGE,
            "initial_score": 0,
            "initial_mean": CENTER,
            "feature_weight_squared_norm": 1,
            "steps": STEPS,
            "point_loss": "0.5 * (actual_age_mean - target)**2",
            "score_step": "-eta * (49.5 - code[target])",
            "analytic_mean_derivative": "-kappa * (49.5-code[target])",
            "analytic_point_derivative": "-kappa * (49.5-target) * (49.5-code[target])",
            "ce_score_lipschitz_bound": MAX_AGE / 4,
            "ce_descent_step_bound": 8 / MAX_AGE,
        },
        "tolerances": {
            "smallest_step_point_derivative_rtol": DERIVATIVE_RTOL,
            "smallest_step_point_derivative_atol": DERIVATIVE_ATOL,
            "probability_mass_atol": PROBABILITY_ATOL,
        },
        "cases": results,
        "target_step_rows": len(rows),
        "all_checks_passed": all(result["passed"] for result in results),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    with (output_dir / "target-steps.tsv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(
            stream, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verification", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = audit(verification=args.verification, output_dir=args.output_dir)
    print(
        json.dumps({
            "all_checks_passed": report["all_checks_passed"],
            "target_step_rows": report["target_step_rows"],
            "output_dir": str(args.output_dir.resolve()),
        })
    )
    if not report["all_checks_passed"]:
        raise SystemExit("Centered assignment checks failed; inspect the saved audit")


if __name__ == "__main__":
    main()
