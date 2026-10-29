import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.special import rel_entr

from src.experiments.utkface.losses import MAX_AGE, NUM_BINS
from src.experiments.utkface.mechanism_population import (
    AGES,
    PopulationProblem,
    file_sha256,
    population_problem,
    provenance,
    repository_expected_loss,
)
from src.experiments.utkface.run import write_json

SEED = 2026102917
TOLERANCE = 1e-9


def cdf_values(*, probabilities: np.ndarray) -> np.ndarray:
    values = np.clip(np.cumsum(probabilities), 0, 1)
    values[-1] = 1
    return values


def orbit(*, probabilities: np.ndarray, shift: float) -> np.ndarray:
    cdf = cdf_values(probabilities=probabilities)
    upper = 1 + cdf * np.expm1(shift)
    lower = 1 + np.r_[0.0, cdf[:-1]] * np.expm1(shift)
    return probabilities * np.exp(shift) / (upper * lower)


def calibration_row(*, name: str, p: np.ndarray, q: np.ndarray) -> dict:
    problem = population_problem(q=q)
    value, gradient = problem.value_gradient(probabilities=p)
    cdf = cdf_values(probabilities=p)
    direction = p * (1 - cdf - np.r_[0.0, cdf[:-1]])
    target_mean, mean = float(q @ AGES), float(p @ AGES)
    identity = (target_mean - mean) / MAX_AGE
    dot = float(gradient @ direction)
    gap = max(0.0, float(p @ gradient - gradient.min()))
    r, s = target_mean / MAX_AGE, mean / MAX_AGE
    kl = float(rel_entr([r, 1 - r], [s, 1 - s]).sum())
    shifts = [-4.0, -1.0, 1.0, 4.0]
    if 0 < r < 1 and 0 < s < 1:
        shifts.append(float(np.log(s) + np.log1p(-r) - np.log(r) - np.log1p(-s)))
    orbit_rows = []
    for shift in shifts:
        candidate = orbit(probabilities=p, shift=shift)
        actual = problem.value_gradient(probabilities=candidate)[0] - value
        formula = float(((target_mean - MAX_AGE) * shift
                         + np.log1p(cdf[:-1] * np.expm1(shift)).sum()) / MAX_AGE)
        bernoulli = float((r - 1) * shift + np.log(s + (1 - s) * np.exp(shift)))
        orbit_rows.append({
            "shift": shift, "risk_change": actual, "formula": formula,
            "formula_error": abs(actual - formula), "jensen_slack": bernoulli - actual,
            "bernoulli_expression": bernoulli, "kl": kl,
            "minimum_probability": float(candidate.min()),
            "simplex_error": abs(float(candidate.sum()) - 1),
        })
    row = {
        "name": name, "target_mean": target_mean, "policy_mean": mean,
        "target_simplex_error": abs(float(q.sum()) - 1),
        "policy_simplex_error": abs(float(p.sum()) - 1),
        "cdf_rounding_correction": float(np.abs(cdf - np.cumsum(p)).max()),
        "tangent_mass_error": abs(float(direction.sum())),
        "tangent_cdf_error": float(np.abs(np.cumsum(direction) - cdf * (1 - cdf)).max()),
        "two_sided_feasibility_minimum": float(min((p + direction).min(), (p - direction).min())),
        "gradient_dot_tangent": dot, "expected_identity": identity,
        "identity_error": abs(dot - identity),
        "gradient_dot_probability_error": abs(float(p @ gradient) + 1),
        "full_simplex_frank_wolfe_gap": gap, "mean_bound_years": MAX_AGE * gap,
        "absolute_mean_error_years": abs(mean - target_mean), "orbits": orbit_rows,
    }
    errors = [row[key] for key in (
        "target_simplex_error", "policy_simplex_error", "tangent_mass_error",
        "tangent_cdf_error", "identity_error", "gradient_dot_probability_error",
    )]
    row["passed"] = bool(max(errors) <= TOLERANCE
                         and row["two_sided_feasibility_minimum"] >= -TOLERANCE
                         and abs(mean - target_mean) <= MAX_AGE * gap + TOLERANCE
                         and all(item["formula_error"] <= TOLERANCE
                                 and item["jensen_slack"] >= -TOLERANCE for item in orbit_rows))
    if len(shifts) == 5:
        row["kl_minimum_expression_error"] = abs(orbit_rows[-1]["bernoulli_expression"] + kl)
        row["orbit_improvement_beyond_kl"] = -orbit_rows[-1]["risk_change"] - kl
        row["passed"] &= (row["kl_minimum_expression_error"] <= TOLERANCE
                          and row["orbit_improvement_beyond_kl"] >= -TOLERANCE)
    return row


def inverse_kkt(*, name: str, p: np.ndarray) -> dict:
    problem = population_problem(q=np.full(NUM_BINS, 1 / NUM_BINS))
    masses = problem.windows @ p
    matrix = (problem.windows / masses[:, None]).reshape(NUM_BINS, MAX_AGE, NUM_BINS).mean(1)
    raw, _, rank, singular = np.linalg.lstsq(matrix.T, np.ones(NUM_BINS), rcond=1e-13)
    residual = float(np.abs(matrix.T @ raw - 1).max())
    accepted = bool(raw.min() >= -1e-11 and abs(float(raw.sum()) - 1) <= TOLERANCE
                    and residual <= TOLERANCE and rank == NUM_BINS)
    row = {
        "name": name, "accepted": accepted, "negative_q_rejection": bool(raw.min() < -1e-11),
        "raw_q_minimum": float(raw.min()), "raw_q_simplex_error": abs(float(raw.sum()) - 1),
        "matrix_residual": residual, "rank": int(rank),
        "condition_number": float(singular[0] / singular[-1]),
    }
    if accepted:
        q = np.maximum(raw, 0)
        q /= q.sum()
        row["q"] = q.tolist()
        row["q_rounding_correction"] = float(np.abs(q - raw).max())
        row["certificate"] = calibration_row(name=name, p=p, q=q)
        row["kkt_residual_after_correction"] = float(np.abs(matrix.T @ q - 1).max())
    return row


def scope_controls() -> dict:
    q, p = np.zeros(NUM_BINS), np.zeros(NUM_BINS)
    q[:3], p[:3] = np.array([35, 19, 13]) / 67, [1 / 2, 1 / 3, 1 / 6]
    original = population_problem(q=q)
    radius_weights = np.tile(np.r_[2.0, np.ones(MAX_AGE - 1)], 3)
    weighted = PopulationProblem(q=q, windows=original.windows,
                                 weights=original.weights * radius_weights * MAX_AGE / NUM_BINS)
    _, gradient = weighted.value_gradient(probabilities=p)
    gap = max(0.0, float(p @ gradient - gradient.min()))
    score = torch.tensor(np.log(1 / 3), dtype=torch.float64, requires_grad=True)
    ages = torch.arange(NUM_BINS, dtype=torch.float64)
    coefficients = torch.lgamma(torch.tensor(float(NUM_BINS), dtype=torch.float64)) - torch.lgamma(ages + 1) - torch.lgamma(NUM_BINS - ages)
    labels = np.zeros(NUM_BINS)
    labels[[0, MAX_AGE]] = [0.75, 0.25]
    loss = repository_expected_loss(logits=coefficients + ages * score, q=labels, method="tailrl_population")
    derivative = float(torch.autograd.grad(loss, score)[0])
    return {
        "doubled_radius_zero": {"mean_difference": float((p - q) @ AGES),
            "expected_difference": -1 / 201, "full_simplex_gap": gap,
            "passed": bool(gap <= TOLERANCE and abs(float((p - q) @ AGES) + 1 / 201) <= TOLERANCE)},
        "restricted_binomial_tail": {"score": float(score.detach()), "target_mean": float(labels @ AGES),
            "policy_mean": float(MAX_AGE * score.detach().sigmoid()), "score_derivative": derivative,
            "passed": bool(abs(derivative) > 1e-8)},
    }


def saved_population(*, path: Path) -> dict:
    saved = json.loads(path.read_text())
    rows = []
    for entry in saved["population_cases"] + saved["noise_population_cases"]:
        row = calibration_row(name=entry["name"], p=np.asarray(entry["tail_population_candidate"]["probabilities"]),
                              q=np.asarray(entry["target"]["probabilities"]))
        row["old_certificate"] = entry["solver"]["chosen_certificate"]
        rows.append(row)
    return {"source": str(path.resolve()), "source_sha256": file_sha256(path=path), "rows": rows}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--population", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(1)
    generator = np.random.default_rng(SEED)
    cases = []
    for index in range(12):
        p = 0.01 / NUM_BINS + 0.99 * generator.dirichlet(np.ones(NUM_BINS))
        q = generator.dirichlet(np.ones(NUM_BINS))
        cases.append(calibration_row(name=f"positive_{index}", p=p, q=q))
    for index in range(6):
        support = generator.choice(NUM_BINS, size=24, replace=False)
        p, q = np.zeros(NUM_BINS), np.zeros(NUM_BINS)
        p[support] = generator.dirichlet(np.ones(len(support)))
        q[support[:8]] = generator.dirichlet(np.ones(8))
        cases.append(calibration_row(name=f"sparse_{index}", p=p, q=q))
    x = np.linspace(-1, 1, NUM_BINS)
    inverse = []
    for name, values in (("uniform", np.ones(NUM_BINS)), ("tilt", np.exp(0.2 * x)),
                         ("asymmetric", np.exp(x + 0.2 * x**2)), ("curved", np.exp(-0.5 * x**2)),
                         ("oscillatory", np.exp(0.3 * np.sin(6 * x)))):
        inverse.append(inverse_kkt(name=name, p=values / values.sum()))
    controls = scope_controls()
    metadata = provenance()
    metadata["source_sha256"]["src/experiments/utkface/mechanism_calibration.py"] = file_sha256(path=Path(__file__))
    report = {"provenance": metadata, "seed": SEED, "tolerance": TOLERANCE,
        "assumptions": ["Complete radii 0..98 with equal weights on the 100-age grid; survival_floor=0.",
            "Unrestricted simplex or a family closed under the CDF-odds orbit; finite risk.",
            "Floating checks report residuals; they are not interval-arithmetic certificates."],
        "cases": cases, "inverse_kkt": inverse, "scope_controls": controls}
    if args.population is not None:
        report["saved_population"] = saved_population(path=args.population)
    report["checks_passed"] = bool(all(row["passed"] for row in cases)
        and all(row["certificate"]["passed"] for row in inverse if row["accepted"])
        and any(row["accepted"] for row in inverse) and all(row["passed"] for row in controls.values()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(path=args.output, payload=report)
    print(json.dumps({"output": str(args.output), "checks_passed": report["checks_passed"],
                      "inverse_kkt_accepted": sum(row["accepted"] for row in inverse)}))
    if not report["checks_passed"]:
        raise AssertionError("Calibration audit failed; inspect the saved residuals")


if __name__ == "__main__":
    main()
