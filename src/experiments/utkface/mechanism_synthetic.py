import argparse
import csv
import hashlib
import json
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
import scipy
import torch
from scipy.linalg import qr
from scipy.optimize import minimize
from scipy.sparse.linalg import LinearOperator, cg
from scipy.special import expit
from threadpoolctl import threadpool_limits

from src.experiments.utkface.losses import MAX_AGE, NUM_BINS, log_policy
from src.experiments.utkface.mechanism_population import file_sha256
from src.experiments.utkface.run import write_json

DIMENSION = 12
BASE_SEED = 2026102919
LAWS = ("binomial", "endpoint")
AGES = torch.arange(NUM_BINS, dtype=torch.float64)
COEFFICIENTS = (
    torch.lgamma(torch.tensor(float(NUM_BINS), dtype=torch.float64))
    - torch.lgamma(AGES + 1) - torch.lgamma(NUM_BINS - AGES)
)
CENTERED = AGES - AGES.mean()
with threadpool_limits(limits=1):
    BASIS = torch.from_numpy(qr(np.column_stack((np.ones(NUM_BINS), CENTERED.numpy())),
                               mode="full")[0][:, 2:].copy())


def payload_hash(*, value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def array_hash(*, values: np.ndarray) -> str:
    prefix = f"{values.dtype}:{values.shape}:".encode()
    return hashlib.sha256(prefix + np.ascontiguousarray(values).tobytes()).hexdigest()


def write_tsv(*, path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def design_rows(*, count: int, seed: int, split: int) -> tuple[np.ndarray, np.ndarray]:
    generator = np.random.default_rng(np.random.SeedSequence([BASE_SEED, seed, split, 0]))
    x = generator.normal(size=(count, DIMENSION))
    mean = MAX_AGE * expit(0.7 * x[:, 0] + 0.4 * x[:, 1] - 0.3 * x[:, 2])
    return np.column_stack((np.ones(count), x)), mean


def observed_labels(*, mean: np.ndarray, seed: int, split: int, law: str) -> np.ndarray:
    stream = LAWS.index(law) + 1
    generator = np.random.default_rng(np.random.SeedSequence([BASE_SEED, seed, split, stream]))
    if law == "binomial":
        return generator.binomial(MAX_AGE, mean / MAX_AGE)
    if law == "endpoint":
        return MAX_AGE * (generator.random(len(mean)) < mean / MAX_AGE).astype(np.int64)
    raise ValueError(f"Unknown conditional label law: {law}")


def model_logits(*, design: torch.Tensor, parameters: torch.Tensor, tied: bool) -> torch.Tensor:
    width = design.shape[1]
    logits = COEFFICIENTS + (design @ parameters[:width])[:, None] * AGES
    if not tied:
        residual = BASIS @ parameters[width:].reshape(NUM_BINS - 2, width)
        logits = logits + design @ residual.T
    return logits


def fit_model(*, design: np.ndarray, targets: np.ndarray, tied: bool,
              penalty: float, score_ridge: float, max_iterations: int) -> tuple[np.ndarray, dict]:
    x, y = torch.from_numpy(design), torch.from_numpy(targets).long()
    width = design.shape[1]
    raw_count = width if tied else (NUM_BINS - 1) * width

    def objective(values: np.ndarray) -> tuple[float, np.ndarray]:
        parameters = torch.tensor(values, dtype=torch.float64, requires_grad=True)
        log_probs = log_policy(logits=model_logits(design=x, parameters=parameters, tied=tied))
        value = -log_probs[torch.arange(len(y)), y].mean()
        value = value + 0.5 * score_ridge * parameters[:width].square().sum()
        if not tied:
            value = value + 0.5 * penalty * parameters[width:].square().sum()
        gradient = torch.autograd.grad(value, parameters)[0]
        return float(value.detach()), gradient.detach().numpy()

    started = time.monotonic()
    result = minimize(objective, np.zeros(raw_count), method="L-BFGS-B", jac=True,
                      options={"maxiter": max_iterations, "maxfun": 4 * max_iterations,
                               "gtol": 1e-8, "ftol": 1e-12, "maxls": 40, "maxcor": 20})
    theta = result.x.copy()
    initial_value, initial_gradient = objective(theta)
    ridge = torch.full((raw_count,), penalty, dtype=torch.float64)
    ridge[:width] = score_ridge
    polishing_steps = hvp_evaluations = backtracks = 0
    cg_statuses = []
    termination = "step_limit"
    for _ in range(12):
        value, gradient = objective(theta)
        if float(np.abs(gradient).max()) <= 1e-8:
            termination = "gradient_tolerance"
            break
        with torch.inference_mode():
            p = log_policy(logits=model_logits(design=x, parameters=torch.from_numpy(theta), tied=tied)).exp()
            variance = ((p @ AGES.square()) - (p @ AGES).square()).clamp_min(0)
            diagonal = x.square().T @ variance / len(x) + score_ridge
            if not tied:
                variance_basis = ((p @ BASIS.square()) - (p @ BASIS).square()).clamp_min(0)
                diagonal = torch.cat((diagonal, (variance_basis.T @ x.square() / len(x) + penalty).flatten()))

        def product(vector: np.ndarray) -> np.ndarray:
            nonlocal hvp_evaluations
            hvp_evaluations += 1
            direction = torch.from_numpy(vector)
            logits_direction = (x @ direction[:width])[:, None] * AGES
            if not tied:
                logits_direction += x @ (BASIS @ direction[width:].reshape(NUM_BINS - 2, width)).T
            weighted = p * (logits_direction - (p * logits_direction).sum(-1, keepdim=True)) / len(x)
            answer = x.T @ (weighted @ AGES)
            if not tied:
                answer = torch.cat((answer, (BASIS.T @ (weighted.T @ x)).flatten()))
            return (answer + ridge * direction).numpy()

        operator = LinearOperator((raw_count, raw_count), matvec=product, dtype=np.float64)
        preconditioner = LinearOperator(operator.shape, matvec=lambda v: v / diagonal.numpy(), dtype=np.float64)
        direction, status = cg(operator, -gradient, M=preconditioner, rtol=1e-10, atol=1e-14, maxiter=400)
        cg_statuses.append(int(status))
        slope = float(gradient @ direction)
        if not np.isfinite(direction).all() or slope >= 0:
            termination = "non_descent_newton_direction"
            break
        step, accepted = 1.0, False
        rounding = 32 * np.finfo(np.float64).eps * max(1.0, abs(value))
        for _ in range(30):
            candidate_value, _ = objective(theta + step * direction)
            if candidate_value <= value + 1e-4 * step * slope + rounding:
                theta += step * direction
                accepted = True
                polishing_steps += 1
                break
            step *= 0.5
            backtracks += 1
        if not accepted:
            termination = "line_search_limit"
            break
    value, gradient = objective(theta)
    if not np.isfinite(theta).all() or not np.isfinite(gradient).all():
        raise RuntimeError(f"Nonfinite convex fit: {result.message}")
    norm = float(np.abs(gradient).max())
    projected = np.zeros((NUM_BINS, width)) if tied else (
        BASIS.numpy() @ theta[width:].reshape(NUM_BINS - 2, width)
    )
    return theta, {
        "scipy_success": bool(result.success), "scipy_status": int(result.status),
        "scipy_message": str(result.message), "iterations": int(result.nit),
        "function_evaluations": int(result.nfev), "objective": value,
        "gradient_infinity_norm": norm, "gradient_tolerance_met": bool(norm <= 1e-8),
        "lbfgs_gradient_infinity_norm": float(np.abs(initial_gradient).max()),
        "polishing_steps": polishing_steps, "polishing_hvp_evaluations": hvp_evaluations,
        "polishing_backtracks": backtracks, "polishing_cg_max_abs_status": max((abs(status) for status in cg_statuses), default=0),
        "polishing_cg_nonconverged": sum(status != 0 for status in cg_statuses),
        "polishing_termination": "gradient_tolerance" if norm <= 1e-8 else termination,
        "strong_convexity_lower_bound": score_ridge if tied else min(score_ridge, penalty),
        "objective_gap_upper_bound": float(gradient @ gradient) / (2 * (score_ridge if tied else min(score_ridge, penalty))),
        "polishing_objective_change": value - initial_value,
        "wall_seconds": time.monotonic() - started, "raw_parameters": raw_count,
        "effective_parameters": width if tied else (NUM_BINS - 1) * width,
        "residual_constant_component_max": float(np.abs(projected.mean(0)).max()),
        "residual_age_component_max": float(np.abs(CENTERED.numpy() @ projected).max()),
    }


def predictions(*, design: np.ndarray, parameters: np.ndarray, tied: bool) -> np.ndarray:
    pieces = []
    with torch.inference_mode():
        theta = torch.from_numpy(parameters)
        for start in range(0, len(design), 2048):
            x = torch.from_numpy(design[start:start + 2048])
            logits = model_logits(design=x, parameters=theta, tied=tied)
            pieces.append((log_policy(logits=logits).exp() @ AGES).numpy())
    return np.concatenate(pieces)


def paired_rows(*, fits: list[dict], selected: list[dict]) -> tuple[list[dict], list[dict]]:
    rows = []
    for selection in selected:
        members = [row for row in fits if all(row[key] == selection[key]
                   for key in ("law", "training_size", "seed"))]
        tied = next(row for row in members if row["tied"])
        comparisons = [(row["fit_id"], row) for row in members if not row["tied"]]
        for label, key in (("validation_selected", "selected_fit"),
                           ("validation_selected_full", "selected_full_fit")):
            comparisons.append((label, next(row for row in members if row["fit_id"] == selection[key])))
        for comparison, row in comparisons:
            rows.append({"law": row["law"], "training_size": row["training_size"],
                         "seed": row["seed"], "comparison": comparison,
                         "fit_id": row["fit_id"], "tied_fit_id": tied["fit_id"],
                         "residual_penalty": row["residual_penalty"],
                         "oracle_mse_difference": row["oracle_mean_mse"] - tied["oracle_mean_mse"],
                         "oracle_mae_difference": row["oracle_mean_mae"] - tied["oracle_mean_mae"]})
    summaries = []
    groups = {(row["law"], row["training_size"], row["comparison"] if
               row["comparison"].startswith("validation_") else row["residual_penalty"]) for row in rows}
    for law, size, contrast in sorted(groups):
        values = [row for row in rows if row["law"] == law and row["training_size"] == size
                  and (row["comparison"] if row["comparison"].startswith("validation_")
                       else row["residual_penalty"]) == contrast]
        summary = {"law": law, "training_size": size, "contrast": contrast,
                   "seeds": [row["seed"] for row in values], "count": len(values)}
        for metric in ("oracle_mse_difference", "oracle_mae_difference"):
            differences = np.asarray([row[metric] for row in values])
            summary[metric + "_mean"] = float(differences.mean())
            summary[metric + "_sd"] = float(differences.std(ddof=1)) if len(values) > 1 else None
        summaries.append(summary)
    return rows, summaries


def run_study(*, args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    config = {"dimension": DIMENSION, "base_seed": BASE_SEED,
              "seeds": [101] if args.smoke else [101, 102, 103, 104, 105],
              "training_sizes": [128] if args.smoke else [128, 512, 2048],
              "validation_size": 256 if args.smoke else 2048,
              "oracle_test_size": 1000 if args.smoke else 20000,
              "oracle_split": 2 if args.smoke else 3,
              "residual_penalties": [0.1] if args.smoke else [0.001, 0.1, 10.0],
              "score_ridge": args.score_ridge, "max_iterations": args.max_iterations,
              "laws": list(LAWS), "true_score": "0.7*X0+0.4*X1-0.3*X2",
              "selection": "Minimum observed-label validation MSE; ties favor fewer effective parameters.",
              "penalty": "Mean categorical CE + score_ridge*||beta||^2/2 + residual_penalty*||projected_R||^2/2; includes intercepts.",
              "initialization": "All score and raw residual parameters are zero for every fit.",
              "parameterization": "Residual logits use an orthonormal 98-dimensional complement to constant and age vectors, with no redundant coordinates. This preserves the former projected-100-row function class and residual L2 penalty. The full class contains the tied class.",
              "polishing": "Up to 12 preconditioned Newton-CG steps, 400 CG iterations per step, Armijo backtracking with a float64 rounding allowance. Exact GLM Hessian-vector products; objective gap <= ||gradient||^2/(2*minimum ridge).",
              "residual_basis_sha256": array_hash(values=BASIS.numpy()),
              "pairing": "Inputs are shared across laws, penalties and nested sample sizes within each independent seed block. Label RNG streams differ across laws. Oracle inputs are shared across all fits within a seed.",
              "metric_scope": "Oracle mean MSE is excess observed-label squared risk. Oracle mean MAE measures error to the conditional mean; it is not excess observed-label absolute risk.",
              "scope": "Designed conditional label laws with the same mean and different shape/noise; this is not a new image experiment."}
    write_json(path=args.output_dir / "protocol.json", payload=config)
    root = Path(__file__).resolve().parents[3]
    report = {"status": "fitting", "config": config, "config_sha256": payload_hash(value=config),
              "source_sha256": {name: file_sha256(path=root / name) for name in
                  ("src/experiments/utkface/mechanism_synthetic.py", "src/experiments/utkface/losses.py",
                   "src/experiments/utkface/mechanism_population.py", "src/experiments/utkface/run.py", "pyproject.toml")},
              "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
              "environment": {"python": platform.python_version(), "numpy": np.__version__,
                  "scipy": scipy.__version__, "torch": str(torch.__version__),
                  "device": "cpu", "precision": "float64", "threads": 1},
              "fits": [], "selections": [], "oracle_test_data": []}
    parameters = {}
    for seed in config["seeds"]:
        train_x, train_mean = design_rows(count=max(config["training_sizes"]), seed=seed, split=0)
        val_x, val_mean = design_rows(count=config["validation_size"], seed=seed, split=1)
        for law in LAWS:
            train_y = observed_labels(mean=train_mean, seed=seed, split=0, law=law)
            val_y = observed_labels(mean=val_mean, seed=seed, split=1, law=law)
            for size in config["training_sizes"]:
                members = []
                for penalty in ["tied", *config["residual_penalties"]]:
                    tied = penalty == "tied"
                    fit_id = f"{law}_n{size}_seed{seed}_penalty{penalty}"
                    theta, diagnostics = fit_model(design=train_x[:size], targets=train_y[:size],
                        tied=tied, penalty=0.0 if tied else float(penalty),
                        score_ridge=args.score_ridge, max_iterations=args.max_iterations)
                    val_pred = predictions(design=val_x, parameters=theta, tied=tied)
                    row = {"fit_id": fit_id, "law": law, "training_size": size, "seed": seed,
                           "tied": bool(tied), "residual_penalty": str(penalty), **diagnostics,
                           "validation_observed_mse": float(np.mean((val_pred - val_y)**2)),
                           "validation_observed_mae": float(np.mean(np.abs(val_pred - val_y))),
                           "train_x_sha256": array_hash(values=train_x[:size]),
                           "train_y_sha256": array_hash(values=train_y[:size]),
                           "validation_x_sha256": array_hash(values=val_x),
                           "validation_y_sha256": array_hash(values=val_y),
                           "parameters_sha256": array_hash(values=theta)}
                    parameters[fit_id] = theta
                    report["fits"].append(row)
                    members.append(row)
                    write_json(path=args.output_dir / "report.json", payload=report)
                    print(json.dumps({"fit": fit_id, "val_mse": row["validation_observed_mse"],
                                      "grad_inf": row["gradient_infinity_norm"],
                                      "seconds": row["wall_seconds"]}), flush=True)
                def ranking(row: dict) -> tuple:
                    return row["validation_observed_mse"], row["effective_parameters"], row["fit_id"]
                chosen = min(members, key=ranking)
                full = min((row for row in members if not row["tied"]), key=ranking)
                report["selections"].append({"law": law, "training_size": size, "seed": seed,
                    "selected_fit": chosen["fit_id"], "selected_full_fit": full["fit_id"],
                    "validation_mse": chosen["validation_observed_mse"]})
    np.savez_compressed(args.output_dir / "parameters.npz", **parameters)
    freeze = {"config_sha256": report["config_sha256"], "selections": report["selections"],
              "parameters_sha256": file_sha256(path=args.output_dir / "parameters.npz"),
              "oracle_test_generated": False}
    write_json(path=args.output_dir / "frozen-selection.json", payload=freeze)
    report["frozen_selection_sha256"] = file_sha256(path=args.output_dir / "frozen-selection.json")
    report["status"] = "oracle_evaluation"
    write_json(path=args.output_dir / "report.json", payload=report)
    selected_ids = {row["selected_fit"] for row in freeze["selections"]}
    for seed in config["seeds"]:
        test_x, test_mean = design_rows(count=config["oracle_test_size"], seed=seed, split=config["oracle_split"])
        report["oracle_test_data"].append({"seed": seed,
            "x_sha256": array_hash(values=test_x), "mean_sha256": array_hash(values=test_mean)})
        for row in (row for row in report["fits"] if row["seed"] == seed):
            pred = predictions(design=test_x, parameters=parameters[row["fit_id"]], tied=row["tied"])
            row["oracle_mean_mse"] = float(np.mean((pred - test_mean)**2))
            row["oracle_mean_mae"] = float(np.mean(np.abs(pred - test_mean)))
            row["validation_selected"] = bool(row["fit_id"] in selected_ids)
    pairs, summaries = paired_rows(fits=report["fits"], selected=freeze["selections"])
    report["paired_differences"], report["paired_summaries"] = pairs, summaries
    report["all_gradients_converged"] = bool(all(row["gradient_tolerance_met"] for row in report["fits"]))
    report["all_scipy_success"] = bool(all(row["scipy_success"] for row in report["fits"]))
    report["wall_seconds"] = time.monotonic() - started
    report["status"] = "complete"
    write_json(path=args.output_dir / "report.json", payload=report)
    write_tsv(path=args.output_dir / "fits.tsv", rows=report["fits"])
    write_tsv(path=args.output_dir / "paired-differences.tsv", rows=pairs)
    print(json.dumps({"output": str(args.output_dir), "fits": len(report["fits"]),
                      "wall_seconds": report["wall_seconds"],
                      "all_gradients_converged": report["all_gradients_converged"]}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--score-ridge", type=float, default=1e-3)
    parser.add_argument("--max-iterations", type=int, default=500)
    args = parser.parse_args()
    if not np.isfinite(args.score_ridge) or args.score_ridge <= 0 or args.max_iterations < 1:
        raise ValueError("Use a finite positive score ridge and positive iteration budget")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    with threadpool_limits(limits=1):
        run_study(args=args)


if __name__ == "__main__":
    main()
