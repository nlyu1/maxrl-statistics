import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np
import polars as pl

from src.experiments.utkface.config import NoiseConfig
from src.experiments.utkface.data import read_manifest
from src.experiments.utkface.noise import noisy_labels
from src.experiments.utkface.policy_centering import METHODS, write_json
from src.experiments.utkface.policy_examples import sha256

COLORS = {"tailrl_population": "#146d69", "ce": "#4b67d3"}
NAMES = {"tailrl_population": "TailRL", "ce": "CE"}
CENTERS = ("mean", "median", "mode")


def demean(*, values: np.ndarray, ages: np.ndarray) -> np.ndarray:
    result = values.astype(np.float64).copy()
    for age in np.unique(ages):
        mask = ages == age
        result[..., mask] -= result[..., mask].mean(-1, keepdims=True)
    return result


def slope(*, x: np.ndarray, y: np.ndarray) -> float:
    if x @ x <= 1e-12:
        raise ValueError("A slope requires nonzero variation in label error")
    return float(x @ y / (x @ x))


def policy_statistics(
    *, log_probs: np.ndarray, ages: np.ndarray, labels: np.ndarray
) -> dict:
    p = np.exp(log_probs.astype(np.float64))
    p /= p.sum(1, keepdims=True)
    actions = np.arange(100)
    mean = p @ actions
    centered = actions[None] - mean[:, None]
    variance = (p * centered**2).sum(1)
    quantiles = {
        f"q{int(q * 100):02}": (p.cumsum(1) < q).sum(1).clip(max=99)
        for q in (0.05, 0.25, 0.5, 0.75, 0.95)
    }
    clean_distance = np.abs(actions[None] - ages[:, None])
    noisy_distance = np.abs(actions[None] - labels[:, None])
    clean_side = (clean_distance < noisy_distance) + 0.5 * (
        clean_distance == noisy_distance
    )
    return {
        "mean": mean,
        "median": quantiles["q50"],
        "mode": log_probs.argmax(1),
        "std_policy": np.sqrt(variance),
        "entropy": -(
            p
            * (log_probs - np.log(np.exp(log_probs.astype(np.float64)).sum(1))[:, None])
        ).sum(1),
        "mean_minus_median": mean - quantiles["q50"],
        "skewness": (p * centered**3).sum(1) / np.maximum(variance, 1e-12) ** 1.5,
        "mode_probability": p.max(1),
        "clean_nll": -log_probs[np.arange(len(ages)), ages],
        "noisy_probe_nll": -log_probs[np.arange(len(ages)), labels],
        "clean_side_mass": (p * clean_side).sum(1),
        "mass_clean_pm2": (p * (clean_distance <= 2)).sum(1),
        "mass_noisy_probe_pm2": (p * (noisy_distance <= 2)).sum(1),
        "expected_clean_mae": (p * clean_distance).sum(1),
        "expected_noisy_probe_mae": (p * noisy_distance).sum(1),
        **quantiles,
    }


def summarize(
    *,
    ages: np.ndarray,
    labels: np.ndarray,
    values: dict,
    baseline: dict,
    controls: np.ndarray,
    mask: np.ndarray,
) -> dict:
    age, noisy = ages[mask], labels[mask]
    delta = noisy - age
    changed = delta != 0
    result = {"count": int(mask.sum()), "changed_count": int(changed.sum())}
    for key in (
        "std_policy",
        "entropy",
        "mode_probability",
        "skewness",
        "clean_nll",
        "noisy_probe_nll",
        "expected_clean_mae",
        "expected_noisy_probe_mae",
    ):
        result[key] = float(values[key][mask].mean())
    result.update({
        "iqr": float((values["q75"][mask] - values["q25"][mask]).mean()),
        "width90": float((values["q95"][mask] - values["q05"][mask]).mean()),
        "coverage90_clean": float(
            ((values["q05"][mask] <= age) & (age <= values["q95"][mask])).mean()
        ),
        "coverage90_noisy_probe": float(
            ((values["q05"][mask] <= noisy) & (noisy <= values["q95"][mask])).mean()
        ),
        "coverage50_clean": float(
            ((values["q25"][mask] <= age) & (age <= values["q75"][mask])).mean()
        ),
        "coverage50_noisy_probe": float(
            ((values["q25"][mask] <= noisy) & (noisy <= values["q75"][mask])).mean()
        ),
        "absolute_mean_minus_median": float(
            np.abs(values["mean_minus_median"][mask]).mean()
        ),
        "mass_clean_pm2": float(values["mass_clean_pm2"][mask].mean()),
        "mass_noisy_probe_pm2": float(values["mass_noisy_probe_pm2"][mask].mean()),
    })
    if changed.any():
        result["clean_side_mass_changed"] = float(
            values["clean_side_mass"][mask][changed].mean()
        )
        result["clean_label_likelihood_wins_changed"] = float(
            (
                values["clean_nll"][mask][changed]
                < values["noisy_probe_nll"][mask][changed]
            ).mean()
        )
        result["clean_log_likelihood_advantage_changed"] = float(
            (
                values["noisy_probe_nll"][mask][changed]
                - values["clean_nll"][mask][changed]
            ).mean()
        )
    centers = {}
    for key in CENTERS:
        center = values[key][mask]
        residual = center - age
        shift = center - baseline[key][mask]
        entry = {
            "mae_clean": float(np.abs(residual).mean()),
            "mae_noisy_probe": float(np.abs(center - noisy).mean()),
            "rmse_clean": float(np.sqrt(np.mean(residual**2))),
            "rmse_noisy_probe": float(np.sqrt(np.mean((center - noisy) ** 2))),
        }
        if changed.any():
            x = delta[changed]
            relative = residual[changed] / x
            difference = np.abs(residual[changed]) - np.abs(
                center[changed] - noisy[changed]
            )
            x_demeaned = demean(values=delta, ages=age)
            residual_demeaned = demean(values=residual, ages=age)
            shift_demeaned = demean(values=shift, ages=age)
            entry.update({
                "closer_clean": float((difference < -1e-7).mean()),
                "closer_noisy": float((difference > 1e-7).mean()),
                "equidistant": float((np.abs(difference) <= 1e-7).mean()),
                "between": float(((relative >= 0) & (relative <= 1)).mean()),
                "position_below_clean": float((relative < 0).mean()),
                "position_clean_half": float(
                    ((relative >= 0) & (relative <= 0.5)).mean()
                ),
                "position_noisy_half": float(
                    ((relative > 0.5) & (relative <= 1)).mean()
                ),
                "position_beyond_noisy": float((relative > 1).mean()),
                "position_median": float(np.median(relative)),
                "slope_raw": slope(x=delta, y=residual),
                "slope_age_adjusted": slope(x=x_demeaned, y=residual_demeaned)
                if x_demeaned @ x_demeaned > 1e-12
                else None,
                "paired_slope_raw": slope(x=delta, y=shift),
                "paired_slope_age_adjusted": slope(x=x_demeaned, y=shift_demeaned)
                if x_demeaned @ x_demeaned > 1e-12
                else None,
            })
            control_delta = controls[:, mask] - age[None]
            control_delta = demean(values=control_delta, ages=age)
            valid = (control_delta**2).sum(1) > 1e-12
            control_delta = control_delta[valid]
            reference = control_delta @ residual_demeaned / (control_delta**2).sum(1)
            paired_reference = (
                control_delta @ shift_demeaned / (control_delta**2).sum(1)
            )
            entry["negative_control_slope_q025_q50_q975"] = (
                np.quantile(reference, [0.025, 0.5, 0.975]).tolist()
                if len(reference)
                else None
            )
            entry["negative_control_paired_slope_q025_q50_q975"] = (
                np.quantile(paired_reference, [0.025, 0.5, 0.975]).tolist()
                if len(paired_reference)
                else None
            )
        centers[key] = entry
    result["centers"] = centers
    return result


def analyze(*, root: Path) -> tuple[dict, dict, dict]:
    audit = json.loads((root / "inference-audit.json").read_text())
    if len(audit["runs"]) != 10:
        raise ValueError("Full inference must finish before analysis")
    payload = read_manifest(path=Path(audit["manifest"]))
    all_ages = [record["age"] for record in payload["records"]]
    first = np.load(audit["runs"][0]["artifact"])
    indices, ages = first["indices"], first["ages"]
    train_mask = np.arange(len(indices)) < audit["train_count"]
    baseline, policies, statistics = {}, {}, {}
    for run in audit["runs"]:
        path = Path(run["artifact"])
        if sha256(path=path) != run["artifact_sha256"]:
            raise ValueError(f"Changed inference artifact: {path}")
        saved = np.load(path)
        if not np.array_equal(indices, saved["indices"]) or not np.array_equal(
            ages, saved["ages"]
        ):
            raise ValueError("Inference cohorts must be identical")
        key = run["method"], int(run["std"])
        labels = noisy_labels(
            ages=all_ages,
            config=NoiseConfig(
                mode="none" if key[1] == 0 else "fixed", std_years=key[1], seed=2026
            ),
            epoch=1,
        ).numpy()[indices]
        policies[key] = saved["log_probs"]
        statistics[key] = policy_statistics(
            log_probs=saved["log_probs"], ages=ages, labels=labels
        )
        if key[1] == 0:
            baseline[run["method"]] = statistics[key]
    report = {
        "inference_commit": audit["commit"],
        "data_fingerprint": audit["data_fingerprint"],
        "training_fingerprint": audit["training_fingerprint"],
        "train_count": audit["train_count"],
        "test_count": audit["test_count"],
        "negative_control_fields": 128,
        "negative_control_seeds": [100000, 100127],
        "definition": "For each center c, slope regresses c-original on the actual rounded/clipped label error after demeaning both variables within exact original age; 0 denotes no association and 1 one-for-one association. Paired slope replaces c-original with c(noise)-c(clean checkpoint).",
        "test_label_definition": "Test labels remained clean. All noisy test labels in this analysis are explicitly counterfactual probes generated from unused positions of the original corruption field, never training targets.",
        "normalization": "Float32 log-softmax probabilities are renormalized in float64 for distribution summaries; collected row-sum errors are below 1e-6.",
        "runs": [],
    }
    frames = []
    for std in (0, 1, 3, 5, 10):
        labels = noisy_labels(
            ages=all_ages,
            config=NoiseConfig(
                mode="none" if std == 0 else "fixed", std_years=std, seed=2026
            ),
            epoch=1,
        ).numpy()[indices]
        controls = np.stack([
            noisy_labels(
                ages=all_ages,
                config=NoiseConfig(
                    mode="none" if std == 0 else "fixed", std_years=std, seed=seed
                ),
                epoch=1,
            ).numpy()[indices]
            for seed in range(100000, 100128)
        ])
        rounded = np.rint(
            np.array(all_ages)
            + np.random.default_rng(np.random.SeedSequence([2026, 0])).normal(
                0, std, len(all_ages)
            )
        )[indices]
        clipped = (rounded < 0) | (rounded > 99)
        delta = labels - ages
        for method in METHODS:
            run = next(
                r for r in audit["runs"] if r["method"] == method and r["std"] == std
            )
            values = statistics[method, std]
            groups = {}
            for split, split_mask in (("train", train_mask), ("test", ~train_mask)):
                masks = {
                    "all": split_mask,
                    "changed": split_mask & (delta != 0),
                    "abs_noise_ge3": split_mask & (np.abs(delta) >= 3),
                    "abs_noise_ge5": split_mask & (np.abs(delta) >= 5),
                    "interior20_79": split_mask
                    & (ages >= 20)
                    & (ages <= 79)
                    & ~clipped,
                    "clipped": split_mask & clipped,
                    "positive_noise": split_mask & (delta > 0),
                    "negative_noise": split_mask & (delta < 0),
                    **{
                        f"decade{decade}": split_mask & (ages // 10 == decade)
                        for decade in range(10)
                    },
                }
                for name, mask in masks.items():
                    if mask.sum() >= 10:
                        groups[f"{split}:{name}"] = summarize(
                            ages=ages,
                            labels=labels,
                            values=values,
                            baseline=baseline[method],
                            controls=controls,
                            mask=mask,
                        )
            report["runs"].append({
                "method": method,
                "std": std,
                "epoch": run["epoch"],
                "groups": groups,
            })
            frame = pl.DataFrame({
                "method": [method] * len(indices),
                "std": [std] * len(indices),
                "split": np.where(train_mask, "train", "test"),
                "index": indices,
                "original_label": ages,
                "corrupted_training_label": [
                    int(labels[i]) if train_mask[i] else None
                    for i in range(len(indices))
                ],
                "counterfactual_test_probe_label": [
                    int(labels[i]) if not train_mask[i] else None
                    for i in range(len(indices))
                ],
                "label_error_or_test_probe_error": delta,
                "label_clipped": clipped,
                **values,
            })
            frames.append(frame)
    pl.concat(frames).write_csv(root / "per-image-statistics.csv")
    write_json(path=root / "centering-statistics.json", data=report)
    return (
        report,
        policies,
        {
            "indices": indices,
            "ages": ages,
            "train": train_mask,
            "statistics": statistics,
            "manifest": payload,
            "manifest_path": audit["manifest"],
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    report, _, _ = analyze(root=args.root)
    for run in report["runs"]:
        group = run["groups"]["train:all"]
        print(
            json.dumps({
                "method": run["method"],
                "std": run["std"],
                "epoch": run["epoch"],
                "summary": group,
            }),
            flush=True,
        )


if __name__ == "__main__":
    main()
