import argparse
import base64
import html
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from PIL import Image

from src.experiments.utkface.config import NoiseConfig
from src.experiments.utkface.data import image_transform, read_manifest
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
        "mass_exact_two_labels": p[np.arange(len(ages)), ages]
        + p[np.arange(len(ages)), labels] * (ages != labels),
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
        result["coverage90_both_changed"] = float(
            (
                (values["q05"][mask][changed] <= age[changed])
                & (age[changed] <= values["q95"][mask][changed])
                & (values["q05"][mask][changed] <= noisy[changed])
                & (noisy[changed] <= values["q95"][mask][changed])
            ).mean()
        )
        result["mass_exact_two_labels_changed"] = float(
            values["mass_exact_two_labels"][mask][changed].mean()
        )
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
        "analysis_source_sha256": sha256(path=Path(__file__)),
        "inference_audit_sha256": sha256(path=root / "inference-audit.json"),
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


def figures(*, root: Path, report: dict, policies: dict, data: dict) -> list[dict]:
    plt.rcParams.update({
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "savefig.dpi": 170,
    })
    lookup = {(r["method"], r["std"]): r for r in report["runs"]}
    levels = [1, 3, 5, 10]
    full_levels = [0, *levels]

    def group(method: str, std: int, split: str = "train") -> dict:
        return lookup[method, std]["groups"][f"{split}:all"]

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), layout="constrained")
    for method in METHODS:
        color, name = COLORS[method], NAMES[method]
        clean_mae = [
            group(method, s)["centers"]["mean"]["mae_clean"] for s in full_levels
        ]
        noisy_mae = [
            group(method, s)["centers"]["mean"]["mae_noisy_probe"] for s in full_levels
        ]
        axes[0, 0].plot(
            full_levels, clean_mae, "o-", color=color, label=f"{name}: original label"
        )
        axes[0, 0].plot(
            full_levels, noisy_mae, "s--", color=color, label=f"{name}: noisy label"
        )
        axes[0, 1].plot(
            levels,
            [group(method, s)["centers"]["mean"]["slope_age_adjusted"] for s in levels],
            "o-",
            color=color,
            label=f"{name}: own training noise",
        )
    control = np.array([
        group("tailrl_population", s)["centers"]["mean"][
            "negative_control_slope_q025_q50_q975"
        ]
        for s in levels
    ])
    axes[0, 1].fill_between(
        levels,
        control[:, 0],
        control[:, 2],
        color="#aeb7c5",
        alpha=0.35,
        label="TailRL: independent-field 95% range",
    )
    axes[0, 1].plot(
        levels,
        [
            group("tailrl_population", s, "test")["centers"]["mean"][
                "slope_age_adjusted"
            ]
            for s in levels
        ],
        "^:",
        color="#747e8e",
        label="TailRL: unused test-label probe",
    )
    for y in (0, 1):
        axes[0, 1].axhline(y, color="#aeb7c5", lw=0.8)
    axes[0, 0].set(
        title="Point accuracy alone does not identify denoising",
        ylabel="Training policy-mean MAE (years)",
    )
    axes[0, 0].legend(fontsize=8.5)
    axes[0, 1].set(
        title="TailRL follows some of each image's own corruption",
        ylabel="Age-adjusted noise association slope",
        ylim=(-0.18, 1.12),
    )
    axes[0, 1].legend(fontsize=8, loc="upper right")
    for center, marker in zip(CENTERS, ("o", "s", "^"), strict=True):
        axes[1, 0].plot(
            levels,
            [
                100 * group("tailrl_population", s)["centers"][center]["between"]
                for s in levels
            ],
            marker=marker,
            label=center.capitalize(),
        )
    axes[1, 0].set(
        title="Most centers are not literally between the two labels",
        ylabel="Between original and noisy labels (%)",
        ylim=(0, 100),
    )
    axes[1, 0].legend()
    for center, marker in zip(CENTERS, ("o", "s", "^"), strict=True):
        axes[1, 1].plot(
            levels,
            [
                100 * group("tailrl_population", s)["centers"][center]["closer_clean"]
                for s in levels
            ],
            marker=marker,
            label=f"{center.capitalize()} closer to original",
        )
    axes[1, 1].plot(
        levels,
        [
            100 * group("tailrl_population", s)["clean_label_likelihood_wins_changed"]
            for s in levels
        ],
        "d--",
        color="#a76e27",
        label="p(original) > p(noisy)",
    )
    axes[1, 1].axhline(50, color="#aeb7c5", lw=0.8)
    axes[1, 1].set(
        title="Mean position and exact-label likelihood can disagree",
        ylabel="Training faces with changed labels (%)",
        ylim=(0, 100),
    )
    axes[1, 1].legend(fontsize=8.5)
    for ax in axes.flat:
        ax.set_xlabel("Training-noise SD (years)")
        ax.set_xticks(full_levels if ax is axes[0, 0] else levels)
        ax.grid(alpha=0.18)
    fig.suptitle(
        "Fixed-noise UTKFace: full-cohort evidence\n4,096 training faces; 2,308 held-out test probes; validation-selected checkpoints",
        fontsize=13,
    )
    fig.savefig(root / "centering-evidence.png")
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), layout="constrained")
    for method in METHODS:
        for split, style in (("train", "o-"), ("test", "s--")):
            axes[0, 0].plot(
                full_levels,
                [group(method, s, split)["std_policy"] for s in full_levels],
                style,
                color=COLORS[method],
                label=f"{NAMES[method]} {split}",
            )
    axes[0, 0].set(
        title="Policies broaden as training noise increases",
        ylabel="Mean within-policy SD (years)",
    )
    axes[0, 0].legend(fontsize=9)
    for label, key, color, style in (
        ("Original label", "coverage90_clean", COLORS["tailrl_population"], "o-"),
        ("Noisy training label", "coverage90_noisy_probe", "#b55a12", "s--"),
        (
            "Both labels, changed-label faces",
            "coverage90_both_changed",
            "#747e8e",
            "^:",
        ),
    ):
        axes[0, 1].plot(
            levels,
            [100 * group("tailrl_population", s)[key] for s in levels],
            style,
            color=color,
            label=label,
        )
    axes[0, 1].set(
        title="Broad support often covers both labels",
        ylabel="Inside equal-tailed 90% policy interval (%)",
        ylim=(0, 100),
    )
    axes[0, 1].legend(fontsize=8.5)
    ages = data["ages"]
    all_ages = [record["age"] for record in data["manifest"]["records"]]
    delta = (
        noisy_labels(
            ages=all_ages,
            config=NoiseConfig(mode="fixed", std_years=5, seed=2026),
            epoch=1,
        ).numpy()[data["indices"]]
        - ages
    )
    mask = data["train"]
    x = demean(values=delta[mask], ages=ages[mask])
    response = data["statistics"]["tailrl_population", 5]["mean"][mask] - ages[mask]
    y = demean(values=response, ages=ages[mask])
    axes[1, 0].hexbin(
        x, y, gridsize=40, mincnt=1, bins="log", cmap="BuGn", extent=(-20, 20, -25, 25)
    )
    xx = np.array([-20, 20])
    axes[1, 0].plot(
        xx, xx, "--", color="#b55a12", lw=1, label="One-for-one noise following"
    )
    axes[1, 0].plot(
        xx,
        xx * group("tailrl_population", 5)["centers"]["mean"]["slope_age_adjusted"],
        color="#172337",
        lw=1.5,
        label="Observed slope 0.51",
    )
    axes[1, 0].axhline(0, color="#aeb7c5", lw=0.8)
    axes[1, 0].set(
        title="SD 5: partial association, substantial image-wise residuals",
        xlabel="Within-age corruption residual (years)",
        ylabel="Within-age policy-mean error (years)",
        xlim=(-20, 20),
        ylim=(-25, 25),
    )
    axes[1, 0].legend(fontsize=8)
    positions = (
        "position_below_clean",
        "position_clean_half",
        "position_noisy_half",
        "position_beyond_noisy",
    )
    position_names = (
        "Outside, original side",
        "Between, original half",
        "Between, noisy half",
        "Outside, noisy side",
    )
    shades = ("#206d67", "#88bcb6", "#e7be83", "#b76721")
    bottom = np.zeros(4)
    for key, label, color in zip(positions, position_names, shades, strict=True):
        height = np.array([
            100 * group("tailrl_population", s)["centers"]["mean"][key] for s in levels
        ])
        axes[1, 1].bar(np.arange(4), height, bottom=bottom, label=label, color=color)
        bottom += height
    axes[1, 1].set(
        title="TailRL mean locations vary across individual faces",
        ylabel="Changed-label training faces (%)",
        xticks=np.arange(4),
        xticklabels=levels,
        xlabel="Training-noise SD (years)",
        ylim=(0, 100),
    )
    axes[1, 1].legend(
        fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2
    )
    for ax in (axes[0, 0], axes[0, 1]):
        ax.set_xlabel("Training-noise SD (years)")
        ax.set_xticks(full_levels if ax is axes[0, 0] else levels)
        ax.grid(alpha=0.18)
    fig.suptitle(
        "Distribution shape: broadening does not imply a clean-centered or calibrated posterior",
        fontsize=13,
    )
    fig.savefig(root / "distribution-characterization.png")
    plt.close(fig)

    std = 10
    labels = noisy_labels(
        ages=all_ages,
        config=NoiseConfig(mode="fixed", std_years=std, seed=2026),
        epoch=1,
    ).numpy()[data["indices"]]
    delta = labels - ages
    means = data["statistics"]["tailrl_population", std]["mean"]
    eligible = mask & (ages >= 20) & (ages <= 79) & (np.abs(delta) >= 8)
    ratio = np.divide(means - ages, delta, out=np.zeros_like(means), where=delta != 0)
    definitions = (
        ("Near the original label", eligible & (np.abs(ratio) <= 0.2), 0.0),
        ("Near the noisy label", eligible & (np.abs(ratio - 1) <= 0.2), 1.0),
        ("Between the labels", eligible & (ratio >= 0.35) & (ratio <= 0.65), 0.5),
        ("Outside the two labels", eligible & (ratio < -0.4), -0.5),
    )
    selected = []
    fig, axes = plt.subplots(
        4, 2, figsize=(13, 12), width_ratios=[1, 5], layout="constrained"
    )
    transform = image_transform(training=False).transforms[:2]
    for row, (name, candidate_mask, target) in enumerate(definitions):
        candidates = np.flatnonzero(candidate_mask)
        if not len(candidates):
            raise ValueError(f"No qualifying illustrative example: {name}")
        index = int(candidates[np.argmin(np.abs(ratio[candidates] - target))])
        selected.append({
            "case": name,
            "record_index": int(data["indices"][index]),
            "original": int(ages[index]),
            "noisy": int(labels[index]),
            "mean": float(means[index]),
            "relative_position": float(ratio[index]),
            "rule": "Age 20-79, absolute corruption at least 8 years; closest relative mean position to the pre-specified category target.",
        })
        record = data["manifest"]["records"][data["indices"][index]]
        with Image.open(
            Path(data["manifest_path"]).parent / record["relative_path"]
        ) as source:
            rgb = source.convert("RGB")
            for operation in transform:
                rgb = operation(rgb)
            axes[row, 0].imshow(rgb)
        axes[row, 0].axis("off")
        ax = axes[row, 1]
        probs = np.exp(policies["tailrl_population", std][index].astype(np.float64))
        ax.bar(
            np.arange(100),
            probs,
            width=0.88,
            color=COLORS["tailrl_population"],
            alpha=0.85,
            label="TailRL",
        )
        ce = np.exp(policies["ce", std][index].astype(np.float64))
        ax.plot(np.arange(100), ce, color=COLORS["ce"], lw=1.1, alpha=0.8, label="CE")
        ax.axvline(ages[index], color="#172337", lw=1.4, label="Original label")
        ax.axvline(labels[index], color="#b55a12", ls="--", lw=1.4, label="Noisy label")
        ax.axvline(
            means[index],
            color=COLORS["tailrl_population"],
            ls=":",
            lw=1.8,
            label="TailRL mean",
        )
        ax.set(
            title=f"{name}: original {ages[index]}, noisy {labels[index]}, mean {means[index]:.2f}, mode {probs.argmax()}",
            xlabel="Age (years)",
            ylabel="Probability",
            xlim=(-1, 100),
        )
        ax.grid(axis="y", alpha=0.18)
        if row == 0:
            ax.legend(fontsize=8.5, ncol=5)
    fig.suptitle(
        "Examples at fixed SD 10: all four behaviors exist\nSelected to illustrate categories; their frequencies come from the full-cohort tables",
        fontsize=13,
    )
    fig.savefig(root / "centering-examples.png")
    plt.close(fig)
    write_json(path=root / "illustrative-examples.json", data={"selection": selected})
    return selected


def html_report(*, root: Path, report: dict) -> None:
    selected = [
        r for r in report["runs"] if r["method"] == "tailrl_population" and r["std"] > 0
    ]
    rows = []
    for run in selected:
        g = run["groups"]["train:all"]
        c = g["centers"]["mean"]
        rows.append(
            f"<tr><td>{run['std']}</td><td>{run['epoch']}</td><td>{c['mae_clean']:.2f}</td><td>{c['mae_noisy_probe']:.2f}</td><td>{c['slope_age_adjusted']:.2f}</td><td>{100 * c['closer_clean']:.1f}%</td><td>{100 * c['between']:.1f}%</td><td>{g['std_policy']:.2f}</td></tr>"
        )
    table = (
        "<table><thead><tr><th>Noise SD</th><th>Epoch</th><th>MAE to original</th><th>MAE to noisy</th><th>Noise slope</th><th>Closer to original</th><th>Literally between</th><th>Policy SD</th></tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table>"
    )
    figures_html = []
    for filename, caption in (
        (
            "centering-evidence.png",
            "Point summaries, own-noise association, and counterevidence to a universal center.",
        ),
        (
            "distribution-characterization.png",
            "Distribution widths, interval coverage, and individual heterogeneity.",
        ),
        (
            "centering-examples.png",
            "Illustrative faces selected to show original-centered, noisy-centered, intermediate, and outside behavior. Full frequencies are in the tables above.",
        ),
    ):
        image = base64.b64encode((root / filename).read_bytes()).decode()
        figures_html.append(
            f'<figure><img src="data:image/png;base64,{image}" alt="{html.escape(caption)}"><figcaption>{html.escape(caption)}</figcaption></figure>'
        )
    doc = """<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>TailRL fixed-noise centering evidence</title><style>body{font:16px/1.6 system-ui,sans-serif;color:#182337;background:#f5f6f8;margin:0}main{max-width:1240px;margin:auto;padding:32px 24px}h1{font-size:30px;line-height:1.2}h2{font-size:21px;margin-top:30px}p{max-width:1100px}table{border-collapse:collapse;background:white;width:100%;font-size:14px}td,th{padding:10px;border-bottom:1px solid #d9dee7;text-align:left}figure{margin:28px 0;background:#fff}img{width:100%;height:auto;display:block}figcaption{font-size:13px;padding:10px;color:#526176}.table{overflow:auto}.note{border-left:3px solid #b55a12;padding:4px 16px}code{overflow-wrap:anywhere}a{color:#146d69}li{margin:6px 0}@media print{body{background:white}figure{break-inside:avoid}main{padding:0}h2{break-after:avoid}}</style></head><body><main>
<h1>Where is the TailRL policy centered under fixed label noise?</h1>
<p><strong>The evidence supports partial, heterogeneous noise-following.</strong> Neither a universal original-label center nor a universal noisy-label center fits these checkpoints. An intermediate association describes the average response, but most per-image means are outside the interval between the two labels.</p>
<p>Re-evaluated all 4,096 training faces and 2,308 held-out test faces for TailRL and CE at noise SD 0, 1, 3, 5, and 10. All policies use deterministic evaluation crops and bfloat16 CNN inference. Original test metrics were reproduced within 0.02 years/nats; exact counts, checkpoint hashes, and corruption fields were verified.</p>
<div class="table">__TABLE__</div>
<p>MAE and policy SD are in years. Closer/between percentages use only changed-label training images. The noise slope regresses policy mean minus original label against actual rounded/clipped corruption, after removing mean differences within each exact original age. A slope of 0 means no within-age association with the corruption; a slope of 1 means one-for-one association. It is not a per-image mixing weight and does not say that prediction errors vanish at slope 0.</p>
<h2>Evidence for and against the three hypotheses</h2>
<ul><li><strong>Always centered on the original label: contradicted.</strong> Own-noise slopes are 0.58, 0.61, 0.51, and 0.24. At SD 5, independent corruption fields give a central 95% reference range of about [-0.032, 0.032], whereas the observed slope is 0.507. The clean-to-noisy checkpoint shift gives almost the same slope, 0.507; restricting to unclipped original ages 20-79 gives 0.505.</li>
<li><strong>Always centered on the noisy label: also contradicted.</strong> The slopes are well below 1, especially at SD 10. There, original-label MAE is 6.19 years versus 8.33 for the noisy label, and 63.7% of changed-label means are closer to the original. But at SD 3 the noisy label is slightly closer on average, and at SD 5 its exact label has higher probability on 57.3% of changed-label faces.</li>
<li><strong>Always centered between the labels: useful only as an aggregate description.</strong> At SD 5 the noise association is about half-way, but only 36.6% of individual means are literally between the labels. The corresponding median and mode fractions are 45.5% and 40.4%. Prediction error, face ambiguity, and model bias leave many centers on either side of the interval. A mean between labels does not establish two peaks or a mixture of their labels.</li></ul>
<h2>What the distributions look like</h2>
<p>They broaden with noise: mean training-policy SD increases from 1.52 years without corruption to 3.07, 3.79, 4.57, and 8.82 years. At SD 5, the mean 90% interval width is 14.63 years; its marginal original/noisy-label inclusion rates are 87.4% and 88.0%. At SD 10 the interval width is 28.69 years. Broad support can retain probability at both labels even when the mean leans toward one.</p>
<p>Mean, median, and mode tell a broadly consistent story of partial noise-following. At SD 5 their within-age slopes are 0.507, 0.512, and 0.529. Yet likelihood can disagree with mean distance: the means slightly favor the original label overall, while the noisy label more often wins the exact-probability comparison. These distributions should not be called clean-centered, Gaussian, or calibrated posteriors from those facts alone.</p>
<h2>A useful CE counterexample</h2>
<p>At SD 1, CE almost memorizes the noisy labels (training noisy-label MAE 0.13 years; noise slope 0.98), while TailRL has noisy-label MAE 2.45 years and slope 0.58. At SD 3 and 5, however, TailRL follows more of the own-label noise than the selected CE checkpoint (slopes 0.61 vs 0.35, and 0.51 vs 0.28), while still having lower original-label MAE. Better accuracy and less noise-following are distinct properties. CE and TailRL are selected at different epochs, so this comparison does not isolate an objective-only causal effect.</p>
<h2>Why this can happen</h2>
<p>The implemented TailRL loss is the average negative log probability of nested age windows around the supplied training label, including the radius-zero window. For one fixed noisy label, every term is nonnegative and a point mass at that noisy label makes all terms zero. Any policy with less than all its mass on that label has a positive radius-zero term. Thus the objective itself has no instruction to center on the unobserved original training label. A shared network can instead learn common visual structure and only partly fit image-specific corruption; selecting checkpoints using clean validation labels can also favor incomplete fitting. This is an interpretation of the observed behavior, not a separately identified mechanism.</p>
<h2>Negative controls and limits</h2>
<p>For each setting, 128 independent rounded/clipped Gaussian corruption fields were assigned to the same faces without changing the model. The displayed 95% range is variation across these counterfactual fields, not a confidence interval over training runs. Held-out test labels were never corrupted during training: their synthetic noisy labels here are explicitly unused probes. Their slopes fall inside their independent-field reference ranges; the SD 3/5/10 slopes are -0.007, 0.016, and -0.002.</p>
<p class="note">One optimization seed, one training corruption field, one fixed data split, and validation-selected checkpoints. TailRL selected epochs 23, 22, 19, and 12 as noise increases; the decreasing slope cannot be attributed entirely to noise strength or to an intrinsic denoising guarantee. Original UTKFace annotations are the reference labels, not independently verified true ages. No final-epoch checkpoints, noise-seed replications, or matched-epoch causal comparison were added.</p>
__FIGURES__
<p>Data fingerprint: <code>__FINGERPRINT__</code>. Complete per-image statistics, subgroup summaries, source artifacts, checkpoint hashes, and independently generated field settings are retained beside this report.</p>
</main></body></html>"""
    (root / "report.html").write_text(
        doc
        .replace("__TABLE__", table)
        .replace("__FIGURES__", "".join(figures_html))
        .replace("__FINGERPRINT__", html.escape(report["data_fingerprint"]))
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    report, policies, data = analyze(root=args.root)
    figures(root=args.root, report=report, policies=policies, data=data)
    html_report(root=args.root, report=report)
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
