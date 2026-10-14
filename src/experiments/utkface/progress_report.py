import argparse
import hashlib
import json
import math
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import polars as pl
from matplotlib.lines import Line2D

from src.experiments.utkface.data import read_manifest, training_indices
from src.experiments.utkface.scheduler import charged_seconds, selected_result
from src.experiments.utkface.study import StudyPlan


def read(*, path: Path) -> dict:
    return json.loads(path.read_text())


def label(*, run: dict) -> str:
    names = {
        "tailrl_population": "TailRL population",
        "mse": "Direct MSE",
        "mae": "Direct MAE",
        "ce": "CE",
        "stat_maxrl_mae_population": "Stat MaxRL MAE",
        "stat_maxrl_mse_population": "Stat MaxRL MSE",
        "grpo_mae_population": "GRPO MAE",
        "grpo_mse_population": "GRPO MSE",
    }
    name = names[run["method"]]
    if run["method"].startswith("stat_"):
        name += f", width {run['width']:g}"
    return name + (" scalar" if run["head"] == "scalar" else "")


def compatible(*, first: dict, second: dict) -> None:
    for key in (
        "cohort",
        "fingerprint",
        "trainingFingerprint",
        "size",
        "epochs",
        "seed",
        "subsetSeed",
        "width",
    ):
        if first[key] != second[key]:
            raise ValueError(f"Incompatible {key}: {first['id']} / {second['id']}")


def audit(
    *,
    plan_path: Path,
    report_dir: Path,
    coordination_path: Path,
    archived_extension_plan: Path,
    data: dict,
) -> dict:
    plan = StudyPlan.model_validate_json(plan_path.read_text())
    state = read(path=plan.output_dir / "state.json")
    control = read(path=coordination_path)
    if control["max_concurrent"] != 8:
        raise ValueError("Study concurrency cap changed")
    expected = "1ecd0d0a6a1cae81ad1c8e2567ce226dbc47d5d7577ab027f97cb4ba08345fb6"
    if hashlib.sha256(plan_path.read_bytes()).hexdigest() != expected:
        raise ValueError("Original study plan changed")
    for path, digest in plan.training_source_sha256.items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Frozen training source changed: {path}")
    manifest = read_manifest(path=plan.jobs[0].config.manifest)
    if manifest["fingerprint"] != plan.data_fingerprint:
        raise ValueError("Data split changed")
    subsets = {
        size: training_indices(payload=manifest, limit=size, subset_seed=2026)
        for size in (0, 256, 1024, 4096)
    }
    if not set(subsets[1024]) <= set(subsets[4096]) <= set(subsets[0]):
        raise ValueError("Training subsets are not nested")
    fingerprints = {
        size: hashlib.sha256(json.dumps(indices).encode()).hexdigest()
        for size, indices in subsets.items()
    }
    records = json.loads((report_dir / "results.json").read_text())
    reported = {run["id"]: run for run in data["runs"]}
    cohort = hashlib.sha256(
        json.dumps(plan.training_source_sha256, sort_keys=True).encode()
    ).hexdigest()
    decisions, noise_fields = [], {}
    stages = Counter()
    for record in records:
        job, result = record["study_job"], record["result"]
        config = result["config"]
        history = json.loads((Path(config["output_dir"]) / "history.json").read_text())
        assert len(history) == config["epochs"]
        assert [h["epoch"] for h in history] == list(range(1, config["epochs"] + 1))
        best = min(history, key=lambda h: h["validation"]["mae_mean"])
        assert result["best_epoch"] == best["epoch"]
        assert result["validation"] == best["validation"]
        assert result["data_fingerprint"] == plan.data_fingerprint
        run = reported[job["identifier"]]
        assert run["cohort"] == cohort and run["fingerprint"] == plan.data_fingerprint
        assert run["bestEpoch"] == best["epoch"]
        for phase, values in (
            ("best", result["validation"]),
            ("final", history[-1]["validation"]),
            ("test", result.get("test", {})),
        ):
            assert all(value == values.get(key) for key, value in run[phase].items())
        assert config["subset_seed"] == 2026 and config["precision"] == "bf16"
        assert result["training_fingerprint"] == fingerprints[config["train_limit"]]
        assert result["train_unique_images"] == len(subsets[config["train_limit"]])
        for key, value in job["config"].items():
            if key not in {"output_dir", "learning_rate", "device"}:
                assert config[key] == value, (job["identifier"], key)
        for h in history:
            assert math.isfinite(h["train_loss"]) and math.isfinite(h["grad_norm"])
            assert h["validation"]["count"] == (config["eval_limit"] or 2308)
            batch = result["global_batch_size"]
            assert h["train_count"] == result["train_unique_images"] // batch * batch
        assert ("test" in result) == config["evaluate_test"]
        if config["evaluate_test"]:
            assert result["test"]["count"] == (config["eval_limit"] or 2308)
        if config["head"] == "scalar":
            assert result["validation"]["label_nll"] is None
            assert result.get("test", {}).get("label_nll") is None
        if job["tune_from"]:
            assert all(
                state["jobs"][j]["status"] == "completed" for j in job["tune_from"]
            )
            winner = selected_result(
                identifiers=tuple(job["tune_from"]),
                state=state,
                rank=job["tuning_rank"],
            )
            assert config["learning_rate"] == winner["config"]["learning_rate"]
            assert "test" not in winner
            decisions.append({
                "id": job["identifier"],
                "candidates": job["tune_from"],
                "rank": job["tuning_rank"],
                "selected_lr": config["learning_rate"],
                "selected_validation_mae": winner["validation"]["mae_mean"],
                "selected_validation_epoch": winner["best_epoch"],
            })
        else:
            assert config["learning_rate"] == job["config"]["learning_rate"]
        if config["noise"]["mode"] != "none":
            noise = config["noise"]
            key = (
                config["train_limit"],
                noise["mode"],
                noise["std_years"],
                noise["seed"],
                config["epochs"],
            )
            measurements = tuple(h["label_noise"]["noise_mse_years2"] for h in history)
            if key in noise_fields:
                assert noise_fields[key] == measurements
            noise_fields[key] = measurements
        stages[job["stage"]] += 1
    now = time.time()
    hours = charged_seconds(state=state, now=now) / 3600
    if control["enforce_gpu_hour_budget"]:
        assert hours <= plan.budget_gpu_hours
    assert not any(r["status"] == "failed" for r in state["jobs"].values())
    extension_root = archived_extension_plan.parent
    assert (
        hashlib.sha256((extension_root / "plan.json").read_bytes()).hexdigest()
        == "9b28c9dee4fff9c6158952f10813e8bd3c7abf06445e5c0a8015d4d74c273619"
    )
    extension = read(path=extension_root / "state.json")
    assert extension["status"] == "deadline_reached"
    assert all(
        r["status"] == "skipped" and not r["attempts"]
        for r in extension["jobs"].values()
    )
    before = read(path=extension_root / "original_state_before_handoff_1791920738.json")
    preserved = 0
    for identifier, record in before["jobs"].items():
        current = state["jobs"][identifier]
        assert len(current["attempts"]) >= len(record["attempts"])
        for first, second in zip(record["attempts"], current["attempts"]):
            for key in ("pid", "started_at", "config_path", "output_dir"):
                assert first[key] == second[key]
            if "seconds" in first:
                assert first["seconds"] == second["seconds"]
            preserved += 1
    events = []
    for record in state["jobs"].values():
        for attempt in record["attempts"]:
            end = attempt.get(
                "ended_at",
                attempt["started_at"] + attempt["seconds"]
                if "seconds" in attempt
                else now,
            )
            events.extend([(attempt["started_at"], 1), (end, -1)])
    active = peak = 0
    for _, delta in sorted(events):
        active += delta
        peak = max(peak, active)
    assert peak <= control["max_concurrent"]
    return {
        "audit_time": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "completed_records": len(records),
        "completed_stages": dict(stages),
        "frozen_training_sources_verified": True,
        "plan_unchanged": True,
        "data_fingerprint": plan.data_fingerprint,
        "source_cohort": cohort,
        "reported_phase_values_verified_against_raw_results": True,
        "training_fingerprints_by_limit": fingerprints,
        "nested_subsets_verified": True,
        "epoch_counts_and_finite_training_verified": True,
        "validation_only_lr_and_checkpoint_selection_verified": True,
        "clean_heldout_counts_verified_by_frozen_evaluation_path": True,
        "shared_noise_field_measurements_verified": True,
        "scalar_label_nll_undefined": True,
        "gpu_hours": hours,
        "budget_gpu_hours": plan.budget_gpu_hours,
        "gpu_hour_cap_enforced": control["enforce_gpu_hour_budget"],
        "max_concurrent": control["max_concurrent"],
        "attempts": sum(len(r["attempts"]) for r in state["jobs"].values()),
        "historical_peak_gpu_leases": peak,
        "before_handoff_attempts_preserved": preserved,
        "extension_remains_closed": True,
        "tuning_decisions": decisions,
    }


def compare(*, data: dict, tie_years: float) -> dict:
    runs = [
        r
        for r in data["runs"]
        if r["status"] == "completed" and r["seed"] == 42 and r["size"] == 4096
    ]
    clean = [r for r in runs if r["stage"] == "subset"]

    def baseline(run):
        matches = [
            r
            for r in clean
            if (r["method"], r["head"], r["width"])
            == (
                run["method"],
                "categorical" if run["head"] == "digits" else run["head"],
                run["width"],
            )
        ]
        assert len(matches) == 1
        compatible(first=run, second=matches[0])
        return matches[0]

    def outcome(delta):
        return (
            "better"
            if delta < -tie_years
            else "worse"
            if delta > tie_years
            else "practical tie"
        )

    pairs = {"schedule": [], "digits": [], "noise": []}
    for run in runs:
        if run["stage"] not in {"flat", "digits", "noise"}:
            continue
        base = baseline(run)
        delta = run["test"]["mae_mean"] - base["test"]["mae_mean"]
        kind = {"flat": "schedule", "digits": "digits", "noise": "noise"}[run["stage"]]
        pairs[kind].append({
            "id": run["id"],
            "baseline_id": base["id"],
            "label": label(run=run),
            "test_mae_delta": delta,
            "outcome": outcome(delta),
            "final_validation_mae_delta": run["final"]["mae_mean"]
            - base["final"]["mae_mean"],
        })
    return {
        "snapshot": data["generated"],
        "optimization_seed": 42,
        "training_size": 4096,
        "tie_threshold_years": tie_years,
        "pairs": pairs,
    }


def figures(*, data: dict, comparisons: dict, output_dir: Path) -> None:
    by_id = {r["id"]: r for r in data["runs"]}
    column = data["curveColumns"].index("mae_mean")

    def curve(ax, run, color, style, name):
        ax.plot(
            [c[0] for c in run["curves"]],
            [c[column] for c in run["curves"]],
            style,
            color=color,
            label=name,
            lw=1.4,
        )
        best = run["curves"][run["bestEpoch"] - 1]
        ax.scatter(best[0], best[column], color=color, s=24, zorder=4)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Validation mean-age MAE (years)")
        ax.grid(alpha=0.2)

    def save(fig, name):
        for suffix in ("png", "pdf"):
            fig.savefig(output_dir / f"{name}.{suffix}", dpi=180)
        plt.close(fig)

    clean = [
        r
        for r in data["runs"]
        if r["stage"] == "subset"
        and r["status"] == "completed"
        and r["seed"] == 42
        and r["size"] == 4096
    ]
    colors = {r["id"]: plt.get_cmap("tab10")(i) for i, r in enumerate(clean)}
    if clean:
        fig, axes = plt.subplots(1, 2, figsize=(16, 6), layout="constrained")
        for r in clean:
            curve(
                axes[0], r, colors[r["id"]], "-", f"{label(run=r)} (e{r['bestEpoch']})"
            )
            axes[1].scatter(
                r["test"]["mae_mean"], label(run=r), color=colors[r["id"]], s=65
            )
            axes[1].annotate(
                f" {r['test']['mae_mean']:.3f}",
                (r["test"]["mae_mean"], label(run=r)),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=9,
            )
        axes[0].legend(fontsize=8)
        axes[1].set_xlabel("Clean test MAE at selected checkpoint (years)")
        axes[1].margins(x=0.15)
        axes[1].grid(axis="x", alpha=0.2)
        fig.suptitle(
            "Fixed 4096-image subset, 50 epochs; one optimization seed (42)\nDots on curves identify validation-MAE-selected checkpoints"
        )
        save(fig, "subset")
    if comparisons["pairs"]["schedule"]:
        fig, axes = plt.subplots(
            1, 2, figsize=(16, 5), sharey=True, layout="constrained"
        )
        for p in comparisons["pairs"]["schedule"]:
            r, base = by_id[p["id"]], by_id[p["baseline_id"]]
            for ax, phase in zip(axes, ("test", "final"), strict=True):
                ax.plot(
                    [base[phase]["mae_mean"], r[phase]["mae_mean"]],
                    [p["label"]] * 2,
                    color="0.7",
                    lw=1,
                )
                ax.scatter(
                    base[phase]["mae_mean"],
                    p["label"],
                    color="#2467B5",
                    marker="o",
                    s=55,
                )
                ax.scatter(
                    r[phase]["mae_mean"], p["label"], color="#BD4D26", marker="s", s=55
                )
        axes[0].set_xlabel("Clean test MAE at selected checkpoint (years)")
        axes[1].set_xlabel("Final-epoch validation MAE (years)")
        for ax in axes:
            ax.grid(axis="x", alpha=0.2)
        fig.suptitle(
            "Schedule comparison: cosine + warmup (blue circles) / constant (orange squares)\nFixed 4096 images, 50 epochs, seed 42; independently validation-tuned learning rates"
        )
        save(fig, "schedule")
    noise = comparisons["pairs"]["noise"]
    if noise:
        bases = list(dict.fromkeys(p["baseline_id"] for p in noise))
        fig, axes = plt.subplots(
            math.ceil(len(bases) / 3),
            3,
            figsize=(18, 4 * math.ceil(len(bases) / 3)),
            squeeze=False,
            layout="constrained",
        )
        for ax, identifier in zip(axes.flat, bases):
            base = by_id[identifier]
            curve(
                ax,
                base,
                "0.35",
                "--",
                f"Clean: test {base['test']['mae_mean']:.3f}, e{base['bestEpoch']}",
            )
            for p in noise:
                r = by_id[p["id"]]
                if p["baseline_id"] == identifier and r["std"] == 5:
                    curve(
                        ax,
                        r,
                        "#BD4D26" if r["noise"] == "fixed" else "#2467B5",
                        "-",
                        f"{r['noise']}: test {r['test']['mae_mean']:.3f}, e{r['bestEpoch']}",
                    )
            ax.set_title(label(run=base))
            ax.legend(fontsize=8)
        for ax in list(axes.flat)[len(bases) :]:
            ax.set_visible(False)
        fig.suptitle(
            "Training-label Gaussian SD 5: fixed/fresh perturbations, clean validation/test\n4096 images, 50 epochs, optimization seed 42 and noise seed 2026; selected checkpoints marked"
        )
        save(fig, "noise")
        fig, axes = plt.subplots(
            1, 2, figsize=(14, 5), sharey=True, layout="constrained"
        )
        for p in noise:
            r = by_id[p["id"]]
            ax = axes[0 if r["noise"] == "fixed" else 1]
            ax.scatter(
                r["snr"], p["test_mae_delta"], color=colors[p["baseline_id"]], s=65
            )
        for ax, mode in zip(axes, ("fixed", "fresh"), strict=True):
            ax.set_title(mode)
            ax.set_xlabel("Realized label-noise SNR (dB)")
            ax.axhline(0, color="0.35", ls="--", lw=1)
            ax.grid(alpha=0.2)
        axes[0].set_ylabel("Test MAE minus matched clean baseline (years)")
        axes[1].legend(
            handles=[
                Line2D(
                    [],
                    [],
                    marker="o",
                    ls="",
                    color=colors[b],
                    label=label(run=by_id[b]),
                )
                for b in bases
            ],
            fontsize=8,
        )
        fig.suptitle(
            "Observed noise settings only; one training seed and one noise field\nClean-reference delta is zero; infinite clean-label SNR is not plotted"
        )
        save(fig, "snr")
    digits = comparisons["pairs"]["digits"]
    if digits:
        fig, axes = plt.subplots(
            1,
            len(digits),
            figsize=(6 * len(digits), 5),
            squeeze=False,
            layout="constrained",
        )
        for ax, p in zip(axes.flat, digits, strict=True):
            r, base = by_id[p["id"]], by_id[p["baseline_id"]]
            curve(
                ax,
                base,
                "#2467B5",
                "--",
                f"Flat: test MAE {base['test']['mae_mean']:.3f}, e{base['bestEpoch']}",
            )
            curve(
                ax,
                r,
                "#BD4D26",
                "-",
                f"Digits: test MAE {r['test']['mae_mean']:.3f}, e{r['bestEpoch']}",
            )
            ax.set_title(p["label"])
            ax.text(
                0.04,
                0.08,
                f"Test label NLL: {base['test']['label_nll']:.3f} → {r['test']['label_nll']:.3f}",
                transform=ax.transAxes,
                fontsize=9,
            )
            ax.legend(fontsize=8)
        fig.suptitle(
            "Flat 100-bin versus conditional 10×10 policy; one matched seed (42)\nFixed 4096 images, 50 epochs; heads and LR candidate sets differ"
        )
        save(fig, "digits")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--coordination", type=Path, required=True)
    parser.add_argument("--archived-extension-plan", type=Path, required=True)
    parser.add_argument("--tie-years", type=float, required=True)
    args = parser.parse_args()
    data = read(path=args.report_dir / "dashboard-data.json")
    integrity = audit(
        plan_path=args.plan,
        report_dir=args.report_dir,
        coordination_path=args.coordination,
        archived_extension_plan=args.archived_extension_plan,
        data=data,
    )
    assert integrity["completed_records"] == sum(
        r["status"] == "completed" and r["batch"] == "utkface-study-20261013"
        for r in data["runs"]
    )
    comparisons = compare(data=data, tie_years=args.tie_years)
    integrity["report_snapshot"] = data["generated"]
    (args.report_dir / "progress_integrity.json").write_text(
        json.dumps(integrity, indent=2) + "\n"
    )
    (args.report_dir / "followup-comparisons.json").write_text(
        json.dumps(comparisons, indent=2) + "\n"
    )
    rows = [
        {
            **{
                k: r[k]
                for k in (
                    "id",
                    "stage",
                    "method",
                    "head",
                    "size",
                    "seed",
                    "epochs",
                    "lr",
                    "bestEpoch",
                    "noise",
                    "std",
                    "noiseSeed",
                    "snr",
                )
            },
            **{
                f"{phase}_{metric}": r[phase][metric]
                for phase in ("best", "final", "test")
                for metric in ("mae_mean", "rmse_mean", "median_spearman", "label_nll")
            },
        }
        for r in data["runs"]
        if r["status"] == "completed"
        and r["stage"] in {"subset", "flat", "noise", "digits"}
    ]
    if rows:
        pl.DataFrame(rows).write_csv(args.report_dir / "followup-metrics.csv")
    figures(data=data, comparisons=comparisons, output_dir=args.report_dir)
    print(json.dumps({k: v for k, v in integrity.items() if k != "tuning_decisions"}))
    print(json.dumps(comparisons))


if __name__ == "__main__":
    main()
