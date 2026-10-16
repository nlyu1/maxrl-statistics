import argparse
import hashlib
import json
import statistics
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from src.experiments.utkface.coordination import CoordinationConfig
from src.experiments.utkface.progress_report import audit, read
from src.experiments.utkface.scheduler import charged_seconds, selected_result
from src.experiments.utkface.study import REDUCED_VARIANTS, StudyPlan
from src.experiments.utkface.study_report import FINAL_STAGES

METRICS = ("mae_mean", "rmse_mean", "median_spearman", "label_nll")
METHODS = tuple(method for method, _ in REDUCED_VARIANTS)
SCIENTIFIC_STAGES = FINAL_STAGES | {"subset_replication", "matched_steps"}


def name(*, run: dict) -> str:
    method = run["method"]
    stem = method.removesuffix("_population")
    names = {
        "tailrl": "TailRL",
        "binary_maxrl": "Binary MaxRL / CE",
        "mae": "Direct MAE",
        "mse": "Direct MSE",
        "ce": "CE",
        "stat_maxrl_mae": "Stat MAE",
        "stat_maxrl_mse": "Stat MSE",
        "grpo_mae": "GRPO MAE",
        "grpo_mse": "GRPO MSE",
        "rloo_mae": "RLOO MAE",
        "rloo_mse": "RLOO MSE",
    }
    result = names[stem]
    if stem.startswith("stat_"):
        result += f", width {run['width']:g}"
    if stem not in {"mae", "mse", "ce"}:
        result += " (population)" if method.endswith("_population") else " (64 samples)"
    return result + (" · scalar" if run["head"] == "scalar" else "")


def select(
    *, runs: list[dict], stages: tuple[str, ...], size: int, method: str, head: str
) -> list[dict]:
    result = [
        r
        for r in runs
        if r["stage"] in stages
        and r["size"] == size
        and r["method"] == method
        and r["head"] == head
        and (not method.startswith("stat_") or r["width"] == 2)
    ]
    return sorted(result, key=lambda r: r["seed"])


def one(
    *, runs: list[dict], stages: tuple[str, ...], size: int, method: str, head: str
) -> dict:
    matches = [
        r
        for r in select(runs=runs, stages=stages, size=size, method=method, head=head)
        if r["seed"] == 42
    ]
    if len(matches) != 1:
        raise ValueError((stages, size, method, head, len(matches)))
    return matches[0]


def mean_sd(*, values: list) -> dict:
    defined = [v for v in values if v is not None]
    return {
        "mean": statistics.mean(defined) if defined else None,
        "sd": statistics.stdev(defined) if len(defined) > 1 else None,
        "count": len(defined),
    }


def paired(*, kind: str, first: list[dict], second: list[dict]) -> dict:
    if not first or [r["seed"] for r in first] != [r["seed"] for r in second]:
        raise ValueError("Paired comparisons require identical complete seed sets")
    for a, b in zip(first, second, strict=True):
        for key in (
            "cohort",
            "fingerprint",
            "subsetSeed",
            "seed",
            "size",
            "trainingFingerprint",
        ):
            if a[key] != b[key]:
                raise ValueError((kind, key, a["id"], b["id"]))
        if kind != "matched_updates" and a["epochs"] != b["epochs"]:
            raise ValueError("Training horizons differ")
        if kind == "noise" and a["lr"] != b["lr"]:
            raise ValueError("Noise comparisons must reuse the clean learning rate")
    values = [
        a["test"]["mae_mean"] - b["test"]["mae_mean"]
        for a, b in zip(first, second, strict=True)
    ]
    return {
        "kind": kind,
        "label": name(run=first[0]),
        "first_ids": [r["id"] for r in first],
        "second_ids": [r["id"] for r in second],
        "seeds": [r["seed"] for r in first],
        "test_mae_deltas": values,
        "test_mae_delta": mean_sd(values=values),
        "best_validation_mae_delta": mean_sd(
            values=[
                a["best"]["mae_mean"] - b["best"]["mae_mean"]
                for a, b in zip(first, second, strict=True)
            ]
        ),
        "final_validation_mae_delta": mean_sd(
            values=[
                a["final"]["mae_mean"] - b["final"]["mae_mean"]
                for a, b in zip(first, second, strict=True)
            ]
        ),
    }


def summarize(*, runs: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for r in runs:
        family = (
            "full"
            if r["stage"] in {"main", "scalar"}
            else "subset"
            if r["stage"] in {"subset", "subset_replication", "matched_steps"}
            else r["stage"]
        )
        keys = (
            family,
            r["method"],
            r["head"],
            r["width"],
            r["size"],
            r["epochs"],
            r["schedule"],
            r["noise"],
            r["std"],
            r["noiseSeed"],
            r["subsetSeed"],
            r["cohort"],
            r["fingerprint"],
            r["trainingFingerprint"],
        )
        groups[keys].append(r)
    summaries = []
    for key, values in groups.items():
        values.sort(key=lambda r: r["seed"])
        if (
            len({r["seed"] for r in values}) != len(values)
            or len({r["lr"] for r in values}) != 1
        ):
            raise ValueError("Aggregation mixes duplicate seeds or learning rates")
        first = values[0]
        summaries.append({
            **{
                k: first[k]
                for k in (
                    "method",
                    "head",
                    "width",
                    "size",
                    "epochs",
                    "schedule",
                    "noise",
                    "std",
                    "noiseSeed",
                    "subsetSeed",
                    "cohort",
                    "fingerprint",
                    "trainingFingerprint",
                )
            },
            "family": key[0],
            "label": name(run=first),
            "n": len(values),
            "seeds": [r["seed"] for r in values],
            "best_epochs": [r["bestEpoch"] for r in values],
            "ids": [r["id"] for r in values],
            "lr": first["lr"],
            **{
                phase: {
                    metric: mean_sd(values=[r[phase][metric] for r in values])
                    for metric in METRICS
                }
                for phase in ("best", "final", "test")
            },
        })
    return summaries


def completion_audit(
    *, control: CoordinationConfig, original: dict, data: dict
) -> dict:
    plans = [StudyPlan.model_validate_json(p.read_text()) for p in control.peer_plans]
    assert len(plans) == 2
    states = [read(path=p.output_dir / "state.json") for p in plans]
    combined = {"jobs": {k: v for s in states for k, v in s["jobs"].items()}}
    by_id = {r["id"]: r for r in data["runs"]}
    decisions = []
    for plan, state in zip(plans, states, strict=True):
        assert state["status"] == "completed"
        assert plan.training_source_sha256 == plans[0].training_source_sha256
        assert plan.data_fingerprint == plans[0].data_fingerprint
        for job in plan.jobs:
            record = state["jobs"][job.identifier]
            if job.identifier in control.held_jobs:
                assert record["status"] == "pending" and not record["attempts"]
                continue
            assert record["status"] == "completed"
        if plan == plans[0]:
            continue
        for job in plan.jobs:
            record = state["jobs"][job.identifier]
            result = read(path=Path(record["result_path"]))
            config = result["config"]
            history = json.loads(
                (Path(config["output_dir"]) / "history.json").read_text()
            )
            assert len(history) == config["epochs"]
            assert [h["epoch"] for h in history] == list(range(1, config["epochs"] + 1))
            best = min(history, key=lambda h: h["validation"]["mae_mean"])
            assert (
                result["best_epoch"] == best["epoch"]
                and result["validation"] == best["validation"]
            )
            assert result["data_fingerprint"] == plan.data_fingerprint
            assert (
                result["training_fingerprint"]
                == original["training_fingerprints_by_limit"][config["train_limit"]]
            )
            for key, value in job.config.model_dump(mode="json").items():
                if key not in {"output_dir", "learning_rate", "device"}:
                    assert config[key] == value
            assert config["noise"]["mode"] == "none" and config["precision"] == "bf16"
            assert result["test"]["count"] == 2308
            for h in history:
                assert (
                    h["train_count"] == config["train_limit"]
                    and h["validation"]["count"] == 2308
                )
                assert np.isfinite(h["train_loss"]) and np.isfinite(h["grad_norm"])
            for phase, values in (
                ("best", best["validation"]),
                ("final", history[-1]["validation"]),
                ("test", result["test"]),
            ):
                assert all(
                    value == values.get(key)
                    for key, value in by_id[job.identifier][phase].items()
                )
            winner = selected_result(
                identifiers=job.tune_from, state=combined, rank=job.tuning_rank
            )
            assert (
                config["learning_rate"] == winner["config"]["learning_rate"]
                and "test" not in winner
            )
            decisions.append({
                "id": job.identifier,
                "lr": config["learning_rate"],
                "validation_mae": winner["validation"]["mae_mean"],
                "tuning_ids": list(job.tune_from),
            })
    activation = read(path=plans[1].output_dir / "requeue-audit.json")
    before = read(path=plans[1].output_dir / "original_state_before_update.json")
    for identifier, record in before["jobs"].items():
        assert (
            states[0]["jobs"][identifier]["attempts"][: len(record["attempts"])]
            == record["attempts"]
        )
    assert (
        hashlib.sha256(control.peer_plans[0].read_bytes()).hexdigest()
        == activation["original_plan_sha256"]
    )
    assert Counter(j.stage for j in plans[1].jobs) == {
        "subset_replication": 16,
        "matched_steps": 8,
    }
    attempts = [a for s in states for r in s["jobs"].values() for a in r["attempts"]]
    events = sorted(
        (t, delta)
        for a in attempts
        for t, delta in ((a["started_at"], 1), (a["ended_at"], -1))
    )
    active = peak = 0
    for _, delta in events:
        active += delta
        peak = max(peak, active)
    assert peak <= control.max_concurrent and active == 0
    admitted = [a for a in attempts if a.get("gpu_admission")]
    claims = [a for a in admitted if a["gpu_admission"] == "two_idle_checks"]
    assert all(a["idle_observed_seconds"] >= 10 for a in claims)
    reuses = [a for a in admitted if a["gpu_admission"] == "reuse"]
    gaps = [
        a["started_at"]
        - max(
            b["ended_at"]
            for b in attempts
            if b["gpu_uuid"] == a["gpu_uuid"] and b["ended_at"] <= a["started_at"]
        )
        for a in reuses
    ]
    interrupted = [a for a in attempts if a.get("yielded_to_foreign_process")]
    return {
        **original,
        "audit_time": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "completed_records": sum(
            r["status"] == "completed" for s in states for r in s["jobs"].values()
        ),
        "completed_test_evaluations": sum(
            r["status"] == "completed"
            and r["test"]["mae_mean"] is not None
            and r["stage"] in SCIENTIFIC_STAGES
            for r in data["runs"]
        ),
        "preflight_test_evaluations": sum(
            r["stage"] == "smoke" and r["status"] == "completed" for r in data["runs"]
        ),
        "held_records": len(control.held_jobs),
        "excluded_noise_realizations": 32,
        "total_gpu_hours": sum(charged_seconds(state=s, now=0) / 3600 for s in states),
        "total_attempts": len(attempts),
        "recorded_preemptions": len(interrupted),
        "preempted_gpu_hours": sum(a["seconds"] for a in interrupted) / 3600,
        "combined_peak_gpu_leases": peak,
        "post_update_attempts": len(admitted),
        "successful_gpu_reuses": len(reuses),
        "new_gpu_claims": len(claims),
        "median_reuse_gap_seconds": statistics.median(gaps),
        "minimum_new_claim_idle_seconds": min(
            a["idle_observed_seconds"] for a in claims
        ),
        "finished_at": datetime.fromtimestamp(
            max(a["ended_at"] for a in attempts), ZoneInfo("Asia/Shanghai")
        ).isoformat(),
        "original_attempts_before_update_preserved": activation[
            "original_attempts_preserved"
        ],
        "continuation_training_and_report_values_verified": True,
        "continuation_validation_only_tuning_decisions": decisions,
        "selected_work_complete": True,
        "report_snapshot": data["generated"],
    }


def comparisons(*, runs: list[dict]) -> dict:
    pairs = defaultdict(list)
    for method in ("mae", "mse"):
        scalar = select(
            runs=runs, stages=("scalar",), size=18462, method=method, head="scalar"
        )
        categorical = select(
            runs=runs, stages=("main",), size=18462, method=method, head="categorical"
        )
        pairs["scalar"].append(paired(kind="scalar", first=scalar, second=categorical))
    scalar = select(
        runs=runs, stages=("scalar",), size=18462, method="mae", head="scalar"
    )
    stat = select(
        runs=runs,
        stages=("main",),
        size=18462,
        method="stat_maxrl_mae_population",
        head="categorical",
    )
    pairs["scalar_vs_stat_mae2"].append(
        paired(kind="scalar_vs_stat_mae2", first=scalar, second=stat)
    )
    for method in METHODS:
        short = one(
            runs=runs, stages=("subset",), size=1024, method=method, head="categorical"
        )
        long = one(
            runs=runs,
            stages=("matched_steps",),
            size=1024,
            method=method,
            head="categorical",
        )
        pairs["matched_updates"].append(
            paired(kind="matched_updates", first=[long], second=[short])
        )
    for r in runs:
        if r["stage"] not in {"flat", "noise", "digits"}:
            continue
        method = (
            "binary_maxrl_population"
            if r["size"] == 18462 and r["method"] == "ce"
            else r["method"]
        )
        base = one(
            runs=runs,
            stages=("main",) if r["size"] == 18462 else ("subset",),
            size=r["size"],
            method=method,
            head="categorical" if r["head"] == "digits" else r["head"],
        )
        kind = {"flat": "schedule", "noise": "noise", "digits": "digits"}[r["stage"]]
        pairs[kind].append(paired(kind=kind, first=[r], second=[base]))
    return dict(pairs)


def figures(*, runs: list[dict], pairs: dict, output_dir: Path) -> None:
    by_id = {r["id"]: r for r in runs}

    def save(*, fig, name):
        for suffix in ("png", "pdf"):
            fig.savefig(output_dir / f"{name}.{suffix}", dpi=170)
        plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), layout="constrained")
    for row, method in enumerate(("mae", "mse")):
        for head, color in (("categorical", "#2467B5"), ("scalar", "#BD4D26")):
            selected = select(
                runs=runs,
                stages=("main", "scalar"),
                size=18462,
                method=method,
                head=head,
            )
            values = np.array([[c[1] for c in r["curves"]] for r in selected])
            mean, sd = values.mean(0), values.std(0, ddof=1)
            axes[row, 0].plot(range(1, 51), mean, color=color, label=f"{head}, n=3")
            axes[row, 0].fill_between(
                range(1, 51), mean - sd, mean + sd, color=color, alpha=0.15
            )
            for r in selected:
                axes[row, 1].scatter(
                    r["seed"],
                    r["test"]["mae_mean"],
                    color=color,
                    marker="o" if head == "categorical" else "s",
                )
        axes[row, 0].set_title(f"Direct {method.upper()}: validation curves")
        axes[row, 0].set_xlabel("Epoch")
        axes[row, 0].set_ylabel("Validation MAE, years")
        axes[row, 0].legend()
        axes[row, 1].set_title("Selected-checkpoint clean test MAE")
        axes[row, 1].set_xticks((42, 43, 44))
        axes[row, 1].set_xlabel("Optimization seed")
        for ax in axes[row]:
            ax.grid(alpha=0.2)
    fig.suptitle(
        "Scalar versus categorical-mean heads · full data, 50 epochs\nSame learning rate 0.001; curves show mean ± sample SD across three seeds"
    )
    save(fig=fig, name="final-scalar")

    fig, ax = plt.subplots(figsize=(13, 7), layout="constrained")
    for index, method in enumerate(METHODS):
        selected = select(
            runs=runs,
            stages=("subset", "subset_replication"),
            size=4096,
            method=method,
            head="categorical",
        )
        assert [r["seed"] for r in selected] == [42, 43, 44]
        values = [r["test"]["mae_mean"] for r in selected]
        ax.errorbar(
            statistics.mean(values),
            index,
            xerr=statistics.stdev(values),
            color="0.5",
            marker="D",
            capsize=4,
        )
        for offset, r, color in zip(
            (-0.13, 0, 0.13), selected, ("#2467B5", "#BD4D26", "#26936F"), strict=True
        ):
            ax.scatter(
                r["test"]["mae_mean"],
                index + offset,
                color=color,
                label=f"seed {r['seed']}" if index == 0 else None,
            )
    ax.set_yticks(
        range(len(METHODS)),
        [
            name(
                run=one(
                    runs=runs,
                    stages=("subset",),
                    size=4096,
                    method=m,
                    head="categorical",
                )
            )
            for m in METHODS
        ],
    )
    ax.invert_yaxis()
    ax.set_xlabel("Clean test MAE at validation-selected checkpoint, years")
    ax.grid(axis="x", alpha=0.2)
    ax.legend()
    fig.suptitle(
        "Fixed 4096 images × 50 epochs · three matched optimization seeds\nDiamonds and bars: mean ± sample SD; colored points: individual runs"
    )
    save(fig=fig, name="final-replication")

    fig, axes = plt.subplots(2, 4, figsize=(19, 9), layout="constrained")
    for ax, method in zip(axes.flat, METHODS, strict=True):
        selections = (
            (4096, ("subset",), "4096 × 50", "#2467B5"),
            (1024, ("subset",), "1024 × 50", "#26936F"),
            (1024, ("matched_steps",), "1024 × 200", "#BD4D26"),
        )
        for size, stages, tag, color in selections:
            r = one(
                runs=runs, stages=stages, size=size, method=method, head="categorical"
            )
            ax.plot(
                [c[-1] for c in r["curves"]],
                [c[1] for c in r["curves"]],
                color=color,
                label=f"{tag}, e{r['bestEpoch']}",
            )
            best = r["curves"][r["bestEpoch"] - 1]
            ax.scatter(best[-1], best[1], color=color, s=22)
        ax.set_title(name(run=r), fontsize=10)
        ax.set_xlabel("Optimizer updates")
        ax.set_ylabel("Validation MAE, years")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=7)
    fig.suptitle(
        "Fixed training subsets · matched seed 42 · dots identify MAE-selected checkpoints\n1024 × 200 and 4096 × 50 both use 3200 updates; cosine schedule and validation frequency also differ"
    )
    save(fig=fig, name="final-small-subsets")

    fig, axes = plt.subplots(1, 2, figsize=(17, 7), layout="constrained")
    for ax, size in zip(axes, (4096, 18462), strict=True):
        selected = [
            p for p in pairs["digits"] if by_id[p["first_ids"][0]]["size"] == size
        ]
        for index, pair in enumerate(selected):
            digit, flat = by_id[pair["first_ids"][0]], by_id[pair["second_ids"][0]]
            ax.plot(
                [flat["test"]["mae_mean"], digit["test"]["mae_mean"]],
                [index] * 2,
                color="0.65",
            )
            ax.scatter(
                flat["test"]["mae_mean"],
                index,
                color="#2467B5",
                label="Flat 100 bins" if index == 0 else None,
            )
            ax.scatter(
                digit["test"]["mae_mean"],
                index,
                color="#BD4D26",
                marker="s",
                label="Conditional 10×10" if index == 0 else None,
            )
        ax.set_yticks(range(len(selected)), [p["label"] for p in selected], fontsize=9)
        ax.invert_yaxis()
        ax.set_title(f"Training images: {size}")
        ax.set_xlabel("Clean test MAE, years")
        ax.grid(axis="x", alpha=0.2)
        ax.legend()
    fig.suptitle(
        "Categorical parameterization pilots · matched seed 42, 50 epochs\nHead capacity and LR tuning differ; full-data digit policies reuse rates tuned on 4096 images"
    )
    save(fig=fig, name="final-digits")


def tables(
    *, summaries: list[dict], pairs: dict, runs: list[dict], output_dir: Path
) -> None:
    def formatted(value):
        if value["mean"] is None:
            return "n/a"
        return f"{value['mean']:.3f}" + (
            f" ± {value['sd']:.3f}" if value["sd"] is not None else ""
        )

    lines = [
        "# Complete UTKFace result tables",
        "",
        "Test metrics evaluate the validation-MAE-selected checkpoint. Best/final columns refer to validation. ± denotes sample SD across optimization seeds on one fixed split, not confidence intervals. Scalar Spearman uses continuous predictions; categorical Spearman uses the policy median. Scalar label NLL is undefined.",
        "",
    ]
    for title, selected in (
        ("Full data, 50 epochs", [s for s in summaries if s["family"] == "full"]),
        (
            "Fixed subsets, clean labels",
            [s for s in summaries if s["family"] == "subset"],
        ),
    ):
        lines += [
            f"## {title}",
            "",
            "| Method | Images × epochs | Seeds | Best epochs | Best val MAE | Final val MAE | Test MAE | Test RMSE | Spearman | Label NLL |",
            "|---|---|---|---|---:|---:|---:|---:|---:|---:|",
        ]
        for s in sorted(
            selected,
            key=lambda s: (s["size"], s["epochs"], s["test"]["mae_mean"]["mean"]),
        ):
            cells = [
                s["label"],
                f"{s['size']} × {s['epochs']}",
                ",".join(map(str, s["seeds"])),
                ",".join(map(str, s["best_epochs"])),
                formatted(s["best"]["mae_mean"]),
                formatted(s["final"]["mae_mean"]),
                *[formatted(s["test"][m]) for m in METRICS],
            ]
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")
    by_id = {r["id"]: r for r in runs}
    for kind in ("schedule", "matched_updates", "digits", "scalar"):
        lines += [
            f"## Paired {kind}",
            "",
            "Differences are first minus reference; negative MAE differences favor the first setting. Every seed pair is listed in final-comparisons.json.",
            "",
            "| Method | First setting | Reference | Seeds | Test MAE difference | Final validation MAE difference | Best epochs, first / reference |",
            "|---|---|---|---|---:|---:|---|",
        ]
        for p in pairs[kind]:
            a, b = by_id[p["first_ids"][0]], by_id[p["second_ids"][0]]
            setting = lambda r: (
                f"{r['size']}×{r['epochs']}, {r['head']}, {r['schedule']}"
            )
            lines.append(
                "| "
                + " | ".join([
                    p["label"],
                    setting(a),
                    setting(b),
                    ",".join(map(str, p["seeds"])),
                    formatted(p["test_mae_delta"]),
                    formatted(p["final_validation_mae_delta"]),
                    ",".join(str(by_id[i]["bestEpoch"]) for i in p["first_ids"])
                    + " / "
                    + ",".join(str(by_id[i]["bestEpoch"]) for i in p["second_ids"]),
                ])
                + " |"
            )
        lines.append("")
    for mode in ("fixed", "fresh"):
        lines += [
            f"## {mode.capitalize()} training-label noise, seed 42 / field 2026",
            "",
            "Each cell shows clean-test MAE (selected epoch). All noise arms reuse the clean 4096-subset learning rate; held-out labels are clean.",
            "",
            "| Method | Clean | SD 1 | SD 3 | SD 5 | SD 10 |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        clean = [r for r in runs if r["stage"] == "subset" and r["size"] == 4096]
        for base in clean:
            selected = [
                next(
                    r
                    for r in runs
                    if r["stage"] == "noise"
                    and r["method"] == base["method"]
                    and r["head"] == base["head"]
                    and r["noise"] == mode
                    and r["std"] == std
                )
                for std in (1, 3, 5, 10)
            ]
            lines.append(
                "| "
                + " | ".join([
                    name(run=base),
                    *[
                        f"{r['test']['mae_mean']:.3f} (e{r['bestEpoch']})"
                        for r in [base, *selected]
                    ],
                ])
                + " |"
            )
        lines.append("")
    lines += [
        "## Measured noise SNR",
        "",
        "SNR uses clean-subset age variance divided by realized perturbation MSE after rounding/clipping. Fresh-noise MSE is averaged over epochs before conversion to dB.",
        "",
        "| SD, years | Fixed SNR, dB | Fresh SNR, dB |",
        "|---|---:|---:|",
    ]
    for std in (1, 3, 5, 10):
        values = [
            next(
                r["snr"]
                for r in runs
                if r["stage"] == "noise" and r["std"] == std and r["noise"] == mode
            )
            for mode in ("fixed", "fresh")
        ]
        lines.append(f"| {std} | {values[0]:.3f} | {values[1]:.3f} |")
    (output_dir / "FINAL_TABLES.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--coordination", type=Path, required=True)
    parser.add_argument("--archived-extension-plan", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    args = parser.parse_args()
    control = CoordinationConfig.model_validate_json(args.coordination.read_text())
    data = read(path=args.report_dir / "dashboard-data.json")
    original = audit(
        plan_path=control.peer_plans[0],
        report_dir=args.report_dir,
        coordination_path=args.coordination,
        archived_extension_plan=args.archived_extension_plan,
        data=data,
    )
    integrity = completion_audit(control=control, original=original, data=data)
    runs = [
        r
        for r in data["runs"]
        if r["status"] == "completed"
        and r["test"]["mae_mean"] is not None
        and r["stage"] in SCIENTIFIC_STAGES
    ]
    assert len(runs) == 222 and integrity["completed_records"] == 514
    summaries = summarize(runs=runs)
    pairs = comparisons(runs=runs)
    for filename, payload in (
        ("final-audit.json", integrity),
        ("final-comparisons.json", pairs),
        ("final-summaries.json", summaries),
    ):
        (args.report_dir / filename).write_text(json.dumps(payload, indent=2) + "\n")
    flat = [
        {
            **{k: v for k, v in s.items() if k not in {"best", "final", "test"}},
            **{
                f"{phase}_{metric}_{stat}": value
                for phase in ("best", "final", "test")
                for metric, stats in s[phase].items()
                for stat, value in stats.items()
            },
        }
        for s in summaries
    ]
    for row in flat:
        for key in ("seeds", "best_epochs", "ids"):
            row[key] = json.dumps(row[key])
    pl.DataFrame(flat, infer_schema_length=None).write_csv(
        args.report_dir / "final-summaries.csv"
    )
    tables(summaries=summaries, pairs=pairs, runs=runs, output_dir=args.report_dir)
    figures(runs=runs, pairs=pairs, output_dir=args.report_dir)
    print(
        json.dumps(
            {
                k: v
                for k, v in integrity.items()
                if k
                not in {
                    "tuning_decisions",
                    "continuation_validation_only_tuning_decisions",
                }
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
