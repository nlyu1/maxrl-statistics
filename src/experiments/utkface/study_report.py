import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import polars as pl

from src.experiments.utkface.run import write_json
from src.experiments.utkface.scheduler import status_summary
from src.experiments.utkface.study import StudyPlan

FINAL_STAGES = {
    "main",
    "scalar",
    "subset",
    "flat",
    "noise",
    "noise_replication",
    "digits",
}
GROUPS = ("stage", "variant", "train_limit", "lr_schedule", "noise_mode", "noise_std")
METRICS = (
    "mae_mean",
    "mae_median",
    "rmse_mean",
    "median_spearman",
    "label_nll",
    "expected_mae",
    "entropy",
)


def value(*, row: dict, metric: str) -> str:
    mean, sd = row[f"{metric}_mean"], row[f"{metric}_sd"]
    if mean is None:
        return "n/a"
    return f"{mean:.3f}" if sd is None else f"{mean:.3f} ± {sd:.3f}"


def report(*, plan_path: Path, output_dir: Path, figures: bool) -> None:
    plan = StudyPlan.model_validate_json(plan_path.read_text())
    state = json.loads((plan.output_dir / "state.json").read_text())
    output_dir.mkdir(parents=True, exist_ok=True)
    progress = status_summary(plan=plan, state=state)
    write_json(path=output_dir / "progress.json", payload=progress)
    records, results, curves = [], [], []
    for job in plan.jobs:
        record = state["jobs"][job.identifier]
        if record["status"] != "completed":
            continue
        result = json.loads(Path(record["result_path"]).read_text())
        config = result["config"]
        if result["data_fingerprint"] != plan.data_fingerprint:
            raise ValueError("Completed run uses a different data split")
        history = json.loads((Path(config["output_dir"]) / "history.json").read_text())
        if len(history) != config["epochs"]:
            raise ValueError("Completed run did not finish its planned epochs")
        noise_mse = sum(
            row["label_noise"]["noise_mse_years2"] for row in history
        ) / len(history)
        signal = history[0]["label_noise"]["signal_variance_years2"]
        row = {
            "identifier": job.identifier,
            "stage": job.stage,
            "variant": job.variant,
            "head": config["head"],
            "seed": config["seed"],
            "train_limit": config["train_limit"],
            "train_unique_images": result["train_unique_images"],
            "training_fingerprint": result["training_fingerprint"],
            "epochs": config["epochs"],
            "learning_rate": config["learning_rate"],
            "lr_schedule": config["schedule"]["lr_schedule"],
            "noise_mode": config["noise"]["mode"],
            "noise_std": config["noise"]["std_years"],
            "effective_noise_rmse": math.sqrt(noise_mse),
            "snr_db": 10 * math.log10(signal / noise_mse) if noise_mse else None,
            "best_epoch": result["best_epoch"],
            "validation_mae": result["validation"]["mae_mean"],
            "gpu_hours": sum(a["seconds"] for a in record["attempts"]) / 3600,
        }
        row.update({metric: result.get("test", {}).get(metric) for metric in METRICS})
        records.append(row)
        results.append({"study_job": job.model_dump(mode="json"), "result": result})
        curves.extend(
            {
                "identifier": job.identifier,
                "stage": job.stage,
                "variant": job.variant,
                "seed": config["seed"],
                "epoch": item["epoch"],
                "learning_rate": item["learning_rate"],
                "train_loss": item["train_loss"],
                "grad_norm": item["grad_norm"],
                "train_count": item["train_count"],
                "validation_mae": item["validation"]["mae_mean"],
                "validation_median_spearman": item["validation"]["median_spearman"],
                "validation_label_nll": item["validation"]["label_nll"],
                **item["label_noise"],
            }
            for item in history
        )
    lines = [
        "# Scaled UTKFace study",
        "",
        f"State: **{progress['status']}**. GPU time: **{progress['gpu_hours']:.2f} / {plan.budget_gpu_hours:g} hours**. Completed primary runs: **{sum(job.stage == 'main' and state['jobs'][job.identifier]['status'] == 'completed' for job in plan.jobs)} / 66**.",
        "",
        "Metrics use clean held-out labels. All tuning and scalar-head gates use validation data. Values are means ± sample SD across distinct training seeds; single-seed values have no SD. Label NLL is unavailable for scalar heads. Population zero-one MaxRL equals CE. Training subsets are fixed and nested.",
        "",
    ]
    if records:
        frame = pl.DataFrame(records).with_columns(
            pl.col(metric).cast(pl.Float64) for metric in METRICS
        )
        frame.write_csv(output_dir / "runs.csv")
        pl.DataFrame(curves).write_csv(output_dir / "curves.csv")
        write_json(path=output_dir / "results.json", payload=results)
        final = frame.filter(pl.col("stage").is_in(FINAL_STAGES))
        if len(final):
            checks = final.group_by(*GROUPS).agg(
                pl.len().alias("runs"),
                pl.col("seed").n_unique().alias("seeds"),
                pl.col("learning_rate").n_unique().alias("learning_rates"),
                pl.col("training_fingerprint").n_unique().alias("subsets"),
            )
            if checks.filter(
                (pl.col("runs") != pl.col("seeds"))
                | (pl.col("learning_rates") != 1)
                | (pl.col("subsets") != 1)
            ).height:
                raise ValueError(
                    "Result groups mix seeds, learning rates or training subsets"
                )
            summary = (
                final
                .group_by(*GROUPS)
                .agg(
                    pl.len().alias("seeds"),
                    *[
                        expression
                        for metric in METRICS
                        for expression in (
                            pl.col(metric).mean().alias(metric + "_mean"),
                            pl.col(metric).std().alias(metric + "_sd"),
                        )
                    ],
                )
                .sort("stage", "train_limit", "noise_std", "mae_mean_mean")
            )
            summary.write_csv(output_dir / "summary.csv")
            write_json(path=output_dir / "summary.json", payload=summary.to_dicts())
            lines += [
                "| Stage | Variant | Training size | Noise | Seeds | Mean MAE ↓ | Median MAE ↓ | Median Spearman ↑ | Label NLL ↓ |",
                "|---|---|---:|---|---:|---:|---:|---:|---:|",
            ]
            for row in summary.to_dicts():
                lines.append(
                    f"| {row['stage']} | {row['variant']} | {row['train_limit'] or 'full'} | {row['noise_mode']} {row['noise_std']:g} | {row['seeds']} | "
                    + " | ".join(
                        value(row=row, metric=metric)
                        for metric in (
                            "mae_mean",
                            "mae_median",
                            "median_spearman",
                            "label_nll",
                        )
                    )
                    + " |"
                )
            if figures:
                primary = summary.filter(
                    pl.col("stage").is_in(["main", "scalar"])
                ).sort("mae_mean_mean")
                if len(primary):
                    figure, axes = plt.subplots(
                        1,
                        3,
                        figsize=(18, max(7, len(primary) * 0.35)),
                        sharey=True,
                        layout="constrained",
                    )
                    for axis, metric, title in zip(
                        axes,
                        ("mae_mean", "median_spearman", "label_nll"),
                        (
                            "Mean prediction MAE (years; lower is better)",
                            "Median Spearman (higher is better)",
                            "Label NLL (nats; lower is better)",
                        ),
                        strict=True,
                    ):
                        axis.barh(
                            range(len(primary)),
                            primary[metric + "_mean"].to_numpy(),
                            xerr=primary[metric + "_sd"].fill_null(0).to_numpy(),
                            capsize=3,
                        )
                        axis.set_xlabel(title)
                        axis.grid(axis="x", alpha=0.2)
                    axes[0].set_yticks(
                        range(len(primary)), primary["variant"].to_list(), fontsize=8
                    )
                    axes[0].invert_yaxis()
                    figure.suptitle(
                        "Cleaned UTKFace, 50 epochs; mean ± sample SD across available seeds"
                    )
                    figure.savefig(output_dir / "main.png", dpi=180)
                    figure.savefig(output_dir / "main.pdf")
                    plt.close(figure)
                    lines += ["", "![Main comparison](main.png)"]
    lines += [
        "",
        "GPU time charges every launched attempt, including failures, retries, process startup, final test evaluation and polling overhead. Waiting for free GPUs consumes no GPU budget. Optional studies start after required studies finish. SNR is 10 log10(variance of clean subset ages / mean squared realized label perturbation), after rounding and clipping. Clean-label SNR is infinite and stored as null.",
        "",
        "[Machine-readable progress](progress.json).",
        "",
    ]
    if records:
        lines += [
            "[Per-run values and LR tuning](runs.csv). [Full configs and metrics](results.json). [Epoch curves and realized noise statistics](curves.csv).",
            "",
        ]
    (output_dir / "REPORT.md").write_text("\n".join(lines))
    print(json.dumps(progress, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--figures", action="store_true")
    args = parser.parse_args()
    report(plan_path=args.plan, output_dir=args.output_dir, figures=args.figures)


if __name__ == "__main__":
    main()
