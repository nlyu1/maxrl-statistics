import argparse
import asyncio
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import tinker
from transformers import AutoTokenizer

from src.experiments.pass_rate_calibration.config import (
    PILOT,
    PREDICTORS,
    SERIES_COLORS,
    PredictorModel,
)
from src.experiments.pass_rate_calibration.data import read_labels, read_problems
from src.experiments.pass_rate_calibration.ground_truth import markdown_table
from src.experiments.pass_rate_calibration.metrics import prediction_metrics
from src.experiments.pass_rate_calibration.predictor import (
    parse_completion,
    predictor_tokens,
    stop_tokens,
)
from src.experiments.utkface.run import write_json


def pilot_problems(*, root: Path) -> pl.DataFrame:
    train = (
        read_problems(root=root).filter(pl.col("split") == "train").sort("problem_id")
    )
    chosen = np.random.default_rng(PILOT.seed).permutation(train.height)[
        : PILOT.problems
    ]
    return (
        train
        .with_row_index("row")
        .filter(pl.col("row").is_in(chosen.tolist()))
        .drop("row")
    )


async def sample_problem(
    *, client: tinker.SamplingClient, tokens: list[int], stop: list[int]
) -> tuple[tinker.SampleResponse, float]:
    began = time.perf_counter()
    response = await client.sample(
        prompt=tinker.ModelInput.from_ints(tokens),
        num_samples=PILOT.samples,
        sampling_params=tinker.SamplingParams(
            max_tokens=PILOT.max_tokens,
            temperature=PILOT.temperature,
            stop=stop,
        ),
    )
    return response, time.perf_counter() - began


async def sample_all(
    *, client: tinker.SamplingClient, prompts: list[list[int]], stop: list[int]
) -> list[tuple[tinker.SampleResponse, float]]:
    return await asyncio.gather(
        *(sample_problem(client=client, tokens=tokens, stop=stop) for tokens in prompts)
    )


def run_model(*, root: Path, model: PredictorModel) -> None:
    problems = pilot_problems(root=root)
    tokenizer = AutoTokenizer.from_pretrained(model.repository, revision=model.revision)
    client = (
        tinker
        .ServiceClient()
        .create_sampling_client(base_model=model.repository)
        .result()
    )
    prompts = [
        predictor_tokens(tokenizer=tokenizer, problem=problem, thinking=True)
        for problem in problems["problem"]
    ]
    began = time.perf_counter()
    responses = asyncio.run(
        sample_all(
            client=client, prompts=prompts, stop=stop_tokens(tokenizer=tokenizer)
        )
    )
    wall = time.perf_counter() - began
    rows = []
    for problem_id, prompt, (response, seconds) in zip(
        problems["problem_id"], prompts, responses, strict=True
    ):
        if len(response.sequences) != PILOT.samples:
            raise RuntimeError(f"{problem_id}: expected {PILOT.samples} samples")
        for sample, sequence in enumerate(response.sequences):
            if sequence.logprobs is None:
                raise RuntimeError("Tinker returned no sampled-token logprobs")
            rows.append({
                "model": model.name,
                "problem_id": problem_id,
                "sample": sample,
                "prompt_tokens": len(prompt),
                "completion_tokens": len(sequence.tokens),
                "stop_reason": sequence.stop_reason,
                "request_seconds": seconds,
                "completion": tokenizer.decode(sequence.tokens),
                **parse_completion(
                    tokenizer=tokenizer,
                    tokens=list(sequence.tokens),
                    logprobs=list(sequence.logprobs),
                    stop_reason=sequence.stop_reason,
                ),
            })
    directory = root / "pilot"
    directory.mkdir(exist_ok=True)
    pl.DataFrame(rows, infer_schema_length=None).write_parquet(
        directory / f"{model.name}.parquet"
    )
    write_json(
        path=directory / f"{model.name}.json",
        payload={
            "model": model.model_dump(),
            "wall_seconds": wall,
            "pilot": PILOT.model_dump(),
        },
    )


def phase3_cost(
    *, model: PredictorModel, prompt: float, completion: float, rollouts: int
) -> float:
    prefill = prompt * (
        model.prefill_price + (rollouts - 1) * model.cached_prefill_price
    )
    sampled = rollouts * completion * model.sample_price
    trained = rollouts * (prompt + completion) * model.train_price
    return PILOT.phase3_problems_per_step * (prefill + sampled + trained) / 1e6


def model_summary(*, root: Path, model: PredictorModel) -> dict:
    frame = pl.read_parquet(root / "pilot" / f"{model.name}.parquet")
    wall = json.loads((root / "pilot" / f"{model.name}.json").read_text())[
        "wall_seconds"
    ]
    lengths = frame["completion_tokens"].to_numpy()
    prompts = frame.group_by("problem_id").agg(pl.col("prompt_tokens").first())[
        "prompt_tokens"
    ]
    valid = frame.filter(pl.col("status") == "valid")
    measured = (
        prompts.sum()
        * (model.prefill_price + (PILOT.samples - 1) * model.cached_prefill_price)
        + lengths.sum() * model.sample_price
    ) / 1e6
    summary = {
        "model": model.name,
        "rollouts": frame.height,
        "distinct_completions_per_problem": float(
            frame
            .group_by("problem_id")
            .agg(pl.col("completion").n_unique())["completion"]
            .mean()
        ),
        "completion_mean": float(lengths.mean()),
        "completion_median": float(np.median(lengths)),
        "completion_p90": float(np.quantile(lengths, 0.9)),
        "abort_rate": float((frame["status"] == "aborted").mean()),
        "malformed_rate": float((frame["status"] == "malformed").mean()),
        "yes_rate": float((valid["answer"] == "Yes").mean()),
        "mean_confidence": float(valid["confidence"].mean()),
        "confidence_above_0.99": float((valid["confidence"] > 0.99).mean()),
        "mean_request_seconds": float(
            frame
            .group_by("problem_id")
            .agg(pl.col("request_seconds").first())["request_seconds"]
            .mean()
        ),
        "wall_seconds": wall,
        "measured_cost_usd": measured,
        "measured_cost_per_rollout_usd": measured / frame.height,
        "phase3": [
            {
                "rollouts": rollouts,
                "per_step_usd": phase3_cost(
                    model=model,
                    prompt=float(prompts.mean()),
                    completion=float(lengths.mean()),
                    rollouts=rollouts,
                ),
                "per_config_usd": PILOT.phase3_steps
                * phase3_cost(
                    model=model,
                    prompt=float(prompts.mean()),
                    completion=float(lengths.mean()),
                    rollouts=rollouts,
                ),
            }
            for rollouts in PILOT.phase3_rollouts
        ],
    }
    return summary


def untrained_metrics(
    *, root: Path, model: PredictorModel, labels: pl.DataFrame
) -> dict:
    per_problem = (
        pl
        .read_parquet(root / "pilot" / f"{model.name}.parquet")
        .filter(pl.col("status") == "valid")
        .group_by("problem_id")
        .agg(q=(pl.col("answer") == "Yes").mean())
        .join(labels.select("problem_id", "p_hat"), on="problem_id")
    )
    return {
        "model": model.name,
        "problems": per_problem.height,
        **prediction_metrics(
            q=per_problem["q"].to_numpy(), p=per_problem["p_hat"].to_numpy()
        ),
    }


def metric_table(*, summaries: list[dict]) -> str:
    keys = [key for key in summaries[0] if key not in ("model", "phase3")]
    lines = [
        "| metric | " + " | ".join(summary["model"] for summary in summaries) + " |",
        "|---|" + "---|" * len(summaries),
    ]
    for key in keys:
        cells = [summary[key] for summary in summaries]
        lines.append(
            f"| {key} | "
            + " | ".join(
                f"{cell:.4g}" if isinstance(cell, float) else str(cell)
                for cell in cells
            )
            + " |"
        )
    return "\n".join(lines)


def plot_lengths(*, root: Path, models: list[PredictorModel], path: Path) -> None:
    figure, axes = plt.subplots(
        1,
        len(models),
        figsize=(6 * len(models), 4),
        constrained_layout=True,
        sharey=True,
        squeeze=False,
    )
    bins = np.linspace(0, PILOT.max_tokens, 65)
    for axis, color, model in zip(axes[0], SERIES_COLORS, models, strict=False):
        frame = pl.read_parquet(root / "pilot" / f"{model.name}.parquet")
        axis.hist(
            frame["completion_tokens"].to_numpy(),
            bins=bins,
            color=color,
            edgecolor="white",
            linewidth=1,
        )
        axis.axvline(PILOT.max_tokens, color="#52514e", linestyle="--", linewidth=1)
        axis.set_title(
            f"{model.name}: reasoning length, mean {frame['completion_tokens'].mean():.0f}, "
            f"aborted {(frame['status'] == 'aborted').mean():.1%}",
            fontsize=10,
        )
        axis.set_xlabel("completion tokens (thinking + answer)")
        axis.grid(axis="y", alpha=0.3, linewidth=0.5)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0, 0].set_ylabel("rollouts")
    figure.savefig(path, dpi=150)
    plt.close(figure)


def report(*, root: Path, names: list[str], with_labels: bool) -> None:
    models = [PREDICTORS[name] for name in names]
    summaries = [model_summary(root=root, model=model) for model in models]
    plot_lengths(
        root=root, models=models, path=root / "pilot" / "pilot-length-histograms.png"
    )
    costs = pl.DataFrame([
        {"model": summary["model"], **row}
        for summary in summaries
        for row in summary["phase3"]
    ])
    lines = [
        "# Phase 1 predictor pilot",
        "",
        (
            f"{PILOT.problems} training problems x {PILOT.samples} samples, thinking on, "
            f"temperature {PILOT.temperature}, cap {PILOT.max_tokens} tokens."
        ),
        "",
        metric_table(summaries=summaries),
        "",
        "## Projected Phase 3 training cost per config (measured mean lengths, evaluation excluded)",
        "",
        markdown_table(frame=costs),
        "",
        "![lengths](pilot-length-histograms.png)",
        "",
    ]
    untrained = []
    if with_labels:
        labels = read_labels(root=root)
        untrained = [
            untrained_metrics(root=root, model=model, labels=labels) for model in models
        ]
        lines += [
            "## Untrained predictor vs p̂ (per-problem Yes fraction)",
            "",
            markdown_table(frame=pl.DataFrame(untrained)),
            "",
        ]
    write_json(
        path=root / "pilot" / "summary.json",
        payload={"models": summaries, "untrained": untrained},
    )
    (root / "pilot" / "pilot.md").write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("run", "report"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--models", nargs="+", choices=tuple(PREDICTORS), required=True)
    parser.add_argument("--with-labels", action="store_true")
    args = parser.parse_args()
    if args.command == "run":
        for name in args.models:
            run_model(root=args.root, model=PREDICTORS[name])
    else:
        report(root=args.root, names=args.models, with_labels=args.with_labels)


if __name__ == "__main__":
    main()
