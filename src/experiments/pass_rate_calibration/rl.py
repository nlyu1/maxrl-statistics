import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import tinker
import torch
import wandb
from pydantic import ConfigDict
from pydantic.dataclasses import dataclass
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from src.experiments.bernoulli_calibration.distribution_shift import (
    pointwise_maxrl_probability,
)
from src.experiments.bernoulli_calibration.objectives import grouped_advantages
from src.experiments.pass_rate_calibration.config import (
    ENOTATION_PATTERN,
    PREDICTOR_PROMPT,
    RL_PREDICTOR,
    SERIES_COLORS,
    SPLITS,
    WANDB_PROJECT,
    RLRun,
    RLSweep,
    Split,
    rl_sweep,
)
from src.experiments.pass_rate_calibration.data import (
    read_labels,
    read_problems,
    require_gate,
)
from src.experiments.pass_rate_calibration.ground_truth import markdown_table
from src.experiments.pass_rate_calibration.metrics import prediction_metrics
from src.experiments.pass_rate_calibration.predictor import (
    parse_completion,
    predictor_tokens,
    stop_tokens,
)
from src.experiments.utkface.run import write_json

SWEEPS: tuple[RLSweep, ...] = ("smoke", "smoke-resume", "lr-pilot", "long")


@dataclass(kw_only=True, frozen=True, config=ConfigDict(arbitrary_types_allowed=True))
class Problems:
    frame: pl.DataFrame
    prompts: tuple[tuple[int, ...], ...]
    train: np.ndarray


@dataclass(kw_only=True, frozen=True)
class Rollout:
    row: int
    sample: int
    tokens: tuple[int, ...]
    logprobs: tuple[float, ...]
    stop_reason: str
    max_tokens: int
    status: str
    answer: str
    confidence: float


def load_problems(*, root: Path, tokenizer: PreTrainedTokenizerBase) -> Problems:
    require_gate(root=root)
    frame = (
        read_problems(root=root)
        .join(
            read_labels(root=root).select("problem_id", "p_hat"),
            on="problem_id",
            how="inner",
            validate="1:1",
        )
        .sort("problem_id")
        .with_columns(
            enotation=pl
            .col("answer")
            .str.strip_chars()
            .str.contains(f"^{ENOTATION_PATTERN}$")
        )
    )
    if frame.height != read_problems(root=root).height:
        raise ValueError("Some problems have no label")
    return Problems(
        frame=frame,
        prompts=tuple(
            tuple(predictor_tokens(tokenizer=tokenizer, problem=problem, thinking=True))
            for problem in frame["problem"]
        ),
        train=np.flatnonzero(frame["split"].to_numpy() == "train"),
    )


def sampling_params(
    *, max_tokens: int, temperature: float, tokenizer: PreTrainedTokenizerBase
) -> tinker.SamplingParams:
    return tinker.SamplingParams(
        max_tokens=max_tokens,
        temperature=temperature,
        stop=stop_tokens(tokenizer=tokenizer),
    )


def split_caps(*, run: RLRun) -> dict[Split, int]:
    return {"train": run.max_tokens, "val": run.max_tokens, "test": run.test_max_tokens}


def sample_rollouts(
    *,
    client: tinker.SamplingClient,
    problems: Problems,
    requests: list[tuple[int, tinker.SamplingParams]],
    samples: int,
    tokenizer: PreTrainedTokenizerBase,
) -> list[Rollout]:
    futures = [
        client.sample(
            prompt=tinker.ModelInput.from_ints(list(problems.prompts[row])),
            num_samples=samples,
            sampling_params=params,
        )
        for row, params in requests
    ]
    rollouts = []
    for (row, params), future in zip(requests, futures, strict=True):
        sequences = future.result().sequences
        if len(sequences) != samples:
            raise RuntimeError(f"Expected {samples} samples, got {len(sequences)}")
        for sample, sequence in enumerate(sequences):
            if sequence.logprobs is None or not sequence.tokens:
                raise RuntimeError("Tinker returned an empty sequence or no logprobs")
            tokens, logprobs = list(sequence.tokens), list(sequence.logprobs)
            parsed = parse_completion(
                tokenizer=tokenizer,
                tokens=tokens,
                logprobs=logprobs,
                stop_reason=sequence.stop_reason,
            )
            rollouts.append(
                Rollout(
                    row=int(row),
                    sample=sample,
                    tokens=tuple(tokens),
                    logprobs=tuple(logprobs),
                    stop_reason=sequence.stop_reason,
                    max_tokens=params.max_tokens,
                    status=parsed["status"],
                    answer=parsed["answer"] or "",
                    confidence=float("nan")
                    if parsed["confidence"] is None
                    else parsed["confidence"],
                )
            )
    return rollouts


def rollout_frame(
    *,
    rollouts: list[Rollout],
    problems: Problems,
    tokenizer: PreTrainedTokenizerBase,
) -> pl.DataFrame:
    identifiers = problems.frame["problem_id"]
    return pl.DataFrame(
        [
            {
                "problem_id": identifiers[rollout.row],
                "sample": rollout.sample,
                "completion_tokens": len(rollout.tokens),
                "stop_reason": rollout.stop_reason,
                "max_tokens": rollout.max_tokens,
                "status": rollout.status,
                "answer": rollout.answer,
                "confidence": rollout.confidence,
                "token_nll": -float(np.sum(rollout.logprobs)),
                "text": tokenizer.decode(list(rollout.tokens)),
            }
            for rollout in rollouts
        ],
        infer_schema_length=None,
    )


def request_cost(
    *, prompt_tokens: int, samples: int, completion_tokens: int, trained_tokens: int
) -> float:
    return (
        prompt_tokens
        * (
            RL_PREDICTOR.prefill_price
            + (samples - 1) * RL_PREDICTOR.cached_prefill_price
        )
        + completion_tokens * RL_PREDICTOR.sample_price
        + trained_tokens * RL_PREDICTOR.train_price
    ) / 1e6


def sampling_cost(*, frame: pl.DataFrame, problems: Problems, samples: int) -> float:
    lengths = dict(
        zip(
            problems.frame["problem_id"],
            map(len, problems.prompts),
            strict=True,
        )
    )
    return request_cost(
        prompt_tokens=sum(lengths[key] for key in frame["problem_id"].unique()),
        samples=samples,
        completion_tokens=int(frame["completion_tokens"].sum()),
        trained_tokens=0,
    )


def objective_optimum(*, run: RLRun, p: np.ndarray) -> np.ndarray:
    if run.method == "maxrl":
        return pointwise_maxrl_probability(
            truth=p, protocol=run.protocol, num_rollouts=run.rollouts
        )
    return (p > 0.5).astype(np.float64)


def evaluation_metrics(
    *, frame: pl.DataFrame, problems: Problems, run: RLRun
) -> dict[str, dict[str, float]]:
    rollouts = frame.join(
        problems.frame.select("problem_id", "split"), on="problem_id", how="left"
    )
    predictions = (
        rollouts
        .group_by("problem_id")
        .agg(
            valid=(pl.col("status") == "valid").sum(),
            yes=(pl.col("answer") == "Yes").sum(),
        )
        .with_columns(
            q=pl.when(pl.col("valid") > 0).then(pl.col("yes") / pl.col("valid"))
        )
        .join(
            problems.frame.select("problem_id", "split", "p_hat", "enotation"),
            on="problem_id",
        )
    )
    result = {}
    for split in SPLITS:
        part = predictions.filter(pl.col("split") == split)
        if part.height == 0:
            continue
        sampled = rollouts.filter(pl.col("split") == split)
        valid = sampled.filter(pl.col("status") == "valid")
        scored = part.filter(pl.col("q").is_not_null())
        q, p = scored["q"].to_numpy(), scored["p_hat"].to_numpy()
        optimum = objective_optimum(run=run, p=p)
        dead = (p > 0) & (p < 1) & np.isin(optimum, (0.0, 1.0))
        metrics = {
            **prediction_metrics(q=q, p=p),
            "problems": part.height,
            "undefined_q": part.height - scored.height,
            "boundary_fraction": float(np.isin(q, (0.0, 1.0)).mean()),
            "optimum_mae": float(np.abs(q - optimum).mean()),
            "dead_zone_collapse": float((q[dead] == optimum[dead]).mean())
            if dead.any()
            else None,
            "yes_rate": float((valid["answer"] == "Yes").mean()),
            "abort_rate": float((sampled["status"] == "aborted").mean()),
            "malformed_rate": float((sampled["status"] == "malformed").mean()),
            "completion_mean": float(sampled["completion_tokens"].mean()),
            "confidence_mean": float(valid["confidence"].mean()),
            "answer_entropy_bits": float(binary_entropy(q=q).mean()),
        }
        if "token_nll" in sampled.columns:
            metrics["token_entropy"] = float(
                sampled["token_nll"].sum() / sampled["completion_tokens"].sum()
            )
        clean = scored.filter(~pl.col("enotation"))
        if clean.height < scored.height:
            subset = prediction_metrics(
                q=clean["q"].to_numpy(), p=clean["p_hat"].to_numpy()
            )
            metrics |= {
                f"{key}_no_enotation": subset[key]
                for key in ("mae", "rmse", "pearson", "spearman", "bias")
            }
        result[split] = {
            key: value for key, value in metrics.items() if value is not None
        }
    return result


def binary_entropy(*, q: np.ndarray) -> np.ndarray:
    clipped = np.clip(q, 1e-12, 1 - 1e-12)
    return -(clipped * np.log2(clipped) + (1 - clipped) * np.log2(1 - clipped))


def prefixed(*, metrics: dict[str, float], prefix: str) -> dict[str, float]:
    return {f"{prefix}/{key}": value for key, value in metrics.items()}


def test_subset(*, root: Path, problems: Problems, run: RLRun) -> frozenset[str]:
    test = problems.frame.filter(pl.col("split") == "test")
    subjects = test.group_by("subject").len().sort("subject")
    shares = subjects["len"].to_numpy() * run.test_subset_size / test.height
    quota = np.floor(shares).astype(int)
    quota[np.argsort(quota - shares)[: run.test_subset_size - quota.sum()]] += 1
    generator = np.random.default_rng((run.seed, 2))
    chosen = []
    for subject, count in zip(subjects["subject"], quota, strict=True):
        pool = (
            test
            .filter(pl.col("subject") == subject)
            .sort("p_hat", "problem_id")["problem_id"]
            .to_numpy()
        )
        for tercile, slots in zip(
            np.array_split(pool, 3), np.array_split(np.arange(count), 3), strict=True
        ):
            chosen += generator.choice(tercile, size=len(slots), replace=False).tolist()
    subset = frozenset(chosen)
    path = root / "rl" / f"test-subset-{run.test_subset_size}.json"
    if path.exists():
        if frozenset(json.loads(path.read_text())["problem_ids"]) != subset:
            raise ValueError(f"{path} disagrees with the stratified test subset")
        return subset
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(
        path=path,
        payload={
            "problem_ids": sorted(subset),
            "quota": dict(zip(subjects["subject"], map(int, quota), strict=True)),
            "seed": run.seed,
        },
    )
    return subset


def eval_logs(
    *, frame: pl.DataFrame, problems: Problems, run: RLRun, subset: frozenset[str]
) -> dict[str, float]:
    metrics = evaluation_metrics(frame=frame, problems=problems, run=run)
    logged = {}
    for split, prefix in (("train", "train_eval"), ("val", "val")):
        if split in metrics:
            logged |= prefixed(metrics=metrics[split], prefix=prefix)
    tested = set(
        frame.join(
            problems.frame.select("problem_id", "split"), on="problem_id"
        ).filter(pl.col("split") == "test")["problem_id"]
    )
    if len(tested) == int((problems.frame["split"] == "test").sum()):
        logged |= prefixed(metrics=metrics["test"], prefix="test")
    if not subset <= tested:
        raise ValueError("An evaluation must cover the whole test subset")
    logged |= prefixed(
        metrics=evaluation_metrics(
            frame=frame.filter(pl.col("problem_id").is_in(sorted(subset))),
            problems=problems,
            run=run,
        )["test"],
        prefix="test_subset",
    )
    return logged


def step_rows(*, problems: Problems, run: RLRun, step: int) -> np.ndarray:
    per_epoch = len(problems.train) // run.problems_per_step
    epoch, offset = divmod(step - 1, per_epoch)
    order = np.random.default_rng((run.seed, epoch)).permutation(len(problems.train))
    chosen = order[
        offset * run.problems_per_step : (offset + 1) * run.problems_per_step
    ]
    return problems.train[chosen]


def step_labels(*, p: np.ndarray, run: RLRun, step: int) -> np.ndarray:
    count = 1 if run.protocol == "shared" else run.rollouts
    draws = np.random.default_rng((run.seed, step, 1)).random((len(p), count))
    return (draws < p[:, None]).astype(np.float32)


def rollout_datum(
    *, prompt: tuple[int, ...], rollout: Rollout, advantage: float
) -> tinker.Datum:
    tokens = [*prompt, *rollout.tokens]
    pad = len(prompt) - 1
    return tinker.Datum(
        model_input=tinker.ModelInput.from_ints(tokens[:-1]),
        loss_fn_inputs={
            "target_tokens": tokens[1:],
            "logprobs": [0.0] * pad + list(rollout.logprobs),
            "advantages": [0.0] * pad + [advantage] * len(rollout.tokens),
        },
    )


def group_diagnostics(
    *, rewards: np.ndarray, labels: np.ndarray, run: RLRun
) -> dict[str, float]:
    if run.protocol == "shared":
        counts = np.full((len(rewards), 1), rewards.shape[1])
        successes = rewards.sum(-1, keepdims=True)
    else:
        counts = np.stack([(labels == label).sum(-1) for label in (0, 1)], -1)
        successes = np.stack(
            [(rewards * (labels == label)).sum(-1) for label in (0, 1)], -1
        )
    active = counts > 0
    diagnostics = {
        "no_success_group_fraction": float(
            ((successes == 0) & active).sum() / active.sum()
        ),
        "all_success_group_fraction": float(
            ((successes == counts) & active).sum() / active.sum()
        ),
    }
    if run.protocol == "split":
        diagnostics |= {
            "empty_group_fraction": float((counts == 0).mean()),
            "singleton_group_fraction": float((counts == 1).mean()),
        }
    return diagnostics


def train_step(
    *,
    run: RLRun,
    problems: Problems,
    tokenizer: PreTrainedTokenizerBase,
    sampler: tinker.SamplingClient,
    training: tinker.TrainingClient,
    step: int,
) -> tuple[dict[str, float], pl.DataFrame]:
    began = time.perf_counter()
    rows = step_rows(problems=problems, run=run, step=step)
    p = problems.frame["p_hat"].to_numpy()[rows]
    labels = np.broadcast_to(
        step_labels(p=p, run=run, step=step), (len(rows), run.rollouts)
    ).copy()
    params = sampling_params(
        max_tokens=run.max_tokens, temperature=run.temperature, tokenizer=tokenizer
    )
    rollouts = sample_rollouts(
        client=sampler,
        problems=problems,
        requests=[(int(row), params) for row in rows],
        samples=run.rollouts,
        tokenizer=tokenizer,
    )
    sampled = time.perf_counter()
    shape = (len(rows), run.rollouts)
    yes = np.array([rollout.answer == "Yes" for rollout in rollouts]).reshape(shape)
    no = np.array([rollout.answer == "No" for rollout in rollouts]).reshape(shape)
    rewards = np.where(labels == 1, yes, no).astype(np.float32)
    advantages = grouped_advantages(
        rewards=torch.from_numpy(rewards),
        labels=torch.from_numpy(labels),
        method=run.method,
        protocol=run.protocol,
        std_floor=run.std_floor,
    ).numpy()
    scale = 1 / rewards.size
    trained = [
        (rollout, float(advantage))
        for rollout, advantage in zip(rollouts, advantages.ravel(), strict=True)
        if advantage != 0
    ]
    gaps = np.zeros(0)
    if trained:
        datums = [
            rollout_datum(
                prompt=problems.prompts[rollout.row],
                rollout=rollout,
                advantage=advantage * scale,
            )
            for rollout, advantage in trained
        ]
        backward = training.forward_backward(datums, "importance_sampling")
        update = training.optim_step(tinker.AdamParams(learning_rate=run.learning_rate))
        outputs = backward.result().loss_fn_outputs
        update.result()
        gaps = np.concatenate([
            output["logprobs"].to_numpy()[len(problems.prompts[rollout.row]) - 1 :]
            - np.asarray(rollout.logprobs)
            for output, (rollout, _) in zip(outputs, trained, strict=True)
        ])
    updated = time.perf_counter()
    frame = rollout_frame(
        rollouts=rollouts, problems=problems, tokenizer=tokenizer
    ).with_columns(
        label=pl.Series(labels.ravel()),
        reward=pl.Series(rewards.ravel()),
        advantage=pl.Series(advantages.ravel()),
    )
    lengths = frame["completion_tokens"].to_numpy()
    valid = frame.filter(pl.col("status") == "valid")
    answered = yes | no
    counts = answered.sum(-1)
    scored = counts > 0
    batch_q = yes.sum(-1)[scored] / counts[scored]
    cost = request_cost(
        prompt_tokens=sum(len(problems.prompts[row]) for row in rows),
        samples=run.rollouts,
        completion_tokens=int(lengths.sum()),
        trained_tokens=sum(
            len(problems.prompts[rollout.row]) + len(rollout.tokens)
            for rollout, _ in trained
        ),
    )
    metrics = {
        "rollout/sampled_reward": float(rewards.mean()),
        "rollout/label_one_fraction": float(labels.mean()),
        "rollout/yes_rate": float((valid["answer"] == "Yes").mean()),
        "rollout/abort_rate": float((frame["status"] == "aborted").mean()),
        "rollout/malformed_rate": float((frame["status"] == "malformed").mean()),
        "rollout/completion_mean": float(lengths.mean()),
        "rollout/completion_p90": float(np.quantile(lengths, 0.9)),
        "rollout/confidence_mean": float(valid["confidence"].mean()),
        "rollout/nonzero_advantage_fraction": float((advantages != 0).mean()),
        "rollout/advantage_abs_max": float(np.abs(advantages).max()),
        "rollout/batch_bias": float(batch_q.mean() - p[scored].mean()),
        "rollout/batch_mae": float(np.abs(batch_q - p[scored]).mean()),
        "rollout/answer_entropy_bits": float(binary_entropy(q=batch_q).mean()),
        "rollout/token_entropy": float(
            -sum(np.sum(rollout.logprobs) for rollout in rollouts) / lengths.sum()
        ),
        **{
            f"rollout/yes_rate_label_{name}": float(yes[mask].mean())
            for value, name in ((1, "yes"), (0, "no"))
            for mask in [(labels == value) & answered]
            if mask.any()
        },
        "train/datums": len(trained),
        "train/logprob_gap_mean": float(gaps.mean()) if gaps.size else 0.0,
        "train/logprob_gap_abs_mean": float(np.abs(gaps).mean()) if gaps.size else 0.0,
        "cost/step_usd": cost,
        "time/sample_seconds": sampled - began,
        "time/train_seconds": updated - sampled,
        **prefixed(
            metrics=group_diagnostics(rewards=rewards, labels=labels, run=run),
            prefix="rollout",
        ),
    }
    return metrics, frame


def base_eval_groups(*, run: RLRun) -> dict[int, tuple[Split, ...]]:
    groups: dict[int, tuple[Split, ...]] = {}
    for split, cap in split_caps(run=run).items():
        groups[cap] = (*groups.get(cap, ()), split)
    return groups


def base_eval_directory(*, root: Path, run: RLRun, max_tokens: int) -> Path:
    return (
        root
        / "rl"
        / "base-eval"
        / f"samples{run.eval_samples}-tokens{max_tokens}-temp{run.temperature:g}"
    )


def evaluate(
    *,
    client: tinker.SamplingClient,
    problems: Problems,
    run: RLRun,
    tokenizer: PreTrainedTokenizerBase,
    rows: np.ndarray,
) -> pl.DataFrame:
    params = {
        split: sampling_params(
            max_tokens=cap, temperature=run.temperature, tokenizer=tokenizer
        )
        for split, cap in split_caps(run=run).items()
    }
    splits = problems.frame["split"].to_numpy()
    return rollout_frame(
        rollouts=sample_rollouts(
            client=client,
            problems=problems,
            requests=[(int(row), params[splits[row]]) for row in rows],
            samples=run.eval_samples,
            tokenizer=tokenizer,
        ),
        problems=problems,
        tokenizer=tokenizer,
    )


def base_eval(*, root: Path, run: RLRun) -> None:
    tokenizer = AutoTokenizer.from_pretrained(
        RL_PREDICTOR.repository, revision=RL_PREDICTOR.revision
    )
    problems = load_problems(root=root, tokenizer=tokenizer)
    test_subset(root=root, problems=problems, run=run)
    client = (
        tinker
        .ServiceClient()
        .create_sampling_client(base_model=RL_PREDICTOR.repository)
        .result()
    )
    for cap, splits in base_eval_groups(run=run).items():
        directory = base_eval_directory(root=root, run=run, max_tokens=cap)
        if (directory / "meta.json").exists():
            meta = json.loads((directory / "meta.json").read_text())
            if meta["prompt"] != PREDICTOR_PROMPT:
                raise ValueError(f"{directory} was sampled with a different prompt")
            continue
        began = time.perf_counter()
        frame = evaluate(
            client=client,
            problems=problems,
            run=run,
            tokenizer=tokenizer,
            rows=np.flatnonzero(problems.frame["split"].is_in(splits).to_numpy()),
        )
        directory.mkdir(parents=True, exist_ok=True)
        frame.write_parquet(directory / "rollouts.parquet")
        write_json(
            path=directory / "meta.json",
            payload={
                "prompt": PREDICTOR_PROMPT,
                "model": RL_PREDICTOR.model_dump(),
                "splits": list(splits),
                "eval_samples": run.eval_samples,
                "max_tokens": cap,
                "temperature": run.temperature,
                "seconds": time.perf_counter() - began,
                "cost_usd": sampling_cost(
                    frame=frame, problems=problems, samples=run.eval_samples
                ),
            },
        )


def load_base_eval(*, root: Path, run: RLRun, problems: Problems) -> pl.DataFrame:
    frames = []
    for cap, splits in base_eval_groups(run=run).items():
        directory = base_eval_directory(root=root, run=run, max_tokens=cap)
        frame = (
            pl
            .read_parquet(directory / "rollouts.parquet")
            .join(problems.frame.select("problem_id", "split"), on="problem_id")
            .filter(pl.col("split").is_in(splits))
            .drop("split")
        )
        if "max_tokens" not in frame.columns:
            frame = frame.with_columns(max_tokens=pl.lit(cap, dtype=pl.Int64))
        expected = int(problems.frame["split"].is_in(splits).sum())
        if frame["problem_id"].n_unique() != expected:
            raise ValueError(f"{directory} does not cover the splits {splits}")
        frames.append(frame.select(sorted(frame.columns)))
    return pl.concat(frames)


def read_history(*, directory: Path) -> list[dict]:
    path = directory / "history.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines()]


def finished(*, directory: Path, run: RLRun) -> bool:
    path = directory / "result.json"
    return path.exists() and json.loads(path.read_text())["steps"] >= run.steps


def comparable(*, run: RLRun) -> dict:
    return run.model_dump(exclude={"steps"})


def train(*, root: Path, sweep: RLSweep, run: RLRun) -> None:
    directory = root / "rl" / run.identifier
    tokenizer = AutoTokenizer.from_pretrained(
        RL_PREDICTOR.repository, revision=RL_PREDICTOR.revision
    )
    problems = load_problems(root=root, tokenizer=tokenizer)
    if len(problems.train) % run.problems_per_step:
        raise ValueError("Problems per step must divide the training set")
    steps_per_epoch = len(problems.train) // run.problems_per_step
    base = load_base_eval(root=root, run=run, problems=problems)
    subset = test_subset(root=root, problems=problems, run=run)
    identifiers = problems.frame["problem_id"].to_numpy()
    periodic_rows = np.flatnonzero(
        (problems.frame["split"].to_numpy() == "val")
        | np.isin(identifiers, sorted(subset))
    )
    service = tinker.ServiceClient()
    state_path = directory / "state.json"
    if state_path.exists():
        if json.loads((directory / "config.json").read_text())["comparable"] != (
            comparable(run=run)
        ):
            raise ValueError(f"{directory} was trained with a different config")
        state = json.loads(state_path.read_text())
        training = service.create_training_client_from_state_with_optimizer(
            path=state["state_path"]
        ).result()
        sampler = service.create_sampling_client(
            model_path=state["sampler_path"]
        ).result()
        start = state["step"] + 1
        history = [
            row
            for row in read_history(directory=directory)
            if row["step"] <= state["step"]
        ]
    else:
        if directory.exists():
            abandoned = directory.with_name(
                f"{directory.name}.abandoned-{time.strftime('%Y%m%d-%H%M%S')}"
            )
            directory.rename(abandoned)
            print(f"Moved a run without a checkpoint to {abandoned}", flush=True)
        directory.mkdir(parents=True)
        state = {"wandb_id": wandb.util.generate_id()}
        write_json(
            path=directory / "config.json",
            payload={"run": run.model_dump(), "comparable": comparable(run=run)},
        )
        training = service.create_lora_training_client(
            base_model=RL_PREDICTOR.repository, rank=run.lora_rank, seed=run.seed
        ).result()
        sampler = training.save_weights_and_get_sampling_client().result()
        start = 1
        history = []
    wandb.init(
        project=WANDB_PROJECT,
        name=f"phase3/{run.identifier}",
        group=f"phase3-{sweep}",
        tags=[root.name, sweep],
        id=state["wandb_id"],
        resume="allow",
        config={
            **run.model_dump(),
            "root": root.name,
            "prompt": PREDICTOR_PROMPT,
            "model": RL_PREDICTOR.repository,
        },
        dir=str(directory),
    )
    (directory / "rollouts").mkdir(exist_ok=True)
    (directory / "eval").mkdir(exist_ok=True)

    def record(metrics: dict[str, float]) -> None:
        history.append(metrics)
        (directory / "history.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in history)
        )
        wandb.log(
            {key: value for key, value in metrics.items() if key != "step"},
            step=metrics["step"],
        )
        print(
            json.dumps({
                key: metrics[key]
                for key in (
                    "step",
                    "rollout/sampled_reward",
                    "rollout/yes_rate",
                    "val/mae",
                    "test_subset/mae",
                    "cost/cumulative_usd",
                )
                if key in metrics
            }),
            flush=True,
        )

    if start == 1:
        record({
            "step": 0,
            "cost/cumulative_usd": 0.0,
            **eval_logs(frame=base, problems=problems, run=run, subset=subset),
        })
    for step in range(start, run.steps + 1):
        began = time.perf_counter()
        metrics, frame = train_step(
            run=run,
            problems=problems,
            tokenizer=tokenizer,
            sampler=sampler,
            training=training,
            step=step,
        )
        frame.write_parquet(directory / "rollouts" / f"step-{step:04d}.parquet")
        if step % steps_per_epoch == 0:
            epoch = pl.concat([
                pl.read_parquet(directory / "rollouts" / f"step-{index:04d}.parquet")
                for index in range(step - steps_per_epoch + 1, step + 1)
            ])
            metrics |= prefixed(
                metrics=evaluation_metrics(frame=epoch, problems=problems, run=run)[
                    "train"
                ],
                prefix="train_epoch",
            )
        saved = time.perf_counter()
        if step % run.eval_every == 0 or step == run.steps:
            sampler_path = (
                training.save_weights_for_sampler(name=f"step-{step:04d}").result().path
            )
            sampler = service.create_sampling_client(model_path=sampler_path).result()
            evaluated = evaluate(
                client=sampler,
                problems=problems,
                run=run,
                tokenizer=tokenizer,
                rows=np.arange(problems.frame.height)
                if step == run.steps
                else periodic_rows,
            )
            evaluated.write_parquet(directory / "eval" / f"step-{step:04d}.parquet")
            metrics |= eval_logs(
                frame=evaluated, problems=problems, run=run, subset=subset
            )
            metrics["cost/eval_usd"] = sampling_cost(
                frame=evaluated, problems=problems, samples=run.eval_samples
            )
            state |= {
                "step": step,
                "sampler_path": sampler_path,
                "state_path": training
                .save_state(name=f"step-{step:04d}")
                .result()
                .path,
            }
            write_json(path=state_path, payload=state)
            metrics["time/eval_seconds"] = time.perf_counter() - saved
        else:
            sampler = training.save_weights_and_get_sampling_client().result()
            metrics["time/save_seconds"] = time.perf_counter() - saved
        metrics["time/step_seconds"] = time.perf_counter() - began
        metrics["cost/cumulative_usd"] = (
            history[-1]["cost/cumulative_usd"]
            + metrics["cost/step_usd"]
            + metrics.get("cost/eval_usd", 0.0)
        )
        record({"step": step, **metrics})
    final = history[-1]
    write_json(
        path=directory / "result.json",
        payload={
            "config": run.model_dump(),
            "steps": run.steps,
            "prompt": PREDICTOR_PROMPT,
            "final": final,
            "state": state,
        },
    )
    wandb.finish()


def sweep_runs(*, sweep: RLSweep) -> list[RLRun]:
    runs = rl_sweep(name=sweep)
    if len({(run.eval_samples, run.max_tokens, run.temperature) for run in runs}) != 1:
        raise ValueError("Runs in a sweep must share the evaluation sampling config")
    return runs


def launch(*, root: Path, sweep: RLSweep) -> None:
    runs = sweep_runs(sweep=sweep)
    base_eval(root=root, run=runs[0])
    (root / "logs").mkdir(exist_ok=True)
    processes = []
    for run in runs:
        if finished(directory=root / "rl" / run.identifier, run=run):
            continue
        with (root / "logs" / f"rl-{run.identifier}.log").open("a") as log:
            processes.append((
                run.identifier,
                subprocess.Popen(
                    [
                        sys.executable,
                        "-m",
                        "src.experiments.pass_rate_calibration.rl",
                        "train",
                        "--root",
                        str(root),
                        "--sweep",
                        sweep,
                        "--identifier",
                        run.identifier,
                    ],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                ),
            ))
    failed = [identifier for identifier, process in processes if process.wait()]
    report(root=root, sweep=sweep)
    if failed:
        raise RuntimeError(f"Failed runs: {failed}")


LINE_STYLES = {
    ("rloo", "shared"): "--",
    ("grpo", "shared"): "-.",
    ("maxrl", "shared"): "-",
    ("maxrl", "split"): ":",
}
PANEL_ROWS = (
    (
        "rollout/sampled_reward",
        "rollout/yes_rate",
        "rollout/yes_rate_label_yes",
        "rollout/yes_rate_label_no",
        "rollout/batch_bias",
    ),
    (
        "rollout/abort_rate",
        "rollout/completion_mean",
        "rollout/token_entropy",
        "rollout/answer_entropy_bits",
        "rollout/nonzero_advantage_fraction",
    ),
    (
        "val/mae",
        "val/spearman",
        "test_subset/mae",
        "test_subset/spearman",
        "test_subset/bias",
    ),
)
SMOOTHING_STEPS = 5


def trailing_mean(*, values: np.ndarray, window: int) -> np.ndarray:
    kernel = np.ones(window)
    sums = np.convolve(values, kernel)[: len(values)]
    counts = np.convolve(np.ones_like(values), kernel)[: len(values)]
    return sums / counts


def report(*, root: Path, sweep: RLSweep) -> None:
    runs = sweep_runs(sweep=sweep)
    histories = {
        run.identifier: read_history(directory=root / "rl" / run.identifier)
        for run in runs
    }
    rates = sorted({run.learning_rate for run in runs})
    families = sorted({(run.method, run.protocol) for run in runs})
    keyed = rates if len(rates) > 1 else families
    if len(keyed) > len(SERIES_COLORS):
        raise ValueError("More series than series colors")
    figure, axes = plt.subplots(
        len(PANEL_ROWS),
        len(PANEL_ROWS[0]),
        figsize=(4 * len(PANEL_ROWS[0]), 3.4 * len(PANEL_ROWS)),
        constrained_layout=True,
    )
    for run in runs:
        history = histories[run.identifier]
        if not history:
            continue
        key = run.learning_rate if len(rates) > 1 else (run.method, run.protocol)
        style = {
            "color": SERIES_COLORS[keyed.index(key)],
            "linestyle": LINE_STYLES[(run.method, run.protocol)]
            if len(rates) > 1
            else "-",
            "linewidth": 2,
            "label": f"{run.method}-{run.protocol} lr {run.learning_rate:g}",
        }
        for row, panels in enumerate(PANEL_ROWS):
            evaluated = row == len(PANEL_ROWS) - 1
            for axis, panel in zip(axes[row], panels, strict=True):
                points = [
                    (item["step"], item[panel]) for item in history if panel in item
                ]
                if not points:
                    continue
                steps, values = (
                    np.asarray(part, dtype=float) for part in zip(*points, strict=True)
                )
                if evaluated:
                    axis.plot(steps, values, marker="o", markersize=4, **style)
                else:
                    axis.plot(
                        steps,
                        trailing_mean(values=values, window=SMOOTHING_STEPS),
                        **style,
                    )
    for row, panels in enumerate(PANEL_ROWS):
        for axis, panel in zip(axes[row], panels, strict=True):
            suffix = (
                "" if row == len(PANEL_ROWS) - 1 else f" ({SMOOTHING_STEPS}-step mean)"
            )
            axis.set_title(panel + suffix, fontsize=9)
            axis.grid(alpha=0.3, linewidth=0.5)
            axis.spines[["top", "right"]].set_visible(False)
            axis.set_xlabel("step")
    axes[0, 0].legend(frameon=False, fontsize=8)
    directory = root / "rl"
    figure.savefig(directory / f"{sweep}-curves.png", dpi=130)
    plt.close(figure)
    columns = (
        "step",
        "val/mae",
        "val/spearman",
        "val/bias",
        "test_subset/mae",
        "test_subset/spearman",
        "test/mae",
        "test/spearman",
        "test/bias",
        "test/abort_rate",
        "train_eval/mae",
        "cost/cumulative_usd",
    )
    rows = []
    for identifier, history in histories.items():
        evaluated = [row for row in history if "val/mae" in row]
        for row in evaluated[:1] + evaluated[1:][-1:]:
            rows.append({
                "run": identifier,
                **{key: row.get(key) for key in columns},
            })
    lines = [
        f"# Phase 3 sweep `{sweep}`",
        "",
        (
            "Step 0 is the shared base model; the last row per run is its final "
            "evaluation. `test_subset` is the fixed stratified JEEBench subset, and "
            "`test` and `train_eval` are the full sets (step 0 and the last step)."
        ),
        "",
        markdown_table(frame=pl.DataFrame(rows, infer_schema_length=None))
        if rows
        else "No evaluations yet.",
        "",
        f"![curves]({sweep}-curves.png)",
        "",
    ]
    (directory / f"{sweep}.md").write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("base-eval", "train", "launch", "report"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--sweep", choices=SWEEPS, required=True)
    parser.add_argument("--identifier")
    args = parser.parse_args()
    runs = {run.identifier: run for run in sweep_runs(sweep=args.sweep)}
    if args.command == "base-eval":
        base_eval(root=args.root, run=next(iter(runs.values())))
    elif args.command == "train":
        train(root=args.root, sweep=args.sweep, run=runs[args.identifier])
    elif args.command == "launch":
        launch(root=args.root, sweep=args.sweep)
    else:
        report(root=args.root, sweep=args.sweep)


if __name__ == "__main__":
    main()
