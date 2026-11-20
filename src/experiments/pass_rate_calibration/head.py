import argparse
import json
import math
import os
import subprocess
import time
from collections.abc import Callable, Iterable
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import torch
import torch.distributed as dist
import torch.nn.functional as F
import wandb
from pydantic import ConfigDict
from pydantic.dataclasses import dataclass
from torch import Tensor, nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import CPUOffloadPolicy, MixedPrecisionPolicy, fully_shard
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.experiments.pass_rate_calibration.config import (
    FROZEN_LEARNING_RATES,
    FULL_LEARNING_RATES,
    HEAD_BACKBONE,
    SERIES_COLORS,
    SPLITS,
    WANDB_PROJECT,
    HeadRun,
    head_runs,
)
from src.experiments.pass_rate_calibration.data import read_labels, read_problems
from src.experiments.pass_rate_calibration.gpu import (
    claimed_gpus,
    visible_devices,
    wait_for_gpus,
)
from src.experiments.pass_rate_calibration.ground_truth import markdown_table
from src.experiments.pass_rate_calibration.metrics import prediction_metrics
from src.experiments.pass_rate_calibration.predictor import (
    answer_tokens,
    predictor_tokens,
)
from src.experiments.utkface.distributed import DistributedContext, distributed_context
from src.experiments.utkface.run import write_json

EVAL_BATCH = 16
FEATURE_BATCH = 8
CHECK_TOLERANCE = 0.5


@dataclass(kw_only=True, frozen=True, config=ConfigDict(arbitrary_types_allowed=True))
class LabelledProblems:
    ids: tuple[str, ...]
    splits: np.ndarray
    p: Tensor
    train: Tensor


def load_labelled(*, root: Path) -> LabelledProblems:
    if not json.loads((root / "ground-truth.json").read_text())["gate_passed"]:
        raise RuntimeError(
            "Phase 0 pass@1 gate failed; reconcile the solver settings first"
        )
    labels = read_labels(root=root).sort("problem_id")
    splits = labels["split"].to_numpy()
    return LabelledProblems(
        ids=tuple(labels["problem_id"]),
        splits=splits,
        p=torch.tensor(labels["p_hat"].to_numpy(), dtype=torch.float32),
        train=torch.tensor(np.flatnonzero(splits == "train")),
    )


def backbone_sequences(*, root: Path) -> tuple[list[list[int]], int, tuple[int, int]]:
    tokenizer = AutoTokenizer.from_pretrained(
        HEAD_BACKBONE.repository, revision=HEAD_BACKBONE.revision
    )
    problems = read_problems(root=root).sort("problem_id")
    sequences = [
        predictor_tokens(tokenizer=tokenizer, problem=problem, thinking=False)
        for problem in problems["problem"]
    ]
    return sequences, tokenizer.pad_token_id, answer_tokens(tokenizer=tokenizer)


def padded(*, sequences: list[list[int]], pad: int) -> tuple[Tensor, Tensor, Tensor]:
    ids = torch.full((len(sequences), max(map(len, sequences))), pad, dtype=torch.long)
    mask = torch.zeros_like(ids)
    for row, tokens in enumerate(sequences):
        ids[row, : len(tokens)] = torch.tensor(tokens)
        mask[row, : len(tokens)] = 1
    return ids, mask, mask.sum(-1) - 1


class LinearHead(nn.Module):
    def __init__(self, *, weight: Tensor) -> None:
        super().__init__()
        self.linear = nn.Linear(weight.numel(), 1)
        with torch.no_grad():
            self.linear.weight.copy_(weight[None])
            self.linear.bias.zero_()

    def forward(self, hidden: Tensor) -> Tensor:
        return self.linear(hidden).squeeze(-1)


class BackbonePredictor(nn.Module):
    def __init__(self, *, backbone: nn.Module, weight: Tensor) -> None:
        super().__init__()
        self.backbone = backbone
        self.head = LinearHead(weight=weight)

    def forward(
        self, *, input_ids: Tensor, attention_mask: Tensor, last: Tensor
    ) -> Tensor:
        state = self.backbone(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state
        return self.head(state[torch.arange(len(last), device=last.device), last])


def extract_features(*, root: Path, gpu: int) -> None:
    directory = root / "heads"
    directory.mkdir(parents=True, exist_ok=True)
    sequences, pad, (yes, no) = backbone_sequences(root=root)
    with claimed_gpus(indices=(gpu,)) as claimed:
        os.environ["CUDA_VISIBLE_DEVICES"] = claimed[0]["uuid"]
        device = torch.device("cuda:0")
        model = (
            AutoModelForCausalLM
            .from_pretrained(
                HEAD_BACKBONE.repository,
                revision=HEAD_BACKBONE.revision,
                dtype=torch.bfloat16,
            )
            .to(device)
            .eval()
        )
        weight = (model.lm_head.weight[yes] - model.lm_head.weight[no]).float().cpu()
        hidden, readout = [], []
        with (
            torch.cuda.device(device),
            torch.no_grad(),
            torch.autocast(device_type="cuda", dtype=torch.bfloat16),
        ):
            for start in range(0, len(sequences), FEATURE_BATCH):
                ids, mask, last = (
                    tensor.to(device)
                    for tensor in padded(
                        sequences=sequences[start : start + FEATURE_BATCH], pad=pad
                    )
                )
                rows = torch.arange(len(last), device=device)
                hidden.append(
                    model
                    .model(input_ids=ids, attention_mask=mask)
                    .last_hidden_state[rows, last]
                    .float()
                    .cpu()
                )
                logits = (
                    model(input_ids=ids, attention_mask=mask).logits[rows, last].float()
                )
                readout.append((logits[:, yes] - logits[:, no]).cpu())
    hidden, readout = torch.cat(hidden), torch.cat(readout)
    error = float((hidden @ weight - readout).abs().max())
    if error > CHECK_TOLERANCE:
        raise RuntimeError(
            f"Head initialization does not reproduce the Yes/No logit: {error}"
        )
    torch.save(
        {
            "problem_ids": read_problems(root=root)
            .sort("problem_id")["problem_id"]
            .to_list(),
            "hidden": hidden,
            "head_weight": weight,
            "zero_shot_logit": readout,
            "yes_token": yes,
            "no_token": no,
            "revision": HEAD_BACKBONE.revision,
            "max_check_error": error,
        },
        directory / "features.pt",
    )


def training_loss(*, run: HeadRun, logits: Tensor, targets: Tensor) -> Tensor:
    if run.loss == "ce":
        return F.binary_cross_entropy_with_logits(logits, targets)
    return (logits.sigmoid() - targets).square().mean()


def native_loss(*, run: HeadRun, logits: Tensor, p: Tensor) -> Tensor:
    if run.loss == "ce":
        return F.binary_cross_entropy_with_logits(logits, p, reduction="none")
    return (logits.sigmoid() - p).square() + p * (1 - p)


def train(
    *,
    run: HeadRun,
    labels: LabelledProblems,
    predict: Callable[[Tensor], Tensor],
    evaluate: Callable[[], Tensor],
    parameters: Iterable[nn.Parameter],
    context: DistributedContext,
    directory: Path,
) -> None:
    if run.problems_per_step % context.world_size:
        raise ValueError("Problems per step must divide evenly across ranks")
    order = torch.Generator().manual_seed(run.seed)
    label_draws = torch.Generator().manual_seed(run.seed + 1)
    optimizer = torch.optim.AdamW(parameters, lr=run.learning_rate, weight_decay=0.0)
    history, steps, snapshots, queue = [], [], [], []
    if context.primary:
        directory.mkdir(parents=True, exist_ok=True)
        write_json(path=directory / "config.json", payload=run.model_dump())
        wandb.init(
            project=WANDB_PROJECT,
            name=f"phase2/{run.identifier}",
            group=f"phase2-{run.backbone}",
            config=run.model_dump(),
            dir=str(directory),
        )

    def record(step: int) -> None:
        with torch.no_grad():
            logits = evaluate().float().cpu()
        if not context.primary:
            return
        q, p = logits.sigmoid().numpy(), labels.p.numpy()
        natives = native_loss(run=run, logits=logits, p=labels.p).numpy()
        logged = {}
        for split in SPLITS:
            mask = labels.splits == split
            row = {
                "step": step,
                "split": split,
                "native_loss": float(natives[mask].mean()),
                **prediction_metrics(q=q[mask], p=p[mask]),
            }
            history.append(row)
            logged |= {
                f"{split}/{key}": value
                for key, value in row.items()
                if key not in ("step", "split") and value is not None
            }
        wandb.log(logged, step=step)
        snapshots.append(pl.DataFrame({"step": step, "problem_id": labels.ids, "q": q}))

    record(0)
    for step in range(1, run.steps + 1):
        while len(queue) < run.problems_per_step:
            queue += labels.train[
                torch.randperm(len(labels.train), generator=order)
            ].tolist()
        batch = torch.tensor(queue[: run.problems_per_step])
        queue = queue[run.problems_per_step :]
        probabilities = labels.p[batch]
        targets = (
            torch.bernoulli(probabilities, generator=label_draws)
            if run.target == "binary"
            else probabilities
        )
        local = slice(context.rank, None, context.world_size)
        began = time.perf_counter()
        logits = predict(batch[local]).float()
        loss = training_loss(
            run=run, logits=logits, targets=targets[local].to(logits.device)
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if context.device.type == "cuda":
            torch.cuda.synchronize(context.device)
        seconds = time.perf_counter() - began
        mean_loss = (
            float(context.sum(values=loss.detach().reshape(1)).item())
            / context.world_size
        )
        if context.primary:
            steps.append({"step": step, "loss": mean_loss, "seconds": seconds})
            wandb.log(
                {"train/loss": mean_loss, "train/step_seconds": seconds}, step=step
            )
        if step % run.eval_every == 0 or step == run.steps:
            record(step)
    if not context.primary:
        return
    pl.DataFrame(history, infer_schema_length=None).write_parquet(
        directory / "history.parquet"
    )
    pl.DataFrame(steps).write_parquet(directory / "steps.parquet")
    pl.concat(snapshots).write_parquet(directory / "predictions.parquet")
    final = {row["split"]: row for row in history if row["step"] == run.steps}
    write_json(
        path=directory / "result.json",
        payload={
            "config": run.model_dump(),
            "final": final,
            "mean_step_seconds": float(np.mean([row["seconds"] for row in steps])),
        },
    )
    wandb.finish()


def train_frozen(*, root: Path) -> None:
    labels = load_labelled(root=root)
    features = torch.load(root / "heads" / "features.pt")
    if tuple(features["problem_ids"]) != labels.ids:
        raise ValueError("Feature cache and labels disagree on problem order")
    hidden = features["hidden"]
    context = DistributedContext(rank=0, world_size=1, device=torch.device("cpu"))
    for run in head_runs(backbone="frozen", learning_rates=FROZEN_LEARNING_RATES):
        directory = root / "heads" / run.identifier
        if (directory / "result.json").exists():
            continue
        torch.manual_seed(run.seed)
        head = LinearHead(weight=features["head_weight"])
        train(
            run=run,
            labels=labels,
            predict=lambda batch, head=head: head(hidden[batch]),
            evaluate=lambda head=head: head(hidden),
            parameters=head.parameters(),
            context=context,
            directory=directory,
        )


def full_run(*, identifier: str) -> HeadRun:
    runs = {
        run.identifier: run
        for run in head_runs(backbone="full", learning_rates=FULL_LEARNING_RATES)
    }
    return runs[identifier]


def train_full(*, root: Path, identifier: str, cpu_offload: bool) -> None:
    run = full_run(identifier=identifier)
    labels = load_labelled(root=root)
    sequences, pad, (yes, no) = backbone_sequences(root=root)
    input_ids, mask, last = padded(sequences=sequences, pad=pad)
    torch.manual_seed(run.seed)
    with distributed_context(device="cuda:0") as context:
        model = AutoModelForCausalLM.from_pretrained(
            HEAD_BACKBONE.repository,
            revision=HEAD_BACKBONE.revision,
            dtype=torch.float32,
        )
        weight = (model.lm_head.weight[yes] - model.lm_head.weight[no]).detach().clone()
        backbone = model.model
        del model
        backbone.config.use_cache = False
        backbone.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        predictor = BackbonePredictor(backbone=backbone, weight=weight)
        sharding = {
            "mesh": init_device_mesh("cuda", (context.world_size,)),
            "mp_policy": MixedPrecisionPolicy(
                param_dtype=torch.bfloat16, reduce_dtype=torch.float32
            ),
            "offload_policy": CPUOffloadPolicy() if cpu_offload else None,
        }
        sharding = {key: value for key, value in sharding.items() if value is not None}
        for layer in backbone.layers:
            fully_shard(layer, **sharding)
        fully_shard(predictor, **sharding)
        predictor.train()

        def forward(batch: Tensor) -> Tensor:
            width = int(mask[batch].sum(-1).max())
            with (
                torch.cuda.device(context.device),
                torch.autocast(device_type="cuda", dtype=torch.bfloat16),
            ):
                return predictor(
                    input_ids=input_ids[batch, :width].to(context.device),
                    attention_mask=mask[batch, :width].to(context.device),
                    last=last[batch].to(context.device),
                )

        def evaluate() -> Tensor:
            predictor.eval()
            owned = torch.arange(len(sequences))[context.rank :: context.world_size]
            per_rank = math.ceil(len(sequences) / context.world_size)
            padded_owned = torch.cat([owned, owned[-1:].repeat(per_rank - len(owned))])
            values = torch.cat([
                forward(padded_owned[start : start + EVAL_BATCH]).float().cpu()
                for start in range(0, per_rank, EVAL_BATCH)
            ])[: len(owned)]
            predictor.train()
            gathered = [(owned, values)]
            if context.world_size > 1:
                gathered = [None] * context.world_size
                dist.all_gather_object(gathered, (owned, values))
            logits = torch.empty(len(sequences))
            for indices, outputs in gathered:
                logits[indices] = outputs
            return logits

        train(
            run=run,
            labels=labels,
            predict=forward,
            evaluate=evaluate,
            parameters=predictor.parameters(),
            context=context,
            directory=root / "heads" / run.identifier,
        )


def queue_full(
    *, root: Path, gpus: int, cpu_offload: bool, poll_seconds: float
) -> None:
    (root / "logs").mkdir(exist_ok=True)
    for run in head_runs(backbone="full", learning_rates=FULL_LEARNING_RATES):
        if (root / "heads" / run.identifier / "result.json").exists():
            continue
        while True:
            indices = wait_for_gpus(count=gpus, poll_seconds=poll_seconds)
            try:
                with (
                    claimed_gpus(indices=indices) as claimed,
                    (root / "logs" / f"{run.identifier}.log").open("a") as log,
                ):
                    code = subprocess.run(
                        [
                            "uv",
                            "run",
                            "python",
                            "-m",
                            "torch.distributed.run",
                            "--standalone",
                            f"--nproc-per-node={gpus}",
                            "-m",
                            "src.experiments.pass_rate_calibration.head",
                            "train-full",
                            "--root",
                            str(root),
                            "--identifier",
                            run.identifier,
                            *(["--cpu-offload"] if cpu_offload else []),
                        ],
                        env={
                            **os.environ,
                            "CUDA_VISIBLE_DEVICES": visible_devices(gpus=claimed),
                            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
                        },
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        check=False,
                    ).returncode
                break
            except (RuntimeError, BlockingIOError):
                time.sleep(poll_seconds)
        if code:
            raise RuntimeError(f"{run.identifier} failed with exit code {code}")


def result_rows(*, root: Path) -> pl.DataFrame:
    rows = []
    for path in sorted((root / "heads").glob("*/result.json")):
        result = json.loads(path.read_text())
        config = result["config"]
        row = {
            key: config[key]
            for key in ("identifier", "backbone", "loss", "target", "learning_rate")
        }
        row["step_seconds"] = result["mean_step_seconds"]
        for split, metrics in result["final"].items():
            for key in (
                "native_loss",
                "mae",
                "rmse",
                "pearson",
                "spearman",
                "expected_reward",
                "bias",
            ):
                row[f"{split}_{key}"] = metrics[key]
        rows.append(row)
    return pl.DataFrame(rows, infer_schema_length=None)


def plot_selected(*, root: Path, selected: pl.DataFrame, path: Path) -> None:
    labels = read_labels(root=root).select("problem_id", "split", "p_hat")
    figure, axes = plt.subplots(
        1,
        selected.height,
        figsize=(3.6 * selected.height, 3.8),
        constrained_layout=True,
        squeeze=False,
    )
    for axis, identifier in zip(axes[0], selected["identifier"], strict=True):
        predictions = pl.read_parquet(
            root / "heads" / identifier / "predictions.parquet"
        )
        final = predictions.filter(pl.col("step") == predictions["step"].max()).join(
            labels, on="problem_id"
        )
        for color, split in zip(SERIES_COLORS, ("val", "test"), strict=False):
            frame = final.filter(pl.col("split") == split)
            axis.scatter(
                frame["p_hat"],
                frame["q"],
                s=14,
                color=color,
                alpha=0.8,
                edgecolors="white",
                linewidths=0.5,
                label=split,
            )
        axis.plot([0, 1], [0, 1], color="#52514e", linewidth=1, linestyle="--")
        axis.set(
            xlim=(-0.02, 1.02),
            ylim=(-0.02, 1.02),
            xlabel="p̂ (mean@256)",
            title=identifier,
        )
        axis.title.set_fontsize(9)
        axis.spines[["top", "right"]].set_visible(False)
        axis.legend(frameon=False, fontsize=8)
    axes[0, 0].set_ylabel("predicted q")
    figure.savefig(path, dpi=150)
    plt.close(figure)


def report(*, root: Path) -> None:
    results = result_rows(root=root)
    selected = (
        results
        .sort("val_native_loss")
        .group_by("backbone", "loss", "target", maintain_order=True)
        .first()
        .sort("backbone", "loss", "target")
    )
    features = torch.load(root / "heads" / "features.pt")
    labels = read_labels(root=root).sort("problem_id")
    zero_shot = torch.sigmoid(features["zero_shot_logit"]).numpy()
    baseline = [
        {
            "split": split,
            **prediction_metrics(q=zero_shot[mask], p=labels["p_hat"].to_numpy()[mask]),
        }
        for split in SPLITS
        for mask in [labels["split"].to_numpy() == split]
    ]
    plot_selected(
        root=root, selected=selected, path=root / "heads" / "selected-scatter.png"
    )
    columns = [
        "identifier",
        "step_seconds",
        *[
            f"{split}_{key}"
            for split in SPLITS
            for key in ("mae", "rmse", "pearson", "spearman")
        ],
        "val_native_loss",
        "test_expected_reward",
        "test_bias",
    ]
    lines = [
        "# Phase 2 linear-head baselines",
        "",
        "Selection: lowest final validation native loss per (backbone, loss, target).",
        "",
        "## Selected runs",
        "",
        markdown_table(frame=selected.select(columns)),
        "",
        "## Untrained non-thinking Yes/No readout (the shared starting point)",
        "",
        markdown_table(
            frame=pl.DataFrame(baseline).select(
                "split", "mae", "rmse", "pearson", "spearman", "bias"
            )
        ),
        "",
        "## All runs",
        "",
        markdown_table(
            frame=results.sort("backbone", "loss", "target", "learning_rate").select(
                "identifier",
                "val_native_loss",
                "train_mae",
                "val_mae",
                "test_mae",
                "test_pearson",
            )
        ),
        "",
        "![selected](selected-scatter.png)",
        "",
    ]
    (root / "heads" / "phase2.md").write_text("\n".join(lines))
    write_json(
        path=root / "heads" / "summary.json",
        payload={
            "selected": selected.to_dicts(),
            "zero_shot": baseline,
            "max_check_error": features["max_check_error"],
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        choices=("features", "train-frozen", "train-full", "queue-full", "report"),
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--gpu", type=int)
    parser.add_argument("--identifier")
    parser.add_argument("--gpus", type=int)
    parser.add_argument("--cpu-offload", action="store_true")
    parser.add_argument("--poll-seconds", type=float, default=300.0)
    args = parser.parse_args()
    if args.command == "features":
        extract_features(root=args.root, gpu=args.gpu)
    elif args.command == "train-frozen":
        train_frozen(root=args.root)
    elif args.command == "train-full":
        train_full(
            root=args.root, identifier=args.identifier, cpu_offload=args.cpu_offload
        )
    elif args.command == "queue-full":
        queue_full(
            root=args.root,
            gpus=args.gpus,
            cpu_offload=args.cpu_offload,
            poll_seconds=args.poll_seconds,
        )
    else:
        report(root=args.root)


if __name__ == "__main__":
    main()
