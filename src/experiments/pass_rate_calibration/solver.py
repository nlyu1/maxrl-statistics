import argparse
import multiprocessing
import os
import subprocess
import time
from pathlib import Path

import polars as pl

from src.experiments.pass_rate_calibration.config import SOLVER
from src.experiments.pass_rate_calibration.data import prepare_problems, read_problems
from src.experiments.pass_rate_calibration.gpu import claimed_gpus
from src.experiments.pass_rate_calibration.grading import score_pair
from src.experiments.utkface.run import write_json

WORKER = (
    "uv",
    "run",
    "--isolated",
    "--no-project",
    "--python",
    "3.12",
    "--with",
    f"vllm=={SOLVER.vllm_version}",
    "--with",
    "polars",
    "--with",
    "python-dotenv",
    "python",
    "-m",
    "src.experiments.pass_rate_calibration.solver_worker",
)


def expected_chunks(*, problems: int, shards: int) -> int:
    return sum(
        -(-len(range(shard, problems, shards)) // SOLVER.chunk_problems)
        for shard in range(shards)
    )


def launch(*, root: Path, gpus: tuple[int, ...]) -> None:
    problems = prepare_problems(root=root)
    (root / "logs").mkdir(parents=True, exist_ok=True)
    with claimed_gpus(indices=gpus) as claimed:
        write_json(
            path=root / "solver.json",
            payload={
                "solver": SOLVER.model_dump(),
                "source_revision": subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], text=True
                ).strip(),
                "gpus": claimed,
                "started_at": time.time(),
            },
        )
        processes = []
        for shard, gpu in enumerate(claimed):
            log = (root / "logs" / f"solver-shard{shard}.log").open("a")
            processes.append(
                subprocess.Popen(
                    [
                        *WORKER,
                        "--root",
                        str(root),
                        "--shard",
                        str(shard),
                        "--shards",
                        str(len(claimed)),
                    ],
                    env={
                        **os.environ,
                        "CUDA_VISIBLE_DEVICES": gpu["uuid"],
                        "VLLM_USAGE_STATS_DISABLED": "1",
                        "DO_NOT_TRACK": "1",
                    },
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
            )
        codes = [process.wait() for process in processes]
    if any(codes):
        raise RuntimeError(f"Solver workers failed with exit codes {codes}")
    written = len(list((root / "rollouts").glob("*.parquet")))
    expected = expected_chunks(problems=problems.height, shards=len(gpus))
    if written != expected:
        raise RuntimeError(f"Expected {expected} rollout chunks, found {written}")


def grade(*, root: Path, processes: int) -> None:
    answers = read_problems(root=root).select("problem_id", "answer")
    rollouts = pl.read_parquet(sorted((root / "rollouts").glob("*.parquet"))).join(
        answers, on="problem_id", how="left"
    )
    if rollouts["answer"].null_count():
        raise ValueError("Rollouts reference unknown problems")
    pairs = list(zip(rollouts["response"], rollouts["answer"], strict=True))
    with multiprocessing.get_context("fork").Pool(processes) as pool:
        scores = pool.map(score_pair, pairs, chunksize=64)
    graded = rollouts.drop("answer").with_columns(
        correct=pl.Series([value for value, _ in scores], dtype=pl.Float64),
        grader_error=pl.Series([error for _, error in scores], dtype=pl.Boolean),
    )
    graded.sort("problem_id", "sample").write_parquet(root / "rollouts.parquet")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "launch", "grade"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--gpus", type=int, nargs="+")
    parser.add_argument("--processes", type=int, default=48)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare_problems(root=args.root)
    elif args.command == "launch":
        launch(root=args.root, gpus=tuple(args.gpus))
    else:
        grade(root=args.root, processes=args.processes)


if __name__ == "__main__":
    main()
