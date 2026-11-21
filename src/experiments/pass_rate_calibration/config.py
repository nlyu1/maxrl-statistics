from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from src.experiments.bernoulli_calibration.config import Protocol

STATE_ROOT = Path.home() / ".local/state/maxrl-statistics"
WANDB_PROJECT = "maxrl-pass-rate-calibration"
SERIES_COLORS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")

DatasetName = Literal["minerva", "jeebench"]
Split = Literal["train", "val", "test"]
SPLITS: tuple[Split, ...] = ("train", "val", "test")
Loss = Literal["ce", "mse"]
Target = Literal["binary", "soft"]
Backbone = Literal["frozen", "full"]
PolicyMethod = Literal["rloo", "grpo", "maxrl"]
RLSweep = Literal["smoke", "smoke-resume", "lr-pilot", "long"]


class Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class DatasetSource(Frozen):
    name: DatasetName
    repository: str
    revision: str
    split: str
    question_field: str
    reference_pass_at_1: float = Field(gt=0, lt=1)


class SolverConfig(Frozen):
    repository: str
    revision: str
    suffix: str
    temperature: float
    top_p: float
    top_k: int
    max_tokens: int
    stop_token_ids: tuple[int, ...]
    samples: int
    seed: int
    max_model_len: int
    gpu_memory_utilization: float
    chunk_problems: int
    vllm_version: str
    gate_tolerance: float


class PredictorModel(Frozen):
    name: Literal["qwen3-8b", "qwen3.5-4b"]
    repository: str
    revision: str
    prefill_price: float
    cached_prefill_price: float
    sample_price: float
    train_price: float


class PilotConfig(Frozen):
    problems: int
    samples: int
    max_tokens: int
    temperature: float
    seed: int
    phase3_problems_per_step: int
    phase3_rollouts: tuple[int, ...]
    phase3_steps: int


class HeadRun(Frozen):
    identifier: str
    backbone: Backbone
    loss: Loss
    target: Target
    learning_rate: float = Field(gt=0)
    steps: int = Field(gt=0)
    problems_per_step: int = Field(gt=0)
    eval_every: int = Field(gt=0)
    seed: int


class RLRun(Frozen):
    identifier: str
    method: PolicyMethod
    protocol: Protocol
    rollouts: int = Field(ge=2)
    learning_rate: float = Field(gt=0)
    steps: int = Field(gt=0)
    problems_per_step: int = Field(gt=0)
    eval_every: int = Field(gt=0)
    eval_samples: int = Field(gt=0)
    test_subset_size: int = Field(gt=0)
    lora_rank: int = Field(gt=0)
    max_tokens: int = Field(gt=0)
    test_max_tokens: int = Field(gt=0)
    temperature: float = Field(gt=0)
    std_floor: float = Field(gt=0)
    seed: int


SOURCES = (
    DatasetSource(
        name="minerva",
        repository="math-ai/minervamath",
        revision="ee46ddc498933b1977577953250ca5c66be64f96",
        split="test",
        question_field="question",
        reference_pass_at_1=0.295,
    ),
    DatasetSource(
        name="jeebench",
        repository="daman1209arora/jeebench_numeric",
        revision="39623e54b6753784e191f393a764b8585d22313b",
        split="test",
        question_field="problem",
        reference_pass_at_1=0.25,
    ),
)

SOLVER = SolverConfig(
    repository="ftajwar/qwen3_1.7B_Base_MaxRL_Polaris_1000_steps",
    revision="a3d4b1f9c1fe3a85fb00ee257a6cfbf79511325c",
    suffix="\nPlease reason step by step, and put your final answer within \\boxed{{}}.",
    temperature=0.6,
    top_p=0.95,
    top_k=-1,
    max_tokens=4096,
    stop_token_ids=(151643, 151645),
    samples=256,
    seed=20261120,
    max_model_len=8192,
    gpu_memory_utilization=0.88,
    chunk_problems=16,
    vllm_version="0.32.0",
    gate_tolerance=0.05,
)

VALIDATION_PROBLEMS = 32
SPLIT_SEED = 20261120

PREDICTOR_PROMPT = (
    "An independent small 1.7B base model will try to solve the following problem. "
    "You should predict whether that model will get a correct answer. Do not solve "
    "the question yourself. Respond with exactly one word: Yes or No.\n\n"
    "Problem to be solved by the other model:\n```\n{problem}\n```"
)
ANSWERS = ("Yes", "No")

PREDICTORS = {
    "qwen3-8b": PredictorModel(
        name="qwen3-8b",
        repository="Qwen/Qwen3-8B",
        revision="b968826d9c46dd6066d109eabc6255188de91218",
        prefill_price=0.195,
        cached_prefill_price=0.039,
        sample_price=0.60,
        train_price=0.44,
    ),
    "qwen3.5-4b": PredictorModel(
        name="qwen3.5-4b",
        repository="Qwen/Qwen3.5-4B",
        revision="851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
        prefill_price=0.33,
        cached_prefill_price=0.066,
        sample_price=1.005,
        train_price=0.737,
    ),
}
HEAD_BACKBONE = PREDICTORS["qwen3-8b"]

PILOT = PilotConfig(
    problems=32,
    samples=8,
    max_tokens=4096,
    temperature=1.0,
    seed=20261120,
    phase3_problems_per_step=16,
    phase3_rollouts=(16, 64, 128),
    phase3_steps=300,
)

HEAD_STEPS = 300
HEAD_PROBLEMS_PER_STEP = 16
HEAD_EVAL_EVERY = 15
HEAD_SEED = 42
FROZEN_LEARNING_RATES = (1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2)
FULL_LEARNING_RATES = (1e-5, 3e-6)
MAXRL_ROLLOUTS = (8, 16, 32, 64, 128)


def head_runs(
    *, backbone: Backbone, learning_rates: tuple[float, ...]
) -> list[HeadRun]:
    return [
        HeadRun(
            identifier=f"{backbone}-{loss}-{target}-lr{rate:g}",
            backbone=backbone,
            loss=loss,
            target=target,
            learning_rate=rate,
            steps=HEAD_STEPS,
            problems_per_step=HEAD_PROBLEMS_PER_STEP,
            eval_every=HEAD_EVAL_EVERY,
            seed=HEAD_SEED,
        )
        for rate in learning_rates
        for loss in ("ce", "mse")
        for target in ("binary", "soft")
    ]


RL_PREDICTOR = PREDICTORS["qwen3-8b"]
RL_LORA_RANK = 32
RL_STD_FLOOR = 1e-8
RL_SEED = 20261121
RL_TEST_MAX_TOKENS = 8192
RL_TEST_SUBSET_SIZE = 48
ENOTATION_PATTERN = r"[-+]?(\d+\.?\d*|\.\d+)[eE][-+]?\d+"


def rl_run(
    *,
    method: PolicyMethod,
    protocol: Protocol,
    rollouts: int,
    learning_rate: float,
    steps: int,
    problems_per_step: int,
    eval_every: int,
    eval_samples: int,
) -> RLRun:
    if method != "maxrl" and protocol != "shared":
        raise ValueError("Only MaxRL distinguishes shared and split labels")
    return RLRun(
        identifier=f"{method}-{protocol}-n{rollouts}-lr{learning_rate:g}",
        method=method,
        protocol=protocol,
        rollouts=rollouts,
        learning_rate=learning_rate,
        steps=steps,
        problems_per_step=problems_per_step,
        eval_every=eval_every,
        eval_samples=eval_samples,
        test_subset_size=RL_TEST_SUBSET_SIZE,
        lora_rank=RL_LORA_RANK,
        max_tokens=PILOT.max_tokens,
        test_max_tokens=RL_TEST_MAX_TOKENS,
        temperature=PILOT.temperature,
        std_floor=RL_STD_FLOOR,
        seed=RL_SEED,
    )


def rl_sweep(*, name: RLSweep) -> list[RLRun]:
    if name in ("smoke", "smoke-resume"):
        return [
            rl_run(
                method="maxrl",
                protocol="shared",
                rollouts=4,
                learning_rate=1e-5,
                steps=2 if name == "smoke" else 3,
                problems_per_step=2,
                eval_every=1,
                eval_samples=2,
            )
        ]
    if name == "long":
        return [
            rl_run(
                method=method,
                protocol=protocol,
                rollouts=16,
                learning_rate=1e-5,
                steps=300,
                problems_per_step=16,
                eval_every=15,
                eval_samples=16,
            )
            for method, protocol in (
                ("rloo", "shared"),
                ("maxrl", "shared"),
                ("maxrl", "split"),
            )
        ]
    grid: tuple[tuple[PolicyMethod, Protocol, tuple[float, ...]], ...] = (
        ("rloo", "shared", (5e-6, 1e-5, 3e-5, 1e-4)),
        ("maxrl", "shared", (5e-6, 1e-5, 3e-5, 1e-4)),
        ("maxrl", "split", (5e-6, 1e-5, 3e-5, 1e-4)),
    )
    return [
        rl_run(
            method=method,
            protocol=protocol,
            rollouts=16,
            learning_rate=learning_rate,
            steps=30,
            problems_per_step=16,
            eval_every=15,
            eval_samples=16,
        )
        for method, protocol, learning_rates in grid
        for learning_rate in learning_rates
    ]
