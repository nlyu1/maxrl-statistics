import argparse
import json
import time
from pathlib import Path

import polars as pl
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt

from src.experiments.pass_rate_calibration.config import SOLVER


def solver_tokens(*, tokenizer, problem: str) -> list[int]:
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": problem + SOLVER.suffix}],
        tokenize=False,
        add_generation_prompt=True,
    )
    if not text.endswith("<|im_start|>assistant\n") or "<think>" in text:
        raise ValueError(f"Unexpected solver prompt rendering: {text[-120:]!r}")
    return tokenizer.encode(text, add_special_tokens=False)


def chunk_path(*, root: Path, shard: int, shards: int, chunk: int) -> Path:
    return root / "rollouts" / f"shard{shard}-of{shards}-chunk{chunk:03d}.parquet"


def generate(*, root: Path, shard: int, shards: int) -> None:
    problems = (
        pl
        .read_parquet(root / "problems.parquet")
        .sort("problem_id")
        .with_row_index("seed_offset")
        .gather_every(shards, offset=shard)
    )
    (root / "rollouts").mkdir(exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(
        SOLVER.repository, revision=SOLVER.revision
    )
    llm = LLM(
        model=SOLVER.repository,
        revision=SOLVER.revision,
        tokenizer_revision=SOLVER.revision,
        dtype="bfloat16",
        seed=SOLVER.seed,
        max_model_len=SOLVER.max_model_len,
        gpu_memory_utilization=SOLVER.gpu_memory_utilization,
        generation_config="vllm",
    )
    for chunk, start in enumerate(range(0, problems.height, SOLVER.chunk_problems)):
        path = chunk_path(root=root, shard=shard, shards=shards, chunk=chunk)
        if path.exists():
            continue
        rows = problems.slice(start, SOLVER.chunk_problems)
        prompts = [
            TokensPrompt(
                prompt_token_ids=solver_tokens(tokenizer=tokenizer, problem=text)
            )
            for text in rows["problem"]
        ]
        parameters = [
            SamplingParams(
                n=SOLVER.samples,
                temperature=SOLVER.temperature,
                top_p=SOLVER.top_p,
                top_k=SOLVER.top_k,
                max_tokens=SOLVER.max_tokens,
                stop_token_ids=list(SOLVER.stop_token_ids),
                seed=SOLVER.seed + offset,
            )
            for offset in rows["seed_offset"]
        ]
        began = time.time()
        outputs = llm.generate(prompts, parameters, use_tqdm=False)
        records = [
            {
                "problem_id": problem_id,
                "sample": completion.index,
                "response": completion.text,
                "prompt_tokens": len(prompt["prompt_token_ids"]),
                "completion_tokens": len(completion.token_ids),
                "finish_reason": completion.finish_reason,
            }
            for problem_id, prompt, output in zip(
                rows["problem_id"], prompts, outputs, strict=True
            )
            for completion in output.outputs
        ]
        frame = pl.DataFrame(records)
        if frame.height != rows.height * SOLVER.samples:
            raise RuntimeError(f"Incomplete generation for {path.name}")
        temporary = path.with_suffix(".tmp")
        frame.write_parquet(temporary)
        temporary.replace(path)
        print(
            json.dumps({
                "chunk": path.name,
                "problems": rows.height,
                "seconds": round(time.time() - began, 1),
                "completion_tokens": int(frame["completion_tokens"].sum()),
                "truncated": int((frame["finish_reason"] == "length").sum()),
            }),
            flush=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--shard", type=int, required=True)
    parser.add_argument("--shards", type=int, required=True)
    args = parser.parse_args()
    generate(root=args.root, shard=args.shard, shards=args.shards)


if __name__ == "__main__":
    main()
