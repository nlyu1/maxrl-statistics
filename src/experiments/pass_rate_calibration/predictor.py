import math

from transformers import PreTrainedTokenizerBase

from src.experiments.pass_rate_calibration.config import ANSWERS, PREDICTOR_PROMPT


def predictor_tokens(
    *, tokenizer: PreTrainedTokenizerBase, problem: str, thinking: bool
) -> list[int]:
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": PREDICTOR_PROMPT.format(problem=problem)}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=thinking,
    )
    return tokenizer.encode(text, add_special_tokens=False)


def single_token(*, tokenizer: PreTrainedTokenizerBase, text: str) -> int:
    tokens = tokenizer.encode(text, add_special_tokens=False)
    if len(tokens) != 1:
        raise ValueError(f"{text!r} is not a single token: {tokens}")
    return tokens[0]


def answer_tokens(*, tokenizer: PreTrainedTokenizerBase) -> tuple[int, int]:
    yes, no = (single_token(tokenizer=tokenizer, text=answer) for answer in ANSWERS)
    return yes, no


def stop_tokens(*, tokenizer: PreTrainedTokenizerBase) -> list[int]:
    return [
        single_token(tokenizer=tokenizer, text=marker)
        for marker in ("<|im_end|>", "<|endoftext|>")
    ]


def parse_completion(
    *,
    tokenizer: PreTrainedTokenizerBase,
    tokens: list[int],
    logprobs: list[float],
    stop_reason: str,
) -> dict:
    if len(tokens) != len(logprobs):
        raise ValueError("Sampled tokens and logprobs are misaligned")
    stops = set(stop_tokens(tokenizer=tokenizer))
    while tokens and tokens[-1] in stops:
        tokens, logprobs = tokens[:-1], logprobs[:-1]
    think_end = single_token(tokenizer=tokenizer, text="</think>")
    ends = [index for index, token in enumerate(tokens) if token == think_end]
    result = {"answer": None, "confidence": None, "answer_text": None}
    if stop_reason == "length":
        return {**result, "status": "aborted"}
    if not ends:
        return {**result, "status": "malformed"}
    tail = range(ends[-1] + 1, len(tokens))
    text = tokenizer.decode([tokens[index] for index in tail]).strip()
    if text not in ANSWERS:
        return {**result, "status": "malformed", "answer_text": text[:200]}
    content = [index for index in tail if tokenizer.decode([tokens[index]]).strip()]
    return {
        "status": "valid",
        "answer": text,
        "answer_text": text,
        "confidence": math.exp(sum(logprobs[index] for index in content)),
    }
