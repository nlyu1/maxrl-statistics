from math_verify.errors import TimeoutException
from math_verify.metric import math_metric
from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig

VERIFY = math_metric(
    gold_extraction_target=(LatexExtractionConfig(),),
    pred_extraction_target=(ExprExtractionConfig(), LatexExtractionConfig()),
)


def score(*, response: str, answer: str) -> tuple[float, bool]:
    try:
        value, _ = VERIFY(["\\boxed{" + answer + "}"], [response])
    except (Exception, TimeoutException):  # noqa: BLE001
        return 0.0, True
    return float(value), False


def score_pair(pair: tuple[str, str]) -> tuple[float, bool]:
    return score(response=pair[0], answer=pair[1])
