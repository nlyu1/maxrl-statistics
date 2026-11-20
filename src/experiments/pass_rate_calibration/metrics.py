import numpy as np
from scipy.stats import pearsonr, spearmanr


def correlation(*, method, q: np.ndarray, p: np.ndarray) -> float | None:
    if np.ptp(q) == 0 or np.ptp(p) == 0:
        return None
    return float(method(q, p).statistic)


def prediction_metrics(*, q: np.ndarray, p: np.ndarray) -> dict[str, float | None]:
    error = q - p
    return {
        "mae": float(np.abs(error).mean()),
        "rmse": float(np.sqrt((error**2).mean())),
        "pearson": correlation(method=pearsonr, q=q, p=p),
        "spearman": correlation(method=spearmanr, q=q, p=p),
        "expected_reward": float((q * p + (1 - q) * (1 - p)).mean()),
        "bias": float(error.mean()),
        "mean_q": float(q.mean()),
        "mean_p": float(p.mean()),
    }
