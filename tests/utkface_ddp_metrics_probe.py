import json
from pathlib import Path

import torch
from scipy.stats import spearmanr
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.experiments.utkface.config import RunConfig, canonical_config
from src.experiments.utkface.distributed import distributed_context
from src.experiments.utkface.run import evaluate


def main() -> None:
    base = canonical_config(manifest=Path("unused"), output_dir=Path("unused"))
    config = RunConfig.model_validate({
        **base.model_dump(),
        "device": "cpu",
        "precision": "fp32",
    })
    targets = torch.arange(65) * 7 % 100
    medians = (targets + (torch.arange(65) % 7) - 3).clamp(0, 99)
    logits = torch.full((65, 100), -20.0)
    logits[torch.arange(65), medians] = 20
    with distributed_context(device="cpu") as context:
        indices = torch.arange(65)[context.rank :: context.world_size]
        loader = DataLoader(
            TensorDataset(logits[indices], targets[indices], indices), batch_size=8
        )
        result = evaluate(
            model=nn.Identity(), loader=loader, config=config, context=context
        )
        assert result["count"] == 65
        assert (
            abs(
                result["median_spearman"]
                - spearmanr(medians.tolist(), targets.tolist()).statistic
            )
            < 1e-12
        )
        expected_nll = torch.nn.functional.cross_entropy(logits, targets).item()
        assert abs(result["label_nll"] - expected_nll) < 1e-5
        if context.primary:
            print(
                json.dumps({
                    "world_size": context.world_size,
                    "count": result["count"],
                    "median_spearman": result["median_spearman"],
                    "label_nll": result["label_nll"],
                })
            )


if __name__ == "__main__":
    main()
