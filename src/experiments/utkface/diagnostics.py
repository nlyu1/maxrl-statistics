import argparse
import json
from pathlib import Path

import polars as pl
import torch
import torch.nn.functional as F

from src.experiments.utkface.advantages import batched_advantages
from src.experiments.utkface.config import canonical_config
from src.experiments.utkface.losses import policy_loss
from src.experiments.utkface.run import write_json


def estimator_diagnostics(
    *, device: torch.device, output_dir: Path, repetitions: int
) -> list[dict]:
    torch.manual_seed(2026)
    base = canonical_config(manifest=Path("unused"), output_dir=output_dir).loss
    logits = torch.randn(1, 100, device=device, dtype=torch.float64, requires_grad=True)
    target = torch.tensor([37], device=device)
    probabilities = logits.softmax(-1).detach()[0]
    rows = []
    for estimator in ("rloo", "grpo", "binary_maxrl", "tailrl"):
        method = (
            estimator + "_mae_population"
            if estimator in {"rloo", "grpo"}
            else estimator + "_population"
        )
        config = base.model_copy(update={"method": method})
        loss = policy_loss(logits=logits, targets=target, config=config)
        exact = torch.autograd.grad(loss, logits, retain_graph=True)[0][0]
        for count in (8, 64, 512, 4096):
            actions = torch.multinomial(
                probabilities, repetitions * count, replacement=True
            ).reshape(repetitions, count)
            rewards = (
                (actions == target).double()
                if estimator == "binary_maxrl"
                else 1 - (actions - target).abs().double() / 99
            )
            advantages = batched_advantages(
                rewards=rewards,
                estimator=estimator,
                survival_floor=0.0,
                std_floor=base.std_floor,
            )
            estimated = torch.zeros(
                repetitions, 100, device=device, dtype=torch.float64
            )
            estimated.scatter_add_(dim=-1, index=actions, src=-advantages / count)
            estimated += advantages.mean(-1, keepdim=True) * probabilities
            mean = estimated.mean(0)
            rows.append({
                "estimator": estimator,
                "rollouts": count,
                "repetitions": repetitions,
                "cosine": F.cosine_similarity(mean, exact, dim=0).item(),
                "relative_error": ((mean - exact).norm() / exact.norm()).item(),
                "norm_ratio": (mean.norm() / exact.norm()).item(),
                "rms_standard_error": (
                    estimated.var(0, unbiased=True).sum() / repetitions
                )
                .sqrt()
                .item(),
                "exact_gradient_norm": exact.norm().item(),
                "binary_expected_attenuation": 1
                - (1 - probabilities[target.item()].item()) ** (count - 1)
                if estimator == "binary_maxrl"
                else 1.0,
            })
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(path=output_dir / "estimators.json", payload=rows)
    pl.DataFrame(rows).write_csv(output_dir / "estimators.csv")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=2048)
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("Run these diagnostics on a CUDA GPU")
    with (
        torch.cuda.device(device),
        torch.device(device),
        torch.autocast(device_type="cuda", dtype=torch.bfloat16),
    ):
        rows = estimator_diagnostics(
            device=device, output_dir=args.output_dir, repetitions=args.repetitions
        )
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
