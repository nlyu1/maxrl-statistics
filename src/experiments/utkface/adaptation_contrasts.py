import argparse
import json
from pathlib import Path

import numpy as np

from src.experiments.utkface.run import write_json

LOSSES = ("cat_mse", "ce", "tail", "grpo_mse", "scalar_mae")


def frozen_id(*, source: str, std: int, loss: str) -> str:
    if source == "imagenet":
        return f"aug_original_frozen_{loss}_sd{std}"
    return f"head_{source}_sd{std}_{loss}"


def collect_contrasts(*, root: Path) -> list[dict]:
    rows = []

    def add(*, suite: str, first: str, second: str, **fields):
        rows.append({"suite": suite, "first": first, "second": second, **fields})

    for pipeline in ("original", "fixed"):
        for mode in ("full", "frozen"):
            for std in (0, 5, 10):
                prefix = f"aug_{pipeline}_{mode}"
                add(
                    suite="augmentation_tail_minus_mse",
                    pipeline=pipeline,
                    mode=mode,
                    std=std,
                    first=f"{prefix}_tail_sd{std}",
                    second=f"{prefix}_cat_mse_sd{std}",
                )
    for mode in ("full", "frozen"):
        for std in (0, 5, 10):
            for loss in LOSSES:
                add(
                    suite="fixed_minus_original",
                    mode=mode,
                    std=std,
                    loss=loss,
                    first=f"aug_fixed_{mode}_{loss}_sd{std}",
                    second=f"aug_original_{mode}_{loss}_sd{std}",
                )
    for source in ("imagenet", "cat_mse", "ce"):
        for std in (0, 1, 3, 5, 10):
            add(
                suite="frozen_tail_minus_mse",
                source=source,
                std=std,
                first=frozen_id(source=source, std=std, loss="tail"),
                second=frozen_id(source=source, std=std, loss="cat_mse"),
            )
    for std in (0, 1, 3, 5, 10):
        for loss in LOSSES:
            add(
                suite="ce_features_minus_sl_features",
                std=std,
                loss=loss,
                first=frozen_id(source="ce", std=std, loss=loss),
                second=frozen_id(source="cat_mse", std=std, loss=loss),
            )
    for mode in ("full", "frozen"):
        for std in (0, 5, 10):
            for loss in ("ce", "tail", "grpo_mse"):
                add(
                    suite="post_minus_mse_control",
                    mode=mode,
                    std=std,
                    loss=loss,
                    first=f"post_sl25_sd{std}_{mode}_{loss}",
                    second=f"post_sl25_sd{std}_{mode}_cat_mse",
                )
            for loss in LOSSES[:4]:
                add(
                    suite="post_minus_original_sl",
                    mode=mode,
                    std=std,
                    loss=loss,
                    first=f"post_sl25_sd{std}_{mode}_{loss}",
                    second=f"aug_original_full_cat_mse_sd{std}",
                )
    for std in (0, 5, 10):
        for loss in LOSSES:
            add(
                suite="scratch_constant_minus_cosine",
                std=std,
                loss=loss,
                first=f"scratch_warmup_constant_{loss}_sd{std}",
                second=f"scratch_cosine_{loss}_sd{std}",
            )
        for recipe in ("scratch_cosine", "scratch_warmup_constant"):
            add(
                suite="scratch_tail_minus_mse",
                recipe=recipe,
                std=std,
                first=f"{recipe}_tail_sd{std}",
                second=f"{recipe}_cat_mse_sd{std}",
            )
    plan = json.loads((root / "plan.json").read_text())
    known = {job["identifier"] for job in plan["jobs"] if job["kind"] == "train"}
    if any(row[side] not in known for row in rows for side in ("first", "second")):
        raise ValueError("Contrast references an unknown training job")
    return rows


def report(*, root: Path, output_dir: Path, draws: int) -> dict:
    plan = json.loads((root / "plan.json").read_text())
    jobs = {job["identifier"]: job for job in plan["jobs"] if job["kind"] == "train"}
    results = {}
    errors = {}
    identity = None
    for identifier, job in jobs.items():
        directory = Path(job["output_dir"])
        result = json.loads((directory / "result.json").read_text())
        prediction = json.loads((directory / "best-test-predictions.json").read_text())
        values = sorted(prediction["rows"], key=lambda row: row["manifest_index"])
        current = [(row["manifest_index"], row["target"]) for row in values]
        if identity is None:
            identity = current
        elif current != identity:
            raise ValueError(f"Paired test images differ: {identifier}")
        error = np.asarray([abs(row["mean"] - row["target"]) for row in values])
        if abs(float(error.mean()) - result["best_test"]["mae_mean"]) > 1e-4:
            raise ValueError(f"Archived predictions differ from result: {identifier}")
        results[identifier] = result
        errors[identifier] = error
    contrasts = collect_contrasts(root=root)
    differences = np.stack([
        errors[row["first"]] - errors[row["second"]] for row in contrasts
    ])
    rng = np.random.default_rng(20261026)
    estimates = np.empty((len(contrasts), draws))
    n = differences.shape[1]
    for start in range(0, draws, 128):
        count = min(128, draws - start)
        indices = rng.integers(0, n, size=(count, n), dtype=np.int32)
        estimates[:, start : start + count] = differences[:, indices].mean(-1)
    for index, row in enumerate(contrasts):
        first = results[row["first"]]["best_test"]
        second = results[row["second"]]["best_test"]
        row.update({
            "first_mae": first["mae_mean"],
            "second_mae": second["mae_mean"],
            "mae_difference": float(differences[index].mean()),
            "bootstrap_95": np.quantile(estimates[index], [0.025, 0.975]).tolist(),
            "rmse_difference": first["rmse_mean"] - second["rmse_mean"],
            "spearman_difference": (
                first["spearman_mean"] - second["spearman_mean"]
                if first["spearman_mean"] is not None
                and second["spearman_mean"] is not None
                else None
            ),
        })
    summary = {
        "contrasts": contrasts,
        "test_images": n,
        "draws": draws,
        "uncertainty_scope": "Paired test-image bootstrap conditional on the trained models, one training seed and one fixed corruption field; intervals are unadjusted for multiplicity.",
        "direction": "Negative first-minus-second MAE favors first.",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(path=output_dir / "contrasts.json", payload=summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=5000)
    args = parser.parse_args()
    result = report(root=args.root, output_dir=args.output_dir, draws=args.draws)
    print(
        json.dumps({
            "contrasts": len(result["contrasts"]),
            "test_images": result["test_images"],
            "draws": args.draws,
        })
    )


if __name__ == "__main__":
    main()
