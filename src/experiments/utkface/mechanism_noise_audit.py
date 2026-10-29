import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl
import torch

from src.experiments.utkface.adaptation_metrics import prediction_values
from src.experiments.utkface.adaptation_train import (
    file_sha256,
    json_sha256,
    source_root,
)
from src.experiments.utkface.config import NoiseConfig
from src.experiments.utkface.data import read_manifest, training_indices
from src.experiments.utkface.losses import NUM_BINS, log_policy
from src.experiments.utkface.mechanism_train import tensor_sha256 as tensor_hash
from src.experiments.utkface.noise import noisy_labels
from src.experiments.utkface.policy_centering_analysis import demean
from src.experiments.utkface.run import write_json

REFERENCE_SEED_START = 80_000_000
BANDS = {
    "all": (0, 99),
    "ages_0_19": (0, 19),
    "ages_20_79": (20, 79),
    "ages_80_99": (80, 99),
}
ADJUSTMENTS = ("raw", "within_exact_age")
AUDIT_SOURCES = (
    "mechanism_noise_audit.py",
    "noise.py",
    "policy_centering_analysis.py",
    "adaptation_metrics.py",
    "losses.py",
    "mechanism_train.py",
)


def checked_archive(*, path: Path) -> tuple[dict, dict, dict]:
    archive = torch.load(path, map_location="cpu", weights_only=True)
    if archive["schema"] != "utkface-mechanism-features-v1":
        raise ValueError(f"Unsupported feature archive: {path}")
    provenance = archive["provenance"]
    source = provenance["source"]
    config = source["training_provenance"]["mechanism_config"]
    run = config["adaptation"]["run"]
    manifest = read_manifest(path=Path(run["manifest"]))
    identity = {
        "manifest_fingerprint": manifest["fingerprint"],
        "manifest_sha256": file_sha256(path=Path(run["manifest"])),
    }
    if any(provenance[key] != value for key, value in identity.items()):
        raise ValueError("Feature archive manifest changed")
    for key in ("checkpoint", "config"):
        if file_sha256(path=Path(source[f"{key}_path"])) != source[f"{key}_sha256"]:
            raise ValueError(f"Archived source {key} changed")
    result_path = Path(run["output_dir"]) / "result.json"
    if file_sha256(path=result_path) != source["result_sha256"]:
        raise ValueError("Archived source result changed")
    if json.loads(Path(source["config_path"]).read_text()) != config:
        raise ValueError("Archived source config differs")
    actual = {
        split: {name: tensor_hash(values=value) for name, value in tensors.items()}
        for split, tensors in archive["splits"].items()
    }
    if actual != archive["tensor_sha256"] or set(actual) != {"train", "val", "test"}:
        raise ValueError("Archive tensor hashes or splits differ")
    clean = torch.tensor([r["age"] for r in manifest["records"]], device="cpu")
    noise = NoiseConfig.model_validate(run["noise"])
    if noise.mode not in {"none", "fixed"}:
        raise ValueError("Noise alignment requires a fixed training field")
    labels = noisy_labels(ages=clean.tolist(), config=noise, epoch=1)
    train = training_indices(
        payload=manifest, limit=run["train_limit"], subset_seed=run["subset_seed"]
    )
    for split, values in archive["splits"].items():
        indices = values["indices"]
        expected_indices = (
            train if split == "train" else sorted(manifest["splits"][split])
        )
        expected_labels = labels[indices] if split == "train" else clean[indices]
        if (
            indices.tolist() != expected_indices
            or not torch.equal(values["clean_ages"], clean[indices])
            or not torch.equal(values["supplied_labels"], expected_labels)
        ):
            raise ValueError(f"Archive images or labels differ: {split}")
        if tensor_hash(values=indices) != provenance["indices_sha256"][split]:
            raise ValueError("Archived index hash differs")
    if (
        json_sha256(train) != source["training_provenance"]["training_indices_sha256"]
        or tensor_hash(values=labels[train]) != source["source_training_labels_sha256"]
    ):
        raise ValueError("Archive does not reproduce source training supervision")
    return archive, config, manifest


def reference_fields(
    *, manifest: dict, std: float, seeds: list[int], output_dir: Path
) -> dict:
    metadata = {
        "manifest_fingerprint": manifest["fingerprint"],
        "std": std,
        "seeds": seeds,
    }
    path = output_dir / f"reference-fields-{json_sha256(metadata)[:16]}.pt"
    ages = [row["age"] for row in manifest["records"]]
    fields = []
    for seed in seeds:
        noise = NoiseConfig(mode="fixed" if std else "none", std_years=std, seed=seed)
        fields.append(noisy_labels(ages=ages, config=noise, epoch=1))
    labels = torch.stack(fields)
    cached = {
        "metadata": metadata,
        "labels": labels,
        "labels_sha256": tensor_hash(values=labels),
    }
    torch.save(cached, path)
    return {
        **cached,
        "path": str(path.resolve()),
        "sha256": file_sha256(path=path),
    }


def alignment(
    *,
    noise: np.ndarray,
    error: np.ndarray,
    controls: np.ndarray,
    ages: np.ndarray,
    adjustment: str,
) -> dict:
    if len(noise) == 0:
        return {"defined": False, "slope": None, "reference_slopes": []}
    varied = float(noise.var()) > 1e-12
    if adjustment == "within_exact_age":
        noise, error, controls = (
            demean(values=x, ages=ages) for x in (noise, error, controls)
        )
    noise, error = noise - noise.mean(), error - error.mean()
    controls = controls - controls.mean(1, keepdims=True)
    denominator = float(noise @ noise)
    defined = varied and denominator > 1e-12
    slope = float(noise @ error / denominator) if defined else None
    denominators = (controls**2).sum(1)
    reference = [
        float(x @ error / d) if d > 1e-12 else None
        for x, d in zip(controls, denominators, strict=True)
    ]
    valid = [value for value in reference if value is not None]
    quantiles = np.quantile(valid, [0.025, 0.5, 0.975]) if valid else [None] * 3
    return {
        "defined": defined,
        "slope": slope,
        "noise_variance": denominator / len(noise),
        "covariance": float(noise @ error / len(noise)),
        "reference_defined_fields": len(valid),
        **{
            f"reference_q{name}": value
            for name, value in zip(("025", "50", "975"), quantiles, strict=True)
        },
        "reference_slopes": reference,
    }


def source_rows(*, archive: dict, config: dict, fields: dict) -> list[dict]:
    source = archive["provenance"]["source"]
    scalar = config["adaptation"]["run"]["head"] == "scalar"
    values = archive["splits"]["train"]
    ages, labels = (
        values[name].numpy().astype(np.float64)
        for name in ("clean_ages", "supplied_labels")
    )
    outputs = values["native_outputs"]
    means = prediction_values(outputs=outputs, scalar=scalar)[:, 0].double().numpy()
    if not np.isfinite(means).all():
        raise ValueError("Archived native means are non-finite")
    entropy, variance = None, None
    if not scalar:
        lp = log_policy(logits=values["native_outputs"].double()).numpy()
        p = np.exp(lp)
        entropy = -(p * lp).sum(1)
        variance = (
            p * (np.arange(NUM_BINS)[None] - (p @ np.arange(NUM_BINS))[:, None]) ** 2
        ).sum(1)
    controls = fields["labels"][:, values["indices"]].numpy().astype(np.float64) - ages
    identity = {
        key: source[key]
        for key in ("identifier", "block", "noise_sd", "variant", "checkpoint", "epoch")
    }
    rows = []
    for band, (low, high) in BANDS.items():
        mask = (ages >= low) & (ages <= high)
        count = int(mask.sum())
        metrics = {"count": count, "split": "train", "band": band}
        for name, targets in (("clean", ages), ("supplied", labels)):
            error = means[mask] - targets[mask]
            metrics[f"{name}_mae"] = float(np.abs(error).mean()) if count else None
            metrics[f"{name}_mse"] = float((error**2).mean()) if count else None
        metrics.update({
            "entropy": float(entropy[mask].mean()) if count and not scalar else None,
            "policy_variance": float(variance[mask].mean())
            if count and not scalar
            else None,
        })
        for adjustment in ADJUSTMENTS:
            fit = alignment(
                noise=labels[mask] - ages[mask],
                error=means[mask] - ages[mask],
                controls=controls[:, mask],
                ages=ages[mask],
                adjustment=adjustment,
            )
            rows.append({**identity, **metrics, "adjustment": adjustment, **fit})
    return rows


def write_table(*, path: Path, rows: list[dict]) -> None:
    pl.DataFrame(rows, infer_schema_length=None).drop("reference_slopes").write_csv(
        path, separator="\t"
    )


def audit(*, probe_dir: Path, output_dir: Path, reference_count: int) -> dict:
    paths = sorted((probe_dir / "features").glob("*.pt"))
    if reference_count < 1 or not paths:
        raise ValueError("Require retained archives and at least one reference field")
    seeds = list(range(REFERENCE_SEED_START, REFERENCE_SEED_START + reference_count))
    for path in paths:
        archive = torch.load(path, map_location="cpu", weights_only=True)
        noise = archive["provenance"]["source"]["training_provenance"][
            "mechanism_config"
        ]["adaptation"]["run"]["noise"]
        if noise["seed"] in seeds:
            raise ValueError("Predeclared reference seeds overlap a training field")
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "schema": "utkface-mechanism-noise-audit-v1",
        "probe_dir": str(probe_dir.resolve()),
        "reference_seeds": seeds,
        "reference_scope": "Conditional randomization references over unused fields, given fixed faces and predictions; not training uncertainty, confidence intervals, or exact randomization p-values.",
        "reference_generation": "Existing noisy_labels over FULL manifest ages, then select archived global indices.",
        "slope_definition": "Cov(prediction-clean_age, supplied-clean_age)/Var(supplied-clean_age); within_exact_age demeans both variables separately within each exact reference age.",
        "audit_source_sha256": {
            name: file_sha256(path=source_root() / "src/experiments/utkface" / name)
            for name in AUDIT_SOURCES
        },
        "sources": [],
    }
    field_cache, all_rows = {}, []
    with torch.device("cpu"), torch.inference_mode():
        for path in paths:
            archive, config, manifest = checked_archive(path=path)
            std = config["adaptation"]["run"]["noise"]["std_years"]
            key = (manifest["fingerprint"], std)
            if key not in field_cache:
                field_cache[key] = reference_fields(
                    manifest=manifest, std=std, seeds=seeds, output_dir=output_dir
                )
            fields = field_cache[key]
            rows = source_rows(archive=archive, config=config, fields=fields)
            entry = {
                "archive": str(path.resolve()),
                "archive_sha256": file_sha256(path=path),
                "provenance": archive["provenance"],
                "tensor_sha256": archive["tensor_sha256"],
                "reference_artifact": {
                    key: fields[key]
                    for key in ("path", "sha256", "metadata", "labels_sha256")
                },
                "rows": rows,
            }
            write_json(path=output_dir / f"{path.stem}.json", payload=entry)
            write_table(path=output_dir / f"{path.stem}.tsv", rows=rows)
            report["sources"].append(entry)
            all_rows.extend(rows)
    write_json(path=output_dir / "report.json", payload=report)
    write_table(path=output_dir / "summary.tsv", rows=all_rows)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reference-fields", type=int, default=128)
    args = parser.parse_args()
    audit(
        probe_dir=args.probe_dir,
        output_dir=args.output_dir,
        reference_count=args.reference_fields,
    )


if __name__ == "__main__":
    main()
