from pathlib import Path

import pytest
import torch
from PIL import Image
from scipy.stats import spearmanr

from src.experiments.utkface.config import NoiseConfig, canonical_config
from src.experiments.utkface.data import (
    create_manifest,
    rgb_fingerprint,
    training_indices,
)
from src.experiments.utkface.losses import (
    normalized_kernel_advantages,
    policy_loss,
    scalar_loss,
)
from src.experiments.utkface.metrics import median_spearman
from src.experiments.utkface.model import DigitHead, ScalarHead
from src.experiments.utkface.noise import noise_summary, noisy_labels


def test_deduplication_precedes_split_and_drops_all_conflicting_labels(
    tmp_path: Path,
) -> None:
    image_dir = tmp_path / "UTKFace"
    image_dir.mkdir()
    for decade in range(10):
        for index in range(30):
            Image.new("RGB", (12, 12), (decade * 20, index * 7, 51)).save(
                image_dir / f"{decade * 10}_0_0_{index:04d}.jpg", format="PNG"
            )
    Image.new("RGB", (12, 12), (0, 0, 51)).save(
        image_dir / "0_0_0_9999.jpg", format="PNG"
    )
    Image.new("RGB", (12, 12), (0, 7, 51)).save(
        image_dir / "31_0_0_9998.jpg", format="PNG"
    )
    payload = create_manifest(
        image_dir=image_dir,
        destination=tmp_path / "manifest.json",
        split_seed=42,
        deduplicate=True,
    )
    assert len(payload["records"]) == 299
    assert payload["preprocessing"]["conflicting_images_dropped"] == 2
    assert payload["preprocessing"]["consistent_duplicate_copies_dropped"] == 1
    digests = [
        rgb_fingerprint(path=tmp_path / r["relative_path"]) for r in payload["records"]
    ]
    assert len(set(digests)) == 299
    assert all(
        "9998" not in r["relative_path"] and "0001" not in r["relative_path"]
        for r in payload["records"]
        if r["age"] == 0
    )


def test_subsets_are_nested_stratified_fixed_and_train_only() -> None:
    payload = {
        "records": [{"age": i % 100} for i in range(10000)],
        "splits": {
            "train": list(range(8000)),
            "val": list(range(8000, 9000)),
            "test": list(range(9000, 10000)),
        },
    }
    large = training_indices(payload=payload, limit=4096, subset_seed=2026)
    small = training_indices(payload=payload, limit=1024, subset_seed=2026)
    assert len(large) == 4096 and len(small) == 1024
    assert set(small) <= set(large) <= set(payload["splits"]["train"])
    assert large == training_indices(payload=payload, limit=4096, subset_seed=2026)
    assert len({payload["records"][i]["age"] // 10 for i in small}) == 10


@pytest.mark.parametrize("kind", ["mae", "mse"])
def test_sampled_statistical_maxrl_has_finite_nonzero_gradient(kind: str) -> None:
    torch.manual_seed(41)
    config = canonical_config(
        manifest=Path("unused"), output_dir=Path("unused")
    ).loss.model_copy(
        update={"method": f"stat_maxrl_{kind}", "kernel_scale_years": 2.0}
    )
    logits = torch.randn(3, 100, requires_grad=True)
    loss = policy_loss(logits=logits, targets=torch.tensor([0, 49, 99]), config=config)
    gradient = torch.autograd.grad(loss, logits)[0]
    assert torch.isfinite(gradient).all() and gradient.norm() > 0


def test_kernel_advantages_are_stable_and_centered() -> None:
    log_kernel = torch.tensor([[-10000.0, -10001.0, -10003.0], [-1.0, -1.0, -1.0]])
    advantages = normalized_kernel_advantages(log_kernel=log_kernel)
    assert torch.isfinite(advantages).all()
    torch.testing.assert_close(advantages.sum(-1), torch.zeros(2), atol=1e-6, rtol=0)
    torch.testing.assert_close(advantages[1], torch.zeros(3))


def test_self_normalized_estimator_converges_to_population_kernel_gradient() -> None:
    torch.manual_seed(2)
    probabilities = torch.tensor([0.2, 0.3, 0.5], dtype=torch.float64)
    kernel = torch.tensor([0.1, 0.5, 1.0], dtype=torch.float64)
    actions = torch.multinomial(probabilities, 512 * 2048, replacement=True).reshape(
        512, 2048
    )
    advantages = normalized_kernel_advantages(log_kernel=kernel[actions].log())
    scores = torch.nn.functional.one_hot(actions, num_classes=3) - probabilities
    actual = -(advantages[:, :, None] * scores).mean((0, 1))
    expected = probabilities - probabilities * kernel / (probabilities * kernel).sum()
    torch.testing.assert_close(actual, expected, atol=0.002, rtol=0)


def test_spearman_handles_ties_and_constant_predictions() -> None:
    pairs = [(1.0, 7), (1.0, 3), (4.0, 9), (2.0, 4), (8.0, 8)]
    got = median_spearman(pairs=pairs)
    assert got["median_spearman"] == pytest.approx(
        spearmanr([p for p, _ in pairs], [t for _, t in pairs]).statistic
    )
    assert median_spearman(pairs=[(2.0, 1), (2.0, 5)])["median_spearman"] is None


def test_fixed_and_fresh_noise_are_reproducible_rounded_and_bounded() -> None:
    ages = list(range(100)) * 100
    fixed = NoiseConfig(mode="fixed", std_years=5, seed=2026)
    fresh = NoiseConfig(mode="fresh", std_years=5, seed=2026)
    torch.testing.assert_close(
        noisy_labels(ages=ages, config=fixed, epoch=1),
        noisy_labels(ages=ages, config=fixed, epoch=7),
    )
    first = noisy_labels(ages=ages, config=fresh, epoch=1)
    assert first.dtype == torch.int64 and first.min() >= 0 and first.max() <= 99
    assert not torch.equal(first, noisy_labels(ages=ages, config=fresh, epoch=2))
    torch.testing.assert_close(first, noisy_labels(ages=ages, config=fresh, epoch=1))
    summary = noise_summary(clean=torch.tensor(ages), noisy=first)
    assert 4 < summary["noise_rmse_years"] < 6 and summary["snr_db"] > 10


def test_digit_policy_is_normalized_and_scalar_losses_use_one_prediction() -> None:
    torch.manual_seed(6)
    features = torch.randn(3, 512)
    logits = DigitHead(features=512)(features)
    torch.testing.assert_close(logits.exp().sum(-1), torch.ones(3))
    predictions = ScalarHead(features=512)(features)
    assert (
        predictions.shape == (3, 1) and predictions.min() > 0 and predictions.max() < 99
    )
    config = canonical_config(manifest=Path("unused"), output_dir=Path("unused")).loss
    targets = torch.tensor([0, 49, 99])
    expected = (predictions.flatten() - targets).abs().mean() / 99
    torch.testing.assert_close(
        scalar_loss(
            predictions=predictions,
            targets=targets,
            config=config.model_copy(update={"method": "mae"}),
        ),
        expected,
    )
