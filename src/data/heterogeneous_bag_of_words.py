from math import isfinite

import numpy as np

from src.data.bag_of_words import BagOfWordsDatasetConfig

"""
The base bag-of-words has homoskedastic noise.

This variant injects **row-level heteroskedastic noise** along a hardness
quantile. Each row i is assigned a quantile u_i on a random stratified grid,
giving a median-pegged exponential noise-std multiplier
    ã(u) = 2 ** ((u - 1/2) / snr_halflife_in_quantile)
which is then RMS-normalized so E[a²] = 1. This preserves the **global**
correlation Corr(signal, target) = corr exactly; per-row local correlation
rho_i = corr / sqrt(corr² + (1 - corr²) · a_i²).

snr_halflife_in_quantile is the quantile-distance over which SNR halves / doubles.
"""


class RowHeterogeneousBagOfWordsDatasetConfig(BagOfWordsDatasetConfig):
    """Bag-of-words with median-pegged exponential row-noise heteroskedasticity."""

    snr_halflife_in_quantile: float

    @classmethod
    def initialize(
        cls, *, snr_halflife_in_quantile: float, **base_cls_kwargs,
    ) -> "RowHeterogeneousBagOfWordsDatasetConfig":
        if not isfinite(snr_halflife_in_quantile) or snr_halflife_in_quantile <= 0:
            raise ValueError("snr_halflife_in_quantile must be finite and positive")
        return cls(
            **cls._base_kwargs(**base_cls_kwargs),
            snr_halflife_in_quantile=snr_halflife_in_quantile,
        )

    def _compute_split(
        self, *, split: str, rng: np.random.Generator
    ) -> "BagOfWordsDatasetConfig.SplitArtifact":
        n = {"train": self.num_train_samples, "val": self.num_val_samples}[split]
        words = list(self.word_density)
        probs = np.array([self.word_density[w] for w in words], dtype=float)
        vals = np.array([self.word_values[w] for w in words], dtype=float)
        vocab = np.asarray(words, dtype=object)

        idx = rng.choice(len(words), size=(n, self.prompt_length), p=probs)
        raw = vals[idx].sum(axis=1)
        signal = (
            raw / ((self.prompt_length**0.5) * self.unnormalized_signal_std) * self.corr
        )

        # Median-pegged exponential schedule on stratified hardness quantiles.
        u = (rng.permutation(n) + 0.5) / n
        tilde_a = 2.0 ** ((u - 0.5) / self.snr_halflife_in_quantile)
        a = tilde_a / np.sqrt(np.mean(tilde_a**2))

        noise_std_global = (1 - self.corr**2) ** 0.5
        targets = signal + noise_std_global * a * rng.standard_normal(n)
        prompts = [" ".join(vocab[row]) for row in idx]
        return self.SplitArtifact(prompts=prompts, signal=signal, targets=targets)
