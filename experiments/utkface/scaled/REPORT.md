# Scaled UTKFace study

State: **running**. GPU time: **13.34 / 64 hours**. Completed primary runs: **0 / 66**.

Metrics use clean held-out labels. All tuning and scalar-head gates use validation data. Values are means ± sample SD across distinct training seeds; single-seed values have no SD. Label NLL is unavailable for scalar heads. Population zero-one MaxRL equals CE. Training subsets are fixed and nested.

| Noise mode | Requested SD, years | Realized RMSE, years | Realized SNR, dB |
|---|---:|---:|---:|
| fixed | 1 | 1.027 | 25.66 |
| fixed | 3 | 2.957 | 16.48 |
| fixed | 5 | 4.875 | 12.13 |
| fixed | 10 | 9.632 | 6.22 |
| fresh | 1 | 1.033 | 25.61 |
| fresh | 3 | 2.956 | 16.48 |
| fresh | 5 | 4.883 | 12.12 |
| fresh | 10 | 9.617 | 6.23 |

These are precomputed perturbation statistics for the fixed 4,096-image subset; fresh-noise values combine all 50 epochs. They are independent of model outcomes. [Noise design data](noise_design.json).


GPU time charges every launched attempt, including failures, retries, process startup, final test evaluation and polling overhead. Waiting for free GPUs consumes no GPU budget. Optional studies start after required studies finish. SNR is 10 log10(variance of clean subset ages / mean squared realized label perturbation), after rounding and clipping. Clean-label SNR is infinite and stored as null.

[Machine-readable progress](progress.json).

[Per-run values and LR tuning](runs.csv). [Full configs and metrics](results.json). [Epoch curves and realized noise statistics](curves.csv).
