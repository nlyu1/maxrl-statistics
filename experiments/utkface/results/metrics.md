# UTKFace measured results

45 runs, 15 configurations. Epoch budgets: [12]. Seeds: [42, 43, 44].

Values are means ± sample standard deviations across seeds. Single-seed rows show the measured value alone. These are not confidence intervals. Checkpoints minimize validation MAE of the policy mean.

Test examples: [2368]; singleton-image test examples: [2288].

| Objective | Seeds | Mean age MAE, years | Mean age RMSE, years | Singleton test MAE, years | Macro decade MAE, years |
|---|---:|---:|---:|---:|---:|
| Population TailRL | 3 | 4.57 ± 0.01 | 6.62 ± 0.05 | 4.53 ± 0.02 | 5.54 ± 0.08 |
| TailRL, 64 samples | 3 | 4.60 ± 0.08 | 6.65 ± 0.13 | 4.56 ± 0.07 | 5.88 ± 0.25 |
| CE / population binary MaxRL | 3 | 4.70 ± 0.06 | 6.77 ± 0.11 | 4.66 ± 0.06 | 5.83 ± 0.10 |
| Statistical MaxRL, MAE, 5 years | 3 | 4.71 ± 0.07 | 6.80 ± 0.05 | 4.68 ± 0.07 | 5.86 ± 0.19 |
| Statistical MaxRL, MAE, 2 years | 3 | 4.72 ± 0.06 | 6.82 ± 0.09 | 4.68 ± 0.06 | 5.57 ± 0.17 |
| Statistical MaxRL, MSE, 2 years | 3 | 4.72 ± 0.05 | 6.77 ± 0.04 | 4.70 ± 0.05 | 5.63 ± 0.34 |
| Binary MaxRL, 64 samples | 3 | 4.78 ± 0.07 | 6.91 ± 0.08 | 4.74 ± 0.09 | 6.12 ± 0.17 |
| MSE of policy mean | 3 | 4.81 ± 0.22 | 6.72 ± 0.13 | 4.77 ± 0.23 | 5.88 ± 0.22 |
| Statistical MaxRL, MSE, 5 years | 3 | 4.85 ± 0.17 | 6.86 ± 0.11 | 4.81 ± 0.17 | 5.94 ± 0.25 |
| Statistical MaxRL, MSE, 10 years | 3 | 5.04 ± 0.20 | 6.99 ± 0.17 | 5.00 ± 0.20 | 6.87 ± 0.70 |
| Statistical MaxRL, MAE, 10 years | 3 | 5.06 ± 0.28 | 7.06 ± 0.21 | 5.02 ± 0.28 | 6.86 ± 0.75 |
| Population GRPO, MAE | 3 | 6.48 ± 0.04 | 9.22 ± 0.45 | 6.45 ± 0.03 | 12.82 ± 0.89 |
| Population RLOO, MAE | 3 | 6.66 ± 0.07 | 9.44 ± 0.32 | 6.62 ± 0.09 | 13.04 ± 0.84 |
| Population RLOO, MSE | 3 | 6.85 ± 0.50 | 9.48 ± 0.64 | 6.81 ± 0.50 | 12.91 ± 1.03 |
| Population GRPO, MSE | 3 | 6.87 ± 0.25 | 9.46 ± 0.41 | 6.83 ± 0.25 | 12.99 ± 0.88 |

Macro decade MAE gives equal weight to each observed age decade. The original test split is imbalanced. The singleton slice retains only images with unique decoded pixels in the entire dataset; training and checkpoint selection use the original split.

| Objective | Expected policy draw MAE, years | Entropy, nats | Categorical NLL, nats | Oracle best-of-64 MAE, years |
|---|---:|---:|---:|---:|
| Population TailRL | 5.46 ± 0.05 | 2.46 ± 0.04 | 3.06 ± 0.04 | 0.54 ± 0.06 |
| TailRL, 64 samples | 5.41 ± 0.07 | 2.12 ± 0.06 | 5.01 ± 0.39 | 0.80 ± 0.05 |
| CE / population binary MaxRL | 6.00 ± 0.08 | 2.46 ± 0.08 | 3.04 ± 0.05 | 0.40 ± 0.04 |
| Statistical MaxRL, MAE, 5 years | 5.17 ± 0.08 | 0.76 ± 0.08 | 9.37 ± 0.60 | 2.10 ± 0.14 |
| Statistical MaxRL, MAE, 2 years | 5.62 ± 0.02 | 1.20 ± 0.06 | 6.14 ± 0.23 | 1.17 ± 0.10 |
| Statistical MaxRL, MSE, 2 years | 5.78 ± 0.06 | 1.62 ± 0.07 | 5.80 ± 0.54 | 0.94 ± 0.09 |
| Binary MaxRL, 64 samples | 6.16 ± 0.03 | 2.11 ± 0.02 | 3.70 ± 0.04 | 0.58 ± 0.02 |
| MSE of policy mean | 11.73 ± 0.37 | 2.74 ± 0.04 | 5.58 ± 0.49 | 1.01 ± 0.19 |
| Statistical MaxRL, MSE, 5 years | 5.39 ± 0.21 | 0.81 ± 0.07 | 10.03 ± 0.60 | 2.28 ± 0.28 |
| Statistical MaxRL, MSE, 10 years | 5.36 ± 0.21 | 0.30 ± 0.04 | 14.22 ± 0.72 | 3.65 ± 0.22 |
| Statistical MaxRL, MAE, 10 years | 5.38 ± 0.30 | 0.29 ± 0.05 | 13.77 ± 0.75 | 3.56 ± 0.18 |
| Population GRPO, MAE | 6.59 ± 0.05 | 0.07 ± 0.01 | 22.85 ± 1.08 | 6.02 ± 0.05 |
| Population RLOO, MAE | 6.77 ± 0.06 | 0.10 ± 0.03 | 26.34 ± 0.42 | 6.10 ± 0.10 |
| Population RLOO, MSE | 7.01 ± 0.51 | 0.07 ± 0.03 | 17.63 ± 0.50 | 6.32 ± 0.64 |
| Population GRPO, MSE | 6.97 ± 0.25 | 0.05 ± 0.00 | 23.81 ± 0.77 | 6.48 ± 0.24 |

Policy draw errors and oracle errors are evaluated exactly over all 100 actions. Oracle best-of-64 uses the target to select the closest draw; it is a distribution diagnostic.

![Point and policy errors](comparison.png)

![Validation learning curves](learning_curves.png)

Per-seed values: [runs.csv](runs.csv). Aggregates: [summary.csv](summary.csv). Full metrics, configs and provenance: [results.json](results.json). Age-decade metrics: [decade_metrics.csv](decade_metrics.csv).
