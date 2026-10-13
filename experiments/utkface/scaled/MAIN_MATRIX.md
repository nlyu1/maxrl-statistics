# Complete UTKFace main matrix

Snapshot: 2026-10-14T06:27:41.095533+08:00. All **66/66 runs**, **22 methods**, **three seeds each**, **50 full-data epochs**.

Best validation metrics refer to the checkpoint selected by minimum validation mean-prediction MAE. Final validation metrics are epoch 50. Test metrics evaluate the selected checkpoint; final-epoch test models were not retained. Values after ± are sample SDs across training seeds on one fixed cleaned split, not confidence intervals.

| Method | Best val MAE | Final val MAE | Test MAE, years | Test median Spearman | Test label NLL, nats | Best epochs, 42/43/44 |
|---|---:|---:|---:|---:|---:|---|
| Stat MaxRL MAE, width 2 (population) | 4.852 | 5.125 | 4.718 ± 0.086 | 0.9015 | 7.439 | 14, 16, 16 |
| TailRL (64 samples) | 4.762 | 4.917 | 4.724 ± 0.019 | 0.9046 | 6.698 | 41, 26, 22 |
| TailRL (population) | 4.823 | 5.031 | 4.752 ± 0.037 | 0.9028 | 3.214 | 13, 18, 7 |
| Direct MAE, categorical mean | 4.860 | 4.921 | 4.762 ± 0.193 | 0.9052 | 9.451 | 40, 36, 32 |
| Stat MaxRL MAE, width 2 (64 samples) | 4.877 | 5.175 | 4.769 ± 0.086 | 0.9000 | 10.010 | 23, 20, 18 |
| Stat MaxRL MAE, width 5 (64 samples) | 4.916 | 5.026 | 4.793 ± 0.113 | 0.9038 | 18.014 | 45, 22, 38 |
| Direct MSE, categorical mean | 4.902 | 4.987 | 4.817 ± 0.191 | 0.9084 | 8.099 | 47, 33, 32 |
| Stat MaxRL MAE, width 5 (population) | 4.949 | 5.127 | 4.871 ± 0.118 | 0.8941 | 13.023 | 30, 31, 24 |
| Stat MaxRL MSE, width 2 (population) | 4.987 | 5.260 | 4.892 ± 0.031 | 0.8982 | 6.234 | 9, 14, 7 |
| Stat MaxRL MSE, width 2 (64 samples) | 4.962 | 5.338 | 4.923 ± 0.096 | 0.8998 | 7.150 | 13, 10, 8 |
| Binary MaxRL / CE (population) | 4.926 | 5.314 | 4.933 ± 0.030 | 0.8974 | 3.195 | 24, 23, 18 |
| Binary MaxRL (64 samples) | 5.076 | 5.439 | 4.961 ± 0.065 | 0.8907 | 4.387 | 23, 18, 22 |
| Stat MaxRL MSE, width 5 (population) | 5.086 | 5.205 | 5.002 ± 0.228 | 0.8983 | 15.107 | 40, 22, 30 |
| Stat MaxRL MSE, width 5 (64 samples) | 5.146 | 5.280 | 5.095 ± 0.306 | 0.8960 | 14.354 | 15, 40, 18 |
| GRPO MAE (64 samples) | 6.452 | 6.535 | 6.342 ± 0.109 | 0.8907 | 28.755 | 44, 42, 44 |
| GRPO MAE (population) | 6.584 | 6.639 | 6.526 ± 0.126 | 0.8881 | 60.694 | 41, 48, 49 |
| RLOO MAE (64 samples) | 6.667 | 6.729 | 6.573 ± 0.288 | 0.8842 | 17.929 | 45, 34, 47 |
| RLOO MAE (population) | 6.632 | 6.703 | 6.603 ± 0.336 | 0.8838 | 18.375 | 33, 45, 36 |
| GRPO MSE (64 samples) | 6.676 | 6.759 | 6.614 ± 0.242 | 0.8915 | 29.489 | 47, 36, 48 |
| RLOO MSE (population) | 6.742 | 6.800 | 6.640 ± 0.450 | 0.8881 | 39.942 | 44, 47, 46 |
| RLOO MSE (64 samples) | 6.710 | 6.720 | 6.657 ± 0.465 | 0.8851 | 41.929 | 50, 49, 41 |
| GRPO MSE (population) | 6.856 | 6.933 | 6.808 ± 0.281 | 0.8867 | 35.231 | 41, 48, 41 |

## Population versus 64 samples

Each row is **population minus sampled**, paired by training seed. Negative MAE/NLL differences favor population. Learning rates and MAE-selected checkpoints differ between the two estimators, so these are tuned-method comparisons.

| Objective | Paired MAE difference, years | Paired NLL difference, nats |
|---|---:|---:|
| Binary MaxRL / CE | -0.028 ± 0.069 | -1.192 ± 0.267 |
| GRPO MAE | +0.184 ± 0.123 | +31.939 ± 5.141 |
| GRPO MSE | +0.194 ± 0.401 | +5.742 ± 2.430 |
| RLOO MAE | +0.031 ± 0.050 | +0.446 ± 0.057 |
| RLOO MSE | -0.017 ± 0.083 | -1.987 ± 0.936 |
| Stat MaxRL MAE, width 2 | -0.051 ± 0.060 | -2.571 ± 1.521 |
| Stat MaxRL MAE, width 5 | +0.078 ± 0.153 | -4.990 ± 2.585 |
| Stat MaxRL MSE, width 2 | -0.031 ± 0.066 | -0.915 ± 0.437 |
| Stat MaxRL MSE, width 5 | -0.094 ± 0.090 | +0.753 ± 4.037 |
| TailRL | +0.028 ± 0.054 | -3.484 ± 0.736 |

[Interactive curves and all evaluation metrics](dashboard.html), [complete endpoints](main-endpoints.json), [paired estimator data](estimator_comparisons.json), and [interpretation](ANALYSIS.md).
