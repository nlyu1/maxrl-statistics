# UTKFace shared learning rates: previous versus current

59 pre-specified reruns replace their predecessors; 163 evaluations are reused. The split, subset fingerprints, seeds, training horizons and all executed config fields except learning rate/output/device are unchanged. No new test scores select replacements. This is a follow-up sensitivity study on the same split.

All numbers below are clean test MAE in years at the validation-mean-MAE-selected checkpoint. Values with three seeds show sample SD, not confidence intervals. Noise and 1024-image comparisons have one optimization seed. Rank changes are descriptive; no statistical significance is claimed.

The rerun trainer additionally records epoch-zero validation. Its source diff was reviewed separately; both source hashes are retained in cohortAudit. Fixed-LR and digit-policy runs retain their historical rates and are not reinterpreted as interventions isolating only the schedule or architecture.

## Full data, 50 epochs

| Method | n | Peak LR old -> new | Rank old -> new | Mean MAE old | Mean MAE new | Mode MAE old | Mode MAE new |
|---|---:|---|---|---:|---:|---:|---:|
| Direct MAE · scalar | 3 | 0.001 -> 0.001 | 1 -> 1 | 4.632 +/- 0.023 | 4.632 +/- 0.023 | 4.632 +/- 0.023 | 4.632 +/- 0.023 |
| Direct MSE · scalar | 3 | 0.001 -> 0.001 | 2 -> 2 | 4.701 +/- 0.025 | 4.701 +/- 0.025 | 4.701 +/- 0.025 | 4.701 +/- 0.025 |
| Stat MAE, width 2 (population) | 3 | 0.001 -> 0.001 | 3 -> 3 | 4.718 +/- 0.086 | 4.718 +/- 0.086 | 5.143 +/- 0.155 | 5.143 +/- 0.155 |
| TailRL (64 samples) | 3 | 0.001 -> 0.001 | 4 -> 4 | 4.724 +/- 0.019 | 4.724 +/- 0.019 | 4.928 +/- 0.076 | 4.928 +/- 0.076 |
| TailRL (population) | 3 | 0.0001 -> 0.001 | 5 -> 5 | 4.752 +/- 0.037 | 4.754 +/- 0.028 | 4.940 +/- 0.112 | 4.922 +/- 0.045 |
| Direct MAE | 3 | 0.001 -> 0.001 | 6 -> 6 | 4.762 +/- 0.193 | 4.762 +/- 0.193 | 5.226 +/- 0.183 | 5.226 +/- 0.183 |
| Stat MAE, width 2 (64 samples) | 3 | 0.001 -> 0.001 | 7 -> 7 | 4.769 +/- 0.086 | 4.769 +/- 0.086 | 5.073 +/- 0.108 | 5.073 +/- 0.108 |
| Stat MAE, width 5 (64 samples) | 3 | 0.001 -> 0.001 | 8 -> 8 | 4.793 +/- 0.113 | 4.793 +/- 0.113 | 4.874 +/- 0.119 | 4.874 +/- 0.119 |
| Stat MAE, width 5 (population) | 3 | 0.0003 -> 0.001 | 10 -> 9 | 4.871 +/- 0.118 | 4.796 +/- 0.045 | 5.051 +/- 0.097 | 4.915 +/- 0.078 |
| Direct MSE | 3 | 0.001 -> 0.001 | 9 -> 10 | 4.817 +/- 0.191 | 4.817 +/- 0.191 | 5.553 +/- 0.162 | 5.553 +/- 0.162 |
| Binary MaxRL / CE (population) | 3 | 0.003 -> 0.001 | 13 -> 11 | 4.933 +/- 0.030 | 4.858 +/- 0.080 | 5.432 +/- 0.069 | 5.295 +/- 0.077 |
| Stat MSE, width 2 (population) | 3 | 0.0003 -> 0.0003 | 11 -> 12 | 4.892 +/- 0.031 | 4.892 +/- 0.031 | 5.518 +/- 0.089 | 5.518 +/- 0.089 |
| Stat MSE, width 2 (64 samples) | 3 | 0.0003 -> 0.0003 | 12 -> 13 | 4.923 +/- 0.096 | 4.923 +/- 0.096 | 5.469 +/- 0.151 | 5.469 +/- 0.151 |
| Binary MaxRL / CE (64 samples) | 3 | 0.001 -> 0.001 | 14 -> 14 | 4.961 +/- 0.065 | 4.961 +/- 0.065 | 5.439 +/- 0.039 | 5.439 +/- 0.039 |
| Stat MSE, width 5 (population) | 3 | 0.001 -> 0.0003 | 15 -> 15 | 5.002 +/- 0.228 | 5.060 +/- 0.232 | 5.160 +/- 0.270 | 5.354 +/- 0.342 |
| Stat MSE, width 5 (64 samples) | 3 | 0.0003 -> 0.0003 | 16 -> 16 | 5.095 +/- 0.306 | 5.095 +/- 0.306 | 5.369 +/- 0.257 | 5.369 +/- 0.257 |
| GRPO MAE (64 samples) | 3 | 0.0003 -> 0.0003 | 17 -> 17 | 6.342 +/- 0.109 | 6.342 +/- 0.109 | 6.409 +/- 0.097 | 6.409 +/- 0.097 |
| GRPO MAE (population) | 3 | 0.001 -> 0.0003 | 18 -> 18 | 6.526 +/- 0.126 | 6.486 +/- 0.142 | 6.560 +/- 0.135 | 6.543 +/- 0.150 |
| RLOO MAE (64 samples) | 3 | 0.0001 -> 0.0001 | 19 -> 19 | 6.573 +/- 0.288 | 6.573 +/- 0.288 | 6.721 +/- 0.325 | 6.721 +/- 0.325 |
| RLOO MAE (population) | 3 | 0.0001 -> 0.0001 | 20 -> 20 | 6.603 +/- 0.336 | 6.603 +/- 0.336 | 6.755 +/- 0.331 | 6.755 +/- 0.331 |
| GRPO MSE (64 samples) | 3 | 0.0003 -> 0.0003 | 21 -> 21 | 6.614 +/- 0.242 | 6.614 +/- 0.242 | 6.680 +/- 0.244 | 6.680 +/- 0.244 |
| RLOO MSE (population) | 3 | 0.001 -> 0.001 | 22 -> 22 | 6.640 +/- 0.450 | 6.640 +/- 0.450 | 6.702 +/- 0.465 | 6.702 +/- 0.465 |
| RLOO MSE (64 samples) | 3 | 0.001 -> 0.001 | 23 -> 23 | 6.657 +/- 0.465 | 6.657 +/- 0.465 | 6.734 +/- 0.477 | 6.734 +/- 0.477 |
| GRPO MSE (population) | 3 | 0.0003 -> 0.0003 | 24 -> 24 | 6.808 +/- 0.281 | 6.808 +/- 0.281 | 6.880 +/- 0.280 | 6.880 +/- 0.280 |

## 4096 images, 50 epochs

| Method | n | Peak LR old -> new | Rank old -> new | Mean MAE old | Mean MAE new | Mode MAE old | Mode MAE new |
|---|---:|---|---|---:|---:|---:|---:|
| Direct MAE · scalar | 1 | 0.001 -> 0.001 | 1 -> 1 | 5.145 | 5.145 | 5.145 | 5.145 |
| Direct MSE · scalar | 1 | 0.003 -> 0.001 | 3 -> 2 | 5.314 | 5.164 | 5.314 | 5.164 |
| Direct MSE | 3 | 0.001 -> 0.001 | 2 -> 3 | 5.297 +/- 0.174 | 5.297 +/- 0.174 | 6.508 +/- 0.283 | 6.508 +/- 0.283 |
| TailRL (population) | 3 | 0.0003 -> 0.001 | 4 -> 4 | 5.359 +/- 0.097 | 5.411 +/- 0.074 | 5.887 +/- 0.101 | 5.744 +/- 0.132 |
| Direct MAE | 3 | 0.003 -> 0.001 | 8 -> 5 | 6.523 +/- 1.355 | 5.505 +/- 0.277 | 7.759 +/- 0.900 | 6.495 +/- 0.203 |
| Stat MSE, width 2 (population) | 3 | 0.001 -> 0.001 | 5 -> 6 | 5.820 +/- 0.044 | 5.820 +/- 0.044 | 6.486 +/- 0.143 | 6.486 +/- 0.143 |
| Stat MAE, width 2 (population) | 3 | 0.001 -> 0.001 | 6 -> 7 | 5.873 +/- 0.068 | 5.873 +/- 0.068 | 6.364 +/- 0.090 | 6.364 +/- 0.090 |
| CE | 3 | 0.003 -> 0.001 | 7 -> 8 | 6.001 +/- 0.083 | 5.924 +/- 0.050 | 6.884 +/- 0.158 | 6.826 +/- 0.148 |
| GRPO MAE (population) | 3 | 0.001 -> 0.001 | 9 -> 9 | 7.018 +/- 0.132 | 7.018 +/- 0.132 | 7.085 +/- 0.159 | 7.085 +/- 0.159 |
| GRPO MSE (population) | 3 | 0.0003 -> 0.0003 | 10 -> 10 | 7.242 +/- 0.325 | 7.242 +/- 0.325 | 7.349 +/- 0.365 | 7.349 +/- 0.365 |

## 1024 images, 200 epochs

| Method | n | Peak LR old -> new | Rank old -> new | Mean MAE old | Mean MAE new | Mode MAE old | Mode MAE new |
|---|---:|---|---|---:|---:|---:|---:|
| Direct MSE | 1 | 0.0003 -> 0.001 | 2 -> 1 | 6.371 | 6.100 | 10.188 | 8.218 |
| Direct MAE | 1 | 0.001 -> 0.001 | 1 -> 2 | 6.208 | 6.208 | 7.456 | 7.456 |
| TailRL (population) | 1 | 0.001 -> 0.001 | 3 -> 3 | 6.565 | 6.565 | 7.305 | 7.305 |
| Stat MSE, width 2 (population) | 1 | 0.001 -> 0.001 | 4 -> 4 | 7.098 | 7.098 | 8.080 | 8.080 |
| Stat MAE, width 2 (population) | 1 | 0.001 -> 0.001 | 5 -> 5 | 7.215 | 7.215 | 8.088 | 8.088 |
| CE | 1 | 0.001 -> 0.001 | 6 -> 6 | 7.280 | 7.280 | 8.681 | 8.681 |
| GRPO MAE (population) | 1 | 0.0003 -> 0.0003 | 7 -> 7 | 7.953 | 7.953 | 8.202 | 8.202 |
| GRPO MSE (population) | 1 | 0.0003 -> 0.0003 | 8 -> 8 | 8.314 | 8.314 | 8.482 | 8.482 |

## Fixed label noise, SD 1

| Method | n | Peak LR old -> new | Rank old -> new | Mean MAE old | Mean MAE new | Mode MAE old | Mode MAE new |
|---|---:|---|---|---:|---:|---:|---:|
| Direct MSE · scalar | 1 | 0.003 -> 0.001 | 3 -> 1 | 5.364 | 5.187 | 5.364 | 5.187 |
| Direct MAE · scalar | 1 | 0.001 -> 0.001 | 1 -> 2 | 5.198 | 5.198 | 5.198 | 5.198 |
| Direct MSE | 1 | 0.001 -> 0.001 | 2 -> 3 | 5.254 | 5.254 | 6.486 | 6.486 |
| TailRL (population) | 1 | 0.0003 -> 0.001 | 4 -> 4 | 5.379 | 5.311 | 5.957 | 5.530 |
| Stat MSE, width 2 (population) | 1 | 0.001 -> 0.001 | 6 -> 5 | 5.653 | 5.653 | 6.196 | 6.196 |
| Stat MAE, width 2 (population) | 1 | 0.001 -> 0.001 | 7 -> 6 | 5.866 | 5.866 | 6.310 | 6.310 |
| Direct MAE | 1 | 0.003 -> 0.001 | 5 -> 7 | 5.650 | 5.882 | 7.043 | 6.852 |
| CE | 1 | 0.003 -> 0.001 | 8 -> 8 | 5.887 | 6.056 | 6.760 | 6.857 |
| GRPO MAE (population) | 1 | 0.001 -> 0.001 | 9 -> 9 | 6.950 | 6.950 | 7.029 | 7.029 |
| GRPO MSE (population) | 1 | 0.0003 -> 0.0003 | 10 -> 10 | 7.125 | 7.125 | 7.237 | 7.237 |

## Fixed label noise, SD 3

| Method | n | Peak LR old -> new | Rank old -> new | Mean MAE old | Mean MAE new | Mode MAE old | Mode MAE new |
|---|---:|---|---|---:|---:|---:|---:|
| Direct MAE · scalar | 1 | 0.001 -> 0.001 | 1 -> 1 | 5.351 | 5.351 | 5.351 | 5.351 |
| Direct MSE · scalar | 1 | 0.003 -> 0.001 | 3 -> 2 | 5.526 | 5.384 | 5.526 | 5.384 |
| Direct MSE | 1 | 0.001 -> 0.001 | 2 -> 3 | 5.449 | 5.449 | 6.717 | 6.717 |
| TailRL (population) | 1 | 0.0003 -> 0.001 | 4 -> 4 | 5.597 | 5.656 | 5.961 | 6.132 |
| Direct MAE | 1 | 0.003 -> 0.001 | 8 -> 5 | 6.337 | 5.828 | 7.317 | 7.025 |
| Stat MSE, width 2 (population) | 1 | 0.001 -> 0.001 | 5 -> 6 | 5.927 | 5.927 | 6.584 | 6.584 |
| Stat MAE, width 2 (population) | 1 | 0.001 -> 0.001 | 6 -> 7 | 5.978 | 5.978 | 6.628 | 6.628 |
| CE | 1 | 0.003 -> 0.001 | 7 -> 8 | 6.283 | 6.108 | 7.414 | 7.168 |
| GRPO MAE (population) | 1 | 0.001 -> 0.001 | 9 -> 9 | 6.994 | 6.994 | 7.073 | 7.073 |
| GRPO MSE (population) | 1 | 0.0003 -> 0.0003 | 10 -> 10 | 7.051 | 7.051 | 7.155 | 7.155 |

## Fixed label noise, SD 5

| Method | n | Peak LR old -> new | Rank old -> new | Mean MAE old | Mean MAE new | Mode MAE old | Mode MAE new |
|---|---:|---|---|---:|---:|---:|---:|
| Direct MSE | 1 | 0.001 -> 0.001 | 1 -> 1 | 5.588 | 5.588 | 6.814 | 6.814 |
| Direct MSE · scalar | 1 | 0.003 -> 0.001 | 2 -> 2 | 5.741 | 5.678 | 5.741 | 5.678 |
| TailRL (population) | 1 | 0.0003 -> 0.001 | 6 -> 3 | 5.967 | 5.741 | 6.461 | 6.212 |
| Direct MAE · scalar | 1 | 0.001 -> 0.001 | 3 -> 4 | 5.764 | 5.764 | 5.764 | 5.764 |
| Stat MAE, width 2 (population) | 1 | 0.001 -> 0.001 | 5 -> 5 | 5.934 | 5.934 | 6.954 | 6.954 |
| Direct MAE | 1 | 0.003 -> 0.001 | 4 -> 6 | 5.896 | 6.095 | 6.736 | 6.997 |
| Stat MSE, width 2 (population) | 1 | 0.001 -> 0.001 | 7 -> 7 | 6.262 | 6.262 | 6.770 | 6.770 |
| CE | 1 | 0.003 -> 0.001 | 8 -> 8 | 6.312 | 6.284 | 7.695 | 7.691 |
| GRPO MAE (population) | 1 | 0.001 -> 0.001 | 9 -> 9 | 7.025 | 7.025 | 7.065 | 7.065 |
| GRPO MSE (population) | 1 | 0.0003 -> 0.0003 | 10 -> 10 | 7.070 | 7.070 | 7.156 | 7.156 |

## Fixed label noise, SD 10

| Method | n | Peak LR old -> new | Rank old -> new | Mean MAE old | Mean MAE new | Mode MAE old | Mode MAE new |
|---|---:|---|---|---:|---:|---:|---:|
| Direct MSE | 1 | 0.001 -> 0.001 | 1 -> 1 | 6.252 | 6.252 | 8.200 | 8.200 |
| Direct MAE | 1 | 0.003 -> 0.001 | 10 -> 2 | 9.425 | 6.396 | 9.742 | 7.769 |
| Direct MSE · scalar | 1 | 0.003 -> 0.001 | 4 -> 3 | 6.874 | 6.617 | 6.874 | 6.617 |
| Direct MAE · scalar | 1 | 0.001 -> 0.001 | 3 -> 4 | 6.734 | 6.734 | 6.734 | 6.734 |
| TailRL (population) | 1 | 0.0003 -> 0.001 | 2 -> 5 | 6.717 | 7.155 | 7.907 | 8.052 |
| CE | 1 | 0.003 -> 0.001 | 7 -> 6 | 7.363 | 7.161 | 8.253 | 9.575 |
| Stat MAE, width 2 (population) | 1 | 0.001 -> 0.001 | 5 -> 7 | 7.201 | 7.201 | 10.214 | 10.214 |
| Stat MSE, width 2 (population) | 1 | 0.001 -> 0.001 | 6 -> 8 | 7.357 | 7.357 | 10.692 | 10.692 |
| GRPO MSE (population) | 1 | 0.0003 -> 0.0003 | 8 -> 9 | 7.547 | 7.547 | 7.626 | 7.626 |
| GRPO MAE (population) | 1 | 0.001 -> 0.001 | 9 -> 10 | 7.697 | 7.697 | 7.802 | 7.802 |

## Fresh label noise, SD 1

| Method | n | Peak LR old -> new | Rank old -> new | Mean MAE old | Mean MAE new | Mode MAE old | Mode MAE new |
|---|---:|---|---|---:|---:|---:|---:|
| Direct MAE · scalar | 1 | 0.001 -> 0.001 | 1 -> 1 | 5.127 | 5.127 | 5.127 | 5.127 |
| Direct MSE · scalar | 1 | 0.003 -> 0.001 | 3 -> 2 | 5.275 | 5.154 | 5.275 | 5.154 |
| Direct MSE | 1 | 0.001 -> 0.001 | 2 -> 3 | 5.248 | 5.248 | 6.435 | 6.435 |
| TailRL (population) | 1 | 0.0003 -> 0.001 | 4 -> 4 | 5.350 | 5.397 | 5.754 | 5.613 |
| CE | 1 | 0.003 -> 0.001 | 6 -> 5 | 5.702 | 5.545 | 6.066 | 5.945 |
| Stat MSE, width 2 (population) | 1 | 0.001 -> 0.001 | 7 -> 6 | 5.765 | 5.765 | 6.106 | 6.106 |
| Stat MAE, width 2 (population) | 1 | 0.001 -> 0.001 | 8 -> 7 | 5.814 | 5.814 | 6.309 | 6.309 |
| Direct MAE | 1 | 0.003 -> 0.001 | 5 -> 8 | 5.695 | 5.838 | 7.068 | 7.102 |
| GRPO MAE (population) | 1 | 0.001 -> 0.001 | 9 -> 9 | 6.915 | 6.915 | 6.941 | 6.941 |
| GRPO MSE (population) | 1 | 0.0003 -> 0.0003 | 10 -> 10 | 7.020 | 7.020 | 7.182 | 7.182 |

## Fresh label noise, SD 3

| Method | n | Peak LR old -> new | Rank old -> new | Mean MAE old | Mean MAE new | Mode MAE old | Mode MAE new |
|---|---:|---|---|---:|---:|---:|---:|
| Direct MSE · scalar | 1 | 0.003 -> 0.001 | 4 -> 1 | 5.322 | 5.100 | 5.322 | 5.100 |
| Direct MAE · scalar | 1 | 0.001 -> 0.001 | 2 -> 2 | 5.249 | 5.249 | 5.249 | 5.249 |
| Direct MSE | 1 | 0.001 -> 0.001 | 3 -> 3 | 5.258 | 5.258 | 6.589 | 6.589 |
| TailRL (population) | 1 | 0.0003 -> 0.001 | 1 -> 4 | 5.242 | 5.367 | 5.623 | 5.606 |
| CE | 1 | 0.003 -> 0.001 | 7 -> 5 | 5.435 | 5.382 | 5.713 | 5.681 |
| Stat MSE, width 2 (population) | 1 | 0.001 -> 0.001 | 5 -> 6 | 5.422 | 5.422 | 5.913 | 5.913 |
| Stat MAE, width 2 (population) | 1 | 0.001 -> 0.001 | 6 -> 7 | 5.423 | 5.423 | 5.806 | 5.806 |
| Direct MAE | 1 | 0.003 -> 0.001 | 8 -> 8 | 6.136 | 5.836 | 7.281 | 6.620 |
| GRPO MAE (population) | 1 | 0.001 -> 0.001 | 9 -> 9 | 6.919 | 6.919 | 6.980 | 6.980 |
| GRPO MSE (population) | 1 | 0.0003 -> 0.0003 | 10 -> 10 | 7.075 | 7.075 | 7.131 | 7.131 |

## Fresh label noise, SD 5

| Method | n | Peak LR old -> new | Rank old -> new | Mean MAE old | Mean MAE new | Mode MAE old | Mode MAE new |
|---|---:|---|---|---:|---:|---:|---:|
| Direct MSE · scalar | 1 | 0.003 -> 0.001 | 7 -> 1 | 5.489 | 5.223 | 5.489 | 5.223 |
| Direct MSE | 1 | 0.001 -> 0.001 | 2 -> 2 | 5.231 | 5.231 | 6.539 | 6.539 |
| Direct MAE · scalar | 1 | 0.001 -> 0.001 | 3 -> 3 | 5.262 | 5.262 | 5.262 | 5.262 |
| Stat MSE, width 2 (population) | 1 | 0.001 -> 0.001 | 4 -> 4 | 5.280 | 5.280 | 5.811 | 5.811 |
| CE | 1 | 0.003 -> 0.001 | 6 -> 5 | 5.439 | 5.366 | 5.879 | 5.641 |
| TailRL (population) | 1 | 0.0003 -> 0.001 | 1 -> 6 | 5.215 | 5.377 | 5.628 | 5.690 |
| Stat MAE, width 2 (population) | 1 | 0.001 -> 0.001 | 5 -> 7 | 5.409 | 5.409 | 5.853 | 5.853 |
| Direct MAE | 1 | 0.003 -> 0.001 | 8 -> 8 | 5.609 | 5.526 | 7.010 | 6.854 |
| GRPO MSE (population) | 1 | 0.0003 -> 0.0003 | 9 -> 9 | 6.932 | 6.932 | 7.025 | 7.025 |
| GRPO MAE (population) | 1 | 0.001 -> 0.001 | 10 -> 10 | 7.051 | 7.051 | 7.140 | 7.140 |

## Fresh label noise, SD 10

| Method | n | Peak LR old -> new | Rank old -> new | Mean MAE old | Mean MAE new | Mode MAE old | Mode MAE new |
|---|---:|---|---|---:|---:|---:|---:|
| CE | 1 | 0.003 -> 0.001 | 7 -> 1 | 5.767 | 5.348 | 7.362 | 6.634 |
| Direct MSE · scalar | 1 | 0.003 -> 0.001 | 5 -> 2 | 5.542 | 5.361 | 5.542 | 5.361 |
| Direct MAE · scalar | 1 | 0.001 -> 0.001 | 2 -> 3 | 5.389 | 5.389 | 5.389 | 5.389 |
| Stat MAE, width 2 (population) | 1 | 0.001 -> 0.001 | 3 -> 4 | 5.440 | 5.440 | 6.998 | 6.998 |
| Stat MSE, width 2 (population) | 1 | 0.001 -> 0.001 | 4 -> 5 | 5.455 | 5.455 | 6.963 | 6.963 |
| TailRL (population) | 1 | 0.0003 -> 0.001 | 1 -> 6 | 5.283 | 5.482 | 5.796 | 6.055 |
| Direct MAE | 1 | 0.003 -> 0.001 | 10 -> 7 | 8.251 | 5.710 | 10.933 | 7.610 |
| Direct MSE | 1 | 0.001 -> 0.001 | 6 -> 8 | 5.719 | 5.719 | 6.693 | 6.693 |
| GRPO MAE (population) | 1 | 0.001 -> 0.001 | 8 -> 9 | 7.046 | 7.046 | 7.141 | 7.141 |
| GRPO MSE (population) | 1 | 0.0003 -> 0.0003 | 9 -> 10 | 7.054 | 7.054 | 7.113 | 7.113 |
