# Complete UTKFace result tables

Test metrics evaluate the validation-MAE-selected checkpoint. Best/final columns refer to validation. ± denotes sample SD across optimization seeds on one fixed split, not confidence intervals. Scalar Spearman uses continuous predictions; categorical Spearman uses the policy median. Scalar label NLL is undefined.

## Full data, 50 epochs

| Method | Images × epochs | Seeds | Best epochs | Best val MAE | Final val MAE | Test MAE | Test RMSE | Spearman | Label NLL |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| Direct MAE · scalar | 18462 × 50 | 42,43,44 | 41,44,42 | 4.667 ± 0.068 | 4.716 ± 0.029 | 4.632 ± 0.023 | 6.765 ± 0.062 | 0.911 ± 0.001 | n/a |
| Direct MSE · scalar | 18462 × 50 | 42,43,44 | 39,42,49 | 4.771 ± 0.029 | 4.817 ± 0.024 | 4.701 ± 0.025 | 6.733 ± 0.056 | 0.911 ± 0.002 | n/a |
| Stat MAE, width 2 (population) | 18462 × 50 | 42,43,44 | 14,16,16 | 4.852 ± 0.080 | 5.125 ± 0.073 | 4.718 ± 0.086 | 6.964 ± 0.176 | 0.902 ± 0.006 | 7.439 ± 0.451 |
| TailRL (64 samples) | 18462 × 50 | 42,43,44 | 41,26,22 | 4.762 ± 0.029 | 4.917 ± 0.034 | 4.724 ± 0.019 | 6.931 ± 0.069 | 0.905 ± 0.002 | 6.698 ± 0.626 |
| TailRL (population) | 18462 × 50 | 42,43,44 | 13,18,7 | 4.823 ± 0.043 | 5.031 ± 0.056 | 4.752 ± 0.037 | 6.871 ± 0.085 | 0.903 ± 0.004 | 3.214 ± 0.222 |
| Direct MAE | 18462 × 50 | 42,43,44 | 40,36,32 | 4.860 ± 0.230 | 4.921 ± 0.193 | 4.762 ± 0.193 | 6.897 ± 0.109 | 0.905 ± 0.005 | 9.451 ± 1.390 |
| Stat MAE, width 2 (64 samples) | 18462 × 50 | 42,43,44 | 23,20,18 | 4.877 ± 0.063 | 5.175 ± 0.060 | 4.769 ± 0.086 | 7.020 ± 0.170 | 0.900 ± 0.008 | 10.010 ± 1.083 |
| Stat MAE, width 5 (64 samples) | 18462 × 50 | 42,43,44 | 45,22,38 | 4.916 ± 0.090 | 5.026 ± 0.117 | 4.793 ± 0.113 | 7.024 ± 0.077 | 0.904 ± 0.001 | 18.014 ± 2.437 |
| Direct MSE | 18462 × 50 | 42,43,44 | 47,33,32 | 4.902 ± 0.199 | 4.987 ± 0.241 | 4.817 ± 0.191 | 6.836 ± 0.078 | 0.908 ± 0.002 | 8.099 ± 0.864 |
| Stat MAE, width 5 (population) | 18462 × 50 | 42,43,44 | 30,31,24 | 4.949 ± 0.073 | 5.127 ± 0.068 | 4.871 ± 0.118 | 7.237 ± 0.192 | 0.894 ± 0.003 | 13.023 ± 0.674 |
| Stat MSE, width 2 (population) | 18462 × 50 | 42,43,44 | 9,14,7 | 4.987 ± 0.054 | 5.260 ± 0.142 | 4.892 ± 0.031 | 7.132 ± 0.060 | 0.898 ± 0.001 | 6.234 ± 0.592 |
| Stat MSE, width 2 (64 samples) | 18462 × 50 | 42,43,44 | 13,10,8 | 4.962 ± 0.057 | 5.338 ± 0.086 | 4.923 ± 0.096 | 7.095 ± 0.128 | 0.900 ± 0.004 | 7.150 ± 0.318 |
| Binary MaxRL / CE (population) | 18462 × 50 | 42,43,44 | 24,23,18 | 4.926 ± 0.037 | 5.314 ± 0.035 | 4.933 ± 0.030 | 7.338 ± 0.061 | 0.897 ± 0.002 | 3.195 ± 0.124 |
| Binary MaxRL / CE (64 samples) | 18462 × 50 | 42,43,44 | 23,18,22 | 5.076 ± 0.040 | 5.439 ± 0.003 | 4.961 ± 0.065 | 7.487 ± 0.253 | 0.891 ± 0.005 | 4.387 ± 0.288 |
| Stat MSE, width 5 (population) | 18462 × 50 | 42,43,44 | 40,22,30 | 5.086 ± 0.194 | 5.205 ± 0.195 | 5.002 ± 0.228 | 7.230 ± 0.180 | 0.898 ± 0.001 | 15.107 ± 1.851 |
| Stat MSE, width 5 (64 samples) | 18462 × 50 | 42,43,44 | 15,40,18 | 5.146 ± 0.210 | 5.280 ± 0.182 | 5.095 ± 0.306 | 7.275 ± 0.330 | 0.896 ± 0.008 | 14.354 ± 2.295 |
| GRPO MAE (64 samples) | 18462 × 50 | 42,43,44 | 44,42,44 | 6.452 ± 0.160 | 6.535 ± 0.105 | 6.342 ± 0.109 | 9.202 ± 0.433 | 0.891 ± 0.007 | 28.755 ± 1.572 |
| GRPO MAE (population) | 18462 × 50 | 42,43,44 | 41,48,49 | 6.584 ± 0.081 | 6.639 ± 0.105 | 6.526 ± 0.126 | 9.405 ± 0.490 | 0.888 ± 0.003 | 60.694 ± 5.382 |
| RLOO MAE (64 samples) | 18462 × 50 | 42,43,44 | 45,34,47 | 6.667 ± 0.359 | 6.729 ± 0.322 | 6.573 ± 0.288 | 9.224 ± 0.501 | 0.884 ± 0.003 | 17.929 ± 0.795 |
| RLOO MAE (population) | 18462 × 50 | 42,43,44 | 33,45,36 | 6.632 ± 0.296 | 6.703 ± 0.289 | 6.603 ± 0.336 | 9.283 ± 0.535 | 0.884 ± 0.003 | 18.375 ± 0.743 |
| GRPO MSE (64 samples) | 18462 × 50 | 42,43,44 | 47,36,48 | 6.676 ± 0.193 | 6.759 ± 0.215 | 6.614 ± 0.242 | 9.358 ± 0.270 | 0.891 ± 0.005 | 29.489 ± 0.792 |
| RLOO MSE (population) | 18462 × 50 | 42,43,44 | 44,47,46 | 6.742 ± 0.434 | 6.800 ± 0.431 | 6.640 ± 0.450 | 9.290 ± 0.679 | 0.888 ± 0.006 | 39.942 ± 0.837 |
| RLOO MSE (64 samples) | 18462 × 50 | 42,43,44 | 50,49,41 | 6.710 ± 0.406 | 6.720 ± 0.421 | 6.657 ± 0.465 | 9.313 ± 0.685 | 0.885 ± 0.006 | 41.929 ± 1.607 |
| GRPO MSE (population) | 18462 × 50 | 42,43,44 | 41,48,41 | 6.856 ± 0.242 | 6.933 ± 0.276 | 6.808 ± 0.281 | 9.444 ± 0.471 | 0.887 ± 0.007 | 35.231 ± 2.314 |

## Fixed subsets, clean labels

| Method | Images × epochs | Seeds | Best epochs | Best val MAE | Final val MAE | Test MAE | Test RMSE | Spearman | Label NLL |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| Direct MAE | 1024 × 50 | 42 | 33 | 6.183 | 6.297 | 6.087 | 8.799 | 0.866 | 4.792 |
| Direct MAE · scalar | 1024 × 50 | 42 | 44 | 6.167 | 6.240 | 6.177 | 8.509 | 0.872 | n/a |
| Direct MSE | 1024 × 50 | 42 | 41 | 6.407 | 6.476 | 6.301 | 8.811 | 0.859 | 4.403 |
| Direct MSE · scalar | 1024 × 50 | 42 | 44 | 6.389 | 6.405 | 6.367 | 8.776 | 0.866 | n/a |
| TailRL (population) | 1024 × 50 | 42 | 29 | 6.566 | 6.757 | 6.403 | 9.271 | 0.831 | 4.432 |
| Stat MSE, width 2 (population) | 1024 × 50 | 42 | 40 | 6.986 | 7.021 | 6.835 | 9.921 | 0.802 | 6.846 |
| Stat MAE, width 2 (population) | 1024 × 50 | 42 | 31 | 7.016 | 7.103 | 6.899 | 9.927 | 0.801 | 6.651 |
| CE | 1024 × 50 | 42 | 30 | 7.337 | 7.381 | 7.329 | 10.461 | 0.779 | 4.832 |
| GRPO MAE (population) | 1024 × 50 | 42 | 31 | 7.985 | 8.040 | 7.732 | 10.991 | 0.805 | 13.191 |
| GRPO MSE (population) | 1024 × 50 | 42 | 47 | 7.915 | 7.965 | 7.784 | 11.011 | 0.814 | 14.247 |
| Direct MAE | 1024 × 200 | 42 | 71 | 6.409 | 6.775 | 6.208 | 9.040 | 0.850 | 6.490 |
| Direct MSE | 1024 × 200 | 42 | 51 | 6.475 | 6.604 | 6.371 | 8.922 | 0.845 | 5.054 |
| TailRL (population) | 1024 × 200 | 42 | 134 | 6.620 | 6.687 | 6.565 | 9.647 | 0.811 | 5.253 |
| Stat MSE, width 2 (population) | 1024 × 200 | 42 | 168 | 7.338 | 7.355 | 7.098 | 10.383 | 0.787 | 8.173 |
| Stat MAE, width 2 (population) | 1024 × 200 | 42 | 198 | 7.224 | 7.246 | 7.215 | 10.513 | 0.794 | 8.020 |
| CE | 1024 × 200 | 42 | 64 | 7.540 | 7.789 | 7.280 | 10.513 | 0.779 | 5.188 |
| GRPO MAE (population) | 1024 × 200 | 42 | 115 | 7.895 | 8.011 | 7.953 | 11.302 | 0.792 | 16.024 |
| GRPO MSE (population) | 1024 × 200 | 42 | 103 | 8.486 | 8.581 | 8.314 | 11.478 | 0.807 | 18.282 |
| Direct MAE · scalar | 4096 × 50 | 42 | 41 | 5.200 | 5.202 | 5.145 | 7.334 | 0.899 | n/a |
| Direct MSE | 4096 × 50 | 42,43,44 | 38,40,40 | 5.469 ± 0.266 | 5.518 ± 0.263 | 5.297 ± 0.174 | 7.509 ± 0.075 | 0.888 ± 0.003 | 5.726 ± 0.623 |
| Direct MSE · scalar | 4096 × 50 | 42 | 48 | 5.455 | 5.483 | 5.314 | 7.470 | 0.892 | n/a |
| TailRL (population) | 4096 × 50 | 42,43,44 | 39,33,29 | 5.482 ± 0.058 | 5.542 ± 0.081 | 5.359 ± 0.097 | 7.869 ± 0.182 | 0.869 ± 0.004 | 4.171 ± 0.079 |
| Stat MSE, width 2 (population) | 4096 × 50 | 42,43,44 | 18,27,12 | 5.821 ± 0.090 | 6.106 ± 0.159 | 5.820 ± 0.044 | 8.590 ± 0.059 | 0.860 ± 0.009 | 6.741 ± 0.992 |
| Stat MAE, width 2 (population) | 4096 × 50 | 42,43,44 | 34,17,29 | 5.804 ± 0.075 | 6.023 ± 0.126 | 5.873 ± 0.068 | 8.725 ± 0.183 | 0.851 ± 0.015 | 7.572 ± 0.960 |
| CE | 4096 × 50 | 42,43,44 | 18,16,18 | 5.932 ± 0.047 | 6.315 ± 0.046 | 6.001 ± 0.083 | 8.849 ± 0.347 | 0.860 ± 0.003 | 3.546 ± 0.083 |
| Direct MAE | 4096 × 50 | 42,43,44 | 44,45,43 | 6.498 ± 1.392 | 6.597 ± 1.384 | 6.523 ± 1.355 | 9.166 ± 1.566 | 0.864 ± 0.008 | 13.541 ± 5.416 |
| GRPO MAE (population) | 4096 × 50 | 42,43,44 | 44,40,42 | 6.982 ± 0.147 | 7.178 ± 0.062 | 7.018 ± 0.132 | 10.117 ± 0.456 | 0.854 ± 0.005 | 33.156 ± 0.301 |
| GRPO MSE (population) | 4096 × 50 | 42,43,44 | 40,21,27 | 7.348 ± 0.249 | 7.440 ± 0.231 | 7.242 ± 0.325 | 10.056 ± 0.547 | 0.861 ± 0.011 | 21.218 ± 1.459 |

## Paired schedule

Differences are first minus reference; negative MAE differences favor the first setting. Every seed pair is listed in final-comparisons.json.

| Method | First setting | Reference | Seeds | Test MAE difference | Final validation MAE difference | Best epochs, first / reference |
|---|---|---|---|---:|---:|---|
| TailRL (population) | 4096×50, categorical, flat | 4096×50, categorical, cosine | 42 | 0.167 | 0.423 | 10 / 39 |
| Direct MSE | 4096×50, categorical, flat | 4096×50, categorical, cosine | 42 | -0.048 | -0.016 | 40 / 38 |
| Direct MAE | 4096×50, categorical, flat | 4096×50, categorical, cosine | 42 | -0.015 | 0.344 | 49 / 44 |
| CE | 4096×50, categorical, flat | 4096×50, categorical, cosine | 42 | -0.163 | 0.461 | 14 / 18 |
| Stat MAE, width 2 (population) | 4096×50, categorical, flat | 4096×50, categorical, cosine | 42 | 0.012 | 0.497 | 6 / 34 |
| Stat MSE, width 2 (population) | 4096×50, categorical, flat | 4096×50, categorical, cosine | 42 | -0.074 | 0.182 | 9 / 18 |
| GRPO MAE (population) | 4096×50, categorical, flat | 4096×50, categorical, cosine | 42 | 0.153 | 0.357 | 41 / 44 |
| GRPO MSE (population) | 4096×50, categorical, flat | 4096×50, categorical, cosine | 42 | 0.334 | 0.498 | 40 / 40 |
| Direct MSE · scalar | 4096×50, scalar, flat | 4096×50, scalar, cosine | 42 | 0.012 | 0.107 | 49 / 48 |
| Direct MAE · scalar | 4096×50, scalar, flat | 4096×50, scalar, cosine | 42 | 0.176 | 0.381 | 49 / 41 |

## Paired matched_updates

Differences are first minus reference; negative MAE differences favor the first setting. Every seed pair is listed in final-comparisons.json.

| Method | First setting | Reference | Seeds | Test MAE difference | Final validation MAE difference | Best epochs, first / reference |
|---|---|---|---|---:|---:|---|
| TailRL (population) | 1024×200, categorical, cosine | 1024×50, categorical, cosine | 42 | 0.163 | -0.070 | 134 / 29 |
| Direct MSE | 1024×200, categorical, cosine | 1024×50, categorical, cosine | 42 | 0.071 | 0.128 | 51 / 41 |
| Direct MAE | 1024×200, categorical, cosine | 1024×50, categorical, cosine | 42 | 0.120 | 0.478 | 71 / 33 |
| CE | 1024×200, categorical, cosine | 1024×50, categorical, cosine | 42 | -0.049 | 0.407 | 64 / 30 |
| Stat MAE, width 2 (population) | 1024×200, categorical, cosine | 1024×50, categorical, cosine | 42 | 0.316 | 0.144 | 198 / 31 |
| Stat MSE, width 2 (population) | 1024×200, categorical, cosine | 1024×50, categorical, cosine | 42 | 0.262 | 0.334 | 168 / 40 |
| GRPO MAE (population) | 1024×200, categorical, cosine | 1024×50, categorical, cosine | 42 | 0.221 | -0.029 | 115 / 31 |
| GRPO MSE (population) | 1024×200, categorical, cosine | 1024×50, categorical, cosine | 42 | 0.530 | 0.616 | 103 / 47 |

## Paired digits

Differences are first minus reference; negative MAE differences favor the first setting. Every seed pair is listed in final-comparisons.json.

| Method | First setting | Reference | Seeds | Test MAE difference | Final validation MAE difference | Best epochs, first / reference |
|---|---|---|---|---:|---:|---|
| TailRL (population) | 4096×50, digits, cosine | 4096×50, categorical, cosine | 42 | -0.030 | 0.207 | 23 / 39 |
| TailRL (population) | 18462×50, digits, cosine | 18462×50, categorical, cosine | 42 | 0.090 | -0.131 | 22 / 13 |
| Direct MSE | 4096×50, digits, cosine | 4096×50, categorical, cosine | 42 | -0.016 | -0.148 | 50 / 38 |
| Direct MSE | 18462×50, digits, cosine | 18462×50, categorical, cosine | 42 | -0.068 | -0.007 | 36 / 47 |
| Direct MAE | 4096×50, digits, cosine | 4096×50, categorical, cosine | 42 | -0.556 | -0.299 | 45 / 44 |
| Direct MAE | 18462×50, digits, cosine | 18462×50, categorical, cosine | 42 | -0.044 | -0.139 | 42 / 40 |
| CE | 4096×50, digits, cosine | 4096×50, categorical, cosine | 42 | -0.190 | -0.046 | 7 / 18 |
| CE | 18462×50, digits, cosine | 18462×50, categorical, cosine | 42 | -0.142 | -0.191 | 21 / 24 |
| Stat MAE, width 2 (population) | 4096×50, digits, cosine | 4096×50, categorical, cosine | 42 | -0.242 | -0.147 | 21 / 34 |
| Stat MAE, width 2 (population) | 18462×50, digits, cosine | 18462×50, categorical, cosine | 42 | 0.157 | 0.030 | 25 / 14 |
| Stat MSE, width 2 (population) | 4096×50, digits, cosine | 4096×50, categorical, cosine | 42 | -0.040 | -0.157 | 39 / 18 |
| Stat MSE, width 2 (population) | 18462×50, digits, cosine | 18462×50, categorical, cosine | 42 | -0.224 | -0.096 | 14 / 9 |
| GRPO MAE (population) | 4096×50, digits, cosine | 4096×50, categorical, cosine | 42 | -0.233 | -0.226 | 49 / 44 |
| GRPO MAE (population) | 18462×50, digits, cosine | 18462×50, categorical, cosine | 42 | -0.631 | -0.656 | 50 / 41 |
| GRPO MSE (population) | 4096×50, digits, cosine | 4096×50, categorical, cosine | 42 | -0.130 | -0.177 | 40 / 40 |
| GRPO MSE (population) | 18462×50, digits, cosine | 18462×50, categorical, cosine | 42 | -0.201 | -0.259 | 37 / 41 |

## Paired scalar

Differences are first minus reference; negative MAE differences favor the first setting. Every seed pair is listed in final-comparisons.json.

| Method | First setting | Reference | Seeds | Test MAE difference | Final validation MAE difference | Best epochs, first / reference |
|---|---|---|---|---:|---:|---|
| Direct MAE · scalar | 18462×50, scalar, cosine | 18462×50, categorical, cosine | 42,43,44 | -0.130 ± 0.173 | -0.205 ± 0.195 | 41,44,42 / 40,36,32 |
| Direct MSE · scalar | 18462×50, scalar, cosine | 18462×50, categorical, cosine | 42,43,44 | -0.116 ± 0.167 | -0.169 ± 0.224 | 39,42,49 / 47,33,32 |

## Fixed training-label noise, seed 42 / field 2026

Each cell shows clean-test MAE (selected epoch). All noise arms reuse the clean 4096-subset learning rate; held-out labels are clean.

| Method | Clean | SD 1 | SD 3 | SD 5 | SD 10 |
|---|---:|---:|---:|---:|---:|
| TailRL (population) | 5.406 (e39) | 5.379 (e33) | 5.597 (e18) | 5.967 (e10) | 6.717 (e4) |
| Direct MSE | 5.216 (e38) | 5.254 (e38) | 5.449 (e44) | 5.588 (e21) | 6.252 (e14) |
| Direct MAE | 5.576 (e44) | 5.650 (e46) | 6.337 (e46) | 5.896 (e44) | 9.425 (e40) |
| CE | 6.097 (e18) | 5.887 (e18) | 6.283 (e20) | 6.312 (e14) | 7.363 (e15) |
| Stat MAE, width 2 (population) | 5.835 (e34) | 5.866 (e23) | 5.978 (e18) | 5.934 (e14) | 7.201 (e4) |
| Stat MSE, width 2 (population) | 5.769 (e18) | 5.653 (e42) | 5.927 (e21) | 6.262 (e8) | 7.357 (e9) |
| GRPO MAE (population) | 7.038 (e44) | 6.950 (e39) | 6.994 (e39) | 7.025 (e38) | 7.697 (e35) |
| GRPO MSE (population) | 6.922 (e40) | 7.125 (e50) | 7.051 (e37) | 7.070 (e27) | 7.547 (e15) |
| Direct MSE · scalar | 5.314 (e48) | 5.364 (e45) | 5.526 (e48) | 5.741 (e45) | 6.874 (e6) |
| Direct MAE · scalar | 5.145 (e41) | 5.198 (e48) | 5.351 (e45) | 5.764 (e35) | 6.734 (e24) |

## Fresh training-label noise, seed 42 / field 2026

Each cell shows clean-test MAE (selected epoch). All noise arms reuse the clean 4096-subset learning rate; held-out labels are clean.

| Method | Clean | SD 1 | SD 3 | SD 5 | SD 10 |
|---|---:|---:|---:|---:|---:|
| TailRL (population) | 5.406 (e39) | 5.350 (e25) | 5.242 (e50) | 5.215 (e40) | 5.283 (e46) |
| Direct MSE | 5.216 (e38) | 5.248 (e42) | 5.258 (e44) | 5.231 (e50) | 5.719 (e41) |
| Direct MAE | 5.576 (e44) | 5.695 (e50) | 6.136 (e48) | 5.609 (e46) | 8.251 (e47) |
| CE | 6.097 (e18) | 5.702 (e48) | 5.435 (e44) | 5.439 (e44) | 5.767 (e42) |
| Stat MAE, width 2 (population) | 5.835 (e34) | 5.814 (e39) | 5.423 (e38) | 5.409 (e38) | 5.440 (e47) |
| Stat MSE, width 2 (population) | 5.769 (e18) | 5.765 (e36) | 5.422 (e38) | 5.280 (e44) | 5.455 (e48) |
| GRPO MAE (population) | 7.038 (e44) | 6.915 (e40) | 6.919 (e48) | 7.051 (e42) | 7.046 (e47) |
| GRPO MSE (population) | 6.922 (e40) | 7.020 (e31) | 7.075 (e40) | 6.932 (e47) | 7.054 (e40) |
| Direct MSE · scalar | 5.314 (e48) | 5.275 (e50) | 5.322 (e44) | 5.489 (e44) | 5.542 (e48) |
| Direct MAE · scalar | 5.145 (e41) | 5.127 (e48) | 5.249 (e40) | 5.262 (e49) | 5.389 (e44) |

## Measured noise SNR

SNR uses clean-subset age variance divided by realized perturbation MSE after rounding/clipping. Fresh-noise MSE is averaged over epochs before conversion to dB.

| SD, years | Fixed SNR, dB | Fresh SNR, dB |
|---|---:|---:|
| 1 | 25.664 | 25.611 |
| 3 | 16.475 | 16.480 |
| 5 | 12.134 | 12.120 |
| 10 | 6.219 | 6.233 |
