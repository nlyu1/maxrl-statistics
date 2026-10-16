# Scaled UTKFace study

State: **completed**. GPU time: **62.07 hours**; original planning reference: **64 hours**. Completed primary runs: **66 / 66**.

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

| Stage | Variant | Training size | Noise | Seeds | Mean MAE ↓ | Median MAE ↓ | Median Spearman ↑ | Label NLL ↓ |
|---|---|---:|---|---:|---:|---:|---:|---:|
| digits | mae_digits | full | none 0 | 1 | 4.631 | 4.920 | 0.903 | 4.054 |
| digits | stat_maxrl_mse_population_s2_digits | full | none 0 | 1 | 4.638 | 4.662 | 0.905 | 6.112 |
| digits | mse_digits | full | none 0 | 1 | 4.640 | 4.939 | 0.907 | 6.424 |
| digits | stat_maxrl_mae_population_s2_digits | full | none 0 | 1 | 4.776 | 4.888 | 0.900 | 8.378 |
| digits | ce_digits | full | none 0 | 1 | 4.784 | 4.853 | 0.897 | 3.236 |
| digits | tailrl_population_digits | full | none 0 | 1 | 4.808 | 4.781 | 0.906 | 3.183 |
| digits | grpo_mae_population_digits | full | none 0 | 1 | 6.010 | 6.071 | 0.901 | 66.424 |
| digits | grpo_mse_population_digits | full | none 0 | 1 | 6.319 | 6.377 | 0.898 | 54.502 |
| digits | mae_digits | 4096 | none 0 | 1 | 5.020 | 5.317 | 0.886 | 3.885 |
| digits | mse_digits | 4096 | none 0 | 1 | 5.200 | 5.650 | 0.885 | 4.608 |
| digits | tailrl_population_digits | 4096 | none 0 | 1 | 5.376 | 5.435 | 0.884 | 3.650 |
| digits | stat_maxrl_mae_population_s2_digits | 4096 | none 0 | 1 | 5.593 | 5.885 | 0.861 | 6.849 |
| digits | stat_maxrl_mse_population_s2_digits | 4096 | none 0 | 1 | 5.729 | 5.984 | 0.860 | 9.546 |
| digits | ce_digits | 4096 | none 0 | 1 | 5.907 | 6.065 | 0.867 | 3.288 |
| digits | grpo_mse_population_digits | 4096 | none 0 | 1 | 6.792 | 6.879 | 0.870 | 31.589 |
| digits | grpo_mae_population_digits | 4096 | none 0 | 1 | 6.805 | 7.014 | 0.856 | 30.390 |
| flat | mse | 4096 | none 0 | 1 | 5.169 | 5.536 | 0.884 | 5.003 |
| flat | mae_scalar | 4096 | none 0 | 1 | 5.321 | 5.321 | 0.889 | n/a |
| flat | mse_scalar | 4096 | none 0 | 1 | 5.326 | 5.326 | 0.895 | n/a |
| flat | mae | 4096 | none 0 | 1 | 5.561 | 5.671 | 0.881 | 6.773 |
| flat | tailrl_population | 4096 | none 0 | 1 | 5.573 | 5.592 | 0.875 | 3.525 |
| flat | stat_maxrl_mse_population_s2 | 4096 | none 0 | 1 | 5.695 | 5.900 | 0.859 | 5.714 |
| flat | stat_maxrl_mae_population_s2 | 4096 | none 0 | 1 | 5.847 | 5.905 | 0.861 | 4.993 |
| flat | ce | 4096 | none 0 | 1 | 5.934 | 6.063 | 0.864 | 3.721 |
| flat | grpo_mae_population | 4096 | none 0 | 1 | 7.191 | 7.274 | 0.840 | 23.082 |
| flat | grpo_mse_population | 4096 | none 0 | 1 | 7.255 | 7.442 | 0.845 | 17.288 |
| main | stat_maxrl_mae_population_s2 | full | none 0 | 3 | 4.718 ± 0.086 | 4.828 ± 0.120 | 0.902 ± 0.006 | 7.439 ± 0.451 |
| main | tailrl | full | none 0 | 3 | 4.724 ± 0.019 | 4.765 ± 0.020 | 0.905 ± 0.002 | 6.698 ± 0.626 |
| main | tailrl_population | full | none 0 | 3 | 4.752 ± 0.037 | 4.730 ± 0.081 | 0.903 ± 0.004 | 3.214 ± 0.222 |
| main | mae | full | none 0 | 3 | 4.762 ± 0.193 | 4.905 ± 0.197 | 0.905 ± 0.005 | 9.451 ± 1.390 |
| main | stat_maxrl_mae_s2 | full | none 0 | 3 | 4.769 ± 0.086 | 4.871 ± 0.088 | 0.900 ± 0.008 | 10.010 ± 1.083 |
| main | stat_maxrl_mae_s5 | full | none 0 | 3 | 4.793 ± 0.113 | 4.859 ± 0.106 | 0.904 ± 0.001 | 18.014 ± 2.437 |
| main | mse | full | none 0 | 3 | 4.817 ± 0.191 | 4.956 ± 0.179 | 0.908 ± 0.002 | 8.099 ± 0.864 |
| main | stat_maxrl_mae_population_s5 | full | none 0 | 3 | 4.871 ± 0.118 | 5.028 ± 0.127 | 0.894 ± 0.003 | 13.023 ± 0.674 |
| main | stat_maxrl_mse_population_s2 | full | none 0 | 3 | 4.892 ± 0.031 | 5.024 ± 0.029 | 0.898 ± 0.001 | 6.234 ± 0.592 |
| main | stat_maxrl_mse_s2 | full | none 0 | 3 | 4.923 ± 0.096 | 5.026 ± 0.073 | 0.900 ± 0.004 | 7.150 ± 0.318 |
| main | binary_maxrl_population | full | none 0 | 3 | 4.933 ± 0.030 | 4.969 ± 0.024 | 0.897 ± 0.002 | 3.195 ± 0.124 |
| main | binary_maxrl | full | none 0 | 3 | 4.961 ± 0.065 | 5.070 ± 0.102 | 0.891 ± 0.005 | 4.387 ± 0.288 |
| main | stat_maxrl_mse_population_s5 | full | none 0 | 3 | 5.002 ± 0.228 | 5.096 ± 0.217 | 0.898 ± 0.001 | 15.107 ± 1.851 |
| main | stat_maxrl_mse_s5 | full | none 0 | 3 | 5.095 ± 0.306 | 5.303 ± 0.284 | 0.896 ± 0.008 | 14.354 ± 2.295 |
| main | grpo_mae | full | none 0 | 3 | 6.342 ± 0.109 | 6.409 ± 0.099 | 0.891 ± 0.007 | 28.755 ± 1.572 |
| main | grpo_mae_population | full | none 0 | 3 | 6.526 ± 0.126 | 6.559 ± 0.136 | 0.888 ± 0.003 | 60.694 ± 5.382 |
| main | rloo_mae | full | none 0 | 3 | 6.573 ± 0.288 | 6.720 ± 0.318 | 0.884 ± 0.003 | 17.929 ± 0.795 |
| main | rloo_mae_population | full | none 0 | 3 | 6.603 ± 0.336 | 6.746 ± 0.337 | 0.884 ± 0.003 | 18.375 ± 0.743 |
| main | grpo_mse | full | none 0 | 3 | 6.614 ± 0.242 | 6.677 ± 0.243 | 0.891 ± 0.005 | 29.489 ± 0.792 |
| main | rloo_mse_population | full | none 0 | 3 | 6.640 ± 0.450 | 6.702 ± 0.465 | 0.888 ± 0.006 | 39.942 ± 0.837 |
| main | rloo_mse | full | none 0 | 3 | 6.657 ± 0.465 | 6.734 ± 0.476 | 0.885 ± 0.006 | 41.929 ± 1.607 |
| main | grpo_mse_population | full | none 0 | 3 | 6.808 ± 0.281 | 6.881 ± 0.279 | 0.887 ± 0.007 | 35.231 ± 2.314 |
| noise | mae_scalar | 4096 | fresh 1 | 1 | 5.127 | 5.127 | 0.892 | n/a |
| noise | mae_scalar | 4096 | fixed 1 | 1 | 5.198 | 5.198 | 0.889 | n/a |
| noise | mse | 4096 | fresh 1 | 1 | 5.248 | 5.410 | 0.889 | 5.363 |
| noise | mse | 4096 | fixed 1 | 1 | 5.254 | 5.407 | 0.884 | 5.341 |
| noise | mse_scalar | 4096 | fresh 1 | 1 | 5.275 | 5.275 | 0.893 | n/a |
| noise | tailrl_population | 4096 | fresh 1 | 1 | 5.350 | 5.524 | 0.874 | 3.910 |
| noise | mse_scalar | 4096 | fixed 1 | 1 | 5.364 | 5.364 | 0.892 | n/a |
| noise | tailrl_population | 4096 | fixed 1 | 1 | 5.379 | 5.638 | 0.861 | 4.262 |
| noise | mae | 4096 | fixed 1 | 1 | 5.650 | 6.126 | 0.866 | 9.470 |
| noise | stat_maxrl_mse_population_s2 | 4096 | fixed 1 | 1 | 5.653 | 5.984 | 0.857 | 7.769 |
| noise | mae | 4096 | fresh 1 | 1 | 5.695 | 6.091 | 0.859 | 9.331 |
| noise | ce | 4096 | fresh 1 | 1 | 5.702 | 5.804 | 0.867 | 4.347 |
| noise | stat_maxrl_mse_population_s2 | 4096 | fresh 1 | 1 | 5.765 | 6.006 | 0.858 | 7.243 |
| noise | stat_maxrl_mae_population_s2 | 4096 | fresh 1 | 1 | 5.814 | 6.078 | 0.856 | 6.795 |
| noise | stat_maxrl_mae_population_s2 | 4096 | fixed 1 | 1 | 5.866 | 6.087 | 0.854 | 6.666 |
| noise | ce | 4096 | fixed 1 | 1 | 5.887 | 6.032 | 0.863 | 3.745 |
| noise | grpo_mae_population | 4096 | fresh 1 | 1 | 6.915 | 6.934 | 0.854 | 32.969 |
| noise | grpo_mae_population | 4096 | fixed 1 | 1 | 6.950 | 7.023 | 0.861 | 31.264 |
| noise | grpo_mse_population | 4096 | fresh 1 | 1 | 7.020 | 7.186 | 0.856 | 21.166 |
| noise | grpo_mse_population | 4096 | fixed 1 | 1 | 7.125 | 7.244 | 0.852 | 19.984 |
| noise | tailrl_population | 4096 | fresh 3 | 1 | 5.242 | 5.295 | 0.884 | 3.551 |
| noise | mae_scalar | 4096 | fresh 3 | 1 | 5.249 | 5.249 | 0.893 | n/a |
| noise | mse | 4096 | fresh 3 | 1 | 5.258 | 5.412 | 0.886 | 5.214 |
| noise | mse_scalar | 4096 | fresh 3 | 1 | 5.322 | 5.322 | 0.891 | n/a |
| noise | mae_scalar | 4096 | fixed 3 | 1 | 5.351 | 5.351 | 0.887 | n/a |
| noise | stat_maxrl_mse_population_s2 | 4096 | fresh 3 | 1 | 5.422 | 5.453 | 0.878 | 4.773 |
| noise | stat_maxrl_mae_population_s2 | 4096 | fresh 3 | 1 | 5.423 | 5.496 | 0.881 | 5.143 |
| noise | ce | 4096 | fresh 3 | 1 | 5.435 | 5.450 | 0.874 | 3.471 |
| noise | mse | 4096 | fixed 3 | 1 | 5.449 | 5.550 | 0.877 | 5.180 |
| noise | mse_scalar | 4096 | fixed 3 | 1 | 5.526 | 5.526 | 0.882 | n/a |
| noise | tailrl_population | 4096 | fixed 3 | 1 | 5.597 | 5.705 | 0.869 | 4.010 |
| noise | stat_maxrl_mse_population_s2 | 4096 | fixed 3 | 1 | 5.927 | 6.195 | 0.856 | 6.732 |
| noise | stat_maxrl_mae_population_s2 | 4096 | fixed 3 | 1 | 5.978 | 6.233 | 0.843 | 6.161 |
| noise | mae | 4096 | fresh 3 | 1 | 6.136 | 6.270 | 0.870 | 11.416 |
| noise | ce | 4096 | fixed 3 | 1 | 6.283 | 6.501 | 0.851 | 3.902 |
| noise | mae | 4096 | fixed 3 | 1 | 6.337 | 6.527 | 0.860 | 13.943 |
| noise | grpo_mae_population | 4096 | fresh 3 | 1 | 6.919 | 6.987 | 0.865 | 31.279 |
| noise | grpo_mae_population | 4096 | fixed 3 | 1 | 6.994 | 7.062 | 0.860 | 30.713 |
| noise | grpo_mse_population | 4096 | fixed 3 | 1 | 7.051 | 7.146 | 0.859 | 20.871 |
| noise | grpo_mse_population | 4096 | fresh 3 | 1 | 7.075 | 7.127 | 0.858 | 21.718 |
| noise | tailrl_population | 4096 | fresh 5 | 1 | 5.215 | 5.228 | 0.888 | 3.330 |
| noise | mse | 4096 | fresh 5 | 1 | 5.231 | 5.416 | 0.888 | 5.209 |
| noise | mae_scalar | 4096 | fresh 5 | 1 | 5.262 | 5.262 | 0.895 | n/a |
| noise | stat_maxrl_mse_population_s2 | 4096 | fresh 5 | 1 | 5.280 | 5.272 | 0.889 | 4.169 |
| noise | stat_maxrl_mae_population_s2 | 4096 | fresh 5 | 1 | 5.409 | 5.469 | 0.883 | 4.236 |
| noise | ce | 4096 | fresh 5 | 1 | 5.439 | 5.421 | 0.881 | 3.375 |
| noise | mse_scalar | 4096 | fresh 5 | 1 | 5.489 | 5.489 | 0.889 | n/a |
| noise | mse | 4096 | fixed 5 | 1 | 5.588 | 5.718 | 0.873 | 5.003 |
| noise | mae | 4096 | fresh 5 | 1 | 5.609 | 5.741 | 0.876 | 8.033 |
| noise | mse_scalar | 4096 | fixed 5 | 1 | 5.741 | 5.741 | 0.877 | n/a |
| noise | mae_scalar | 4096 | fixed 5 | 1 | 5.764 | 5.764 | 0.874 | n/a |
| noise | mae | 4096 | fixed 5 | 1 | 5.896 | 6.093 | 0.860 | 11.393 |
| noise | stat_maxrl_mae_population_s2 | 4096 | fixed 5 | 1 | 5.934 | 6.218 | 0.845 | 5.036 |
| noise | tailrl_population | 4096 | fixed 5 | 1 | 5.967 | 6.014 | 0.859 | 3.589 |
| noise | stat_maxrl_mse_population_s2 | 4096 | fixed 5 | 1 | 6.262 | 6.253 | 0.849 | 4.281 |
| noise | ce | 4096 | fixed 5 | 1 | 6.312 | 6.455 | 0.854 | 3.609 |
| noise | grpo_mse_population | 4096 | fresh 5 | 1 | 6.932 | 7.020 | 0.866 | 22.058 |
| noise | grpo_mae_population | 4096 | fixed 5 | 1 | 7.025 | 7.058 | 0.856 | 31.220 |
| noise | grpo_mae_population | 4096 | fresh 5 | 1 | 7.051 | 7.113 | 0.863 | 31.798 |
| noise | grpo_mse_population | 4096 | fixed 5 | 1 | 7.070 | 7.158 | 0.857 | 21.013 |
| noise | tailrl_population | 4096 | fresh 10 | 1 | 5.283 | 5.121 | 0.896 | 3.390 |
| noise | mae_scalar | 4096 | fresh 10 | 1 | 5.389 | 5.389 | 0.889 | n/a |
| noise | stat_maxrl_mae_population_s2 | 4096 | fresh 10 | 1 | 5.440 | 5.470 | 0.888 | 4.309 |
| noise | stat_maxrl_mse_population_s2 | 4096 | fresh 10 | 1 | 5.455 | 5.297 | 0.888 | 3.944 |
| noise | mse_scalar | 4096 | fresh 10 | 1 | 5.542 | 5.542 | 0.887 | n/a |
| noise | mse | 4096 | fresh 10 | 1 | 5.719 | 5.914 | 0.878 | 5.657 |
| noise | ce | 4096 | fresh 10 | 1 | 5.767 | 5.682 | 0.876 | 3.585 |
| noise | mse | 4096 | fixed 10 | 1 | 6.252 | 6.571 | 0.846 | 4.913 |
| noise | tailrl_population | 4096 | fixed 10 | 1 | 6.717 | 6.697 | 0.857 | 3.739 |
| noise | mae_scalar | 4096 | fixed 10 | 1 | 6.734 | 6.734 | 0.852 | n/a |
| noise | mse_scalar | 4096 | fixed 10 | 1 | 6.874 | 6.874 | 0.848 | n/a |
| noise | grpo_mae_population | 4096 | fresh 10 | 1 | 7.046 | 7.139 | 0.858 | 30.712 |
| noise | grpo_mse_population | 4096 | fresh 10 | 1 | 7.054 | 7.125 | 0.860 | 21.001 |
| noise | stat_maxrl_mae_population_s2 | 4096 | fixed 10 | 1 | 7.201 | 7.553 | 0.802 | 4.711 |
| noise | stat_maxrl_mse_population_s2 | 4096 | fixed 10 | 1 | 7.357 | 7.756 | 0.824 | 4.585 |
| noise | ce | 4096 | fixed 10 | 1 | 7.363 | 7.190 | 0.837 | 3.865 |
| noise | grpo_mse_population | 4096 | fixed 10 | 1 | 7.547 | 7.631 | 0.839 | 18.742 |
| noise | grpo_mae_population | 4096 | fixed 10 | 1 | 7.697 | 7.795 | 0.819 | 29.680 |
| noise | mae | 4096 | fresh 10 | 1 | 8.251 | 8.751 | 0.841 | 16.327 |
| noise | mae | 4096 | fixed 10 | 1 | 9.425 | 9.486 | 0.760 | 23.843 |
| scalar | mae_scalar | full | none 0 | 3 | 4.632 ± 0.023 | 4.632 ± 0.023 | 0.911 ± 0.001 | n/a |
| scalar | mse_scalar | full | none 0 | 3 | 4.701 ± 0.025 | 4.701 ± 0.025 | 0.911 ± 0.002 | n/a |
| subset | mae | 1024 | none 0 | 1 | 6.087 | 6.371 | 0.866 | 4.792 |
| subset | mae_scalar | 1024 | none 0 | 1 | 6.177 | 6.177 | 0.872 | n/a |
| subset | mse | 1024 | none 0 | 1 | 6.301 | 6.660 | 0.859 | 4.403 |
| subset | mse_scalar | 1024 | none 0 | 1 | 6.367 | 6.367 | 0.866 | n/a |
| subset | tailrl_population | 1024 | none 0 | 1 | 6.403 | 6.642 | 0.831 | 4.432 |
| subset | stat_maxrl_mse_population_s2 | 1024 | none 0 | 1 | 6.835 | 7.276 | 0.802 | 6.846 |
| subset | stat_maxrl_mae_population_s2 | 1024 | none 0 | 1 | 6.899 | 7.351 | 0.801 | 6.651 |
| subset | ce | 1024 | none 0 | 1 | 7.329 | 7.600 | 0.779 | 4.832 |
| subset | grpo_mae_population | 1024 | none 0 | 1 | 7.732 | 7.853 | 0.805 | 13.191 |
| subset | grpo_mse_population | 1024 | none 0 | 1 | 7.784 | 7.949 | 0.814 | 14.247 |
| subset | mae_scalar | 4096 | none 0 | 1 | 5.145 | 5.145 | 0.899 | n/a |
| subset | mse | 4096 | none 0 | 1 | 5.216 | 5.392 | 0.890 | 5.371 |
| subset | mse_scalar | 4096 | none 0 | 1 | 5.314 | 5.314 | 0.892 | n/a |
| subset | tailrl_population | 4096 | none 0 | 1 | 5.406 | 5.646 | 0.866 | 4.260 |
| subset | mae | 4096 | none 0 | 1 | 5.576 | 5.901 | 0.868 | 9.432 |
| subset | stat_maxrl_mse_population_s2 | 4096 | none 0 | 1 | 5.769 | 6.036 | 0.868 | 6.185 |
| subset | stat_maxrl_mae_population_s2 | 4096 | none 0 | 1 | 5.835 | 6.128 | 0.849 | 8.519 |
| subset | ce | 4096 | none 0 | 1 | 6.097 | 6.273 | 0.856 | 3.583 |
| subset | grpo_mse_population | 4096 | none 0 | 1 | 6.922 | 7.014 | 0.865 | 20.499 |
| subset | grpo_mae_population | 4096 | none 0 | 1 | 7.038 | 7.112 | 0.848 | 33.041 |

![Main comparison](main.png)

GPU time charges every launched attempt, including failures, retries, process startup, final test evaluation and polling overhead. Waiting for free GPUs consumes no GPU budget. Optional studies start after required studies finish. SNR is 10 log10(variance of clean subset ages / mean squared realized label perturbation), after rounding and clipping. Clean-label SNR is infinite and stored as null.

[Machine-readable progress](progress.json).

[Per-run values and LR tuning](runs.csv). [Full configs and metrics](results.json). [Epoch curves and realized noise statistics](curves.csv).
