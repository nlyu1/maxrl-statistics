# Full-data scalar heads: three-seed results

All six scalar confirmations completed, using seeds **42, 43 and 44** and 50 full-data epochs. Both scalar and corresponding categorical-mean heads use learning rate **0.001**, the same pretrained ResNet-18 backbone, batch 64, bf16, AdamW and cosine schedule. Scalar output is `99 * sigmoid(linear(features))`.

| Objective and head | Test MAE, mean ± sample SD | Best validation MAE | Final validation MAE | Best epochs, 42/43/44 |
|---|---:|---:|---:|---|
| MAE, scalar | **4.632 ± 0.023** | 4.667 | 4.716 | 41 / 44 / 42 |
| MAE, categorical mean | 4.762 ± 0.193 | 4.860 | 4.921 | 40 / 36 / 32 |
| MSE, scalar | **4.701 ± 0.025** | 4.771 | 4.817 | 39 / 42 / 49 |
| MSE, categorical mean | 4.817 ± 0.191 | 4.902 | 4.987 | 47 / 33 / 32 |

Scalar-minus-categorical MAE differences are **−0.064, −0.326, +0.000** years for the MAE objective, averaging **−0.130 ± 0.173**. For MSE they are **−0.029, −0.309, −0.011**, averaging **−0.116 ± 0.167**. The corresponding-head mean improvements are influenced strongly by categorical seed 43; scalar MAE is effectively tied on seed 44.

Against population Stat-MAE width 2, scalar MAE improves by **0.008, 0.106 and 0.143 years**, averaging **0.085 ± 0.070**. This is the lowest observed three-seed full-data mean in the completed study, with only three optimization seeds on one fixed split.

[Three-seed curves and endpoints](final-scalar.pdf) · [All metrics](FINAL_TABLES.md) · [Exact paired differences](final-comparisons.json) · [Interactive report](dashboard.html).

Scalar Spearman uses continuous outputs, reaching approximately **0.9108 for MAE** and **0.9114 for MSE**. Categorical Spearman uses the policy median, so the decision rules differ. Scalar label NLL is undefined. Test metrics evaluate the minimum-validation-MAE checkpoint; final-epoch test models were not retained.

The 4096-image scalar MAE/MSE results are **5.145 / 5.314** at selected epochs **41 / 48**; the 1024-image results are **6.177 / 6.367**, both at epoch **44**. These subset results have **one seed**, and the tuned rates differ across heads. They do not supply the same three-seed architectural comparison as full-data training.
