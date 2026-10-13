# Scalar versus categorical heads: matched seed 42

Snapshot: 2026-10-14T07:07:21.781072+08:00. Both heads train for 50 full-data epochs on the same split and training-source cohort. Checkpoints minimize validation mean-prediction MAE. Test labels remain clean.

| Objective | Head | LR | Best epoch | Best val MAE | Final val MAE | Test MAE | Spearman |
|---|---|---:|---:|---:|---:|---:|---:|
| MSE | categorical | 0.001 | 47 | 4.8206 | 4.8930 | 4.7087 | 0.9086 |
| MSE | scalar | 0.001 | 39 | 4.7511 | 4.7924 | 4.6798 | 0.9124 |
| MAE | categorical | 0.001 | 40 | 4.8106 | 4.8649 | 4.6749 | 0.9007 |
| MAE | scalar | 0.001 | 41 | 4.7318 | 4.7485 | 4.6112 | 0.9111 |

MSE: scalar-minus-categorical test MAE is -0.0289 years (-0.61%). This is one matched seed, not a replicated advantage.
MAE: scalar-minus-categorical test MAE is -0.0636 years (-1.36%). This is one matched seed, not a replicated advantage.

Scalar label NLL is undefined. Scalar Spearman ranks continuous predictions, whereas categorical Spearman ranks policy medians. Learning rates are validation-selected independently. Final-epoch values are validation metrics; final-epoch test models were not retained.

Completed scalar confirmations: 2/6. Missing scalar seeds and objectives remain visible in the dashboard.

[Curves and selected test endpoints](scalar.pdf). [Exact paired metrics](scalar-comparison.json).
