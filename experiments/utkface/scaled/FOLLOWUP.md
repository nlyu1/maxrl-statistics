# UTKFace follow-up results

Snapshot: **2026-10-14 21:24 GMT+8**. [Interactive report](dashboard.html), [exact paired differences](followup-comparisons.json), [best/final/test metrics and best epochs](followup-metrics.csv), [current integrity audit](progress_integrity.json).

The original queue has completed **302 jobs**, including tuning, with **290/430 required jobs complete** and **140 required jobs pending**. It has used **46.892/64 GPU-hours across 325 attempts**. The 66-run main matrix is unchanged. Since the post-cutoff publication, 57 additional LR pilots and 29 final evaluations have completed. The deadline extension remains closed, with 56 skipped jobs and no attempts or GPU-hours used.

The comparisons below use **one matched optimization seed, 42**, fixed training subsets, 50 epochs and the same frozen training-source cohort. Noise uses field 2026. They are descriptive single-seed comparisons. Learning rates and checkpoints were selected using validation mean-prediction MAE. Test values evaluate that selected checkpoint; final-epoch values refer to validation.

| Work | Complete | Planned |
|---|---:|---:|
| Main categorical matrix, three seeds each | 66 | 66 |
| Full-data scalar confirmations | 2 | 6 |
| Fixed 4096-image subset | 6 | 10 |
| Fixed 1024-image subset | 0 | 10 |
| Constant-LR confirmations | 6 | 10 |
| Required SD-1/3/5 noise confirmations | 14 | 60 |
| Optional digit-policy confirmations | 3 | 16 |

The reduced matrix contains eight categorical objectives plus the two qualified scalar heads. The four missing 4096-image methods are population Stat-MAE/MSE width 2 and population GRPO MAE/MSE. All three prioritized digit pilots, CE/MSE/population TailRL, are complete. Other optional digit runs, SD-10 noise and optimization-seed noise repeats remain queued.

## Fixed 4096-image subset

| Method | Best epoch | Best val MAE | Final val MAE | Test MAE | Test Spearman | Test label NLL |
|---|---:|---:|---:|---:|---:|---:|
| Direct MAE, scalar | 41 | 5.200 | 5.202 | 5.145 | 0.8989 | n/a |
| Direct MSE, categorical mean | 38 | 5.440 | 5.463 | 5.216 | 0.8898 | 5.371 |
| Direct MSE, scalar | 48 | 5.455 | 5.483 | 5.314 | 0.8921 | n/a |
| Population TailRL | 39 | 5.440 | 5.463 | 5.406 | 0.8656 | 4.260 |
| Direct MAE, categorical mean | 44 | 5.509 | 5.634 | 5.576 | 0.8678 | 9.432 |
| CE | 18 | 5.980 | 6.303 | 6.097 | 0.8563 | 3.583 |

Scalar MAE has the lowest observed MAE among these six completed configurations. Its gap against the categorical-mean MAE head is **−0.431 years**; scalar MSE instead has **+0.098 years** higher MAE than categorical-mean MSE, with similar RMSE, **7.470 versus 7.483**. The tuned rates differ between heads, so this does not isolate architecture at a fixed optimizer setting. Scalar Spearman uses the continuous prediction; categorical Spearman uses the policy median. Scalar label NLL is undefined.

Full-data 50-epoch training uses 14,400 updates, versus 3,200 for 4096 images and 800 for 1024. Dataset size and optimization exposure therefore change together. No 1024-image confirmation or matched-update extension run is available. [Curves and selected-checkpoint endpoints](subset.pdf).

## Constant learning rate

Each schedule has its own validation-only LR sweep. “Practical tie” means an absolute test-MAE difference of at most 0.1 years; it is not a statistical equivalence test. Constant LR removes both warmup and cosine decay.

| Method | Cosine test MAE | Constant test MAE | Constant minus cosine | Outcome | Final-val MAE difference |
|---|---:|---:|---:|---|---:|
| Population TailRL | 5.406 | 5.573 | +0.167 | Worse | +0.423 |
| Direct MSE, categorical mean | 5.216 | 5.169 | −0.048 | Practical tie | −0.016 |
| Direct MAE, categorical mean | 5.576 | 5.561 | −0.015 | Practical tie | +0.344 |
| CE | 6.097 | 5.934 | −0.163 | Better | +0.461 |
| Direct MSE, scalar | 5.314 | 5.326 | +0.012 | Practical tie | +0.107 |
| Direct MAE, scalar | 5.145 | 5.321 | +0.176 | Worse | +0.381 |

CE's selected-checkpoint test improvement does not extend to final validation: its final validation MAE worsens by 0.461 years, and its best validation MAE is also worse under constant LR. Direct categorical MAE likewise has a selected-test practical tie while final validation worsens. This limits any general claim that removing decay helps. Final-epoch test models were not retained. [Endpoint comparison](schedule.pdf).

## Label noise and SNR

All six completed methods have both fixed and fresh SD-5 Gaussian-noise runs. Labels are rounded/clipped to 0–99; validation and test labels remain clean. Measured SNR is **12.134 dB fixed / 12.120 dB fresh**. Their nearly equal perturbation magnitudes do not remove the difference between reusing one corrupt label field and resampling labels every epoch.

| Method | Clean test MAE | Fixed SD-5 | Fresh SD-5 | Best epochs, clean/fixed/fresh |
|---|---:|---:|---:|---|
| Population TailRL | 5.406 | 5.967 | 5.215 | 39 / 10 / 40 |
| Direct MSE, categorical mean | 5.216 | 5.588 | 5.231 | 38 / 21 / 50 |
| Direct MAE, categorical mean | 5.576 | 5.896 | 5.609 | 44 / 44 / 46 |
| CE | 6.097 | 6.312 | 5.439 | 18 / 14 / 44 |
| Direct MSE, scalar | 5.314 | 5.741 | 5.489 | 48 / 45 / 44 |
| Direct MAE, scalar | 5.145 | 5.764 | 5.262 | 41 / 35 / 49 |

Fresh SD-5 noise has lower test MAE than fixed SD-5 noise for all six completed methods. Against clean training, fresh noise improves TailRL by **0.191 years** and CE by **0.658**, approximately preserves categorical MSE/MAE, and worsens the scalar heads. This is consistent with a regularization hypothesis, but one training seed and one noise field cannot establish the mechanism or general robustness.

The other two completed noise runs use fixed SD 1, measured SNR **25.664 dB**: CE reaches **5.887** at epoch 18, and scalar MAE **5.198** at epoch 48. SD-3 results and fresh SD-1 results are missing, so these observations do not establish a noise dose-response curve. Independent noise fields 2027/2028 were deadline-skipped. Fresh-noise categorical MSE selects epoch 50, leaving possible horizon truncation. [Validation curves](noise.pdf), [observed SNR and error changes](snr.pdf).

## Conditional 10×10 policy

| Objective | Flat test MAE | Digit test MAE | Flat → digit NLL | Best epochs, flat/digit |
|---|---:|---:|---|---|
| Population TailRL | 5.406 | 5.376 | 4.260 → 3.650 | 39 / 23 |
| Direct MSE | 5.216 | 5.200 | 5.371 → 4.608 | 38 / 50 |
| CE | 6.097 | 5.907 | 3.583 → 3.288 | 18 / 7 |

TailRL and MSE have small observed MAE differences; CE improves by **0.190 years**. Label NLL is lower for all three digit pilots. These results do not establish equivalence or a replicated advantage: the conditional head has 73,364 parameters versus 51,300 for the flat head, and its LR sweep has three candidates versus four. Digit MSE selects the final available epoch. Both heads enumerate the same 100 ages; this experiment changes the distribution parameterization. [Matched-seed curves](digits.pdf).

## Validation and remaining work

The fresh audit checks all 302 completed histories, complete epoch counts, finite training values, common split/subset fingerprints, reported phase metrics against raw results, validation-only LR selection and clean held-out counts. Noise-field statistics agree across methods for shared realizations. All 11 frozen training sources and both immutable plans remain intact. All 224 attempts present before supervisor handoff remain preserved; every subsequent attempt is retained and charged. Historical peak study concurrency is six GPUs, matching the cap.

The new report and comparison figures are reproducible remotely with `uv run --python 3.12 python -m src.experiments.utkface.progress_report --plan /tmp/utkface-study-20261013/plan.json --report-dir /tmp/utkface-study-20261013/report --coordination /tmp/utkface-extension-20261014/coordination.json --tie-years 0.1`, after the standard report refresh. JavaScript interaction checks passed on this snapshot. Browser rendering remains unverified because the in-app browser blocks file URLs.

The original supervisor is healthy and waiting for free GPUs. The priority gaps are the remaining four reduced categorical objectives, all 1024-image comparisons, full-data scalar replications, and missing noise settings. The original 64 GPU-hour cap and shared six-GPU cap remain enforced. [The 07:30 extension cutoff audit](COVERAGE.md) is a historical snapshot, distinct from this progress report.
