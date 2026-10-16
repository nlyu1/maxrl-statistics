# UTKFace completed study

Verified October 16, 2026, GMT+8. Training finished **October 15 at 12:15:54 GMT+8**. The interactive report was refreshed at **14:47 on October 16**; its timestamp identifies retrieval and analysis, not new training.

[Interactive report](dashboard.html) · [all final tables and selected epochs](FINAL_TABLES.md) · [grouped metrics](final-summaries.csv) · [exact paired comparisons](final-comparisons.json) · [completion audit](final-audit.json) · [individual runs](dashboard-runs.csv) · [all validation curves](dashboard-curves.csv).

## Completion and scope

All **212 jobs selected in the October 15 update completed**. Across both queues there are **514 completed jobs**: 288 learning-rate pilots, four real-image preflight checks and **222 scientific final evaluations**. Every selected run completed its planned horizon. The original queue contributes 490 completed jobs and the continuation 24. Both supervisors exited normally after completion.

Total recorded usage is **65.265 GPU-hours across 542 attempts**, including interrupted attempts: 62.070 hours in the original queue and 3.194 hours in the continuation. GPU-hour enforcement was disabled as requested; accounting was retained.

| Scientific comparison | Completed runs | Optimization seeds |
|---|---:|---|
| Main 22-method categorical matrix, full data × 50 epochs | 66 | 42, 43, 44 for every method |
| Full-data scalar MAE/MSE heads × 50 epochs | 6 | 42, 43, 44 for each head |
| Clean 4096-image subset × 50 epochs | 26 | Three per categorical method; one per scalar head |
| Clean 1024-image subset × 50 epochs | 10 | 42 |
| Clean 1024-image subset × 200 epochs | 8 | 42, categorical methods |
| Constant-LR comparisons, 4096 images × 50 epochs | 10 | 42 |
| Fixed/fresh noise, SD 1/3/5/10, 4096 images × 50 epochs | 80 | Optimization seed 42; noise seed 2026 |
| Conditional 10×10 digit policies, subset/full data × 50 epochs | 16 | 42 |

**Forty optional noise training-seed repetitions remain held and unrun.** Thirty-two independent SD-5 noise-field experiments were excluded at the user's request. These are deliberate scope exclusions, not failed runs. The old 56-job deadline extension remains an unchanged historical ledger; its selected 24 jobs completed in the new continuation. No selected work remains pending.

## Full-data comparison

All values below are clean-test MAE at the checkpoint selected by minimum validation mean-age MAE. ± denotes sample SD across three optimization seeds on one fixed split, not a confidence interval. The 66-run categorical matrix is unchanged from the earlier report; the scalar confirmations now supply three-seed evidence.

| Method | Test MAE, years | Test label NLL, nats | Best epochs, seeds 42/43/44 |
|---|---:|---:|---|
| Direct MAE, scalar | **4.632 ± 0.023** | Undefined | 41 / 44 / 42 |
| Direct MSE, scalar | 4.701 ± 0.025 | Undefined | 39 / 42 / 49 |
| Population Stat MAE, width 2 | 4.718 ± 0.086 | 7.439 | 14 / 16 / 16 |
| Sampled TailRL, 64 actions | 4.724 ± 0.019 | 6.698 | 41 / 26 / 22 |
| Population TailRL | 4.752 ± 0.037 | 3.214 | 13 / 18 / 7 |
| Direct MAE, categorical mean | 4.762 ± 0.193 | 9.451 | 40 / 36 / 32 |
| Direct MSE, categorical mean | 4.817 ± 0.191 | 8.099 | 47 / 33 / 32 |
| Population binary MaxRL / CE | 4.933 ± 0.030 | 3.195 | 24 / 23 / 18 |

Scalar MAE has the lowest observed three-seed mean MAE. Its paired differences against population Stat-MAE width 2 are **−0.008, −0.106 and −0.143 years**, averaging **−0.085 ± 0.070**. It improves on that comparator on all three seeds, although the absolute differences are small and three seeds on one split do not establish a general advantage.

Against the corresponding categorical-mean objective, scalar MAE improves by **0.130 years on average**, while scalar MSE improves by **0.116**. Those averages are influenced strongly by a weaker categorical seed 43. Scalar MAE is effectively tied with categorical MAE on seed 44; scalar MSE improves over categorical MSE on all three seeds. Both full-data head comparisons use learning rate **0.001**, so this comparison avoids the different-rate confound present in the small-subset head comparisons. [Three-seed curves and endpoints](final-scalar.pdf).

The distributional tradeoff remains important. Population TailRL has similar point error to the leading categorical objectives and much lower label NLL than mean-regression or narrow statistical-kernel objectives. Scalar heads do not supply a categorical likelihood. Scalar Spearman uses the continuous output, while categorical Spearman uses the policy median; those rank metrics use different decision rules.

The eight GRPO/RLOO variants in the main matrix retain substantially higher MAE, approximately **6.34–6.81 years**, and high label NLL. Population estimators do not uniformly improve generalization: five of ten population-versus-sampled objective pairs improve mean MAE and five worsen. [Complete categorical matrix and paired differences](MAIN_MATRIX.md).

## Fixed subsets and training exposure

The categorical 4096-image comparisons now have all three matched seeds:

| Categorical method | Test MAE, years | Best epochs, 42/43/44 |
|---|---:|---|
| Direct MSE | **5.297 ± 0.174** | 38 / 40 / 40 |
| Population TailRL | 5.359 ± 0.097 | 39 / 33 / 29 |
| Population Stat MSE, width 2 | 5.820 ± 0.044 | 18 / 27 / 12 |
| Population Stat MAE, width 2 | 5.873 ± 0.068 | 34 / 17 / 29 |
| CE | 6.001 ± 0.083 | 18 / 16 / 18 |
| Direct MAE | 6.523 ± 1.355 | 44 / 45 / 43 |
| Population GRPO MAE | 7.018 ± 0.132 | 44 / 40 / 42 |
| Population GRPO MSE | 7.242 ± 0.325 | 40 / 21 / 27 |

MSE and TailRL have the lowest means among the replicated small-subset categorical methods. Statistical MaxRL's favorable full-data ranking does not carry over to this setting. The MSE–TailRL difference is small relative to their observed seed variability; these results do not establish a reliable winner between them.

The clean 4096-image curves show late validation degradation, especially for CE: its three-seed mean rises from **5.932** at the individually selected checkpoints, epochs **18 / 16 / 18**, to **6.315** at epoch 50. Stat MSE rises **5.821 → 6.106** and Stat MAE **5.804 → 6.023**. MSE and TailRL degrade less, **5.469 → 5.518** and **5.482 → 5.542**. This pattern is consistent with overfitting and supports checkpoint selection, although it does not establish that every objective is overfitting equally. Optimization instability and the seed-44 MAE failure below are separate concerns.

Categorical MAE is notably seed-sensitive at its validation-selected rate **0.003**: test MAEs are **5.576, 5.917 and 8.074**. The last run also has best validation MAE **8.089**, so its poor outcome is visible without examining test performance. It completed with finite training values and is retained in the analysis. A rate selected using a single 20-epoch seed-42 pilot did not transfer reliably across optimization seeds. [All three-seed points](final-replication.pdf).

The small-subset scalar heads have only seed 42: MAE/MSE reach **5.145 / 5.314** years at 4096 images and **6.177 / 6.367** at 1024. They should not be treated as three-seed subset results. At 1024 images, categorical MAE reaches **6.087** in the single-seed 50-epoch experiment.

At batch 64, full-data × 50 uses **14,400 updates**, 4096 × 50 uses **3,200**, and 1024 × 50 uses **800**. The eight reinstated 1024 × 200 runs match 4096 × 50 at 3,200 updates and 204,800 training presentations.

| Categorical method, seed 42 | 4096 × 50 | 1024 × 50 | 1024 × 200 | 200 minus 50 epochs |
|---|---:|---:|---:|---:|
| Population TailRL | 5.406 | 6.403 | 6.565 | +0.163 |
| Direct MSE | 5.216 | 6.301 | 6.371 | +0.071 |
| Direct MAE | 5.576 | 6.087 | 6.208 | +0.120 |
| CE | 6.097 | 7.329 | 7.280 | −0.049 |
| Population Stat MAE, width 2 | 5.835 | 6.899 | 7.215 | +0.316 |
| Population Stat MSE, width 2 | 5.769 | 6.835 | 7.098 | +0.262 |
| Population GRPO MAE | 7.038 | 7.732 | 7.953 | +0.221 |
| Population GRPO MSE | 6.922 | 7.784 | 8.314 | +0.530 |

The longer setting closes none of the gap to 4096 images. Using the predefined 0.1-year descriptive threshold, six methods worsen and two are practical ties. However, changing the horizon also stretches warmup from 2.5 to 10 epochs and stretches cosine decay; rates were inherited from 20-epoch 1024-image tuning. Validation is checked every 16 updates at 1024 images versus every 64 at 4096. This compares complete training settings, not a pure intervention on the number of updates. [Curves on the optimizer-update axis](final-small-subsets.pdf).

## Learning-rate schedule

All ten independently validation-tuned constant-versus-cosine comparisons are complete at 4096 images, 50 epochs and seed 42. Under the descriptive 0.1-year threshold, constant LR is better on selected-checkpoint test MAE for **CE**, worse for **TailRL, GRPO MAE, GRPO MSE and scalar MAE**, and a practical tie for the other five methods. This threshold is not a statistical equivalence test.

Final-epoch validation MAE is higher under constant LR for nine of ten methods; categorical MSE has a small improvement. CE illustrates the checkpoint dependence: selected-checkpoint test MAE improves by 0.163 years, while final validation MAE worsens by 0.461. No final-epoch test model was retained. Removing the schedule also removes warmup, so this does not isolate decay alone. No test outcome was used to retune learning rates. [Endpoints](schedule.pdf), [all schedule differences and selected epochs](FINAL_TABLES.md#paired-schedule).

## Training-label noise and SNR

The full **80-run** grid is complete: ten methods × fixed/fresh noise × requested SD 1/3/5/10 years. Every run uses the fixed 4096 subset for 50 epochs, optimization seed 42 and noise seed 2026. Rates are reused from each method's clean subset tuning. Labels are rounded and clipped to 0–99, including for scalar heads; validation and test labels remain clean. Fresh noise changes each epoch. Fixed/fresh share their first realization and methods share their corruption fields.

| Method | Clean | Fixed SD 5 | Fresh SD 5 | Fixed SD 10 | Fresh SD 10 |
|---|---:|---:|---:|---:|---:|
| Population TailRL | 5.406 | 5.967 | **5.215** | 6.717 | **5.283** |
| Direct MSE, categorical mean | 5.216 | 5.588 | 5.231 | **6.252** | 5.719 |
| Direct MAE, categorical mean | 5.576 | 5.896 | 5.609 | 9.425 | 8.251 |
| CE | 6.097 | 6.312 | 5.439 | 7.363 | 5.767 |
| Population Stat MAE, width 2 | 5.835 | 5.934 | 5.409 | 7.201 | 5.440 |
| Population Stat MSE, width 2 | 5.769 | 6.262 | 5.280 | 7.357 | 5.455 |
| Population GRPO MAE | 7.038 | 7.025 | 7.051 | 7.697 | 7.046 |
| Population GRPO MSE | 6.922 | 7.070 | 6.932 | 7.547 | 7.054 |
| Direct MSE, scalar | 5.314 | 5.741 | 5.489 | 6.874 | 5.542 |
| Direct MAE, scalar | 5.145 | 5.764 | 5.262 | 6.734 | 5.389 |

Fresh noise improves on fixed noise for **nine of ten methods at SD 5** and **all ten at SD 10**. Fresh SD-5 improves TailRL, CE and both statistical objectives against their own clean seed-42 baselines. Direct categorical MSE is nearly unchanged, while scalar MAE/MSE worsen. The earlier six-method observation therefore extends broadly, but it is not universal across objectives or comparisons to clean training.

Measured SNR is approximately **25.6, 16.5, 12.1 and 6.2 dB** at SD 1/3/5/10. Fixed/fresh perturbation magnitudes are nearly identical, so SNR alone does not describe the effect of reusing versus resampling labels. Clipping also changes the conditional noise distribution near age boundaries. Fresh-noise improvements are consistent with regularization, especially for overfitting CE, but neither a mechanism nor general robustness is established by one optimization seed and one noise field. The independent-field runs were deliberately excluded, and optimization-seed repetitions remain held. [SD-5 curves](noise.pdf), [observed SNR and matched-clean differences](snr.pdf), [all SDs, SNR values and best epochs](FINAL_TABLES.md).

## Conditional 10×10 categorical policies

All sixteen digit-policy pilots completed: eight objectives on 4096 images and eight on full data, each at seed 42 for 50 epochs. The model represents `p(tens | image) p(units | image, tens)` and enumerates the same 100 joint ages as the flat head.

On 4096 images all eight digit pilots have lower observed test MAE than their matched flat-head baselines. Improvements range from 0.016 to 0.556 years; the largest is direct MAE, **5.576 → 5.020**. Its rate also changes from 0.003 to 0.0003, so this cannot be attributed to parameterization alone. TailRL and direct MSE have small differences, −0.030 and −0.016 years. Label NLL improves for six of eight objectives, with exceptions for Stat MSE and GRPO MSE.

Full-data outcomes are mixed. CE improves **4.926 → 4.784**, Stat MSE improves **4.861 → 4.638**, and GRPO MAE improves **6.641 → 6.010**. Stat MAE worsens **4.619 → 4.776**; TailRL changes **4.719 → 4.808**. Direct MAE/MSE have small changes, −0.044/−0.068 years. Full-data NLL improves for four of eight methods.

These are single-seed pilots with different heads and tuning protocols: 73,364 head parameters for the conditional policy versus 51,300 for the flat head; three LR candidates for digit policies versus four for subset categorical tuning. Full-data digit runs reuse rates selected on the 4096 subset, while full-data categorical baselines use their full-data tuning. The comparisons identify promising complete settings, not an isolated architecture effect. [Both data sizes](final-digits.pdf), [subset curves](digits.pdf), [all matched endpoints and best epochs](FINAL_TABLES.md#paired-digits).

## Operational results and validation

There were **28 recorded contention preemptions**, charged **1.854 GPU-hours**, and no permanently failed jobs. Following the policy update, 217 attempts produced 212 completed jobs and five preemptions. The dispatcher recorded **149 immediate reuses**, with a **1.010-second median recorded completion-to-launch gap**, and 68 new claims, with a minimum observed idle-confirmation span of **10.814 seconds**. Historical routine reuse gaps were approximately 41 seconds. Actual combined peak concurrency was **six**, below the authorized ceiling of eight; availability limited utilization.

The completion audit checks all 490 original histories and all 24 continuation histories against raw metrics, complete epoch counts, common split/subset fingerprints, validation-only LR/checkpoint selection, finite training values and clean held-out counts. It verifies the combined GPU lease history, all 325 attempts present at the policy handoff, shared noise realizations and unchanged frozen training sources. It preserves the old closed extension. Report interactions are tested by an isolated QuickJS DOM harness; browser rendering remains unverified because the in-app browser blocks file URLs.

The experiment uses exact RGB deduplication before splitting, not person-disjoint or near-duplicate-disjoint evaluation. All replications share the same cleaned split. Method rankings apply to these independently tuned settings, and many follow-ups have one optimization seed. Final-epoch metrics are validation metrics; clean test evaluates the validation-MAE-selected checkpoint. All retained and excluded work is listed explicitly above.
