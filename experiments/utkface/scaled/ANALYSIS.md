# UTKFace: interpretation of the available evidence

Snapshot: **2026-10-14 21:24 GMT+8**. [Open the interactive report](dashboard.html), [new follow-up results and coverage](FOLLOWUP.md), [full 22-method endpoint table](MAIN_MATRIX.md), [full-data scalar comparison](SCALAR.md), [complete metrics](dashboard-runs.csv), [epoch curves](dashboard-curves.csv), [main scientific figure](main.pdf), and [experiment protocol](PLAN.md).

The categorical main comparison remains **complete: 66/66 runs, 22 configurations, three training seeds each, and 50 full-data epochs per run**. Full-data scalar MAE/MSE confirmations still have one seed each. Since the post-cutoff report, **86 more jobs have completed: 57 LR pilots and 29 final evaluations**. New final results comprise six 4096-image baselines, six constant-LR runs, 14 noise runs and three digit-policy pilots. These follow-ups use one optimization seed, 42; they do not add three-seed evidence. The original supervisor is healthy and currently waiting for free GPUs. The main-method numbers and full-data scalar results below are unchanged.

**All 56 extension jobs remain deadline-skipped without an attempt:** 16 clean-subset replications, eight matched-update runs and 32 independent-noise-field runs. Their allocation used **0/24 GPU-hours** and remains closed. The new results come from the original queue, which has **140 required jobs pending**, including tuning. The original allocation has used **46.892/64 GPU-hours**. The [cutoff audit](cutoff-audit.json) and [cutoff coverage table](COVERAGE.md) remain historical snapshots; [FOLLOWUP.md](FOLLOWUP.md) gives current coverage.

## Main methods

These are clean **test metrics at the validation-MAE-selected checkpoint**. SD describes variation across optimization seeds on one fixed data split. The dashboard separately provides best-checkpoint and final-epoch validation metrics, all evaluation metrics, and per-metric validation extrema. Historical runs did not retain final-epoch test models.

| Method | Seeds | Test MAE, years | Median Spearman | Label NLL, nats | Best epochs, seed order |
|---|---:|---:|---:|---:|---|
| Population Stat MaxRL, MAE width 2 | 3 | 4.718 ± 0.086 | 0.9015 | 7.439 | 14, 16, 16 |
| Sampled TailRL | 3 | 4.724 ± 0.019 | 0.9046 | 6.698 | 41, 26, 22 |
| Population TailRL | 3 | 4.752 ± 0.037 | 0.9028 | 3.214 | 13, 18, 7 |
| Direct MAE, categorical mean | 3 | 4.762 ± 0.193 | 0.9052 | 9.451 | 40, 36, 32 |
| Direct MSE, categorical mean | 3 | 4.817 ± 0.191 | 0.9084 | 8.099 | 47, 33, 32 |
| Population binary MaxRL / CE | 3 | 4.933 ± 0.030 | 0.8974 | 3.195 | 24, 23, 18 |

Population Stat MaxRL with MAE width 2 has the lowest three-seed mean MAE, but its advantage over direct MAE is small and inconsistent: paired differences are **−0.056, −0.220, +0.143 years**, averaging **−0.044 ± 0.182**. This does not establish a reliable improvement over direct MAE. Its width-2 result improves on population MAE width 5 on all three seeds, by **0.153 ± 0.044 years**, and on population MSE width 2 by **0.174 ± 0.060 years**. These compare independently tuned methods; they do not isolate a kernel change at identical learning rates.

Equal numeric kernel widths also do not imply equal spread: the Laplace MAE kernel with scale 2 has variance 8 years², while the Gaussian MSE kernel with standard deviation 2 has variance 4 years². Their comparison changes both shape and effective spread.

Direct MAE improves on direct MSE on all three matched seeds, by **0.055 ± 0.021 years**. Direct MSE has higher median Spearman, illustrating that point error and rank correlation can prefer different objectives.

**Both TailRL estimators now support complete three-seed comparisons.** Population TailRL's paired MAE differences against direct MAE are **+0.044, −0.191, +0.117 years**, averaging **−0.010 ± 0.161**. Its observed mean point error is similar to direct MAE, while its label NLL is much lower: **3.214 versus 9.451 nats**. Against CE it improves MAE on all three seeds by **0.181 ± 0.057 years**, with similar mean NLL, **3.214 versus 3.195**. Against population Stat-MAE width 2 its MAE difference is **+0.034 ± 0.063 years**, with lower NLL, **3.214 versus 7.439**. These are descriptive tradeoffs at independently validation-selected rates and checkpoints, not tests of equivalence or a clear point-error winner.

Sampled TailRL reaches **4.724 ± 0.019 years MAE**. Its paired gap against population Stat-MAE width 2 is only **+0.006 ± 0.105 years**; against direct MAE it is **−0.038 ± 0.203**. Those comparisons do not establish a stable point-error advantage. Population TailRL is **+0.028 ± 0.054 years** worse than sampled TailRL on MAE, but has **3.484 ± 0.736 nats lower NLL**, with the NLL improvement present on all three seeds. Their tuned rates differ, 1e-4 for population and 1e-3 for sampled, so this does not isolate estimator variance. [Additional paired comparisons](additional_paired_differences.json).

Across the ten population-versus-sampled objective pairs, population improves mean MAE in five and worsens it in five. The direction depends on the objective; exact population evaluation is not a uniform generalization improvement here. The [full paired table](MAIN_MATRIX.md) and [estimator comparison data](estimator_comparisons.json) include each seed's differences, their mean/SD and median-Spearman differences.

Population binary MaxRL equals CE in this implementation. Its point error is close to the sampled zero-one estimator: paired population-minus-sampled difference **−0.028 ± 0.069 years**. Its label NLL is lower, 3.195 versus 4.387 nats. Direct losses on the categorical mean do not uniquely constrain the distribution, so good mean-age predictions need not imply good label probabilities.

The eight GRPO/RLOO configurations complete all three seeds with MAE **6.342–6.808 years**, worse than the completed direct, binary and statistical baselines here. Their test entropies are only **0.038–0.085 nats**, with label NLL **17.93–60.69 nats**: evidence of highly concentrated policies assigning little probability to many true ages. This describes these objectives, estimators, tuning ranges and schedules; it is not a general impossibility result for GRPO/RLOO. The extra metric selector exposes expected policy error, median/mode error and oracle best-of-K error to inspect objective/decision-rule differences. [Original paired comparisons](paired_differences.json).

## Checkpoint dependence

Population Stat-MAE width 2 reaches its selected checkpoint at epochs 14/16/16, while the direct baselines peak much later. Its mean validation MAE moves from **4.852 at selection to 5.125 at epoch 50**; label NLL rises from **7.255 to 11.082** and entropy falls from **1.387 to 0.615**. CE similarly moves from validation MAE **4.926 to 5.314**, with NLL **3.226 to 6.600**. Direct MAE's validation MAE changes less, **4.860 to 4.921**. Thus a best-checkpoint ranking does not describe the final-epoch model. Curves and separate final-validation columns make this visible. Test evaluation remains at the selected checkpoint for every method.

Population TailRL selects epochs **13/18/7**. Its mean validation MAE changes from **4.823 at selection to 5.031 at epoch 50**, and NLL from **3.218 to 4.386**. Its good selected-checkpoint performance therefore also depends on checkpoint selection.

Sampled TailRL selects later, at **41/26/22**, and ends at mean validation MAE **4.917**, compared with **4.921** for direct MAE and **5.125** for population Stat-MAE width 2. Thus the final-validation ordering differs from the selected-checkpoint test ordering. These are different evaluation splits and checkpoints; final test comparisons remain unavailable.

## Smaller training sets and the extension

Six of ten qualified 4096-image configurations are complete: categorical population TailRL, MSE, MAE and CE, plus scalar MSE/MAE. Their selected-checkpoint test MAEs are **5.406, 5.216, 5.576, 6.097, 5.314 and 5.145 years**, respectively. All use optimization seed 42 and the same subset fingerprint. Population Statistical MaxRL MAE/MSE width 2 and GRPO MAE/MSE remain pending, as do all 1024-image confirmations. The completed subset ranking therefore does not cover the reduced categorical matrix. [Best/final metrics and epochs](FOLLOWUP.md), [curves](subset.pdf).

Both smaller subsets are fixed, nested and stratified by age decade. At batch 64, **full-data×50 gives 14,400 updates**, **4096×50 gives 3,200**, and **1024×50 gives 800**. Dataset size and optimization exposure change together. The extension's planned **1024×200** arm would match updates and presentations to 4096×50, while also changing validation-check frequency and inheriting the 20-epoch subset LR selection. None of those eight runs launched, so the matched-update question remains unanswered.

Two extra optimization seeds on the clean 4096 subset were planned to test whether single-seed method rankings persist. All 16 were deadline-skipped. The [extension](EXTENSION.md) ended at **07:30 GMT+8**, with its 24 GPU-hour cap untouched. Subsequent original-queue progress uses its existing 64 GPU-hour allocation and the unchanged shared six-GPU cap.

## Learning-rate schedule

Six matched-seed schedule comparisons are complete, with separate validation-only LR sweeps. Using a 0.1-year test-MAE threshold, constant LR is better for **CE (−0.163 years)**, worse for **population TailRL (+0.167)** and **scalar MAE (+0.176)**, and a practical tie for categorical MSE/MAE and scalar MSE. These are single-seed descriptions, not statistical equivalence tests. Constant LR removes **both warmup and decay**, so the comparison does not isolate cosine decay alone. [Exact outcomes](FOLLOWUP.md), [endpoint figure](schedule.pdf).

CE's apparent selected-test improvement does not extend to validation: its best validation MAE is worse under constant LR and its final validation MAE worsens by **0.461 years**. Categorical MAE likewise has a selected-test practical tie but **0.344 years** worse final validation. A final-epoch claim therefore differs from the selected-checkpoint test result; final-epoch test models were not retained. No test result was used to retune the rates.

## Scalar heads

Both scalar candidates passed the predefined validation-only qualification gate: scalar/categorical validation-MAE ratios **0.9948 for MAE** and **0.9781 for MSE**. Those are 30-epoch tuning outcomes. Both **seed-42** confirmations now finish 50 epochs:

| Objective | Head | Best epoch | Best validation MAE | Final validation MAE | Selected-checkpoint test MAE |
|---|---|---:|---:|---:|---:|
| MAE | Categorical mean | 40 | 4.811 | 4.865 | 4.675 |
| MAE | Scalar | 41 | 4.732 | 4.748 | 4.611 |
| MSE | Categorical mean | 47 | 4.821 | 4.893 | 4.709 |
| MSE | Scalar | 39 | 4.751 | 4.792 | 4.680 |

All four runs use LR **0.001**, selected independently using validation. Scalar-minus-categorical test MAE is **−0.064 years (−1.36%) for MAE** and **−0.029 years (−0.61%) for MSE**. Scalar MSE also lowers test RMSE slightly, **6.722 versus 6.747 years**. Scalar MAE is close to population Stat-MAE width 2 on this same seed, **4.611 versus 4.619 years**. This is a single matched optimization seed per objective and does not establish a replicated head advantage or a new three-seed winner. **2/6 scalar confirmations** are complete; seeds 43/44 remain pending. [Exact comparison](scalar-comparison.json), [curves and endpoints](scalar.pdf).

Within scalar seed 42, the MAE-trained head has lower test MAE, while the MSE-trained head has lower test RMSE, **6.722 versus 6.802 years**. This illustrates an objective-dependent tradeoff at validation-MAE-selected checkpoints; neither head was selected using test RMSE.

The 4096-image comparison is less uniform: scalar MAE reaches **5.145 versus 5.576 years** for the categorical-mean MAE head, a **0.431-year** improvement, while scalar MSE reaches **5.314 versus 5.216**, a **0.098-year** increase. Scalar/categorical MSE RMSE is similar, **7.470 versus 7.483**. Rates differ between the independently tuned heads, and only seed 42 is complete, so the result does not establish a general scalar-head advantage. [Subset comparison](subset.pdf).

The common-seed control preserves actual completion counts while restricting displayed summaries and completed curves to matching optimization seeds. The scalar section therefore compares seed 42 instead of contrasting one scalar seed with three-seed baseline means. Scalar heads have no categorical label NLL; their Spearman ranks continuous predictions, whereas the categorical metric ranks policy medians. Scalar seed-42 test Spearman is **0.9111 for MAE** and **0.9124 for MSE**, but these head comparisons also change the prediction used for ranking. The reproducible main-figure command now restricts that figure to categorical main runs and prints actual seed counts; scalar comparisons have their own matched-seed figure.

## Label noise and SNR

Fourteen trained noise runs are complete on 4096 images: fixed/fresh SD 5 for all six completed baselines, plus fixed SD 1 for CE and scalar MAE. Optimization seed is 42 and noise field is 2026; labels are rounded/clipped to 0–99 and held-out labels stay clean. Measured SD-5 SNR is **12.134 dB fixed / 12.120 dB fresh**. Fresh noise has lower test MAE than fixed noise for all six methods. Against clean training, fresh noise improves TailRL by **0.191 years** and CE by **0.658**, approximately preserves categorical MSE/MAE, and worsens scalar MSE/MAE by **0.175/0.117 years**. A regularization explanation is plausible, but one training seed and one noise field do not establish the mechanism or general robustness. [All metrics and best epochs](FOLLOWUP.md), [curves](noise.pdf), [observed SNR/error changes](snr.pdf).

The fixed-SD-1 results have measured SNR **25.664 dB**: CE reaches **5.887 years** and scalar MAE **5.198**, compared with clean baselines 6.097 and 5.145. SD-3 and fresh-SD-1 results remain missing. The SD-3 SNR near **16.5 dB** is still a design statistic, not a trained performance observation. These incomplete settings do not establish a noise dose-response curve. [Noise design](noise_design.json).

The extension's independent fields 2027/2028 were deadline-skipped, so sensitivity to which labels are corrupted remains unmeasured. Original optional optimization-seed repeats also remain pending. Those two sources of variability stay separate in the dashboard. Fixed corruption can be memorized, while fresh corruption changes each epoch; rounding/clipping introduces boundary effects. SNR measures clean-age variance relative to realized perturbation MSE, not image informativeness or intrinsic annotation noise. SD-10 and training-seed repeats are optional. Fresh-noise categorical MSE selects epoch 50, so the training horizon may truncate further improvement.

## Two-digit policy

All three prioritized pilots are complete on 4096 images, seed 42. Digit versus flat test MAE is **5.376 versus 5.406 for population TailRL**, **5.200 versus 5.216 for MSE**, and **5.907 versus 6.097 for CE**. TailRL and MSE have small point-error differences; CE improves by **0.190 years**. Label NLL decreases for all three. These observations establish neither equivalence nor a replicated advantage. [Metrics and best epochs](FOLLOWUP.md), [curves](digits.pdf).

The conditional head computes `p(tens|image) p(units|image,tens)` over the same 100 ages. Its head has **73,364 parameters**, versus **51,300** for the flat head, and its three-rate exploratory tuning differs from the four-rate baseline sweep. Representation, capacity and tuning differ together. Digit MSE selects epoch 50, leaving a horizon limitation. Other digit configurations remain optional and unfinished.

## Reliability and availability

The real-image bf16 preflights passed for categorical statistical losses, scalar MAE and digit CE. Subsequent audits verify frozen training-source hashes, immutable plans, complete epoch counts, matching training fingerprints within each subset, validation-only LR selection and clean 2,308-image test labels. Exact RGB duplicates were removed before splitting; person-level and near-duplicate independence remain unverified.

The fresh audit validates all **302 completed histories**, raw versus reported phase metrics, epoch counts, finite training, validation-only LR selection and common subset fingerprints. Shared noise-field measurements agree across methods. All **224 attempts recorded before supervisor handoff** remain intact; **325 original attempts** now account for **46.892/64 GPU-hours**. Historical peak GPU leases are **six**, matching the cap. Both immutable plans and all 11 frozen training sources remain intact. The extension still has no attempts or charges and keeps its explicit deadline-skipped reasons. No permanently failed job is present. [Current integrity and tuning decisions](progress_integrity.json), [coordination audit](coordination_integrity.json), [historical cutoff audit](cutoff-audit.json), [main integrity](main_integrity.json), [pilot integrity](pilot_integrity.json).

Report metric/selection tests pass, as do 18 scheduler checks. Interaction logic passed an isolated JavaScript DOM harness covering all six sections, sorting, hiding, filters, exports and finite chart geometry. The static scientific figure was visually inspected. Browser rendering of the HTML remains unverified because the in-app browser blocks file URLs. [Interaction validation](interaction_validation.json).

The first report was generated at **04:01 GMT+8**, and the post-cutoff report was retrieved at **07:35**, both before the 08:30 deadline. The CPU watcher ended normally at **08:25**. This **21:24** refresh includes substantial new original-queue results and uses a one-off report generation. Monitoring continues while the original allocation remains active. Snapshot timestamps and actual seed counts identify the evidence at each refresh.
