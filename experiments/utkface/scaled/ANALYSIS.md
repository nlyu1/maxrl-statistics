# UTKFace: interpretation of the available evidence

Snapshot: **2026-10-14 07:08 GMT+8**. [Open the interactive report](dashboard.html), [full 22-method endpoint table](MAIN_MATRIX.md), [scalar comparison](SCALAR.md), [complete metrics](dashboard-runs.csv), [epoch curves](dashboard-curves.csv), [scientific figure](main.pdf), and [experiment protocol](PLAN.md).

The categorical main comparison remains **complete: 66/66 runs, 22 configurations, three training seeds each, and 50 full-data epochs per run**. Scalar MAE and MSE now each complete one 50-epoch confirmation, seed 42. The new result is scalar MSE. Smaller-set, constant-LR, noise and digit-policy comparisons have no completed final runs at this snapshot; their report sections show missing coverage instead of inferred results. Both supervisors are healthy and waiting for free GPUs. The categorical main numbers below are unchanged from the completed-matrix publication.

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

Both original smaller subsets are fixed, nested and stratified by age decade. At batch 64, **4096×50 gives 3,200 updates**, while **1024×50 gives 800**. The new **1024×200** arm matches updates and presentations, helping distinguish training-set size from optimization exposure. It also evaluates validation 200 times instead of 50 and reuses the LR selected by the original 20-epoch subset sweep; those differences limit interpretation of best-checkpoint comparisons.

Two extra optimization seeds on the clean 4096 subset test whether single-seed method rankings persist. They inherit the same validation-selected subset LR and noise-free split. The [56-run extension](EXTENSION.md) is a candidate queue, not a promise that all jobs finish: it has a **24 GPU-hour cap**, a **07:30 GMT+8 cutoff**, and shares a **six-GPU cap** with the original queue. At this snapshot it has launched no attempts: GPU scarcity has delayed its original clean-subset tuning dependencies. The temporarily available GPU finished the main matrix and both scalar seed-42 confirmations; all GPUs are occupied by other workloads now. Required original work can continue within its existing 64 GPU-hour allocation.

## Learning-rate schedule

No completed schedule comparison is available yet. The dashboard will compare matched-seed 4096-image results after separate validation-only LR sweeps for each schedule. “Good/bad/practical tie” uses a configurable 0.1-year MAE threshold, not a statistical test. Constant LR removes **both warmup and decay**, so this experiment does not isolate cosine decay alone. The displayed endpoint is the test evaluation available at run completion, at the selected checkpoint.

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

The common-seed control preserves actual completion counts while restricting displayed summaries and completed curves to matching optimization seeds. The scalar section therefore compares seed 42 instead of contrasting one scalar seed with three-seed baseline means. Scalar heads have no categorical label NLL; their Spearman ranks continuous predictions, whereas the categorical metric ranks policy medians. Scalar seed-42 test Spearman is **0.9111 for MAE** and **0.9124 for MSE**, but these head comparisons also change the prediction used for ranking. The reproducible main-figure command now restricts that figure to categorical main runs and prints actual seed counts; scalar comparisons have their own matched-seed figure.

## Label noise and SNR

The original matrix covers fixed/fresh Gaussian perturbations with requested SD 1/3/5 years on 4096 images. Noise seed 2026 is shared across methods; training labels are rounded/clipped to 0–99 and held-out labels stay clean. Precomputed realized SNR is approximately **25.6, 16.5 and 12.1 dB**, respectively. These are perturbation-design statistics, **not trained robustness results**. [Noise design](noise_design.json).

The extension adds noise fields 2027/2028 at SD 5 while holding optimization seed 42 fixed. This probes sensitivity to which labels are corrupted. Original optional training-seed repeats change optimization seed while retaining field 2026; the two sources of variability remain separate in the dashboard. Fixed corruption permits memorization, whereas fresh corruption changes each epoch. Rounding/clipping also introduces boundary effects. SNR measures clean-age variance relative to realized perturbation MSE, not image informativeness or the dataset's intrinsic annotation noise.

## Two-digit policy

Prioritized pilots compare CE, direct MSE and population TailRL on 4096 images. The conditional head computes the complete joint distribution `p(tens|image) p(units|image,tens)` over the same ages 0–99, so it does not reduce the enumerated action space. Its head has **73,364 parameters**, versus **51,300** for the flat 100-bin head. Its parameter sharing and nonlinearity differ, and its three-rate exploratory tuning differs from the baseline four-rate sweep. No completed pilot comparison is available yet.

## Reliability and availability

The real-image bf16 preflights passed for categorical statistical losses, scalar MAE and digit CE. Subsequent audits verify the frozen training-source hashes, original plan hash, complete epoch counts, common training fingerprint, validation-only LR selection and clean 2,308-image test labels. Exact RGB duplicates were removed before splitting; person-level and near-duplicate independence remain unverified.

The coordination audit preserves all **224 attempts recorded before supervisor handoff**, with **230 original attempts** and **41.231/64 GPU-hours** charged at 07:08. The report progress read agrees; waiting for an idle GPU consumes no additional GPU budget. The extension has launched no attempts and has charged **0/24 GPU-hours**. Observed historical concurrency is five GPUs; the enforced combined cap is six. External preemptions remain charged, and no permanently failed job remains after the adoption-classification repair. [Coordination audit](coordination_integrity.json), [main integrity](main_integrity.json), [pilot integrity](pilot_integrity.json).

Report metric/selection tests pass, as do 18 scheduler checks. Interaction logic passed an isolated JavaScript DOM harness covering all six sections, sorting, hiding, filters, exports and finite chart geometry. The static scientific figure was visually inspected. Browser rendering of the HTML remains unverified because the in-app browser blocks file URLs. [Interaction validation](interaction_validation.json).

The first report was generated at **04:01 GMT+8**, before the 08:30 deadline. A server-side CPU watcher refreshes it every five minutes through **08:25**; the existing 20-minute follow-up retrieves and publishes meaningful updates. Snapshot timestamps and actual seed counts identify the available evidence at each refresh.
