# Scaled UTKFace: interim test results

Snapshot: 2026-10-14 00:44 Asia/Shanghai. **25/66 categorical main runs** have completed all fifty full-data epochs; eight configurations have all three seeds and population RLOO-MAE has one seed. All **96 initial learning-rate pilots** and **48 longer confirmations** are complete. Five main runs are active. Charged allocation at the snapshot is **27.617/64 GPU-hours**. There are no permanently failed jobs.

These are partial results from the requested main comparison. Test outcomes have not been used to change learning rates, scalar gates or the method matrix. All rates were selected using thirty-epoch validation confirmations on seed 42; main runs use seeds 42, 43 and 44. Every run trains all fifty epochs and evaluates its validation-selected checkpoint on the clean 2,308-image test split. Each full-data epoch presents 18,432 images from the common 18,462-image training pool.

## Main test comparison

Values are means ± sample SD across the available training seeds. A single-seed value has no SD. TailRL, Statistical MaxRL and most RLOO main results are still pending; this table does not establish the full study's winner.

![Partial main comparison](main.png)

[PDF figure](main.pdf).

| Method | Seeds | Test mean MAE, years | Test median Spearman | Test label NLL, nats |
|---|---:|---:|---:|---:|
| Direct MAE, categorical mean | 3 | 4.762 ± 0.193 | 0.9052 ± 0.0047 | 9.451 ± 1.390 |
| Direct MSE, categorical mean | 3 | 4.817 ± 0.191 | 0.9084 ± 0.0019 | 8.099 ± 0.864 |
| Population binary MaxRL (CE) | 3 | 4.933 ± 0.030 | 0.8974 ± 0.0016 | 3.195 ± 0.124 |
| 64-sample binary MaxRL | 3 | 4.961 ± 0.065 | 0.8907 ± 0.0047 | 4.387 ± 0.288 |
| 64-sample GRPO, MAE reward | 3 | 6.342 ± 0.109 | 0.8907 ± 0.0066 | 28.755 ± 1.572 |
| Population RLOO, MAE reward | 1 | 6.426 | 0.8867 | 17.807 |
| Population GRPO, MAE reward | 3 | 6.526 ± 0.126 | 0.8881 ± 0.0027 | 60.694 ± 5.382 |
| 64-sample GRPO, MSE reward | 3 | 6.614 ± 0.242 | 0.8915 ± 0.0048 | 29.489 ± 0.792 |
| Population GRPO, MSE reward | 3 | 6.808 ± 0.281 | 0.8867 ± 0.0074 | 35.231 ± 2.314 |

Direct categorical-mean MAE and MSE reach 4.762 and 4.817 years, respectively. MAE is lower on all three matched seeds, by 0.034, 0.055 and 0.076 years. The paired MAE-minus-MSE difference is -0.055 ± 0.021 years across seeds. This is a small, consistent advantage in the present runs; broader conclusions still require the remaining methods and settings.

Population binary MaxRL, which equals CE, and the 64-sample estimator have similar point errors, 4.933 versus 4.961 years. Their paired population-minus-sampled MAE difference is -0.028 ± 0.069 years. CE has lower label NLL, 3.195 versus 4.387 nats, and lower NLL than the direct point-loss baselines. Direct mean MAE/MSE do not uniquely determine the categorical distribution, so their point error and label NLL assess different aspects of behavior.

The four GRPO configurations reach 6.342–6.808 years MAE, worse than these completed baselines. Their mean entropies are only 0.041–0.061 nats, with label NLL 28.75–60.69 nats. This is evidence of very concentrated policies with poor probability on observed labels. It does not isolate why their point error is worse. Population RLOO-MAE is 6.426 years on one seed; its remaining seeds are running.

Population GRPO with the MAE reward is worse than its sampled counterpart on each matched seed; its paired gap is +0.184 ± 0.123 years. With the MSE reward, the population-minus-sampled gap is +0.194 ± 0.401 years and is driven by one seed. These are descriptive comparisons at each method's validation-selected learning rate, not confidence intervals or an isolated estimator intervention. [Paired seed differences](paired_differences.json).

The missing main seeds are recorded explicitly in [main integrity and coverage](main_integrity.json). At this snapshot, eight methods contribute three seeds, one method contributes one seed, and thirteen methods contribute none. The main matrix has 41 unfinished runs. Both qualified scalar heads, smaller fixed subsets, constant-LR studies, Gaussian-noise studies and optional digit policies have no final training comparisons yet.

## Completed learning-rate confirmations

All 24 configurations finished both thirty-epoch confirmations. This table contains validation metrics from the selected rate and checkpoint on training seed 42; these values are not test results.

| Method | LR | Best epoch | Mean MAE, years | Median Spearman | Label NLL, nats | Entropy, nats |
|---|---:|---:|---:|---:|---:|---:|
| 64-sample TailRL | 0.001 | 24 | 4.667 | 0.9065 | 6.038 | 1.993 |
| Direct MAE, scalar head | 0.001 | 30 | 4.690 | 0.9068 | n/a | n/a |
| Direct MAE, categorical mean | 0.001 | 27 | 4.714 | 0.9045 | 7.301 | 1.951 |
| Direct MSE, scalar head | 0.001 | 25 | 4.722 | 0.9091 | n/a | n/a |
| 64-sample Statistical MaxRL, MAE width 5 | 0.001 | 24 | 4.739 | 0.9027 | 13.640 | 0.540 |
| Population TailRL | 0.0001 | 9 | 4.742 | 0.9049 | 3.045 | 2.522 |
| 64-sample Statistical MaxRL, MSE width 5 | 0.0003 | 20 | 4.751 | 0.9062 | 13.044 | 0.489 |
| Population Statistical MaxRL, MAE width 5 | 0.0003 | 20 | 4.752 | 0.8991 | 10.291 | 0.626 |
| Population Statistical MaxRL, MAE width 2 | 0.001 | 15 | 4.768 | 0.8998 | 7.477 | 1.388 |
| 64-sample Statistical MaxRL, MAE width 2 | 0.001 | 25 | 4.808 | 0.8978 | 10.080 | 0.823 |
| Population Statistical MaxRL, MSE width 2 | 0.0003 | 13 | 4.813 | 0.8984 | 6.264 | 1.376 |
| Population Statistical MaxRL, MSE width 5 | 0.001 | 27 | 4.822 | 0.9042 | 12.866 | 0.797 |
| Direct MSE, categorical mean | 0.001 | 27 | 4.827 | 0.9031 | 6.130 | 2.259 |
| 64-sample Statistical MaxRL, MSE width 2 | 0.0003 | 8 | 4.876 | 0.9048 | 6.102 | 1.868 |
| Population binary MaxRL (CE) | 0.003 | 20 | 4.890 | 0.9030 | 3.172 | 2.418 |
| 64-sample binary MaxRL | 0.001 | 21 | 4.951 | 0.8843 | 4.278 | 1.812 |
| 64-sample GRPO, MAE reward | 0.0003 | 25 | 6.410 | 0.8940 | 24.424 | 0.098 |
| Population RLOO, MAE reward | 0.0001 | 26 | 6.494 | 0.8875 | 16.149 | 0.110 |
| 64-sample RLOO, MAE reward | 0.0001 | 28 | 6.562 | 0.8887 | 15.935 | 0.087 |
| Population GRPO, MAE reward | 0.001 | 29 | 6.575 | 0.8857 | 46.938 | 0.056 |
| 64-sample GRPO, MSE reward | 0.0003 | 26 | 6.606 | 0.8865 | 26.443 | 0.056 |
| Population GRPO, MSE reward | 0.0003 | 29 | 6.606 | 0.8910 | 27.935 | 0.052 |
| 64-sample RLOO, MSE reward | 0.001 | 25 | 6.805 | 0.8806 | 32.668 | 0.052 |
| Population RLOO, MSE reward | 0.001 | 29 | 6.896 | 0.8749 | 34.400 | 0.072 |

TailRL has the lowest validation point error among these confirmations: 4.667 years with 64 samples and 4.742 years with the population estimator. Sampled Statistical MaxRL with an MAE-width-5 kernel reaches 4.739 years. Their fifty-epoch multi-seed test runs are still needed before judging the method comparison. Population TailRL has validation label NLL 3.045 nats, versus CE's 3.172 nats at its MAE-selected checkpoint.

Both scalar heads pass the planned 5% eligibility gate. Scalar MAE's validation MAE is 0.9948 times its categorical counterpart; scalar MSE's ratio is 0.9781. Each qualifies for three full-data seeds and inclusion in the later reduced studies. Eligibility does not establish test superiority. Scalar label NLL is unavailable; scalar Spearman ranks the continuous prediction, while categorical Spearman ranks the policy median.

![Learning-rate tuning](pilots.png)

The figure compares the best of four ten-epoch pilots with the best of two thirty-epoch confirmations. Each point uses validation-selected metrics from a separate run; connecting lines are not continued training trajectories. Warmup and decay timing change with each run's horizon. [PDF figure](pilots.pdf). All candidate rates and selected pairs are in [tuning integrity data](pilot_integrity.json).

## Integrity and remaining work

The immutable plan and all eleven frozen training-source hashes match the original execution metadata. All completed runs passed epoch-count, common split/subset fingerprint, finite-loss/gradient and validation-checkpoint checks. Every main run uses its intended validation-selected rate, fifty epochs, the same full training pool, and 2,308 clean test images. Result groups have distinct seeds and a common learning rate and training fingerprint. Artifact-only Git commits differ across runs; their training implementation is identical. [Main integrity](main_integrity.json), [validation record](validation.json), and [execution metadata](execution.json).

All four real-image bf16 preflights passed, covering sampled Statistical MaxRL MAE/MSE, scalar MAE and the conditional digit policy with CE. Each trained on 256 images and evaluated 65 validation and 65 test images. These are pipeline checks, not main comparison results. [Preflight validation](preflight_validation.json).

The split contains 23,078 unique RGB images after dropping conflicting-age duplicate groups and redundant consistent copies. Independent audits found no exact RGB duplicates within or across splits. This does not establish subject-disjoint or near-duplicate-disjoint testing. [Data audit](data_audit.json).

The supervisor remains PID 1425732 on cmu-L40-live. It currently uses five GPUs; observed peak concurrency is five, below the six-GPU cap. Every attempt, including interruptions and retries, remains charged to the same 64 GPU-hour allocation. Waiting for idle GPUs consumes no allocation. The server's /tmp has about 22 GB free. Earlier external-preemption recovery is preserved in the [recovery record](queue_recovery.json).

Noise design is measured, but no label-noise training comparison is complete. For rounded and clipped Gaussian noise with requested standard deviations 1/3/5 years on the fixed 4096-image subset, fixed-noise SNRs are 25.66/16.48/12.13 dB; fresh-noise SNRs averaged across fifty epochs are 25.61/16.48/12.12 dB. Validation and test labels remain clean. [Noise design](noise_design.json).

The queue continues the remaining categorical main runs, six qualified scalar runs, fixed 4096/1024-subset studies, constant-LR sweeps and required noise studies. Optional digit policies and extra noise follow required work and must fit the allocation. The plan, state and previous attempts have not been replaced. The hourly follow-up remains active; the server queue runs independently of the desktop app.

[Generated report](REPORT.md), [aggregate test values](summary.csv), [per-run records](runs.csv), [full configs and metrics](results.json), [epoch curves](curves.csv), [progress snapshot](progress.json), and [protocol](PLAN.md).
