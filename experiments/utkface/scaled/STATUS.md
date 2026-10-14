# UTKFace visual report and study status

**The interactive report is available:** [dashboard.html](dashboard.html). It was first generated at **04:01 GMT+8 on October 14**, before the requested 08:30 deadline. [Read the interpretation](ANALYSIS.md).

The latest snapshot is **21:24 GMT+8 on October 14**. **302 jobs are complete**, including LR tuning; **290/430 required jobs** are complete and 140 remain pending. Since the post-cutoff publication, 86 more jobs finished: 57 LR pilots and 29 final evaluations. [New follow-up results](FOLLOWUP.md), [complete main table](MAIN_MATRIX.md).

| Experiment | Completed / planned |
|---|---:|
| Main categorical matrix, 22 methods × three seeds | **66/66** |
| Full-data scalar confirmations | 2/6 |
| Fixed 4096-image subset | 6/10 |
| Fixed 1024-image subset | 0/10 |
| Constant-LR confirmations | 6/10 |
| Required SD-1/3/5 noise confirmations | 14/60 |
| Optional digit-policy confirmations | 3/16 |

The reduced sets include eight categorical methods and two qualified scalar heads. Completed follow-up comparisons use one optimization seed, 42. All three priority digit pilots are complete. The missing 4096-image categorical objectives are population Stat-MAE/MSE width 2 and population GRPO MAE/MSE. The original 96 wide LR pilots and 48 longer full-data confirmations remain complete; new subset/constant/digit tuning has completed 24/24/9 additional pilots.

The current audit records **46.892/64 GPU-hours across 325 original attempts**, including preemptions. The original supervisor is healthy and currently waiting for idle GPUs; no permanently failed job is present. Historical peak study GPU leases are **six**, matching the enforced cap. The extension remains closed at **0/24 GPU-hours**, with all 56 jobs explicitly deadline-skipped and no attempts launched. Its 07:30 cutoff accounting is a historical record, not current original-queue coverage. [Current integrity](progress_integrity.json), [coordination audit](coordination_integrity.json), [historical cutoff](COVERAGE.md).

Both immutable plans, all 224 attempts present before handoff, and all 11 frozen training sources remain intact. The fresh audit checks all 302 completed histories, epoch counts, raw versus reported metrics, common split/subset fingerprints, shared noise fields and validation-only LR selection. The CPU watcher ended normally at 08:25; this refresh uses one-off report generation. The closed extension will not restart or spend its expired allocation. Monitoring continues for the original queue within its existing cap.

The dashboard offers sorting, method hiding, common-seed comparisons with actual completion counts retained, best epochs, best/final validation metrics, selected-checkpoint test metrics, validation curves, subset/update axes, paired schedule outcomes, scalar comparisons, noise/SNR controls and digit-policy comparisons. Additional evaluation metrics are selectable; complete run and epoch CSVs are downloadable. Best checkpoint always means minimum validation mean-age MAE. Final-epoch test models were not retained.

The completed categorical matrix is unchanged: its lowest test-MAE mean is population Statistical MaxRL with MAE width 2, **4.718 ± 0.086 years**. Sampled TailRL reaches **4.724 ± 0.019**, with a paired difference of only **+0.006 ± 0.105 years**; no clear point-error winner is established. Population TailRL reaches **4.752 ± 0.037**, with substantially lower label NLL than sampled TailRL, **3.214 versus 6.698**. Population does not improve mean MAE uniformly across objectives: five of ten estimator pairs improve and five worsen. [Scientific figure](main.pdf), [full analysis](ANALYSIS.md).

New 4096-image test MAEs include **5.145 for scalar MAE**, **5.216 for categorical-mean MSE**, and **5.406 for population TailRL**. At SD 5, fresh noise has lower test MAE than fixed noise for all six completed methods; independent noise fields and most other noise levels remain missing. Constant LR has mixed results. Digit policies have small MAE differences for TailRL/MSE and a 0.190-year CE improvement, with lower NLL for all three pilots. These are single-seed observations. [Tables, limitations and best epochs](FOLLOWUP.md), [subset](subset.pdf), [schedule](schedule.pdf), [noise](noise.pdf), [SNR](snr.pdf), [digits](digits.pdf).

The dashboard and interaction validation capture **21:24:21 GMT+8**. Best/final validation metrics remain distinct from selected-checkpoint test metrics; scalar NLL is undefined. Final-epoch test models were not retained. Full-data scalar results remain 4.611/4.680 years MAE for MAE/MSE at seed 42; their other two seeds remain pending.

[Current integrity and tuning decisions](progress_integrity.json), [data audit](data_audit.json), [interaction validation](interaction_validation.json), [follow-up metrics](followup-metrics.csv), [all metrics](dashboard-runs.csv), and [all curves](dashboard-curves.csv) support this snapshot. Earlier main/preflight audits remain historical evidence for unchanged runs. Report interactions passed an isolated JavaScript harness; browser rendering is unverified because the in-app browser blocks local file URLs.
