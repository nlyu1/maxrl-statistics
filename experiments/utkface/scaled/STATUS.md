# UTKFace study: completed

All selected work finished **October 15, 2026 at 12:15:54 GMT+8**. The latest scientific snapshot was retrieved **October 16 at 14:47 GMT+8**. Both supervisors exited normally. There are no active attempts or selected pending jobs.

**[Open the interactive report](dashboard.html)** · [Final interpretation](FINAL.md) · [All metrics and best epochs](FINAL_TABLES.md) · [Completion audit](final-audit.json).

The study completed **514 jobs**: 288 LR pilots, four real-image preflight checks and **222 scientific final evaluations**. All **212 jobs selected in the October 15 update** completed. The original and continuation queues contribute 490 and 24 completed jobs, respectively.

| Scientific work | Completed |
|---|---:|
| Full-data categorical matrix, 22 methods × three seeds | 66/66 |
| Full-data scalar heads, two objectives × three seeds | 6/6 |
| Clean 4096-image comparisons, including 16 reinstated replications | 26/26 |
| Clean 1024-image × 50-epoch comparisons | 10/10 |
| Reinstated 1024-image × 200-epoch comparisons | 8/8 |
| Constant-LR comparisons | 10/10 |
| Fixed/fresh SD-1/3/5/10 noise comparisons | 80/80 |
| Conditional 10×10 policy comparisons, subset and full data | 16/16 |

Forty optional noise training-seed repetitions remain held and unrun. Thirty-two independent SD-5 noise-field runs were deliberately excluded. The old deadline extension remains an unchanged historical record; its selected 24 jobs completed in the new continuation. Held and excluded work supplies no additional replication evidence.

Usage totals **65.265 GPU-hours across 542 attempts**, including **28 recorded contention preemptions** costing 1.854 hours. No jobs are permanently failed. The authorized combined concurrency ceiling was eight; actual peak concurrency was six. All prior charges remain intact. The updated dispatcher recorded 149 immediate reuses with a **1.010-second median recorded turnaround**, and 68 new claims with at least **10.814 seconds** of idle confirmation.

The new scientific result is the three-seed scalar comparison: scalar MAE reaches **4.632 ± 0.023 years**, the lowest observed three-seed full-data mean. At 4096 images, categorical MSE and population TailRL reach **5.297 ± 0.174** and **5.359 ± 0.097**, while categorical MAE varies substantially across seeds. The 1024-image 200-epoch runs do not close the gap to 4096 images. Fresh noise usually outperforms fixed noise, but all noise comparisons still have only one optimization seed and one corruption field. Digit-policy pilots have mixed full-data outcomes.

All 514 completed histories, frozen training sources, split/subset fingerprints, held-out counts and validation-only LR/checkpoint choices have been audited. All 325 attempts present at the policy handoff were preserved. The continuation plan and shared coordination match their authorized Git artifacts. The refreshed dashboard passed thirteen isolated QuickJS interaction checks. Browser rendering remains unverified because the in-app browser blocks file URLs.

Best/final epoch metrics refer to validation; clean test evaluates the validation-MAE-selected checkpoint. Scalar label NLL is undefined. Scalar Spearman uses continuous predictions; categorical Spearman uses policy medians. Final-epoch test models were not retained. Sample SD across optimization seeds on one fixed split is not a confidence interval.

The follow-up automation was deleted after completion. All nine refreshed scientific figures were visually inspected; the downloaded 40-file artifact archive passed SHA256 and member-list verification.

[Final audit](final-audit.json) · [Plan verification](final-plan-verification.json) · [All grouped metrics](final-summaries.csv) · [Exact paired comparisons](final-comparisons.json) · [Original raw results](results.json) · [Continuation raw results](continuation-results.json) · [Interactive checks](interaction_validation.json).
