# UTKFace shared learning rates, October 22

This report incorporates all 59 pre-specified reruns and reuses 163 unchanged evaluations. The reruns finished at 04:59:47 GMT+8 on October 22 and consumed 9.251 GPU-hours, including unsuccessful earlier attempts. Rendering these artifacts adds no training runs.

- [Previous/current values, test rankings and learning rates](LR_COMPARISON.md)
- [Five metrics and paired differences](lr-comparison.csv)
- [Per-seed differences and source IDs](lr-comparison.json)
- [Current 222-evaluation snapshot and raw validation curves](report-data.json)
- [Executed rerun results](rerun-results.json)
- [Replacement and history audit](refresh-audit.json)
- [Training-source compatibility audit](cohort-audit.json)
- [Frozen rerun plan](plan.json) and [replacement mapping](migration.json)
- [Updated seven-figure viewer](../bundle-20261018/viewer.html)

The original [dashboard snapshot](../dashboard-data.json), [raw results](../results.json) and [continuation results](../continuation-results.json) retain the previous rate regime. Its tuning records and scientific outcomes are historical evidence, not additional candidates for choosing the current results. The current curves replace the designated runs regardless of whether their new test score improves.

Checkpoint selection always minimizes clean validation mean-age MAE. Mode MAE uses the same checkpoint and the global mode over all 100 ages, including the joint digit policy. Scalar models have no label NLL. Bands and plus/minus values are sample SD over optimization seeds, not confidence intervals. Noise and 1024-image comparisons have one optimization seed. The same split and test set are reused, so these are descriptive follow-up comparisons rather than a fresh held-out confirmation.

Both trainer source cohorts are preserved. The only audited training-source difference adds epoch-zero evaluation/logging, restores random state after that evaluation, and changes timing. The optimizer, training loop, data path and objectives are unchanged. Every executed configuration differs from its predecessor only in learning rate, output path and device assignment. Recorded or reconstructed initialization provenance is retained; epoch zero remains outside plotted moving means.

The full-data test ordering is broadly stable: scalar MAE remains first at 4.632 +/- 0.023 years, and GRPO/RLOO remain worse. Categorical MSE becomes the best mean-age predictor at 1024 images x 200 epochs (6.371 -> 6.100 years). At 4096 images, categorical MAE improves from 6.523 +/- 1.355 to 5.505 +/- 0.277 years. With fixed noise SD 10, categorical MAE improves from 9.425 to 6.396, while population TailRL worsens from 6.717 to 7.155.

Point accuracy and distribution quality differ. Full-data population TailRL's MAE is nearly unchanged, while its NLL worsens from 3.214 to 3.832 nats. Population GRPO-MAE's NLL improves from 60.694 to 33.385 but remains much worse than CE's 3.053. These changes do not establish statistical significance or general noise robustness.

## Rebuild on the server

All Python execution uses uv on cmu-L40-live after the usual local commit, GitHub push and server pull. Datasets, checkpoints and retained generated outputs reside under the persistent state root.

    UTKFACE_STATE="$HOME/.local/state/maxrl-statistics"
    REFRESH="$UTKFACE_STATE/reports/utkface-shared-lr-20261022"
    mkdir -p "$REFRESH"
    git show 23ce46a:experiments/utkface/scaled/bundle-20261018/bundle-data.json > "$REFRESH/historical-bundle.json"
    uv run --no-sync python -m src.experiments.utkface.lr_refresh \
      --original experiments/utkface/scaled/dashboard-data.json \
      --reruns "$UTKFACE_STATE/utkface-shared-lr-20261022" \
      --historical-bundle "$REFRESH/historical-bundle.json" \
      --output-dir "$REFRESH/data" \
      --bundle-dir "$REFRESH/bundle" \
      --mode-pdf "$REFRESH/pdf/utkface-mode-mae-six-figures.pdf" \
      --writeup-data "$REFRESH/writeup/curves.js"

The viewer retains its seven original comparisons; the mode-MAE PDF retains the first six, including statistical widths. The writeup retains its six experiments, including digit policies instead of the separate width figure. The displayed noise plots retain their original five methods for comparison; the tables above include all ten evaluated noise arms.
