# UTKFace completed follow-up comparisons

Scientific snapshot: **October 16, 2026 at 14:47 GMT+8**. All selected follow-up work is complete. [Final interpretation](FINAL.md), [all metrics and selected epochs](FINAL_TABLES.md), [interactive report](dashboard.html), [completion audit](final-audit.json).

| Work | Completion and replication |
|---|---|
| Clean 4096-image subset | Eight categorical methods × three seeds; two scalar heads × one seed |
| Clean 1024-image subset, 50 epochs | Ten methods × seed 42 |
| Clean 1024-image subset, 200 epochs | Eight categorical methods × seed 42 |
| Constant LR | Ten independently validation-tuned methods × seed 42 |
| Gaussian noise | Ten methods × fixed/fresh × SD 1/3/5/10; seed 42, noise field 2026 |
| Conditional digit policies | Eight objectives × 4096/full-data settings × seed 42 |
| Full-data scalar confirmations | MAE and MSE × seeds 42/43/44 |

MSE and population TailRL lead the replicated 4096-image categorical comparisons. Categorical MAE has a large seed-44 degradation, retained in its three-seed mean. The matched-update 1024-image setting does not close the gap to 4096. Constant LR has mixed selected-test outcomes and mostly worse final validation. Fresh noise usually improves on fixed noise, with different effects across objectives. Full-data digit-policy outcomes are mixed.

See [three-seed subset endpoints](final-replication.pdf), [subset curves by updates](final-small-subsets.pdf), [scalar curves](final-scalar.pdf), [schedule endpoints](schedule.pdf), [SD-5 curves](noise.pdf), [SNR](snr.pdf), [subset digit curves](digits.pdf), and [digit endpoints at both data sizes](final-digits.pdf).

Every noise arm uses the same fixed 4096-image training subset and its clean validation-selected LR; labels are rounded/clipped to 0–99 and held-out labels stay clean. Noise repetitions and independent noise fields were not run. The full noise grid therefore supplies variation across magnitudes and temporal sampling, not independent realization uncertainty.

Both queues finished normally. Forty optional noise training-seed repetitions remain held; thirty-two independent SD-5 noise-field jobs are excluded. The historical 56-job deadline extension remains unchanged, and its selected 24 jobs completed in the continuation.

The final audit and figures can be regenerated remotely with `uv run --python 3.12 python -m src.experiments.utkface.final_report --coordination /tmp/utkface-followup-20261015/coordination.json --archived-extension-plan /tmp/utkface-extension-20261014/plan.json --report-dir /tmp/utkface-study-20261013/report`, after refreshing the original study_report and the two-plan visual_report. Best/final metrics are validation metrics; clean test evaluates the MAE-selected checkpoint. Browser rendering remains unverified; interaction logic passed the isolated QuickJS harness.
