# First learning-rate pilot results, 2026-10-13

The four initial learning-rate pilots for 64-sample binary MaxRL completed ten full-data epochs each, with training seed 42. They use the same 18,462-image training pool and 2,308 clean validation images; each epoch presents 18,432 training images. No pilot evaluated test data. Metrics below come from the checkpoint selected by validation mean-prediction MAE.

| LR | Best epoch | Validation mean MAE, years | Median Spearman | Label NLL, nats |
|---:|---:|---:|---:|---:|
| 1e-4 | 8 | 5.170 | 0.8878 | 3.434 |
| 3e-4 | 10 | 4.909 | 0.8961 | 3.674 |
| 1e-3 | 10 | 4.956 | 0.8953 | 3.895 |
| 3e-3 | 10 | 5.426 | 0.8876 | 4.089 |

Rates 3e-4 and 1e-3 qualify for the planned 30-epoch confirmations. This is an initial single-seed tuning result; the final LR remains unset until confirmation. The lowest-MAE checkpoint differs from the lowest-NLL checkpoint. [Per-run records](runs.csv), [full validation metrics and configs](results.json), and [epoch curves](curves.csv).

All four real-image bf16 preflights passed: sampled Statistical MaxRL with MAE/MSE kernels, scalar MAE, and the conditional digit policy with CE. Each trained on 256 images and evaluated 65 validation and 65 test images. Losses, gradients and distribution metrics were finite; scalar NLL was unavailable as intended. These are pipeline checks, not method-comparison results. [Preflight validation](preflight_validation.json).

The first full-data LR pilot was preempted three times by another workload appearing on its GPU. The original three-attempt limit stopped the queue. Scheduler revision `50ac82d` replaces that limit with per-GPU cooldowns and retries external interruptions within the same budget. Genuine training errors still halt dispatch. The original state is backed up, every attempt and its charge is preserved, and training source hashes and the plan are unchanged. [Recovery record](queue_recovery.json).

The recovered supervisor is running on `cmu-L40-live` with PID `1425732`. The snapshot has eight completed nodes (four preflights and four LR pilots) and 522 pending nodes, including **0/66 completed main runs**. Charged time is **0.477/64 GPU-hours**. A further external interruption of the population binary pilot was queued for retry without stopping the queue. GPU 0 is temporarily cooling down; the other nine GPUs currently have compute processes. The queue uses at most six GPUs concurrently and charges no waiting time. Observed peak study concurrency so far is one GPU.

State, process ownership, attempts, logs and budget accounting are under `/tmp/utkface-study-20261013/`. The [protocol](PLAN.md) includes inspection and report commands. Training sources are frozen in the [execution metadata](execution.json).

An hourly heartbeat, `UTKFace scaled study follow-up`, is active in this thread to inspect progress, recover the supervisor if necessary, and collect the final analysis. The server queue runs independently of the desktop app. Scheduled Codex follow-ups require the desktop app to remain running, as described in the [scheduled task documentation](https://learn.chatgpt.com/docs/automations).

The [interim report](REPORT.md) contains the measured noise design, preflight runs and completed pilots. [Validation](validation.json) records the original 64 passing focused tests, clean-data audits, two-process DDP metric checks, passed GPU preflights, 14 passing scheduler/report checks after the recovery fix, and completed-pilot integrity checks.
