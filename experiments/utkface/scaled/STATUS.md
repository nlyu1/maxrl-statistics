# Learning-rate pilot results, 2026-10-13

At 14:43 Asia/Shanghai, 27 of 96 initial full-data learning-rate pilots were complete. Six methods have finished the four-rate grid; three direct categorical MAE pilots have also completed. Each run uses ten epochs, training seed 42, the same 18,462-image training pool and 2,308 clean validation images; each epoch presents 18,432 training images. No pilot evaluated test data. The table shows each completed grid's lowest validation mean-prediction MAE, with metrics from that checkpoint.

| Method | LR | Best epoch | Mean MAE, years | Median Spearman | Label NLL, nats | Entropy, nats |
|---|---:|---:|---:|---:|---:|---:|
| Population binary MaxRL (CE) | 1e-3 | 9 | 4.718 | 0.9017 | 2.998 | 2.471 |
| 64-sample binary MaxRL | 3e-4 | 10 | 4.909 | 0.8961 | 3.674 | 2.148 |
| 64-sample GRPO, MAE reward | 3e-4 | 9 | 6.501 | 0.8850 | 18.081 | 0.094 |
| Population GRPO, MAE reward | 1e-3 | 10 | 6.638 | 0.8855 | 30.485 | 0.090 |
| 64-sample GRPO, MSE reward | 3e-4 | 10 | 6.721 | 0.8806 | 19.872 | 0.086 |
| Population GRPO, MSE reward | 3e-4 | 10 | 6.765 | 0.8829 | 21.789 | 0.056 |

Population binary MaxRL qualifies rates 1e-3 and 3e-3 for 30-epoch confirmation. Sampled MSE-GRPO qualifies 3e-4 and 1e-4; the other four completed grids qualify 3e-4 and 1e-3. Final learning rates remain unset until confirmation. GRPO's low entropy and high NLL show very concentrated predictions at these early checkpoints; the longer runs will test whether its point error improves. These are single-seed tuning results. [Per-run records](runs.csv), [full validation metrics and configs](results.json), [epoch curves](curves.csv), and [integrity checks and all pilot metrics](pilot_integrity.json).

All four real-image bf16 preflights passed: sampled Statistical MaxRL with MAE/MSE kernels, scalar MAE, and the conditional digit policy with CE. Each trained on 256 images and evaluated 65 validation and 65 test images. Losses, gradients and distribution metrics were finite; scalar NLL was unavailable as intended. These are pipeline checks, not method-comparison results. [Preflight validation](preflight_validation.json).

The first full-data LR pilot was preempted three times by another workload appearing on its GPU. The original three-attempt limit stopped the queue. Scheduler revision `50ac82d` replaces that limit with per-GPU cooldowns and retries external interruptions within the same budget. Genuine training errors still halt dispatch. The original state is backed up, every attempt and its charge is preserved, and training source hashes and the plan are unchanged. [Recovery record](queue_recovery.json).

The recovered supervisor is running on `cmu-L40-live` with PID `1425732`. The snapshot has 31 completed nodes (four preflights and 27 LR pilots), 498 pending nodes and one active pilot on GPU 1, including **0/66 completed main runs**. Charged time at the report snapshot is **2.211/64 GPU-hours**. The direct categorical MAE grid is in progress. There are no permanently failed jobs. The queue uses at most six GPUs concurrently and charges no waiting time. Observed peak study concurrency so far is one GPU.

State, process ownership, attempts, logs and budget accounting are under `/tmp/utkface-study-20261013/`. The [protocol](PLAN.md) includes inspection and report commands. Training sources are frozen in the [execution metadata](execution.json).

An hourly heartbeat, `UTKFace scaled study follow-up`, is active in this thread to inspect progress, recover the supervisor if necessary, and collect the final analysis. The server queue runs independently of the desktop app. Scheduled Codex follow-ups require the desktop app to remain running, as described in the [scheduled task documentation](https://learn.chatgpt.com/docs/automations).

The [interim report](REPORT.md) contains the measured noise design, preflight runs and completed pilots. [Validation](validation.json) records the original 64 passing focused tests, clean-data audits, two-process DDP metric checks, passed GPU preflights, 14 passing scheduler/report checks after the recovery fix, and completed-pilot integrity checks.
