# Status after preflight and queue recovery, 2026-10-13

All four real-image bf16 preflights passed: sampled Statistical MaxRL with MAE/MSE kernels, scalar MAE, and the conditional digit policy with CE. Each trained on 256 images and evaluated 65 validation and 65 test images. Losses, gradients and distribution metrics were finite; scalar NLL was unavailable as intended. These are pipeline checks, not method-comparison results. [Preflight validation](preflight_validation.json).

The first full-data LR pilot was preempted three times by another workload appearing on its GPU. The original three-attempt limit stopped the queue. Scheduler revision `50ac82d` replaces that limit with per-GPU cooldowns and retries external interruptions within the same budget. Genuine training errors still halt dispatch. The original state is backed up, every attempt and its charge is preserved, and training source hashes and the plan are unchanged. [Recovery record](queue_recovery.json).

The recovered supervisor is running on `cmu-L40-live` with PID `1425732`. The snapshot has four completed preflights and 526 pending nodes, including **0/66 completed main runs**. Charged time is **0.112/64 GPU-hours**. All ten GPUs currently have other workloads; the queue will resume on an idle GPU. It uses at most six GPUs concurrently and charges no waiting time.

State, process ownership, attempts, logs and budget accounting are under `/tmp/utkface-study-20261013/`. The [protocol](PLAN.md) includes inspection and report commands. Training sources are frozen in the [execution metadata](execution.json).

An hourly heartbeat, `UTKFace scaled study follow-up`, is active in this thread to inspect progress, recover the supervisor if necessary, and collect the final analysis. The server queue runs independently of the desktop app. Scheduled Codex follow-ups require the desktop app to remain running, as described in the [scheduled task documentation](https://learn.chatgpt.com/docs/automations).

The [interim report](REPORT.md) contains the measured noise design and the four preflight runs. [Validation](validation.json) records the original 64 passing focused tests, clean-data audits, two-process DDP metric checks, passed GPU preflights, and 14 passing scheduler/report checks after the recovery fix.
