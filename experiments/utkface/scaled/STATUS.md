# Launch status, 2026-10-13

The durable supervisor is running on `cmu-L40-live`, with a 64 GPU-hour allocation and a maximum of six study GPUs concurrently. Its first snapshot has 530 pending DAG nodes, including tuning, conditional scalar branches and optional studies. The required main matrix has 66 runs. **No training attempt has started:** all ten server GPUs currently have other workloads. The queue waits for an idle GPU automatically, without charging waiting time.

The supervisor PID at launch is `1213286`. State, process ownership, attempts, logs and budget accounting are under `/tmp/utkface-study-20261013/`. The [protocol](PLAN.md) includes inspection and report commands. Training sources are frozen in the [execution metadata](execution.json); the supervisor was launched after pulling commit `406f1d5`.

An hourly heartbeat, `UTKFace scaled study follow-up`, is active in this thread to inspect progress, recover the supervisor if necessary, and collect the final analysis. The server queue runs independently of the desktop app. Scheduled Codex follow-ups require the desktop app to remain running, as described in the [scheduled task documentation](https://learn.chatgpt.com/docs/automations).

The [interim report](REPORT.md) currently contains the measured noise design and zero completed main runs. It is a launch snapshot, not a training result. [Validation](validation.json) records 64 passing focused tests, clean-data audits, two-process DDP metric checks and pending real-image GPU preflights.
