# UTKFace orchestration update, October 15

The user removed GPU-hour enforcement, raised combined concurrency to eight GPUs and selected which unfinished extension work should resume. GPU-hours remain measured across every attempt. The 64/24-hour values in historical plans are planning references, not operating caps under the updated coordination policy.

The original plan, results and attempt ledger stay at `/tmp/utkface-study-20261013`. The closed October 14 extension remains an unchanged historical record at `/tmp/utkface-extension-20261014`. A new continuation at `/tmp/utkface-followup-20261015` restores only its sixteen clean 4096-image seed replications and eight 1024-image, 200-epoch runs. The thirty-two independent SD-5 noise-field runs are excluded.

The two active queues use `/tmp/utkface-followup-20261015/coordination.json` and the existing shared dispatch lock. This policy overrides the historical per-plan concurrency and GPU-hour limits. The continuation has no training deadline.

| Pending work at authorization | Active | Held |
|---|---:|---:|
| Remaining original required work, including tuning | 140 | 0 |
| Reinstated clean 4096-image replications | 16 | 0 |
| Reinstated 1024-image × 200-epoch comparisons | 8 | 0 |
| Original optional SD-10 noise settings | 20 | 0 |
| Original additional digit-policy LR pilots | 15 | 0 |
| Original additional digit-policy confirmations | 13 | 0 |
| Original optional noise training-seed repetitions | 0 | 40 |
| Total | **212** | **40** |

The sixteen clean-subset replications remain active because the user explicitly selected them. The later instruction to hold extra seeds applies to the original 88 optional jobs, retaining 48 new-setting jobs and holding 40 noise training-seed repetitions. Held entries remain intact in the original ledger and are excluded from dispatch and active pending counts.

After a successful owned process exits, the dispatcher wakes through a process-exit descriptor and can reuse its GPU in the same iteration. Every launch still requires an immediately preceding inventory check with no compute processes, at most 1024 MiB and at most 5% utilization. New GPU claims require two idle observations at least ten seconds apart. Busy observations reset readiness. Interrupted GPUs retain their contention cooldown and do not use the successful-completion shortcut. The dispatcher never signals other workloads.

Training implementation, data, subset fingerprints, learning-rate selection, epochs and noise settings remain unchanged. The update records its orchestration source revision separately. All previous attempts and their charges are preserved. Historical cutoff documents remain dated evidence; the current dashboard uses the original and continuation plans and labels held jobs explicitly.

Deployment and validation are pending. The source must be committed locally, pushed to GitHub and pulled remotely before tests, preparation or execution, following `server.md`.
