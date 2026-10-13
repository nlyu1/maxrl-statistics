# Coverage at the extension cutoff

Audit: 2026-10-14T07:30:58.498619+08:00. Extension cutoff: 07:30 GMT+8, October 14.

The extension has stopped admitting work and its supervisor has exited. Its uncompleted jobs are explicitly marked deadline-skipped; owned attempts and charges are retained. The original queue may continue within its existing 64 GPU-hour budget. Waiting for free GPUs is not an experimental failure.

| Queue | Stage | Optional | Completed | Planned | Pending | Deadline-skipped |
|---|---|---|---:|---:|---:|---:|
| utkface-extension-20261014 | matched_steps | no | 0 | 8 | 0 | 8 |
| utkface-extension-20261014 | noise_realization | no | 0 | 32 | 0 | 32 |
| utkface-extension-20261014 | subset_replication | no | 0 | 16 | 0 | 16 |
| utkface-study-20261013 | digits | yes | 0 | 16 | 16 | 0 |
| utkface-study-20261013 | digits_tuning | yes | 0 | 24 | 24 | 0 |
| utkface-study-20261013 | flat | no | 0 | 10 | 10 | 0 |
| utkface-study-20261013 | flat_tuning | no | 0 | 40 | 40 | 0 |
| utkface-study-20261013 | main | no | 66 | 66 | 0 | 0 |
| utkface-study-20261013 | noise | no | 0 | 60 | 60 | 0 |
| utkface-study-20261013 | noise | yes | 0 | 20 | 20 | 0 |
| utkface-study-20261013 | noise_replication | yes | 0 | 40 | 40 | 0 |
| utkface-study-20261013 | refine | no | 48 | 48 | 0 | 0 |
| utkface-study-20261013 | scalar | no | 2 | 6 | 4 | 0 |
| utkface-study-20261013 | smoke | no | 4 | 4 | 0 | 0 |
| utkface-study-20261013 | subset | no | 0 | 20 | 20 | 0 |
| utkface-study-20261013 | subset_tuning | no | 0 | 80 | 80 | 0 |
| utkface-study-20261013 | wide | no | 96 | 96 | 0 | 0 |

“Required” includes tuning, preflight and final comparisons; it is not the number of final training runs. The reduced categorical matrix has eight methods. The qualified scalar MAE/MSE heads add two methods to the original subset, schedule and noise arms. SD-10 noise, noise training-seed repeats and digit-policy jobs are optional.

utkface-study-20261013: 216/430 required jobs complete, 214 required jobs pending; 41.230570/64 GPU-hours charged across 230 attempts.
utkface-extension-20261014: 0/56 required jobs complete, 0 required jobs pending; 0.000000/24 GPU-hours charged across 0 attempts.

Missing cells provide no evidence about subset rankings, constant-LR effects, label-noise robustness, SNR sensitivity or digit-policy equivalence. Main results retain three seeds per configuration; scalar head confirmations have one matched seed each. Final-epoch test models were not retained.

[Exact audit and all job identifiers](cutoff-audit.json). [Interactive report](dashboard.html).
