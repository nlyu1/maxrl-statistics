# Scaled UTKFace study: original 64 GPU-hour planning reference

**The selected study is complete.** All 514 selected jobs, including tuning and preflight, finished; 222 provide scientific final evaluations. Forty optional seed repetitions remain held and thirty-two independent noise-field runs were excluded. [Final report](FINAL.md), [all result tables](FINAL_TABLES.md), [completion audit](final-audit.json).

On October 15 the user removed GPU-hour enforcement, raised combined concurrency to eight and revised the retained follow-up scope. [Current orchestration, queue and deployment audit](ORCHESTRATION.md). The immutable original plan retains its initial planning values; current dispatch uses the explicit coordination policy.

This study uses a new split created after image deduplication. The earlier [12-epoch report](../RESULTS.md) describes the original split and is a separate experiment. Code is on `codex/utkface-scaled-study`.

[Interactive report](dashboard.html), [interpretation](ANALYSIS.md), [launch status](STATUS.md), [generated tables and noise SNR](REPORT.md), [deadline extension](EXTENSION.md), [data audit](data_audit.json), [fixed subset indices](training_subsets.json), [frozen execution metadata](execution.json), and [validation record](validation.json).

The pinned archive has 23,671 valid images aged 0–99. Preprocessing drops all 417 images from 204 conflicting-label duplicate groups and removes 176 redundant copies from consistently labeled groups. It retains 23,078 unique decoded RGB images, then applies the original decade-stratified 80/10/10 split with seed 42: **18,462 train / 2,308 validation / 2,308 test**. An independent audit finds zero duplicate groups and zero cross-split image overlap. This checks exact pixels; it does not establish person-level independence or remove near-duplicates.

The cleaned manifest fingerprint is `873be56dfe1904d383e5a8952d7b558d6957cc322c5d04a7a0c0a73ff81573f2`. The 4,096-example subset is stratified by age decade, and the 1,024-example subset is stratified within it, using subset seed 2026. Both subsets remain fixed across epochs and methods. Validation/test use every held-out image. Full-data training drops its final incomplete batch, presenting 18,432 images per epoch; the smaller sizes are exact multiples of batch 64.

All runs fine-tune pretrained ResNet-18, use bf16 CNN execution, float32 probability math, AdamW with weight decay 1e-4, batch 64, gradient clipping at 5, and the existing cosine schedule with 5% warmup and minimum LR ratio 0.1. Each scheduled run uses one GPU. The implementation retains torchrun/DDP support.

The primary matrix has 22 configurations and three training seeds (42, 43, 44), totaling **66 main runs of 50 epochs**:

| Family | 64 sampled actions | Exact population |
|---|---|---|
| TailRL | Yes | Yes |
| Zero-one binary MaxRL | Yes | Yes, identical to CE |
| Statistical MaxRL MAE, widths 2 and 5 years | Both widths | Both widths |
| Statistical MaxRL MSE, widths 2 and 5 years | Both widths | Both widths |
| GRPO, MAE and MSE rewards | Both rewards | Both rewards |
| RLOO, MAE and MSE rewards | Both rewards | Both rewards |

Direct MSE and MAE on the categorical policy's mean age add two configurations to the 20 above. Gaussian kernels use `exp(-0.5 * (error / width)^2)` and Laplace kernels use `exp(-abs(error) / width)`. Sampled Statistical MaxRL uses detached, centered, self-normalized kernel weights computed in log space. Its finite-sample gradient is biased; it converges to the population gradient as the rollout count grows. Zero-one success means exact integer-age equality.

Learning rates are selected using clean validation MAE of the policy mean. Each primary configuration runs the grid **1e-4, 3e-4, 1e-3, 3e-3** for 10 full-data epochs, then reruns its two best rates for 30 full-data epochs. The best validation checkpoint among the two confirmations determines the LR for all three 50-epoch runs. Tuning never evaluates the test split. Test evaluation happens after each final run's checkpoint selection, and never controls later LR selection or the scalar-head gate.

The reduced matrix has eight configurations: population TailRL; direct categorical MSE, MAE and CE; population Statistical MaxRL MAE/MSE at width 2; and population GRPO with MAE/MSE rewards.

| Study | Required final runs | Training images | Epochs | Seeds |
|---|---:|---:|---:|---|
| Primary categorical matrix | 66 | Full | 50 | 42, 43, 44 |
| Fixed smaller training sets | 16 | 4,096 or 1,024 | 50 | 42 |
| Constant-LR comparison | 8 | 4,096 | 50 | 42 |
| Training-label noise, 2 modes × 3 levels | 48 | 4,096 | 50 | 42 |

Each smaller-data configuration tunes the same four LRs for 20 epochs on its own fixed subset before its 50-epoch run. The constant-LR arm tunes four rates for 20 epochs on 4,096 images and runs its winner for 50. Its comparator is the corresponding cosine arm. Constant LR starts at the selected rate on step one; this removes warmup as well as cosine decay. AdamW weight decay remains 1e-4.

Two scalar-head candidates use a single ResNet output followed by `99 * sigmoid(logit)`, matching the categorical mean's age range. They optimize direct MSE or MAE with the same normalization by 99. Each receives the same full-data LR grid and confirmations as its categorical counterpart. A candidate qualifies when its best 30-epoch validation MAE is no more than 1.05 times that counterpart's best validation MAE. Qualifying heads add three full-data seeds and join the subset, constant-LR and noise comparisons. The decision uses validation data only.

Noise affects training labels only. Each clean integer age receives Gaussian noise with requested standard deviation **1, 3 or 5 years**, then is rounded to the nearest integer and clipped to 0–99. Fixed noise is sampled once per image; fresh noise changes each epoch, hence each training presentation. Both modes share their first realization, and all methods use the same noise seed 2026. Clean validation and test labels remain intact. The noise arms reuse the corresponding clean-subset LR, isolating robustness at fixed hyperparameters.

Every epoch records realized noise MSE, bias and standard deviation after rounding/clipping. SNR is `10 * log10(Var(clean subset ages) / E[(noisy - clean)^2])` in dB. The report averages the realized MSE across epochs before computing SNR. Clean-label SNR is infinite and stored as null. Optional training-seed replications reuse the corruption field and measure optimization variability, rather than independent noise realizations.

Evaluation includes mean/median/mode MAE, mean RMSE, **Spearman correlation between predicted median age and clean target age**, and **label NLL = -log p(target age)**. Spearman uses the entire evaluation split, with average ranks for ties; under DDP, predictions and targets are gathered across unpadded shards before computing it. Constant predictions yield an undefined correlation, stored as null with an explicit flag. Scalar regression supplies no categorical label distribution, so its NLL is unavailable. Checkpoint selection remains validation MAE of the policy mean or scalar prediction.

Optional work begins only after required work finishes. It includes a noise standard deviation of 10 years; a two-token digit-policy pilot; and a second training seed for noise levels 3 and 5. The digit policy models `p(tens | image) * p(units | image, tens)` with a 10-way tens head and a shared conditional units head. It computes all 100 joint age probabilities, allowing the same population objectives and age metrics. The reduced matrix tunes three LRs for 20 epochs on 4,096 images, followed by 50-epoch subset/full confirmations if budget permits. This is an exploratory architecture comparison; its tuning protocol differs from the primary full-data matrix.

The queue records elapsed time for every launched GPU attempt, including startup, validation, final test evaluation, failed attempts, retries and polling overhead. Waiting for a free GPU is uncharged. GPU-hour enforcement has been removed; the original 64-hour value remains a planning reference. Combined concurrency is now **at most eight GPUs**, and the selected work is described in ORCHESTRATION.md.

The dispatcher wakes on owned-process completion and otherwise polls every ten seconds. Successful completion permits immediate reuse after the final free-GPU check. New claims require two idle observations at least ten seconds apart. A GPU must have no compute process, use at most 1,024 MiB and remain at or below 5% utilization. It rechecks immediately before dispatch and assigns by GPU UUID. If another workload appears during a run, it yields its own process group and retries the interrupted run in a fresh attempt directory. The interrupted GPU receives a cooldown of 15 minutes per observed interruption, capped at one hour. Genuine training errors and timeouts halt dispatch for investigation. It never signals another workload. Supervisor and shared dispatch locks prevent duplicate queues and enforce combined concurrency; persisted state allows adoption. Frozen training hashes are verified before dispatch.

All Python execution runs remotely through `uv`, following `server.md`. Data, checkpoints, plans, logs and budget accounting live under `/tmp`. Code reaches the server through committed GitHub pushes and remote pulls.

The following block records initial data and plan preparation. Do not recreate retained plans or ledgers.

```bash
uv run --python 3.12 python -m src.experiments.utkface.clean \
  --manifest /tmp/utkface-data/manifest.json --destination /tmp/utkface-clean-v2
uv run --python 3.12 python -m src.experiments.utkface.audit \
  --manifest /tmp/utkface-clean-v2/manifest.json --output-dir /tmp/utkface-clean-v2/audit
uv run --python 3.12 python -m src.experiments.utkface.study \
  --manifest /tmp/utkface-clean-v2/manifest.json --output-dir /tmp/utkface-study-20261013 \
  --budget-gpu-hours 64 --max-concurrent 6
```

Current read-only status and reporting commands:

```bash
uv run --python 3.12 python -m src.experiments.utkface.scheduler \
  --plan /tmp/utkface-study-20261013/plan.json \
  --coordination /tmp/utkface-followup-20261015/coordination.json --status
uv run --python 3.12 python -m src.experiments.utkface.scheduler \
  --plan /tmp/utkface-followup-20261015/plan.json \
  --coordination /tmp/utkface-followup-20261015/coordination.json --status
uv run --python 3.12 python -m src.experiments.utkface.study_report \
  --plan /tmp/utkface-study-20261013/plan.json \
  --output-dir /tmp/utkface-study-20261013/report --figures
```

`plan.json` describes every dependency and conditional branch. `state.json` records process ownership, GPU assignments, attempts and budget usage. `supervisor.log` and per-attempt logs expose failures. `study_report.py` exports interim/final CSVs, configs, epoch curves and measured SNR, with main PNG/PDF figures once main runs finish. Partial groups show their actual seed count. Old experiment configs belong to the schema at commit `5f4d631`; this study generates fresh configs.
