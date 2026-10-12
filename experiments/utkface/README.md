# UTKFace: categorical age policies

Every method uses ImageNet-pretrained ResNet-18 with a `Linear(512, 100)` head. Action `a` is the integer age `0,...,99`; `π(a|x) = softmax(logits)[a]`. There is one image forward pass per batch, even for sampled methods with many rollouts. All backbone parameters are trained.

The task follows [cont-maxrl's ordinal experiment](https://github.com/stablegradients/cont-maxrl/tree/b29aeaad894f42b1ddd26f09784a9cf3e0f29449/experiment5_ordinal). Batched advantages and the NCCL/DDP/SyncBatchNorm approach follow [TailRL's localization experiment](https://github.com/Zanette-Labs/TailRL/tree/5682c6ac03387355e017ce966693266bb148fa10/experiments/imagenet_localization). Statistical MaxRL reuses this repository's `src/maxrl_estimator.py` for finite-degree sampled estimates.

## Code and training flow

```text
src/experiments/utkface/
  canonical.py             explicit scientific defaults
  config.py                immutable validated configs and method names
  data.py                  original age filtering, transforms, split; distributed loaders
  prepare.py               image verification, content hash, split audit
  model.py                 ResNet-18, 100 logits
  objectives.py            exact 100-action population losses and sampled PG surrogates
  estimators.py            batched TailRL, binary MaxRL, GRPO, RLOO advantages
  distributed.py           torchrun ranks, NCCL, DDP, SyncBatchNorm, bf16 contexts
  evaluate.py              common policy/point metrics and age subgroups
  train.py                 SGD, cosine schedule, validation, checkpoints, test
  run.py / __main__.py     single training run CLI
  validate_distributed.py  global-gradient equivalence, uneven evaluation, ResNet smoke
  study.py                 multi-seed / learning-rate / temperature study
  analyze.py               validation selection, CSV summaries, learning curves
```

`run.py → canonical/config → train.py → data/model/distributed → forward → objective → backward → DDP reduction → clipping → SGD → scheduler`. Each epoch ends with validation and an atomic checkpoint. The best checkpoint by validation mean-prediction MAE is evaluated on the test split after training. No test metric chooses the checkpoint or hyperparameters.

## Objectives

Let `e=a-y`, `p_a=π(a|x)`, `μ=Σ_a p_a r_a`, and `σ²=Σ_a p_a(r_a-μ)²`. Losses below are averaged over images. The default reward is `r_a=1-|e|/99`; `--reward mse` uses `1-(e/99)²`, and `--reward binary` uses `1[a=y]` for GRPO/RLOO/TailRL.

| CLI method | Population definition / update |
|---|---|
| `mse` | `((Σ_a p_a a-y)/99)²` |
| `ce` | `-log p_y` |
| `rloo_population` | `-Σ_a p_a r_a`; the exact expected RLOO policy gradient |
| `grpo_population` | gradient `-Σ_a p_a ((r_a-μ)/(σ+ε)) ∇log p_a`, with statistics detached and zero update if `σ<1e-4` |
| `maxrl_binary_population` | `-log P(a=y)`, exactly CE |
| `tailrl_population` | `-∫₀¹ log P(r_a≥t) dt`, evaluated exactly over the 100 actions |
| `stat_maxrl_mse_population` | `-log Σ_a p_a exp(-e²/(2τ²))` |
| `stat_maxrl_mae_population` | `-log Σ_a p_a exp(-|e|/τ)` |

`τ` is in age years; the canonical value is 5. Statistical MaxRL uses Gaussian/Laplace observation kernels, without their parameter-independent normalization constants. These population objectives use the exact logarithm, corresponding to infinite truncation degree. `--degree` only affects the sampled Statistical MaxRL estimators.

Population GRPO is a detached policy-gradient surrogate. Its displayed training loss is not a scalar objective whose ordinary derivative includes the derivatives of `μ` and `σ`, and its loss values should not be compared with CE or TailRL. This implements population reward normalization without PPO clipping, an old policy, KL regularization, or multiple updates per batch.

For MAE reward, population TailRL equals the original ordinal loss mathematically: `-(1/99)Σ_{d=0}^{98} log P(|a-y|≤d)`. Sorted log probabilities and `logcumsumexp` avoid the original subtraction of cumulative probabilities and its `1e-12` numerical clamp. The canonical survival floor is zero. Setting a positive `--survival-floor` clamps population log survival and masks the corresponding empirical intervals in sampled TailRL, following TailRL's estimator.

Sampled `grpo`, `rloo`, `maxrl_binary`, `tailrl`, `stat_maxrl_mse`, and `stat_maxrl_mae` sample `[batch, num_rollouts]` categorical actions. Advantages are detached, computed across the rollout dimension, and used in `-mean(advantage * log_prob)`. Binary MaxRL requires `reward='binary'` (set by its canonical config), and all-failure groups contribute zero policy gradient. Finite-group TailRL and binary MaxRL have sampling bias; they are not advertised as unbiased estimators of their population logarithms. The existing degree-D Statistical MaxRL estimator is unbiased for the degree-D truncated objective, rather than the infinite-degree logarithm.

The MSE baseline deliberately retains the same categorical head as every other method and regresses its mean. The original task's MSE implementation used a separate scalar sigmoid head. Matching the mean does not identify the policy's variance or sampled accuracy.

## Data and reproducibility

Use the aligned/cropped [UTKFace dataset](https://susanqq.github.io/UTKFace/). Supply a directory containing the original JPG names, or its parent containing `UTKFace/`. Ages above 99 are excluded. Filename matching and lexicographic ordering follow the original loader. The original 80/10/10 split uses `train_test_split` twice, stratified by age decade, seed 42; training seeds never change the split.

Training uses random resized crops to 224, scale `(0.8,1.0)`, horizontal flips, ColorJitter 0.2, and ImageNet normalization. Evaluation resizes to 256 and center-crops to 224. Smaller image sizes and stratified sample caps are available for smoke tests; they change the experiment and must be reported.

Default training: bf16 forward, fp32 objective arithmetic, SGD with momentum 0.9, weight decay `1e-4`, global batch 128, LR `1e-3`, 20 epochs, 5% linear warmup then cosine decay, gradient clipping at 10. These are explicit starting settings, not validation-tuned claims. Different losses have different gradient scales; compare a validation-selected LR grid before drawing conclusions.

Each run writes configuration, environment/git metadata, split names/ages/file sizes and hash, metrics JSON/Parquet, and last/best/milestone checkpoints. Dataset auditing additionally hashes every image's contents. Checkpoints preserve optimizer, scheduler, sampler epoch, and each rank's Python/NumPy/CPU/CUDA/DataLoader random states. Epoch-boundary resumption requires the same scientific config, split manifest, and world size.

## Server commands

Follow `server.md`: edit and commit locally, push through GitHub, pull into `cmu-L40-live:~/maxrl`, and execute there. Data and generated artifacts belong under `/tmp`, outside the checkout.

```bash
cd ~/maxrl
uv sync --dev
uv run python -m src.experiments.utkface.prepare \
  --data-dir /tmp/utkface/UTKFace --output /tmp/utkface-study/dataset.json

uv run pytest tests/test_utkface_objectives.py tests/test_utkface_data_training.py \
  tests/test_utkface_analysis.py tests/test_maxrl_estimator.py -q
uv run ruff check src/experiments/utkface tests/test_utkface*.py

CUDA_VISIBLE_DEVICES=0,1 uv run torchrun --standalone --nproc-per-node=2 \
  -m src.experiments.utkface.validate_distributed \
  --output /tmp/utkface-study/ddp_validation.json

CUDA_VISIBLE_DEVICES=0,1 uv run torchrun --standalone --nproc-per-node=2 \
  -m src.experiments.utkface --data-dir /tmp/utkface/UTKFace \
  --output-dir /tmp/utkface-study/smoke --method tailrl_population \
  --epochs 1 --max-train-samples 512 --max-eval-samples 128
```

For one GPU, replace the torchrun prefix with `uv run python`. `--batch-size` always means the global image batch and must divide evenly across ranks. GPU DDP uses SyncBatchNorm so normalization uses the global batch. Evaluation uses disjoint shards without padding, bypasses DDP's forward collectives, and reduces metric sums/counts; uneven final batches do not duplicate images or deadlock. CPU/gloo exists for validation. FSDP is not implemented; ResNet-18 fits comfortably with DDP.

```bash
# 8 population arms × 3 paired seeds, independent two-GPU workers
uv run python -m src.experiments.utkface.study \
  --data-dir /tmp/utkface/UTKFace --output-dir /tmp/utkface-study/main \
  --gpu-groups 0,1 2,3 --seeds 42 43 44

# Optional validation tuning, with Gaussian/Laplace temperature sensitivity
uv run python -m src.experiments.utkface.study \
  --data-dir /tmp/utkface/UTKFace --output-dir /tmp/utkface-study/tuned \
  --gpu-groups 0,1 2,3 --learning-rates 0.001 0.01 0.1 \
  --temperatures 2 5 10 --seeds 42 43 44

uv run python -m src.experiments.utkface.analyze \
  --study-dir /tmp/utkface-study/main --output-dir /tmp/utkface-study/report
```

Re-running a study resumes its runs and skips completed summaries after checking their configurations and split manifests. Use disjoint GPU groups and the same group sizes when resuming. Logs live under the study's `logs/`; failures propagate instead of silently continuing. Analysis rejects missing runs unless `--allow-incomplete` is explicit, checks shared splits and paired initial weights, selects hyperparameters by mean validation MAE, and reports test mean ± sample SD across seeds.

Metrics include mean/greedy/median point MAE, sampled expected MAE/MSE, exact-match probability, within-five-years probability, entropy, CE, TailRL, both statistical objectives, and exact expected best-of-1/8/64 reward. Best-of-k assumes an oracle selects the sample closest to the known age. Per-decade and original easy/medium/hard age groups are also recorded; age groups are not a substitute for a demographic fairness evaluation.

## Validation and experiment status

See [RESULTS.md](RESULTS.md) for executed checks and measured runs. Commands above describe the reproducible workflow; their presence is not evidence that they have passed.
