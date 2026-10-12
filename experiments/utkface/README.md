# UTKFace categorical age estimation

ImageNet-pretrained ResNet-18 has eight residual blocks, global average pooling and a 512-to-100 linear head. Softmax gives a categorical policy on integer ages 0–99. All methods fine-tune the entire network with AdamW, bf16 CNN execution, float32 probability math, warmup/cosine scheduling and gradient clipping. Ages outside this range are excluded. The original 224px augmentation and decade-stratified split seed 42 are retained. [Provenance](SOURCES.md).

The reusable implementation is in `src/experiments/utkface/`: `prepare.py` verifies the pinned archive and writes a hashed split manifest; `data.py` implements transforms and sharding; `model.py` builds the common policy; `advantages.py` contains sampled estimators; `losses.py` contains exact objectives; `distributed.py` initializes torchrun; `run.py` trains, selects the best validation checkpoint, and optionally evaluates test data; `sweep.py` runs independent comparisons through the existing GPU pool.

For probabilities p(a), target y and D=99:

| CLI method | Loss / update |
|---|---|
| `mse` | ((Σ p(a)a − y)/D)²; MSE of the mean, not expected action MSE |
| `ce` | −log p(y) |
| `binary_maxrl_population` | −log p(y), exactly CE |
| `rloo_mae_population`, `rloo_mse_population` | −E[r], with r=1−abs(a−y)/D or 1−((a−y)/D)² |
| `grpo_mae_population`, `grpo_mse_population` | same expected reward gradient divided by detached population reward std; std below 1e−4 yields zero |
| `tailrl_population` | −(1/D) Σ from d=0 to D−1 log P(abs(a−y)≤d) |
| `stat_maxrl_mse_population` | −log Σ p(a) exp(−0.5((a−y)/σ)²) |
| `stat_maxrl_mae_population` | −log Σ p(a) exp(−abs(a−y)/b) |
| `grpo_mae`, `grpo_mse`, `rloo_mae`, `rloo_mse`, `binary_maxrl`, `tailrl` | sampled score-function estimates, batched over images and rollouts |

Kernel scales are in years. Gaussian normalization constants are omitted because they have no parameter gradient. Population GRPO is a semi-gradient, so its logged scalar is not a comparable objective value. RLOO's baseline changes sampled variance but disappears from the exact population gradient. Binary MaxRL uses exact age equality, with zero update for a sampled group containing no successes. Its finite-rollout estimate is biased; it must not be conflated with population CE.

Use `--survival-floor` to clamp population TailRL log-survival and mask matching sampled increments. The default zero preserves the ordinal objective. Rollouts for each image stay together on their rank; DDP distributes images, not action groups. NCCL, SyncBatchNorm and distributed training samplers are enabled under torchrun. Evaluation uses unpadded shards and sums counts and metrics across ranks. FSDP is not implemented.

Follow `server.md`: edit and commit locally, push to GitHub, pull on `cmu-L40-live:~/maxrl-statistics`, then execute there. All commands below run on that server.

```bash
uv sync --dev --python 3.12
uv run --python 3.12 python -m src.experiments.utkface.prepare --destination /tmp/utkface-data
uv run --python 3.12 pytest tests/test_utkface_losses.py tests/test_utkface_data.py -q
uv run --python 3.12 python -m src.experiments.utkface.run \
  --manifest /tmp/utkface-data/manifest.json --output-dir /tmp/utkface-ce \
  --method ce --device cuda:1 --epochs 20 --evaluate-test
CUDA_VISIBLE_DEVICES=1,5 uv run --python 3.12 torchrun --standalone --nproc-per-node=2 \
  -m src.experiments.utkface.run --manifest /tmp/utkface-data/manifest.json \
  --output-dir /tmp/utkface-ddp --method tailrl_population --batch-size 32
```

Batch size is per rank, so the DDP example keeps global batch 64. Use a fresh output directory for every run. `config.json`, `history.json`, `result.json` and optionally `best.pt` record configs, split fingerprint, environment, commit, counts, validation curves and test metrics. Test evaluation is disabled unless requested. The selected checkpoint minimizes validation MAE of the policy mean for every method. Mean, median and mode age errors, RMSE, policy entropy, expected action errors, CE, TailRL and age-decade errors are reported. Oracle best-of-K errors use the true target to select the closest of K draws; they are distribution diagnostics, not deployable predictions.

The pilot compares learning rates 1e−4, 3e−4, 1e−3 and kernel scales 2, 5, 10 on 4096 training / 512 validation examples for three epochs, without accessing test data. The main sweep selects a learning rate per objective and scale, then retains all three kernel scales on the full split and independent seeds: 45 pilot runs and 45 full runs with three seeds. CE also represents population binary MaxRL; a separate sampled binary arm tests the finite-rollout behavior.

```bash
uv run --python 3.12 python -m src.experiments.utkface.sweep --stage pilot \
  --manifest /tmp/utkface-data/manifest.json --output-dir /tmp/utkface-pilot --gpu-ids 1 5 7 9
uv run --python 3.12 python -m src.experiments.utkface.sweep --stage select \
  --manifest /tmp/utkface-data/manifest.json --output-dir /tmp/utkface-pilot --gpu-ids 1 \
  --selection /tmp/utkface-selection.json
uv run --python 3.12 python -m src.experiments.utkface.sweep --stage main \
  --manifest /tmp/utkface-data/manifest.json --output-dir /tmp/utkface-main --gpu-ids 1 5 7 9 \
  --selection /tmp/utkface-selection.json --epochs 20 --seeds 42 43 44
```
