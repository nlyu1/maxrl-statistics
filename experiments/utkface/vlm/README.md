# Qwen3.5 inference evaluation

Three model sizes (0.8B, 2B, 4B), each with Base, post-trained thinking-off and post-trained thinking-on: nine configurations. Six pinned checkpoints share one pinned vLLM Docker image. Each configuration generates eight samples for each image in the same decade-stratified 1024-image subset of the cleaned 2308-image validation split. The subset seed is 20261020. Sixty-four disjoint validation images support pipeline checks and tuning.

## Prompt and budget follow-up, 2026-10-22

The completed 20261020 study remains unchanged. `vlm_eval followup` copies its exact subset file and pinned checkpoint/template specifications into a new study, preserving the eight samples, seeds and temperature/top-p/top-k/presence penalty. It resolves the manifest to persistent storage and records the reference plan/subset hashes.

Follow-up prompt: `Estimate the person's age in years from their appearance in this photo. Give your single best estimate, even if uncertain. Your final answer must be exactly one integer from 0 to 99 inside \boxed{}, with no other text or range.` Thinking-on retains its internal reasoning; the restriction applies to the final answer.

Base/off have a 1024-token cap. Thinking-on pilots try 4096 then 8192, choosing the first cap with at most 3% length-limited completions lacking a valid boxed answer. If neither passes, the run uses the authorized 8192 cap and explicitly records `token_budget_limited=true`; it does not claim the pilot criterion passed.

The follow-up uses bf16, FlashAttention 2, CUDA graphs/compilation, 256 active sequences, an 8192-token batching budget and a 12288-token context. `--enforce-eager` in the original study disabled CUDA graphs; its logs confirm that attention already used FA2. Gated DeltaNet layers use their own Triton/FLA kernels.

Each mode benchmarks 32 and 64 concurrent HTTP requests on all 64 disjoint pilot images, with eight generations per image and the selected token cap. An unmeasured n=8 warm-up with up to 256 tokens warms decoding and image processing first. The candidates run twice in opposite orders (32, 64, 64, 32), and the lowest concurrency within 5% of peak median images/second is selected. Each request offers eight sequences: 32 HTTP requests can already supply 256 sequences. The pilot retains tokens/second, GPU utilization/memory, running/waiting sequences, KV-cache usage and preemption counts. This fixes the original 16-image throughput pilot, which could not fill its 32-request limit. Short pilot timings are throughput estimates, not full-pass runtime predictions.

Checkpoint files are reused through a Docker volume bound to persistent checkpoint storage. All new study state, outputs and logs live under `~/.local/state/maxrl-statistics/utkface-vlm-20261022-followup`. The existing idle-GPU checks, process exceptions and contention handling apply.

```bash
uv run --no-sync python -m src.experiments.utkface.vlm_eval followup \
  --root "$HOME/.local/state/maxrl-statistics/utkface-vlm-20261022-followup" \
  --reference /tmp/utkface-vlm-20261020 \
  --cache-volume utkface-vlm-qwen35-persistent
uv run --no-sync python -m src.experiments.utkface.vlm_queue \
  --root "$HOME/.local/state/maxrl-statistics/utkface-vlm-20261022-followup" --detach
```

## Original 20261020 protocol

Prompt: `What is the person's age in years? Put your final answer in \boxed{}.`

Every model uses temperature 0.7, top-p 0.8, top-k 20 and no presence penalty. Base weights use their own tokenizer and processor with the paired post-trained checkpoint's official chat template and thinking disabled. Post-trained modes explicitly set `enable_thinking`. Neither labels nor filenames are included in model requests. Numeric boxed final answers must lie in 0–99; invalid answers remain in saved responses and coverage counts.

Pilot rates of length-limited responses without a valid final boxed answer select the smallest cap among 128/256/512 tokens for Base and thinking-off, and 256/512/1024/2048 for thinking-on. All length-limited responses remain counted as truncated, including those with a parseable answer. Pilot throughput selects 8 or 32 concurrent HTTP requests. These choices do not use age accuracy. The serving engine admits at most 128 active sequences and uses bf16 with a 4096-token context. Post-trained thinking modes share a loaded model. At most three models run concurrently on separate GPUs.

Metrics are MAE, RMSE and Spearman correlation for pooled single draws, per-image sample means and medians, and each of eight 1024-image sampling replicates. The report separates complete-eight aggregates from aggregates using available valid samples. Replicate means and sample SDs describe generation variability. Full raw responses, token/truncation counts, invalid rates and explicitly labeled worst-case invalid penalties are retained.

The server queue reuses `scheduler.py` GPU inventory and idle checks: no blocking compute process, at most 1024 MiB used and at most 5% utilization, confirmed twice at least ten seconds apart and rechecked immediately before launch. Explicitly approved process exceptions require matching GPU UUID, PID, owner UID, process start time, command hash and server boot ID. Only those verified instances are excluded from dispatch and contention checks; their memory and utilization remain included. The shared training scheduler keeps its original process checks. GPUs are selected by UUID. Contention causes only the study's own worker/container to stop, with saved responses retained for resumption and a 15-minute escalating GPU cooldown. Infrastructure errors stop dispatch for investigation. Downloads run in a CPU-only container and use a named Docker volume, avoiding dataset/weights growth in `/tmp`.

All Python commands run on `cmu-L40-live` through `uv`, following `server.md`. Commit/push locally and pull remotely before execution. Run artifacts are retained under `~/.local/state/maxrl-statistics/utkface-vlm-20261020`; the historical `/tmp/utkface-vlm-20261020` path remains a symlink.

```bash
UTKFACE_ROOT="$HOME/.local/state/maxrl-statistics"
uv run --no-sync python -m src.experiments.utkface.vlm_eval prepare \
  --root "$UTKFACE_ROOT/utkface-vlm-20261020" \
  --manifest "$UTKFACE_ROOT/utkface-clean-v2/manifest.json" \
  --image vllm/vllm-openai@sha256:PINNED_DIGEST
uv run --no-sync python -m src.experiments.utkface.vlm_queue \
  --root "$UTKFACE_ROOT/utkface-vlm-20261020" --detach
```

`state.json`, `supervisor.log` and `downloads.json` record autonomous progress. The supervisor refreshes `summary.json`, `summary.csv` and `REPORT.md`, including the final report when all nine configurations complete. Restart the same queue without recreating the plan to adopt active workers or resume saved image responses. Containers and their logs remain identifiable by study labels.
