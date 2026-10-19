# Qwen3.5 inference evaluation

Three model sizes (0.8B, 2B, 4B), each with Base, post-trained thinking-off and post-trained thinking-on: nine configurations. Six pinned checkpoints share one pinned vLLM Docker image. Each configuration generates eight samples for each image in the same decade-stratified 1024-image subset of the cleaned 2308-image validation split. The subset seed is 20261020. Sixty-four disjoint validation images support pipeline checks and tuning.

Prompt: `What is the person's age in years? Put your final answer in \boxed{}.`

Every model uses temperature 0.7, top-p 0.8, top-k 20 and no presence penalty. Base weights use their own tokenizer and processor with the paired post-trained checkpoint's official chat template and thinking disabled. Post-trained modes explicitly set `enable_thinking`. Neither labels nor filenames are included in model requests. Numeric boxed final answers must lie in 0–99; invalid answers remain in saved responses and coverage counts.

Pilot completion/truncation rates select the smallest cap among 128/256/512 tokens for Base and thinking-off, and 256/512/1024/2048 for thinking-on. Pilot throughput selects 8 or 32 concurrent HTTP requests. These choices do not use age accuracy. The serving engine admits at most 128 active sequences and uses bf16 with a 4096-token context. Post-trained thinking modes share a loaded model. At most three models run concurrently on separate GPUs.

Metrics are MAE, RMSE and Spearman correlation for pooled single draws, per-image sample means and medians, and each of eight 1024-image sampling replicates. The report separates complete-eight aggregates from aggregates using available valid samples. Replicate means and sample SDs describe generation variability. Full raw responses, token/truncation counts, invalid rates and explicitly labeled worst-case invalid penalties are retained.

The server queue reuses `scheduler.py` GPU inventory and idle checks: no compute process, at most 1024 MiB used and at most 5% utilization, confirmed twice at least ten seconds apart and rechecked immediately before launch. GPUs are selected by UUID. Contention causes only the study's own worker/container to stop, with saved responses retained for resumption and a 15-minute escalating GPU cooldown. Infrastructure errors stop dispatch for investigation. Downloads run in a CPU-only container and use a named Docker volume, avoiding dataset/weights growth in `/tmp`.

All Python commands run on `cmu-L40-live` through `uv`, following `server.md`. Commit/push locally and pull remotely before execution. Run artifacts belong under `/tmp/utkface-vlm-20261020`.

```bash
uv run --no-sync python -m src.experiments.utkface.vlm_eval prepare \
  --root /tmp/utkface-vlm-20261020 \
  --manifest /tmp/utkface-clean-v2/manifest.json \
  --image vllm/vllm-openai@sha256:PINNED_DIGEST
uv run --no-sync python -m src.experiments.utkface.vlm_queue \
  --root /tmp/utkface-vlm-20261020 --detach
```

`state.json`, `supervisor.log` and `downloads.json` record autonomous progress. The supervisor refreshes `summary.json`, `summary.csv` and `REPORT.md`, including the final report when all nine configurations complete. Restart the same queue without recreating the plan to adopt active workers or resume saved image responses. Containers and their logs remain identifiable by study labels.
