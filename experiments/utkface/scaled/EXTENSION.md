# Deadline extension, 2026-10-14

The extension ends at **07:30 Asia/Shanghai on October 14** and has a separate **24 GPU-hour cap**. Both queues share one dispatch lock and a combined six-GPU cap. The original 64 GPU-hour budget, immutable plan, saved attempts and results remain intact. The original queue may continue if required work remains after the extension deadline. A first interactive report will be available before 08:30, with incomplete cells and actual seed counts shown explicitly.

The extension adds 56 candidate runs using the same frozen training implementation and cleaned split:

- Sixteen 4096-image clean-subset replications: the eight reduced categorical methods on training seeds 43 and 44, using the original validation-selected subset rates.
- Eight 1024-image runs of 200 epochs, matching the training presentations of the 4096-image, 50-epoch setting. These separate dataset size from optimization-step count; their rates come from the original 1024-image tuning.
- Thirty-two independent-noise-realization runs: the eight reduced methods, fixed/fresh Gaussian noise with requested SD 5 years, and noise seeds 2027/2028. Training seed remains 42. The original noise realization is seed 2026. These differ from the original optional repeats, which change the training seed while preserving noise seed 2026.

The coordination overlay promotes the original optional 4096-image two-digit-policy pilots for CE, MSE and population TailRL. They compare the autoregressive 10×10 head with the 100-bin categorical baselines. Original job identifiers, epoch counts and learning-rate candidate sets remain unchanged. The overlay favors finishing the main matrix, obtaining a scalar comparison on seed 42, and producing small-subset, LR-schedule and noise comparisons progressively rather than waiting for every tuning grid before any confirmation.

Dispatch requires an idle GPU. Another user's workload is never stopped. At the extension deadline, its owned attempts are stopped and retained as incomplete attempts; unstarted jobs are marked deadline-skipped. Every attempt remains charged. GPU scarcity can leave experiments unfinished; the report deadline does not depend on their completion.

Epoch curves and best/final epoch metrics refer to validation data. Historical training saved the test evaluation of the validation-selected checkpoint, not the final-epoch model. The report labels those quantities separately and will not invent final-epoch test metrics. Noise realization and optimization seed are separate report dimensions; runs with different sources, subsets, horizons or noise fields are not silently pooled.
