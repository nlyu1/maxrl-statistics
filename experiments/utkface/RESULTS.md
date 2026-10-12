# UTKFace execution status

The implementation is prepared on branch `codex/utkface-population-study`. No training result or runtime test is claimed yet.

Executed locally within the allowed boundary:

- `uvx ruff check src/experiments/utkface tests/test_utkface_objectives.py tests/test_utkface_data_training.py tests/test_utkface_analysis.py`: passed.
- `uvx ruff format --check` on the same paths: passed.
- Static review against the pinned task code and TailRL estimators: completed. Runtime identity checks are included in the tests and remain pending.

On 2026-10-12, `ssh cmu-L40-live` failed with `Connection refused`, and the tunnel-process check specified by `server.md` found no matching process. `server.md` requires the user to start that tunnel and prohibits local Python, tests, and experiments. Runtime validation and measured comparisons are pending server access.

The prepared workflow covers objective identities/gradients, original stratified splits, exact checkpoint recovery, validation-only hyperparameter selection, two-rank global-gradient comparison, uneven distributed evaluation, bf16 ResNet forward/backward for every method, and an eight-arm, three-seed training study.
