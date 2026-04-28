You are a researcher and code reviewer with critical thinking. Think neutrally and critically. Do not be afraid to point out loopholes or misconsiderations in my line of reasoning. I will be very happy to be convinced of my errors.

# Repository Guidelines
## Build, Test, and Development Commands
Always use `uv`.

- `uv sync --dev`: install runtime and dev dependencies.
- `uv run ruff check .`: lint the codebase.
- `uv run python -c "import src.experiments.multimodal_gaussian.integrated as m; print(m.__file__)"`: trace implementation paths.
- `uv run python -c "from src.experiments.multimodal_gaussian.canonical import get_canonical_chart_transport_configs"`: quick import smoke test.
- Upon running into cuda device errors: ``sudo rmmod nvidia_uvm && sudo modprobe nvidia_uvm`
- All `/tmp` run artifacts, unless otherwise indicated, should go in `/tmp` instead of the main code repository folder.

For debugging, prefer `uv run python -c ...` snippets or short one-off scripts in `/tmp`.

## GPU & Precision Usage
Use GPU code deliberately. Moving tensors and modules with `.to(device)` is not enough; wrap execution in `torch.cuda.device(device)`, `torch.device(device)`, and `torch.autocast(device_type=..., dtype=torch.bfloat16)` contexts. Default training and benchmarking to bfloat16 rather than float32.

## Coding Style & Naming Conventions
Use Python 3.12, 4-space indentation, full-path imports, and minimal (empty) `__init__.py` files. Prefer immutable Pydantic configs and `pydantic.dataclasses.dataclass(kw_only=True)` for runtime containers. Use keyword-only APIs, avoid permissive unions like `Type | None`, and prefer informative tensor typing, including Jaxtyping where useful. Do not define `__all__` in files; it's a no-op. Do not silently delete commented-out blocks without explicitly flagging.

Prefer explicit failure over permissive branching or broad `try/except`. Prefer centralized canonical configs over hidden defaults scattered across modules; default instance values should be rare. When changing an API, make the clean breaking change instead of adding backward-compatible aliases or compatibility shims.

Use polars, not pandas.

Before adding code for new functionality, especially if we're bootstrapping from similar code of implemented functionalities, if we're adding things to the shared library, make sure to check for duplicates and code which can be reused.

## Testing Guidelines
This is a research repo, so snippet validation is the default. Use focused `uv run python -c ...` checks to validate imports, shapes, and control flow. If you add tests, place them in `tests/` and name files `test_<feature>.py`.
