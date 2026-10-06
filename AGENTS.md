# Repository Guidelines

## Project Structure & Module Organization

- `vram_mcp/` contains the Python package. `server.py` exposes MCP tools; `core.py` coordinates VRAM decisions. Keep MCP imports confined to `server.py`.
- `gpu.py`, `nvml.py`, `procinfo.py`, `ollama.py`, and `ollama_correlate.py` collect and correlate GPU, process, and model data. `claims.py` manages shared claims/reservations; `audit.py` records events and samples; `_util.py` holds shared helpers.
- `tests/test_*.py` mirrors package modules. Design specifications and implementation plans live under `docs/superpowers/`. Packaging is configured in `pyproject.toml`; CI lives in `.github/workflows/ci.yml`.

## Build, Test, and Development Commands

Use Python 3.10+ in a virtual environment.

- `python -m pip install -e ".[dev]"` installs the editable package and pytest.
- `python -m pytest -q` runs the full suite, matching CI.
- `python -m pytest tests/test_core.py -q` runs a focused module suite.
- `python -m pip wheel --no-deps . -w dist` builds a wheel using Hatchling.
- `vram-mcp` starts the MCP stdio server, normally launched by an MCP client. Local model operations require a reachable Ollama instance.

## Coding Style & Naming Conventions

Follow existing Python style: four-space indentation, `snake_case` functions/modules, `PascalCase` classes, and `UPPER_CASE` constants. Use type hints and concise docstrings for contracts and non-obvious behavior. No formatter or linter is configured; avoid unrelated formatting changes.

Keep core logic injectable and testable. MCP tools are asynchronous; dispatch blocking HTTP, subprocess, NVML, and file operations through `anyio.to_thread.run_sync`. Preserve explicit unknown/null readings when telemetry is unavailable.

## Testing Guidelines

Use pytest with descriptive `test_<behavior>` names. Mock hardware, subprocesses, and HTTP; tests should not require a GPU or Ollama daemon. Use `tmp_path`, `monkeypatch`, and injected clocks for ledger/audit isolation; never touch live `~/.cache/vram-mcp/` files.

Add regression tests for changed behavior, including unavailable telemetry and claim/reservation protection. No coverage threshold is configured. CI tests Python 3.10-3.14 on Windows and Linux; server tests skip when `mcp` is unavailable.

## Commit & Pull Request Guidelines

Follow Git history's Conventional Commit style, such as `fix(core): correct spill attribution` or `test(server): isolate reservation state`. Keep commits focused.

PRs should describe the problem, resulting behavior, and validation commands/results. Link relevant issues and update `README.md` when tools or configuration change. Keep personal session notes in ignored `research/`, outside committed project documentation.
