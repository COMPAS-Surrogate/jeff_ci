# AGENTS.md

This file is a short guide for humans (and coding agents) making changes to this codebase.

## What this repo is
`cosmic_integration` is a Python package for:

- **Rates / cosmic integration** utilities (see `cosmic_integration/ratesSampler/`).
- **Log-likelihood computation** for COMPAS + observations (`cosmic_integration/lnl_computer.py`, `cosmic_integration/observation/`).
- **A GP surrogate for log-likelihood** with active learning / BO-style training (`cosmic_integration/lnl_surrogate/`).
- A small set of **CLI entry scripts** (`cosmic_integration/cli_tools/`).

If you’re adding new features, prefer keeping core logic in the package modules and leaving CLIs as thin wrappers.

## Repository map
- `cosmic_integration/lnl_surrogate/`
  - `lnl_surrogate.py`: main `LnLSurrogate` Likelihood wrapper + training entrypoint
  - `active_learning/`: GP/active-learning loop plumbing
  - `adaptive_robust_scalar.py`: target transform / scaling utilities
  - `run_sampler.py`: sampling utilities using a trained surrogate
- `cosmic_integration/observation/`: observation models and plotting
- `cosmic_integration/ratesSampler/`: binned cosmic integrator + data reduction
- `cosmic_integration/cli_tools/`: command-line scripts (Click)

## Setup (local dev)
Recommended: create a dedicated virtual environment.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
```

Then install the project dependencies.

- If you have a `pyproject.toml` / `requirements.txt` in the repo root, use that.
- If you don’t, document the install step you used in the PR/commit message.

## How to run common tasks
### Train a surrogate (example)
The project includes CLIs under `cosmic_integration/cli_tools/`.

Example (see `cli_tools/train_surrogate.py`):

```bash
python -m cosmic_integration.cli_tools.train_surrogate \
  path/to/COMPAS.h5 \
  --observation-file path/to/observation.json \
  --outdir outdir
```

(Adjust arguments to match your local data files.)

### Run other tools
Other useful scripts live in `cosmic_integration/cli_tools/`, e.g.

- `run_surrogate_workflow.py`
- `diagnose_surrogate_chain.py`
- `run_1d_lnl_check.py`

Prefer `python -m cosmic_integration.cli_tools.<script_name>` so imports resolve consistently.

## Tests (pytest)
This project uses **pytest**.

Run the full test suite:

```bash
pytest
```

Common variants:

```bash
pytest -q
pytest -k <pattern>
pytest tests/<test_file>.py -q
```

### Test expectations
- New features and bug fixes should include **new tests**.
- Tests should be deterministic: set seeds where randomness is used.
- If a test needs large data files (e.g., HDF5 grids), prefer:
  - small synthetic fixtures, or
  - marking as slow/integration (e.g. `@pytest.mark.slow`) and skipping by default.

## Coding conventions
### Style
- Keep functions small and well-named; prefer readability over cleverness.
- Prefer **f-strings** for formatting.
- Add type hints for public APIs (and for any non-trivial internal functions).

### Logging
- Use the `logging` module (not `print`) for library code.
- CLIs may `click.echo(...)` for user-facing output.

### Data / I/O
- Avoid hard-coding paths.
- Treat `.h5` / cache files as potentially large; do not load whole files unless necessary.

## Making changes safely
### When touching the surrogate / scaling code
Files like `lnl_surrogate/adaptive_robust_scalar.py` affect:

- the target transformation (`LnL` → transformed space)
- stability near the optimum
- tail behaviour (soft clipping / lower clip)

If you change anything there:
- add/extend tests that cover the transform,
- include before/after plots or numeric diagnostics in the PR description when possible,
- be explicit about any changed default hyperparameters.

### Backwards compatibility
- Don’t change public function signatures without a clear reason.
- If you must change an API, add a short migration note in the PR description.

## Agent workflow checklist
Before opening a PR (or finishing an automated edit):
1. Run `pytest` locally.
2. Ensure new code is covered by tests.
3. Keep diffs focused: avoid drive-by refactors.
4. Update docstrings/comments where behaviour changed.

