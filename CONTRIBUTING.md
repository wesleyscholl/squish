# Contributing to Squish

Thank you for your interest in contributing!  Squish is an Apple Silicon–only
local LLM inference library.  Contributions that improve load-time performance,
accuracy fidelity, or API compatibility are especially welcome.

---

## Setting Up the Development Environment

```bash
# 1. Clone and enter the repo
git clone https://github.com/wesleyscholl/squish.git
cd squish

# 2. Create a virtual environment (Python 3.12 recommended)
python3 -m venv .venv
source .venv/bin/activate

# 3. Install in editable mode with dev extras
pip install -e ".[dev]"

# 4. (Optional) Build the Rust INT8 quantiser extension
cd squish_quant_rs
pip install maturin
maturin develop --release
cd ..
```

---

## Running Tests

```bash
pytest tests/ -v
```

The test suite does **not** require model weights — it uses mocked tensors and
stub loaders.  Tests that do require weights are skipped automatically when
`~/models/` is absent.

---

## Linting

Squish uses [ruff](https://docs.astral.sh/ruff/) with a line length of 100:

```bash
ruff check squish/ tests/
ruff format squish/ tests/
```

CI will fail if `ruff check` reports any errors.

---

## What Makes a Good PR

- **Focused** — one logical change per PR
- **No hardcoded paths** — use `Path.home()` or environment variables
- **Tests pass** — `pytest tests/` green before opening the PR
- **No model weights committed** — `models/` is gitignored for a reason
- **Performance-sensitive changes** include a before/after `squish bench` run

---

## Reporting Bugs

Use [GitHub Issues](https://github.com/wesleyscholl/squish/issues) with the
**Bug report** template.  Please include your chip generation (M1/M2/M3/M4),
macOS version, and the model you were loading.

---

## Branch Strategy

| Branch | Purpose |
|---|---|
| `main` | Stable, tagged releases |
| `research` | Eval scripts, benchmark notebooks, paper tooling |
| Feature branches | `feat/<short-name>`, opened as PRs against `main` |
