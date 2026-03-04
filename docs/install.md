# Installation

Squish runs on **macOS with Apple Silicon (M1–M5)**. Pick the install method that suits you.

---

## Option 1 — Homebrew (recommended)

```bash
brew tap wesleyscholl/squish
brew install squish
```

Homebrew manages dependencies, creates the `squish` binary in your `PATH`, and makes upgrades easy:

```bash
brew upgrade squish
```

---

## Option 2 — pip

```bash
pip install squish
```

Requires Python 3.10+ and installs the `squish` CLI entry point automatically.

For a virtual environment (recommended):

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install squish
```

---

## Option 3 — curl installer

```bash
curl -sSL https://raw.githubusercontent.com/wesleyscholl/squish/main/install.sh | bash
```

!!! warning "Review before running"
    Piping `curl` to `bash` executes arbitrary code. Review [install.sh](https://github.com/wesleyscholl/squish/blob/main/install.sh) first.

---

## Option 4 — From source

```bash
git clone https://github.com/wesleyscholl/squish.git
cd squish
pip install -e ".[dev]"
```

This also installs development extras (`pytest`, `ruff`, `mypy`, `httpx`).

---

## Verify the installation

```bash
squish --version
squish --help
```

You should see version info and the command list.

---

## Requirements

| Requirement | Version |
|---|---|
| macOS | 13 Ventura or later |
| Apple Silicon | M1 / M2 / M3 / M4 / M5 |
| Python | 3.10+ |
| MLX | ≥ 0.18 |
| Free RAM | ≥ 8 GB (16 GB+ recommended) |
| Free disk | ≥ 5 GB per model |

---

## Uninstall

=== "Homebrew"
    ```bash
    brew uninstall squish
    brew untap wesleyscholl/squish
    ```

=== "pip"
    ```bash
    pip uninstall squish
    ```

=== "Source"
    ```bash
    pip uninstall squish
    rm -rf /path/to/squish
    ```
