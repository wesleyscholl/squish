#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════════
#  Squish — One-Command Installer
#  curl -fsSL https://raw.githubusercontent.com/wesleyscholl/squish/main/install.sh | bash
# ═══════════════════════════════════════════════════════════════════════════════
set -euo pipefail

SQUISH_REPO="$HOME/.squish"
SQUISH_HOME="$HOME/.squish"
POC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

GREEN="\033[32m"
CYAN="\033[36m"
RED="\033[31m"
YELLOW="\033[33m"
BOLD="\033[1m"
DIM="\033[2m"
RESET="\033[0m"

# ── Banner ────────────────────────────────────────────────────────────────────
cat <<'BANNER'

  ███████╗ ██████╗ ██╗   ██╗██╗███████╗██╗  ██╗
  ██╔════╝██╔═══██╗██║   ██║██║██╔════╝██║  ██║
  ███████╗██║   ██║██║   ██║██║███████╗███████║
  ╚════██║██║▄▄ ██║██║   ██║██║╚════██║██╔══██║
  ███████║╚██████╔╝╚██████╔╝██║███████║██║  ██║
  ╚══════╝ ╚══▀▀═╝  ╚═════╝ ╚═╝╚══════╝╚═╝  ╚═╝

  Local LLM inference — No cloud, no API keys, no rate limits.
  54× faster cold load than standard safetensors loading.

BANNER

# ── Detect platform ───────────────────────────────────────────────────────────
OS="$(uname -s)"
ARCH="$(uname -m)"

if [[ "$OS" != "Darwin" ]]; then
    echo -e "${RED}Squish currently requires macOS (Apple Silicon).${RESET}"
    echo "Linux support is coming. Track progress: github.com/wesleyscholl/squish"
    exit 1
fi

if [[ "$ARCH" != "arm64" ]]; then
    echo -e "${YELLOW}Warning: x86_64 Mac detected — performance will be limited (no MLX).${RESET}"
fi

echo -e "${CYAN}▸ Platform: ${OS} / ${ARCH}${RESET}"

# ── Python version ────────────────────────────────────────────────────────────
PYTHON=""
for py in python3.12 python3.11 python3.10 python3; do
    if command -v "$py" &>/dev/null; then
        VER="$("$py" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
        MAJOR="${VER%%.*}"
        MINOR="${VER##*.}"
        if [[ "$MAJOR" -ge 3 && "$MINOR" -ge 10 ]]; then
            PYTHON="$py"
            break
        fi
    fi
done

if [[ -z "$PYTHON" ]]; then
    echo -e "${RED}Python 3.10+ required but not found.${RESET}"
    echo "Install with: brew install python@3.12"
    exit 1
fi
echo -e "${CYAN}▸ Python:   $($PYTHON --version)${RESET}"

# ── pip dependencies ──────────────────────────────────────────────────────────
echo -e "\n${CYAN}▸ Installing Python dependencies…${RESET}"
"$PYTHON" -m pip install --quiet --upgrade \
    mlx \
    mlx-lm \
    fastapi \
    'uvicorn[standard]' \
    sse-starlette \
    zstandard \
    huggingface_hub \
    numpy \
    safetensors \
    transformers

echo -e "${GREEN}  ✓ Dependencies installed${RESET}"

# ── Squish home dir ───────────────────────────────────────────────────────────
mkdir -p "$SQUISH_HOME/models"
echo -e "${CYAN}▸ Squish home: ${SQUISH_HOME}${RESET}"

# ── squish binary ─────────────────────────────────────────────────────────────
CLI_SCRIPT="$POC_DIR/cli.py"

if [[ ! -f "$CLI_SCRIPT" ]]; then
    echo -e "${RED}cli.py not found at $CLI_SCRIPT${RESET}"
    echo "Are you running install.sh from the squish poc directory?"
    exit 1
fi

# Try /usr/local/bin, then ~/.local/bin
for bin_dir in /usr/local/bin "$HOME/.local/bin"; do
    if [[ -w "$bin_dir" ]] || mkdir -p "$bin_dir" 2>/dev/null; then
        INSTALL_DIR="$bin_dir"
        break
    fi
done

SQUISH_BIN="$INSTALL_DIR/squish"
cat > "$SQUISH_BIN" <<WRAPPER
#!/usr/bin/env bash
exec "$PYTHON" "$CLI_SCRIPT" "\$@"
WRAPPER
chmod +x "$SQUISH_BIN"
echo -e "${GREEN}  ✓ Installed $(squish --version 2>/dev/null || echo squish) → $SQUISH_BIN${RESET}"

# Ensure bin dir is in PATH
SHELL_RC="$HOME/.zshrc"
[[ "$SHELL" == *"bash"* ]] && SHELL_RC="$HOME/.bashrc"

if ! grep -q "squish" "$SHELL_RC" 2>/dev/null; then
    if [[ "$INSTALL_DIR" == "$HOME/.local/bin" ]]; then
        echo 'export PATH="$HOME/.local/bin:$PATH"' >> "$SHELL_RC"
        echo -e "${CYAN}  Added ~/.local/bin to PATH in $SHELL_RC${RESET}"
    fi
fi

# ── Quick smoke test ──────────────────────────────────────────────────────────
echo -e "\n${CYAN}▸ Verifying install…${RESET}"
if "$PYTHON" -c "import mlx.core, mlx_lm, fastapi, uvicorn" 2>/dev/null; then
    echo -e "${GREEN}  ✓ Core imports OK${RESET}"
else
    echo -e "${YELLOW}  ⚠ Some imports failed — check your Python environment${RESET}"
fi

# ── Print next steps ──────────────────────────────────────────────────────────
cat <<DONE

${BOLD}${GREEN}✓ Squish installed!${RESET}

${BOLD}Get started in 3 commands:${RESET}

  # 1. Download + compress a model (one time, ~5-10 min for 7B):
  ${CYAN}squish pull Qwen/Qwen2.5-7B-Instruct${RESET}

  # 2. Ask a question:
  ${CYAN}squish run "What is attention in transformers?"${RESET}

  # 3. Interactive chat:
  ${CYAN}squish chat${RESET}

${BOLD}For your git commit workflow:${RESET}
  ${CYAN}cd any-git-repo && squish git${RESET}

${BOLD}Drop-in OpenAI replacement:${RESET}
  ${CYAN}squish serve --port 8000${RESET}
  ${DIM}# Then set OPENAI_BASE_URL=http://localhost:8000/v1 in any client${RESET}

${BOLD}Benchmarks (reproduces the 54× numbers):${RESET}
  ${CYAN}squish bench${RESET}

DONE
