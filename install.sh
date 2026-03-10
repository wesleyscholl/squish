#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════════
#  Squish — One-Command Installer
#
#  Usage (recommended):
#    curl -fsSL https://raw.githubusercontent.com/wesleyscholl/squish/main/install.sh | bash
#
#  Or clone and run locally:
#    git clone https://github.com/wesleyscholl/squish.git
#    bash squish/install.sh
# ═══════════════════════════════════════════════════════════════════════════════
set -euo pipefail

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

  Local LLM inference · 54× faster cold load · No cloud · No API key
  Apple Silicon native · Linux x86_64/arm64 · OpenAI + Ollama drop-in compatible

BANNER

# ── Platform check ────────────────────────────────────────────────────────────
OS="$(uname -s)"
ARCH="$(uname -m)"

if [[ "$OS" == "Darwin" ]]; then
    if [[ "$ARCH" != "arm64" ]]; then
        echo -e "${YELLOW}  ⚠  x86_64 Mac detected — MLX requires Apple Silicon (M1–M5).${RESET}"
        echo "     Install will continue but inference will fall back to CPU."
    fi
    BACKEND="macos"
elif [[ "$OS" == "Linux" ]]; then
    BACKEND="linux"
    echo -e "${CYAN}  ▸ Linux detected — will install torch/bitsandbytes backend.${RESET}"
else
    echo -e "${RED}  ✗  Unsupported OS: ${OS}.  Squish supports macOS and Linux.${RESET}"
    exit 1
fi

echo -e "${CYAN}  ▸ Platform : ${OS} / ${ARCH}${RESET}"

# ── Python ────────────────────────────────────────────────────────────────────
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
    echo -e "${RED}  ✗  Python 3.10+ required but not found.${RESET}"
    echo "     Install with: brew install python@3.12"
    echo "     Then re-run this script."
    exit 1
fi

PYTHON_VERSION="$("$PYTHON" --version 2>&1)"
echo -e "${CYAN}  ▸ Python   : ${PYTHON_VERSION}${RESET}"

# ── Install squish from PyPI (or editable if inside a checkout) ───────────────
echo ""
echo -e "${CYAN}  ▸ Installing squish …${RESET}"

# Detect if we're running from inside a git checkout of squish
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" 2>/dev/null && pwd || pwd)"
if [[ -f "$SCRIPT_DIR/pyproject.toml" ]] && grep -q 'name = "squish"' "$SCRIPT_DIR/pyproject.toml" 2>/dev/null; then
    echo -e "${DIM}    (local checkout detected — installing in editable mode)${RESET}"
    "$PYTHON" -m pip install --quiet -e "$SCRIPT_DIR[${BACKEND}]"
else
    "$PYTHON" -m pip install --quiet --upgrade "squish[${BACKEND}]"
fi

echo -e "${GREEN}  ✓ squish installed${RESET}"

# ── Verify install ────────────────────────────────────────────────────────────
echo ""
echo -e "${CYAN}  ▸ Verifying …${RESET}"

if "$PYTHON" -m squish.cli --help &>/dev/null; then
    echo -e "${GREEN}  ✓ squish CLI is working${RESET}"
else
    echo -e "${YELLOW}  ⚠  CLI check failed — try: ${PYTHON} -m squish.cli --help${RESET}"
fi

# Check that squish binary is on PATH
if command -v squish &>/dev/null; then
    echo -e "${GREEN}  ✓ 'squish' command found on PATH$(command -v squish | xargs printf ': %s')${RESET}"
else
    # Suggest adding pip's local bin to PATH if needed
    LOCAL_BIN="$("$PYTHON" -m site --user-base)/bin"
    echo -e "${YELLOW}  ⚠  'squish' not found on PATH.${RESET}"
    echo -e "${DIM}     Add to PATH: export PATH=\"${LOCAL_BIN}:\$PATH\"${RESET}"
    SHELL_RC="$HOME/.zshrc"
    [[ "${SHELL:-}" == *"bash"* ]] && SHELL_RC="$HOME/.bashrc"
    if ! grep -q "$LOCAL_BIN" "$SHELL_RC" 2>/dev/null; then
        echo "export PATH=\"${LOCAL_BIN}:\$PATH\"" >> "$SHELL_RC"
        echo -e "${CYAN}     Added to ${SHELL_RC} — restart your shell or run: source ${SHELL_RC}${RESET}"
    fi
fi

# Run squish doctor
echo ""
echo -e "${CYAN}  ▸ Running squish doctor …${RESET}"
"$PYTHON" -m squish.cli doctor 2>/dev/null || true

# ── Print next steps ──────────────────────────────────────────────────────────
cat <<DONE

${BOLD}${GREEN}✓ Squish installed successfully!${RESET}

${BOLD}Next steps — run these 3 commands:${RESET}

  ${CYAN}squish catalog${RESET}           ${DIM}# Browse available models${RESET}
  ${CYAN}squish pull qwen3:8b${RESET}     ${DIM}# Download + compress (once, ~5 min for 8B)${RESET}
  ${CYAN}squish run qwen3:8b${RESET}      ${DIM}# Start server + open web UI${RESET}

${BOLD}Then open:${RESET}
  ${CYAN}http://localhost:11435/chat${RESET}    ${DIM}# Web chat UI${RESET}

${BOLD}Or chat in the terminal:${RESET}
  ${CYAN}squish chat qwen3:8b${RESET}

${BOLD}OpenAI drop-in:${RESET}
  ${DIM}export OPENAI_BASE_URL=http://localhost:11435/v1${RESET}
  ${DIM}export OPENAI_API_KEY=squish${RESET}

${BOLD}Ollama drop-in:${RESET}
  ${DIM}export OLLAMA_HOST=http://localhost:11435${RESET}

${BOLD}All models (29 available):${RESET}
  ${CYAN}squish catalog${RESET}           ${DIM}# qwen3, gemma3, deepseek-r1, llama3, phi4, mistral …${RESET}

${BOLD}Get help:${RESET}
  ${DIM}squish --help${RESET}
  ${DIM}https://github.com/wesleyscholl/squish${RESET}

DONE
