#!/usr/bin/env python3
"""
squish/cli.py

Entry point for the Squish local-inference CLI.

Sub-commands
───────────
  squish pull   MODEL               Download + compress a model
  squish catalog                    Browse available models
  squish run    [MODEL] [OPTIONS]   Start the inference server
  squish serve  [MODEL] [OPTIONS]   Alias for `squish run`
  squish chat   [MODEL] [OPTIONS]   Interactive terminal chat (no browser needed)
  squish models                     List local models (auto-discovers ~/.squish/models/)
  squish info                       System info: Metal, RAM, disk
  squish bench  [MODEL] [OPTIONS]   Quick throughput/latency benchmark
  squish doctor                     Check all dependencies
  squish daemon start|stop|status   Manage background server
  squish compress MODEL             Compress a model to npy-dir format

MODEL shorthand resolves via the Squish catalog:
  qwen3:8b, gemma3:4b, deepseek-r1:7b, llama3.2:3b, phi4:14b …
  Legacy aliases still work: 7b, 14b, 1.5b, 32b, 72b
  Any path starting with ~ or / → used as-is

Usage:
    python3 -m squish.cli pull qwen3:8b
    python3 -m squish.cli run 7b
    python3 -m squish.cli chat 7b
    python3 -m squish.cli catalog

After `pip install -e .`:
    squish pull qwen3:8b
    squish run qwen3:8b
    squish chat qwen3:8b
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

# When running as `python3 squish/cli.py` (not via `-m`), the repo root is NOT
# on sys.path, which breaks `from squish.X import ...` inside subcommands like
# compress (AWQ path) and convert.  Inject the repo root so the package is
# always importable regardless of invocation style.
_cli_dir = os.path.dirname(os.path.abspath(__file__))   # …/squish/squish
_repo_root = os.path.dirname(_cli_dir)                   # …/squish
if _repo_root not in sys.path:  # pragma: no cover
    sys.path.insert(0, _repo_root)
del _cli_dir, _repo_root

try:
    from squish.catalog import (
        list_catalog,
    )
    from squish.catalog import (
        pull as _catalog_pull,
    )
    from squish.catalog import (
        resolve as _catalog_resolve,
    )
    _CATALOG_AVAILABLE = True
except Exception:  # pragma: no cover
    _CATALOG_AVAILABLE = False


# ── Model registry ───────────────────────────────────────────────────────────

# Resolve models directory: SQUISH_MODELS_DIR env var → ~/.squish/models → <repo>/models → ~/models (legacy)
def _resolve_models_dir() -> Path:
    env_override = os.environ.get("SQUISH_MODELS_DIR", "").strip()
    if env_override:
        return Path(env_override).expanduser()
    # Check ~/.squish/models (canonical install location)
    primary = Path.home() / ".squish" / "models"  # pragma: no cover
    if primary.exists():  # pragma: no cover
        return primary
    # Check <squish repo root>/models/ — works when running directly from the repo
    repo_models = Path(__file__).resolve().parent.parent / "models"  # pragma: no cover
    if repo_models.exists():  # pragma: no cover
        return repo_models
    # Check ~/models (legacy location)
    legacy = Path.home() / "models"  # pragma: no cover
    if legacy.exists():  # pragma: no cover
        return legacy
    return primary  # pragma: no cover  # default even if absent — gives a consistent error path

_MODELS_DIR = _resolve_models_dir()

# Legacy shorthand → directory name (kept for backward compatibility).
# New models should be added to squish/catalog.py instead.
_MODEL_SHORTHAND = {
    # Qwen 2.5
    "1.5b":  "Qwen2.5-1.5B-Instruct-bf16",
    "7b":    "Qwen2.5-7B-Instruct-bf16",
    "14b":   "Qwen2.5-14B-Instruct-bf16",
    "32b":   "Qwen2.5-32B-Instruct-bf16",
    "72b":   "Qwen2.5-72B-Instruct-bf16",
    # Qwen 3
    "qwen3:0.6b":   "Qwen3-0.6B-bf16",
    "qwen3:1.7b":   "Qwen3-1.7B-bf16",
    "qwen3:4b":     "Qwen3-4B-bf16",
    "qwen3:8b":     "Qwen3-8B-bf16",
    "qwen3:14b":    "Qwen3-14B-bf16",
    "qwen3:30b-a3b":"Qwen3-30B-A3B-bf16",
    "qwen3:32b":    "Qwen3-32B-bf16",
    # Llama 3.x
    "llama3.2:1b":  "Llama-3.2-1B-Instruct-bf16",
    "llama3.2:3b":  "Llama-3.2-3B-Instruct-bf16",
    "llama3.1:8b":  "Meta-Llama-3.1-8B-Instruct-bf16",
    # Gemma 3
    "gemma3:1b":    "gemma-3-1b-it-bf16",
    "gemma3:4b":    "gemma-3-4b-it-bf16",
    "gemma3:12b":   "gemma-3-12b-it-bf16",
    "gemma3:27b":   "gemma-3-27b-it-bf16",
    # DeepSeek-R1
    "deepseek-r1:7b":  "DeepSeek-R1-Distill-Qwen-7B-bf16",
    "deepseek-r1:14b": "DeepSeek-R1-Distill-Qwen-14B-bf16",
    "deepseek-r1:32b": "DeepSeek-R1-Distill-Qwen-32B-bf16",
    "r1:7b":           "DeepSeek-R1-Distill-Qwen-7B-bf16",
    "r1:14b":          "DeepSeek-R1-Distill-Qwen-14B-bf16",
    # Phi-4
    "phi4:14b":     "phi-4-bf16",
    # Mistral
    "mistral:7b":   "Mistral-7B-Instruct-v0.3-bf16",
    # SmolLM2
    "smollm2:135m": "SmolLM2-135M-Instruct-bf16",
    "smollm2:360m": "SmolLM2-360M-Instruct-bf16",
    "smollm2:1.7b": "SmolLM2-1.7B-Instruct-bf16",
}

# Compressed dir naming convention
_COMPRESSED_SUFFIX = "-compressed"

# Default server port
_DEFAULT_PORT = 11435


def _resolve_model(name: str | None) -> tuple[Path, Path]:  # pragma: no cover
    """
    Resolve MODEL shorthand / path to (model_dir, compressed_dir).
    Raises SystemExit if the path doesn't exist.
    """
    if name is None:
        # Auto-pick: prefer 7B if available, else first available
        for shorthand in ("7b", "14b", "1.5b"):
            candidate = _MODELS_DIR / _MODEL_SHORTHAND[shorthand]
            if candidate.exists():
                name = shorthand
                break
        if name is None:
            _die("No model specified and no default found in ~/models/\n"
                 "Usage: squish run 7b  or  squish run ~/models/my-model")

    if name in _MODEL_SHORTHAND:
        model_dir = _MODELS_DIR / _MODEL_SHORTHAND[name]
    elif _CATALOG_AVAILABLE:
        # Try the dynamic catalog (handles qwen3:8b, gemma3:4b, etc.)
        entry = _catalog_resolve(name)
        if entry is not None:
            model_dir = _MODELS_DIR / entry.dir_name
        else:
            model_dir = Path(name).expanduser()
    else:
        model_dir = Path(name).expanduser()

    if not model_dir.exists():
        hint = name if (name and "/" not in str(name)) else "qwen3:8b"
        _die(
            f"Model directory not found: {model_dir}\n"
            f"  Run:  squish pull {hint}  to download it.\n"
            f"  Browse available models: squish catalog"
        )

    compressed_dir = Path(str(model_dir) + _COMPRESSED_SUFFIX)
    if not compressed_dir.exists():
        # Try squish_4bit subdir (mlx_lm native 4-bit)
        squish4bit = model_dir.parent / (model_dir.name.replace("-bf16", "") + "-4bit")
        if squish4bit.exists():
            compressed_dir = squish4bit
        else:
            print(f"  ⚠  No compressed dir found at {compressed_dir}")
            print(f"     To compress: python3 -m squish.convert --model-dir {model_dir} --output {compressed_dir}")
            print("     Starting with uncompressed model (slower load)…")
            compressed_dir = model_dir

    return model_dir, compressed_dir


def _die(msg: str) -> None:
    print(f"\n  ✗  {msg}\n", file=sys.stderr)
    sys.exit(1)


def _box(lines: list[str]) -> None:
    """Print a simple box around lines."""
    width = max(len(ln) for ln in lines) + 4
    print("┌" + "─" * width + "┐")
    for ln in lines:
        print(f"│  {ln:<{width-2}}│")
    print("└" + "─" * width + "┘")


# ── squish models ─────────────────────────────────────────────────────────────

def cmd_models(args):
    """List available local models."""
    print()
    print(f"  Local models in {_MODELS_DIR}:")
    print()
    if not _MODELS_DIR.exists():
        print("  (directory not found)")
        return

    rows = []
    for d in sorted(_MODELS_DIR.iterdir()):
        if not d.is_dir():
            continue
        if d.name.startswith("."):
            continue
        compressed = Path(str(d) + _COMPRESSED_SUFFIX)
        comp_str = "✓ compressed" if compressed.exists() else "  (raw only)"
        # estimate disk size
        try:
            total = sum(f.stat().st_size for f in d.rglob("*") if f.is_file())
            size_str = f"{total / 1e9:.1f} GB"
        except Exception:
            size_str = "?"
        rows.append((d.name, size_str, comp_str))

    if not rows:
        print("  No model directories found.")
        print("  Download a model with: squish pull qwen3:8b")
        print("  Browse all models    : squish catalog")
        return

    # Column widths
    w0 = max(len(r[0]) for r in rows) + 2
    print(f"  {'Model':<{w0}} {'Disk':>8}  {'Compressed'}")
    print(f"  {'─'*w0} {'─'*8}  {'─'*14}")
    for name, size, comp in rows:
        print(f"  {name:<{w0}} {size:>8}  {comp}")

    print()
    print("  Legacy aliases : 1.5b, 7b, 14b, 32b, 72b")
    print("  Catalog IDs    : qwen3:8b, gemma3:4b, deepseek-r1:7b, llama3.2:3b …")
    print("  Browse catalog : squish catalog")
    print("  Download model : squish pull qwen3:8b")
    print()


# ── squish rm ────────────────────────────────────────────────────────────────

def cmd_rm(args):  # pragma: no cover
    """Remove a local model (raw weights and/or compressed dir)."""
    import shutil

    name = args.model

    # Resolve directories without requiring them to exist
    model_dir: Path | None = None
    compressed_dir: Path | None = None

    # Try catalog first (so short names like qwen3:8b work)
    try:
        from squish.catalog import list_catalog
        entries = {e.id: e for e in list_catalog()}
        if name in entries:
            model_dir       = _MODELS_DIR / entries[name].dir_name
            compressed_dir  = Path(str(model_dir) + _COMPRESSED_SUFFIX)
        elif name in _MODEL_SHORTHAND:
            model_dir       = _MODELS_DIR / _MODEL_SHORTHAND[name]
            compressed_dir  = Path(str(model_dir) + _COMPRESSED_SUFFIX)
        else:
            p = Path(name).expanduser()
            model_dir       = p if p.is_absolute() else _MODELS_DIR / name
            compressed_dir  = Path(str(model_dir) + _COMPRESSED_SUFFIX)
    except Exception:
        p = Path(name).expanduser()
        model_dir       = p if p.is_absolute() else _MODELS_DIR / name
        compressed_dir  = Path(str(model_dir) + _COMPRESSED_SUFFIX)

    has_raw  = model_dir.exists()
    has_comp = compressed_dir.exists() and compressed_dir != model_dir

    if not has_raw and not has_comp:
        _die(f"No local files found for model '{name}'.\n"
             f"  Expected raw dir : {model_dir}\n"
             f"  Expected comp dir: {compressed_dir}")

    # Build list of what will be removed
    targets: list[tuple[str, Path]] = []
    if has_raw and (args.compressed_only is False or not args.compressed_only):
        targets.append(("raw weights", model_dir))
    if has_comp and not args.raw_only:
        targets.append(("compressed weights", compressed_dir))

    if not targets:
        print("Nothing to remove (flags excluded all targets).")
        return

    print()
    print(f"  Will remove the following directories for '{name}':")
    total_bytes = 0
    for label, path in targets:
        try:
            sz = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
        except Exception:
            sz = 0
        total_bytes += sz
        print(f"    [{label}]  {path}  ({sz / 1e9:.2f} GB)")
    print(f"  Total: {total_bytes / 1e9:.2f} GB will be freed")
    print()

    if args.dry_run:
        print("  --dry-run: no files removed.")
        return

    # Confirm unless --yes
    if not args.yes:
        ans = input("  Type 'yes' to confirm deletion: ").strip().lower()
        if ans != "yes":
            print("  Aborted.")
            return

    for label, path in targets:
        print(f"  Removing {label}: {path} …", end=" ", flush=True)
        try:
            shutil.rmtree(path)
            print("done.")
        except Exception as exc:
            print(f"ERROR: {exc}")

    print()
    print("  Done. Run 'squish models' to verify.")
    print()


# ── squish search ─────────────────────────────────────────────────────────────

def cmd_search(args):
    """Search the catalog for models matching a query string."""
    from squish.catalog import search

    hits = search(args.query)

    if not hits:
        print(f"  No catalog entries match '{args.query}'.")
        return

    print()
    print(f"  Catalog search results for '{args.query}':")
    print()
    w_id   = max(len(e.id) for e in hits) + 2
    w_para = max(len(str(getattr(e, 'params', ''))) for e in hits) + 2
    print(f"  {'ID':<{w_id}} {'Params':>{w_para}}  Tags")
    print(f"  {'─'*w_id} {'─'*max(w_para,6)}  {'─'*24}")
    for e in hits:
        tags_str = ", ".join(getattr(e, "tags", [])) or "—"
        params   = str(getattr(e, "params", "—"))
        print(f"  {e.id:<{w_id}} {params:>{w_para}}  {tags_str}")
    print()
    print("  Pull a model: squish pull <id>")
    print()


# ── squish info ───────────────────────────────────────────────────────────────

def cmd_info(args):  # pragma: no cover
    """Print system info relevant to local inference."""
    import platform
    import subprocess as sp

    print()
    _box(["Squish — System Info"])
    print()

    # macOS / chip info
    print(f"  OS            : {platform.system()} {platform.release()}")
    try:
        chip = sp.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            text=True, stderr=sp.DEVNULL).strip()
        print(f"  Chip          : {chip}")
    except Exception:
        pass

    # Unified memory
    try:
        mem_bytes = int(sp.check_output(
            ["sysctl", "-n", "hw.memsize"], text=True).strip())
        mem_gb = mem_bytes / 1e9
        print(f"  Unified RAM   : {mem_gb:.0f} GB")
        print(f"  Metal budget  : {mem_gb * 0.90:.1f} GB  (90% of RAM — Phase 0.1)")
    except Exception:
        pass

    # Disk space
    m = _MODELS_DIR
    if m.exists():
        stat = shutil.disk_usage(m)
        print(f"  Models dir    : {m}")
        print(f"  Disk free     : {stat.free / 1e9:.1f} GB")
        print(f"  Disk used     : {stat.used / 1e9:.1f} GB  (total: {stat.total / 1e9:.0f} GB)")

    # Python / MLX
    try:
        import mlx.core as mx
        print(f"  MLX           : v{mx.__version__}  (Metal backend active)")
    except Exception:
        print("  MLX           : not installed")
    print(f"  Python        : {sys.version.split()[0]}")

    # Models available
    if m.exists():
        n_models = sum(1 for d in m.iterdir() if d.is_dir() and not d.name.startswith("."))
        n_comp   = sum(1 for d in m.iterdir() if d.is_dir() and (Path(str(d)+_COMPRESSED_SUFFIX).exists()))
        print(f"  Local models  : {n_models} model(s),  {n_comp} compressed")

    # Server status
    import socket
    s = socket.socket()
    s.settimeout(0.5)
    try:
        s.connect(("127.0.0.1", _DEFAULT_PORT))
        print(f"  Server        : ✓ running on :{_DEFAULT_PORT}")
    except Exception:
        print("  Server        : not running  (start with: squish run 7b)")
    finally:
        s.close()
    print()


# ── squish run ────────────────────────────────────────────────────────────────

def cmd_run(args):  # pragma: no cover
    """Start the Squish inference server."""
    model_dir, compressed_dir = _resolve_model(args.model)

    server_script = Path(__file__).resolve().parent / "server.py"
    if not server_script.exists():
        _die(f"server.py not found at {server_script}")

    port     = args.port or _DEFAULT_PORT
    host     = args.host or "127.0.0.1"
    api_key  = args.api_key or "squish"

    # Warn when binding to a non-loopback address — server will be reachable on LAN
    if host not in ("127.0.0.1", "localhost", "::1"):
        import warnings
        warnings.warn(
            f"\n  ⚠  CORS is wide-open and the server will be reachable from your LAN "
            f"at http://{host}:{port}/v1.\n"
            f"  Set a strong SQUISH_API_KEY env var or bind to 127.0.0.1 if unintended.",
            stacklevel=0,
        )

    print()
    _box([
        "  Squish — Local Inference Server  ",
        f"  Model     : {model_dir.name}",
        f"  Endpoint  : http://{host}:{port}/v1",
        f"  Web UI    : http://{host}:{port}/chat",
        f"  API key   : {api_key}",
        "",
        "  OpenAI drop-in:",
        f"    OPENAI_BASE_URL=http://{host}:{port}/v1",
        f"    OPENAI_API_KEY={api_key}",
        "",
        "  Ollama drop-in:",
        f"    OLLAMA_HOST=http://{host}:{port}",
        "",
        "  Press Ctrl+C to stop",
    ])
    print()

    cmd = [
        sys.executable, str(server_script),
        "--model-dir",      str(model_dir),
        "--compressed-dir", str(compressed_dir),
        "--port",           str(port),
        "--host",           host,
        # API key is passed via env var to avoid exposure in `ps aux`
    ]
    # Inject into the environ that execv will inherit
    os.environ.setdefault("SQUISH_API_KEY", api_key)
    if args.draft_model:
        cmd += ["--draft-model", args.draft_model]
    if args.batch_scheduler:
        cmd += ["--batch-scheduler", "--batch-size", str(args.batch_size)]
    if args.kv_cache_mode and args.kv_cache_mode != "fp16":
        cmd += ["--kv-cache-mode", args.kv_cache_mode]
    if getattr(args, "log_level", "warning") != "warning":
        cmd += ["--log-level", args.log_level]

    try:
        os.execv(sys.executable, cmd)  # replace this process — clean signals
    except Exception as e:
        _die(f"Failed to start server: {e}")


# ── squish chat ───────────────────────────────────────────────────────────────

def cmd_chat(args):  # pragma: no cover
    """
    Interactive terminal chat against a running (or auto-started) server.

    If no server is running, starts one in a subprocess first.
    Uses Server-Sent Events streaming for real-time token display.
    """
    import socket
    import urllib.error
    import urllib.request

    port    = args.port or _DEFAULT_PORT
    host    = args.host or "127.0.0.1"
    api_key = args.api_key or "squish"
    base_url = f"http://{host}:{port}/v1"

    # ── Auto-start server if not running ─────────────────────────────────────
    _server_proc = None

    def _server_up() -> bool:
        s = socket.socket()
        s.settimeout(1.0)
        try:
            s.connect((host, port))
            s.close()
            return True
        except Exception:
            return False

    if not _server_up():
        model_dir, compressed_dir = _resolve_model(args.model)
        print(f"  Starting server for {model_dir.name} …")
        server_script = Path(__file__).resolve().parent / "server.py"
        _server_proc = subprocess.Popen([
            sys.executable, str(server_script),
            "--model-dir",      str(model_dir),
            "--compressed-dir", str(compressed_dir),
            "--port",           str(port),
            "--host",           host,
            # API key via env var — keeps it out of `ps aux`
        ], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
           env={**os.environ, "SQUISH_API_KEY": api_key})

        # Wait (up to 30s) for server to come up
        for _ in range(60):
            time.sleep(0.5)
            if _server_up():
                break
        else:
            _server_proc.terminate()
            _die("Server did not start within 30s. Check logs above.")

        print(f"  ✓ Server ready on {base_url}")

    # ── Chat loop ─────────────────────────────────────────────────────────────
    messages = []
    model    = args.chat_model or "squish"

    SYSTEM = (args.system or
              "You are a knowledgeable, concise assistant running entirely locally on "
              "Apple Silicon. You have full privacy — nothing leaves this machine.")
    if SYSTEM:
        messages.append({"role": "system", "content": SYSTEM})

    print()
    print("  Squish Chat  (type /quit to exit, /clear to reset, /system to change system prompt)")
    print("  ─────────────────────────────────────────────────────────────────")
    print()

    def _stream_chat(msgs: list) -> str:
        """Send messages, stream tokens to stdout, return full response."""
        payload = json.dumps({
            "model":       model,
            "messages":    msgs,
            "max_tokens":  args.max_tokens,
            "temperature": args.temperature,
            "stream":      True,
        }).encode()
        req = urllib.request.Request(
            f"{base_url}/chat/completions",
            data    = payload,
            headers = {
                "Content-Type":  "application/json",
                "Authorization": f"Bearer {api_key}",
            },
        )
        full = ""
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                for raw_line in resp:
                    line = raw_line.decode("utf-8").strip()
                    if not line.startswith("data: "):
                        continue
                    payload_str = line[6:]
                    if payload_str == "[DONE]":
                        break
                    try:
                        chunk = json.loads(payload_str)
                        delta = chunk["choices"][0]["delta"].get("content", "")
                        if delta:
                            print(delta, end="", flush=True)
                            full += delta
                    except Exception:
                        pass
        except urllib.error.URLError as e:
            print(f"\n  ✗ Request failed: {e}")
        print()
        return full

    try:
        while True:
            try:
                user_input = input("  You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n  Goodbye.")
                break

            if not user_input:
                continue
            if user_input.lower() in ("/quit", "/exit", "quit", "exit"):
                print("  Goodbye.")
                break
            if user_input.lower() == "/clear":
                messages = [m for m in messages if m["role"] == "system"]
                print("  ✓ Conversation cleared.")
                continue
            if user_input.lower().startswith("/system "):
                new_sys = user_input[8:].strip()
                messages = [m for m in messages if m["role"] != "system"]
                messages.insert(0, {"role": "system", "content": new_sys})
                print("  ✓ System prompt updated.")
                continue
            if user_input.lower() == "/help":
                print("  Commands: /quit  /clear  /system <text>  /help")
                continue

            messages.append({"role": "user", "content": user_input})
            print("\n  Assistant: ", end="", flush=True)
            reply = _stream_chat(messages)
            if reply:
                messages.append({"role": "assistant", "content": reply})
            # Rolling context window — keeps non-system messages within limit
            max_hist = getattr(args, "max_history", 40)
            non_sys = [m for m in messages if m["role"] != "system"]
            if len(non_sys) > max_hist:
                system_msgs = [m for m in messages if m["role"] == "system"]
                messages = system_msgs + non_sys[-max_hist:]
            print()

    finally:
        if _server_proc is not None:
            _server_proc.terminate()


# ── squish bench ──────────────────────────────────────────────────────────────

def cmd_bench(args):  # pragma: no cover
    """Quick throughput benchmark against a running server."""
    import socket
    import urllib.request

    port    = args.port or _DEFAULT_PORT
    host    = args.host or "127.0.0.1"
    api_key = args.api_key or "squish"
    base_url = f"http://{host}:{port}/v1"

    # Check server up
    s = socket.socket()
    s.settimeout(1.0)
    try:
        s.connect((host, port))
    except Exception:
        _die(f"No server running on {host}:{port}. Start with: squish run 7b")
    finally:
        s.close()

    # ── Optional cold-start timing ─────────────────────────────────────────
    if getattr(args, "cold_start", False):
        print("  \u23f1  Cold-start: timing model load via /v1/models (may take up to 2 min) \u2026")
        sys.stdout.flush()
        t_cs = time.perf_counter()
        try:
            req_cs = urllib.request.Request(
                f"{base_url}/models",
                headers={"Authorization": f"Bearer {api_key}"},
            )
            with urllib.request.urlopen(req_cs, timeout=120) as _r:
                _r.read()
            cold_ms = (time.perf_counter() - t_cs) * 1000
            print(f"  Cold-start load time : {cold_ms:,.0f} ms  ({cold_ms/1000:.2f}s)")
            print("  Tip: restart the server before running --cold-start for a true cold measurement.")
        except Exception as _e:
            print(f"  Cold-start check failed: {_e}")
        print()

    prompts = [
        "Explain quantum entanglement in two sentences.",
        "What is the time complexity of quicksort?",
        "Write a Python function that reverses a string.",
        "What causes the Northern Lights?",
    ]

    print(f"\n  Squish bench — {len(prompts)} prompts, {args.max_tokens} max tokens")
    print(f"  Server: {base_url}")
    print()

    results = []
    prompted = []
    for i, prompt in enumerate(prompts):
        payload = json.dumps({
            "model": "squish",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": args.max_tokens,
            "temperature": 0.7,
            "stream": True,
        }).encode()
        req = urllib.request.Request(
            f"{base_url}/chat/completions", data=payload,
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {api_key}"},
        )
        t0 = time.perf_counter()
        ttft = None
        n_toks = 0
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                for raw_line in resp:
                    line = raw_line.decode().strip()
                    if not line.startswith("data: "):
                        continue
                    payload_str = line[6:]
                    if payload_str == "[DONE]":
                        break
                    try:
                        chunk = json.loads(payload_str)
                        delta = chunk["choices"][0]["delta"].get("content", "")
                        if delta:
                            if ttft is None:
                                ttft = time.perf_counter() - t0
                            n_toks += len(delta.split())  # approximate
                    except Exception:
                        pass
        except Exception as e:
            print(f"  [{i+1}] ERROR: {e}")
            continue

        total = time.perf_counter() - t0
        tps   = n_toks / total if total > 0 else 0
        print(f"  [{i+1}] {prompt[:50]:<50}  TTFT={ttft*1000:.0f}ms  "
              f"{n_toks:>4} tok  {tps:.1f} tok/s")
        results.append({"ttft": ttft, "tps": tps, "n_toks": n_toks})
        prompted.append(prompt)

    if results:
        print()
        avg_ttft = sum(r["ttft"] for r in results if r["ttft"]) / len(results)
        avg_tps  = sum(r["tps"] for r in results) / len(results)
        print(f"  Average TTFT: {avg_ttft*1000:.0f} ms")
        print(f"  Average throughput: {avg_tps:.1f} tok/s (≈word/s)")

        if getattr(args, "markdown", False) or getattr(args, "save", None):
            import platform as _plat
            try:
                chip = subprocess.check_output(
                    ["sysctl", "-n", "machdep.cpu.brand_string"],
                    text=True, stderr=subprocess.DEVNULL,
                ).strip()
            except Exception:
                chip = _plat.machine()
            try:
                mem_bytes = int(subprocess.check_output(
                    ["sysctl", "-n", "hw.memsize"], text=True,
                ).strip())
                mem_gb = f"{mem_bytes/1e9:.0f} GB"
            except Exception:
                mem_gb = "?"
            lines = [
                f"## Squish Benchmark — {time.strftime('%Y-%m-%d')}",
                "",
                f"> Hardware: {chip} · {mem_gb} unified memory",
                f"> Server: {base_url} · {args.max_tokens} max tokens",
                "",
                "| Prompt | TTFT (ms) | Tokens | Tok/s |",
                "|--------|----------:|-------:|------:|",
            ]
            for prompt, r in zip(prompted, results, strict=False):
                lines.append(
                    f"| {prompt[:55]:<55} | "
                    f"{r['ttft']*1000:.0f} | {r['n_toks']} | {r['tps']:.1f} |"
                )
            lines += [
                f"| **Average** | **{avg_ttft*1000:.0f}** | — | **{avg_tps:.1f}** |",
                "",
                "_Reproduced with: `squish bench --markdown`_",
            ]
            md = "\n".join(lines)
            save_path = Path(getattr(args, "save", None) or "squish_bench.md")
            save_path.write_text(md)
            print(f"\n  Markdown table saved to {save_path}")
    print()


# ── CLI entry point ───────────────────────────────────────────────────────────

def cmd_doctor(args):
    """Check that all squish components are installed correctly."""
    import platform as _platform
    import socket

    print()
    _box(["squish doctor — dependency check"])
    print()

    ok = True

    def _check(label: str, passed: bool, fix: str = "") -> None:
        nonlocal ok
        sym = "✓" if passed else "✗"
        print(f"  {sym}  {label}")
        if not passed:
            ok = False
            if fix:
                print(f"       Fix: {fix}")

    # OS
    _check("macOS / Apple Silicon",
           _platform.system() == "Darwin" and _platform.machine() == "arm64",
           "squish requires macOS on Apple Silicon (M-series)")

    def _ver_ok(found: str, required: str) -> bool:
        """Return True if found >= required using tuple comparison."""
        try:
            def _to_tuple(v: str):
                return tuple(int(x) for x in v.split("+")[0].split(".") if x.isdigit())
            return _to_tuple(found) >= _to_tuple(required)
        except Exception:  # pragma: no cover
            return True  # unknown format → assume ok

    # MLX
    try:
        import mlx.core as mx
        _check(f"mlx ≥ 0.18  (found {mx.__version__})",
               _ver_ok(mx.__version__, "0.18"),
               "pip install --upgrade mlx")
    except ImportError:  # pragma: no cover
        _check("mlx", False, "pip install mlx")

    # mlx-lm
    try:
        import mlx_lm
        version = getattr(mlx_lm, "__version__", "0")
        _check(f"mlx-lm ≥ 0.19  (found {version})",
               _ver_ok(version, "0.19"),
               "pip install --upgrade mlx-lm")
    except ImportError:  # pragma: no cover
        _check("mlx-lm", False, "pip install mlx-lm")

    # numpy
    try:
        import numpy as np
        _check(f"numpy ≥ 1.26  (found {np.__version__})",
               _ver_ok(np.__version__, "1.26"),
               "pip install --upgrade numpy")
    except ImportError:  # pragma: no cover
        _check("numpy", False, "pip install numpy")

    # transformers
    try:
        import transformers
        _check(f"transformers ≥ 4.40  (found {transformers.__version__})",
               _ver_ok(transformers.__version__, "4.40"),
               "pip install --upgrade transformers")
    except ImportError:  # pragma: no cover
        _check("transformers", False, "pip install transformers")

    # zstandard
    try:
        import zstandard
        _check(f"zstandard ≥ 0.22  (found {zstandard.__version__})",
               _ver_ok(zstandard.__version__, "0.22"),
               "pip install --upgrade zstandard")
    except ImportError:  # pragma: no cover
        _check("zstandard (optional zstd entropy layer)", False, "pip install zstandard")

    # squish_quant Rust extension
    try:
        import squish_quant  # noqa: F401
        _check("squish_quant Rust extension (6 GB/s quantizer)", True)
    except ImportError:  # pragma: no cover
        _check("squish_quant Rust extension (optional — 4× faster quantization)", False,
               "cd squish_quant_rs && python3 -m maturin build --release && pip install .")

    # squish.quantizer self-test
    try:
        import numpy as np

        from squish.quantizer import (
            mean_cosine_similarity,
            quantize_embeddings,
            reconstruct_embeddings,
        )
        rng = np.random.default_rng(0)
        emb = rng.standard_normal((32, 128)).astype(np.float32)
        r   = quantize_embeddings(emb, group_size=64)
        rec = reconstruct_embeddings(r)
        sim = mean_cosine_similarity(emb, rec)
        _check(f"squish.quantizer round-trip  (cosine={sim:.5f})", sim > 0.999,
               "Run: python3 -m squish.quantizer")
    except Exception as e:  # pragma: no cover
        _check(f"squish.quantizer self-test: {e}", False)

    # Models directory
    models_dir = _MODELS_DIR
    if models_dir.exists():
        n = sum(1 for d in models_dir.iterdir() if d.is_dir() and not d.name.startswith("."))
        _check(f"models dir {models_dir}  ({n} model(s))", True)
    else:  # pragma: no cover
        _check(f"models dir {models_dir}", False,
               f"mkdir -p {models_dir}")

    # Disk space in models dir
    try:
        import shutil as _shutil
        _disk_path = _MODELS_DIR if _MODELS_DIR.exists() else Path.home()
        _stat = _shutil.disk_usage(_disk_path)
        _free_gb = _stat.free / 1e9
        _check(
            f"Disk free: {_free_gb:.1f} GB  (\u2265 5 GB recommended for small models)",
            _free_gb >= 5.0,
            "Free at least 5 GB of disk space before pulling a model",
        )
    except Exception:  # pragma: no cover
        pass  # non-fatal

    # Server status
    s = socket.socket()
    s.settimeout(0.5)
    try:
        s.connect(("127.0.0.1", _DEFAULT_PORT))
        _check(f"server running on :{_DEFAULT_PORT}", True)  # pragma: no cover
    except Exception:  # pragma: no cover
        _check("server not running (optional)", True)  # not an error
    finally:
        s.close()

    print()
    if ok:
        print("  All checks passed. squish is ready.\n")
    else:
        print("  Some checks failed. See fixes above.\n")


def cmd_daemon(args):  # pragma: no cover
    """Start, stop, or check the Squish daemon (persistent background server)."""
    import signal

    pid_file = Path.home() / ".squish" / "daemon.pid"
    log_file = Path.home() / ".squish" / "daemon.log"
    pid_file.parent.mkdir(parents=True, exist_ok=True)

    def _read_pid() -> int | None:
        try:
            return int(pid_file.read_text().strip())
        except Exception:
            return None

    def _is_running(pid: int) -> bool:
        try:
            os.kill(pid, 0)
            return True
        except (ProcessLookupError, PermissionError):
            return False

    action = args.daemon_action or "status"

    if action == "status":
        pid = _read_pid()
        if pid and _is_running(pid):
            print(f"\n  ✓  Squish daemon running  (pid {pid})")
            print(f"     Endpoint : http://{args.host}:{args.port}/v1")
            print(f"     Log      : {log_file}\n")
        else:
            print("\n  ✗  Squish daemon not running  (start with: squish daemon start)\n")
        return

    if action == "stop":
        pid = _read_pid()
        if pid and _is_running(pid):
            try:
                os.kill(pid, signal.SIGTERM)
                time.sleep(1)
                if _is_running(pid):
                    os.kill(pid, signal.SIGKILL)
                pid_file.unlink(missing_ok=True)
                print(f"\n  ✓  Daemon stopped  (pid {pid})\n")
            except Exception as e:
                _die(f"Could not stop daemon: {e}")
        else:
            print("\n  Daemon is not running.\n")
            pid_file.unlink(missing_ok=True)
        return

    if action == "start":
        pid = _read_pid()
        if pid and _is_running(pid):
            print(f"\n  Daemon already running  (pid {pid}).")
            print("  Stop first with: squish daemon stop\n")
            return

        model_dir, compressed_dir = _resolve_model(args.model)
        server_script = Path(__file__).resolve().parent / "server.py"
        if not server_script.exists():
            _die(f"server.py not found at {server_script}")

        port    = args.port
        host    = args.host
        api_key = args.api_key

        print(f"\n  Starting Squish daemon for {model_dir.name} …")
        print(f"  Endpoint : http://{host}:{port}/v1")
        print(f"  Log      : {log_file}\n")

        with open(log_file, "a") as log:
            proc = subprocess.Popen(
                [
                    sys.executable, str(server_script),
                    "--model-dir",      str(model_dir),
                    "--compressed-dir", str(compressed_dir),
                    "--port",           str(port),
                    "--host",           host,
                    # API key via env var — keeps it out of `ps aux`
                ],
                stdout=log,
                stderr=log,
                stdin=subprocess.DEVNULL,
                start_new_session=True,
                env={**os.environ, "SQUISH_API_KEY": api_key},
            )

        pid_file.write_text(str(proc.pid))

        # Wait up to 30s for server to respond
        import socket as _sock
        for _ in range(60):
            time.sleep(0.5)
            s = _sock.socket()
            s.settimeout(0.5)
            try:
                s.connect((host, port))
                s.close()
                print(f"  ✓  Daemon ready  (pid {proc.pid})\n")
                return
            except Exception:
                pass
        print(f"  ⚠  Daemon started (pid {proc.pid}) but hasn't responded yet.")
        print(f"     Check logs: tail -f {log_file}\n")


def cmd_compress(args):  # pragma: no cover
    """Compress a model directory to Squish npy-dir format (INT8 or INT4)."""
    # Resolve model path (accept shorthand or full path)
    if args.model in _MODEL_SHORTHAND:
        model_dir = _MODELS_DIR / _MODEL_SHORTHAND[args.model]
    elif _CATALOG_AVAILABLE:
        entry = _catalog_resolve(args.model)
        if entry is not None:
            model_dir = _MODELS_DIR / entry.dir_name
        else:
            model_dir = Path(args.model).expanduser()
    else:
        model_dir = Path(args.model).expanduser()

    if not model_dir.exists():
        _die(f"Model directory not found: {model_dir}")

    output_dir = Path(args.output).expanduser() if args.output else Path(str(model_dir) + _COMPRESSED_SUFFIX)

    quant_label = "INT4 (~44% disk savings, ≤2% accuracy delta)" if getattr(args, "int4", False) else "INT8 group-64"
    print(f"\n  Compressing: {model_dir}")
    print(f"  Quantization: {quant_label}")
    print(f"  Output:      {output_dir}\n")

    # ── Optional AWQ calibration pass ────────────────────────────────────────
    awq_scales_dir = None
    if getattr(args, "awq", False):
        n_samples = getattr(args, "awq_samples", 20)
        print(f"  Running AWQ calibration ({n_samples} samples)...")
        print("  Note: loads full model in memory — may take 2–5 min for large models.")
        try:
            import tempfile

            import mlx_lm
            model_awq, tokenizer_awq = mlx_lm.load(str(model_dir))
            from squish.awq import collect_activation_scales, save_awq_scales
            scales = collect_activation_scales(
                model_awq, tokenizer_awq,
                n_samples=n_samples, verbose=True,
            )
            awq_scales_dir = tempfile.mkdtemp(prefix="squish_awq_")
            save_awq_scales(scales, awq_scales_dir, verbose=False)
            print(f"  ✓  AWQ scales collected → {awq_scales_dir}")
            del model_awq
            try:
                import mlx.core as mx
                mx.clear_cache()
            except Exception:
                pass
        except ImportError as _e:
            print(f"  Warning: AWQ skipped — {_e}. Install mlx-lm to enable AWQ.")
        except Exception as _e:
            print(f"  Warning: AWQ calibration failed — {_e}. Continuing without AWQ.")

    cmd = [
        sys.executable, "-m", "squish.convert",
        "--model-dir", str(model_dir),
        "--output",    str(output_dir),
        "--format",    "npy-dir",
    ]
    if args.passthrough:
        cmd += ["--passthrough"] + args.passthrough
    if args.outlier_threshold != 20.0:
        cmd += ["--outlier-threshold", str(args.outlier_threshold)]
    if getattr(args, "int4", False):
        cmd.append("--int4")
    if awq_scales_dir:
        cmd += ["--awq-scales", awq_scales_dir]
    if args.verbose:
        cmd.append("--verbose")

    print("  Compressing weights — this may take 3–8 min for large models …")
    sys.stdout.flush()
    import threading as _threading

    _compress_done = False

    def _heartbeat():
        elapsed = 0
        while not _compress_done:
            time.sleep(15)
            elapsed += 15
            if not _compress_done:
                print(f"  … still compressing ({elapsed}s elapsed) — please wait", flush=True)

    _hb_thread = _threading.Thread(target=_heartbeat, daemon=True)
    _hb_thread.start()
    # Inherit the environment and inject the repo root into PYTHONPATH so that
    # the subprocess can always find squish.convert regardless of whether squish
    # is pip-installed in sys.executable's site-packages.
    _repo_root_str = str(Path(__file__).parent.parent)
    _env = os.environ.copy()
    _env["PYTHONPATH"] = _repo_root_str + os.pathsep + _env.get("PYTHONPATH", "")
    try:
        result = subprocess.run(cmd, cwd=_repo_root_str, env=_env)
    finally:
        _compress_done = True

    if result.returncode != 0:
        _die("Compression failed — see output above.")
    print(f"\n  ✓  Compressed model saved to {output_dir}")

    # ── Optional zstd entropy pass ──────────────────────────────────────────
    zstd_level = getattr(args, "zstd_level", 0)
    if zstd_level and zstd_level > 0:
        tensors_dir = output_dir / "tensors"
        if tensors_dir.exists():
            print(f"  Applying zstd entropy compression at level {zstd_level} …")
            try:
                from squish.entropy import compress_npy_dir
                compress_npy_dir(tensors_dir, level=zstd_level, verbose=True)
                print("  ✓  Entropy compression complete.")
            except ImportError:
                print("  Warning: zstandard not installed — skipping entropy pass.")
                print("  Install with: pip install zstandard")
        else:
            print(f"  Warning: tensors/ not found at {tensors_dir} — skipping entropy pass.")

    print(f"     Run with: squish run {model_dir}\n")


# ── squish pull ───────────────────────────────────────────────────────────────

def cmd_pull(args):  # pragma: no cover
    """
    Download and compress a model from the Squish catalog.

    If pre-compressed Squish weights exist on HuggingFace they are downloaded
    directly (no on-device compression needed).  Otherwise the bf16 MLX weights
    are fetched and compressed locally.

    Examples
    --------
      squish pull qwen3:8b
      squish pull gemma3:4b --int4
      squish pull deepseek-r1:7b --token hf_…
      squish pull llama3.1:8b --models-dir /Volumes/SSD/models
    """
    if not _CATALOG_AVAILABLE:
        _die("squish.catalog is not available. Ensure the package is properly installed.")

    name = args.model
    models_dir = Path(args.models_dir).expanduser() if args.models_dir else _MODELS_DIR
    token = args.token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

    # Resolve first so we can print a clear error before any download starts
    entry = _catalog_resolve(name)
    if entry is None:
        _die(
            f"Unknown model: {name!r}\n"
            f"Run `squish catalog` to browse available models."
        )

    quant_label = "INT4" if args.int4 else "INT8"
    print()
    _box([
        "  squish pull",
        f"  Model      : {entry.id}  —  {entry.name}",
        f"  Parameters : {entry.params}",
        f"  Raw size   : ~{entry.size_gb:.1f} GB",
        f"  Compressed : ~{entry.squished_size_gb:.1f} GB  ({quant_label})",
        f"  Context    : {entry.context:,} tokens",
        f"  Dest       : {models_dir}",
    ])
    print()

    try:
        compressed_dir = _catalog_pull(
            name=name,
            models_dir=models_dir,
            int4=args.int4,
            token=token,
            refresh_catalog=args.refresh_catalog,
            verbose=args.verbose,
        )
    except ImportError as exc:
        _die(str(exc))
    except ValueError as exc:
        _die(str(exc))
    except RuntimeError as exc:
        _die(str(exc))

    print()
    _box([
        f"  ✓  {entry.id} ready!",
        f"  Run  : squish run {entry.id}",
        f"  Chat : squish chat {entry.id}",
        f"  Path : {compressed_dir}",
    ])
    print()


# ── squish catalog ────────────────────────────────────────────────────────────

def cmd_catalog(args):
    """Browse the Squish model catalog."""
    if not _CATALOG_AVAILABLE:  # pragma: no cover
        _die("squish.catalog is not available. Ensure the package is properly installed.")

    entries = list_catalog(
        tag=args.tag or None,
        refresh=args.refresh,
    )

    if not entries:
        tag_msg = f" (tag: {args.tag})" if args.tag else ""
        print(f"\n  No models found{tag_msg}.")
        return

    print()
    _box([
        "  Squish Model Catalog",
        f"  {len(entries)} model(s) available",
        "  Run `squish pull <id>` to download",
    ])
    print()

    # Header
    col_id   = max(len(e.id) for e in entries) + 2
    print(f"  {'ID':<{col_id}} {'Params':>7}  {'Raw':>7}  {'Squished':>9}  {'Prebuilt':>9}  Notes")
    print(f"  {'─'*col_id} {'─'*7}  {'─'*7}  {'─'*9}  {'─'*9}  {'─'*24}")

    for e in entries:
        prebuilt = "⚡ yes" if e.has_prebuilt else "compress"
        notes = e.notes if e.notes else ", ".join(e.tags)
        print(
            f"  {e.id:<{col_id}} {e.params:>7}  "
            f"{e.size_gb:>6.1f}G  {e.squished_size_gb:>8.1f}G  "
            f"{prebuilt:>9}  {notes}"
        )

    print()
    print("  Prebuilt ⚡ = pre-compressed weights on HuggingFace (instant download)")
    print()

    if args.tag:
        print(f"  Showing tag: {args.tag!r}")
        print("  Other tags: small, fast, balanced, large, reasoning, moe, edge")
    else:
        print("  Filter by tag: squish catalog --tag reasoning")
        print("  Refresh list : squish catalog --refresh")
    print()


# ── EAGLE-3 head download (Phase 1B) ─────────────────────────────────────────

_EAGLE_HEAD_CATALOG: dict[str, str] = {
    # model-alias → HuggingFace repo for the EAGLE-3 head
    "qwen3:8b":       "yuhuili/EAGLE3-Qwen3-Instruct-8B",
    "qwen3:4b":       "yuhuili/EAGLE3-Qwen3-Instruct-4B",
    "qwen3:14b":      "yuhuili/EAGLE3-Qwen3-Instruct-14B",
    "qwen3:30b-a3b":  "yuhuili/EAGLE3-Qwen3-Instruct-30B-A3B",
    "qwen2.5:7b":     "yuhuili/EAGLE3-Qwen2.5-Instruct-7B",
    "qwen2.5:14b":    "yuhuili/EAGLE3-Qwen2.5-Instruct-14B",
    "llama3.1:8b":    "yuhuili/EAGLE3-LLaMA3.1-Instruct-8B",
    "llama3.2:3b":    "yuhuili/EAGLE3-Llama-3.2-Instruct-3B",
}


def cmd_pull_head(args):  # pragma: no cover
    """
    Download an EAGLE-3 draft head from HuggingFace and convert it to MLX format.

    Examples
    --------
      squish pull-head qwen3:8b
      squish pull-head yuhuili/EAGLE3-Qwen3-Instruct-8B --output ~/.squish/heads/qwen3-8b
      squish pull-head qwen3:8b --token hf_…
    """
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        _die(
            "huggingface_hub is required for pull-head.\n"
            "Install it with: pip install huggingface-hub"
        )

    model_arg = args.model
    token = args.token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

    # Resolve alias → HF repo
    hf_repo = _EAGLE_HEAD_CATALOG.get(model_arg.lower(), model_arg)
    if "/" not in hf_repo:
        _die(
            f"Unknown model alias {model_arg!r}.\n"
            f"Pass a full HuggingFace repo (e.g. yuhuili/EAGLE3-Qwen3-Instruct-8B) "
            f"or one of: {', '.join(_EAGLE_HEAD_CATALOG)}"
        )

    # Determine output directory
    if args.output:
        out_dir = Path(args.output).expanduser()
    else:
        slug = hf_repo.split("/")[-1].lower()
        out_dir = Path.home() / ".squish" / "eagle-heads" / slug

    out_dir.mkdir(parents=True, exist_ok=True)

    print()
    _box([
        "  squish pull-head",
        f"  HF repo    : {hf_repo}",
        f"  Output dir : {out_dir}",
    ])
    print()

    # Download from HuggingFace Hub
    print(f"  Downloading {hf_repo} …")
    raw_dir = snapshot_download(
        repo_id=hf_repo,
        local_dir=str(out_dir / "_raw"),
        token=token or None,
        ignore_patterns=["*.bin"],  # prefer safetensors / MLX
    )

    # If weights are already in MLX format (config.json + *.safetensors),
    # just symlink / copy; otherwise convert via mlx_lm.
    import shutil, json as _json
    raw_path = Path(raw_dir)
    has_mlx = (raw_path / "config.json").exists() and any(raw_path.glob("*.safetensors"))

    if has_mlx:
        mlx_dir = out_dir
        for f in raw_path.iterdir():
            dst = mlx_dir / f.name
            if not dst.exists():
                shutil.copy2(f, dst)
        print("  Weights already in MLX safetensors format — no conversion needed.")
    else:
        # Convert PyTorch / BF16 safetensors to MLX
        print("  Converting to MLX format …")
        try:
            from mlx_lm import convert as _mlx_convert  # type: ignore
            _mlx_convert(
                hf_path=raw_dir,
                mlx_path=str(out_dir),
                quantize=False,  # EAGLE heads are already compact; keep fp16
            )
        except ImportError:
            _die("mlx_lm is required for conversion. Install with: pip install mlx-lm")
        except Exception as exc:
            _die(f"Conversion failed: {exc}")

    # Clean up raw download directory
    shutil.rmtree(out_dir / "_raw", ignore_errors=True)

    print()
    print(f"  EAGLE-3 head saved to: {out_dir}")
    print()
    print("  Start with EAGLE-3 speculative decoding:")
    print(f"    squish run --model <your-model> --eagle-head-dir {out_dir}")
    print()


def cmd_convert_model(args):
    """
    Convert and optionally quantize a model with mixed-precision quantization.

    Runs two mlx_lm.convert passes for different precision per layer group:
      - FFN layers (all linear except lm_head + embed_tokens): --ffn-bits
      - Embedding/output layers (lm_head, embed_tokens): --embed-bits

    Usage:
      squish convert-model --source-path path/to/model \\
        --output-path path/to/output \\
        --ffn-bits 4 --embed-bits 6
    """
    source_path = Path(args.source_path).expanduser().resolve()
    output_path = Path(args.output_path).expanduser().resolve()

    if not source_path.exists():
        _die(f"Source path not found: {source_path}")

    if args.dry_run:
        print(f"  [dry-run] source      : {source_path}")
        print(f"  [dry-run] output      : {output_path}")
        print(f"  [dry-run] ffn-bits    : {args.ffn_bits}")
        print(f"  [dry-run] embed-bits  : {args.embed_bits}")
        return

    output_path.mkdir(parents=True, exist_ok=True)

    try:
        import mlx_lm as _mlx_lm
    except ImportError:
        _die("mlx_lm is required for convert-model. Install with: pip install mlx-lm>=0.19")

    print(f"  Quantizing FFN layers to {args.ffn_bits}-bit …")
    try:
        _mlx_lm.convert(
            hf_path=str(source_path),
            mlx_path=str(output_path),
            quantize=True,
            q_bits=args.ffn_bits,
            linear_class_predicate=lambda m: (
                "lm_head" not in m.name and "embed_tokens" not in m.name
            ),
        )
    except Exception as exc:
        _die(f"FFN quantization failed: {exc}")

    if args.embed_bits != args.ffn_bits:
        print(f"  Re-quantizing embed/lm_head to {args.embed_bits}-bit …")
        try:
            _mlx_lm.convert(
                hf_path=str(output_path),
                mlx_path=str(output_path),
                quantize=True,
                q_bits=args.embed_bits,
                linear_class_predicate=lambda m: (
                    "lm_head" in m.name or "embed_tokens" in m.name
                ),
            )
        except Exception as exc:
            _die(f"Embed quantization failed: {exc}")

    print(f"\n  Mixed-precision model saved to: {output_path}")
    print(f"  Load with: squish run --mlx-model-dir {output_path}")


def main():
    ap = argparse.ArgumentParser(
        prog="squish",
        description="Squish — private local inference for Apple Silicon",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  squish catalog                     Browse available models
  squish catalog --tag reasoning     Filter by tag (reasoning, small, large, moe…)
  squish pull qwen3:8b               Download + compress Qwen3-8B
  squish pull gemma3:4b --int4       Download with INT4 compression (~44% disk savings)
  squish run qwen3:8b                Start server on :11435
  squish run 7b --batch-scheduler    Legacy shorthand, with continuous batching
  squish chat qwen3:8b               Interactive terminal chat
  squish chat                        Chat against already-running server
  squish models                      List local models
  squish info                        System info + server status
  squish doctor                      Check all dependencies
  squish daemon start qwen3:8b       Start persistent background server
  squish daemon status               Check daemon status
  squish daemon stop                 Stop daemon
  squish bench                       Quick throughput benchmark
  squish compress qwen3:8b           Compress a local model to INT8 npy-dir format

Model IDs (sample):
  qwen3:8b   gemma3:4b   deepseek-r1:7b   llama3.2:3b   phi4:14b
  Legacy: 7b  14b  1.5b  32b  72b

OpenAI drop-in (after squish run):
  export OPENAI_BASE_URL=http://localhost:11435/v1
  export OPENAI_API_KEY=squish
  # Any openai-compatible client/agent framework now works locally

Ollama drop-in:
  export OLLAMA_HOST=http://localhost:11435
  ollama list    # or use any Ollama-compatible tool
""",
    )
    sub = ap.add_subparsers(dest="command")

    ap.add_argument(
        "--version", action="version",
        version="squish 1.0.1",
        help="Show squish version and exit",
    )

    # ── run ──
    p_run = sub.add_parser("run", help="Start the inference server")
    p_run.add_argument("model", nargs="?", help="Model: 7b, 14b, 1.5b, or path")
    p_run.add_argument("--port",    type=int, default=_DEFAULT_PORT)
    p_run.add_argument("--host",    default="127.0.0.1",
                       help="0.0.0.0 to expose on LAN")
    p_run.add_argument("--api-key", default="squish")
    p_run.add_argument("--draft-model",      default="",
                       help="Path to draft model for speculative decoding")
    p_run.add_argument("--batch-scheduler",  action="store_true",
                       help="Enable continuous batching (improves concurrent throughput)")
    p_run.add_argument("--batch-size",       type=int, default=8)
    p_run.add_argument("--kv-cache-mode",    choices=["fp16", "int8", "snap"], default="fp16")
    p_run.add_argument("--log-level",
                       choices=["critical", "error", "warning", "info", "debug", "trace"],
                       default="warning",
                       help="Server log verbosity (default: warning)")
    p_run.set_defaults(func=cmd_run)

    # ── serve (alias for run) ──
    p_serve = sub.add_parser("serve", help="Start the inference server (alias for 'run')")
    p_serve.add_argument("model", nargs="?", help="Model: qwen3:8b, 7b, 14b, or path")
    p_serve.add_argument("--port",    type=int, default=_DEFAULT_PORT)
    p_serve.add_argument("--host",    default="127.0.0.1",
                         help="0.0.0.0 to expose on LAN")
    p_serve.add_argument("--api-key", default="squish")
    p_serve.add_argument("--draft-model",      default="")
    p_serve.add_argument("--batch-scheduler",  action="store_true")
    p_serve.add_argument("--batch-size",       type=int, default=8)
    p_serve.add_argument("--kv-cache-mode",    choices=["fp16", "int8", "snap"], default="fp16")
    p_serve.add_argument("--log-level",
                         choices=["critical", "error", "warning", "info", "debug", "trace"],
                         default="warning",
                         help="Server log verbosity (default: warning)")
    p_serve.set_defaults(func=cmd_run)

    # ── chat ──
    p_chat = sub.add_parser("chat", help="Interactive terminal chat")
    p_chat.add_argument("model", nargs="?", help="Model shorthand or path (auto-starts server if needed)")
    p_chat.add_argument("--port",        type=int, default=_DEFAULT_PORT)
    p_chat.add_argument("--host",        default="127.0.0.1")
    p_chat.add_argument("--api-key",     default="squish")
    p_chat.add_argument("--chat-model",  default="squish",
                        help="Model ID to send in requests (default: squish)")
    p_chat.add_argument("--system",      default="",
                        help="System prompt (default: private local assistant)")
    p_chat.add_argument("--max-tokens",  type=int, default=1024)
    p_chat.add_argument("--temperature", type=float, default=0.7)
    p_chat.add_argument("--max-history", type=int, default=40,
                        help="Keep at most this many non-system messages in context "
                             "(prevents unbounded token growth; default 40)")
    p_chat.set_defaults(func=cmd_chat)

    # ── models ──
    p_models = sub.add_parser("models", help="List local models")
    p_models.set_defaults(func=cmd_models)

    # ── info ──
    p_info = sub.add_parser("info", help="System info")
    p_info.set_defaults(func=cmd_info)

    # ── bench ──
    p_bench = sub.add_parser("bench", help="Quick throughput benchmark")
    p_bench.add_argument("--port",       type=int, default=_DEFAULT_PORT)
    p_bench.add_argument("--host",       default="127.0.0.1")
    p_bench.add_argument("--api-key",    default="squish")
    p_bench.add_argument("--max-tokens", type=int, default=128)
    p_bench.add_argument("--markdown",   action="store_true",
                         help="Print a markdown table after benchmarking")
    p_bench.add_argument("--save",       default="", metavar="FILE",
                         help="Save markdown table to FILE (implies --markdown; "
                              "default: squish_bench.md)")
    p_bench.add_argument("--cold-start",  action="store_true",
                         help="Time model load (first-request latency) before the benchmark prompts")
    p_bench.set_defaults(func=cmd_bench)

    # ── doctor ──
    p_doctor = sub.add_parser("doctor", help="Check all dependencies and system requirements")
    p_doctor.set_defaults(func=cmd_doctor)

    # ── daemon ──
    p_daemon = sub.add_parser("daemon", help="Manage the persistent background server daemon")
    p_daemon.add_argument("daemon_action", nargs="?",
                          choices=["start", "stop", "status"],
                          default="status",
                          help="start | stop | status (default: status)")
    p_daemon.add_argument("model", nargs="?", help="Model shorthand or path (for start)")
    p_daemon.add_argument("--port",    type=int, default=_DEFAULT_PORT)
    p_daemon.add_argument("--host",    default="127.0.0.1")
    p_daemon.add_argument("--api-key", default="squish")
    p_daemon.set_defaults(func=cmd_daemon)

    # ── compress ──
    p_compress = sub.add_parser("compress", help="Compress (squish) a model to INT8 npy-dir format")
    p_compress.add_argument("model", help="Model path (e.g. ~/.squish/models/llama3.1-8b-4bit) or shorthand (7b, 14b)")
    p_compress.add_argument("--output",            default=None,
                            help="Output directory (default: <model>-compressed)")
    p_compress.add_argument("--passthrough",       nargs="*", default=[], metavar="PATTERN",
                            help="Tensor substrings to keep as float32 (e.g. embed lm_head)")
    p_compress.add_argument("--outlier-threshold", type=float, default=20.0)
    p_compress.add_argument("--int4",    action="store_true",
                            help="INT4 nibble-packed (~44%% disk savings vs INT8). "
                                 "Requires squish_quant Rust ext. "
                                 "⚠ Not recommended for models < 3B — use INT8 for best quality.")
    p_compress.add_argument("--zstd-level", type=int, default=0, metavar="N",
                            help="Apply zstd entropy compression at level N (1-22) after "
                                 "quantization.  Level 3 is a good default; 0 = skip (default). "
                                 "Requires: pip install zstandard")
    p_compress.add_argument("--awq", action="store_true", default=False,
                            help="Run Activation-aware Weight Quantization (AWQ) calibration "
                                 "before INT8/INT4 compression. Improves accuracy at the cost "
                                 "of ~2-5 min extra calibration time.")
    p_compress.add_argument("--awq-samples", type=int, default=20, metavar="N",
                            help="Number of calibration samples for AWQ (default: 20)")
    p_compress.add_argument("--verbose",           action="store_true")
    p_compress.set_defaults(func=cmd_compress)

    # ── pull ──
    p_pull = sub.add_parser(
        "pull",
        help="Download + compress a model from the Squish catalog",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            "Download a model and compress it with Squish.\n\n"
            "Pre-compressed weights are downloaded directly when available.\n"
            "Otherwise the bf16 MLX variant is fetched and compressed locally.\n\n"
            "Examples:\n"
            "  squish pull qwen3:8b\n"
            "  squish pull gemma3:4b --int4\n"
            "  squish pull deepseek-r1:14b --token hf_…"
        ),
    )
    p_pull.add_argument("model", help="Model ID (e.g. qwen3:8b, gemma3:4b, 7b)")
    p_pull.add_argument("--int4", action="store_true",
                        help="Use INT4 nibble-packed compression (~44%% disk savings, ≤2%% accuracy delta). "
                             "⚠ Not recommended for models < 3B.")
    p_pull.add_argument("--token", default="",
                        help="HuggingFace access token (or set $HF_TOKEN)")
    p_pull.add_argument("--models-dir", default="",
                        help=f"Override models directory (default: {_MODELS_DIR})")
    p_pull.add_argument("--refresh-catalog", action="store_true",
                        help="Force-refresh the online catalog before resolving")
    p_pull.add_argument("--verbose", action="store_true")
    p_pull.set_defaults(func=cmd_pull)

    # ── pull-head (EAGLE-3) ──
    p_head = sub.add_parser(
        "pull-head",
        help="Download an EAGLE-3 draft head for speculative decoding",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            "Download an EAGLE-3 draft head and convert it to MLX format.\n\n"
            "An EAGLE head pairs with a specific target model and typically\n"
            "achieves 75-85%% draft acceptance versus 55-65%% for a separate\n"
            "draft model, at a fraction of the memory cost.\n\n"
            "Examples:\n"
            "  squish pull-head qwen3:8b\n"
            "  squish pull-head yuhuili/EAGLE3-Qwen3-Instruct-8B --output ./my-head\n"
            "  squish pull-head qwen3:8b --token hf_…"
        ),
    )
    p_head.add_argument("model",
                        help="Model alias (e.g. qwen3:8b) or full HF repo "
                             "(e.g. yuhuili/EAGLE3-Qwen3-Instruct-8B)")
    p_head.add_argument("--output", default="",
                        help="Output directory (default: ~/.squish/eagle-heads/<slug>)")
    p_head.add_argument("--token", default="",
                        help="HuggingFace API token (or set HF_TOKEN env var)")
    p_head.set_defaults(func=cmd_pull_head)

    # ── catalog ──
    p_catalog = sub.add_parser("catalog", help="Browse available models in the Squish catalog")
    p_catalog.add_argument("--tag", default="",
                           help="Filter by tag: small, fast, balanced, large, reasoning, moe, edge")
    p_catalog.add_argument("--refresh", action="store_true",
                           help="Force-refresh the catalog from HuggingFace")
    p_catalog.set_defaults(func=cmd_catalog)

    p_rm = sub.add_parser("rm", help="Remove a local model (frees disk space)")
    p_rm.add_argument("model", help="Model ID, alias, or path (e.g. qwen3:8b, 7b, ~/models/Llama-3)")
    p_rm.add_argument("--compressed-only", action="store_true",
                      help="Remove only the compressed (-compressed) directory")
    p_rm.add_argument("--raw-only", action="store_true",
                      help="Remove only the raw weights directory")
    p_rm.add_argument("--dry-run", action="store_true",
                      help="Show what would be removed without deleting anything")
    p_rm.add_argument("-y", "--yes", action="store_true",
                      help="Skip confirmation prompt")
    p_rm.set_defaults(func=cmd_rm, compressed_only=False, raw_only=False)  # pragma: no cover

    p_search = sub.add_parser("search", help="Search the model catalog")
    p_search.add_argument("query", help="Search query (matched against ID, tags, params, description)")
    p_search.set_defaults(func=cmd_search)

    # ── convert-model ──
    p_convert = sub.add_parser(
        "convert-model",
        help="Create a mixed-precision quantized model (different bits per layer group)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            "Quantize an existing model with different precision per layer group.\n\n"
            "Applies --ffn-bits to all linear layers except lm_head and embed_tokens,\n"
            "then --embed-bits to those omitted layers. Produces a single merged model.\n\n"
            "Example:\n"
            "  squish convert-model \\\n"
            "    --source-path ~/.squish/models/qwen3-8b \\\n"
            "    --output-path ~/.squish/models/qwen3-8b-mixed \\\n"
            "    --ffn-bits 4 --embed-bits 6\n"
        ),
    )
    p_convert.add_argument("--source-path", required=True, metavar="PATH",
                           help="Source model directory (HF format or mlx_lm format)")
    p_convert.add_argument("--output-path", required=True, metavar="PATH",
                           help="Output directory for mixed-precision model")
    p_convert.add_argument("--ffn-bits", type=int, default=4, metavar="N",
                           help="Quantization bits for FFN layers (default: 4)")
    p_convert.add_argument("--embed-bits", type=int, default=6, metavar="N",
                           help="Quantization bits for lm_head + embed_tokens (default: 6)")
    p_convert.add_argument("--dry-run", action="store_true", default=False,
                           help="Print what would be done without converting")
    p_convert.set_defaults(func=cmd_convert_model)

    args = ap.parse_args()

    if not args.command:
        ap.print_help()
        sys.exit(0)

    args.func(args)  # pragma: no cover


if __name__ == "__main__":
    main()
