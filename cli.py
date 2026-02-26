#!/usr/bin/env python3
"""
squish — CLI for Squish local LLM inference

The goal: replace every cloud API call, every `ollama run` call,
every rate-limited Gemini/OpenAI dependency with a single command
that works offline, instantly, for free.

Commands:
  squish run   "prompt"          — single-shot generation, streams to stdout
  squish chat                    — interactive chat session
  squish serve                   — start the OpenAI-compatible API server
  squish pull  MODEL_ID          — download + compress a model from HuggingFace
  squish list                    — list compressed models available locally
  squish ps                      — show if a server is running and on what port
  squish bench [--model MODEL]   — run the full benchmark suite
  squish git   [path]            — generate a git commit message (runs on staged diff)

Examples:
  squish run "Explain attention mechanisms in one paragraph"
  squish git                     # in any git repo — generates + stages a commit
  squish pull Qwen/Qwen2.5-7B-Instruct
  squish serve --port 8000
  squish chat --model ~/models/Qwen2.5-7B-Instruct-bf16

Point existing tools at it:
  export OPENAI_BASE_URL=http://localhost:8000/v1
  export OPENAI_API_KEY=squish
"""

import argparse
import json
import os
import re
import signal
import socket
import subprocess
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path

POC_DIR = Path(__file__).parent.resolve()

# ── ANSI ──────────────────────────────────────────────────────────────────────
GREEN   = ""
RED     = ""
YELLOW  = ""
CYAN    = ""
MAGENTA = ""
BOLD    = ""
DIM     = ""
RESET   = ""

# Default server address
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000

# Where Squish stores its compressed models
SQUISH_HOME = Path(os.environ.get("SQUISH_HOME", Path.home() / ".squish"))
MODELS_DIR  = SQUISH_HOME / "models"

# ── helpers ───────────────────────────────────────────────────────────────────

def _server_url(port: int = DEFAULT_PORT) -> str:
    return f"http://{DEFAULT_HOST}:{port}"


def _is_server_up(port: int = DEFAULT_PORT, timeout: float = 0.5) -> bool:
    """Fast TCP probe — no HTTP overhead."""
    try:
        with socket.create_connection((DEFAULT_HOST, port), timeout=timeout):
            return True
    except (ConnectionRefusedError, TimeoutError, OSError):
        return False


def _wait_for_server(port: int = DEFAULT_PORT, max_wait: float = 60.0) -> bool:
    """Wait up to max_wait seconds for the server to accept connections."""
    deadline = time.time() + max_wait
    first = True
    while time.time() < deadline:
        if _is_server_up(port):
            return True
        if first:
            print(f"{DIM}  waiting for server on :{port} …{RESET}", end="", flush=True)
            first = False
        else:
            print(".", end="", flush=True)
        time.sleep(0.2)
    print()
    return False


def _find_model(model_hint: str | None) -> tuple[str | None, str | None]:
    """
    Find model_dir and compressed_dir for a model hint.
    Returns (model_dir, compressed_dir).
    """
    models_root = Path.home() / "models"

    # Explicit path
    if model_hint and Path(model_hint).expanduser().exists():
        p = Path(model_hint).expanduser()
        comp = Path(str(p) + "-compressed")
        return str(p), str(comp) if comp.exists() else None

    # Search ~/models for matching prefix
    if models_root.exists():
        candidates = sorted(models_root.iterdir())
        # Prefer bf16 variants
        for d in candidates:
            if d.is_dir() and (not model_hint or model_hint.lower() in d.name.lower()):
                if "bf16" in d.name and "compressed" not in d.name:
                    comp = Path(str(d) + "-compressed")
                    return str(d), str(comp) if comp.exists() else None
        # Fall back to any match
        for d in candidates:
            if d.is_dir() and (not model_hint or model_hint.lower() in d.name.lower()):
                if "compressed" not in d.name:
                    comp = Path(str(d) + "-compressed")
                    return str(d), str(comp) if comp.exists() else None

    # SQUISH_HOME models
    if MODELS_DIR.exists():
        for d in sorted(MODELS_DIR.iterdir()):
            if d.is_dir() and (not model_hint or model_hint.lower() in d.name.lower()):
                if "compressed" not in d.name:
                    comp = MODELS_DIR / (d.name + "-compressed")
                    return str(d), str(comp) if comp.exists() else None

    return None, None


def _start_server_background(model_dir: str, compressed_dir: str | None, port: int) -> int:
    """
    Start server.py in a detached background process.
    Returns the PID.
    """
    cmd = [
        sys.executable, str(POC_DIR / "squish" / "server.py"),
        "--model-dir", model_dir,
        "--port", str(port),
    ]
    if compressed_dir and Path(compressed_dir).exists():
        cmd += ["--compressed-dir", compressed_dir]

    # Write PID file so `squish ps` can find it
    pidfile = SQUISH_HOME / f"server_{port}.pid"
    SQUISH_HOME.mkdir(parents=True, exist_ok=True)

    log_path = SQUISH_HOME / f"server_{port}.log"
    log_f = open(log_path, "w")

    proc = subprocess.Popen(
        cmd,
        stdout=log_f,
        stderr=log_f,
        close_fds=True,
        start_new_session=True,
    )
    with open(pidfile, "w") as f:
        f.write(str(proc.pid))

    return proc.pid


def _http_stream(url: str, payload: dict) -> None:
    """Stream a chat completion response to stdout."""
    data = json.dumps(payload).encode()
    req  = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            buffer = b""
            for raw in resp:
                buffer += raw
                while b"\n" in buffer:
                    line, buffer = buffer.split(b"\n", 1)
                    line = line.strip()
                    if not line:
                        continue
                    if line.startswith(b"data: "):
                        chunk = line[6:]
                        if chunk == b"[DONE]":
                            return
                        try:
                            obj  = json.loads(chunk)
                            text = (obj.get("choices", [{}])[0]
                                       .get("delta", {})
                                       .get("content") or "")
                            if text:
                                print(text, end="", flush=True)
                        except json.JSONDecodeError:
                            pass
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")
        print(f"\n{RED}Server error {e.code}: {body}{RESET}", file=sys.stderr)
    except urllib.error.URLError as e:
        print(f"\n{RED}Connection error: {e.reason}{RESET}", file=sys.stderr)


def _http_post(url: str, payload: dict) -> dict | None:
    """Non-streaming POST, returns parsed JSON."""
    data = json.dumps(payload).encode()
    req  = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")
        print(f"{RED}Server error {e.code}: {body}{RESET}", file=sys.stderr)
        return None
    except urllib.error.URLError as e:
        print(f"{RED}Connection error: {e.reason}{RESET}", file=sys.stderr)
        return None


def _ensure_server(model_hint: str | None, port: int, auto_start: bool = True) -> bool:
    """
    Return True if a server is available on `port`.
    If not running and auto_start=True, find a model and start it.
    """
    if _is_server_up(port):
        return True

    if not auto_start:
        print(f"{RED}No Squish server running on :{port}.  Run: squish serve{RESET}",
              file=sys.stderr)
        return False

    model_dir, compressed_dir = _find_model(model_hint)
    if not model_dir:
        print(f"{RED}No model found.  Run: squish pull Qwen/Qwen2.5-7B-Instruct{RESET}",
              file=sys.stderr)
        return False

    model_name = Path(model_dir).name
    print(f"{CYAN}▸ Starting Squish server  [{model_name}]  port {port}{RESET}")
    pid = _start_server_background(model_dir, compressed_dir, port)
    print(f"{DIM}  server PID {pid}  log → {SQUISH_HOME}/server_{port}.log{RESET}")

    if not _wait_for_server(port, max_wait=90):
        print(f"\n{RED}Server failed to start.  "
              f"Check log: {SQUISH_HOME}/server_{port}.log{RESET}", file=sys.stderr)
        return False

    print(f"\n{GREEN}✓ Server ready{RESET}")
    return True


# ═══════════════════════════════════════════════════════════════════════════════
#  Commands
# ═══════════════════════════════════════════════════════════════════════════════

def cmd_run(args):
    """Single-shot generation — stream to stdout and exit."""
    prompt = " ".join(args.prompt) if args.prompt else ""

    # Read from stdin if no prompt given (allows: echo "prompt" | squish run)
    if not prompt and not sys.stdin.isatty():
        prompt = sys.stdin.read().strip()

    if not prompt:
        print(f"{RED}Usage: squish run \"your prompt here\"{RESET}", file=sys.stderr)
        print(f"{DIM}  Or pipe:  echo \"prompt\" | squish run{RESET}", file=sys.stderr)
        sys.exit(1)

    port = args.port
    if not _ensure_server(args.model, port):
        sys.exit(1)

    payload = {
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "stream": True,
    }
    _http_stream(f"{_server_url(port)}/v1/chat/completions", payload)
    print()   # trailing newline


def cmd_chat(args):
    """Interactive REPL chat session."""
    port = args.port
    if not _ensure_server(args.model, port):
        sys.exit(1)

    print(f"\n{BOLD}{CYAN}Squish — Interactive Chat{RESET}")
    print(f"{DIM}  Server: {_server_url(port)}{RESET}")
    print(f"{DIM}  Type 'exit' or Ctrl-D to quit.  '/reset' to clear history.{RESET}\n")

    history: list[dict] = []
    if args.system:
        history.append({"role": "system", "content": args.system})

    while True:
        try:
            user_input = input(f"{BOLD}{GREEN}you{RESET}  › ").strip()
        except (EOFError, KeyboardInterrupt):
            print(f"\n{DIM}Bye.{RESET}")
            break

        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit", "bye"):
            print(f"{DIM}Bye.{RESET}")
            break
        if user_input == "/reset":
            history.clear()
            if args.system:
                history.append({"role": "system", "content": args.system})
            print(f"{DIM}  History cleared.{RESET}")
            continue
        if user_input.startswith("/system "):
            new_sys = user_input[8:].strip()
            history = [m for m in history if m["role"] != "system"]
            history.insert(0, {"role": "system", "content": new_sys})
            print(f"{DIM}  System message updated.{RESET}")
            continue

        history.append({"role": "user", "content": user_input})
        print(f"\n{BOLD}{CYAN}squish{RESET} › ", end="", flush=True)

        payload = {
            "messages": history,
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "stream": True,
        }

        # Collect streamed response into buffer so we can add to history
        parts = []
        data  = json.dumps(payload).encode()
        req   = urllib.request.Request(
            f"{_server_url(port)}/v1/chat/completions",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            buf = b""
            with urllib.request.urlopen(req, timeout=120) as resp:
                for raw in resp:
                    buf += raw
                    while b"\n" in buf:
                        line, buf = buf.split(b"\n", 1)
                        line = line.strip()
                        if not line or not line.startswith(b"data: "):
                            continue
                        chunk = line[6:]
                        if chunk == b"[DONE]":
                            break
                        try:
                            obj  = json.loads(chunk)
                            text = (obj.get("choices", [{}])[0]
                                       .get("delta", {})
                                       .get("content") or "")
                            if text:
                                print(text, end="", flush=True)
                                parts.append(text)
                        except json.JSONDecodeError:
                            pass
        except (urllib.error.URLError, urllib.error.HTTPError) as e:
            print(f"\n{RED}Error: {e}{RESET}", file=sys.stderr)

        print("\n")
        assistant_text = "".join(parts)
        if assistant_text:
            history.append({"role": "assistant", "content": assistant_text})


def cmd_serve(args):
    """Launch the API server in the foreground."""
    model_dir, compressed_dir = _find_model(args.model)
    if not model_dir:
        print(f"{RED}No model found.  Specify --model or run: squish pull MODEL_ID{RESET}",
              file=sys.stderr)
        sys.exit(1)

    cmd = [
        sys.executable, str(POC_DIR / "squish" / "server.py"),
        "--model-dir", model_dir,
        "--port", str(args.port),
        "--host", args.host,
    ]
    if compressed_dir and Path(compressed_dir).exists():
        cmd += ["--compressed-dir", compressed_dir]
    if args.verbose:
        cmd += ["--verbose"]
    if args.reference:
        cmd += ["--reference"]

    os.execv(sys.executable, cmd)   # replace this process


def cmd_pull(args):
    """Download a model from HuggingFace and compress it."""
    model_id = args.model_id
    if not model_id:
        print(f"{RED}Usage: squish pull Qwen/Qwen2.5-7B-Instruct{RESET}", file=sys.stderr)
        sys.exit(1)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    short_name = model_id.split("/")[-1]

    print(f"\n{BOLD}{CYAN}Squish Pull — {model_id}{RESET}\n")

    # ── Step 1: download mlx version if possible, else bf16 convert ──────────
    mlx_model_id = f"mlx-community/{short_name}-bf16"
    model_dir    = MODELS_DIR / f"{short_name}-bf16"
    comp_dir     = MODELS_DIR / f"{short_name}-bf16-compressed"

    if model_dir.exists():
        print(f"{GREEN}✓ Model already downloaded: {model_dir}{RESET}")
    else:
        print(f"{CYAN}▸ Downloading {mlx_model_id} (bf16 MLX) …{RESET}")
        try:
            from huggingface_hub import snapshot_download
            snapshot_download(
                repo_id  = mlx_model_id,
                local_dir = str(model_dir),
                ignore_patterns=["*.pt", "*.bin", "original/"],
            )
            print(f"{GREEN}✓ Downloaded to {model_dir}{RESET}")
        except Exception as e:
            print(f"{YELLOW}⚠ mlx-community version not found: {e}{RESET}")
            print(f"{CYAN}▸ Falling back: download {model_id} and convert to bf16 …{RESET}")
            raw_dir = MODELS_DIR / short_name
            try:
                from huggingface_hub import snapshot_download
                snapshot_download(
                    repo_id  = model_id,
                    local_dir = str(raw_dir),
                    ignore_patterns=["*.pt", "original/"],
                )
            except ImportError:
                print(f"{RED}huggingface_hub not installed.  "
                      f"Run: pip install huggingface_hub{RESET}", file=sys.stderr)
                sys.exit(1)

            print(f"{CYAN}▸ Converting to bf16 with mlx_lm …{RESET}")
            result = subprocess.run(
                [sys.executable, "-m", "mlx_lm.convert",
                 "--hf-path", str(raw_dir),
                 "--mlx-path", str(model_dir),
                 "--dtype", "bfloat16"],
                capture_output=False,
            )
            if result.returncode != 0:
                print(f"{RED}Conversion failed{RESET}", file=sys.stderr)
                sys.exit(1)
            print(f"{GREEN}✓ Converted to bf16: {model_dir}{RESET}")

    # ── Step 2: compress with Vectro ──────────────────────────────────────────
    if comp_dir.exists() and (comp_dir / ".squish_ready").exists():
        print(f"{GREEN}✓ Already compressed: {comp_dir}{RESET}")
    else:
        print(f"\n{CYAN}▸ Compressing with Vectro INT8 (one-time, ~60-300s for 7B) …{RESET}")
        result = subprocess.run(
            [sys.executable, str(POC_DIR / "squish" / "convert.py"),
             "--model-dir", str(model_dir),
             "--output-dir", str(comp_dir)],
            capture_output=False,
        )
        if result.returncode != 0:
            print(f"{RED}Compression failed{RESET}", file=sys.stderr)
            sys.exit(1)
        # Trigger Tier 2 cache build (run_poc phase 5 equivalent)
        print(f"\n{CYAN}▸ Building MLX safetensors cache (Tier 2) …{RESET}")
        sys.path.insert(0, str(POC_DIR))
        from compressed_loader import load_from_npy_dir
        _, _, stats = load_from_npy_dir(
            str(comp_dir),
            model_dir=str(model_dir),
            verbose=True,
            return_stats=True,
        )
        print(f"\n{GREEN}✓ Cache built — loader: {stats.get('loader')}{RESET}")

    # ── Step 3: print summary ─────────────────────────────────────────────────
    import shutil
    model_sz = sum(f.stat().st_size for f in model_dir.rglob("*.safetensors")) / 1e9
    comp_sz  = sum(f.stat().st_size for f in comp_dir.rglob("*.npy"))  / 1e9
    tier2_f  = comp_dir / "squish_weights.safetensors"
    tier2_sz = tier2_f.stat().st_size / 1e9 if tier2_f.exists() else 0

    print(f"\n{BOLD}Pull complete — {short_name}{RESET}")
    print(f"  Original safetensors: {model_sz:.1f} GB")
    print(f"  Squish npy-dir:       {comp_sz:.1f} GB")
    print(f"  Tier 2 cache:         {tier2_sz:.1f} GB")
    print(f"\n{CYAN}▸ Start the server:{RESET}")
    print(f"  squish serve --model {short_name}")
    print(f"\n{CYAN}▸ Or run directly:{RESET}")
    print(f"  squish run \"Hello, what can you do?\"")
    print()


def cmd_list(args):
    """List compressed models available locally."""
    sources = []
    if (Path.home() / "models").exists():
        sources.append(Path.home() / "models")
    if MODELS_DIR.exists():
        sources.append(MODELS_DIR)

    print(f"\n{BOLD}Squish — Available Models{RESET}\n")
    found = False
    for src in sources:
        for d in sorted(src.iterdir()):
            if not d.is_dir() or "compressed" not in d.name:
                continue
            tier2  = d / "squish_weights.safetensors"
            ready  = (d / ".squish_ready").exists()
            status = f"{GREEN}Tier 2 ⚡⚡{RESET}" if (tier2.exists() and ready) else \
                     f"{YELLOW}Tier 1  ⚡{RESET}"  if (d / "finalized" / ".ready").exists() else \
                     f"{DIM}Tier 0  (slow){RESET}"
            sz  = sum(f.stat().st_size for f in d.rglob("*.npy")) / 1e9
            t2s = tier2.stat().st_size / 1e9 if tier2.exists() else 0
            print(f"  {d.name:<55} {status}  {sz:.1f} GB npy  {t2s:.1f} GB cache")
            found = True

    if not found:
        print(f"  {DIM}No compressed models found.{RESET}")
        print(f"  Run: {CYAN}squish pull Qwen/Qwen2.5-7B-Instruct{RESET}")
    print()


def cmd_ps(args):
    """Show running Squish server processes."""
    print(f"\n{BOLD}Squish — Running Servers{RESET}\n")
    found = False
    if SQUISH_HOME.exists():
        for pid_file in SQUISH_HOME.glob("server_*.pid"):
            port = int(pid_file.stem.split("_")[1])
            try:
                pid = int(pid_file.read_text().strip())
                # Check if process is alive
                os.kill(pid, 0)
                up_str = f"{GREEN}up{RESET}" if _is_server_up(port) else f"{YELLOW}starting{RESET}"
                print(f"  port {port:5d}  PID {pid:6d}  {up_str}")
                found = True
            except (ProcessLookupError, ValueError):
                pid_file.unlink(missing_ok=True)

    if not found:
        print(f"  {DIM}No servers running.{RESET}")
        print(f"  Start one: {CYAN}squish serve{RESET}")
    print()


def cmd_stop(args):
    """Stop a running Squish server."""
    port = args.port
    pid_file = SQUISH_HOME / f"server_{port}.pid"
    if not pid_file.exists():
        print(f"{YELLOW}No PID file for port {port}.{RESET}")
        return
    try:
        pid = int(pid_file.read_text().strip())
        os.kill(pid, signal.SIGTERM)
        pid_file.unlink(missing_ok=True)
        print(f"{GREEN}✓ Stopped server (PID {pid}) on port {port}{RESET}")
    except (ProcessLookupError, ValueError):
        pid_file.unlink(missing_ok=True)
        print(f"{YELLOW}Server was not running.{RESET}")


def cmd_git(args):
    """
    Generate a git commit message from staged changes and optionally commit.

    Drop-in replacement for the ollama-based git-commit-push-script.
    Works in any git repository.
    """
    repo_path = Path(args.path).resolve() if args.path else Path.cwd()

    # ── Get staged diff ───────────────────────────────────────────────────────
    diff_result = subprocess.run(
        ["git", "-C", str(repo_path), "diff", "--cached", "--stat"],
        capture_output=True, text=True
    )
    if diff_result.returncode != 0:
        print(f"{RED}Not a git repository or git not found.{RESET}", file=sys.stderr)
        sys.exit(1)

    stat_summary = diff_result.stdout.strip()
    if not stat_summary:
        print(f"{YELLOW}No staged changes.  Run: git add <files>{RESET}")
        sys.exit(0)

    # Full diff (truncated for speed)
    diff_full = subprocess.run(
        ["git", "-C", str(repo_path), "diff", "--cached"],
        capture_output=True, text=True
    )
    diff_text = diff_full.stdout[:3000] if diff_full.returncode == 0 else ""

    # Branch name (for context)
    branch_result = subprocess.run(
        ["git", "-C", str(repo_path), "rev-parse", "--abbrev-ref", "HEAD"],
        capture_output=True, text=True
    )
    branch = branch_result.stdout.strip() if branch_result.returncode == 0 else "main"

    # Jira/ticket detection from branch name
    ticket_match = re.search(r"([A-Z]+-\d+)", branch)
    ticket = ticket_match.group(1) if ticket_match else None

    # ── Build prompt ──────────────────────────────────────────────────────────
    prompt = f"""You are a git commit message generator.  Write a single commit message
for the following staged changes.  Rules:
- Maximum 72 characters
- Imperative mood ("Add feature" not "Added feature")
- No period at end
- No quotes, no backticks, no markdown
- Just the commit message, nothing else

Branch: {branch}
Files changed:
{stat_summary}

Diff (truncated):
{diff_text[:2000]}

Commit message:"""

    # ── Ensure server + generate ──────────────────────────────────────────────
    port = args.port
    if not _ensure_server(args.model, port):
        sys.exit(1)

    print(f"{CYAN}▸ Generating commit message…{RESET}", end="", flush=True)
    t0 = time.perf_counter()

    payload = {
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 80,
        "temperature": 0.2,
        "stream": False,
        "stop": ["\n", ".", "\"", "`"],
    }

    result = _http_post(f"{_server_url(port)}/v1/chat/completions", payload)
    elapsed = time.perf_counter() - t0

    if not result:
        print(f"\n{RED}Failed to generate commit message.{RESET}", file=sys.stderr)
        sys.exit(1)

    raw_message = result["choices"][0]["message"]["content"].strip()

    # Clean up
    commit_msg = raw_message
    for pat in [r'^"(.*)"$', r"^'(.*)'$", r"^`(.*)`$",
                r"^Commit message:\s*", r"^commit message:\s*"]:
        commit_msg = re.sub(pat, r"\1" if "(" in pat else "", commit_msg, flags=re.IGNORECASE).strip()
    # Remove trailing period
    commit_msg = commit_msg.rstrip(".")
    # Trim to 72 chars at word boundary
    if len(commit_msg) > 72:
        commit_msg = commit_msg[:72].rsplit(" ", 1)[0]

    if ticket:
        commit_msg = f"{ticket} {commit_msg}"

    print(f"\r{GREEN}✓{RESET} Generated in {elapsed:.2f}s\n")
    print(f"  {BOLD}{commit_msg}{RESET}\n")

    # ── Optionally commit ──────────────────────────────────────────────────────
    if args.yes or args.commit:
        do_commit = True
    elif args.no_commit:
        do_commit = False
    else:
        try:
            ans = input(f"  Commit with this message? [{GREEN}Y{RESET}/n] ").strip().lower()
            do_commit = ans in ("", "y", "yes")
        except (EOFError, KeyboardInterrupt):
            do_commit = False

    if do_commit:
        commit_result = subprocess.run(
            ["git", "-C", str(repo_path), "commit", "-m", commit_msg],
            capture_output=False,
        )
        if commit_result.returncode == 0:
            print(f"\n{GREEN}✓ Committed!{RESET}")
            if args.push:
                print(f"{CYAN}▸ Pushing…{RESET}")
                subprocess.run(["git", "-C", str(repo_path), "push"])
        else:
            print(f"\n{RED}Commit failed.{RESET}", file=sys.stderr)
    else:
        print(f"  {DIM}(not committed){RESET}")


def cmd_bench(args):
    """Run the full benchmark suite."""
    port = args.port
    if not _ensure_server(args.model, port):
        sys.exit(1)

    bench_script = POC_DIR / "evals" / "run_eval.py"
    os.execv(sys.executable, [
        sys.executable, str(bench_script),
        "--tasks", args.tasks,
        "--limit", str(args.limit),
        "--runs",  str(args.runs),
    ])


def cmd_version(args):
    print(f"squish 0.1.0 — {DIM}local LLM inference, no cloud required{RESET}")


# ═══════════════════════════════════════════════════════════════════════════════
#  Splash
# ═══════════════════════════════════════════════════════════════════════════════

_SPLASH = f"""{BOLD}{CYAN}
  ███████╗ ██████╗ ██╗   ██╗██╗███████╗██╗  ██╗
  ██╔════╝██╔═══██╗██║   ██║██║██╔════╝██║  ██║
  ███████╗██║   ██║██║   ██║██║███████╗███████║
  ╚════██║██║▄▄ ██║██║   ██║██║╚════██║██╔══██║
  ███████║╚██████╔╝╚██████╔╝██║███████║██║  ██║
  ╚══════╝ ╚══▀▀═╝  ╚═════╝ ╚═╝╚══════╝╚═╝  ╚═╝
{RESET}{DIM}  Local LLM inference.  No API key.  No cloud.  Free.{RESET}
"""


# ═══════════════════════════════════════════════════════════════════════════════
#  Argument parser
# ═══════════════════════════════════════════════════════════════════════════════

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="squish",
        description="Local LLM inference — drop-in OpenAI API replacement",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  squish run "What is gradient descent?"
  squish chat
  squish git                         # AI commit message for staged changes
  squish git --push                  # commit + push in one command
  squish serve --port 8000
  squish pull Qwen/Qwen2.5-7B-Instruct
  squish list
  squish ps
  squish bench --tasks arc_easy,hellaswag --runs 3
        """,
    )
    ap.add_argument("--port",    type=int, default=DEFAULT_PORT)
    ap.add_argument("--model",   default=None, help="Model name hint or path")
    ap.add_argument("--version", action="store_true")

    sub = ap.add_subparsers(dest="command")

    # run
    p_run = sub.add_parser("run", help="Single-shot generation, stream to stdout")
    p_run.add_argument("prompt", nargs="*")
    p_run.add_argument("--max-tokens", type=int, default=1024)
    p_run.add_argument("--temperature", type=float, default=0.0)
    p_run.add_argument("--port",  type=int, default=DEFAULT_PORT)
    p_run.add_argument("--model", default=None)

    # chat
    p_chat = sub.add_parser("chat", help="Interactive multi-turn chat")
    p_chat.add_argument("--system", default=None, help="System message")
    p_chat.add_argument("--max-tokens", type=int, default=2048)
    p_chat.add_argument("--temperature", type=float, default=0.7)
    p_chat.add_argument("--port",  type=int, default=DEFAULT_PORT)
    p_chat.add_argument("--model", default=None)

    # serve
    p_serve = sub.add_parser("serve", help="Start the OpenAI-compatible API server")
    p_serve.add_argument("--port",    type=int,   default=DEFAULT_PORT)
    p_serve.add_argument("--host",    default="127.0.0.1")
    p_serve.add_argument("--model",   default=None)
    p_serve.add_argument("--verbose", action="store_true")
    p_serve.add_argument("--reference", action="store_true",
                         help="Use original safetensors (slower, for comparison)")

    # pull
    p_pull = sub.add_parser("pull", help="Download + compress a model from HuggingFace")
    p_pull.add_argument("model_id")
    p_pull.add_argument("--force", action="store_true", help="Re-download even if present")

    # list
    sub.add_parser("list", help="List locally available compressed models")
    sub.add_parser("ls",   help="Alias for list")

    # ps
    sub.add_parser("ps",   help="Show running Squish server processes")

    # stop
    p_stop = sub.add_parser("stop", help="Stop a running server")
    p_stop.add_argument("--port", type=int, default=DEFAULT_PORT)

    # git
    p_git = sub.add_parser("git", help="AI-generated git commit message")
    p_git.add_argument("path", nargs="?", default=None, help="Git repo path (default: cwd)")
    p_git.add_argument("-y", "--yes", action="store_true", help="Auto-commit without prompting")
    p_git.add_argument("--commit", action="store_true",    help="Commit without pushing")
    p_git.add_argument("--push",   action="store_true",    help="Commit and push")
    p_git.add_argument("--no-commit", action="store_true", help="Print message only, don't commit")
    p_git.add_argument("--port",   type=int, default=DEFAULT_PORT)
    p_git.add_argument("--model",  default=None)

    # bench
    p_bench = sub.add_parser("bench", help="Run accuracy benchmarks")
    p_bench.add_argument("--tasks",  default="arc_easy,hellaswag,winogrande,piqa")
    p_bench.add_argument("--limit",  type=int, default=200)
    p_bench.add_argument("--runs",   type=int, default=1)
    p_bench.add_argument("--port",   type=int, default=DEFAULT_PORT)
    p_bench.add_argument("--model",  default=None)

    return ap


def main():
    ap  = build_parser()
    args = ap.parse_args()

    if args.version or (not args.command and args.version):
        cmd_version(args)
        return

    if not args.command:
        print(_SPLASH)
        ap.print_help()
        return

    dispatch = {
        "run":   cmd_run,
        "chat":  cmd_chat,
        "serve": cmd_serve,
        "pull":  cmd_pull,
        "list":  cmd_list,
        "ls":    cmd_list,
        "ps":    cmd_ps,
        "stop":  cmd_stop,
        "git":   cmd_git,
        "bench": cmd_bench,
    }

    fn = dispatch.get(args.command)
    if fn:
        fn(args)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
