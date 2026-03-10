#!/usr/bin/env python3
"""
_run_all.py — Start the squish server and run all benchmarks sequentially.

Avoids shell heredocs/dquotes by driving everything from Python.
Results are printed to stdout and tee'd to /tmp/squish_bench_results.txt

Server logs → ~/.squish/squish.log
Server PID  → ~/.squish/squish.pid
"""
from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
import urllib.request
import urllib.error

ROOT              = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR         = os.path.join(ROOT, "models", "Qwen3-8B-mlx-int4")
PORT              = 11435
NO_COMPILE_PORT   = PORT + 1   # companion server for compile=ON vs OFF comparison
API_KEY           = "squish"
LOG_FILE    = "/tmp/squish_bench_results.txt"

SQUISH_DIR  = os.path.expanduser("~/.squish")
SERVER_LOG  = os.path.join(SQUISH_DIR, "squish.log")
SERVER_PID  = os.path.join(SQUISH_DIR, "squish.pid")

BENCH_DIR   = os.path.join(ROOT, "benchmarks")

# ── helpers ───────────────────────────────────────────────────────────────────

def tee(msg: str) -> None:
    print(msg, flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(msg + "\n")

def wait_for_server(timeout: int = 120, port: int = PORT) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = urllib.request.urlopen(f"http://localhost:{port}/health", timeout=3)
            if r.status == 200:
                return True
        except Exception:
            pass
        time.sleep(2)
    return False

def run_bench(script: str, extra_args: list[str]) -> int:
    """Run a benchmark script as a subprocess, streaming output."""
    cmd = [sys.executable, "-u", os.path.join(BENCH_DIR, script)]
    cmd.extend(extra_args)
    tee(f"\n{'='*70}")
    tee(f"RUNNING: {' '.join(cmd)}")
    tee(f"{'='*70}")
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    proc = subprocess.run(cmd, env=env)
    return proc.returncode

# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    # Truncate results log
    with open(LOG_FILE, "w") as f:
        f.write(f"Squish Benchmark Run — {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 70 + "\n\n")

    os.makedirs(SQUISH_DIR, exist_ok=True)
    tee(f"Starting squish server (mlx INT4, no-compile for memory safety)…")
    tee(f"Server log  → {SERVER_LOG}")

    server_env = os.environ.copy()
    server_env["SQUISH_API_KEY"] = API_KEY
    server_env["PYTHONUNBUFFERED"] = "1"

    server_cmd = [
        sys.executable, "-u", "-m", "squish.server",
        "--mlx-model-dir", MODEL_DIR,
        "--port", str(PORT),
        "--kv-cache-mode", "fp16",
        "--log-level", "info",
    ]

    # Optional verbose inference tracing: SQUISH_TRACE=1 ./benchmarks/_run_all.py
    if os.environ.get("SQUISH_TRACE"):
        _trace_log = os.path.join(SQUISH_DIR, "squish_trace.log")
        server_cmd += ["--trace", "--trace-file", _trace_log]
        tee(f"Trace log   → {_trace_log}")

    _srv_log = open(SERVER_LOG, "w", buffering=1)  # line-buffered
    srv = subprocess.Popen(
        server_cmd,
        cwd=ROOT,
        env=server_env,
        stdout=_srv_log,
        stderr=_srv_log,
    )
    # Write PID file
    with open(SERVER_PID, "w") as _pf:
        _pf.write(str(srv.pid))

    tee(f"Server PID: {srv.pid}  — waiting for health check (up to 90s)…")

    # Tail the server log to stdout so startup errors are visible in real-time
    _tail_stop = threading.Event()
    def _tail_log() -> None:
        try:
            with open(SERVER_LOG, "r") as _lf:
                while not _tail_stop.is_set():
                    line = _lf.readline()
                    if line:
                        print(f"[server] {line}", end="", flush=True)
                    else:
                        time.sleep(0.1)
        except Exception:
            pass
    _tailer = threading.Thread(target=_tail_log, daemon=True)
    _tailer.start()

    if not wait_for_server(timeout=120):
        _tail_stop.set()
        tee("ERROR: server did not start in 120s — aborting")
        tee(f"Check server log for details: {SERVER_LOG}")
        # Print last 40 lines of log for immediate context
        try:
            with open(SERVER_LOG) as _lf:
                lines = _lf.readlines()
            tee("--- Last 40 lines of server log ---")
            for _l in lines[-40:]:
                tee(_l.rstrip())
            tee("--- end ---")
        except Exception:
            pass
        srv.terminate()
        sys.exit(1)

    _tail_stop.set()
    tee("Server is ready.\n")

    results: dict[str, int] = {}

    # 1. Local-only optimization benchmarks (no heavy inference, runs fast)
    results["bench_optimizations_local"] = run_bench(
        "bench_optimizations.py",
        ["--port", str(PORT), "--suite", "local"],
    )

    # 2. Warmup: single inference to trigger Metal JIT
    tee("\n-- JIT warmup (first inference compiles Metal kernels) --")
    try:
        import json as _json
        payload = _json.dumps({
            "model": "squish",
            "messages": [{"role": "user", "content": "hi /no_think"}],
            "max_tokens": 8,
            "stream": False,
        }).encode()
        req = urllib.request.Request(
            f"http://localhost:{PORT}/v1/chat/completions",
            data=payload,
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {API_KEY}"},
        )
        t0 = time.perf_counter()
        resp = urllib.request.urlopen(req, timeout=300)
        body = _json.loads(resp.read())
        warmup_s = time.perf_counter() - t0
        content = body["choices"][0]["message"]["content"]
        tee(f"Warmup done in {warmup_s:.1f}s — response: {content!r}")
    except Exception as e:
        tee(f"Warmup failed: {e}")

    # Start no-compile companion server for compile=ON vs OFF comparison (Phase 0)
    no_compile_srv = None
    _nc_log_path = os.path.join(SQUISH_DIR, "squish_nc.log")
    try:
        tee(f"\n-- Starting no-compile companion server on port {NO_COMPILE_PORT} --")
        nc_env = os.environ.copy()
        nc_env["SQUISH_API_KEY"] = API_KEY
        nc_env["PYTHONUNBUFFERED"] = "1"
        nc_cmd = [
            sys.executable, "-u", "-m", "squish.server",
            "--mlx-model-dir", MODEL_DIR,
            "--port", str(NO_COMPILE_PORT),
            "--no-compile",
            "--kv-cache-mode", "fp16",
            "--log-level", "warning",
        ]
        _nc_log_f = open(_nc_log_path, "w", buffering=1)
        no_compile_srv = subprocess.Popen(
            nc_cmd, cwd=ROOT, env=nc_env,
            stdout=_nc_log_f, stderr=_nc_log_f,
        )
        if wait_for_server(timeout=120, port=NO_COMPILE_PORT):
            tee(f"No-compile server ready on port {NO_COMPILE_PORT}.")
        else:
            tee("Warning: no-compile server did not start in time — compile comparison skipped.")
            no_compile_srv.terminate()
            no_compile_srv = None
    except Exception as _nc_err:
        tee(f"Warning: could not start no-compile server: {_nc_err}")
        no_compile_srv = None

    # 3. Agent capability benchmark (tools + reasoning + agentic)
    results["bench_agent_8b"] = run_bench(
        "bench_agent_8b.py",
        ["--port", str(PORT), "--suite", "all",
         "--timeout", "120", "--no-think"],
    )

    # 4. Commit-quality benchmark (1 round, fast)
    results["bench_commit"] = run_bench(
        "bench_commit.py",
        ["--port", str(PORT), "--models", "squish",
         "--rounds", "1", "--timeout", "120", "--no-think"],
    )

    # 5. Server-side optimization benchmarks (requires running server)
    _opt_server_args = ["--port", str(PORT), "--suite", "server"]
    if no_compile_srv is not None:
        _opt_server_args += ["--no-compile-port", str(NO_COMPILE_PORT)]
    results["bench_optimizations_server"] = run_bench(
        "bench_optimizations.py",
        _opt_server_args,
    )

    # ── Summary ───────────────────────────────────────────────────────────────
    tee("\n" + "="*70)
    tee("BENCHMARK SUITE COMPLETE")
    tee("="*70)
    for name, rc in results.items():
        status = "OK" if rc == 0 else f"FAILED (rc={rc})"
        tee(f"  {name:<40} {status}")
    tee(f"\nFull output: {LOG_FILE}")

    if no_compile_srv is not None:
        tee("Stopping no-compile companion server…")
        no_compile_srv.terminate()
        try:
            no_compile_srv.wait(timeout=10)
        except subprocess.TimeoutExpired:
            no_compile_srv.kill()

    tee("\nStopping server…")
    srv.terminate()
    try:
        srv.wait(timeout=10)
    except subprocess.TimeoutExpired:
        srv.kill()
    _srv_log.close()
    tee(f"Server log saved to: {SERVER_LOG}")
    tee("Done.")


if __name__ == "__main__":
    main()
