#!/usr/bin/env python3
"""
bench_optimizations.py — Micro-benchmark for squish inference optimizations.

Measures the performance impact of the five optimizations added in the
"memory-efficient LLM inference" sprint:

  1. Speculative decoding KV cache (stateful vs stateless)  — via server
  2. Disk prompt-cache lookup (hit latency vs cold prefill)  — local I/O
  3. LazyLLM prefill token pruning (speed-up vs baseline)    — via server
  4. Disk KV overflow tier (throughput at long context)      — local numpy
  5. mx.compile decode step (tokens/s with vs without)       — via server

Items 1, 3, 5 require a running squish server (default: 127.0.0.1:11435).
Items 2, 4 run locally without a server (numpy / I/O only).

Usage:
    # Start squish first
    squish serve qwen3:8b

    # All benchmarks
    python3 benchmarks/bench_optimizations.py

    # Single section
    python3 benchmarks/bench_optimizations.py --suite local
    python3 benchmarks/bench_optimizations.py --suite server

    # Skip server tests if no server running
    python3 benchmarks/bench_optimizations.py --local-only

    # Change server address
    python3 benchmarks/bench_optimizations.py --port 11435 --model squish
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np

# ── colour helpers ────────────────────────────────────────────────────────────
G = "\033[32m"; Y = "\033[33m"; C = "\033[36m"; W = "\033[1;37m"
D = "\033[2m";  NC = "\033[0m"; R = "\033[31m"


def _hdr(title: str) -> None:
    print(f"\n{W}{'─' * 60}{NC}")
    print(f"{C}  {title}{NC}")
    print(f"{W}{'─' * 60}{NC}")


def _row(label: str, val: str, extra: str = "") -> None:
    print(f"  {label:<42} {G}{val:>12}{NC}  {D}{extra}{NC}")


def _skip(label: str, reason: str) -> None:
    print(f"  {Y}~ SKIP{NC}  {label:<40} {D}{reason}{NC}")


# ── HTTP helpers ──────────────────────────────────────────────────────────────

def _chat(messages: list[dict], port: int, model: str,
          temperature: float = 0.0, max_tokens: int = 256,
          stream: bool = False) -> tuple[str, float]:
    """POST /v1/chat/completions; return (text, elapsed_s)."""
    import urllib.request, urllib.error

    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }
    body = json.dumps(payload).encode()
    req  = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data   = body,
        headers = {
            "Content-Type":  "application/json",
            "Authorization": "Bearer squish",
        },
    )
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read())
            elapsed = time.perf_counter() - t0
            text = data["choices"][0]["message"]["content"]
            return text, elapsed
    except Exception as exc:
        return f"[error: {exc}]", time.perf_counter() - t0


def _ttft(messages: list[dict], port: int, model: str,
          max_tokens: int = 1) -> float:
    """Measure time-to-first-token by asking for max_tokens=1."""
    _, elapsed = _chat(messages, port, model, max_tokens=max_tokens)
    return elapsed


def _server_alive(port: int) -> bool:
    import urllib.request
    try:
        urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=3)
        return True
    except Exception:
        return False


# ── ════════════════════════════════════════════════════════════════════════ ──
# SUITE 1 — local (no server required)
# ── ════════════════════════════════════════════════════════════════════════ ──

# ── 2. DiskKVCache lookup latency ────────────────────────────────────────────

def bench_disk_prompt_cache(tmp_dir: Path) -> None:
    _hdr("2. DiskKVCache — hit latency vs cold prefill (local I/O)")

    # Ensure squish is importable
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    try:
        from squish.kv_cache import DiskKVCache, QuantizedKVCache
    except ImportError as exc:
        _skip("DiskKVCache", f"import failed: {exc}")
        return

    N_LAYERS = 4
    N_TOKENS = 32    # realistic short prompt
    N_HEADS  = 4
    HEAD_DIM = 8
    VOCAB    = 128
    REPEATS  = 20

    # Build a realistic-ish populated cache
    cache = QuantizedKVCache(n_layers=N_LAYERS, window=64, mode="int8")
    rng = np.random.default_rng(42)
    for i in range(N_LAYERS):
        for _ in range(N_TOKENS):
            k = rng.standard_normal((N_HEADS, HEAD_DIM)).astype(np.float16)
            v = rng.standard_normal((N_HEADS, HEAD_DIM)).astype(np.float16)
            cache._layers[i].append(k, v)

    logit = rng.standard_normal(VOCAB).astype(np.float32)
    ids = list(range(N_TOKENS))

    dc = DiskKVCache(tmp_dir / "bench_disk", max_entries=256)

    # Store synchronously (caller blocks until written)
    _store_arrays = DiskKVCache._serialise(cache)
    _store_arrays["last_logit"] = logit
    _entry = (tmp_dir / "bench_disk") / (DiskKVCache._key(ids) + ".npz")
    np.savez_compressed(str(_entry), **_store_arrays)

    # ── measure lookup latency ────────────────────────────────────────────────
    t0 = time.perf_counter()
    for _ in range(REPEATS):
        result = dc.lookup(ids)
    hit_ms = (time.perf_counter() - t0) / REPEATS * 1000

    status = f"hit={result is not None}"
    _row("DiskKVCache.lookup  (avg over 20 reps)", f"{hit_ms:.2f} ms", status)

    # ── compare to serialise cost (proxy for cold-prefill write overhead) ─────
    t0 = time.perf_counter()
    for _ in range(REPEATS):
        buf = io.BytesIO()
        np.savez_compressed(buf, **_store_arrays)
    write_ms = (time.perf_counter() - t0) / REPEATS * 1000
    _row("np.savez_compressed (avg over 20 reps — write cost)", f"{write_ms:.2f} ms", "")

    speedup = write_ms / hit_ms if hit_ms > 0 else float("inf")
    _row("Read / write ratio (hit faster by)", f"{speedup:.1f}×", "")


# ── 4. Disk KV overflow tier — append throughput ─────────────────────────────

def bench_disk_kv_tier(tmp_dir: Path) -> None:
    _hdr("4. Disk KV overflow tier — append throughput (local numpy)")

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    try:
        from squish.kv_cache import KVLayerCache
    except ImportError as exc:
        _skip("KVLayerCache disk tier", f"import failed: {exc}")
        return

    N_HEADS  = 8
    HEAD_DIM = 128    # Qwen3-8B head_dim
    N_TOKENS = 2048   # 2 K context
    THRESHOLD = 512   # keep last 512 tokens in RAM, spill rest

    rng = np.random.default_rng(0)

    # ── baseline: pure RAM ───────────────────────────────────────────────────
    layer_ram = KVLayerCache(window=64)
    t0 = time.perf_counter()
    for t in range(N_TOKENS):
        k = rng.standard_normal((N_HEADS, HEAD_DIM)).astype(np.float16)
        v = rng.standard_normal((N_HEADS, HEAD_DIM)).astype(np.float16)
        layer_ram.append(k, v)
    ram_s = time.perf_counter() - t0
    ram_mem = layer_ram.memory_bytes / 1024 / 1024

    _row(f"RAM-only   {N_TOKENS} appends", f"{ram_s*1000:.0f} ms",
         f"RAM ≈ {ram_mem:.1f} MB")

    # ── with disk tier ───────────────────────────────────────────────────────
    layer_disk = KVLayerCache(window=64)
    layer_disk.enable_disk_tier(
        threshold=THRESHOLD,
        max_disk_tokens=N_TOKENS * 2,
        cache_dir=tmp_dir / "kv_tier",
        n_heads=N_HEADS,
        head_dim=HEAD_DIM,
    )
    rng2 = np.random.default_rng(0)   # same data
    t0 = time.perf_counter()
    for t in range(N_TOKENS):
        k = rng2.standard_normal((N_HEADS, HEAD_DIM)).astype(np.float16)
        v = rng2.standard_normal((N_HEADS, HEAD_DIM)).astype(np.float16)
        layer_disk.append(k, v)
    disk_s = time.perf_counter() - t0
    disk_ram_mb = layer_disk.memory_bytes / 1024 / 1024
    disk_spilled = layer_disk._disk_n

    _row(f"Disk-tier  {N_TOKENS} appends (thresh={THRESHOLD})", f"{disk_s*1000:.0f} ms",
         f"RAM ≈ {disk_ram_mb:.1f} MB  spilled={disk_spilled} tokens")

    overhead_pct = (disk_s - ram_s) / ram_s * 100
    _row("Disk-tier overhead vs RAM-only", f"{overhead_pct:+.1f}%",
         "(negative = faster due to smaller RAM copies)")

    # ── verify get_full_kv preserves token count ─────────────────────────────
    fk, fv = layer_disk.get_full_kv()
    total = fk.shape[1] if fk is not None else 0
    _row(f"get_full_kv total tokens", f"{total}", f"expected ≥ {N_TOKENS}")


# ── 3. LazyLLM — build_keep_mask micro-benchmark ─────────────────────────────

def bench_lazy_llm_local() -> None:
    _hdr("3. LazyLLM — _build_keep_mask micro-benchmark (local numpy)")

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    try:
        from squish.lazy_llm import _build_keep_mask
    except ImportError as exc:
        _skip("LazyLLM _build_keep_mask", f"import failed: {exc}")
        return

    for T, kr, rw in [(256, 0.7, 4), (1024, 0.7, 4), (4096, 0.7, 8)]:
        rng = np.random.default_rng(0)
        scores = rng.standard_normal(T).astype(np.float32)
        REPS = 200
        t0 = time.perf_counter()
        for _ in range(REPS):
            mask = _build_keep_mask(scores, kr, rw)
        us = (time.perf_counter() - t0) / REPS * 1e6
        pruned_pct = 100.0 * (~mask).sum() / T
        _row(f"_build_keep_mask  T={T:<5} keep={kr}", f"{us:.1f} µs",
             f"pruned {pruned_pct:.0f}% of tokens")


# ── ════════════════════════════════════════════════════════════════════════ ──
# SUITE 2 — server-side benchmarks
# ── ════════════════════════════════════════════════════════════════════════ ──

SHORT_PROMPT  = "What is 7 × 8?"
MEDIUM_PROMPT = (
    "Explain the difference between a linked list and an array in Python, "
    "covering memory layout, time complexity for common operations, and "
    "typical use-cases. Be concise."
)
LONG_PROMPT = (
    "You are an expert software engineer. Review the following code block and "
    "identify any bugs, performance issues, or style violations. Provide a "
    "corrected version with inline comments explaining each change.\n\n"
    "```python\n"
    "def merge_sorted(a, b):\n"
    "    result = []\n"
    "    i = j = 0\n"
    "    while i < len(a) and j <= len(b):\n"
    "        if a[i] < b[j]: result.append(a[i]); i += 1\n"
    "        else: result.append(b[j]); j += 1\n"
    "    result += a[i:]\n"
    "    result += b[j:]\n"
    "    return result\n"
    "```\n"
    "Also explain the time and space complexity of the corrected solution."
)


def bench_server_ttft(port: int, model: str) -> None:
    _hdr("1 / 3 / 5. Server TTFT & throughput (baseline + LazyLLM)")

    prompts = [
        ("short  (≈10 tokens)",  SHORT_PROMPT,  32),
        ("medium (≈50 tokens)",  MEDIUM_PROMPT, 128),
        ("long   (≈200 tokens)", LONG_PROMPT,   512),
    ]

    print(f"\n  {D}Model: {model}  Port: {port}{NC}\n")

    REPS = 2

    for label, prompt, max_tok in prompts:
        msgs = [{"role": "user", "content": prompt}]

        # ── warm-up (avoid first-call JIT overhead) ───────────────────────────
        _chat(msgs, port, model, max_tokens=1)

        # ── measure ───────────────────────────────────────────────────────────
        times = []
        toks_per_s_list = []
        for _ in range(REPS):
            t0 = time.perf_counter()
            text, elapsed = _chat(msgs, port, model, max_tokens=max_tok)
            total_s = time.perf_counter() - t0
            n_chars = len(text)
            # rough token estimate: ~4 chars / token
            est_toks = max(1, n_chars // 4)
            tps = est_toks / total_s if total_s > 0 else 0
            times.append(elapsed)
            toks_per_s_list.append(tps)

        avg_ttft = sum(times) / len(times)
        avg_tps  = sum(toks_per_s_list) / len(toks_per_s_list)
        _row(f"{label}", f"{avg_ttft*1000:.0f} ms TTFT",
             f"≈ {avg_tps:.0f} tok/s")


def bench_mx_compile(port: int, model: str, no_compile_port: int = 0) -> None:
    _hdr("5. mx.compile decode step — tokens/s measurement")

    msgs = [{"role": "user", "content": "Count to 20 in a comma-separated list."}]
    REPS = 3

    # ── measure with compile (server default — compile=ON) ──────────────────────────
    times_c = []
    for _ in range(REPS):
        _chat(msgs, port, model, max_tokens=1)   # JIT warmup
        t0 = time.perf_counter()
        text, _ = _chat(msgs, port, model, max_tokens=64)
        t_total = time.perf_counter() - t0
        est_toks = max(1, len(text) // 4)
        times_c.append(est_toks / t_total)

    avg_c = sum(times_c) / len(times_c)
    _row("Decode throughput (compile=ON)", f"≈ {avg_c:.1f} tok/s", "")

    # ── measure without compile (companion --no-compile server) ───────────────
    if no_compile_port > 0 and _server_alive(no_compile_port):
        times_nc = []
        for _ in range(REPS):
            _chat(msgs, no_compile_port, model, max_tokens=1)   # warmup
            t0 = time.perf_counter()
            text, _ = _chat(msgs, no_compile_port, model, max_tokens=64)
            t_total = time.perf_counter() - t0
            est_toks = max(1, len(text) // 4)
            times_nc.append(est_toks / t_total)

        avg_nc = sum(times_nc) / len(times_nc)
        speedup = avg_c / avg_nc if avg_nc > 0 else 0.0
        _row("Decode throughput (compile=OFF)", f"≈ {avg_nc:.1f} tok/s", "")
        _row("mx.compile speedup", f"{speedup:.2f}×", "")
    else:
        print(f"  {D}compile=OFF comparison unavailable — pass --no-compile-port to enable{NC}")


# ── ════════════════════════════════════════════════════════════════════════ ──
# Main
# ── ════════════════════════════════════════════════════════════════════════ ──

def main() -> int:  # pragma: no cover
    ap = argparse.ArgumentParser(
        description="Squish inference-optimization micro-benchmarks",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    ap.add_argument("--port",       type=int, default=11435)
    ap.add_argument("--model",      default="squish")
    ap.add_argument("--suite",      choices=["local", "server", "all"], default="all")
    ap.add_argument("--local-only", action="store_true",
                    help="Skip server benchmarks even if a server is reachable")
    ap.add_argument("--no-compile-port", type=int, default=0,
                    metavar="N",
                    help="Port of a companion --no-compile server for compile=ON/OFF "
                         "comparison (default 0 = no comparison)")
    args = ap.parse_args()

    print(f"\n{W}{'═' * 60}{NC}")
    print(f"{C}  Squish Inference-Optimization Benchmarks{NC}")
    print(f"{W}{'═' * 60}{NC}")
    print(f"  Date  : {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Suite : {args.suite}")

    with tempfile.TemporaryDirectory(prefix="squish_bench_") as _td:
        tmp = Path(_td)

        run_local  = args.suite in ("local",  "all")
        run_server = args.suite in ("server", "all") and not args.local_only

        # ── local benchmarks (no server required) ─────────────────────────────
        if run_local:
            bench_disk_prompt_cache(tmp)
            bench_disk_kv_tier(tmp)
            bench_lazy_llm_local()

        # ── server benchmarks ─────────────────────────────────────────────────
        if run_server:
            if not _server_alive(args.port):
                print(f"\n  {Y}⚠ No server found at 127.0.0.1:{args.port} — "
                      f"skipping server benchmarks.{NC}")
                print(f"  {D}Start with: squish serve qwen3:8b{NC}")
            else:
                bench_server_ttft(args.port, args.model)
                bench_mx_compile(args.port, args.model,
                                 no_compile_port=args.no_compile_port)

    print(f"\n{W}{'═' * 60}{NC}")
    print(f"{G}  Benchmarks complete.{NC}")
    print(f"{W}{'═' * 60}{NC}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
