#!/usr/bin/env python3
"""
bench_eoe.py — End-to-end hardware benchmark for Squish on Apple Silicon.

Measures wall-clock latency and throughput for real inference against a running
Squish server, with optional comparison against an Ollama server.

Metrics collected per (model, engine) pair
──────────────────────────────────────────
  • load_s       — cold server startup time (seconds); omitted if already warm
  • ttft_ms      — time to first token (milliseconds)
  • tps          — decode throughput (tokens / second)
  • total_toks   — total tokens generated
  • total_s      — full request wall-clock (seconds)

Results are printed as a table and optionally saved as a JSON file so they can
be pasted into doc/benchmark_* or shared in the README.

Prerequisites
─────────────
  # Start Squish on the default port
  squish serve --model qwen2.5:1.5b --port 11435

  # (Optional) Start Ollama on its default port
  ollama serve

Usage
─────
  # Squish only (default model at default port)
  python3 dev/benchmarks/bench_eoe.py

  # Compare Squish vs Ollama, N runs, save JSON
  python3 dev/benchmarks/bench_eoe.py \\
      --squish-port 11435 --squish-model squish \\
      --ollama-port 11434 --ollama-model qwen2.5:1.5b \\
      --runs 5 --output results/eoe_$(date +%Y%m%d).json

  # Quick sanity check — 2 runs, short prompt, no file output
  python3 dev/benchmarks/bench_eoe.py --runs 2 --max-tokens 64
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

# ── Terminal colour helpers ────────────────────────────────────────────────────
_USE_COLOR = sys.stdout.isatty()
G  = "\033[32m"  if _USE_COLOR else ""
Y  = "\033[33m"  if _USE_COLOR else ""
C  = "\033[36m"  if _USE_COLOR else ""
R  = "\033[31m"  if _USE_COLOR else ""
W  = "\033[1;37m" if _USE_COLOR else ""
D  = "\033[2m"   if _USE_COLOR else ""
NC = "\033[0m"   if _USE_COLOR else ""


def _hdr(title: str) -> None:
    print(f"\n{W}{'─' * 64}{NC}")
    print(f"{C}  {title}{NC}")
    print(f"{W}{'─' * 64}{NC}")


def _row(label: str, value: str, note: str = "") -> None:
    print(f"  {label:<40} {G}{value:>14}{NC}  {D}{note}{NC}")


def _warn(msg: str) -> None:
    print(f"  {Y}WARN{NC}  {msg}", file=sys.stderr)


def _die(msg: str) -> None:
    print(f"  {R}ERROR{NC}  {msg}", file=sys.stderr)
    sys.exit(1)


# ── HTTP helpers ───────────────────────────────────────────────────────────────

def _chat_stream(
    base_url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    temperature: float = 0.0,
    timeout: float = 120.0,
    api_key: str = "squish",
) -> dict[str, Any]:
    """
    Send a streaming /v1/chat/completions request; return timing stats.

    Returns
    -------
    dict with keys:
        ttft_ms   – time to first token chunk (ms)
        total_s   – wall-clock for complete response
        total_toks – approximate token count (len(text) // 4 if no usage field)
        tps        – tokens/second (total_toks / decode_seconds)
        text       – full generated text
    """
    payload = json.dumps({
        "model":       model,
        "messages":    [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens":  max_tokens,
        "stream":      True,
    }).encode()

    req = urllib.request.Request(
        f"{base_url}/v1/chat/completions",
        data=payload,
        headers={
            "Content-Type":  "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )

    t_start = time.perf_counter()
    ttft: float | None = None
    text_chunks: list[str] = []
    total_toks: int = 0

    with urllib.request.urlopen(req, timeout=timeout) as resp:
        for raw_line in resp:
            line = raw_line.decode("utf-8").strip()
            if not line.startswith("data:"):
                continue
            data_str = line[len("data:"):].strip()
            if data_str == "[DONE]":
                break
            try:
                chunk = json.loads(data_str)
            except json.JSONDecodeError:
                continue

            delta = chunk.get("choices", [{}])[0].get("delta", {})
            content = delta.get("content", "")
            if content:
                if ttft is None:
                    ttft = (time.perf_counter() - t_start) * 1000.0
                text_chunks.append(content)

            # Prefer server-reported usage when available
            usage = chunk.get("usage") or {}
            if usage.get("completion_tokens"):
                total_toks = int(usage["completion_tokens"])

    t_end = time.perf_counter()
    total_s = t_end - t_start

    full_text = "".join(text_chunks)
    if total_toks == 0:
        # Rough fallback: 1 token ≈ 4 chars
        total_toks = max(1, len(full_text) // 4)

    decode_s = max(total_s - (ttft or 0) / 1000.0, 0.001)
    tps = total_toks / decode_s

    return {
        "ttft_ms":    ttft or (total_s * 1000.0),
        "total_s":    total_s,
        "total_toks": total_toks,
        "tps":        tps,
        "text":       full_text,
    }


def _is_server_up(base_url: str, timeout: float = 30.0) -> bool:
    try:
        urllib.request.urlopen(f"{base_url}/health", timeout=timeout)
        return True
    except Exception:
        pass
    # Fallback: models endpoint
    try:
        urllib.request.urlopen(f"{base_url}/v1/models", timeout=timeout)
        return True
    except Exception:
        return False


# ── Benchmark runner ───────────────────────────────────────────────────────────

_PROMPTS: list[str] = [
    "Explain the difference between a transformer and an RNN in one paragraph.",
    "Write a Python function that returns the nth Fibonacci number.",
    "What are the main causes of the French Revolution? Answer in 3 sentences.",
]


def _run_suite(
    label: str,
    base_url: str,
    model: str,
    runs: int,
    max_tokens: int,
    api_key: str = "squish",
) -> list[dict[str, Any]]:
    """Run `runs` inference calls and return raw per-run stats."""
    _hdr(f"{label}  ({model})")

    if not _is_server_up(base_url):
        _warn(f"server at {base_url} is not responding — skipping")
        return []

    results: list[dict[str, Any]] = []
    prompt = _PROMPTS[0]   # single representative prompt for comparability

    # Warm up the model (first generation triggers Metal JIT; discard result)
    print(f"  {D}Warming up…{NC}", end="", flush=True)
    try:
        _chat_stream(base_url, model, "Say hi.", 8, api_key=api_key)
        print(f"\r  {D}Warmup done.{NC}            ")
    except Exception as exc:
        print()
        _warn(f"warmup failed: {exc}")

    for i in range(runs):
        try:
            stats = _chat_stream(base_url, model, prompt, max_tokens, api_key=api_key)
        except Exception as exc:
            _warn(f"run {i+1} failed: {exc}")
            continue

        results.append(stats)
        _row(
            f"run {i+1:02d}",
            f"{stats['tps']:.1f} tok/s",
            f"TTFT {stats['ttft_ms']:.0f}ms  {stats['total_toks']} toks",
        )

    if not results:
        return results

    ttfts = [r["ttft_ms"]  for r in results]
    tpses = [r["tps"]      for r in results]

    print()
    _row("mean TTFT",      f"{statistics.mean(ttfts):.0f} ms",
         f"± {statistics.stdev(ttfts):.0f} ms" if len(ttfts) > 1 else "")
    _row("mean throughput", f"{statistics.mean(tpses):.1f} tok/s",
         f"± {statistics.stdev(tpses):.1f}" if len(tpses) > 1 else "")
    _row("best TTFT",      f"{min(ttfts):.0f} ms", "")
    _row("peak throughput", f"{max(tpses):.1f} tok/s", "")

    return results


# ── Summary table ──────────────────────────────────────────────────────────────

def _print_summary(all_results: dict[str, list[dict]]) -> None:
    _hdr("Summary")
    hdr_fmt = f"  {'Engine':<22} {'Mean TTFT':>10} {'Mean tok/s':>12} {'Runs':>6}"
    print(f"{W}{hdr_fmt}{NC}")
    print(f"  {'─'*22} {'─'*10} {'─'*12} {'─'*6}")
    for label, runs in all_results.items():
        if not runs:
            print(f"  {label:<22} {'—':>10} {'—':>12} {'0':>6}")
            continue
        ttfts = [r["ttft_ms"] for r in runs]
        tpses = [r["tps"]     for r in runs]
        print(
            f"  {label:<22} "
            f"{statistics.mean(ttfts):>9.0f}ms "
            f"{statistics.mean(tpses):>12.1f} "
            f"{len(runs):>6}"
        )


# ── Entry point ────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="End-to-end benchmark: Squish (and optionally Ollama) on Apple Silicon",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    p.add_argument("--squish-port",  type=int, default=11435,
                   help="Squish server port (default: 11435)")
    p.add_argument("--squish-model", default="squish",
                   help="Model name to use for Squish requests (default: squish)")
    p.add_argument("--squish-key",   default="squish",
                   help="API key for Squish server (default: squish)")
    p.add_argument("--ollama-port",  type=int, default=0,
                   help="Ollama server port for comparison (0 = disabled, default: 0)")
    p.add_argument("--ollama-model", default="qwen2.5:1.5b",
                   help="Ollama model name (default: qwen2.5:1.5b)")
    p.add_argument("--runs",        type=int, default=5,
                   help="Number of inference runs per engine (default: 5)")
    p.add_argument("--max-tokens",  type=int, default=256,
                   help="Max tokens to generate per run (default: 256)")
    p.add_argument("--output",      default="",
                   help="Save raw JSON results to this path (e.g. results/eoe.json)")
    return p


def main() -> None:
    args = _build_parser().parse_args()

    squish_url = f"http://127.0.0.1:{args.squish_port}"
    all_results: dict[str, list[dict]] = {}

    # ── Squish ──
    squish_runs = _run_suite(
        label="Squish",
        base_url=squish_url,
        model=args.squish_model,
        runs=args.runs,
        max_tokens=args.max_tokens,
        api_key=args.squish_key,
    )
    all_results["Squish"] = squish_runs

    # ── Ollama (optional) ──
    if args.ollama_port > 0:
        ollama_url = f"http://127.0.0.1:{args.ollama_port}"
        ollama_runs = _run_suite(
            label="Ollama",
            base_url=ollama_url,
            model=args.ollama_model,
            runs=args.runs,
            max_tokens=args.max_tokens,
        )
        all_results["Ollama"] = ollama_runs

        # Speed-up ratio
        if squish_runs and ollama_runs:
            sq_tps  = statistics.mean(r["tps"]      for r in squish_runs)
            ol_tps  = statistics.mean(r["tps"]      for r in ollama_runs)
            sq_ttft = statistics.mean(r["ttft_ms"]  for r in squish_runs)
            ol_ttft = statistics.mean(r["ttft_ms"]  for r in ollama_runs)
            _hdr("Squish vs Ollama")
            _row("throughput ratio", f"{sq_tps / ol_tps:.2f}×",
                 "Squish / Ollama  (>1 = Squish faster)")
            _row("TTFT ratio",       f"{ol_ttft / sq_ttft:.2f}×",
                 "Ollama / Squish  (>1 = Squish lower TTFT)")

    _print_summary(all_results)

    # ── Save JSON ──
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {
            "squish_port":  args.squish_port,
            "squish_model": args.squish_model,
            "ollama_port":  args.ollama_port,
            "ollama_model": args.ollama_model,
            "runs":         args.runs,
            "max_tokens":   args.max_tokens,
            "results":      {k: v for k, v in all_results.items()},
        }
        # Compute aggregate stats per engine for convenience
        payload["summary"] = {}
        for label, runs in all_results.items():
            if not runs:
                payload["summary"][label] = {}
                continue
            ttfts = [r["ttft_ms"] for r in runs]
            tpses = [r["tps"]     for r in runs]
            payload["summary"][label] = {
                "ttft_mean_ms":  round(statistics.mean(ttfts), 1),
                "ttft_stdev_ms": round(statistics.stdev(ttfts), 1) if len(ttfts) > 1 else 0.0,
                "tps_mean":      round(statistics.mean(tpses), 2),
                "tps_stdev":     round(statistics.stdev(tpses), 2) if len(tpses) > 1 else 0.0,
                "n_runs":        len(runs),
            }
        out_path.write_text(json.dumps(payload, indent=2))
        print(f"\n  {G}Saved → {out_path}{NC}")


if __name__ == "__main__":
    main()
