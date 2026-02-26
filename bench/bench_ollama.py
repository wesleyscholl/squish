#!/usr/bin/env python3
"""
bench_ollama.py

Measure Ollama cold-load time for qwen2.5:7b on identical hardware.
Produces mean ± stddev across N runs suitable for arXiv comparison table.

Cold load definition: time from request initiation (model NOT in memory)
to receipt of the first response token. This is directly comparable to
Squish's "load time" metric (time to model-ready state).

Methodology:
    1. Unload model from memory between every run (ensures cold state)
    2. POST to /api/generate with stream=true, timing from request
       start to first "done=false" chunk
    3. Record N timings, compute mean ± stddev

Usage:
    # Ensure Ollama is running:  ollama serve
    # Ensure model is pulled:   ollama pull qwen2.5:7b
    python3 bench_ollama.py [--model qwen2.5:7b] [--runs 10]
"""
import argparse
import json
import statistics
import subprocess
import sys
import time
import urllib.request
import urllib.error

OLLAMA_BASE = "http://localhost:11434"
DEFAULT_MODEL = "qwen2.5:7b"
DEFAULT_RUNS  = 10
WARMUP_PROMPT = "hello"


def ollama_unload(model: str) -> None:
    """Unload a model from Ollama's in-memory cache."""
    # Ollama unloads a model by setting keep_alive=0 on a generate/chat call
    # POST /api/generate with keep_alive=0 forces eviction
    payload = json.dumps({
        "model":      model,
        "prompt":     "",
        "keep_alive": 0,
        "stream":     False,
    }).encode()
    req = urllib.request.Request(
        f"{OLLAMA_BASE}/api/generate",
        data    = payload,
        headers = {"Content-Type": "application/json"},
        method  = "POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            resp.read()
    except Exception:
        pass  # model may not be loaded; that's fine


def time_cold_load(model: str) -> float:
    """
    Unload model, then time from request start to first token response.

    Returns seconds as float.
    """
    ollama_unload(model)
    time.sleep(0.5)  # brief pause to ensure eviction is complete

    payload = json.dumps({
        "model":  model,
        "prompt": WARMUP_PROMPT,
        "stream": True,
        "options": {
            "num_predict": 1,    # generate exactly 1 token — minimises generation time
            "temperature": 0.0,
        },
    }).encode()
    req = urllib.request.Request(
        f"{OLLAMA_BASE}/api/generate",
        data    = payload,
        headers = {"Content-Type": "application/json"},
        method  = "POST",
    )

    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=120) as resp:
        # Read until we get the first non-empty chunk with response text
        while True:
            line = resp.readline()
            if not line:
                break
            chunk = json.loads(line.decode().strip())
            # Ollama streams chunks; the first chunk with response != "" is our signal
            if chunk.get("response") or chunk.get("done"):
                break
    t1 = time.perf_counter()
    return t1 - t0


def check_model_exists(model: str) -> bool:
    req = urllib.request.Request(
        f"{OLLAMA_BASE}/api/tags",
        method = "GET",
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
        names = [m["name"] for m in data.get("models", [])]
        return model in names or any(m.startswith(model.split(":")[0]) for m in names)
    except Exception:
        return False


def check_ollama_running() -> bool:
    try:
        urllib.request.urlopen(f"{OLLAMA_BASE}/api/tags", timeout=5)
        return True
    except Exception:
        return False


def main():
    ap = argparse.ArgumentParser(description="Benchmark Ollama cold-load time")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--runs",  type=int, default=DEFAULT_RUNS)
    args = ap.parse_args()

    print(f"Ollama Cold-Load Benchmark")
    print(f"  Model : {args.model}")
    print(f"  Runs  : {args.runs}")
    print(f"  Metric: time from cold state to first token (stream)")
    print()

    if not check_ollama_running():
        print("ERROR: Ollama not running. Start with: ollama serve")
        sys.exit(1)

    if not check_model_exists(args.model):
        print(f"ERROR: Model {args.model} not pulled. Run: ollama pull {args.model}")
        sys.exit(1)

    # Warm up Ollama's process itself (not the model) — one throw-away ping
    check_ollama_running()

    timings = []
    for i in range(args.runs):
        t = time_cold_load(args.model)
        timings.append(t)
        status = f"  Run {i+1:2d}/{args.runs}: {t:.3f}s"
        print(status, flush=True)

    mean   = statistics.mean(timings)
    stdev  = statistics.stdev(timings) if len(timings) > 1 else 0.0
    mn     = min(timings)
    mx     = max(timings)

    print()
    print("=" * 50)
    print(f"  Model              : {args.model}")
    print(f"  Runs               : {args.runs}")
    print(f"  Mean cold-load time: {mean:.3f}s")
    print(f"  Stddev             : ±{stdev:.3f}s")
    print(f"  Min / Max          : {mn:.3f}s / {mx:.3f}s")
    print()
    print(f"  arXiv table entry:")
    print(f"  | Ollama ({args.model}) | {mean:.1f}s ±{stdev:.1f}s | measured on identical hardware |")
    print("=" * 50)

    # Write JSON for RESULTS.md integration
    import pathlib, datetime
    out = pathlib.Path("/Users/wscholl/poc/bench_ollama_results.json")
    with open(out, "w") as f:
        json.dump({
            "model":   args.model,
            "runs":    args.runs,
            "timings": timings,
            "mean_s":  round(mean, 4),
            "std_s":   round(stdev, 4),
            "min_s":   round(mn, 4),
            "max_s":   round(mx, 4),
            "measured_at": datetime.datetime.now().isoformat(),
            "hardware": "Apple Silicon M-series 16GB unified memory",
        }, f, indent=2)
    print(f"\n  Results saved → {out}")


if __name__ == "__main__":
    main()
