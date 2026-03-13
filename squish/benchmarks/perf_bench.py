# squish/benchmarks/perf_bench.py
"""Track E — Performance Benchmark (TTFT, TPS, RAM, tokens/watt, batch throughput).

Measures latency and throughput for a running inference engine using only
stdlib modules (urllib, subprocess, asyncio).  Works with any
OpenAI-compatible server (squish, ollama, llama.cpp, etc.).

Metrics collected
─────────────────
  warm_ttft_ms        Warm time-to-first-token in milliseconds (median of 3 runs)
  tps                 Output tokens per second (median of 3 runs)
  ram_delta_mb        RSS increase from before to after the first request (MB)
  long_ctx_tps        TPS at 8× the default prompt length (single run)
  batch_p50_ms        P50 end-to-end latency for N=8 concurrent requests (ms)
  batch_p99_ms        P99 end-to-end latency for N=8 concurrent requests (ms)
  batch_throughput_tps  Total TPS across all concurrent requests
  tokens_per_watt     Tokens/joule via macOS powermetrics (darwin only; 0.0 on other OS)
"""
from __future__ import annotations

import asyncio
import json
import os
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from squish.benchmarks.base import (
    BenchmarkRunner,
    EngineClient,
    EngineConfig,
    ResultRecord,
)

__all__ = ["PerfBenchConfig", "PerfBenchRunner"]


# ---------------------------------------------------------------------------
# Warm prompts used for latency / TPS measurement
# ---------------------------------------------------------------------------

_WARM_PROMPTS: List[str] = [
    "Explain the concept of entropy in one sentence.",
    "What is the capital of Japan?",
    "Write a Python one-liner that squares all even numbers in a list.",
]

_LONG_CTX_MULTIPLIER = 8  # repeat base prompt N times to stress long context


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class PerfBenchConfig:
    """Configuration for the performance benchmark runner."""
    warm_reps: int = 3
    batch_concurrency: int = 8
    max_tokens: int = 128
    temperature: float = 0.0
    powermetrics_sample_ms: int = 500
    powermetrics_duration_s: float = 5.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _count_tokens(text: str) -> int:
    """Approximate token count; uses whitespace split as a proxy."""
    return max(1, len(text.split()))


def _rss_mb() -> float:
    """Return current process RSS in MB (best-effort; 0.0 on failure)."""
    try:
        import resource
        rss_bytes = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # macOS returns bytes, Linux returns kilobytes
        if sys.platform == "darwin":
            return rss_bytes / (1024 * 1024)
        return rss_bytes / 1024
    except Exception:  # noqa: BLE001
        return 0.0


def _warm_ttft_and_tps(
    client: EngineClient,
    model: str,
    config: PerfBenchConfig,
) -> Dict[str, float]:
    """Measure warm TTFT and TPS over _WARM_PROMPTS × warm_reps."""
    ttfts: List[float] = []
    tps_vals: List[float] = []

    for rep in range(config.warm_reps):
        prompt = _WARM_PROMPTS[rep % len(_WARM_PROMPTS)]
        messages = [{"role": "user", "content": prompt}]
        t0 = time.perf_counter()
        ttft_recorded = False
        n_tokens = 0
        try:
            for delta, ttft_s, total_s in client.chat_stream(
                model,
                messages,
                max_tokens=config.max_tokens,
                temperature=config.temperature,
            ):
                if not ttft_recorded:
                    ttfts.append(ttft_s * 1000)  # → ms
                    ttft_recorded = True
                n_tokens += _count_tokens(delta)
            elapsed = time.perf_counter() - t0
            if elapsed > 0 and n_tokens > 0:
                tps_vals.append(n_tokens / elapsed)
        except Exception:  # noqa: BLE001
            pass  # skip failed rep

    median_ttft = statistics.median(ttfts) if ttfts else 0.0
    median_tps = statistics.median(tps_vals) if tps_vals else 0.0
    return {"warm_ttft_ms": round(median_ttft, 2), "tps": round(median_tps, 2)}


def _long_ctx_tps(
    client: EngineClient,
    model: str,
    config: PerfBenchConfig,
) -> float:
    """Run a single long-context request and return TPS."""
    base_prompt = _WARM_PROMPTS[0]
    long_prompt = (base_prompt + " ") * _LONG_CTX_MULTIPLIER
    messages = [{"role": "user", "content": long_prompt}]
    n_tokens = 0
    t0 = time.perf_counter()
    try:
        for delta, _ttft, _total in client.chat_stream(
            model, messages,
            max_tokens=config.max_tokens,
            temperature=config.temperature,
        ):
            n_tokens += _count_tokens(delta)
    except Exception:  # noqa: BLE001
        return 0.0
    elapsed = time.perf_counter() - t0
    return round(n_tokens / elapsed, 2) if elapsed > 0 and n_tokens > 0 else 0.0


def _batch_throughput(
    engine: EngineConfig,
    model: str,
    config: PerfBenchConfig,
) -> Dict[str, float]:
    """Fire N concurrent requests and compute P50/P99 latency + batch TPS."""

    async def _single(session_model: str) -> Dict[str, float]:
        prompt = _WARM_PROMPTS[0]
        payload = json.dumps({
            "model": session_model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": config.max_tokens,
            "temperature": config.temperature,
            "stream": False,
        }).encode()
        req = urllib.request.Request(
            engine.chat_url(),
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {engine.api_key}",
            },
            method="POST",
        )
        t0 = time.perf_counter()
        try:
            resp_data = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: urllib.request.urlopen(req, timeout=engine.timeout).read(),
            )
            elapsed_ms = (time.perf_counter() - t0) * 1000
            parsed = json.loads(resp_data)
            content = parsed.get("choices", [{}])[0].get("message", {}).get("content", "")
            n_tok = _count_tokens(content)
            return {"latency_ms": elapsed_ms, "n_tokens": n_tok}
        except Exception:  # noqa: BLE001
            elapsed_ms = (time.perf_counter() - t0) * 1000
            return {"latency_ms": elapsed_ms, "n_tokens": 0}

    async def _run_all() -> List[Dict[str, float]]:
        tasks = [_single(model) for _ in range(config.batch_concurrency)]
        return await asyncio.gather(*tasks)

    try:
        results = asyncio.run(_run_all())
    except RuntimeError:
        # Already inside an event loop (e.g. Jupyter / test environment)
        loop = asyncio.new_event_loop()
        results = loop.run_until_complete(_run_all())
        loop.close()

    latencies = sorted(r["latency_ms"] for r in results)
    total_tokens = sum(r["n_tokens"] for r in results)
    max_latency_s = max(r["latency_ms"] for r in results) / 1000 if results else 1.0

    n = len(latencies)
    p50 = latencies[int(n * 0.50)] if n else 0.0
    p99 = latencies[min(int(n * 0.99), n - 1)] if n else 0.0
    batch_tps = round(total_tokens / max_latency_s, 2) if max_latency_s > 0 else 0.0

    return {
        "batch_p50_ms":          round(p50, 2),
        "batch_p99_ms":          round(p99, 2),
        "batch_throughput_tps":  batch_tps,
    }


def _tokens_per_watt(
    client: EngineClient,
    model: str,
    config: PerfBenchConfig,
) -> float:
    """macOS-only: measure tokens/watt using powermetrics subprocess.

    Returns 0.0 on non-darwin platforms or if powermetrics is unavailable.
    """
    if sys.platform != "darwin":  # pragma: no cover
        return 0.0

    # Start powermetrics in background
    try:
        pm_proc = subprocess.Popen(
            [
                "sudo", "-n", "powermetrics",
                "--samplers", "cpu_power",
                "-i", str(config.powermetrics_sample_ms),
                "-f", "plist",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except Exception:  # pragma: no cover
        return 0.0

    # Run inference for powermetrics_duration_s
    t0 = time.perf_counter()
    n_tokens = 0
    try:
        while time.perf_counter() - t0 < config.powermetrics_duration_s:
            prompt = _WARM_PROMPTS[0]
            messages = [{"role": "user", "content": prompt}]
            for delta, _ttft, _total in client.chat_stream(
                model, messages,
                max_tokens=config.max_tokens,
                temperature=config.temperature,
            ):
                n_tokens += _count_tokens(delta)
    except Exception:  # noqa: BLE001
        pass
    finally:
        pm_proc.terminate()

    elapsed = time.perf_counter() - t0

    # Parse watts from powermetrics plist output
    watts = 0.0
    try:
        raw = pm_proc.stdout.read() if pm_proc.stdout else ""
        # Look for CPU + GPU package power line: e.g. "CPU Power: 4321 mW"
        import re
        matches = re.findall(r"CPU Power:\s+(\d+)\s+mW", raw)
        if matches:
            avg_mw = statistics.mean(float(m) for m in matches)
            watts = avg_mw / 1000.0
    except Exception:  # noqa: BLE001
        pass

    if watts <= 0 or elapsed <= 0 or n_tokens == 0:
        return 0.0

    tps = n_tokens / elapsed
    return round(tps / watts, 4)  # tokens per watt


# ---------------------------------------------------------------------------
# PerfBenchRunner
# ---------------------------------------------------------------------------

class PerfBenchRunner(BenchmarkRunner):
    """Track E: performance benchmark — latency, throughput, efficiency."""

    def __init__(self, config: Optional[PerfBenchConfig] = None) -> None:
        self._config = config or PerfBenchConfig()

    @property
    def track_name(self) -> str:
        return "perf"

    def run(
        self,
        engine: EngineConfig,
        model: str,
        *,
        limit: Optional[int] = None,
    ) -> ResultRecord:
        """Run the performance track and return a ResultRecord.

        ``limit`` is interpreted as max_tokens when provided.
        """
        config = self._config
        if limit is not None:
            from dataclasses import replace
            config = replace(config, max_tokens=limit)

        client = EngineClient(engine)

        # RAM baseline
        rss_before = _rss_mb()

        # Warm TTFT + TPS
        warm = _warm_ttft_and_tps(client, model, config)

        # RAM after warm requests
        rss_after = _rss_mb()
        ram_delta = max(0.0, rss_after - rss_before)

        # Long-context TPS
        lc_tps = _long_ctx_tps(client, model, config)

        # Batch throughput
        batch = _batch_throughput(engine, model, config)

        # Tokens/watt (darwin only)
        tpw = _tokens_per_watt(client, model, config)

        metrics: Dict[str, Any] = {
            **warm,
            "ram_delta_mb":         round(ram_delta, 2),
            "long_ctx_tps":         lc_tps,
            **batch,
            "tokens_per_watt":      tpw,
        }

        return ResultRecord(
            track=self.track_name,
            engine=engine.name,
            model=model,
            metrics=metrics,
            metadata={
                "warm_reps":        config.warm_reps,
                "batch_concurrency": config.batch_concurrency,
                "max_tokens":        config.max_tokens,
                "platform":          sys.platform,
            },
        )
