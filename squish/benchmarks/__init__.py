# squish/benchmarks/__init__.py
"""Squish Benchmark Suite — 5-track cross-engine comparison.

Tracks
------
A: quality   — MMLU, ARC, HellaSwag, WinoGrande, TruthfulQA, GSM8K
B: code      — HumanEval, MBPP (pass@1)
C: tools     — BFCL v3 tool-use schema compliance
D: agent     — 20 hand-authored agentic scenarios with fixture replay
E: perf      — TTFT, TPS, RAM, tokens/watt, batch throughput

Usage::

    squish bench --track quality --limit 50
    squish bench --track all --report
"""
from __future__ import annotations

from squish.benchmarks.base import (
    BenchmarkRunner,
    EngineClient,
    EngineConfig,
    ResultRecord,
    SQUISH_ENGINE,
    OLLAMA_ENGINE,
    LMSTUDIO_ENGINE,
    MLXLM_ENGINE,
    LLAMACPP_ENGINE,
    ENGINE_REGISTRY,
    parse_engines,
)

__all__ = [
    "BenchmarkRunner",
    "EngineClient",
    "EngineConfig",
    "ResultRecord",
    "SQUISH_ENGINE",
    "OLLAMA_ENGINE",
    "LMSTUDIO_ENGINE",
    "MLXLM_ENGINE",
    "LLAMACPP_ENGINE",
    "ENGINE_REGISTRY",
    "parse_engines",
]
