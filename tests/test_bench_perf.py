#!/usr/bin/env python3
"""
tests/test_bench_perf.py

Unit tests for squish/benchmarks/perf_bench.py (Track E).

All tests run without a live server — network calls are patched.
"""
from __future__ import annotations

import json
import statistics
import sys
import time
from unittest.mock import MagicMock, patch, call

import pytest

from squish.benchmarks.base import EngineConfig, ResultRecord
from squish.benchmarks.perf_bench import (
    PerfBenchConfig,
    PerfBenchRunner,
    _count_tokens,
    _rss_mb,
    _warm_ttft_and_tps,
    _long_ctx_tps,
    _batch_throughput,
    _tokens_per_watt,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

_ENGINE = EngineConfig("squish", "http://localhost:11434")
_MODEL = "test-model"


def _make_runner(config: PerfBenchConfig | None = None) -> PerfBenchRunner:
    return PerfBenchRunner(config)


def _stream_chunks(text: str = "hello world foo bar baz"):
    """Yield (delta, ttft_s, total_s) tuples like EngineClient.chat_stream."""
    words = text.split()
    for i, w in enumerate(words):
        ttft = 0.05 if i == 0 else 0.05
        yield (w + " ", ttft, 0.1 + i * 0.02)


# ---------------------------------------------------------------------------
# _count_tokens
# ---------------------------------------------------------------------------

class TestCountTokens:
    def test_single_word(self):
        assert _count_tokens("hello") == 1

    def test_multiple_words(self):
        assert _count_tokens("hello world foo") == 3

    def test_empty_string_returns_one(self):
        assert _count_tokens("") == 1

    def test_whitespace_only_returns_one(self):
        assert _count_tokens("   ") == 1


# ---------------------------------------------------------------------------
# PerfBenchConfig
# ---------------------------------------------------------------------------

class TestPerfBenchConfig:
    def test_defaults(self):
        cfg = PerfBenchConfig()
        assert cfg.warm_reps == 3
        assert cfg.batch_concurrency == 8
        assert cfg.max_tokens == 128
        assert cfg.temperature == 0.0

    def test_custom_values(self):
        cfg = PerfBenchConfig(warm_reps=1, batch_concurrency=2, max_tokens=64)
        assert cfg.warm_reps == 1
        assert cfg.batch_concurrency == 2
        assert cfg.max_tokens == 64


# ---------------------------------------------------------------------------
# PerfBenchRunner
# ---------------------------------------------------------------------------

class TestPerfBenchRunnerTrackName:
    def test_track_name_is_perf(self):
        assert _make_runner().track_name == "perf"

    def test_is_benchmark_runner_subclass(self):
        from squish.benchmarks.base import BenchmarkRunner
        assert isinstance(_make_runner(), BenchmarkRunner)


class TestPerfBenchRunnerResultRecord:
    def _mock_client_stream(self):
        """Return a mock EngineClient whose chat_stream yields tokens."""
        client = MagicMock()
        client.chat_stream.return_value = iter(_stream_chunks())
        return client

    def test_run_returns_result_record(self):
        runner = _make_runner(PerfBenchConfig(warm_reps=1, batch_concurrency=1))
        with patch("squish.benchmarks.perf_bench.EngineClient") as MockClient, \
             patch("squish.benchmarks.perf_bench._batch_throughput") as mock_batch, \
             patch("squish.benchmarks.perf_bench._tokens_per_watt", return_value=0.0):
            mock_client_instance = MockClient.return_value
            mock_client_instance.chat_stream.return_value = iter(_stream_chunks())
            mock_batch.return_value = {
                "batch_p50_ms": 50.0,
                "batch_p99_ms": 90.0,
                "batch_throughput_tps": 100.0,
            }
            result = runner.run(_ENGINE, _MODEL)
        assert isinstance(result, ResultRecord)

    def test_result_track_is_perf(self):
        runner = _make_runner(PerfBenchConfig(warm_reps=1, batch_concurrency=1))
        with patch("squish.benchmarks.perf_bench.EngineClient") as MockClient, \
             patch("squish.benchmarks.perf_bench._batch_throughput") as mock_batch, \
             patch("squish.benchmarks.perf_bench._tokens_per_watt", return_value=0.0):
            mock_client_instance = MockClient.return_value
            mock_client_instance.chat_stream.return_value = iter(_stream_chunks())
            mock_batch.return_value = {
                "batch_p50_ms": 50.0,
                "batch_p99_ms": 90.0,
                "batch_throughput_tps": 100.0,
            }
            result = runner.run(_ENGINE, _MODEL)
        assert result.track == "perf"

    def test_result_contains_required_metrics(self):
        runner = _make_runner(PerfBenchConfig(warm_reps=1, batch_concurrency=1))
        with patch("squish.benchmarks.perf_bench.EngineClient") as MockClient, \
             patch("squish.benchmarks.perf_bench._batch_throughput") as mock_batch, \
             patch("squish.benchmarks.perf_bench._tokens_per_watt", return_value=0.0):
            mock_client_instance = MockClient.return_value
            mock_client_instance.chat_stream.return_value = iter(_stream_chunks())
            mock_batch.return_value = {
                "batch_p50_ms": 50.0,
                "batch_p99_ms": 90.0,
                "batch_throughput_tps": 100.0,
            }
            result = runner.run(_ENGINE, _MODEL)
        required = {
            "warm_ttft_ms", "tps", "ram_delta_mb", "long_ctx_tps",
            "batch_p50_ms", "batch_p99_ms", "batch_throughput_tps",
            "tokens_per_watt",
        }
        assert required.issubset(result.metrics.keys()), \
            f"Missing keys: {required - result.metrics.keys()}"

    def test_limit_overrides_max_tokens(self):
        """limit param becomes max_tokens in config."""
        runner = _make_runner(PerfBenchConfig(warm_reps=1, batch_concurrency=1))
        captured_configs = []

        original_warm = __import__(
            "squish.benchmarks.perf_bench", fromlist=["_warm_ttft_and_tps"]
        )._warm_ttft_and_tps

        def capture_warm(client, model, config):
            captured_configs.append(config.max_tokens)
            return {"warm_ttft_ms": 50.0, "tps": 10.0}

        with patch("squish.benchmarks.perf_bench.EngineClient"), \
             patch("squish.benchmarks.perf_bench._warm_ttft_and_tps", side_effect=capture_warm), \
             patch("squish.benchmarks.perf_bench._long_ctx_tps", return_value=5.0), \
             patch("squish.benchmarks.perf_bench._batch_throughput", return_value={
                 "batch_p50_ms": 50.0, "batch_p99_ms": 90.0, "batch_throughput_tps": 100.0
             }), \
             patch("squish.benchmarks.perf_bench._tokens_per_watt", return_value=0.0):
            runner.run(_ENGINE, _MODEL, limit=64)

        assert captured_configs[0] == 64

    def test_metadata_contains_platform(self):
        runner = _make_runner(PerfBenchConfig(warm_reps=1, batch_concurrency=1))
        with patch("squish.benchmarks.perf_bench.EngineClient") as MockClient, \
             patch("squish.benchmarks.perf_bench._batch_throughput") as mock_batch, \
             patch("squish.benchmarks.perf_bench._tokens_per_watt", return_value=0.0):
            mock_client_instance = MockClient.return_value
            mock_client_instance.chat_stream.return_value = iter(_stream_chunks())
            mock_batch.return_value = {
                "batch_p50_ms": 50.0,
                "batch_p99_ms": 90.0,
                "batch_throughput_tps": 100.0,
            }
            result = runner.run(_ENGINE, _MODEL)
        assert "platform" in result.metadata
        assert result.metadata["platform"] == sys.platform


# ---------------------------------------------------------------------------
# _tokens_per_watt (platform guard)
# ---------------------------------------------------------------------------

class TestTokensPerWatt:
    def test_returns_zero_on_non_darwin(self):
        """On non-darwin platforms the function must return 0.0 without calling powermetrics."""
        if sys.platform == "darwin":
            # On macOS: patch sys.platform to simulate non-darwin
            with patch("squish.benchmarks.perf_bench.sys") as mock_sys:
                mock_sys.platform = "linux"
                client = MagicMock()
                result = _tokens_per_watt(client, _MODEL, PerfBenchConfig())
            assert result == 0.0
        else:
            client = MagicMock()
            result = _tokens_per_watt(client, _MODEL, PerfBenchConfig())
            assert result == 0.0

    def test_returns_float(self):
        """Result is always a float."""
        client = MagicMock()
        with patch("squish.benchmarks.perf_bench.sys") as mock_sys:
            mock_sys.platform = "linux"
            result = _tokens_per_watt(client, _MODEL, PerfBenchConfig())
        assert isinstance(result, float)


# ---------------------------------------------------------------------------
# _warm_ttft_and_tps — unit via mock stream
# ---------------------------------------------------------------------------

class TestWarmTtftAndTps:
    def test_returns_dict_with_warm_ttft_ms_and_tps(self):
        client = MagicMock()
        client.chat_stream.return_value = iter(_stream_chunks())
        result = _warm_ttft_and_tps(client, _MODEL, PerfBenchConfig(warm_reps=1))
        assert "warm_ttft_ms" in result
        assert "tps" in result

    def test_ttft_is_non_negative(self):
        client = MagicMock()
        client.chat_stream.return_value = iter(_stream_chunks())
        result = _warm_ttft_and_tps(client, _MODEL, PerfBenchConfig(warm_reps=1))
        assert result["warm_ttft_ms"] >= 0.0

    def test_no_stream_returns_zeros(self):
        """If chat_stream yields nothing, both metrics should be 0.0."""
        client = MagicMock()
        client.chat_stream.return_value = iter([])
        result = _warm_ttft_and_tps(client, _MODEL, PerfBenchConfig(warm_reps=1))
        assert result["warm_ttft_ms"] == 0.0
        assert result["tps"] == 0.0


# ---------------------------------------------------------------------------
# _rss_mb
# ---------------------------------------------------------------------------

class TestRssMb:
    def test_returns_float(self):
        result = _rss_mb()
        assert isinstance(result, float)

    def test_returns_non_negative(self):
        result = _rss_mb()
        assert result >= 0.0
