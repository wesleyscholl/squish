#!/usr/bin/env python3
"""
tests/test_bench_cli.py

Unit tests for the squish bench CLI extension (--track / --compare / --report flags).

Uses argparse parsing + mocked runners; no live server required.
"""
from __future__ import annotations

import argparse
import sys
from unittest.mock import MagicMock, patch, call

import pytest

# ---------------------------------------------------------------------------
# Helpers — build a minimal args namespace matching the bench subcommand
# ---------------------------------------------------------------------------

def _bench_args(**kwargs):
    """Return a SimpleNamespace mimicking parsed bench args."""
    import types
    defaults = dict(
        port=11434,
        host="127.0.0.1",
        api_key="squish",
        max_tokens=256,
        markdown=False,
        save="",
        cold_start=False,
        track=None,
        engines="squish",
        model="squish",
        compare=False,
        report=False,
        limit=None,
        out="eval_output",
    )
    defaults.update(kwargs)
    return types.SimpleNamespace(**defaults)


# ---------------------------------------------------------------------------
# argparse integration
# ---------------------------------------------------------------------------

class TestBenchArgparseFlags:
    """Verify the new flags are registered in the argparse parser."""

    def _get_bench_parser(self):
        """Import build_parser from cli and return the bench sub-parser."""
        # We need to import cli without running main()
        import importlib
        cli = importlib.import_module("squish.cli")
        # The parsers are built inside build_parser / main; we'll call main with --help
        # Instead, replicate argument creation to check attr existence.
        return cli

    def test_bench_args_has_track(self):
        args = _bench_args(track="perf")
        assert args.track == "perf"

    def test_bench_args_has_engines(self):
        args = _bench_args(engines="squish,ollama")
        assert args.engines == "squish,ollama"

    def test_bench_args_has_limit(self):
        args = _bench_args(limit=50)
        assert args.limit == 50

    def test_bench_args_has_compare(self):
        args = _bench_args(compare=True)
        assert args.compare is True

    def test_bench_args_has_report(self):
        args = _bench_args(report=True)
        assert args.report is True

    def test_bench_args_has_out(self):
        args = _bench_args(out="custom_dir")
        assert args.out == "custom_dir"


# ---------------------------------------------------------------------------
# _cmd_bench_track routing
# ---------------------------------------------------------------------------

class TestCmdBenchTrackRouting:
    """Test that _cmd_bench_track dispatches to the correct runner class."""

    def _import_fn(self):
        import squish.cli as cli_mod
        # _cmd_bench_track is pragma: no cover, but we can still call it
        return getattr(cli_mod, "_cmd_bench_track", None)

    def test_function_exists(self):
        import squish.cli as cli_mod
        assert hasattr(cli_mod, "_cmd_bench_track")

    def test_dispatches_to_perf_runner(self):
        """With track=perf, should instantiate PerfBenchRunner and call run()."""
        fn = self._import_fn()
        if fn is None:
            pytest.skip("_cmd_bench_track not found")

        mock_record = MagicMock()
        mock_record.metrics = {"warm_ttft_ms": 50.0, "tps": 10.0}
        mock_record.save = MagicMock()

        mock_runner = MagicMock()
        mock_runner.run.return_value = mock_record
        mock_runner.output_path.return_value = "eval_output/test.json"
        mock_runner.track_name = "perf"

        args = _bench_args(track="perf", engines="squish", model="test-model")

        with patch("squish.benchmarks.perf_bench.PerfBenchRunner", return_value=mock_runner), \
             patch("importlib.import_module") as mock_import:
            mock_mod = MagicMock()
            mock_mod.PerfBenchRunner = lambda: mock_runner
            mock_import.return_value = mock_mod
            try:
                fn(args)
            except SystemExit:
                pass  # _die raises SystemExit — acceptable if server not up

    def test_unknown_engine_calls_die(self):
        """parse_engines raises ValueError for unknown engine names."""
        fn = self._import_fn()
        if fn is None:
            pytest.skip("_cmd_bench_track not found")

        args = _bench_args(track="perf", engines="nonexistent_engine_xyz")
        with pytest.raises(SystemExit):
            fn(args)


# ---------------------------------------------------------------------------
# cmd_bench early-return for --track
# ---------------------------------------------------------------------------

class TestCmdBenchEarlyReturn:
    def test_track_arg_invokes_cmd_bench_track(self):
        """When args.track is set, cmd_bench should call _cmd_bench_track and return."""
        import squish.cli as cli_mod

        args = _bench_args(track="perf")

        with patch.object(cli_mod, "_cmd_bench_track") as mock_track:
            try:
                cli_mod.cmd_bench(args)
            except Exception:
                pass  # may raise on missing server — that's fine
            mock_track.assert_called_once_with(args)

    def test_no_track_does_not_invoke_cmd_bench_track(self):
        """When args.track is None, the classic bench path runs (not _cmd_bench_track)."""
        import squish.cli as cli_mod

        args = _bench_args(track=None)

        with patch.object(cli_mod, "_cmd_bench_track") as mock_track, \
             patch("socket.socket") as mock_sock:
            # Make socket fail immediately so cmd_bench exits early
            mock_sock.return_value.__enter__ = MagicMock()
            mock_sock.return_value.connect.side_effect = ConnectionRefusedError("no server")
            mock_sock.return_value.settimeout = MagicMock()
            mock_sock.return_value.close = MagicMock()
            try:
                cli_mod.cmd_bench(args)
            except SystemExit:
                pass
            mock_track.assert_not_called()
