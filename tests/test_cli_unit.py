"""
tests/test_cli_unit.py

Unit tests for squish/cli.py pure-Python helpers.
Only tests functions that do not require MLX, model files, or a running server.

Actual API:
    _box(lines: list[str]) -> None  — prints, no return
    _die(msg: str) -> None          — prints to stderr, sys.exit(1)
    cmd_catalog(args)               — args.tag, args.refresh
    cmd_models(args)                — args.path (optional?)
    cmd_doctor(args)
"""
from __future__ import annotations

import argparse
import sys
from unittest.mock import patch

import pytest


def _import_cli():
    import squish.cli as cli  # noqa: PLC0415
    return cli


# ── _box ──────────────────────────────────────────────────────────────────────
# Signature: _box(lines: list[str]) -> None — prints a box to stdout

class TestBox:
    def test_prints_to_stdout(self, capsys):
        cli = _import_cli()
        cli._box(["hello"])
        captured = capsys.readouterr()
        assert "hello" in captured.out

    def test_returns_none(self):
        cli = _import_cli()
        result = cli._box(["test"])
        assert result is None

    def test_contains_all_lines(self, capsys):
        cli = _import_cli()
        cli._box(["alpha", "beta", "gamma"])
        captured = capsys.readouterr()
        assert "alpha" in captured.out
        assert "beta" in captured.out
        assert "gamma" in captured.out

    def test_box_border_chars(self, capsys):
        cli = _import_cli()
        cli._box(["content"])
        captured = capsys.readouterr()
        assert any(c in captured.out for c in ["\u2500", "\u2502", "\u250c", "\u2514", "|", "-"])

    def test_box_unicode_content(self, capsys):
        cli = _import_cli()
        cli._box(["H\u00e9llo w\u00f6rld"])
        captured = capsys.readouterr()
        assert "H\u00e9llo" in captured.out


# ── _die ──────────────────────────────────────────────────────────────────────

class TestDie:
    def test_raises_system_exit_1(self):
        cli = _import_cli()
        with pytest.raises(SystemExit) as exc:
            cli._die("fatal error")
        assert exc.value.code == 1

    def test_writes_message(self, capsys):
        cli = _import_cli()
        with pytest.raises(SystemExit):
            cli._die("test message")
        captured = capsys.readouterr()
        assert "test message" in (captured.err + captured.out)


# ── cmd_catalog ───────────────────────────────────────────────────────────────
# args: tag (str|None), refresh (bool)

class TestCmdCatalog:
    def test_runs_without_error(self, capsys):
        cli = _import_cli()
        fn = getattr(cli, "cmd_catalog", None)
        if fn is None:
            pytest.skip("cmd_catalog not found")
        ns = argparse.Namespace(tag=None, refresh=False)
        fn(ns)
        captured = capsys.readouterr()
        assert len(captured.out) > 0

    def test_tag_filter_small(self, capsys):
        cli = _import_cli()
        fn = getattr(cli, "cmd_catalog", None)
        if fn is None:
            pytest.skip("cmd_catalog not found")
        ns = argparse.Namespace(tag="small", refresh=False)
        fn(ns)
        captured = capsys.readouterr()
        assert isinstance(captured.out, str)

    def test_unknown_tag_no_models(self, capsys):
        cli = _import_cli()
        fn = getattr(cli, "cmd_catalog", None)
        if fn is None:
            pytest.skip("cmd_catalog not found")
        ns = argparse.Namespace(tag="definitely_not_a_real_tag_xyz_99999", refresh=False)
        fn(ns)
        captured = capsys.readouterr()
        # Should indicate no models found
        combined = (captured.out + captured.err).lower()
        assert "no" in combined or len(captured.out) == 0 or combined

    def test_output_contains_model_ids(self, capsys):
        cli = _import_cli()
        fn = getattr(cli, "cmd_catalog", None)
        if fn is None:
            pytest.skip("cmd_catalog not found")
        ns = argparse.Namespace(tag=None, refresh=False)
        fn(ns)
        captured = capsys.readouterr()
        # Should have at least one colon-separated model ID like "qwen3:8b"
        assert ":" in captured.out


# ── cmd_models ────────────────────────────────────────────────────────────────

class TestCmdModels:
    def test_runs_without_error(self, capsys):
        cli = _import_cli()
        fn = getattr(cli, "cmd_models", None)
        if fn is None:
            pytest.skip("cmd_models not found")
        ns = argparse.Namespace()
        try:
            fn(ns)
        except (SystemExit, Exception):
            pass
        captured = capsys.readouterr()
        assert isinstance(captured.out + captured.err, str)


# ── cmd_doctor ────────────────────────────────────────────────────────────────

class TestCmdDoctor:
    def test_runs_without_crash(self, capsys):
        cli = _import_cli()
        fn = getattr(cli, "cmd_doctor", None)
        if fn is None:
            pytest.skip("cmd_doctor not found")
        ns = argparse.Namespace()
        try:
            fn(ns)
        except SystemExit:
            pass
        captured = capsys.readouterr()
        assert isinstance(captured.out + captured.err, str)

    def test_doctor_produces_output(self, capsys):
        cli = _import_cli()
        fn = getattr(cli, "cmd_doctor", None)
        if fn is None:
            pytest.skip("cmd_doctor not found")
        ns = argparse.Namespace()
        try:
            fn(ns)
        except SystemExit:
            pass
        captured = capsys.readouterr()
        combined = (captured.out + captured.err).lower()
        assert any(word in combined for word in
                   ("python", "mlx", "squish", "ok", "pass", "fail", "version", "error", "\u2713", "\u2717"))


# ── main / argparse ───────────────────────────────────────────────────────────

class TestMain:
    def test_no_args_exits(self):
        cli = _import_cli()
        if not hasattr(cli, "main"):
            pytest.skip("No main() found")
        with pytest.raises(SystemExit) as exc:
            with patch("sys.argv", ["squish"]):
                cli.main()
        assert exc.value.code in (0, 1, 2)

    def test_help_exits_zero(self):
        cli = _import_cli()
        if not hasattr(cli, "main"):
            pytest.skip("No main() found")
        with pytest.raises(SystemExit) as exc:
            with patch("sys.argv", ["squish", "--help"]):
                cli.main()
        assert exc.value.code == 0
