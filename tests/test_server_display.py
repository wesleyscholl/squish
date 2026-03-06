"""
tests/test_server_display.py

Branch coverage for the terminal display / formatting functions in
squish/server.py that are otherwise unreachable in headless test runs
because they depend on _TTY / _TRUE_COLOR / _trace_file globals.

Covers:
  _gradient     — true-colour path (text non-empty), plain fallback, empty text
  _cprint       — with value, without value
  _ok / _info / _warn — smoke tests (no branches, just line execution)
  _section      — with title, without title (empty string)
  _print_banner — _TTY=True path, _TTY=False path
  _tlog         — basic trace, trace with _trace_file set
"""
from __future__ import annotations

import io
import sys
import time

import pytest

import squish.server as srv


# ── helpers ───────────────────────────────────────────────────────────────────

def _capture_stdout(fn, *args, **kwargs) -> str:
    """Run *fn* and return everything it printed to stdout."""
    buf = io.StringIO()
    old, sys.stdout = sys.stdout, buf
    try:
        fn(*args, **kwargs)
    finally:
        sys.stdout = old
    return buf.getvalue()


def _capture_stderr(fn, *args, **kwargs) -> str:
    """Run *fn* and return everything it printed to stderr."""
    buf = io.StringIO()
    old, sys.stderr = sys.stderr, buf
    try:
        fn(*args, **kwargs)
    finally:
        sys.stderr = old
    return buf.getvalue()


# ── _gradient ─────────────────────────────────────────────────────────────────

class TestGradient:

    def test_true_color_returns_escape_codes(self, monkeypatch):
        monkeypatch.setattr(srv, "_TRUE_COLOR", True)
        result = srv._gradient("Hi", [(255, 0, 0), (0, 0, 255)])
        assert "\033[38;2;" in result
        # Each character should be present somewhere in the result
        assert "H" in result
        assert "i" in result

    def test_false_color_returns_text_unchanged(self, monkeypatch):
        monkeypatch.setattr(srv, "_TRUE_COLOR", False)
        result = srv._gradient("Hi", [(255, 0, 0), (0, 0, 255)])
        assert result == "Hi"

    def test_empty_text_returns_empty(self, monkeypatch):
        monkeypatch.setattr(srv, "_TRUE_COLOR", True)
        result = srv._gradient("", [(255, 0, 0), (0, 0, 255)])
        assert result == ""

    def test_single_char(self, monkeypatch):
        monkeypatch.setattr(srv, "_TRUE_COLOR", True)
        result = srv._gradient("X", [(0, 128, 255), (255, 128, 0)])
        assert "X" in result

    def test_multi_stop_gradient(self, monkeypatch):
        monkeypatch.setattr(srv, "_TRUE_COLOR", True)
        stops = [(88, 28, 135), (236, 72, 153), (34, 211, 238)]
        result = srv._gradient("SQUISH", stops)
        assert len(result) > len("SQUISH")


# ── _cprint ───────────────────────────────────────────────────────────────────

class TestCprint:

    def test_with_value_prints_both(self):
        output = _capture_stdout(srv._cprint, "COL", "label", "val")
        assert "label" in output
        assert "val" in output

    def test_without_value_prints_label_only(self):
        output = _capture_stdout(srv._cprint, "COL", "label")
        assert "label" in output

    def test_custom_end(self):
        output = _capture_stdout(srv._cprint, "COL", "lbl", end="")
        # No trailing newline in output
        assert not output.endswith("\n")


# ── _ok / _info / _warn ───────────────────────────────────────────────────────

class TestSimplePrinters:

    def test_ok_prints(self):
        output = _capture_stdout(srv._ok, "everything is fine")
        assert "everything is fine" in output

    def test_info_prints_label_and_value(self):
        output = _capture_stdout(srv._info, "port", "11435")
        assert "port" in output
        assert "11435" in output

    def test_warn_prints_message(self):
        output = _capture_stdout(srv._warn, "something fishy")
        assert "something fishy" in output


# ── _section ──────────────────────────────────────────────────────────────────

class TestSection:

    def test_section_with_title_prints_divider_and_title(self):
        output = _capture_stdout(srv._section, "Startup")
        assert "─" in output or "-" in output  # divider line
        assert "Startup" in output

    def test_section_empty_title_no_title_line(self):
        output = _capture_stdout(srv._section, "")
        # Divider still printed, but no extra title line required
        assert len(output) > 0


# ── _print_banner ─────────────────────────────────────────────────────────────

class TestPrintBanner:

    def test_tty_path_prints_ascii_art(self, monkeypatch):
        """When _TTY is True the SQUISH ASCII-art block is emitted."""
        monkeypatch.setattr(srv, "_TTY", True)
        monkeypatch.setattr(srv, "_TRUE_COLOR", False)   # keep output plain
        output = _capture_stdout(srv._print_banner)
        # Logo lines or character from banner
        assert "SQUISH" in output or "squish" in output.lower() or "◕" in output or "═" in output

    def test_non_tty_path_prints_plain_text(self, monkeypatch):
        """When _TTY is False the plain-text fallback is used."""
        monkeypatch.setattr(srv, "_TTY", False)
        output = _capture_stdout(srv._print_banner)
        assert "SQUISH" in output or "Squish" in output


# ── _tlog ─────────────────────────────────────────────────────────────────────

class TestTlog:

    def test_tlog_writes_to_stderr(self, monkeypatch):
        monkeypatch.setattr(srv, "_trace_file", None)
        output = _capture_stderr(srv._tlog, "hello trace")
        assert "hello trace" in output

    def test_tlog_writes_to_trace_file(self, monkeypatch):
        """When _trace_file is set, _tlog also writes a plain-text line."""
        buf = io.StringIO()
        monkeypatch.setattr(srv, "_trace_file", buf)
        _capture_stderr(srv._tlog, "file trace")
        written = buf.getvalue()
        assert "file trace" in written

    def test_tlog_trace_file_exception_silenced(self, monkeypatch):
        """If _trace_file.write() raises, _tlog must not propagate the exception."""

        class _Broken:
            def write(self, _s):
                raise OSError("disk full")
            def flush(self):
                pass

        monkeypatch.setattr(srv, "_trace_file", _Broken())
        # Should complete without raising
        _capture_stderr(srv._tlog, "broken file test")
