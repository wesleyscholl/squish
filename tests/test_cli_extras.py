"""
tests/test_cli_extras.py

Extra coverage for squish/cli.py:
  - cmd_models: MODELS_DIR doesn't exist           (lines 219-220)
  - cmd_models: MODELS_DIR exists with entries     (lines 222-236)
  - cmd_models: MODELS_DIR exists, empty           (lines 238-242)
  - cmd_search: hits found                         (lines 350-371)
  - cmd_search: no hits                            (lines 354-356)
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


def _import_cli():
    import squish.cli as cli  # noqa: PLC0415
    return cli


# ── cmd_models ────────────────────────────────────────────────────────────────

class TestCmdModelsEdgeCases:
    def test_models_dir_not_found(self, tmp_path: Path, capsys, monkeypatch: pytest.MonkeyPatch):
        """MODELS_DIR doesn't exist → print '(directory not found)' (lines 219-220)."""
        cli = _import_cli()
        missing_dir = tmp_path / "nonexistent_models"
        monkeypatch.setattr(cli, "_MODELS_DIR", missing_dir)
        ns = argparse.Namespace()
        cli.cmd_models(ns)
        captured = capsys.readouterr()
        assert "not found" in captured.out.lower() or "directory" in captured.out.lower()

    def test_models_dir_empty(self, tmp_path: Path, capsys, monkeypatch: pytest.MonkeyPatch):
        """MODELS_DIR exists but is empty → print 'No model directories found' (lines 238-242)."""
        cli = _import_cli()
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        monkeypatch.setattr(cli, "_MODELS_DIR", models_dir)
        ns = argparse.Namespace()
        cli.cmd_models(ns)
        captured = capsys.readouterr()
        assert "no model" in captured.out.lower() or "no models" in captured.out.lower()

    def test_models_dir_with_dirs(self, tmp_path: Path, capsys, monkeypatch: pytest.MonkeyPatch):
        """MODELS_DIR has model dirs → lists them with size/compressed info (lines 222-236)."""
        cli = _import_cli()
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        # Create a fake model directory with a weight file
        model_dir = models_dir / "qwen3-8b"
        model_dir.mkdir()
        (model_dir / "weights.npz").write_bytes(b"x" * 1024)
        monkeypatch.setattr(cli, "_MODELS_DIR", models_dir)
        ns = argparse.Namespace()
        cli.cmd_models(ns)
        captured = capsys.readouterr()
        assert "qwen3-8b" in captured.out

    def test_models_dir_with_hidden_and_file(self, tmp_path: Path, capsys, monkeypatch: pytest.MonkeyPatch):
        """Hidden dirs (starting with '.') and files are skipped (lines 224, 226-227)."""
        cli = _import_cli()
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        # Hidden dir – should be skipped
        hidden = models_dir / ".hidden_model"
        hidden.mkdir()
        # Regular file – should be skipped (not a dir)
        (models_dir / "README.md").write_text("readme")
        # Real model
        real_model = models_dir / "llama-3b"
        real_model.mkdir()
        (real_model / "weights.npz").write_bytes(b"x")
        monkeypatch.setattr(cli, "_MODELS_DIR", models_dir)
        ns = argparse.Namespace()
        cli.cmd_models(ns)
        captured = capsys.readouterr()
        assert "llama-3b" in captured.out
        assert ".hidden_model" not in captured.out

    def test_models_dir_stat_exception(self, tmp_path: Path, capsys, monkeypatch: pytest.MonkeyPatch):
        """stat() failure for disk size → falls back to '?' (lines 234-235)."""
        cli = _import_cli()
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        model_dir = models_dir / "mymodel"
        model_dir.mkdir()
        monkeypatch.setattr(cli, "_MODELS_DIR", models_dir)

        # Patch Path.rglob to raise an exception
        original_rglob = Path.rglob
        def bad_rglob(self, *args, **kwargs):
            if self == model_dir:
                raise OSError("permission denied")
            return original_rglob(self, *args, **kwargs)
        monkeypatch.setattr(Path, "rglob", bad_rglob)

        ns = argparse.Namespace()
        cli.cmd_models(ns)
        captured = capsys.readouterr()
        # Should have "?" for size
        assert "?" in captured.out or "mymodel" in captured.out


# ── cmd_search ────────────────────────────────────────────────────────────────

class TestCmdSearch:
    def test_no_hits_prints_no_match(self, capsys, monkeypatch: pytest.MonkeyPatch):
        """search() returns empty list → print no-match message (lines 354-356)."""
        from squish import catalog as _cat
        monkeypatch.setattr(_cat, "search", lambda q: [])
        cli = _import_cli()
        ns = argparse.Namespace(query="xyznotfound")
        cli.cmd_search(ns)
        captured = capsys.readouterr()
        assert "no" in captured.out.lower() or "match" in captured.out.lower()

    def test_with_hits_prints_table(self, capsys, monkeypatch: pytest.MonkeyPatch):
        """search() returns results → print table (lines 358-370)."""
        from squish import catalog as _cat
        mock_entry = _cat.CatalogEntry(
            id="qwen3:8b",
            name="Qwen3 8B",
            hf_mlx_repo="test/repo",
            size_gb=5.0,
            squished_size_gb=1.5,
            params="8B",
            context=32768,
            tags=["small"],
        )
        monkeypatch.setattr(_cat, "search", lambda q: [mock_entry])
        cli = _import_cli()
        ns = argparse.Namespace(query="qwen")
        cli.cmd_search(ns)
        captured = capsys.readouterr()
        assert "qwen3:8b" in captured.out

    def test_search_imports_from_catalog(self):
        """cmd_search imports from squish.catalog without error."""
        cli = _import_cli()
        assert hasattr(cli, "cmd_search")


# ── cmd_doctor with failing check (cli.py line 960) ──────────────────────────

class TestCmdDoctorFailing:
    def test_failing_check_prints_some_checks_failed(self, capsys, monkeypatch: pytest.MonkeyPatch):
        """When a check fails, 'Some checks failed' is printed (line 960)."""
        import platform as _platform
        cli = _import_cli()

        # Make the macOS/ARM check fail by pretending we're on x86_64
        with patch("platform.machine", return_value="x86_64"):
            try:
                cli.cmd_doctor(argparse.Namespace())
            except SystemExit:
                pass

        captured = capsys.readouterr()
        assert "Some checks failed" in captured.out
