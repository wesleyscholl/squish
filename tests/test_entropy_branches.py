"""
tests/test_entropy_branches.py

Branch coverage for squish/entropy.py:
  - compress_npy_dir: sentinel exists (line 75)
  - benchmark_compression: plain .npy items (lines 221-225)
"""
from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pytest


# ── compress_npy_dir: sentinel already present ───────────────────────────────

class TestCompressNpyDirSentinel:
    def test_sentinel_present_verbose_prints_skip(self, tmp_path: Path, capsys):
        zstd = pytest.importorskip("zstandard")
        from squish.entropy import compress_npy_dir

        # Write at least one .npy file so the dir check passes
        arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        tensors_dir = tmp_path / "tensors"
        tensors_dir.mkdir()
        np.save(tensors_dir / "layer.npy", arr)

        # Create the sentinel file
        sentinel = tensors_dir.parent / ".squish_zst_ready"
        sentinel.touch()

        result = compress_npy_dir(tensors_dir, verbose=True)
        captured = capsys.readouterr()
        assert result == {}
        assert "Already compressed" in captured.out or "skipping" in captured.out.lower()

    def test_sentinel_present_silent(self, tmp_path: Path):
        zstd = pytest.importorskip("zstandard")
        from squish.entropy import compress_npy_dir

        arr = np.array([1.0], dtype=np.float32)
        tensors_dir = tmp_path / "tensors"
        tensors_dir.mkdir()
        np.save(tensors_dir / "layer.npy", arr)

        sentinel = tensors_dir.parent / ".squish_zst_ready"
        sentinel.touch()

        result = compress_npy_dir(tensors_dir, verbose=False)
        assert result == {}


# ── benchmark_compression ─────────────────────────────────────────────────────

class TestBenchmarkCompression:
    def test_runs_with_npy_files(self, tmp_path: Path, capsys):
        """benchmark_compression on a dir with .npy files produces output."""
        zstd = pytest.importorskip("zstandard")
        from squish.entropy import benchmark_compression

        arr = np.zeros((4, 4), dtype=np.float32)
        tensors_dir = tmp_path / "tensors"
        tensors_dir.mkdir()
        np.save(tensors_dir / "weights.npy", arr)

        benchmark_compression(tensors_dir)
        captured = capsys.readouterr()
        assert "weights.npy" in captured.out or "Tensor" in captured.out


# ── decompress_npy_dir: sentinel absent (branch [152, 155]) ──────────────────

class TestDecompressNpyDirNoSentinel:
    def test_no_sentinel_decompresses_and_continues(self, tmp_path: Path, capsys):
        """
        When sentinel file does NOT exist, the unlink step is skipped and
        execution continues to the verbose print  (line 152→155).
        """
        zstd = pytest.importorskip("zstandard")
        import zstandard
        from squish.entropy import decompress_npy_dir

        # Create a .npy.zst file
        tensors_dir = tmp_path / "tensors"
        tensors_dir.mkdir()
        arr = np.array([1.0, 2.0], dtype=np.float32)
        npy_buf = io.BytesIO()
        np.save(npy_buf, arr)

        cctx = zstandard.ZstdCompressor()
        zst_data = cctx.compress(npy_buf.getvalue())
        (tensors_dir / "layer.npy.zst").write_bytes(zst_data)

        # Ensure sentinel does NOT exist
        sentinel = tensors_dir.parent / ".squish_zst_ready"
        assert not sentinel.exists()

        decompress_npy_dir(tensors_dir, verbose=True)

        captured = capsys.readouterr()
        assert "Decompressed" in captured.out or "tensor" in captured.out.lower()
