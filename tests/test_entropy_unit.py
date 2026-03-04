"""
tests/test_entropy_unit.py

Unit tests for squish/entropy.py using temporary directories.
Tests compress/decompress/load_npy_zst without external hardware.
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

import numpy as np
import pytest

# ── skip if zstandard not available ────────────────────────────────────────
zstd = pytest.importorskip("zstandard", reason="zstandard not installed")
from squish import entropy  # noqa: E402


# ── helpers ──────────────────────────────────────────────────────────────────

def _make_npy(tmp_path: Path, name: str, arr: np.ndarray | None = None) -> Path:
    """Write a small .npy file and return its Path."""
    if arr is None:
        arr = (np.random.default_rng(0).integers(0, 128, size=(16, 16)) - 64).astype(np.int8)
    p = tmp_path / name
    np.save(str(p), arr)
    return p


def _make_npy_dir(tmp_path: Path, n: int = 3) -> Path:
    """Create a tensors/ sub-dir with n small .npy files.  Returns the dir."""
    tensors_dir = tmp_path / "tensors"
    tensors_dir.mkdir()
    rng = np.random.default_rng(42)
    for i in range(n):
        arr = (rng.integers(0, 128, size=(8, 8)) - 64).astype(np.int8)
        np.save(str(tensors_dir / f"tensor_{i}.npy"), arr)
    return tensors_dir


# ── _require_zstd ─────────────────────────────────────────────────────────────

class TestRequireZstd:
    def test_returns_module(self):
        mod = entropy._require_zstd()
        assert hasattr(mod, "ZstdCompressor")

    def test_does_not_exit_when_available(self):
        # If zstandard is installed this should return without raising SystemExit
        mod = entropy._require_zstd()
        assert mod is not None


# ── load_npy_zst ─────────────────────────────────────────────────────────────

class TestLoadNpyZst:
    def test_roundtrip(self, tmp_path):
        arr = np.arange(100, dtype=np.float32).reshape(10, 10)
        npy_path = tmp_path / "test.npy"
        np.save(str(npy_path), arr)

        # Compress manually with size so frame header includes content length
        import zstandard as _zstd
        cctx = _zstd.ZstdCompressor(level=1)
        zst_path = npy_path.with_suffix(".npy.zst")
        orig_size = npy_path.stat().st_size
        with open(npy_path, "rb") as src, open(zst_path, "wb") as dst:
            cctx.copy_stream(src, dst, size=orig_size)

        loaded = entropy.load_npy_zst(zst_path)
        np.testing.assert_array_equal(loaded, arr)

    def test_accepts_dctx_arg(self, tmp_path):
        arr = np.ones((5, 5), dtype=np.int8)
        npy_path = tmp_path / "ones.npy"
        np.save(str(npy_path), arr)

        import zstandard as _zstd
        cctx = _zstd.ZstdCompressor(level=1)
        dctx = _zstd.ZstdDecompressor()
        zst_path = npy_path.with_suffix(".npy.zst")
        orig_size = npy_path.stat().st_size
        with open(npy_path, "rb") as src, open(zst_path, "wb") as dst:
            cctx.copy_stream(src, dst, size=orig_size)

        loaded = entropy.load_npy_zst(zst_path, dctx=dctx)
        np.testing.assert_array_equal(loaded, arr)

    def test_preserves_dtype(self, tmp_path):
        for dtype in [np.float32, np.float16, np.int8, np.int32]:
            arr = np.array([1, 2, 3], dtype=dtype)
            npy_path = tmp_path / f"arr_{dtype.__name__}.npy"
            np.save(str(npy_path), arr)
            import zstandard as _zstd
            cctx = _zstd.ZstdCompressor(level=1)
            zst_path = npy_path.with_suffix(".npy.zst")
            orig_size = npy_path.stat().st_size
            with open(npy_path, "rb") as src, open(zst_path, "wb") as dst:
                cctx.copy_stream(src, dst, size=orig_size)
            loaded = entropy.load_npy_zst(zst_path)
            assert loaded.dtype == dtype


# ── compress_npy_dir ──────────────────────────────────────────────────────────

class TestCompressNpyDir:
    def test_creates_zst_files(self, tmp_path):
        tensors_dir = _make_npy_dir(tmp_path, n=2)
        stats = entropy.compress_npy_dir(tensors_dir, verbose=False)
        zst_files = list(tensors_dir.glob("*.npy.zst"))
        assert len(zst_files) == 2

    def test_removes_original_npy(self, tmp_path):
        tensors_dir = _make_npy_dir(tmp_path, n=2)
        entropy.compress_npy_dir(tensors_dir, verbose=False)
        npy_files = list(tensors_dir.glob("*.npy"))
        assert len(npy_files) == 0

    def test_creates_sentinel(self, tmp_path):
        tensors_dir = _make_npy_dir(tmp_path, n=2)
        entropy.compress_npy_dir(tensors_dir, verbose=False)
        sentinel = tensors_dir.parent / ".squish_zst_ready"
        assert sentinel.exists()

    def test_returns_stats_dict(self, tmp_path):
        tensors_dir = _make_npy_dir(tmp_path, n=2)
        stats = entropy.compress_npy_dir(tensors_dir, verbose=False)
        assert isinstance(stats, dict)
        assert "files" in stats
        assert stats["files"] == 2
        assert "ratio" in stats
        assert stats["ratio"] > 0

    def test_skips_if_already_compressed(self, tmp_path):
        tensors_dir = _make_npy_dir(tmp_path, n=2)
        # Create sentinel manually
        sentinel = tensors_dir.parent / ".squish_zst_ready"
        sentinel.write_text("squish-zst-v1")
        stats = entropy.compress_npy_dir(tensors_dir, verbose=False)
        assert stats == {}

    def test_raises_on_missing_dir(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            entropy.compress_npy_dir(tmp_path / "nonexistent", verbose=False)

    def test_raises_on_empty_dir(self, tmp_path):
        tensors_dir = tmp_path / "empty"
        tensors_dir.mkdir()
        with pytest.raises(ValueError):
            entropy.compress_npy_dir(tensors_dir, verbose=False)

    def test_verbose_output(self, tmp_path, capsys):
        tensors_dir = _make_npy_dir(tmp_path, n=1)
        entropy.compress_npy_dir(tensors_dir, verbose=True)
        captured = capsys.readouterr()
        assert len(captured.out) > 0

    def test_compressed_files_readable(self, tmp_path):
        tensors_dir = _make_npy_dir(tmp_path, n=2)
        entropy.compress_npy_dir(tensors_dir, verbose=False)
        for zst_path in sorted(tensors_dir.glob("*.npy.zst")):
            arr = entropy.load_npy_zst(zst_path)
            assert isinstance(arr, np.ndarray)


# ── decompress_npy_dir ────────────────────────────────────────────────────────

class TestDecompressNpyDir:
    def _compress_dir(self, tensors_dir: Path):
        entropy.compress_npy_dir(tensors_dir, verbose=False)

    def test_restores_npy_files(self, tmp_path):
        tensors_dir = _make_npy_dir(tmp_path, n=2)
        original_arrays = {
            p.name: np.load(str(p)) for p in sorted(tensors_dir.glob("*.npy"))
        }
        self._compress_dir(tensors_dir)
        entropy.decompress_npy_dir(tensors_dir, verbose=False)
        for name, orig in original_arrays.items():
            restored = np.load(str(tensors_dir / name))
            np.testing.assert_array_equal(restored, orig)

    def test_removes_sentinel(self, tmp_path):
        tensors_dir = _make_npy_dir(tmp_path, n=2)
        self._compress_dir(tensors_dir)
        sentinel = tensors_dir.parent / ".squish_zst_ready"
        assert sentinel.exists()
        entropy.decompress_npy_dir(tensors_dir, verbose=False)
        assert not sentinel.exists()

    def test_no_zst_files_prints_nothing(self, tmp_path, capsys):
        tensors_dir = tmp_path / "empty_tensors"
        tensors_dir.mkdir()
        # No .npy.zst files present
        entropy.decompress_npy_dir(tensors_dir, verbose=True)
        captured = capsys.readouterr()
        assert "No" in (captured.out + captured.err)

    def test_verbose_output(self, tmp_path, capsys):
        tensors_dir = _make_npy_dir(tmp_path, n=2)
        self._compress_dir(tensors_dir)
        entropy.decompress_npy_dir(tensors_dir, verbose=True)
        captured = capsys.readouterr()
        assert len(captured.out) > 0
