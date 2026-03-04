"""
tests/test_loader_utils_extras.py

Extra coverage for squish/loader_utils.py:
  - _reconstruct_numpy: __pt key without __shape key (line 167)
  - _dequantize_npy: INT4 path when squish_quant available (lines 210-212)
"""
from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pytest


# ── _reconstruct_numpy with __pt but no __shape ───────────────────────────────

class TestReconstructNumpyPtNoShape:
    def test_pt_without_shape_key(self, tmp_path: Path):
        """
        NPZ with '{sk}__pt' but no '{sk}__shape' →
        original_shape = npz[sk + '__pt'].shape (line 167)
        """
        from squish.loader_utils import _dequantize

        sk = "model.layers.0.weight"
        arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)

        # Create NPZ with __pt but WITHOUT __shape
        npz_path = tmp_path / "test.npz"
        np.savez(npz_path, **{f"{sk}__pt": arr})

        with np.load(npz_path, allow_pickle=False) as npz:
            result = _dequantize(npz, sk)

        assert result.shape == (2, 2)
        np.testing.assert_allclose(result, arr, rtol=1e-5)

    def test_pt_with_shape_key(self, tmp_path: Path):
        """Normal case: __shape key present, uses it to reshape."""
        from squish.loader_utils import _dequantize

        sk = "layer"
        arr = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)
        shape = np.array([2, 2], dtype=np.int64)

        npz_path = tmp_path / "test2.npz"
        np.savez(npz_path, **{f"{sk}__pt": arr.ravel(), f"{sk}__shape": shape})

        with np.load(npz_path, allow_pickle=False) as npz:
            result = _dequantize(npz, sk)

        assert result.shape == (2, 2)


# ── _dequantize_npy INT4 path ─────────────────────────────────────────────────

class TestDequantizeNpyInt4:
    def test_int4_roundtrip(self, tmp_path: Path):
        """
        When __q4.npy and __s4.npy exist, use INT4 dequantization (lines 210-212).
        Requires squish_quant extension.
        """
        try:
            import squish_quant  # noqa: F401
        except ImportError:
            pytest.skip("squish_quant not available")

        from squish.loader_utils import _dequantize_npy
        from squish.quantizer import quantize_int4

        rng = np.random.default_rng(42)
        emb = rng.standard_normal((4, 128)).astype(np.float32)

        packed, scales = quantize_int4(emb, group_size=64)

        sk = "weight"
        tensor_dir = tmp_path / "tensors"
        tensor_dir.mkdir()

        np.save(tensor_dir / f"{sk}__q4.npy", packed)
        np.save(tensor_dir / f"{sk}__s4.npy", scales)

        result = _dequantize_npy(tensor_dir, sk)
        assert isinstance(result, np.ndarray)
        assert result.shape == emb.shape or result.shape[1] >= emb.shape[1]


# ── _get_zstd_dctx cached path (branch [39, 45]) ─────────────────────────────

class TestGetZstdDctxCached:
    def test_second_call_returns_cached_dctx(self):
        """Second call to _get_zstd_dctx() hits the cached path (line 39 → 45)."""
        zstd = pytest.importorskip("zstandard")
        import squish.loader_utils as _lu

        # Reset the module-level cache so first call initializes it
        _lu._zstd_dctx = None
        dctx1 = _lu._get_zstd_dctx()
        assert dctx1 is not None

        # Second call should return the cached object (skips the if-block at line 39)
        dctx2 = _lu._get_zstd_dctx()
        assert dctx2 is dctx1  # same object — pulled from cache
