"""
tests/test_quantizer_extras.py

Extra coverage for squish/quantizer.py:
  - quantize_embeddings: non-2D input raises ValueError (line 246)
  - _quantize_numpy_asymmetric: grouped path (lines 154-169)
  - _reconstruct_numpy: asymmetric per-row path (line 180), grouped path (lines 182-191)
  - _reconstruct_numpy: symmetric per-row (line 194), grouped (lines 196-206)
"""
from __future__ import annotations

import numpy as np
import pytest

from squish.quantizer import QuantizationResult, quantize_embeddings, reconstruct_embeddings


# ── quantize_embeddings with non-2D input ─────────────────────────────────────

class TestQuantizeEmbeddingsValidation:
    def test_1d_raises_value_error(self):
        """1D array → ValueError (line 246)."""
        emb = np.ones(10, dtype=np.float32)
        with pytest.raises(ValueError, match="2-D"):
            quantize_embeddings(emb)

    def test_3d_raises_value_error(self):
        """3D array → ValueError (line 246)."""
        emb = np.ones((2, 3, 4), dtype=np.float32)
        with pytest.raises(ValueError, match="2-D"):
            quantize_embeddings(emb)


# ── Asymmetric grouped quantize + reconstruct ─────────────────────────────────

class TestAsymmetricGrouped:
    def test_asymmetric_grouped_exact_divisible(self):
        """dim=8, group_size=4 → no padding; roundtrip produces correct shape."""
        rng = np.random.default_rng(0)
        emb = rng.standard_normal((4, 8)).astype(np.float32)
        result = quantize_embeddings(emb, group_size=4, asymmetric=True)
        rec    = reconstruct_embeddings(result)
        assert rec.shape == emb.shape

    def test_asymmetric_per_row(self):
        """group_size=0 → per-row asymmetric (non-grouped path, lines 144-153)."""
        rng = np.random.default_rng(1)
        emb = rng.standard_normal((4, 32)).astype(np.float32)
        result = quantize_embeddings(emb, group_size=0, asymmetric=True)
        assert result.zero_points is not None
        rec = reconstruct_embeddings(result)
        assert rec.shape == emb.shape

    def test_asymmetric_grouped_has_zero_points(self):
        """Asymmetric result includes zero_points; grouped path (line 168)."""
        rng = np.random.default_rng(2)
        emb = rng.standard_normal((4, 64)).astype(np.float32)
        result = quantize_embeddings(emb, group_size=64, asymmetric=True)
        assert result.zero_points is not None

    def test_reconstruct_asymmetric_1d_scales(self):
        """Asymmetric with 1D scales → scales[:, None] * (q - zp[:, None]) path (line 180)."""
        rng = np.random.default_rng(4)
        emb = rng.standard_normal((3, 16)).astype(np.float32)
        result = quantize_embeddings(emb, group_size=0, asymmetric=True)
        assert result.scales.ndim == 1
        rec    = reconstruct_embeddings(result)
        assert rec.shape == emb.shape

    def test_reconstruct_asymmetric_grouped_2d_scales(self):
        """Asymmetric with 2D scales → grouped reconstruct path (lines 181-191)."""
        rng = np.random.default_rng(5)
        emb = rng.standard_normal((4, 64)).astype(np.float32)
        result = quantize_embeddings(emb, group_size=16, asymmetric=True)  # 4 groups/row → 2D
        assert result.scales.ndim == 2
        rec    = reconstruct_embeddings(result)
        assert rec.shape == emb.shape


# ── Symmetric grouped path ────────────────────────────────────────────────────

class TestSymmetricGrouped:
    def test_symmetric_grouped_exact(self):
        """Symmetric grouped with exact divisibility (lines 195-206)."""
        rng = np.random.default_rng(6)
        emb = rng.standard_normal((4, 64)).astype(np.float32)
        result = quantize_embeddings(emb, group_size=64, asymmetric=False, backend="numpy")
        rec    = reconstruct_embeddings(result, backend="numpy")
        assert rec.shape == emb.shape

    def test_symmetric_1d_scales_path(self):
        """Symmetric per-row (1D scales) → q * scales[:, None] (line 194)."""
        rng = np.random.default_rng(7)
        emb = rng.standard_normal((3, 16)).astype(np.float32)
        result = quantize_embeddings(emb, group_size=0, asymmetric=False, backend="numpy")
        assert result.scales.ndim == 1
        assert result.zero_points is None
        rec = reconstruct_embeddings(result)
        assert rec.shape == emb.shape
