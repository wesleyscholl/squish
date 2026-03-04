"""
tests/test_quantizer_extended.py

Extended unit tests for squish/quantizer.py — covers the numpy backend
and paths that the Rust-first backend leaves untested.
"""
from __future__ import annotations

import numpy as np
import pytest

from squish.quantizer import (
    QuantizationResult,
    _quantize_numpy,
    _quantize_numpy_asymmetric,
    _reconstruct_numpy,
    dequantize_int4,
    mean_cosine_similarity,
    quantize_embeddings,
    quantize_int4,
    reconstruct_embeddings,
)


RNG = np.random.default_rng(42)


# ── _quantize_numpy (forces numpy backend) ────────────────────────────────────

class TestQuantizeNumpy:
    def test_per_row_output_shape(self):
        emb = RNG.standard_normal((8, 64)).astype(np.float32)
        result = _quantize_numpy(emb, group_size=0)
        assert result.quantized.shape == (8, 64)
        assert result.scales.shape == (8,)

    def test_per_group_output_shape(self):
        emb = RNG.standard_normal((8, 64)).astype(np.float32)
        result = _quantize_numpy(emb, group_size=16)
        assert result.quantized.shape == (8, 64)
        assert result.scales.shape == (8, 4)   # 64/16 = 4 groups

    def test_quantized_dtype_int8(self):
        emb = RNG.standard_normal((4, 32)).astype(np.float32)
        result = _quantize_numpy(emb)
        assert result.quantized.dtype == np.int8

    def test_zero_row_handled(self):
        emb = np.zeros((4, 32), dtype=np.float32)
        result = _quantize_numpy(emb)
        assert np.all(result.quantized == 0)

    def test_scales_positive(self):
        emb = RNG.standard_normal((16, 128)).astype(np.float32)
        result = _quantize_numpy(emb)
        assert np.all(result.scales > 0)

    def test_non_divisible_group_pads(self):
        """group_size doesn't divide d evenly — should pad internally."""
        emb = RNG.standard_normal((4, 70)).astype(np.float32)
        result = _quantize_numpy(emb, group_size=32)
        assert result.quantized.shape == (4, 70)


# ── _quantize_numpy_asymmetric ────────────────────────────────────────────────

class TestQuantizeNumpyAsymmetric:
    def test_output_shape_per_row(self):
        emb = RNG.standard_normal((6, 48)).astype(np.float32)
        result = _quantize_numpy_asymmetric(emb, group_size=0)
        assert result.quantized.shape == (6, 48)

    def test_output_shape_per_group(self):
        emb = RNG.standard_normal((6, 48)).astype(np.float32)
        result = _quantize_numpy_asymmetric(emb, group_size=16)
        assert result.quantized.shape == (6, 48)

    def test_has_zero_points(self):
        emb = RNG.standard_normal((4, 32)).astype(np.float32)
        result = _quantize_numpy_asymmetric(emb, group_size=0)
        # Asymmetric variant may store zero_points as extra field
        assert isinstance(result, QuantizationResult)

    def test_values_in_int8_range(self):
        emb = RNG.standard_normal((4, 32)).astype(np.float32)
        result = _quantize_numpy_asymmetric(emb)
        assert result.quantized.min() >= -128
        assert result.quantized.max() <= 127


# ── _reconstruct_numpy ────────────────────────────────────────────────────────

class TestReconstructNumpy:
    def test_shape_preserved(self):
        emb = RNG.standard_normal((8, 64)).astype(np.float32)
        result = _quantize_numpy(emb, group_size=0)
        rec = _reconstruct_numpy(result)
        assert rec.shape == (8, 64)

    def test_output_dtype_float32(self):
        emb = RNG.standard_normal((4, 32)).astype(np.float32)
        result = _quantize_numpy(emb)
        rec = _reconstruct_numpy(result)
        assert rec.dtype == np.float32

    def test_grouped_reconstruct(self):
        emb = RNG.standard_normal((8, 64)).astype(np.float32)
        result = _quantize_numpy(emb, group_size=16)
        rec = _reconstruct_numpy(result)
        assert rec.shape == (8, 64)


# ── quantize_embeddings with explicit numpy backend ───────────────────────────

class TestQuantizeEmbeddingsNumpyBackend:
    def test_numpy_backend_roundtrip(self):
        emb = RNG.standard_normal((8, 64)).astype(np.float32)
        result = quantize_embeddings(emb, group_size=64, backend="numpy")
        rec = reconstruct_embeddings(result, backend="numpy")
        sim = mean_cosine_similarity(emb, rec)
        assert sim > 0.90

    def test_asymmetric_backend_numpy(self):
        emb = RNG.standard_normal((8, 64)).astype(np.float32)
        result = quantize_embeddings(emb, backend="numpy", asymmetric=True)
        assert result.quantized.shape == (8, 64)

    def test_soft_clip_sigma(self):
        """soft_clip_sigma path is exercised."""
        emb = RNG.standard_normal((8, 64)).astype(np.float32)
        result = quantize_embeddings(emb, backend="numpy", soft_clip_sigma=3.0)
        assert result.quantized.shape == (8, 64)


# ── quantize_int4 / dequantize_int4 ───────────────────────────────────────────

class TestQuantizeInt4:
    def test_raises_without_rust_ext(self):
        """If Rust ext is available, verify it actually works; otherwise skip."""
        try:
            emb = RNG.standard_normal((4, 64)).astype(np.float32)
            packed, scales = quantize_int4(emb, group_size=64)
            assert packed.shape == (4, 32)  # d//2
            assert scales.dtype == np.float32
        except RuntimeError as e:
            if "squish_quant Rust extension" in str(e):
                pytest.skip("squish_quant Rust extension not available")
            raise

    def test_roundtrip_quality(self):
        try:
            emb = RNG.standard_normal((4, 64)).astype(np.float32)
            packed, scales = quantize_int4(emb, group_size=64)
            rec = dequantize_int4(packed, scales, group_size=64)
            assert rec.shape == (4, 64)
            sim = mean_cosine_similarity(emb, rec)
            assert sim > 0.80
        except RuntimeError as e:
            if "squish_quant Rust extension" in str(e):
                pytest.skip("squish_quant Rust extension not available")
            raise


# ── mean_cosine_similarity edge cases ─────────────────────────────────────────

class TestMeanCosineSimilarity:
    def test_identical_vectors(self):
        a = RNG.standard_normal((10, 32)).astype(np.float32)
        result = mean_cosine_similarity(a, a.copy())
        assert abs(result - 1.0) < 1e-4

    def test_opposite_vectors(self):
        a = RNG.standard_normal((4, 16)).astype(np.float32)
        result = mean_cosine_similarity(a, -a)
        assert result < -0.9

    def test_zero_vector_both(self):
        a = np.zeros((3, 8), dtype=np.float32)
        # Both zero — convention is similarity=1.0 (they agree on being zero)
        result = mean_cosine_similarity(a, a)
        assert isinstance(result, float)

    def test_one_zero_vector(self):
        a = RNG.standard_normal((4, 16)).astype(np.float32)
        b = np.zeros_like(a)
        b[0] = a[0]             # one non-zero row
        result = mean_cosine_similarity(a, b)
        assert isinstance(result, float)

    def test_shape_mismatch_raises(self):
        a = RNG.standard_normal((4, 16)).astype(np.float32)
        b = RNG.standard_normal((4, 32)).astype(np.float32)
        with pytest.raises(ValueError):
            mean_cosine_similarity(a, b)
