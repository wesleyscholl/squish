"""
tests/test_quantizer_full.py

Comprehensive unit tests for squish/quantizer.py.
Covers QuantizationResult, quantize_embeddings, reconstruct_embeddings,
quantize_int4, dequantize_int4, mean_cosine_similarity, get_backend_info,
asymmetric path, grouped path, and edge cases.
"""
from __future__ import annotations

import numpy as np
import pytest

from squish.quantizer import (
    QuantizationResult,
    dequantize_int4,
    get_backend_info,
    mean_cosine_similarity,
    quantize_embeddings,
    quantize_int4,
    reconstruct_embeddings,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

RNG = np.random.default_rng(42)


def _rand(n: int, d: int) -> np.ndarray:
    return RNG.standard_normal((n, d)).astype(np.float32)


# ── QuantizationResult ────────────────────────────────────────────────────────

class TestQuantizationResult:
    def test_fields_accessible(self):
        q = np.zeros((4, 8), dtype=np.int8)
        s = np.ones(4, dtype=np.float32)
        r = QuantizationResult(quantized=q, scales=s, dims=8, n=4)
        assert r.dims == 8
        assert r.n == 4
        assert r.zero_points is None

    def test_with_zero_points(self):
        q = np.zeros((2, 4), dtype=np.int8)
        s = np.ones(2, dtype=np.float32)
        zp = np.zeros(2, dtype=np.int32)
        r = QuantizationResult(quantized=q, scales=s, dims=4, n=2, zero_points=zp)
        assert r.zero_points is not None
        assert r.zero_points.shape == (2,)

    def test_is_named_tuple(self):
        r = QuantizationResult(np.zeros((1, 2), dtype=np.int8),
                               np.ones(1, dtype=np.float32), 2, 1)
        assert hasattr(r, "_fields")
        assert "quantized" in r._fields


# ── quantize_embeddings / reconstruct_embeddings ──────────────────────────────

class TestQuantizeReconstructRoundTrip:
    def test_basic_shape(self):
        arr = _rand(16, 64)
        r = quantize_embeddings(arr)
        assert isinstance(r, QuantizationResult)
        assert r.quantized.shape == (16, 64)
        assert r.scales.shape == (16,)
        assert r.dims == 64
        assert r.n == 16

    def test_round_trip_cosine(self):
        arr = _rand(32, 128)
        r = quantize_embeddings(arr)
        rec = reconstruct_embeddings(r)
        sim = mean_cosine_similarity(arr, rec)
        assert sim > 0.999, f"cosine {sim:.6f} < 0.999"

    def test_round_trip_large(self):
        arr = _rand(64, 512)
        r = quantize_embeddings(arr)
        rec = reconstruct_embeddings(r)
        assert rec.shape == arr.shape
        assert np.allclose(arr, rec, atol=0.1)

    def test_output_dtype(self):
        arr = _rand(8, 16)
        r = quantize_embeddings(arr)
        assert r.quantized.dtype == np.int8
        assert r.scales.dtype == np.float32

    def test_reconstruct_dtype(self):
        arr = _rand(4, 8)
        r = quantize_embeddings(arr)
        rec = reconstruct_embeddings(r)
        assert rec.dtype == np.float32

    def test_zero_vector(self):
        arr = np.zeros((4, 16), dtype=np.float32)
        r = quantize_embeddings(arr)
        rec = reconstruct_embeddings(r)
        assert np.allclose(rec, 0, atol=1e-6)

    def test_single_row(self):
        arr = _rand(1, 64)
        r = quantize_embeddings(arr)
        rec = reconstruct_embeddings(r)
        assert mean_cosine_similarity(arr, rec) > 0.999

    def test_grouped_quantization(self):
        arr = _rand(8, 128)
        r = quantize_embeddings(arr, group_size=32)
        assert r.scales.shape == (8, 4)  # 128/32 = 4 groups
        rec = reconstruct_embeddings(r)
        assert mean_cosine_similarity(arr, rec) > 0.999

    def test_grouped_non_divisible_raises(self):
        # Rust backend enforces n_cols % group_size == 0
        arr = _rand(4, 70)
        with pytest.raises((ValueError, Exception)):
            quantize_embeddings(arr, group_size=32)

    def test_grouped_divisible(self):
        # 70 columns, group_size=14 (70/14=5 groups) — cleanly divisible
        arr = _rand(4, 70)
        r = quantize_embeddings(arr, group_size=14)
        rec = reconstruct_embeddings(r)
        assert rec.shape == (4, 70)
        assert mean_cosine_similarity(arr, rec) > 0.9

    def test_group_size_larger_than_d_falls_back_to_per_row(self):
        arr = _rand(4, 16)
        r = quantize_embeddings(arr, group_size=64)
        assert r.scales.ndim == 1
        assert r.scales.shape == (4,)

    def test_accepts_float64_input(self):
        arr = _rand(4, 16).astype(np.float64)
        r = quantize_embeddings(arr)
        rec = reconstruct_embeddings(r)
        assert rec.dtype == np.float32

    def test_accepts_c_contiguous_input(self):
        arr = np.ascontiguousarray(_rand(4, 16))
        r = quantize_embeddings(arr)
        assert r.quantized.dtype == np.int8

    def test_high_dynamic_range(self):
        arr = np.array([[100.0, -200.0, 0.0, 50.0]], dtype=np.float32)
        r = quantize_embeddings(arr)
        rec = reconstruct_embeddings(r)
        assert mean_cosine_similarity(arr, rec) > 0.999


# ── Asymmetric quantization ───────────────────────────────────────────────────

class TestAsymmetricQuantization:
    def test_asymmetric_round_trip(self):
        from squish.quantizer import _quantize_numpy_asymmetric
        arr = _rand(8, 64)
        r = _quantize_numpy_asymmetric(arr)
        assert r.zero_points is not None
        rec = reconstruct_embeddings(r)
        assert mean_cosine_similarity(arr, rec) > 0.995

    def test_asymmetric_with_groups(self):
        from squish.quantizer import _quantize_numpy_asymmetric
        arr = _rand(4, 64)
        r = _quantize_numpy_asymmetric(arr, group_size=16)
        assert r.zero_points is not None
        assert r.scales.ndim == 2
        rec = reconstruct_embeddings(r)
        assert rec.shape == arr.shape

    def test_asymmetric_positive_only(self):
        from squish.quantizer import _quantize_numpy_asymmetric
        arr = np.abs(_rand(4, 32)) + 1.0  # all positive
        r = _quantize_numpy_asymmetric(arr)
        rec = reconstruct_embeddings(r)
        assert mean_cosine_similarity(arr, rec) > 0.99

    def test_asymmetric_has_zero_points(self):
        from squish.quantizer import _quantize_numpy_asymmetric
        arr = _rand(4, 16)
        r = _quantize_numpy_asymmetric(arr)
        assert r.zero_points is not None
        assert r.zero_points.dtype == np.int32


# ── INT4 quantization ─────────────────────────────────────────────────────────

class TestInt4:
    def test_basic_shape(self):
        arr = _rand(4, 32)
        packed, scales = quantize_int4(arr, group_size=8)
        # Each pair of int4 values packed into one uint8 → 16 bytes per row
        assert packed.shape[0] == 4
        assert packed.dtype == np.uint8
        assert scales.shape[0] == 4

    def test_round_trip(self):
        arr = _rand(8, 64)
        packed, scales = quantize_int4(arr, group_size=16)
        rec = dequantize_int4(packed, scales, group_size=16)
        assert rec.shape == arr.shape
        assert rec.dtype == np.float32
        # INT4 accuracy is coarser; cosine similarity should still be high
        sim = mean_cosine_similarity(arr, rec)
        assert sim > 0.97

    def test_default_group_size(self):
        arr = _rand(4, 64)
        packed, scales = quantize_int4(arr)
        rec = dequantize_int4(packed, scales)
        assert rec.shape == arr.shape

    def test_packed_dtype(self):
        arr = _rand(4, 32)
        packed, scales = quantize_int4(arr, group_size=8)
        assert packed.dtype == np.uint8

    def test_scales_dtype(self):
        arr = _rand(4, 32)
        packed, scales = quantize_int4(arr, group_size=8)
        assert scales.dtype == np.float32


# ── mean_cosine_similarity ────────────────────────────────────────────────────

class TestMeanCosineSimilarity:
    def test_identical(self):
        a = _rand(10, 64)
        assert mean_cosine_similarity(a, a) == pytest.approx(1.0, abs=1e-5)

    def test_opposite(self):
        a = _rand(1, 32)
        sim = mean_cosine_similarity(a, -a)
        assert sim == pytest.approx(-1.0, abs=1e-5)

    def test_orthogonal(self):
        a = np.array([[1.0, 0.0]], dtype=np.float32)
        b = np.array([[0.0, 1.0]], dtype=np.float32)
        assert mean_cosine_similarity(a, b) == pytest.approx(0.0, abs=1e-5)

    def test_batch(self):
        a = _rand(16, 64)
        b = _rand(16, 64)
        sim = mean_cosine_similarity(a, b)
        assert -1.0 <= sim <= 1.0

    def test_returns_float(self):
        a = _rand(4, 8)
        sim = mean_cosine_similarity(a, a)
        assert isinstance(sim, float)


# ── get_backend_info ──────────────────────────────────────────────────────────

class TestGetBackendInfo:
    def test_returns_dict(self):
        info = get_backend_info()
        assert isinstance(info, dict)

    def test_non_empty(self):
        info = get_backend_info()
        assert len(info) > 0

    def test_values_are_bool(self):
        info = get_backend_info()
        for k, v in info.items():
            assert isinstance(v, bool), f"{k}: {v!r} is not bool"

    def test_has_numpy_key(self):
        info = get_backend_info()
        assert "numpy" in info

    def test_numpy_available(self):
        info = get_backend_info()
        # numpy must always be True
        assert info["numpy"] is True
