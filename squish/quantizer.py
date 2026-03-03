#!/usr/bin/env python3
"""
squish/quantizer.py

Self-contained INT8/INT4 quantization engine — no external dependencies beyond
numpy (+ optional squish_quant Rust extension for 4× throughput).

This module replaces the vectro/python/interface.py dependency entirely so that
squish installs and runs on any Apple Silicon machine with `pip install squish`
without needing a sibling ~/vectro checkout.

Public API (mirrors vectro interface for drop-in compatibility):
    QuantizationResult          — NamedTuple: (quantized, scales, dims, n)
    quantize_embeddings(arr)    — float32 (n,d) → QuantizationResult
    reconstruct_embeddings(r)   — QuantizationResult → float32 (n,d)
    quantize_int4(arr, gs)      — float32 (n,d) → (packed_uint8, scales)
    dequantize_int4(p, s, gs)   — (packed_uint8, scales) → float32 (n,d)
    mean_cosine_similarity(a,b) — float
    get_backend_info()          — dict of available backends

Backend priority (auto):
    1. squish_quant  (Rust/Rayon/SIMD — 6+ GB/s, `pip install squish[quant]`)
    2. numpy vectorised broadcast (~1.5 GB/s, always available)
"""
from __future__ import annotations

from typing import NamedTuple

import numpy as np

# ---------------------------------------------------------------------------
# Optional Rust extension (squish_quant — built with maturin)
# ---------------------------------------------------------------------------
_squish_quant = None
try:
    import squish_quant as _squish_quant  # type: ignore[import]
except ImportError:
    pass


# ---------------------------------------------------------------------------
# QuantizationResult
# ---------------------------------------------------------------------------

class QuantizationResult(NamedTuple):
    """Result of INT8 quantization."""
    quantized: np.ndarray   # int8,   shape (n, d)  or padded to group boundary
    scales:    np.ndarray   # float32 shape (n,)     per-row
                            #         or   (n, n_groups) per-group
    dims:      int          # original column dimension d
    n:         int          # number of rows n


# ---------------------------------------------------------------------------
# Rust-backed paths
# ---------------------------------------------------------------------------

def _quantize_rust(embeddings: np.ndarray, group_size: int = 0) -> QuantizationResult:
    emb = np.ascontiguousarray(embeddings, dtype=np.float32)
    n, d = emb.shape
    if group_size <= 0 or group_size >= d:
        q, scales = _squish_quant.quantize_int8_f32(emb)
    else:
        q, scales = _squish_quant.quantize_int8_grouped(emb, group_size)
    return QuantizationResult(quantized=q, scales=scales, dims=d, n=n)


def _reconstruct_rust(result: QuantizationResult) -> np.ndarray:
    q = np.ascontiguousarray(result.quantized, dtype=np.int8)
    s = np.ascontiguousarray(result.scales, dtype=np.float32)
    if s.ndim == 1:
        return _squish_quant.dequantize_int8_f32(q, s)
    group_size = result.dims // s.shape[1]
    return _squish_quant.dequantize_int8_grouped(q, s, group_size)


# ---------------------------------------------------------------------------
# Pure-NumPy vectorised paths
# ---------------------------------------------------------------------------

def _quantize_numpy(embeddings: np.ndarray, group_size: int = 0) -> QuantizationResult:
    """Vectorised per-row or per-group INT8 quantisation.

    group_size=0  → 1 scale per row   (classic approach, smallest scale overhead)
    group_size=N  → 1 scale per group of N columns (better accuracy, more scales)
    """
    emb = np.ascontiguousarray(embeddings, dtype=np.float32)
    n, d = emb.shape

    if group_size <= 0 or group_size >= d:
        # ── per-row ────────────────────────────────────────────────────────
        row_max = np.max(np.abs(emb), axis=1)                              # (n,)
        scales  = np.where(row_max == 0, 1.0, row_max / 127.0).astype(np.float32)
        q = np.clip(np.round(emb / scales[:, None]), -127, 127).astype(np.int8)
        return QuantizationResult(quantized=q, scales=scales, dims=d, n=n)
    else:
        # ── per-group ─────────────────────────────────────────────────────
        pad = (-d) % group_size
        emb_pad = np.pad(emb, ((0, 0), (0, pad))) if pad else emb
        n_groups = emb_pad.shape[1] // group_size
        grouped  = emb_pad.reshape(n * n_groups, group_size)              # (n*G, group_size)
        gmax   = np.max(np.abs(grouped), axis=1)                          # (n*G,)
        gscale = np.where(gmax == 0, 1.0, gmax / 127.0).astype(np.float32)
        q_groups = np.clip(
            np.round(grouped / gscale[:, None]), -127, 127
        ).astype(np.int8)
        q      = q_groups.reshape(n, -1)[:, :d]                           # trim pad
        scales = gscale.reshape(n, n_groups).astype(np.float32)
        return QuantizationResult(quantized=q, scales=scales, dims=d, n=n)


def _reconstruct_numpy(result: QuantizationResult) -> np.ndarray:
    """Reconstruct float32 from QuantizationResult using numpy broadcast."""
    q      = result.quantized.astype(np.float32)
    scales = np.asarray(result.scales, dtype=np.float32)
    if scales.ndim == 1:
        return q * scales[:, None]
    # grouped: scales is (n, n_groups), q is (n, d) — expand groups
    n, d = q.shape
    n_groups = scales.shape[1]
    group_size = result.dims // n_groups
    pad = (-d) % group_size
    if pad:
        q = np.pad(q, ((0, 0), (0, pad)))
    expanded_scales = np.repeat(scales, group_size, axis=1)[:, :d + pad]
    recon = (q * expanded_scales)[:, :d]
    return recon.astype(np.float32)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_backend_info() -> dict:
    """Return dict describing which backends are available."""
    return {
        "squish_quant_rust": _squish_quant is not None,
        "numpy":             True,
    }


def quantize_embeddings(
    embeddings: np.ndarray,
    group_size: int = 64,
    backend: str = "auto",
) -> QuantizationResult:
    """Quantize a float32 (n, d) matrix to INT8.

    Args:
        embeddings: 2D float32 array of shape (n, d).
        group_size: Columns per quantization group.  0 = per-row (legacy).
                    Default 64 gives noticeably better accuracy at no disk cost.
        backend:    'auto' | 'rust' | 'numpy'

    Returns:
        QuantizationResult(quantized, scales, dims, n)
    """
    if embeddings.ndim != 2:
        raise ValueError(f"embeddings must be 2-D, got shape {embeddings.shape}")

    use_rust = (
        _squish_quant is not None
        and backend in ("auto", "rust")
    )

    if use_rust:
        return _quantize_rust(embeddings, group_size)
    return _quantize_numpy(embeddings, group_size)


def reconstruct_embeddings(
    result: QuantizationResult,
    backend: str = "auto",
) -> np.ndarray:
    """Reconstruct float32 (n, d) from a QuantizationResult.

    Args:
        result:  QuantizationResult produced by quantize_embeddings().
        backend: 'auto' | 'rust' | 'numpy'

    Returns:
        Approximated float32 array of shape (n, dims).
    """
    use_rust = (
        _squish_quant is not None
        and backend in ("auto", "rust")
    )
    if use_rust:
        return _reconstruct_rust(result)
    return _reconstruct_numpy(result)


def quantize_int4(
    embeddings: np.ndarray,
    group_size: int = 64,
) -> tuple[np.ndarray, np.ndarray]:
    """Nibble-packed INT4 quantisation (requires squish_quant Rust extension).

    Returns:
        (packed_uint8, scales_float32)
        packed has shape (n, d//2); scales has shape (n, d//group_size).

    Raises:
        RuntimeError if squish_quant Rust extension is not installed.
    """
    if _squish_quant is None:
        raise RuntimeError(
            "squish_quant Rust extension required for INT4.\n"
            "  Build: cd squish/squish_quant_rs && python3 -m maturin build --release\n"
            "  Or:    pip install squish[quant]"
        )
    emb = np.ascontiguousarray(embeddings, dtype=np.float32)
    return _squish_quant.quantize_int4_grouped(emb, group_size)


def dequantize_int4(
    packed: np.ndarray,
    scales: np.ndarray,
    group_size: int = 64,
) -> np.ndarray:
    """Reconstruct float32 from nibble-packed INT4 weights.

    Args:
        packed:     (n, d//2) uint8 — from quantize_int4().
        scales:     (n, d//group_size) float32.
        group_size: Must match the value used during quantize_int4().

    Returns:
        (n, d) float32.

    Raises:
        RuntimeError if squish_quant Rust extension is not installed.
    """
    if _squish_quant is None:
        raise RuntimeError(
            "squish_quant Rust extension required for INT4 dequantization.\n"
            "  Build: cd squish/squish_quant_rs && python3 -m maturin build --release\n"
            "  Or:    pip install squish[quant]"
        )
    return _squish_quant.dequantize_int4_grouped(
        np.ascontiguousarray(packed, dtype=np.uint8),
        np.ascontiguousarray(scales, dtype=np.float32),
        group_size,
    )


def mean_cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute mean cosine similarity between corresponding rows of a and b."""
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    if a.shape != b.shape:
        raise ValueError(f"Shape mismatch: {a.shape} vs {b.shape}")
    norms_a = np.linalg.norm(a, axis=1)
    norms_b = np.linalg.norm(b, axis=1)
    # Safe normalise — avoid division by zero
    a_safe = a / np.where(norms_a == 0, 1.0, norms_a)[:, None]
    b_safe = b / np.where(norms_b == 0, 1.0, norms_b)[:, None]
    dots = np.sum(a_safe * b_safe, axis=1)
    both_zero = (norms_a == 0) & (norms_b == 0)
    one_zero  = (norms_a == 0) ^ (norms_b == 0)
    dots[both_zero] = 1.0
    dots[one_zero]  = 0.0
    return float(np.mean(dots))


if __name__ == "__main__":
    print("squish.quantizer — backend info:")
    for k, v in get_backend_info().items():
        status = "✓" if v else "✗"
        print(f"  {status}  {k}")
    print()
    rng = np.random.default_rng(42)
    emb = rng.standard_normal((256, 4096)).astype(np.float32)
    result = quantize_embeddings(emb, group_size=64)
    recon  = reconstruct_embeddings(result)
    sim    = mean_cosine_similarity(emb, recon)
    print(f"  Round-trip cosine similarity (INT8 g64, 256×4096): {sim:.6f}")
    print(f"  Scales shape: {result.scales.shape}")
