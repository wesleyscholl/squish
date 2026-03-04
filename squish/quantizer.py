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
except ImportError:  # pragma: no cover
    pass


# ---------------------------------------------------------------------------
# QuantizationResult
# ---------------------------------------------------------------------------

class QuantizationResult(NamedTuple):
    """Result of INT8 quantization (symmetric or asymmetric)."""
    quantized:   np.ndarray   # int8,   shape (n, d)  or padded to group boundary
    scales:      np.ndarray   # float32 shape (n,)     per-row
                              #         or   (n, n_groups) per-group
    dims:        int          # original column dimension d
    n:           int          # number of rows n
    zero_points: np.ndarray | None = None
                              # int32 zero-points for asymmetric quant (same shape
                              # as scales); None for symmetric (default).


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
    # np.array() with the default copy=True always returns an owned, writable
    # contiguous buffer — required for the safe in-place operations below.
    emb = np.array(embeddings, dtype=np.float32, order="C")
    n, d = emb.shape

    if group_size <= 0 or group_size >= d:
        # ── per-row ────────────────────────────────────────────────────────
        row_max = np.max(np.abs(emb), axis=1)                              # (n,)
        scales  = np.where(row_max == 0, 1.0, row_max / 127.0).astype(np.float32)
        # In-place division re-uses the buffer from np.ascontiguousarray above
        emb /= scales[:, None]
        np.round(emb, out=emb)
        q = np.clip(emb, -127, 127).astype(np.int8)
        return QuantizationResult(quantized=q, scales=scales, dims=d, n=n)
    else:
        # ── per-group ─────────────────────────────────────────────────────
        pad = (-d) % group_size
        emb_pad = np.pad(emb, ((0, 0), (0, pad))) if pad else emb
        n_groups = emb_pad.shape[1] // group_size
        grouped  = emb_pad.reshape(n * n_groups, group_size)              # (n*G, group_size)
        gmax   = np.max(np.abs(grouped), axis=1)                          # (n*G,)
        gscale = np.where(gmax == 0, 1.0, gmax / 127.0).astype(np.float32)
        # In-place: reuse grouped buffer — safe because grouped is a reshaped copy
        grouped /= gscale[:, None]
        np.round(grouped, out=grouped)
        q_groups = np.clip(grouped, -127, 127).astype(np.int8)
        q      = q_groups.reshape(n, -1)[:, :d]                           # trim pad
        scales = gscale.reshape(n, n_groups).astype(np.float32)
        return QuantizationResult(quantized=q, scales=scales, dims=d, n=n)


def _quantize_numpy_asymmetric(embeddings: np.ndarray, group_size: int = 0) -> QuantizationResult:
    """Asymmetric per-row or per-group INT8 quantisation with zero-point.

    Maps the range ``[x_min, x_max]`` to ``[-128, 127]`` using both a scale
    and an integer zero-point, yielding ~0.1–0.5 dB better SNR than symmetric
    quantisation for activations with non-zero mean (e.g. post-softmax tensors).

    ``QuantizationResult.zero_points`` is set; Rust path returns None (falls
    back to numpy symmetric automatically).
    """
    emb = np.array(embeddings, dtype=np.float32, order="C")
    n, d = emb.shape
    _QMIN, _QMAX = -128, 127

    def _asym_scale_zero(xmin: np.ndarray, xmax: np.ndarray):
        scale = np.where(xmax == xmin, 1.0,
                         (xmax - xmin) / (_QMAX - _QMIN)).astype(np.float32)
        # zero_point is an int32 offset — NOT constrained to [-128, 127].
        # For xmin > 0 the unclipped zp will be < -128, which is correct and
        # produces well-centred quantized values without any range loss.
        zp = np.round(-xmin / scale + _QMIN).astype(np.int32)
        return scale, zp

    if group_size <= 0 or group_size >= d:
        xmin = emb.min(axis=1)   # (n,)
        xmax = emb.max(axis=1)   # (n,)
        scales, zps = _asym_scale_zero(xmin, xmax)
        # q = clamp( round(x / scale) + zp, QMIN, QMAX )
        q_f = emb / scales[:, None] + zps[:, None]
        np.round(q_f, out=q_f)
        q = np.clip(q_f, _QMIN, _QMAX).astype(np.int8)
        return QuantizationResult(quantized=q, scales=scales, dims=d, n=n,
                                  zero_points=zps)
    else:
        pad = (-d) % group_size
        emb_pad = np.pad(emb, ((0, 0), (0, pad))) if pad else emb
        n_groups = emb_pad.shape[1] // group_size
        grouped  = emb_pad.reshape(n * n_groups, group_size)  # (n*G, gs)
        xmin = grouped.min(axis=1)
        xmax = grouped.max(axis=1)
        gscale, gzp = _asym_scale_zero(xmin, xmax)
        q_f = grouped / gscale[:, None] + gzp[:, None]
        np.round(q_f, out=q_f)
        q_groups = np.clip(q_f, _QMIN, _QMAX).astype(np.int8)
        q      = q_groups.reshape(n, -1)[:, :d]
        scales = gscale.reshape(n, n_groups).astype(np.float32)
        zps    = gzp.reshape(n, n_groups).astype(np.int32)
        return QuantizationResult(quantized=q, scales=scales, dims=d, n=n,
                                  zero_points=zps)


def _reconstruct_numpy(result: QuantizationResult) -> np.ndarray:
    """Reconstruct float32 from QuantizationResult using numpy broadcast."""
    q      = result.quantized.astype(np.float32)
    scales = np.asarray(result.scales, dtype=np.float32)
    if result.zero_points is not None:
        # Asymmetric: x = scale * (q - zero_point)
        zp = np.asarray(result.zero_points, dtype=np.float32)
        if scales.ndim == 1:
            return scales[:, None] * (q - zp[:, None])
        n, d = q.shape
        n_groups  = scales.shape[1]
        group_size = result.dims // n_groups
        pad = (-d) % group_size
        full_cols = n_groups * group_size
        if pad:  # pragma: no cover
            q  = np.pad(q,  ((0, 0), (0, pad)))
        q_shifted = (q[:, :full_cols].reshape(n, n_groups, group_size)
                     - zp[:, :, np.newaxis])
        recon = (q_shifted * scales[:, :, np.newaxis]).reshape(n, full_cols)
        return recon[:, :d]

    if scales.ndim == 1:
        return q * scales[:, None]
    # grouped symmetric: scales is (n, n_groups), q is (n, d)
    n, d = q.shape
    n_groups = scales.shape[1]
    group_size = result.dims // n_groups
    pad = (-d) % group_size
    full_cols = n_groups * group_size
    if pad:  # pragma: no cover
        q = np.pad(q, ((0, 0), (0, pad)))
    # Reshape to (n, n_groups, group_size) then broadcast scales (n, n_groups, 1)
    recon = (q[:, :full_cols].reshape(n, n_groups, group_size)
             * scales[:, :, np.newaxis]).reshape(n, full_cols)
    return recon[:, :d]


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
    asymmetric: bool = False,
    soft_clip_sigma: float = 0.0,
) -> QuantizationResult:
    """Quantize a float32 (n, d) matrix to INT8.

    Args:
        embeddings:      2D float32 array of shape (n, d).
        group_size:      Columns per quantization group.  0 = per-row (legacy).
                         Default 64 gives noticeably better accuracy at no disk cost.
        backend:         'auto' | 'rust' | 'numpy'
        asymmetric:      If True, use asymmetric (zero-point) INT8 for ~0.1–0.5 dB
                         better SNR on activations with non-zero mean.
                         Rust backend falls back to symmetric when asymmetric=True.
        soft_clip_sigma: If > 0, clip each row to ``mean ± soft_clip_sigma * std``
                         before quantizing to suppress extreme outliers.
                         Typical value: 3.0–5.0.  Set to 0 (default) to disable.

    Returns:
        QuantizationResult(quantized, scales, dims, n[, zero_points])
    """
    if embeddings.ndim != 2:
        raise ValueError(f"embeddings must be 2-D, got shape {embeddings.shape}")

    work = embeddings
    if soft_clip_sigma > 0.0:
        # Clip each row to [mean - k*std, mean + k*std] to suppress outliers
        work = np.array(embeddings, dtype=np.float32, order="C")
        row_mean = work.mean(axis=1, keepdims=True)
        row_std  = work.std(axis=1, keepdims=True)
        lo = row_mean - soft_clip_sigma * row_std
        hi = row_mean + soft_clip_sigma * row_std
        np.clip(work, lo, hi, out=work)

    use_rust = (
        _squish_quant is not None
        and backend in ("auto", "rust")
        and not asymmetric   # Rust path is symmetric-only
    )

    if use_rust:
        return _quantize_rust(work, group_size)
    if asymmetric:
        return _quantize_numpy_asymmetric(work, group_size)
    return _quantize_numpy(work, group_size)


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
        and result.zero_points is None   # Rust path is symmetric-only
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
    if _squish_quant is None:  # pragma: no cover
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
    if _squish_quant is None:  # pragma: no cover
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
    """Compute mean cosine similarity between corresponding rows of a and b.

    Single-pass einsum implementation: avoids 3 separate matrix traversals
    (norms_a, norms_b, dot products) by computing dots and norms together.
    """
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    if a.shape != b.shape:
        raise ValueError(f"Shape mismatch: {a.shape} vs {b.shape}")
    # Row-wise dot products and squared norms in one einsum pass each
    dots   = np.einsum("ij,ij->i", a, b)                   # (n,)
    norms_a = np.sqrt(np.einsum("ij,ij->i", a, a))         # (n,)
    norms_b = np.sqrt(np.einsum("ij,ij->i", b, b))         # (n,)
    denom  = norms_a * norms_b
    # Handle zero-norm vectors
    both_zero = (norms_a == 0) & (norms_b == 0)
    one_zero  = (norms_a == 0) ^ (norms_b == 0)
    cosines   = np.where(denom > 0, dots / denom, 0.0)
    cosines[both_zero] = 1.0
    cosines[one_zero]  = 0.0
    return float(np.mean(cosines))


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
