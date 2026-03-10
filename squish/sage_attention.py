"""SageAttention — INT8 quantized attention compute.

Implements the per-block INT8 smoothed QK^T matmul technique from
SageAttention (ICLR 2025, arXiv:2410.02367) that achieves ~2.1× speedup
over FlashAttention2 with near-lossless accuracy.

The core insight: the two large matrix multiplications inside attention
(Q×K^T and P×V) can be quantized to INT8 without meaningful accuracy loss
when K channel outliers are smoothed before quantization.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np


@dataclass
class SageAttentionConfig:
    """Configuration for SageAttention INT8 quantization.

    Args:
        head_dim: Dimension of each attention head.
        n_heads: Number of attention heads.
        block_size: Block size for per-block quantization statistics.
        smooth_k: Apply channel-wise K smoothing to reduce outliers before INT8.
        smooth_alpha: EMA factor for updating the K outlier scale (0 < alpha <= 1).
        qk_bits: Bit-width for Q and K quantization (8 or 4).
        pv_bits: Bit-width for P×V computation (8 or 16).
        fallback_threshold: Per-block max-abs value above which full precision fallback triggers.
    """

    head_dim: int = 128
    n_heads: int = 32
    block_size: int = 64
    smooth_k: bool = True
    smooth_alpha: float = 0.01
    qk_bits: int = 8
    pv_bits: int = 16
    fallback_threshold: float = 100.0

    def __post_init__(self) -> None:
        if self.head_dim <= 0:
            raise ValueError("head_dim must be positive")
        if self.n_heads <= 0:
            raise ValueError("n_heads must be positive")
        if self.block_size <= 0:
            raise ValueError("block_size must be positive")
        if not 0 < self.smooth_alpha <= 1.0:
            raise ValueError("smooth_alpha must be in (0, 1]")
        if self.qk_bits not in (4, 8):
            raise ValueError("qk_bits must be 4 or 8")
        if self.pv_bits not in (8, 16):
            raise ValueError("pv_bits must be 8 or 16")
        if self.fallback_threshold < 0:
            raise ValueError("fallback_threshold must be non-negative")

    @property
    def scale(self) -> float:
        """Canonical 1/sqrt(head_dim) attention scale."""
        return 1.0 / math.sqrt(self.head_dim)

    @property
    def qk_clamp(self) -> float:
        """Symmetric INT clamp bound for qk_bits."""
        return float(2 ** (self.qk_bits - 1) - 1)


@dataclass
class KSmoother:
    """Online per-channel K outlier smoother (EMA of per-channel max-abs).

    Maintains a running scale estimate so that K / scale has reduced
    outliers, making INT8 quantization more accurate.
    """

    config: SageAttentionConfig
    _scales: np.ndarray | None = field(default=None, repr=False)

    def update_and_smooth(self, k: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Update EMA scales and return (smoothed_k, scales_used).

        Args:
            k: Key tensor, shape (..., seq_len, head_dim).

        Returns:
            Tuple of (smoothed_k, per_channel_scales).
        """
        per_channel_max = np.abs(k).max(axis=tuple(range(k.ndim - 1))).clip(min=1e-6)
        if self._scales is None:
            self._scales = per_channel_max.copy()
        else:
            alpha = self.config.smooth_alpha
            self._scales = (1 - alpha) * self._scales + alpha * per_channel_max
        smoothed = k / self._scales
        return smoothed, self._scales.copy()

    def reset(self) -> None:
        """Reset the running scale estimates."""
        self._scales = None


def _quantize_to_int8(x: np.ndarray, clamp: float = 127.0) -> tuple[np.ndarray, np.ndarray]:
    """Symmetric per-block INT8 quantization.

    Args:
        x: Input float array, shape (n_blocks, block_size, ...).
        clamp: Maximum INT value (127 for INT8).

    Returns:
        Tuple of (quantized_int8, per_block_scale).
    """
    block_max = np.abs(x).max(axis=-1, keepdims=True).clip(min=1e-6)
    scale = block_max / clamp
    x_int = np.round(x / scale).clip(-clamp, clamp).astype(np.int8)
    return x_int, scale.astype(np.float32)


def _dequantize(x_int: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """Dequantize INT8 back to float32."""
    return x_int.astype(np.float32) * scale


def simulate_sage_qk(
    q: np.ndarray,
    k: np.ndarray,
    config: SageAttentionConfig,
    k_scales: np.ndarray | None = None,
) -> tuple[np.ndarray, SageAttentionStats]:
    """Simulate the SageAttention INT8 QK^T computation.

    Implements the algorithmic core:
    1. Apply K channel smoothing.
    2. Quantize Q and K to INT8 per block.
    3. Compute INT8 Q×K^T, dequantize, apply scale.
    4. Track fallback blocks where max-abs exceeds threshold.

    Args:
        q: Query, shape (n_heads, seq_q, head_dim).
        k: Key, shape (n_heads, seq_k, head_dim).
        config: SageAttentionConfig instance.
        k_scales: Optional pre-computed smoothing scales.

    Returns:
        Tuple of (attn_logits float32, SageAttentionStats).
    """
    n_heads, seq_q, head_dim = q.shape
    _, seq_k, _ = k.shape
    clamp = config.qk_clamp

    fallback_blocks = 0
    total_blocks = 0

    attn_logits = np.zeros((n_heads, seq_q, seq_k), dtype=np.float32)

    for h in range(n_heads):
        q_h = q[h]  # (seq_q, head_dim)
        k_h = k[h]  # (seq_k, head_dim)

        # Apply smoothing to K
        if config.smooth_k and k_scales is not None:
            k_h = k_h / k_scales.clip(min=1e-6)

        # Block quantization of Q
        q_blocks = _block_split(q_h, config.block_size)  # list of (bs, d)

        block_logits = np.zeros((seq_q, seq_k), dtype=np.float32)

        for qi, qb in enumerate(q_blocks):
            q_start = qi * config.block_size
            q_end = min(q_start + config.block_size, seq_q)

            # Check fallback threshold
            if np.abs(qb).max() > config.fallback_threshold:
                fallback_blocks += 1
                # Full precision for this block
                q_fp = qb.astype(np.float32)
                k_fp = k_h.astype(np.float32)
                block_logits[q_start:q_end, :] = q_fp @ k_fp.T * config.scale
            else:
                q_int, q_scale = _quantize_to_int8(qb, clamp)
                k_int, k_scale = _quantize_to_int8(k_h, clamp)
                # Simulate INT8 matmul and dequantize
                result = q_int.astype(np.int32) @ k_int.astype(np.int32).T
                dq = result.astype(np.float32) * (q_scale * k_scale.T) * config.scale
                block_logits[q_start:q_end, :] = dq

            total_blocks += 1

        attn_logits[h] = block_logits

    stats = SageAttentionStats(
        total_blocks=total_blocks,
        fallback_blocks=fallback_blocks,
        qk_bits=config.qk_bits,
        pv_bits=config.pv_bits,
    )
    return attn_logits, stats


def _block_split(x: np.ndarray, block_size: int):
    """Split (seq, d) along seq into a list of (block_size, d) arrays."""
    seq = x.shape[0]
    blocks = []
    for start in range(0, seq, block_size):
        blocks.append(x[start : start + block_size])
    return blocks


@dataclass
class SageAttentionStats:
    """Runtime statistics for SageAttention."""

    total_blocks: int = 0
    fallback_blocks: int = 0
    qk_bits: int = 8
    pv_bits: int = 16

    @property
    def fallback_rate(self) -> float:
        """Fraction of blocks that fell back to full precision."""
        if self.total_blocks == 0:
            return 0.0
        return self.fallback_blocks / self.total_blocks

    @property
    def int_compute_fraction(self) -> float:
        """Fraction of blocks computed in quantized mode."""
        return 1.0 - self.fallback_rate

    @property
    def estimated_speedup_vs_fp16(self) -> float:
        """Rough estimated speedup based on paper's ~2.1× for INT8.

        Assumes fallback blocks run at FP16 speed (1.0×),
        INT8 blocks at 2.1×.
        """
        int_speedup = 2.1 if self.qk_bits == 8 else 3.1
        # weighted average of quantized + fallback blocks
        return 1.0 / (
            self.fallback_rate * 1.0 + self.int_compute_fraction * (1.0 / int_speedup)
        )

    def merge(self, other: SageAttentionStats) -> SageAttentionStats:
        """Return merged stats from two forward passes."""
        return SageAttentionStats(
            total_blocks=self.total_blocks + other.total_blocks,
            fallback_blocks=self.fallback_blocks + other.fallback_blocks,
            qk_bits=self.qk_bits,
            pv_bits=self.pv_bits,
        )


class SageAttentionKernel:
    """Stateful SageAttention kernel wrapping KSmoother + forward pass.

    Usage::

        kernel = SageAttentionKernel(config)
        logits, stats = kernel.forward(q, k, v)
    """

    def __init__(self, config: SageAttentionConfig) -> None:
        self.config = config
        self._smoother = KSmoother(config=config)
        self._cumulative_stats = SageAttentionStats(
            qk_bits=config.qk_bits, pv_bits=config.pv_bits
        )

    def forward(
        self, q: np.ndarray, k: np.ndarray, v: np.ndarray
    ) -> tuple[np.ndarray, SageAttentionStats]:
        """Run SageAttention forward pass.

        Args:
            q: (n_heads, seq_q, head_dim)
            k: (n_heads, seq_k, head_dim)
            v: (n_heads, seq_k, head_dim)

        Returns:
            Tuple of (output float32, SageAttentionStats for this call).
        """
        # Step 1: smooth K
        k_flat = k.reshape(-1, self.config.head_dim)
        k_smoothed_flat, k_scales = self._smoother.update_and_smooth(k_flat)
        k_smoothed = k_smoothed_flat.reshape(k.shape)

        # Step 2: quantized QK^T
        attn_logits, stats = simulate_sage_qk(q, k_smoothed, self.config, k_scales)

        # Step 3: softmax
        attn_logits -= attn_logits.max(axis=-1, keepdims=True)
        attn_weights = np.exp(attn_logits)
        attn_weights /= attn_weights.sum(axis=-1, keepdims=True) + 1e-9

        # Step 4: PV matmul (FP16 simulation in FP32 here)
        output = attn_weights @ v.astype(np.float32)

        self._cumulative_stats = self._cumulative_stats.merge(stats)
        return output, stats

    @property
    def cumulative_stats(self) -> SageAttentionStats:
        return self._cumulative_stats

    def reset_stats(self) -> None:
        self._cumulative_stats = SageAttentionStats(
            qk_bits=self.config.qk_bits, pv_bits=self.config.pv_bits
        )

    def reset_smoother(self) -> None:
        self._smoother.reset()
