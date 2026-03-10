"""SageAttention2 — INT4 QK + FP8 PV with per-warp smoothing.

Extends SageAttention (ICML 2025, arXiv:2411.10958) by pushing quantization
from INT8 → INT4 for Q/K and adding FP8 for P×V.  The key challenge of INT4's
restricted range [-7, +7] is overcome via per-warp smoothing: statistics are
computed at warp granularity (groups of 32 elements) rather than per-tensor.

Estimated speedup vs FlashAttention2: ~3.1× (INT8 fallback: ~2.1×).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np

# INT4 clamp constant (symmetric: -7 .. +7 to avoid min-int issues)
_INT4_CLAMP = 7.0
_INT8_CLAMP = 127.0
_WARP_SIZE = 32  # emulated warp granularity


@dataclass
class SageAttention2Config:
    """Configuration for SageAttention2.

    Args:
        head_dim: Dimension of each attention head.
        n_heads: Number of attention heads.
        block_size: Block size for per-block statistics (must be multiple of warp_size).
        warp_size: Emulated warp size for per-warp smoothing (default 32).
        use_int4: Use INT4 for QK (falls back to INT8 per block if range exceeded).
        use_fp8_pv: Use FP8 precision for P×V computation.
        int4_fallback_threshold: Per-warp max-abs above which a block falls back to INT8.
        smooth_alpha: EMA factor for updating per-channel K smoothing scales.
    """

    head_dim: int = 128
    n_heads: int = 32
    block_size: int = 64
    warp_size: int = 32
    use_int4: bool = True
    use_fp8_pv: bool = True
    int4_fallback_threshold: float = 6.5
    smooth_alpha: float = 0.01

    def __post_init__(self) -> None:
        if self.head_dim <= 0:
            raise ValueError("head_dim must be positive")
        if self.n_heads <= 0:
            raise ValueError("n_heads must be positive")
        if self.block_size < self.warp_size:
            raise ValueError("block_size must be >= warp_size")
        if not 0 < self.smooth_alpha <= 1.0:
            raise ValueError("smooth_alpha must be in (0, 1]")
        if self.int4_fallback_threshold <= 0:
            raise ValueError("int4_fallback_threshold must be positive")

    @property
    def scale(self) -> float:
        return 1.0 / math.sqrt(self.head_dim)

    @property
    def active_qk_bits(self) -> int:
        return 4 if self.use_int4 else 8

    @property
    def active_pv_bits(self) -> int:
        return 8 if self.use_fp8_pv else 16


@dataclass
class WarpQuantResult:
    """Result of per-warp INT4 quantization for one block."""

    data_int: np.ndarray      # quantized values as int8 array (fits ±7)
    warp_scales: np.ndarray   # per-warp scales, shape (n_warps,)
    used_int4: bool           # True = INT4, False = fell back to INT8


def warp_quantize_int4(
    x: np.ndarray, warp_size: int, fallback_threshold: float
) -> WarpQuantResult:
    """Per-warp INT4 quantization with INT8 fallback.

    Splits the last dimension of x into warp-sized chunks; computes
    per-warp max-abs scale; checks threshold and returns INT4 if safe.

    Args:
        x: Input array, shape (..., dim).
        warp_size: Elements per warp group.
        fallback_threshold: Max per-warp element above which we use INT8.

    Returns:
        WarpQuantResult with quantized data.
    """
    flat = x.reshape(-1)
    n_warps = max(1, (flat.size + warp_size - 1) // warp_size)
    padded_size = n_warps * warp_size
    padded = np.zeros(padded_size, dtype=np.float32)
    padded[: flat.size] = flat.astype(np.float32)
    warps = padded.reshape(n_warps, warp_size)

    warp_max = np.abs(warps).max(axis=1).clip(min=1e-6)  # (n_warps,)

    # Decide precision: if any warp max exceeds threshold, fall back to INT8
    use_int4 = bool(warp_max.max() <= fallback_threshold)
    clamp = _INT4_CLAMP if use_int4 else _INT8_CLAMP

    scales = warp_max / clamp  # (n_warps,)
    normalized = warps / scales[:, None]
    quantized = np.round(normalized).clip(-clamp, clamp).astype(np.int8)
    return WarpQuantResult(
        data_int=quantized.reshape(x.shape[:-1] + (-1,)),
        warp_scales=scales,
        used_int4=use_int4,
    )


def _fp8_simulate(x: np.ndarray) -> np.ndarray:
    """Simulate FP8 E4M3 precision by rounding to 4-bit mantissa."""
    # Approximate FP8 E4M3: max value ~448, 4 mantissa bits → 1/16 precision
    sign = np.sign(x)
    abs_x = np.abs(x).clip(max=448.0)
    # Quantize mantissa to 4 bits: round to nearest 1/16 of the exponent scale
    exponent = np.floor(np.log2(abs_x.clip(min=1e-38)))
    scale = 2.0 ** (exponent - 3)  # 2^(exp-3) = step size for E4M3
    quantized = np.round(abs_x / scale.clip(min=1e-38)) * scale.clip(min=1e-38)
    return (sign * quantized).astype(np.float32)


def simulate_sage2_attention(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    config: SageAttention2Config,
    k_scales: np.ndarray | None = None,
) -> tuple[np.ndarray, SageAttention2Stats]:
    """Full SageAttention2 forward pass simulation.

    Args:
        q: (n_heads, seq_q, head_dim)
        k: (n_heads, seq_k, head_dim)
        v: (n_heads, seq_k, head_dim)
        config: SageAttention2Config
        k_scales: Optional per-channel K smoothing scales.

    Returns:
        Tuple of (output float32, SageAttention2Stats).
    """
    n_heads, seq_q, head_dim = q.shape
    _, seq_k, _ = k.shape

    int4_blocks = 0
    int8_fallback_blocks = 0
    total_blocks = 0

    output = np.zeros_like(q, dtype=np.float32)

    for h in range(n_heads):
        q_h = q[h].astype(np.float32)
        k_h = k[h].astype(np.float32)
        v_h = v[h].astype(np.float32)

        if k_scales is not None:
            k_h = k_h / k_scales.clip(min=1e-6)

        # FP8 for V
        if config.use_fp8_pv:
            v_h = _fp8_simulate(v_h)

        # Block-level QK^T with per-warp INT4/INT8 quantization
        attn_logits = np.zeros((seq_q, seq_k), dtype=np.float32)

        for q_start in range(0, seq_q, config.block_size):
            q_end = min(q_start + config.block_size, seq_q)
            qb = q_h[q_start:q_end]  # (bs, d)

            wqr = warp_quantize_int4(qb, config.warp_size, config.int4_fallback_threshold)
            wkr = warp_quantize_int4(k_h, config.warp_size, config.int4_fallback_threshold)

            # Reconstruct float for matmul (simulate integer matmul)
            q_recon = _reconstruct(wqr, config.warp_size)[:q_end - q_start]
            k_recon = _reconstruct(wkr, config.warp_size)[:seq_k]

            attn_logits[q_start:q_end] = (q_recon @ k_recon.T) * config.scale

            if wqr.used_int4 and wkr.used_int4:
                int4_blocks += 1
            else:
                int8_fallback_blocks += 1
            total_blocks += 1

        # Softmax
        attn_logits -= attn_logits.max(axis=-1, keepdims=True)
        weights = np.exp(attn_logits)
        weights /= weights.sum(axis=-1, keepdims=True) + 1e-9

        # PV (FP8 simulated V)
        output[h] = weights @ v_h

    stats = SageAttention2Stats(
        total_blocks=total_blocks,
        int4_blocks=int4_blocks,
        int8_fallback_blocks=int8_fallback_blocks,
        used_fp8_pv=config.use_fp8_pv,
    )
    return output, stats


def _reconstruct(wqr: WarpQuantResult, warp_size: int) -> np.ndarray:
    """Reconstruct float array from WarpQuantResult."""
    flat_int = wqr.data_int.reshape(-1).astype(np.float32)
    n_warps = wqr.warp_scales.size
    padded_size = n_warps * warp_size
    padded = np.zeros(padded_size, dtype=np.float32)
    padded[: flat_int.size] = flat_int
    warps = padded.reshape(n_warps, warp_size)
    recon_flat = (warps * wqr.warp_scales[:, None]).reshape(-1)
    return recon_flat[: wqr.data_int.reshape(-1).size].reshape(wqr.data_int.shape)


@dataclass
class SageAttention2Stats:
    """Runtime statistics for SageAttention2."""

    total_blocks: int = 0
    int4_blocks: int = 0
    int8_fallback_blocks: int = 0
    used_fp8_pv: bool = True

    @property
    def int4_rate(self) -> float:
        if self.total_blocks == 0:
            return 0.0
        return self.int4_blocks / self.total_blocks

    @property
    def int8_rate(self) -> float:
        if self.total_blocks == 0:
            return 0.0
        return self.int8_fallback_blocks / self.total_blocks

    @property
    def estimated_speedup_vs_fa2(self) -> float:
        """Weighted speedup: INT4 blocks at 3.1×, INT8 fallbacks at 2.1×."""
        if self.total_blocks == 0:
            return 1.0
        return 1.0 / (
            self.int4_rate / 3.1 + self.int8_rate / 2.1
        )

    def merge(self, other: SageAttention2Stats) -> SageAttention2Stats:
        return SageAttention2Stats(
            total_blocks=self.total_blocks + other.total_blocks,
            int4_blocks=self.int4_blocks + other.int4_blocks,
            int8_fallback_blocks=self.int8_fallback_blocks + other.int8_fallback_blocks,
            used_fp8_pv=self.used_fp8_pv,
        )


class SageAttention2Kernel:
    """Stateful SageAttention2 kernel with per-channel K smoothing."""

    def __init__(self, config: SageAttention2Config) -> None:
        self.config = config
        self._k_scales: np.ndarray | None = None
        self._cumulative_stats = SageAttention2Stats(used_fp8_pv=config.use_fp8_pv)

    def _update_k_scales(self, k: np.ndarray) -> np.ndarray:
        """EMA update of per-channel K scales for smoothing."""
        flat = k.reshape(-1, self.config.head_dim)
        per_ch = np.abs(flat).max(axis=0).clip(min=1e-6)
        if self._k_scales is None:
            self._k_scales = per_ch.copy()
        else:
            a = self.config.smooth_alpha
            self._k_scales = (1 - a) * self._k_scales + a * per_ch
        return self._k_scales.copy()

    def forward(
        self, q: np.ndarray, k: np.ndarray, v: np.ndarray
    ) -> tuple[np.ndarray, SageAttention2Stats]:
        k_scales = self._update_k_scales(k)
        output, stats = simulate_sage2_attention(q, k, v, self.config, k_scales)
        self._cumulative_stats = self._cumulative_stats.merge(stats)
        return output, stats

    @property
    def cumulative_stats(self) -> SageAttention2Stats:
        return self._cumulative_stats

    def reset(self) -> None:
        self._k_scales = None
        self._cumulative_stats = SageAttention2Stats(used_fp8_pv=self.config.use_fp8_pv)
