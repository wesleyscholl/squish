"""FlashDecode — Split-KV parallel decode attention.

Standard attention at decode time reads the *entire* KV cache for each query
token.  Flash Decode (Dao et al., 2023; MLSys 2024) splits the KV cache into
``n_splits`` independent chunks, computes a partial softmax per chunk in
parallel, then merges the partial results using the log-sum-exp trick.  This
allows maximum KV-read parallelism and is memory-bandwidth-optimal for the
single-query decode setting (batch_size × seq_len = 1 × seq_len).

On CPUs the split reduces contiguous L2/L3 evictions; on Apple Silicon the
splits map naturally to ANE tiles.

Usage::

    from squish.flash_decode import FlashDecodeAttention, FlashDecodeConfig

    cfg   = FlashDecodeConfig(n_heads=32, head_dim=128, n_splits=8)
    attn  = FlashDecodeAttention(cfg)
    out   = attn.decode(q, k_cache, v_cache)  # q: (n_heads, head_dim)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

__all__ = [
    "FlashDecodeConfig",
    "FlashDecodeAttention",
    "FlashDecodeSplit",
    "merge_split_results",
]


@dataclass
class FlashDecodeConfig:
    """Configuration for FlashDecode split-KV attention.

    Attributes:
        n_heads: Number of query heads.
        head_dim: Dimension per head.
        n_splits: Number of KV-cache splits.  Should be a power of 2.
        softmax_scale: Optional override; defaults to ``1/sqrt(head_dim)``.
        kv_n_heads: KV heads for GQA.  Defaults to ``n_heads``.
    """

    n_heads: int = 32
    head_dim: int = 128
    n_splits: int = 8
    softmax_scale: Optional[float] = None
    kv_n_heads: Optional[int] = None

    def __post_init__(self) -> None:
        if self.n_heads < 1:
            raise ValueError(f"n_heads must be positive; got {self.n_heads}")
        if self.head_dim < 1:
            raise ValueError(f"head_dim must be positive; got {self.head_dim}")
        if self.n_splits < 1:
            raise ValueError(f"n_splits must be positive; got {self.n_splits}")
        if self.softmax_scale is None:
            object.__setattr__(self, "softmax_scale", 1.0 / math.sqrt(self.head_dim))
        if self.kv_n_heads is None:
            object.__setattr__(self, "kv_n_heads", self.n_heads)

    @property
    def effective_scale(self) -> float:
        return self.softmax_scale  # type: ignore[return-value]

    @property
    def kv_group_size(self) -> int:
        return self.n_heads // self.kv_n_heads  # type: ignore[operator]


@dataclass
class FlashDecodeSplit:
    """Partial attention result for one KV split.

    Attributes:
        output: Weighted value sum, shape ``(n_heads, head_dim)``, float32.
        log_sum_exp: Running log-sum-exp for numerically stable merge,
            shape ``(n_heads,)``, float32.
        max_score: Per-head max logit for this split, shape ``(n_heads,)``.
    """

    output: np.ndarray
    log_sum_exp: np.ndarray
    max_score: np.ndarray


def _softmax_with_lse(scores: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute softmax + log-sum-exp for a score vector.

    Args:
        scores: Float32 array of shape ``(seq_len,)``.

    Returns:
        (softmax weights, log_sum_exp scalar, max_score scalar)
    """
    m = float(np.max(scores))
    shifted = scores - m
    exp_s = np.exp(shifted)
    lse = float(np.log(np.sum(exp_s))) + m
    return (exp_s / np.sum(exp_s)).astype(np.float32), np.float32(lse), np.float32(m)


def merge_split_results(splits: list[FlashDecodeSplit]) -> np.ndarray:
    """Merge partial FlashDecode results using the log-sum-exp trick.

    Args:
        splits: List of :class:`FlashDecodeSplit` instances, one per KV split.

    Returns:
        Final attention output, shape ``(n_heads, head_dim)``, float32.
    """
    if not splits:
        raise ValueError("splits list must be non-empty")
    n_heads, head_dim = splits[0].output.shape

    # Global max log-sum-exp per head for numerical stability
    global_lse = np.full(n_heads, -1e30, dtype=np.float32)
    for s in splits:
        global_lse = np.maximum(global_lse, s.log_sum_exp)

    # Weighted sum: weight = exp(local_lse - global_lse)
    output = np.zeros((n_heads, head_dim), dtype=np.float32)
    total_weight = np.zeros(n_heads, dtype=np.float32)
    for s in splits:
        w = np.exp(s.log_sum_exp - global_lse)  # shape (n_heads,)
        output += w[:, None] * s.output
        total_weight += w

    denom = np.where(total_weight > 1e-30, total_weight, 1.0)[:, None]
    return (output / denom).astype(np.float32)


class FlashDecodeAttention:
    """Flash-decode attention for single-query decode step.

    Splits the KV cache into ``n_splits`` chunks, computes per-split
    partial softmax attention, then merges results with the log-sum-exp
    reduction.

    Args:
        config: :class:`FlashDecodeConfig` instance.
    """

    def __init__(self, config: FlashDecodeConfig) -> None:
        self.config = config
        self._step_count: int = 0
        self._total_kv_len: int = 0

    def _compute_split(
        self,
        q: np.ndarray,
        k_split: np.ndarray,
        v_split: np.ndarray,
    ) -> FlashDecodeSplit:
        """Attend to one KV split.

        Args:
            q: Query, shape ``(n_heads, head_dim)``.
            k_split: Key slice, shape ``(kv_n_heads, split_len, head_dim)``.
            v_split: Value slice, shape ``(kv_n_heads, split_len, head_dim)``.

        Returns:
            :class:`FlashDecodeSplit` partial result.
        """
        cfg = self.config
        n_heads, head_dim = q.shape
        kv_n_heads = k_split.shape[0]
        group = n_heads // kv_n_heads

        outputs = np.zeros((n_heads, head_dim), dtype=np.float32)
        lse     = np.zeros(n_heads, dtype=np.float32)
        max_s   = np.zeros(n_heads, dtype=np.float32)

        for h in range(n_heads):
            kv_h = h // group
            scores = k_split[kv_h] @ q[h] * cfg.effective_scale  # (split_len,)
            w, lse_h, m_h = _softmax_with_lse(scores)
            out_h = v_split[kv_h].T @ w                          # (head_dim,)
            outputs[h] = out_h
            lse[h]     = lse_h
            max_s[h]   = m_h

        return FlashDecodeSplit(output=outputs, log_sum_exp=lse, max_score=max_s)

    def decode(
        self,
        q: np.ndarray,
        k_cache: np.ndarray,
        v_cache: np.ndarray,
    ) -> np.ndarray:
        """Run flash-decode attention for a single query token.

        Args:
            q: Query tensor, shape ``(n_heads, head_dim)`` float32.
            k_cache: Full key cache, shape ``(kv_n_heads, seq_len, head_dim)``.
            v_cache: Full value cache, shape ``(kv_n_heads, seq_len, head_dim)``.

        Returns:
            Attention output, shape ``(n_heads, head_dim)``, float32.
        """
        cfg = self.config
        seq_len = k_cache.shape[1]
        n_splits = min(cfg.n_splits, seq_len)
        split_len = (seq_len + n_splits - 1) // n_splits

        splits: list[FlashDecodeSplit] = []
        for s in range(n_splits):
            start = s * split_len
            end   = min(start + split_len, seq_len)
            if start >= seq_len:
                break
            k_s = k_cache[:, start:end, :]
            v_s = v_cache[:, start:end, :]
            splits.append(self._compute_split(q, k_s, v_s))

        self._step_count += 1
        self._total_kv_len += seq_len
        return merge_split_results(splits)

    @property
    def avg_kv_len(self) -> float:
        """Average KV sequence length across all decode calls."""
        if self._step_count == 0:
            return 0.0
        return self._total_kv_len / self._step_count

    def reset_stats(self) -> None:
        """Reset accumulated decode statistics."""
        self._step_count = 0
        self._total_kv_len = 0
