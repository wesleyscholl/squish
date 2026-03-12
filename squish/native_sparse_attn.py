"""squish/native_sparse_attn.py

NativeSparseAttention — Block-sparse attention with local sliding window,
inspired by DeepSeek-V3 Native Sparse Attention (NSA).

Standard full-attention is O(N²) in both compute and memory with respect to
sequence length N.  For long-context inference this becomes prohibitive.  Two
complementary sparsity patterns together cover the key attention dependencies
found in practice:

1. **Local sliding window** — the last ``window_size`` key/value tokens are
   attended to with full causal attention, capturing recency and local
   syntactic structure.

2. **Global sparse blocks** — the full key sequence is partitioned into
   non-overlapping blocks of ``block_size`` tokens.  For each query block, a
   mean-pooled representative is computed and used to score every KV block via
   dot-product; the ``top_k_blocks`` highest-scoring blocks are then included
   in the attention mask.  This captures long-range dependencies without
   attending to every position.

The combined pattern attends to at most
``window_size + top_k_blocks * block_size`` key positions per query token,
giving sub-quadratic cost when ``block_size`` and ``top_k_blocks`` are small
relative to N.

Causal masking is applied after both selection steps, so no future key
positions are leaked even when a "future" block is selected by the block
scorer.

The :attr:`~NativeSparseAttention.sparsity` property reports the fraction of
attention weight pairs that were zeroed in the most recent
:meth:`~NativeSparseAttention.forward` call.

Example usage::

    import numpy as np
    from squish.native_sparse_attn import NSAConfig, NativeSparseAttention

    cfg = NSAConfig(n_heads=4, head_dim=32, block_size=8,
                    top_k_blocks=2, window_size=32)
    nsa = NativeSparseAttention(cfg)

    rng = np.random.default_rng(7)
    seq = 64
    q   = rng.standard_normal((4, seq, 32)).astype(np.float32)
    k   = rng.standard_normal((4, seq, 32)).astype(np.float32)
    v   = rng.standard_normal((4, seq, 32)).astype(np.float32)
    out = nsa.forward(q, k, v)
    print(out.shape)              # (4, 64, 32)
    print(f"sparsity={nsa.sparsity:.3f}")
"""

from __future__ import annotations

__all__ = ["NSAConfig", "NativeSparseAttention"]

import math
from dataclasses import dataclass

import numpy as np


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class NSAConfig:
    """Configuration for Native Sparse Attention.

    Attributes:
        n_heads:      Number of attention heads.
        head_dim:     Dimension of each attention head.
        block_size:   Number of tokens per KV block used for sparse block
                      selection.  Must be >= 1.
        top_k_blocks: Number of non-local KV blocks each query block attends
                      to (selected by mean-pooled dot product).  Must be >= 1.
        window_size:  Number of recent tokens in the local causal sliding
                      window.  Must be >= 1.
    """

    n_heads:      int
    head_dim:     int
    block_size:   int = 64
    top_k_blocks: int = 4
    window_size:  int = 256

    def __post_init__(self) -> None:
        if self.n_heads < 1:
            raise ValueError(f"n_heads must be >= 1, got {self.n_heads}")
        if self.head_dim < 1:
            raise ValueError(f"head_dim must be >= 1, got {self.head_dim}")
        if self.block_size < 1:
            raise ValueError(f"block_size must be >= 1, got {self.block_size}")
        if self.top_k_blocks < 1:
            raise ValueError(
                f"top_k_blocks must be >= 1, got {self.top_k_blocks}"
            )
        if self.window_size < 1:
            raise ValueError(f"window_size must be >= 1, got {self.window_size}")


# ---------------------------------------------------------------------------
# Sparse attention
# ---------------------------------------------------------------------------


class NativeSparseAttention:
    """Block-sparse + sliding window attention.

    Each query position attends to:

    * The local sliding window of the most recent ``window_size`` causal
      tokens (positions ``max(0, abs_pos - window_size + 1)`` to
      ``abs_pos`` inclusive).
    * The ``top_k_blocks`` most-relevant non-local KV blocks as scored by
      mean-pooled query-key dot products at block granularity.

    Both selection steps are combined via a boolean OR, then a strict causal
    mask is applied so queries never attend to future keys regardless of
    what the block selector chose.

    Args:
        config: An :class:`NSAConfig` instance.
    """

    def __init__(self, config: NSAConfig) -> None:
        self._cfg            = config
        self._last_sparsity: float = 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def forward(
        self,
        q: np.ndarray,
        k: np.ndarray,
        v: np.ndarray,
    ) -> np.ndarray:
        """Apply sparse causal attention over the full sequence.

        Args:
            q: Query tensor of shape ``(n_heads, seq_q, head_dim)`` (float32).
            k: Key tensor of shape ``(n_heads, seq_k, head_dim)`` (float32).
            v: Value tensor of shape ``(n_heads, seq_k, head_dim)`` (float32).
               Must have the same shape as *k*.

        Returns:
            Output tensor of shape ``(n_heads, seq_q, head_dim)`` (float32).

        Raises:
            ValueError: If any tensor has the wrong number of dimensions,
                        mismatched head counts / head dims, or *v* and *k*
                        have different shapes.
        """
        cfg = self._cfg
        q   = np.asarray(q, dtype=np.float32)
        k   = np.asarray(k, dtype=np.float32)
        v   = np.asarray(v, dtype=np.float32)

        self._validate_qkv(q, k, v)

        n_heads, seq_q, head_dim = q.shape
        seq_k = k.shape[1]
        scale = 1.0 / math.sqrt(head_dim)

        # Initialise the combined sparse mask: False = masked out.
        mask = np.zeros((n_heads, seq_q, seq_k), dtype=np.bool_)

        # ── 1. Local sliding window ───────────────────────────────────
        # For query at position qi (relative to the query window), map to its
        # absolute key-sequence position abs_qi.  Callers are responsible for
        # supplying q/k/v consistently (e.g., the last seq_q positions of the
        # full sequence share the last seq_k KV tokens).
        for qi in range(seq_q):
            abs_qi       = (seq_k - seq_q) + qi
            window_start = max(0, abs_qi - cfg.window_size + 1)
            mask[:, qi, window_start : abs_qi + 1] = True

        # ── 2. Top-k global sparse blocks ────────────────────────────
        block_indices = self._select_top_blocks(q, k)
        # block_indices: (n_heads, n_q_blocks, effective_top_k)
        n_q_blocks     = block_indices.shape[1]
        effective_top_k = block_indices.shape[2]
        bs             = cfg.block_size

        for qi_block in range(n_q_blocks):
            q_start = qi_block * bs
            q_end   = min(q_start + bs, seq_q)
            for h in range(n_heads):
                for bi in range(effective_top_k):
                    kv_block_idx = int(block_indices[h, qi_block, bi])
                    k_start      = kv_block_idx * bs
                    k_end        = min(k_start + bs, seq_k)
                    mask[h, q_start:q_end, k_start:k_end] = True

        # ── 3. Strict causal mask ─────────────────────────────────────
        # Build a causal mask: qi can only attend to abs_qi and earlier keys.
        causal_mask = np.zeros((seq_q, seq_k), dtype=np.bool_)
        for qi in range(seq_q):
            abs_qi = (seq_k - seq_q) + qi
            causal_mask[qi, : abs_qi + 1] = True

        mask &= causal_mask[np.newaxis, :, :]  # broadcast over heads

        # ── 4. Sparse scaled dot-product attention ────────────────────
        # scores: (n_heads, seq_q, seq_k)
        scores = np.einsum("hqd,hkd->hqk", q, k) * scale

        # Mask out non-attended positions with a large negative constant.
        _NEG_INF = np.float32(-1e9)
        scores   = np.where(mask, scores, _NEG_INF)

        # Numerically stable softmax.
        scores_max = np.max(scores, axis=-1, keepdims=True)
        exp_s      = np.exp(scores - scores_max)
        # Zero out rows that were entirely masked to prevent NaN division.
        any_valid  = mask.any(axis=-1, keepdims=True)  # (n_heads, seq_q, 1)
        exp_s      = np.where(any_valid, exp_s, np.float32(0.0))
        denom      = np.sum(exp_s, axis=-1, keepdims=True)
        safe_denom = np.where(denom < 1e-12, np.float32(1.0), denom)
        attn       = exp_s / safe_denom  # (n_heads, seq_q, seq_k)

        # Output: (n_heads, seq_q, head_dim)
        output = np.einsum("hqk,hkd->hqd", attn, v).astype(np.float32)

        # Record sparsity for the ``sparsity`` property.
        total_pairs    = n_heads * seq_q * seq_k
        attended_pairs = int(mask.sum())
        self._last_sparsity = (
            1.0 - attended_pairs / total_pairs if total_pairs > 0 else 0.0
        )

        return output

    def _select_top_blocks(
        self,
        q: np.ndarray,
        k: np.ndarray,
    ) -> np.ndarray:
        """Select the top-k most-relevant KV blocks for each query block.

        Keys are split into non-overlapping blocks of size ``block_size``.
        Each block's representative is the element-wise mean of its key
        vectors.  For each query block (also mean-pooled over its tokens) the
        dot product with every KV-block representative is computed.  The
        ``top_k_blocks`` highest-scoring KV blocks are returned per query
        block per head.

        Args:
            q: Query tensor of shape ``(n_heads, seq_q, head_dim)``.
            k: Key tensor of shape ``(n_heads, seq_k, head_dim)``.

        Returns:
            Integer array of shape
            ``(n_heads, n_q_blocks, min(top_k_blocks, n_kv_blocks))``
            containing selected KV block indices.
        """
        cfg         = self._cfg
        n_heads     = q.shape[0]
        seq_q       = q.shape[1]
        seq_k       = k.shape[1]
        bs          = cfg.block_size
        n_q_blocks  = math.ceil(seq_q / bs)
        n_kv_blocks = math.ceil(seq_k / bs)

        # Mean-pool query blocks: (n_heads, n_q_blocks, head_dim).
        q_blocks = np.zeros((n_heads, n_q_blocks, cfg.head_dim), dtype=np.float32)
        for bi in range(n_q_blocks):
            q_blocks[:, bi, :] = q[:, bi * bs : (bi + 1) * bs, :].mean(axis=1)

        # Mean-pool key blocks: (n_heads, n_kv_blocks, head_dim).
        k_blocks = np.zeros((n_heads, n_kv_blocks, cfg.head_dim), dtype=np.float32)
        for bi in range(n_kv_blocks):
            k_blocks[:, bi, :] = k[:, bi * bs : (bi + 1) * bs, :].mean(axis=1)

        # Block-level attention scores: (n_heads, n_q_blocks, n_kv_blocks).
        block_scores = np.einsum("hqd,hkd->hqk", q_blocks, k_blocks)

        effective_top_k = min(cfg.top_k_blocks, n_kv_blocks)
        result = np.zeros(
            (n_heads, n_q_blocks, effective_top_k), dtype=np.int64
        )

        for h in range(n_heads):
            for qb in range(n_q_blocks):
                row = block_scores[h, qb]  # (n_kv_blocks,)
                if effective_top_k >= n_kv_blocks:
                    result[h, qb] = np.argsort(-row)[:effective_top_k]
                else:
                    part         = np.argpartition(-row, effective_top_k)[:effective_top_k]
                    result[h, qb] = part[np.argsort(-row[part])]

        return result

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def sparsity(self) -> float:
        """Fraction of attention weight pairs zeroed in the last forward call.

        Returns ``0.0`` before the first call to :meth:`forward`.  A value of
        ``0.9`` means 90 % of the ``(seq_q, seq_k)`` attention matrix was
        masked out.
        """
        return self._last_sparsity

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _validate_qkv(
        self,
        q: np.ndarray,
        k: np.ndarray,
        v: np.ndarray,
    ) -> None:
        cfg = self._cfg
        for name, arr in (("q", q), ("k", k), ("v", v)):
            if arr.ndim != 3:
                raise ValueError(
                    f"{name} must be 3-D (n_heads, seq, head_dim), "
                    f"got shape {arr.shape}."
                )
        if q.shape[0] != cfg.n_heads:
            raise ValueError(
                f"q n_heads={q.shape[0]} does not match "
                f"config.n_heads={cfg.n_heads}."
            )
        if q.shape[2] != cfg.head_dim:
            raise ValueError(
                f"q head_dim={q.shape[2]} does not match "
                f"config.head_dim={cfg.head_dim}."
            )
        if k.shape[0] != cfg.n_heads or k.shape[2] != cfg.head_dim:
            raise ValueError(
                f"k must have shape (n_heads={cfg.n_heads}, seq_k, "
                f"head_dim={cfg.head_dim}), got {k.shape}."
            )
        if v.shape != k.shape:
            raise ValueError(
                f"v must have the same shape as k ({k.shape}), "
                f"got {v.shape}."
            )
