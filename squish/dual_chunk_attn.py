"""squish/dual_chunk_attn.py

DualChunkAttention — Intra-chunk full attention plus inter-chunk compressed
attention for 1 M+ token context lengths.

Standard full self-attention is O(N²) in memory, making it infeasible for
sequences of one million or more tokens.  Dual-Chunk Attention (DCA) decomposes
the long sequence into fixed-size chunks and applies two complementary attention
patterns:

1. **Intra-chunk attention** — each chunk token attends to all other tokens
   *within the same chunk* using standard causal scaled dot-product attention.
   Memory cost per chunk is ``O(chunk_size²)``, not ``O(N²)``.

2. **Inter-chunk attention** — each chunk also attends to a compressed
   representation of the most-relevant past chunks.  Each past chunk is
   summarised by a single ``(n_heads, head_dim)`` vector produced by
   :meth:`encode_chunk` (the attention-weighted mean of the chunk's value
   vectors, using the mean key as the query for the summary weighting).
   The ``inter_chunk_top_k`` most relevant chunks — scored by the dot product
   of the current chunk's mean query with each chunk summary — are selected
   and attended to as a short sequence of representative tokens.

The combined DCA output is the sum of the intra-chunk output and the
inter-chunk output, giving each token access to local context (high
resolution) plus a compressed view of salient distant past (low resolution).

Callers are responsible for accumulating chunk summaries returned by
:meth:`encode_chunk` as the sequence is processed and passing them as the
``past_chunks`` argument to :meth:`forward`.

Example usage::

    import numpy as np
    from squish.dual_chunk_attn import DCAConfig, DualChunkAttention

    cfg = DCAConfig(n_heads=4, head_dim=32, chunk_size=64, inter_chunk_top_k=2)
    dca = DualChunkAttention(cfg)

    rng = np.random.default_rng(1)
    q   = rng.standard_normal((4, 64, 32)).astype(np.float32)
    k   = rng.standard_normal((4, 64, 32)).astype(np.float32)
    v   = rng.standard_normal((4, 64, 32)).astype(np.float32)

    # No past chunks for the first chunk.
    out0   = dca.forward(q, k, v, past_chunks=None)
    print(out0.shape)  # (4, 64, 32)

    # Encode this chunk and use it for the next.
    summary = dca.encode_chunk(k, v)
    print(summary.shape)  # (4, 32)

    out1 = dca.forward(q, k, v, past_chunks=[summary])
    print(out1.shape)  # (4, 64, 32)
"""

from __future__ import annotations

__all__ = ["DCAConfig", "DualChunkAttention"]

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class DCAConfig:
    """Configuration for Dual-Chunk Attention.

    Attributes:
        n_heads:            Number of attention heads.
        head_dim:           Dimension of each attention head.
        chunk_size:         Number of tokens per chunk.  Memory per chunk is
                            O(chunk_size²) rather than O(seq_len²).  Must be
                            >= 1.
        inter_chunk_top_k:  Number of past chunk summaries to attend to in the
                            inter-chunk attention step.  Set to 0 to disable
                            inter-chunk attention entirely.  Must be >= 0.
    """

    n_heads:            int
    head_dim:           int
    chunk_size:         int = 512
    inter_chunk_top_k:  int = 4

    def __post_init__(self) -> None:
        if self.n_heads < 1:
            raise ValueError(f"n_heads must be >= 1, got {self.n_heads}")
        if self.head_dim < 1:
            raise ValueError(f"head_dim must be >= 1, got {self.head_dim}")
        if self.chunk_size < 1:
            raise ValueError(f"chunk_size must be >= 1, got {self.chunk_size}")
        if self.inter_chunk_top_k < 0:
            raise ValueError(
                f"inter_chunk_top_k must be >= 0, got {self.inter_chunk_top_k}"
            )


# ---------------------------------------------------------------------------
# Attention
# ---------------------------------------------------------------------------


class DualChunkAttention:
    """Dual-chunk attention (DCA) for very long context windows.

    Each call to :meth:`forward` processes a single chunk of tokens using:

    * **Intra-chunk path**: full causal scaled dot-product attention within
      the chunk itself.
    * **Inter-chunk path**: the chunk's mean query is used to score all
      provided ``past_chunks`` summaries; the ``inter_chunk_top_k`` highest-
      scoring summaries are attended to as representative tokens, and the
      output is added to the intra-chunk result.

    Chunk summaries must be pre-computed by the caller using
    :meth:`encode_chunk` and accumulated between chunk calls.

    Args:
        config: A :class:`DCAConfig` instance.
    """

    def __init__(self, config: DCAConfig) -> None:
        self._cfg = config

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def encode_chunk(self, k: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Compress a chunk into a single per-head representative vector.

        The summary is computed as the attention-weighted mean of the chunk's
        value vectors, where the attention weights are derived from the
        softmax dot-product between the chunk's mean key (used as a query)
        and each position's key.  This produces a content-aware summary that
        emphasises the most salient value vectors in the chunk.

        Args:
            k: Key tensor of shape ``(n_heads, chunk_size, head_dim)``
               (float32).
            v: Value tensor of shape ``(n_heads, chunk_size, head_dim)``
               (float32).  Must have the same shape as *k*.

        Returns:
            Summary tensor of shape ``(n_heads, head_dim)`` (float32) that
            serves as both the key and value representative for this chunk in
            subsequent :meth:`forward` inter-chunk attention steps.

        Raises:
            ValueError: If *k* or *v* have unexpected shapes.
        """
        cfg = self._cfg
        k   = np.asarray(k, dtype=np.float32)
        v   = np.asarray(v, dtype=np.float32)
        self._validate_kv(k, v, "encode_chunk")

        # Mean key across the chunk as the query for internal scoring.
        # q_mean: (n_heads, head_dim)
        q_mean = k.mean(axis=1)

        # Scores: q_mean[h] · k[h, s] — shape (n_heads, chunk_size).
        scale  = 1.0 / math.sqrt(cfg.head_dim)
        scores = np.einsum("hd,hsd->hs", q_mean, k) * scale  # (n_heads, chunk_size)

        # Softmax over the chunk dimension.
        scores -= scores.max(axis=-1, keepdims=True)
        exp_s   = np.exp(scores)
        attn    = exp_s / exp_s.sum(axis=-1, keepdims=True)  # (n_heads, chunk_size)

        # Weighted mean of value vectors: summary shape (n_heads, head_dim).
        summary = np.einsum("hs,hsd->hd", attn, v).astype(np.float32)
        return summary

    def forward(
        self,
        q: np.ndarray,
        k: np.ndarray,
        v: np.ndarray,
        past_chunks: Optional[list[np.ndarray]] = None,
    ) -> np.ndarray:
        """Compute dual-chunk attention output for the current chunk.

        Applies intra-chunk causal attention first, then adds inter-chunk
        attention over the top-k selected past chunk summaries (when
        ``past_chunks`` is provided and ``inter_chunk_top_k > 0``).

        Args:
            q:           Query tensor of shape ``(n_heads, seq_q, head_dim)``
                         (float32).
            k:           Key tensor of shape ``(n_heads, seq_q, head_dim)``
                         (float32).  For intra-chunk attention, ``seq_q``
                         should equal ``chunk_size`` or be shorter for the
                         final partial chunk.
            v:           Value tensor of shape ``(n_heads, seq_q, head_dim)``
                         (float32).  Must have the same shape as *k*.
            past_chunks: Optional list of chunk summaries, each of shape
                         ``(n_heads, head_dim)`` as returned by
                         :meth:`encode_chunk`.  The most recent summary
                         should be last.  Pass ``None`` or an empty list for
                         the first chunk.

        Returns:
            Output tensor of shape ``(n_heads, seq_q, head_dim)`` (float32).

        Raises:
            ValueError: If *q*, *k*, or *v* have unexpected shapes, or if a
                        past chunk summary has an unexpected shape.
        """
        cfg = self._cfg
        q   = np.asarray(q, dtype=np.float32)
        k   = np.asarray(k, dtype=np.float32)
        v   = np.asarray(v, dtype=np.float32)

        self._validate_qkv(q, k, v)

        n_heads, seq_q, head_dim = q.shape
        scale = 1.0 / math.sqrt(head_dim)

        # ── Intra-chunk causal attention ──────────────────────────────
        # scores: (n_heads, seq_q, seq_q)
        scores_intra = np.einsum("hqd,hkd->hqk", q, k) * scale

        # Causal mask: position i can only attend to j <= i.
        causal_mask = np.tril(np.ones((seq_q, seq_q), dtype=np.bool_))
        scores_intra = np.where(
            causal_mask[np.newaxis, :, :],
            scores_intra,
            np.float32(-1e9),
        )

        scores_intra -= scores_intra.max(axis=-1, keepdims=True)
        exp_intra     = np.exp(scores_intra)
        # Rows that are fully masked (first token) are handled by the causal
        # mask allowing at least position 0, so no all-zero rows occur.
        attn_intra    = exp_intra / exp_intra.sum(axis=-1, keepdims=True)
        out_intra     = np.einsum("hqk,hkd->hqd", attn_intra, v)  # (n_heads, seq_q, head_dim)

        # ── Inter-chunk attention ─────────────────────────────────────
        if (
            past_chunks is None
            or len(past_chunks) == 0
            or cfg.inter_chunk_top_k == 0
        ):
            return out_intra.astype(np.float32)

        # Validate and stack summaries.
        for ci, chunk in enumerate(past_chunks):
            chunk_arr = np.asarray(chunk, dtype=np.float32)
            if chunk_arr.shape != (n_heads, head_dim):
                raise ValueError(
                    f"past_chunks[{ci}] must have shape ({n_heads}, {head_dim}), "
                    f"got {chunk_arr.shape}."
                )

        # chunk_summaries: (n_chunks, n_heads, head_dim)
        chunk_summaries = np.stack(
            [np.asarray(c, dtype=np.float32) for c in past_chunks], axis=0
        )
        n_chunks = chunk_summaries.shape[0]

        # Select top-k chunks using the mean query of the current chunk.
        # q_mean: (n_heads, head_dim)
        q_mean = q.mean(axis=1)

        # scores_chunks[h, c] = q_mean[h] · summary[c, h]
        # shape: (n_heads, n_chunks) → sum over heads for global selection.
        scores_chunks = np.einsum(
            "hd,chd->hc", q_mean, chunk_summaries
        )  # (n_heads, n_chunks)
        global_scores = scores_chunks.sum(axis=0)  # (n_chunks,)

        effective_k = min(cfg.inter_chunk_top_k, n_chunks)
        if effective_k < n_chunks:
            top_chunk_idx = np.argpartition(-global_scores, effective_k)[
                :effective_k
            ]
            # Sort by descending score for reproducibility.
            top_chunk_idx = top_chunk_idx[
                np.argsort(-global_scores[top_chunk_idx])
            ]
        else:
            top_chunk_idx = np.argsort(-global_scores)[:effective_k]

        # selected: (effective_k, n_heads, head_dim) → (n_heads, effective_k, head_dim)
        selected      = chunk_summaries[top_chunk_idx]  # (effective_k, n_heads, head_dim)
        selected_kv   = selected.transpose(1, 0, 2)     # (n_heads, effective_k, head_dim)

        # Inter-chunk attention: q attends to selected summaries.
        # Each summary acts as both key and value (single-token compressed chunk).
        scores_inter  = (
            np.einsum("hqd,hkd->hqk", q, selected_kv) * scale
        )  # (n_heads, seq_q, effective_k)
        scores_inter -= scores_inter.max(axis=-1, keepdims=True)
        exp_inter      = np.exp(scores_inter)
        attn_inter     = exp_inter / exp_inter.sum(axis=-1, keepdims=True)
        out_inter      = np.einsum(
            "hqk,hkd->hqd", attn_inter, selected_kv
        )  # (n_heads, seq_q, head_dim)

        return (out_intra + out_inter).astype(np.float32)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _validate_kv(
        self, k: np.ndarray, v: np.ndarray, caller: str
    ) -> None:
        cfg = self._cfg
        for name, arr in (("k", k), ("v", v)):
            if arr.ndim != 3:
                raise ValueError(
                    f"{caller}: {name} must be 3-D "
                    f"(n_heads, chunk_size, head_dim), got {arr.shape}."
                )
        if k.shape[0] != cfg.n_heads or k.shape[2] != cfg.head_dim:
            raise ValueError(
                f"{caller}: k must have shape "
                f"(n_heads={cfg.n_heads}, chunk_size, head_dim={cfg.head_dim}), "
                f"got {k.shape}."
            )
        if v.shape != k.shape:
            raise ValueError(
                f"{caller}: v must have the same shape as k ({k.shape}), "
                f"got {v.shape}."
            )

    def _validate_qkv(
        self, q: np.ndarray, k: np.ndarray, v: np.ndarray
    ) -> None:
        cfg = self._cfg
        for name, arr in (("q", q), ("k", k), ("v", v)):
            if arr.ndim != 3:
                raise ValueError(
                    f"forward: {name} must be 3-D "
                    f"(n_heads, seq_q, head_dim), got {arr.shape}."
                )
        if q.shape[0] != cfg.n_heads or q.shape[2] != cfg.head_dim:
            raise ValueError(
                f"forward: q must have shape "
                f"(n_heads={cfg.n_heads}, seq_q, head_dim={cfg.head_dim}), "
                f"got {q.shape}."
            )
        if k.shape != q.shape:
            raise ValueError(
                f"forward: k must have the same shape as q ({q.shape}), "
                f"got {k.shape}."
            )
        if v.shape != k.shape:
            raise ValueError(
                f"forward: v must have the same shape as k ({k.shape}), "
                f"got {v.shape}."
            )
