"""squish/context_summarizer.py

ContextSummarizer — Inference-time context compression via importance scoring.

Long-context models can easily exceed hardware KV cache capacity or inference
time budgets.  When a context exceeds a user-defined token *budget*,
ContextSummarizer selects the most important tokens to retain using one of
three strategies:

**importance** (default)
    Rank tokens in the non-recent prefix by the L2 norm of their embedding
    vector.  High-norm embeddings are empirically correlated with tokens that
    carry dense semantic content (numerals, proper nouns, rare words).  The
    ``min_keep_recent`` most-recent tokens are always appended to preserve
    causal coherence.

**stride**
    Keep ``n_from_prefix`` evenly spaced tokens from the non-recent prefix
    (using ``np.linspace`` for uniform spacing), then append the most-recent
    ``min_keep_recent`` tokens.

**recency**
    Discard everything except the most recent ``budget`` tokens.  This
    degenerates to a simple sliding-window truncation.

For all methods the ``min_keep_recent`` guard is honoured (clamped to
``budget`` if it exceeds the budget), and the surviving tokens are always
returned in their original (causal) order.

Example usage::

    import numpy as np
    from squish.context_summarizer import SummaryConfig, ContextSummarizer

    cfg        = SummaryConfig(method="importance", budget=128, min_keep_recent=32)
    summarizer = ContextSummarizer(cfg)

    rng        = np.random.default_rng(0)
    token_ids  = rng.integers(0, 32000, size=512, dtype=np.int32)
    embeddings = rng.standard_normal((512, 256)).astype(np.float32)

    compressed_ids, stats = summarizer.summarize(token_ids, embeddings)
    print(stats)
"""

from __future__ import annotations

__all__ = ["SummaryConfig", "SummaryStats", "ContextSummarizer"]

import dataclasses
from typing import Optional

import numpy as np


_VALID_METHODS: frozenset[str] = frozenset({"importance", "stride", "recency"})


# ---------------------------------------------------------------------------
# Configuration and stats
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class SummaryConfig:
    """Configuration for :class:`ContextSummarizer`.

    Attributes:
        method:           Token selection strategy.  One of ``"importance"``,
                          ``"stride"``, or ``"recency"``.
        budget:           Maximum number of tokens in the compressed output.
                          Must be >= 1.
        min_keep_recent:  Minimum number of most-recent tokens to always
                          retain (applied for ``"importance"`` and
                          ``"stride"``; implicitly satisfied by
                          ``"recency"``).  Clamped to *budget* if larger.
                          Must be >= 0.
    """

    method:           str = "importance"
    budget:           int = 512
    min_keep_recent:  int = 64

    def __post_init__(self) -> None:
        if self.method not in _VALID_METHODS:
            raise ValueError(
                f"method must be one of {sorted(_VALID_METHODS)}, "
                f"got '{self.method}'"
            )
        if self.budget < 1:
            raise ValueError(f"budget must be >= 1, got {self.budget}")
        if self.min_keep_recent < 0:
            raise ValueError(
                f"min_keep_recent must be >= 0, got {self.min_keep_recent}"
            )


@dataclasses.dataclass
class SummaryStats:
    """Statistics returned by :meth:`ContextSummarizer.summarize`.

    Attributes:
        n_tokens_in:       Sequence length before compression.
        n_tokens_out:      Sequence length after compression.
        compression_ratio: ``n_tokens_out / n_tokens_in``.  A value of
                           ``1.0`` means no compression was applied.
        method_used:       The strategy string that was applied.
    """

    n_tokens_in:       int
    n_tokens_out:      int
    compression_ratio: float
    method_used:       str


# ---------------------------------------------------------------------------
# Summarizer
# ---------------------------------------------------------------------------


class ContextSummarizer:
    """Inference-time context compressor.

    When the input sequence length exceeds ``config.budget``, selects at
    most ``budget`` tokens to forward to the model.  The
    ``min_keep_recent`` most-recent tokens are always included in the
    retained set for the ``"importance"`` and ``"stride"`` methods.

    Args:
        config: :class:`SummaryConfig` controlling compression behaviour.
    """

    def __init__(self, config: SummaryConfig) -> None:
        self._cfg = config

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def needs_compression(self, seq_len: int) -> bool:
        """Return ``True`` if *seq_len* exceeds the configured budget.

        Args:
            seq_len: Current sequence length.

        Returns:
            ``True`` when ``seq_len > config.budget``.
        """
        return seq_len > self._cfg.budget

    def summarize(
        self,
        token_ids: np.ndarray,
        embeddings: Optional[np.ndarray] = None,
    ) -> tuple[np.ndarray, SummaryStats]:
        """Compress *token_ids* to at most ``config.budget`` tokens.

        If ``seq_len <= budget`` the input is returned unchanged (no copy is
        made).

        Args:
            token_ids:  int32 array of shape ``(seq_len,)``.
            embeddings: Optional float32 array of shape ``(seq_len, dim)``.
                        Required when ``method == "importance"``; ignored
                        for ``"stride"`` and ``"recency"``.

        Returns:
            Tuple ``(compressed_ids, stats)`` where ``compressed_ids`` is
            an int32 array with at most ``budget`` entries in their original
            causal order.

        Raises:
            ValueError: If *token_ids* is not 1-D, if *embeddings* is
                        required but not provided, or if shapes are
                        inconsistent.
        """
        token_ids = np.asarray(token_ids, dtype=np.int32)
        if token_ids.ndim != 1:
            raise ValueError(
                f"token_ids must be 1-D (seq_len,), got shape {token_ids.shape}"
            )
        seq_len = token_ids.shape[0]

        if not self.needs_compression(seq_len):
            stats = SummaryStats(
                n_tokens_in=seq_len,
                n_tokens_out=seq_len,
                compression_ratio=1.0,
                method_used=self._cfg.method,
            )
            return token_ids, stats

        if embeddings is not None:
            embeddings = np.asarray(embeddings, dtype=np.float32)
            if embeddings.ndim != 2 or embeddings.shape[0] != seq_len:
                raise ValueError(
                    f"embeddings must have shape (seq_len={seq_len}, dim), "
                    f"got {embeddings.shape}"
                )

        method = self._cfg.method
        if method == "importance":
            keep_indices = self._select_importance(seq_len, embeddings)
        elif method == "stride":
            keep_indices = self._select_stride(seq_len)
        else:  # method == "recency"
            keep_indices = self._select_recency(seq_len)

        # Always return tokens in original causal order.
        keep_indices_sorted = np.sort(keep_indices)
        compressed          = token_ids[keep_indices_sorted]

        n_out = compressed.shape[0]
        stats = SummaryStats(
            n_tokens_in=seq_len,
            n_tokens_out=n_out,
            compression_ratio=float(n_out) / float(seq_len),
            method_used=method,
        )
        return compressed, stats

    # ------------------------------------------------------------------
    # Internal selection strategies
    # ------------------------------------------------------------------

    def _select_recency(self, seq_len: int) -> np.ndarray:
        """Keep the most recent ``budget`` tokens."""
        budget = self._cfg.budget
        start  = seq_len - budget
        return np.arange(start, seq_len, dtype=np.int64)

    def _select_stride(self, seq_len: int) -> np.ndarray:
        """Keep evenly-spaced tokens from the non-recent prefix, plus recent
        tokens.

        The prefix is divided into ``n_from_prefix`` evenly spaced samples
        using ``np.linspace``.  The most-recent ``min_keep_recent`` tokens
        are appended unconditionally.
        """
        budget      = self._cfg.budget
        min_recent  = min(self._cfg.min_keep_recent, budget)
        n_from_prefix = budget - min_recent

        recent_start   = seq_len - min_recent
        recent_indices = np.arange(recent_start, seq_len, dtype=np.int64)

        prefix_len = recent_start  # == seq_len - min_recent
        if n_from_prefix <= 0 or prefix_len <= 0:
            return recent_indices

        # Evenly spaced float positions, rounded to nearest integer index.
        raw_positions  = np.linspace(0, prefix_len - 1, n_from_prefix)
        stride_indices = np.unique(np.round(raw_positions).astype(np.int64))

        return np.concatenate([stride_indices, recent_indices])

    def _select_importance(
        self,
        seq_len: int,
        embeddings: Optional[np.ndarray],
    ) -> np.ndarray:
        """Keep highest embedding-norm tokens from the non-recent prefix, plus
        the most-recent tokens.

        Args:
            seq_len:    Full sequence length.
            embeddings: float32 array ``(seq_len, dim)``.  Required.

        Raises:
            ValueError: If *embeddings* is ``None``.
        """
        if embeddings is None:
            raise ValueError(
                "embeddings must be provided when method='importance'"
            )

        budget        = self._cfg.budget
        min_recent    = min(self._cfg.min_keep_recent, budget)
        n_from_prefix = budget - min_recent

        recent_start   = seq_len - min_recent
        recent_indices = np.arange(recent_start, seq_len, dtype=np.int64)

        prefix_len = recent_start
        if n_from_prefix <= 0 or prefix_len <= 0:
            return recent_indices

        prefix_embeddings = embeddings[:prefix_len]
        norms = np.linalg.norm(prefix_embeddings, axis=1)  # (prefix_len,)

        if n_from_prefix >= prefix_len:
            important_indices = np.arange(prefix_len, dtype=np.int64)
        else:
            # Partial sort: O(n) to find top-k, then no full sort needed.
            part              = np.argpartition(-norms, n_from_prefix - 1)
            important_indices = part[:n_from_prefix].astype(np.int64)

        return np.concatenate([important_indices, recent_indices])
