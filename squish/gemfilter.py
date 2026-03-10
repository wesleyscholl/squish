"""
GemFilter — Early-Layer Input Token Compression.

ICLR 2025. arxiv.org/abs/2409.17422
github.com/SalesforceAIResearch/GemFilter
Salesforce AI Research.

Key insight: LLMs can identify relevant tokens ("gems") in early layers,
before generating answers.  GemFilter runs only the first `r` layers of the
model, reads the attention weights from the last query token, and discards
all but the top-k attended tokens.  The full forward pass then runs only on
those k tokens.

No auxiliary model needed — the target model filters itself.

Results:
  - 2.4× speedup, 30% GPU memory reduction.
  - 1000× compression on 108K-token needle-in-haystack tasks.
  - Input reduced to 8% at 1024-token context, 32% at 4096 tokens.

This module provides:
  - GemFilterConfig — configuration (filter layers, top-k budget)
  - AttentionScoreBuffer — collects attention maps from early layers
  - GemSelector — computes gem token scores and returns selected indices
  - GemFilterStats — compression statistics
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class GemFilterConfig:
    """Configuration for GemFilter.

    Args:
        filter_layer:   Which model layer's attention map to read.  For an
                        8B model the paper uses layers 13–19; 15 is a
                        safe default.
        top_k_tokens:   Maximum number of tokens to retain after filtering.
                        Set to None to use top_k_fraction instead.
        top_k_fraction: Fraction of input tokens to retain (0.0–1.0).
                        Used if top_k_tokens is None.
        always_keep_first: Always retain the system prompt / first
                        keep_prefix_tokens tokens regardless of attention
                        score.  Prevents loss of instruction context.
        keep_prefix_tokens: How many leading tokens to always keep.
        always_keep_last:  Always retain the most recent keep_suffix_tokens
                        tokens (the actual query).
        keep_suffix_tokens: How many trailing tokens to always keep.
        aggregation:    How to aggregate multi-head attention to a single
                        per-token score.  'mean' | 'max' | 'last_head'.
    """

    filter_layer: int = 15
    top_k_tokens: int | None = None
    top_k_fraction: float = 0.10
    always_keep_first: bool = True
    keep_prefix_tokens: int = 4
    always_keep_last: bool = True
    keep_suffix_tokens: int = 4
    aggregation: str = "mean"

    def __post_init__(self) -> None:
        if self.filter_layer < 0:
            raise ValueError("filter_layer must be >= 0")
        if self.top_k_tokens is not None and self.top_k_tokens < 1:
            raise ValueError("top_k_tokens must be >= 1 when set")
        if not (0.0 < self.top_k_fraction <= 1.0):
            raise ValueError("top_k_fraction must be in (0, 1]")
        valid_agg = {"mean", "max", "last_head"}
        if self.aggregation not in valid_agg:
            raise ValueError(f"aggregation must be one of {valid_agg}")
        if self.keep_prefix_tokens < 0:
            raise ValueError("keep_prefix_tokens must be >= 0")
        if self.keep_suffix_tokens < 0:
            raise ValueError("keep_suffix_tokens must be >= 0")

    def budget(self, seq_len: int) -> int:
        """Number of tokens to retain for a sequence of length *seq_len*."""
        if self.top_k_tokens is not None:
            return min(self.top_k_tokens, seq_len)
        return max(1, min(seq_len, int(math.ceil(seq_len * self.top_k_fraction))))


# ---------------------------------------------------------------------------
# Attention score buffer
# ---------------------------------------------------------------------------

class AttentionScoreBuffer:
    """Collects attention maps from early forward passes.

    During the partial (filter-layer) forward pass, hook this into the
    model's attention layers to record attention weights at :attr:`target_layer`.

    Then call :meth:`get_scores` to obtain a per-token importance score.
    """

    def __init__(self, config: GemFilterConfig) -> None:
        self._config = config
        self._attn_maps: list[np.ndarray] = []
        """Stored attention maps; each is (n_heads, n_queries, seq_len) or
        (n_queries, seq_len) for single-head."""

    def record(self, layer_idx: int, attn_map: np.ndarray) -> None:
        """Record attention map if *layer_idx* is the filter layer.

        Args:
            layer_idx: Which transformer layer this map came from.
            attn_map:  Float array.  Accepted shapes:
                       - (n_heads, n_queries, seq_len) — batch of heads
                       - (n_queries, seq_len) — single head
                       - (seq_len,) — already aggregated
        """
        if layer_idx != self._config.filter_layer:
            return
        self._attn_maps.append(np.asarray(attn_map, dtype=np.float32))

    def get_scores(self) -> np.ndarray | None:
        """Aggregate all recorded attention maps to a 1-D per-token score.

        Returns:
            1-D float array of length *seq_len*, or None if no maps recorded.
        """
        if not self._attn_maps:
            return None

        agg = self._config.aggregation
        scores_list: list[np.ndarray] = []

        for attn in self._attn_maps:
            # Normalise to 3-D: (n_heads, n_queries, seq_len)
            if attn.ndim == 1:
                flat = attn
            elif attn.ndim == 2:
                # (n_queries, seq_len) → use last query row (the actual query)
                flat = attn[-1]
            else:
                # (n_heads, n_queries, seq_len) — last query row of each head
                last_q = attn[:, -1, :]  # (n_heads, seq_len)
                if agg == "mean":
                    flat = last_q.mean(axis=0)
                elif agg == "max":
                    flat = last_q.max(axis=0)
                else:  # last_head
                    flat = last_q[-1]

            scores_list.append(flat)

        # Average across all recorded layers at the filter layer
        stacked = np.vstack([s.reshape(1, -1) for s in scores_list])
        return stacked.mean(axis=0)

    def reset(self) -> None:
        self._attn_maps.clear()


# ---------------------------------------------------------------------------
# Gem Selector
# ---------------------------------------------------------------------------

class GemSelector:
    """Selects "gem" tokens given per-token importance scores.

    Usage::

        selector = GemSelector(config)
        kept_indices = selector.select(scores, seq_len)
        filtered_input_ids = input_ids[kept_indices]
    """

    def __init__(self, config: GemFilterConfig) -> None:
        self._config = config

    def select(
        self, scores: np.ndarray, seq_len: int | None = None
    ) -> np.ndarray:
        """Return sorted indices of tokens to keep.

        Args:
            scores:  1-D float array of per-token importance scores.
            seq_len: Actual sequence length (defaults to len(scores)).

        Returns:
            1-D int array of token indices to retain, in ascending order.
        """
        cfg = self._config
        n = len(scores) if seq_len is None else seq_len
        scores = np.asarray(scores[:n], dtype=np.float32)

        budget = cfg.budget(n)
        forced: set = set()

        if cfg.always_keep_first:
            for i in range(min(cfg.keep_prefix_tokens, n)):
                forced.add(i)

        if cfg.always_keep_last:
            for i in range(max(0, n - cfg.keep_suffix_tokens), n):
                forced.add(i)

        remaining_budget = max(0, budget - len(forced))

        # Zero out forced indices before selecting top-k to avoid double counting
        free_scores = scores.copy()
        for idx in forced:
            free_scores[idx] = -np.inf

        if remaining_budget > 0 and n > len(forced):
            top_k_indices = np.argsort(free_scores)[-remaining_budget:]
        else:
            top_k_indices = np.array([], dtype=np.int64)

        all_indices = sorted(set(top_k_indices.tolist()) | forced)
        return np.array(all_indices, dtype=np.int64)

    def compression_ratio(self, n_original: int, n_kept: int) -> float:
        """1 - n_kept/n_original."""
        return 1.0 - n_kept / max(1, n_original)


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

@dataclass
class GemFilterStats:
    """Cumulative compression statistics for GemFilter."""

    total_input_tokens: int = 0
    total_kept_tokens: int = 0
    n_calls: int = 0

    def record(self, n_input: int, n_kept: int) -> None:
        object.__setattr__(self, "total_input_tokens",
                           self.total_input_tokens + n_input)
        object.__setattr__(self, "total_kept_tokens",
                           self.total_kept_tokens + n_kept)
        object.__setattr__(self, "n_calls", self.n_calls + 1)

    @property
    def mean_compression_ratio(self) -> float:
        """Average fraction of tokens discarded."""
        if self.total_input_tokens == 0:
            return 0.0
        return 1.0 - self.total_kept_tokens / self.total_input_tokens

    @property
    def mean_kept_fraction(self) -> float:
        if self.total_input_tokens == 0:
            return 1.0
        return self.total_kept_tokens / self.total_input_tokens

    @property
    def mean_speedup_estimate(self) -> float:
        """Estimated prefill + decode speedup from reduced token count.

        Based on paper's 2.4× figure at ~10% retention.  Linear interpolation.
        """
        if self.mean_kept_fraction >= 1.0:
            return 1.0
        # Paper: ~10% retention → 2.4× speedup; interpolate linearly
        return 1.0 / max(0.01, self.mean_kept_fraction) * 0.24 + 1.0
