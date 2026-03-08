"""
squish/spe_cache.py

SpeCache — Speculative KV-Cache Prefetching.

Inspired by:
  "SpeCache: Speculative Key-Value Caching for Efficient LLM Multi-Turn
   Dialogue" — arXiv:2501.17188 (Jan 2025)

Problem
-------
Multi-turn conversations reuse long shared KV caches.  Decoding stalls
while prior-turn KV is reloaded from disk / CPU memory to device.

SpeCache Idea
-------------
During the *decode* phase of turn N, speculatively prefetch the KV blocks
that are *predicted* to be needed for turn N+1 — in the background.

Prefetch prediction uses a lightweight heuristic:
  1. Count attention sink tokens (first K tokens): always prefetch.
  2. Count high-attention blocks from the current decode step as "hot".
  3. Sort candidate blocks by combined score = α·recency + β·attention_mass.
  4. Prefetch top-``budget`` blocks while idle GPU cycles exist.

This module provides:
  * ``SpeCacheConfig`` — prefetch budget and scoring weights.
  * ``BlockScoreTracker`` — accumulates attention scores per block.
  * ``SpeCachePrefetcher`` — decides which blocks to prefetch next turn.
  * ``InMemoryBlockStore`` — simple dict-backed block store (testable).

Integration::

    from squish.spe_cache import SpeCacheConfig, SpeCachePrefetcher, InMemoryBlockStore

    store    = InMemoryBlockStore(block_size=64)
    prefetch = SpeCachePrefetcher(config=SpeCacheConfig(), store=store)

    # Each decode step
    prefetch.record_attention(attn_scores_np)

    # End of turn N → schedule prefetch for turn N+1
    hot_blocks = prefetch.predict_next_turn_blocks(total_blocks=len(kv_store))
    for block_id in hot_blocks:
        prefetch.prefetch(block_id)
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Set, Tuple

import numpy as np

__all__ = [
    "SpeCacheConfig",
    "BlockScoreTracker",
    "SpeCachePrefetcher",
    "InMemoryBlockStore",
]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class SpeCacheConfig:
    """
    Configuration for SpeCache speculative prefetching.

    Parameters
    ----------
    block_size : int
        Number of tokens per KV block.
    prefetch_budget : int
        Maximum blocks to prefetch per turn boundary.
    sink_blocks : int
        Always-prefetch "attention sink" blocks (first N blocks).
    alpha_recency : float
        Weight for recency in block scoring (higher = prefer recent).
    beta_attention : float
        Weight for accumulated attention mass in block scoring.
    top_k_attention : int
        Top-K attention positions tracked per decode step.
    """
    block_size:       int   = 64
    prefetch_budget:  int   = 8
    sink_blocks:      int   = 1
    alpha_recency:    float = 0.4
    beta_attention:   float = 0.6
    top_k_attention:  int   = 32

    def __post_init__(self) -> None:
        if self.block_size < 1:
            raise ValueError("block_size must be ≥ 1")
        if self.prefetch_budget < 0:
            raise ValueError("prefetch_budget must be ≥ 0")
        if self.sink_blocks < 0:
            raise ValueError("sink_blocks must be ≥ 0")
        if not 0.0 <= self.alpha_recency <= 1.0:
            raise ValueError("alpha_recency must be in [0, 1]")
        if not 0.0 <= self.beta_attention <= 1.0:
            raise ValueError("beta_attention must be in [0, 1]")
        if self.top_k_attention < 1:
            raise ValueError("top_k_attention must be ≥ 1")


# ---------------------------------------------------------------------------
# Block Score Tracker
# ---------------------------------------------------------------------------

class BlockScoreTracker:
    """
    Accumulates attention scores per KV block across decode steps.

    Parameters
    ----------
    block_size : int
        Token positions per block.
    top_k : int
        Only the top-K attention positions are counted per step.
    """

    def __init__(self, block_size: int = 64, top_k: int = 32) -> None:
        self._block_size = block_size
        self._top_k      = top_k
        # block_id -> float (cumulative attention mass)
        self._scores: Dict[int, float] = {}
        self._step_count = 0

    def record(self, attn_scores: np.ndarray) -> None:
        """
        Update block scores with one decode step's attention distribution.

        Parameters
        ----------
        attn_scores : (seq_len,) float — attention weights over key positions
                      (averaged over heads is fine).
        """
        scores = np.asarray(attn_scores, dtype=np.float32).flatten()
        seq_len = len(scores)
        if seq_len == 0:
            return

        # Normalise
        total = scores.sum()
        if total > 0:
            scores = scores / total

        # Top-K positions
        k   = min(self._top_k, seq_len)
        top = np.argpartition(scores, -k)[-k:]

        for pos in top:
            block_id = int(pos) // self._block_size
            self._scores[block_id] = (
                self._scores.get(block_id, 0.0) + float(scores[pos])
            )

        self._step_count += 1

    def get_scores(self) -> Dict[int, float]:
        """Return accumulated attention scores per block (copy)."""
        return dict(self._scores)

    def reset(self) -> None:
        """Clear scores for the next conversation turn."""
        self._scores.clear()
        self._step_count = 0

    @property
    def step_count(self) -> int:
        return self._step_count


# ---------------------------------------------------------------------------
# In-Memory Block Store (for testing and prototyping)
# ---------------------------------------------------------------------------

class InMemoryBlockStore:
    """
    Simple dict-backed KV block store for testing.

    A block is identified by an integer ``block_id``.
    Data can be any Python object (typically numpy arrays).
    """

    def __init__(self, block_size: int = 64) -> None:
        self.block_size = block_size
        self._blocks: Dict[int, object] = {}
        self._prefetched: Set[int] = set()

    def store(self, block_id: int, data: object) -> None:
        """Store a block."""
        self._blocks[block_id] = data

    def load(self, block_id: int) -> Optional[object]:
        """Load a block by ID (returns None if not present)."""
        return self._blocks.get(block_id)

    def prefetch(self, block_id: int) -> None:
        """Mark block as prefetched (no-op in memory store)."""
        self._prefetched.add(block_id)

    def is_prefetched(self, block_id: int) -> bool:
        return block_id in self._prefetched

    def __len__(self) -> int:
        return len(self._blocks)


# ---------------------------------------------------------------------------
# SpeCache Prefetcher
# ---------------------------------------------------------------------------

class SpeCachePrefetcher:
    """
    Decides which KV blocks to prefetch for the next conversation turn.

    Parameters
    ----------
    config : SpeCacheConfig
    store  : object with a ``.prefetch(block_id)`` method (see InMemoryBlockStore).
    """

    def __init__(
        self,
        config: SpeCacheConfig,
        store:  object,
    ) -> None:
        self._cfg     = config
        self._store   = store
        self._tracker = BlockScoreTracker(
            block_size = config.block_size,
            top_k      = config.top_k_attention,
        )
        self._lock    = threading.Lock()

    def record_attention(self, attn_scores: np.ndarray) -> None:
        """
        Accumulate one decode step's attention pattern.

        Call once per decode step with shape ``(seq_len,)`` attention weights
        (already averaged over heads/layers is fine).
        """
        with self._lock:
            self._tracker.record(attn_scores)

    def predict_next_turn_blocks(self, total_blocks: int) -> List[int]:
        """
        Predict which blocks to prefetch for the next turn.

        Scoring formula for each block b:
            score(b) = alpha * recency(b) + beta * attn_mass(b)

        where ``recency(b) = b / total_blocks`` (later blocks are more recent).

        Parameters
        ----------
        total_blocks : int — total number of KV blocks in the current cache.

        Returns
        -------
        List of block IDs sorted by descending priority.
        """
        cfg = self._cfg
        if total_blocks <= 0:
            return []

        with self._lock:
            attn_scores = self._tracker.get_scores()

        # Candidate blocks: all blocks in [0, total_blocks)
        candidates: List[Tuple[float, int]] = []
        for block_id in range(total_blocks):
            recency   = block_id / max(total_blocks - 1, 1)
            attn_mass = attn_scores.get(block_id, 0.0)
            score     = cfg.alpha_recency * recency + cfg.beta_attention * attn_mass
            candidates.append((score, block_id))

        # Always include sink blocks
        sink_set = set(range(min(cfg.sink_blocks, total_blocks)))

        # Sort by score descending
        candidates.sort(reverse=True)

        # Select top budget, ensuring sinks are always included
        selected: List[int] = []
        seen: Set[int] = set()

        for bid in sink_set:
            if bid < total_blocks:  # pragma: no cover
                selected.append(bid)
                seen.add(bid)

        for score, block_id in candidates:
            if len(selected) >= cfg.prefetch_budget:
                break
            if block_id not in seen:
                selected.append(block_id)
                seen.add(block_id)

        return selected

    def prefetch(self, block_id: int) -> None:
        """Trigger prefetch of a single block via the store."""
        self._store.prefetch(block_id)

    def prefetch_batch(self, block_ids: List[int]) -> None:
        """Prefetch a list of blocks (optionally in background thread)."""
        for bid in block_ids:
            self._store.prefetch(bid)

    def end_of_turn(self, total_blocks: int) -> List[int]:
        """
        Convenience method: predict, prefetch, and reset tracker.

        Call at the end of each dialogue turn.

        Returns
        -------
        List of prefetched block IDs.
        """
        blocks = self.predict_next_turn_blocks(total_blocks)
        self.prefetch_batch(blocks)
        with self._lock:
            self._tracker.reset()
        return blocks

    @property
    def step_count(self) -> int:
        """Number of decode steps recorded since last reset."""
        return self._tracker.step_count
