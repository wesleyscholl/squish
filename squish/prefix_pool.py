"""PrefixPool — Cross-request KV prefix sharing pool.

When multiple concurrent requests share the same system prompt (e.g. RAG
pipeline, chatbot with fixed instructions), the KV activations for that
prefix need only be computed once and can be shared across all requests.
PrefixPool stores (hash → KV tensor) entries and provides O(1) lookup.

Reference:
    Zheng et al., "Efficiently Programming Large Language Models using
    SGLang", arXiv 2023. https://arxiv.org/abs/2312.07104

    Pope et al., "Efficiently Scaling Transformer Inference", MLSys 2023.
    https://arxiv.org/abs/2211.05102

Usage example::

    import numpy as np
    from squish.prefix_pool import PrefixPoolConfig, PrefixPool

    config = PrefixPoolConfig(max_entries=128, n_heads=32, head_dim=128)
    pool = PrefixPool(config)

    tokens = [1, 2, 3, 4, 5]
    keys = np.random.randn(32, 5, 128).astype(np.float32)
    values = np.random.randn(32, 5, 128).astype(np.float32)

    prefix_hash = pool.put(tokens, keys, values)
    cached = pool.get(tokens)
    print(f"Cache hit: {cached is not None}")
    print(f"Hit rate: {pool.hit_rate:.3f}, KV tokens saved: {pool.total_kv_saved}")
"""

from __future__ import annotations

__all__ = [
    "PrefixPoolConfig",
    "PrefixEntry",
    "PrefixPool",
    "PrefixPoolStats",
]

import hashlib
import time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PrefixPoolConfig:
    """Configuration for the cross-request KV prefix pool.

    Attributes:
        max_entries: Maximum number of cached prefixes before eviction.
        n_heads: Number of query/key heads in the attention layer.
        head_dim: Per-head key/value dimension.
        kv_n_heads: Number of key/value heads (GQA/MQA support).  Defaults
            to *n_heads* when ``None``.
        eviction_policy: Either ``"lru"`` (least recently used) or
            ``"lfu"`` (least frequently used).
    """

    max_entries: int = 256
    n_heads: int = 32
    head_dim: int = 128
    kv_n_heads: Optional[int] = None
    eviction_policy: str = "lru"

    def __post_init__(self) -> None:
        if self.max_entries <= 0:
            raise ValueError(
                f"max_entries must be a positive integer, got {self.max_entries}"
            )
        if self.n_heads <= 0:
            raise ValueError(
                f"n_heads must be a positive integer, got {self.n_heads}"
            )
        if self.head_dim <= 0:
            raise ValueError(
                f"head_dim must be a positive integer, got {self.head_dim}"
            )
        if self.eviction_policy not in ("lru", "lfu"):
            raise ValueError(
                f"eviction_policy must be 'lru' or 'lfu', "
                f"got '{self.eviction_policy}'"
            )
        if self.kv_n_heads is None:
            object.__setattr__(self, "kv_n_heads", self.n_heads)
        elif self.kv_n_heads <= 0:  # type: ignore[operator]
            raise ValueError(
                f"kv_n_heads must be a positive integer, got {self.kv_n_heads}"
            )


# ---------------------------------------------------------------------------
# Cache entry
# ---------------------------------------------------------------------------

@dataclass
class PrefixEntry:
    """A single cached KV prefix.

    Attributes:
        prefix_hash: SHA-256 hex digest of the prefix token sequence.
        keys: Key tensor of shape ``(kv_n_heads, seq_len, head_dim)``.
        values: Value tensor of the same shape as *keys*.
        hit_count: Number of times this entry has been retrieved.
        last_used: Unix timestamp of the most recent retrieval (or insertion).
    """

    prefix_hash: str
    keys: np.ndarray   # (kv_n_heads, seq_len, head_dim)
    values: np.ndarray  # (kv_n_heads, seq_len, head_dim)
    hit_count: int = 0
    last_used: float = 0.0

    def __post_init__(self) -> None:
        if self.last_used == 0.0:
            self.last_used = time.time()


# ---------------------------------------------------------------------------
# Pool
# ---------------------------------------------------------------------------

def _hash_tokens(prefix_tokens: list) -> str:
    """Return a SHA-256 hex digest of the token ID sequence."""
    token_bytes = np.asarray(prefix_tokens, dtype=np.int64).tobytes()
    return hashlib.sha256(token_bytes).hexdigest()


class PrefixPool:
    """O(1) lookup pool for cached KV prefix tensors.

    Stores up to ``config.max_entries`` entries.  When the pool is full a
    single entry is evicted according to ``config.eviction_policy`` before
    inserting the new entry.
    """

    def __init__(self, config: PrefixPoolConfig) -> None:
        self._config = config
        self._entries: Dict[str, PrefixEntry] = {}
        self._n_hits: int = 0
        self._n_misses: int = 0
        self._n_evictions: int = 0
        self._total_kv_saved: int = 0

    # ------------------------------------------------------------------
    # Insertion
    # ------------------------------------------------------------------

    def put(
        self,
        prefix_tokens: list,
        keys: np.ndarray,
        values: np.ndarray,
    ) -> str:
        """Cache KV tensors for *prefix_tokens*.

        If the pool is at capacity, evicts one entry according to the
        configured policy before inserting.

        Args:
            prefix_tokens: Sequence of integer token IDs representing the
                shared prefix.
            keys: Key tensor of shape ``(kv_n_heads, seq_len, head_dim)``.
            values: Value tensor of the same shape as *keys*.

        Returns:
            The SHA-256 hash string identifying this cache entry.

        Raises:
            ValueError: If *keys* or *values* have an unexpected number of
                dimensions.
        """
        if keys.ndim != 3:
            raise ValueError(
                f"keys must be 3-D (kv_n_heads, seq_len, head_dim), "
                f"got shape {keys.shape}"
            )
        if values.shape != keys.shape:
            raise ValueError(
                f"values shape {values.shape} must match keys shape {keys.shape}"
            )
        h = _hash_tokens(prefix_tokens)
        # Re-insertion updates tensors without counting as a new entry
        if h not in self._entries:
            if len(self._entries) >= self._config.max_entries:
                self._evict()
        self._entries[h] = PrefixEntry(
            prefix_hash=h,
            keys=keys.astype(np.float32, copy=False),
            values=values.astype(np.float32, copy=False),
            last_used=time.time(),
        )
        return h

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def get(
        self, prefix_tokens: list
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Return cached ``(keys, values)`` for *prefix_tokens*, or ``None``.

        Updates *hit_count* and *last_used* on a cache hit.
        """
        h = _hash_tokens(prefix_tokens)
        if h not in self._entries:
            self._n_misses += 1
            return None
        entry = self._entries[h]
        entry.hit_count += 1
        entry.last_used = time.time()
        seq_len = entry.keys.shape[1]
        self._n_hits += 1
        self._total_kv_saved += seq_len
        return entry.keys, entry.values

    def contains(self, prefix_tokens: list) -> bool:
        """Return ``True`` if the prefix is currently cached."""
        return _hash_tokens(prefix_tokens) in self._entries

    # ------------------------------------------------------------------
    # Eviction
    # ------------------------------------------------------------------

    def evict_lru(self) -> None:
        """Remove the least recently used (oldest *last_used*) entry."""
        if not self._entries:
            return
        lru_key = min(
            self._entries, key=lambda h: self._entries[h].last_used
        )
        del self._entries[lru_key]
        self._n_evictions += 1

    def evict_lfu(self) -> None:
        """Remove the least frequently used (lowest *hit_count*) entry."""
        if not self._entries:
            return
        lfu_key = min(
            self._entries, key=lambda h: self._entries[h].hit_count
        )
        del self._entries[lfu_key]
        self._n_evictions += 1

    def _evict(self) -> None:
        """Evict one entry according to the configured policy."""
        if self._config.eviction_policy == "lru":
            self.evict_lru()
        else:
            self.evict_lfu()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def size(self) -> int:
        """Current number of cached entries."""
        return len(self._entries)

    @property
    def hit_rate(self) -> float:
        """Cache hit rate: ``hits / (hits + misses)``."""
        total = self._n_hits + self._n_misses
        if total == 0:
            return 0.0
        return self._n_hits / total

    @property
    def total_kv_saved(self) -> int:
        """Total number of KV tokens not recomputed due to cache hits.

        Each hit contributes the ``seq_len`` of the retrieved entry.
        """
        return self._total_kv_saved

    def get_stats(self) -> "PrefixPoolStats":
        """Return a snapshot of current pool statistics."""
        return PrefixPoolStats(
            n_hits=self._n_hits,
            n_misses=self._n_misses,
            n_evictions=self._n_evictions,
            total_tokens_saved=self._total_kv_saved,
        )


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

@dataclass
class PrefixPoolStats:
    """Aggregate statistics for a :class:`PrefixPool` session.

    Attributes:
        n_hits: Number of successful cache lookups.
        n_misses: Number of unsuccessful cache lookups.
        n_evictions: Number of entries evicted due to capacity pressure.
        total_tokens_saved: Cumulative KV tokens not re-computed.
    """

    n_hits: int = 0
    n_misses: int = 0
    n_evictions: int = 0
    total_tokens_saved: int = 0

    @property
    def hit_rate(self) -> float:
        """Fraction of lookups that were cache hits."""
        total = self.n_hits + self.n_misses
        if total == 0:
            return 0.0
        return self.n_hits / total

    @property
    def eviction_rate(self) -> float:
        """Fraction of all events (hits + misses + evictions) that were evictions."""
        total = self.n_hits + self.n_misses + self.n_evictions
        if total == 0:
            return 0.0
        return self.n_evictions / total
