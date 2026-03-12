"""squish/semantic_response_cache.py

SemanticResponseCache — Embedding-similarity response deduplication cache.

Serving the same or semantically equivalent requests from cache can
dramatically reduce inference cost and p99 latency.  Exact-match caching
(e.g. on a hash of the prompt string) misses near-duplicates — paraphrased
questions, minor spelling variants, or multi-language equivalents that all map
to the same answer.

SemanticResponseCache operates in the embedding space.  It stores
``(embedding, response)`` pairs and, on lookup, performs a linear scan over
all stored embeddings to find the one with the highest cosine similarity to
the query.  If that similarity meets or exceeds ``similarity_threshold`` the
cached response is returned; otherwise a cache miss is signalled and the
caller is expected to generate a fresh response and call :meth:`store`.

Eviction follows an LRU policy implemented with ``collections.OrderedDict``:
entries are ordered from least-recently-used (front) to most-recently-used
(back).  A successful lookup promotes the matched entry to the MRU position.
When the cache reaches capacity the LRU entry (front) is evicted.

For the recommended capacity of <=256 the O(n) linear scan adds negligible
overhead compared to a model forward pass.

Example usage::

    import numpy as np
    from squish.semantic_response_cache import CacheConfig, SemanticResponseCache

    cfg   = CacheConfig(capacity=64, similarity_threshold=0.95, embedding_dim=32)
    cache = SemanticResponseCache(cfg)

    emb = np.random.randn(32).astype(np.float32)
    cache.store(emb, "The sky is blue.")

    # Nearly identical embedding — should hit.
    hit = cache.lookup(emb + np.random.randn(32) * 0.01)
    print(hit)          # "The sky is blue."
    print(cache.stats)
"""

from __future__ import annotations

__all__ = ["CacheConfig", "CacheStats", "SemanticResponseCache"]

import collections
from dataclasses import dataclass
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class CacheConfig:
    """Configuration for :class:`SemanticResponseCache`.

    Attributes:
        capacity:             Maximum number of ``(embedding, response)``
                              pairs to store before LRU eviction.
        similarity_threshold: Minimum cosine similarity in ``[0, 1]``
                              required for a cache hit.
        embedding_dim:        Expected dimensionality of every embedding
                              vector.
    """

    capacity: int = 256
    similarity_threshold: float = 0.95
    embedding_dim: int = 64

    def __post_init__(self) -> None:
        if self.capacity < 1:
            raise ValueError(f"capacity must be >= 1, got {self.capacity}")
        if not (0.0 <= self.similarity_threshold <= 1.0):
            raise ValueError(
                f"similarity_threshold must be in [0, 1], "
                f"got {self.similarity_threshold}"
            )
        if self.embedding_dim < 1:
            raise ValueError(
                f"embedding_dim must be >= 1, got {self.embedding_dim}"
            )


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------


@dataclass
class CacheStats:
    """Runtime statistics for :class:`SemanticResponseCache`.

    Attributes:
        n_hits:   Number of successful cache lookups (similarity >= threshold).
        n_misses: Number of lookup misses (no entry met the threshold).
        n_stored: Cumulative number of store operations performed.
        hit_rate: ``n_hits / (n_hits + n_misses)``, or 0.0 if no lookups have
                  been attempted.
    """

    n_hits: int
    n_misses: int
    n_stored: int
    hit_rate: float


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------


class SemanticResponseCache:
    """LRU response cache with cosine-similarity-based lookup.

    Entries are keyed internally by a monotonically increasing integer.
    Lookup performs a linear cosine-similarity scan over all stored embeddings
    and returns the response associated with the best match when its
    similarity meets the threshold.

    Args:
        config: A :class:`CacheConfig` controlling capacity, similarity
                threshold, and expected embedding dimensionality.
    """

    def __init__(self, config: CacheConfig) -> None:
        self._cfg = config
        # OrderedDict from entry_id (int) → (embedding, response).
        # Ordered least-recently-used → most-recently-used.
        self._store: collections.OrderedDict[int, tuple[np.ndarray, str]] = (
            collections.OrderedDict()
        )
        self._n_hits: int = 0
        self._n_misses: int = 0
        self._n_stored: int = 0
        self._next_id: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def lookup(self, embedding: np.ndarray) -> Optional[str]:
        """Find and return a cached response for *embedding*.

        Performs a linear cosine-similarity scan.  The entry with the highest
        similarity is promoted to the MRU position and its response is
        returned when similarity >= ``similarity_threshold``.

        Args:
            embedding: Float array of shape ``(embedding_dim,)``.

        Returns:
            The cached response string on a hit, or ``None`` on a miss.

        Raises:
            ValueError: If *embedding* has the wrong shape.
        """
        embedding = self._validate_embedding(embedding)
        if not self._store:
            self._n_misses += 1
            return None

        best_key: Optional[int] = None
        best_sim: float = -2.0

        for key, (stored_emb, _) in self._store.items():
            sim = self.similarity(embedding, stored_emb)
            if sim > best_sim:
                best_sim = sim
                best_key = key

        if best_sim >= self._cfg.similarity_threshold and best_key is not None:
            # Promote matched entry to MRU.
            self._store.move_to_end(best_key)
            self._n_hits += 1
            return self._store[best_key][1]

        self._n_misses += 1
        return None

    def store(self, embedding: np.ndarray, response: str) -> None:
        """Add an ``(embedding, response)`` pair to the cache.

        When the cache is at capacity the least-recently-used entry is evicted
        before inserting the new one.  A copy of *embedding* is stored so
        later mutations of the caller's array do not corrupt the cache.

        Args:
            embedding: Float array of shape ``(embedding_dim,)``.
            response:  The response string to cache.
        """
        embedding = self._validate_embedding(embedding)
        if len(self._store) >= self._cfg.capacity:
            # Evict LRU entry (front of the ordered dict).
            self._store.popitem(last=False)
        key = self._next_id
        self._next_id += 1
        self._store[key] = (embedding.copy(), response)
        self._n_stored += 1

    def similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Cosine similarity between two embedding vectors.

        Args:
            a: Float array of shape ``(embedding_dim,)``.
            b: Float array of shape ``(embedding_dim,)``.

        Returns:
            Cosine similarity in ``[-1, 1]``.  Returns 0.0 when either
            vector has near-zero norm to avoid division by zero.
        """
        a = np.asarray(a, dtype=np.float64)
        b = np.asarray(b, dtype=np.float64)
        norm_a = float(np.linalg.norm(a))
        norm_b = float(np.linalg.norm(b))
        if norm_a < 1e-12 or norm_b < 1e-12:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    @property
    def stats(self) -> CacheStats:
        """Runtime cache statistics snapshot."""
        total = self._n_hits + self._n_misses
        hit_rate = self._n_hits / total if total > 0 else 0.0
        return CacheStats(
            n_hits=self._n_hits,
            n_misses=self._n_misses,
            n_stored=self._n_stored,
            hit_rate=hit_rate,
        )

    @property
    def size(self) -> int:
        """Current number of entries held in the cache."""
        return len(self._store)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _validate_embedding(self, embedding: np.ndarray) -> np.ndarray:
        """Return *embedding* as a float32 1-D array after shape checks."""
        embedding = np.asarray(embedding, dtype=np.float32)
        if embedding.ndim != 1:
            raise ValueError(
                f"embedding must be 1-D, got shape {embedding.shape}"
            )
        if embedding.shape[0] != self._cfg.embedding_dim:
            raise ValueError(
                f"embedding dim ({embedding.shape[0]}) does not match "
                f"config.embedding_dim ({self._cfg.embedding_dim})"
            )
        return embedding
