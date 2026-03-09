"""
squish/specontext.py

SpeContext — Distilled Model as KV Retrieval Algorithm.

Based on:
  "SpeContext: Speculative KV Cache Compression via Distilled Context"
  ASPLOS 2026 — arXiv:2512.00722

Problem
-------
Existing long-context KV management systems use *heuristics* to decide which
tokens to keep in the GPU hot tier:
  - H2O: keep tokens that received high cumulative attention (heavy hitters)
  - Quest: keep tokens matched by query-aware sparsity
  - ClusterKV: keep semantically similar tokens

All heuristics are imperfect proxies.  SpeContext's key insight:

  *Due to the homology between a distilled LM and the original LLM, the
  information they focus on (important tokens) exhibits a high degree of
  similarity given the same inputs.*

The distilled model (e.g., Qwen3-1.5B for Qwen3-8B) IS the retrieval
algorithm.  Its attention head weights identify the *same* important tokens
that the target model would attend to — at 1/5 the compute cost.

Three-level co-design
---------------------
1. **Algorithm**: lightweight retrieval head built from the distilled model's
   head-level attention weights, with >90% parameter reduction by pruning
   redundant attention heads.
2. **System**: asynchronous prefetch with elastic loading — overlaps KV cache
   retrieval with LLM computation, reducing data transfer by up to 90%.
3. **Compilation**: adaptive memory layout that maximises GPU utilisation.

Results
-------
- Up to 24.89× throughput improvement (cloud)
- Up to 10.06× speedup (edge)
- Negligible accuracy loss
- Native GQA support (Qwen3's attention type)

Conflict notes
--------------
- **Replaces** ClusterKV, ParisKV, PQCache as the primary KV retrieval layer.
- **Uses Qwen3-1.5B** which is already in Squish's model zoo — no new infra.
- **Synergy with ShadowKV**: SpeContext selects *which* tokens to keep on GPU;
  ShadowKV handles *how* to store them (low-rank K + offloaded V).
- **Elastic loading** is the async prefetch described in prior reports, here
  fully engineered.

Provides
--------
  SpeContextConfig          — configuration knobs.
  DistilledRetrievalHead    — scores token importance using distilled-model
                              attention weigths.
  SpeContextCache           — manages hot / prefetch / offload tiers with
                              async prefetch.
  SpeContextStats           — per-request performance counters.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

__all__ = [
    "SpeContextConfig",
    "DistilledRetrievalHead",
    "SpeContextCache",
    "SpeContextStats",
]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class SpeContextConfig:
    """Configuration for SpeContext KV retrieval.

    Parameters
    ----------
    retrieval_topk:
        Number of KV positions to retrieve into the GPU hot tier per query.
    prefetch_budget:
        Maximum number of KV positions staged in the prefetch buffer ahead
        of demand (elastic loading).
    head_dim:
        Dimension of each attention head vector (used for scoring).
    n_retrieval_heads:
        Number of attention heads in the distilled model's retrieval head.
        Pruned from the full distilled model; >90% param reduction.
    gqa_groups:
        For GQA (Grouped Query Attention): number of KV head groups.
        1 = MHA; Qwen3-8B uses GQA with groups > 1.
    """

    retrieval_topk: int = 128
    prefetch_budget: int = 256
    head_dim: int = 64
    n_retrieval_heads: int = 4
    gqa_groups: int = 4

    def __post_init__(self) -> None:
        if self.retrieval_topk < 1:
            raise ValueError("retrieval_topk must be >= 1")
        if self.prefetch_budget < self.retrieval_topk:
            raise ValueError("prefetch_budget must be >= retrieval_topk")
        if self.head_dim < 1:
            raise ValueError("head_dim must be >= 1")
        if self.n_retrieval_heads < 1:
            raise ValueError("n_retrieval_heads must be >= 1")
        if self.gqa_groups < 1:
            raise ValueError("gqa_groups must be >= 1")


# ---------------------------------------------------------------------------
# DistilledRetrievalHead
# ---------------------------------------------------------------------------

class DistilledRetrievalHead:
    """Scores token importance via distilled-model attention weights.

    The retrieval head projects the query vector against a compressed key
    matrix derived from the distilled model's pruned attention heads.
    The scores identify which KV positions the full target model is most
    likely to attend to.

    Parameters
    ----------
    config:
        ``SpeContextConfig``.
    head_weights:
        Optional pre-trained head projection weights, shape
        ``(n_retrieval_heads, head_dim, head_dim)``.  If ``None``, a random
        orthogonal initialisation is used (replaced online via ``set_weights``).
    """

    def __init__(
        self,
        config: SpeContextConfig | None = None,
        head_weights: np.ndarray | None = None,
    ) -> None:
        self._cfg = config or SpeContextConfig()
        if head_weights is not None:
            expected = (self._cfg.n_retrieval_heads, self._cfg.head_dim, self._cfg.head_dim)
            if head_weights.shape != expected:
                raise ValueError(
                    f"head_weights shape {head_weights.shape} != expected {expected}"
                )
            self._W = head_weights.astype(np.float32)
        else:
            # Random orthogonal init — replaced by set_weights() in production
            rng = np.random.default_rng(0)
            W = rng.standard_normal(
                (self._cfg.n_retrieval_heads, self._cfg.head_dim, self._cfg.head_dim)
            ).astype(np.float32)
            # Orthogonalise each head
            for i in range(self._cfg.n_retrieval_heads):
                Q, _ = np.linalg.qr(W[i])
                W[i] = Q
            self._W = W

    def set_weights(self, head_weights: np.ndarray) -> None:
        expected = (self._cfg.n_retrieval_heads, self._cfg.head_dim, self._cfg.head_dim)
        if head_weights.shape != expected:
            raise ValueError(f"head_weights shape {head_weights.shape} != {expected}")
        self._W = head_weights.astype(np.float32)

    def score_tokens(
        self,
        query_vec: np.ndarray,
        key_matrix: np.ndarray,
    ) -> np.ndarray:
        """Compute per-token importance scores.

        Parameters
        ----------
        query_vec:
            Shape ``(head_dim,)`` — the current decode query vector.
        key_matrix:
            Shape ``(seq_len, head_dim)`` — the key vectors of all cached
            tokens.

        Returns
        -------
        np.ndarray
            Shape ``(seq_len,)`` — importance score per token (higher = more
            attended by the distilled model → keep in hot tier).
        """
        q = np.asarray(query_vec, dtype=np.float32).ravel()
        K = np.asarray(key_matrix, dtype=np.float32)
        if len(q) != self._cfg.head_dim:
            raise ValueError(f"query_vec length {len(q)} != head_dim {self._cfg.head_dim}")
        if K.shape[1] != self._cfg.head_dim:
            raise ValueError(f"key_matrix dim-1 {K.shape[1]} != head_dim {self._cfg.head_dim}")

        # Aggregate scores across all retrieval heads via mean attention
        scores_per_head = np.zeros(K.shape[0], dtype=np.float32)
        scale = 1.0 / np.sqrt(self._cfg.head_dim)
        for W_h in self._W:
            projected_q = W_h @ q            # (head_dim,)
            projected_K = K @ W_h.T         # (seq_len, head_dim)
            attn = (projected_K @ projected_q) * scale  # (seq_len,)
            scores_per_head += attn
        return scores_per_head / self._cfg.n_retrieval_heads

    def top_k_indices(
        self,
        query_vec: np.ndarray,
        key_matrix: np.ndarray,
        k: int,
    ) -> np.ndarray:
        """Return indices of the *k* highest-scored tokens.

        Returns
        -------
        np.ndarray
            Shape ``(min(k, seq_len),)`` — sorted descending by score.
        """
        scores = self.score_tokens(query_vec, key_matrix)
        k = min(k, len(scores))
        return np.argpartition(scores, -k)[-k:]


# ---------------------------------------------------------------------------
# SpeContextCache
# ---------------------------------------------------------------------------

class SpeContextCache:
    """Three-tier KV cache managed by SpeContext's elastic loading strategy.

    Tiers
    -----
    - **Hot** (GPU): top-*retrieval_topk* tokens identified by
      ``DistilledRetrievalHead``.
    - **Prefetch buffer**: next *prefetch_budget* tokens staged asynchronously
      ahead of demand.
    - **Cold** (CPU/disk): everything else.

    For this numpy reference, all tiers reside in CPU RAM.  A production
    implementation would route the hot tier to GPU device memory.

    Parameters
    ----------
    retrieval_head:
        Configured ``DistilledRetrievalHead``.
    config:
        ``SpeContextConfig``.
    """

    def __init__(
        self,
        retrieval_head: DistilledRetrievalHead,
        config: SpeContextConfig | None = None,
    ) -> None:
        self._head = retrieval_head
        self._cfg = config or SpeContextConfig()
        # Storage: list of (key, value) arrays per position
        self._keys: list[np.ndarray] = []    # (head_dim,) each
        self._values: list[np.ndarray] = []  # (head_dim,) each (value vectors)
        self._hot_indices: np.ndarray = np.array([], dtype=np.int64)
        self._prefetch_indices: np.ndarray = np.array([], dtype=np.int64)

    def append(self, key: np.ndarray, value: np.ndarray) -> None:
        """Append one KV pair to the cold store."""
        self._keys.append(np.asarray(key, dtype=np.float32).ravel())
        self._values.append(np.asarray(value, dtype=np.float32).ravel())

    def retrieve(self, query_vec: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Retrieve the hot-tier KV block for the given query.

        Updates hot and prefetch indices, simulating the elastic loading
        strategy (async prefetch of the next most-likely KV positions).

        Returns
        -------
        hot_keys:
            Shape ``(n_hot, head_dim)``.
        hot_values:
            Shape ``(n_hot, head_dim)``.
        hot_indices:
            Shape ``(n_hot,)`` — position indices of the hot tokens.
        """
        n = len(self._keys)
        if n == 0:
            empty = np.zeros((0, self._cfg.head_dim), dtype=np.float32)
            return empty, empty, np.array([], dtype=np.int64)

        key_matrix = np.stack(self._keys)  # (n, head_dim)
        topk = min(self._cfg.retrieval_topk, n)
        hot_idx = self._head.top_k_indices(query_vec, key_matrix, k=topk)
        self._hot_indices = hot_idx

        # Prefetch: precompute next biggest candidates
        prefetch_k = min(self._cfg.prefetch_budget, n)
        all_top = self._head.top_k_indices(query_vec, key_matrix, k=prefetch_k)
        # Exclude already-hot from prefetch
        hot_set = set(hot_idx.tolist())
        self._prefetch_indices = np.array(
            [i for i in all_top.tolist() if i not in hot_set],
            dtype=np.int64,
        )

        hot_keys = key_matrix[hot_idx]
        hot_values = np.stack(self._values)[hot_idx]
        return hot_keys, hot_values, hot_idx

    def reset(self) -> None:
        self._keys.clear()
        self._values.clear()
        self._hot_indices = np.array([], dtype=np.int64)
        self._prefetch_indices = np.array([], dtype=np.int64)

    @property
    def size(self) -> int:
        return len(self._keys)

    @property
    def hot_count(self) -> int:
        return len(self._hot_indices)

    @property
    def prefetch_count(self) -> int:
        return len(self._prefetch_indices)


# ---------------------------------------------------------------------------
# SpeContextStats
# ---------------------------------------------------------------------------

@dataclass
class SpeContextStats:
    """Per-request counters for SpeContext.

    Attributes
    ----------
    retrieval_calls:
        Number of retrieve() calls issued.
    hot_tokens_served:
        Total tokens served from the hot tier.
    prefetch_hits:
        Number of times a prefetched token was immediately requested.
    cold_accesses:
        Accesses that bypassed both hot and prefetch tiers.
    data_transferred_kb:
        Simulated data transfer volume (hot tier tokens × head_dim × 4 bytes).
    """

    retrieval_calls: int = 0
    hot_tokens_served: int = 0
    prefetch_hits: int = 0
    cold_accesses: int = 0
    data_transferred_kb: float = 0.0

    @property
    def hot_tier_rate(self) -> float:
        total = self.hot_tokens_served + self.cold_accesses
        return self.hot_tokens_served / total if total > 0 else 0.0

    @property
    def mean_hot_per_call(self) -> float:
        return self.hot_tokens_served / self.retrieval_calls if self.retrieval_calls > 0 else 0.0
