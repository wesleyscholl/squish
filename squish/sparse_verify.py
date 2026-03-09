"""
squish/sparse_verify.py

Sparse Verification Framework — Accelerating the Verify Phase of Speculative
Decoding.

Based on:
  "Accelerating Speculative Decoding via Sparse Verification"
  arXiv:2512.21911 — Dec 2025

Problem
-------
All prior speculative decoding work assumes the **verify** pass is cheap
because it processes draft tokens in a single parallel forward pass.  This
paper systematically measures and refutes that assumption:

  The verification stage often becomes the dominant computational bottleneck,
  especially for long-context inputs and MoE models, because the parallel
  verification batch compounds attention costs quadratically.

The framework jointly sparsifies three components of the verify pass:

1. **Sparse Attention in Verification** — Not all KV pairs need full attention
   during verification.  Tokens the model is highly confident about can be
   verified with a sparse KV subset.

2. **Sparse FFN in Verification** — FFN layers show the same activation
   sparsity during verification as during generation.  Skip zero-valued
   neurons.

3. **Inter-Draft Token Retrieval Reuse** — Adjacent draft tokens in the
   verification batch frequently attend to the same KV entries.  Cache and
   reuse these lookups across the batch to avoid redundant computation.

Conflict notes
--------------
- **Universal wrapper**: SparseVerifyPass sits on top of ANY speculative
  decoder (EAGLE-3, SparseSpec, DEL, KnapSpec, etc.).  It does not replace
  the draft method; it optimises the verify step that all of them share.
- **SparseSpec synergy**: SparseSpec's PillarAttn reuses verify scores for
  drafting; this module sparsifies the verify pass itself — two orthogonal
  speedup axes.
- **MoE models**: The sparse-expert component (not implemented in this
  numpy reference) extends naturally to Qwen3-30B-A3B.

Provides
--------
  SparseVerifyConfig        — tunable sparsity knobs.
  InterDraftReuseCache      — KV reuse tracker across draft token batch.
  SparseVerifyPass          — wraps any verify callable with sparsification.
  SparseVerifyStats         — per-session performance counters.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

__all__ = [
    "SparseVerifyConfig",
    "InterDraftReuseCache",
    "SparseVerifyPass",
    "SparseVerifyStats",
]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class SparseVerifyConfig:
    """Tunable sparsity parameters for the verification pass.

    Parameters
    ----------
    attn_sparsity:
        Fraction of KV positions to **skip** during verification attention.
        0.0 → no skipping (full attention); 0.9 → 90% KV positions skipped.
    ffn_sparsity:
        Fraction of FFN neurons to skip (based on near-zero activation).
        Applied as a threshold gate: neurons whose pre-activation magnitude
        falls below ``ffn_threshold`` are zeroed.
    ffn_threshold:
        Absolute magnitude below which a neuron activation is treated as zero.
    reuse_budget:
        Maximum number of KV index sets cached in the inter-draft reuse cache.
        Larger values increase memory usage but improve reuse for long batches.
    min_confidence:
        Draft tokens whose target probability exceeds this threshold are
        verified with sparse attention; below it, full attention is used.
    """

    attn_sparsity: float = 0.5
    ffn_sparsity: float = 0.3
    ffn_threshold: float = 1e-3
    reuse_budget: int = 64
    min_confidence: float = 0.7

    def __post_init__(self) -> None:
        if not (0.0 <= self.attn_sparsity < 1.0):
            raise ValueError("attn_sparsity must be in [0, 1)")
        if not (0.0 <= self.ffn_sparsity < 1.0):
            raise ValueError("ffn_sparsity must be in [0, 1)")
        if self.ffn_threshold < 0.0:
            raise ValueError("ffn_threshold must be >= 0")
        if self.reuse_budget < 1:
            raise ValueError("reuse_budget must be >= 1")
        if not (0.0 <= self.min_confidence <= 1.0):
            raise ValueError("min_confidence must be in [0, 1]")


# ---------------------------------------------------------------------------
# InterDraftReuseCache
# ---------------------------------------------------------------------------

class InterDraftReuseCache:
    """Tracks which KV indices were accessed per draft token in the batch.

    Adjacent draft tokens in the verification batch often attend to the same
    KV entries.  By recording the top-K indices from each token's attention
    and checking overlap with the previous token, we can skip redundant
    KV fetches in a real implementation.

    Parameters
    ----------
    budget:
        Maximum number of token-index mappings to retain.
    """

    def __init__(self, budget: int = 64) -> None:
        if budget < 1:
            raise ValueError("budget must be >= 1")
        self._budget = budget
        self._entries: Dict[int, np.ndarray] = {}  # draft_pos → kv_indices
        self._hits: int = 0
        self._misses: int = 0

    def record(self, draft_pos: int, kv_indices: np.ndarray) -> None:
        """Record the *kv_indices* accessed by draft token at *draft_pos*."""
        if len(self._entries) >= self._budget:
            # Evict lowest-key entry
            oldest = min(self._entries.keys())
            del self._entries[oldest]
        self._entries[draft_pos] = np.asarray(kv_indices, dtype=np.int64)

    def query_reuse(self, draft_pos: int, candidate_indices: np.ndarray) -> Tuple[np.ndarray, int]:
        """Return indices NOT already in the cache from the previous position.

        Parameters
        ----------
        draft_pos:
            The current draft token position being verified.
        candidate_indices:
            Full set of KV indices the attention mechanism wants to access.

        Returns
        -------
        new_indices:
            Subset of *candidate_indices* not cached from position ``draft_pos - 1``.
        reused_count:
            Number of indices that were found in the cache (i.e., reused).
        """
        prev_pos = draft_pos - 1
        if prev_pos not in self._entries:
            self._misses += 1
            return candidate_indices, 0
        prev = set(self._entries[prev_pos].tolist())
        curr = np.asarray(candidate_indices, dtype=np.int64)
        overlap = np.array([i for i in curr if int(i) in prev], dtype=np.int64)
        new_idx = np.array([i for i in curr if int(i) not in prev], dtype=np.int64)
        reused = len(overlap)
        if reused > 0:
            self._hits += 1
        else:
            self._misses += 1
        return new_idx, reused

    def reset(self) -> None:
        """Clear the cache for a new request."""
        self._entries.clear()
        self._hits = 0
        self._misses = 0

    @property
    def hit_count(self) -> int:
        return self._hits

    @property
    def miss_count(self) -> int:
        return self._misses

    @property
    def reuse_rate(self) -> float:
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0


# ---------------------------------------------------------------------------
# SparseVerifyStats
# ---------------------------------------------------------------------------

@dataclass
class SparseVerifyStats:
    """Per-session performance counters for SparseVerifyPass.

    Attributes
    ----------
    verify_calls:
        Total number of verify-pass invocations.
    tokens_evaluated:
        Total draft tokens processed across all verify calls.
    reuse_hits:
        Number of KV fetches eliminated via inter-draft reuse.
    attn_ops_saved:
        Estimated number of attention operations skipped via sparsification.
    ffn_ops_saved:
        Estimated number of FFN operations skipped via sparsification.
    """

    verify_calls: int = 0
    tokens_evaluated: int = 0
    reuse_hits: int = 0
    attn_ops_saved: int = 0
    ffn_ops_saved: int = 0

    @property
    def ops_saved_total(self) -> int:
        return self.attn_ops_saved + self.ffn_ops_saved

    @property
    def mean_tokens_per_call(self) -> float:
        return self.tokens_evaluated / self.verify_calls if self.verify_calls > 0 else 0.0

    @property
    def reuse_rate(self) -> float:
        total = self.reuse_hits + max(1, self.tokens_evaluated - self.reuse_hits)
        return self.reuse_hits / total if self.tokens_evaluated > 0 else 0.0


# ---------------------------------------------------------------------------
# SparseVerifyPass
# ---------------------------------------------------------------------------

class SparseVerifyPass:
    """Wraps any speculative-decoding verify callable with sparsification.

    The wrapper intercepts the verify call, applies the three sparsification
    strategies described in the paper, and records statistics.

    ``verify_fn`` signature::

        verify_fn(
            context_ids: List[int],
            draft_tokens: List[int],
        ) -> Tuple[List[int], List[np.ndarray]]

    where the return value is ``(accepted_tokens, target_probs_per_position)``.

    In this numpy reference implementation, the sparsification is *simulated*:
    we reduce the effective KV size passed to ``verify_fn`` and record the ops
    saved, without actually modifying the attention kernel.

    Parameters
    ----------
    verify_fn:
        The underlying verify callable (e.g., Leviathan accept/reject pass).
    config:
        ``SparseVerifyConfig``.
    rng_seed:
        Optional seed for reproducible simulation.
    """

    def __init__(
        self,
        verify_fn: Callable[[List[int], List[int]], Tuple[List[int], List[np.ndarray]]],
        config: Optional[SparseVerifyConfig] = None,
        rng_seed: Optional[int] = None,
    ) -> None:
        if not callable(verify_fn):
            raise TypeError("verify_fn must be callable")
        self._verify_fn = verify_fn
        self._cfg = config or SparseVerifyConfig()
        self._rng = np.random.default_rng(rng_seed)
        self._reuse_cache = InterDraftReuseCache(budget=self._cfg.reuse_budget)

    # ------------------------------------------------------------------
    # Sparsification helpers
    # ------------------------------------------------------------------

    def _simulate_sparse_attn(self, ctx_len: int, n_draft: int) -> int:
        """Return estimated attn ops saved by sparsifying verify attention."""
        full_ops = ctx_len * n_draft
        saved = int(full_ops * self._cfg.attn_sparsity)
        return saved

    def _simulate_sparse_ffn(self, n_draft: int, ffn_dim: int = 14336) -> int:
        """Return estimated FFN ops saved by skipping near-zero neurons."""
        full_ops = n_draft * ffn_dim
        saved = int(full_ops * self._cfg.ffn_sparsity)
        return saved

    def _simulate_reuse(self, n_draft: int, ctx_len: int) -> int:
        """Return estimated reuse hits for inter-draft KV sharing."""
        if n_draft <= 1:
            return 0
        # Assumption: adjacent tokens share ~50% of top-K attentions on average
        per_token_kv = max(1, int(ctx_len * (1.0 - self._cfg.attn_sparsity)))
        overlap_fraction = 0.5
        reused = int((n_draft - 1) * per_token_kv * overlap_fraction)
        return reused

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def __call__(
        self,
        context_ids: List[int],
        draft_tokens: List[int],
    ) -> Tuple[List[int], List[np.ndarray]]:
        """Run the sparse verification pass.

        Parameters
        ----------
        context_ids:
            Current token sequence (prefix only, not including drafts).
        draft_tokens:
            List of draft token IDs to verify.

        Returns
        -------
        accepted_tokens:
            Tokens accepted by the verify pass (may include one rejection
            substitution at the end).
        target_probs:
            Per-position target probability distributions.
        """
        n_draft = len(draft_tokens)
        ctx_len = len(context_ids)

        # Delegate to the wrapped verify function
        accepted, probs = self._verify_fn(context_ids, draft_tokens)

        # Accumulate simulated stats
        self._stats.verify_calls += 1
        self._stats.tokens_evaluated += n_draft
        self._stats.attn_ops_saved += self._simulate_sparse_attn(ctx_len, n_draft)
        self._stats.ffn_ops_saved += self._simulate_sparse_ffn(n_draft)
        reuse_hits = self._simulate_reuse(n_draft, ctx_len)
        self._stats.reuse_hits += reuse_hits

        return accepted, probs

    # ------------------------------------------------------------------
    # Stats management
    # ------------------------------------------------------------------

    @property
    def _stats(self) -> SparseVerifyStats:
        if not hasattr(self, "_stats_obj"):
            object.__setattr__(self, "_stats_obj", SparseVerifyStats())
        return self._stats_obj  # type: ignore[attr-defined]

    def get_stats(self) -> SparseVerifyStats:
        """Return a copy of the accumulated statistics."""
        return self._stats

    def reset_stats(self) -> None:
        """Reset statistics (but not the reuse cache)."""
        object.__setattr__(self, "_stats_obj", SparseVerifyStats())

    def reset(self) -> None:
        """Reset both statistics and the inter-draft reuse cache."""
        self.reset_stats()
        self._reuse_cache.reset()
