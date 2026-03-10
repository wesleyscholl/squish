"""SmallKV — Small-Model-Assisted KV Eviction Compensation.

Implements the saliency shift compensation mechanism from SmallKV
(NeurIPS 2025 Spotlight, arXiv:2508.02751) that exploits the empirical
observation: LLMs within the same model family (e.g., Qwen3-8B and
Qwen3-1.5B) exhibit highly consistent attention patterns, so the smaller
model's attention scores are a reliable proxy for the larger model's.

Two compensation mechanisms:
  1. Saliency shift detection — re-identify tokens that were evicted but
     have regained importance according to the small model's attention.
  2. Marginal token V-only retention — keep V cache for medium-importance
     tokens (discard K) and approximate the K×Q computation using the
     small model's attention weights.

Reported: 1.75–2.56× throughput improvement while retaining performance
with only 10% KV cache budget.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import numpy as np


@dataclass
class SmallKVConfig:
    """Configuration for SmallKV compensation.

    Args:
        n_layers: Target model layer count.
        kv_budget_fraction: Fraction of tokens to keep in KV cache.
        saliency_shift_threshold: Attention score increase (small model) that
            triggers a token's recall from evicted state.
        marginal_v_only_fraction: Fraction of tokens to keep as V-only (no K).
        score_ema_alpha: EMA factor for smoothing small-model attention scores.
        recall_top_k: Max number of evicted tokens recalled per step.
        proxy_weight: Weight given to small model's attention proxy vs. direct
            attention score when the target model re-examines a token.
    """

    n_layers: int = 32
    kv_budget_fraction: float = 0.10
    saliency_shift_threshold: float = 0.05
    marginal_v_only_fraction: float = 0.15
    score_ema_alpha: float = 0.1
    recall_top_k: int = 8
    proxy_weight: float = 0.7

    def __post_init__(self) -> None:
        if self.n_layers <= 0:
            raise ValueError("n_layers must be positive")
        if not 0 < self.kv_budget_fraction <= 1.0:
            raise ValueError("kv_budget_fraction must be in (0, 1]")
        if not 0 < self.marginal_v_only_fraction <= 1.0:
            raise ValueError("marginal_v_only_fraction must be in (0, 1]")
        if not 0 < self.score_ema_alpha <= 1.0:
            raise ValueError("score_ema_alpha must be in (0, 1]")
        if self.recall_top_k <= 0:
            raise ValueError("recall_top_k must be positive")
        if not 0 <= self.proxy_weight <= 1.0:
            raise ValueError("proxy_weight must be in [0, 1]")
        if self.kv_budget_fraction + self.marginal_v_only_fraction > 1.0:
            raise ValueError(
                "kv_budget_fraction + marginal_v_only_fraction must be <= 1.0"
            )


@dataclass
class SaliencyTracker:
    """Per-layer saliency tracker using small-model attention as proxy.

    Maintains EMA-smoothed per-token attention scores from the small model
    and detects saliency shifts: tokens whose importance has increased
    materially since eviction.
    """

    config: SmallKVConfig
    layer_idx: int
    _scores: np.ndarray | None = field(default=None, repr=False)
    _prev_scores: np.ndarray | None = field(default=None, repr=False)
    _evicted: set[int] = field(default_factory=set)

    def update_scores(self, small_model_attn: np.ndarray) -> None:
        """Update EMA scores from small model attention on the current step.

        Args:
            small_model_attn: (seq_len,) or (n_heads, seq_len) attention weights.
        """
        if small_model_attn.ndim > 1:
            scores = small_model_attn.mean(axis=0).astype(np.float32)
        else:
            scores = small_model_attn.astype(np.float32)

        self._prev_scores = self._scores.copy() if self._scores is not None else None

        if self._scores is None or self._scores.size != scores.size:
            self._scores = scores.copy()
        else:
            a = self.config.score_ema_alpha
            self._scores = (1 - a) * self._scores + a * scores

    def detect_saliency_shifts(self) -> list[int]:
        """Return token indices that have experienced significant saliency increase.

        Returns:
            List of token indices (at most `recall_top_k`) that were evicted but
            now show scores above saliency_shift_threshold.
        """
        if self._scores is None or not self._evicted:
            return []

        shifts = []
        for tok_idx in self._evicted:
            if tok_idx < self._scores.size:
                if self._scores[tok_idx] >= self.config.saliency_shift_threshold:
                    shifts.append(tok_idx)

        # Sort by score descending; return top-k
        shifts.sort(key=lambda i: -self._scores[i])
        return shifts[: self.config.recall_top_k]

    def mark_evicted(self, token_indices: list[int]) -> None:
        """Record newly evicted tokens."""
        self._evicted.update(token_indices)

    def mark_recalled(self, token_indices: list[int]) -> None:
        """Remove recalled tokens from evicted set."""
        for i in token_indices:
            self._evicted.discard(i)

    @property
    def n_evicted(self) -> int:
        return len(self._evicted)

    @property
    def current_scores(self) -> np.ndarray | None:
        return self._scores

    def reset(self) -> None:
        self._scores = None
        self._prev_scores = None
        self._evicted.clear()


@dataclass
class MarginalVCache:
    """V-only cache entry for marginal tokens.

    Keeps V vectors for medium-importance tokens while discarding K.
    The small model's attention score approximates the K×Q dot product.
    """

    token_idx: int
    v_vector: np.ndarray        # (head_dim,) or (n_heads, head_dim)
    proxy_attn_score: float     # small model's attention score for this token


class SmallKVCache:
    """Full SmallKV cache: full KV for critical, V-only for marginal, none for evicted.

    This class manages three storage tiers and answers "what KV do I have
    for token i at layer l?" queries from the target model.
    """

    def __init__(self, config: SmallKVConfig) -> None:
        self.config = config
        # layer_idx -> {token_idx: (k_vec, v_vec) or None (v-only)}
        self._kv_store: dict[int, dict[int, tuple[np.ndarray, np.ndarray] | None]] = {}
        # layer_idx -> {token_idx: MarginalVCache}
        self._v_only_store: dict[int, dict[int, MarginalVCache]] = {}
        self._trackers: dict[int, SaliencyTracker] = {
            l: SaliencyTracker(config=config, layer_idx=l)
            for l in range(config.n_layers)
        }
        self._stats = SmallKVStats()

    def ingest(
        self,
        layer_idx: int,
        token_indices: np.ndarray,
        keys: np.ndarray,
        values: np.ndarray,
        importance_scores: np.ndarray,
    ) -> None:
        """Classify and store tokens for one layer.

        Args:
            layer_idx: Target model layer.
            token_indices: (seq_len,) integer token positions.
            keys: (seq_len, head_dim) key vectors.
            values: (seq_len, head_dim) value vectors.
            importance_scores: (seq_len,) importance score per token.
        """
        seq_len = len(token_indices)
        budget = max(1, int(seq_len * self.config.kv_budget_fraction))
        v_budget = max(1, int(seq_len * self.config.marginal_v_only_fraction))

        sorted_by_importance = np.argsort(-importance_scores)
        critical_pos = sorted_by_importance[:budget]
        marginal_pos = sorted_by_importance[budget : budget + v_budget]
        evicted_pos = sorted_by_importance[budget + v_budget :]

        if layer_idx not in self._kv_store:
            self._kv_store[layer_idx] = {}
            self._v_only_store[layer_idx] = {}

        for pos in critical_pos:
            self._kv_store[layer_idx][int(token_indices[pos])] = (
                keys[pos].copy(),
                values[pos].copy(),
            )

        for pos in marginal_pos:
            tok_idx = int(token_indices[pos])
            self._v_only_store[layer_idx][tok_idx] = MarginalVCache(
                token_idx=tok_idx,
                v_vector=values[pos].copy(),
                proxy_attn_score=float(importance_scores[pos]),
            )

        evicted_list = [int(token_indices[p]) for p in evicted_pos]
        self._trackers[layer_idx].mark_evicted(evicted_list)
        self._stats.record_ingest(
            n_critical=len(critical_pos),
            n_marginal=len(marginal_pos),
            n_evicted=len(evicted_list),
        )

    def check_and_recall(
        self, layer_idx: int, small_model_attn: np.ndarray
    ) -> list[int]:
        """Update saliency tracker and recall shifted tokens.

        Args:
            layer_idx: Layer to update.
            small_model_attn: (seq_len,) or (n_heads, seq_len) small model attention.

        Returns:
            List of token indices recalled from eviction.
        """
        tracker = self._trackers.get(layer_idx)
        if tracker is None:
            return []
        tracker.update_scores(small_model_attn)
        recalled = tracker.detect_saliency_shifts()
        if recalled:
            tracker.mark_recalled(recalled)
            self._stats.total_recalls += len(recalled)
        return recalled

    def get_kv(
        self, layer_idx: int, token_idx: int
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Retrieve (K, V) for a token.  Returns (None, None) if evicted."""
        kv = self._kv_store.get(layer_idx, {}).get(token_idx)
        if kv is not None:
            return kv
        vc = self._v_only_store.get(layer_idx, {}).get(token_idx)
        if vc is not None:
            # V-only: return (None, V) — K must be approximated from small model
            return None, vc.v_vector
        return None, None

    def reset(self, layer_idx: int | None = None) -> None:
        if layer_idx is not None:
            self._kv_store.pop(layer_idx, None)
            self._v_only_store.pop(layer_idx, None)
            if layer_idx in self._trackers:
                self._trackers[layer_idx].reset()
        else:
            self._kv_store.clear()
            self._v_only_store.clear()
            for t in self._trackers.values():
                t.reset()

    @property
    def stats(self) -> SmallKVStats:
        return self._stats


@dataclass
class SmallKVStats:
    """Aggregate statistics for SmallKV operations."""

    total_critical: int = 0
    total_marginal: int = 0
    total_evicted: int = 0
    total_recalls: int = 0

    def record_ingest(self, n_critical: int, n_marginal: int, n_evicted: int) -> None:
        self.total_critical += n_critical
        self.total_marginal += n_marginal
        self.total_evicted += n_evicted

    @property
    def total_tokens(self) -> int:
        return self.total_critical + self.total_marginal + self.total_evicted

    @property
    def retention_rate(self) -> float:
        if self.total_tokens == 0:
            return 0.0
        return (self.total_critical + self.total_marginal) / self.total_tokens

    @property
    def full_kv_rate(self) -> float:
        if self.total_tokens == 0:
            return 0.0
        return self.total_critical / self.total_tokens

    @property
    def recall_rate(self) -> float:
        if self.total_evicted == 0:
            return 0.0
        return self.total_recalls / self.total_evicted

    @property
    def estimated_throughput_multiplier(self) -> float:
        """Rough estimate based on paper's 1.75–2.56× range.

        Scales linearly with eviction rate (more eviction → more speedup).
        """
        evict_rate = 1.0 - self.retention_rate
        return 1.0 + evict_rate * 1.56  # max 2.56× at full eviction
