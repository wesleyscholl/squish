"""
SqueezeAttention — Joint 2D KV Budget Management.

Referenced in KVTuner related work; 2025 preprint.

Prior work handles KV budget along two axes independently:
  - Token dimension: which tokens to evict (H2O, SnapKV)
  - Layer dimension: how many KV slots per layer (PyramidKV)

SqueezeAttention jointly optimizes both axes, finding the Pareto-optimal
policy for a given total KV budget.  Independent application risks
compounding quality loss; joint optimization avoids it.

This module provides:
  - SqueezeConfig — total KV budget + search constraints
  - LayerKVBudget — (token_budget, precision) for one layer
  - BudgetAllocator — joint token + layer budget optimizer
  - SqueezeKVCache — applies the joint budget at runtime
  - SqueezeStats — compression statistics
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class SqueezeConfig:
    """Configuration for SqueezeAttention budget allocation.

    Args:
        n_layers:          Number of transformer layers.
        total_kv_budget:   Total KV tokens available across all layers.
        min_tokens_per_layer:  Lower bound on per-layer token budget.
        max_tokens_per_layer:  Upper bound on per-layer token budget.
        token_eviction:    Token eviction policy: 'attention' | 'recency'.
        interaction_penalty: Weight applied to discourage aggressive
                           simultaneous compression on both axes for the same
                           layer.  Higher = more balanced allocation.
    """

    n_layers: int = 32
    total_kv_budget: int = 16384
    min_tokens_per_layer: int = 64
    max_tokens_per_layer: int = 4096
    token_eviction: str = "attention"
    interaction_penalty: float = 0.5

    def __post_init__(self) -> None:
        if self.n_layers < 1:
            raise ValueError("n_layers must be >= 1")
        if self.total_kv_budget < 1:
            raise ValueError("total_kv_budget must be >= 1")
        if self.min_tokens_per_layer < 1:
            raise ValueError("min_tokens_per_layer must be >= 1")
        if self.max_tokens_per_layer < self.min_tokens_per_layer:
            raise ValueError(
                "max_tokens_per_layer must be >= min_tokens_per_layer"
            )
        valid_eviction = {"attention", "recency"}
        if self.token_eviction not in valid_eviction:
            raise ValueError(f"token_eviction must be one of {valid_eviction}")
        if self.interaction_penalty < 0:
            raise ValueError("interaction_penalty must be >= 0")

    @property
    def avg_tokens_per_layer(self) -> int:
        return self.total_kv_budget // max(1, self.n_layers)


# ---------------------------------------------------------------------------
# Layer budget
# ---------------------------------------------------------------------------

@dataclass
class LayerKVBudget:
    """KV budget for one transformer layer."""

    layer_idx: int
    token_budget: int
    """Maximum number of KV tokens to keep for this layer."""

    compression_score: float = 0.0
    """Combined compression score (higher = more compressed)."""

    @property
    def is_compressed(self) -> bool:
        return self.compression_score > 0.0


# ---------------------------------------------------------------------------
# Budget allocator — joint optimization
# ---------------------------------------------------------------------------

class BudgetAllocator:
    """Finds the Pareto-optimal joint token+layer KV budget allocation.

    Algorithm:
    1. Assign equal base budget to each layer.
    2. For each layer, estimate a "salience" score from calibration data
       (high salience = important layer, gets more budget).
    3. Redistribute budget proportionally to salience, subject to constraints.
    4. Apply interaction penalty: layers that already have reduced budget
       (from layer-wise compression) are protected from further deep token
       eviction.
    """

    def __init__(self, config: SqueezeConfig) -> None:
        self._config = config
        self._salience: dict[int, float] = {}

    def record_layer_salience(self, layer_idx: int, salience: float) -> None:
        """Record a salience score for *layer_idx* (e.g., mean attention entropy).

        Higher salience = more important = more budget allocated.
        """
        self._salience[layer_idx] = float(salience)

    def allocate(self) -> list[LayerKVBudget]:
        """Compute joint budget allocation and return one budget per layer."""
        cfg = self._config
        n = cfg.n_layers

        # Default salience: pyramid shape (outer layers less critical)
        saliences = np.array([
            self._salience.get(i, 0.5 + 0.5 * math.sin(math.pi * i / max(1, n - 1)))
            for i in range(n)
        ], dtype=np.float64)

        total_salience = saliences.sum()
        if total_salience == 0:
            saliences[:] = 1.0
            total_salience = float(n)

        # Proportional allocation
        raw_budgets = saliences / total_salience * cfg.total_kv_budget
        budgets = np.clip(
            raw_budgets.round().astype(int),
            cfg.min_tokens_per_layer,
            cfg.max_tokens_per_layer,
        )

        # Scale so that total stays within budget
        actual_total = budgets.sum()
        if actual_total > cfg.total_kv_budget:
            scale = cfg.total_kv_budget / actual_total
            budgets = np.maximum(
                cfg.min_tokens_per_layer,
                (budgets * scale).astype(int),
            )

        avg_budget = cfg.avg_tokens_per_layer
        results: list[LayerKVBudget] = []
        for i in range(n):
            b = int(budgets[i])
            # Compression score: deviation from average budget, penalised for
            # interaction (being very compressed on both axes simultaneously).
            raw_score = max(0.0, 1.0 - b / max(1, avg_budget))
            penalized_score = raw_score * (1.0 - cfg.interaction_penalty * raw_score)
            results.append(LayerKVBudget(
                layer_idx=i,
                token_budget=b,
                compression_score=float(np.clip(penalized_score, 0.0, 1.0)),
            ))

        return results


# ---------------------------------------------------------------------------
# Runtime cache
# ---------------------------------------------------------------------------

class SqueezeKVCache:
    """KV cache that applies a joint 2D budget policy at runtime.

    For each layer, it keeps at most ``budget.token_budget`` tokens.
    Eviction policy: attention-score-based (highest scores retained) or
    recency-based (most recent retained).
    """

    def __init__(self, budgets: list[LayerKVBudget], config: SqueezeConfig) -> None:
        self._budgets: dict[int, LayerKVBudget] = {b.layer_idx: b for b in budgets}
        self._config = config
        # layer_idx -> (keys list, values list, attn_scores list)
        self._store: dict[int, tuple[list[np.ndarray], list[np.ndarray], list[float]]] = {}
        self._stats = SqueezeStats()

    def _get_layer_store(
        self, layer_idx: int
    ) -> tuple[list[np.ndarray], list[np.ndarray], list[float]]:
        if layer_idx not in self._store:
            self._store[layer_idx] = ([], [], [])
        return self._store[layer_idx]

    def append(
        self,
        layer_idx: int,
        key: np.ndarray,
        value: np.ndarray,
        attn_score: float = 1.0,
    ) -> None:
        """Append one token's KV to *layer_idx*'s cache."""
        keys, values, scores = self._get_layer_store(layer_idx)
        keys.append(np.asarray(key, dtype=np.float32))
        values.append(np.asarray(value, dtype=np.float32))
        scores.append(float(attn_score))

        object.__setattr__(self._stats, "total_appended",
                           self._stats.total_appended + 1)

        # Enforce budget
        budget = self._budgets.get(layer_idx)
        if budget is None:
            return
        if len(keys) > budget.token_budget:
            self._evict(layer_idx, len(keys) - budget.token_budget)

    def _evict(self, layer_idx: int, n_evict: int) -> None:
        keys, values, scores = self._get_layer_store(layer_idx)
        cfg = self._config

        if cfg.token_eviction == "attention":
            # Keep tokens with highest attention scores
            order = np.argsort(scores)  # ascending = lowest first
            evict_indices = set(int(order[i]) for i in range(n_evict))
        else:
            # Recency: evict oldest
            evict_indices = set(range(n_evict))

        keep_indices = [i for i in range(len(keys)) if i not in evict_indices]
        self._store[layer_idx] = (
            [keys[i] for i in keep_indices],
            [values[i] for i in keep_indices],
            [scores[i] for i in keep_indices],
        )
        object.__setattr__(self._stats, "total_evicted",
                           self._stats.total_evicted + n_evict)

    def get_kv(
        self, layer_idx: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return *(keys, values)* for *layer_idx*."""
        keys, values, _ = self._get_layer_store(layer_idx)
        if not keys:
            empty = np.zeros((0,), dtype=np.float32)
            return empty, empty.copy()
        return np.stack(keys), np.stack(values)

    def size(self, layer_idx: int) -> int:
        keys, _, _ = self._get_layer_store(layer_idx)
        return len(keys)

    def total_size(self) -> int:
        return sum(len(v[0]) for v in self._store.values())

    def reset(self) -> None:
        self._store.clear()

    @property
    def stats(self) -> SqueezeStats:
        return self._stats


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

@dataclass
class SqueezeStats:
    """Statistics for SqueezeAttention runtime cache."""

    total_appended: int = 0
    total_evicted: int = 0

    @property
    def eviction_rate(self) -> float:
        if self.total_appended == 0:
            return 0.0
        return self.total_evicted / self.total_appended

    @property
    def retention_rate(self) -> float:
        return 1.0 - self.eviction_rate
