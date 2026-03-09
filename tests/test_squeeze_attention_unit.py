"""Unit tests for squish.squeeze_attention — Joint 2D KV budget management."""

import pytest
import numpy as np
from squish.squeeze_attention import (
    SqueezeConfig,
    LayerKVBudget,
    BudgetAllocator,
    SqueezeKVCache,
    SqueezeStats,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cfg(n_layers=8, total_kv_budget=1024, min_tokens_per_layer=32,
         max_tokens_per_layer=512, **kw) -> SqueezeConfig:
    return SqueezeConfig(
        n_layers=n_layers,
        total_kv_budget=total_kv_budget,
        min_tokens_per_layer=min_tokens_per_layer,
        max_tokens_per_layer=max_tokens_per_layer,
        **kw,
    )


def _allocate(n_layers=8, total=1024) -> "list[LayerKVBudget]":
    cfg = _cfg(n_layers=n_layers, total_kv_budget=total)
    alloc = BudgetAllocator(cfg)
    rng = np.random.default_rng(0)
    for i in range(n_layers):
        alloc.record_layer_salience(i, rng.uniform(0.2, 1.0))
    return alloc.allocate()


# ---------------------------------------------------------------------------
# TestSqueezeConfig
# ---------------------------------------------------------------------------

class TestSqueezeConfig:
    def test_defaults(self):
        cfg = SqueezeConfig()
        assert cfg.n_layers == 32
        assert cfg.total_kv_budget == 16384

    def test_invalid_n_layers(self):
        with pytest.raises(ValueError):
            SqueezeConfig(n_layers=0)

    def test_invalid_total_kv_budget(self):
        with pytest.raises(ValueError):
            SqueezeConfig(total_kv_budget=0)

    def test_invalid_min_tokens(self):
        with pytest.raises(ValueError):
            SqueezeConfig(min_tokens_per_layer=0)

    def test_max_less_than_min_invalid(self):
        with pytest.raises(ValueError):
            SqueezeConfig(min_tokens_per_layer=100, max_tokens_per_layer=50)

    def test_invalid_token_eviction(self):
        with pytest.raises(ValueError):
            SqueezeConfig(token_eviction="random")

    def test_invalid_interaction_penalty(self):
        with pytest.raises(ValueError):
            SqueezeConfig(interaction_penalty=-1.0)

    def test_avg_tokens_per_layer(self):
        cfg = SqueezeConfig(n_layers=8, total_kv_budget=800)
        assert cfg.avg_tokens_per_layer == 100


# ---------------------------------------------------------------------------
# TestLayerKVBudget
# ---------------------------------------------------------------------------

class TestLayerKVBudget:
    def test_is_compressed_when_positive_score(self):
        b = LayerKVBudget(layer_idx=0, token_budget=100, compression_score=0.3)
        assert b.is_compressed

    def test_not_compressed_when_zero_score(self):
        b = LayerKVBudget(layer_idx=0, token_budget=100, compression_score=0.0)
        assert not b.is_compressed


# ---------------------------------------------------------------------------
# TestBudgetAllocator
# ---------------------------------------------------------------------------

class TestBudgetAllocator:
    def test_allocate_returns_one_per_layer(self):
        budgets = _allocate(n_layers=8)
        assert len(budgets) == 8

    def test_all_within_bounds(self):
        cfg = _cfg()
        budgets = _allocate()
        for b in budgets:
            assert b.token_budget >= cfg.min_tokens_per_layer
            assert b.token_budget <= cfg.max_tokens_per_layer

    def test_total_budget_not_exceeded(self):
        cfg = _cfg(n_layers=8, total_kv_budget=1024)
        alloc = BudgetAllocator(cfg)
        for i in range(8):
            alloc.record_layer_salience(i, 0.5)
        budgets = alloc.allocate()
        total = sum(b.token_budget for b in budgets)
        assert total <= cfg.total_kv_budget + 8  # small tolerance

    def test_compression_scores_in_range(self):
        budgets = _allocate()
        for b in budgets:
            assert 0.0 <= b.compression_score <= 1.0

    def test_higher_salience_gets_more_budget(self):
        cfg = _cfg(n_layers=2, total_kv_budget=500,
                   min_tokens_per_layer=50, max_tokens_per_layer=400)
        alloc = BudgetAllocator(cfg)
        alloc.record_layer_salience(0, 0.9)   # high
        alloc.record_layer_salience(1, 0.1)   # low
        budgets = alloc.allocate()
        assert budgets[0].token_budget >= budgets[1].token_budget

    def test_without_salience_data_uses_heuristic(self):
        cfg = _cfg(n_layers=4)
        alloc = BudgetAllocator(cfg)
        budgets = alloc.allocate()
        assert len(budgets) == 4


# ---------------------------------------------------------------------------
# TestSqueezeKVCache
# ---------------------------------------------------------------------------

class TestSqueezeKVCache:
    def _make_cache(self, token_budget=10, eviction="attention") -> SqueezeKVCache:
        budgets = [LayerKVBudget(layer_idx=0, token_budget=token_budget)]
        cfg = SqueezeConfig(n_layers=1, total_kv_budget=token_budget,
                            token_eviction=eviction)
        return SqueezeKVCache(budgets=budgets, config=cfg)

    def test_append_and_retrieve(self):
        cache = self._make_cache(token_budget=20)
        k = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        v = np.array([4.0, 5.0, 6.0], dtype=np.float32)
        cache.append(0, k, v)
        rk, rv = cache.get_kv(0)
        assert rk.shape[0] == 1

    def test_budget_enforced_attention_eviction(self):
        cache = self._make_cache(token_budget=5)
        for i in range(10):
            k = np.array([float(i)])
            v = np.array([float(i)])
            cache.append(0, k, v, attn_score=float(i))
        assert cache.size(0) <= 5

    def test_budget_enforced_recency_eviction(self):
        cache = self._make_cache(token_budget=5, eviction="recency")
        for i in range(10):
            k = np.array([float(i)])
            v = np.array([float(i)])
            cache.append(0, k, v)
        assert cache.size(0) <= 5

    def test_eviction_increments_stats(self):
        cache = self._make_cache(token_budget=3)
        for i in range(6):
            cache.append(0, np.array([float(i)]), np.array([float(i)]))
        assert cache.stats.total_evicted > 0

    def test_total_size_across_layers(self):
        budgets = [
            LayerKVBudget(layer_idx=0, token_budget=10),
            LayerKVBudget(layer_idx=1, token_budget=10),
        ]
        cfg = SqueezeConfig(n_layers=2, total_kv_budget=20)
        cache = SqueezeKVCache(budgets=budgets, config=cfg)
        cache.append(0, np.ones(4), np.ones(4))
        cache.append(1, np.ones(4), np.ones(4))
        assert cache.total_size() == 2

    def test_reset_clears_all(self):
        cache = self._make_cache(token_budget=10)
        cache.append(0, np.ones(4), np.ones(4))
        cache.reset()
        assert cache.total_size() == 0

    def test_get_kv_empty_layer(self):
        cache = self._make_cache()
        k, v = cache.get_kv(99)  # non-existent layer
        assert k.shape[0] == 0


# ---------------------------------------------------------------------------
# TestSqueezeStats
# ---------------------------------------------------------------------------

class TestSqueezeStats:
    def test_eviction_rate_zero_initially(self):
        stats = SqueezeStats()
        assert stats.eviction_rate == 0.0

    def test_retention_rate_one_initially(self):
        stats = SqueezeStats()
        assert stats.retention_rate == 1.0

    def test_eviction_rate_calculation(self):
        stats = SqueezeStats(total_appended=10, total_evicted=3)
        assert stats.eviction_rate == pytest.approx(0.3)

    def test_retention_is_complement_of_eviction(self):
        stats = SqueezeStats(total_appended=10, total_evicted=3)
        assert stats.retention_rate == pytest.approx(1.0 - stats.eviction_rate)
