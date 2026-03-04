"""
tests/test_layerwise_unit.py

Unit tests for pure helpers in squish/layerwise_loader.py.
Covers LayerCache (no eviction), LoadStats, _layer_dir, recommend_cache_size.
"""
from __future__ import annotations

import pytest

from squish.layerwise_loader import (
    LayerCache,
    LoadStats,
    _layer_dir,
    recommend_cache_size,
)
from pathlib import Path


class TestLayerDir:
    def test_returns_path_with_3digit_index(self):
        root = Path("/tmp/shard")
        p = _layer_dir(root, 5)
        assert p.name == "layer_005"

    def test_joined_root(self):
        root = Path("/data/model")
        p = _layer_dir(root, 42)
        assert p.parent == root

    def test_zero_padded(self):
        root = Path("/r")
        assert _layer_dir(root, 0).name == "layer_000"
        assert _layer_dir(root, 100).name == "layer_100"


class TestLayerCachePure:
    def test_init_capacity_stored(self):
        c = LayerCache(capacity=4)
        assert c.capacity == 4

    def test_init_capacity_zero_raises(self):
        with pytest.raises(ValueError):
            LayerCache(capacity=0)

    def test_get_returns_none_when_empty(self):
        c = LayerCache(capacity=4)
        assert c.get(0) is None

    def test_put_and_get(self):
        c = LayerCache(capacity=4)
        obj = object()
        c.put(0, obj)
        assert c.get(0) is obj

    def test_contains_true(self):
        c = LayerCache(capacity=4)
        c.put(1, "layer1")
        assert 1 in c

    def test_contains_false(self):
        c = LayerCache(capacity=4)
        assert 2 not in c

    def test_len(self):
        c = LayerCache(capacity=4)
        assert len(c) == 0
        c.put(0, "a")
        assert len(c) == 1
        c.put(1, "b")
        assert len(c) == 2

    def test_cached_indices(self):
        c = LayerCache(capacity=4)
        c.put(3, "x")
        c.put(1, "y")
        indices = c.cached_indices
        assert set(indices) == {1, 3}

    def test_put_duplicate_does_not_grow(self):
        c = LayerCache(capacity=4)
        c.put(0, "a")
        c.put(0, "b")  # same key
        assert len(c) == 1

    def test_lru_ordering_after_get(self):
        """After get(), the item should move to end of LRU order."""
        c = LayerCache(capacity=4)
        c.put(0, "a")
        c.put(1, "b")
        c.get(0)  # move 0 to end
        indices = c.cached_indices
        assert indices[-1] == 0  # 0 should be most recently used


class TestLoadStats:
    def test_initial_values(self):
        s = LoadStats()
        assert s.cache_hits == 0
        assert s.cache_misses == 0
        assert s.total_loaded_bytes == 0
        assert s.total_load_time_s == 0.0
        assert s.prefetch_hits == 0

    def test_hit_rate_zero_when_empty(self):
        s = LoadStats()
        assert s.hit_rate == 0.0

    def test_hit_rate_calculation(self):
        s = LoadStats(cache_hits=3, cache_misses=1)
        assert abs(s.hit_rate - 0.75) < 1e-6

    def test_hit_rate_all_hits(self):
        s = LoadStats(cache_hits=10, cache_misses=0)
        assert abs(s.hit_rate - 1.0) < 1e-6

    def test_str_representation(self):
        s = LoadStats(cache_hits=5, cache_misses=2, total_loaded_bytes=1024)
        text = str(s)
        assert "hits=5" in text
        assert "misses=2" in text


class TestRecommendCacheSize:
    def test_basic_calculation(self):
        # 70B model, 80 layers, 16GB Metal
        size = recommend_cache_size(140.0, 80, 16.0)
        assert 2 <= size <= 80

    def test_minimum_is_2(self):
        # Very small Metal memory relative to model
        size = recommend_cache_size(1000.0, 80, 1.0)
        assert size == 2

    def test_maximum_is_n_layers(self):
        # Tiny model, huge Metal
        size = recommend_cache_size(0.1, 10, 1000.0)
        assert size == 10

    def test_safety_factor_applied(self):
        size_80 = recommend_cache_size(10.0, 10, 16.0, safety_factor=0.80)
        size_50 = recommend_cache_size(10.0, 10, 16.0, safety_factor=0.50)
        assert size_80 >= size_50
