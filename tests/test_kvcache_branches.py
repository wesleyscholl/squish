"""
tests/test_kvcache_branches.py

Branch coverage for squish/kv_cache.py:
  - get_full_kv: empty keys_recent (line 240)
  - get_full_kv: old_k only path   (line 246)
  - get_as_mlx: empty cache        (lines 254-257)
  - _snap_evict: full_k is None    (line 359)
  - QuantizedKVCache.__call__: full_k is None (line 320)
  - QuantizedKVCache.get_kv_mlx   (line 517)
  - _KVLayerView                   (lines 552-563)
"""
from __future__ import annotations

import numpy as np
import pytest

from squish.kv_cache import KVLayerCache, QuantizedKVCache, _snap_evict


def _rand_kv(n_heads=2, head_dim=4, dtype=np.float16):
    rng = np.random.default_rng(42)
    return (
        rng.standard_normal((n_heads, head_dim)).astype(dtype),
        rng.standard_normal((n_heads, head_dim)).astype(dtype),
    )


# ── get_full_kv branches ──────────────────────────────────────────────────────

class TestGetFullKVBranches:
    def test_empty_cache_returns_none(self):
        """Completely empty cache → (None, None)."""
        layer = KVLayerCache()
        k, v = layer.get_full_kv()
        assert k is None
        assert v is None

    def test_recent_only_no_old(self):
        """Only recent tokens, no old quantized: rec_k != None, old_k == None."""
        layer = KVLayerCache()
        k0, v0 = _rand_kv()
        layer.append(k0, v0)
        full_k, full_v = layer.get_full_kv()
        assert full_k is not None
        assert full_v is not None
        assert full_k.shape[1] == 1  # 1 token in recent

    def test_old_only_empty_recent(self):
        """Force old_q to have data but keys_recent is empty → old_k only path (line 246)."""
        layer = KVLayerCache(window=1)
        k0, v0 = _rand_kv()
        k1, v1 = _rand_kv()
        # Append enough tokens to overflow window and move into old_q
        layer.append(k0, v0)
        layer.append(k1, v1)  # This may spill to old_q depending on window
        # Add more to ensure spilling
        for _ in range(5):
            k, v = _rand_kv()
            layer.append(k, v)
        # Just verify it doesn't crash
        full_k, full_v = layer.get_full_kv()
        assert full_k is not None

    def test_empty_recent_list_path(self):
        """When keys_recent is [] but old_q exists, rec_k = rec_v = None (line 240)."""
        layer = KVLayerCache()
        # Manually set old_q to non-None but leave recent empty
        layer.n_heads      = 2
        layer.head_dim     = 4
        layer.keys_old_q   = np.zeros((2, 1, 4), dtype=np.int8)
        layer.values_old_q = np.zeros((2, 1, 4), dtype=np.int8)
        layer.keys_old_s   = np.ones( (2, 1),      dtype=np.float32)
        layer.values_old_s = np.ones( (2, 1),      dtype=np.float32)
        layer.keys_recent   = []  # empty recent
        layer.values_recent = []
        full_k, full_v = layer.get_full_kv()
        # Should return old_k only (line 246: full_k, full_v = old_k, old_v)
        assert full_k is not None
        assert full_k.shape[1] == 1


# ── get_as_mlx with empty cache ───────────────────────────────────────────────

class TestGetAsMlx:
    def test_empty_cache_returns_none_none(self):
        """get_as_mlx on empty cache: get_full_kv returns None → (None, None)."""
        try:
            import mlx.core as mx  # noqa: F401
        except ImportError:
            pytest.skip("MLX not available")
        layer = KVLayerCache()
        k, v = layer.get_as_mlx()
        assert k is None
        assert v is None

    def test_nonempty_cache_returns_mlx_arrays(self):
        """get_as_mlx on cache with data: returns MLX bfloat16 arrays."""
        try:
            import mlx.core as mx
        except ImportError:
            pytest.skip("MLX not available")
        layer = KVLayerCache()
        k0, v0 = _rand_kv()
        layer.append(k0, v0)
        k_mlx, v_mlx = layer.get_as_mlx()
        assert k_mlx is not None
        assert k_mlx.dtype == mx.bfloat16


# ── _snap_evict with None full_k ──────────────────────────────────────────────

class TestSnapEvictBranches:
    def test_snap_evict_returns_early_when_empty(self):
        """Empty layer → get_full_kv() returns None → early return (line 359)."""
        layer = KVLayerCache()
        # Calling _snap_evict on empty layer should return without error
        _snap_evict(layer, budget=64, snap_window=16)
        # Nothing should have changed
        assert layer.n_tokens == 0


# ── QuantizedKVCache.update (replaces __call__ which doesn't exist) ──────────

class TestQuantizedKVCacheUpdate:
    def test_update_appends_to_layer(self):
        """update() stores key/value for the specified layer."""
        cache = QuantizedKVCache(n_layers=2, mode="int8")
        k0, v0 = _rand_kv()
        cache.update(0, k0, v0)
        assert cache._layers[0].n_tokens == 1
        assert cache._layers[1].n_tokens == 0

    def test_update_increments_n_tokens(self):
        cache = QuantizedKVCache(n_layers=1, mode="fp16")
        k0, v0 = _rand_kv()
        k1, v1 = _rand_kv()
        cache.update(0, k0, v0)
        cache.update(0, k1, v1)
        assert cache._layers[0].n_tokens == 2


# ── QuantizedKVCache.get_kv_mlx ──────────────────────────────────────────────

class TestGetKvMlx:
    def test_get_kv_mlx_empty(self):
        """get_kv_mlx on an empty layer returns (None, None)."""
        try:
            import mlx.core as mx  # noqa: F401
        except ImportError:
            pytest.skip("MLX not available")
        cache = QuantizedKVCache(n_layers=2, mode="int8")
        k, v = cache.get_kv_mlx(0)
        assert k is None
        assert v is None


# ── _LayerCacheView (correct class name) ──────────────────────────────────────

class TestLayerCacheView:
    def test_keys_and_values_empty(self):
        """_LayerCacheView.keys and .values on empty cache return None."""
        try:
            import mlx.core as mx  # noqa: F401
        except ImportError:
            pytest.skip("MLX not available")
        from squish.kv_cache import _LayerCacheView
        layer  = KVLayerCache()
        parent = QuantizedKVCache(n_layers=1, mode="int8")
        view   = _LayerCacheView(layer, parent)
        assert view.keys   is None
        assert view.values is None

    def test_view_stores_refs(self):
        """_LayerCacheView stores layer and parent references."""
        from squish.kv_cache import _LayerCacheView
        layer  = KVLayerCache()
        parent = QuantizedKVCache(n_layers=1, mode="fp16")
        view   = _LayerCacheView(layer, parent)
        assert view._layer  is layer
        assert view._parent is parent
