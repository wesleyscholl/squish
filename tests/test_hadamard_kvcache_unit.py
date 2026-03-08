"""tests/test_hadamard_kvcache_unit.py — 100% coverage for HadamardKVCache."""
from __future__ import annotations

import sys

import numpy as np
import pytest

from squish.kv_cache import HadamardKVCache


N_HEADS  = 2
HEAD_DIM = 4   # power-of-two for default tests


def _rand_kv(n_heads=N_HEADS, head_dim=HEAD_DIM, seed=0):
    rng = np.random.default_rng(seed)
    k = rng.standard_normal((n_heads, head_dim)).astype(np.float16)
    v = rng.standard_normal((n_heads, head_dim)).astype(np.float16)
    return k, v


# ---------------------------------------------------------------------------
# __init__
# ---------------------------------------------------------------------------

class TestHadamardKVCacheInit:
    def test_basic_init_stores_seed(self):
        cache = HadamardKVCache(n_layers=2, window=64, mode="int8", seed=7)
        assert cache._seed == 7

    def test_h_caches_start_empty(self):
        cache = HadamardKVCache(n_layers=1, seed=42)
        assert cache._H_k == {}
        assert cache._H_v == {}


# ---------------------------------------------------------------------------
# _build_hadamard
# ---------------------------------------------------------------------------

class TestBuildHadamard:
    def test_power_of_two_shape_and_dtype(self):
        """Power-of-2 dim → Walsh–Hadamard path (lines 1123-1131)."""
        rng = np.random.default_rng(0)
        H = HadamardKVCache._build_hadamard(4, rng)
        assert H.shape == (4, 4)
        assert H.dtype == np.float16

    def test_power_of_two_8(self):
        rng = np.random.default_rng(1)
        H = HadamardKVCache._build_hadamard(8, rng)
        assert H.shape == (8, 8)
        assert H.dtype == np.float16

    def test_non_power_of_two_shape_and_dtype(self):
        """Non-power-of-2 dim → QR fallback path (lines 1132-1135)."""
        rng = np.random.default_rng(2)
        H = HadamardKVCache._build_hadamard(3, rng)
        assert H.shape == (3, 3)
        assert H.dtype == np.float16

    def test_non_power_of_two_5(self):
        rng = np.random.default_rng(3)
        H = HadamardKVCache._build_hadamard(5, rng)
        assert H.shape == (5, 5)


# ---------------------------------------------------------------------------
# _get_H_k / _get_H_v  (lazy build + cache)
# ---------------------------------------------------------------------------

class TestGetHCached:
    def test_get_H_k_first_call_builds(self):
        """First call: head_dim not in _H_k → builds and caches (lines 1141-1143)."""
        cache = HadamardKVCache(n_layers=1, seed=42)
        assert HEAD_DIM not in cache._H_k
        H = cache._get_H_k(HEAD_DIM)
        assert H.shape == (HEAD_DIM, HEAD_DIM)
        assert HEAD_DIM in cache._H_k

    def test_get_H_k_second_call_uses_cache(self):
        """Second call hits cache → branch 1141→False→1144 (line 1144 only)."""
        cache = HadamardKVCache(n_layers=1, seed=42)
        H1 = cache._get_H_k(HEAD_DIM)
        H2 = cache._get_H_k(HEAD_DIM)   # cache hit
        assert H1 is H2

    def test_get_H_v_first_call_builds(self):
        """First call: head_dim not in _H_v → builds and caches (lines 1148-1150)."""
        cache = HadamardKVCache(n_layers=1, seed=42)
        assert HEAD_DIM not in cache._H_v
        H = cache._get_H_v(HEAD_DIM)
        assert H.shape == (HEAD_DIM, HEAD_DIM)
        assert HEAD_DIM in cache._H_v

    def test_get_H_v_second_call_uses_cache(self):
        """Cache hit on _get_H_v → branch 1148→False (line 1151 only)."""
        cache = HadamardKVCache(n_layers=1, seed=42)
        H1 = cache._get_H_v(HEAD_DIM)
        H2 = cache._get_H_v(HEAD_DIM)
        assert H1 is H2


# ---------------------------------------------------------------------------
# update — rotate K/V then delegate to parent QuantizedKVCache
# ---------------------------------------------------------------------------

class TestHadamardKVCacheUpdate:
    def test_update_stores_tokens(self):
        """update() rotates K/V and delegates to parent; layer gets tokens."""
        cache = HadamardKVCache(n_layers=1, window=64, mode="int8", seed=42)
        k, v = _rand_kv()
        cache.update(0, k, v)
        assert cache._layers[0].n_tokens == 1

    def test_update_multiple_times(self):
        cache = HadamardKVCache(n_layers=1, window=64, mode="int8", seed=42)
        for i in range(3):
            k, v = _rand_kv(seed=i)
            cache.update(0, k, v)
        assert cache._layers[0].n_tokens == 3

    def test_update_builds_h_matrices(self):
        """Calling update populates _H_k and _H_v caches."""
        cache = HadamardKVCache(n_layers=1, window=64, mode="int8", seed=42)
        k, v = _rand_kv()
        assert HEAD_DIM not in cache._H_k
        cache.update(0, k, v)
        assert HEAD_DIM in cache._H_k
        assert HEAD_DIM in cache._H_v


# ---------------------------------------------------------------------------
# get_kv_mlx — un-rotate on read-back
# ---------------------------------------------------------------------------

class TestHadamardKVCacheGetKvMlx:
    def test_returns_correct_shape(self):
        """After update, get_kv_mlx returns un-rotated MLX arrays."""
        cache = HadamardKVCache(n_layers=1, window=64, mode="int8", seed=42)
        k, v = _rand_kv()
        cache.update(0, k, v)
        keys_out, vals_out = cache.get_kv_mlx(0)
        assert keys_out.shape[-1] == HEAD_DIM
        assert vals_out.shape[-1] == HEAD_DIM

    def test_output_dtype_is_bfloat16(self):
        import mlx.core as mx
        cache = HadamardKVCache(n_layers=1, window=64, mode="int8", seed=42)
        k, v = _rand_kv()
        cache.update(0, k, v)
        keys_out, vals_out = cache.get_kv_mlx(0)
        assert keys_out.dtype == mx.bfloat16
        assert vals_out.dtype == mx.bfloat16

    def test_no_mlx_raises_runtime_error(self, monkeypatch):
        """mlx.core not importable → RuntimeError (lines 1191-1192)."""
        monkeypatch.setitem(sys.modules, "mlx.core", None)
        cache = HadamardKVCache(n_layers=1, window=64, mode="int8", seed=42)
        # Must have tokens first (update uses numpy only)
        k, v = _rand_kv()
        cache.update(0, k, v)
        with pytest.raises(RuntimeError, match="mlx.core not available"):
            cache.get_kv_mlx(0)
