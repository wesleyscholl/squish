"""Unit tests for squish.yoco — You Only Cache Once."""

import numpy as np
import pytest

from squish.yoco import (
    YOCOConfig,
    YOCOKVStore,
    YOCOLayerSpec,
    YOCOSchedule,
    YOCOStats,
)

# ---------------------------------------------------------------------------
# TestYOCOConfig
# ---------------------------------------------------------------------------

class TestYOCOConfig:
    def test_defaults(self):
        cfg = YOCOConfig()
        assert cfg.n_layers == 32
        assert cfg.n_self_attn_layers == 16

    def test_n_cross_attn_layers(self):
        cfg = YOCOConfig(n_layers=32, n_self_attn_layers=16)
        assert cfg.n_cross_attn_layers == 16

    def test_invalid_n_layers(self):
        with pytest.raises(ValueError):
            YOCOConfig(n_layers=1)

    def test_invalid_self_attn_layers_zero(self):
        with pytest.raises(ValueError):
            YOCOConfig(n_layers=8, n_self_attn_layers=0)

    def test_invalid_self_attn_equals_n_layers(self):
        with pytest.raises(ValueError):
            YOCOConfig(n_layers=8, n_self_attn_layers=8)

    def test_invalid_head_dim(self):
        with pytest.raises(ValueError):
            YOCOConfig(head_dim=0)

    def test_invalid_n_kv_heads(self):
        with pytest.raises(ValueError):
            YOCOConfig(n_kv_heads=0)


# ---------------------------------------------------------------------------
# TestYOCOSchedule
# ---------------------------------------------------------------------------

class TestYOCOSchedule:
    def test_from_config_basic(self):
        cfg = YOCOConfig(n_layers=8, n_self_attn_layers=4)
        sched = YOCOSchedule.from_config(cfg)
        assert len(sched.specs) == 8
        for i in range(4):
            assert sched.specs[i].is_self_attn
        for i in range(4, 8):
            assert sched.specs[i].is_cross_attn

    def test_self_and_cross_are_mutually_exclusive(self):
        cfg = YOCOConfig(n_layers=8, n_self_attn_layers=4)
        sched = YOCOSchedule.from_config(cfg)
        for spec in sched.specs:
            assert spec.is_self_attn != spec.is_cross_attn

    def test_kv_cache_reduction_factor_half(self):
        cfg = YOCOConfig(n_layers=8, n_self_attn_layers=4)
        sched = YOCOSchedule.from_config(cfg)
        assert sched.kv_cache_reduction_factor() == pytest.approx(0.5)

    def test_summary_string(self):
        cfg = YOCOConfig(n_layers=8, n_self_attn_layers=4)
        sched = YOCOSchedule.from_config(cfg)
        s = sched.summary()
        assert "YOCO" in s

    def test_spec_for(self):
        cfg = YOCOConfig(n_layers=4, n_self_attn_layers=2)
        sched = YOCOSchedule.from_config(cfg)
        assert sched.spec_for(0).is_self_attn
        assert sched.spec_for(2).is_cross_attn

    def test_self_attn_layers_list(self):
        cfg = YOCOConfig(n_layers=8, n_self_attn_layers=3)
        sched = YOCOSchedule.from_config(cfg)
        assert sched.self_attn_layers == [0, 1, 2]

    def test_cross_attn_layers_list(self):
        cfg = YOCOConfig(n_layers=8, n_self_attn_layers=3)
        sched = YOCOSchedule.from_config(cfg)
        assert sched.cross_attn_layers == [3, 4, 5, 6, 7]


# ---------------------------------------------------------------------------
# TestYOCOLayerSpec
# ---------------------------------------------------------------------------

class TestYOCOLayerSpec:
    def test_self_attn_repr(self):
        spec = YOCOLayerSpec(layer_idx=0, role="self")
        assert "SELF" in repr(spec)

    def test_cross_attn_repr(self):
        spec = YOCOLayerSpec(layer_idx=4, role="cross")
        assert "CROSS" in repr(spec)


# ---------------------------------------------------------------------------
# TestYOCOKVStore
# ---------------------------------------------------------------------------

class TestYOCOKVStore:
    def _cfg(self):
        return YOCOConfig(n_layers=8, n_self_attn_layers=4,
                          n_kv_heads=2, head_dim=8)

    def test_initial_empty(self):
        store = YOCOKVStore(self._cfg())
        assert store.size == 0
        assert store.is_empty

    def test_append_increments_size(self):
        store = YOCOKVStore(self._cfg())
        k = np.ones((2, 8), dtype=np.float32)
        v = np.ones((2, 8), dtype=np.float32)
        store.append(k, v)
        assert store.size == 1
        assert not store.is_empty

    def test_get_shared_kv_shape(self):
        cfg = self._cfg()
        store = YOCOKVStore(cfg)
        for _ in range(5):
            k = np.ones((cfg.n_kv_heads, cfg.head_dim), dtype=np.float32)
            v = np.ones((cfg.n_kv_heads, cfg.head_dim), dtype=np.float32)
            store.append(k, v)
        keys, values = store.get_shared_kv()
        assert keys.shape[0] == 5

    def test_get_kv_empty_returns_zero_array(self):
        store = YOCOKVStore(self._cfg())
        k, v = store.get_shared_kv()
        assert k.shape[0] == 0

    def test_reset_clears_store(self):
        store = YOCOKVStore(self._cfg())
        k = np.ones((2, 8))
        store.append(k, k)
        store.reset()
        assert store.size == 0

    def test_stats_track_writes(self):
        store = YOCOKVStore(self._cfg())
        k = np.ones((2, 8))
        store.append(k, k)
        store.append(k, k)
        assert store.stats.tokens_written == 2

    def test_stats_track_reads(self):
        store = YOCOKVStore(self._cfg())
        k = np.ones((2, 8))
        store.append(k, k)
        store.get_shared_kv()
        store.get_shared_kv()
        assert store.stats.reads == 2


# ---------------------------------------------------------------------------
# TestYOCOStats
# ---------------------------------------------------------------------------

class TestYOCOStats:
    def test_kv_memory_bytes(self):
        stats = YOCOStats(tokens_written=100)
        mem = stats.kv_memory_bytes(n_kv_heads=8, head_dim=128, dtype_bytes=2)
        assert mem == 2 * 100 * 8 * 128 * 2

    def test_standard_kv_bytes_larger(self):
        stats = YOCOStats(tokens_written=100)
        yoco = stats.kv_memory_bytes(8, 128)
        std = stats.standard_kv_memory_bytes(32, 8, 128)
        assert std > yoco

    def test_reduction_ratio_positive(self):
        stats = YOCOStats(tokens_written=100)
        r = stats.kv_memory_reduction_ratio(32, 8, 128)
        assert 0.0 < r < 1.0

    def test_reduction_ratio_zero_tokens(self):
        stats = YOCOStats(tokens_written=0)
        r = stats.kv_memory_reduction_ratio(32, 8, 128)
        assert r == 0.0
