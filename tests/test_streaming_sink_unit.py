"""tests/test_streaming_sink_unit.py — unit tests for squish.streaming_sink"""

import numpy as np
import pytest

from squish.streaming_sink import (
    SinkConfig,
    SinkKVCache,
    SinkStats,
)

HEAD_DIM = 8
RNG = np.random.default_rng(55)


def _kv():
    return (
        RNG.standard_normal(HEAD_DIM).astype(np.float32),
        RNG.standard_normal(HEAD_DIM).astype(np.float32),
    )


# ---------------------------------------------------------------------------
# SinkConfig
# ---------------------------------------------------------------------------

class TestSinkConfig:
    def test_defaults(self):
        cfg = SinkConfig()
        assert cfg.num_sinks == 4
        assert cfg.window_size == 1024

    def test_capacity(self):
        cfg = SinkConfig(num_sinks=4, window_size=100)
        assert cfg.capacity == 104

    def test_zero_sinks_allowed(self):
        cfg = SinkConfig(num_sinks=0, window_size=512)
        assert cfg.capacity == 512

    @pytest.mark.parametrize("field,val", [
        ("num_sinks", -1),
        ("window_size", 0),
        ("head_dim", 0),
    ])
    def test_invalid(self, field, val):
        with pytest.raises(ValueError):
            SinkConfig(**{field: val})


# ---------------------------------------------------------------------------
# SinkKVCache
# ---------------------------------------------------------------------------

class TestSinkKVCache:
    def _cache(self, num_sinks=2, window_size=5):
        cfg = SinkConfig(num_sinks=num_sinks, window_size=window_size, head_dim=HEAD_DIM)
        return SinkKVCache(cfg)

    def test_empty_cache(self):
        cache = self._cache()
        keys, vals, pos = cache.get_kv()
        assert keys.shape[0] == 0

    def test_size_increments(self):
        cache = self._cache()
        for i in range(4):
            cache.append(*_kv())
        assert cache.size == 4

    def test_sinks_pinned(self):
        cache = self._cache(num_sinks=2, window_size=3)
        for i in range(5):
            cache.append(*_kv())
        assert cache.n_sinks == 2

    def test_window_bounded(self):
        cache = self._cache(num_sinks=2, window_size=3)
        for i in range(20):
            cache.append(*_kv())
        assert cache.n_window <= 3

    def test_total_size_bounded_by_capacity(self):
        cfg = SinkConfig(num_sinks=2, window_size=3, head_dim=HEAD_DIM)
        cache = SinkKVCache(cfg)
        for i in range(100):
            cache.append(*_kv())
        assert cache.size <= cfg.capacity

    def test_get_kv_shape(self):
        cache = self._cache(num_sinks=2, window_size=4)
        for _ in range(8):
            cache.append(*_kv())
        keys, vals, pos = cache.get_kv()
        assert keys.shape[1] == HEAD_DIM
        assert vals.shape == keys.shape
        assert pos.shape == (keys.shape[0],)

    def test_sinks_come_first_in_positions(self):
        """Sink positions (first tokens) should be lowest in positions array."""
        cache = self._cache(num_sinks=2, window_size=5)
        for _ in range(8):
            cache.append(*_kv())
        _, _, pos = cache.get_kv()
        assert pos[0] < pos[-1]

    def test_reset(self):
        cache = self._cache()
        for _ in range(10):
            cache.append(*_kv())
        cache.reset()
        assert cache.size == 0
        assert cache.stats.total_appended == 0

    def test_is_full(self):
        cache = self._cache(num_sinks=2, window_size=3)
        for _ in range(5):
            cache.append(*_kv())
        assert cache.is_full

    def test_not_full_before_capacity(self):
        cache = self._cache(num_sinks=2, window_size=3)
        cache.append(*_kv())
        assert not cache.is_full

    def test_evictions_tracked(self):
        cache = self._cache(num_sinks=2, window_size=3)
        for _ in range(10):
            cache.append(*_kv())
        assert cache.stats.evictions > 0

    def test_sink_preserved_counted(self):
        cache = self._cache(num_sinks=2, window_size=5)
        for _ in range(8):
            cache.append(*_kv())
        assert cache.stats.sink_preserved == 2

    def test_no_sinks_mode(self):
        cfg = SinkConfig(num_sinks=0, window_size=4, head_dim=HEAD_DIM)
        cache = SinkKVCache(cfg)
        for _ in range(6):
            cache.append(*_kv())
        assert cache.n_sinks == 0
        assert cache.n_window <= 4


# ---------------------------------------------------------------------------
# SinkStats
# ---------------------------------------------------------------------------

class TestSinkStats:
    def test_defaults(self):
        s = SinkStats()
        assert s.total_appended == 0
        assert s.evictions == 0
        assert s.eviction_rate == 0.0
        assert s.sink_preservation_rate == 0.0

    def test_eviction_rate(self):
        s = SinkStats(total_appended=10, sink_preserved=2, evictions=4)
        rate = s.eviction_rate
        assert 0.0 <= rate <= 1.0

    def test_sink_preservation_rate(self):
        s = SinkStats(total_appended=8, sink_preserved=2)
        assert abs(s.sink_preservation_rate - 0.25) < 1e-6

    def test_window_tokens_total(self):
        s = SinkStats(total_appended=10, sink_preserved=2)
        assert s.window_tokens_total == 8
