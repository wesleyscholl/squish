"""Unit tests for squish.kvsharer — KVSharer cross-layer KV cache sharing."""

import pytest
import numpy as np
from squish.kvsharer import (
    KVSharerConfig,
    KVSharerCalibrator,
    KVShareMap,
    KVLayerCache,
    KVSharerStats,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config(n_layers=8, **kw) -> KVSharerConfig:
    return KVSharerConfig(n_layers=n_layers, **kw)


def _calibrated_map(n_layers=8, prefer_dissimilar=True) -> KVShareMap:
    cfg = KVSharerConfig(n_layers=n_layers, prefer_dissimilar=prefer_dissimilar,
                         max_share_fraction=0.40)
    cal = KVSharerCalibrator(cfg)
    rng = np.random.default_rng(42)
    for li in range(n_layers):
        keys = rng.standard_normal((4, 16)).astype(np.float32)
        vals = rng.standard_normal((4, 16)).astype(np.float32)
        cal.record_layer_kv(li, keys, vals)
    return cal.compute_share_map()


# ---------------------------------------------------------------------------
# TestKVSharerConfig
# ---------------------------------------------------------------------------

class TestKVSharerConfig:
    def test_defaults(self):
        cfg = KVSharerConfig()
        assert cfg.n_layers == 32
        assert cfg.max_share_fraction == 0.30
        assert cfg.prefer_dissimilar is True

    def test_invalid_similarity_threshold(self):
        with pytest.raises(ValueError):
            KVSharerConfig(similarity_threshold=1.5)

    def test_invalid_share_fraction(self):
        with pytest.raises(ValueError):
            KVSharerConfig(max_share_fraction=0.0)

    def test_invalid_share_fraction_high(self):
        with pytest.raises(ValueError):
            KVSharerConfig(max_share_fraction=1.1)

    def test_invalid_n_layers(self):
        with pytest.raises(ValueError):
            KVSharerConfig(n_layers=1)


# ---------------------------------------------------------------------------
# TestKVSharerCalibrator
# ---------------------------------------------------------------------------

class TestKVSharerCalibrator:
    def test_record_stores_data(self):
        cfg = _make_config()
        cal = KVSharerCalibrator(cfg)
        cal.record_layer_kv(0, np.ones((4, 8)), np.ones((4, 8)))
        assert 0 in cal._kv_means

    def test_compute_share_map_returns_kvshare_map(self):
        share_map = _calibrated_map()
        assert isinstance(share_map, KVShareMap)

    def test_share_fraction_within_limit(self):
        share_map = _calibrated_map(n_layers=8)
        assert share_map.share_fraction <= 0.41  # small tolerance

    def test_no_self_sharing(self):
        share_map = _calibrated_map()
        for recipient, donor in share_map.share_map.items():
            assert recipient != donor

    def test_donors_not_recipients(self):
        share_map = _calibrated_map()
        for recipient in share_map.share_map:
            assert recipient not in share_map.donor_recipients

    def test_calibrate_without_data_uses_heuristic(self):
        cfg = KVSharerConfig(n_layers=4, max_share_fraction=0.4)
        cal = KVSharerCalibrator(cfg)
        share_map = cal.compute_share_map()
        # No data → should still produce a valid (possibly empty) map
        assert isinstance(share_map, KVShareMap)

    def test_prefer_similar_opposite_ordering(self):
        m_dissimilar = _calibrated_map(prefer_dissimilar=True)
        m_similar = _calibrated_map(prefer_dissimilar=False)
        # Both are valid share maps; results may differ
        assert isinstance(m_dissimilar, KVShareMap)
        assert isinstance(m_similar, KVShareMap)


# ---------------------------------------------------------------------------
# TestKVShareMap
# ---------------------------------------------------------------------------

class TestKVShareMap:
    def test_n_shared_counts_recipients(self):
        share_map = _calibrated_map()
        assert share_map.n_shared == len(share_map.share_map)

    def test_donor_for_returns_self_when_not_shared(self):
        share_map = _calibrated_map()
        for li in range(8):
            if not share_map.is_shared(li):
                assert share_map.donor_for(li) == li

    def test_donor_for_redirects_recipients(self):
        share_map = _calibrated_map()
        for recipient, donor in share_map.share_map.items():
            assert share_map.donor_for(recipient) == donor

    def test_recipient_layers_sorted(self):
        share_map = _calibrated_map()
        rl = share_map.recipient_layers
        assert rl == sorted(rl)

    def test_summary_is_string(self):
        share_map = _calibrated_map()
        s = share_map.summary()
        assert isinstance(s, str)
        assert "KVShareMap" in s

    def test_kv_ops_saved_fraction_positive(self):
        share_map = _calibrated_map()
        assert share_map.kv_ops_saved_fraction() >= 0.0


# ---------------------------------------------------------------------------
# TestKVLayerCache
# ---------------------------------------------------------------------------

class TestKVLayerCache:
    def test_store_and_retrieve_non_shared(self):
        share_map = KVShareMap(share_map={}, donor_recipients={},
                               n_layers=4, config=_make_config(n_layers=4))
        cache = KVLayerCache(share_map)
        k = np.ones((4, 8), dtype=np.float32)
        v = np.ones((4, 8), dtype=np.float32) * 2
        cache.store(0, k, v)
        rk, rv = cache.retrieve(0)
        np.testing.assert_array_equal(k, rk)
        np.testing.assert_array_equal(v, rv)

    def test_recipient_reads_donor_kv(self):
        share_map = KVShareMap(
            share_map={2: 0},
            donor_recipients={0: [2]},
            n_layers=4,
            config=_make_config(n_layers=4),
        )
        cache = KVLayerCache(share_map)
        k = np.arange(4, dtype=np.float32)
        v = np.arange(4, dtype=np.float32) + 10
        cache.store(0, k, v)  # store at donor
        rk, rv = cache.retrieve(2)  # read as recipient
        np.testing.assert_array_equal(k, rk)
        np.testing.assert_array_equal(v, rv)

    def test_retrieve_nonexistent_returns_none(self):
        share_map = KVShareMap(share_map={}, donor_recipients={},
                               n_layers=4, config=_make_config(n_layers=4))
        cache = KVLayerCache(share_map)
        assert cache.retrieve(3) is None

    def test_reset_clears_store(self):
        share_map = KVShareMap(share_map={}, donor_recipients={},
                               n_layers=4, config=_make_config(n_layers=4))
        cache = KVLayerCache(share_map)
        cache.store(0, np.ones(4), np.ones(4))
        cache.reset()
        assert cache.retrieve(0) is None

    def test_redirect_increments_stats(self):
        share_map = KVShareMap(
            share_map={1: 0},
            donor_recipients={0: [1]},
            n_layers=2,
            config=_make_config(n_layers=2),
        )
        cache = KVLayerCache(share_map)
        cache.store(1, np.ones(4), np.ones(4))  # recipient write
        assert cache.stats.redirect_writes == 1
        cache.store(0, np.ones(4), np.ones(4))
        cache.retrieve(1)  # recipient read
        assert cache.stats.redirect_reads == 1

    def test_n_cached_layers(self):
        share_map = KVShareMap(share_map={}, donor_recipients={},
                               n_layers=4, config=_make_config(n_layers=4))
        cache = KVLayerCache(share_map)
        cache.store(0, np.ones(4), np.ones(4))
        cache.store(1, np.ones(4), np.ones(4))
        assert cache.n_cached_layers == 2


# ---------------------------------------------------------------------------
# TestKVSharerStats
# ---------------------------------------------------------------------------

class TestKVSharerStats:
    def test_defaults_zero(self):
        stats = KVSharerStats()
        assert stats.redirect_writes == 0
        assert stats.redirect_reads == 0

    def test_total_redirects(self):
        stats = KVSharerStats(redirect_writes=3, redirect_reads=5)
        assert stats.total_redirects == 8

    def test_estimated_compute_savings_range(self):
        stats = KVSharerStats(redirect_writes=10, redirect_reads=10)
        s = stats.estimated_compute_savings
        assert 0.0 <= s <= 1.0
