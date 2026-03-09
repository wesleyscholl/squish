"""Unit tests for squish.svdq — SVD per-head key cache mixed precision."""

import pytest
import numpy as np
from squish.svdq import (
    SVDqConfig,
    HeadSVDProfile,
    SVDqCalibrator,
    SVDqPrecisionMap,
    SVDqStats,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config(**kw) -> SVDqConfig:
    return SVDqConfig(n_layers=4, n_heads=4, head_dim=16,
                      candidate_bits=(2, 4, 8), target_avg_bits=4.0, **kw)


def _make_calibrated(n_layers=4, n_heads=4, head_dim=16) -> SVDqPrecisionMap:
    cfg = SVDqConfig(n_layers=n_layers, n_heads=n_heads, head_dim=head_dim,
                     candidate_bits=(2, 4, 8), target_avg_bits=4.0)
    cal = SVDqCalibrator(cfg)
    rng = np.random.default_rng(7)
    for li in range(n_layers):
        for hi in range(n_heads):
            km = rng.standard_normal((8, head_dim)).astype(np.float32)
            cal.record_head_keys(li, hi, km)
    return cal.search()


# ---------------------------------------------------------------------------
# TestSVDqConfig
# ---------------------------------------------------------------------------

class TestSVDqConfig:
    def test_defaults(self):
        cfg = SVDqConfig()
        assert cfg.n_layers == 32
        assert cfg.energy_threshold == 0.95

    def test_invalid_n_heads(self):
        with pytest.raises(ValueError):
            SVDqConfig(n_heads=0)

    def test_invalid_energy_threshold(self):
        with pytest.raises(ValueError):
            SVDqConfig(energy_threshold=0.0)

    def test_invalid_min_rank(self):
        with pytest.raises(ValueError):
            SVDqConfig(min_rank=0)

    def test_invalid_target_avg_bits(self):
        with pytest.raises(ValueError):
            SVDqConfig(target_avg_bits=-1.0)


# ---------------------------------------------------------------------------
# TestHeadSVDProfile
# ---------------------------------------------------------------------------

class TestHeadSVDProfile:
    def _make_profile(self, n_svs=16):
        svs = np.linspace(10.0, 0.1, n_svs)
        return HeadSVDProfile(layer_idx=0, head_idx=0, singular_values=svs)

    def test_total_energy_positive(self):
        p = self._make_profile()
        assert p.total_energy > 0.0

    def test_effective_rank_leq_total(self):
        p = self._make_profile()
        assert p.effective_rank(0.95) <= len(p.singular_values)

    def test_effective_rank_at_threshold_1(self):
        p = self._make_profile()
        # At threshold=1.0, all dims needed
        assert p.effective_rank(1.0) == len(p.singular_values)

    def test_compressibility_range(self):
        p = self._make_profile()
        c = p.compressibility(0.95)
        assert 0.0 <= c <= 1.0

    def test_uniform_svs_zero_compressibility(self):
        svs = np.ones(16)
        p = HeadSVDProfile(layer_idx=0, head_idx=0, singular_values=svs)
        # Uniform singular values: first one captures 1/16 = 6.25% energy,
        # so effective rank ≈ 15 at 0.95 → compressibility close to 0
        c = p.compressibility(0.95)
        assert c <= 0.15  # nearly incompressible

    def test_zero_energy_effective_rank_is_one(self):
        svs = np.zeros(8)
        p = HeadSVDProfile(layer_idx=0, head_idx=0, singular_values=svs)
        assert p.effective_rank(0.95) == 1


# ---------------------------------------------------------------------------
# TestSVDqCalibrator
# ---------------------------------------------------------------------------

class TestSVDqCalibrator:
    def test_search_returns_precision_map(self):
        pm = _make_calibrated()
        assert isinstance(pm, SVDqPrecisionMap)

    def test_all_heads_have_bits(self):
        pm = _make_calibrated(n_layers=4, n_heads=4)
        for li in range(4):
            for hi in range(4):
                b = pm.bits_for_head(li, hi)
                assert b in (2, 4, 8)

    def test_all_heads_have_rank(self):
        pm = _make_calibrated(n_layers=4, n_heads=4, head_dim=16)
        for li in range(4):
            for hi in range(4):
                r = pm.rank_for_head(li, hi)
                assert 1 <= r <= 16

    def test_search_without_data_uses_heuristic(self):
        cfg = _make_config()
        cal = SVDqCalibrator(cfg)
        pm = cal.search()
        assert isinstance(pm, SVDqPrecisionMap)

    def test_min_rank_respected(self):
        cfg = SVDqConfig(n_layers=2, n_heads=2, head_dim=16,
                         candidate_bits=(2, 4), target_avg_bits=3.0, min_rank=8)
        cal = SVDqCalibrator(cfg)
        pm = cal.search()
        for key in pm.rank_map:
            assert pm.rank_map[key] >= 8


# ---------------------------------------------------------------------------
# TestSVDqPrecisionMap
# ---------------------------------------------------------------------------

class TestSVDqPrecisionMap:
    def test_avg_bits_in_range(self):
        pm = _make_calibrated()
        assert 1.0 <= pm.avg_bits <= 16.0

    def test_avg_rank_positive(self):
        pm = _make_calibrated()
        assert pm.avg_rank > 0

    def test_rank_compression_ratio_in_range(self):
        pm = _make_calibrated()
        r = pm.rank_compression_ratio()
        assert 0.0 <= r <= 1.0

    def test_default_bits_fallback(self):
        cfg = _make_config()
        pm = SVDqPrecisionMap(bits_map={}, rank_map={}, profiles={}, config=cfg)
        assert pm.bits_for_head(99, 99) == 4

    def test_default_rank_fallback(self):
        cfg = _make_config()
        pm = SVDqPrecisionMap(bits_map={}, rank_map={}, profiles={}, config=cfg)
        assert pm.rank_for_head(99, 99) == 16


# ---------------------------------------------------------------------------
# TestSVDqStats
# ---------------------------------------------------------------------------

class TestSVDqStats:
    def test_combined_compression_ratio_less_than_one(self):
        pm = _make_calibrated()
        stats = SVDqStats(precision_map=pm)
        r = stats.combined_compression_ratio()
        assert 0.0 < r < 1.0

    def test_avg_bits_matches_map(self):
        pm = _make_calibrated()
        stats = SVDqStats(precision_map=pm)
        assert stats.avg_bits == pm.avg_bits

    def test_rank_compression_positive(self):
        pm = _make_calibrated()
        stats = SVDqStats(precision_map=pm)
        assert 0.0 <= stats.rank_compression_ratio <= 1.0
