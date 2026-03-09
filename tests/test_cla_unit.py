"""Unit tests for squish.cla — Cross-Layer Attention architecture."""

import pytest
from squish.cla import (
    CLAConfig,
    CLALayerSpec,
    CLASchedule,
    CLAStats,
)


# ---------------------------------------------------------------------------
# TestCLAConfig
# ---------------------------------------------------------------------------

class TestCLAConfig:
    def test_defaults(self):
        cfg = CLAConfig()
        assert cfg.n_layers == 32
        assert cfg.sharing_factor == 2

    def test_invalid_n_layers(self):
        with pytest.raises(ValueError):
            CLAConfig(n_layers=1)

    def test_invalid_sharing_factor(self):
        with pytest.raises(ValueError):
            CLAConfig(sharing_factor=0)

    def test_invalid_generator_stride(self):
        with pytest.raises(ValueError):
            CLAConfig(sharing_factor=2, generator_stride=2)

    def test_n_cross_attn_layers_property(self):
        cfg = CLAConfig(n_layers=16, sharing_factor=2)
        # n_self_attn_layers is not a property of CLAConfig, test schedule property
        cfg2 = CLAConfig(n_layers=16, sharing_factor=2)
        schedule = CLASchedule.from_config(cfg2)
        assert schedule.n_generators + schedule.n_borrowers == 16


# ---------------------------------------------------------------------------
# TestCLASchedule
# ---------------------------------------------------------------------------

class TestCLASchedule:
    def test_from_config_basic_sf2(self):
        cfg = CLAConfig(n_layers=8, sharing_factor=2)
        sched = CLASchedule.from_config(cfg)
        assert len(sched.specs) == 8
        # With sf=2 and gs=0: layers 0,2,4,6 are generators
        assert sched.specs[0].is_generator
        assert sched.specs[1].is_borrower
        assert sched.specs[2].is_generator
        assert sched.specs[3].is_borrower

    def test_first_layer_always_generator(self):
        cfg = CLAConfig(n_layers=6, sharing_factor=3, generator_stride=1,
                        allow_first_layer_borrow=False)
        sched = CLASchedule.from_config(cfg)
        assert sched.specs[0].is_generator

    def test_borrower_has_donor(self):
        cfg = CLAConfig(n_layers=8, sharing_factor=2)
        sched = CLASchedule.from_config(cfg)
        for spec in sched.specs:
            if spec.is_borrower:
                assert spec.borrows_from is not None
                assert spec.borrows_from != spec.layer_idx

    def test_generator_no_donor(self):
        cfg = CLAConfig(n_layers=8, sharing_factor=2)
        sched = CLASchedule.from_config(cfg)
        for spec in sched.specs:
            if spec.is_generator:
                assert spec.borrows_from is None

    def test_kv_cache_reduction_factor_sf2(self):
        cfg = CLAConfig(n_layers=8, sharing_factor=2)
        sched = CLASchedule.from_config(cfg)
        # 4 generators out of 8 layers → 0.5
        assert abs(sched.kv_cache_reduction_factor() - 0.5) < 0.1

    def test_summary_is_string(self):
        cfg = CLAConfig(n_layers=8, sharing_factor=2)
        sched = CLASchedule.from_config(cfg)
        s = sched.summary()
        assert isinstance(s, str)
        assert "CLASchedule" in s

    def test_spec_for(self):
        cfg = CLAConfig(n_layers=4, sharing_factor=2)
        sched = CLASchedule.from_config(cfg)
        assert sched.spec_for(0).layer_idx == 0

    def test_generator_and_borrower_lists_partition_layers(self):
        cfg = CLAConfig(n_layers=8, sharing_factor=2)
        sched = CLASchedule.from_config(cfg)
        all_layers = set(range(8))
        gen = set(sched.generator_layers)
        bor = set(sched.borrower_layers)
        assert gen | bor == all_layers
        assert gen & bor == set()


# ---------------------------------------------------------------------------
# TestCLALayerSpec
# ---------------------------------------------------------------------------

class TestCLALayerSpec:
    def test_is_generator_mutually_exclusive(self):
        gen = CLALayerSpec(layer_idx=0, is_generator=True, borrows_from=None)
        bor = CLALayerSpec(layer_idx=1, is_generator=False, borrows_from=0)
        assert gen.is_generator and not gen.is_borrower
        assert bor.is_borrower and not bor.is_generator

    def test_repr_generator(self):
        spec = CLALayerSpec(layer_idx=0, is_generator=True, borrows_from=None)
        assert "GENERATOR" in repr(spec)

    def test_repr_borrower(self):
        spec = CLALayerSpec(layer_idx=1, is_generator=False, borrows_from=0)
        assert "BORROWER" in repr(spec)


# ---------------------------------------------------------------------------
# TestCLAStats
# ---------------------------------------------------------------------------

class TestCLAStats:
    def _make_stats(self):
        cfg = CLAConfig(n_layers=8, sharing_factor=2)
        sched = CLASchedule.from_config(cfg)
        return CLAStats(schedule=sched, n_kv_heads=4, head_dim=64, seq_len=512)

    def test_kv_bytes_cla_less_than_standard(self):
        stats = self._make_stats()
        assert stats.kv_bytes_cla < stats.kv_bytes_standard

    def test_kv_memory_reduction_ratio_positive(self):
        stats = self._make_stats()
        assert 0.0 < stats.kv_memory_reduction_ratio < 1.0

    def test_kv_cache_multiplier_less_than_one(self):
        stats = self._make_stats()
        assert stats.kv_cache_multiplier < 1.0

    def test_sf1_no_reduction(self):
        cfg = CLAConfig(n_layers=8, sharing_factor=1)
        sched = CLASchedule.from_config(cfg)
        stats = CLAStats(schedule=sched, n_kv_heads=4, head_dim=64, seq_len=512)
        # All layers are generators when sf=1
        assert stats.kv_cache_multiplier == pytest.approx(1.0)
