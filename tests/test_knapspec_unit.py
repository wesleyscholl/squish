"""tests/test_knapspec_unit.py — 100% line and branch coverage for squish/knapspec.py"""
import numpy as np
import pytest

from squish.knapspec import KnapSpecConfig, KnapSpecSelector


# ---------------------------------------------------------------------------
# KnapSpecConfig
# ---------------------------------------------------------------------------

class TestKnapSpecConfig:
    def test_defaults(self):
        cfg = KnapSpecConfig()
        assert cfg.num_layers == 32
        assert cfg.attn_base_latency == 1.0
        assert cfg.attn_context_coeff == 0.001
        assert cfg.mlp_latency == 1.5
        assert cfg.budget_fraction == 0.5
        assert cfg.dp_resolution == 200

    def test_custom_values(self):
        cfg = KnapSpecConfig(
            num_layers=8,
            attn_base_latency=2.0,
            mlp_latency=3.0,
            budget_fraction=0.7,
            dp_resolution=50,
        )
        assert cfg.num_layers == 8
        assert cfg.budget_fraction == 0.7

    @pytest.mark.parametrize("kwargs, match", [
        ({"num_layers": 0}, "num_layers"),
        ({"attn_base_latency": -0.1}, "attn_base_latency"),
        ({"attn_context_coeff": -0.01}, "attn_context_coeff"),
        ({"mlp_latency": -1.0}, "mlp_latency"),
        ({"budget_fraction": 0.0}, "budget_fraction"),
        ({"budget_fraction": 1.1}, "budget_fraction"),
        ({"dp_resolution": 0}, "dp_resolution"),
    ])
    def test_validation(self, kwargs, match):
        with pytest.raises(ValueError, match=match):
            KnapSpecConfig(**kwargs)

    def test_budget_fraction_1_valid(self):
        cfg = KnapSpecConfig(budget_fraction=1.0)
        assert cfg.budget_fraction == 1.0

    def test_zero_latency_valid(self):
        cfg = KnapSpecConfig(attn_base_latency=0.0, mlp_latency=0.0)
        assert cfg.attn_base_latency == 0.0


# ---------------------------------------------------------------------------
# KnapSpecSelector — helpers
# ---------------------------------------------------------------------------

class TestKnapSpecSelectorHelpers:
    def _make(self, n=4, attn_base=1.0, attn_coeff=0.0, mlp=1.0, frac=0.5):
        cfg = KnapSpecConfig(
            num_layers=n,
            attn_base_latency=attn_base,
            attn_context_coeff=attn_coeff,
            mlp_latency=mlp,
            budget_fraction=frac,
        )
        return KnapSpecSelector(cfg)

    def test_full_model_latency_no_context(self):
        sel = self._make(n=4, attn_base=1.0, attn_coeff=0.0, mlp=1.0)
        # Each layer: attn=1 + mlp=1 = 2; 4 layers = 8
        assert sel.full_model_latency(0) == pytest.approx(8.0)

    def test_full_model_latency_with_context(self):
        sel = self._make(n=2, attn_base=1.0, attn_coeff=1.0, mlp=1.0)
        # context_len=10: attn_cost = 1+10=11; layer cost = 11+1=12; total=24
        assert sel.full_model_latency(10) == pytest.approx(24.0)

    def test_block_costs_layout(self):
        sel = self._make(n=2, attn_base=2.0, attn_coeff=0.0, mlp=3.0)
        costs = sel.block_costs(0)
        assert len(costs) == 4  # 2 attn + 2 mlp
        # Indices 0,2 = attn = 2.0; indices 1,3 = mlp = 3.0
        np.testing.assert_allclose(costs[0::2], 2.0)
        np.testing.assert_allclose(costs[1::2], 3.0)

    def test_block_costs_context_dependence(self):
        sel = self._make(n=2, attn_base=1.0, attn_coeff=0.1, mlp=1.0)
        c100 = sel.block_costs(100)
        c0   = sel.block_costs(0)
        # Attention at context 100 is larger than at context 0
        assert c100[0] > c0[0]
        # MLP unchanged
        assert c100[1] == c0[1]


# ---------------------------------------------------------------------------
# KnapSpecSelector — select() branches
# ---------------------------------------------------------------------------

class TestKnapSpecSelectorSelect:
    def _make(self, n=4, attn_base=1.0, attn_coeff=0.0, mlp=1.0,
              frac=0.5, res=20):
        cfg = KnapSpecConfig(
            num_layers=n,
            attn_base_latency=attn_base,
            attn_context_coeff=attn_coeff,
            mlp_latency=mlp,
            budget_fraction=frac,
            dp_resolution=res,
        )
        return KnapSpecSelector(cfg)

    def test_full_budget_returns_all(self):
        """budget_fraction=1.0 → keep everything."""
        sel = self._make(frac=1.0)
        attn, mlp = sel.select(0)
        assert attn == list(range(4))
        assert mlp  == list(range(4))

    def test_zero_cost_model_returns_all(self):
        """All zero latencies → degenerate case, keep everything."""
        sel = self._make(n=4, attn_base=0.0, attn_coeff=0.0, mlp=0.0, frac=0.5)
        attn, mlp = sel.select(0)
        assert attn == list(range(4))
        assert mlp  == list(range(4))

    def test_very_small_budget_returns_few(self):
        """Very small budget_fraction → many blocks skipped."""
        # budget = 5% of total → very few blocks kept
        sel = self._make(n=8, frac=0.05, res=50)
        attn, mlp = sel.select(0)
        total_kept = len(attn) + len(mlp)
        assert total_kept <= 8  # fewer than half the blocks

    def test_partial_budget_runs_dp(self):
        """Normal budget: DP selects a proper subset."""
        sel = self._make(n=4, attn_base=1.0, mlp=2.0, frac=0.4, res=100)
        attn, mlp = sel.select(0)
        # Not all block types must be empty, not all must be full
        assert len(attn) + len(mlp) < 8  # something was skipped

    def test_custom_quality_weights(self):
        """Higher-quality blocks should be preferred when budget is tight."""
        n = 4
        cfg = KnapSpecConfig(
            num_layers=n,
            attn_base_latency=1.0,
            attn_context_coeff=0.0,
            mlp_latency=1.0,
            budget_fraction=0.25,  # very tight budget
            dp_resolution=200,
        )
        sel = KnapSpecSelector(cfg)
        # Give block 0 (attn_0) very high quality; all others low
        quality = np.zeros(2 * n, dtype=np.float64)
        quality[0] = 100.0  # attn_0
        attn, _ = sel.select(0, quality_weights=quality)
        # Layer 0 attention should be included
        assert 0 in attn

    def test_context_length_affects_attn_cost(self):
        """At long context, attention becomes more expensive → fewer attn kept."""
        cfg_short = KnapSpecConfig(
            num_layers=8,
            attn_base_latency=0.1,
            attn_context_coeff=0.01,
            mlp_latency=1.0,
            budget_fraction=0.5,
            dp_resolution=100,
        )
        cfg_long = KnapSpecConfig(
            num_layers=8,
            attn_base_latency=0.1,
            attn_context_coeff=100.0,  # very expensive at long context
            mlp_latency=1.0,
            budget_fraction=0.5,
            dp_resolution=100,
        )
        sel_short = KnapSpecSelector(cfg_short)
        sel_long  = KnapSpecSelector(cfg_long)
        attn_short, _ = sel_short.select(0)
        attn_long,  _ = sel_long.select(10000)
        # Long context with expensive attention → fewer attention blocks kept
        assert len(attn_long) <= len(attn_short)

    def test_returns_sorted_indices(self):
        """Returned lists must be sorted."""
        sel = self._make(n=8, frac=0.5, res=50)
        attn, mlp = sel.select(0)
        assert attn == sorted(attn)
        assert mlp  == sorted(mlp)

    def test_indices_in_valid_range(self):
        """All returned indices must be in [0, num_layers-1]."""
        sel = self._make(n=8, frac=0.5, res=50)
        attn, mlp = sel.select(0)
        for idx in attn:
            assert 0 <= idx < 8
        for idx in mlp:
            assert 0 <= idx < 8

    def test_tiny_dp_resolution(self):
        """Resolution=1 should still work without error."""
        sel = self._make(n=4, frac=0.5, res=1)
        attn, mlp = sel.select(0)
        assert isinstance(attn, list)
        assert isinstance(mlp, list)

    def test_traceback_skips_unchosen_items(self):
        """Ensure the traceback correctly skips items that chose=False."""
        # With uniform costs and qualities, DP is deterministic
        sel = self._make(n=3, attn_base=1.0, mlp=1.0, frac=0.5, res=100)
        attn, mlp = sel.select(0)
        total = len(attn) + len(mlp)
        # Half budget means roughly half the blocks
        assert 0 < total < 6  # not zero, not all

    def test_zero_budget_fraction_returns_empty(self):
        """Force budget<=0 path by setting budget_fraction to negative value.

        This is a dead-code guard; we bypass validation to exercise the branch.
        """
        cfg = KnapSpecConfig(num_layers=4, attn_base_latency=1.0,
                             attn_context_coeff=0.0, mlp_latency=1.0,
                             budget_fraction=0.5)
        # Bypass validation: set budget_fraction negative → budget < 0
        cfg.budget_fraction = -0.1
        sel = KnapSpecSelector(cfg)
        attn, mlp = sel.select(0)
        assert attn == []
        assert mlp == []
