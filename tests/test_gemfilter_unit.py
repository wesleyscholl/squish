"""Unit tests for squish.gemfilter — Early-layer input token compression."""

import numpy as np
import pytest

from squish.gemfilter import (
    AttentionScoreBuffer,
    GemFilterConfig,
    GemFilterStats,
    GemSelector,
)

# ---------------------------------------------------------------------------
# TestGemFilterConfig
# ---------------------------------------------------------------------------

class TestGemFilterConfig:
    def test_defaults(self):
        cfg = GemFilterConfig()
        assert cfg.filter_layer == 15
        assert cfg.top_k_fraction == 0.10
        assert cfg.aggregation == "mean"

    def test_invalid_filter_layer(self):
        with pytest.raises(ValueError):
            GemFilterConfig(filter_layer=-1)

    def test_invalid_top_k_tokens(self):
        with pytest.raises(ValueError):
            GemFilterConfig(top_k_tokens=0)

    def test_invalid_top_k_fraction(self):
        with pytest.raises(ValueError):
            GemFilterConfig(top_k_fraction=0.0)

    def test_invalid_top_k_fraction_high(self):
        with pytest.raises(ValueError):
            GemFilterConfig(top_k_fraction=1.5)

    def test_invalid_aggregation(self):
        with pytest.raises(ValueError):
            GemFilterConfig(aggregation="sum")

    def test_budget_with_top_k_tokens(self):
        cfg = GemFilterConfig(top_k_tokens=50)
        assert cfg.budget(200) == 50

    def test_budget_capped_by_seq_len(self):
        cfg = GemFilterConfig(top_k_tokens=500)
        assert cfg.budget(100) == 100

    def test_budget_with_fraction(self):
        cfg = GemFilterConfig(top_k_tokens=None, top_k_fraction=0.10)
        assert cfg.budget(1000) == 100

    def test_invalid_keep_prefix_tokens(self):
        with pytest.raises(ValueError):
            GemFilterConfig(keep_prefix_tokens=-1)


# ---------------------------------------------------------------------------
# TestAttentionScoreBuffer
# ---------------------------------------------------------------------------

class TestAttentionScoreBuffer:
    def _cfg(self):
        return GemFilterConfig(filter_layer=3)

    def test_no_record_returns_none(self):
        buf = AttentionScoreBuffer(self._cfg())
        assert buf.get_scores() is None

    def test_ignores_wrong_layer(self):
        buf = AttentionScoreBuffer(self._cfg())
        attn = np.ones((4, 10, 50))
        buf.record(5, attn)  # wrong layer
        assert buf.get_scores() is None

    def test_records_correct_layer_1d(self):
        buf = AttentionScoreBuffer(self._cfg())
        scores = np.array([0.1, 0.5, 0.3, 0.9, 0.2])
        buf.record(3, scores)
        out = buf.get_scores()
        assert out is not None
        assert len(out) == 5

    def test_records_2d_uses_last_query_row(self):
        buf = AttentionScoreBuffer(self._cfg())
        attn = np.array([[0.1, 0.2, 0.7], [0.3, 0.4, 0.3]])  # (n_queries, seq_len)
        buf.record(3, attn)
        out = buf.get_scores()
        assert len(out) == 3  # seq_len

    def test_aggregation_mean_3d(self):
        cfg = GemFilterConfig(filter_layer=3, aggregation="mean")
        buf = AttentionScoreBuffer(cfg)
        attn = np.ones((4, 2, 10)) * 0.5  # (n_heads, n_queries, seq_len)
        buf.record(3, attn)
        out = buf.get_scores()
        assert out is not None
        assert len(out) == 10
        np.testing.assert_allclose(out, 0.5, atol=1e-5)

    def test_aggregation_max_3d(self):
        cfg = GemFilterConfig(filter_layer=3, aggregation="max")
        buf = AttentionScoreBuffer(cfg)
        attn = np.arange(48, dtype=np.float32).reshape(4, 2, 6)
        buf.record(3, attn)
        out = buf.get_scores()
        assert out is not None
        assert len(out) == 6

    def test_reset_clears_buffer(self):
        buf = AttentionScoreBuffer(self._cfg())
        buf.record(3, np.ones(5))
        buf.reset()
        assert buf.get_scores() is None


# ---------------------------------------------------------------------------
# TestGemSelector
# ---------------------------------------------------------------------------

class TestGemSelector:
    def test_select_returns_correct_count(self):
        cfg = GemFilterConfig(top_k_tokens=5,
                               always_keep_first=False, always_keep_last=False)
        sel = GemSelector(cfg)
        rng = np.random.default_rng(0)
        scores = rng.random(100)
        indices = sel.select(scores)
        assert len(indices) == 5

    def test_select_sorted_ascending(self):
        cfg = GemFilterConfig(top_k_tokens=10,
                               always_keep_first=False, always_keep_last=False)
        sel = GemSelector(cfg)
        scores = np.random.default_rng(0).random(100)
        indices = sel.select(scores)
        assert list(indices) == sorted(indices)

    def test_always_keep_prefix(self):
        cfg = GemFilterConfig(top_k_tokens=5, always_keep_first=True,
                               keep_prefix_tokens=3, always_keep_last=False)
        sel = GemSelector(cfg)
        scores = np.zeros(50)  # all zero except high scores elsewhere
        scores[40:45] = 1.0
        indices = sel.select(scores)
        for i in range(3):
            assert i in indices

    def test_always_keep_suffix(self):
        cfg = GemFilterConfig(top_k_tokens=5, always_keep_last=True,
                               keep_suffix_tokens=3, always_keep_first=False)
        sel = GemSelector(cfg)
        scores = np.zeros(50)
        scores[0:5] = 1.0
        indices = sel.select(scores)
        for i in range(47, 50):
            assert i in indices

    def test_budget_not_exceeded(self):
        cfg = GemFilterConfig(top_k_fraction=0.1,
                               always_keep_first=False, always_keep_last=False)
        sel = GemSelector(cfg)
        scores = np.random.default_rng(0).random(200)
        indices = sel.select(scores)
        assert len(indices) <= cfg.budget(200) + 1  # +1 tolerance

    def test_compression_ratio(self):
        sel = GemSelector(GemFilterConfig())
        r = sel.compression_ratio(1000, 100)
        assert r == pytest.approx(0.9)


# ---------------------------------------------------------------------------
# TestGemFilterStats
# ---------------------------------------------------------------------------

class TestGemFilterStats:
    def test_initial_state(self):
        stats = GemFilterStats()
        assert stats.n_calls == 0
        assert stats.mean_compression_ratio == 0.0

    def test_record_updates_counts(self):
        stats = GemFilterStats()
        stats.record(1000, 100)
        assert stats.n_calls == 1
        assert stats.total_input_tokens == 1000
        assert stats.total_kept_tokens == 100

    def test_mean_compression_ratio(self):
        stats = GemFilterStats()
        stats.record(1000, 100)
        assert stats.mean_compression_ratio == pytest.approx(0.9)

    def test_mean_kept_fraction(self):
        stats = GemFilterStats()
        stats.record(1000, 100)
        assert stats.mean_kept_fraction == pytest.approx(0.1)

    def test_mean_speedup_gt_one_when_compressed(self):
        stats = GemFilterStats()
        stats.record(1000, 80)  # 8% kept
        assert stats.mean_speedup_estimate > 1.0

    def test_no_compression_speedup_approx_one(self):
        stats = GemFilterStats()
        stats.record(100, 100)
        assert stats.mean_speedup_estimate >= 1.0
