"""tests/test_sparse_verify_unit.py — unit tests for squish.sparse_verify"""

import numpy as np
import pytest

from squish.sparse_verify import (
    InterDraftReuseCache,
    SparseVerifyConfig,
    SparseVerifyPass,
    SparseVerifyStats,
)

VOCAB = 16


def _accept_all_fn(context_ids, draft_tokens):
    """Verify fn: accepts all draft tokens, returns uniform probs."""
    probs = [np.ones(VOCAB, dtype=np.float32) / VOCAB for _ in draft_tokens]
    return list(draft_tokens), probs


def _reject_all_fn(context_ids, draft_tokens):
    """Verify fn: accepts nothing, returns first token as substitute."""
    p = np.ones(VOCAB, dtype=np.float32) / VOCAB
    return [0], [p]


# ---------------------------------------------------------------------------
# SparseVerifyConfig
# ---------------------------------------------------------------------------

class TestSparseVerifyConfig:
    def test_defaults(self):
        cfg = SparseVerifyConfig()
        assert cfg.attn_sparsity == 0.5
        assert cfg.ffn_sparsity == 0.3
        assert cfg.reuse_budget == 64

    def test_custom(self):
        cfg = SparseVerifyConfig(attn_sparsity=0.8, reuse_budget=128)
        assert cfg.attn_sparsity == 0.8

    @pytest.mark.parametrize("field,val", [
        ("attn_sparsity", -0.1),
        ("attn_sparsity", 1.0),
        ("ffn_sparsity", 1.0),
        ("ffn_threshold", -1.0),
        ("reuse_budget", 0),
        ("min_confidence", -0.1),
        ("min_confidence", 1.1),
    ])
    def test_invalid(self, field, val):
        with pytest.raises(ValueError):
            SparseVerifyConfig(**{field: val})


# ---------------------------------------------------------------------------
# InterDraftReuseCache
# ---------------------------------------------------------------------------

class TestInterDraftReuseCache:
    def test_empty_query(self):
        cache = InterDraftReuseCache(budget=8)
        new_idx, reused = cache.query_reuse(1, np.array([0, 1, 2]))
        assert reused == 0
        assert len(new_idx) == 3

    def test_record_and_reuse(self):
        cache = InterDraftReuseCache(budget=8)
        cache.record(0, np.array([2, 5, 7]))
        new_idx, reused = cache.query_reuse(1, np.array([2, 5, 7, 9]))
        assert reused == 3

    def test_partial_overlap(self):
        cache = InterDraftReuseCache(budget=8)
        cache.record(0, np.array([1, 3]))
        new_idx, reused = cache.query_reuse(1, np.array([1, 4, 5]))
        assert reused == 1
        assert 4 in new_idx.tolist() or 5 in new_idx.tolist()

    def test_budget_eviction(self):
        cache = InterDraftReuseCache(budget=2)
        cache.record(0, np.array([0]))
        cache.record(1, np.array([1]))
        cache.record(2, np.array([2]))  # evicts key=0
        assert 0 not in cache._entries

    def test_reset(self):
        cache = InterDraftReuseCache(budget=4)
        cache.record(0, np.array([1, 2]))
        cache.reset()
        assert len(cache._entries) == 0
        assert cache.hit_count == 0

    def test_reuse_rate(self):
        cache = InterDraftReuseCache(budget=8)
        cache.record(0, np.array([1, 2, 3]))
        cache.query_reuse(1, np.array([1, 2, 3]))
        assert cache.reuse_rate > 0.0

    def test_invalid_budget(self):
        with pytest.raises(ValueError):
            InterDraftReuseCache(budget=0)


# ---------------------------------------------------------------------------
# SparseVerifyStats
# ---------------------------------------------------------------------------

class TestSparseVerifyStats:
    def test_defaults(self):
        s = SparseVerifyStats()
        assert s.ops_saved_total == 0
        assert s.mean_tokens_per_call == 0.0
        assert s.reuse_rate == 0.0

    def test_ops_saved(self):
        s = SparseVerifyStats(attn_ops_saved=100, ffn_ops_saved=50)
        assert s.ops_saved_total == 150

    def test_mean_tokens(self):
        s = SparseVerifyStats(verify_calls=4, tokens_evaluated=20)
        assert s.mean_tokens_per_call == 5.0

    def test_reuse_rate(self):
        s = SparseVerifyStats(reuse_hits=30, tokens_evaluated=60)
        assert s.reuse_rate > 0.0


# ---------------------------------------------------------------------------
# SparseVerifyPass
# ---------------------------------------------------------------------------

class TestSparseVerifyPass:
    def test_non_callable(self):
        with pytest.raises(TypeError):
            SparseVerifyPass("not_callable")

    def test_passthrough_accept_all(self):
        svp = SparseVerifyPass(_accept_all_fn)
        accepted, probs = svp([0, 1, 2], [3, 4, 5])
        assert accepted == [3, 4, 5]
        assert len(probs) == 3

    def test_passthrough_reject(self):
        svp = SparseVerifyPass(_reject_all_fn)
        accepted, probs = svp([0, 1], [5, 6, 7])
        assert len(accepted) == 1

    def test_stats_increment(self):
        svp = SparseVerifyPass(_accept_all_fn)
        svp([0], [1, 2])
        svp([0], [3, 4, 5])
        stats = svp.get_stats()
        assert stats.verify_calls == 2
        assert stats.tokens_evaluated == 5

    def test_attn_ops_saved_positive(self):
        cfg = SparseVerifyConfig(attn_sparsity=0.8)
        svp = SparseVerifyPass(_accept_all_fn, config=cfg)
        svp(list(range(100)), list(range(4)))
        assert svp.get_stats().attn_ops_saved > 0

    def test_ffn_ops_saved_positive(self):
        cfg = SparseVerifyConfig(ffn_sparsity=0.5)
        svp = SparseVerifyPass(_accept_all_fn, config=cfg)
        svp([0], [1, 2, 3])
        assert svp.get_stats().ffn_ops_saved > 0

    def test_reset_stats(self):
        svp = SparseVerifyPass(_accept_all_fn)
        svp([0], [1])
        svp.reset_stats()
        assert svp.get_stats().verify_calls == 0

    def test_full_reset(self):
        svp = SparseVerifyPass(_accept_all_fn)
        svp([0], [1])
        svp.reset()
        assert svp.get_stats().tokens_evaluated == 0

    def test_default_config(self):
        svp = SparseVerifyPass(_accept_all_fn)
        assert svp._cfg.attn_sparsity == 0.5

    def test_zero_sparsity_no_ops_saved(self):
        cfg = SparseVerifyConfig(attn_sparsity=0.0, ffn_sparsity=0.0)
        svp = SparseVerifyPass(_accept_all_fn, config=cfg)
        svp([0], [1])
        assert svp.get_stats().attn_ops_saved == 0
        assert svp.get_stats().ffn_ops_saved == 0
