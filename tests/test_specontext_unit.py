"""tests/test_specontext_unit.py — unit tests for squish.specontext"""

import numpy as np
import pytest

from squish.specontext import (
    DistilledRetrievalHead,
    SpeContextCache,
    SpeContextConfig,
    SpeContextStats,
)

HEAD_DIM = 8
RNG = np.random.default_rng(99)


def _make_config(**kw):
    defaults = dict(
        retrieval_topk=4,
        prefetch_budget=8,
        head_dim=HEAD_DIM,
        n_retrieval_heads=2,
        gqa_groups=2,
    )
    defaults.update(kw)
    return SpeContextConfig(**defaults)


def _make_head(config=None):
    cfg = config or _make_config()
    W = RNG.standard_normal((cfg.n_retrieval_heads, cfg.head_dim, cfg.head_dim)).astype(np.float32)
    return DistilledRetrievalHead(cfg, head_weights=W)


# ---------------------------------------------------------------------------
# SpeContextConfig
# ---------------------------------------------------------------------------

class TestSpeContextConfig:
    def test_defaults(self):
        cfg = SpeContextConfig()
        assert cfg.retrieval_topk == 128
        assert cfg.gqa_groups == 4

    def test_custom(self):
        cfg = _make_config()
        assert cfg.retrieval_topk == 4

    @pytest.mark.parametrize("field,val", [
        ("retrieval_topk", 0),
        ("prefetch_budget", 1),   # < retrieval_topk
        ("head_dim", 0),
        ("n_retrieval_heads", 0),
        ("gqa_groups", 0),
    ])
    def test_invalid(self, field, val):
        base = dict(retrieval_topk=4, prefetch_budget=8, head_dim=8, n_retrieval_heads=2, gqa_groups=2)
        base[field] = val
        with pytest.raises(ValueError):
            SpeContextConfig(**base)


# ---------------------------------------------------------------------------
# DistilledRetrievalHead
# ---------------------------------------------------------------------------

class TestDistilledRetrievalHead:
    def test_score_tokens_shape(self):
        head = _make_head()
        q = RNG.standard_normal(HEAD_DIM).astype(np.float32)
        K = RNG.standard_normal((20, HEAD_DIM)).astype(np.float32)
        scores = head.score_tokens(q, K)
        assert scores.shape == (20,)

    def test_top_k_indices_count(self):
        head = _make_head()
        q = RNG.standard_normal(HEAD_DIM).astype(np.float32)
        K = RNG.standard_normal((30, HEAD_DIM)).astype(np.float32)
        idx = head.top_k_indices(q, K, k=5)
        assert len(idx) == 5

    def test_top_k_k_larger_than_seq(self):
        head = _make_head()
        q = RNG.standard_normal(HEAD_DIM).astype(np.float32)
        K = RNG.standard_normal((3, HEAD_DIM)).astype(np.float32)
        idx = head.top_k_indices(q, K, k=10)
        assert len(idx) == 3

    def test_score_query_dim_mismatch(self):
        head = _make_head()
        q = RNG.standard_normal(HEAD_DIM + 2).astype(np.float32)
        K = RNG.standard_normal((10, HEAD_DIM)).astype(np.float32)
        with pytest.raises(ValueError):
            head.score_tokens(q, K)

    def test_score_key_dim_mismatch(self):
        head = _make_head()
        q = RNG.standard_normal(HEAD_DIM).astype(np.float32)
        K = RNG.standard_normal((10, HEAD_DIM + 1)).astype(np.float32)
        with pytest.raises(ValueError):
            head.score_tokens(q, K)

    def test_set_weights_wrong_shape(self):
        head = _make_head()
        bad_W = RNG.standard_normal((3, HEAD_DIM, HEAD_DIM)).astype(np.float32)
        with pytest.raises(ValueError):
            head.set_weights(bad_W)

    def test_wrong_weight_shape_on_init(self):
        cfg = _make_config()
        bad_W = RNG.standard_normal((1, HEAD_DIM, HEAD_DIM)).astype(np.float32)
        with pytest.raises(ValueError):
            DistilledRetrievalHead(cfg, head_weights=bad_W)

    def test_default_init_no_error(self):
        cfg = _make_config()
        head = DistilledRetrievalHead(cfg)  # random orthogonal init
        q = RNG.standard_normal(HEAD_DIM).astype(np.float32)
        K = RNG.standard_normal((5, HEAD_DIM)).astype(np.float32)
        scores = head.score_tokens(q, K)
        assert scores.shape == (5,)


# ---------------------------------------------------------------------------
# SpeContextCache
# ---------------------------------------------------------------------------

class TestSpeContextCache:
    def test_empty_retrieve(self):
        head = _make_head()
        cfg = _make_config()
        cache = SpeContextCache(head, cfg)
        q = RNG.standard_normal(HEAD_DIM).astype(np.float32)
        keys, vals, idx = cache.retrieve(q)
        assert keys.shape[0] == 0

    def test_append_and_retrieve_shape(self):
        head = _make_head()
        cfg = _make_config()
        cache = SpeContextCache(head, cfg)
        for _ in range(20):
            k = RNG.standard_normal(HEAD_DIM).astype(np.float32)
            v = RNG.standard_normal(HEAD_DIM).astype(np.float32)
            cache.append(k, v)
        q = RNG.standard_normal(HEAD_DIM).astype(np.float32)
        keys, vals, idx = cache.retrieve(q)
        assert keys.shape == (min(cfg.retrieval_topk, 20), HEAD_DIM)
        assert vals.shape == (min(cfg.retrieval_topk, 20), HEAD_DIM)

    def test_hot_count(self):
        head = _make_head()
        cfg = _make_config()
        cache = SpeContextCache(head, cfg)
        for _ in range(10):
            cache.append(
                RNG.standard_normal(HEAD_DIM).astype(np.float32),
                RNG.standard_normal(HEAD_DIM).astype(np.float32),
            )
        q = RNG.standard_normal(HEAD_DIM).astype(np.float32)
        cache.retrieve(q)
        assert cache.hot_count == min(cfg.retrieval_topk, 10)

    def test_reset(self):
        head = _make_head()
        cfg = _make_config()
        cache = SpeContextCache(head, cfg)
        cache.append(np.ones(HEAD_DIM), np.ones(HEAD_DIM))
        cache.reset()
        assert cache.size == 0

    def test_size_increments(self):
        head = _make_head()
        cfg = _make_config()
        cache = SpeContextCache(head, cfg)
        for i in range(5):
            cache.append(np.ones(HEAD_DIM), np.ones(HEAD_DIM))
        assert cache.size == 5


# ---------------------------------------------------------------------------
# SpeContextStats
# ---------------------------------------------------------------------------

class TestSpeContextStats:
    def test_defaults(self):
        s = SpeContextStats()
        assert s.hot_tier_rate == 0.0
        assert s.mean_hot_per_call == 0.0

    def test_hot_tier_rate(self):
        s = SpeContextStats(hot_tokens_served=80, cold_accesses=20)
        assert abs(s.hot_tier_rate - 0.8) < 1e-6

    def test_mean_hot(self):
        s = SpeContextStats(retrieval_calls=5, hot_tokens_served=50)
        assert s.mean_hot_per_call == 10.0
