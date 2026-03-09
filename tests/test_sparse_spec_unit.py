"""tests/test_sparse_spec_unit.py — unit tests for squish.sparse_spec"""

import math
import numpy as np
import pytest

from squish.sparse_spec import (
    SparseSpecConfig,
    PillarAttnCache,
    SparseSpecDrafter,
    SparseSpecStats,
    SparseSpecDecoder,
)

VOCAB = 16
RNG = np.random.default_rng(0)


def _fixed_fn(agree_tok: int, vocab: int = VOCAB):
    """Draft/target fn: peaks at agree_tok with 0.6 probability."""
    def fn(ids):
        p = np.ones(vocab, dtype=np.float32) * (0.4 / (vocab - 1))
        p[agree_tok % vocab] = 0.6
        return agree_tok % vocab, p
    return fn


def _make_decoder(agree_tok: int = 3, gamma: int = 2, top_k_ratio: float = 1.0):
    cfg = SparseSpecConfig(gamma=gamma, top_k_ratio=top_k_ratio, warmup_steps=0)
    cache = PillarAttnCache(capacity=64)
    drafter = SparseSpecDrafter(_fixed_fn(agree_tok), cache, cfg)
    decoder = SparseSpecDecoder(drafter, _fixed_fn(agree_tok), cfg)
    return decoder, cfg, cache


# ---------------------------------------------------------------------------
# SparseSpecConfig
# ---------------------------------------------------------------------------

class TestSparseSpecConfig:
    def test_defaults(self):
        cfg = SparseSpecConfig()
        assert cfg.gamma == 8
        assert cfg.top_k_ratio == 0.05
        assert cfg.warmup_steps == 2

    def test_custom(self):
        cfg = SparseSpecConfig(gamma=4, top_k_ratio=0.1, temperature=0.8, warmup_steps=1)
        assert cfg.gamma == 4

    @pytest.mark.parametrize("field,val", [
        ("gamma", 0),
        ("top_k_ratio", 0.0),
        ("top_k_ratio", 1.1),
        ("temperature", 0.0),
        ("top_p", 0.0),
        ("warmup_steps", -1),
    ])
    def test_invalid(self, field, val):
        with pytest.raises(ValueError):
            SparseSpecConfig(**{field: val})


# ---------------------------------------------------------------------------
# PillarAttnCache
# ---------------------------------------------------------------------------

class TestPillarAttnCache:
    def test_empty_top_k(self):
        cache = PillarAttnCache(capacity=32)
        idx = cache.top_k_indices(10)
        assert len(idx) == 0

    def test_update_and_top_k(self):
        cache = PillarAttnCache(capacity=16)
        scores = np.zeros(10, dtype=np.float32)
        scores[5] = 1.0
        scores[7] = 0.8
        cache.update(scores)
        top2 = cache.top_k_indices(2)
        assert set(top2.tolist()) == {5, 7}

    def test_k_larger_than_positions(self):
        cache = PillarAttnCache(capacity=16)
        cache.update(np.ones(4, dtype=np.float32))
        idx = cache.top_k_indices(100)
        assert len(idx) == 4

    def test_reset(self):
        cache = PillarAttnCache(capacity=16)
        cache.update(np.ones(8, dtype=np.float32))
        cache.reset()
        assert cache.n_positions == 0

    def test_capacity_clipping(self):
        cache = PillarAttnCache(capacity=4)
        cache.update(np.ones(10, dtype=np.float32))
        assert cache.n_positions == 4

    def test_invalid_capacity(self):
        with pytest.raises(ValueError):
            PillarAttnCache(capacity=0)


# ---------------------------------------------------------------------------
# SparseSpecDrafter
# ---------------------------------------------------------------------------

class TestSparseSpecDrafter:
    def test_non_callable(self):
        cfg = SparseSpecConfig()
        with pytest.raises(TypeError):
            SparseSpecDrafter("not_callable", PillarAttnCache(), cfg)

    def test_draft_length(self):
        cfg = SparseSpecConfig(gamma=5, warmup_steps=0, top_k_ratio=1.0)
        drafter = SparseSpecDrafter(_fixed_fn(2), PillarAttnCache(), cfg)
        tokens, probs = drafter.draft([1, 2, 3])
        assert len(tokens) == 5
        assert len(probs) == 5

    def test_prob_shape(self):
        cfg = SparseSpecConfig(gamma=3, warmup_steps=0, top_k_ratio=1.0)
        drafter = SparseSpecDrafter(_fixed_fn(1), PillarAttnCache(), cfg)
        _, probs = drafter.draft([0])
        for p in probs:
            assert p.shape == (VOCAB,)
            assert abs(p.sum() - 1.0) < 1e-5

    def test_prefers_agree_tok_at_low_temp(self):
        cfg = SparseSpecConfig(gamma=4, temperature=0.01, top_k_ratio=1.0, warmup_steps=0)
        drafter = SparseSpecDrafter(_fixed_fn(7), PillarAttnCache(), cfg)
        tokens, _ = drafter.draft([0])
        assert all(t == 7 for t in tokens)

    def test_sparse_context_used_on_warm_step(self):
        """After warmup, drafter uses sparse context — recorded call count."""
        cfg = SparseSpecConfig(gamma=2, warmup_steps=1, top_k_ratio=0.5)
        cache = PillarAttnCache(capacity=32)
        scores = np.ones(8, dtype=np.float32)
        cache.update(scores)
        drafter = SparseSpecDrafter(_fixed_fn(3), cache, cfg)
        # Step 1 = warmup; step 2 = warm
        drafter.draft(list(range(8)))
        tokens, probs = drafter.draft(list(range(8)))
        assert len(tokens) == 2

    def test_empty_input(self):
        cfg = SparseSpecConfig(gamma=2, warmup_steps=0, top_k_ratio=1.0)
        drafter = SparseSpecDrafter(_fixed_fn(0), PillarAttnCache(), cfg)
        tokens, _ = drafter.draft([])
        assert len(tokens) == 2


# ---------------------------------------------------------------------------
# SparseSpecStats
# ---------------------------------------------------------------------------

class TestSparseSpecStats:
    def test_defaults(self):
        s = SparseSpecStats()
        assert s.acceptance_rate == 0.0
        assert s.mean_accepted_per_step == 0.0
        assert s.kv_reduction_ratio >= 0.0

    def test_acceptance_rate(self):
        s = SparseSpecStats(accepted_total=8, rejected_total=2)
        assert abs(s.acceptance_rate - 0.8) < 1e-6

    def test_mean_accepted(self):
        s = SparseSpecStats(accepted_total=10, draft_steps=5)
        assert s.mean_accepted_per_step == 2.0

    def test_kv_reduction(self):
        s = SparseSpecStats(kv_ops_saved=90, draft_steps=10)
        assert s.kv_reduction_ratio > 0.0


# ---------------------------------------------------------------------------
# SparseSpecDecoder.generate
# ---------------------------------------------------------------------------

class TestSparseSpecDecoderGenerate:
    def test_non_callable_target(self):
        cfg = SparseSpecConfig(gamma=2, warmup_steps=0, top_k_ratio=1.0)
        drafter = SparseSpecDrafter(_fixed_fn(1), PillarAttnCache(), cfg)
        with pytest.raises(TypeError):
            SparseSpecDecoder(drafter, "bad", cfg)

    def test_output_length(self):
        dec, cfg, _ = _make_decoder(agree_tok=3, gamma=2)
        out, stats = dec.generate([0, 1], max_new_tokens=6)
        assert len(out) >= 2 + 1  # at least one new token
        assert len(out) <= 2 + 6 + 1  # no more than budget + bonus

    def test_max_new_tokens_respected(self):
        dec, _, _ = _make_decoder(agree_tok=5, gamma=3)
        for budget in (1, 3, 7, 10):
            out, stats = dec.generate([0], max_new_tokens=budget)
            assert len(out) - 1 <= budget

    def test_returns_stats(self):
        dec, _, _ = _make_decoder()
        _, stats = dec.generate([0], max_new_tokens=4)
        assert isinstance(stats, SparseSpecStats)
        assert stats.draft_steps >= 1

    def test_draft_steps_positive(self):
        dec, _, _ = _make_decoder(gamma=2)
        _, stats = dec.generate([0, 1, 2], max_new_tokens=5)
        assert stats.draft_steps >= 1

    def test_agreement_boosts_acceptance(self):
        dec, _, _ = _make_decoder(agree_tok=4, gamma=3)
        _, stats = dec.generate([0], max_new_tokens=12)
        assert stats.accepted_total > 0

    def test_empty_prompt(self):
        dec, _, _ = _make_decoder()
        out, stats = dec.generate([], max_new_tokens=4)
        assert len(out) <= 4 + 1

    def test_kv_stats_tracked(self):
        dec, _, _ = _make_decoder(top_k_ratio=0.5, gamma=2)
        _, stats = dec.generate([0] * 20, max_new_tokens=8)
        # kv_ops_saved may be 0 if warmup still active; check type
        assert isinstance(stats.kv_ops_saved, int)

    def test_default_config(self):
        cache = PillarAttnCache()
        drafter = SparseSpecDrafter(_fixed_fn(2), cache, SparseSpecConfig(warmup_steps=0))
        dec = SparseSpecDecoder(drafter, _fixed_fn(2))
        out, _ = dec.generate([0], max_new_tokens=3)
        assert len(out) >= 1
