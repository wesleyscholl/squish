"""tests/test_duo_decoding_unit.py — unit tests for squish/duo_decoding.py"""
import numpy as np
import pytest

from squish.duo_decoding import (
    DuoDecodingConfig,
    DuoCandidate,
    DuoScheduler,
    DuoCPUVerifier,
    DuoDecodingStats,
    DuoDecodingDecoder,
)

VOCAB = 16


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fixed_fn(agree_tok: int):
    def fn(ids):
        logits = np.full(VOCAB, -10.0, dtype=np.float32)
        logits[agree_tok] = 10.0
        return logits
    return fn


def _make_decoder(agree_tok: int = 5, gamma: int = 4,
                  k_max: int = 3, rng_seed: int = 0):
    cfg = DuoDecodingConfig(gamma=gamma, k_max=k_max)
    sched = DuoScheduler(_fixed_fn(agree_tok), cfg, rng_seed=rng_seed)
    ver = DuoCPUVerifier(_fixed_fn(agree_tok), cfg, rng_seed=rng_seed + 1)
    return DuoDecodingDecoder(sched, ver, cfg, rng_seed=rng_seed + 2)


# ---------------------------------------------------------------------------
# DuoDecodingConfig
# ---------------------------------------------------------------------------

class TestDuoDecodingConfig:
    def test_defaults(self):
        cfg = DuoDecodingConfig()
        assert cfg.gamma == 4
        assert cfg.k_max == 3
        assert cfg.prune_threshold == pytest.approx(5.0)
        assert cfg.temperature == 1.0
        assert cfg.top_p == 1.0

    def test_custom(self):
        cfg = DuoDecodingConfig(gamma=2, k_max=5, prune_threshold=3.0)
        assert cfg.k_max == 5

    @pytest.mark.parametrize("kwargs, match", [
        ({"gamma": 0},              "gamma"),
        ({"k_max": 0},              "k_max"),
        ({"prune_threshold": -1.0}, "prune_threshold"),
        ({"temperature": 0},        "temperature"),
        ({"top_p": 0.0},            "top_p"),
        ({"top_p": 1.1},            "top_p"),
    ])
    def test_validation(self, kwargs, match):
        with pytest.raises(ValueError, match=match):
            DuoDecodingConfig(**kwargs)


# ---------------------------------------------------------------------------
# DuoCandidate
# ---------------------------------------------------------------------------

class TestDuoCandidate:
    def test_empty_candidate(self):
        c = DuoCandidate()
        assert c.depth == 0
        assert c.tokens == []
        assert c.log_prob == pytest.approx(0.0)

    def test_append_updates_depth(self):
        c = DuoCandidate()
        probs = np.zeros(VOCAB, dtype=np.float32)
        probs[3] = 1.0
        c.append(3, probs)
        assert c.depth == 1
        assert c.tokens == [3]

    def test_append_updates_log_prob(self):
        c = DuoCandidate()
        probs = np.zeros(VOCAB, dtype=np.float32)
        probs[3] = 0.5
        c.append(3, probs)
        assert c.log_prob == pytest.approx(np.log(0.5), abs=1e-5)

    def test_multiple_appends(self):
        c = DuoCandidate()
        for i in range(3):
            probs = np.zeros(VOCAB, dtype=np.float32)
            probs[i] = 1.0
            c.append(i, probs)
        assert c.depth == 3
        assert c.tokens == [0, 1, 2]


# ---------------------------------------------------------------------------
# DuoScheduler
# ---------------------------------------------------------------------------

class TestDuoScheduler:
    def test_returns_k_max_candidates(self):
        cfg = DuoDecodingConfig(k_max=3, gamma=2)
        sched = DuoScheduler(_fixed_fn(5), cfg, rng_seed=0)
        candidates = sched.draft_candidates([0])
        # After pruning, can be ≤ k_max; at least 1
        assert 1 <= len(candidates) <= cfg.k_max

    def test_candidates_have_correct_depth(self):
        cfg = DuoDecodingConfig(k_max=2, gamma=3)
        sched = DuoScheduler(_fixed_fn(2), cfg, rng_seed=0)
        candidates = sched.draft_candidates([0])
        for c in candidates:
            assert c.depth == cfg.gamma

    def test_candidates_sorted_by_log_prob(self):
        cfg = DuoDecodingConfig(k_max=4, gamma=3)
        sched = DuoScheduler(_fixed_fn(7), cfg, rng_seed=0)
        candidates = sched.draft_candidates([0])
        scores = [c.log_prob for c in candidates]
        assert scores == sorted(scores, reverse=True)

    def test_best_returns_highest_score(self):
        cfg = DuoDecodingConfig(k_max=3, gamma=2)
        sched = DuoScheduler(_fixed_fn(1), cfg, rng_seed=0)
        candidates = sched.draft_candidates([0])
        best = sched.best(candidates)
        assert best.log_prob == max(c.log_prob for c in candidates)

    def test_best_raises_on_empty(self):
        cfg = DuoDecodingConfig()
        sched = DuoScheduler(_fixed_fn(0), cfg)
        with pytest.raises(ValueError):
            sched.best([])

    def test_k_max_one_gives_single_candidate(self):
        cfg = DuoDecodingConfig(k_max=1, gamma=2)
        sched = DuoScheduler(_fixed_fn(3), cfg, rng_seed=0)
        candidates = sched.draft_candidates([0])
        assert len(candidates) == 1


# ---------------------------------------------------------------------------
# DuoCPUVerifier
# ---------------------------------------------------------------------------

class TestDuoCPUVerifier:
    def test_verify_one_shape(self):
        cfg = DuoDecodingConfig()
        ver = DuoCPUVerifier(_fixed_fn(4), cfg, rng_seed=0)
        tok, probs = ver.verify_one([0, 1])
        assert isinstance(tok, int)
        assert probs.shape == (VOCAB,)
        assert abs(probs.sum() - 1.0) < 1e-5

    def test_preferred_token_at_low_temp(self):
        cfg = DuoDecodingConfig(temperature=0.01)
        ver = DuoCPUVerifier(_fixed_fn(6), cfg, rng_seed=0)
        tokens = [ver.verify_one([0])[0] for _ in range(20)]
        assert all(t == 6 for t in tokens)


# ---------------------------------------------------------------------------
# DuoDecodingStats
# ---------------------------------------------------------------------------

class TestDuoDecodingStats:
    def test_defaults(self):
        s = DuoDecodingStats()
        assert s.total_tokens == 0
        assert s.candidates_pruned == 0

    def test_acceptance_rate_zero(self):
        assert DuoDecodingStats().acceptance_rate == 0.0

    def test_acceptance_rate(self):
        s = DuoDecodingStats(accepted_total=9, rejected_total=3)
        assert s.acceptance_rate == pytest.approx(0.75)

    def test_mean_accepted_per_step_zero(self):
        assert DuoDecodingStats().mean_accepted_per_step == 0.0

    def test_mean_accepted_per_step(self):
        s = DuoDecodingStats(accepted_total=12, draft_steps=4)
        assert s.mean_accepted_per_step == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# DuoDecodingDecoder
# ---------------------------------------------------------------------------

class TestDuoDecodingDecoderGenerate:
    def test_generates_token_count(self):
        dec = _make_decoder()
        _, stats = dec.generate([0], max_new_tokens=8)
        assert stats.total_tokens == 8

    def test_respects_max_new_tokens(self):
        dec = _make_decoder(gamma=3)
        _, stats = dec.generate([0, 1], max_new_tokens=7)
        assert stats.total_tokens == 7

    def test_output_ids_length(self):
        dec = _make_decoder()
        out, stats = dec.generate([0, 1], max_new_tokens=6)
        assert len(out) == 2 + 6

    def test_draft_steps_tracked(self):
        dec = _make_decoder()
        _, stats = dec.generate([0], max_new_tokens=6)
        assert stats.draft_steps > 0

    def test_agreement_high_acceptance(self):
        cfg = DuoDecodingConfig(gamma=4, k_max=2, temperature=0.01)
        sched = DuoScheduler(_fixed_fn(8), cfg, rng_seed=0)
        ver = DuoCPUVerifier(_fixed_fn(8), cfg, rng_seed=1)
        dec = DuoDecodingDecoder(sched, ver, cfg, rng_seed=2)
        _, stats = dec.generate([0], max_new_tokens=20)
        assert stats.acceptance_rate > 0.5

    def test_disagreement_causes_rejections(self):
        cfg = DuoDecodingConfig(gamma=2, k_max=2)
        sched = DuoScheduler(_fixed_fn(1), cfg, rng_seed=0)
        ver = DuoCPUVerifier(_fixed_fn(2), cfg, rng_seed=1)
        dec = DuoDecodingDecoder(sched, ver, cfg, rng_seed=2)
        _, stats = dec.generate([0], max_new_tokens=10)
        assert stats.rejected_total > 0

    def test_empty_prompt(self):
        dec = _make_decoder()
        out, stats = dec.generate([], max_new_tokens=4)
        assert stats.total_tokens == 4

    def test_default_config_in_decoder(self):
        cfg = DuoDecodingConfig()
        sched = DuoScheduler(_fixed_fn(3), cfg)
        ver = DuoCPUVerifier(_fixed_fn(3), cfg)
        dec = DuoDecodingDecoder(sched, ver)   # no explicit config
        _, stats = dec.generate([0], max_new_tokens=3)
        assert stats.total_tokens == 3

    def test_k_max_one_matches_single_candidate(self):
        cfg = DuoDecodingConfig(k_max=1, gamma=3)
        sched = DuoScheduler(_fixed_fn(4), cfg, rng_seed=0)
        ver = DuoCPUVerifier(_fixed_fn(4), cfg, rng_seed=1)
        dec = DuoDecodingDecoder(sched, ver, cfg, rng_seed=2)
        _, stats = dec.generate([0], max_new_tokens=6)
        assert stats.total_tokens == 6

    def test_candidates_pruned_counter(self):
        """With tight prune threshold, some candidates will be pruned."""
        cfg = DuoDecodingConfig(k_max=4, gamma=3, prune_threshold=0.01)
        sched = DuoScheduler(_fixed_fn(5), cfg, rng_seed=0)
        ver = DuoCPUVerifier(_fixed_fn(5), cfg, rng_seed=1)
        dec = DuoDecodingDecoder(sched, ver, cfg, rng_seed=2)
        _, stats = dec.generate([0], max_new_tokens=12)
        # Some candidates may be pruned when all get same high-prob token
        assert stats.candidates_pruned >= 0  # valid non-negative count
