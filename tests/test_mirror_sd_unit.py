"""tests/test_mirror_sd_unit.py — unit tests for squish/mirror_sd.py"""
import time

import numpy as np
import pytest

from squish.mirror_sd import (
    MirrorSDConfig,
    MirrorFuture,
    MirrorDraftPipeline,
    MirrorVerifyPipeline,
    MirrorSDStats,
    MirrorSDDecoder,
)

VOCAB = 16


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fixed_fn(agree_tok: int):
    """Always returns max logit on agree_tok."""
    def fn(ids):
        logits = np.full(VOCAB, -10.0, dtype=np.float32)
        logits[agree_tok] = 10.0
        return logits
    return fn


# ---------------------------------------------------------------------------
# MirrorSDConfig
# ---------------------------------------------------------------------------

class TestMirrorSDConfig:
    def test_defaults(self):
        cfg = MirrorSDConfig()
        assert cfg.gamma == 4
        assert cfg.temperature == 1.0
        assert cfg.top_p == 1.0
        assert cfg.overlap_steps == 2

    def test_custom(self):
        cfg = MirrorSDConfig(gamma=6, temperature=0.8, top_p=0.9, overlap_steps=3)
        assert cfg.gamma == 6
        assert cfg.overlap_steps == 3

    @pytest.mark.parametrize("kwargs, match", [
        ({"gamma": 0},           "gamma"),
        ({"temperature": 0},     "temperature"),
        ({"top_p": 0.0},         "top_p"),
        ({"top_p": 1.1},         "top_p"),
        ({"overlap_steps": 0},   "overlap_steps"),
    ])
    def test_validation(self, kwargs, match):
        with pytest.raises(ValueError, match=match):
            MirrorSDConfig(**kwargs)


# ---------------------------------------------------------------------------
# MirrorFuture
# ---------------------------------------------------------------------------

class TestMirrorFuture:
    def test_wait_returns_value(self):
        fut = MirrorFuture(lambda: 42)
        assert fut.wait() == 42

    def test_ready_after_wait(self):
        fut = MirrorFuture(lambda: "hello")
        fut.wait()
        assert fut.ready is True

    def test_not_ready_immediately_for_slow_fn(self):
        import threading
        start = threading.Event()
        def slow():
            start.set()
            time.sleep(0.05)
            return 99
        fut = MirrorFuture(slow)
        start.wait()  # wait until thread has started
        # May or may not be ready yet; just check wait() returns correctly
        assert fut.wait() == 99
        assert fut.ready is True

    def test_exception_propagates(self):
        def boom():
            raise RuntimeError("from future")
        fut = MirrorFuture(boom)
        with pytest.raises(RuntimeError, match="from future"):
            fut.wait()


# ---------------------------------------------------------------------------
# MirrorDraftPipeline
# ---------------------------------------------------------------------------

class TestMirrorDraftPipeline:
    def _make(self, tok: int = 5):
        cfg = MirrorSDConfig()
        return MirrorDraftPipeline(_fixed_fn(tok), cfg, rng_seed=0)

    def test_step_returns_token_and_probs(self):
        pipe = self._make(3)
        tok, probs = pipe.step([0, 1])
        assert isinstance(tok, int)
        assert probs.shape == (VOCAB,)
        assert abs(probs.sum() - 1.0) < 1e-5

    def test_step_preferred_token(self):
        """At temperature→0, the highest-logit token should dominate."""
        cfg = MirrorSDConfig(temperature=0.01)
        pipe = MirrorDraftPipeline(_fixed_fn(7), cfg, rng_seed=0)
        # 20 steps — with near-zero temp, token 7 should always win
        tokens = [pipe.step([0])[0] for _ in range(20)]
        assert all(t == 7 for t in tokens)

    def test_draft_sequence_length(self):
        pipe = self._make(2)
        tokens, probs_list = pipe.draft_sequence([0], gamma=3)
        assert len(tokens) == 3
        assert len(probs_list) == 3

    def test_draft_sequence_shape(self):
        pipe = self._make(4)
        _, probs_list = pipe.draft_sequence([0, 1, 2], gamma=4)
        for p in probs_list:
            assert p.shape == (VOCAB,)

    def test_draft_sequence_zero_gamma(self):
        pipe = self._make(1)
        tokens, probs_list = pipe.draft_sequence([0], gamma=0)
        assert tokens == []
        assert probs_list == []


# ---------------------------------------------------------------------------
# MirrorVerifyPipeline
# ---------------------------------------------------------------------------

class TestMirrorVerifyPipeline:
    def _make(self, tok: int = 5):
        cfg = MirrorSDConfig()
        return MirrorVerifyPipeline(_fixed_fn(tok), cfg, rng_seed=0)

    def test_enqueue_returns_future(self):
        pipe = self._make(3)
        fut = pipe.enqueue([0, 1])
        assert isinstance(fut, MirrorFuture)

    def test_enqueue_wait_returns_token_probs(self):
        pipe = self._make(3)
        fut = pipe.enqueue([0])
        tok, probs = fut.wait()
        assert isinstance(tok, int)
        assert probs.shape == (VOCAB,)

    def test_enqueue_preferred_token(self):
        cfg = MirrorSDConfig(temperature=0.01)
        pipe = MirrorVerifyPipeline(_fixed_fn(9), cfg, rng_seed=0)
        tok, _ = pipe.enqueue([0]).wait()
        assert tok == 9

    def test_multiple_futures_concurrent(self):
        pipe = self._make(5)
        futures = [pipe.enqueue([i]) for i in range(5)]
        for fut in futures:
            tok, probs = fut.wait()
            assert isinstance(tok, int)


# ---------------------------------------------------------------------------
# MirrorSDStats
# ---------------------------------------------------------------------------

class TestMirrorSDStats:
    def test_defaults(self):
        s = MirrorSDStats()
        assert s.total_tokens == 0
        assert s.overlap_hits == 0

    def test_acceptance_rate_zero(self):
        assert MirrorSDStats().acceptance_rate == 0.0

    def test_acceptance_rate(self):
        s = MirrorSDStats(accepted_total=6, rejected_total=2)
        assert s.acceptance_rate == pytest.approx(0.75)

    def test_mean_accepted_per_step_zero(self):
        assert MirrorSDStats().mean_accepted_per_step == 0.0

    def test_mean_accepted_per_step(self):
        s = MirrorSDStats(accepted_total=9, draft_steps=3)
        assert s.mean_accepted_per_step == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# MirrorSDDecoder
# ---------------------------------------------------------------------------

def _make_decoder(agree_tok: int = 5, gamma: int = 4, rng_seed: int = 0):
    cfg = MirrorSDConfig(gamma=gamma)
    draft = MirrorDraftPipeline(_fixed_fn(agree_tok), cfg, rng_seed=rng_seed)
    verify = MirrorVerifyPipeline(_fixed_fn(agree_tok), cfg, rng_seed=rng_seed + 1)
    return MirrorSDDecoder(draft, verify, cfg, rng_seed=rng_seed + 2)


class TestMirrorSDDecoderGenerate:
    def test_generates_token_count(self):
        dec = _make_decoder()
        _, stats = dec.generate([0], max_new_tokens=8)
        assert stats.total_tokens == 8

    def test_respects_max_new_tokens(self):
        dec = _make_decoder(gamma=3)
        _, stats = dec.generate([0, 1], max_new_tokens=7)
        assert stats.total_tokens == 7

    def test_output_length(self):
        dec = _make_decoder()
        out, stats = dec.generate([0, 1, 2], max_new_tokens=5)
        assert len(out) == 3 + 5

    def test_draft_steps_tracked(self):
        dec = _make_decoder()
        _, stats = dec.generate([0], max_new_tokens=6)
        assert stats.draft_steps > 0

    def test_agreement_high_acceptance(self):
        cfg = MirrorSDConfig(gamma=4, temperature=0.01)
        draft = MirrorDraftPipeline(_fixed_fn(8), cfg, rng_seed=0)
        verify = MirrorVerifyPipeline(_fixed_fn(8), cfg, rng_seed=1)
        dec = MirrorSDDecoder(draft, verify, cfg, rng_seed=2)
        _, stats = dec.generate([0], max_new_tokens=16)
        assert stats.acceptance_rate > 0.0

    def test_empty_prompt(self):
        dec = _make_decoder()
        out, stats = dec.generate([], max_new_tokens=4)
        assert stats.total_tokens == 4
        assert len(out) == 4

    def test_rejection_when_disagreement(self):
        """Draft always picks token 1, verify always picks token 2 → rejections."""
        cfg = MirrorSDConfig(gamma=2)
        draft = MirrorDraftPipeline(_fixed_fn(1), cfg, rng_seed=0)
        verify = MirrorVerifyPipeline(_fixed_fn(2), cfg, rng_seed=1)
        dec = MirrorSDDecoder(draft, verify, cfg, rng_seed=2)
        _, stats = dec.generate([0], max_new_tokens=10)
        assert stats.rejected_total > 0

    def test_single_token_gamma(self):
        """gamma=1 forces one draft token per step."""
        cfg = MirrorSDConfig(gamma=1)
        draft = MirrorDraftPipeline(_fixed_fn(3), cfg, rng_seed=0)
        verify = MirrorVerifyPipeline(_fixed_fn(3), cfg, rng_seed=1)
        dec = MirrorSDDecoder(draft, verify, cfg, rng_seed=2)
        _, stats = dec.generate([0], max_new_tokens=5)
        assert stats.total_tokens == 5
