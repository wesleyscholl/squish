"""tests/test_long_spec_unit.py — unit tests for squish/long_spec.py"""
import numpy as np
import pytest

from squish.long_spec import (
    LongSpecConfig,
    LongSpecDecoder,
    LongSpecHead,
    LongSpecStats,
)

# ---------------------------------------------------------------------------
# LongSpecConfig
# ---------------------------------------------------------------------------

class TestLongSpecConfig:
    def test_defaults(self):
        cfg = LongSpecConfig()
        assert cfg.gamma == 4
        assert cfg.hidden_size == 4096
        assert cfg.vocab_size == 32000
        assert cfg.max_context_len == 65536
        assert cfg.temperature == 1.0
        assert cfg.top_p == 1.0

    def test_custom(self):
        cfg = LongSpecConfig(gamma=6, hidden_size=512, vocab_size=100)
        assert cfg.gamma == 6
        assert cfg.hidden_size == 512
        assert cfg.vocab_size == 100

    @pytest.mark.parametrize("kwargs, match", [
        ({"gamma": 0},              "gamma"),
        ({"hidden_size": 0},        "hidden_size"),
        ({"vocab_size": 1},         "vocab_size"),
        ({"max_context_len": 0},    "max_context_len"),
        ({"temperature": 0},        "temperature"),
        ({"top_p": 0.0},            "top_p"),
        ({"top_p": 1.01},           "top_p"),
    ])
    def test_validation(self, kwargs, match):
        params = {"hidden_size": 64, "vocab_size": 20}
        params.update(kwargs)
        with pytest.raises(ValueError, match=match):
            LongSpecConfig(**params)


# ---------------------------------------------------------------------------
# LongSpecHead
# ---------------------------------------------------------------------------

class TestLongSpecHead:
    def test_output_shape(self):
        head = LongSpecHead(vocab_size=20, hidden_size=8)
        h    = np.ones(8, dtype=np.float32)
        out  = head.forward(h)
        assert out.shape == (20,)

    def test_forward_dtype(self):
        head = LongSpecHead(vocab_size=10, hidden_size=4)
        out  = head.forward(np.ones(4, dtype=np.float16))
        assert out.dtype == np.float32

    def test_load_weights(self):
        head = LongSpecHead(vocab_size=10, hidden_size=4, rng_seed=0)
        W1   = np.eye(4, dtype=np.float32)
        b1   = np.zeros(4, dtype=np.float32)
        W2   = np.zeros((10, 4), dtype=np.float32)
        b2   = np.arange(10, dtype=np.float32)
        head.load_weights(W1, b1, W2, b2)
        out  = head.forward(np.ones(4, dtype=np.float32))
        # b2 = [0..9] and W2=0, so output logits = arange after adding b2
        assert np.allclose(out, b2, atol=1e-5)

    def test_different_seeds_differ(self):
        h1 = LongSpecHead(vocab_size=20, hidden_size=8, rng_seed=0)
        h2 = LongSpecHead(vocab_size=20, hidden_size=8, rng_seed=99)
        x  = np.ones(8, dtype=np.float32)
        assert not np.allclose(h1.forward(x), h2.forward(x))

    def test_invalid_vocab(self):
        with pytest.raises(ValueError, match="vocab_size"):
            LongSpecHead(vocab_size=1, hidden_size=4)

    def test_invalid_hidden(self):
        with pytest.raises(ValueError, match="hidden_size"):
            LongSpecHead(vocab_size=10, hidden_size=0)


# ---------------------------------------------------------------------------
# LongSpecStats
# ---------------------------------------------------------------------------

class TestLongSpecStats:
    def test_defaults(self):
        s = LongSpecStats()
        assert s.total_tokens == 0
        assert s.draft_steps == 0
        assert s.accepted_total == 0
        assert s.rejected_total == 0

    def test_acceptance_rate_empty(self):
        assert LongSpecStats().acceptance_rate == 0.0

    def test_acceptance_rate(self):
        s = LongSpecStats(accepted_total=7, rejected_total=3)
        assert s.acceptance_rate == pytest.approx(0.7)

    def test_mean_accepted_per_step_empty(self):
        assert LongSpecStats().mean_accepted_per_step == 0.0

    def test_mean_accepted_per_step(self):
        s = LongSpecStats(accepted_total=9, draft_steps=3)
        assert s.mean_accepted_per_step == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# LongSpecDecoder helpers
# ---------------------------------------------------------------------------

VOCAB = 16
HIDDEN = 8


def _make_target(agree_tok: int):
    """Target function that always peaks at agree_tok."""
    def fn(ids):
        logits = np.full(VOCAB, -10.0, dtype=np.float32)
        logits[agree_tok] = 10.0
        return logits
    return fn


def _make_hidden():
    """Returns a fixed (HIDDEN,) hidden vector."""
    def fn(ids):
        return np.ones(HIDDEN, dtype=np.float32)
    return fn


def _make_head(agree_tok: int):
    """Draft head always peaks at agree_tok."""
    head = LongSpecHead(vocab_size=VOCAB, hidden_size=HIDDEN, rng_seed=0)
    W2   = np.zeros((VOCAB, HIDDEN), dtype=np.float32)
    b2   = np.full(VOCAB, -10.0, dtype=np.float32)
    b2[agree_tok] = 10.0
    head.load_weights(
        W1=np.eye(HIDDEN, dtype=np.float32),
        b1=np.zeros(HIDDEN, dtype=np.float32),
        W2=W2,
        b2=b2,
    )
    return head


# ---------------------------------------------------------------------------
# LongSpecDecoder — generate
# ---------------------------------------------------------------------------

class TestLongSpecDecoderGenerate:
    def test_generates_correct_count(self):
        cfg = LongSpecConfig(gamma=2, hidden_size=HIDDEN, vocab_size=VOCAB)
        dec = LongSpecDecoder(
            target_fn=_make_target(5),
            hidden_fn=_make_hidden(),
            head=_make_head(5),
            config=cfg,
            rng_seed=0,
        )
        out, stats = dec.generate([0, 1], max_new_tokens=8)
        assert stats.total_tokens == 8
        assert len(out) == 10

    def test_respects_max_new_tokens(self):
        cfg = LongSpecConfig(gamma=3, hidden_size=HIDDEN, vocab_size=VOCAB)
        dec = LongSpecDecoder(
            target_fn=_make_target(3),
            hidden_fn=_make_hidden(),
            head=_make_head(3),
            config=cfg,
            rng_seed=1,
        )
        _, stats = dec.generate([0], max_new_tokens=5)
        assert stats.total_tokens == 5

    def test_agreement_produces_acceptances(self):
        cfg = LongSpecConfig(
            gamma=3, hidden_size=HIDDEN, vocab_size=VOCAB, temperature=0.01
        )
        dec = LongSpecDecoder(
            target_fn=_make_target(9),
            hidden_fn=_make_hidden(),
            head=_make_head(9),
            config=cfg,
            rng_seed=0,
        )
        _, stats = dec.generate([0], max_new_tokens=12)
        assert stats.accepted_total > 0

    def test_disagreement_produces_rejections(self):
        cfg = LongSpecConfig(
            gamma=3, hidden_size=HIDDEN, vocab_size=VOCAB, temperature=0.01
        )
        dec = LongSpecDecoder(
            target_fn=_make_target(2),
            hidden_fn=_make_hidden(),
            head=_make_head(9),   # draft always says 9, target always says 2
            config=cfg,
            rng_seed=0,
        )
        _, stats = dec.generate([0], max_new_tokens=12)
        assert stats.rejected_total > 0

    def test_draft_steps_counted(self):
        cfg = LongSpecConfig(gamma=2, hidden_size=HIDDEN, vocab_size=VOCAB)
        head = LongSpecHead(vocab_size=VOCAB, hidden_size=HIDDEN)
        dec  = LongSpecDecoder(
            target_fn=_make_target(1),
            hidden_fn=_make_hidden(),
            head=head,
            config=cfg,
        )
        _, stats = dec.generate([0], max_new_tokens=6)
        assert stats.draft_steps > 0

    def test_empty_prompt(self):
        cfg = LongSpecConfig(gamma=2, hidden_size=HIDDEN, vocab_size=VOCAB)
        dec = LongSpecDecoder(
            target_fn=_make_target(0),
            hidden_fn=_make_hidden(),
            head=_make_head(0),
            config=cfg,
        )
        out, stats = dec.generate([], max_new_tokens=4)
        assert stats.total_tokens == 4
