"""tests/test_token_swift_unit.py — unit tests for squish/token_swift.py"""
import numpy as np
import pytest

from squish.token_swift import (
    TokenSwiftConfig,
    MultiTokenHead,
    PartialKVManager,
    TokenSwiftStats,
    TokenSwiftDecoder,
)


# ---------------------------------------------------------------------------
# TokenSwiftConfig
# ---------------------------------------------------------------------------

class TestTokenSwiftConfig:
    def test_defaults(self):
        cfg = TokenSwiftConfig()
        assert cfg.n_heads == 3
        assert cfg.window_size == 512
        assert cfg.ngram_penalty == 0.0
        assert cfg.ngram_n == 3
        assert cfg.vocab_size == 32000
        assert cfg.temperature == 1.0
        assert cfg.top_p == 1.0
        assert cfg.gamma == 3

    def test_custom(self):
        cfg = TokenSwiftConfig(n_heads=5, window_size=256, ngram_penalty=0.3)
        assert cfg.n_heads == 5
        assert cfg.window_size == 256
        assert cfg.ngram_penalty == pytest.approx(0.3)

    @pytest.mark.parametrize("kwargs, match", [
        ({"n_heads": 0},          "n_heads"),
        ({"window_size": 0},      "window_size"),
        ({"ngram_penalty": -0.1}, "ngram_penalty"),
        ({"ngram_n": 0},          "ngram_n"),
        ({"vocab_size": 1},       "vocab_size"),
        ({"temperature": 0},      "temperature"),
        ({"top_p": 0.0},          "top_p"),
        ({"top_p": 1.1},          "top_p"),
        ({"gamma": 0},            "gamma"),
    ])
    def test_validation(self, kwargs, match):
        params = {"vocab_size": 20}
        params.update(kwargs)
        with pytest.raises(ValueError, match=match):
            TokenSwiftConfig(**params)


# ---------------------------------------------------------------------------
# MultiTokenHead
# ---------------------------------------------------------------------------

class TestMultiTokenHead:
    def test_shape(self):
        heads = MultiTokenHead(n_heads=3, hidden_size=8, vocab_size=16)
        assert len(heads.weights) == 3
        assert heads.weights[0].shape == (16, 8)

    def test_predict_returns_k_arrays(self):
        heads  = MultiTokenHead(n_heads=3, hidden_size=8, vocab_size=16)
        result = heads.predict(np.ones(8, dtype=np.float32))
        assert len(result) == 3
        for arr in result:
            assert arr.shape == (16,)
            assert arr.dtype == np.float32

    def test_load_head_weights(self):
        heads = MultiTokenHead(n_heads=2, hidden_size=4, vocab_size=8)
        W     = np.zeros((8, 4), dtype=np.float32)
        b     = np.arange(8, dtype=np.float32)
        heads.load_head_weights(head_idx=0, W=W, b=b)
        out = heads.predict(np.ones(4, dtype=np.float32))
        assert np.allclose(out[0], b, atol=1e-5)  # W=0, so logits = b

    def test_load_head_weights_no_bias(self):
        heads = MultiTokenHead(n_heads=2, hidden_size=4, vocab_size=8)
        W     = np.zeros((8, 4), dtype=np.float32)
        heads.load_head_weights(head_idx=1, W=W)   # bias unchanged
        out   = heads.predict(np.ones(4, dtype=np.float32))
        assert np.allclose(out[1], heads.biases[1], atol=1e-5)

    def test_load_head_out_of_range(self):
        heads = MultiTokenHead(n_heads=2, hidden_size=4, vocab_size=8)
        with pytest.raises(IndexError):
            heads.load_head_weights(head_idx=2, W=np.zeros((8, 4)))

    @pytest.mark.parametrize("kwargs, match", [
        ({"n_heads": 0},     "n_heads"),
        ({"hidden_size": 0}, "hidden_size"),
        ({"vocab_size": 1},  "vocab_size"),
    ])
    def test_init_validation(self, kwargs, match):
        params = {"n_heads": 2, "hidden_size": 4, "vocab_size": 8}
        params.update(kwargs)
        with pytest.raises(ValueError, match=match):
            MultiTokenHead(**params)


# ---------------------------------------------------------------------------
# PartialKVManager
# ---------------------------------------------------------------------------

class TestPartialKVManager:
    def test_initial_state(self):
        mgr = PartialKVManager(prompt_len=10, window_size=3)
        assert mgr.total_len == 10
        assert mgr.window_positions == []
        assert mgr.frozen_positions == list(range(10))
        assert mgr.evict_fraction() == 0.0

    def test_add_tokens_updates_window(self):
        mgr = PartialKVManager(prompt_len=5, window_size=3)
        mgr.add_tokens(2)
        assert mgr.total_len == 7
        assert mgr.window_positions == [5, 6]  # both within window

    def test_window_rolls_past_capacity(self):
        mgr = PartialKVManager(prompt_len=5, window_size=3)
        mgr.add_tokens(5)
        # window covers [7, 8, 9]; positions [5, 6] now outside
        assert mgr.window_positions == [7, 8, 9]
        assert mgr.window_start == 7

    def test_evict_fraction_none_evicted(self):
        mgr = PartialKVManager(prompt_len=5, window_size=10)
        mgr.add_tokens(5)
        assert mgr.evict_fraction() == 0.0  # 5 gen, window=10, none evicted

    def test_evict_fraction_partial(self):
        mgr = PartialKVManager(prompt_len=5, window_size=3)
        mgr.add_tokens(5)
        # 5 generated, 2 outside window
        assert mgr.evict_fraction() == pytest.approx(2 / 5)

    def test_negative_add_clamps(self):
        mgr = PartialKVManager(prompt_len=5, window_size=3)
        mgr.add_tokens(-10)
        assert mgr._gen_len == 0

    def test_invalid_prompt_len(self):
        with pytest.raises(ValueError, match="prompt_len"):
            PartialKVManager(prompt_len=-1, window_size=3)

    def test_invalid_window_size(self):
        with pytest.raises(ValueError, match="window_size"):
            PartialKVManager(prompt_len=5, window_size=0)

    def test_zero_prompt_len(self):
        mgr = PartialKVManager(prompt_len=0, window_size=4)
        assert mgr.frozen_positions == []
        mgr.add_tokens(2)
        assert mgr.window_positions == [0, 1]


# ---------------------------------------------------------------------------
# TokenSwiftStats
# ---------------------------------------------------------------------------

class TestTokenSwiftStats:
    def test_defaults(self):
        s = TokenSwiftStats()
        assert s.total_tokens == 0
        assert s.penalty_applied == 0

    def test_acceptance_rate_zero(self):
        assert TokenSwiftStats().acceptance_rate == 0.0

    def test_acceptance_rate(self):
        s = TokenSwiftStats(accepted_total=6, rejected_total=2)
        assert s.acceptance_rate == pytest.approx(0.75)

    def test_mean_accepted_per_step(self):
        s = TokenSwiftStats(accepted_total=12, draft_steps=4)
        assert s.mean_accepted_per_step == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# TokenSwiftDecoder helpers
# ---------------------------------------------------------------------------

VOCAB   = 16
HIDDEN  = 8


def _fixed_target(agree_tok: int):
    def fn(ids):
        logits = np.full(VOCAB, -10.0, dtype=np.float32)
        logits[agree_tok] = 10.0
        return logits
    return fn


def _fixed_hidden():
    return lambda ids: np.ones(HIDDEN, dtype=np.float32)


def _make_heads(agree_tok: int, n: int = 3):
    heads = MultiTokenHead(n_heads=n, hidden_size=HIDDEN, vocab_size=VOCAB)
    for i in range(n):
        W = np.zeros((VOCAB, HIDDEN), dtype=np.float32)
        b = np.full(VOCAB, -10.0, dtype=np.float32)
        b[agree_tok] = 10.0
        heads.load_head_weights(i, W, b)
    return heads


# ---------------------------------------------------------------------------
# TokenSwiftDecoder — generate
# ---------------------------------------------------------------------------

class TestTokenSwiftDecoderGenerate:
    def test_generates_count_with_heads(self):
        cfg  = TokenSwiftConfig(n_heads=3, window_size=4, vocab_size=VOCAB)
        dec  = TokenSwiftDecoder(
            target_fn=_fixed_target(5),
            config=cfg,
            hidden_fn=_fixed_hidden(),
            heads=_make_heads(5),
            rng_seed=0,
        )
        out, stats = dec.generate([0, 1], max_new_tokens=6)
        assert stats.total_tokens == 6
        assert len(out) == 8

    def test_generates_count_without_heads(self):
        """Fallback (no heads) path: uses target_fn for drafting."""
        cfg = TokenSwiftConfig(n_heads=2, window_size=4, vocab_size=VOCAB)
        dec = TokenSwiftDecoder(
            target_fn=_fixed_target(3),
            config=cfg,
            rng_seed=0,
        )
        _, stats = dec.generate([0], max_new_tokens=4)
        assert stats.total_tokens == 4

    def test_respects_max_new_tokens(self):
        cfg = TokenSwiftConfig(n_heads=3, window_size=4, vocab_size=VOCAB)
        dec = TokenSwiftDecoder(
            target_fn=_fixed_target(2),
            config=cfg,
            hidden_fn=_fixed_hidden(),
            heads=_make_heads(2),
            rng_seed=42,
        )
        _, stats = dec.generate([0], max_new_tokens=7)
        assert stats.total_tokens == 7

    def test_ngram_penalty_fires(self):
        """With high penalty and repeated token, penalty_applied should > 0."""
        agree_tok = 4
        cfg = TokenSwiftConfig(
            n_heads=2, window_size=8, vocab_size=VOCAB,
            ngram_penalty=5.0, ngram_n=2, gamma=2,
        )
        dec = TokenSwiftDecoder(
            target_fn=_fixed_target(agree_tok),
            config=cfg,
            rng_seed=0,
        )
        # Seed ids so the bigram has a history
        _, stats = dec.generate([agree_tok] * 5, max_new_tokens=10)
        assert stats.penalty_applied > 0

    def test_ngram_penalty_disabled(self):
        cfg = TokenSwiftConfig(
            n_heads=2, window_size=8, vocab_size=VOCAB, ngram_penalty=0.0
        )
        dec = TokenSwiftDecoder(
            target_fn=_fixed_target(6),
            config=cfg,
            rng_seed=0,
        )
        _, stats = dec.generate([0], max_new_tokens=6)
        assert stats.penalty_applied == 0

    def test_draft_steps_tracked(self):
        cfg = TokenSwiftConfig(n_heads=2, window_size=4, vocab_size=VOCAB)
        dec = TokenSwiftDecoder(
            target_fn=_fixed_target(1),
            config=cfg,
            rng_seed=0,
        )
        _, stats = dec.generate([0], max_new_tokens=6)
        assert stats.draft_steps > 0

    def test_agreement_high_acceptance(self):
        cfg = TokenSwiftConfig(
            n_heads=3, window_size=4, vocab_size=VOCAB, temperature=0.01
        )
        dec = TokenSwiftDecoder(
            target_fn=_fixed_target(8),
            config=cfg,
            hidden_fn=_fixed_hidden(),
            heads=_make_heads(8),
            rng_seed=0,
        )
        _, stats = dec.generate([0], max_new_tokens=12)
        assert stats.accepted_total > 0
        assert stats.acceptance_rate > 0.0
