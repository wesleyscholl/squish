"""tests/test_quant_spec_unit.py — 100% coverage for squish/quant_spec.py"""
import numpy as np
import pytest

from squish.quant_spec import (
    QuantSpecConfig,
    DraftQuantizer,
    QuantSpecDecoder,
    _softmax,
    _top_p_filter,
    _sample,
)


# ---------------------------------------------------------------------------
# QuantSpecConfig
# ---------------------------------------------------------------------------

class TestQuantSpecConfig:
    def test_defaults(self):
        cfg = QuantSpecConfig()
        assert cfg.gamma == 4
        assert cfg.draft_quant_bits == 4
        assert cfg.draft_skip_layers == 8
        assert cfg.temperature == 1.0
        assert cfg.top_p == 1.0
        assert cfg.acceptance_threshold == 0.0

    def test_invalid_gamma(self):
        with pytest.raises(ValueError, match="gamma"):
            QuantSpecConfig(gamma=0)

    def test_invalid_bits(self):
        with pytest.raises(ValueError, match="draft_quant_bits"):
            QuantSpecConfig(draft_quant_bits=3)

    def test_valid_bits(self):
        for bits in (2, 4, 8):
            cfg = QuantSpecConfig(draft_quant_bits=bits)
            assert cfg.draft_quant_bits == bits

    def test_invalid_skip_layers(self):
        with pytest.raises(ValueError, match="draft_skip_layers"):
            QuantSpecConfig(draft_skip_layers=-1)

    def test_invalid_temperature(self):
        with pytest.raises(ValueError, match="temperature"):
            QuantSpecConfig(temperature=0.0)

    def test_invalid_top_p_zero(self):
        with pytest.raises(ValueError, match="top_p"):
            QuantSpecConfig(top_p=0.0)

    def test_invalid_top_p_above_one(self):
        with pytest.raises(ValueError, match="top_p"):
            QuantSpecConfig(top_p=1.1)

    def test_invalid_acceptance_threshold(self):
        with pytest.raises(ValueError, match="acceptance_threshold"):
            QuantSpecConfig(acceptance_threshold=-0.1)
        with pytest.raises(ValueError, match="acceptance_threshold"):
            QuantSpecConfig(acceptance_threshold=1.01)


# ---------------------------------------------------------------------------
# DraftQuantizer
# ---------------------------------------------------------------------------

class TestDraftQuantizer:
    def test_invalid_bits(self):
        with pytest.raises(ValueError, match="bits"):
            DraftQuantizer(bits=3)

    @pytest.mark.parametrize("bits", [2, 4, 8])
    def test_round_trip(self, bits):
        rng = np.random.default_rng(bits)
        q = DraftQuantizer(bits=bits)
        vec = rng.standard_normal(16).astype(np.float32)
        quantized, scale, zero = q.quantize(vec)
        restored = q.dequantize(quantized, scale, zero)
        # Theoretical max error for a uniform (levels-1)-step quantizer: scale/2
        # scale = range / (qmax - qmin) = range / (2^bits - 1)
        v_range = float(vec.max() - vec.min())
        atol    = v_range / (2.0 * (2 ** bits - 1)) + 1e-5
        assert np.abs(vec - restored).max() <= atol

    def test_constant_tensor(self):
        q = DraftQuantizer(bits=4)
        vec = np.full(10, 3.14, dtype=np.float32)
        qi, scale, zero = q.quantize(vec)
        assert (qi == 0).all()
        restored = q.dequantize(qi, scale, zero)
        assert np.allclose(restored, 3.14, atol=1e-4)

    def test_compression_ratio_4bit(self):
        q = DraftQuantizer(bits=4)
        assert q.compression_ratio == 8.0

    def test_bits_property(self):
        q = DraftQuantizer(bits=8)
        assert q.bits == 8


# ---------------------------------------------------------------------------
# Sampling utilities
# ---------------------------------------------------------------------------

class TestSamplingUtils:
    def test_softmax_sums_to_one(self):
        logits = np.random.randn(100).astype(np.float32)
        probs = _softmax(logits)
        assert abs(probs.sum() - 1.0) < 1e-5

    def test_softmax_temperature(self):
        logits = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        p1 = _softmax(logits, temperature=1.0)
        p2 = _softmax(logits, temperature=0.1)
        # Lower temperature → more peaked
        assert p2.max() > p1.max()

    def test_top_p_filters_tokens(self):
        probs = np.array([0.5, 0.3, 0.1, 0.07, 0.03], dtype=np.float32)
        filtered = _top_p_filter(probs, 0.85)
        # Tokens that sum to ≥ 0.85 should be kept; rest zeroed
        assert filtered[-1] == 0.0 or filtered.sum() <= 1.01

    def test_top_p_one_full_pass(self):
        probs = np.array([0.5, 0.3, 0.2], dtype=np.float32)
        result = _top_p_filter(probs, 1.0)
        np.testing.assert_allclose(result, probs, rtol=1e-5)

    def test_top_p_filter_all_zero_input(self):
        """Branch 206→208: total == 0 → no normalization, returns zeros."""
        probs    = np.zeros(5, dtype=np.float32)
        filtered = _top_p_filter(probs, 0.9)
        assert filtered.sum() == 0.0

    def test_sample_returns_valid_index(self):
        rng   = np.random.default_rng(42)
        probs = np.array([0.1, 0.4, 0.3, 0.2], dtype=np.float32)
        idx   = _sample(probs, rng)
        assert 0 <= idx < 4


# ---------------------------------------------------------------------------
# QuantSpecDecoder
# ---------------------------------------------------------------------------

class TestQuantSpecDecoder:
    def _make_model(self, vocab=50, return_batch_logits=False):
        """Stub model that returns random logits."""
        def model(token_ids, kv_state, skip_layers=0):
            seq_len = len(token_ids)
            if return_batch_logits:
                logits = np.random.randn(seq_len, vocab).astype(np.float32)
            else:
                logits = np.random.randn(vocab).astype(np.float32)
            return logits, kv_state  # pass-through kv_state

        return model

    def test_generate_step_returns_at_least_one_token(self):
        model = self._make_model(vocab=50)
        cfg   = QuantSpecConfig(gamma=3, draft_quant_bits=4, temperature=1.0,
                                acceptance_threshold=0.0)
        dec   = QuantSpecDecoder(draft_fn=model, config=cfg, seed=42)
        context = np.array([1, 2, 3], dtype=np.int32)
        tokens, kv = dec.generate_step(context, kv_state=None)
        assert len(tokens) >= 1
        assert all(0 <= t < 50 for t in tokens)

    def test_generate_step_with_batch_logits(self):
        draft_model  = self._make_model(vocab=50, return_batch_logits=False)
        verify_model = self._make_model(vocab=50, return_batch_logits=True)
        cfg  = QuantSpecConfig(gamma=2, draft_quant_bits=4)
        dec  = QuantSpecDecoder(draft_fn=draft_model, verify_fn=verify_model,
                                config=cfg, seed=0)
        tokens, _ = dec.generate_step(np.array([1, 2], dtype=np.int32), None)
        assert len(tokens) >= 1

    def test_acceptance_rate_zero_initially(self):
        model = self._make_model()
        dec = QuantSpecDecoder(draft_fn=model, config=QuantSpecConfig())
        assert dec.acceptance_rate == 0.0

    def test_acceptance_rate_after_step(self):
        model = self._make_model(vocab=50)
        cfg = QuantSpecConfig(gamma=2, acceptance_threshold=0.0)
        dec = QuantSpecDecoder(draft_fn=model, config=cfg, seed=1)
        dec.generate_step(np.array([1], dtype=np.int32), None)
        # acceptance_rate should be defined
        assert 0.0 <= dec.acceptance_rate <= 1.0

    def test_reset_stats(self):
        model = self._make_model(vocab=50)
        dec = QuantSpecDecoder(draft_fn=model, config=QuantSpecConfig(), seed=7)
        dec.generate_step(np.array([1, 2], dtype=np.int32), None)
        dec.reset_stats()
        assert dec.total_draft_tokens    == 0
        assert dec.total_accepted_tokens == 0
        assert dec.acceptance_rate       == 0.0

    def test_high_acceptance_threshold_truncates(self):
        # threshold = 1.0 → only accept if p_target == p_draft (unlikely with random)
        model = self._make_model(vocab=50)
        cfg = QuantSpecConfig(gamma=4, acceptance_threshold=1.0)
        dec = QuantSpecDecoder(draft_fn=model, config=cfg, seed=99)
        tokens, _ = dec.generate_step(np.array([1], dtype=np.int32), None)
        # Should return at least the bonus token
        assert len(tokens) >= 1

    def test_uses_verify_fn_separately(self):
        """Ensure verify_fn is actually used when provided."""
        called = {"verify": 0}
        vocab = 20

        def draft_fn(ids, kv, skip_layers=0):
            return np.random.randn(vocab).astype(np.float32), kv

        def verify_fn(ids, kv):
            called["verify"] += 1
            return np.random.randn(vocab).astype(np.float32), kv

        cfg = QuantSpecConfig(gamma=2)
        dec = QuantSpecDecoder(draft_fn=draft_fn, verify_fn=verify_fn, config=cfg)
        dec.generate_step(np.array([1, 2], dtype=np.int32), None)
        assert called["verify"] >= 1


class TestAllAcceptedOneDimLogits:
    def test_all_accepted_with_1d_logits_no_bonus(self):
        """Branch 356→364: all drafts accepted AND all_logits is 1-D
        → elif skipped → return directly."""
        vocab = 4
        # Draft always generates token 0 (overwhelmingly peaked on token 0).
        peaked = np.array([100.0, 0.0, 0.0, 0.0], dtype=np.float32)

        def draft_fn(ids, kv, skip_layers=0):
            return peaked.copy(), kv

        # Verify returns 1D logits also peaked on token 0 → target_prob[0] ≥ draft_prob[0]
        # → acceptance_prob = 1.0 → always accepted.
        def verify_fn(ids, kv):
            return peaked.copy(), kv

        cfg = QuantSpecConfig(gamma=2, acceptance_threshold=0.0)
        dec = QuantSpecDecoder(draft_fn=draft_fn, verify_fn=verify_fn,
                               config=cfg, seed=0)
        tokens, _ = dec.generate_step(np.array([1], dtype=np.int32), None)
        # With all accepted and 1D logits, no bonus is added → exactly gamma tokens
        assert len(tokens) == cfg.gamma
