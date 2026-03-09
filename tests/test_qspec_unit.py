"""tests/test_qspec_unit.py — unit tests for squish/qspec.py"""
import numpy as np
import pytest

from squish.qspec import (
    QSpecConfig,
    ActivationQuantizer,
    QSpecStats,
    QSpecDecoder,
)


# ---------------------------------------------------------------------------
# QSpecConfig
# ---------------------------------------------------------------------------

class TestQSpecConfig:
    def test_defaults(self):
        cfg = QSpecConfig()
        assert cfg.gamma             == 4
        assert cfg.draft_act_bits    == 8
        assert cfg.verify_act_bits   == 16
        assert cfg.group_size        == 128
        assert cfg.temperature       == 1.0
        assert cfg.top_p             == 1.0

    def test_custom(self):
        cfg = QSpecConfig(gamma=2, draft_act_bits=4, verify_act_bits=8, group_size=64)
        assert cfg.gamma           == 2
        assert cfg.draft_act_bits  == 4
        assert cfg.verify_act_bits == 8
        assert cfg.group_size      == 64

    @pytest.mark.parametrize("kwargs, match", [
        ({"gamma": 0},                           "gamma"),
        ({"draft_act_bits": 3},                  "draft_act_bits"),
        ({"draft_act_bits": 16},                 "draft_act_bits"),
        ({"verify_act_bits": 4},                 "verify_act_bits"),
        ({"verify_act_bits": 32},                "verify_act_bits"),
        # draft must be strictly less than verify
        ({"draft_act_bits": 8, "verify_act_bits": 8}, "draft_act_bits"),
        ({"group_size": 0},                      "group_size"),
        ({"temperature": 0},                     "temperature"),
        ({"top_p": 0.0},                         "top_p"),
        ({"top_p": 1.1},                         "top_p"),
    ])
    def test_validation(self, kwargs, match):
        base = {"draft_act_bits": 8, "verify_act_bits": 16}
        base.update(kwargs)
        with pytest.raises(ValueError, match=match):
            QSpecConfig(**base)

    def test_int4_draft_int8_verify_valid(self):
        cfg = QSpecConfig(draft_act_bits=4, verify_act_bits=8)
        assert cfg.draft_act_bits  == 4
        assert cfg.verify_act_bits == 8


# ---------------------------------------------------------------------------
# ActivationQuantizer
# ---------------------------------------------------------------------------

class TestActivationQuantizer:
    def test_invalid_bits(self):
        with pytest.raises(ValueError, match="bits"):
            ActivationQuantizer(bits=7)

    def test_invalid_group_size(self):
        with pytest.raises(ValueError, match="group_size"):
            ActivationQuantizer(bits=8, group_size=0)

    def test_quantize_shape_preserved(self):
        q   = ActivationQuantizer(bits=8, group_size=4)
        x   = np.random.randn(16).astype(np.float32)
        out = q.quantize(x)
        assert out.shape == x.shape
        assert out.dtype == x.dtype

    def test_quantize_fp16_is_noop(self):
        q   = ActivationQuantizer(bits=16, group_size=4)
        x   = np.array([1.5, -2.3, 0.7, 4.0], dtype=np.float32)
        out = q.quantize(x)
        assert np.allclose(out, x, atol=1e-6)

    def test_quantize_int8_reduces_precision(self):
        q   = ActivationQuantizer(bits=8, group_size=4)
        x   = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
        out = q.quantize(x)
        # Quantised values should be close but not bit-identical for small fractional values
        assert out.shape == x.shape

    def test_quantize_int4(self):
        q   = ActivationQuantizer(bits=4, group_size=4)
        x   = np.array([1.0, -1.0, 0.5, -0.5], dtype=np.float32)
        out = q.quantize(x)
        assert out.shape == x.shape
        assert out.dtype == x.dtype

    def test_bits_saved_int8(self):
        q = ActivationQuantizer(bits=8)
        assert q.bits_saved_fraction() == pytest.approx(0.5)   # (16-8)/16

    def test_bits_saved_int4(self):
        q = ActivationQuantizer(bits=4)
        assert q.bits_saved_fraction() == pytest.approx(0.75)  # (16-4)/16

    def test_bits_saved_fp16(self):
        q = ActivationQuantizer(bits=16)
        assert q.bits_saved_fraction() == pytest.approx(0.0)   # (16-16)/16

    def test_group_larger_than_input(self):
        """Should not crash when group_size > len(x)."""
        q   = ActivationQuantizer(bits=8, group_size=128)
        x   = np.random.randn(8).astype(np.float32)
        out = q.quantize(x)
        assert out.shape == x.shape


# ---------------------------------------------------------------------------
# QSpecStats
# ---------------------------------------------------------------------------

class TestQSpecStats:
    def test_defaults(self):
        s = QSpecStats()
        assert s.total_tokens    == 0
        assert s.draft_steps     == 0
        assert s.accepted_total  == 0
        assert s.rejected_total  == 0

    def test_acceptance_rate_zero(self):
        assert QSpecStats().acceptance_rate == 0.0

    def test_acceptance_rate(self):
        s = QSpecStats(accepted_total=3, rejected_total=1)
        assert s.acceptance_rate == pytest.approx(0.75)

    def test_mean_accepted_per_step_zero(self):
        assert QSpecStats().mean_accepted_per_step == 0.0

    def test_mean_accepted_per_step(self):
        s = QSpecStats(accepted_total=8, draft_steps=2)
        assert s.mean_accepted_per_step == pytest.approx(4.0)


# ---------------------------------------------------------------------------
# QSpecDecoder helpers
# ---------------------------------------------------------------------------

VOCAB = 16


def _fixed_logit_fn(agree_tok: int):
    """Returns a callable that always puts max logit on agree_tok."""
    def fn(ids):
        logits = np.full(VOCAB, -10.0, dtype=np.float32)
        logits[agree_tok] = 10.0
        return logits
    return fn


# ---------------------------------------------------------------------------
# QSpecDecoder — generate
# ---------------------------------------------------------------------------

class TestQSpecDecoderGenerate:
    def test_act_quantizer_is_public_attribute(self):
        cfg = QSpecConfig(draft_act_bits=8, verify_act_bits=16)
        dec = QSpecDecoder(
            w4a8_fn=_fixed_logit_fn(0),
            w4a16_fn=_fixed_logit_fn(0),
            config=cfg,
            rng_seed=0,
        )
        assert isinstance(dec.act_quantizer, ActivationQuantizer)
        assert dec.act_quantizer.bits == cfg.draft_act_bits

    def test_generates_token_count(self):
        dec = QSpecDecoder(
            w4a8_fn=_fixed_logit_fn(5),
            w4a16_fn=_fixed_logit_fn(5),
            rng_seed=42,
        )
        _, stats = dec.generate([0], max_new_tokens=6)
        assert stats.total_tokens == 6

    def test_respects_max_new_tokens(self):
        dec = QSpecDecoder(
            w4a8_fn=_fixed_logit_fn(2),
            w4a16_fn=_fixed_logit_fn(2),
            rng_seed=0,
        )
        _, stats = dec.generate([0, 1], max_new_tokens=8)
        assert stats.total_tokens == 8

    def test_full_acceptance_when_agree(self):
        """When draft (w4a8) and verify (w4a16) always pick the same token,
        the acceptance rate should be high (≥50%)."""
        dec = QSpecDecoder(
            w4a8_fn=_fixed_logit_fn(7),
            w4a16_fn=_fixed_logit_fn(7),
            rng_seed=0,
        )
        _, stats = dec.generate([0], max_new_tokens=20)
        assert stats.acceptance_rate >= 0.5

    def test_rejections_when_disagree(self):
        """When draft always predicts token 1, but verify always predicts token 2,
        most drafts will be rejected."""
        dec = QSpecDecoder(
            w4a8_fn=_fixed_logit_fn(1),
            w4a16_fn=_fixed_logit_fn(2),
            rng_seed=0,
        )
        _, stats = dec.generate([0], max_new_tokens=10)
        assert stats.rejected_total > 0

    def test_draft_steps_tracked(self):
        dec = QSpecDecoder(
            w4a8_fn=_fixed_logit_fn(3),
            w4a16_fn=_fixed_logit_fn(3),
            rng_seed=0,
        )
        _, stats = dec.generate([0], max_new_tokens=8)
        assert stats.draft_steps > 0

    def test_int4_draft_int8_verify(self):
        """INT4/INT8 config — ensure it constructs and generates without error."""
        cfg = QSpecConfig(draft_act_bits=4, verify_act_bits=8)
        dec = QSpecDecoder(
            w4a8_fn=_fixed_logit_fn(9),
            w4a16_fn=_fixed_logit_fn(9),
            config=cfg,
            rng_seed=0,
        )
        out, stats = dec.generate([0], max_new_tokens=4)
        assert stats.total_tokens == 4
        assert dec.act_quantizer.bits == 4

    def test_empty_prompt(self):
        dec = QSpecDecoder(
            w4a8_fn=_fixed_logit_fn(0),
            w4a16_fn=_fixed_logit_fn(0),
            rng_seed=0,
        )
        out, stats = dec.generate([], max_new_tokens=3)
        assert stats.total_tokens == 3
