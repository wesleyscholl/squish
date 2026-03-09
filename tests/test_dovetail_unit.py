"""tests/test_dovetail_unit.py — unit tests for squish/dovetail.py"""
import numpy as np
import pytest

from squish.dovetail import (
    DovetailConfig,
    DovetailDraftRunner,
    DovetailCPUVerifier,
    DovetailStats,
    DovetailDecoder,
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


def _make_decoder(agree_tok: int = 5, gamma: int = 4, rng_seed: int = 0):
    cfg = MirrorCfg = DovetailConfig(gamma=gamma)
    runner = DovetailDraftRunner(_fixed_fn(agree_tok), cfg, rng_seed=rng_seed)
    verifier = DovetailCPUVerifier(_fixed_fn(agree_tok), cfg, rng_seed=rng_seed + 1)
    return DovetailDecoder(runner, verifier, cfg, rng_seed=rng_seed + 2)


# ---------------------------------------------------------------------------
# DovetailConfig
# ---------------------------------------------------------------------------

class TestDovetailConfig:
    def test_defaults(self):
        cfg = DovetailConfig()
        assert cfg.gamma == 4
        assert cfg.temperature == 1.0
        assert cfg.top_p == 1.0

    def test_custom(self):
        cfg = DovetailConfig(gamma=6, temperature=0.7, top_p=0.95)
        assert cfg.gamma == 6

    @pytest.mark.parametrize("kwargs, match", [
        ({"gamma": 0},       "gamma"),
        ({"temperature": 0}, "temperature"),
        ({"top_p": 0.0},     "top_p"),
        ({"top_p": 1.1},     "top_p"),
    ])
    def test_validation(self, kwargs, match):
        with pytest.raises(ValueError, match=match):
            DovetailConfig(**kwargs)


# ---------------------------------------------------------------------------
# DovetailDraftRunner
# ---------------------------------------------------------------------------

class TestDovetailDraftRunner:
    def test_run_returns_correct_length(self):
        cfg = DovetailConfig()
        runner = DovetailDraftRunner(_fixed_fn(3), cfg, rng_seed=0)
        tokens, probs = runner.run([0, 1], gamma=4)
        assert len(tokens) == 4
        assert len(probs) == 4

    def test_probs_sum_to_one(self):
        cfg = DovetailConfig()
        runner = DovetailDraftRunner(_fixed_fn(2), cfg, rng_seed=0)
        _, probs = runner.run([0], gamma=3)
        for p in probs:
            assert abs(p.sum() - 1.0) < 1e-5

    def test_preferred_token_at_low_temp(self):
        cfg = DovetailConfig(temperature=0.01)
        runner = DovetailDraftRunner(_fixed_fn(7), cfg, rng_seed=0)
        tokens, _ = runner.run([0], gamma=10)
        assert all(t == 7 for t in tokens)

    def test_invalid_draft_fn(self):
        cfg = DovetailConfig()
        with pytest.raises(TypeError):
            DovetailDraftRunner("not_callable", cfg)

    def test_zero_gamma(self):
        cfg = DovetailConfig()
        runner = DovetailDraftRunner(_fixed_fn(1), cfg, rng_seed=0)
        tokens, probs = runner.run([0], gamma=0)
        assert tokens == []
        assert probs == []


# ---------------------------------------------------------------------------
# DovetailCPUVerifier
# ---------------------------------------------------------------------------

class TestDovetailCPUVerifier:
    def test_verify_one_shape(self):
        cfg = DovetailConfig()
        ver = DovetailCPUVerifier(_fixed_fn(4), cfg, rng_seed=0)
        tok, probs = ver.verify_one([0, 1])
        assert isinstance(tok, int)
        assert probs.shape == (VOCAB,)
        assert abs(probs.sum() - 1.0) < 1e-5

    def test_preferred_token_at_low_temp(self):
        cfg = DovetailConfig(temperature=0.01)
        ver = DovetailCPUVerifier(_fixed_fn(6), cfg, rng_seed=0)
        tokens = [ver.verify_one([0])[0] for _ in range(20)]
        assert all(t == 6 for t in tokens)

    def test_invalid_target_fn(self):
        cfg = DovetailConfig()
        with pytest.raises(TypeError):
            DovetailCPUVerifier(42, cfg)


# ---------------------------------------------------------------------------
# DovetailStats
# ---------------------------------------------------------------------------

class TestDovetailStats:
    def test_defaults(self):
        s = DovetailStats()
        assert s.total_tokens == 0
        assert s.draft_steps == 0

    def test_acceptance_rate_zero(self):
        assert DovetailStats().acceptance_rate == 0.0

    def test_acceptance_rate(self):
        s = DovetailStats(accepted_total=3, rejected_total=1)
        assert s.acceptance_rate == pytest.approx(0.75)

    def test_mean_accepted_per_step_zero(self):
        assert DovetailStats().mean_accepted_per_step == 0.0

    def test_mean_accepted_per_step(self):
        s = DovetailStats(accepted_total=8, draft_steps=2)
        assert s.mean_accepted_per_step == pytest.approx(4.0)


# ---------------------------------------------------------------------------
# DovetailDecoder
# ---------------------------------------------------------------------------

class TestDovetailDecoderGenerate:
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
        out, stats = dec.generate([0, 1, 2], max_new_tokens=5)
        assert len(out) == 3 + 5

    def test_draft_steps_tracked(self):
        dec = _make_decoder()
        _, stats = dec.generate([0], max_new_tokens=6)
        assert stats.draft_steps > 0

    def test_agreement_gives_high_acceptance(self):
        cfg = DovetailConfig(gamma=4, temperature=0.01)
        runner = DovetailDraftRunner(_fixed_fn(8), cfg, rng_seed=0)
        ver = DovetailCPUVerifier(_fixed_fn(8), cfg, rng_seed=1)
        dec = DovetailDecoder(runner, ver, cfg, rng_seed=2)
        _, stats = dec.generate([0], max_new_tokens=20)
        assert stats.acceptance_rate > 0.5

    def test_disagreement_causes_rejections(self):
        cfg = DovetailConfig(gamma=2)
        runner = DovetailDraftRunner(_fixed_fn(1), cfg, rng_seed=0)
        ver = DovetailCPUVerifier(_fixed_fn(2), cfg, rng_seed=1)
        dec = DovetailDecoder(runner, ver, cfg, rng_seed=2)
        _, stats = dec.generate([0], max_new_tokens=10)
        assert stats.rejected_total > 0

    def test_empty_prompt(self):
        dec = _make_decoder()
        out, stats = dec.generate([], max_new_tokens=4)
        assert stats.total_tokens == 4
        assert len(out) == 4

    def test_default_config_in_decoder(self):
        cfg = DovetailConfig()
        runner = DovetailDraftRunner(_fixed_fn(3), cfg)
        ver = DovetailCPUVerifier(_fixed_fn(3), cfg)
        dec = DovetailDecoder(runner, ver)   # no explicit config
        _, stats = dec.generate([0], max_new_tokens=3)
        assert stats.total_tokens == 3

    def test_single_gamma(self):
        cfg = DovetailConfig(gamma=1)
        runner = DovetailDraftRunner(_fixed_fn(5), cfg, rng_seed=0)
        ver = DovetailCPUVerifier(_fixed_fn(5), cfg, rng_seed=1)
        dec = DovetailDecoder(runner, ver, cfg, rng_seed=2)
        _, stats = dec.generate([0], max_new_tokens=5)
        assert stats.total_tokens == 5
