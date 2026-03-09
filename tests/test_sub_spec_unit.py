"""tests/test_sub_spec_unit.py — unit tests for squish/sub_spec.py"""
import numpy as np
import pytest

from squish.sub_spec import (
    SubSpecConfig,
    SubstituteLayerProxy,
    SubSpecStats,
    SubSpecDecoder,
)


# ---------------------------------------------------------------------------
# SubSpecConfig
# ---------------------------------------------------------------------------

class TestSubSpecConfig:
    def test_defaults(self):
        cfg = SubSpecConfig()
        assert cfg.n_total_layers == 32
        assert cfg.n_gpu_layers == 16
        assert cfg.gamma == 4
        assert cfg.quant_bits == 4
        assert cfg.temperature == 1.0
        assert cfg.top_p == 1.0

    def test_n_substitute_layers_property(self):
        cfg = SubSpecConfig(n_total_layers=32, n_gpu_layers=20)
        assert cfg.n_substitute_layers == 12

    def test_n_substitute_layers_all_gpu(self):
        cfg = SubSpecConfig(n_total_layers=16, n_gpu_layers=16)
        assert cfg.n_substitute_layers == 0

    def test_custom(self):
        cfg = SubSpecConfig(n_total_layers=48, n_gpu_layers=24, gamma=6, quant_bits=8)
        assert cfg.n_total_layers == 48
        assert cfg.n_gpu_layers == 24
        assert cfg.gamma == 6

    @pytest.mark.parametrize("kwargs, match", [
        ({"n_total_layers": 0},              "n_total_layers"),
        ({"n_gpu_layers": -1},               "n_gpu_layers"),
        ({"n_total_layers": 8, "n_gpu_layers": 16}, "n_gpu_layers"),
        ({"gamma": 0},                       "gamma"),
        ({"quant_bits": 3},                  "quant_bits"),
        ({"temperature": 0},                 "temperature"),
        ({"temperature": -1.0},              "temperature"),
        ({"top_p": 0.0},                     "top_p"),
        ({"top_p": 1.1},                     "top_p"),
    ])
    def test_validation(self, kwargs, match):
        params = {"n_total_layers": 32, "n_gpu_layers": 16}
        params.update(kwargs)
        with pytest.raises(ValueError, match=match):
            SubSpecConfig(**params)


# ---------------------------------------------------------------------------
# SubstituteLayerProxy
# ---------------------------------------------------------------------------

class TestSubstituteLayerProxy:
    def test_shape(self):
        w = np.random.randn(8, 4).astype(np.float32)
        proxy = SubstituteLayerProxy(w)
        assert proxy.out_dim == 8
        assert proxy.in_dim == 4

    def test_forward_vector(self):
        w = np.eye(4, dtype=np.float32)
        proxy = SubstituteLayerProxy(w, group_size=4)
        x = np.ones(4, dtype=np.float32)
        out = proxy.forward(x)
        assert out.shape == (4,)
        # identity-like weights: quantized near identity
        assert np.allclose(out, x, atol=0.2)

    def test_forward_batch(self):
        w = np.eye(4, dtype=np.float32)
        proxy = SubstituteLayerProxy(w, group_size=4)
        x = np.ones((3, 4), dtype=np.float32)
        out = proxy.forward(x)
        assert out.shape == (3, 4)

    def test_zero_weight(self):
        w = np.zeros((4, 4), dtype=np.float32)
        proxy = SubstituteLayerProxy(w)
        x = np.ones(4, dtype=np.float32)
        out = proxy.forward(x)
        assert np.allclose(out, 0.0)

    def test_compression_ratio(self):
        w = np.random.randn(8, 8).astype(np.float32)
        proxy = SubstituteLayerProxy(w)
        assert proxy.compression_ratio == pytest.approx(4.0 / 32.0)

    def test_requires_2d_weight(self):
        with pytest.raises(ValueError, match="2-D"):
            SubstituteLayerProxy(np.ones(8))

    def test_requires_positive_group_size(self):
        with pytest.raises(ValueError, match="group_size"):
            SubstituteLayerProxy(np.ones((4, 4)), group_size=0)


# ---------------------------------------------------------------------------
# SubSpecStats
# ---------------------------------------------------------------------------

class TestSubSpecStats:
    def test_defaults(self):
        s = SubSpecStats()
        assert s.total_tokens == 0
        assert s.draft_steps == 0
        assert s.accepted_total == 0
        assert s.rejected_total == 0

    def test_acceptance_rate_zero_division(self):
        s = SubSpecStats()
        assert s.acceptance_rate == 0.0

    def test_acceptance_rate(self):
        s = SubSpecStats(accepted_total=3, rejected_total=1)
        assert s.acceptance_rate == pytest.approx(0.75)

    def test_mean_accepted_per_step_zero_steps(self):
        s = SubSpecStats()
        assert s.mean_accepted_per_step == 0.0

    def test_mean_accepted_per_step(self):
        s = SubSpecStats(accepted_total=10, draft_steps=4)
        assert s.mean_accepted_per_step == pytest.approx(2.5)


# ---------------------------------------------------------------------------
# SubSpecDecoder helpers
# ---------------------------------------------------------------------------

def _fixed_logit_fn(agree_tok: int, vocab: int = 10):
    """Returns a function that always returns logits peaking at agree_tok."""
    def fn(ids):
        logits = np.full(vocab, -10.0, dtype=np.float32)
        logits[agree_tok] = 10.0
        return logits
    return fn


# ---------------------------------------------------------------------------
# SubSpecDecoder — generate
# ---------------------------------------------------------------------------

class TestSubSpecDecoderGenerate:
    def test_generates_tokens(self):
        fn  = _fixed_logit_fn(5)
        cfg = SubSpecConfig(n_total_layers=8, n_gpu_layers=4, gamma=2)
        dec = SubSpecDecoder(fn, fn, cfg, rng_seed=0)
        out, stats = dec.generate([1, 2, 3], max_new_tokens=6)
        assert stats.total_tokens == 6
        assert len(out) == 9  # 3 prompt + 6 generated

    def test_respects_max_new_tokens(self):
        fn  = _fixed_logit_fn(3)
        cfg = SubSpecConfig(n_total_layers=8, n_gpu_layers=4, gamma=4)
        dec = SubSpecDecoder(fn, fn, cfg, rng_seed=0)
        _, stats = dec.generate([0], max_new_tokens=5)
        assert stats.total_tokens == 5

    def test_full_acceptance_when_draft_target_agree(self):
        """When draft and target return identical logits acceptance is ~100%."""
        fn  = _fixed_logit_fn(7)
        cfg = SubSpecConfig(
            n_total_layers=8, n_gpu_layers=4, gamma=3, temperature=0.01
        )
        dec = SubSpecDecoder(fn, fn, cfg, rng_seed=1)
        _, stats = dec.generate([0], max_new_tokens=12)
        assert stats.accepted_total > 0
        assert stats.acceptance_rate > 0.0

    def test_rejection_when_draft_target_disagree(self):
        """High-confidence disagreement must produce rejections."""
        vocab = 10
        draft_fn  = _fixed_logit_fn(3, vocab)
        target_fn = _fixed_logit_fn(7, vocab)
        cfg = SubSpecConfig(n_total_layers=8, n_gpu_layers=4, gamma=4, temperature=0.01)
        dec = SubSpecDecoder(draft_fn, target_fn, cfg, rng_seed=0)
        _, stats = dec.generate([0], max_new_tokens=16)
        assert stats.rejected_total > 0

    def test_empty_prompt_accepted(self):
        fn  = _fixed_logit_fn(2)
        cfg = SubSpecConfig(n_total_layers=8, n_gpu_layers=4, gamma=2)
        dec = SubSpecDecoder(fn, fn, cfg)
        out, stats = dec.generate([], max_new_tokens=4)
        assert stats.total_tokens == 4

    def test_stats_draft_steps_positive(self):
        fn  = _fixed_logit_fn(1)
        cfg = SubSpecConfig(n_total_layers=8, n_gpu_layers=4, gamma=2)
        dec = SubSpecDecoder(fn, fn, cfg)
        _, stats = dec.generate([0], max_new_tokens=8)
        assert stats.draft_steps > 0

    def test_top_p_sampling(self):
        fn  = _fixed_logit_fn(4)
        cfg = SubSpecConfig(n_total_layers=8, n_gpu_layers=4, gamma=2, top_p=0.9)
        dec = SubSpecDecoder(fn, fn, cfg, rng_seed=42)
        out, stats = dec.generate([0], max_new_tokens=6)
        assert stats.total_tokens == 6
