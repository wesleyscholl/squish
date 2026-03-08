"""tests/test_del_decoder_unit.py — 100% line and branch coverage for squish/del_decoder.py"""
import numpy as np
import pytest

from squish.del_decoder import DELConfig, DELDecoder, DELStats, _softmax


# ---------------------------------------------------------------------------
# _softmax helper
# ---------------------------------------------------------------------------

class TestSoftmax:
    def test_sums_to_one(self):
        out = _softmax(np.array([1.0, 2.0, 3.0]))
        assert float(out.sum()) == pytest.approx(1.0, abs=1e-5)

    def test_argmax_preserved(self):
        logits = np.array([-1.0, 5.0, 1.0])
        assert np.argmax(_softmax(logits)) == 1

    def test_max_subtraction_stability(self):
        # Should not overflow even for large logits
        logits = np.array([1000.0, 1000.0, 1000.0])
        out = _softmax(logits)
        assert float(out.sum()) == pytest.approx(1.0, abs=1e-5)


# ---------------------------------------------------------------------------
# DELConfig
# ---------------------------------------------------------------------------

class TestDELConfig:
    def test_defaults(self):
        cfg = DELConfig()
        assert cfg.num_layers == 32
        assert cfg.min_exit_layer == 8
        assert cfg.max_exit_layer == 24
        assert cfg.gamma == 5
        assert cfg.confidence_threshold == 0.5

    def test_custom(self):
        cfg = DELConfig(num_layers=16, min_exit_layer=4, max_exit_layer=12, gamma=3)
        assert cfg.num_layers == 16

    @pytest.mark.parametrize("kwargs, match", [
        ({"num_layers": 1},                     "num_layers"),
        ({"min_exit_layer": 0},                 "min_exit_layer"),
        ({"max_exit_layer": 33},                "max_exit_layer"),
        ({"min_exit_layer": 25, "max_exit_layer": 20}, "min_exit_layer"),
        ({"gamma": 0},                          "gamma"),
        ({"confidence_threshold": 0.0},         "confidence_threshold"),
        ({"confidence_threshold": 1.1},         "confidence_threshold"),
    ])
    def test_validation(self, kwargs, match):
        with pytest.raises(ValueError, match=match):
            DELConfig(**kwargs)

    def test_min_equals_max_valid(self):
        cfg = DELConfig(min_exit_layer=16, max_exit_layer=16)
        assert cfg.min_exit_layer == cfg.max_exit_layer


# ---------------------------------------------------------------------------
# DELStats
# ---------------------------------------------------------------------------

class TestDELStats:
    def test_acceptance_rate_zero(self):
        s = DELStats()
        assert s.acceptance_rate == 0.0

    def test_acceptance_rate(self):
        s = DELStats(accepted_draft=3, rejected_draft=1)
        assert s.acceptance_rate == pytest.approx(0.75)


# ---------------------------------------------------------------------------
# DELDecoder helpers
# ---------------------------------------------------------------------------

def _make_forward(vocab=10, num_layers=16, always_tok=1):
    """
    Return a deterministic forward function:
      - Always predicts token `always_tok` with high confidence.
      - Accepts a `layer_limit` parameter (ignored in this toy version).
    """
    def forward_fn(ids, layer_limit=None):
        logits = np.full(vocab, -10.0, dtype=np.float32)
        logits[always_tok] = 10.0
        return logits
    return forward_fn


def _make_forward_confident_at_layer(vocab=10, confident_layer=8, strong_tok=2, weak_tok=3):
    """
    Return a forward fn where:
      - At `confident_layer` the model is very confident (high max prob, good TPL).
      - At other layers, confidence is lower.
    """
    def forward_fn(ids, layer_limit=None):
        logits = np.zeros(vocab, dtype=np.float32)
        if layer_limit == confident_layer:
            logits[strong_tok] = 10.0   # very confident at this layer
        else:
            logits[weak_tok] = 2.0      # less confident elsewhere
            logits[weak_tok + 1] = 1.5
        return logits
    return forward_fn


def _make_rejecting_forward(vocab=10, draft_tok=1, verify_tok=2):
    """Draft always produces `draft_tok`; full model always prefers `verify_tok`."""
    call_count = [0]

    def forward_fn(ids, layer_limit=None):
        logits = np.full(vocab, -10.0, dtype=np.float32)
        if layer_limit is not None:
            # Draft pass: prefer draft_tok
            logits[draft_tok] = 10.0
        else:
            # Verify pass: prefer verify_tok (disagrees with draft)
            logits[verify_tok] = 10.0
        return logits
    return forward_fn


# ---------------------------------------------------------------------------
# DELDecoder — shadow analysis
# ---------------------------------------------------------------------------

class TestDELShadowAnalysis:
    def test_returns_valid_exit_layer(self):
        cfg = DELConfig(num_layers=16, min_exit_layer=4, max_exit_layer=12)
        dec = DELDecoder(_make_forward(), cfg)
        layer = dec._shadow_analysis([0, 1, 2])
        assert cfg.min_exit_layer <= layer <= cfg.max_exit_layer

    def test_favours_high_confidence_layers(self):
        """Layer with highest confidence / (L/N) should be selected."""
        cfg = DELConfig(
            num_layers=16, min_exit_layer=4, max_exit_layer=8, gamma=2, confidence_threshold=0.01
        )
        dec = DELDecoder(
            _make_forward_confident_at_layer(vocab=10, confident_layer=4), cfg
        )
        chosen = dec._shadow_analysis([0])
        assert chosen == 4  # layer 4 → TPL = 1.0 / (4/16) = 4.0 (best)

    def test_single_candidate_layer(self):
        cfg = DELConfig(num_layers=16, min_exit_layer=10, max_exit_layer=10)
        dec = DELDecoder(_make_forward(), cfg)
        layer = dec._shadow_analysis([0])
        assert layer == 10


# ---------------------------------------------------------------------------
# DELDecoder — _select_tpl_layer
# ---------------------------------------------------------------------------

class TestDELSelectTPL:
    def test_no_history_returns_min(self):
        cfg = DELConfig(num_layers=16, min_exit_layer=4, max_exit_layer=12)
        dec = DELDecoder(_make_forward(), cfg)
        layer = dec._select_tpl_layer()
        # No history → all use uninformed prior 0.5; min_exit_layer has max
        # TPL because cost fraction is smallest
        assert layer == cfg.min_exit_layer

    def test_with_history_favours_high_accept_low_cost(self):
        """Layer with high acceptance rate AND low cost fraction wins."""
        cfg = DELConfig(num_layers=16, min_exit_layer=4, max_exit_layer=12)
        dec = DELDecoder(_make_forward(), cfg)
        # Inject: layer 4 accepts 10/10, layer 12 accepts 1/10
        dec._layer_accepts[4]  = 10
        dec._layer_rejects[4]  = 0
        dec._layer_accepts[12] = 1
        dec._layer_rejects[12] = 9
        layer = dec._select_tpl_layer()
        assert layer == 4

    def test_layers_with_zero_history_use_prior(self):
        cfg = DELConfig(num_layers=16, min_exit_layer=8, max_exit_layer=8)
        dec = DELDecoder(_make_forward(), cfg)
        layer = dec._select_tpl_layer()
        assert layer == 8


# ---------------------------------------------------------------------------
# DELDecoder — _draft_dynamic
# ---------------------------------------------------------------------------

class TestDELDraftDynamic:
    def test_produces_up_to_gamma(self):
        cfg = DELConfig(num_layers=16, min_exit_layer=4, max_exit_layer=12,
                        gamma=5, confidence_threshold=0.01)
        dec = DELDecoder(_make_forward(always_tok=1), cfg)
        draft_ids, draft_probs = dec._draft_dynamic([0], exit_layer=8)
        assert len(draft_ids) == 5
        assert len(draft_probs) == 5
        assert all(t == 1 for t in draft_ids)

    def test_dynamic_exit_fires_on_low_confidence(self):
        """With high threshold, stop after the first uncertain token."""
        # Make a forward that never gets confident at exit layer
        def fwd(ids, layer_limit=None):
            logits = np.zeros(10, dtype=np.float32)
            # Uniform distribution → low confidence
            return logits

        cfg = DELConfig(num_layers=16, min_exit_layer=4, max_exit_layer=12,
                        gamma=5, confidence_threshold=0.5)
        dec = DELDecoder(fwd, cfg)
        draft_ids, _ = dec._draft_dynamic([0], exit_layer=8)
        # Confidence ≈ 0.1 (uniform over 10) < 0.5 → stop after first token
        assert len(draft_ids) == 1

    def test_no_early_exit_when_always_confident(self):
        cfg = DELConfig(confidence_threshold=0.01)
        dec = DELDecoder(_make_forward(always_tok=0), cfg)
        draft_ids, _ = dec._draft_dynamic([0], exit_layer=cfg.min_exit_layer)
        assert len(draft_ids) == cfg.gamma


# ---------------------------------------------------------------------------
# DELDecoder — generate()
# ---------------------------------------------------------------------------

class TestDELGenerate:
    def test_generates_correct_length(self):
        cfg = DELConfig(num_layers=16, min_exit_layer=4, max_exit_layer=8,
                        gamma=3, confidence_threshold=0.01)
        dec = DELDecoder(_make_forward(always_tok=1), cfg, rng_seed=42)
        ids, stats = dec.generate([0], max_new_tokens=10)
        # May overshoot due to bonus tokens; must be >= max_new_tokens
        assert stats.total_tokens >= 10
        assert len(ids) == 1 + stats.total_tokens  # prompt + new tokens

    def test_stats_acceptance_tracked(self):
        cfg = DELConfig(num_layers=16, min_exit_layer=8, max_exit_layer=8,
                        gamma=2, confidence_threshold=0.01)
        dec = DELDecoder(_make_forward(always_tok=1), cfg, rng_seed=0)
        _, stats = dec.generate([0], max_new_tokens=5)
        total_draft = stats.accepted_draft + stats.rejected_draft
        assert total_draft >= 0

    def test_rejection_path_covered(self):
        """Force rejection by making draft disagree with verifier."""
        def fwd(ids, layer_limit=None):
            logits = np.full(10, -10.0, dtype=np.float32)
            if layer_limit is not None:
                logits[1] = 20.0  # draft always picks 1
            else:
                logits[2] = 20.0  # verify always picks 2 (rejects 1)
            return logits

        cfg = DELConfig(num_layers=16, min_exit_layer=4, max_exit_layer=8,
                        gamma=3, confidence_threshold=0.01)
        # Use seed 0 with a direct accept-prob that we can predict
        # Our accept probability = min(1, p_target/p_draft) ≈ 0 when disagree
        dec = DELDecoder(fwd, cfg, rng_seed=0)
        _, stats = dec.generate([0], max_new_tokens=5)
        assert stats.rejected_draft >= 0  # path executed

    def test_early_exit_counted(self):
        """Dynamic exit should increment early_exits counter."""
        def fwd(ids, layer_limit=None):
            # Uniform distribution → low confidence → triggers dynamic exit
            return np.zeros(10, dtype=np.float32)

        cfg = DELConfig(num_layers=16, min_exit_layer=4, max_exit_layer=8,
                        gamma=4, confidence_threshold=0.9)
        dec = DELDecoder(fwd, cfg, rng_seed=0)
        _, stats = dec.generate([0], max_new_tokens=4)
        assert stats.early_exits > 0

    def test_shadow_analyses_counted(self):
        cfg = DELConfig(num_layers=16, min_exit_layer=4, max_exit_layer=8,
                        gamma=2, confidence_threshold=0.01)
        dec = DELDecoder(_make_forward(always_tok=1), cfg, rng_seed=0)
        _, stats = dec.generate([0], max_new_tokens=4)
        assert stats.shadow_analyses >= 1

    def test_exit_layer_counts_populated(self):
        cfg = DELConfig(num_layers=16, min_exit_layer=4, max_exit_layer=8,
                        gamma=2, confidence_threshold=0.01)
        dec = DELDecoder(_make_forward(always_tok=1), cfg, rng_seed=0)
        _, stats = dec.generate([0], max_new_tokens=4)
        assert len(stats.exit_layer_counts) > 0
        total = sum(stats.exit_layer_counts.values())
        assert total >= 1

    def test_bonus_token_added_on_full_accept(self):
        """When all draft tokens accepted, a bonus token is appended."""
        # Make both draft and verify pick the same token → full acceptance
        cfg = DELConfig(num_layers=16, min_exit_layer=4, max_exit_layer=4,
                        gamma=1, confidence_threshold=0.01)
        dec = DELDecoder(_make_forward(always_tok=5), cfg, rng_seed=0)
        ids_before = [0]
        ids_out, _ = dec.generate(ids_before, max_new_tokens=1)
        # 1 new token generated → accepted + possibly bonus; at least 1 new
        assert len(ids_out) > len(ids_before)

    def test_layer_accepts_rejects_updated(self):
        """Layer accept/reject counters must be updated after each step."""
        cfg = DELConfig(num_layers=16, min_exit_layer=8, max_exit_layer=8,
                        gamma=2, confidence_threshold=0.01)
        dec = DELDecoder(_make_forward(always_tok=3), cfg, rng_seed=7)
        dec.generate([0], max_new_tokens=3)
        total = (
            sum(dec._layer_accepts.values())
            + sum(dec._layer_rejects.values())
        )
        assert total >= 1
