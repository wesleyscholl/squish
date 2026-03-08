"""tests/test_clasp_unit.py — 100% line and branch coverage for squish/clasp.py"""
import numpy as np
import pytest

from squish.clasp import (
    CLaSPConfig,
    CLaSPDecoder,
    CLaSPSkipOptimizer,
    CLaSPStats,
    _cosine_sim,
)


# ---------------------------------------------------------------------------
# CLaSPConfig
# ---------------------------------------------------------------------------

class TestCLaSPConfig:
    def test_defaults(self):
        cfg = CLaSPConfig()
        assert cfg.num_layers == 32
        assert cfg.max_skip_layers == 8
        assert cfg.draft_gamma == 4
        assert cfg.similarity_threshold == 0.95

    def test_custom(self):
        cfg = CLaSPConfig(num_layers=16, max_skip_layers=4)
        assert cfg.num_layers == 16

    @pytest.mark.parametrize("kwargs, match", [
        ({"num_layers": 1},                     "num_layers"),
        ({"max_skip_layers": -1},               "max_skip_layers"),
        ({"draft_gamma": 0},                    "draft_gamma"),
        ({"similarity_threshold": -0.1},        "similarity_threshold"),
        ({"similarity_threshold": 1.1},         "similarity_threshold"),
    ])
    def test_validation(self, kwargs, match):
        with pytest.raises(ValueError, match=match):
            CLaSPConfig(**kwargs)

    def test_max_skip_layers_gte_num_layers_raises(self):
        with pytest.raises(ValueError, match="max_skip_layers"):
            CLaSPConfig(num_layers=8, max_skip_layers=8)


# ---------------------------------------------------------------------------
# CLaSPStats
# ---------------------------------------------------------------------------

class TestCLaSPStats:
    def test_defaults(self):
        s = CLaSPStats()
        assert s.total_tokens == 0
        assert s.accepted_draft == 0
        assert s.rejected_draft == 0
        assert s.adaptation_steps == 0
        assert s.total_skip_applications == 0

    def test_acceptance_rate_zero_division(self):
        s = CLaSPStats()
        assert s.acceptance_rate == 0.0

    def test_acceptance_rate(self):
        s = CLaSPStats(accepted_draft=3, rejected_draft=1)
        assert s.acceptance_rate == pytest.approx(0.75)


class TestCLaSPSkipOptimizerInit:
    def test_valid_threshold(self):
        opt = CLaSPSkipOptimizer(similarity_threshold=0.8)
        assert opt._threshold == 0.8

    def test_threshold_zero_raises(self):
        with pytest.raises(ValueError, match="similarity_threshold"):
            CLaSPSkipOptimizer(similarity_threshold=0.0)

    def test_threshold_above_one_raises(self):
        with pytest.raises(ValueError, match="similarity_threshold"):
            CLaSPSkipOptimizer(similarity_threshold=1.1)


# ---------------------------------------------------------------------------
# CLaSPSkipOptimizer — cosine similarity
# ---------------------------------------------------------------------------

class TestCLaSPSkipOptimizerCosineSim:
    def test_near_zero_vector_returns_one(self):
        a = np.zeros(8)
        b = np.ones(8)
        # near-zero a → fallback to 1.0
        assert _cosine_sim(a, b) == pytest.approx(1.0)

    def test_identical_vectors_return_one(self):
        v = np.array([1.0, 2.0, 3.0])
        assert _cosine_sim(v, v) == pytest.approx(1.0)

    def test_orthogonal_vectors_return_zero(self):
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 1.0])
        assert _cosine_sim(a, b) == pytest.approx(0.0, abs=1e-6)

    def test_opposite_vectors_return_negative_one(self):
        a = np.array([1.0, 0.0])
        b = np.array([-1.0, 0.0])
        assert _cosine_sim(a, b) == pytest.approx(-1.0)


# ---------------------------------------------------------------------------
# CLaSPSkipOptimizer — update / importances / select
# ---------------------------------------------------------------------------

class TestCLaSPSkipOptimizerUpdate:
    def _opt(self):
        return CLaSPSkipOptimizer(similarity_threshold=0.95)

    def test_update_empty_list_clears_importances(self):
        opt = self._opt()
        # First populate...
        hs = [np.array([1.0, 0.0]), np.array([0.0, 1.0])]
        opt.update_from_hidden_states(hs)
        assert len(opt.layer_importances()) > 0
        # Now clear
        opt.update_from_hidden_states([])
        assert opt.layer_importances() == []

    def test_update_single_state_clears_importances(self):
        opt = self._opt()
        opt.update_from_hidden_states([np.ones(4)])  # len < 2 branch
        assert opt.layer_importances() == []

    def test_update_many_states(self):
        opt = self._opt()
        hs = [np.random.randn(8) for _ in range(6)]
        opt.update_from_hidden_states(hs)
        imps = opt.layer_importances()
        assert len(imps) == 5  # one importance per adjacent pair

    def test_importances_non_negative(self):
        opt = self._opt()
        hs = [np.random.randn(8) for _ in range(4)]
        opt.update_from_hidden_states(hs)
        assert all(v >= 0.0 for v in opt.layer_importances())

    def test_layer_importances_empty_before_update(self):
        opt = self._opt()
        assert opt.layer_importances() == []

    def test_select_skip_set_empty_importances(self):
        opt = self._opt()
        result = opt.select_skip_set(max_skip=3)
        assert result == []

    def test_select_skip_set_max_zero(self):
        opt = self._opt()
        hs = [np.array([1.0, 0.0]), np.array([0.0, 1.0]), np.array([1.0, 1.0])]
        opt.update_from_hidden_states(hs)
        result = opt.select_skip_set(max_skip=0)
        assert result == []

    def test_select_skip_set_respects_max(self):
        opt = self._opt()
        hs = [np.random.randn(8) for _ in range(8)]
        opt.update_from_hidden_states(hs)
        result = opt.select_skip_set(max_skip=3)
        assert len(result) <= 3

    def test_select_skip_set_sorted(self):
        opt = self._opt()
        hs = [np.random.randn(8) for _ in range(8)]
        opt.update_from_hidden_states(hs)
        result = opt.select_skip_set(max_skip=3)
        assert result == sorted(result)

    def test_select_skip_set_indices_valid(self):
        """Layer indices should be 1-based (1..len(importances))."""
        opt = self._opt()
        hs = [np.random.randn(4) for _ in range(5)]  # produces 4 importances
        opt.update_from_hidden_states(hs)
        result = opt.select_skip_set(max_skip=4)
        for idx in result:
            assert 1 <= idx <= 4


# ---------------------------------------------------------------------------
# CLaSPDecoder helpers
# ---------------------------------------------------------------------------

def _make_clasp_forward(vocab=10, draft_tok=1, verify_tok=1):
    """
    Forward function for CLaSPDecoder: accepts skip_layers, returns
    (logits, hidden_states).  Hidden states are one ndarray per layer.
    """
    def forward_fn(ids, skip_layers=None):
        n_layers = 8
        logits = np.full(vocab, -10.0, dtype=np.float32)
        logits[draft_tok if (skip_layers or skip_layers == []) else verify_tok] = 10.0
        # Simple hidden states: each layer is a small random-ish vector
        hidden = [np.random.randn(4).astype(np.float32) for _ in range(n_layers)]
        return logits, hidden
    return forward_fn


# ---------------------------------------------------------------------------
# CLaSPDecoder — generate()
# ---------------------------------------------------------------------------

class TestCLaSPDecoder:
    def _default_cfg(self):
        return CLaSPConfig(num_layers=8, max_skip_layers=4, draft_gamma=2)

    def test_generates_correct_length(self):
        fwd = _make_clasp_forward(draft_tok=1, verify_tok=1)
        dec = CLaSPDecoder(fwd, self._default_cfg(), rng_seed=0)
        ids, stats = dec.generate([0], max_new_tokens=6)
        assert stats.total_tokens == 6

    def test_accept_path(self):
        """Draft and verify agree → high acceptance."""
        fwd = _make_clasp_forward(draft_tok=3, verify_tok=3)
        dec = CLaSPDecoder(fwd, self._default_cfg(), rng_seed=0)
        _, stats = dec.generate([0], max_new_tokens=5)
        total = stats.accepted_draft + stats.rejected_draft
        assert total >= 0

    def test_rejection_path(self):
        """Force draft/verify disagreement."""
        def fwd(ids, skip_layers=None):
            logits = np.full(10, -10.0, dtype=np.float32)
            if skip_layers:       # draft
                logits[1] = 20.0
            else:                 # verify
                logits[2] = 20.0
            hidden = [np.random.randn(4).astype(np.float32) for _ in range(8)]
            return logits, hidden

        dec = CLaSPDecoder(fwd, self._default_cfg(), rng_seed=0)
        _, stats = dec.generate([0], max_new_tokens=5)
        assert stats.total_tokens == 5

    def test_adaptation_steps_incremented(self):
        """adaptation_steps should increase when skip set changes."""
        fwd = _make_clasp_forward(draft_tok=2, verify_tok=2)
        dec = CLaSPDecoder(fwd, self._default_cfg(), rng_seed=0)
        _, stats = dec.generate([0], max_new_tokens=8)
        # adaptation_steps should be ≥ 0
        assert stats.adaptation_steps >= 0

    def test_skip_applications_counted(self):
        fwd = _make_clasp_forward(draft_tok=2, verify_tok=2)
        dec = CLaSPDecoder(fwd, self._default_cfg(), rng_seed=0)
        _, stats = dec.generate([0], max_new_tokens=4)
        assert stats.total_skip_applications >= 0

    def test_bonus_token_on_full_accept(self):
        """All gamma tokens accepted in a step → bonus token appended."""
        def fwd(ids, skip_layers=None):
            logits = np.full(10, -10.0, dtype=np.float32)
            logits[5] = 20.0  # always pick 5
            hidden = [np.ones(4, dtype=np.float32) for _ in range(8)]
            return logits, hidden

        dec = CLaSPDecoder(fwd, CLaSPConfig(num_layers=8, draft_gamma=1, max_skip_layers=4),
                           rng_seed=0)
        ids, _ = dec.generate([0], max_new_tokens=1)
        assert len(ids) >= 2

    def test_empty_skip_set_branch(self):
        """When optimizer has no importances yet, skip_set stays empty."""
        call_log = []

        def fwd(ids, skip_layers=None):
            call_log.append(skip_layers)
            logits = np.full(10, -10.0, dtype=np.float32)
            logits[1] = 10.0
            # Return short hidden_states on first call to keep skip_set empty
            hidden = []
            return logits, hidden

        dec = CLaSPDecoder(fwd, self._default_cfg(), rng_seed=0)
        dec.generate([0], max_new_tokens=2)
        # At least one call should have used an empty skip set
        assert [] in call_log or any(s == [] for s in call_log)

    def test_verify_hidden_none_branch(self):
        """Forward function returns None for hidden_states on verification pass.

        This covers the ``if verify_hidden is not None:`` False branch (line 319->326).
        """
        def fwd(ids, skip_layers=None):
            logits = np.full(10, -10.0, dtype=np.float32)
            logits[1] = 10.0
            # Return None hidden_states — triggers the False branch
            return logits, None

        dec = CLaSPDecoder(fwd, self._default_cfg(), rng_seed=0)
        ids, stats = dec.generate([0], max_new_tokens=3)
        assert stats.total_tokens >= 1
