"""tests/test_fsm_gamma_unit.py — 100% line and branch coverage for
FSMGammaController and SpeculativeGenerator._update_fsm in squish/speculative.py.
"""
from __future__ import annotations

import types

import pytest

from squish.speculative import FSMGammaController, SpeculativeGenerator, _MAX_SPEC_TOKENS


# ---------------------------------------------------------------------------
# FSMGammaController — __init__
# ---------------------------------------------------------------------------

class TestFSMGammaControllerInit:
    def test_defaults(self):
        fsm = FSMGammaController()
        assert fsm.gamma == 4
        assert fsm.min_gamma == 2
        assert fsm.max_gamma == 8

    def test_custom(self):
        fsm = FSMGammaController(initial_gamma=6, min_gamma=3, max_gamma=10)
        assert fsm.gamma == 6
        assert fsm.min_gamma == 3
        assert fsm.max_gamma == 10

    def test_initial_gamma_clamped_below_min(self):
        fsm = FSMGammaController(initial_gamma=1, min_gamma=3, max_gamma=8)
        assert fsm.gamma == 3

    def test_initial_gamma_clamped_above_max(self):
        fsm = FSMGammaController(initial_gamma=20, min_gamma=2, max_gamma=8)
        assert fsm.gamma == 8

    def test_initial_gamma_within_range_unchanged(self):
        fsm = FSMGammaController(initial_gamma=5, min_gamma=2, max_gamma=8)
        assert fsm.gamma == 5

    def test_min_gamma_zero_raises(self):
        with pytest.raises(ValueError, match="min_gamma"):
            FSMGammaController(min_gamma=0)

    def test_min_gamma_negative_raises(self):
        with pytest.raises(ValueError, match="min_gamma"):
            FSMGammaController(min_gamma=-1)

    def test_max_gamma_less_than_min_raises(self):
        with pytest.raises(ValueError, match="max_gamma"):
            FSMGammaController(min_gamma=5, max_gamma=4)

    def test_min_equals_max_valid(self):
        fsm = FSMGammaController(initial_gamma=3, min_gamma=3, max_gamma=3)
        assert fsm.gamma == 3


# ---------------------------------------------------------------------------
# FSMGammaController — step()
# ---------------------------------------------------------------------------

class TestFSMGammaControllerStep:
    def test_full_accept_increments_gamma(self):
        fsm = FSMGammaController(initial_gamma=4, min_gamma=2, max_gamma=8)
        new = fsm.step(n_accepted=4, n_proposed=4)  # full accept
        assert new == 5
        assert fsm.gamma == 5

    def test_partial_reject_decrements_gamma(self):
        fsm = FSMGammaController(initial_gamma=4, min_gamma=2, max_gamma=8)
        new = fsm.step(n_accepted=2, n_proposed=4)  # rejection
        assert new == 3
        assert fsm.gamma == 3

    def test_gamma_clamped_at_max(self):
        fsm = FSMGammaController(initial_gamma=8, min_gamma=2, max_gamma=8)
        new = fsm.step(n_accepted=8, n_proposed=8)
        assert new == 8  # already at max, stays there

    def test_gamma_clamped_at_min(self):
        fsm = FSMGammaController(initial_gamma=2, min_gamma=2, max_gamma=8)
        new = fsm.step(n_accepted=0, n_proposed=2)
        assert new == 2  # already at min, stays there

    def test_zero_proposed_counts_as_full_accept(self):
        """n_accepted (0) >= n_proposed (0) → full accept → increment."""
        fsm = FSMGammaController(initial_gamma=4, min_gamma=2, max_gamma=8)
        new = fsm.step(n_accepted=0, n_proposed=0)
        assert new == 5

    def test_step_returns_updated_gamma(self):
        fsm = FSMGammaController(initial_gamma=4, min_gamma=2, max_gamma=8)
        returned = fsm.step(5, 5)
        assert returned == fsm.gamma

    def test_multiple_steps_converge(self):
        """Repeated full accepts should drive gamma to max."""
        fsm = FSMGammaController(initial_gamma=2, min_gamma=2, max_gamma=8)
        for _ in range(10):
            fsm.step(4, 4)
        assert fsm.gamma == 8

    def test_multiple_rejects_drive_to_min(self):
        fsm = FSMGammaController(initial_gamma=8, min_gamma=2, max_gamma=8)
        for _ in range(10):
            fsm.step(0, 4)
        assert fsm.gamma == 2


# ---------------------------------------------------------------------------
# FSMGammaController — reset()
# ---------------------------------------------------------------------------

class TestFSMGammaControllerReset:
    def test_reset_to_explicit_value(self):
        fsm = FSMGammaController(initial_gamma=7, min_gamma=2, max_gamma=8)
        fsm.reset(gamma=5)
        assert fsm.gamma == 5

    def test_reset_clamps_above_max(self):
        fsm = FSMGammaController(initial_gamma=4, min_gamma=2, max_gamma=8)
        fsm.reset(gamma=100)
        assert fsm.gamma == 8

    def test_reset_clamps_below_min(self):
        fsm = FSMGammaController(initial_gamma=4, min_gamma=2, max_gamma=8)
        fsm.reset(gamma=0)
        assert fsm.gamma == 2

    def test_reset_none_gives_midpoint_even(self):
        fsm = FSMGammaController(initial_gamma=4, min_gamma=2, max_gamma=8)
        fsm.reset(gamma=None)
        assert fsm.gamma == (2 + 8) // 2  # 5

    def test_reset_none_gives_midpoint_odd_range(self):
        fsm = FSMGammaController(initial_gamma=5, min_gamma=1, max_gamma=6)
        fsm.reset()
        assert fsm.gamma == (1 + 6) // 2  # 3

    def test_reset_at_exact_min(self):
        fsm = FSMGammaController(initial_gamma=6, min_gamma=3, max_gamma=9)
        fsm.reset(gamma=3)
        assert fsm.gamma == 3

    def test_reset_at_exact_max(self):
        fsm = FSMGammaController(initial_gamma=6, min_gamma=3, max_gamma=9)
        fsm.reset(gamma=9)
        assert fsm.gamma == 9


# ---------------------------------------------------------------------------
# SpeculativeGenerator._update_fsm
# ---------------------------------------------------------------------------

def _make_mock_model():
    """Minimal mock that lets SpeculativeGenerator.__init__ complete."""
    # _try_make_model_cache checks for model.model (MLX pattern); if absent, returns None.
    return types.SimpleNamespace()


def _make_gen(fsm_gamma=True, k=4, fsm_min=2, fsm_max=8):
    model = _make_mock_model()
    tok   = types.SimpleNamespace()
    return SpeculativeGenerator(
        target_model=model,
        target_tokenizer=tok,
        k=k,
        fsm_gamma=fsm_gamma,
        fsm_min=fsm_min,
        fsm_max=fsm_max,
    )


class TestSpeculativeGeneratorUpdateFSM:
    def test_fsm_disabled_no_effect_on_k(self):
        """When fsm_gamma=False, _update_fsm is a no-op."""
        gen = _make_gen(fsm_gamma=False, k=4)
        initial_k = gen._k
        gen._update_fsm(n_accepted=4, n_proposed=4)
        assert gen._k == initial_k

    def test_fsm_enabled_full_accept_increments_k(self):
        """Full acceptance increments k through the FSM."""
        gen = _make_gen(fsm_gamma=True, k=4, fsm_min=2, fsm_max=8)
        gen._update_fsm(n_accepted=4, n_proposed=4)
        assert gen._k == 5

    def test_fsm_enabled_rejection_decrements_k(self):
        """Any rejection decrements k via the FSM."""
        gen = _make_gen(fsm_gamma=True, k=4, fsm_min=2, fsm_max=8)
        gen._update_fsm(n_accepted=1, n_proposed=4)
        assert gen._k == 3

    def test_fsm_k_capped_at_max_spec_tokens(self):
        """k never exceeds _MAX_SPEC_TOKENS (=8) even if FSM would go higher."""
        gen = _make_gen(fsm_gamma=True, k=_MAX_SPEC_TOKENS, fsm_min=2,
                        fsm_max=_MAX_SPEC_TOKENS)
        gen._update_fsm(n_accepted=8, n_proposed=8)
        assert gen._k <= _MAX_SPEC_TOKENS

    def test_fsm_k_clamped_at_fsm_min(self):
        gen = _make_gen(fsm_gamma=True, k=2, fsm_min=2, fsm_max=8)
        gen._update_fsm(n_accepted=0, n_proposed=2)
        assert gen._k == 2  # min_gamma floor

    def test_fsm_none_branch_skipped(self):
        """Ensure the 'if self._fsm is not None' branch with None is covered."""
        gen = _make_gen(fsm_gamma=False)
        assert gen._fsm is None
        # This call must be safe and must NOT change _k
        gen._k = 5
        gen._update_fsm(3, 4)
        assert gen._k == 5
