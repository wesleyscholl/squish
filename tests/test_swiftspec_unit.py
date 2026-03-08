"""tests/test_swiftspec_unit.py — 100% line and branch coverage for squish/swiftspec.py"""
import numpy as np
import pytest
import threading

from squish.swiftspec import SwiftSpecConfig, SwiftSpecDecoder, SwiftSpecStats


# ---------------------------------------------------------------------------
# SwiftSpecConfig
# ---------------------------------------------------------------------------

class TestSwiftSpecConfig:
    def test_defaults(self):
        cfg = SwiftSpecConfig()
        assert cfg.gamma == 5
        assert cfg.max_workers == 2

    def test_custom(self):
        cfg = SwiftSpecConfig(gamma=3, max_workers=4)
        assert cfg.gamma == 3

    @pytest.mark.parametrize("kwargs, match", [
        ({"gamma": 0},       "gamma"),
        ({"max_workers": 0}, "max_workers"),
    ])
    def test_validation(self, kwargs, match):
        with pytest.raises(ValueError, match=match):
            SwiftSpecConfig(**kwargs)


# ---------------------------------------------------------------------------
# SwiftSpecStats
# ---------------------------------------------------------------------------

class TestSwiftSpecStats:
    def test_defaults(self):
        s = SwiftSpecStats()
        assert s.total_tokens == 0
        assert s.draft_steps == 0
        assert s.accepted_total == 0

    def test_mean_accepted_per_step_zero_steps(self):
        s = SwiftSpecStats()
        assert s.mean_accepted_per_step == 0.0

    def test_mean_accepted_per_step_nonzero(self):
        s = SwiftSpecStats(draft_steps=4, accepted_total=12)
        assert s.mean_accepted_per_step == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# SwiftSpecDecoder helpers
# ---------------------------------------------------------------------------

def _make_funcs(vocab=10, agree_tok=7, draft_gamma=2, max_accept=None):
    """
    Build draft_fn and verify_fn that agree on `agree_tok`.

    Parameters
    ----------
    max_accept : int or None
        If set, only accept up to this many tokens per verify call.
    """
    def draft_fn(ids, gamma):
        return [agree_tok] * gamma

    def verify_fn(ids, draft_ids):
        n = len(draft_ids) if max_accept is None else min(len(draft_ids), max_accept)
        accepted = draft_ids[:n]   # only new tokens, not existing context
        extras = [agree_tok]
        return accepted, extras

    return draft_fn, verify_fn


def _make_rejecting_funcs(vocab=10, draft_tok=1, verify_tok=2):
    """Draft produces draft_tok, verifier always disagrees → partial acceptance."""
    def draft_fn(ids, gamma):
        return [draft_tok] * gamma

    def verify_fn(ids, draft_ids):
        # Accept zero tokens (immediate rejection) — return only verifier's choice
        accepted = [verify_tok]   # correction token only, no drafts accepted
        extras = []
        return accepted, extras

    return draft_fn, verify_fn


# ---------------------------------------------------------------------------
# SwiftSpecDecoder — basic generate
# ---------------------------------------------------------------------------

class TestSwiftSpecDecoderGenerate:
    def test_generates_correct_length(self):
        draft_fn, verify_fn = _make_funcs(max_accept=2)
        cfg = SwiftSpecConfig(gamma=2, max_workers=2)
        dec = SwiftSpecDecoder(draft_fn, verify_fn, cfg)
        ids, stats = dec.generate([0], max_new_tokens=8)
        assert stats.total_tokens == 8

    def test_stats_draft_steps_nonzero(self):
        draft_fn, verify_fn = _make_funcs(max_accept=2)
        cfg = SwiftSpecConfig(gamma=2, max_workers=2)
        dec = SwiftSpecDecoder(draft_fn, verify_fn, cfg)
        _, stats = dec.generate([0], max_new_tokens=6)
        assert stats.draft_steps > 0

    def test_mean_accepted_per_step_positive(self):
        draft_fn, verify_fn = _make_funcs(max_accept=2)
        cfg = SwiftSpecConfig(gamma=2, max_workers=2)
        dec = SwiftSpecDecoder(draft_fn, verify_fn, cfg)
        _, stats = dec.generate([0], max_new_tokens=6)
        assert stats.mean_accepted_per_step > 0.0

    def test_rejection_path_covered(self):
        """Verify fn rejects all draft tokens → each step produces exactly 1 token."""
        draft_fn, verify_fn = _make_rejecting_funcs()
        cfg = SwiftSpecConfig(gamma=3, max_workers=2)
        dec = SwiftSpecDecoder(draft_fn, verify_fn, cfg)
        _, stats = dec.generate([0], max_new_tokens=4)
        assert stats.total_tokens == 4

    def test_output_ids_length_matches_total_tokens(self):
        draft_fn, verify_fn = _make_funcs(max_accept=2)
        cfg = SwiftSpecConfig(gamma=2, max_workers=2)
        dec = SwiftSpecDecoder(draft_fn, verify_fn, cfg)
        ids, stats = dec.generate([0, 1], max_new_tokens=5)
        # Output = prompt + new tokens
        assert len(ids) == 2 + stats.total_tokens

    def test_multi_worker_runs_correctly(self):
        """max_workers=4 should not break correctness."""
        draft_fn, verify_fn = _make_funcs(max_accept=3)
        cfg = SwiftSpecConfig(gamma=3, max_workers=4)
        dec = SwiftSpecDecoder(draft_fn, verify_fn, cfg)
        ids, stats = dec.generate([0], max_new_tokens=9)
        assert stats.total_tokens == 9

    def test_single_gamma(self):
        """gamma=1 should still work without issues."""
        draft_fn, verify_fn = _make_funcs(max_accept=1)
        cfg = SwiftSpecConfig(gamma=1, max_workers=2)
        dec = SwiftSpecDecoder(draft_fn, verify_fn, cfg)
        _, stats = dec.generate([0], max_new_tokens=4)
        assert stats.total_tokens == 4


# ---------------------------------------------------------------------------
# SwiftSpecDecoder — cancel path (next_draft_future.cancel() branch)
# ---------------------------------------------------------------------------

class TestSwiftSpecDecoderCancelPath:
    def test_cancel_future_exercised_on_early_termination(self):
        """
        The loop terminates while a next_draft_future is in flight.
        This covers the `next_draft_future.cancel()` branch.

        We make verify_fn return exactly max_new_tokens tokens in a single
        call so the loop exits on the very first iteration.
        """
        max_new = 5

        def draft_fn(ids, gamma):
            return [7] * gamma

        def verify_fn(ids, draft_ids):
            # Return max_new_tokens new tokens so the loop exits immediately
            accepted = [7] * max_new
            return accepted, [7]

        cfg = SwiftSpecConfig(gamma=3, max_workers=2)
        dec = SwiftSpecDecoder(draft_fn, verify_fn, cfg)
        ids, stats = dec.generate([0], max_new_tokens=max_new)
        assert stats.total_tokens >= max_new

    def test_generates_with_zero_extras_from_verify(self):
        """verify_fn returns no bonus extras (empty list) — should not crash."""
        def draft_fn(ids, gamma):
            return [3] * gamma

        def verify_fn(ids, draft_ids):
            accepted = draft_ids[:1]   # only 1 new token
            return accepted, []   # no extra token

        cfg = SwiftSpecConfig(gamma=2, max_workers=2)
        dec = SwiftSpecDecoder(draft_fn, verify_fn, cfg)
        _, stats = dec.generate([0], max_new_tokens=4)
        assert stats.total_tokens >= 0  # just ensure it completes without error

    def test_zero_max_new_tokens_exits_immediately(self):
        """max_new_tokens=0 → while condition is False from the start (covers 179->200 branch)."""
        draft_fn, verify_fn = _make_funcs(max_accept=2)
        cfg = SwiftSpecConfig(gamma=2, max_workers=2)
        dec = SwiftSpecDecoder(draft_fn, verify_fn, cfg)
        ids, stats = dec.generate([0, 1], max_new_tokens=0)
        assert stats.total_tokens == 0
        assert ids == [0, 1]  # unchanged
