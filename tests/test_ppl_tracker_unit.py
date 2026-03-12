"""
tests/test_ppl_tracker_unit.py

Unit tests for squish/ppl_tracker.py — 100% coverage.
"""

import math

import numpy as np
import pytest

from squish.ppl_tracker import PPLAlert, PPLStats, PPLTracker, PPLWindow


# ---------------------------------------------------------------------------
# PPLWindow
# ---------------------------------------------------------------------------


class TestPPLWindow:
    def test_initial_empty(self):
        w = PPLWindow(window_size=5)
        assert w.values == []

    def test_push_appends(self):
        w = PPLWindow(window_size=5)
        w.push(10.0)
        assert w.values == [10.0]

    def test_push_evicts_oldest(self):
        w = PPLWindow(window_size=3)
        for v in [1.0, 2.0, 3.0, 4.0]:
            w.push(v)
        assert len(w.values) == 3
        assert w.values[0] == 2.0
        assert w.values[-1] == 4.0

    def test_push_exactly_full(self):
        w = PPLWindow(window_size=2)
        w.push(5.0)
        w.push(6.0)
        assert len(w.values) == 2

    def test_mean_empty(self):
        assert math.isnan(PPLWindow(3).mean)

    def test_mean_single(self):
        w = PPLWindow(3)
        w.push(7.0)
        assert w.mean == 7.0

    def test_mean_multiple(self):
        w = PPLWindow(5)
        for v in [2.0, 4.0, 6.0]:
            w.push(v)
        assert abs(w.mean - 4.0) < 1e-9

    def test_std_empty(self):
        assert math.isnan(PPLWindow(3).std)

    def test_std_single(self):
        w = PPLWindow(3)
        w.push(5.0)
        assert math.isnan(w.std)

    def test_std_multiple(self):
        w = PPLWindow(5)
        for v in [1.0, 2.0, 3.0]:
            w.push(v)
        assert not math.isnan(w.std)

    def test_min_empty(self):
        assert math.isnan(PPLWindow(3).min)

    def test_min_value(self):
        w = PPLWindow(5)
        for v in [3.0, 1.0, 4.0]:
            w.push(v)
        assert w.min == 1.0

    def test_max_empty(self):
        assert math.isnan(PPLWindow(3).max)

    def test_max_value(self):
        w = PPLWindow(5)
        for v in [3.0, 1.0, 4.0]:
            w.push(v)
        assert w.max == 4.0

    def test_window_size_zero_raises(self):
        with pytest.raises(ValueError, match="window_size"):
            PPLWindow(window_size=0)


# ---------------------------------------------------------------------------
# PPLStats
# ---------------------------------------------------------------------------


class TestPPLStats:
    def test_range_ppl(self):
        s = PPLStats(100, 10, min_ppl=5.0, max_ppl=20.0)
        assert abs(s.range_ppl - 15.0) < 1e-9


# ---------------------------------------------------------------------------
# PPLTracker — construction validation
# ---------------------------------------------------------------------------


class TestPPLTrackerConstruction:
    def test_defaults(self):
        t = PPLTracker()
        assert t.step == 0
        assert t._baseline_ppl is None
        assert not t.is_degraded

    def test_window_size_zero_raises(self):
        with pytest.raises(ValueError, match="window_size"):
            PPLTracker(window_size=0)

    def test_alert_threshold_one_raises(self):
        with pytest.raises(ValueError, match="alert_threshold"):
            PPLTracker(alert_threshold=1.0)

    def test_alert_threshold_less_than_one_raises(self):
        with pytest.raises(ValueError, match="alert_threshold"):
            PPLTracker(alert_threshold=0.5)

    def test_baseline_ppl_zero_raises(self):
        with pytest.raises(ValueError, match="baseline_ppl"):
            PPLTracker(baseline_ppl=0.0)

    def test_baseline_ppl_negative_raises(self):
        with pytest.raises(ValueError, match="baseline_ppl"):
            PPLTracker(baseline_ppl=-1.0)

    def test_baseline_ppl_set_on_construction(self):
        t = PPLTracker(baseline_ppl=10.0)
        assert t._baseline_ppl == 10.0


# ---------------------------------------------------------------------------
# PPLTracker — record
# ---------------------------------------------------------------------------


def _make_peaked_logits(seq_len, vocab_size, peak_scale=10.0, seed=0):
    """High-probability logits for random targets (low PPL)."""
    rng = np.random.default_rng(seed)
    targets = rng.integers(0, vocab_size, size=seq_len)
    logits = np.zeros((seq_len, vocab_size), dtype=np.float32)
    for i, t in enumerate(targets):
        logits[i, t] = peak_scale
    return logits, targets


def _make_uniform_logits(seq_len, vocab_size):
    """Uniform logits → high PPL."""
    logits = np.zeros((seq_len, vocab_size), dtype=np.float32)
    targets = np.zeros(seq_len, dtype=np.int64)
    return logits, targets


class TestPPLTrackerRecord:
    def test_step_increments(self):
        t = PPLTracker()
        logits, targets = _make_peaked_logits(5, 100)
        t.record(logits, targets)
        assert t.step == 1
        t.record(logits, targets)
        assert t.step == 2

    def test_rolling_ppl_nan_before_record(self):
        t = PPLTracker()
        assert math.isnan(t.rolling_ppl)

    def test_rolling_ppl_positive_after_record(self):
        t = PPLTracker()
        logits, targets = _make_peaked_logits(5, 100)
        t.record(logits, targets)
        rp = t.rolling_ppl
        assert not math.isnan(rp)
        assert rp > 0.0

    def test_ppl_low_for_peaked_logits(self):
        t = PPLTracker()
        logits, targets = _make_peaked_logits(10, 100, peak_scale=20.0)
        t.record(logits, targets)
        assert t.rolling_ppl < 2.0  # near-certain predictions → PPL close to 1

    def test_ppl_high_for_uniform_logits(self):
        t = PPLTracker()
        logits, targets = _make_uniform_logits(5, 1000)
        t.record(logits, targets)
        assert t.rolling_ppl > 100.0

    def test_record_empty_sequence_noop(self):
        t = PPLTracker()
        logits = np.zeros((0, 50), dtype=np.float32)
        targets = np.array([], dtype=np.int64)
        t.record(logits, targets)
        assert t.step == 0

    def test_window_fills_and_geometric_mean(self):
        t = PPLTracker(window_size=3)
        logits, targets = _make_peaked_logits(5, 100, peak_scale=15.0)
        for _ in range(5):
            t.record(logits, targets)
        # Window should have 3 values
        assert len(t._window.values) == 3
        # rolling_ppl = geometric mean ≈ each value (they're all the same)
        expected = t._window.values[0]
        assert abs(t.rolling_ppl - expected) < 1e-3


# ---------------------------------------------------------------------------
# PPLTracker — is_degraded and alerts
# ---------------------------------------------------------------------------


class TestPPLTrackerDegradation:
    def test_no_baseline_not_degraded(self):
        t = PPLTracker()
        logits, targets = _make_uniform_logits(3, 50)
        t.record(logits, targets)
        assert not t.is_degraded

    def test_at_baseline_not_degraded(self):
        t = PPLTracker(alert_threshold=2.0)
        logits, targets = _make_peaked_logits(5, 100, peak_scale=20.0)
        t.record(logits, targets)
        t.set_baseline()
        assert not t.is_degraded

    def test_degradation_triggers_is_degraded(self):
        t = PPLTracker(alert_threshold=1.5, baseline_ppl=2.0, window_size=5)
        logits, targets = _make_uniform_logits(5, 1000)
        for _ in range(5):
            t.record(logits, targets)
        assert t.rolling_ppl > 2.0 * 1.5
        assert t.is_degraded

    def test_alert_fired_when_degraded(self):
        t = PPLTracker(alert_threshold=1.5, baseline_ppl=2.0, window_size=5)
        logits, targets = _make_uniform_logits(5, 1000)
        for _ in range(5):
            t.record(logits, targets)
        assert len(t.alerts) > 0
        a = t.alerts[0]
        assert a.ratio > 1.5
        assert "PPL degradation" in a.message
        assert a.baseline_ppl == 2.0

    def test_alert_list_is_copy(self):
        t = PPLTracker(alert_threshold=1.5, baseline_ppl=2.0, window_size=5)
        logits, targets = _make_uniform_logits(5, 1000)
        for _ in range(5):
            t.record(logits, targets)
        a1 = t.alerts
        a2 = t.alerts
        assert a1 is not a2


# ---------------------------------------------------------------------------
# PPLTracker — set_baseline
# ---------------------------------------------------------------------------


class TestPPLTrackerSetBaseline:
    def test_set_baseline_explicit(self):
        t = PPLTracker()
        t.set_baseline(ppl=25.0)
        assert t._baseline_ppl == 25.0

    def test_set_baseline_from_rolling(self):
        t = PPLTracker()
        logits, targets = _make_peaked_logits(5, 100)
        t.record(logits, targets)
        rp = t.rolling_ppl
        t.set_baseline()
        assert abs(t._baseline_ppl - rp) < 1e-6

    def test_set_baseline_empty_window_raises(self):
        t = PPLTracker()
        with pytest.raises(RuntimeError, match="empty"):
            t.set_baseline()

    def test_set_baseline_zero_raises(self):
        t = PPLTracker()
        with pytest.raises(ValueError, match="baseline_ppl"):
            t.set_baseline(ppl=0.0)


# ---------------------------------------------------------------------------
# PPLTracker — reset
# ---------------------------------------------------------------------------


class TestPPLTrackerReset:
    def test_reset_clears_window_and_step(self):
        t = PPLTracker()
        logits, targets = _make_peaked_logits(5, 100)
        for _ in range(3):
            t.record(logits, targets)
        t.reset()
        assert t.step == 0
        assert math.isnan(t.rolling_ppl)

    def test_reset_clears_alerts(self):
        t = PPLTracker(alert_threshold=1.5, baseline_ppl=2.0, window_size=5)
        logits, targets = _make_uniform_logits(5, 1000)
        for _ in range(5):
            t.record(logits, targets)
        assert len(t.alerts) > 0
        t.reset()
        assert len(t.alerts) == 0

    def test_reset_preserves_baseline(self):
        t = PPLTracker(alert_threshold=2.0)
        logits, targets = _make_peaked_logits(5, 100)
        t.record(logits, targets)
        t.set_baseline()
        original_baseline = t._baseline_ppl
        t.reset()
        assert t._baseline_ppl == original_baseline

    def test_is_degraded_false_after_reset_empty_window(self):
        t = PPLTracker(alert_threshold=1.5, baseline_ppl=2.0, window_size=5)
        logits, targets = _make_uniform_logits(5, 1000)
        for _ in range(5):
            t.record(logits, targets)
        assert t.is_degraded
        t.reset()
        assert not t.is_degraded  # window is empty → nan → not degraded


# ---------------------------------------------------------------------------
# PPLTracker — ppl_stats
# ---------------------------------------------------------------------------


class TestPPLTrackerStats:
    def test_stats_empty(self):
        t = PPLTracker()
        s = t.ppl_stats()
        assert s.total_steps == 0
        assert s.total_tokens == 0
        assert math.isnan(s.min_ppl)
        assert math.isnan(s.max_ppl)

    def test_stats_after_records(self):
        t = PPLTracker()
        logits, targets = _make_peaked_logits(5, 100)
        for _ in range(3):
            t.record(logits, targets)
        s = t.ppl_stats()
        assert s.total_steps == 3
        assert s.total_tokens == 15
        assert s.min_ppl <= s.max_ppl
        assert not math.isnan(s.min_ppl)
        assert not math.isnan(s.max_ppl)

    def test_range_ppl(self):
        t = PPLTracker()
        logits_low, targets = _make_peaked_logits(5, 100, peak_scale=20.0)
        logits_high, _ = _make_uniform_logits(5, 1000)
        t.record(logits_low, targets)
        t.record(logits_high, np.zeros(5, dtype=np.int64))
        s = t.ppl_stats()
        assert s.range_ppl > 0.0
