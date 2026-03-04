"""
tests/test_scheduler_extended.py

Unit tests for BatchScheduler lifecycle + stats (no real model needed).
Covers __init__, start/stop/is_running lifecycle, stats().
The worker loop and generation methods are not called.
"""
from __future__ import annotations

from unittest.mock import MagicMock

from squish.scheduler import BatchScheduler


def _make_scheduler(**kwargs) -> BatchScheduler:
    """Build a BatchScheduler with mock model + tokenizer."""
    model = MagicMock()
    tokenizer = MagicMock()
    tokenizer.pad_token_id = 0
    tokenizer.eos_token_id = 2
    tokenizer.encode.return_value = [1, 2, 3]
    defaults = dict(
        model=model,
        tokenizer=tokenizer,
        max_batch_size=4,
        batch_window_ms=10.0,
    )
    defaults.update(kwargs)
    return BatchScheduler(**defaults)


class TestBatchSchedulerInit:
    def test_initial_running_state_false(self):
        sched = _make_scheduler()
        assert sched.is_running() is False

    def test_initial_metrics_zero(self):
        sched = _make_scheduler()
        assert sched.total_batches == 0
        assert sched.total_tokens_gen == 0
        assert sched.total_requests == 0

    def test_max_batch_stored(self):
        sched = _make_scheduler(max_batch_size=16)
        assert sched._max_batch == 16

    def test_window_ms_stored(self):
        sched = _make_scheduler(batch_window_ms=50.0)
        assert sched._window_ms == 50.0

    def test_pad_id_from_tokenizer(self):
        sched = _make_scheduler()
        assert isinstance(sched._pad_id, int)


class TestBatchSchedulerLifecycle:
    def test_start_returns_self(self):
        sched = _make_scheduler()
        result = sched.start()
        assert result is sched
        sched.stop(timeout=1.0)

    def test_is_running_true_after_start(self):
        sched = _make_scheduler()
        sched.start()
        assert sched.is_running() is True
        sched.stop(timeout=1.0)

    def test_start_twice_is_idempotent(self):
        sched = _make_scheduler()
        sched.start()
        sched.start()  # second call — thread already alive
        assert sched.is_running() is True
        sched.stop(timeout=1.0)

    def test_stop_sets_not_running(self):
        sched = _make_scheduler()
        sched.start()
        sched.stop(timeout=2.0)
        assert sched.is_running() is False

    def test_stop_without_start_is_safe(self):
        sched = _make_scheduler()
        sched.stop(timeout=0.5)  # should not raise


class TestBatchSchedulerStats:
    def test_stats_keys_present(self):
        sched = _make_scheduler()
        s = sched.stats()
        expected_keys = {
            "running", "total_batches", "total_tokens_gen",
            "total_requests", "pending_queue", "max_batch_size",
            "batch_window_ms",
        }
        assert expected_keys == set(s.keys())

    def test_stats_running_false_when_stopped(self):
        sched = _make_scheduler()
        s = sched.stats()
        assert s["running"] is False

    def test_stats_running_true_after_start(self):
        sched = _make_scheduler()
        sched.start()
        s = sched.stats()
        assert s["running"] is True
        sched.stop(timeout=1.0)

    def test_stats_pending_queue_zero_initially(self):
        sched = _make_scheduler()
        assert sched.stats()["pending_queue"] == 0
