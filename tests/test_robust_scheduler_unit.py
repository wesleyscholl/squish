"""Unit tests for squish.robust_scheduler — Interval-based robust scheduling."""

import pytest
from squish.robust_scheduler import (
    RobustSchedulerConfig,
    LengthInterval,
    Request,
    AMaxScheduler,
    ABalancedScheduler,
    RobustSchedulerStats,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _req(rid: str, input_len: int = 10, lo: int = 50, hi: int = 100) -> Request:
    return Request(
        request_id=rid,
        input_len=input_len,
        length_interval=LengthInterval(lo=lo, hi=hi),
    )


def _cfg(max_batch_tokens=2000, max_batch_size=10, **kw) -> RobustSchedulerConfig:
    return RobustSchedulerConfig(
        max_batch_tokens=max_batch_tokens, max_batch_size=max_batch_size, **kw
    )


# ---------------------------------------------------------------------------
# TestRobustSchedulerConfig
# ---------------------------------------------------------------------------

class TestRobustSchedulerConfig:
    def test_defaults(self):
        cfg = RobustSchedulerConfig()
        assert cfg.max_batch_tokens == 32768
        assert cfg.alpha == 0.5

    def test_invalid_max_batch_tokens(self):
        with pytest.raises(ValueError):
            RobustSchedulerConfig(max_batch_tokens=0)

    def test_invalid_max_batch_size(self):
        with pytest.raises(ValueError):
            RobustSchedulerConfig(max_batch_size=0)

    def test_invalid_memory_pressure_threshold(self):
        with pytest.raises(ValueError):
            RobustSchedulerConfig(memory_pressure_threshold=1.5)

    def test_invalid_alpha(self):
        with pytest.raises(ValueError):
            RobustSchedulerConfig(alpha=-0.1)

    def test_invalid_preemption_penalty(self):
        with pytest.raises(ValueError):
            RobustSchedulerConfig(preemption_penalty=-1.0)


# ---------------------------------------------------------------------------
# TestLengthInterval
# ---------------------------------------------------------------------------

class TestLengthInterval:
    def test_lo_gt_hi_invalid(self):
        with pytest.raises(ValueError):
            LengthInterval(lo=100, hi=50)

    def test_negative_lo_invalid(self):
        with pytest.raises(ValueError):
            LengthInterval(lo=-1, hi=50)

    def test_midpoint(self):
        iv = LengthInterval(lo=50, hi=150)
        assert iv.midpoint == 100

    def test_range_width(self):
        iv = LengthInterval(lo=50, hi=200)
        assert iv.range_width == 150

    def test_effective_length_at_alpha_zero(self):
        iv = LengthInterval(lo=50, hi=100)
        assert iv.effective_length(0.0) == 50

    def test_effective_length_at_alpha_one(self):
        iv = LengthInterval(lo=50, hi=100)
        assert iv.effective_length(1.0) == 100

    def test_from_point_creates_interval(self):
        iv = LengthInterval.from_point(100, uncertainty=0.25)
        assert iv.lo <= 100 <= iv.hi
        assert iv.point_estimate == 100


# ---------------------------------------------------------------------------
# TestRequest
# ---------------------------------------------------------------------------

class TestRequest:
    def test_tokens_at_hi(self):
        r = _req("r1", input_len=10, lo=50, hi=100)
        assert r.tokens_at_hi == 110

    def test_tokens_at_lo(self):
        r = _req("r1", input_len=10, lo=50, hi=100)
        assert r.tokens_at_lo == 60

    def test_tokens_at_alpha_half(self):
        r = _req("r1", input_len=10, lo=50, hi=100)
        assert r.tokens_at_alpha(0.5) == 10 + 75


# ---------------------------------------------------------------------------
# TestAMaxScheduler
# ---------------------------------------------------------------------------

class TestAMaxScheduler:
    def test_schedule_returns_requests(self):
        sched = AMaxScheduler(_cfg())
        sched.enqueue(_req("r1"))
        sched.enqueue(_req("r2"))
        batch = sched.schedule_batch()
        assert len(batch) >= 1

    def test_queue_size_decreases_after_batch(self):
        sched = AMaxScheduler(_cfg())
        for i in range(5):
            sched.enqueue(_req(f"r{i}"))
        batch = sched.schedule_batch()
        assert sched.queue_size == 5 - len(batch)

    def test_budget_constraint_respected(self):
        cfg = _cfg(max_batch_tokens=150)  # very tight budget
        sched = AMaxScheduler(cfg)
        # Each request at hi = 10 + 100 = 110 tokens
        for i in range(3):
            sched.enqueue(_req(f"r{i}", input_len=10, lo=50, hi=100))
        batch = sched.schedule_batch()
        assert len(batch) == 1  # only one fits

    def test_schedules_shortest_upper_bound_first(self):
        cfg = _cfg(max_batch_tokens=300)
        sched = AMaxScheduler(cfg)
        # Two requests: one short, one long at hi
        sched.enqueue(_req("long", input_len=10, lo=50, hi=200))
        sched.enqueue(_req("short", input_len=10, lo=50, hi=50))
        batch = sched.schedule_batch()
        ids = [r.request_id for r in batch]
        assert "short" in ids

    def test_empty_queue_returns_empty_batch(self):
        sched = AMaxScheduler(_cfg())
        assert sched.schedule_batch() == []

    def test_stats_track_scheduled(self):
        sched = AMaxScheduler(_cfg())
        sched.enqueue(_req("r1"))
        sched.schedule_batch()
        assert sched.stats.total_scheduled >= 1
        assert sched.stats.total_batches == 1


# ---------------------------------------------------------------------------
# TestABalancedScheduler
# ---------------------------------------------------------------------------

class TestABalancedScheduler:
    def test_initial_alpha(self):
        cfg = _cfg(alpha=0.5)
        sched = ABalancedScheduler(cfg)
        assert sched.current_alpha == 0.5

    def test_schedule_returns_requests(self):
        sched = ABalancedScheduler(_cfg())
        sched.enqueue(_req("r1"))
        batch = sched.schedule_batch()
        assert len(batch) >= 1

    def test_alpha_adapts_under_no_pressure(self):
        cfg = _cfg(max_batch_tokens=100000, alpha=0.5)
        sched = ABalancedScheduler(cfg)
        sched.enqueue(_req("r1"))
        sched.schedule_batch()
        # With plenty of memory, alpha should decrease (optimistic)
        assert sched.current_alpha <= 0.5

    def test_preemption_increments_stats(self):
        sched = ABalancedScheduler(_cfg())
        sched.handle_preemption("r1")
        assert sched.stats.preemptions == 1

    def test_preemption_rate(self):
        sched = ABalancedScheduler(_cfg())
        sched.enqueue(_req("r1"))
        sched.schedule_batch()  # schedules 1
        sched.handle_preemption("r1")
        assert sched.stats.preemption_rate == pytest.approx(1.0)

    def test_budget_constraint_respected(self):
        # At alpha=0.5: tokens_at_alpha = 10 + int(0.5*100 + 0.5*50) = 10+75 = 85
        # Budget=150: one request fits (85), two would need 170 > 150
        cfg = _cfg(max_batch_tokens=150)
        sched = ABalancedScheduler(cfg)
        for i in range(3):
            sched.enqueue(_req(f"r{i}", input_len=10, lo=50, hi=100))
        batch = sched.schedule_batch()
        assert len(batch) >= 1

    def test_empty_queue_returns_empty(self):
        sched = ABalancedScheduler(_cfg())
        assert sched.schedule_batch() == []


# ---------------------------------------------------------------------------
# TestRobustSchedulerStats
# ---------------------------------------------------------------------------

class TestRobustSchedulerStats:
    def test_mean_batch_size_zero_when_no_batches(self):
        stats = RobustSchedulerStats()
        assert stats.mean_batch_size == 0.0

    def test_mean_batch_size_calculation(self):
        stats = RobustSchedulerStats(total_scheduled=10, total_batches=2)
        assert stats.mean_batch_size == 5.0

    def test_preemption_rate_zero_when_no_requests(self):
        stats = RobustSchedulerStats()
        assert stats.preemption_rate == 0.0
