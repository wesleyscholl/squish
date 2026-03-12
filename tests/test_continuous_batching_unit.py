"""
tests/test_continuous_batching_unit.py

Unit tests for squish/continuous_batching.py — 100% coverage.
"""

import pytest

from squish.continuous_batching import (
    CBConfig,
    CBScheduler,
    CBStats,
    InFlightRequest,
    RequestState,
)


# ---------------------------------------------------------------------------
# CBConfig
# ---------------------------------------------------------------------------


class TestCBConfig:
    def test_defaults(self):
        cfg = CBConfig()
        assert cfg.max_batch_size == 32
        assert cfg.max_seq_len == 2048
        assert cfg.priority_policy == "fifo"

    @pytest.mark.parametrize(
        "kwargs, match",
        [
            ({"max_batch_size": 0}, "max_batch_size"),
            ({"max_batch_size": -1}, "max_batch_size"),
            ({"max_seq_len": 0}, "max_seq_len"),
            ({"priority_policy": "random"}, "priority_policy"),
        ],
    )
    def test_validation(self, kwargs, match):
        with pytest.raises(ValueError, match=match):
            CBConfig(**kwargs)

    def test_valid_policies(self):
        CBConfig(priority_policy="fifo")
        CBConfig(priority_policy="sjf")

    def test_frozen(self):
        cfg = CBConfig()
        with pytest.raises((AttributeError, TypeError)):
            cfg.max_batch_size = 64  # type: ignore[misc]


# ---------------------------------------------------------------------------
# RequestState
# ---------------------------------------------------------------------------


class TestRequestState:
    def test_string_values(self):
        assert RequestState.WAITING == "waiting"
        assert RequestState.RUNNING == "running"
        assert RequestState.FINISHED == "finished"
        assert RequestState.PREEMPTED == "preempted"


# ---------------------------------------------------------------------------
# InFlightRequest
# ---------------------------------------------------------------------------


class TestInFlightRequest:
    def test_defaults(self):
        req = InFlightRequest("r1", [1, 2, 3], max_new_tokens=10)
        assert req.state == RequestState.WAITING
        assert req.priority == 0
        assert req.generated_tokens == []

    def test_tokens_remaining(self):
        req = InFlightRequest("r1", [1], max_new_tokens=5)
        assert req.tokens_remaining == 5
        req.generated_tokens.append(1)
        assert req.tokens_remaining == 4

    def test_is_finished(self):
        req = InFlightRequest("r1", [1], max_new_tokens=2)
        assert not req.is_finished
        req.generated_tokens = [10, 20]
        assert req.is_finished

    def test_max_new_tokens_zero_raises(self):
        with pytest.raises(ValueError, match="max_new_tokens"):
            InFlightRequest("r1", [1], max_new_tokens=0)

    def test_independent_generated_tokens(self):
        r1 = InFlightRequest("r1", [1], max_new_tokens=5)
        r2 = InFlightRequest("r2", [1], max_new_tokens=5)
        r1.generated_tokens.append(99)
        assert r2.generated_tokens == []


# ---------------------------------------------------------------------------
# CBScheduler — construction and basic properties
# ---------------------------------------------------------------------------


class TestCBSchedulerProperties:
    def test_initial_state(self):
        s = CBScheduler(CBConfig(max_batch_size=4))
        assert s.n_waiting == 0
        assert s.n_running == 0
        assert s.n_finished == 0
        assert s.n_steps == 0
        assert s.throughput == 0.0

    def test_submit_increments_waiting(self):
        s = CBScheduler(CBConfig())
        s.submit(InFlightRequest("r1", [1], max_new_tokens=5))
        assert s.n_waiting == 1

    def test_submit_duplicate_raises(self):
        s = CBScheduler(CBConfig())
        r = InFlightRequest("r1", [1], max_new_tokens=5)
        s.submit(r)
        r2 = InFlightRequest("r1", [2], max_new_tokens=3)
        with pytest.raises(ValueError, match="r1"):
            s.submit(r2)


# ---------------------------------------------------------------------------
# CBScheduler — step_batch, FIFO
# ---------------------------------------------------------------------------


class TestCBSchedulerFIFO:
    def _sched(self, max_batch=2):
        return CBScheduler(CBConfig(max_batch_size=max_batch, priority_policy="fifo"))

    def test_step_batch_promotes_up_to_max(self):
        s = self._sched(max_batch=2)
        for i in range(4):
            s.submit(InFlightRequest(f"r{i}", [1], max_new_tokens=5))
        batch = s.step_batch()
        assert len(batch) == 2
        assert s.n_running == 2
        assert s.n_waiting == 2
        assert s.n_steps == 1

    def test_step_batch_fifo_order(self):
        s = self._sched(max_batch=2)
        s.submit(InFlightRequest("first", [1], max_new_tokens=5))
        s.submit(InFlightRequest("second", [1], max_new_tokens=5))
        s.submit(InFlightRequest("third", [1], max_new_tokens=5))
        batch = s.step_batch()
        ids = {r.request_id for r in batch}
        assert ids == {"first", "second"}

    def test_step_promotes_after_completion(self):
        s = self._sched(max_batch=2)
        s.submit(InFlightRequest("r1", [1], max_new_tokens=1))
        s.submit(InFlightRequest("r2", [1], max_new_tokens=5))
        s.submit(InFlightRequest("r3", [1], max_new_tokens=5))
        s.step_batch()  # r1, r2 running
        s.complete_token("r1", 42)  # r1 finishes
        assert s.n_running == 1
        s.step_batch()  # r3 should be promoted
        assert s.n_running == 2

    def test_no_waiting_requests_batch_unchanged(self):
        s = self._sched(max_batch=2)
        s.submit(InFlightRequest("r1", [1], max_new_tokens=5))
        batch = s.step_batch()
        batch2 = s.step_batch()
        assert len(batch2) == 1

    def test_empty_scheduler_step(self):
        s = self._sched()
        batch = s.step_batch()
        assert batch == []
        assert s.n_steps == 1


# ---------------------------------------------------------------------------
# CBScheduler — step_batch, SJF
# ---------------------------------------------------------------------------


class TestCBSchedulerSJF:
    def test_sjf_promotes_shortest_first(self):
        s = CBScheduler(CBConfig(max_batch_size=2, priority_policy="sjf"))
        s.submit(InFlightRequest("long", [1], max_new_tokens=100))
        s.submit(InFlightRequest("short", [1], max_new_tokens=1))
        s.submit(InFlightRequest("medium", [1], max_new_tokens=10))
        batch = s.step_batch()
        ids = {r.request_id for r in batch}
        assert ids == {"short", "medium"}


# ---------------------------------------------------------------------------
# CBScheduler — complete_token
# ---------------------------------------------------------------------------


class TestCBSchedulerCompleteToken:
    def test_complete_token_appends(self):
        s = CBScheduler(CBConfig())
        r = InFlightRequest("r1", [1], max_new_tokens=3)
        s.submit(r)
        s.step_batch()
        s.complete_token("r1", 99)
        assert r.generated_tokens == [99]
        assert r.tokens_remaining == 2

    def test_complete_token_finishes_request(self):
        s = CBScheduler(CBConfig())
        r = InFlightRequest("r1", [1], max_new_tokens=1)
        s.submit(r)
        s.step_batch()
        s.complete_token("r1", 42)
        assert r.state == RequestState.FINISHED
        assert s.n_finished == 1
        assert s.n_running == 0

    def test_complete_token_unknown_id_raises(self):
        s = CBScheduler(CBConfig())
        with pytest.raises(KeyError):
            s.complete_token("ghost", 1)

    def test_complete_token_non_running_raises(self):
        s = CBScheduler(CBConfig())
        r = InFlightRequest("r1", [1], max_new_tokens=5)
        s.submit(r)
        # r1 is WAITING, not RUNNING
        with pytest.raises(ValueError, match="not running"):
            s.complete_token("r1", 1)


# ---------------------------------------------------------------------------
# CBScheduler — preempt
# ---------------------------------------------------------------------------


class TestCBSchedulerPreempt:
    def test_preempt_moves_to_preempted(self):
        s = CBScheduler(CBConfig())
        r = InFlightRequest("r1", [1], max_new_tokens=5)
        s.submit(r)
        s.step_batch()
        s.preempt("r1")
        assert r.state == RequestState.PREEMPTED
        assert s.n_running == 0

    def test_preempt_unknown_raises(self):
        s = CBScheduler(CBConfig())
        with pytest.raises(KeyError):
            s.preempt("ghost")

    def test_preempt_non_running_raises(self):
        s = CBScheduler(CBConfig())
        r = InFlightRequest("r1", [1], max_new_tokens=1)
        s.submit(r)
        s.step_batch()
        s.complete_token("r1", 42)  # finished
        with pytest.raises(ValueError, match="not running"):
            s.preempt("r1")


# ---------------------------------------------------------------------------
# CBScheduler — throughput and stats
# ---------------------------------------------------------------------------


class TestCBSchedulerStats:
    def test_throughput_zero_steps(self):
        s = CBScheduler(CBConfig())
        assert s.throughput == 0.0

    def test_throughput_after_tokens(self):
        s = CBScheduler(CBConfig(max_batch_size=4))
        for i in range(3):
            s.submit(InFlightRequest(f"r{i}", [1], max_new_tokens=5))
        s.step_batch()  # step 1
        for i in range(3):
            s.complete_token(f"r{i}", i)
        s.step_batch()  # step 2
        # 3 tokens / 2 steps
        assert abs(s.throughput - 1.5) < 1e-9

    def test_scheduler_stats(self):
        s = CBScheduler(CBConfig(max_batch_size=4))
        r1 = InFlightRequest("r1", [1], max_new_tokens=1)
        r2 = InFlightRequest("r2", [1], max_new_tokens=5)
        s.submit(r1)
        s.submit(r2)
        s.step_batch()
        s.complete_token("r1", 42)
        s.preempt("r2")
        st = s.scheduler_stats()
        assert st.total_submitted == 2
        assert st.total_completed == 1
        assert st.total_preemptions == 1
        assert st.total_tokens_generated == 1
        assert st.n_steps == 1
        assert st.avg_batch_size == 1.0
        assert abs(st.completion_rate - 0.5) < 1e-9


# ---------------------------------------------------------------------------
# CBStats
# ---------------------------------------------------------------------------


class TestCBStats:
    def test_avg_batch_size_zero_steps(self):
        st = CBStats(0, 0, 0, 0, 0)
        assert st.avg_batch_size == 0.0

    def test_completion_rate_zero_submitted(self):
        st = CBStats(0, 0, 0, 0, 1)
        assert st.completion_rate == 0.0
