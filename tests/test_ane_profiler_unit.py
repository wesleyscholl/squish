"""
tests/test_ane_profiler_unit.py

Unit tests for squish/ane_profiler.py — 100% coverage.
"""

import pytest

from squish.ane_profiler import (
    ANEMetrics,
    ANEOpRecord,
    ANEProfiler,
    ANEProfilingSession,
    OpDevice,
)


# ---------------------------------------------------------------------------
# OpDevice constants
# ---------------------------------------------------------------------------


class TestOpDevice:
    def test_values(self):
        assert OpDevice.ANE == "ane"
        assert OpDevice.GPU == "gpu"
        assert OpDevice.CPU == "cpu"


# ---------------------------------------------------------------------------
# ANEMetrics
# ---------------------------------------------------------------------------


class TestANEMetrics:
    def test_ane_fraction_zero_ops(self):
        m = ANEMetrics(0, 0, 0, 0, 0.0, 0.0)
        assert m.ane_fraction == 0.0

    def test_ane_fraction(self):
        m = ANEMetrics(4, 2, 1, 1, 100.0, 60.0)
        assert abs(m.ane_fraction - 0.5) < 1e-9

    def test_ane_latency_fraction_zero_latency(self):
        m = ANEMetrics(1, 1, 0, 0, 0.0, 0.0)
        assert m.ane_latency_fraction == 0.0

    def test_ane_latency_fraction(self):
        m = ANEMetrics(2, 1, 1, 0, 200.0, 100.0)
        assert abs(m.ane_latency_fraction - 0.5) < 1e-9

    def test_avg_latency_us_zero_ops(self):
        m = ANEMetrics(0, 0, 0, 0, 0.0, 0.0)
        assert m.avg_latency_us == 0.0

    def test_avg_latency_us(self):
        m = ANEMetrics(4, 0, 4, 0, 200.0, 0.0)
        assert abs(m.avg_latency_us - 50.0) < 1e-9


# ---------------------------------------------------------------------------
# ANEProfiler — construction and validation
# ---------------------------------------------------------------------------


class TestANEProfilerConstruction:
    def test_default_threshold(self):
        p = ANEProfiler()
        assert p._threshold == 65_536

    def test_custom_threshold(self):
        p = ANEProfiler(ane_threshold_elements=1024)
        assert p._threshold == 1024

    def test_invalid_threshold_raises(self):
        with pytest.raises(ValueError, match="ane_threshold_elements"):
            ANEProfiler(ane_threshold_elements=0)


# ---------------------------------------------------------------------------
# ANEProfiler — record_op classification
# ---------------------------------------------------------------------------


class TestANEProfilerClassification:
    def test_float32_always_cpu(self):
        p = ANEProfiler()
        p.record_op("matmul", (1024, 1024), dtype="float32", latency_us=50.0)
        assert p._records[-1].device == OpDevice.CPU

    def test_float16_large_ane(self):
        p = ANEProfiler(ane_threshold_elements=1000)
        p.record_op("matmul", (100, 100), dtype="float16", latency_us=10.0)
        assert p._records[-1].device == OpDevice.ANE

    def test_float16_small_gpu(self):
        p = ANEProfiler(ane_threshold_elements=100_000)
        p.record_op("tiny", (10, 10), dtype="float16", latency_us=1.0)
        assert p._records[-1].device == OpDevice.GPU

    def test_bfloat16_large_ane(self):
        p = ANEProfiler(ane_threshold_elements=100)
        p.record_op("big", (32, 32), dtype="bfloat16", latency_us=5.0)
        assert p._records[-1].device == OpDevice.ANE

    def test_int8_gpu(self):
        p = ANEProfiler()
        p.record_op("quant", (1024, 1024), dtype="int8", latency_us=2.0)
        assert p._records[-1].device == OpDevice.GPU

    def test_unknown_dtype_gpu(self):
        p = ANEProfiler()
        p.record_op("op", (64, 64), dtype="uint8", latency_us=1.0)
        assert p._records[-1].device == OpDevice.GPU

    def test_empty_shape_treated_as_one_element(self):
        p = ANEProfiler(ane_threshold_elements=2)
        p.record_op("scalar", (), dtype="float16", latency_us=0.0)
        # 1 element <= 2 threshold → GPU
        assert p._records[-1].device == OpDevice.GPU

    def test_default_latency_is_zero(self):
        p = ANEProfiler()
        p.record_op("op", (10,), dtype="float32")
        assert p._records[-1].latency_us == 0.0

    def test_record_stores_shape_as_tuple(self):
        p = ANEProfiler()
        p.record_op("op", (4, 8), dtype="float32")
        assert p._records[-1].shape == (4, 8)


# ---------------------------------------------------------------------------
# ANEProfiler — summary
# ---------------------------------------------------------------------------


class TestANEProfilerSummary:
    def test_empty_summary(self):
        p = ANEProfiler()
        m = p.summary()
        assert m.total_ops == 0
        assert m.ane_ops == 0
        assert m.gpu_ops == 0
        assert m.cpu_ops == 0
        assert m.total_latency_us == 0.0
        assert m.ane_latency_us == 0.0

    def test_summary_counts(self):
        p = ANEProfiler(ane_threshold_elements=100)
        p.record_op("a", (200,), dtype="float16", latency_us=10.0)  # ANE
        p.record_op("b", (10,), dtype="float16", latency_us=5.0)   # GPU
        p.record_op("c", (50,), dtype="float32", latency_us=20.0)  # CPU
        m = p.summary()
        assert m.total_ops == 3
        assert m.ane_ops == 1
        assert m.gpu_ops == 1
        assert m.cpu_ops == 1
        assert abs(m.total_latency_us - 35.0) < 1e-9
        assert abs(m.ane_latency_us - 10.0) < 1e-9


# ---------------------------------------------------------------------------
# ANEProfiler — op_breakdown
# ---------------------------------------------------------------------------


class TestANEProfilerOpBreakdown:
    def test_op_breakdown_empty(self):
        p = ANEProfiler()
        assert p.op_breakdown() == {}

    def test_op_breakdown_aggregates(self):
        p = ANEProfiler(ane_threshold_elements=10)
        p.record_op("matmul", (100,), dtype="float16", latency_us=10.0)
        p.record_op("matmul", (100,), dtype="float16", latency_us=20.0)
        p.record_op("norm", (5,), dtype="float16", latency_us=2.0)
        bd = p.op_breakdown()
        assert bd["matmul"]["n_calls"] == 2
        assert abs(bd["matmul"]["total_us"] - 30.0) < 1e-9
        assert bd["matmul"]["device"] == OpDevice.ANE
        assert bd["norm"]["n_calls"] == 1
        assert bd["norm"]["device"] == OpDevice.GPU

    def test_op_breakdown_reflects_last_device(self):
        """Device in breakdown should be the device of the last recorded call."""
        p = ANEProfiler(ane_threshold_elements=50)
        p.record_op("op", (100,), dtype="float16")  # ANE
        p.record_op("op", (10,), dtype="float16")   # GPU (smaller)
        bd = p.op_breakdown()
        assert bd["op"]["device"] == OpDevice.GPU


# ---------------------------------------------------------------------------
# ANEProfiler — n_ops and reset
# ---------------------------------------------------------------------------


class TestANEProfilerReset:
    def test_n_ops(self):
        p = ANEProfiler()
        assert p.n_ops == 0
        p.record_op("a", (10,), dtype="float32")
        p.record_op("b", (10,), dtype="float32")
        assert p.n_ops == 2

    def test_reset_clears_records(self):
        p = ANEProfiler()
        p.record_op("a", (10,), dtype="float32")
        p.reset()
        assert p.n_ops == 0
        m = p.summary()
        assert m.total_ops == 0


# ---------------------------------------------------------------------------
# ANEProfilingSession
# ---------------------------------------------------------------------------


class TestANEProfilingSession:
    def test_metrics_none_before_exit(self):
        p = ANEProfiler()
        sess = ANEProfilingSession(p)
        assert sess.metrics is None

    def test_enter_resets_profiler(self):
        p = ANEProfiler()
        p.record_op("pre", (10,), dtype="float32")
        assert p.n_ops == 1
        with ANEProfilingSession(p):
            pass
        # After exit, pre-existing op is gone (reset on enter)
        # We exited cleanly; profiler should have 0 ops from before enter
        # (but session ops inside the block may exist if any were recorded)
        assert p.n_ops == 0

    def test_session_captures_metrics(self):
        p = ANEProfiler(ane_threshold_elements=100)
        with ANEProfilingSession(p) as sess:
            p.record_op("op", (200,), dtype="float16", latency_us=50.0)
        assert sess.metrics is not None
        assert sess.metrics.total_ops == 1
        assert sess.metrics.ane_ops == 1
        assert abs(sess.metrics.total_latency_us - 50.0) < 1e-9

    def test_session_with_no_ops(self):
        p = ANEProfiler()
        with ANEProfilingSession(p) as sess:
            pass
        assert sess.metrics is not None
        assert sess.metrics.total_ops == 0

    def test_session_returns_self(self):
        p = ANEProfiler()
        with ANEProfilingSession(p) as sess:
            assert isinstance(sess, ANEProfilingSession)
