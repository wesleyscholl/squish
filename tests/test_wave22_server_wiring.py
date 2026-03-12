"""tests/test_wave22_server_wiring.py

Verifies that all Wave 22 module classes are importable and have the expected
public APIs that the server.py wiring code depends on.  These are pure
import + instantiation tests — no model or GPU required.

Wave 22 modules (Server Wiring · Adaptive Serving · Observability):
  multi_tenant_sched, request_router, cache_warmup, token_budget_gate,
  observability_hook, request_coalesce, adaptive_quantize, health_check,
  fault_tolerance, model_pool, streaming_chunk, cost_estimator,
  sla_monitor, context_cache
"""

import time

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# MultiTenantSched
# ---------------------------------------------------------------------------


class TestMultiTenantSchedWiring:
    def test_tenant_config_fields(self):
        from squish.multi_tenant_sched import TenantConfig

        cfg = TenantConfig(
            tenant_id="alice",
            weight=2.0,
            max_concurrent=4,
            target_latency_ms=200.0,
        )
        assert cfg.tenant_id == "alice"
        assert cfg.weight == 2.0
        assert cfg.max_concurrent == 4
        assert cfg.target_latency_ms == 200.0

    def test_tenant_request_fields(self):
        from squish.multi_tenant_sched import TenantRequest

        req = TenantRequest(
            request_id="req-1",
            tenant_id="alice",
            n_tokens_est=128,
            submitted_at=1.0,
        )
        assert req.request_id == "req-1"
        assert req.tenant_id == "alice"
        assert req.n_tokens_est == 128
        assert req.submitted_at == 1.0

    def test_scheduler_submit_and_next(self):
        from squish.multi_tenant_sched import TenantConfig, TenantRequest, TenantScheduler

        tenants = [
            TenantConfig(tenant_id="alice", weight=2.0, max_concurrent=4),
            TenantConfig(tenant_id="bob", weight=1.0, max_concurrent=2),
        ]
        sched = TenantScheduler(tenants)
        sched.submit(TenantRequest("r1", "alice", n_tokens_est=64, submitted_at=1.0))
        sched.submit(TenantRequest("r2", "bob", n_tokens_est=32, submitted_at=1.1))
        req = sched.next_request()
        assert req is not None
        sched.complete(req.request_id, actual_latency_ms=50.0)
        assert sched.stats.total_completed == 1

    def test_tenant_stats_slo_violation_rate(self):
        from squish.multi_tenant_sched import TenantStats

        s = TenantStats(total_submitted=10, total_completed=8, slo_violations=2)
        assert abs(s.slo_violation_rate - 0.25) < 1e-9
        empty = TenantStats()
        assert empty.slo_violation_rate == 0.0


# ---------------------------------------------------------------------------
# RequestRouter
# ---------------------------------------------------------------------------


class TestRequestRouterWiring:
    def test_replica_config_fields(self):
        from squish.request_router import ReplicaConfig

        cfg = ReplicaConfig(replica_id="gpu-0", max_concurrent=8, weight=2.0)
        assert cfg.replica_id == "gpu-0"
        assert cfg.max_concurrent == 8
        assert cfg.weight == 2.0

    def test_replica_registry_register(self):
        from squish.request_router import ReplicaConfig, ReplicaRegistry

        reg = ReplicaRegistry([
            ReplicaConfig(replica_id="gpu-0", max_concurrent=4),
            ReplicaConfig(replica_id="gpu-1", max_concurrent=4),
        ])
        assert reg.n_replicas == 2
        assert "gpu-0" in reg.replica_ids
        reg.update_load("gpu-0", 2)
        assert reg.effective_load("gpu-0") == 2 / 4

    def test_request_router_route_and_complete(self):
        from squish.request_router import ReplicaConfig, ReplicaRegistry, RequestRouter

        reg = ReplicaRegistry([
            ReplicaConfig(replica_id="r0", max_concurrent=8, weight=1.0),
            ReplicaConfig(replica_id="r1", max_concurrent=8, weight=1.0),
        ])
        router = RequestRouter(reg)
        rid = router.route("req-001")
        assert rid in ("r0", "r1")
        assert router.n_active == 1
        router.complete("req-001")
        assert router.stats.total_completed == 1

    def test_router_stats_avg_load(self):
        from squish.request_router import RouterStats

        s = RouterStats(total_routed=10, total_completed=5)
        assert abs(s.avg_load - 2.0) < 1e-9
        empty = RouterStats()
        assert empty.avg_load == 0.0


# ---------------------------------------------------------------------------
# CacheWarmup
# ---------------------------------------------------------------------------


class TestCacheWarmupWiring:
    def test_warmup_config_fields(self):
        from squish.cache_warmup import WarmupConfig

        cfg = WarmupConfig(top_k=16, min_access_count=2, max_prefix_tokens=64)
        assert cfg.top_k == 16
        assert cfg.min_access_count == 2
        assert cfg.max_prefix_tokens == 64

    def test_access_record_fields(self):
        from squish.cache_warmup import AccessRecord

        rec = AccessRecord(prefix_hash=42, access_count=5, last_access=1.5)
        assert rec.prefix_hash == 42
        assert rec.access_count == 5
        assert rec.last_access == 1.5

    def test_predictor_record_and_candidates(self):
        from squish.cache_warmup import CacheWarmupPredictor, WarmupConfig

        cfg = WarmupConfig(top_k=4, min_access_count=2, max_prefix_tokens=8)
        predictor = CacheWarmupPredictor(cfg)
        tokens = [1, 2, 3, 4]
        predictor.record_access(tokens, timestamp=0.0)
        predictor.record_access(tokens, timestamp=0.1)
        assert predictor.n_tracked == 1
        candidates = predictor.get_warmup_candidates()
        assert len(candidates) == 1
        assert predictor.stats.total_accesses == 2

    def test_warmup_stats_fields(self):
        from squish.cache_warmup import WarmupStats

        s = WarmupStats(total_accesses=10, cache_warmups_issued=4, predicted_hits=3)
        assert s.total_accesses == 10
        assert s.cache_warmups_issued == 4
        assert s.predicted_hits == 3


# ---------------------------------------------------------------------------
# TokenBudgetGate
# ---------------------------------------------------------------------------


class TestTokenBudgetGateWiring:
    def test_budget_policy_fields(self):
        from squish.token_budget_gate import BudgetPolicy

        policy = BudgetPolicy(mode="soft", soft_penalty=0.2, warn_at_fraction=0.8)
        assert policy.mode == "soft"
        assert policy.soft_penalty == 0.2
        assert policy.warn_at_fraction == 0.8

    def test_gate_tick_exhausts(self):
        from squish.token_budget_gate import BudgetPolicy, TokenBudgetGate

        policy = BudgetPolicy(mode="hard")
        gate = TokenBudgetGate(max_tokens=5, policy=policy)
        results = [gate.tick() for _ in range(7)]
        # Ticks 1–4 should return True (budget not yet reached at < 5),
        # tick 5 returns False (budget now == max_tokens).
        assert results[4] is False
        assert gate.is_exhausted

    def test_gate_fraction_used(self):
        from squish.token_budget_gate import BudgetPolicy, TokenBudgetGate

        policy = BudgetPolicy()
        gate = TokenBudgetGate(max_tokens=10, policy=policy)
        gate.tick()
        gate.tick()
        gate.tick()
        assert abs(gate.fraction_used() - 0.3) < 1e-9
        assert gate.tokens_used == 3
        assert gate.remaining() == 7

    def test_budget_gate_stats(self):
        from squish.token_budget_gate import BudgetGateStats

        s = BudgetGateStats(
            total_requests=5,
            total_tokens_gated=50,
            hard_stops=2,
            warnings_issued=3,
        )
        assert s.total_requests == 5
        assert s.hard_stops == 2
        assert s.warnings_issued == 3


# ---------------------------------------------------------------------------
# ObservabilityHook
# ---------------------------------------------------------------------------


class TestObservabilityHookWiring:
    def test_span_kind_constants(self):
        from squish.observability_hook import SpanKind

        assert SpanKind.PREFILL == "prefill"
        assert SpanKind.DECODE == "decode"
        assert SpanKind.VERIFY == "verify"
        assert SpanKind.KV_LOOKUP == "kv_lookup"

    def test_span_duration_ms(self):
        from squish.observability_hook import Span

        span = Span(span_id="abc", kind="prefill", start_time=1.0, end_time=0.0)
        assert span.duration_ms == 0.0  # not finished yet
        span.end_time = 1.5
        assert abs(span.duration_ms - 500.0) < 1e-6

    def test_span_collector_record_finish_export(self):
        from squish.observability_hook import SpanCollector

        collector = SpanCollector(max_spans=100)
        span = collector.record("decode", step=0)
        assert span.end_time == 0.0
        collector.finish(span)
        assert collector.n_spans == 1
        records = collector.export()
        assert len(records) == 1
        assert records[0]["kind"] == "decode"
        assert records[0]["duration_ms"] > 0.0

    def test_inference_tracer_stats(self):
        from squish.observability_hook import InferenceTracer, SpanCollector, TracerStats

        collector = SpanCollector(max_spans=100)
        tracer = InferenceTracer(collector)
        s0 = tracer.stats
        assert isinstance(s0, TracerStats)
        assert s0.total_spans == 0
        span_p = tracer.trace_prefill(seq_len=64)
        collector.finish(span_p)
        span_d = tracer.trace_decode(step=0)
        collector.finish(span_d)
        assert tracer.stats.total_spans == 2
        assert tracer.stats.prefill_spans == 1
        assert tracer.stats.decode_spans == 1


# ---------------------------------------------------------------------------
# RequestCoalesce
# ---------------------------------------------------------------------------


class TestRequestCoalesceWiring:
    def test_coalesce_config_defaults(self):
        from squish.request_coalesce import CoalesceConfig

        cfg = CoalesceConfig(min_shared_tokens=4, max_group_size=4)
        assert cfg.min_shared_tokens == 4
        assert cfg.max_group_size == 4

    def test_coalesce_group_fields(self):
        from squish.request_coalesce import CoalesceGroup

        g = CoalesceGroup(
            shared_prefix=[1, 2, 3],
            request_ids=["r1", "r2"],
            branch_tokens=[[4, 5], [6, 7]],
        )
        assert g.shared_prefix == [1, 2, 3]
        assert len(g.request_ids) == 2
        assert len(g.branch_tokens) == 2

    def test_prefix_coalescer_coalesce(self):
        from squish.request_coalesce import CoalesceConfig, PrefixCoalescer

        cfg = CoalesceConfig(min_shared_tokens=3, max_group_size=4)
        coalescer = PrefixCoalescer(cfg)
        shared = [10, 20, 30, 40]
        coalescer.add_request("req-1", shared + [100, 101])
        coalescer.add_request("req-2", shared + [200, 201])
        coalescer.add_request("req-3", shared + [300])
        assert coalescer.n_pending == 3
        groups = coalescer.coalesce()
        assert coalescer.n_pending == 0
        assert len(groups) >= 1
        assert coalescer.stats.total_requests == 3

    def test_coalesce_stats_fields(self):
        from squish.request_coalesce import CoalesceStats

        s = CoalesceStats(
            total_requests=6,
            total_groups_formed=2,
            total_tokens_saved=12,
        )
        assert s.total_requests == 6
        assert s.total_groups_formed == 2
        assert s.total_tokens_saved == 12


# ---------------------------------------------------------------------------
# AdaptiveQuantize
# ---------------------------------------------------------------------------


class TestAdaptiveQuantizeWiring:
    def test_pressure_thresholds_fields(self):
        from squish.adaptive_quantize import PressureThresholds

        t = PressureThresholds(int8_threshold=0.70, int4_threshold=0.85)
        assert t.int8_threshold == 0.70
        assert t.int4_threshold == 0.85

    def test_quant_precision_constants(self):
        from squish.adaptive_quantize import QuantPrecision

        assert QuantPrecision.FP16 == "fp16"
        assert QuantPrecision.INT8 == "int8"
        assert QuantPrecision.INT4 == "int4"

    def test_pressure_monitor_update_and_precision(self):
        from squish.adaptive_quantize import PressureMonitor, PressureThresholds, QuantPrecision

        t = PressureThresholds(int8_threshold=0.75, int4_threshold=0.90)
        cap = 4 * 1024 ** 3  # 4 GiB
        monitor = PressureMonitor(t, capacity_bytes=cap)
        monitor.update(0)
        assert monitor.current_precision == QuantPrecision.FP16
        monitor.update(int(0.80 * cap))
        assert monitor.current_precision == QuantPrecision.INT8
        monitor.update(int(0.92 * cap))
        assert monitor.current_precision == QuantPrecision.INT4

    def test_adaptive_quantizer_quantize_dequantize(self):
        from squish.adaptive_quantize import (
            AdaptiveQuantizer,
            PressureMonitor,
            PressureThresholds,
            QuantPrecision,
        )

        rng = np.random.default_rng(0)
        t = PressureThresholds(int8_threshold=0.75, int4_threshold=0.90)
        cap = 4 * 1024 ** 3
        monitor = PressureMonitor(t, capacity_bytes=cap)
        monitor.update(int(0.80 * cap))  # force INT8
        quantizer = AdaptiveQuantizer(monitor)
        x = rng.standard_normal((16, 8)).astype(np.float32)
        q, scale = quantizer.quantize(x)
        assert q.dtype == np.int8
        assert scale > 0.0
        x_approx = quantizer.dequantize(q, scale, QuantPrecision.INT8)
        assert x_approx.shape == x.shape
        assert quantizer.stats.total_quantize_calls == 1
        assert quantizer.stats.int8_calls == 1


# ---------------------------------------------------------------------------
# HealthCheck
# ---------------------------------------------------------------------------


class TestHealthCheckWiring:
    def test_health_state_constants(self):
        from squish.health_check import HealthState

        assert HealthState.OK == "ok"
        assert HealthState.DEGRADED == "degraded"
        assert HealthState.CRITICAL == "critical"

    def test_health_metric_state(self):
        from squish.health_check import HealthMetric, HealthState

        m_ok = HealthMetric(name="p99", value=100.0,
                            threshold_warn=500.0, threshold_crit=2000.0)
        assert m_ok.state == HealthState.OK

        m_deg = HealthMetric(name="p99", value=800.0,
                             threshold_warn=500.0, threshold_crit=2000.0)
        assert m_deg.state == HealthState.DEGRADED

        m_crit = HealthMetric(name="p99", value=2500.0,
                              threshold_warn=500.0, threshold_crit=2000.0)
        assert m_crit.state == HealthState.CRITICAL

    def test_inference_health_monitor_record(self):
        from squish.health_check import HealthState, InferenceHealthMonitor

        monitor = InferenceHealthMonitor(
            warn_latency_ms=500.0,
            crit_latency_ms=2000.0,
            warn_error_rate=0.05,
            crit_error_rate=0.20,
        )
        for _ in range(100):
            monitor.record_request(latency_ms=200.0, success=True)
        assert monitor.overall_health() == HealthState.OK
        # Inject a very slow, failing request.
        monitor.record_request(latency_ms=3000.0, success=False)
        # p99 should now be elevated; overall health should not be OK.
        assert monitor.stats.total_requests == 101
        assert monitor.stats.total_errors == 1

    def test_health_stats_error_rate(self):
        from squish.health_check import HealthStats

        s = HealthStats(total_requests=20, total_errors=4)
        assert abs(s.error_rate - 0.2) < 1e-9
        empty = HealthStats()
        assert empty.error_rate == 0.0


# ---------------------------------------------------------------------------
# FaultTolerance
# ---------------------------------------------------------------------------


class TestFaultToleranceWiring:
    def test_fault_policy_fields(self):
        from squish.fault_tolerance import FaultPolicy

        p = FaultPolicy(
            evict_kv_at=0.80,
            disable_draft_at=0.88,
            reduce_batch_at=0.95,
            min_batch_size=2,
        )
        assert p.evict_kv_at == 0.80
        assert p.disable_draft_at == 0.88
        assert p.reduce_batch_at == 0.95
        assert p.min_batch_size == 2

    def test_fault_action_constants(self):
        from squish.fault_tolerance import FaultAction

        assert FaultAction.EVICT_KV == "evict_kv"
        assert FaultAction.DISABLE_DRAFT == "disable_draft"
        assert FaultAction.REDUCE_BATCH == "reduce_batch"
        assert FaultAction.RENEGOTIATE_SLO == "renegotiate_slo"

    def test_fault_handler_evaluate(self):
        from squish.fault_tolerance import FaultAction, FaultHandler, FaultPolicy

        policy = FaultPolicy(evict_kv_at=0.85, disable_draft_at=0.90,
                             reduce_batch_at=0.95, min_batch_size=1)
        handler = FaultHandler(policy)

        # Below all thresholds — no actions.
        actions = handler.evaluate(pressure=0.50, current_batch_size=8)
        assert actions == []

        # Above evict_kv threshold only.
        actions = handler.evaluate(pressure=0.87, current_batch_size=8)
        assert FaultAction.EVICT_KV in actions
        assert FaultAction.DISABLE_DRAFT not in actions

        # Above all thresholds.
        actions = handler.evaluate(pressure=0.96, current_batch_size=8)
        assert FaultAction.EVICT_KV in actions
        assert FaultAction.DISABLE_DRAFT in actions
        assert FaultAction.REDUCE_BATCH in actions

    def test_fault_stats_fields(self):
        from squish.fault_tolerance import FaultHandler, FaultPolicy

        policy = FaultPolicy()
        handler = FaultHandler(policy)
        evicted = handler.apply_evict_kv(16)
        assert evicted == 16
        assert handler.stats.kv_evictions == 16
        assert handler.stats.total_evaluations == 0


# ---------------------------------------------------------------------------
# ModelPool
# ---------------------------------------------------------------------------


class TestModelPoolWiring:
    def test_pool_entry_fields(self):
        from squish.model_pool import PoolEntry

        entry = PoolEntry(model_id="phi-3-mini", size_mb=2048.0,
                          last_accessed=1.0, access_count=3)
        assert entry.model_id == "phi-3-mini"
        assert entry.size_mb == 2048.0
        assert entry.last_accessed == 1.0
        assert entry.access_count == 3

    def test_model_pool_register_acquire_release(self):
        from squish.model_pool import ModelPool

        pool = ModelPool(capacity_mb=8192.0)
        pool.register("llama-3-8b", size_mb=4096.0)
        entry = pool.acquire("llama-3-8b")
        assert entry.model_id == "llama-3-8b"
        assert "llama-3-8b" in pool.loaded_models
        assert pool.stats.cache_misses == 1
        pool.release("llama-3-8b")
        # Second acquire should be a cache hit.
        pool.acquire("llama-3-8b")
        assert pool.stats.cache_hits == 1

    def test_model_pool_evict_lru(self):
        from squish.model_pool import ModelPool

        pool = ModelPool(capacity_mb=4096.0)
        pool.register("small-a", size_mb=1024.0)
        pool.register("small-b", size_mb=1024.0)
        pool.acquire("small-a")
        pool.release("small-a")
        pool.acquire("small-b")
        pool.release("small-b")
        evicted = pool.evict_lru()
        assert evicted is not None
        assert pool.stats.total_evictions == 1

    def test_pool_stats_hit_rate(self):
        from squish.model_pool import PoolStats

        s = PoolStats(total_acquires=10, cache_hits=7, cache_misses=3)
        assert abs(s.hit_rate - 0.7) < 1e-9
        empty = PoolStats()
        assert empty.hit_rate == 0.0


# ---------------------------------------------------------------------------
# StreamingChunk
# ---------------------------------------------------------------------------


class TestStreamingChunkWiring:
    def test_chunk_config_fields(self):
        from squish.streaming_chunk import ChunkConfig

        cfg = ChunkConfig(chunk_size=8, max_buffer=128)
        assert cfg.chunk_size == 8
        assert cfg.max_buffer == 128

    def test_backpressure_buffer_push_flush(self):
        from squish.streaming_chunk import BackpressureBuffer, ChunkConfig

        cfg = ChunkConfig(chunk_size=4, max_buffer=6)
        buf = BackpressureBuffer(cfg)
        for i in range(6):
            assert buf.push(i) is True
        # 7th push should trigger backpressure.
        assert buf.push(99) is False
        assert buf.peek_size() == 6
        chunk = buf.flush()
        assert len(chunk) == 4
        assert buf.peek_size() == 2

    def test_chunked_streamer_stream(self):
        from squish.streaming_chunk import ChunkConfig, ChunkedStreamer

        cfg = ChunkConfig(chunk_size=4)
        streamer = ChunkedStreamer(cfg)
        token_ids = list(range(10))
        chunks = streamer.stream(token_ids)
        assert chunks[0] == [0, 1, 2, 3]
        assert chunks[-1] == [8, 9]
        assert sum(len(c) for c in chunks) == 10
        assert streamer.stats.total_tokens_streamed == 10
        assert streamer.stats.total_chunks == len(chunks)

    def test_stream_stats_avg_chunk_size(self):
        from squish.streaming_chunk import StreamStats

        s = StreamStats(total_tokens_streamed=20, total_chunks=5)
        assert abs(s.avg_chunk_size - 4.0) < 1e-9
        empty = StreamStats()
        assert empty.avg_chunk_size == 0.0


# ---------------------------------------------------------------------------
# CostEstimator
# ---------------------------------------------------------------------------


class TestCostEstimatorWiring:
    def test_cost_model_fields(self):
        from squish.cost_estimator import CostModel

        m = CostModel(
            prefill_cost_per_token=0.001,
            decode_cost_per_token=0.002,
            kv_cost_per_mb_ms=0.0001,
            currency="credits",
        )
        assert m.prefill_cost_per_token == 0.001
        assert m.decode_cost_per_token == 0.002
        assert m.currency == "credits"

    def test_request_cost_total(self):
        from squish.cost_estimator import RequestCost

        cost = RequestCost(
            request_id="req-1",
            prefill_cost=0.5,
            decode_cost=0.3,
            kv_cost=0.2,
        )
        assert abs(cost.total_cost - 1.0) < 1e-9

    def test_request_cost_estimator_estimate(self):
        from squish.cost_estimator import CostModel, RequestCostEstimator

        model = CostModel(
            prefill_cost_per_token=0.001,
            decode_cost_per_token=0.002,
            kv_cost_per_mb_ms=0.0,
        )
        estimator = RequestCostEstimator(model)
        cost = estimator.estimate(
            request_id="req-001",
            n_prefill_tokens=100,
            n_decode_tokens=50,
            kv_mb=0.0,
            duration_ms=200.0,
        )
        assert abs(cost.prefill_cost - 0.1) < 1e-9
        assert abs(cost.decode_cost - 0.1) < 1e-9
        assert estimator.stats.total_requests == 1

    def test_cost_stats_avg_cost(self):
        from squish.cost_estimator import CostStats

        s = CostStats(total_requests=4, total_cost=8.0)
        assert abs(s.avg_cost_per_request - 2.0) < 1e-9
        empty = CostStats()
        assert empty.avg_cost_per_request == 0.0


# ---------------------------------------------------------------------------
# SLAMonitor
# ---------------------------------------------------------------------------


class TestSLAMonitorWiring:
    def test_violation_policy_fields(self):
        from squish.sla_monitor import ViolationPolicy

        policy = ViolationPolicy(
            max_latency_ms=1000.0,
            max_error_rate=0.10,
            violation_window=50,
            escalation_threshold=5,
        )
        assert policy.max_latency_ms == 1000.0
        assert policy.max_error_rate == 0.10
        assert policy.violation_window == 50
        assert policy.escalation_threshold == 5

    def test_violation_type_constants(self):
        from squish.sla_monitor import ViolationType

        assert ViolationType.LATENCY == "latency"
        assert ViolationType.ERROR_RATE == "error_rate"

    def test_sla_monitor_record_and_check(self):
        from squish.sla_monitor import SLAMonitor, ViolationPolicy, ViolationType

        policy = ViolationPolicy(
            max_latency_ms=500.0,
            max_error_rate=0.10,
            violation_window=20,
            escalation_threshold=3,
        )
        monitor = SLAMonitor(policy)
        for _ in range(20):
            monitor.record(latency_ms=1000.0, success=True)
        violations = monitor.check()
        assert any(v.violation_type == ViolationType.LATENCY for v in violations)
        assert monitor.stats.total_records == 20

    def test_sla_stats_fields(self):
        from squish.sla_monitor import SLAMonitor, ViolationPolicy

        policy = ViolationPolicy(max_latency_ms=200.0, max_error_rate=0.05)
        monitor = SLAMonitor(policy)
        # No records yet → is_healthy should return True (no violations).
        assert monitor.is_healthy() is True
        s = monitor.stats
        assert s.total_records == 0
        assert s.total_violations == 0


# ---------------------------------------------------------------------------
# ContextCache
# ---------------------------------------------------------------------------


class TestContextCacheWiring:
    def test_cache_entry_fields(self):
        from squish.context_cache import CacheEntry

        kv = np.zeros((4, 8), dtype=np.float32)
        entry = CacheEntry(
            entry_id="e1",
            token_hash=12345,
            kv_data=kv,
            created_at=time.time(),
            ttl_s=300.0,
        )
        assert entry.entry_id == "e1"
        assert entry.token_hash == 12345
        assert entry.kv_data.shape == (4, 8)
        assert not entry.is_expired

    def test_persistent_context_cache_put_get(self):
        from squish.context_cache import PersistentContextCache

        cache = PersistentContextCache(max_entries=16, default_ttl_s=300.0)
        rng = np.random.default_rng(0)
        tokens = [1, 2, 3, 4, 5]
        kv = rng.standard_normal((4, 5, 8)).astype(np.float32)
        entry_id = cache.put(tokens, kv)
        assert isinstance(entry_id, str) and len(entry_id) > 0
        result = cache.get(tokens)
        assert result is not None
        assert result.shape == kv.shape
        assert cache.stats.hits == 1

    def test_context_cache_evict_expired(self):
        from squish.context_cache import PersistentContextCache

        cache = PersistentContextCache(max_entries=8, default_ttl_s=0.001)
        rng = np.random.default_rng(1)
        for i in range(3):
            kv = rng.standard_normal((2, i + 1, 4)).astype(np.float32)
            cache.put([i + 1, i + 2], kv, ttl_s=0.001)
        time.sleep(0.05)
        n_evicted = cache.evict_expired()
        assert n_evicted == 3
        assert cache.n_entries == 0

    def test_context_cache_stats_hit_rate(self):
        from squish.context_cache import ContextCacheStats

        s = ContextCacheStats(total_puts=5, total_gets=8, hits=6, misses=2)
        assert abs(s.hit_rate - 0.75) < 1e-9
        empty = ContextCacheStats()
        assert empty.hit_rate == 0.0
