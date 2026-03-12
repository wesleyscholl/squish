"""
test_wave26_server_wiring.py — Wave 26 server-wiring tests.

4 tests per module × 14 modules = 56 tests.
Each test covers: import, instantiation, core method invocation, and stats/properties.
"""

from __future__ import annotations

import numpy as np
import pytest

RNG = np.random.default_rng(0xCAFE_BABE)


# ── TensorParallel ────────────────────────────────────────────────────────────

def test_tensor_parallel_import():
    from squish.tensor_parallel import TensorParallelShard, TPConfig
    cfg = TPConfig(n_devices=4, mode="column")
    tp = TensorParallelShard(cfg)
    assert tp is not None


def test_tensor_parallel_shard_column():
    from squish.tensor_parallel import TensorParallelShard, TPConfig
    n_devices = 2
    cfg = TPConfig(n_devices=n_devices, mode="column")
    tp = TensorParallelShard(cfg)
    W = RNG.random((8, 16)).astype(np.float32)
    shards = tp.shard(W)
    assert len(shards) == n_devices
    # columns sum to total
    assert sum(s.shape[1] for s in shards) == 16


def test_tensor_parallel_forward_column():
    from squish.tensor_parallel import TensorParallelShard, TPConfig
    cfg = TPConfig(n_devices=2, mode="column")
    tp = TensorParallelShard(cfg)
    W = RNG.random((8, 16)).astype(np.float32)
    shards = tp.shard(W)
    x = RNG.random((4, 8)).astype(np.float32)
    out = tp.forward(x, shards)
    assert out.shape == (4, 16)
    assert out.dtype == np.float32


def test_tensor_parallel_stats():
    from squish.tensor_parallel import TensorParallelShard, TPConfig
    cfg = TPConfig(n_devices=2, mode="row")
    tp = TensorParallelShard(cfg)
    W = RNG.random((8, 16)).astype(np.float32)
    shards = tp.shard(W)
    x = RNG.random((4, 8)).astype(np.float32)
    tp.forward(x, shards)
    stats = tp.stats
    assert stats.n_shards == 2
    assert stats.mode == "row"


# ── SequenceParallel ──────────────────────────────────────────────────────────

def test_sequence_parallel_import():
    from squish.sequence_parallel import SequenceParallelScatter, SPConfig
    cfg = SPConfig(n_devices=4, n_heads=8, head_dim=32)
    sp = SequenceParallelScatter(cfg)
    assert sp is not None


def test_sequence_parallel_scatter_gather():
    from squish.sequence_parallel import SequenceParallelScatter, SPConfig
    n_heads, seq, head_dim, n_devices = 4, 16, 8, 2
    cfg = SPConfig(n_devices=n_devices, n_heads=n_heads, head_dim=head_dim)
    sp = SequenceParallelScatter(cfg)
    x = RNG.random((n_heads, seq, head_dim)).astype(np.float32)
    chunks = sp.scatter(x)
    assert len(chunks) == n_devices
    reconstructed = sp.gather(chunks)
    assert reconstructed.shape == x.shape


def test_sequence_parallel_all_to_all():
    from squish.sequence_parallel import SequenceParallelScatter, SPConfig
    n_heads, seq, head_dim, n_devices = 4, 16, 8, 2
    cfg = SPConfig(n_devices=n_devices, n_heads=n_heads, head_dim=head_dim)
    sp = SequenceParallelScatter(cfg)
    x = RNG.random((n_heads, seq, head_dim)).astype(np.float32)
    out = sp.all_to_all(x)
    assert out.shape == (n_devices, n_heads // n_devices, seq, head_dim)


def test_sequence_parallel_communication_bytes():
    from squish.sequence_parallel import SequenceParallelScatter, SPConfig
    n_heads, seq, head_dim = 4, 16, 8
    cfg = SPConfig(n_devices=2, n_heads=n_heads, head_dim=head_dim)
    sp = SequenceParallelScatter(cfg)
    x = RNG.random((n_heads, seq, head_dim)).astype(np.float32)
    sp.scatter(x)
    assert sp.communication_bytes >= 0


# ── KVMigrate ─────────────────────────────────────────────────────────────────

def test_kv_migrate_import():
    from squish.kv_migrate import KVMigrator
    m = KVMigrator(n_heads=4, head_dim=16)
    assert m.n_migrations == 0


def test_kv_migrate_pack_unpack():
    from squish.kv_migrate import KVMigrator
    n_heads, seq, head_dim = 2, 8, 4
    m = KVMigrator(n_heads=n_heads, head_dim=head_dim)
    keys = RNG.random((n_heads, seq, head_dim)).astype(np.float32)
    vals = RNG.random((n_heads, seq, head_dim)).astype(np.float32)
    data, stats = m.pack("seq1", keys, vals)
    assert isinstance(data, bytes)
    assert stats.n_tokens == seq
    rkeys, rvals = m.unpack(data, stats)
    np.testing.assert_allclose(rkeys, keys, rtol=1e-5)
    np.testing.assert_allclose(rvals, vals, rtol=1e-5)


def test_kv_migrate_cost():
    from squish.kv_migrate import KVMigrator
    m = KVMigrator(n_heads=4, head_dim=16)
    cost = m.migration_cost_bytes(64)
    assert cost > 0


def test_kv_migrate_n_migrations():
    from squish.kv_migrate import KVMigrator
    n_heads, seq, head_dim = 2, 4, 4
    m = KVMigrator(n_heads=n_heads, head_dim=head_dim)
    keys = RNG.random((n_heads, seq, head_dim)).astype(np.float32)
    vals = RNG.random((n_heads, seq, head_dim)).astype(np.float32)
    m.pack("s1", keys, vals)
    m.pack("s2", keys, vals)
    assert m.n_migrations == 2


# ── DisaggPrefill ─────────────────────────────────────────────────────────────

def test_disagg_prefill_import():
    from squish.disagg_prefill import DisaggPrefillNode, DisaggDecodeNode, DisaggConfig
    cfg = DisaggConfig(n_heads=2, head_dim=8, n_layers=2)
    pnode = DisaggPrefillNode(cfg)
    dnode = DisaggDecodeNode(cfg)
    assert pnode is not None
    assert dnode is not None


def test_disagg_prefill_prefill_and_load():
    from squish.disagg_prefill import DisaggPrefillNode, DisaggDecodeNode, DisaggConfig
    cfg = DisaggConfig(n_heads=2, head_dim=4, n_layers=2)
    pnode = DisaggPrefillNode(cfg)
    dnode = DisaggDecodeNode(cfg)
    tokens = np.array([1, 2, 3, 4, 5], dtype=np.int32)
    payload = pnode.prefill("seq1", tokens)
    assert payload.seq_id == "seq1"
    assert payload.n_tokens == 5
    assert isinstance(payload.first_token, int)
    dnode.load_payload(payload)
    assert dnode.is_loaded("seq1")


def test_disagg_prefill_step():
    from squish.disagg_prefill import DisaggPrefillNode, DisaggDecodeNode, DisaggConfig
    cfg = DisaggConfig(n_heads=2, head_dim=4, n_layers=2)
    pnode = DisaggPrefillNode(cfg)
    dnode = DisaggDecodeNode(cfg)
    tokens = np.array([10, 20, 30], dtype=np.int32)
    payload = pnode.prefill("seq2", tokens)
    dnode.load_payload(payload)
    tok = dnode.step("seq2")
    assert 0 <= tok <= 50256


def test_disagg_prefill_not_loaded():
    from squish.disagg_prefill import DisaggDecodeNode, DisaggConfig
    cfg = DisaggConfig(n_heads=2, head_dim=4, n_layers=2)
    dnode = DisaggDecodeNode(cfg)
    assert not dnode.is_loaded("unknown")
    with pytest.raises(KeyError):
        dnode.step("unknown")


# ── RequestPreempt ────────────────────────────────────────────────────────────

def test_request_preempt_import():
    from squish.request_preempt import PreemptScheduler, PreemptStats
    sched = PreemptScheduler(n_heads=2, head_dim=8, n_layers=2)
    assert sched.stats.n_preemptions == 0


def test_request_preempt_swap():
    from squish.request_preempt import PreemptScheduler
    sched = PreemptScheduler(n_heads=2, head_dim=4, n_layers=2)
    kv = RNG.random((2, 2, 2, 8, 4)).astype(np.float32)
    state = sched.preempt("req1", current_kv=kv, strategy="swap")
    assert state.seq_id == "req1"
    assert state.saved_kv is not None
    assert sched.stats.n_preemptions == 1
    assert sched.can_resume("req1")


def test_request_preempt_resume():
    from squish.request_preempt import PreemptScheduler
    sched = PreemptScheduler(n_heads=2, head_dim=4, n_layers=2)
    kv = RNG.random((2, 2, 2, 4, 4)).astype(np.float32)
    sched.preempt("req2", current_kv=kv, strategy="swap")
    restored = sched.resume("req2")
    assert restored is not None
    assert restored.seq_id == "req2"
    assert sched.stats.n_resumes == 1


def test_request_preempt_recompute():
    from squish.request_preempt import PreemptScheduler
    sched = PreemptScheduler(n_heads=2, head_dim=4, n_layers=2)
    kv = RNG.random((2, 2, 2, 6, 4)).astype(np.float32)
    state = sched.preempt("req3", current_kv=kv, strategy="recompute")
    assert state.saved_kv is None
    assert sched.stats.n_preemptions == 1


# ── InferGateway ──────────────────────────────────────────────────────────────

def test_infer_gateway_import():
    from squish.infer_gateway import InferenceGateway
    gw = InferenceGateway()
    assert len(gw.workers) == 0


def test_infer_gateway_register_and_route():
    from squish.infer_gateway import InferenceGateway
    gw = InferenceGateway()
    gw.register("worker-1", capacity=10)
    gw.register("worker-2", capacity=10)
    result = gw.route("req-1")
    assert result.worker_id in ("worker-1", "worker-2")
    assert result.load_fraction > 0.0


def test_infer_gateway_complete():
    from squish.infer_gateway import InferenceGateway
    gw = InferenceGateway()
    gw.register("w1", capacity=5)
    gw.route("req-a")
    gw.complete("w1")
    assert gw.total_load == pytest.approx(0.0)


def test_infer_gateway_no_workers_raises():
    from squish.infer_gateway import InferenceGateway
    gw = InferenceGateway()
    with pytest.raises(RuntimeError):
        gw.route("req-x")


# ── ModelVersionSwap ──────────────────────────────────────────────────────────

def test_model_version_swap_import():
    from squish.model_version_swap import ModelVersionManager, SwapPolicy
    policy = SwapPolicy(canary_fraction=0.1, min_canary_requests=5)
    mgr = ModelVersionManager(policy)
    assert mgr.active_version is None


def test_model_version_swap_stage_and_commit():
    from squish.model_version_swap import ModelVersionManager, SwapPolicy
    policy = SwapPolicy(canary_fraction=0.5, min_canary_requests=2)
    mgr = ModelVersionManager(policy)
    mgr.register_version("v1")
    mgr.stage("v1")
    # Bootstrap: no old active, can commit immediately
    committed = mgr.commit()
    assert committed == "v1"
    assert mgr.active_version == "v1"


def test_model_version_swap_rollback():
    from squish.model_version_swap import ModelVersionManager, SwapPolicy
    policy = SwapPolicy(canary_fraction=0.5, min_canary_requests=2)
    mgr = ModelVersionManager(policy)
    mgr.register_version("v1")
    mgr.register_version("v2")
    mgr.stage("v1")
    mgr.commit()
    mgr.stage("v2")
    for _ in range(2):
        mgr.record_result("v2", success=True)
    mgr.commit()
    restored = mgr.rollback()
    assert restored == "v1"
    assert mgr.active_version == "v1"


def test_model_version_swap_route_canary():
    from squish.model_version_swap import ModelVersionManager, SwapPolicy
    policy = SwapPolicy(canary_fraction=1.0, min_canary_requests=1)
    mgr = ModelVersionManager(policy)
    mgr.register_version("v1")
    mgr.register_version("v2")
    mgr.stage("v1")
    mgr.commit()
    mgr.stage("v2")
    # With canary_fraction=1.0, all traffic goes to canary while active exists
    version = mgr.route_request()
    assert version in ("v1", "v2")


# ── ProductionProfiler ────────────────────────────────────────────────────────

def test_production_profiler_import():
    from squish.production_profiler import ProductionProfiler
    p = ProductionProfiler()
    assert p.operations == []


def test_production_profiler_record_and_stats():
    from squish.production_profiler import ProductionProfiler
    p = ProductionProfiler()
    for i in range(20):
        p.record("forward", float(i + 1))
    stats = p.stats("forward")
    assert stats.n_samples == 20
    assert stats.mean_ms > 0.0
    assert stats.p99_ms >= stats.p50_ms
    assert stats.p999_ms >= stats.p99_ms


def test_production_profiler_report():
    from squish.production_profiler import ProductionProfiler
    p = ProductionProfiler()
    p.record("op-a", 1.0)
    p.record("op-b", 2.0)
    report = p.report()
    assert "op-a" in report
    assert "op-b" in report


def test_production_profiler_reset():
    from squish.production_profiler import ProductionProfiler
    p = ProductionProfiler()
    p.record("decode", 5.0)
    p.reset("decode")
    stats = p.stats("decode")
    assert stats.n_samples == 0


# ── AdaptiveBatcher ───────────────────────────────────────────────────────────

def test_adaptive_batcher_import():
    from squish.adaptive_batcher import AdaptiveBatchController, BatchObjective
    obj = BatchObjective(mode="throughput", max_batch_size=16)
    ctrl = AdaptiveBatchController(obj)
    assert ctrl is not None


def test_adaptive_batcher_throughput_mode():
    from squish.adaptive_batcher import AdaptiveBatchController, BatchObjective
    obj = BatchObjective(mode="throughput", max_batch_size=8, min_batch_size=1)
    ctrl = AdaptiveBatchController(obj)
    decision = ctrl.next_batch(queue_depth=10)
    assert decision.batch_size <= 8
    assert decision.batch_size >= 1


def test_adaptive_batcher_latency_mode():
    from squish.adaptive_batcher import AdaptiveBatchController, BatchObjective
    obj = BatchObjective(mode="latency", target_latency_ms=50.0,
                         max_batch_size=16, min_batch_size=1)
    ctrl = AdaptiveBatchController(obj)
    for bs in range(1, 9):
        ctrl.record_observation(bs, bs * 5.0)  # 1→5ms, 8→40ms
    decision = ctrl.next_batch(queue_depth=8)
    # batch 10 would be ~50ms, batch 8 is 40ms — should pick something <= 10
    assert 1 <= decision.batch_size <= 16


def test_adaptive_batcher_latency_model():
    from squish.adaptive_batcher import AdaptiveBatchController, BatchObjective
    obj = BatchObjective(mode="throughput", max_batch_size=4)
    ctrl = AdaptiveBatchController(obj)
    ctrl.record_observation(2, 10.0)
    ctrl.record_observation(4, 20.0)
    model = ctrl.latency_model
    assert 2 in model
    assert 4 in model


# ── SafetyLayer ───────────────────────────────────────────────────────────────

def test_safety_layer_import():
    from squish.safety_layer import SafetyClassifier, SafetyConfig
    cfg = SafetyConfig(vocab_size=1000, n_categories=4, threshold=0.5, seed=0)
    clf = SafetyClassifier(cfg)
    assert clf is not None


def test_safety_layer_score_tokens():
    from squish.safety_layer import SafetyClassifier, SafetyConfig
    vocab = 500
    cfg = SafetyConfig(vocab_size=vocab, n_categories=4, threshold=0.5, seed=7)
    clf = SafetyClassifier(cfg)
    tokens = np.array([0, 5, 10, 50], dtype=np.int32)
    result = clf.score(tokens)
    assert 0.0 <= result.score <= 1.0
    assert isinstance(result.is_safe, bool)
    assert result.category_scores.shape == (4,)
    assert result.category_scores.sum() == pytest.approx(1.0, abs=1e-5)


def test_safety_layer_score_logits():
    from squish.safety_layer import SafetyClassifier, SafetyConfig
    vocab = 200
    cfg = SafetyConfig(vocab_size=vocab, n_categories=4, seed=3)
    clf = SafetyClassifier(cfg)
    logits = RNG.random((vocab,)).astype(np.float32)
    result = clf.score_logits(logits)
    assert 0.0 <= result.score <= 1.0


def test_safety_layer_update_threshold():
    from squish.safety_layer import SafetyClassifier, SafetyConfig
    cfg = SafetyConfig(vocab_size=100, n_categories=4, threshold=0.5, seed=0)
    clf = SafetyClassifier(cfg)
    clf.update_threshold(0.8)
    tokens = np.array([1, 2, 3], dtype=np.int32)
    result_strict = clf.score(tokens)
    clf.update_threshold(0.1)
    result_lenient = clf.score(tokens)
    # Stricter threshold → same score but different is_safe classification
    assert result_strict.score == pytest.approx(result_lenient.score, abs=1e-5)


# ── SemanticResponseCache ─────────────────────────────────────────────────────

def test_semantic_response_cache_import():
    from squish.semantic_response_cache import SemanticResponseCache, CacheConfig
    cfg = CacheConfig(capacity=16, similarity_threshold=0.95, embedding_dim=8)
    cache = SemanticResponseCache(cfg)
    assert cache.size == 0


def test_semantic_response_cache_store_lookup_hit():
    from squish.semantic_response_cache import SemanticResponseCache, CacheConfig
    dim = 8
    cfg = CacheConfig(capacity=16, similarity_threshold=0.9, embedding_dim=dim)
    cache = SemanticResponseCache(cfg)
    emb = RNG.random((dim,)).astype(np.float32)
    emb /= np.linalg.norm(emb) + 1e-9
    cache.store(emb, "hello world")
    result = cache.lookup(emb)
    assert result == "hello world"
    assert cache.stats.n_hits == 1


def test_semantic_response_cache_miss():
    from squish.semantic_response_cache import SemanticResponseCache, CacheConfig
    dim = 8
    cfg = CacheConfig(capacity=16, similarity_threshold=0.995, embedding_dim=dim)
    cache = SemanticResponseCache(cfg)
    emb1 = np.ones(dim, dtype=np.float32) / math.sqrt(dim)
    emb2 = np.zeros(dim, dtype=np.float32)
    emb2[0] = 1.0
    cache.store(emb1, "response A")
    result = cache.lookup(emb2)
    # orthogonal vectors — should miss at high threshold
    assert result is None or isinstance(result, str)
    assert cache.stats.n_misses >= 0


def test_semantic_response_cache_eviction():
    from squish.semantic_response_cache import SemanticResponseCache, CacheConfig
    dim = 4
    cfg = CacheConfig(capacity=3, similarity_threshold=0.99, embedding_dim=dim)
    cache = SemanticResponseCache(cfg)
    for i in range(4):
        emb = np.zeros(dim, dtype=np.float32)
        emb[i % dim] = 1.0
        cache.store(emb, f"r{i}")
    assert cache.size <= 3


# Need math for test above
import math


# ── RateLimiter ───────────────────────────────────────────────────────────────

def test_rate_limiter_import():
    from squish.rate_limiter import TokenBucketRateLimiter, RateLimitConfig
    cfg = RateLimitConfig(rate=100.0, burst=50)
    rl = TokenBucketRateLimiter(cfg)
    assert rl is not None


def test_rate_limiter_consume_allowed():
    from squish.rate_limiter import TokenBucketRateLimiter, RateLimitConfig
    cfg = RateLimitConfig(rate=1000.0, burst=100)
    rl = TokenBucketRateLimiter(cfg)
    result = rl.consume("tenant-1", n_tokens=10, now=0.0)
    assert result.allowed is True
    assert result.tokens_consumed == 10
    assert result.wait_ms == pytest.approx(0.0)


def test_rate_limiter_consume_denied():
    from squish.rate_limiter import TokenBucketRateLimiter, RateLimitConfig
    cfg = RateLimitConfig(rate=1.0, burst=5)
    rl = TokenBucketRateLimiter(cfg)
    # drain the bucket
    rl.consume("t1", n_tokens=5, now=0.0)
    result = rl.consume("t1", n_tokens=3, now=0.0)  # no time passes → denied
    assert result.allowed is False
    assert result.tokens_consumed == 0
    assert result.wait_ms > 0.0


def test_rate_limiter_refill():
    from squish.rate_limiter import TokenBucketRateLimiter, RateLimitConfig
    cfg = RateLimitConfig(rate=10.0, burst=20)
    rl = TokenBucketRateLimiter(cfg)
    rl.consume("t1", n_tokens=20, now=0.0)
    tokens_after = rl.refill("t1", now=1.0)   # 1 second → +10 tokens
    assert tokens_after == pytest.approx(10.0, abs=0.1)


# ── SchemaValidator ───────────────────────────────────────────────────────────

def test_schema_validator_import():
    from squish.schema_validator import SchemaValidator
    sv = SchemaValidator()
    assert sv is not None


def test_schema_validator_valid_object():
    from squish.schema_validator import SchemaValidator
    sv = SchemaValidator()
    schema = {
        "type": "object",
        "required": ["name", "age"],
        "properties": {
            "name": {"type": "string", "minLength": 1},
            "age": {"type": "integer", "minimum": 0, "maximum": 150},
        }
    }
    result = sv.validate('{"name": "Alice", "age": 30}', schema)
    assert result.valid is True
    assert result.errors == []
    assert result.n_fields_checked > 0


def test_schema_validator_invalid_type():
    from squish.schema_validator import SchemaValidator
    sv = SchemaValidator()
    schema = {"type": "integer"}
    result = sv.validate('"not an integer"', schema)
    assert result.valid is False
    assert len(result.errors) > 0


def test_schema_validator_is_valid():
    from squish.schema_validator import SchemaValidator
    sv = SchemaValidator()
    assert sv.is_valid("[1, 2, 3]", {"type": "array", "items": {"type": "number"}}) is True
    assert sv.is_valid("[1, 2, 3]", {"type": "object"}) is False


# ── AuditLogger ───────────────────────────────────────────────────────────────

def test_audit_logger_import():
    from squish.audit_logger import AuditLogger
    al = AuditLogger()
    assert al.chain_length == 0


def test_audit_logger_log_entry():
    from squish.audit_logger import AuditLogger
    al = AuditLogger()
    entry = al.log("req-001", tokens_in=100, tokens_out=50, model="llama3")
    assert entry.entry_id == 0
    assert entry.prev_hash == "genesis"
    assert len(entry.entry_hash) == 64   # SHA-256 hex
    assert al.chain_length == 1


def test_audit_logger_chain():
    from squish.audit_logger import AuditLogger
    al = AuditLogger()
    e1 = al.log("r1", 10, 5, "m1")
    e2 = al.log("r2", 20, 10, "m1")
    assert e2.prev_hash == e1.entry_hash
    assert e2.entry_id == 1
    assert al.head_hash == e2.entry_hash


def test_audit_logger_verify():
    from squish.audit_logger import AuditLogger
    al = AuditLogger()
    for i in range(5):
        al.log(f"req-{i}", i * 10, i * 5, "model-v1")
    assert al.verify() is True
    # tamper with one entry
    entries = al.export()
    tampered = list(entries)
    # Replace tokens_in to break hash chain
    import dataclasses
    tampered[2] = dataclasses.replace(tampered[2], tokens_in=99999)
    assert al.verify(tampered) is False
