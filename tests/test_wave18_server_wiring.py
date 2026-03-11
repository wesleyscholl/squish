"""tests/test_wave18_server_wiring.py

Verifies that all Wave 18 module classes are importable and have the expected
public APIs that the server.py wiring code depends on.  These are pure
import + instantiation tests — no model or GPU required.

Wave 18 modules (Adaptive Compute + Model Intelligence + Evaluation):
  vptq, layer_skip, swift, spec_reason, mirror_sd, sparse_verify,
  robust_scheduler, block_expert_archive, disc_router, self_learning,
  semantic_cache, ipw, power_monitor, diffusion_draft
"""

import tempfile

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Detect optional heavyweight dependencies once at module scope
# ---------------------------------------------------------------------------

try:
    import sqlite3 as _sqlite3

    import sqlite_vec as _sqlite_vec  # noqa: F401

    _conn = _sqlite3.connect(":memory:")
    _conn.enable_load_extension(True)  # raises AttributeError if Python built without support
    _conn.close()
    _HAS_SQLITE_VEC = True
except (ImportError, AttributeError):
    _HAS_SQLITE_VEC = False


# ---------------------------------------------------------------------------
# VPTQ (Vector Post-Training Quantization)
# ---------------------------------------------------------------------------


class TestVPTQWiring:
    def test_import(self):
        from squish.vptq import VPTQConfig, VPTQQuantizer

        cfg   = VPTQConfig(n_codebook_entries=16, group_size=4)
        quant = VPTQQuantizer(cfg)
        assert quant is not None

    def test_config_defaults(self):
        from squish.vptq import VPTQConfig

        cfg = VPTQConfig()
        assert cfg.n_codebook_entries >= 2
        assert cfg.group_size >= 1
        assert cfg.n_fit_iters >= 1

    def test_compress_decompress_roundtrip(self):
        from squish.vptq import VPTQConfig, VPTQLayer, VPTQQuantizer

        rng    = np.random.default_rng(0)
        cfg    = VPTQConfig(n_codebook_entries=16, group_size=4, n_fit_iters=5)
        quant  = VPTQQuantizer(cfg)
        weight = rng.standard_normal((16, 8)).astype(np.float32)
        layer  = quant.compress(weight)
        assert isinstance(layer, VPTQLayer)
        approx = quant.decompress(layer)
        assert approx.shape == weight.shape

    def test_codebook_fit_encode_decode(self):
        from squish.vptq import VPTQCodebook

        rng = np.random.default_rng(1)
        cb  = VPTQCodebook(group_size=4, n_codebook_entries=8, n_fit_iters=5)
        assert not cb.is_fitted
        data = rng.standard_normal((32, 4)).astype(np.float32)
        cb.fit(data)
        assert cb.is_fitted
        indices = cb.encode(data[:4])
        assert indices.shape == (4,)
        decoded = cb.decode(indices)
        assert decoded.shape == (4, 4)

    def test_decompress_layer_function(self):
        from squish.vptq import VPTQConfig, VPTQQuantizer, decompress_layer

        rng    = np.random.default_rng(2)
        cfg    = VPTQConfig(n_codebook_entries=8, group_size=4, n_fit_iters=5)
        quant  = VPTQQuantizer(cfg)
        weight = rng.standard_normal((8, 4)).astype(np.float32)
        layer  = quant.compress(weight)
        approx = decompress_layer(layer)
        assert approx.shape == weight.shape


# ---------------------------------------------------------------------------
# LayerSkip (EarlyExitDecoder)
# ---------------------------------------------------------------------------


class TestLayerSkipWiring:
    def test_import(self):
        from squish.layer_skip import ConfidenceEstimator, EarlyExitConfig, EarlyExitDecoder

        cfg = EarlyExitConfig(num_layers=8, exit_layer=4, confidence_threshold=0.9)
        est = ConfidenceEstimator(metric="max_prob")
        rng = np.random.default_rng(0)

        def full_forward(token_ids, exit_layer=None):
            return rng.standard_normal(32).astype(np.float32)

        dec = EarlyExitDecoder(full_forward, cfg)
        assert dec is not None
        assert est is not None

    def test_config_defaults(self):
        from squish.layer_skip import EarlyExitConfig

        cfg = EarlyExitConfig()
        assert cfg.num_layers >= 1
        assert 0 < cfg.exit_layer < cfg.num_layers
        assert 0.0 < cfg.confidence_threshold <= 1.0
        assert cfg.mode in ("early_exit", "self_spec", "hybrid")

    def test_confidence_estimator(self):
        from squish.layer_skip import ConfidenceEstimator

        est = ConfidenceEstimator(metric="max_prob")
        # One-hot-ish logits → high confidence
        logits = np.zeros(32, dtype=np.float32)
        logits[5] = 10.0
        conf = est.estimate(logits)
        assert 0.0 <= conf <= 1.0
        assert conf > 0.9
        tok = est.top_token(logits)
        assert tok == 5

    def test_stats_counters(self):
        from squish.layer_skip import EarlyExitStats

        s = EarlyExitStats()
        assert s.early_exit_rate == 0.0
        assert s.acceptance_rate == 0.0


# ---------------------------------------------------------------------------
# SWIFT (task-adaptive layer skip)
# ---------------------------------------------------------------------------


class TestSWIFTWiring:
    def test_import(self):
        from squish.swift import SWIFTConfig, SWIFTDecoder, SWIFTLayerConfig

        cfg     = SWIFTConfig(num_layers=8, initial_skip_fraction=0.3)
        lc      = SWIFTLayerConfig(task_type="chat", skip_layers=[2, 4, 6])
        rng     = np.random.default_rng(0)

        def forward_fn(*args, **kwargs):
            return rng.dirichlet(np.ones(32)).astype(np.float32)

        dec = SWIFTDecoder(forward_fn, {"chat": lc}, cfg)
        assert dec is not None

    def test_config_defaults(self):
        from squish.swift import SWIFTConfig

        cfg = SWIFTConfig()
        assert cfg.num_layers >= 1
        assert 0.0 <= cfg.initial_skip_fraction < 1.0
        assert cfg.n_calibration_steps >= 1
        assert 0.0 < cfg.cooling_rate <= 1.0

    def test_layer_config_serialization(self):
        from squish.swift import SWIFTLayerConfig

        lc  = SWIFTLayerConfig(task_type="code", skip_layers=[1, 3, 5], calibration_score=0.85)
        d   = lc.to_dict()
        assert d["task_type"] == "code"
        assert d["skip_layers"] == [1, 3, 5]
        restored = SWIFTLayerConfig.from_dict(d)
        assert restored.task_type == "code"
        assert restored.skip_layers == [1, 3, 5]

    def test_stats_acceptance_rate(self):
        from squish.swift import SWIFTStats

        s = SWIFTStats(total_tokens=10, accepted_draft=7, rejected_draft=3)
        assert abs(s.acceptance_rate - 0.7) < 1e-6


# ---------------------------------------------------------------------------
# SpecReason
# ---------------------------------------------------------------------------


class TestSpecReasonWiring:
    def test_import(self):
        from squish.spec_reason import ReasoningStep, SpecReasonConfig, SpecReasonOrchestrator

        call_n = [0]

        def draft_fn(ctx):
            call_n[0] += 1
            return ReasoningStep(text=f"draft{call_n[0]}", source="draft",
                                 confidence=0.8, tokens_used=5)

        def target_fn(ctx):
            return ReasoningStep(text="target", source="target",
                                 confidence=0.95, tokens_used=10)

        cfg  = SpecReasonConfig(min_acceptance_score=0.75, max_draft_steps=2)
        orch = SpecReasonOrchestrator(cfg, draft_fn=draft_fn, target_fn=target_fn)
        assert orch is not None

    def test_config_defaults(self):
        from squish.spec_reason import SpecReasonConfig

        cfg = SpecReasonConfig()
        assert 0.0 < cfg.min_acceptance_score <= 1.0
        assert cfg.max_step_tokens >= 1
        assert cfg.max_draft_steps >= 1

    def test_generate_step(self):
        from squish.spec_reason import (
            ReasoningStep,
            SpecReasonConfig,
            SpecReasonOrchestrator,
            StepVerdict,
        )

        def draft_fn(ctx):
            return ReasoningStep(text="step", source="draft",
                                 confidence=0.9, tokens_used=3)

        def target_fn(ctx):
            return ReasoningStep(text="verified", source="target",
                                 confidence=0.99, tokens_used=5)

        cfg  = SpecReasonConfig(min_acceptance_score=0.5)
        orch = SpecReasonOrchestrator(cfg, draft_fn, target_fn)
        step, verdict = orch.generate_step("context")
        assert isinstance(step, ReasoningStep)
        assert verdict in (StepVerdict.ACCEPT, StepVerdict.REJECT, StepVerdict.PARTIAL)

    def test_stats_and_reset(self):
        from squish.spec_reason import ReasoningStep, SpecReasonConfig, SpecReasonOrchestrator

        def draft_fn(ctx):
            return ReasoningStep(text="a", source="draft", confidence=0.9, tokens_used=1)

        def target_fn(ctx):
            return ReasoningStep(text="b", source="target", confidence=0.95, tokens_used=2)

        cfg  = SpecReasonConfig(max_draft_steps=2)
        orch = SpecReasonOrchestrator(cfg, draft_fn, target_fn)
        orch.generate_chain("start", max_steps=3)
        s = orch.stats
        assert s.total_steps >= 1
        orch.reset()
        assert orch.stats.total_steps == 0


# ---------------------------------------------------------------------------
# MirrorSD
# ---------------------------------------------------------------------------


class TestMirrorSDWiring:
    def test_import(self):
        from squish.mirror_sd import (
            MirrorDraftPipeline,
            MirrorSDConfig,
            MirrorSDDecoder,
            MirrorVerifyPipeline,
        )

        rng = np.random.default_rng(0)

        def draft_fn(ids):
            return rng.dirichlet(np.ones(32)).astype(np.float32)

        def target_fn(ids):
            return rng.dirichlet(np.ones(32)).astype(np.float32)

        cfg     = MirrorSDConfig(gamma=2)
        drft    = MirrorDraftPipeline(draft_fn, cfg, rng_seed=0)
        vrfy    = MirrorVerifyPipeline(target_fn, cfg, rng_seed=1)
        dec     = MirrorSDDecoder(drft, vrfy, cfg)
        assert dec is not None

    def test_config_defaults(self):
        from squish.mirror_sd import MirrorSDConfig

        cfg = MirrorSDConfig()
        assert cfg.gamma >= 1
        assert 0.0 < cfg.temperature
        assert 0.0 < cfg.top_p <= 1.0
        assert cfg.overlap_steps >= 0

    def test_draft_pipeline_step(self):
        from squish.mirror_sd import MirrorDraftPipeline, MirrorSDConfig

        rng = np.random.default_rng(0)

        def draft_fn(ids):
            return rng.dirichlet(np.ones(16)).astype(np.float32)

        cfg  = MirrorSDConfig(gamma=2, temperature=1.0)
        pipe = MirrorDraftPipeline(draft_fn, cfg, rng_seed=42)
        token, probs = pipe.step([1, 2, 3])
        assert isinstance(token, (int, np.integer))
        assert probs.shape == (16,)
        assert abs(probs.sum() - 1.0) < 1e-4

    def test_stats_mean_accepted(self):
        from squish.mirror_sd import MirrorSDStats

        s = MirrorSDStats(total_tokens=10, draft_steps=5,
                          accepted_total=8, rejected_total=2)
        assert 0.0 <= s.acceptance_rate <= 1.0
        assert s.mean_accepted_per_step >= 0.0


# ---------------------------------------------------------------------------
# SparseVerify
# ---------------------------------------------------------------------------


class TestSparseVerifyWiring:
    def test_import(self):
        from squish.sparse_verify import InterDraftReuseCache, SparseVerifyConfig, SparseVerifyPass

        cfg  = SparseVerifyConfig(attn_sparsity=0.5, reuse_budget=32)
        cache = InterDraftReuseCache(budget=32)

        def verify_fn(ctx, drafts):
            return drafts[:], [np.zeros(len(drafts), dtype=np.float32)]

        sv = SparseVerifyPass(verify_fn, cfg, rng_seed=0)
        assert sv is not None
        assert cache is not None

    def test_config_defaults(self):
        from squish.sparse_verify import SparseVerifyConfig

        cfg = SparseVerifyConfig()
        assert 0.0 <= cfg.attn_sparsity <= 1.0
        assert 0.0 <= cfg.ffn_sparsity <= 1.0
        assert cfg.reuse_budget >= 1
        assert 0.0 <= cfg.min_confidence <= 1.0

    def test_reuse_cache_record_and_query(self):
        from squish.sparse_verify import InterDraftReuseCache

        cache   = InterDraftReuseCache(budget=16)
        indices = np.array([0, 1, 2, 3], dtype=np.int64)
        cache.record(draft_pos=0, kv_indices=indices)
        assert cache.hit_count + cache.miss_count == 0  # no queries yet
        reused, n_hit = cache.query_reuse(0, indices)
        assert n_hit >= 0

    def test_stats_counters(self):
        from squish.sparse_verify import SparseVerifyStats

        s = SparseVerifyStats(verify_calls=5, tokens_evaluated=20,
                              reuse_hits=4, attn_ops_saved=10, ffn_ops_saved=5)
        assert s.ops_saved_total == 15
        assert s.mean_tokens_per_call == 4.0
        assert 0.0 <= s.reuse_rate <= 1.0


# ---------------------------------------------------------------------------
# RobustScheduler
# ---------------------------------------------------------------------------


class TestRobustSchedulerWiring:
    def test_import(self):
        from squish.robust_scheduler import (
            AMaxScheduler,
            LengthInterval,
            Request,
            RobustSchedulerConfig,
        )

        cfg  = RobustSchedulerConfig(max_batch_tokens=1024, max_batch_size=8)
        iv   = LengthInterval(lo=64, hi=256)
        req  = Request(request_id="r1", input_len=32, length_interval=iv)
        sched = AMaxScheduler(cfg)
        sched.enqueue(req)
        assert sched.queue_size == 1

    def test_config_defaults(self):
        from squish.robust_scheduler import RobustSchedulerConfig

        cfg = RobustSchedulerConfig()
        assert cfg.max_batch_tokens >= 1
        assert cfg.max_batch_size >= 1
        assert 0.0 < cfg.memory_pressure_threshold <= 1.0
        assert 0.0 <= cfg.alpha <= 1.0

    def test_length_interval_properties(self):
        from squish.robust_scheduler import LengthInterval

        iv = LengthInterval(lo=100, hi=200)
        assert iv.midpoint == 150
        assert iv.range_width == 100
        eff = iv.effective_length(alpha=0.5)
        assert iv.lo <= eff <= iv.hi

    def test_schedule_batch_and_complete(self):
        from squish.robust_scheduler import (
            AMaxScheduler,
            LengthInterval,
            Request,
            RobustSchedulerConfig,
        )

        cfg   = RobustSchedulerConfig(max_batch_tokens=2048, max_batch_size=4)
        sched = AMaxScheduler(cfg)
        for i in range(4):
            iv  = LengthInterval(lo=32, hi=128)
            req = Request(request_id=f"r{i}", input_len=16, length_interval=iv)
            sched.enqueue(req)
        batch = sched.schedule_batch()
        assert len(batch) >= 1
        sched.complete(batch[0].request_id)
        s = sched.stats
        assert s.total_scheduled >= 1

    def test_balanced_scheduler_pressure(self):
        from squish.robust_scheduler import ABalancedScheduler, RobustSchedulerConfig

        cfg   = RobustSchedulerConfig(max_batch_tokens=512, max_batch_size=4)
        sched = ABalancedScheduler(cfg)
        assert 0.0 <= sched.memory_pressure <= 1.0
        assert 0.0 <= sched.current_alpha <= 1.0


# ---------------------------------------------------------------------------
# BlockExpertArchive
# ---------------------------------------------------------------------------


class TestBlockExpertArchiveWiring:
    def test_import(self):
        from squish.block_expert_archive import BlockExpertArchive, BlockExpertConfig, ExpertRouter

        rng     = np.random.default_rng(0)
        dim     = 16
        n_blk   = 2
        n_exp   = 3
        cfg     = BlockExpertConfig(n_clusters=n_exp, n_iter=5)
        blk_w   = {i: [rng.standard_normal(dim).astype(np.float32) for _ in range(n_exp)]
                   for i in range(n_blk)}
        base_w  = {i: rng.standard_normal(dim).astype(np.float32) for i in range(n_blk)}
        with tempfile.TemporaryDirectory() as tmp:
            archive = BlockExpertArchive.create(tmp, blk_w, base_w, cfg)
            assert archive is not None
            assert isinstance(archive.router, ExpertRouter)

    def test_config_defaults(self):
        from squish.block_expert_archive import BlockExpertConfig

        cfg = BlockExpertConfig()
        assert cfg.n_clusters >= 2
        assert cfg.n_iter >= 1
        assert cfg.delta_bits in (4, 8, 16)

    def test_archive_num_blocks_and_experts(self):
        from squish.block_expert_archive import BlockExpertArchive, BlockExpertConfig

        rng    = np.random.default_rng(1)
        dim    = 8
        n_blk  = 3
        n_exp  = 2
        cfg    = BlockExpertConfig(n_clusters=n_exp, n_iter=3)
        blk_w  = {i: [rng.standard_normal(dim).astype(np.float32) for _ in range(n_exp)]
                  for i in range(n_blk)}
        base_w = {i: rng.standard_normal(dim).astype(np.float32) for i in range(n_blk)}
        with tempfile.TemporaryDirectory() as tmp:
            archive = BlockExpertArchive.create(tmp, blk_w, base_w, cfg)
            assert archive.num_blocks() == n_blk
            assert archive.num_experts(0) == n_exp

    def test_expert_router_route(self):
        from squish.block_expert_archive import BlockExpertArchive, BlockExpertConfig

        rng    = np.random.default_rng(2)
        dim    = 8
        cfg    = BlockExpertConfig(n_clusters=2, n_iter=3)
        blk_w  = {0: [rng.standard_normal(dim).astype(np.float32),
                      rng.standard_normal(dim).astype(np.float32)]}
        base_w = {0: rng.standard_normal(dim).astype(np.float32)}
        with tempfile.TemporaryDirectory() as tmp:
            archive = BlockExpertArchive.create(tmp, blk_w, base_w, cfg)
            cur_w   = rng.standard_normal(dim).astype(np.float32)
            cluster_idx, stats = archive.router.route(0, cur_w)
            assert 0 <= cluster_idx < 2


# ---------------------------------------------------------------------------
# DISCRouter
# ---------------------------------------------------------------------------


class TestDISCRouterWiring:
    def test_import(self):
        from squish.disc_router import DISCRouter, DISCRouterConfig, TaskType

        def llm_fn(system, user, context=""):
            return f"type: {TaskType.QA.value}"

        cfg    = DISCRouterConfig(max_subtasks=4, parallel_execution=False)
        router = DISCRouter(llm_fn, cfg)
        assert router is not None

    def test_task_type_values(self):
        from squish.disc_router import TaskType

        assert TaskType.SUMMARIZE is not None
        assert TaskType.QA is not None
        assert TaskType.CODE is not None

    def test_disc_plan_add_and_len(self):
        from squish.disc_router import DISCPlan, SubTask, TaskType

        plan = DISCPlan()
        task = SubTask(task_id="t1", task_type=TaskType.SUMMARIZE,
                       prompt="summarize this")
        plan.add(task)
        assert len(plan) == 1

    def test_disc_plan_topological_order(self):
        from squish.disc_router import DISCPlan, SubTask, TaskType

        plan = DISCPlan()
        t1   = SubTask(task_id="t1", task_type=TaskType.RETRIEVE,
                       prompt="get docs", depends_on=[])
        t2   = SubTask(task_id="t2", task_type=TaskType.SUMMARIZE,
                       prompt="summarize", depends_on=["t1"])
        plan.add(t1)
        plan.add(t2)
        order = plan.topological_order()
        ids   = [t.task_id for t in order]
        assert ids.index("t1") < ids.index("t2")


# ---------------------------------------------------------------------------
# SelfLearning
# ---------------------------------------------------------------------------


class TestSelfLearningWiring:
    def test_import(self):
        from squish.self_learning import LearnConfig, LearnExample, SelfLearner

        rng      = np.random.default_rng(0)
        base_w   = {0: rng.standard_normal((4, 4)).astype(np.float32)}
        cfg      = LearnConfig(steps=2, lr=1e-4, batch_size=2, max_rank=2)
        learner  = SelfLearner(base_w, cfg)
        assert learner is not None

    def test_config_defaults(self):
        from squish.self_learning import LearnConfig

        cfg = LearnConfig()
        assert cfg.steps >= 1
        assert cfg.lr > 0.0
        assert cfg.batch_size >= 1
        assert cfg.max_rank >= 0

    def test_learn_from_examples(self):
        from squish.self_learning import LearnConfig, LearnExample, LearnResult, SelfLearner

        rng     = np.random.default_rng(0)
        base_w  = {0: rng.standard_normal((4, 4)).astype(np.float32)}
        cfg     = LearnConfig(steps=2, batch_size=2, max_rank=2, seed=42)
        learner = SelfLearner(base_w, cfg)
        examples = [
            LearnExample(input=[1, 2, 3], output=[4, 5, 6]),
            LearnExample(input=[7, 8], output=[9]),
            LearnExample(input=[10, 11, 12], output=[13, 14]),
        ]
        result = learner.learn_from_examples(examples)
        assert isinstance(result, LearnResult)
        assert result.steps_run >= 1
        assert result.elapsed_s >= 0.0

    def test_compute_delta_snr(self):
        from squish.self_learning import compute_delta_snr

        rng   = np.random.default_rng(1)
        base  = rng.standard_normal((8, 8)).astype(np.float32)
        delta = base * 0.01  # small delta → high SNR
        snr   = compute_delta_snr(base, delta)
        assert isinstance(snr, float)
        assert snr > 0.0


# ---------------------------------------------------------------------------
# SemanticCache
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _HAS_SQLITE_VEC, reason="sqlite_vec not installed")
class TestSemanticCacheWiring:
    def test_import(self):
        from squish.semantic_cache import SquishSemanticCache

        cache = SquishSemanticCache(db_path=":memory:")
        assert cache is not None

    def test_store_and_lookup_hit(self):
        from squish.semantic_cache import SquishSemanticCache

        cache = SquishSemanticCache(db_path=":memory:")
        cache.store("hello world", "some response", task_type="default")
        hit = cache.lookup("hello world", task_type="default")
        assert hit is not None

    def test_lookup_miss(self):
        from squish.semantic_cache import SquishSemanticCache

        cache = SquishSemanticCache(db_path=":memory:")
        result = cache.lookup("completely unseen query xyz", task_type="default")
        assert result is None

    def test_stats_and_clear(self):
        from squish.semantic_cache import SquishSemanticCache

        cache = SquishSemanticCache(db_path=":memory:")
        cache.store("query one", "answer one")
        s = cache.stats()
        assert isinstance(s, dict)
        cache.clear()
        empty_stats = cache.stats()
        assert isinstance(empty_stats, dict)


# ---------------------------------------------------------------------------
# IPW (Intelligence-Per-Watt)
# ---------------------------------------------------------------------------


class TestIPWWiring:
    def test_import(self):
        from squish.ipw import IPWConfig, IPWMeasurement, IPWTracker

        cfg     = IPWConfig(energy_unit="mJ", quality_weight=1.0)
        tracker = IPWTracker(cfg)
        m       = IPWMeasurement(quality_score=0.8, energy_mj=10.0,
                                  time_ms=500.0, tokens_generated=64,
                                  task_type="chat")
        assert tracker is not None
        assert m is not None

    def test_config_defaults(self):
        from squish.ipw import IPWConfig

        cfg = IPWConfig()
        assert cfg.energy_unit in ("mJ", "J", "Wh")
        assert cfg.quality_weight > 0.0
        assert cfg.min_energy_mj > 0.0

    def test_measurement_derived_properties(self):
        from squish.ipw import IPWMeasurement

        m = IPWMeasurement(quality_score=0.9, energy_mj=100.0,
                            time_ms=1000.0, tokens_generated=50)
        assert m.ipw > 0.0
        assert m.tokens_per_joule > 0.0
        assert m.tokens_per_second > 0.0

    def test_tracker_record_and_summary(self):
        from squish.ipw import IPWConfig, IPWTracker

        cfg     = IPWConfig()
        tracker = IPWTracker(cfg)
        for i in range(5):
            tracker.record_values(
                quality_score=0.7 + 0.05 * i,
                energy_mj=100.0 + i * 10,
                time_ms=500.0,
                tokens_generated=64,
                task_type="chat",
            )
        assert tracker.total_measurements == 5
        s = tracker.summary()
        assert s.count == 5
        assert s.mean_ipw > 0.0
        tracker.reset()
        assert tracker.total_measurements == 0


# ---------------------------------------------------------------------------
# PowerMonitor
# ---------------------------------------------------------------------------


class TestPowerMonitorWiring:
    def test_import(self):
        from squish.power_monitor import POWER_CONFIGS, PowerModeConfig, PowerMonitor

        assert "performance" in POWER_CONFIGS
        assert "balanced" in POWER_CONFIGS
        assert "battery" in POWER_CONFIGS
        mon = PowerMonitor(poll_interval_s=60.0)
        assert mon is not None

    def test_power_mode_config_fields(self):
        from squish.power_monitor import POWER_CONFIGS

        perf = POWER_CONFIGS["performance"]
        batt = POWER_CONFIGS["battery"]
        assert perf.eagle_k_draft >= 1
        assert perf.batch_window_ms > 0.0
        assert batt.eagle_k_draft <= perf.eagle_k_draft  # battery ≤ performance

    def test_get_power_source_returns_valid(self):
        from squish.power_monitor import PowerMonitor

        mon    = PowerMonitor(poll_interval_s=60.0)
        source = mon.get_power_source()
        assert source in ("battery", "ac")

    def test_get_recommended_mode(self):
        from squish.power_monitor import PowerMonitor

        mon  = PowerMonitor(poll_interval_s=60.0)
        mode = mon.get_recommended_mode()
        assert mode in ("performance", "balanced", "battery")

    def test_apply_mode_function(self):
        from squish.power_monitor import apply_mode

        globs = {}
        apply_mode("balanced", globs)
        # apply_mode should not raise, even with an empty globals dict


# ---------------------------------------------------------------------------
# DiffusionDraft
# ---------------------------------------------------------------------------


class TestDiffusionDraftWiring:
    def test_import(self):
        from squish.diffusion_draft import DiffusionDraftModel

        model = DiffusionDraftModel(model_path="/mock/path",
                                    confidence_threshold=0.7,
                                    max_suitable_tokens=64)
        assert model is not None

    def test_default_constants(self):
        from squish.diffusion_draft import DiffusionDraftModel

        assert DiffusionDraftModel.DEFAULT_CONFIDENCE_THRESHOLD == 0.7
        assert DiffusionDraftModel.DEFAULT_MAX_SUITABLE_TOKENS > 0

    def test_is_available_returns_bool(self):
        from squish.diffusion_draft import DiffusionDraftModel

        model = DiffusionDraftModel(model_path="/nonexistent/path")
        # Without transformers installed, should return False
        result = model.is_available()
        assert isinstance(result, bool)

    def test_is_suitable_for_task(self):
        from squish.diffusion_draft import DiffusionDraftModel

        model = DiffusionDraftModel(model_path="/mock/path", max_suitable_tokens=32)
        assert model.is_suitable_for_task(n_tokens=16) is True
        assert model.is_suitable_for_task(n_tokens=64) is False

    def test_accessors(self):
        from squish.diffusion_draft import DiffusionDraftModel

        model = DiffusionDraftModel(model_path="/mock/path",
                                    confidence_threshold=0.8,
                                    max_suitable_tokens=48)
        assert model.model_path() == "/mock/path"
        assert model.confidence_threshold() == 0.8
        assert model.max_suitable_tokens() == 48
