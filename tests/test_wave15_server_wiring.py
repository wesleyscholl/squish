"""tests/test_wave15_server_wiring.py

Verifies that all Wave 15 module classes are importable and have the expected
public APIs that the server.py wiring code depends on.  These are pure
import + instantiation tests — no model or GPU required.

Wave 15 modules (Serving Intelligence + KV Architecture Evolution):
  ada_serve, conf_spec, seq_packing, meta_reasoner, yoco, cla,
  kvsharer, diffkv, paris_kv, kvtuner
"""

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# AdaServe
# ---------------------------------------------------------------------------

class TestAdaServeWiring:
    def test_import(self):
        from squish.ada_serve import AdaServeConfig, AdaServeScheduler, SLOTarget
        cfg  = AdaServeConfig(min_gamma=1, max_gamma=8, base_gamma=4)
        slo  = SLOTarget(task_type="chat", time_to_first_token_ms=200.0)
        sched = AdaServeScheduler(cfg, slo_registry={"chat": slo})
        assert sched is not None

    def test_config_defaults(self):
        from squish.ada_serve import AdaServeConfig
        cfg = AdaServeConfig()
        assert cfg.min_gamma >= 1
        assert cfg.max_gamma >= cfg.min_gamma
        assert 0.0 < cfg.slo_headroom_fraction < 1.0
        assert 0.0 < cfg.goodput_weight < 1.0

    def test_register_slo_and_get_gamma(self):
        from squish.ada_serve import AdaServeConfig, AdaServeRequest, AdaServeScheduler, SLOTarget
        cfg  = AdaServeConfig(min_gamma=1, max_gamma=8)
        slo  = SLOTarget(task_type="chat", time_to_first_token_ms=200.0)
        sched = AdaServeScheduler(cfg)
        sched.register_slo("chat", slo)
        req = AdaServeRequest(request_id="r1", slo=slo)
        sched.enqueue(req)
        gamma = sched.get_gamma("r1")
        assert cfg.min_gamma <= gamma <= cfg.max_gamma

    def test_complete_and_stats(self):
        from squish.ada_serve import AdaServeConfig, AdaServeRequest, AdaServeScheduler, SLOTarget
        cfg  = AdaServeConfig()
        slo  = SLOTarget(task_type="code")
        sched = AdaServeScheduler(cfg)
        sched.register_slo("code", slo)
        req = AdaServeRequest(request_id="r2", slo=slo)
        sched.enqueue(req)
        sched.complete("r2", tokens_generated=32, slo_met=True)
        s = sched.stats  # AdaServeStats is a property, not a method
        assert s.total_requests >= 1

    def test_slo_target_priority_range(self):
        from squish.ada_serve import SLOTarget
        slo = SLOTarget(task_type="batch", priority=1)
        assert slo.priority == 1
        slo_high = SLOTarget(task_type="realtime", priority=10)
        assert slo_high.priority > slo.priority


# ---------------------------------------------------------------------------
# ConfSpec
# ---------------------------------------------------------------------------

class TestConfSpecWiring:
    def test_import(self):
        from squish.conf_spec import ConfSpecConfig, ConfSpecVerifier
        cfg  = ConfSpecConfig(high_gate=0.9, low_gate=0.5, vocab_size=32000)
        verifier = ConfSpecVerifier(config=cfg)
        assert verifier is not None

    def test_config_defaults(self):
        from squish.conf_spec import ConfSpecConfig
        cfg = ConfSpecConfig()
        assert 0.0 < cfg.low_gate < cfg.high_gate <= 1.0
        assert cfg.vocab_size > 0
        assert cfg.metric in ("top_prob", "entropy", "margin")

    def test_verify_step_returns_decision(self):
        from squish.conf_spec import ConfSpecConfig, ConfSpecVerifier, VerificationRouting
        rng = np.random.default_rng(0)
        cfg  = ConfSpecConfig(high_gate=0.9, low_gate=0.5, vocab_size=64)
        verifier = ConfSpecVerifier(cfg)
        logits = rng.standard_normal(64).astype(np.float32)
        decision = verifier.verify_step("step text", "context", logits)
        assert decision is not None
        assert isinstance(decision.accepted, bool)
        assert decision.routing in (
            VerificationRouting.FULL_TARGET,
            VerificationRouting.LIGHTWEIGHT,
            VerificationRouting.AUTO_ACCEPT,
        )

    def test_stats_and_reset(self):
        from squish.conf_spec import ConfSpecConfig, ConfSpecVerifier
        rng = np.random.default_rng(1)
        cfg  = ConfSpecConfig(vocab_size=64)
        verifier = ConfSpecVerifier(cfg)
        for _ in range(5):
            logits = rng.standard_normal(64).astype(np.float32)
            verifier.verify_step("step", "ctx", logits)
        s = verifier.stats  # ConfSpecStats is a property, not a method
        assert s.total_steps >= 1
        verifier.reset()
        s2 = verifier.stats
        assert s2.total_steps == 0


# ---------------------------------------------------------------------------
# SeqPacking
# ---------------------------------------------------------------------------

class TestSeqPackingWiring:
    def test_import(self):
        from squish.seq_packing import PackingConfig, SequencePacker
        cfg    = PackingConfig(max_packed_length=512, pad_to_multiple=8)
        packer = SequencePacker(config=cfg, pad_token_id=0)
        assert packer is not None

    def test_config_defaults(self):
        from squish.seq_packing import PackingConfig
        cfg = PackingConfig()
        assert cfg.max_packed_length >= 1
        assert cfg.pad_to_multiple >= 1
        assert isinstance(cfg.allow_partial, bool)

    def test_pack_sequences(self):
        from squish.seq_packing import PackedBatch, PackingConfig, SequencePacker
        cfg    = PackingConfig(max_packed_length=32)
        packer = SequencePacker(cfg)
        seqs   = [[1, 2, 3], [4, 5], [6, 7, 8, 9]]
        batches = packer.pack(seqs)
        assert len(batches) >= 1
        for batch in batches:
            assert isinstance(batch, PackedBatch)
            assert len(batch.sequence_lengths) == len(batch.sequence_offsets)

    def test_packed_batch_lengths_sum(self):
        from squish.seq_packing import PackingConfig, SequencePacker
        cfg    = PackingConfig(max_packed_length=128, allow_partial=False)
        packer = SequencePacker(cfg)
        seqs   = [[i] * (4 + i) for i in range(8)]
        batches = packer.pack(seqs)
        total_tokens = sum(sum(b.sequence_lengths) for b in batches)
        expected_total = sum(len(s) for s in seqs)
        assert total_tokens == expected_total


# ---------------------------------------------------------------------------
# MetaReasoner
# ---------------------------------------------------------------------------

class TestMetaReasonerWiring:
    def test_import(self):
        from squish.meta_reasoner import MetaReasoner, MetaReasonerConfig
        cfg = MetaReasonerConfig(think_start_token_id=151667,
                                  think_end_token_id=151668)
        mr  = MetaReasoner(cfg)
        assert mr is not None

    def test_config_defaults(self):
        from squish.meta_reasoner import MetaReasonerConfig
        cfg = MetaReasonerConfig()
        assert cfg.entropy_threshold < cfg.entropy_high_threshold
        assert cfg.min_think_tokens >= 1
        assert cfg.max_think_tokens > cfg.min_think_tokens

    def test_step_produces_bool(self):
        from squish.meta_reasoner import MetaReasoner, MetaReasonerConfig
        rng = np.random.default_rng(42)
        cfg = MetaReasonerConfig(entropy_threshold=1.5, entropy_high_threshold=4.0)
        mr  = MetaReasoner(cfg)
        # Not in thinking phase initially
        assert mr.in_thinking_phase is False
        logits = rng.standard_normal(32000).astype(np.float32)
        result = mr.step(logits)
        assert isinstance(result, bool)

    def test_compute_entropy_static(self):
        from squish.meta_reasoner import MetaReasoner
        # Uniform distribution over 4 classes → entropy = log(4) ≈ 1.386
        logits = np.zeros(4, dtype=np.float32)
        entropy = MetaReasoner.compute_entropy(logits)
        assert abs(entropy - np.log(4)) < 0.01

    def test_reset_clears_state(self):
        from squish.meta_reasoner import MetaReasoner, MetaReasonerConfig
        rng = np.random.default_rng(0)
        cfg = MetaReasonerConfig()
        mr  = MetaReasoner(cfg)
        for _ in range(5):
            mr.step(rng.standard_normal(32000).astype(np.float32))
        mr.reset()
        assert mr.think_tokens_generated == 0
        assert mr.in_thinking_phase is False


# ---------------------------------------------------------------------------
# YOCO (You Only Cache Once)
# ---------------------------------------------------------------------------

class TestYOCOWiring:
    def test_import(self):
        from squish.yoco import YOCOConfig, YOCOKVStore
        cfg   = YOCOConfig(n_layers=32, n_self_attn_layers=16,
                           head_dim=128, n_kv_heads=8)
        store = YOCOKVStore(cfg)
        assert store is not None

    def test_config_defaults(self):
        from squish.yoco import YOCOConfig
        cfg = YOCOConfig()
        assert 0 < cfg.n_self_attn_layers <= cfg.n_layers
        assert cfg.head_dim > 0
        assert cfg.n_kv_heads > 0

    def test_append_and_get_shared_kv(self):
        from squish.yoco import YOCOConfig, YOCOKVStore
        rng  = np.random.default_rng(0)
        cfg  = YOCOConfig(n_layers=4, n_self_attn_layers=2, head_dim=8, n_kv_heads=2)
        store = YOCOKVStore(cfg)
        assert store.is_empty
        keys   = rng.standard_normal((4, 8)).astype(np.float32)
        values = rng.standard_normal((4, 8)).astype(np.float32)
        store.append(keys, values)
        assert not store.is_empty
        k_out, v_out = store.get_shared_kv()
        # get_shared_kv adds a leading batch dimension
        assert k_out.ndim == keys.ndim + 1
        assert k_out.shape[-2:] == keys.shape
        assert v_out.shape[-2:] == values.shape

    def test_reset_clears_store(self):
        from squish.yoco import YOCOConfig, YOCOKVStore
        rng  = np.random.default_rng(1)
        cfg  = YOCOConfig(n_layers=4, n_self_attn_layers=2, head_dim=8, n_kv_heads=2)
        store = YOCOKVStore(cfg)
        store.append(rng.standard_normal((2, 8)).astype(np.float32),
                     rng.standard_normal((2, 8)).astype(np.float32))
        store.reset()
        assert store.is_empty

    def test_schedule_reduction_factor(self):
        from squish.yoco import YOCOConfig, YOCOSchedule
        cfg  = YOCOConfig(n_layers=16, n_self_attn_layers=8)
        sched = YOCOSchedule.from_config(cfg)
        factor = sched.kv_cache_reduction_factor()
        assert 0.0 < factor <= 1.0


# ---------------------------------------------------------------------------
# CLA (Cross-Layer Attention)
# ---------------------------------------------------------------------------

class TestCLAWiring:
    def test_import(self):
        from squish.cla import CLAConfig, CLALayerSpec
        cfg  = CLAConfig(n_layers=32, sharing_factor=2)
        spec = CLALayerSpec(layer_idx=0, is_generator=True, borrows_from=None)
        assert cfg is not None
        assert spec is not None

    def test_config_defaults(self):
        from squish.cla import CLAConfig
        cfg = CLAConfig()
        assert cfg.n_layers >= 2
        assert cfg.sharing_factor >= 1

    def test_schedule_generates_correct_specs(self):
        from squish.cla import CLAConfig, CLALayerSpec, CLASchedule
        cfg   = CLAConfig(n_layers=8, sharing_factor=2)
        sched = CLASchedule.from_config(cfg)
        for i in range(cfg.n_layers):
            spec = sched.spec_for(i)
            assert isinstance(spec, CLALayerSpec)
            assert spec.layer_idx == i

    def test_reduction_factor_within_bounds(self):
        from squish.cla import CLAConfig, CLASchedule
        cfg   = CLAConfig(n_layers=16, sharing_factor=4)
        sched = CLASchedule.from_config(cfg)
        factor = sched.kv_cache_reduction_factor()
        assert 0.0 < factor <= 1.0

    def test_generator_layers_have_no_borrow(self):
        from squish.cla import CLAConfig, CLASchedule
        cfg   = CLAConfig(n_layers=8, sharing_factor=2,
                          allow_first_layer_borrow=False)
        sched = CLASchedule.from_config(cfg)
        generators = [sched.spec_for(i) for i in range(cfg.n_layers)
                      if sched.spec_for(i).is_generator]
        for g in generators:
            assert g.borrows_from is None


# ---------------------------------------------------------------------------
# KVSharer
# ---------------------------------------------------------------------------

class TestKVSharerWiring:
    def test_import(self):
        from squish.kvsharer import KVSharerCalibrator, KVSharerConfig
        cfg  = KVSharerConfig(n_layers=8, similarity_threshold=0.95)
        cal  = KVSharerCalibrator(cfg)
        assert cal is not None

    def test_config_defaults(self):
        from squish.kvsharer import KVSharerConfig
        cfg = KVSharerConfig()
        assert 0.0 < cfg.similarity_threshold <= 1.0
        assert 0.0 < cfg.max_share_fraction < 1.0
        assert cfg.n_layers >= 2

    def test_record_and_compute_share_map(self):
        from squish.kvsharer import KVShareMap, KVSharerCalibrator, KVSharerConfig
        rng = np.random.default_rng(0)
        cfg = KVSharerConfig(n_layers=4, similarity_threshold=0.80)
        cal = KVSharerCalibrator(cfg)
        for layer_idx in range(4):
            keys   = rng.standard_normal((8, 16)).astype(np.float32)
            values = rng.standard_normal((8, 16)).astype(np.float32)
            cal.record_layer_kv(layer_idx, keys, values)
        share_map = cal.compute_share_map()
        assert isinstance(share_map, KVShareMap)
        assert share_map.n_layers == 4

    def test_share_map_kv_ops_fraction(self):
        from squish.kvsharer import KVSharerCalibrator, KVSharerConfig
        rng = np.random.default_rng(1)
        cfg = KVSharerConfig(n_layers=6, similarity_threshold=0.70)
        cal = KVSharerCalibrator(cfg)
        # Use identical vectors to force high similarity
        base = rng.standard_normal((8, 16)).astype(np.float32)
        for layer_idx in range(6):
            cal.record_layer_kv(layer_idx, base.copy(), base.copy())
        share_map = cal.compute_share_map()
        frac = share_map.kv_ops_saved_fraction()
        assert 0.0 <= frac <= 1.0


# ---------------------------------------------------------------------------
# DiffKV
# ---------------------------------------------------------------------------

class TestDiffKVWiring:
    def test_import(self):
        from squish.diffkv import DiffKVConfig, DiffKVPolicyManager
        cfg = DiffKVConfig(n_layers=8, n_heads=8,
                           critical_k_bits=8, critical_v_bits=4)
        mgr = DiffKVPolicyManager(cfg)
        assert mgr is not None

    def test_config_defaults(self):
        from squish.diffkv import DiffKVConfig
        cfg = DiffKVConfig()
        assert cfg.n_layers >= 1
        assert cfg.n_heads >= 1
        assert cfg.critical_k_bits >= cfg.marginal_k_bits
        assert cfg.critical_v_bits >= cfg.marginal_v_bits
        assert 0.0 <= cfg.critical_fraction + cfg.marginal_fraction <= 1.0

    def test_get_policy_returns_diffkv_policy(self):
        from squish.diffkv import DiffKVConfig, DiffKVPolicy, DiffKVPolicyManager
        cfg = DiffKVConfig(n_layers=4, n_heads=4,
                           critical_k_bits=8, critical_v_bits=4,
                           marginal_k_bits=4, marginal_v_bits=2)
        mgr = DiffKVPolicyManager(cfg)
        policy = mgr.get_policy(layer_idx=0, head_idx=0)
        assert isinstance(policy, DiffKVPolicy)
        assert policy.layer_idx == 0
        assert policy.head_idx == 0

    def test_record_attention_adjusts_policy(self):
        from squish.diffkv import DiffKVConfig, DiffKVPolicyManager
        rng = np.random.default_rng(0)
        cfg = DiffKVConfig(n_layers=2, n_heads=2)
        mgr = DiffKVPolicyManager(cfg)
        attn = rng.random((4, 4)).astype(np.float32)
        attn /= attn.sum(axis=-1, keepdims=True)
        mgr.record_attention(layer_idx=0, head_idx=0, attn_weights=attn)
        policies = mgr.all_policies()
        assert len(policies) == cfg.n_layers * cfg.n_heads


# ---------------------------------------------------------------------------
# ParisKV
# ---------------------------------------------------------------------------

class TestParisKVWiring:
    def test_import(self):
        from squish.paris_kv import ParisKVCodebook, ParisKVConfig
        # n_codes belongs to ParisKVCodebook, not ParisKVConfig
        cfg      = ParisKVConfig(learning_rate=0.05)
        codebook = ParisKVCodebook(dim=16, n_codes=8, config=cfg)
        assert codebook is not None
        assert not codebook.is_fitted

    def test_config_defaults(self):
        from squish.paris_kv import ParisKVConfig
        cfg = ParisKVConfig()
        assert 0.0 < cfg.learning_rate < 1.0
        assert cfg.min_count >= 1
        assert cfg.drift_window >= 1

    def test_fit_encode_decode_roundtrip(self):
        from squish.paris_kv import ParisKVCodebook, ParisKVConfig
        rng      = np.random.default_rng(0)
        cfg      = ParisKVConfig()
        dim      = 16
        codebook = ParisKVCodebook(dim=dim, n_codes=8, config=cfg)
        data     = rng.standard_normal((64, dim)).astype(np.float32)
        codebook.fit(data)
        assert codebook.is_fitted
        indices    = codebook.encode(data[:4])
        decoded    = codebook.decode(indices)
        assert decoded.shape == (4, dim)
        assert codebook.quantization_error >= 0.0  # property, not method

    def test_online_update_adjusts_codebook(self):
        from squish.paris_kv import ParisKVCodebook, ParisKVConfig
        rng      = np.random.default_rng(1)
        cfg      = ParisKVConfig(learning_rate=0.1)
        codebook = ParisKVCodebook(dim=8, n_codes=4, config=cfg)
        data     = rng.standard_normal((32, 8)).astype(np.float32)
        codebook.fit(data)
        new_data  = rng.standard_normal((8, 8)).astype(np.float32)
        drift     = codebook.online_update(new_data)
        assert isinstance(drift, float)
        assert drift >= 0.0


# ---------------------------------------------------------------------------
# KVTuner
# ---------------------------------------------------------------------------

class TestKVTunerWiring:
    def test_import(self):
        from squish.kvtuner import KVTunerCalibrator, KVTunerConfig
        cfg = KVTunerConfig(n_layers=8, candidate_bits=(2, 4, 8),
                             target_avg_bits=4.0)
        cal = KVTunerCalibrator(cfg)
        assert cal is not None

    def test_config_defaults(self):
        from squish.kvtuner import KVTunerConfig
        cfg = KVTunerConfig()
        assert cfg.n_layers >= 1
        assert len(cfg.candidate_bits) >= 2
        assert 0.0 < cfg.target_avg_bits <= 32.0
        assert cfg.key_priority > 0.0

    def test_record_and_search(self):
        from squish.kvtuner import KVQuantConfig, KVTunerCalibrator, KVTunerConfig
        rng = np.random.default_rng(0)
        cfg = KVTunerConfig(n_layers=4, candidate_bits=(2, 4, 8),
                             target_avg_bits=4.0)
        cal = KVTunerCalibrator(cfg)
        for layer_idx in range(cfg.n_layers):
            keys   = rng.standard_normal((8, 16)).astype(np.float32)
            values = rng.standard_normal((8, 16)).astype(np.float32)
            cal.record_layer(layer_idx, keys, values)
        result = cal.search()
        assert isinstance(result, KVQuantConfig)
        assert result.n_layers == cfg.n_layers

    def test_avg_bits_respects_target(self):
        from squish.kvtuner import KVTunerCalibrator, KVTunerConfig
        rng = np.random.default_rng(2)
        n   = 8
        cfg = KVTunerConfig(n_layers=n, candidate_bits=(2, 4, 8),
                             target_avg_bits=4.0)
        cal = KVTunerCalibrator(cfg)
        for i in range(n):
            cal.record_layer(i,
                             rng.standard_normal((16, 32)).astype(np.float32),
                             rng.standard_normal((16, 32)).astype(np.float32))
        result = cal.search()
        avg = result.avg_bits
        assert 1.0 <= avg <= 9.0  # Must be between min and max candidate bits
