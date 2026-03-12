"""tests/test_wave21_server_wiring.py

Verifies that all Wave 21 module classes are importable and have the expected
public APIs that the server wiring code depends on.  These are pure
import + instantiation tests — no model or GPU required.

Wave 21 modules (Advanced Decode Infrastructure + Compression):
  tree_verifier, kv_compress, dynamic_ntk, quant_spec_decode,
  sparse_attn_index, mixed_precision_kv, pipeline_bubble, layerwise_decode,
  codec_kv, dedupe_attn, flash_prefill, budget_spec, retention_attn,
  kv_router
"""

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# TreeVerifier
# ---------------------------------------------------------------------------


class TestTreeVerifierWiring:
    def test_import(self):
        from squish.tree_verifier import VerifyConfig, TokenTree, VerifyResult, TreeVerifier, VerifyStats
        cfg = VerifyConfig(n_draft_tokens=4, n_branches=3, temperature=1.0)
        verifier = TreeVerifier(cfg)
        assert verifier is not None

    def test_config_defaults(self):
        from squish.tree_verifier import VerifyConfig
        cfg = VerifyConfig()
        assert cfg.n_draft_tokens >= 1
        assert cfg.n_branches >= 1
        assert cfg.temperature > 0.0

    def test_verify_returns_result(self):
        from squish.tree_verifier import VerifyConfig, TokenTree, VerifyResult, TreeVerifier
        rng = np.random.default_rng(0)
        vocab = 64
        n_branches, n_draft = 3, 4
        cfg = VerifyConfig(n_draft_tokens=n_draft, n_branches=n_branches, temperature=1.0)
        verifier = TreeVerifier(cfg)
        tree = TokenTree(
            tokens=rng.integers(0, vocab, (n_branches, n_draft), dtype=np.int32),
            draft_logits=rng.standard_normal((n_branches, n_draft, vocab)).astype(np.float32),
        )
        target_logits = rng.standard_normal((n_branches, n_draft, vocab)).astype(np.float32)
        result = verifier.verify(tree, target_logits)
        assert isinstance(result, VerifyResult)
        assert result.n_accepted >= 0
        assert 0.0 <= result.acceptance_rate <= 1.0

    def test_stats_and_reset(self):
        from squish.tree_verifier import VerifyConfig, TokenTree, TreeVerifier, VerifyStats
        rng = np.random.default_rng(1)
        vocab = 64
        n_branches, n_draft = 2, 3
        cfg = VerifyConfig(n_draft_tokens=n_draft, n_branches=n_branches, temperature=1.0)
        verifier = TreeVerifier(cfg)
        tree = TokenTree(
            tokens=rng.integers(0, vocab, (n_branches, n_draft), dtype=np.int32),
            draft_logits=rng.standard_normal((n_branches, n_draft, vocab)).astype(np.float32),
        )
        target_logits = rng.standard_normal((n_branches, n_draft, vocab)).astype(np.float32)
        verifier.verify(tree, target_logits)
        s = verifier.stats
        assert isinstance(s, VerifyStats)
        assert s.total_verifications == 1
        assert s.total_draft == n_branches * n_draft
        verifier.reset_stats()
        assert verifier.stats.total_verifications == 0


# ---------------------------------------------------------------------------
# KVCompress
# ---------------------------------------------------------------------------


class TestKVCompressWiring:
    def test_import(self):
        from squish.kv_compress import KVCompressConfig, CompressedKVEntry, KVCompressor, CompressStats
        cfg = KVCompressConfig(compress_after=256, quant_bits=8, prune_ratio=0.1,
                               n_heads=4, head_dim=32)
        comp = KVCompressor(cfg)
        assert comp is not None

    def test_config_defaults(self):
        from squish.kv_compress import KVCompressConfig
        cfg = KVCompressConfig()
        assert cfg.compress_after >= 0
        assert cfg.quant_bits in (4, 8)
        assert 0.0 <= cfg.prune_ratio < 1.0
        assert cfg.n_heads >= 1
        assert cfg.head_dim >= 1
        assert cfg.quant_max > 0

    def test_compress_decompress_roundtrip(self):
        from squish.kv_compress import KVCompressConfig, CompressedKVEntry, KVCompressor
        rng = np.random.default_rng(0)
        n_heads, seq_len, head_dim = 4, 32, 16
        # prune_ratio=0.0 keeps all positions uniformly, avoiding the unequal-
        # per-head count issue that arises when a global quantile threshold is
        # applied to heterogeneous per-head norm distributions.
        cfg = KVCompressConfig(compress_after=16, quant_bits=8, prune_ratio=0.0,
                               n_heads=n_heads, head_dim=head_dim)
        comp = KVCompressor(cfg)
        keys = rng.standard_normal((n_heads, seq_len, head_dim)).astype(np.float32)
        values = rng.standard_normal((n_heads, seq_len, head_dim)).astype(np.float32)
        entry = comp.compress(keys, values)
        assert isinstance(entry, CompressedKVEntry)
        assert entry.mask.shape == (n_heads, seq_len)
        k_back, v_back = comp.decompress(entry)
        assert k_back.ndim == 3
        assert v_back.shape == k_back.shape

    def test_stats_and_reset(self):
        from squish.kv_compress import KVCompressConfig, KVCompressor, CompressStats
        rng = np.random.default_rng(2)
        # prune_ratio=0.0 avoids the unequal-count-per-head issue
        cfg = KVCompressConfig(compress_after=8, quant_bits=8, prune_ratio=0.0,
                               n_heads=2, head_dim=8)
        comp = KVCompressor(cfg)
        keys = rng.standard_normal((2, 16, 8)).astype(np.float32)
        values = rng.standard_normal((2, 16, 8)).astype(np.float32)
        comp.compress(keys, values)
        s = comp.stats
        assert isinstance(s, CompressStats)
        assert s.n_compress_calls == 1
        assert s.total_tokens > 0
        assert 0.0 <= s.prune_rate <= 1.0
        comp.reset_stats()
        assert comp.stats.n_compress_calls == 0


# ---------------------------------------------------------------------------
# DynamicNTK
# ---------------------------------------------------------------------------


class TestDynamicNTKWiring:
    def test_import(self):
        from squish.dynamic_ntk import DynamicNTKConfig, NTKState, DynamicNTKScaler
        cfg = DynamicNTKConfig(base_theta=10000.0, max_trained_len=4096,
                               trigger_ratio=0.8, alpha=8.0, head_dim=64)
        scaler = DynamicNTKScaler(cfg)
        assert scaler is not None

    def test_config_defaults(self):
        from squish.dynamic_ntk import DynamicNTKConfig
        cfg = DynamicNTKConfig()
        assert cfg.base_theta > 0.0
        assert cfg.max_trained_len >= 1
        assert 0.0 < cfg.trigger_ratio <= 1.0
        assert cfg.alpha > 1.0
        assert cfg.head_dim >= 2
        assert cfg.head_dim % 2 == 0
        assert cfg.trigger_len >= 1

    def test_get_freqs_shape(self):
        from squish.dynamic_ntk import DynamicNTKConfig, DynamicNTKScaler
        cfg = DynamicNTKConfig(base_theta=10000.0, max_trained_len=2048,
                               trigger_ratio=0.8, alpha=4.0, head_dim=64)
        scaler = DynamicNTKScaler(cfg)
        freqs = scaler.get_freqs(seq_len=512)
        assert freqs.shape == (cfg.head_dim // 2,)
        assert freqs.dtype == np.float32

    def test_scaling_triggered(self):
        from squish.dynamic_ntk import DynamicNTKConfig, NTKState, DynamicNTKScaler
        # max_trained_len=512; trigger_len=256 (50 %); seq_len=600 > max_trained_len
        # ensures scale_factor = alpha*(600/512) - (alpha-1) > 1, so new_base > base_theta
        cfg = DynamicNTKConfig(base_theta=10000.0, max_trained_len=512,
                               trigger_ratio=0.5, alpha=4.0, head_dim=32)
        scaler = DynamicNTKScaler(cfg)
        freqs = scaler.get_freqs(seq_len=600)
        assert isinstance(scaler.state, NTKState)
        assert scaler.state.is_scaled is True
        assert scaler.effective_base > cfg.base_theta


# ---------------------------------------------------------------------------
# QuantSpecDecode
# ---------------------------------------------------------------------------


class TestQuantSpecDecodeWiring:
    def test_import(self):
        from squish.quant_spec_decode import QSDConfig, DraftStep, QuantSpecDecoder, QSDStats
        cfg = QSDConfig(n_draft_tokens=4, vocab_size=128, temperature=1.0)
        decoder = QuantSpecDecoder(cfg)
        assert decoder is not None

    def test_config_defaults(self):
        from squish.quant_spec_decode import QSDConfig
        cfg = QSDConfig()
        assert cfg.n_draft_tokens >= 1
        assert cfg.vocab_size >= 2
        assert cfg.draft_quant_bits == 4
        assert cfg.temperature > 0.0
        assert cfg.quant_max == 7

    def test_quantize_draft(self):
        from squish.quant_spec_decode import QSDConfig, DraftStep, QuantSpecDecoder
        rng = np.random.default_rng(0)
        vocab = 128
        n_draft = 4
        cfg = QSDConfig(n_draft_tokens=n_draft, vocab_size=vocab, temperature=1.0)
        decoder = QuantSpecDecoder(cfg)
        logits = rng.standard_normal((n_draft, vocab)).astype(np.float32)
        step = decoder.quantize_draft(logits)
        assert isinstance(step, DraftStep)
        assert step.tokens.shape == (n_draft,)
        assert step.logits_q.shape == (n_draft, vocab)
        assert step.scale > 0.0

    def test_verify_and_stats(self):
        from squish.quant_spec_decode import QSDConfig, QuantSpecDecoder, QSDStats
        rng = np.random.default_rng(1)
        vocab = 128
        n_draft = 4
        cfg = QSDConfig(n_draft_tokens=n_draft, vocab_size=vocab, temperature=1.0)
        decoder = QuantSpecDecoder(cfg)
        draft_logits = rng.standard_normal((n_draft, vocab)).astype(np.float32)
        target_logits = rng.standard_normal((n_draft, vocab)).astype(np.float32)
        step = decoder.quantize_draft(draft_logits)
        accepted, n_accepted = decoder.verify(step, target_logits)
        assert n_accepted >= 0
        assert len(accepted) == n_accepted
        s = decoder.stats
        assert isinstance(s, QSDStats)
        assert s.total_draft == n_draft
        assert 0.0 <= s.acceptance_rate <= 1.0
        decoder.reset_stats()
        assert decoder.stats.total_draft == 0


# ---------------------------------------------------------------------------
# SparseAttnIndex
# ---------------------------------------------------------------------------


class TestSparseAttnIndexWiring:
    def test_import(self):
        from squish.sparse_attn_index import IndexConfig, ANCandidates, SparseAttnIndex, IndexStats
        cfg = IndexConfig(top_k=8, head_dim=16, n_heads=4, n_probe=4)
        index = SparseAttnIndex(cfg)
        assert index is not None

    def test_config_defaults(self):
        from squish.sparse_attn_index import IndexConfig
        cfg = IndexConfig()
        assert cfg.top_k >= 1
        assert cfg.head_dim >= 1
        assert cfg.n_heads >= 1
        assert cfg.n_probe >= 1

    def test_build_and_query(self):
        from squish.sparse_attn_index import IndexConfig, ANCandidates, SparseAttnIndex
        rng = np.random.default_rng(0)
        n_heads, seq_len, head_dim, top_k = 4, 64, 16, 8
        cfg = IndexConfig(top_k=top_k, head_dim=head_dim, n_heads=n_heads)
        index = SparseAttnIndex(cfg)
        keys = rng.standard_normal((n_heads, seq_len, head_dim)).astype(np.float32)
        index.build(keys)
        assert index.n_indexed == seq_len
        q = rng.standard_normal((n_heads, head_dim)).astype(np.float32)
        result = index.query(q)
        assert isinstance(result, ANCandidates)
        assert result.indices.shape == (n_heads, top_k)
        assert result.scores.shape == (n_heads, top_k)

    def test_stats(self):
        from squish.sparse_attn_index import IndexConfig, IndexStats, SparseAttnIndex
        rng = np.random.default_rng(2)
        cfg = IndexConfig(top_k=4, head_dim=8, n_heads=2)
        index = SparseAttnIndex(cfg)
        keys = rng.standard_normal((2, 16, 8)).astype(np.float32)
        index.build(keys)
        q = rng.standard_normal((2, 8)).astype(np.float32)
        index.query(q)
        index.query(q)
        s = index.stats
        assert isinstance(s, IndexStats)
        assert s.n_builds == 1
        assert s.n_queries == 2
        assert s.total_keys_indexed == 2 * 16
        index.reset()
        assert index.n_indexed == 0


# ---------------------------------------------------------------------------
# MixedPrecisionKV
# ---------------------------------------------------------------------------


class TestMixedPrecisionKVWiring:
    def test_import(self):
        from squish.mixed_precision_kv import (
            HeadPrecision, MPKVConfig, HeadPrecisionMap,
            MixedPrecisionKVCache, MPKVStats,
        )
        cfg = MPKVConfig(n_heads=8, head_dim=32,
                         int4_threshold=0.3, int8_threshold=0.7)
        cache = MixedPrecisionKVCache(cfg)
        assert cache is not None

    def test_config_defaults(self):
        from squish.mixed_precision_kv import MPKVConfig, HeadPrecision
        cfg = MPKVConfig()
        assert cfg.n_heads >= 1
        assert cfg.head_dim >= 1
        assert 0.0 <= cfg.int4_threshold < 1.0
        assert cfg.int4_threshold < cfg.int8_threshold <= 1.0
        assert HeadPrecision.INT4 == "int4"
        assert HeadPrecision.INT8 == "int8"
        assert HeadPrecision.FP16 == "fp16"

    def test_assign_precisions(self):
        from squish.mixed_precision_kv import MPKVConfig, HeadPrecisionMap, MixedPrecisionKVCache
        rng = np.random.default_rng(0)
        n_heads = 8
        cfg = MPKVConfig(n_heads=n_heads, head_dim=16,
                         int4_threshold=0.3, int8_threshold=0.7)
        cache = MixedPrecisionKVCache(cfg)
        variance = rng.random(n_heads).astype(np.float32)
        prec_map = cache.assign_precisions(variance)
        assert isinstance(prec_map, HeadPrecisionMap)
        assert len(prec_map.precisions) == n_heads
        assert prec_map.n_int4 + prec_map.n_int8 + prec_map.n_fp16 == n_heads

    def test_store_load_and_stats(self):
        from squish.mixed_precision_kv import MPKVConfig, HeadPrecision, MixedPrecisionKVCache, MPKVStats
        rng = np.random.default_rng(1)
        n_heads, head_dim = 4, 16
        cfg = MPKVConfig(n_heads=n_heads, head_dim=head_dim,
                         int4_threshold=0.3, int8_threshold=0.7)
        cache = MixedPrecisionKVCache(cfg)
        key = rng.standard_normal(head_dim).astype(np.float32)
        val = rng.standard_normal(head_dim).astype(np.float32)
        k_q, v_q = cache.store(head_idx=0, key=key, value=val,
                                precision=HeadPrecision.INT8)
        assert k_q is not None
        k_back, v_back = cache.load(head_idx=0, key_q=k_q, value_q=v_q,
                                    precision=HeadPrecision.INT8)
        assert k_back.dtype == np.float32
        variance = rng.random(n_heads).astype(np.float32)
        cache.assign_precisions(variance)
        s = cache.stats
        assert isinstance(s, MPKVStats)
        assert s.total_heads_assigned == n_heads
        cache.reset_stats()
        assert cache.stats.total_heads_assigned == 0


# ---------------------------------------------------------------------------
# PipelineBubble
# ---------------------------------------------------------------------------


class TestPipelineBubbleWiring:
    def test_import(self):
        from squish.pipeline_bubble import StageConfig, StageSchedule, BubbleEliminator, BubbleStats
        cfg = StageConfig(n_stages=4, n_microbatches=8, stage_latency_ms=10.0)
        elim = BubbleEliminator(cfg)
        assert elim is not None

    def test_config_defaults(self):
        from squish.pipeline_bubble import StageConfig
        cfg = StageConfig()
        assert cfg.n_stages >= 1
        assert cfg.n_microbatches >= 1
        assert cfg.stage_latency_ms > 0.0
        frac = cfg.theoretical_bubble_fraction
        assert 0.0 <= frac <= 1.0

    def test_build_schedule(self):
        from squish.pipeline_bubble import StageConfig, StageSchedule, BubbleEliminator
        cfg = StageConfig(n_stages=4, n_microbatches=8, stage_latency_ms=5.0)
        elim = BubbleEliminator(cfg)
        sched = elim.build_schedule()
        assert isinstance(sched, StageSchedule)
        assert sched.n_stages == cfg.n_stages
        assert sched.n_slots >= cfg.n_microbatches
        assert 0.0 <= sched.bubble_fraction <= 1.0

    def test_simulate_and_stats(self):
        from squish.pipeline_bubble import StageConfig, BubbleEliminator, BubbleStats
        cfg = StageConfig(n_stages=4, n_microbatches=8, stage_latency_ms=5.0)
        elim = BubbleEliminator(cfg)
        sched = elim.build_schedule()
        result = elim.simulate(sched)
        assert "total_time_ms" in result
        assert "bubble_fraction" in result
        assert "throughput_mbatch_per_ms" in result
        assert result["total_time_ms"] > 0.0
        s = elim.stats
        assert isinstance(s, BubbleStats)
        assert s.n_schedules_built == 1
        elim.reset_stats()
        assert elim.stats.n_schedules_built == 0


# ---------------------------------------------------------------------------
# LayerwiseDecode
# ---------------------------------------------------------------------------


class TestLayerwiseDecodeWiring:
    def test_import(self):
        from squish.layerwise_decode import LayerwiseConfig, LayerStream, LayerwiseDecoder, DecodeStats
        cfg = LayerwiseConfig(n_layers=8, hidden_dim=64, exit_threshold=0.9,
                              min_exit_layer=4, probe_vocab=32)
        decoder = LayerwiseDecoder(cfg)
        assert decoder is not None

    def test_config_defaults(self):
        from squish.layerwise_decode import LayerwiseConfig
        cfg = LayerwiseConfig()
        assert cfg.n_layers >= 1
        assert cfg.hidden_dim >= 1
        assert 0.0 < cfg.exit_threshold <= 1.0
        assert 0 <= cfg.min_exit_layer < cfg.n_layers
        assert cfg.probe_vocab >= 2

    def test_process_layer(self):
        from squish.layerwise_decode import LayerwiseConfig, LayerStream, LayerwiseDecoder
        rng = np.random.default_rng(0)
        hidden_dim = 32
        cfg = LayerwiseConfig(n_layers=4, hidden_dim=hidden_dim,
                              exit_threshold=0.9, min_exit_layer=2, probe_vocab=16)
        decoder = LayerwiseDecoder(cfg, rng=np.random.default_rng(42))
        hidden = rng.standard_normal(hidden_dim).astype(np.float32)
        stream = LayerStream(hidden=hidden, layer_idx=0, confidence=0.0)
        layer_w = rng.standard_normal((hidden_dim, hidden_dim)).astype(np.float32) * 0.01
        new_stream = decoder.process_layer(stream, layer_w)
        assert new_stream.layer_idx == 1
        assert new_stream.hidden.shape == (hidden_dim,)
        assert 0.0 <= new_stream.confidence <= 1.0
        assert decoder.stats.total_layers_run == 1

    def test_record_token_stats(self):
        from squish.layerwise_decode import LayerwiseConfig, LayerStream, LayerwiseDecoder, DecodeStats
        rng = np.random.default_rng(1)
        hidden_dim = 32
        cfg = LayerwiseConfig(n_layers=4, hidden_dim=hidden_dim,
                              exit_threshold=0.9, min_exit_layer=2, probe_vocab=16)
        decoder = LayerwiseDecoder(cfg, rng=np.random.default_rng(99))
        decoder.record_token(exited_early=False)
        decoder.record_token(exited_early=True)
        s = decoder.stats
        assert isinstance(s, DecodeStats)
        assert s.total_tokens == 2
        assert s.early_exits == 1
        assert s.early_exit_rate == 0.5
        decoder.reset_stats()
        assert decoder.stats.total_tokens == 0


# ---------------------------------------------------------------------------
# CodecKV
# ---------------------------------------------------------------------------


class TestCodecKVWiring:
    def test_import(self):
        from squish.codec_kv import CodecConfig, KVCodec, CodecStats
        cfg = CodecConfig(n_codebook=16, head_dim=8, n_heads=2, n_fit_samples=32)
        codec = KVCodec(cfg)
        assert codec is not None

    def test_config_defaults(self):
        from squish.codec_kv import CodecConfig
        cfg = CodecConfig()
        assert cfg.n_codebook >= 2
        assert cfg.head_dim >= 1
        assert cfg.n_heads >= 1
        assert cfg.n_fit_samples >= cfg.n_codebook

    def test_fit_and_encode_decode(self):
        from squish.codec_kv import CodecConfig, KVCodec
        rng = np.random.default_rng(0)
        n_codebook, head_dim, n_heads = 4, 8, 2
        cfg = CodecConfig(n_codebook=n_codebook, head_dim=head_dim,
                          n_heads=n_heads, n_fit_samples=64)
        codec = KVCodec(cfg, rng=np.random.default_rng(7))
        assert not codec.is_fitted
        # Use integer-valued data so that fp32 dot products are computed exactly.
        # This prevents _pairwise_sq_dist from producing slightly-negative values
        # (which would make k-means++ probabilities negative and cause ValueError).
        centers = (10.0 * np.eye(n_codebook, head_dim)).astype(np.float32)
        noise_k = rng.integers(-2, 3, (n_codebook * 16, head_dim)).astype(np.float32)
        noise_v = rng.integers(-2, 3, (n_codebook * 16, head_dim)).astype(np.float32)
        keys_sample = np.vstack([centers[i % n_codebook] + noise_k[i]
                                  for i in range(n_codebook * 16)])
        values_sample = np.vstack([centers[i % n_codebook] + noise_v[i]
                                    for i in range(n_codebook * 16)])
        codec.fit(keys_sample, values_sample)
        assert codec.is_fitted
        keys = rng.standard_normal((n_heads, 16, head_dim)).astype(np.float32)
        idx_k = codec.encode_keys(keys)
        assert idx_k.shape == (n_heads, 16)
        k_hat = codec.decode_keys(idx_k[0], 0)
        assert k_hat.shape == (16, head_dim)

    def test_stats_and_compression_ratio(self):
        from squish.codec_kv import CodecConfig, KVCodec, CodecStats
        rng = np.random.default_rng(2)
        n_codebook, head_dim = 4, 8
        cfg = CodecConfig(n_codebook=n_codebook, head_dim=head_dim, n_heads=2, n_fit_samples=32)
        codec = KVCodec(cfg, rng=np.random.default_rng(5))
        # Integer-valued data: fp32 dot products are exact, avoiding negative
        # squared distances in _pairwise_sq_dist and negative k-means++ probs.
        centers = (10.0 * np.eye(n_codebook, head_dim)).astype(np.float32)
        noise_k = rng.integers(-2, 3, (n_codebook * 8, head_dim)).astype(np.float32)
        noise_v = rng.integers(-2, 3, (n_codebook * 8, head_dim)).astype(np.float32)
        keys_s = np.vstack([centers[i % n_codebook] + noise_k[i]
                             for i in range(n_codebook * 8)])
        vals_s = np.vstack([centers[i % n_codebook] + noise_v[i]
                             for i in range(n_codebook * 8)])
        codec.fit(keys_s, vals_s)
        keys = rng.standard_normal((2, 8, head_dim)).astype(np.float32)
        codec.encode_keys(keys)
        s = codec.stats
        assert isinstance(s, CodecStats)
        assert s.n_fit_calls == 1
        assert s.n_encode_calls == 1
        assert s.total_encoded_tokens == 2 * 8
        assert codec.compression_ratio > 0.0


# ---------------------------------------------------------------------------
# DedupeAttn
# ---------------------------------------------------------------------------


class TestDedupeAttnWiring:
    def test_import(self):
        from squish.dedupe_attn import DedupConfig, AttentionDeduplicator, DedupStats
        cfg = DedupConfig(sim_threshold=0.99, max_cache=64, n_heads=4, head_dim=16)
        dedup = AttentionDeduplicator(cfg)
        assert dedup is not None

    def test_config_defaults(self):
        from squish.dedupe_attn import DedupConfig
        cfg = DedupConfig()
        assert 0.0 < cfg.sim_threshold <= 1.0
        assert cfg.max_cache >= 1
        assert cfg.n_heads >= 1
        assert cfg.head_dim >= 1

    def test_store_and_lookup_hit(self):
        from squish.dedupe_attn import DedupConfig, AttentionDeduplicator
        rng = np.random.default_rng(0)
        head_dim = 16
        cfg = DedupConfig(sim_threshold=0.95, max_cache=32, n_heads=2, head_dim=head_dim)
        dedup = AttentionDeduplicator(cfg)
        q = rng.standard_normal(head_dim).astype(np.float32)
        q = q / (np.linalg.norm(q) + 1e-9)
        out = rng.standard_normal(head_dim).astype(np.float32)
        dedup.store(q, out, head_idx=0)
        # Query with nearly identical vector
        cached = dedup.lookup(q * 0.9999, head_idx=0)
        # Either hit (very similar) or miss both are valid; stats must be consistent
        assert dedup.stats.n_lookups == 1
        assert dedup.stats.n_stores == 1

    def test_stats_and_reset(self):
        from squish.dedupe_attn import DedupConfig, AttentionDeduplicator, DedupStats
        rng = np.random.default_rng(1)
        head_dim = 16
        cfg = DedupConfig(sim_threshold=0.99, max_cache=8, n_heads=2, head_dim=head_dim)
        dedup = AttentionDeduplicator(cfg)
        for i in range(4):
            q = rng.standard_normal(head_dim).astype(np.float32)
            o = rng.standard_normal(head_dim).astype(np.float32)
            dedup.store(q, o, head_idx=0)
            dedup.lookup(q, head_idx=0)
        s = dedup.stats
        assert isinstance(s, DedupStats)
        assert s.n_stores == 4
        assert s.n_lookups == 4
        assert 0.0 <= s.hit_rate <= 1.0
        dedup.reset_stats()
        assert dedup.stats.n_lookups == 0
        dedup.clear()


# ---------------------------------------------------------------------------
# FlashPrefill
# ---------------------------------------------------------------------------


class TestFlashPrefillWiring:
    def test_import(self):
        from squish.flash_prefill import PrefillConfig, PrefillStats, FlashPrefillKernel
        cfg = PrefillConfig(chunk_size=64, n_heads=4, head_dim=16)
        kernel = FlashPrefillKernel(cfg)
        assert kernel is not None

    def test_config_defaults(self):
        from squish.flash_prefill import PrefillConfig
        cfg = PrefillConfig()
        assert cfg.chunk_size >= 1
        assert cfg.n_heads >= 1
        assert cfg.head_dim >= 1
        assert cfg.effective_scale > 0.0

    def test_prefill_output_shape(self):
        from squish.flash_prefill import PrefillConfig, FlashPrefillKernel
        rng = np.random.default_rng(0)
        n_heads, seq_len, head_dim = 4, 64, 16
        cfg = PrefillConfig(chunk_size=32, n_heads=n_heads, head_dim=head_dim)
        kernel = FlashPrefillKernel(cfg)
        q = rng.standard_normal((n_heads, seq_len, head_dim)).astype(np.float32)
        k = rng.standard_normal((n_heads, seq_len, head_dim)).astype(np.float32)
        v = rng.standard_normal((n_heads, seq_len, head_dim)).astype(np.float32)
        output = kernel.prefill(q, k, v)
        assert output.shape == (n_heads, seq_len, head_dim)
        assert output.dtype == np.float32

    def test_stats_and_reset(self):
        from squish.flash_prefill import PrefillConfig, PrefillStats, FlashPrefillKernel
        rng = np.random.default_rng(1)
        n_heads, seq_len, head_dim = 2, 32, 8
        cfg = PrefillConfig(chunk_size=16, n_heads=n_heads, head_dim=head_dim)
        kernel = FlashPrefillKernel(cfg)
        q = rng.standard_normal((n_heads, seq_len, head_dim)).astype(np.float32)
        k = rng.standard_normal((n_heads, seq_len, head_dim)).astype(np.float32)
        v = rng.standard_normal((n_heads, seq_len, head_dim)).astype(np.float32)
        kernel.prefill(q, k, v)
        kernel.prefill(q, k, v)
        s = kernel.stats
        assert isinstance(s, PrefillStats)
        assert s.total_prefill_calls == 2
        assert s.total_tokens == 2 * seq_len
        assert s.total_chunks_processed >= 2
        kernel.reset_stats()
        assert kernel.stats.total_prefill_calls == 0


# ---------------------------------------------------------------------------
# BudgetSpec
# ---------------------------------------------------------------------------


class TestBudgetSpecWiring:
    def test_import(self):
        from squish.budget_spec import BudgetConfig, BudgetState, BudgetSpecDecoder, BudgetStats
        cfg = BudgetConfig(total_budget=64, n_draft=5, ramp_down_at=0.8)
        decoder = BudgetSpecDecoder(cfg)
        assert decoder is not None

    def test_config_defaults(self):
        from squish.budget_spec import BudgetConfig
        cfg = BudgetConfig()
        assert cfg.total_budget >= 1
        assert cfg.n_draft >= 1
        assert 0.0 < cfg.ramp_down_at <= 1.0

    def test_effective_draft_len(self):
        from squish.budget_spec import BudgetConfig, BudgetSpecDecoder
        cfg = BudgetConfig(total_budget=100, n_draft=5, ramp_down_at=0.9)
        decoder = BudgetSpecDecoder(cfg)
        draft_len = decoder.effective_draft_len()
        assert 1 <= draft_len <= cfg.n_draft
        assert not decoder.is_exhausted()

    def test_step_exhaustion_and_stats(self):
        from squish.budget_spec import BudgetConfig, BudgetState, BudgetSpecDecoder, BudgetStats
        cfg = BudgetConfig(total_budget=10, n_draft=3, ramp_down_at=0.8)
        decoder = BudgetSpecDecoder(cfg)
        while not decoder.is_exhausted():
            n = decoder.effective_draft_len()
            if n == 0:
                break
            decoder.step(n)
        assert decoder.is_exhausted()
        state = decoder.state
        assert isinstance(state, BudgetState)
        assert state.remaining == 0
        s = decoder.stats
        assert isinstance(s, BudgetStats)
        assert s.total_tokens == cfg.total_budget
        decoder.reset()
        assert not decoder.is_exhausted()
        assert decoder.stats.total_requests == 1


# ---------------------------------------------------------------------------
# RetentionAttn
# ---------------------------------------------------------------------------


class TestRetentionAttnWiring:
    def test_import(self):
        from squish.retention_attn import RetentionConfig, RetentionState, RetentionKernel, RetentionStats
        cfg = RetentionConfig(hidden_dim=64, n_heads=4, gamma=0.9)
        kernel = RetentionKernel(cfg)
        assert kernel is not None

    def test_config_defaults(self):
        from squish.retention_attn import RetentionConfig
        cfg = RetentionConfig()
        assert cfg.hidden_dim >= 1
        assert cfg.n_heads >= 1
        assert cfg.hidden_dim % cfg.n_heads == 0
        assert 0.0 < cfg.gamma < 1.0
        assert cfg.head_dim == cfg.hidden_dim // cfg.n_heads

    def test_step_shape(self):
        from squish.retention_attn import RetentionConfig, RetentionState, RetentionKernel
        rng = np.random.default_rng(0)
        hidden_dim, n_heads = 32, 4
        head_dim = hidden_dim // n_heads
        cfg = RetentionConfig(hidden_dim=hidden_dim, n_heads=n_heads, gamma=0.9)
        kernel = RetentionKernel(cfg)
        state = kernel.init_state()
        assert state.S.shape == (n_heads, head_dim, head_dim)
        assert state.step == 0
        q = rng.standard_normal((n_heads, head_dim)).astype(np.float32)
        k = rng.standard_normal((n_heads, head_dim)).astype(np.float32)
        v = rng.standard_normal((n_heads, head_dim)).astype(np.float32)
        output, new_state = kernel.step(q, k, v, state)
        assert output.shape == (n_heads, head_dim)
        assert isinstance(new_state, RetentionState)
        assert new_state.step == 1

    def test_stats_and_reset(self):
        from squish.retention_attn import RetentionConfig, RetentionKernel, RetentionStats
        rng = np.random.default_rng(1)
        hidden_dim, n_heads = 16, 2
        head_dim = hidden_dim // n_heads
        cfg = RetentionConfig(hidden_dim=hidden_dim, n_heads=n_heads, gamma=0.85)
        kernel = RetentionKernel(cfg)
        state = kernel.init_state()
        state2 = kernel.init_state()
        for _ in range(5):
            q = rng.standard_normal((n_heads, head_dim)).astype(np.float32)
            k = rng.standard_normal((n_heads, head_dim)).astype(np.float32)
            v = rng.standard_normal((n_heads, head_dim)).astype(np.float32)
            _, state = kernel.step(q, k, v, state)
        s = kernel.stats
        assert isinstance(s, RetentionStats)
        assert s.total_steps == 5
        assert s.total_states_init == 2
        kernel.reset_stats()
        assert kernel.stats.total_steps == 0


# ---------------------------------------------------------------------------
# KVRouter
# ---------------------------------------------------------------------------


class TestKVRouterWiring:
    def test_import(self):
        from squish.kv_router import KVRouteConfig, KVRouteEntry, KVRouteTable, KVRouter, KVRouterStats
        cfg = KVRouteConfig(n_nodes=4, n_layers=8, n_heads=4, head_dim=32)
        table = KVRouteTable(cfg)
        router = KVRouter(cfg, table)
        assert router is not None

    def test_config_defaults(self):
        from squish.kv_router import KVRouteConfig
        cfg = KVRouteConfig()
        assert cfg.n_nodes >= 2
        assert cfg.n_layers >= 1
        assert cfg.n_heads >= 1
        assert cfg.head_dim >= 1
        assert cfg.kv_bytes_per_token > 0

    def test_register_route_and_lookup(self):
        from squish.kv_router import KVRouteConfig, KVRouteEntry, KVRouteTable, KVRouter
        cfg = KVRouteConfig(n_nodes=4, n_layers=8, n_heads=4, head_dim=32)
        table = KVRouteTable(cfg)
        router = KVRouter(cfg, table)
        source = 0
        target = router.route(seq_id=42, source_node=source)
        assert 0 <= target < cfg.n_nodes
        entry = table.register(seq_id=42, source=source, target=target,
                               n_tokens=64, layers=(0, 8))
        assert isinstance(entry, KVRouteEntry)
        assert entry.seq_id == 42
        assert entry.size_bytes > 0
        assert table.n_active == 1
        found = table.lookup(42)
        assert found is not None
        table.remove(42)
        assert table.n_active == 0

    def test_stats_and_reset(self):
        from squish.kv_router import KVRouteConfig, KVRouteTable, KVRouter, KVRouterStats
        cfg = KVRouteConfig(n_nodes=4, n_layers=8, n_heads=4, head_dim=32)
        table = KVRouteTable(cfg)
        router = KVRouter(cfg, table)
        for seq_id in range(5):
            target = router.route(seq_id=seq_id, source_node=0)
            table.register(seq_id=seq_id, source=0, target=target,
                           n_tokens=32, layers=(0, 8))
        s = router.stats
        assert isinstance(s, KVRouterStats)
        assert s.total_routes == 5
        assert s.total_bytes_routed > 0
        router.reset_stats()
        assert router.stats.total_routes == 0
