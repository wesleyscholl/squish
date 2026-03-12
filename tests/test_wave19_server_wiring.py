"""tests/test_wave19_server_wiring.py

Verifies that all Wave 19 module classes are importable and have the expected
public APIs that the server.py wiring code depends on.  These are pure
import + instantiation tests — no model or GPU required.

Wave 19 modules (Quantization + Attention + Norm + SpecDecode + Serving):
  fp8_quant, mx_quant, flash_decode, paged_kv, gqa,
  sliding_window_attn, rope_scaling, act_sparsity, fused_rmsnorm,
  lora_inference, medusa, eagle3, prefix_pool, token_healer
"""

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# FP8Quant
# ---------------------------------------------------------------------------

class TestFP8QuantWiring:
    def test_import_and_config_properties(self):
        from squish.fp8_quant import FP8Config
        cfg = FP8Config(fmt="e4m3", block_size=128, per_channel=True)
        assert cfg.max_val == 448.0
        assert cfg.mantissa_bits == 3
        assert cfg.exponent_bits == 4
        cfg2 = FP8Config(fmt="e5m2", block_size=64)
        assert cfg2.max_val == 57344.0
        assert cfg2.mantissa_bits == 2
        assert cfg2.exponent_bits == 5

    def test_encode_decode_roundtrip_e4m3(self):
        from squish.fp8_quant import fp8_decode, fp8_encode_e4m3
        rng = np.random.default_rng(0)
        x = rng.standard_normal(256).astype(np.float32)
        codes, scale = fp8_encode_e4m3(x)
        assert codes.dtype == np.uint8
        assert codes.shape == x.shape
        assert scale > 0.0
        decoded = fp8_decode(codes, scale, fmt="e4m3")
        assert decoded.shape == x.shape
        # relative error should be reasonable for FP8
        nonzero = np.abs(x) > 1e-3
        if np.any(nonzero):
            err = np.mean(np.abs(x[nonzero] - decoded[nonzero]) / np.abs(x[nonzero]))
            assert err < 1.0

    def test_quantizer_encode_decode_per_channel(self):
        from squish.fp8_quant import FP8Config, FP8Quantizer, FP8Tensor
        rng = np.random.default_rng(1)
        cfg = FP8Config(fmt="e4m3", block_size=128, per_channel=True)
        quant = FP8Quantizer(cfg)
        w = rng.standard_normal((4, 64)).astype(np.float32)
        enc = quant.encode(w)
        assert isinstance(enc, FP8Tensor)
        assert enc.fmt == "e4m3"
        assert enc.shape == (4, 64)
        assert enc.data.dtype == np.uint8
        rec = quant.decode(enc)
        assert rec.shape == w.shape
        rel_err = quant.relative_error(w, rec)
        assert 0.0 <= rel_err < 1.0

    def test_fp8_tensor_properties_and_e5m2_path(self):
        from squish.fp8_quant import FP8Config, FP8Quantizer, fp8_encode_e5m2
        rng = np.random.default_rng(2)
        x = rng.standard_normal(512).astype(np.float32)
        codes, scale = fp8_encode_e5m2(x)
        assert codes.dtype == np.uint8
        # FP8Tensor compression ratio must be 4x vs float32
        from squish.fp8_quant import FP8Tensor
        t = FP8Tensor(data=codes, scales=np.array([scale], dtype=np.float32),
                      shape=(512,), fmt="e5m2")
        assert t.compression_ratio == 4.0
        assert t.n_elements == 512
        assert t.bits_per_element == 8


# ---------------------------------------------------------------------------
# MXQuant
# ---------------------------------------------------------------------------

class TestMXQuantWiring:
    def test_import_and_config_properties(self):
        from squish.mx_quant import MXConfig
        cfg4 = MXConfig(fmt="mx4", tile_size=32)
        assert cfg4.bits_per_element == 4
        assert cfg4.mantissa_bits == 2
        assert cfg4.compression_ratio == 8.0
        cfg9 = MXConfig(fmt="mx9", tile_size=32)
        assert cfg9.bits_per_element == 9
        assert cfg9.mantissa_bits == 4

    def test_quantizer_encode_decode_roundtrip_mx4(self):
        from squish.mx_quant import MXConfig, MXQuantizer, MXTensor
        rng = np.random.default_rng(3)
        cfg = MXConfig(fmt="mx4", tile_size=32)
        quant = MXQuantizer(cfg)
        w = rng.standard_normal((8, 64)).astype(np.float32)
        enc = quant.encode(w)
        assert isinstance(enc, MXTensor)
        assert enc.shape == (8, 64)
        assert enc.fmt == "mx4"
        assert enc.mantissas.dtype == np.uint8
        rec = quant.decode(enc)
        assert rec.shape == w.shape

    def test_snr_db_mx6(self):
        from squish.mx_quant import MXConfig, MXQuantizer
        rng = np.random.default_rng(4)
        cfg = MXConfig(fmt="mx6", tile_size=32)
        quant = MXQuantizer(cfg)
        x = rng.standard_normal(256).astype(np.float32)
        enc = quant.encode(x)
        rec = quant.decode(enc)
        snr = quant.snr_db(x, rec)
        # SNR should be finite and positive for any non-trivial input
        assert np.isfinite(snr)

    def test_mx_tensor_properties(self):
        from squish.mx_quant import MXConfig, MXQuantizer
        rng = np.random.default_rng(5)
        cfg = MXConfig(fmt="mx9", tile_size=16)
        quant = MXQuantizer(cfg)
        x = rng.standard_normal((3, 48)).astype(np.float32)
        enc = quant.encode(x)
        assert enc.n_elements == 3 * 48
        assert enc.bits_per_element == 9
        assert enc.tile_exps.shape[0] == (3 * 48 + 15) // 16


# ---------------------------------------------------------------------------
# FlashDecode
# ---------------------------------------------------------------------------

class TestFlashDecodeWiring:
    def test_import_and_config_defaults(self):
        from squish.flash_decode import FlashDecodeConfig
        cfg = FlashDecodeConfig(n_heads=8, head_dim=64, n_splits=4)
        assert cfg.effective_scale == pytest.approx(1.0 / 64 ** 0.5, rel=1e-5)
        assert cfg.kv_n_heads == 8
        assert cfg.kv_group_size == 1

    def test_decode_output_shape(self):
        from squish.flash_decode import FlashDecodeAttention, FlashDecodeConfig
        rng = np.random.default_rng(6)
        n_heads, head_dim, seq_len = 4, 32, 64
        cfg = FlashDecodeConfig(n_heads=n_heads, head_dim=head_dim, n_splits=4)
        attn = FlashDecodeAttention(cfg)
        q = rng.standard_normal((n_heads, head_dim)).astype(np.float32)
        k = rng.standard_normal((n_heads, seq_len, head_dim)).astype(np.float32)
        v = rng.standard_normal((n_heads, seq_len, head_dim)).astype(np.float32)
        out = attn.decode(q, k, v)
        assert out.shape == (n_heads, head_dim)
        assert out.dtype == np.float32

    def test_merge_split_results(self):
        from squish.flash_decode import FlashDecodeSplit, merge_split_results
        rng = np.random.default_rng(7)
        n_heads, head_dim = 4, 32
        splits = []
        for _ in range(3):
            out = rng.standard_normal((n_heads, head_dim)).astype(np.float32)
            lse = rng.standard_normal(n_heads).astype(np.float32)
            ms  = rng.standard_normal(n_heads).astype(np.float32)
            splits.append(FlashDecodeSplit(output=out, log_sum_exp=lse, max_score=ms))
        merged = merge_split_results(splits)
        assert merged.shape == (n_heads, head_dim)
        assert np.all(np.isfinite(merged))

    def test_avg_kv_len_and_reset_stats(self):
        from squish.flash_decode import FlashDecodeAttention, FlashDecodeConfig
        rng = np.random.default_rng(8)
        cfg = FlashDecodeConfig(n_heads=4, head_dim=16, n_splits=2)
        attn = FlashDecodeAttention(cfg)
        assert attn.avg_kv_len == 0.0
        q = rng.standard_normal((4, 16)).astype(np.float32)
        k = rng.standard_normal((4, 32, 16)).astype(np.float32)
        v = rng.standard_normal((4, 32, 16)).astype(np.float32)
        attn.decode(q, k, v)
        assert attn.avg_kv_len == 32.0
        attn.reset_stats()
        assert attn.avg_kv_len == 0.0


# ---------------------------------------------------------------------------
# PagedKV
# ---------------------------------------------------------------------------

class TestPagedKVWiring:
    def test_import_and_config_validation(self):
        from squish.paged_kv import PagedKVConfig
        cfg = PagedKVConfig(block_size=16, n_blocks=64, n_heads=8, head_dim=32)
        assert cfg.kv_n_heads == 8
        # GQA config: n_heads must be divisible by kv_n_heads
        cfg_gqa = PagedKVConfig(block_size=16, n_blocks=64, n_heads=8,
                                head_dim=32, kv_n_heads=2)
        assert cfg_gqa.kv_n_heads == 2

    def test_append_and_gather_roundtrip(self):
        from squish.paged_kv import PagedKVCache, PagedKVConfig
        rng = np.random.default_rng(9)
        cfg = PagedKVConfig(block_size=8, n_blocks=16, n_heads=2, head_dim=16)
        cache = PagedKVCache(cfg)
        keys_in, vals_in = [], []
        for _ in range(5):
            k = rng.standard_normal((2, 16)).astype(np.float32)
            v = rng.standard_normal((2, 16)).astype(np.float32)
            cache.append(seq_id=0, key=k, value=v)
            keys_in.append(k)
            vals_in.append(v)
        k_out, v_out = cache.gather(seq_id=0)
        assert k_out.shape == (2, 5, 16)
        assert v_out.shape == (2, 5, 16)
        np.testing.assert_allclose(k_out[:, 0, :], keys_in[0], atol=1e-5)

    def test_free_and_utilization(self):
        from squish.paged_kv import PagedKVCache, PagedKVConfig
        rng = np.random.default_rng(10)
        cfg = PagedKVConfig(block_size=4, n_blocks=8, n_heads=2, head_dim=8)
        cache = PagedKVCache(cfg)
        k = rng.standard_normal((2, 8)).astype(np.float32)
        v = rng.standard_normal((2, 8)).astype(np.float32)
        cache.append(seq_id=1, key=k, value=v)
        assert cache.utilization > 0.0
        assert cache.n_sequences == 1
        cache.free(seq_id=1)
        assert cache.utilization == 0.0
        assert cache.n_sequences == 0

    def test_paged_kv_stats_properties(self):
        from squish.paged_kv import PagedKVCache, PagedKVConfig, PagedKVStats
        rng = np.random.default_rng(11)
        cfg = PagedKVConfig(block_size=4, n_blocks=16, n_heads=2, head_dim=8)
        cache = PagedKVCache(cfg)
        k = rng.standard_normal((2, 8)).astype(np.float32)
        v = rng.standard_normal((2, 8)).astype(np.float32)
        cache.append(0, k, v)
        cache.free(0)
        st = cache.stats
        assert st.total_allocations == 1
        assert st.total_frees == 1
        assert 0.0 <= st.utilization_rate <= 1.0
        standalone = PagedKVStats(total_allocations=10, total_frees=5,
                                  peak_blocks_used=4, n_total_blocks=16)
        assert standalone.utilization_rate == pytest.approx(4 / 16)


# ---------------------------------------------------------------------------
# GQA
# ---------------------------------------------------------------------------

class TestGQAWiring:
    def test_import_and_config_group_size(self):
        from squish.gqa import GQAConfig
        cfg = GQAConfig(n_q_heads=32, n_kv_heads=8, head_dim=128)
        assert cfg.group_size == 4
        assert cfg.softmax_scale == pytest.approx(1.0 / 128 ** 0.5, rel=1e-5)
        # MQA: single KV head
        mqa = GQAConfig(n_q_heads=8, n_kv_heads=1, head_dim=64)
        assert mqa.group_size == 8

    def test_cache_append_and_get_kv(self):
        from squish.gqa import GQACache, GQAConfig
        rng = np.random.default_rng(12)
        cfg = GQAConfig(n_q_heads=4, n_kv_heads=2, head_dim=16, max_seq_len=32)
        cache = GQACache(cfg)
        assert cache.seq_len == 0
        for _ in range(6):
            k = rng.standard_normal((2, 16)).astype(np.float32)
            v = rng.standard_normal((2, 16)).astype(np.float32)
            cache.append(k, v)
        assert cache.seq_len == 6
        keys, values = cache.get_kv()
        assert keys.shape == (2, 6, 16)
        assert values.shape == (2, 6, 16)

    def test_grouped_query_attention_output_shape(self):
        from squish.gqa import GQAConfig, grouped_query_attention
        rng = np.random.default_rng(13)
        cfg = GQAConfig(n_q_heads=4, n_kv_heads=2, head_dim=16, max_seq_len=32)
        seq_q, seq_kv = 3, 10
        q = rng.standard_normal((4, seq_q, 16)).astype(np.float32)
        k = rng.standard_normal((2, seq_kv, 16)).astype(np.float32)
        v = rng.standard_normal((2, seq_kv, 16)).astype(np.float32)
        out = grouped_query_attention(q, k, v, cfg)
        assert out.shape == (4, seq_q, 16)
        assert np.all(np.isfinite(out))

    def test_gqa_stats_memory_ratio_and_reset(self):
        from squish.gqa import GQACache, GQAConfig, GQAStats
        cfg = GQAConfig(n_q_heads=8, n_kv_heads=2, head_dim=16, max_seq_len=32)
        cache = GQACache(cfg)
        rng = np.random.default_rng(14)
        for _ in range(4):
            k = rng.standard_normal((2, 16)).astype(np.float32)
            v = rng.standard_normal((2, 16)).astype(np.float32)
            cache.append(k, v)
        st = cache.stats
        assert st.n_appends == 4
        assert st.memory_ratio == pytest.approx(2 / 8)
        assert st.total_kv_heads_saved == 4 * (8 - 2)
        cache.reset()
        assert cache.seq_len == 0


# ---------------------------------------------------------------------------
# SlidingWindowAttn
# ---------------------------------------------------------------------------

class TestSlidingWindowAttnWiring:
    def test_import_and_config_defaults(self):
        from squish.sliding_window_attn import SWAConfig
        cfg = SWAConfig(window_size=16, n_heads=4, head_dim=32)
        assert cfg.kv_n_heads == 4
        cfg_gqa = SWAConfig(window_size=16, n_heads=8, head_dim=32, kv_n_heads=2)
        assert cfg_gqa.kv_n_heads == 2

    def test_cache_append_fill_ratio_and_eviction(self):
        from squish.sliding_window_attn import SWAConfig, SlidingWindowKVCache
        rng = np.random.default_rng(15)
        cfg = SWAConfig(window_size=8, n_heads=2, head_dim=16)
        cache = SlidingWindowKVCache(cfg)
        for i in range(12):
            k = rng.standard_normal((2, 16)).astype(np.float32)
            v = rng.standard_normal((2, 16)).astype(np.float32)
            cache.append(k, v)
        assert cache.fill_ratio == 1.0
        assert cache.window_used == 8
        assert cache.seq_len == 12
        assert cache.stats.tokens_evicted == 12 - 8

    def test_sliding_window_attention_output_shape(self):
        from squish.sliding_window_attn import (
            SWAConfig,
            SlidingWindowKVCache,
            sliding_window_attention,
        )
        rng = np.random.default_rng(16)
        cfg = SWAConfig(window_size=10, n_heads=4, head_dim=16)
        cache = SlidingWindowKVCache(cfg)
        for _ in range(6):
            k = rng.standard_normal((4, 16)).astype(np.float32)
            v = rng.standard_normal((4, 16)).astype(np.float32)
            cache.append(k, v)
        q = rng.standard_normal((4, 16)).astype(np.float32)
        out = sliding_window_attention(q, cache, cfg)
        assert out.shape == (4, 16)
        assert np.all(np.isfinite(out))

    def test_swa_stats_eviction_rate_and_reset(self):
        from squish.sliding_window_attn import SWAConfig, SlidingWindowKVCache, SWAStats
        rng = np.random.default_rng(17)
        cfg = SWAConfig(window_size=4, n_heads=2, head_dim=8)
        cache = SlidingWindowKVCache(cfg)
        for _ in range(8):
            k = rng.standard_normal((2, 8)).astype(np.float32)
            v = rng.standard_normal((2, 8)).astype(np.float32)
            cache.append(k, v)
        st = cache.stats
        assert st.total_tokens_seen == 8
        assert st.eviction_rate == pytest.approx(4 / 8)
        cache.reset()
        assert cache.seq_len == 0
        assert cache.window_used == 0


# ---------------------------------------------------------------------------
# RoPEScaling
# ---------------------------------------------------------------------------

class TestRoPEScalingWiring:
    def test_import_and_config_scale_factor(self):
        from squish.rope_scaling import RoPEConfig
        cfg = RoPEConfig(head_dim=64, base_theta=10000.0,
                         original_max_len=4096, target_max_len=32768,
                         method="ntk")
        assert cfg.scale_factor == pytest.approx(8.0)
        # Explicit scale_factor overrides computed value
        cfg2 = RoPEConfig(head_dim=64, original_max_len=2048,
                          target_max_len=16384, method="yarn",
                          scale_factor=8.0)
        assert cfg2.scale_factor == 8.0

    def test_create_rope_scaler_factory(self):
        from squish.rope_scaling import (
            LongRoPEScaler,
            NTKScaler,
            RoPEConfig,
            YaRNScaler,
            create_rope_scaler,
        )
        for method, cls in (("ntk", NTKScaler), ("yarn", YaRNScaler),
                            ("longrope", LongRoPEScaler)):
            cfg = RoPEConfig(head_dim=32, original_max_len=512,
                             target_max_len=4096, method=method)
            scaler = create_rope_scaler(cfg)
            assert isinstance(scaler, cls)

    def test_ntk_scaler_apply_output_shape(self):
        from squish.rope_scaling import NTKScaler, RoPEConfig
        rng = np.random.default_rng(18)
        cfg = RoPEConfig(head_dim=32, original_max_len=512,
                         target_max_len=4096, method="ntk")
        scaler = NTKScaler(cfg)
        seq_len, n_heads = 16, 4
        x = rng.standard_normal((seq_len, n_heads, 32)).astype(np.float32)
        pos_ids = np.arange(seq_len)
        out = scaler.apply(x, pos_ids)
        assert out.shape == (seq_len, n_heads, 32)
        assert out.dtype == np.float32

    def test_yarn_longrope_get_freqs_shape(self):
        from squish.rope_scaling import LongRoPEScaler, RoPEConfig, YaRNScaler
        for cls, method in ((YaRNScaler, "yarn"), (LongRoPEScaler, "longrope")):
            cfg = RoPEConfig(head_dim=64, original_max_len=512,
                             target_max_len=4096, method=method)
            scaler = cls(cfg)
            freqs = scaler.get_freqs(seq_len=100)
            assert freqs.shape == (100, 32)  # (seq_len, head_dim // 2)


# ---------------------------------------------------------------------------
# ActSparsity
# ---------------------------------------------------------------------------

class TestActSparsityWiring:
    def test_import_and_config_validation(self):
        from squish.act_sparsity import SparsityConfig
        cfg = SparsityConfig(hidden_dim=256, n_layers=4, threshold=0.02)
        assert cfg.hidden_dim == 256
        assert cfg.threshold == 0.02
        with pytest.raises(ValueError):
            SparsityConfig(hidden_dim=0)
        with pytest.raises(ValueError):
            SparsityConfig(threshold=-0.1)

    def test_predictor_record_and_calibrate(self):
        from squish.act_sparsity import ActSparsityPredictor, SparsityConfig
        rng = np.random.default_rng(19)
        cfg = SparsityConfig(hidden_dim=64, n_layers=4, threshold=0.5)
        pred = ActSparsityPredictor(cfg)
        # Create activations with 50% near-zero values
        acts = rng.uniform(-1, 1, (8, 64)).astype(np.float32)
        pred.record(layer_idx=0, activations=acts)
        sparsity_map = pred.calibrate()
        assert 0 in sparsity_map
        assert 0.0 <= sparsity_map[0] <= 1.0
        # Unrecorded layer returns min_sparsity
        assert pred.get_sparsity(3) == cfg.min_sparsity

    def test_sparse_ffn_gate_apply_and_compression(self):
        from squish.act_sparsity import SparseFFNGate, SparsityConfig
        rng = np.random.default_rng(20)
        cfg = SparsityConfig(hidden_dim=64, n_layers=2, threshold=1.0)
        gate = SparseFFNGate(cfg, layer_idx=0)
        # Before any apply(), compression_ratio == 1.0
        assert gate.compression_ratio() == 1.0
        acts = rng.uniform(-2, 2, (4, 64)).astype(np.float32)
        masked = gate.apply(acts)
        assert masked.shape == acts.shape
        # Values are either zeroed out or at/above threshold
        assert np.all((masked == 0.0) | (np.abs(masked) >= cfg.threshold))
        ratio = gate.compression_ratio()
        assert 0.0 <= ratio <= 1.0

    def test_act_sparsity_stats_properties(self):
        from squish.act_sparsity import ActSparsityStats
        st = ActSparsityStats(total_activations_seen=1000,
                              total_zeros=400, total_skipped_layers=10)
        assert st.sparsity_rate == pytest.approx(0.4)
        zero_st = ActSparsityStats()
        assert zero_st.sparsity_rate == 0.0
        assert zero_st.skip_rate == 0.0


# ---------------------------------------------------------------------------
# FusedRMSNorm
# ---------------------------------------------------------------------------

class TestFusedRMSNormWiring:
    def test_import_and_config_validation(self):
        from squish.fused_rmsnorm import FusedNormConfig
        cfg = FusedNormConfig(hidden_dim=256, eps=1e-5)
        assert cfg.hidden_dim == 256
        assert cfg.eps == 1e-5
        with pytest.raises(ValueError):
            FusedNormConfig(hidden_dim=0)
        with pytest.raises(ValueError):
            FusedNormConfig(eps=-1e-6)

    def test_fused_rmsnorm_forward_shape_and_residual(self):
        from squish.fused_rmsnorm import FusedNormConfig, FusedRMSNorm
        rng = np.random.default_rng(21)
        dim = 64
        cfg = FusedNormConfig(hidden_dim=dim, eps=1e-6, add_residual=True)
        norm = FusedRMSNorm(cfg)
        x = rng.standard_normal((8, dim)).astype(np.float32)
        res = np.zeros_like(x)
        out, res_out = norm.forward(x, residual=res)
        assert out.shape == (8, dim)
        assert res_out is not None
        assert res_out.shape == (8, dim)
        # Without residual
        out2, res_out2 = norm.forward(x, residual=None)
        assert out2.shape == (8, dim)
        assert res_out2 is None

    def test_fused_layer_norm_forward(self):
        from squish.fused_rmsnorm import FusedLayerNorm, FusedNormConfig
        rng = np.random.default_rng(22)
        dim = 32
        cfg = FusedNormConfig(hidden_dim=dim, eps=1e-5, add_residual=False)
        norm = FusedLayerNorm(cfg)
        x = rng.standard_normal((4, dim)).astype(np.float32)
        out, res_out = norm.forward(x)
        assert out.shape == (4, dim)
        assert res_out is None
        # Output should have approximately zero mean per row
        means = out.mean(axis=-1)
        np.testing.assert_allclose(means, 0.0, atol=1e-5)

    def test_fused_add_rms_norm_function(self):
        from squish.fused_rmsnorm import fused_add_rms_norm
        rng = np.random.default_rng(23)
        dim = 16
        x = rng.standard_normal((4, dim)).astype(np.float32)
        residual = rng.standard_normal((4, dim)).astype(np.float32)
        weight = np.ones(dim, dtype=np.float32)
        out, new_res = fused_add_rms_norm(x, residual, weight, eps=1e-6)
        assert out.shape == (4, dim)
        assert new_res.shape == (4, dim)
        # new_residual == x + residual
        np.testing.assert_allclose(new_res, x + residual, atol=1e-5)


# ---------------------------------------------------------------------------
# LoRAInference
# ---------------------------------------------------------------------------

class TestLoRAInferenceWiring:
    def test_import_and_config_scaling(self):
        from squish.lora_inference import LoRAConfig
        cfg = LoRAConfig(rank=16, alpha=32.0)
        assert cfg.rank == 16
        assert cfg.alpha == 32.0
        assert "q_proj" in cfg.target_modules
        with pytest.raises(ValueError):
            LoRAConfig(rank=0)
        with pytest.raises(ValueError):
            LoRAConfig(alpha=-1.0)

    def test_adapter_add_layer_and_apply(self):
        from squish.lora_inference import LoRAConfig, LoRAInferenceAdapter
        rng = np.random.default_rng(24)
        in_f, out_f, r = 64, 64, 4
        cfg = LoRAConfig(rank=r, alpha=8.0)
        adapter = LoRAInferenceAdapter(cfg)
        A = rng.standard_normal((in_f, r)).astype(np.float32) * 0.01
        B = np.zeros((r, out_f), dtype=np.float32)
        adapter.add_layer("q_proj", in_f, out_f, A, B)
        assert "q_proj" in adapter.adapter_names
        x = rng.standard_normal((3, in_f)).astype(np.float32)
        base_out = rng.standard_normal((3, out_f)).astype(np.float32)
        result = adapter.apply("q_proj", x, base_out)
        assert result.shape == (3, out_f)
        # B=0 so delta=0, result must equal base_out
        np.testing.assert_allclose(result, base_out, atol=1e-5)

    def test_lora_layer_forward_and_n_params(self):
        from squish.lora_inference import LoRAConfig, LoRAInferenceAdapter
        rng = np.random.default_rng(25)
        in_f, out_f, r = 32, 48, 8
        cfg = LoRAConfig(rank=r, alpha=16.0)
        adapter = LoRAInferenceAdapter(cfg)
        A = rng.standard_normal((in_f, r)).astype(np.float32)
        B = rng.standard_normal((r, out_f)).astype(np.float32)
        adapter.add_layer("v_proj", in_f, out_f, A, B)
        layer = adapter._layers["v_proj"]
        assert layer.rank == r
        assert layer.n_params == in_f * r + r * out_f
        x = rng.standard_normal((2, in_f)).astype(np.float32)
        delta = layer.forward(x)
        assert delta.shape == (2, out_f)

    def test_merge_into_and_total_params(self):
        from squish.lora_inference import LoRAConfig, LoRAInferenceAdapter
        rng = np.random.default_rng(26)
        in_f, out_f, r = 16, 16, 2
        cfg = LoRAConfig(rank=r, alpha=4.0)
        adapter = LoRAInferenceAdapter(cfg)
        A = rng.standard_normal((in_f, r)).astype(np.float32)
        B = rng.standard_normal((r, out_f)).astype(np.float32)
        adapter.add_layer("q_proj", in_f, out_f, A, B)
        assert adapter.total_params == in_f * r + r * out_f
        W = rng.standard_normal((in_f, out_f)).astype(np.float32)
        merged = adapter.merge_into({"q_proj": W})
        assert merged["q_proj"].shape == W.shape
        # Merged weight must differ from base W by the LoRA delta
        expected_delta = A @ B * (cfg.alpha / r)
        np.testing.assert_allclose(merged["q_proj"], W + expected_delta, atol=1e-4)


# ---------------------------------------------------------------------------
# Medusa
# ---------------------------------------------------------------------------

class TestMedusaWiring:
    def test_import_and_config_validation(self):
        from squish.medusa import MedusaConfig
        cfg = MedusaConfig(n_heads=4, vocab_size=1000, hidden_dim=128)
        assert cfg.n_heads == 4
        assert cfg.top_k_per_head == 10
        with pytest.raises(ValueError):
            MedusaConfig(n_heads=0)
        with pytest.raises(ValueError):
            MedusaConfig(acceptance_threshold=0.0)

    def test_decoder_draft_returns_tree(self):
        from squish.medusa import MedusaConfig, MedusaDecoder, MedusaDraftTree
        rng = np.random.default_rng(27)
        cfg = MedusaConfig(n_heads=3, vocab_size=500, hidden_dim=64)
        decoder = MedusaDecoder(cfg)
        hidden = rng.standard_normal(64).astype(np.float32)
        tree = decoder.draft(hidden)
        assert isinstance(tree, MedusaDraftTree)
        assert tree.depth == 3
        assert len(tree.tokens) == 3
        assert len(tree.probs) == 3
        # path lengths must be 1, 2, 3
        for i, path in enumerate(tree.tokens):
            assert len(path) == i + 1

    def test_decoder_verify_greedy(self):
        from squish.medusa import MedusaConfig, MedusaDecoder
        rng = np.random.default_rng(28)
        cfg = MedusaConfig(n_heads=3, vocab_size=100, hidden_dim=32)
        decoder = MedusaDecoder(cfg)
        # Craft target logits so token 5 has the highest score for all steps
        draft_tokens = [5, 5, 5]
        target_logits = []
        for _ in range(3):
            lg = np.full(100, -10.0, dtype=np.float32)
            lg[5] = 10.0
            target_logits.append(lg)
        accepted, n = decoder.verify(draft_tokens, target_logits)
        assert n == 3
        assert accepted == [5, 5, 5]

    def test_medusa_stats_and_throughput_multiplier(self):
        from squish.medusa import MedusaConfig, MedusaDecoder, MedusaStats
        rng = np.random.default_rng(29)
        cfg = MedusaConfig(n_heads=4, vocab_size=50, hidden_dim=32)
        decoder = MedusaDecoder(cfg)
        # All accepted
        draft = [1, 2, 3, 4]
        logits = [np.array([-10.0] * 50, dtype=np.float32) for _ in range(4)]
        for i, lg in enumerate(logits):
            lg[draft[i]] = 10.0
        decoder.verify(draft, logits)
        assert decoder.acceptance_rate == 1.0
        assert decoder.throughput_multiplier == pytest.approx(1.0 + 4.0)
        st = decoder.get_stats()
        assert isinstance(st, MedusaStats)
        assert st.total_accepted == 4
        assert st.mean_accepted_per_call == 4.0


# ---------------------------------------------------------------------------
# Eagle3
# ---------------------------------------------------------------------------

class TestEagle3Wiring:
    def test_import_and_config_feature_dim_default(self):
        from squish.eagle3 import Eagle3Config
        cfg = Eagle3Config(hidden_dim=128, vocab_size=1000, max_draft_len=5)
        assert cfg.feature_dim == 128  # defaults to hidden_dim
        cfg2 = Eagle3Config(hidden_dim=64, vocab_size=500, feature_dim=32)
        assert cfg2.feature_dim == 32
        with pytest.raises(ValueError):
            Eagle3Config(hidden_dim=0)
        with pytest.raises(ValueError):
            Eagle3Config(acceptance_threshold=0.0)

    def test_draft_head_forward_shapes(self):
        from squish.eagle3 import Eagle3Config, Eagle3DraftHead
        rng = np.random.default_rng(30)
        cfg = Eagle3Config(hidden_dim=64, vocab_size=200, feature_dim=64)
        head = Eagle3DraftHead(cfg)
        hidden = rng.standard_normal(64).astype(np.float32)
        features = head.predict_features(hidden)
        assert features.shape == (64,)
        logits = head.predict_tokens(features)
        assert logits.shape == (200,)
        feats2, logits2 = head.forward(hidden)
        assert feats2.shape == (64,)
        assert logits2.shape == (200,)

    def test_decoder_draft_step_and_verify_step(self):
        from squish.eagle3 import Eagle3Config, Eagle3Decoder
        rng = np.random.default_rng(31)
        cfg = Eagle3Config(hidden_dim=64, vocab_size=200, max_draft_len=4,
                           acceptance_threshold=1e-6)  # near-zero threshold
        decoder = Eagle3Decoder(cfg)
        hidden = rng.standard_normal(64).astype(np.float32)
        steps = decoder.draft_step(hidden, n_steps=4)
        assert len(steps) == 4
        for feats, logits in steps:
            assert feats.shape == (64,)
            assert logits.shape == (200,)
        draft_tokens = [int(np.argmax(lg)) for _, lg in steps]
        # threshold=0.0 means sim > 0 required; test that return types are correct
        accepted, bonus = decoder.verify_step(draft_tokens, hidden)
        assert isinstance(accepted, bool)
        assert isinstance(bonus, int)

    def test_eagle3_stats_mean_similarity_and_acceptance(self):
        from squish.eagle3 import Eagle3Config, Eagle3Decoder, Eagle3Stats
        rng = np.random.default_rng(32)
        cfg = Eagle3Config(hidden_dim=32, vocab_size=100, max_draft_len=3,
                           acceptance_threshold=0.999)  # reject most
        decoder = Eagle3Decoder(cfg)
        h = rng.standard_normal(32).astype(np.float32)
        decoder.draft_step(h, n_steps=3)
        decoder.verify_step([0, 1, 2], h)
        assert decoder.n_total == 1
        st = decoder.get_stats()
        assert isinstance(st, Eagle3Stats)
        assert -1.0 <= st.mean_feature_similarity <= 1.0
        assert 0.0 <= st.acceptance_rate <= 1.0


# ---------------------------------------------------------------------------
# PrefixPool
# ---------------------------------------------------------------------------

class TestPrefixPoolWiring:
    def test_import_and_config_defaults(self):
        from squish.prefix_pool import PrefixPoolConfig
        cfg = PrefixPoolConfig(max_entries=64, n_heads=4, head_dim=16)
        assert cfg.kv_n_heads == 4  # defaults to n_heads
        assert cfg.eviction_policy == "lru"
        with pytest.raises(ValueError):
            PrefixPoolConfig(max_entries=0)
        with pytest.raises(ValueError):
            PrefixPoolConfig(eviction_policy="random")

    def test_put_and_get_cache_hit(self):
        from squish.prefix_pool import PrefixPool, PrefixPoolConfig
        rng = np.random.default_rng(33)
        cfg = PrefixPoolConfig(max_entries=16, n_heads=2, head_dim=8)
        pool = PrefixPool(cfg)
        tokens = [10, 20, 30]
        k = rng.standard_normal((2, 3, 8)).astype(np.float32)
        v = rng.standard_normal((2, 3, 8)).astype(np.float32)
        h = pool.put(tokens, k, v)
        assert isinstance(h, str)
        assert len(h) == 64  # SHA-256 hex digest
        assert pool.contains(tokens)
        result = pool.get(tokens)
        assert result is not None
        k_cached, v_cached = result
        np.testing.assert_allclose(k_cached, k, atol=1e-6)

    def test_eviction_lru_and_lfu(self):
        from squish.prefix_pool import PrefixPool, PrefixPoolConfig
        rng = np.random.default_rng(34)
        cfg = PrefixPoolConfig(max_entries=2, n_heads=1, head_dim=4,
                               eviction_policy="lru")
        pool = PrefixPool(cfg)
        k = rng.standard_normal((1, 2, 4)).astype(np.float32)
        v = rng.standard_normal((1, 2, 4)).astype(np.float32)
        pool.put([1, 2], k, v)
        pool.put([3, 4], k, v)
        assert pool.size == 2
        # Inserting a third entry must evict one
        pool.put([5, 6], k, v)
        assert pool.size == 2

    def test_pool_stats_hit_rate_and_kv_saved(self):
        from squish.prefix_pool import PrefixPool, PrefixPoolConfig
        rng = np.random.default_rng(35)
        cfg = PrefixPoolConfig(max_entries=8, n_heads=1, head_dim=4)
        pool = PrefixPool(cfg)
        k = rng.standard_normal((1, 5, 4)).astype(np.float32)
        v = rng.standard_normal((1, 5, 4)).astype(np.float32)
        pool.put([1, 2, 3, 4, 5], k, v)
        pool.get([1, 2, 3, 4, 5])  # hit
        pool.get([99])              # miss
        st = pool.get_stats()
        assert st.n_hits == 1
        assert st.n_misses == 1
        assert st.hit_rate == pytest.approx(0.5)
        assert pool.total_kv_saved == 5  # seq_len of the hit entry


# ---------------------------------------------------------------------------
# TokenHealer
# ---------------------------------------------------------------------------

class TestTokenHealerWiring:
    def test_import_and_config_validation(self):
        from squish.token_healer import HealerConfig
        cfg = HealerConfig(vocab_size=1000, max_healing_tokens=8)
        assert cfg.vocab_size == 1000
        assert cfg.min_prefix_len == 1
        with pytest.raises(ValueError):
            HealerConfig(vocab_size=0)
        with pytest.raises(ValueError):
            HealerConfig(max_healing_tokens=0)

    def test_find_suffix_overlap_detects_prefix(self):
        from squish.token_healer import HealerConfig, TokenHealer
        # vocab: id 0="_va", id 1="_value", id 2="_var"
        vocab = ["_va", "_value", "_var", " is"]
        cfg = HealerConfig(vocab_size=len(vocab), max_healing_tokens=4)
        healer = TokenHealer(cfg, vocab_list=vocab)
        # Token 0 ("_va") is a proper prefix of "_value" and "_var"
        n, s = healer.find_suffix_overlap([2, 0])
        assert n == 1
        assert s == "_va"
        # No overlap for " is" (not a prefix of anything longer)
        n2, s2 = healer.find_suffix_overlap([3])
        assert n2 == 0

    def test_heal_backs_up_tokens_and_tracks_stats(self):
        from squish.token_healer import HealerConfig, TokenHealer
        vocab = ["_va", "_value", "_var", " is"]
        cfg = HealerConfig(vocab_size=len(vocab), max_healing_tokens=4)
        healer = TokenHealer(cfg, vocab_list=vocab)
        # Prompt ends with token 0 ("_va") — needs healing
        prompt = [2, 2, 0]
        completion = [1, 3]   # "_value is"
        healed = healer.heal(prompt, completions=[completion])
        # The last token of prompt ([0]) should be backed up
        assert healed == [2, 2] + completion
        assert healer.n_healed == 1
        assert healer.avg_overlap_tokens == 1.0

    def test_healer_stats_avg_tokens_per_heal(self):
        from squish.token_healer import HealerConfig, HealerStats, TokenHealer
        vocab = ["ab", "abc", "xyz", "x"]
        cfg = HealerConfig(vocab_size=len(vocab), max_healing_tokens=4)
        healer = TokenHealer(cfg, vocab_list=vocab)
        # Token 3 ("x") is a proper prefix of "xyz" (id=2, but we need "xyz" in vocab)
        # Token 0 ("ab") is a proper prefix of "abc"
        healer.heal([3, 0], completions=[[1]])  # overlap = 1 ("ab")
        st = healer.get_stats()
        assert isinstance(st, HealerStats)
        assert st.total_heals >= 0
        assert st.avg_tokens_per_heal >= 0.0
