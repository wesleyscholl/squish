"""tests/test_wave14_server_wiring.py

Verifies that all Wave 14 module classes are importable and have the expected
public APIs that the server.py wiring code depends on.  These are pure
import + instantiation tests — no model or GPU required.

Wave 14 modules:
  soup_experts, vision_cache, vector_index, sub_spec, del_decoder,
  dfloat11, rans_codec, qspec, quant_spec, copy_spec,
  squeeze_llm, nf4_quant, spin_quant, hetero_vocab_sd, head_infer, life_model
"""

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# SoupOfExperts
# ---------------------------------------------------------------------------

class TestSoupOfExpertsWiring:
    def test_import(self):
        from squish.soup_experts import SoupOfExperts
        soe = SoupOfExperts(tolerance=0.01)
        assert soe is not None

    def test_register_expert(self):
        from squish.soup_experts import SoupOfExperts
        soe = SoupOfExperts()
        soe.register_expert("code", "/tmp/code.safetensors", default_weight=0.5)
        assert "code" in soe._experts

    def test_detect_domain(self):
        from squish.soup_experts import SoupOfExperts
        soe = SoupOfExperts()
        soe.register_expert("legal", "/tmp/legal.safetensors", default_weight=0.0)
        soe.register_expert("code", "/tmp/code.safetensors", default_weight=0.0)
        weights = soe.detect_domain("Write a Python function that parses JSON")
        assert isinstance(weights, dict)
        assert sum(weights.values()) == pytest.approx(1.0, abs=0.02)

    def test_set_mixing_weights_validates_sum(self):
        from squish.soup_experts import SoupOfExperts
        soe = SoupOfExperts(tolerance=0.01)
        soe.register_expert("a", "/tmp/a.st", 0.5)
        soe.register_expert("b", "/tmp/b.st", 0.5)
        with pytest.raises(ValueError):
            soe.set_mixing_weights({"a": 0.4, "b": 0.4})  # sum = 0.8 < 1.0


# ---------------------------------------------------------------------------
# VisionPrefixCache
# ---------------------------------------------------------------------------

class TestVisionPrefixCacheWiring:
    def test_import(self):
        from squish.vision_cache import VisionPrefixCache
        cache = VisionPrefixCache(max_entries=4)
        assert cache is not None

    def test_cache_miss_then_hit(self):
        from squish.vision_cache import VisionPrefixCache
        cache = VisionPrefixCache(max_entries=4)
        image_bytes = b"\x00" * 64

        encoded_calls = []

        def encoder(b):
            encoded_calls.append(True)
            return np.zeros(16)

        result1 = cache.get_or_encode(image_bytes, encoder)
        result2 = cache.get_or_encode(image_bytes, encoder)
        assert len(encoded_calls) == 1   # second call should be a cache hit
        assert np.allclose(result1, result2)

    def test_hit_rate_property(self):
        from squish.vision_cache import VisionPrefixCache
        cache = VisionPrefixCache(max_entries=4)
        assert 0.0 <= cache.hit_rate <= 1.0

    def test_lru_eviction(self):
        from squish.vision_cache import VisionPrefixCache
        cache = VisionPrefixCache(max_entries=2)
        for i in range(3):
            img = bytes([i] * 8)
            cache.get_or_encode(img, lambda b: np.array([float(b[0])]))
        # Cache held only 2 entries; size must not exceed max
        assert len(cache._cache) <= 2


# ---------------------------------------------------------------------------
# MRLIndex (vector_index)
# ---------------------------------------------------------------------------

class TestMRLIndexWiring:
    def test_import(self):
        from squish.vector_index import MRLIndex
        idx = MRLIndex(full_dim=64, coarse_dim=16)
        assert idx is not None

    def test_add_and_search(self):
        from squish.vector_index import MRLIndex
        rng = np.random.default_rng(0)
        idx = MRLIndex(full_dim=32, coarse_dim=8)
        vecs = rng.standard_normal((20, 32)).astype(np.float32)
        ids  = np.arange(20)
        idx.add(vecs, ids)
        query = rng.standard_normal(32).astype(np.float32)
        result_ids, dists = idx.search(query, top_k=5)
        assert len(result_ids) <= 5

    def test_empty_search_returns_empty(self):
        from squish.vector_index import MRLIndex
        idx = MRLIndex(full_dim=16, coarse_dim=8)
        rng = np.random.default_rng(1)
        ids, dists = idx.search(rng.standard_normal(16).astype(np.float32), top_k=3)
        assert len(ids) == 0

    def test_invalid_dims_raise(self):
        from squish.vector_index import MRLIndex
        with pytest.raises(ValueError):
            MRLIndex(full_dim=16, coarse_dim=32)  # coarse > full


# ---------------------------------------------------------------------------
# SubSpec
# ---------------------------------------------------------------------------

class TestSubSpecWiring:
    def test_import(self):
        from squish.sub_spec import SubSpecConfig, SubSpecDecoder
        cfg = SubSpecConfig(n_total_layers=8, n_gpu_layers=4, gamma=2)
        rng = np.random.default_rng(0)
        vocab = 16

        def draft_fn(prefix):
            return rng.standard_normal((len(prefix), vocab)).astype(np.float32)

        def target_fn(prefix):
            return rng.standard_normal((len(prefix), vocab)).astype(np.float32)

        dec = SubSpecDecoder(draft_fn, target_fn, cfg)
        assert dec is not None

    def test_config_defaults(self):
        from squish.sub_spec import SubSpecConfig
        cfg = SubSpecConfig()
        assert cfg.n_total_layers >= cfg.n_gpu_layers
        assert cfg.gamma >= 1
        assert cfg.quant_bits in (2, 4, 8)

    def test_substitute_layer_proxy(self):
        from squish.sub_spec import SubstituteLayerProxy
        rng = np.random.default_rng(0)
        W   = rng.standard_normal((8, 8)).astype(np.float32)
        proxy = SubstituteLayerProxy(W)
        x_in  = rng.standard_normal(8).astype(np.float32)
        out   = proxy.forward(x_in)
        assert out.shape == (8,)

    def test_generate_returns_tokens(self):
        from squish.sub_spec import SubSpecConfig, SubSpecDecoder
        cfg = SubSpecConfig(n_total_layers=4, n_gpu_layers=2, gamma=2)
        rng = np.random.default_rng(2)
        vocab = 16

        def draft_fn(prefix):
            return rng.standard_normal((len(prefix), vocab)).astype(np.float32)

        def target_fn(prefix):
            return rng.standard_normal((len(prefix), vocab)).astype(np.float32)

        dec    = SubSpecDecoder(draft_fn, target_fn, cfg)
        tokens = dec.generate(prompt=[1, 2, 3], max_new_tokens=4)
        assert len(tokens) > 0


# ---------------------------------------------------------------------------
# DELDecoder (Dynamic Exit Layer)
# ---------------------------------------------------------------------------

class TestDELDecoderWiring:
    def test_import(self):
        from squish.del_decoder import DELConfig, DELDecoder
        cfg  = DELConfig(num_layers=8, min_exit_layer=2, max_exit_layer=6, gamma=3)
        rng  = np.random.default_rng(0)
        vocab = 16

        def forward_fn(token_ids, exit_layer=None):
            return rng.standard_normal((len(token_ids), vocab)).astype(np.float32)

        dec = DELDecoder(forward_fn, cfg)
        assert dec is not None

    def test_config_defaults(self):
        from squish.del_decoder import DELConfig
        cfg = DELConfig()
        assert cfg.num_layers >= 2
        assert cfg.min_exit_layer >= 1
        assert cfg.max_exit_layer <= cfg.num_layers
        assert cfg.gamma >= 1

    def test_config_rejects_bad_exit_range(self):
        from squish.del_decoder import DELConfig
        with pytest.raises(ValueError):
            DELConfig(num_layers=8, min_exit_layer=7, max_exit_layer=4)

    def test_generate_returns_tokens(self):
        from squish.del_decoder import DELConfig, DELDecoder
        cfg  = DELConfig(num_layers=8, min_exit_layer=2, max_exit_layer=6, gamma=2)
        rng  = np.random.default_rng(3)
        vocab = 16

        def forward_fn(token_ids, exit_layer=None):
            return rng.standard_normal((len(token_ids), vocab)).astype(np.float32)

        dec    = DELDecoder(forward_fn, cfg)
        tokens = dec.generate(prompt=[1, 2, 3], max_new_tokens=4)
        assert len(tokens) > 0


# ---------------------------------------------------------------------------
# DFloat11
# ---------------------------------------------------------------------------

class TestDFloat11Wiring:
    def test_import(self):
        from squish.dfloat11 import DFloat11Config, DFloat11Compressor
        cfg  = DFloat11Config(block_size=64)
        comp = DFloat11Compressor(cfg)
        assert comp is not None

    def test_config_defaults(self):
        from squish.dfloat11 import DFloat11Config
        cfg = DFloat11Config()
        assert cfg.block_size >= 1
        assert cfg.min_symbol_freq >= 1

    def test_compress_decompress_roundtrip(self):
        from squish.dfloat11 import DFloat11Config, DFloat11Compressor
        rng = np.random.default_rng(0)
        cfg  = DFloat11Config(block_size=64)
        comp = DFloat11Compressor(cfg)
        weights = rng.standard_normal(256).astype(np.float16)
        compressed = comp.compress(weights)
        restored   = comp.decompress(compressed)
        assert restored.dtype == np.float16
        assert restored.shape == weights.shape


# ---------------------------------------------------------------------------
# RANSCodec
# ---------------------------------------------------------------------------

class TestRANSCodecWiring:
    def test_import(self):
        from squish.rans_codec import RANSCodec
        freq  = {0: 100, 1: 80, 2: 60}
        codec = RANSCodec(freq=freq)
        assert codec is not None

    def test_encode_decode_roundtrip(self):
        from squish.rans_codec import RANSCodec
        rng  = np.random.default_rng(0)
        data = [int(x % 4) for x in rng.integers(0, 4, 64)]
        freq = {0: 30, 1: 25, 2: 20, 3: 25}
        codec = RANSCodec(freq=freq)
        state   = codec.encode(data)
        decoded = codec.decode(state, len(data))
        assert decoded == data

    def test_empty_freq_creates_codec(self):
        from squish.rans_codec import RANSCodec
        codec = RANSCodec(freq={})
        assert codec is not None


# ---------------------------------------------------------------------------
# QSpec
# ---------------------------------------------------------------------------

class TestQSpecWiring:
    def test_import(self):
        from squish.qspec import QSpecConfig, QSpecDecoder
        cfg  = QSpecConfig(gamma=2, draft_act_bits=8, verify_act_bits=16)
        rng  = np.random.default_rng(0)
        vocab = 16

        def w4a8_fn(token_ids):
            return rng.standard_normal((len(token_ids), vocab)).astype(np.float32)

        def w4a16_fn(token_ids):
            return rng.standard_normal((len(token_ids), vocab)).astype(np.float32)

        dec = QSpecDecoder(w4a8_fn, w4a16_fn, cfg)
        assert dec is not None

    def test_config_defaults(self):
        from squish.qspec import QSpecConfig
        cfg = QSpecConfig()
        assert cfg.gamma >= 1
        assert cfg.draft_act_bits < cfg.verify_act_bits

    def test_config_rejects_inverted_bits(self):
        from squish.qspec import QSpecConfig
        with pytest.raises(ValueError):
            QSpecConfig(draft_act_bits=8, verify_act_bits=8)

    def test_generate_returns_tokens(self):
        from squish.qspec import QSpecConfig, QSpecDecoder
        cfg  = QSpecConfig(gamma=2)
        rng  = np.random.default_rng(4)
        vocab = 16

        def w4a8_fn(token_ids):
            return rng.standard_normal((len(token_ids), vocab)).astype(np.float32)

        def w4a16_fn(token_ids):
            return rng.standard_normal((len(token_ids), vocab)).astype(np.float32)

        dec    = QSpecDecoder(w4a8_fn, w4a16_fn, cfg)
        tokens = dec.generate(prompt=[1, 2], max_new_tokens=4)
        assert len(tokens) > 0


# ---------------------------------------------------------------------------
# QuantSpec
# ---------------------------------------------------------------------------

class TestQuantSpecWiring:
    def test_import(self):
        from squish.quant_spec import QuantSpecConfig, QuantSpecDecoder
        cfg  = QuantSpecConfig(gamma=2, draft_quant_bits=4, draft_skip_layers=2)
        rng  = np.random.default_rng(0)
        vocab = 16

        def draft_fn(token_ids):
            return rng.standard_normal((len(token_ids), vocab)).astype(np.float32)

        dec = QuantSpecDecoder(draft_fn, cfg)
        assert dec is not None

    def test_config_defaults(self):
        from squish.quant_spec import QuantSpecConfig
        cfg = QuantSpecConfig()
        assert cfg.gamma >= 1
        assert cfg.draft_quant_bits in (2, 4, 8)
        assert cfg.draft_skip_layers >= 0

    def test_generate_returns_tokens(self):
        from squish.quant_spec import QuantSpecConfig, QuantSpecDecoder
        cfg  = QuantSpecConfig(gamma=2)
        rng  = np.random.default_rng(5)
        vocab = 16

        def draft_fn(token_ids):
            return rng.standard_normal((len(token_ids), vocab)).astype(np.float32)

        dec    = QuantSpecDecoder(draft_fn, cfg)
        tokens = dec.generate(prompt=[1, 2], max_new_tokens=4)
        assert len(tokens) > 0


# ---------------------------------------------------------------------------
# CopySpec
# ---------------------------------------------------------------------------

class TestCopySpecWiring:
    def test_import(self):
        from squish.copy_spec import CopySpecConfig, CopySpecDrafter
        cfg    = CopySpecConfig(min_match_len=2, max_draft_len=4)
        drafter = CopySpecDrafter(cfg)
        assert drafter is not None

    def test_config_defaults(self):
        from squish.copy_spec import CopySpecConfig
        cfg = CopySpecConfig()
        assert cfg.min_match_len >= 1
        assert cfg.max_draft_len >= 1
        assert cfg.max_history_len >= cfg.min_match_len

    def test_propose_copies_from_history(self):
        from squish.copy_spec import CopySpecConfig, CopySpecDrafter
        cfg = CopySpecConfig(min_match_len=2, max_draft_len=4)
        d   = CopySpecDrafter(cfg)
        # Feed a repetitive sequence into history
        history = [1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2]
        for i in range(0, len(history) - 1):
            d.extend_history(history[:i+1])
        draft = d.propose(context=history[-2:])
        assert isinstance(draft, list)

    def test_stats_initial_zeros(self):
        from squish.copy_spec import CopySpecConfig, CopySpecDrafter
        d = CopySpecDrafter(CopySpecConfig())
        st = d.stats()
        assert st.total_steps == 0


# ---------------------------------------------------------------------------
# SqueezeLLM
# ---------------------------------------------------------------------------

class TestSqueezeLLMWiring:
    def test_import(self):
        from squish.squeeze_llm import SqueezeLLMConfig, SqueezeLLMQuantizer
        cfg   = SqueezeLLMConfig(quant_bits=4)
        quant = SqueezeLLMQuantizer(cfg)
        assert quant is not None

    def test_config_defaults(self):
        from squish.squeeze_llm import SqueezeLLMConfig
        cfg = SqueezeLLMConfig()
        assert cfg.quant_bits in (2, 3, 4)
        assert 0.0 <= cfg.sparsity_ratio < 1.0

    def test_quantize_returns_layer(self):
        from squish.squeeze_llm import SqueezeLLMConfig, SqueezeLLMQuantizer, SqueezeLLMLayer
        rng   = np.random.default_rng(0)
        cfg   = SqueezeLLMConfig(quant_bits=4, sparsity_ratio=0.01, n_fit_iters=2)
        quant = SqueezeLLMQuantizer(cfg)
        W     = rng.standard_normal((32, 32)).astype(np.float32)
        layer = quant.quantize(W)
        assert isinstance(layer, SqueezeLLMLayer)
        x_in  = rng.standard_normal(32).astype(np.float32)
        out   = layer.forward(x_in)
        assert out.shape == (32,)


# ---------------------------------------------------------------------------
# NF4 quantization
# ---------------------------------------------------------------------------

class TestNF4QuantWiring:
    def test_import(self):
        from squish.nf4_quant import NF4_LEVELS, quantize_nf4, dequantize_nf4
        assert len(NF4_LEVELS) == 16

    def test_nf4_levels_sorted(self):
        from squish.nf4_quant import NF4_LEVELS
        assert np.all(np.diff(NF4_LEVELS) > 0), "NF4_LEVELS must be strictly ascending"

    def test_quantize_returns_int_array(self):
        from squish.nf4_quant import quantize_nf4
        rng = np.random.default_rng(0)
        W   = rng.standard_normal((8, 8)).astype(np.float32)
        q, scales = quantize_nf4(W)
        assert q.dtype in (np.uint8, np.int8, np.uint16, np.int16, np.int32, np.uint32)
        assert scales is not None

    def test_roundtrip_error_small(self):
        from squish.nf4_quant import quantize_nf4, dequantize_nf4
        rng = np.random.default_rng(1)
        W   = rng.standard_normal((16, 16)).astype(np.float32)
        q, scales = quantize_nf4(W)
        restored  = dequantize_nf4(q, scales, W.shape)
        mse = float(np.mean((W - restored) ** 2))
        assert mse < 1.0, f"NF4 roundtrip MSE too large: {mse}"


# ---------------------------------------------------------------------------
# SpinQuant
# ---------------------------------------------------------------------------

class TestSpinQuantWiring:
    def test_import(self):
        from squish.spin_quant import run_rotation
        assert callable(run_rotation)

    def test_rotation_calibration(self):
        from squish.spin_quant import SpinQuantCalibrator
        rng = np.random.default_rng(0)
        W   = rng.standard_normal((8, 8)).astype(np.float32)
        cal = SpinQuantCalibrator(dim=8, seed=42)
        R_init = cal.rotation_matrix
        assert R_init.shape == (8, 8)
        # R should be orthogonal
        prod = R_init @ R_init.T
        assert np.allclose(prod, np.eye(8), atol=1e-5)

    def test_rotation_step_preserves_orthogonality(self):
        from squish.spin_quant import SpinQuantCalibrator
        rng = np.random.default_rng(1)
        W   = rng.standard_normal((8, 8)).astype(np.float32)
        cal = SpinQuantCalibrator(dim=8)
        for _ in range(5):
            cal.step(W, lr=0.01)
        R = cal.rotation_matrix
        prod = R @ R.T
        assert np.allclose(prod, np.eye(8), atol=1e-4)


# ---------------------------------------------------------------------------
# HeteroVocabSD
# ---------------------------------------------------------------------------

class TestHeteroVocabSDWiring:
    def test_import(self):
        from squish.hetero_vocab_sd import HeteroVocabConfig, HeteroVocabDecoder
        cfg  = HeteroVocabConfig(gamma=2, draft_vocab_size=16, target_vocab_size=32)
        rng  = np.random.default_rng(0)

        def drafter_fn(token_ids):
            return rng.standard_normal((len(token_ids), 16)).astype(np.float32)

        def target_fn(token_ids):
            return rng.standard_normal((len(token_ids), 32)).astype(np.float32)

        # Build a trivial token map {i: i} for the first 16 tokens
        token_map = {i: i for i in range(16)}

        dec = HeteroVocabDecoder(drafter_fn, target_fn, cfg, token_map=token_map)
        assert dec is not None

    def test_config_defaults(self):
        from squish.hetero_vocab_sd import HeteroVocabConfig
        cfg = HeteroVocabConfig()
        assert cfg.gamma >= 1
        assert cfg.draft_vocab_size >= 2
        assert cfg.target_vocab_size >= 2

    def test_generate_returns_tokens(self):
        from squish.hetero_vocab_sd import HeteroVocabConfig, HeteroVocabDecoder
        cfg  = HeteroVocabConfig(gamma=2, draft_vocab_size=16, target_vocab_size=16)
        rng  = np.random.default_rng(6)

        def drafter_fn(token_ids):
            return rng.standard_normal((len(token_ids), 16)).astype(np.float32)

        def target_fn(token_ids):
            return rng.standard_normal((len(token_ids), 16)).astype(np.float32)

        dec    = HeteroVocabDecoder(drafter_fn, target_fn, cfg)
        tokens = dec.generate(prompt=[1, 2], max_new_tokens=4)
        assert len(tokens) > 0


# ---------------------------------------------------------------------------
# HeadInfer
# ---------------------------------------------------------------------------

class TestHeadInferWiring:
    def test_import(self):
        from squish.head_infer import HeadInferConfig, HeadAwareKVStore
        cfg   = HeadInferConfig(n_layers=4, n_heads=4, window_size=32)
        store = HeadAwareKVStore(cfg, head_dim=16)
        assert store is not None

    def test_config_defaults(self):
        from squish.head_infer import HeadInferConfig
        cfg = HeadInferConfig()
        assert cfg.n_layers >= 1
        assert cfg.n_heads >= 1
        assert cfg.window_size >= 1
        assert 0.0 < cfg.retrieval_threshold < 1.0

    def test_put_and_get(self):
        from squish.head_infer import HeadInferConfig, HeadAwareKVStore
        rng   = np.random.default_rng(0)
        cfg   = HeadInferConfig(n_layers=2, n_heads=2, window_size=8)
        store = HeadAwareKVStore(cfg, head_dim=8)
        k = rng.standard_normal(8).astype(np.float32)
        v = rng.standard_normal(8).astype(np.float32)
        store.put(layer=0, head=0, key=k, value=v)
        keys, vals = store.get(layer=0, head=0)
        assert keys.shape[-1] == 8

    def test_classifier_import(self):
        from squish.head_infer import HeadInferConfig, HeadClassifier
        cfg = HeadInferConfig(n_layers=2, n_heads=4)
        clf = HeadClassifier(cfg)
        arr = clf.to_labels_array()
        assert arr.shape == (2, 4)


# ---------------------------------------------------------------------------
# LifeModel (LIFE estimator)
# ---------------------------------------------------------------------------

class TestLifeModelWiring:
    def test_import(self):
        from squish.life_model import predict
        assert callable(predict)

    def test_predict_returns_dict(self):
        from squish.life_model import predict
        result = predict(model_dir=None, batch_size=1, seq_len=512, output_len=128)
        assert isinstance(result, dict)

    def test_predict_has_expected_keys(self):
        from squish.life_model import predict
        result = predict(model_dir=None, batch_size=1, seq_len=512, output_len=128)
        assert "ttft_s" in result or "tpot_ms" in result or "tokens_per_s" in result

    def test_predict_positive_throughput(self):
        from squish.life_model import predict
        result = predict(model_dir=None, batch_size=1, seq_len=128, output_len=64)
        # At least one throughput-like key should be positive
        throughput_keys = [k for k in result if "tok" in k.lower() or "throughput" in k.lower()]
        if throughput_keys:
            assert result[throughput_keys[0]] > 0


# ---------------------------------------------------------------------------
# Integration — all Wave 14 modules are importable
# ---------------------------------------------------------------------------

class TestWave14AllImportable:
    """Smoke test: verify every Wave 14 module can be imported."""

    @pytest.mark.parametrize("module,symbols", [
        ("soup_experts",   ["SoupOfExperts"]),
        ("vision_cache",   ["VisionPrefixCache"]),
        ("vector_index",   ["MRLIndex"]),
        ("sub_spec",       ["SubSpecConfig", "SubSpecDecoder"]),
        ("del_decoder",    ["DELConfig", "DELDecoder"]),
        ("dfloat11",       ["DFloat11Config", "DFloat11Compressor"]),
        ("rans_codec",     ["RANSCodec"]),
        ("qspec",          ["QSpecConfig", "QSpecDecoder"]),
        ("quant_spec",     ["QuantSpecConfig", "QuantSpecDecoder"]),
        ("copy_spec",      ["CopySpecConfig", "CopySpecDrafter"]),
        ("squeeze_llm",    ["SqueezeLLMConfig", "SqueezeLLMQuantizer"]),
        ("nf4_quant",      ["NF4_LEVELS", "quantize_nf4", "dequantize_nf4"]),
        ("spin_quant",     ["run_rotation"]),
        ("hetero_vocab_sd",["HeteroVocabConfig", "HeteroVocabDecoder"]),
        ("head_infer",     ["HeadInferConfig", "HeadAwareKVStore"]),
        ("life_model",     ["predict"]),
    ])
    def test_module_importable(self, module, symbols):
        import importlib
        mod = importlib.import_module(f"squish.{module}")
        for sym in symbols:
            assert hasattr(mod, sym), f"squish.{module} missing {sym!r}"
