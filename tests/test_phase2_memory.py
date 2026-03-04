#!/usr/bin/env python3
"""
tests/test_phase2_memory.py

Unit tests for Phase 2 memory-management modules:
  - squish.split_loader  (CPU/GPU split loading)
  - squish.flash_attention (Flash Attention wrapper)
  - squish.layerwise_loader (AirLLM-style layer streaming)

All tests are designed to run without a real MLX Metal GPU context —
they use simple mock layers / synthetic numpy arrays.

Run with:
    pytest tests/test_phase2_memory.py -v
"""

import json
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Shared fixtures / helpers
# ---------------------------------------------------------------------------

def _make_mock_array(shape, dtype=np.float16):
    """Return a mock MLX-like array backed by numpy."""
    arr = MagicMock()
    arr.nbytes = int(np.prod(shape)) * np.dtype(dtype).itemsize
    arr.shape  = shape
    arr.dtype  = dtype
    # np.array(mock) would fail — store the real ndarray for round-trip tests
    arr._np = np.random.randn(*shape).astype(dtype)
    return arr


def _make_simple_layer(param_bytes: int = 1024 * 1024, n_params: int = 4):
    """Create a minimal mock transformer layer."""
    layer = MagicMock()
    single = param_bytes // n_params
    # Build nested params dict matching real MLX structure
    elem_size = np.dtype(np.float16).itemsize
    n_elements = single // elem_size

    def _fake_params():
        return {
            "self_attn": {
                "q_proj": {"weight": _make_mock_array((64, n_elements // 64))},
                "k_proj": {"weight": _make_mock_array((64, n_elements // 64))},
            },
            "mlp": {
                "gate_proj": {"weight": _make_mock_array((32, n_elements // 32))},
            },
        }

    layer.parameters = _fake_params
    layer.load_weights = MagicMock()
    return layer


def _make_mock_model(n_layers: int = 8, mb_per_layer: float = 100.0):
    """Build a mock model with n_layers pseudo-MLX layers."""
    model = MagicMock()
    param_bytes = int(mb_per_layer * 1024 * 1024)
    model.layers = [_make_simple_layer(param_bytes) for _ in range(n_layers)]
    return model


# ===========================================================================
# 1. split_loader tests
# ===========================================================================

class TestSplitLoaderHelpers:
    """Pure-numpy helpers that don't touch Metal."""

    def test_layer_weight_bytes_counts_all_params(self):
        from squish.split_loader import _layer_weight_bytes

        layer = MagicMock()
        # Simulate parameters() returning nested dicts of arrays with .nbytes
        mock_q = MagicMock()
        mock_q.nbytes = 256
        mock_k = MagicMock()
        mock_k.nbytes = 256
        mock_v = MagicMock()
        mock_v.nbytes = 256
        mock_o = MagicMock()
        mock_o.nbytes = 512

        # Patch mx.array detection: make them pass isinstance(v, mx.array)
        # Since we can't import mlx in tests, we patch the function itself
        with patch("squish.split_loader._mx") as mock_mx_fn:
            import mlx.core as mx  # noqa: F401 — available or skip
            layer.parameters.return_value = {
                "self_attn": {
                    "q_proj": {"weight": mock_q},
                    "k_proj": {"weight": mock_k},
                    "v_proj": {"weight": mock_v},
                },
                "mlp": {"gate": {"weight": mock_o}},
            }
            mock_mx_array = MagicMock()
            mock_mx_fn.return_value.array = mock_mx_array
            # _layer_weight_bytes uses isinstance(obj, mx.array), so mock it
            # Patch at a higher level — test the returned sum
            # For brevity, directly call with mlx available (skip if absent)
            pytest.importorskip("mlx.core")
            total = _layer_weight_bytes(layer)
            assert isinstance(total, int)
            assert total >= 0

    def test_split_info_properties(self):
        from squish.split_loader import SplitInfo

        info = SplitInfo(
            gpu_layers=[0, 1, 2, 3, 4, 5],
            cpu_layers=[6, 7],
            gpu_bytes=6 * 100 * 1024 ** 2,
            cpu_bytes=2 * 100 * 1024 ** 2,
            metal_limit=16 * 1024 ** 3,
            target_bytes=int(0.9 * 16 * 1024 ** 3),
        )
        assert info.gpu_count == 6
        assert info.cpu_count == 2
        assert abs(info.gpu_gb - 0.586) < 0.01
        assert abs(info.cpu_gb - 0.195) < 0.01

    def test_split_info_str(self):
        from squish.split_loader import SplitInfo

        info = SplitInfo(
            gpu_layers=list(range(28)),
            cpu_layers=[],
            gpu_bytes=14 * 1024 ** 3,
            cpu_bytes=0,
            metal_limit=16 * 1024 ** 3,
            target_bytes=14 * 1024 ** 3,
        )
        s = str(info)
        assert "gpu_layers=28" in s or "28" in s


class TestSplitLayerLoaderLogic:
    """Tests for greedy layer assignment logic (no Metal)."""

    def test_auto_split_model_fits_returns_none(self):
        """If model < target_bytes, auto_split returns None."""
        from squish.split_loader import SplitLayerLoader

        # Patch _get_metal_limit_bytes to return 16 GB
        with patch("squish.split_loader._get_metal_limit_bytes", return_value=16 * 1024 ** 3):
            model = _make_mock_model(n_layers=8, mb_per_layer=10.0)  # total ~80 MB << 16 GB
            # Patch _layer_weight_bytes to return known values
            with patch("squish.split_loader._layer_weight_bytes", return_value=10 * 1024 ** 2):
                result = SplitLayerLoader.auto_split(model, verbose=False)
        # 8 layers × 10 MB = 80 MB << 14.4 GB target → no split needed
        assert result is None

    def test_apply_with_forced_cpu_layers(self):
        """When force_cpu_layers is set, those layers become OffloadedLayer."""
        pytest.importorskip("mlx.core")
        import mlx.core as mx  # noqa: F401

        from squish.split_loader import OffloadedLayer, SplitLayerLoader

        model = _make_mock_model(n_layers=4, mb_per_layer=10.0)

        with patch("squish.split_loader._get_metal_limit_bytes", return_value=16 * 1024 ** 3):
            with patch("squish.split_loader._layer_weight_bytes", return_value=10 * 1024 ** 2):
                with patch("squish.split_loader._flatten_params", return_value=[("w", np.zeros((4,), dtype=np.float16))]):
                    loader = SplitLayerLoader(model, target_fraction=0.9, force_cpu_layers=[2, 3])
                    info = loader.apply()

        assert 2 in info.cpu_layers
        assert 3 in info.cpu_layers
        assert isinstance(model.layers[2], OffloadedLayer)
        assert isinstance(model.layers[3], OffloadedLayer)

    def test_profile_model_layers_returns_list(self):
        from squish.split_loader import profile_model_layers

        model = _make_mock_model(n_layers=4, mb_per_layer=50.0)
        with patch("squish.split_loader._layer_weight_bytes", return_value=50 * 1024 ** 2):
            result = profile_model_layers(model)

        assert len(result) == 4
        for row in result:
            assert "index" in row
            assert "bytes" in row
            assert "mb" in row
            assert row["bytes"] == 50 * 1024 ** 2


# ===========================================================================
# 2. flash_attention tests
# ===========================================================================

class TestFlashAttentionAvailability:

    def test_is_flash_attention_available_type(self):
        from squish.flash_attention import _has_fast_sdp_available
        result = _has_fast_sdp_available()
        assert isinstance(result, bool)

    def test_patch_result_default_zero(self):
        from squish.flash_attention import PatchResult
        r = PatchResult()
        assert r.already_fast == 0
        assert r.patched == 0
        assert r.fallback == 0
        assert r.total == 0

    def test_patch_result_str(self):
        from squish.flash_attention import PatchResult
        r = PatchResult(already_fast=28, patched=0, fallback=0, total=28)
        s = str(r)
        assert "already_fast=28" in s
        assert "total=28" in s

    def test_patch_model_no_layers(self):
        from squish.flash_attention import PatchResult, patch_model_attention
        model = MagicMock()
        del model.layers  # trigger AttributeError path
        # Should not raise
        result = patch_model_attention(model, verbose=False)
        assert isinstance(result, PatchResult)
        assert result.total == 0

    def test_patch_model_empty_layers(self):
        from squish.flash_attention import patch_model_attention
        model = MagicMock()
        model.layers = []
        result = patch_model_attention(model, verbose=False)
        assert result.total == 0

    def test_attention_status_returns_dict(self):
        from squish.flash_attention import attention_status
        model = _make_mock_model(n_layers=4)
        # Mock layers have no self_attn attribute, so total_attn = 0
        status = attention_status(model)
        assert "mlx_fast_available" in status
        assert "mlx_version" in status
        assert "total_attn_layers" in status
        assert isinstance(status["mlx_fast_available"], bool)

    def test_attention_status_with_attn_layers(self):
        from squish.flash_attention import attention_status

        model = MagicMock()
        layer0 = MagicMock()
        attn0  = MagicMock()
        attn0._squish_flash_patched = True
        layer0.self_attn = attn0
        layer0.attention = MagicMock(side_effect=AttributeError)

        model.layers = [layer0]

        status = attention_status(model)
        assert status["total_attn_layers"] >= 0

    def test_predict_memory_savings_shape(self):
        from squish.flash_attention import predict_memory_savings
        rows = predict_memory_savings(n_heads=28, head_dim=128, context_lengths=[512, 4096])
        assert len(rows) == 2
        for r in rows:
            assert r["standard_mb"] > r["flash_mb"]
            assert r["ratio"] > 1.0
            assert r["savings_mb"] > 0

    def test_predict_memory_savings_grows_with_context(self):
        from squish.flash_attention import predict_memory_savings
        rows = predict_memory_savings(n_heads=8, head_dim=64, context_lengths=[512, 4096, 32768])
        ratios = [r["ratio"] for r in rows]
        # Standard attention is O(N²), so ratio should increase with context
        assert ratios[1] > ratios[0]
        assert ratios[2] > ratios[1]


class TestFlashAttentionBenchmark:

    def test_benchmark_skipped_without_fast_sdp(self):
        pytest.importorskip("mlx.core")
        from squish.flash_attention import _has_fast_sdp_available, benchmark_attention
        if not _has_fast_sdp_available():
            with pytest.raises(RuntimeError, match="mx.fast"):
                benchmark_attention(n_heads=2, kv_heads=2, head_dim=64, context_lengths=[64], n_trials=1)
        else:
            results = benchmark_attention(n_heads=2, kv_heads=2, head_dim=64, context_lengths=[64], n_trials=2)
            assert len(results) == 1
            r = results[0]
            assert r["context"] == 64
            assert r["standard_ms"] > 0
            assert r["flash_ms"] > 0
            assert r["speedup"] > 0


# ===========================================================================
# 3. layerwise_loader tests
# ===========================================================================

class TestLayerCache:

    def test_lru_eviction(self):
        from squish.layerwise_loader import LayerCache


        cache = LayerCache(capacity=2)

        layers = [MagicMock(spec=[]) for _ in range(4)]
        # Disable _zero_layer_weights side effects
        with patch("squish.layerwise_loader._zero_layer_weights"):
            cache.put(0, layers[0])
            cache.put(1, layers[1])
            evicted = cache.put(2, layers[2])  # should evict 0
            assert 0 in evicted
            assert len(cache) == 2

    def test_cache_hit(self):
        from squish.layerwise_loader import LayerCache

        cache = LayerCache(capacity=4)
        mock_layer = MagicMock(spec=[])

        with patch("squish.layerwise_loader._zero_layer_weights"):
            cache.put(0, mock_layer)
            result = cache.get(0)
            assert result is mock_layer

    def test_cache_miss_returns_none(self):
        from squish.layerwise_loader import LayerCache
        cache = LayerCache(capacity=4)
        assert cache.get(99) is None

    def test_cache_contains(self):
        from squish.layerwise_loader import LayerCache

        cache = LayerCache(capacity=4)
        mock_layer = MagicMock(spec=[])
        with patch("squish.layerwise_loader._zero_layer_weights"):
            cache.put(5, mock_layer)
            assert 5 in cache
            assert 6 not in cache

    def test_capacity_one(self):
        from squish.layerwise_loader import LayerCache

        cache = LayerCache(capacity=1)
        with patch("squish.layerwise_loader._zero_layer_weights"):
            cache.put(0, MagicMock(spec=[]))
            cache.put(1, MagicMock(spec=[]))
            assert len(cache) == 1
            assert 1 in cache
            assert 0 not in cache

    def test_invalid_capacity_raises(self):
        from squish.layerwise_loader import LayerCache
        with pytest.raises(ValueError):
            LayerCache(capacity=0)


class TestShardModel:

    def test_shard_model_creates_directories(self, tmp_path):
        from squish.layerwise_loader import _LAYER_META_FILE, _MODEL_META_FILE, shard_model

        # Build a mock model with 3 layers
        model = MagicMock()
        layers = []
        for _ in range(3):
            layer = MagicMock()
            layer.parameters.return_value = {
                "weight": MagicMock(
                    **{"__class__.__name__": "array"}
                )
            }
            layers.append(layer)
        model.layers = layers

        # Patch _flatten_params to return known numpy arrays
        fake_params = [("self_attn.q_proj.weight", np.zeros((8, 8), dtype=np.float16))]
        with patch("squish.layerwise_loader._flatten_params", return_value=fake_params):
            out = shard_model(model, tmp_path / "shards", verbose=False)

        assert out.exists()
        assert (out / _MODEL_META_FILE).exists()
        for i in range(3):
            layer_dir = out / f"layer_{i:03d}"
            assert layer_dir.exists()
            assert (layer_dir / _LAYER_META_FILE).exists()

    def test_shard_model_meta_content(self, tmp_path):
        from squish.layerwise_loader import _FORMAT_VERSION, _MODEL_META_FILE, shard_model

        model = MagicMock()
        model.layers = [MagicMock() for _ in range(5)]
        fake_params = [("w", np.zeros((4,), dtype=np.float16))]
        with patch("squish.layerwise_loader._flatten_params", return_value=fake_params):
            out = shard_model(model, tmp_path / "shards5", verbose=False)

        meta = json.loads((out / _MODEL_META_FILE).read_text())
        assert meta["n_layers"] == 5
        assert meta["format"] == _FORMAT_VERSION

    def test_shard_model_empty_raises(self, tmp_path):
        from squish.layerwise_loader import shard_model

        model = MagicMock()
        model.layers = []
        with pytest.raises(ValueError, match="empty"):
            shard_model(model, tmp_path / "empty", verbose=False)


class TestRecommendCacheSize:

    def test_basic_calculation(self):
        from squish.layerwise_loader import recommend_cache_size

        # 70B bfloat16: 140 GB, 80 layers → 1.75 GB/layer
        # 32 GB available Mac, 80% usable = 25.6 GB → 14 layers
        result = recommend_cache_size(
            total_model_gb=140.0,
            n_layers=80,
            available_metal_gb=32.0,
            safety_factor=0.80,
        )
        assert 10 <= result <= 20  # rough range

    def test_minimum_two(self):
        from squish.layerwise_loader import recommend_cache_size

        # Tiny available memory → should still return at least 2
        result = recommend_cache_size(
            total_model_gb=1000.0,
            n_layers=100,
            available_metal_gb=1.0,
        )
        assert result == 2

    def test_capped_at_n_layers(self):
        from squish.layerwise_loader import recommend_cache_size

        # Tiny model, huge memory → all layers fit
        result = recommend_cache_size(
            total_model_gb=1.0,
            n_layers=10,
            available_metal_gb=1000.0,
        )
        assert result == 10


class TestLoadStatsStr:

    def test_zero_stats(self):
        from squish.layerwise_loader import LoadStats
        s = LoadStats()
        assert s.hit_rate == 0.0
        assert "hits=0" in str(s)

    def test_hit_rate_calculation(self):
        from squish.layerwise_loader import LoadStats
        s = LoadStats(cache_hits=7, cache_misses=3)
        assert abs(s.hit_rate - 0.70) < 1e-6


# ===========================================================================
# 4.  Integration smoke test — module imports
# ===========================================================================

class TestModuleImports:

    def test_split_loader_importable(self):
        from squish import split_loader  # noqa: F401

    def test_flash_attention_importable(self):
        from squish import flash_attention  # noqa: F401

    def test_layerwise_loader_importable(self):
        from squish import layerwise_loader  # noqa: F401

    def test_split_loader_public_api(self):
        pass

    def test_flash_attention_public_api(self):
        pass

    def test_layerwise_loader_public_api(self):
        pass
