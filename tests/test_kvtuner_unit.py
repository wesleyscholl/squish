"""Unit tests for squish.kvtuner — Sensitivity-Aware Mixed Precision KV Quantization."""

import json
import os
import tempfile
import pytest
import numpy as np
from squish.kvtuner import (
    KVTunerConfig,
    ALLOWED_BITS,
    LayerSensitivity,
    KVTunerCalibrator,
    KVQuantConfig,
    KVTunerStats,
    _simulate_quantization_error,
)


# ---------------------------------------------------------------------------
# TestKVTunerConfig
# ---------------------------------------------------------------------------

class TestKVTunerConfig:
    def test_defaults(self):
        cfg = KVTunerConfig()
        assert cfg.n_layers == 32
        assert cfg.target_avg_bits == 4.0
        assert cfg.sensitivity_metric == "mse"

    def test_invalid_n_layers(self):
        with pytest.raises(ValueError):
            KVTunerConfig(n_layers=0)

    def test_invalid_candidate_bits(self):
        with pytest.raises(ValueError):
            KVTunerConfig(candidate_bits=(1, 7))

    def test_invalid_target_avg_bits(self):
        with pytest.raises(ValueError):
            KVTunerConfig(target_avg_bits=0.0)

    def test_invalid_sensitivity_metric(self):
        with pytest.raises(ValueError):
            KVTunerConfig(sensitivity_metric="unknown")

    def test_invalid_key_priority(self):
        with pytest.raises(ValueError):
            KVTunerConfig(key_priority=0.0)


# ---------------------------------------------------------------------------
# TestSimulateQuantizationError
# ---------------------------------------------------------------------------

class TestSimulateQuantizationError:
    def test_mse_higher_at_lower_bits(self):
        rng = np.random.default_rng(0)
        tensor = rng.standard_normal(256).astype(np.float32)
        err2 = _simulate_quantization_error(tensor, 2, "mse")
        err8 = _simulate_quantization_error(tensor, 8, "mse")
        assert err2 > err8

    def test_max_abs_nonnegative(self):
        tensor = np.linspace(-1, 1, 64).astype(np.float32)
        err = _simulate_quantization_error(tensor, 4, "max_abs")
        assert err >= 0.0

    def test_cosine_in_range(self):
        tensor = np.linspace(-1, 1, 64).astype(np.float32)
        err = _simulate_quantization_error(tensor, 4, "cosine")
        assert 0.0 <= err <= 2.0

    def test_empty_tensor_returns_zero(self):
        err = _simulate_quantization_error(np.array([]), 4, "mse")
        assert err == 0.0


# ---------------------------------------------------------------------------
# TestLayerSensitivity
# ---------------------------------------------------------------------------

class TestLayerSensitivity:
    def test_combined_weights_keys_more(self):
        sens = LayerSensitivity(layer_idx=0, key_sensitivity=1.0, value_sensitivity=0.5)
        assert sens.combined_sensitivity == 1.0 * 1.5 + 0.5

    def test_repr_contains_layer_idx(self):
        sens = LayerSensitivity(layer_idx=5, key_sensitivity=0.3, value_sensitivity=0.2)
        assert "5" in repr(sens)


# ---------------------------------------------------------------------------
# TestKVTunerCalibrator
# ---------------------------------------------------------------------------

class TestKVTunerCalibrator:
    def _make_calibrated(self, n_layers=8, target=4.0) -> KVQuantConfig:
        cfg = KVTunerConfig(n_layers=n_layers, target_avg_bits=target,
                            candidate_bits=(2, 4, 8))
        cal = KVTunerCalibrator(cfg)
        rng = np.random.default_rng(42)
        for li in range(n_layers):
            k = rng.standard_normal((16, 32)).astype(np.float32)
            v = rng.standard_normal((16, 32)).astype(np.float32)
            cal.record_layer(li, k, v)
        return cal.search()

    def test_search_returns_kvquantconfig(self):
        qc = self._make_calibrated()
        assert isinstance(qc, KVQuantConfig)

    def test_all_layers_have_bits(self):
        qc = self._make_calibrated(n_layers=8)
        assert len(qc.k_bits) == 8
        assert len(qc.v_bits) == 8

    def test_bits_are_valid(self):
        qc = self._make_calibrated(n_layers=8)
        for li in range(8):
            k, v = qc.bits_for_layer(li)
            assert k in ALLOWED_BITS
            assert v in ALLOWED_BITS

    def test_key_bits_gte_value_bits(self):
        qc = self._make_calibrated(n_layers=8)
        for li in range(8):
            k, v = qc.bits_for_layer(li)
            assert k >= v

    def test_avg_bits_within_range(self):
        qc = self._make_calibrated(n_layers=8, target=4.0)
        assert 1.0 <= qc.avg_bits <= 16.0

    def test_search_without_data_uses_heuristic(self):
        cfg = KVTunerConfig(n_layers=4, candidate_bits=(2, 4, 8))
        cal = KVTunerCalibrator(cfg)
        qc = cal.search()
        assert qc.n_layers == 4

    def test_sensitivities_populated(self):
        qc = self._make_calibrated(n_layers=8)
        assert len(qc.sensitivities) == 8


# ---------------------------------------------------------------------------
# TestKVQuantConfig
# ---------------------------------------------------------------------------

class TestKVQuantConfig:
    def _make_config(self):
        cfg = KVTunerConfig(n_layers=4, candidate_bits=(2, 4, 8))
        cal = KVTunerCalibrator(cfg)
        return cal.search()

    def test_bits_for_layer_default_fallback(self):
        qc = self._make_config()
        k, v = qc.bits_for_layer(100)  # out of range
        assert k == 4 and v == 4

    def test_save_and_load_roundtrip(self):
        qc = self._make_config()
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            qc.save(path)
            loaded = KVQuantConfig.load(path)
            assert loaded.n_layers == qc.n_layers
            assert loaded.k_bits == qc.k_bits
            assert loaded.v_bits == qc.v_bits
        finally:
            os.unlink(path)

    def test_save_creates_valid_json(self):
        qc = self._make_config()
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            path = f.name
        try:
            qc.save(path)
            with open(path) as f:
                data = json.load(f)
            assert "k_bits" in data
            assert "v_bits" in data
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# TestKVTunerStats
# ---------------------------------------------------------------------------

class TestKVTunerStats:
    def _make_stats(self, avg_bits=4.0):
        cfg = KVTunerConfig(n_layers=4, candidate_bits=(4,), target_avg_bits=avg_bits)
        cal = KVTunerCalibrator(cfg)
        qc = cal.search()
        return KVTunerStats(quant_config=qc)

    def test_avg_bits_property(self):
        stats = self._make_stats(avg_bits=4.0)
        assert stats.avg_bits >= 1.0

    def test_memory_reduction_vs_fp16_positive(self):
        stats = self._make_stats()
        assert stats.estimated_memory_reduction_vs_fp16() > 0.0

    def test_memory_reduction_vs_kivi8(self):
        stats = self._make_stats(avg_bits=4.0)
        r = stats.estimated_memory_reduction_vs_kivi8()
        assert r >= 0.0

    def test_throughput_improvement_at_4bit(self):
        stats = self._make_stats(avg_bits=4.0)
        improvement = stats.estimated_throughput_improvement_vs_kivi8()
        assert improvement >= 0.0
