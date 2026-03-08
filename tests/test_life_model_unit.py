"""tests/test_life_model_unit.py — 100% coverage for squish/life_model.py"""
import json
import platform
from pathlib import Path
from unittest import mock

import pytest

from squish.life_model import (
    _detect_chip,
    _count_model_params,
    _quant_bytes_per_param,
    _kv_memory_gb,
    predict,
    _DEFAULT_BW_GB_S,
    _DEFAULT_TFLOPS,
)


# ---------------------------------------------------------------------------
# _detect_chip
# ---------------------------------------------------------------------------

class TestDetectChip:
    def _patch_system(self, system="Darwin"):
        return mock.patch("squish.life_model.platform.system", return_value=system)

    def _patch_subprocess(self, outputs):
        """outputs: list of return values per call; raise CalledProcessError for None."""
        import subprocess
        def side_effect(cmd, **kwargs):
            val = outputs.pop(0) if outputs else ""
            if val is None:
                raise subprocess.CalledProcessError(1, cmd)
            return val
        return mock.patch("squish.life_model.subprocess.check_output",
                          side_effect=side_effect)

    def test_non_darwin_returns_defaults(self):
        with self._patch_system("Linux"):
            name, bw, tflops = _detect_chip()
        assert name    == "unknown"
        assert bw      == _DEFAULT_BW_GB_S
        assert tflops  == _DEFAULT_TFLOPS

    def test_known_chip_detected(self):
        # Use the exact base key; "apple m1" is the first dict entry and
        # matches itself unambiguously (not a prefix of an earlier key).
        with self._patch_system("Darwin"):
            with self._patch_subprocess(["Apple M1"]):
                name, bw, tflops = _detect_chip()
        assert name   == "Apple M1"
        assert bw     == 68.0
        assert tflops == 11.0

    def test_unknown_chip_falls_back_to_defaults(self):
        with self._patch_system("Darwin"):
            with self._patch_subprocess(["Intel Core i9"]):
                name, bw, tflops = _detect_chip()
        assert bw    == _DEFAULT_BW_GB_S
        assert tflops == _DEFAULT_TFLOPS

    def test_first_call_raises_falls_back_to_hw_model(self):
        """First subprocess call raises → falls back to sysctl hw.model."""
        with self._patch_system("Darwin"):
            # "Apple M2" matches key "apple m2" exactly (bw=100.0)
            with self._patch_subprocess([None, "Apple M2"]):
                name, bw, tflops = _detect_chip()
        assert name == "Apple M2"
        assert bw   == 100.0

    def test_both_calls_raise_returns_defaults(self):
        with self._patch_system("Darwin"):
            with self._patch_subprocess([None, None]):
                name, bw, tflops = _detect_chip()
        assert bw    == _DEFAULT_BW_GB_S

    def test_empty_string_result_tries_fallback(self):
        """First call returns empty string → tries second sysctl call."""
        with self._patch_system("Darwin"):
            # "Apple M3" matches key "apple m3" exactly (bw=100.0)
            with self._patch_subprocess(["", "Apple M3"]):
                name, bw, tflops = _detect_chip()
        assert name == "Apple M3"
        assert bw   == 100.0


# ---------------------------------------------------------------------------
# _count_model_params
# ---------------------------------------------------------------------------

class TestCountModelParams:
    def test_none_returns_default(self):
        assert _count_model_params(None) == 7.0

    def test_reads_num_parameters_from_config(self, tmp_path):
        cfg = {"num_parameters": 8_000_000_000}
        (tmp_path / "config.json").write_text(json.dumps(cfg))
        assert _count_model_params(str(tmp_path)) == pytest.approx(8.0)

    def test_computes_from_architecture(self, tmp_path):
        cfg = {
            "hidden_size": 4096,
            "intermediate_size": 11008,
            "num_hidden_layers": 32,
            "vocab_size": 32000,
            "num_attention_heads": 32,
        }
        (tmp_path / "config.json").write_text(json.dumps(cfg))
        result = _count_model_params(str(tmp_path))
        assert result > 0.0

    def test_corrupt_config_falls_back_to_name_heuristic(self, tmp_path):
        (tmp_path / "config.json").write_text("NOT JSON AT ALL {{{")
        # Directory name has no size hint → default 7.0
        result = _count_model_params(str(tmp_path))
        assert result == 7.0

    def test_name_heuristic_72b(self, tmp_path):
        named = tmp_path / "my-model-72b"
        named.mkdir()
        (named / "config.json").write_text("not json")
        assert _count_model_params(str(named)) == 72.0

    def test_name_heuristic_1_5b(self, tmp_path):
        named = tmp_path / "qwen3-1.5b-mlx"
        named.mkdir()
        assert _count_model_params(str(named)) == 1.5

    def test_no_config_no_hint_returns_default(self, tmp_path):
        # No config.json in the directory → directory name has no size hint
        named = tmp_path / "mystery-model"
        named.mkdir()
        assert _count_model_params(str(named)) == 7.0


# ---------------------------------------------------------------------------
# _quant_bytes_per_param
# ---------------------------------------------------------------------------

class TestQuantBytesPerParam:
    def test_none_returns_fp16(self):
        assert _quant_bytes_per_param(None) == 2.0

    def test_reads_bits_from_config(self, tmp_path):
        cfg = {"quantization": {"bits": 4}}
        (tmp_path / "config.json").write_text(json.dumps(cfg))
        assert _quant_bytes_per_param(str(tmp_path)) == pytest.approx(0.5)

    def test_reads_num_bits_from_config(self, tmp_path):
        cfg = {"num_bits": 8}
        (tmp_path / "config.json").write_text(json.dumps(cfg))
        assert _quant_bytes_per_param(str(tmp_path)) == pytest.approx(1.0)

    def test_corrupt_config_uses_name_heuristic(self, tmp_path):
        (tmp_path / "config.json").write_text("{bad json}")
        named = tmp_path / "model-int4"
        named.mkdir()
        assert _quant_bytes_per_param(str(named)) == pytest.approx(0.5)

    def test_name_hint_int8(self, tmp_path):
        named = tmp_path / "model-int8"
        named.mkdir()
        assert _quant_bytes_per_param(str(named)) == pytest.approx(1.0)

    def test_name_hint_mlx_int4(self, tmp_path):
        named = tmp_path / "Qwen3-8B-mlx-int4"
        named.mkdir()
        assert _quant_bytes_per_param(str(named)) == pytest.approx(0.5)

    def test_no_hint_returns_fp16(self, tmp_path):
        named = tmp_path / "generic-model"
        named.mkdir()
        assert _quant_bytes_per_param(str(named)) == pytest.approx(2.0)

    def test_config_without_bits_falls_through_to_name_heuristic(self, tmp_path):
        """Valid config with no quantization key → bits is None → branch 207→212."""
        cfg = {"hidden_size": 4096}  # no 'quantization' or 'num_bits'
        (tmp_path / "config.json").write_text(json.dumps(cfg))
        # Directory name has no int4/int8 hint → returns fp16 default
        assert _quant_bytes_per_param(str(tmp_path)) == pytest.approx(2.0)

    def test_corrupt_config_in_model_dir_hits_except(self, tmp_path):
        """config.json exists but is corrupt → except block executes (line 209)."""
        (tmp_path / "config.json").write_text("{corrupt: json!!!}")
        # Directory name has no int4/int8 hint → returns fp16 default
        assert _quant_bytes_per_param(str(tmp_path)) == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# _kv_memory_gb
# ---------------------------------------------------------------------------

class TestKVMemoryGb:
    def test_returns_positive_float_with_defaults(self):
        result = _kv_memory_gb(None, seq_len=512, output_len=128, batch_size=1)
        assert result > 0.0

    def test_scales_with_seq_len(self):
        small = _kv_memory_gb(None, seq_len=128,  output_len=32,  batch_size=1)
        large = _kv_memory_gb(None, seq_len=1024, output_len=256, batch_size=1)
        assert large > small

    def test_reads_config_json(self, tmp_path):
        cfg = {
            "num_hidden_layers":   4,
            "num_key_value_heads": 4,
            "head_dim":            64,
            "num_attention_heads": 4,
            "hidden_size":         256,
        }
        (tmp_path / "config.json").write_text(json.dumps(cfg))
        result = _kv_memory_gb(str(tmp_path), seq_len=16, output_len=16, batch_size=1)
        # 2 * 4 * 4 * 32 * 64 * 2 / (1024^3)
        expected = 2 * 4 * 4 * 32 * 64 * 2 / (1024 ** 3)
        assert result == pytest.approx(expected, rel=1e-5)

    def test_corrupt_config_uses_defaults(self, tmp_path):
        (tmp_path / "config.json").write_text("!!!bad json!!!")
        result = _kv_memory_gb(str(tmp_path), seq_len=16, output_len=16, batch_size=1)
        assert result > 0.0

    def test_head_dim_derived_from_hidden_size(self, tmp_path):
        cfg = {
            "num_hidden_layers":   2,
            "num_attention_heads": 4,
            "hidden_size":         128,   # head_dim = 128 // 4 = 32
        }
        (tmp_path / "config.json").write_text(json.dumps(cfg))
        result = _kv_memory_gb(str(tmp_path), seq_len=8, output_len=8, batch_size=1)
        assert result > 0.0

    def test_model_dir_without_config_json(self, tmp_path):
        """model_dir given but no config.json → config_path.exists() False → branch 238→244."""
        # tmp_path exists but contains no config.json
        result = _kv_memory_gb(str(tmp_path), seq_len=8, output_len=8, batch_size=1)
        assert result > 0.0


# ---------------------------------------------------------------------------
# predict
# ---------------------------------------------------------------------------

class TestPredict:
    def _mock_chip(self, bw=100.0, tflops=15.0):
        return mock.patch("squish.life_model._detect_chip",
                          return_value=("Mock Chip", bw, tflops))

    def test_returns_expected_keys(self):
        with self._mock_chip():
            result = predict()
        expected_keys = {
            "ttft_ms", "tpot_ms", "tokens_per_sec", "kv_memory_gb",
            "model_memory_gb", "bottleneck", "model_params_b",
            "effective_bw_gb_s", "hardware",
        }
        assert set(result.keys()) == expected_keys

    def test_hardware_name_matches_mock(self):
        with self._mock_chip():
            result = predict()
        assert result["hardware"] == "Mock Chip"

    def test_positive_values(self):
        with self._mock_chip():
            result = predict(batch_size=1, seq_len=512, output_len=128)
        assert result["ttft_ms"]        > 0
        assert result["tpot_ms"]        > 0
        assert result["tokens_per_sec"] > 0
        assert result["kv_memory_gb"]  >= 0
        assert result["model_params_b"] > 0

    def test_memory_bandwidth_bottleneck(self):
        """Low-bandwidth chip → memory-bandwidth bottleneck."""
        with self._mock_chip(bw=10.0, tflops=1000.0):
            result = predict(batch_size=1)
        assert result["bottleneck"] == "memory-bandwidth"

    def test_compute_bottleneck(self):
        """Very high bandwidth, low TFLOPS → compute bound."""
        with self._mock_chip(bw=10000.0, tflops=0.001):
            result = predict(batch_size=1)
        assert result["bottleneck"] == "compute"

    def test_with_model_dir(self, tmp_path):
        """Passing model_dir reads config; no crash."""
        cfg = {"num_parameters": 7_000_000_000, "quantization": {"bits": 4}}
        (tmp_path / "config.json").write_text(json.dumps(cfg))
        with self._mock_chip():
            result = predict(model_dir=str(tmp_path))
        assert result["model_params_b"] == pytest.approx(7.0)
        assert result["model_memory_gb"] > 0

    def test_larger_batch_increases_throughput_or_equal(self):
        """Larger batch_size should give >= single-stream throughput."""
        with self._mock_chip():
            r1 = predict(batch_size=1)
            r4 = predict(batch_size=4)
        assert r4["tokens_per_sec"] >= r1["tokens_per_sec"]
