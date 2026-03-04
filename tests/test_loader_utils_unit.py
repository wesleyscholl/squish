"""
tests/test_loader_utils_unit.py

Unit tests for squish/loader_utils.py.
Covers _load_npy_path, _HF_TO_MLX_TYPE, _safe_key_to_original,
_dequantize, and _dequantize_npy with synthetic data.
"""
from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
import pytest

from squish.loader_utils import (
    _HF_TO_MLX_TYPE,
    _dequantize,
    _dequantize_npy,
    _load_npy_path,
    _safe_key_to_original,
)


# ── _load_npy_path ─────────────────────────────────────────────────────────────
# Signature: _load_npy_path(path: Path, mmap_mode='r') -> np.ndarray

class TestLoadNpyPath:
    def test_load_plain_npy(self, tmp_path):
        arr = np.arange(10, dtype=np.float32)
        p = tmp_path / "test.npy"
        np.save(p, arr)
        loaded = _load_npy_path(p)
        assert np.array_equal(loaded, arr)

    def test_load_npy_2d(self, tmp_path):
        arr = np.random.default_rng(0).standard_normal((16, 64)).astype(np.float32)
        p = tmp_path / "matrix.npy"
        np.save(p, arr)
        loaded = _load_npy_path(p)
        assert loaded.shape == (16, 64)

    def test_load_preserves_float16(self, tmp_path):
        arr = np.ones((4, 8), dtype=np.float16)
        p = tmp_path / "fp16.npy"
        np.save(p, arr)
        loaded = _load_npy_path(p, mmap_mode=None)
        assert loaded.dtype == np.float16

    def test_load_preserves_int32(self, tmp_path):
        arr = np.zeros((2, 3), dtype=np.int32)
        p = tmp_path / "int32.npy"
        np.save(p, arr)
        loaded = _load_npy_path(p, mmap_mode=None)
        assert loaded.dtype == np.int32

    def test_zst_fallback(self, tmp_path):
        """When .npy doesn't exist, falls back to .npy.zst."""
        pytest.importorskip("zstandard")
        import zstandard as zstd  # noqa: PLC0415
        arr = np.arange(12, dtype=np.float32)
        buf = io.BytesIO()
        np.save(buf, arr)
        compressed = zstd.ZstdCompressor().compress(buf.getvalue())
        (tmp_path / "data.npy.zst").write_bytes(compressed)
        # Pass the .npy path (no .zst suffix); function appends .zst internally
        loaded = _load_npy_path(tmp_path / "data.npy")
        assert np.array_equal(loaded, arr)

    def test_missing_raises(self, tmp_path):
        with pytest.raises((FileNotFoundError, OSError, RuntimeError)):
            _load_npy_path(tmp_path / "nonexistent.npy")


# ── _HF_TO_MLX_TYPE ───────────────────────────────────────────────────────────
# Maps HuggingFace model_type strings (lowercase) to mlx_lm module names.

class TestHFToMLXType:
    def test_is_dict(self):
        assert isinstance(_HF_TO_MLX_TYPE, dict)

    def test_contains_llama(self):
        assert "llama" in _HF_TO_MLX_TYPE

    def test_contains_qwen2(self):
        assert "qwen2" in _HF_TO_MLX_TYPE

    def test_contains_mistral(self):
        assert "mistral" in _HF_TO_MLX_TYPE

    def test_contains_gemma(self):
        assert "gemma" in _HF_TO_MLX_TYPE

    def test_values_are_strings(self):
        for k, v in _HF_TO_MLX_TYPE.items():
            assert isinstance(v, str), f"Non-string value for {k!r}"

    def test_keys_are_lowercase(self):
        for k in _HF_TO_MLX_TYPE:
            assert k == k.lower(), f"Key not lowercase: {k!r}"

    def test_llama_maps_to_llama(self):
        assert _HF_TO_MLX_TYPE["llama"] == "llama"

    def test_qwen2_maps_to_qwen2(self):
        assert _HF_TO_MLX_TYPE["qwen2"] == "qwen2"

    def test_non_empty(self):
        assert len(_HF_TO_MLX_TYPE) >= 5


# ── _safe_key_to_original ──────────────────────────────────────────────────────
# Signature: _safe_key_to_original(manifest_path: str) -> dict[str, str]
# Reads manifest.json; returns {safe_key -> original_name} (inverted dict).

class TestSafeKeyToOriginal:
    def _write_manifest(self, tmp_path, mapping: dict) -> str:
        p = tmp_path / "manifest.json"
        p.write_text(json.dumps(mapping))
        return str(p)

    def test_returns_dict(self, tmp_path):
        path = self._write_manifest(tmp_path, {"orig1": "safe1"})
        result = _safe_key_to_original(path)
        assert isinstance(result, dict)

    def test_inverted_mapping(self, tmp_path):
        # manifest: original_name -> safe_key
        # output should be: safe_key -> original_name
        manifest = {"model.embed.weight": "model__embed__weight",
                    "model.proj.weight": "model__proj__weight"}
        path = self._write_manifest(tmp_path, manifest)
        result = _safe_key_to_original(path)
        assert "model__embed__weight" in result
        assert result["model__embed__weight"] == "model.embed.weight"

    def test_empty_manifest(self, tmp_path):
        path = self._write_manifest(tmp_path, {})
        result = _safe_key_to_original(path)
        assert result == {}

    def test_all_values_strings(self, tmp_path):
        manifest = {f"orig_{i}": f"safe_{i}" for i in range(5)}
        path = self._write_manifest(tmp_path, manifest)
        result = _safe_key_to_original(path)
        for k, v in result.items():
            assert isinstance(k, str) and isinstance(v, str)

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises((FileNotFoundError, OSError)):
            _safe_key_to_original(str(tmp_path / "no_manifest.json"))


# ── _dequantize ────────────────────────────────────────────────────────────────
# Signature: _dequantize(npz, sk: str) -> np.ndarray
# npz is an open numpy NpzFile; sk is the safe_key prefix.

class TestDequantize:
    def _make_npz_int8(self, tmp_path, sk="layer", n=4, d=32):
        from squish.quantizer import quantize_embeddings  # noqa: PLC0415
        arr = np.random.default_rng(7).standard_normal((n, d)).astype(np.float32)
        r = quantize_embeddings(arr)
        npz_path = tmp_path / "weights.npz"
        np.savez_compressed(
            npz_path,
            **{
                f"{sk}__q":     r.quantized,
                f"{sk}__s":     r.scales,
                f"{sk}__shape": np.array([n, d]),
            },
        )
        return npz_path, arr

    def test_returns_ndarray(self, tmp_path):
        sk = "model__embed"
        npz_path, _ = self._make_npz_int8(tmp_path, sk=sk)
        with np.load(npz_path) as npz:
            result = _dequantize(npz, sk)
        assert isinstance(result, np.ndarray)

    def test_output_shape(self, tmp_path):
        sk = "model__embed"
        npz_path, orig = self._make_npz_int8(tmp_path, sk=sk, n=4, d=32)
        with np.load(npz_path) as npz:
            result = _dequantize(npz, sk)
        assert result.shape == orig.shape

    def test_output_dtype_float32(self, tmp_path):
        sk = "w"
        npz_path, _ = self._make_npz_int8(tmp_path, sk=sk)
        with np.load(npz_path) as npz:
            result = _dequantize(npz, sk)
        assert result.dtype == np.float32

    def test_passthrough_float16(self, tmp_path):
        """Handles {sk}__pt path (float16 passthrough)."""
        sk = "layer_pt"
        arr = np.random.default_rng(7).standard_normal((4, 16)).astype(np.float16)
        npz_path = tmp_path / "pt.npz"
        np.savez_compressed(npz_path, **{
            f"{sk}__pt":    arr,
            f"{sk}__shape": np.array([4, 16]),
        })
        with np.load(npz_path) as npz:
            result = _dequantize(npz, sk)
        assert result.dtype == np.float32
        assert result.shape == arr.shape


# ── _dequantize_npy ────────────────────────────────────────────────────────────
# Signature: _dequantize_npy(tensor_dir: Path, sk: str) -> np.ndarray
# Reads {sk}__q.npy and {sk}__s.npy from tensor_dir.

class TestDequantizeNpy:
    def test_int8_per_row(self, tmp_path):
        from squish.quantizer import quantize_embeddings  # noqa: PLC0415
        sk = "model__embed"
        n, d = 4, 32
        arr = np.random.default_rng(7).standard_normal((n, d)).astype(np.float32)
        r = quantize_embeddings(arr)
        np.save(tmp_path / f"{sk}__q.npy", r.quantized)
        np.save(tmp_path / f"{sk}__s.npy", r.scales)
        result = _dequantize_npy(tmp_path, sk)
        assert isinstance(result, np.ndarray)
        assert result.shape == (n, d)
        assert result.dtype == np.float32

    def test_passthrough_float16(self, tmp_path):
        sk = "ff_proj"
        arr = np.random.default_rng(9).standard_normal((8, 16)).astype(np.float16)
        np.save(tmp_path / f"{sk}__pt.npy", arr)
        result = _dequantize_npy(tmp_path, sk)
        assert result.dtype == np.float32
        assert result.shape == arr.shape

    def test_output_float32_always(self, tmp_path):
        from squish.quantizer import quantize_embeddings  # noqa: PLC0415
        sk = "w"
        arr = np.random.default_rng(1).standard_normal((4, 16)).astype(np.float32)
        r = quantize_embeddings(arr)
        np.save(tmp_path / f"{sk}__q.npy", r.quantized)
        np.save(tmp_path / f"{sk}__s.npy", r.scales)
        result = _dequantize_npy(tmp_path, sk)
        assert result.dtype == np.float32
