"""
tests/test_remote_unit.py

Unit tests for squish/remote.py.
All tests use synthetic temp files — no network / HF access required.

Actual API:
    _read_safetensors_metadata(path: Path) -> (dict, int)
    _npy_dir_tensors(tensors_dir: Path, base_url_dir: str) -> list[dict]
    _sha256(path: Path) -> str
    build_manifest(model_dir: Path, base_url: str, include_sha256=True) -> dict
"""
from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np
import pytest

from squish.remote import (
    _npy_dir_tensors,
    _read_safetensors_metadata,
    _sha256,
    build_manifest,
)


# ── helpers ────────────────────────────────────────────────────────────────────

def _make_safetensors_file(path: Path, tensors: dict) -> None:
    """Write a minimal valid safetensors file."""
    DTYPE_MAP = {
        np.float32: "F32",
        np.float16: "F16",
        np.int32: "I32",
        np.uint8: "U8",
    }
    metadata: dict = {}
    data_offset = 0
    data_chunks = []
    for name, arr in tensors.items():
        arr = np.ascontiguousarray(arr)
        dtype_str = DTYPE_MAP.get(arr.dtype.type, "F32")
        nbytes = arr.nbytes
        metadata[name] = {
            "dtype": dtype_str,
            "shape": list(arr.shape),
            "data_offsets": [data_offset, data_offset + nbytes],
        }
        data_chunks.append(arr.tobytes())
        data_offset += nbytes
    header_json = json.dumps(metadata).encode()
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(header_json)))
        f.write(header_json)
        for chunk in data_chunks:
            f.write(chunk)


# ── _read_safetensors_metadata ─────────────────────────────────────────────────

class TestReadSafetensorsMetadata:
    """Returns (meta_dict, header_end_bytes) tuple."""

    def test_returns_tuple(self, tmp_path):
        p = tmp_path / "model.safetensors"
        _make_safetensors_file(p, {"weight": np.ones((4, 8), dtype=np.float32)})
        result = _read_safetensors_metadata(p)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_meta_is_dict(self, tmp_path):
        p = tmp_path / "model.safetensors"
        _make_safetensors_file(p, {"weight": np.ones((4, 8), dtype=np.float32)})
        meta, header_len = _read_safetensors_metadata(p)
        assert isinstance(meta, dict)

    def test_header_len_is_int(self, tmp_path):
        p = tmp_path / "model.safetensors"
        _make_safetensors_file(p, {"w": np.zeros(4, dtype=np.float32)})
        meta, header_len = _read_safetensors_metadata(p)
        assert isinstance(header_len, int)
        assert header_len > 8  # at least 8 bytes for the length prefix

    def test_tensor_names_in_meta(self, tmp_path):
        p = tmp_path / "model.safetensors"
        _make_safetensors_file(p, {
            "embed": np.zeros((10, 16), dtype=np.float32),
            "proj": np.eye(8, dtype=np.float32),
        })
        meta, _ = _read_safetensors_metadata(p)
        assert "embed" in meta
        assert "proj" in meta

    def test_meta_has_shape(self, tmp_path):
        p = tmp_path / "m.safetensors"
        arr = np.zeros((3, 5), dtype=np.float32)
        _make_safetensors_file(p, {"w": arr})
        meta, _ = _read_safetensors_metadata(p)
        assert meta["w"]["shape"] == [3, 5]

    def test_meta_has_dtype(self, tmp_path):
        p = tmp_path / "m.safetensors"
        _make_safetensors_file(p, {"w": np.zeros((2, 4), dtype=np.float16)})
        meta, _ = _read_safetensors_metadata(p)
        assert "dtype" in meta["w"]

    def test_meta_has_data_offsets(self, tmp_path):
        p = tmp_path / "m.safetensors"
        _make_safetensors_file(p, {"a": np.ones(4, dtype=np.float32)})
        meta, _ = _read_safetensors_metadata(p)
        assert "data_offsets" in meta["a"]
        assert len(meta["a"]["data_offsets"]) == 2

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises((FileNotFoundError, OSError)):
            _read_safetensors_metadata(tmp_path / "nonexistent.safetensors")


# ── _npy_dir_tensors ──────────────────────────────────────────────────────────

class TestNpyDirTensors:
    """Returns list[dict] with file metadata entries (not arrays)."""

    def test_returns_list(self, tmp_path):
        np.save(tmp_path / "weight.npy", np.ones(4, dtype=np.float32))
        result = _npy_dir_tensors(tmp_path, "http://example.com/tensors")
        assert isinstance(result, list)

    def test_single_npy_one_entry(self, tmp_path):
        np.save(tmp_path / "weight.npy", np.ones(4, dtype=np.float32))
        result = _npy_dir_tensors(tmp_path, "http://example.com")
        assert len(result) == 1

    def test_entry_is_dict(self, tmp_path):
        np.save(tmp_path / "tensor.npy", np.zeros(8, dtype=np.float32))
        result = _npy_dir_tensors(tmp_path, "http://example.com")
        assert isinstance(result[0], dict)

    def test_entry_has_filename(self, tmp_path):
        np.save(tmp_path / "myweight.npy", np.zeros(4))
        result = _npy_dir_tensors(tmp_path, "http://example.com")
        assert "filename" in result[0]
        assert "myweight.npy" in result[0]["filename"]

    def test_entry_has_url(self, tmp_path):
        np.save(tmp_path / "layer_0.npy", np.zeros(4))
        result = _npy_dir_tensors(tmp_path, "http://example.com/dir")
        assert "url" in result[0]
        assert "http://" in result[0]["url"]

    def test_multiple_npy_files(self, tmp_path):
        for i in range(3):
            np.save(tmp_path / f"layer_{i}.npy", np.zeros(i + 1, dtype=np.float32))
        result = _npy_dir_tensors(tmp_path, "http://example.com")
        assert len(result) == 3

    def test_empty_dir_empty_list(self, tmp_path):
        result = _npy_dir_tensors(tmp_path, "http://example.com")
        assert result == []

    def test_ignores_non_npy(self, tmp_path):
        (tmp_path / "readme.txt").write_text("hello")
        np.save(tmp_path / "a.npy", np.zeros(2))
        result = _npy_dir_tensors(tmp_path, "http://example.com")
        assert len(result) == 1
        assert result[0]["filename"] == "a.npy"

    def test_sorted_results(self, tmp_path):
        for i in [3, 1, 2]:
            np.save(tmp_path / f"layer_{i}.npy", np.zeros(2))
        result = _npy_dir_tensors(tmp_path, "http://example.com")
        names = [r["filename"] for r in result]
        assert names == sorted(names)


# ── _sha256 ────────────────────────────────────────────────────────────────────

class TestSha256:
    def test_returns_string(self, tmp_path):
        p = tmp_path / "data.bin"
        p.write_bytes(b"hello world")
        result = _sha256(p)
        assert isinstance(result, str)

    def test_hex_length_64(self, tmp_path):
        p = tmp_path / "data.bin"
        p.write_bytes(b"test")
        assert len(_sha256(p)) == 64

    def test_hex_chars_only(self, tmp_path):
        p = tmp_path / "data.bin"
        p.write_bytes(b"\x00" * 64)
        result = _sha256(p)
        assert all(c in "0123456789abcdef" for c in result)

    def test_deterministic(self, tmp_path):
        p = tmp_path / "data.bin"
        p.write_bytes(b"determinism")
        assert _sha256(p) == _sha256(p)

    def test_different_files_different_hash(self, tmp_path):
        p1 = tmp_path / "a.bin"
        p2 = tmp_path / "b.bin"
        p1.write_bytes(b"aaa")
        p2.write_bytes(b"bbb")
        assert _sha256(p1) != _sha256(p2)

    def test_known_hash(self, tmp_path):
        import hashlib  # noqa: PLC0415
        data = b"squish"
        p = tmp_path / "known.bin"
        p.write_bytes(data)
        assert _sha256(p) == hashlib.sha256(data).hexdigest()

    def test_empty_file(self, tmp_path):
        import hashlib  # noqa: PLC0415
        p = tmp_path / "empty.bin"
        p.write_bytes(b"")
        assert _sha256(p) == hashlib.sha256(b"").hexdigest()

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises((FileNotFoundError, OSError)):
            _sha256(tmp_path / "missing.bin")


# ── build_manifest ────────────────────────────────────────────────────────────

class TestBuildManifest:
    """build_manifest(model_dir: Path, base_url: str, include_sha256=True) -> dict"""

    def _setup_npy_dir(self, tmp_path) -> Path:
        """Create a fake model dir with a tensors/ subdirectory."""
        model_dir = tmp_path / "model"
        tensors_dir = model_dir / "tensors"
        tensors_dir.mkdir(parents=True)
        for i in range(3):
            arr = np.zeros((4, 8), dtype=np.float32)
            np.save(tensors_dir / f"layer_{i}__q.npy", arr)
        return model_dir

    def _setup_safetensors_dir(self, tmp_path) -> Path:
        model_dir = tmp_path / "model_sf"
        model_dir.mkdir()
        _make_safetensors_file(
            model_dir / "weights.safetensors",
            {"embed": np.zeros((4, 8), dtype=np.float32),
             "proj":  np.ones((4, 8), dtype=np.float32)},
        )
        return model_dir

    def test_returns_dict_safetensors(self, tmp_path, capsys):
        model_dir = self._setup_safetensors_dir(tmp_path)
        result = build_manifest(model_dir, "http://example.com/model",
                                include_sha256=False)
        assert isinstance(result, dict)

    def test_has_files_key_safetensors(self, tmp_path, capsys):
        model_dir = self._setup_safetensors_dir(tmp_path)
        result = build_manifest(model_dir, "http://example.com/model",
                                include_sha256=False)
        assert "files" in result

    def test_has_model_name(self, tmp_path, capsys):
        model_dir = self._setup_safetensors_dir(tmp_path)
        result = build_manifest(model_dir, "http://example.com/model",
                                include_sha256=False)
        assert "model_name" in result

    def test_manifest_json_serializable(self, tmp_path, capsys):
        model_dir = self._setup_safetensors_dir(tmp_path)
        result = build_manifest(model_dir, "http://example.com/model",
                                include_sha256=False)
        json.dumps(result)

    def test_npy_dir_format(self, tmp_path, capsys):
        model_dir = self._setup_npy_dir(tmp_path)
        result = build_manifest(model_dir, "http://example.com/model",
                                include_sha256=False)
        assert isinstance(result, dict)
        assert "files" in result

    def test_sha256_included_when_requested(self, tmp_path, capsys):
        model_dir = self._setup_safetensors_dir(tmp_path)
        result = build_manifest(model_dir, "http://example.com/model",
                                include_sha256=True)
        result_str = json.dumps(result)
        import re  # noqa: PLC0415
        assert re.search(r"[0-9a-f]{64}", result_str), "No SHA-256 hashes found"
