"""
tests/test_remote_ext.py

Extended coverage for squish/remote.py:
  - _safetensors_tensors: __metadata__ key skip   (line 44)
  - _safetensors_tensors: None data_offsets skip  (line 47)
  - _npy_dir_tensors: non-npy magic skip          (line 69)
  - build_manifest: tensors/ npy-dir format       (lines 138-144)
  - build_manifest: no valid format → sys.exit(1) (lines 147-148)
  - build_manifest: ancillary files               (line 157)
"""
from __future__ import annotations

import json
import struct
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

import squish.remote as _remote


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_safetensors(path: Path, meta: dict) -> None:
    """Write a minimal safetensors file with the given metadata dict."""
    header_json = json.dumps(meta).encode("utf-8")
    header_len = struct.pack("<Q", len(header_json))
    path.write_bytes(header_len + header_json)


def _write_npy(path: Path, arr: np.ndarray) -> None:
    np.save(path, arr)


# ── _safetensors_tensors ──────────────────────────────────────────────────────

class TestSafetensorsTensors:
    def test_metadata_key_is_skipped(self, tmp_path: Path):
        """__metadata__ key should be skipped (line 44)."""
        sf_path = tmp_path / "model.safetensors"
        meta = {
            "__metadata__": {"format": "pt"},
            "weight": {"dtype": "F32", "shape": [4], "data_offsets": [0, 16]},
        }
        _make_safetensors(sf_path, meta)
        results = list(_remote._safetensors_tensors(sf_path, "http://localhost/model.safetensors"))
        # __metadata__ should be excluded
        names = [r["name"] for r in results]
        assert "__metadata__" not in names
        assert "weight" in names

    def test_none_data_offsets_is_skipped(self, tmp_path: Path):
        """Entry with missing/None data_offsets should be skipped (line 47)."""
        sf_path = tmp_path / "model.safetensors"
        meta = {
            "weight_with_offsets": {"dtype": "F32", "shape": [4], "data_offsets": [0, 16]},
            "weight_no_offsets":   {"dtype": "F32", "shape": [4]},  # no data_offsets
        }
        _make_safetensors(sf_path, meta)
        results = list(_remote._safetensors_tensors(sf_path, "http://localhost/model.safetensors"))
        names = [r["name"] for r in results]
        assert "weight_with_offsets" in names
        assert "weight_no_offsets" not in names

    def test_offset_calculation(self, tmp_path: Path):
        """Offsets should be absolute (header_end + relative_offset)."""
        sf_path = tmp_path / "model.safetensors"
        meta = {
            "layer": {"dtype": "F16", "shape": [2, 4], "data_offsets": [0, 16]},
        }
        _make_safetensors(sf_path, meta)
        results = list(_remote._safetensors_tensors(sf_path, "http://localhost/layer"))
        assert len(results) == 1
        header_json_len = len(json.dumps(meta).encode("utf-8"))
        header_end = 8 + header_json_len
        assert results[0]["data_offsets"][0] == header_end + 0
        assert results[0]["data_offsets"][1] == header_end + 16


# ── _npy_dir_tensors ──────────────────────────────────────────────────────────

class TestNpyDirTensors:
    def test_valid_npy_file_included(self, tmp_path: Path):
        """Valid .npy file should be included in results."""
        arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        npy_path = tmp_path / "layer.npy"
        _write_npy(npy_path, arr)
        results = _remote._npy_dir_tensors(tmp_path, "http://localhost/tensors")
        names = [r["filename"] for r in results]
        assert "layer.npy" in names

    def test_non_npy_magic_is_skipped(self, tmp_path: Path):
        """File with .npy extension but wrong magic bytes skipped (line 69)."""
        bad_npy = tmp_path / "fake.npy"
        bad_npy.write_bytes(b"NOT_NUMPY_MAGIC_AT_ALL")
        results = _remote._npy_dir_tensors(tmp_path, "http://localhost/tensors")
        names = [r["filename"] for r in results]
        assert "fake.npy" not in names

    def test_mixed_valid_and_invalid(self, tmp_path: Path):
        """Mix of valid and invalid .npy files: only valid ones returned."""
        arr = np.array([1.0], dtype=np.float32)
        good = tmp_path / "good.npy"
        bad  = tmp_path / "bad.npy"
        _write_npy(good, arr)
        bad.write_bytes(b"GARBAGE!")
        results = _remote._npy_dir_tensors(tmp_path, "http://localhost/tensors")
        names = [r["filename"] for r in results]
        assert "good.npy" in names
        assert "bad.npy" not in names

    def test_empty_directory(self, tmp_path: Path):
        results = _remote._npy_dir_tensors(tmp_path, "http://localhost/tensors")
        assert results == []


# ── build_manifest ────────────────────────────────────────────────────────────

class TestBuildManifest:
    def test_no_format_raises_system_exit(self, tmp_path: Path):
        """No safetensors and no tensors/ dir → sys.exit(1) (lines 147-148)."""
        model_dir = tmp_path / "empty_model"
        model_dir.mkdir()
        with pytest.raises(SystemExit):
            _remote.build_manifest(model_dir, "http://localhost/model")

    def test_tensors_dir_format(self, tmp_path: Path):
        """model_dir/tensors/*.npy format: npy-dir branch (lines 129-144)."""
        model_dir = tmp_path / "my_model"
        tensors_dir = model_dir / "tensors"
        tensors_dir.mkdir(parents=True)
        arr = np.array([1.0, 2.0], dtype=np.float32)
        _write_npy(tensors_dir / "layer0.npy", arr)
        manifest = _remote.build_manifest(model_dir, "http://localhost/model")
        assert any(f.get("format") == "npy_dir" for f in manifest["files"])

    def test_tensors_dir_with_sha256(self, tmp_path: Path):
        """tensors/ dir + include_sha256=True: sha256 computed (line 138)."""
        model_dir = tmp_path / "my_model"
        tensors_dir = model_dir / "tensors"
        tensors_dir.mkdir(parents=True)
        arr = np.array([1.0, 2.0], dtype=np.float32)
        _write_npy(tensors_dir / "layer0.npy", arr)
        manifest = _remote.build_manifest(model_dir, "http://localhost/model", include_sha256=True)
        items = manifest["files"][0]["items"]
        assert all("sha256" in item for item in items)

    def test_ancillary_files_collected(self, tmp_path: Path):
        """Ancillary files like tokenizer.json collected (line 157)."""
        model_dir = tmp_path / "my_model"
        tensors_dir = model_dir / "tensors"
        tensors_dir.mkdir(parents=True)
        arr = np.array([1.0], dtype=np.float32)
        _write_npy(tensors_dir / "w.npy", arr)
        # Create ancillary file
        (model_dir / "tokenizer.json").write_text('{"version": "1.0"}')
        (model_dir / "config.json").write_text('{}')
        manifest = _remote.build_manifest(model_dir, "http://localhost/model")
        filenames = [a["filename"] for a in manifest["ancillary"]]
        assert "tokenizer.json" in filenames
        assert "config.json" in filenames
