"""
tests/test_catalog_branches.py

Branch coverage tests for squish/catalog.py:
  - _try_refresh_catalog SQUISH_OFFLINE path   (lines 294-316)
  - _try_refresh_catalog fresh TTL path        (lines 310-317)
  - load_catalog(refresh=True) unlinks cache   (line 369)
  - list_catalog with sort_key ValueError      (line 393)
  - _has_squish_weights                        (lines 508-512)
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

import squish.catalog as _cat


# ── Helper: write a minimal valid catalog JSON ────────────────────────────────

def _write_catalog(path: Path, entries=None) -> None:
    if entries is None:
        entries = [
            {
                "id":              "qwen3:8b",
                "name":           "Qwen3 8B",
                "hf_mlx_repo":    "some/repo",
                "size_gb":        5.0,
                "squished_size_gb": 1.5,
                "params":         "8B",
                "context":        32768,
            }
        ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"models": entries}))


# ── SQUISH_OFFLINE mode ───────────────────────────────────────────────────────

class TestCatalogOfflineMode:
    def test_offline_with_no_local_cache(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
        """SQUISH_OFFLINE=1 and no local cache: returns empty/bundled catalog."""
        monkeypatch.setenv("SQUISH_OFFLINE", "1")
        monkeypatch.setattr(_cat, "_CATALOG_CACHE", None)
        # Point LOCAL_CATALOG_PATH to a non-existent file so exists() returns False
        nonexistent = tmp_path / "no_such_catalog.json"
        monkeypatch.setattr(_cat, "LOCAL_CATALOG_PATH", nonexistent)
        catalog = _cat._try_refresh_catalog({})
        assert isinstance(catalog, dict)

    def test_offline_with_local_cache_loads_it(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
        """SQUISH_OFFLINE=1 and local cache exists: load it synchronously."""
        monkeypatch.setenv("SQUISH_OFFLINE", "1")
        monkeypatch.setattr(_cat, "_CATALOG_CACHE", None)
        cache_file = tmp_path / "catalog.json"
        _write_catalog(cache_file)
        monkeypatch.setattr(_cat, "LOCAL_CATALOG_PATH", cache_file)
        catalog = _cat._try_refresh_catalog({})
        assert "qwen3:8b" in catalog

    def test_offline_with_malformed_cache(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
        """SQUISH_OFFLINE=1 with malformed JSON: silently fails, returns what we have."""
        monkeypatch.setenv("SQUISH_OFFLINE", "1")
        monkeypatch.setattr(_cat, "_CATALOG_CACHE", None)
        cache_file = tmp_path / "catalog.json"
        cache_file.write_text("not valid json")
        monkeypatch.setattr(_cat, "LOCAL_CATALOG_PATH", cache_file)
        catalog = _cat._try_refresh_catalog({})
        assert isinstance(catalog, dict)

    def test_offline_with_malformed_entry_skipped(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
        """SQUISH_OFFLINE=1 with entry missing required keys: KeyError swallowed (line 297)."""
        monkeypatch.setenv("SQUISH_OFFLINE", "1")
        monkeypatch.setattr(_cat, "_CATALOG_CACHE", None)
        cache_file = tmp_path / "catalog.json"
        # Entry has 'id' but missing required 'name', 'hf_mlx_repo', etc.
        cache_file.write_text(json.dumps({"models": [{"id": "broken"}]}))
        monkeypatch.setattr(_cat, "LOCAL_CATALOG_PATH", cache_file)
        catalog = _cat._try_refresh_catalog({})
        assert isinstance(catalog, dict)
        assert "broken" not in catalog  # entry was skipped due to missing fields


# ── Fresh TTL path ────────────────────────────────────────────────────────────

class TestCatalogFreshTTL:
    def test_fresh_cache_served_from_disk(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
        """Within TTL: load from disk, skip network."""
        monkeypatch.delenv("SQUISH_OFFLINE", raising=False)
        monkeypatch.setattr(_cat, "_CATALOG_CACHE", None)
        cache_file = tmp_path / "catalog.json"
        _write_catalog(cache_file)
        # Make the file appear fresh (mtime = now)
        monkeypatch.setattr(_cat, "LOCAL_CATALOG_PATH", cache_file)
        monkeypatch.setattr(_cat, "CATALOG_TTL", 9999)  # very long TTL
        catalog = _cat._try_refresh_catalog({})
        assert "qwen3:8b" in catalog

    def test_fresh_cache_with_malformed_json(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
        """Within TTL but malformed JSON: falls through to background refresh."""
        monkeypatch.delenv("SQUISH_OFFLINE", raising=False)
        monkeypatch.setattr(_cat, "_CATALOG_CACHE", None)
        cache_file = tmp_path / "catalog.json"
        cache_file.write_text("{bad json")
        monkeypatch.setattr(_cat, "LOCAL_CATALOG_PATH", cache_file)
        monkeypatch.setattr(_cat, "CATALOG_TTL", 9999)
        # Should not raise
        catalog = _cat._try_refresh_catalog({})
        assert isinstance(catalog, dict)


# ── load_catalog(refresh=True) ────────────────────────────────────────────────

class TestLoadCatalogRefresh:
    def test_refresh_unlinks_local_cache(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
        """load_catalog(refresh=True) should delete the local cache file."""
        monkeypatch.setattr(_cat, "_CATALOG_CACHE", None)
        cache_file = tmp_path / "catalog.json"
        _write_catalog(cache_file)
        assert cache_file.exists()
        monkeypatch.setattr(_cat, "LOCAL_CATALOG_PATH", cache_file)
        monkeypatch.setattr(_cat, "CATALOG_TTL", 0)  # force stale
        # threading is imported locally in _try_refresh_catalog, so patch threading.Thread directly
        with patch("threading.Thread") as mock_thread_cls:
            mock_thread_cls.return_value = MagicMock()
            _cat.load_catalog(refresh=True)
        assert not cache_file.exists()


# ── list_catalog sort_key ValueError ──────────────────────────────────────────

class TestListCatalogSortKey:
    def test_unknown_param_format_sorts_last(self, monkeypatch: pytest.MonkeyPatch):
        """Params like '??' have no recognized unit → sort key returns 9999."""
        monkeypatch.setattr(_cat, "_CATALOG_CACHE", None)
        entry = _cat.CatalogEntry(
            id="test:weird",
            name="Weird Model",
            hf_mlx_repo="x/y",
            size_gb=1.0,
            squished_size_gb=0.3,
            params="??",
            context=4096,
        )
        monkeypatch.setattr(_cat, "_CATALOG_CACHE",
                            {"test:weird": entry})
        entries = _cat.list_catalog()
        assert any(e.id == "test:weird" for e in entries)

    def test_param_with_float_conversion_error(self, monkeypatch: pytest.MonkeyPatch):
        """Params like 'XB' where X is not floatable triggers ValueError in sort_key."""
        monkeypatch.setattr(_cat, "_CATALOG_CACHE", None)
        entry = _cat.CatalogEntry(
            id="test:bad",
            name="Bad Model",
            hf_mlx_repo="x/y",
            size_gb=1.0,
            squished_size_gb=0.3,
            params="NANB",  # ends with B but 'NAN' is valid float → no error
            context=4096,
        )
        # Use a params that will fail float() conversion
        entry2 = _cat.CatalogEntry(
            id="test:bad2",
            name="Bad Model 2",
            hf_mlx_repo="x/y",
            size_gb=1.0,
            squished_size_gb=0.3,
            params="!!B",  # ends with B but '!!' is not floatable → ValueError
            context=4096,
        )
        monkeypatch.setattr(_cat, "_CATALOG_CACHE",
                            {"test:bad": entry, "test:bad2": entry2})
        entries = _cat.list_catalog()
        # Just verify it doesn't crash and returns both
        assert len(entries) == 2


# ── _has_squish_weights ───────────────────────────────────────────────────────

class TestHasSquishWeights:
    def test_returns_true_for_squish_weights_npz(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(_cat, "_hf_list_files", lambda repo, token=None: ["squish_weights.npz"])
        assert _cat._has_squish_weights("some/repo") is True

    def test_returns_true_for_squish_npy_dir(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(_cat, "_hf_list_files", lambda repo, token=None: ["squish_npy/weights.npy"])
        assert _cat._has_squish_weights("some/repo") is True

    def test_returns_false_when_no_squish_weights(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(_cat, "_hf_list_files", lambda repo, token=None: ["config.json", "model.safetensors"])
        assert _cat._has_squish_weights("some/repo") is False

    def test_empty_file_list(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(_cat, "_hf_list_files", lambda repo, token=None: [])
        assert _cat._has_squish_weights("some/repo") is False


# ── Stale cache load path ─────────────────────────────────────────────────────

class TestStaleCacheLoad:
    def test_stale_cache_loads_while_refreshing(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
        """Stale cache: should load from disk immediately while background thread fetches."""
        monkeypatch.delenv("SQUISH_OFFLINE", raising=False)
        monkeypatch.setattr(_cat, "_CATALOG_CACHE", None)
        cache_file = tmp_path / "catalog.json"
        _write_catalog(cache_file)
        # Make the file appear stale (mtime = unix epoch)
        os.utime(cache_file, (0, 0))
        monkeypatch.setattr(_cat, "LOCAL_CATALOG_PATH", cache_file)
        monkeypatch.setattr(_cat, "CATALOG_TTL", 1)  # 1 second TTL
        # threading is imported locally in _try_refresh_catalog, so patch threading.Thread directly
        with patch("threading.Thread") as mock_thread_cls:
            mock_thread_cls.return_value = MagicMock()
            catalog = _cat._try_refresh_catalog({})
        # Should have loaded from stale cache
        assert "qwen3:8b" in catalog
