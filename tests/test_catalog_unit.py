"""
tests/test_catalog_unit.py

Unit tests for squish/catalog.py.
Covers CatalogEntry properties, _entry_from_dict, list_catalog, resolve,
and alias lookup.
"""
from __future__ import annotations

import pytest

from squish.catalog import (
    CatalogEntry,
    _ALIASES,
    _entry_from_dict,
    list_catalog,
    resolve,
)


# ── CatalogEntry properties ────────────────────────────────────────────────────

class TestCatalogEntry:
    def _make(self, **kwargs) -> CatalogEntry:
        defaults = dict(
            id="test:7b",
            name="Test Model 7B",
            hf_mlx_repo="mlx-community/test-model-7b-bf16",
            size_gb=14.4,
            params="7B",
            context=131072,
            squished_size_gb=3.9,
            squish_repo=None,
            tags=["balanced"],
            notes="",
        )
        defaults.update(kwargs)
        return CatalogEntry(**defaults)

    def test_dir_name_from_hf_repo(self):
        e = self._make(hf_mlx_repo="mlx-community/Qwen3-8B-bf16")
        assert e.dir_name == "Qwen3-8B-bf16"

    def test_dir_name_no_org(self):
        e = self._make(hf_mlx_repo="someorg/my-model-bf16")
        assert e.dir_name == "my-model-bf16"

    def test_has_prebuilt_true(self):
        e = self._make(squish_repo="squish-community/test-squished")
        assert e.has_prebuilt is True

    def test_has_prebuilt_false(self):
        e = self._make(squish_repo=None)
        assert e.has_prebuilt is False

    def test_str_contains_id(self):
        e = self._make(id="mymodel:7b")
        assert "mymodel:7b" in str(e)

    def test_str_contains_params(self):
        e = self._make(params="14B")
        assert "14B" in str(e)

    def test_str_contains_size(self):
        e = self._make(size_gb=28.2)
        assert "28.2" in str(e) or "28" in str(e)

    def test_str_prebuilt_marker(self):
        e = self._make(squish_repo="squish-community/x")
        assert "prebuilt" in str(e)

    def test_repr_is_string(self):
        e = self._make()
        assert isinstance(repr(e), str)

    def test_tags_list(self):
        e = self._make(tags=["small", "fast"])
        assert "small" in e.tags
        assert "fast" in e.tags

    def test_context_stored(self):
        e = self._make(context=8192)
        assert e.context == 8192

    def test_id_stored(self):
        e = self._make(id="gemma3:4b")
        assert e.id == "gemma3:4b"


# ── _entry_from_dict ──────────────────────────────────────────────────────────

class TestEntryFromDict:
    def _base_dict(self, **overrides):
        d = {
            "id": "model:7b",
            "name": "Test Model",
            "hf_mlx_repo": "mlx-community/test-bf16",
            "size_gb": 7.0,
            "params": "7B",
            "context": 4096,
            "squished_size_gb": 2.0,
        }
        d.update(overrides)
        return d

    def test_returns_catalog_entry(self):
        e = _entry_from_dict(self._base_dict())
        assert isinstance(e, CatalogEntry)

    def test_fields_set(self):
        e = _entry_from_dict(self._base_dict(id="foo:1b", size_gb=3.5))
        assert e.id == "foo:1b"
        assert e.size_gb == 3.5

    def test_squish_repo_none(self):
        e = _entry_from_dict(self._base_dict())
        assert e.squish_repo is None
        assert e.has_prebuilt is False

    def test_squish_repo_set(self):
        e = _entry_from_dict(self._base_dict(squish_repo="squish-community/x"))
        assert e.has_prebuilt is True

    def test_tags_default_empty(self):
        e = _entry_from_dict(self._base_dict())
        assert isinstance(e.tags, list)

    def test_tags_populated(self):
        e = _entry_from_dict(self._base_dict(tags=["small", "fast"]))
        assert "small" in e.tags


# ── list_catalog ──────────────────────────────────────────────────────────────

class TestListCatalog:
    def test_returns_list(self):
        entries = list_catalog()
        assert isinstance(entries, list)

    def test_all_catalog_entries(self):
        for e in list_catalog():
            assert isinstance(e, CatalogEntry)

    def test_non_empty(self):
        assert len(list_catalog()) > 10

    def test_tag_filter_small(self):
        entries = list_catalog(tag="small")
        assert len(entries) > 0
        for e in entries:
            assert "small" in e.tags

    def test_tag_filter_large(self):
        entries = list_catalog(tag="large")
        for e in entries:
            assert "large" in e.tags

    def test_tag_filter_unknown_empty(self):
        entries = list_catalog(tag="definitely_not_a_real_tag_xyz")
        assert entries == []

    def test_has_size_gb(self):
        for e in list_catalog():
            assert isinstance(e.size_gb, (int, float))
            assert e.size_gb > 0

    def test_has_hf_mlx_repo(self):
        for e in list_catalog():
            assert isinstance(e.hf_mlx_repo, str)
            assert "/" in e.hf_mlx_repo

    def test_has_name(self):
        for e in list_catalog():
            assert isinstance(e.name, str)
            assert len(e.name) > 0

    def test_has_id(self):
        for e in list_catalog():
            assert isinstance(e.id, str)
            assert ":" in e.id

    def test_sort_stability(self):
        a = list_catalog()
        b = list_catalog()
        assert [e.id for e in a] == [e.id for e in b]

    def test_contains_qwen(self):
        all_ids = [e.id for e in list_catalog()]
        assert any("qwen" in i.lower() for i in all_ids)

    def test_contains_gemma(self):
        all_ids = [e.id for e in list_catalog()]
        assert any("gemma" in i.lower() for i in all_ids)


# ── resolve ──────────────────────────────────────────────────────────────────

class TestResolve:
    def test_resolves_canonical_id(self):
        entries = list_catalog()
        e = entries[0]
        resolved = resolve(e.id)
        assert resolved is not None
        assert resolved.id == e.id

    def test_returns_none_for_unknown(self):
        result = resolve("totally/unknown-model-xyz-99999")
        assert result is None

    def test_resolves_to_catalog_entry(self):
        entries = list_catalog()
        e = entries[0]
        result = resolve(e.id)
        assert isinstance(result, CatalogEntry)

    def test_qwen3_8b_resolves(self):
        result = resolve("qwen3:8b")
        assert result is not None
        assert result.params == "8B"

    def test_gemma3_4b_resolves(self):
        result = resolve("gemma3:4b")
        assert result is not None

    def test_alias_lookup(self):
        if not _ALIASES:
            pytest.skip("No aliases defined")
        alias_key = next(iter(_ALIASES))
        result = resolve(alias_key)
        if result is not None:
            assert isinstance(result, CatalogEntry)


# ── _ALIASES dict ─────────────────────────────────────────────────────────────

class TestAliases:
    def test_aliases_is_dict(self):
        assert isinstance(_ALIASES, dict)

    def test_alias_values_are_strings(self):
        for k, v in _ALIASES.items():
            assert isinstance(k, str)
            assert isinstance(v, str)

    def test_aliases_keys_lowercase(self):
        for k in _ALIASES:
            assert k == k.lower(), f"Alias key not lowercase: {k!r}"

    def test_aliases_resolve(self):
        for alias in _ALIASES:
            result = resolve(alias)
            if result is not None:
                assert isinstance(result, CatalogEntry)
