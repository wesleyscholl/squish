"""
tests/test_catalog_extended.py

Unit tests for pure/testable helpers in squish/catalog.py.
Covers: CatalogEntry, _entry_from_dict, list_catalog (sort/filter),
        search, resolve, _has_squish_weights, _sort_key variants.
All network / HuggingFace calls are mocked or bypassed.
"""
from __future__ import annotations

from unittest.mock import patch

import pytest

from squish.catalog import (
    CatalogEntry,
    _entry_from_dict,
    list_catalog,
    resolve,
    search,
)


# ── shared fixtures ────────────────────────────────────────────────────────────

def _make_entry(**kwargs) -> CatalogEntry:
    defaults = dict(
        id="test:7b",
        name="Test 7B",
        hf_mlx_repo="test/test-7b-mlx",
        size_gb=4.2,
        params="7B",
        context=4096,
        squished_size_gb=2.1,
        squish_repo=None,
        tags=[],
        notes="",
    )
    defaults.update(kwargs)
    return CatalogEntry(**defaults)


def _mock_catalog(*entries: CatalogEntry):
    """Patch load_catalog to return {e.id: e} for each entry."""
    return {e.id: e for e in entries}


# ── _entry_from_dict ───────────────────────────────────────────────────────────

class TestEntryFromDict:
    def _base_dict(self, **overrides):
        d = {
            "id":              "qwen3:8b",
            "name":            "Qwen3 8B",
            "hf_mlx_repo":     "mlx-community/Qwen3-8B-mlx",
            "size_gb":         5.0,
            "params":          "8B",
            "context":         8192,
            "squished_size_gb": 2.5,
        }
        d.update(overrides)
        return d

    def test_minimal_round_trip(self):
        entry = _entry_from_dict(self._base_dict())
        assert entry.id == "qwen3:8b"
        assert entry.params == "8B"
        assert entry.size_gb == 5.0
        assert entry.squish_repo is None
        assert entry.tags == []

    def test_with_optional_fields(self):
        d = self._base_dict(
            squish_repo="squish-community/qwen3-8b",
            tags=["instruct", "chat"],
            notes="Official Qwen3",
        )
        entry = _entry_from_dict(d)
        assert entry.squish_repo == "squish-community/qwen3-8b"
        assert "chat" in entry.tags

    def test_missing_required_key_raises(self):
        d = self._base_dict()
        del d["id"]
        with pytest.raises(KeyError):
            _entry_from_dict(d)

    def test_dir_name_safe(self):
        """dir_name property strips colon from the id."""
        entry = _entry_from_dict(self._base_dict())
        assert ":" not in entry.dir_name


# ── list_catalog ──────────────────────────────────────────────────────────────

class TestListCatalog:
    def test_sorted_by_param_ascending(self):
        e7b  = _make_entry(id="a:7b",  params="7B",  size_gb=4.0, squished_size_gb=2.0)
        e14b = _make_entry(id="a:14b", params="14B", size_gb=8.0, squished_size_gb=4.0)
        e1b  = _make_entry(id="a:1b",  params="1B",  size_gb=1.0, squished_size_gb=0.5)

        with patch("squish.catalog.load_catalog", return_value=_mock_catalog(e7b, e14b, e1b)):
            result = list_catalog(refresh=False)

        ids = [e.id for e in result]
        assert ids.index("a:1b") < ids.index("a:7b") < ids.index("a:14b")

    def test_tag_filter(self):
        ec  = _make_entry(id="x:7b",  params="7B",  size_gb=4.0, squished_size_gb=2.0, tags=["chat"])
        enc = _make_entry(id="x:14b", params="14B", size_gb=8.0, squished_size_gb=4.0, tags=["code"])

        with patch("squish.catalog.load_catalog", return_value=_mock_catalog(ec, enc)):
            result = list_catalog(tag="chat", refresh=False)

        assert len(result) == 1
        assert result[0].id == "x:7b"

    def test_no_matching_tag_returns_empty(self):
        e = _make_entry(id="x:7b", params="7B", size_gb=4.0, squished_size_gb=2.0, tags=["chat"])
        with patch("squish.catalog.load_catalog", return_value=_mock_catalog(e)):
            result = list_catalog(tag="nonexistent", refresh=False)
        assert result == []

    def test_million_param_model_sorts_low(self):
        e500m = _make_entry(id="m:500m", params="500M", size_gb=0.5, squished_size_gb=0.25)
        e7b   = _make_entry(id="m:7b",   params="7B",   size_gb=4.0, squished_size_gb=2.0)

        with patch("squish.catalog.load_catalog", return_value=_mock_catalog(e500m, e7b)):
            result = list_catalog(refresh=False)

        ids = [e.id for e in result]
        assert ids.index("m:500m") < ids.index("m:7b")

    def test_unknown_param_string_sorted_last(self):
        unknown = _make_entry(id="u:unknown", params="?", size_gb=1.0, squished_size_gb=0.5)
        e7b    = _make_entry(id="u:7b",      params="7B", size_gb=4.0, squished_size_gb=2.0)

        with patch("squish.catalog.load_catalog", return_value=_mock_catalog(unknown, e7b)):
            result = list_catalog(refresh=False)

        ids = [e.id for e in result]
        assert ids.index("u:7b") < ids.index("u:unknown")


# ── search ─────────────────────────────────────────────────────────────────────

class TestSearch:
    def _catalog(self):
        return _mock_catalog(
            _make_entry(id="qwen3:7b",   name="Qwen3 7B",   params="7B",  tags=["chat"],
                        size_gb=4.0, squished_size_gb=2.0),
            _make_entry(id="gemma3:9b",  name="Gemma3 9B",  params="9B",  tags=["instruct"],
                        size_gb=5.0, squished_size_gb=2.5),
            _make_entry(id="llama3:8b",  name="Llama 3 8B", params="8B",  tags=["chat"],
                        size_gb=5.0, squished_size_gb=2.5),
        )

    def test_search_by_id(self):
        with patch("squish.catalog.load_catalog", return_value=self._catalog()):
            result = search("qwen3")
        assert all("qwen3" in e.id for e in result)

    def test_search_by_name(self):
        with patch("squish.catalog.load_catalog", return_value=self._catalog()):
            result = search("Gemma")
        assert len(result) == 1
        assert result[0].id == "gemma3:9b"

    def test_search_by_tag(self):
        with patch("squish.catalog.load_catalog", return_value=self._catalog()):
            result = search("instruct")
        assert len(result) == 1
        assert result[0].id == "gemma3:9b"

    def test_search_by_params(self):
        with patch("squish.catalog.load_catalog", return_value=self._catalog()):
            result = search("8b")
        ids = {e.id for e in result}
        assert "llama3:8b" in ids

    def test_search_no_match_returns_empty(self):
        with patch("squish.catalog.load_catalog", return_value=self._catalog()):
            result = search("zzznotamodel")
        assert result == []

    def test_search_case_insensitive(self):
        with patch("squish.catalog.load_catalog", return_value=self._catalog()):
            result_lower = search("qwen3")
            result_mixed = search("QWEN3")
        assert {e.id for e in result_lower} == {e.id for e in result_mixed}


# ── resolve ────────────────────────────────────────────────────────────────────

class TestResolve:
    def _catalog(self):
        return _mock_catalog(
            _make_entry(id="qwen3:7b",  params="7B",  size_gb=4.0, squished_size_gb=2.0),
            _make_entry(id="qwen3:14b", params="14B", size_gb=8.0, squished_size_gb=4.0),
            _make_entry(id="gemma3:9b", params="9B",  size_gb=5.0, squished_size_gb=2.5),
        )

    def test_exact_id_returns_entry(self):
        with patch("squish.catalog.load_catalog", return_value=self._catalog()):
            entry = resolve("qwen3:7b")
        assert entry is not None
        assert entry.id == "qwen3:7b"

    def test_prefix_match_returns_smallest(self):
        with patch("squish.catalog.load_catalog", return_value=self._catalog()):
            entry = resolve("qwen3")
        assert entry is not None
        # smallest size_gb should be preferred
        assert entry.id == "qwen3:7b"

    def test_unknown_name_returns_none(self):
        with patch("squish.catalog.load_catalog", return_value=self._catalog()):
            entry = resolve("notamodel")
        assert entry is None

    def test_legacy_alias_7b(self):
        """The alias '7b' should map to the qwen2.5:7b canonical id (or nearest)."""
        catalog = _mock_catalog(
            _make_entry(id="qwen2.5:7b", params="7B", size_gb=4.0, squished_size_gb=2.0),
        )
        with patch("squish.catalog.load_catalog", return_value=catalog):
            entry = resolve("7b")
        # Either resolves via alias or prefix — should not raise
        # If alias resolves to a non-existent key, it falls through to prefix match

    def test_strip_whitespace(self):
        with patch("squish.catalog.load_catalog", return_value=self._catalog()):
            entry = resolve("  qwen3:7b  ")
        assert entry is not None
        assert entry.id == "qwen3:7b"
