"""tests/test_semantic_cache_unit.py

Unit tests for squish/semantic_cache.py — 100% line + branch coverage.

Strategy
--------
The ``SquishSemanticCache`` class requires sqlite-vec, which needs SQLite to be
compiled with ``SQLITE_ENABLE_LOAD_EXTENSION`` — unavailable in the stock macOS
Python.  Instead of a real extension, tests use ``_FakeSqliteConn``:

* Delegates all *responses* table SQL to a real in-memory SQLite connection.
* Intercepts ``CREATE VIRTUAL TABLE … USING vec0`` (no-op fake table).
* Fakes ``cache_vec`` INSERT and DELETE by storing raw embedding blobs in a
  plain Python dict.
* Implements the KNN lookup by computing cosine distances in numpy, matching
  the semantics of ``vec_distance_cosine``.
* Stubs ``enable_load_extension`` and ``load_extension`` to satisfy
  ``sqlite_vec.load(conn)`` without actually loading the C extension.
"""
from __future__ import annotations

import pathlib
import sqlite3 as _stdlib_sqlite3  # saved before any patches
import sys
import time
import unittest.mock as mock

import numpy as np
import pytest

from squish.semantic_cache import (
    _EMBED_DIM,
    _word_hash_embed,
    SquishSemanticCache,
)

# Keep a reference to the real sqlite3.connect so _FakeSqliteConn can use it
# even when squish.semantic_cache.sqlite3.connect is patched.
_REAL_CONNECT = _stdlib_sqlite3.connect


# ── Fake sqlite3 infrastructure ───────────────────────────────────────────────

class _FakeCursor:
    """Minimal cursor-like object returned for vec0 / cache_vec operations."""

    def __init__(self, rows=()):
        self.lastrowid = None
        self._rows = list(rows)

    def fetchall(self):
        return self._rows

    def fetchone(self):
        return self._rows[0] if self._rows else None


class _FakeSqliteConn:
    """
    Drop-in replacement for ``sqlite3.Connection`` that satisfies
    ``SquishSemanticCache.__init__`` without requiring SQLite extension support.

    * Real SQL (responses table, indexes) is forwarded to an in-memory SQLite.
    * ``cache_vec`` operations are handled in Python.
    * Extension stubs allow ``sqlite_vec.load(conn)`` to succeed silently.
    """

    def __init__(self):
        self._real = _REAL_CONNECT(":memory:", check_same_thread=False)
        self._real.row_factory = _stdlib_sqlite3.Row
        self._vecs: dict[int, bytes] = {}  # rowid → raw float32 blob

        # Pre-create the responses table so IF NOT EXISTS is idempotent
        self._real.execute(
            """
            CREATE TABLE IF NOT EXISTS responses (
                id        INTEGER PRIMARY KEY AUTOINCREMENT,
                task_type TEXT    NOT NULL DEFAULT 'default',
                prompt    TEXT    NOT NULL,
                response  TEXT    NOT NULL,
                stored_at REAL    NOT NULL,
                ttl_hours REAL    NOT NULL DEFAULT 48
            )
            """
        )
        self._real.execute(
            "CREATE INDEX IF NOT EXISTS idx_responses_task_stored"
            " ON responses (task_type, stored_at)"
        )
        self._real.commit()

    # ── Property delegation ────────────────────────────────────────────────

    @property
    def row_factory(self):
        return self._real.row_factory

    @row_factory.setter
    def row_factory(self, value):
        self._real.row_factory = value

    # ── Stubs for sqlite-vec extension loading ────────────────────────────

    def enable_load_extension(self, flag: bool) -> None:  # pragma: no cover
        pass

    def load_extension(self, path: str) -> None:  # pragma: no cover
        pass

    # ── SQL router ────────────────────────────────────────────────────────

    def execute(self, sql, params=()):
        sql_norm = " ".join(sql.upper().split())

        # CREATE VIRTUAL TABLE … USING vec0 → skip
        if "VEC0" in sql_norm:
            return _FakeCursor()

        # All cache_vec DML / queries
        if "CACHE_VEC" in sql_norm:
            return self._vec_execute(sql_norm, params)

        # Everything else → real in-memory SQLite
        return self._real.execute(sql, params)

    def _vec_execute(self, sql_norm: str, params) -> _FakeCursor:
        if "INSERT INTO CACHE_VEC" in sql_norm:
            rowid, blob = int(params[0]), bytes(params[1])
            self._vecs[rowid] = blob
            cur = _FakeCursor()
            cur.lastrowid = rowid
            return cur

        if "MATCH" in sql_norm and "EMBEDDING" in sql_norm:
            # KNN: compute cosine distances with numpy (1 − dot product between
            # unit vectors equals vec_distance_cosine semantics).
            q_blob = bytes(params[0])
            q_vec = np.frombuffer(q_blob, dtype=np.float32)
            rows = []
            for rowid, blob in self._vecs.items():
                v = np.frombuffer(blob, dtype=np.float32)
                dist = float(1.0 - float(q_vec @ v))
                rows.append({"rowid": rowid, "distance": dist})
            rows.sort(key=lambda r: r["distance"])
            return _FakeCursor(rows[:20])

        if "DELETE FROM CACHE_VEC" in sql_norm:
            if "ROWID IN" in sql_norm:
                for rowid in params:
                    self._vecs.pop(int(rowid), None)
            elif "ROWID =" in sql_norm:
                self._vecs.pop(int(params[0]), None)
            else:
                self._vecs.clear()
            return _FakeCursor()

        return _FakeCursor()

    def commit(self):
        self._real.commit()

    def close(self):
        self._real.close()


# ── _word_hash_embed ──────────────────────────────────────────────────────────

class TestWordHashEmbed:
    def test_shape(self):
        assert _word_hash_embed("hello world").shape == (_EMBED_DIM,)

    def test_dtype_float32(self):
        assert _word_hash_embed("test").dtype == np.float32

    def test_unit_norm(self):
        vec = _word_hash_embed("the quick brown fox jumps over the lazy dog")
        assert abs(float(np.linalg.norm(vec)) - 1.0) < 1e-5

    def test_custom_dim(self):
        assert _word_hash_embed("hello", dim=64).shape == (64,)

    def test_empty_text_all_finite(self):
        # No n-grams → near-zero vector; must not crash.
        vec = _word_hash_embed("")
        assert np.all(np.isfinite(vec))

    def test_different_texts_differ(self):
        v1 = _word_hash_embed("buy wine online tonight")
        v2 = _word_hash_embed("molecular hydrogen orbital mechanics")
        assert float(v1 @ v2) < 0.99


# ── Test fixture & helpers ────────────────────────────────────────────────────

@pytest.fixture
def cache(tmp_path):
    """SquishSemanticCache backed by _FakeSqliteConn (no real vec0 extension)."""
    with mock.patch(
        "squish.semantic_cache.sqlite3.connect",
        side_effect=lambda *a, **kw: _FakeSqliteConn(),
    ):
        c = SquishSemanticCache(db_path=tmp_path / "test.db")
    yield c
    c._conn.close()


def _insert(
    cache: SquishSemanticCache,
    query: str,
    response: str,
    task_type: str = "default",
    stored_at_offset: float = 0.0,
    ttl_hours: float = 48.0,
) -> int:
    """Bypass store() to inject a row with an arbitrary stored_at timestamp."""
    vec = cache._embed(query[:1024])
    blob = cache._pack_vec(vec)
    now = time.time() + stored_at_offset
    cur = cache._conn.execute(
        "INSERT INTO responses (task_type, prompt, response, stored_at, ttl_hours)"
        " VALUES (?, ?, ?, ?, ?)",
        (task_type, query[:4096], response, now, ttl_hours),
    )
    row_id = cur.lastrowid
    cache._conn.execute(
        "INSERT INTO cache_vec (rowid, embedding) VALUES (?, ?)", (row_id, blob)
    )
    cache._conn.commit()
    return row_id


# ── SquishSemanticCache.__init__ ──────────────────────────────────────────────

_CONN_PATCH = "squish.semantic_cache.sqlite3.connect"


class TestInit:
    def test_raises_import_error_without_sqlite_vec(self, tmp_path):
        """ImportError from __init__ when sqlite_vec is absent."""
        with mock.patch.dict(sys.modules, {"sqlite_vec": None}):
            with pytest.raises(ImportError, match="sqlite-vec is required"):
                SquishSemanticCache(db_path=tmp_path / "err.db")

    def test_default_db_path_created(self, tmp_path):
        """When db_path='' the cache uses ~/.squish/response_cache.db."""
        with mock.patch.object(
            pathlib.Path, "home", new=mock.Mock(return_value=tmp_path)
        ):
            with mock.patch(_CONN_PATCH, side_effect=lambda *a, **kw: _FakeSqliteConn()):
                c = SquishSemanticCache()  # db_path="" → default-path branch
        names = {
            r[0]
            for r in c._conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        assert "responses" in names
        c._conn.close()

    def test_explicit_db_path_initialises(self, tmp_path):
        """Explicit db_path is accepted and default config is applied."""
        with mock.patch(_CONN_PATCH, side_effect=lambda *a, **kw: _FakeSqliteConn()) as mock_c:
            c = SquishSemanticCache(db_path=tmp_path / "cache.db")
        # sqlite3.connect should have been called with the explicit path
        mock_c.assert_called_once_with(str(tmp_path / "cache.db"), check_same_thread=False)
        assert "default" in c._config
        c._conn.close()

    def test_explicit_config_stored(self, tmp_path):
        cfg = {
            "mytype": {"threshold": 0.95, "ttl_hours": 12},
            "default": {"threshold": 0.8, "ttl_hours": 6},
        }
        with mock.patch(_CONN_PATCH, side_effect=lambda *a, **kw: _FakeSqliteConn()):
            c = SquishSemanticCache(db_path=tmp_path / "cfg.db", config=cfg)
        assert c._config["mytype"]["threshold"] == 0.95
        c._conn.close()


# ── _embed and _pack_vec ──────────────────────────────────────────────────────

class TestEmbedAndPackVec:
    def test_embed_returns_unit_vector(self, cache):
        v = cache._embed("recommend a good red wine")
        assert abs(float(np.linalg.norm(v)) - 1.0) < 1e-5

    def test_pack_vec_byte_length(self, cache):
        v = cache._embed("test")
        blob = cache._pack_vec(v)
        # float32 = 4 bytes per element
        assert len(blob) == _EMBED_DIM * 4


# ── store ─────────────────────────────────────────────────────────────────────

class TestStore:
    def test_store_inserts_one_row(self, cache):
        cache.store("what wine pairs with fish?", "Try a Sauvignon Blanc.")
        n = cache._conn.execute("SELECT COUNT(*) FROM responses").fetchone()[0]
        assert n == 1

    def test_store_unknown_task_type_falls_back_to_default_ttl(self, tmp_path):
        """task_type not in config → 'or self._config["default"]' branch."""
        with mock.patch(_CONN_PATCH, side_effect=lambda *a, **kw: _FakeSqliteConn()):
            c = SquishSemanticCache(
                db_path=tmp_path / "s.db",
                config={"default": {"threshold": 0.9, "ttl_hours": 72}},
            )
        c.store("prompt text", "model response", task_type="unknown_task")
        row = c._conn.execute("SELECT ttl_hours FROM responses").fetchone()
        assert row[0] == 72.0
        c._conn.close()


# ── lookup ────────────────────────────────────────────────────────────────────

class TestLookup:
    def test_empty_cache_returns_none(self, cache):
        """No rows in cache_vec → candidates = [] → first early-return."""
        assert cache.lookup("any query") is None

    def test_returns_cached_response(self, cache):
        cache.store("recommend red wine for steak", "Try a Cabernet Sauvignon.")
        result = cache.lookup("recommend red wine for steak")
        assert result == "Try a Cabernet Sauvignon."

    def test_no_close_match_returns_none(self, tmp_path):
        """Candidate exists in vec table but distance > max_cos_dist → close_ids = []."""
        # Very high threshold → max_cos_dist is tiny.
        with mock.patch(_CONN_PATCH, side_effect=lambda *a, **kw: _FakeSqliteConn()):
            c = SquishSemanticCache(
                db_path=tmp_path / "hi.db",
                config={"default": {"threshold": 0.9999, "ttl_hours": 48}},
            )
        c.store("red wine pairing with beef tenderloin", "Cabernet Sauvignon")
        # Completely different domain → cosine distance >> 0.0001
        result = c.lookup("orbital mechanics of Jupiter moons Europa", "default")
        assert result is None
        c._conn.close()

    def test_task_type_mismatch_returns_none(self, cache):
        """close_ids non-empty but no row matches the requested task_type."""
        # Store as "typeA"; exact same text looked up as "typeB".
        _insert(cache, "wine recommendation for salmon", "Pinot Grigio", task_type="typeA")
        # "typeB" not in default config → falls back to default threshold (0.92)
        # distance = 0 (identical embedding) → close_ids is non-empty
        # but WHERE task_type = "typeB" finds nothing → rows = []
        result = cache.lookup("wine recommendation for salmon", "typeB")
        assert result is None

    def test_expired_entry_returns_none_and_deletes_row(self, cache):
        """TTL elapsed → _delete_row() called, None returned."""
        # stored 2 hours ago, TTL = 1 hour → expired
        _insert(
            cache,
            "best champagne for celebration",
            "Moët & Chandon",
            ttl_hours=1.0,
            stored_at_offset=-7200.0,
        )
        result = cache.lookup("best champagne for celebration")
        assert result is None
        n = cache._conn.execute("SELECT COUNT(*) FROM responses").fetchone()[0]
        assert n == 0  # _delete_row removed it

    def test_lookup_unknown_task_type_uses_default_config(self, tmp_path):
        """Config fallback: task_type not in config → uses default threshold."""
        with mock.patch(_CONN_PATCH, side_effect=lambda *a, **kw: _FakeSqliteConn()):
            c = SquishSemanticCache(
                db_path=tmp_path / "u.db",
                config={"default": {"threshold": 0.9, "ttl_hours": 48}},
            )
        c.store("find white wine", "Chardonnay works well.", task_type="default")
        # "unknown_type" not in config → or-branch uses default threshold
        # Same text → distance 0 → close_ids non-empty
        # But WHERE task_type = "unknown_type" → no rows → None
        result = c.lookup("find white wine", "unknown_type")
        assert result is None
        c._conn.close()


# ── evict_expired ─────────────────────────────────────────────────────────────

class TestEvictExpired:
    def test_no_expired_returns_zero(self, cache):
        """if ids: branch — False path (no expired entries)."""
        cache.store("fresh query", "fresh response")
        assert cache.evict_expired() == 0

    def test_empty_cache_returns_zero(self, cache):
        assert cache.evict_expired() == 0

    def test_expired_entries_removed(self, cache):
        """if ids: branch — True path, deletes from both tables."""
        _insert(
            cache,
            "stale query",
            "stale response",
            ttl_hours=1.0,
            stored_at_offset=-7200.0,
        )
        n = cache.evict_expired()
        assert n == 1
        assert cache._conn.execute("SELECT COUNT(*) FROM responses").fetchone()[0] == 0


# ── stats ─────────────────────────────────────────────────────────────────────

class TestStats:
    def test_empty_cache_stats(self, cache):
        s = cache.stats()
        assert s["total_entries"] == 0
        assert s["by_task_type"] == {}

    def test_stats_with_multiple_types(self, cache):
        cache.store("q1", "r1", task_type="code_review")
        cache.store("q2", "r2", task_type="code_review")
        cache.store("q3", "r3", task_type="git_commit")
        s = cache.stats()
        assert s["total_entries"] == 3
        assert s["by_task_type"]["code_review"] == 2
        assert s["by_task_type"]["git_commit"] == 1


# ── clear ─────────────────────────────────────────────────────────────────────

class TestClear:
    def test_clear_removes_all_entries(self, cache):
        cache.store("q1", "r1")
        cache.store("q2", "r2")
        cache.clear()
        assert cache._conn.execute("SELECT COUNT(*) FROM responses").fetchone()[0] == 0

    def test_clear_idempotent_on_empty(self, cache):
        cache.clear()  # must not raise
        assert cache._conn.execute("SELECT COUNT(*) FROM responses").fetchone()[0] == 0


# ── _delete_row ───────────────────────────────────────────────────────────────

class TestDeleteRow:
    def test_deletes_specific_row(self, cache):
        row_id = _insert(cache, "delete me", "response to delete")
        cache._delete_row(row_id)
        # responses table (real sqlite) should be empty
        assert cache._conn.execute("SELECT COUNT(*) FROM responses").fetchone()[0] == 0
        # cache_vec (_vecs dict in fake conn) should also be empty
        assert cache._conn._vecs == {}
