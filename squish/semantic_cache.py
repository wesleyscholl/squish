"""
squish/semantic_cache.py

Semantic response cache backed by sqlite-vec.

Stores model responses keyed by their prompt embedding so that prompts
semantically similar to a cached prompt (cosine similarity ≥ per-task
threshold) return the cached response without running the model.

Install the optional dependency::

    pip install squish[cache]       # adds sqlite-vec

or manually::

    pip install sqlite-vec

Embedding
---------
Uses a lightweight word / bi-gram hashing-trick encoding that requires no
additional ML model.  The prompt is split into words and bi-grams; each
n-gram is hashed to a 384-dim index; the resulting count vector is
L2-normalised to unit length.  Cosine similarity is then the inner
product of two unit vectors, which sqlite-vec computes as::

    cosine_similarity = 1 - vec_distance_cosine(a, b)

For production use, replace ``_embed`` with a sentence-transformer or
MLX embedding model by subclassing and overriding that method.
"""
from __future__ import annotations

import hashlib
import math
import pathlib
import sqlite3
import struct
import time
from typing import Optional

import numpy as np

# Vector dimension — must match the CREATE VIRTUAL TABLE statement.
_EMBED_DIM: int = 384

# ── Lightweight word-bigram hashing trick ────────────────────────────────────

def _word_hash_embed(text: str, dim: int = _EMBED_DIM) -> np.ndarray:
    """
    Lightweight bag-of-n-grams hashing embedding → L2-normalised float32.

    No ML model required.  Works well for prompts that share vocabulary
    (same task domain), which is the dominant use case for this cache.
    """
    vec = np.zeros(dim, dtype=np.float64)
    words = text.lower().split()
    # unigrams + bigrams for richer overlap
    ngrams = words + [f"{a}_{b}" for a, b in zip(words, words[1:])]
    for gram in ngrams:
        digest = hashlib.md5(gram.encode(), usedforsecurity=False).digest()
        # Use two 32-bit slices to pick two indices → double-count each gram
        idx_a = struct.unpack_from("<I", digest, 0)[0] % dim
        idx_b = struct.unpack_from("<I", digest, 4)[0] % dim
        vec[idx_a] += 1.0
        vec[idx_b] += 0.5
    norm = math.sqrt(float(vec @ vec)) + 1e-9
    return (vec / norm).astype(np.float32)


# ── Schema ────────────────────────────────────────────────────────────────────

_CREATE_RESPONSES_SQL = """
CREATE TABLE IF NOT EXISTS responses (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    task_type TEXT    NOT NULL DEFAULT 'default',
    prompt    TEXT    NOT NULL,
    response  TEXT    NOT NULL,
    stored_at REAL    NOT NULL,
    ttl_hours REAL    NOT NULL DEFAULT 48
)
"""

_CREATE_VEC_SQL = f"""
CREATE VIRTUAL TABLE IF NOT EXISTS cache_vec
USING vec0(
    embedding float[{_EMBED_DIM}]
)
"""

# Index to speed up task_type + TTL filtering
_CREATE_IDX_SQL = (
    "CREATE INDEX IF NOT EXISTS idx_responses_task_stored "
    "ON responses (task_type, stored_at)"
)


class SquishSemanticCache:
    """
    sqlite-vec backed semantic response cache.

    Parameters
    ----------
    db_path:
        Path to the SQLite database file (created on first use).
        Defaults to ``~/.squish/response_cache.db``.
    config:
        Per-task-type ``{"threshold": float, "ttl_hours": float}`` dict.
        The ``"default"`` key is the fallback.
    """

    def __init__(
        self,
        db_path: "str | pathlib.Path" = "",
        config: Optional[dict] = None,
    ) -> None:
        try:
            import sqlite_vec as _sqlite_vec  # type: ignore[import]
            self._sqlite_vec = _sqlite_vec
        except ImportError as exc:
            raise ImportError(
                "sqlite-vec is required for the semantic response cache.\n"
                "Install with:  pip install sqlite-vec\n"
                "Or:            pip install 'squish[cache]'"
            ) from exc

        if not db_path:
            db_path = pathlib.Path.home() / ".squish" / "response_cache.db"
        db_path = pathlib.Path(db_path)
        db_path.parent.mkdir(parents=True, exist_ok=True)

        self._config: dict = config or {
            "git_commit":  {"threshold": 0.95, "ttl_hours": 24},
            "devops_plan": {"threshold": 0.88, "ttl_hours": 168},
            "code_review": {"threshold": 0.92, "ttl_hours": 72},
            "email_draft": {"threshold": 0.85, "ttl_hours": 48},
            "default":     {"threshold": 0.92, "ttl_hours": 48},
        }

        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row

        # Load the sqlite-vec extension
        self._conn.enable_load_extension(True)
        self._sqlite_vec.load(self._conn)
        self._conn.enable_load_extension(False)

        self._conn.execute(_CREATE_RESPONSES_SQL)
        self._conn.execute(_CREATE_VEC_SQL)
        self._conn.execute(_CREATE_IDX_SQL)
        self._conn.commit()

    # ── Embedding ─────────────────────────────────────────────────────────────

    def _embed(self, text: str) -> np.ndarray:
        """Return a float32 L2-normalised vector for *text*."""
        return _word_hash_embed(text)

    @staticmethod
    def _pack_vec(vec: np.ndarray) -> bytes:
        """Pack a float32 numpy array into the blob expected by sqlite-vec."""
        return struct.pack(f"{len(vec)}f", *vec.astype(np.float32).tolist())

    # ── Public API ────────────────────────────────────────────────────────────

    def lookup(self, query: str, task_type: str = "default") -> Optional[str]:
        """
        Return a cached response string if a semantically similar entry
        exists and has not yet expired; otherwise return ``None``.

        Parameters
        ----------
        query:     The full prompt text.
        task_type: Task-type key for threshold selection.
        """
        cfg           = self._config.get(task_type) or self._config["default"]
        threshold     = float(cfg["threshold"])
        # cosine_distance = 1 - cosine_similarity; lower is more similar
        max_cos_dist  = 1.0 - threshold
        now           = time.time()

        q_vec  = self._embed(query[:1024])
        q_blob = self._pack_vec(q_vec)

        # Phase 1: approximate KNN from the vec0 index (fast path, top-20 candidates)
        candidates = self._conn.execute(
            """
            SELECT rowid, distance
              FROM cache_vec
             WHERE embedding MATCH ?
               AND k = 20
             ORDER BY distance
            """,
            (q_blob,),
        ).fetchall()

        if not candidates:
            return None

        # Phase 2: filter by cosine threshold
        close_ids = [
            row["rowid"] for row in candidates
            if row["distance"] <= max_cos_dist
        ]
        if not close_ids:
            return None

        # Phase 3: retrieve metadata and apply task_type + TTL filters
        ph = ",".join("?" * len(close_ids))
        rows = self._conn.execute(
            f"""
            SELECT id, response, stored_at, ttl_hours
              FROM responses
             WHERE id IN ({ph})
               AND task_type = ?
             ORDER BY stored_at DESC
             LIMIT 1
            """,
            close_ids + [task_type],
        ).fetchall()

        if not rows:
            return None

        row = rows[0]
        if now - row["stored_at"] > row["ttl_hours"] * 3600:
            self._delete_row(row["id"])
            return None

        return row["response"]

    def store(self, query: str, response: str, task_type: str = "default") -> None:
        """
        Persist a ``query`` → ``response`` pair in the cache.

        Parameters
        ----------
        query:     The original prompt text.
        response:  The full model response text.
        task_type: Task-type key for TTL selection.
        """
        cfg       = self._config.get(task_type) or self._config["default"]
        ttl_hours = float(cfg["ttl_hours"])
        now       = time.time()

        q_vec  = self._embed(query[:1024])
        q_blob = self._pack_vec(q_vec)

        cur = self._conn.execute(
            """
            INSERT INTO responses (task_type, prompt, response, stored_at, ttl_hours)
            VALUES (?, ?, ?, ?, ?)
            """,
            (task_type, query[:4096], response, now, ttl_hours),
        )
        row_id = cur.lastrowid
        self._conn.execute(
            "INSERT INTO cache_vec (rowid, embedding) VALUES (?, ?)",
            (row_id, q_blob),
        )
        self._conn.commit()

    def evict_expired(self) -> int:
        """
        Delete all entries whose TTL has elapsed.

        Returns the number of rows removed.
        """
        now = time.time()
        expired = self._conn.execute(
            "SELECT id FROM responses WHERE (? - stored_at) > ttl_hours * 3600",
            (now,),
        ).fetchall()
        ids = [r["id"] for r in expired]
        if ids:
            ph = ",".join("?" * len(ids))
            self._conn.execute(f"DELETE FROM responses WHERE id IN ({ph})", ids)
            self._conn.execute(f"DELETE FROM cache_vec  WHERE rowid IN ({ph})", ids)
            self._conn.commit()
        return len(ids)

    def stats(self) -> dict:
        """Return basic statistics about the cache contents."""
        total = self._conn.execute(
            "SELECT COUNT(*) AS n FROM responses"
        ).fetchone()["n"]
        by_type = self._conn.execute(
            "SELECT task_type, COUNT(*) AS n FROM responses GROUP BY task_type"
        ).fetchall()
        return {
            "total_entries": total,
            "by_task_type":  {r["task_type"]: r["n"] for r in by_type},
        }

    def clear(self) -> None:
        """Remove all entries from the cache."""
        self._conn.execute("DELETE FROM responses")
        self._conn.execute("DELETE FROM cache_vec")
        self._conn.commit()

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _delete_row(self, row_id: int) -> None:
        """Remove a single expired entry."""
        self._conn.execute("DELETE FROM responses WHERE id = ?",    (row_id,))
        self._conn.execute("DELETE FROM cache_vec  WHERE rowid = ?", (row_id,))
        self._conn.commit()
