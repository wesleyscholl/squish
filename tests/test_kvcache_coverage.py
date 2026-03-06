"""
tests/test_kvcache_coverage.py

Coverage booster for squish/kv_cache.py paths not hit by the existing suites:

  - SVD calibration path in KVLayerCache.append()
  - _svd_project() / _svd_fit_and_flush()
  - SVD back-projection branch in get_full_kv()
  - Async prefetch: _get_pool / start_prefetch / get_full_kv_prefetched
  - QuantizedKVCache.__init__ with svd_rank > 0
  - DiskKVCache._serialise with recent tokens AND SVD basis
  - DiskKVCache._deserialise with recent tokens AND SVD basis
  - DiskKVCache._evict_if_needed error-handler branch (pragma:no cover in code)
  - SessionKVCache: __init__, session_key, load_session, save_session,
    list_sessions, _evict_if_needed
"""
from __future__ import annotations

import time
import numpy as np
import pytest

from squish.kv_cache import (
    KVLayerCache,
    QuantizedKVCache,
    DiskKVCache,
    SessionKVCache,
    _SVD_INIT_TOKENS,
)

# ── helpers ───────────────────────────────────────────────────────────────────

N_HEADS  = 2
HEAD_DIM = 8


def _rand_kv(seed: int = 0):
    rng = np.random.default_rng(seed)
    k = rng.standard_normal((N_HEADS, HEAD_DIM)).astype(np.float16)
    v = rng.standard_normal((N_HEADS, HEAD_DIM)).astype(np.float16)
    return k, v


def _make_svd_layer(svd_rank: int = 4, window: int = 1) -> KVLayerCache:
    """
    Return a KVLayerCache that has completed SVD fit.
    Appends exactly _SVD_INIT_TOKENS + 1 tokens so the fit fires and one
    subsequent token is projected through the fitted basis.
    """
    layer = KVLayerCache(window=window)
    layer._svd_rank = svd_rank
    rng = np.random.default_rng(42)
    # One extra token triggers the flush; one more triggers svd_project path
    for i in range(_SVD_INIT_TOKENS + 2):
        k = rng.standard_normal((N_HEADS, HEAD_DIM)).astype(np.float16)
        v = rng.standard_normal((N_HEADS, HEAD_DIM)).astype(np.float16)
        layer.append(k, v)
    return layer


def _populated_qkv_with_recent(
    n_layers: int = 2, n_tokens: int = 3
) -> QuantizedKVCache:
    """
    Build a QuantizedKVCache with tokens in *both* the INT8 tier (keys_old_q)
    and the FP16 recent window, so _serialise exercises all branches.
    """
    # window=1 forces immediate eviction to INT8; n_tokens=3 → 2 in INT8, 1 recent
    cache = QuantizedKVCache(n_layers=n_layers, window=1, mode="int8")
    rng = np.random.default_rng(7)
    for i in range(n_layers):
        for _ in range(n_tokens):
            k = rng.standard_normal((N_HEADS, HEAD_DIM)).astype(np.float16)
            v = rng.standard_normal((N_HEADS, HEAD_DIM)).astype(np.float16)
            cache._layers[i].append(k, v)
    return cache


# ── SVD path in KVLayerCache ──────────────────────────────────────────────────

class TestSVDPath:

    def test_svd_fit_triggered_after_init_tokens(self):
        """_svd_fit_and_flush() fires once the buffer reaches _SVD_INIT_TOKENS."""
        layer = KVLayerCache(window=1)
        layer._svd_rank = 4
        rng = np.random.default_rng(0)
        # Append exactly _SVD_INIT_TOKENS + 1 tokens — the 65th eviction
        # fills the buffer to 64, triggering _svd_fit_and_flush().
        for i in range(_SVD_INIT_TOKENS + 1):
            k = rng.standard_normal((N_HEADS, HEAD_DIM)).astype(np.float16)
            v = rng.standard_normal((N_HEADS, HEAD_DIM)).astype(np.float16)
            layer.append(k, v)
        assert layer._svd_Vk is not None, "_svd_Vk should be fitted after flush"
        assert layer._svd_Vv is not None
        assert layer._svd_buf_k is None, "calibration buffer should be cleared"

    def test_svd_vk_shape(self):
        """Fitted SVD basis has shape (n_heads, rank, head_dim)."""
        layer = _make_svd_layer(svd_rank=4)
        assert layer._svd_Vk.shape == (N_HEADS, 4, HEAD_DIM)
        assert layer._svd_Vv.shape == (N_HEADS, 4, HEAD_DIM)

    def test_svd_projection_path_in_append(self):
        """After fit, subsequent evictions use _svd_project (not SVD buffer)."""
        layer = _make_svd_layer(svd_rank=4)
        # After _make_svd_layer, _svd_Vk is set; keys_old_q should contain
        # projected+quantized entries from svd_fit_and_flush + one more projection.
        assert layer.keys_old_q is not None
        # Shape: (n_heads, n_old_tokens, rank)  — rank < head_dim
        assert layer.keys_old_q.shape[2] == 4

    def test_get_full_kv_svd_backprojected(self):
        """get_full_kv() back-projects SVD-compressed entries to full head_dim."""
        layer = _make_svd_layer(svd_rank=4)
        k, v = layer.get_full_kv()
        assert k is not None
        assert k.shape[2] == HEAD_DIM, "back-projection should restore head_dim"

    def test_svd_project_directly(self):
        """_svd_project produces shape (n_heads, rank)."""
        layer = _make_svd_layer(svd_rank=4)
        k, _ = _rand_kv()
        projected = layer._svd_project(k, layer._svd_Vk)
        assert projected.shape == (N_HEADS, 4)
        assert projected.dtype == np.float16


# ── Async prefetch ────────────────────────────────────────────────────────────

class TestPrefetch:

    def _layer_with_int8(self):
        """A KVLayerCache with data in the INT8 tier (keys_old_q not None)."""
        layer = KVLayerCache(window=1)
        for i in range(3):
            k, v = _rand_kv(i)
            layer.append(k, v)
        return layer

    def test_get_pool_creates_executor(self):
        """_get_pool() returns a ThreadPoolExecutor on first call."""
        import concurrent.futures as cf
        KVLayerCache._THREAD_POOL = None   # reset class-level pool
        pool = KVLayerCache._get_pool()
        assert isinstance(pool, cf.ThreadPoolExecutor)
        # Second call returns same instance
        assert KVLayerCache._get_pool() is pool

    def test_start_prefetch_submits_future(self):
        """start_prefetch() sets _prefetch_future when keys_old_q exists."""
        layer = self._layer_with_int8()
        layer.start_prefetch()
        future = layer._prefetch_future
        assert future is not None

    def test_start_prefetch_noop_when_no_int8(self):
        """start_prefetch() does nothing when there's no INT8 data."""
        layer = KVLayerCache()
        layer.start_prefetch()
        assert layer._prefetch_future is None

    def test_start_prefetch_noop_if_already_inflight(self):
        """start_prefetch() is idempotent — won't submit twice."""
        layer = self._layer_with_int8()
        layer.start_prefetch()
        first_future = layer._prefetch_future
        layer.start_prefetch()   # second call should not replace the future
        assert layer._prefetch_future is first_future

    def test_get_full_kv_prefetched_with_future(self):
        """get_full_kv_prefetched() waits on the future and returns arrays."""
        layer = self._layer_with_int8()
        layer.start_prefetch()
        k, v = layer.get_full_kv_prefetched()
        assert k is not None
        assert layer._prefetch_future is None  # future consumed

    def test_get_full_kv_prefetched_without_future(self):
        """Falls back to synchronous get_full_kv() when no future is set."""
        layer = self._layer_with_int8()
        # Ensure no future
        assert layer._prefetch_future is None
        k, v = layer.get_full_kv_prefetched()
        assert k is not None


# ── QuantizedKVCache with svd_rank ────────────────────────────────────────────

class TestQuantizedKVCacheSVDRank:

    def test_svd_rank_set_on_all_layers(self):
        """QuantizedKVCache propagates svd_rank > 0 to every KVLayerCache."""
        cache = QuantizedKVCache(n_layers=3, svd_rank=4)
        for layer in cache._layers:
            assert layer._svd_rank == 4

    def test_zero_svd_rank_unchanged(self):
        """svd_rank=0 (default) leaves _svd_rank at 0 on all layers."""
        cache = QuantizedKVCache(n_layers=2, svd_rank=0)
        for layer in cache._layers:
            assert layer._svd_rank == 0


# ── DiskKVCache._serialise with recent + SVD ─────────────────────────────────

class TestSerialiseDeserialise:

    def test_serialise_with_recent_tokens(self):
        """_serialise includes L{i}_keys_recent when n_rec > 0."""
        cache = _populated_qkv_with_recent(n_layers=1, n_tokens=3)
        out   = DiskKVCache._serialise(cache)
        assert out is not None
        # After window=1, at least 2 tokens in INT8 and 1 in recent
        assert "L0_n_recent" in out
        assert int(out["L0_n_recent"]) > 0
        assert "L0_keys_recent" in out
        assert "L0_vals_recent" in out

    def test_serialise_with_svd_basis(self):
        """_serialise persists SVD basis when _svd_Vk is set."""
        layer = _make_svd_layer(svd_rank=4)
        # Wrap in minimal QuantizedKVCache shell
        cache = object.__new__(QuantizedKVCache)
        cache._layers = [layer]
        out = DiskKVCache._serialise(cache)
        assert out is not None
        assert "L0_svd_Vk" in out
        assert "L0_svd_Vv" in out
        assert "L0_svd_rank" in out
        assert int(out["L0_svd_rank"]) == 4

    def test_deserialise_with_recent_tokens(self):
        """_deserialise restores recent tokens from the serialised dict."""
        cache = _populated_qkv_with_recent(n_layers=1, n_tokens=3)
        out   = DiskKVCache._serialise(cache)
        restored = DiskKVCache._deserialise(out)
        assert len(restored._layers) == 1
        assert len(restored._layers[0].keys_recent) > 0

    def test_deserialise_with_svd_basis(self):
        """_deserialise restores _svd_Vk + _svd_Vv from the serialised dict."""
        layer = _make_svd_layer(svd_rank=4)
        cache = object.__new__(QuantizedKVCache)
        cache._layers = [layer]
        out   = DiskKVCache._serialise(cache)
        restored = DiskKVCache._deserialise(out)
        assert restored._layers[0]._svd_Vk is not None
        assert restored._layers[0]._svd_rank == 4

    def test_roundtrip_serialise_deserialise(self):
        """Full round-trip: serialise then deserialise returns equivalent cache."""
        cache = _populated_qkv_with_recent(n_layers=2, n_tokens=5)
        out   = DiskKVCache._serialise(cache)
        r     = DiskKVCache._deserialise(out)
        assert len(r._layers) == 2
        for i in range(2):
            orig = cache._layers[i]
            rest = r._layers[i]
            assert rest.n_heads  == orig.n_heads
            assert rest.head_dim == orig.head_dim

    def test_serialise_no_recent_tokens(self):
        """_serialise with n_rec==0 skips the recent block (branch 1351->1355)."""
        # window=0 forces every appended token immediately into the INT8 tier.
        lay = KVLayerCache(window=0)
        k, v = _rand_kv(seed=77)
        lay.append(k, v)
        assert len(lay.keys_recent) == 0
        assert lay.keys_old_q is not None
        cache = object.__new__(QuantizedKVCache)
        cache._layers = [lay]
        out = DiskKVCache._serialise(cache)
        assert out is not None
        assert int(out["L0_n_recent"]) == 0
        assert "L0_keys_recent" not in out

    def test_deserialise_no_recent_tokens(self):
        """_deserialise with n_rec==0 skips the recent-restore loop (branch 1378->1385)."""
        H, D = N_HEADS, HEAD_DIM
        data = {
            "n_layers": np.array(1, dtype=np.int32),
            "L0_n_heads": np.array(H, dtype=np.int32),
            "L0_head_dim": np.array(D, dtype=np.int32),
            "L0_n_recent": np.array(0, dtype=np.int32),
        }
        qkv = DiskKVCache._deserialise(data)
        assert len(qkv._layers) == 1
        assert qkv._layers[0].keys_recent == []


# ── DiskKVCache.store with last_logit ─────────────────────────────────────────

class TestDiskKVCacheStoreCoverage:

    def _wait(self, dc: DiskKVCache, ids, timeout: float = 2.0) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if dc.lookup(ids) is not None:
                return True
            time.sleep(0.05)
        return False

    def test_store_with_last_logit_round_trips(self, tmp_path):
        """store(last_logit_np=array) persists and lookup recovers the logit."""
        dc    = DiskKVCache(tmp_path)
        cache = _populated_qkv_with_recent(n_layers=1, n_tokens=3)
        logit = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        ids   = [1, 2, 3]
        dc.store(ids, cache, logit)
        assert self._wait(dc, ids), "store thread never wrote file"
        result = dc.lookup(ids)
        assert result is not None
        _, restored_logit = result
        np.testing.assert_allclose(restored_logit, logit, rtol=1e-5)


# ── SessionKVCache ────────────────────────────────────────────────────────────

class TestSessionKVCache:

    def test_init_creates_directory(self, tmp_path):
        """__init__ creates the cache directory if it does not exist."""
        sdir = tmp_path / "sessions"
        sc   = SessionKVCache(cache_dir=sdir)
        assert sdir.is_dir()

    def test_session_key_deterministic(self, tmp_path):
        """session_key() returns the same value for the same messages."""
        sc   = SessionKVCache(cache_dir=tmp_path)
        msgs = [{"role": "user", "content": "hello"}]
        k1   = sc.session_key(msgs)
        k2   = sc.session_key(msgs)
        assert k1 == k2

    def test_session_key_differs_for_different_content(self, tmp_path):
        sc = SessionKVCache(cache_dir=tmp_path)
        k1 = sc.session_key([{"content": "a"}])
        k2 = sc.session_key([{"content": "b"}])
        assert k1 != k2

    def test_session_key_is_32_hex_chars(self, tmp_path):
        sc  = SessionKVCache(cache_dir=tmp_path)
        key = sc.session_key([{"content": "hello world"}])
        assert len(key) == 32
        assert all(c in "0123456789abcdef" for c in key)

    def test_load_session_miss_returns_none(self, tmp_path):
        """load_session() returns None when no matching file exists."""
        sc = SessionKVCache(cache_dir=tmp_path)
        assert sc.load_session("no_such_key") is None

    def test_save_and_load_roundtrip(self, tmp_path):
        """save_session() + load_session() round-trips a QuantizedKVCache."""
        sc    = SessionKVCache(cache_dir=tmp_path)
        cache = _populated_qkv_with_recent(n_layers=2, n_tokens=3)
        key   = sc.session_key([{"content": "test prompt"}])
        sc.save_session(key, cache)

        # Wait for background thread
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            if sc.load_session(key) is not None:
                break
            time.sleep(0.05)

        restored = sc.load_session(key)
        assert restored is not None
        assert len(restored._layers) == 2

    def test_list_sessions_empty(self, tmp_path):
        """list_sessions() returns empty list on a fresh cache."""
        sc = SessionKVCache(cache_dir=tmp_path)
        assert sc.list_sessions() == []

    def test_list_sessions_after_save(self, tmp_path):
        """list_sessions() returns stored session keys."""
        sc    = SessionKVCache(cache_dir=tmp_path)
        cache = _populated_qkv_with_recent(n_layers=1, n_tokens=2)
        key   = sc.session_key([{"content": "abc"}])
        sc.save_session(key, cache)

        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            if sc.list_sessions():
                break
            time.sleep(0.05)

        sessions = sc.list_sessions()
        assert key in sessions

    def test_evict_if_needed_respects_max(self, tmp_path):
        """SessionKVCache._evict_if_needed keeps at most max_entries files."""
        sc = SessionKVCache(cache_dir=tmp_path, max_entries=2)
        cache = _populated_qkv_with_recent(n_layers=1, n_tokens=2)

        keys = []
        for i in range(4):
            msgs = [{"content": f"prompt_{i}"}]
            k    = sc.session_key(msgs)
            keys.append(k)
            sc.save_session(k, cache)
            time.sleep(0.02)  # small delay to stagger mtime

        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline:
            sessions = sc.list_sessions()
            if len(sessions) <= 2:
                break
            time.sleep(0.1)

        assert len(sc.list_sessions()) <= 2

    def test_load_session_returns_none_on_corrupt_file(self, tmp_path):
        """load_session() deletes and returns None for corrupted files."""
        sc  = SessionKVCache(cache_dir=tmp_path)
        key = "badcafe" * 4   # 32 hex chars
        bad = tmp_path / f"{key}.npz"
        bad.write_bytes(b"not valid npz")
        result = sc.load_session(key)
        assert result is None

    def test_session_key_uses_last_8_messages(self, tmp_path):
        """session_key() uses up to the last 8 messages for the hash."""
        sc   = SessionKVCache(cache_dir=tmp_path)
        # 10 messages: last 8 determine the key
        msgs = [{"content": f"msg_{i}"} for i in range(10)]
        key1 = sc.session_key(msgs)
        key2 = sc.session_key(msgs[-8:])
        assert key1 == key2

    def test_save_session_skips_unpopulated_cache(self, tmp_path):
        """save_session early-returns from _worker when cache not yet populated (line 1497)."""
        lay = KVLayerCache()   # n_heads is None — not yet populated
        qkv = object.__new__(QuantizedKVCache)
        qkv._layers = [lay]
        sc = SessionKVCache(cache_dir=tmp_path)
        sc.save_session("unpopulated_key_placeholder_000", qkv)
        time.sleep(0.2)   # let background thread finish
        # _serialise returns None → _worker returns early → no file written
        assert list(tmp_path.glob("*.npz")) == []
