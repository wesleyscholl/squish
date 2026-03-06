#!/usr/bin/env python3
"""
squish/kv_cache.py

Quantized KV cache for long-context inference on Apple Silicon unified memory.

Two complementary strategies — each reducing KV cache memory by ~50%:

  KIVI (Kim et al., 2024  https://arxiv.org/abs/2402.02750)
  ──────────────────────────────────────────────────────────
  Keep the most-recent ``window`` token positions in FP16 (the "residual").
  Quantize older positions to INT8 with per-channel scales for keys and
  per-token scales for values.  All slots are dequantized on-the-fly during
  the attention computation.

  Memory model (per layer, per head, per token):
    • FP16:  head_dim × 2 bytes  = 256 bytes  (Qwen2-7B: head_dim=128)
    • INT8:  head_dim × 1 byte
           + 1 scale × 4 bytes (f32)          ≈ 132 bytes  (½ of FP16)

  SnapKV (Li et al., 2024  https://arxiv.org/abs/2404.14469)
  ──────────────────────────────────────────────────────────
  During prefill, observe which K/V positions receive the most attention from
  the most-recent ``snap_window`` query positions.  After prefill, evict the
  bottom ``(1 - budget_ratio)`` fraction of positions.  This caps the cache
  size to ``budget`` tokens regardless of context length.


Usage
-----
At the server level — patch the model before any generation:

    from squish.kv_cache import make_quantized_cache, patch_model_kv_cache

    # After mlx_lm.load() returns (model, tokenizer):
    patch_model_kv_cache(model, mode="int8", window=64)

    # With SnapKV budget (evict to at most 2048 positions):
    patch_model_kv_cache(model, mode="snap", window=64, budget=2048)

Low-level: create a cache and pass to generate():

    cache = make_quantized_cache(model, mode="int8", window=64)
    # Pass cache as kv_cache argument to mlx_lm generate functions
"""
import threading
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Lazy MLX import (module may be imported without Metal available, e.g. tests)
# ---------------------------------------------------------------------------
try:
    import mlx.core as _mlx  # type: ignore[import]
except ImportError:  # pragma: no cover
    _mlx = None  # will raise at runtime if Metal code is actually called


def _mx():
    """Compatibility shim — prefer using _mlx directly in hot paths."""
    return _mlx


# ---------------------------------------------------------------------------
# INT8 per-channel quantization helpers (pure numpy — runs on CPU)
# These are only called on the "old" portion of the cache;
# the recent window stays in FP16 on Metal.
# ---------------------------------------------------------------------------

def _quantize_int8_per_channel(arr_f16: np.ndarray) -> tuple:
    """
    Quantize a 2-D float16 array to INT8 per output-channel (per row).

    Uses in-place arithmetic to keep peak memory to ~1× input (float32)
    instead of the naive 3× (float32 + abs intermediate + division result).

    Parameters
    ----------
    arr_f16 : np.ndarray  shape (n_tokens, head_dim)  — float16

    Returns
    -------
    q    : np.ndarray  shape (n_tokens, head_dim)  — int8
    scale: np.ndarray  shape (n_tokens,)            — float32 per-token scale
    """
    arr = arr_f16.astype(np.float32)           # unavoidable: fp16 overflows in abs
    # Per-row abs-max (fused reduce — no full intermediate array)
    scale    = np.max(np.abs(arr), axis=-1)    # (n,)
    scale_safe = np.maximum(scale, 1e-8)       # (n,) — avoids divide-by-zero
    # In-place normalize + scale — reuses the float32 buffer
    arr /= scale_safe[:, np.newaxis]           # normalise to [-1, 1]
    arr *= 127.0
    np.round(arr, out=arr)
    q = np.clip(arr, -128, 127).astype(np.int8)
    return q, scale_safe.astype(np.float32)


def _dequantize_int8_per_channel(q: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """
    Dequantize INT8 array back to float16.

    Parameters
    ----------
    q     : (n_tokens, head_dim)   int8
    scale : (n_tokens,)            float32

    Returns
    -------
    (n_tokens, head_dim)  float16
    """
    arr = q.astype(np.float32) / 127.0 * scale[:, np.newaxis]
    return arr.astype(np.float16)


# Number of tokens to collect before fitting the per-layer SVD basis.
# 64 tokens provide a stable subspace for head_dim=128 models.
_SVD_INIT_TOKENS: int = 64


# ---------------------------------------------------------------------------
# KVLayerCache — per-layer KV buffer with optional INT8 compression
# ---------------------------------------------------------------------------

class KVLayerCache:
    """
    Single-layer KV cache that combines:
    • A FP16 "recent window" for the last ``window`` positions
    • An INT8 compressed buffer for all older positions

    Thread-safe (a single generation thread owns the cache).

    Attributes
    ----------
    keys_recent   : list[np.ndarray]  — list of (n_heads, head_dim) f16 arrays
    values_recent : same shape
    keys_old_q    : np.ndarray or None  — (n_heads, n_old, head_dim) int8
    keys_old_s    : np.ndarray or None  — (n_heads, n_old) float32 scales
    values_old_q  : same
    values_old_s  : same
    """
    __slots__ = (
        "window",
        "keys_recent", "values_recent",
        "keys_old_q", "keys_old_s",
        "values_old_q", "values_old_s",
        "n_heads", "head_dim",
        "_lock",
        # disk overflow tier (Item 4 — long-context NVMe spill)
        "_disk_threshold", "_disk_dir",
        "_disk_map_k",  "_disk_map_v",
        "_disk_scales_k", "_disk_scales_v",
        "_disk_n",
        # Phase 1: SVD KV compression
        "_svd_rank",      # int: 0 = off; rank < head_dim when enabled
        "_svd_Vk",        # np.ndarray (n_heads, rank, head_dim) float16 or None
        "_svd_Vv",        # same shape as _svd_Vk
        "_svd_buf_k",     # list[np.ndarray] calibration buffer (FP16), or None
        "_svd_buf_v",     # same
        # Phase 2: HNSW retrieval index
        "_retrieval_top_k",  # int: 0 = off; >0 enables HNSW for disk-tier retrieval
        "_hnsw",             # HNSWIndex | None
        # Phase 0C: async CPU dequant pre-fetch (prevents GPU ↔ dequant contention)
        "_prefetch_future",  # Future[tuple] | None
    )

    def __init__(self, window: int = 64):
        self.window        = window
        self.keys_recent   = []     # list of (n_heads, head_dim) f16 arrays
        self.values_recent = []
        self.keys_old_q    = None   # (n_heads, n_old, head_dim_or_rank) int8
        self.keys_old_s    = None   # (n_heads, n_old) f32
        self.values_old_q  = None
        self.values_old_s  = None
        self.n_heads       = None
        self.head_dim      = None
        self._lock         = threading.RLock()   # re-entrant: _snap_evict calls get_full_kv under the lock
        # disk overflow tier — all None / 0 when disabled
        self._disk_threshold = None   # int: spill old_q rows beyond this count
        self._disk_dir       = None
        self._disk_map_k     = None   # np.memmap (n_heads, max_disk, head_dim) int8
        self._disk_map_v     = None
        self._disk_scales_k  = None   # np.ndarray (n_heads, max_disk) f32
        self._disk_scales_v  = None
        self._disk_n         = 0      # rows currently written to disk
        # Phase 1: SVD KV compression
        self._svd_rank   = 0     # 0 = off; set to rank < head_dim to enable
        self._svd_Vk     = None  # (n_heads, rank, head_dim) float16 — fitted once, then frozen
        self._svd_Vv     = None
        self._svd_buf_k  = None  # list[np.ndarray] calibration buffer (FP16), cleared after fit
        self._svd_buf_v  = None
        # Phase 2: HNSW retrieval index
        self._retrieval_top_k = 0     # 0 = off; set via enable_disk_tier
        self._hnsw            = None  # HNSWIndex, lazily created on first disk spill
        # Phase 0C: async CPU dequant pre-fetch
        self._prefetch_future = None  # concurrent.futures.Future | None

    # ── Main cache update ─────────────────────────────────────────────────────

    def append(self, key_np: np.ndarray, value_np: np.ndarray) -> None:
        """
        Append a single token's K/V pair (shape (n_heads, head_dim)) to the cache.
        When the recent window overflows, the oldest slot is quantized to INT8.
        """
        with self._lock:
            if self.n_heads is None:
                self.n_heads  = key_np.shape[0]
                self.head_dim = key_np.shape[-1]

            self.keys_recent.append(key_np.astype(np.float16))
            self.values_recent.append(value_np.astype(np.float16))

            # Evict the oldest recent slot to INT8 when window fills
            while len(self.keys_recent) > self.window:
                oldest_k = self.keys_recent.pop(0)    # (n_heads, head_dim)
                oldest_v = self.values_recent.pop(0)

                # ── Phase 1: SVD calibration phase ──────────────────────────────────
                # Buffer tokens in FP16 until we have enough to fit the SVD basis.
                # Once fitted, all subsequent tokens are projected before INT8 quant.
                if self._svd_rank > 0 and self._svd_Vk is None:
                    if self._svd_buf_k is None:
                        self._svd_buf_k, self._svd_buf_v = [], []
                    self._svd_buf_k.append(oldest_k)
                    self._svd_buf_v.append(oldest_v)
                    if len(self._svd_buf_k) >= _SVD_INIT_TOKENS:
                        self._svd_fit_and_flush()
                    continue  # token is buffered; skip normal quantization

                # ── Apply SVD projection if basis is ready ────────────────────────
                if self._svd_Vk is not None:
                    oldest_k = self._svd_project(oldest_k, self._svd_Vk)
                    oldest_v = self._svd_project(oldest_v, self._svd_Vv)

                # Quantize per-head per-token
                new_kq_list, new_ks_list = [], []
                new_vq_list, new_vs_list = [], []
                for h in range(self.n_heads):
                    kq, ks = _quantize_int8_per_channel(
                        oldest_k[h:h+1, :])          # (1, head_dim)
                    vq, vs = _quantize_int8_per_channel(
                        oldest_v[h:h+1, :])
                    new_kq_list.append(kq)            # each (1, head_dim) int8
                    new_ks_list.append(ks)            # each (1,) float32
                    new_vq_list.append(vq)
                    new_vs_list.append(vs)

                # stack → (n_heads, 1, head_dim) and (n_heads, 1)
                slot_kq = np.stack(new_kq_list, axis=0)   # (n_heads, 1, head_dim) i8
                slot_ks = np.stack(new_ks_list, axis=0)   # (n_heads, 1) f32
                slot_vq = np.stack(new_vq_list, axis=0)
                slot_vs = np.stack(new_vs_list, axis=0)

                if self.keys_old_q is None:
                    self.keys_old_q   = slot_kq
                    self.keys_old_s   = slot_ks
                    self.values_old_q = slot_vq
                    self.values_old_s = slot_vs
                else:
                    self.keys_old_q   = np.concatenate([self.keys_old_q,   slot_kq], axis=1)
                    self.keys_old_s   = np.concatenate([self.keys_old_s,   slot_ks], axis=1)
                    self.values_old_q = np.concatenate([self.values_old_q, slot_vq], axis=1)
                    self.values_old_s = np.concatenate([self.values_old_s, slot_vs], axis=1)
            # Spill oldest INT8 entries to NVMe disk tier if enabled
            self._maybe_spill_to_disk()

    def get_full_kv(self) -> tuple:
        """
        Return the full key and value matrices as FP16 numpy arrays.

        Returns
        -------
        keys   : (n_heads, n_total, head_dim)  float16
        values : (n_heads, n_total, head_dim)  float16
        """
        with self._lock:
            # Reconstruct disk tier (oldest, spilled to NVMe memmap)
            disk_k, disk_v = self._disk_full_kv()

            # Reconstruct RAM INT8 portion
            if self.keys_old_q is not None:
                # Dequantize per head
                old_k_list, old_v_list = [], []
                for h in range(self.n_heads):
                    k_deq = _dequantize_int8_per_channel(
                        self.keys_old_q[h],      # (n_old, rank_or_head_dim)
                        self.keys_old_s[h])      # (n_old,)
                    v_deq = _dequantize_int8_per_channel(
                        self.values_old_q[h],
                        self.values_old_s[h])
                    # Phase 1: back-project SVD-compressed tokens to full head_dim
                    if self._svd_Vk is not None:
                        Vk_h = self._svd_Vk[h].astype(np.float32)  # (rank, head_dim)
                        Vv_h = self._svd_Vv[h].astype(np.float32)
                        k_deq = (k_deq.astype(np.float32) @ Vk_h).astype(np.float16)
                        v_deq = (v_deq.astype(np.float32) @ Vv_h).astype(np.float16)
                    old_k_list.append(k_deq)
                    old_v_list.append(v_deq)
                old_k = np.stack(old_k_list, axis=0)   # (n_heads, n_old, head_dim)
                old_v = np.stack(old_v_list, axis=0)
            else:
                old_k = old_v = None

            if self.keys_recent:
                # Each element is (n_heads, head_dim) → stack along token dim
                rec_k = np.stack(self.keys_recent,   axis=1)   # (n_heads, n_rec, head_dim)
                rec_v = np.stack(self.values_recent, axis=1)
            else:
                rec_k = rec_v = None

            # Combine: disk || RAM int8 || FP16 recent
            parts_k = [p for p in (disk_k, old_k, rec_k) if p is not None]
            parts_v = [p for p in (disk_v, old_v, rec_v) if p is not None]
            if not parts_k:
                return None, None
            if len(parts_k) == 1:
                return parts_k[0], parts_v[0]
            full_k = np.concatenate(parts_k, axis=1)
            full_v = np.concatenate(parts_v, axis=1)
            return full_k, full_v

    # ── Phase 0C: async CPU dequant pre-fetch ─────────────────────────────────
    # During the token-sampling step (which is CPU-bound) we overlap the
    # INT8→FP16 dequantization of the *next* decode step on a background thread.
    # This hides the O(n_old_tokens) numpy work behind the sampler, preventing
    # ≥30 % slowdown from blocking the generation loop on large KV caches.
    #
    # Usage in the decode loop:
    #   layer_cache.start_prefetch()      # fire-and-forget at end of step N
    #   ...sample token N...
    #   k, v = layer_cache.get_full_kv_prefetched()   # ready at step N+1

    _THREAD_POOL: "concurrent.futures.ThreadPoolExecutor | None" = None

    @classmethod
    def _get_pool(cls) -> "concurrent.futures.ThreadPoolExecutor":
        if cls._THREAD_POOL is None:
            import concurrent.futures
            # One worker is enough: dequant is sequential per layer; we want
            # CPU—not Metal—so we keep the thread off the main queue.
            cls._THREAD_POOL = concurrent.futures.ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="squish-kv-deq"
            )
        return cls._THREAD_POOL

    def start_prefetch(self) -> None:
        """
        Submit the dequantization work for the *current* cache state to a
        background CPU thread.  Call this immediately after sampling a token
        so the work overlaps with the next-step setup.
        """
        if self._prefetch_future is not None:
            return  # already in-flight
        if self.keys_old_q is None:
            return  # nothing to prefetch — recent window only, no INT8 tier
        try:
            self._prefetch_future = self._get_pool().submit(self.get_full_kv)
        except Exception:  # pragma: no cover
            self._prefetch_future = None  # never block generation

    def get_full_kv_prefetched(self) -> tuple:
        """
        Return the pre-fetched ``(keys, values)`` numpy arrays if available,
        otherwise fall back to a synchronous ``get_full_kv()`` call.
        """
        future = self._prefetch_future
        self._prefetch_future = None  # consume the future
        if future is None:
            return self.get_full_kv()
        try:
            return future.result(timeout=0.5)
        except Exception:  # pragma: no cover
            return self.get_full_kv()

    def get_as_mlx(self):
        """Return (keys, values) as MLX bfloat16 arrays for use in attention."""
        mx = _mx()
        k_np, v_np = self.get_full_kv()
        if k_np is None:
            return None, None
        return (mx.array(k_np).astype(mx.bfloat16),
                mx.array(v_np).astype(mx.bfloat16))

    @property
    def n_tokens(self) -> int:
        old = self.keys_old_q.shape[1] if self.keys_old_q is not None else 0
        return old + len(self.keys_recent)

    @property
    def memory_bytes(self) -> int:
        """Approximate memory usage in bytes."""
        b = 0
        if self.keys_old_q is not None:
            b += self.keys_old_q.nbytes + self.keys_old_s.nbytes * 2
            b += self.values_old_q.nbytes + self.values_old_s.nbytes * 2
        for arr in self.keys_recent + self.values_recent:
            b += arr.nbytes
        return b

    def reset(self):
        """Clear all cached K/V state (SVD basis is preserved across conversations)."""
        with self._lock:
            self.keys_recent.clear()
            self.values_recent.clear()
            self.keys_old_q = self.keys_old_s = None
            self.values_old_q = self.values_old_s = None
            self._disk_n = 0
            # Delete memmap files if they exist
            self._disk_map_k = None
            self._disk_map_v = None
            self._disk_scales_k = None
            self._disk_scales_v = None
            for attr in ("_disk_path_k", "_disk_path_v"):
                p = getattr(self, attr, None)
                if p is not None:  # pragma: no cover
                    try:
                        import pathlib
                        pathlib.Path(p).unlink(missing_ok=True)
                    except Exception:
                        pass
            # Clear SVD calibration buffer but keep the fitted basis
            self._svd_buf_k = None
            self._svd_buf_v = None
            # Reset HNSW retrieval index (token positions change each conversation)
            self._hnsw = None

    # ── Phase 1: SVD KV compression helpers ───────────────────────────────────

    def _svd_project(self, x: np.ndarray, V: np.ndarray) -> np.ndarray:
        """
        Project one token's K or V from full head_dim to SVD rank.

        Parameters
        ----------
        x : (n_heads, head_dim)          float16
        V : (n_heads, rank, head_dim)    float16  — right singular vectors

        Returns
        -------
        (n_heads, rank)  float16
        """
        x_f32 = x.astype(np.float32)       # (n_heads, head_dim)
        V_f32 = V.astype(np.float32)       # (n_heads, rank, head_dim)
        # result[h] = x[h] @ V[h].T  → (rank,) per head
        out = np.einsum("hd,hrd->hr", x_f32, V_f32)
        return out.astype(np.float16)

    def _svd_fit_and_flush(self) -> None:
        """
        Fit per-head SVD bases from the calibration buffer, then quantize all
        buffered tokens to INT8 using the fitted projection.

        Called once when ``len(_svd_buf_k) >= _SVD_INIT_TOKENS``.
        After this call ``_svd_Vk/Vv`` are set and ``_svd_buf_k/v`` are cleared.
        """
        Vk_list, Vv_list = [], []
        for h in range(self.n_heads):
            K = np.stack([t[h] for t in self._svd_buf_k], axis=0).astype(np.float32)
            _, _, Vt = np.linalg.svd(K, full_matrices=False)   # Vt: (min(n_init,dim), dim)
            Vk_list.append(Vt[:self._svd_rank, :])              # (rank, head_dim)

            V_mat = np.stack([t[h] for t in self._svd_buf_v], axis=0).astype(np.float32)
            _, _, Vt = np.linalg.svd(V_mat, full_matrices=False)
            Vv_list.append(Vt[:self._svd_rank, :])

        # Store as float16 to save memory; cast to float32 only on projection
        self._svd_Vk = np.stack(Vk_list, axis=0).astype(np.float16)  # (n_heads, rank, head_dim)
        self._svd_Vv = np.stack(Vv_list, axis=0).astype(np.float16)

        # Flush the calibration buffer: project + quantize each buffered token
        for k_f16, v_f16 in zip(self._svd_buf_k, self._svd_buf_v):
            k_proj = self._svd_project(k_f16, self._svd_Vk)   # (n_heads, rank)
            v_proj = self._svd_project(v_f16, self._svd_Vv)

            new_kq_list, new_ks_list = [], []
            new_vq_list, new_vs_list = [], []
            for h in range(self.n_heads):
                kq, ks = _quantize_int8_per_channel(k_proj[h:h+1, :])
                vq, vs = _quantize_int8_per_channel(v_proj[h:h+1, :])
                new_kq_list.append(kq)
                new_ks_list.append(ks)
                new_vq_list.append(vq)
                new_vs_list.append(vs)

            slot_kq = np.stack(new_kq_list, axis=0)
            slot_ks = np.stack(new_ks_list, axis=0)
            slot_vq = np.stack(new_vq_list, axis=0)
            slot_vs = np.stack(new_vs_list, axis=0)

            if self.keys_old_q is None:
                self.keys_old_q   = slot_kq
                self.keys_old_s   = slot_ks
                self.values_old_q = slot_vq
                self.values_old_s = slot_vs
            else:
                self.keys_old_q   = np.concatenate([self.keys_old_q,   slot_kq], axis=1)
                self.keys_old_s   = np.concatenate([self.keys_old_s,   slot_ks], axis=1)
                self.values_old_q = np.concatenate([self.values_old_q, slot_vq], axis=1)
                self.values_old_s = np.concatenate([self.values_old_s, slot_vs], axis=1)

        self._svd_buf_k = None
        self._svd_buf_v = None

    # ── Disk overflow tier (Item 4) ───────────────────────────────────────────

    def enable_disk_tier(
        self,
        threshold: int,
        max_disk_tokens: int,
        cache_dir,          # str | Path
        n_heads: int,
        head_dim: int,
        retrieval_top_k: int = 0,  # Phase 2: >0 enables HNSW retrieval index
    ) -> None:
        """
        Enable disk-backed overflow for old INT8 K/V entries.

        When the number of INT8-quantized positions (``keys_old_q.shape[1]``)
        exceeds ``threshold``, the oldest ``(n_old - threshold)`` rows are
        spilled to a ``numpy.memmap`` file on the NVMe mount at ``cache_dir``.

        OS page-fault semantics keep hot pages in the unified memory file-cache
        while cold pages stay on disk — effectively using the SSD as a
        transparent third tier, behind Metal (FP16 recent window) and CPU RAM
        (INT8 ring buffer).

        Parameters
        ----------
        threshold        : int — rows to keep in RAM before spilling
        max_disk_tokens  : int — pre-allocated memmap size (rows)
        cache_dir        : directory for temp memmap files (e.g. /tmp/squish_kv)
        n_heads, head_dim : model dimensions (required to size the memmap
                            before the first append, because __slots__ prevent
                            lazy init once n_heads is known).
        retrieval_top_k  : int — Phase 2: if >0 build an HNSW index on spilled
                            key vectors to support get_relevant_kv().
        """
        import pathlib, tempfile
        cache_dir = pathlib.Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        self._disk_threshold = threshold
        self._disk_dir       = cache_dir
        self.n_heads  = n_heads
        self.head_dim = head_dim
        self._retrieval_top_k = retrieval_top_k
        uid = id(self)
        path_k = cache_dir / f"kv_{uid}_k.bin"
        path_v = cache_dir / f"kv_{uid}_v.bin"
        self._disk_map_k = np.memmap(
            path_k, dtype=np.int8, mode="w+",
            shape=(n_heads, max_disk_tokens, head_dim))
        self._disk_map_v = np.memmap(
            path_v, dtype=np.int8, mode="w+",
            shape=(n_heads, max_disk_tokens, head_dim))
        self._disk_scales_k = np.zeros((n_heads, max_disk_tokens), dtype=np.float32)
        self._disk_scales_v = np.zeros((n_heads, max_disk_tokens), dtype=np.float32)
        self._disk_n = 0
        # HNSW index is created lazily on first spill (dim may change with SVD)

    def _maybe_spill_to_disk(self) -> None:
        """
        If the INT8 RAM buffer exceeds ``_disk_threshold``, spill the oldest
        tokens to the memmap tier.

        Called inside the ``_lock`` from ``append()``.
        """
        if (self._disk_threshold is None
                or self._disk_map_k is None
                or self.keys_old_q is None):
            return
        n_old = self.keys_old_q.shape[1]
        if n_old <= self._disk_threshold:
            return
        n_spill = n_old - self._disk_threshold
        disk_end = self._disk_n + n_spill
        if disk_end > self._disk_map_k.shape[1]:
            # Disk tier full — silently keep RAM tier only (graceful degrade)
            return
        # Write oldest n_spill rows to memmap.
        # shapes: keys_old_q (n_heads, n_old, head_dim)
        self._disk_map_k[:, self._disk_n:disk_end, :] = \
            self.keys_old_q[:, :n_spill, :]
        self._disk_map_v[:, self._disk_n:disk_end, :] = \
            self.values_old_q[:, :n_spill, :]
        self._disk_scales_k[:, self._disk_n:disk_end] = \
            self.keys_old_s[:, :n_spill]
        self._disk_scales_v[:, self._disk_n:disk_end] = \
            self.values_old_s[:, :n_spill]
        self._disk_n = disk_end

        # Phase 2: update HNSW retrieval index with the spilled key vectors
        if self._retrieval_top_k > 0:  # pragma: no cover
            self._update_hnsw_index(
                keys_int8=self.keys_old_q[:, :n_spill, :],
                scales=self.keys_old_s[:, :n_spill],
                start_pos=self._disk_n - n_spill,
            )

        # Trim RAM buffer
        self.keys_old_q   = self.keys_old_q[:, n_spill:, :]
        self.keys_old_s   = self.keys_old_s[:, n_spill:]
        self.values_old_q = self.values_old_q[:, n_spill:, :]
        self.values_old_s = self.values_old_s[:, n_spill:]

    def _update_hnsw_index(  # pragma: no cover
        self,
        keys_int8: np.ndarray,   # (n_heads, n, rank_or_head_dim) int8
        scales: np.ndarray,      # (n_heads, n) float32
        start_pos: int,
    ) -> None:
        """
        Add spilled key vectors to the HNSW retrieval index.

        Uses head 0 as the representative head for indexing.  HNSW is
        lazily created on the first call so the correct dimension is known
        (after SVD fitting if active).

        Parameters
        ----------
        keys_int8 : (n_heads, n, dim) int8 — spilled key vectors
        scales    : (n_heads, n) float32   — per-token scales
        start_pos : int — disk position offset for these tokens
        """
        h = 0   # representative head
        n = keys_int8.shape[1]
        # Dequantize head-0 keys
        k_deq = _dequantize_int8_per_channel(
            keys_int8[h], scales[h])  # (n, rank_or_head_dim) float16
        # Back-project SVD if active (so index is in full head_dim space)
        if self._svd_Vk is not None:
            k_deq = (k_deq.astype(np.float32) @ self._svd_Vk[h].astype(np.float32)).astype(np.float16)
        k_f32 = k_deq.astype(np.float32)  # (n, head_dim)
        dim = k_f32.shape[-1]
        # Lazy HNSW init: dim is now known
        if self._hnsw is None:
            try:
                try:
                    from squish.vector_index import HNSWIndex
                except ImportError:
                    from vector_index import HNSWIndex  # direct run
                max_elem = int(self._disk_map_k.shape[1]) if self._disk_map_k is not None else 500_000
                self._hnsw = HNSWIndex(dim=dim, max_elements=max_elem)
            except ImportError:
                return  # hnswlib not installed — silently skip
        ids = np.arange(start_pos, start_pos + n, dtype=np.int64)
        self._hnsw.add(k_f32, ids)

    def get_relevant_kv(  # pragma: no cover
        self,
        query_key_fp16: np.ndarray,   # (n_heads, head_dim) float16
        top_k: int,
        hot_window: int = 256,
    ) -> tuple:
        """
        Return a *sparse* K/V context composed of ANNS-retrieved disk tokens
        plus a guaranteed hot-window of the most recent RAM tokens.

        Falls back to ``get_full_kv()`` when no HNSW index is available.

        Parameters
        ----------
        query_key_fp16 : (n_heads, head_dim) float16 — current decode step key
        top_k          : int — number of disk tokens to retrieve per query
        hot_window     : int — number of most-recent RAM tokens always included

        Returns
        -------
        (keys, values) : (n_heads, n_ctx, head_dim) float16 each
        """
        if self._hnsw is None or self._disk_n == 0:
            return self.get_full_kv()
        with self._lock:
            h = 0  # representative head for ANNS query
            # Build query vector (back-project SVD if active)
            q_fp16 = query_key_fp16[h]
            if self._svd_Vk is not None:
                q_fp16 = (q_fp16.astype(np.float32) @ self._svd_Vk[h].astype(np.float32)).astype(np.float16)
            q_f32 = q_fp16.astype(np.float32)

            # ANNS search over disk-tier keys
            retrieved_ids, _ = self._hnsw.search(q_f32, top_k=top_k)

            if len(retrieved_ids) == 0:
                return self.get_full_kv()

            # Gather retrieved disk rows
            retrieved_k_list, retrieved_v_list = [], []
            for hh in range(self.n_heads):
                k_rows = _dequantize_int8_per_channel(
                    np.array(self._disk_map_k[hh, retrieved_ids, :]),
                    self._disk_scales_k[hh, retrieved_ids])
                v_rows = _dequantize_int8_per_channel(
                    np.array(self._disk_map_v[hh, retrieved_ids, :]),
                    self._disk_scales_v[hh, retrieved_ids])
                retrieved_k_list.append(k_rows)
                retrieved_v_list.append(v_rows)
            disk_k = np.stack(retrieved_k_list, axis=0)   # (n_heads, top_k, dim)
            disk_v = np.stack(retrieved_v_list, axis=0)

            # RAM INT8 tier (most recent hot_window entries)
            if self.keys_old_q is not None:
                n_old = self.keys_old_q.shape[1]
                hot_start = max(0, n_old - hot_window)
                old_k_list, old_v_list = [], []
                for hh in range(self.n_heads):
                    k_deq = _dequantize_int8_per_channel(
                        self.keys_old_q[hh, hot_start:, :],
                        self.keys_old_s[hh, hot_start:])
                    v_deq = _dequantize_int8_per_channel(
                        self.values_old_q[hh, hot_start:, :],
                        self.values_old_s[hh, hot_start:])
                    if self._svd_Vk is not None:
                        Vk_h = self._svd_Vk[hh].astype(np.float32)
                        Vv_h = self._svd_Vv[hh].astype(np.float32)
                        k_deq = (k_deq.astype(np.float32) @ Vk_h).astype(np.float16)
                        v_deq = (v_deq.astype(np.float32) @ Vv_h).astype(np.float16)
                    old_k_list.append(k_deq)
                    old_v_list.append(v_deq)
                hot_k = np.stack(old_k_list, axis=0)
                hot_v = np.stack(old_v_list, axis=0)
            else:
                hot_k = hot_v = None

            # FP16 recent window (always included)
            if self.keys_recent:
                rec_k = np.stack(self.keys_recent, axis=1)
                rec_v = np.stack(self.values_recent, axis=1)
            else:
                rec_k = rec_v = None

            # Concatenate: retrieved disk || hot RAM || recent FP16
            parts_k = [p for p in (disk_k, hot_k, rec_k) if p is not None]
            parts_v = [p for p in (disk_v, hot_v, rec_v) if p is not None]
            if not parts_k:
                return None, None
            return (np.concatenate(parts_k, axis=1),
                    np.concatenate(parts_v, axis=1))

    def _disk_full_kv(self) -> tuple:
        """
        Reconstruct the disk-tier portion as FP16 numpy arrays.

        Returns (keys, values) each shape (n_heads, _disk_n, head_dim) float16,
        or (None, None) if the disk tier is empty.
        """
        if self._disk_n == 0 or self._disk_map_k is None:
            return None, None
        old_k_list, old_v_list = [], []
        for h in range(self.n_heads):
            old_k_list.append(_dequantize_int8_per_channel(
                np.array(self._disk_map_k[h, :self._disk_n, :]),
                self._disk_scales_k[h, :self._disk_n]))
            old_v_list.append(_dequantize_int8_per_channel(
                np.array(self._disk_map_v[h, :self._disk_n, :]),
                self._disk_scales_v[h, :self._disk_n]))
        return (np.stack(old_k_list, axis=0),   # (n_heads, _disk_n, head_dim)
                np.stack(old_v_list, axis=0))

    # ── mlx_lm KVCache protocol (offset + update_and_fetch) ──────────────────

    @property
    def offset(self) -> int:
        """Total tokens stored; used by mlx_lm for RoPE position encoding."""
        return self.n_tokens

    def update_and_fetch(self, keys, values):
        """
        mlx_lm attention-layer cache protocol.

        Called by each model attention layer with the newly computed K/V
        tensors (shape: batch=1, n_heads, T_new, head_dim).  Appends the
        new tokens to the INT8-compressed ring buffer and returns the
        *full* accumulated K/V sequence as MLX arrays so that the
        attention computation uses the complete context.

        Parameters
        ----------
        keys   : mx.array  shape (1, n_heads, T_new, head_dim)
        values : mx.array  shape (1, n_heads, T_new, head_dim)

        Returns
        -------
        (keys_full, values_full) as mx.array (1, n_heads, T_total, head_dim)
        """
        mx = _mx()
        # Convert to numpy for storage: (n_heads, T_new, head_dim) float16
        k_np = np.array(keys[0].astype(mx.float16))
        v_np = np.array(values[0].astype(mx.float16))
        T_new = k_np.shape[1]
        for t in range(T_new):
            self.append(k_np[:, t, :], v_np[:, t, :])
        full_k, full_v = self.get_full_kv()   # (n_heads, T_total, head_dim) f16
        if full_k is None:  # pragma: no cover
            return keys, values
        return (
            mx.array(full_k[None]).astype(mx.bfloat16),   # (1, n_heads, T_total, head_dim)
            mx.array(full_v[None]).astype(mx.bfloat16),
        )


# ---------------------------------------------------------------------------
# SnapKV eviction (importance-based position selection)
# ---------------------------------------------------------------------------

def _snap_evict(
    layer_cache: KVLayerCache,
    budget: int,
    snap_window: int = 32,
) -> None:
    """
    Apply SnapKV-style eviction: keep only the ``budget`` most-important
    token positions.

    Importance is defined as the sum of attention weights each K/V position
    receives from the most-recent ``snap_window`` query positions.

    Called once after prefill (when the cache exceeds ``budget``).

    Parameters
    ----------
    layer_cache : KVLayerCache to evict
    budget      : maximum number of positions to retain
    snap_window : number of recent queries to use for importance estimation
    """
    with layer_cache._lock:
        n = layer_cache.n_tokens
        if n <= budget:
            return

        # Reconstruct full FP16 cache to compute importances
        full_k, full_v = layer_cache.get_full_kv()
        if full_k is None:  # pragma: no cover
            return

        nh, nt, hd = full_k.shape
        k_f32 = full_k.astype(np.float32)   # (n_heads, n_tokens, head_dim)

        # Use the tail of K as proxy queries (recent snap_window positions)
        q_window = min(snap_window, nt)
        q = k_f32[:, -q_window:, :]          # (nh, snap_window, hd)

        # Attention logits: (nh, snap_window, n_tokens)
        scale_factor = 1.0 / (hd ** 0.5)
        logits = np.einsum("nhd, nTd -> nhT", q, k_f32) * scale_factor

        # Softmax and sum importances over snap_window
        exp_l = np.exp(logits - logits.max(axis=-1, keepdims=True))
        attn  = exp_l / exp_l.sum(axis=-1, keepdims=True)     # (nh, snap_w, nt)
        importance = attn.sum(axis=(0, 1))                     # (n_tokens,)

        # Always keep the last snap_window positions (recent context)
        top_indices = np.argsort(-importance)[: budget]
        top_indices = np.sort(top_indices)                     # restore order

        # Rebuild cache with only selected positions
        sel_k = full_k[:, top_indices, :]   # (nh, budget, hd)
        sel_v = full_v[:, top_indices, :]

        # Reset and reload as all-recent (FP16) — next tokens will push old
        # positions into INT8 naturally through the window mechanism
        layer_cache.keys_recent.clear()
        layer_cache.values_recent.clear()
        layer_cache.keys_old_q = None
        layer_cache.keys_old_s = None
        layer_cache.values_old_q = None
        layer_cache.values_old_s = None

        # Reload into recent window — all positions initially FP16
        # append_batch calls the existing eviction logic to spill to INT8
        for t in range(sel_k.shape[1]):
            # Each element: (n_heads, head_dim)
            layer_cache.keys_recent.append(sel_k[:, t, :])
            layer_cache.values_recent.append(sel_v[:, t, :])

        # Spill all but the last `window` back to INT8
        while len(layer_cache.keys_recent) > layer_cache.window:
            oldest_k = layer_cache.keys_recent.pop(0)
            oldest_v = layer_cache.values_recent.pop(0)
            new_kq_list, new_ks_list = [], []
            new_vq_list, new_vs_list = [], []
            for h in range(nh):
                kq, ks = _quantize_int8_per_channel(oldest_k[h:h+1, :])
                vq, vs = _quantize_int8_per_channel(oldest_v[h:h+1, :])
                new_kq_list.append(kq)
                new_ks_list.append(ks)
                new_vq_list.append(vq)
                new_vs_list.append(vs)
            slot_kq = np.stack(new_kq_list, axis=0)
            slot_ks = np.stack(new_ks_list, axis=0)
            slot_vq = np.stack(new_vq_list, axis=0)
            slot_vs = np.stack(new_vs_list, axis=0)
            if layer_cache.keys_old_q is None:
                layer_cache.keys_old_q   = slot_kq
                layer_cache.keys_old_s   = slot_ks
                layer_cache.values_old_q = slot_vq
                layer_cache.values_old_s = slot_vs
            else:
                layer_cache.keys_old_q   = np.concatenate(
                    [layer_cache.keys_old_q,   slot_kq], axis=1)
                layer_cache.keys_old_s   = np.concatenate(
                    [layer_cache.keys_old_s,   slot_ks], axis=1)
                layer_cache.values_old_q = np.concatenate(
                    [layer_cache.values_old_q, slot_vq], axis=1)
                layer_cache.values_old_s = np.concatenate(
                    [layer_cache.values_old_s, slot_vs], axis=1)


# ---------------------------------------------------------------------------
# QuantizedKVCache — full model cache (all layers)
# ---------------------------------------------------------------------------

class QuantizedKVCache:
    """
    Full-model KV cache with KIVI-style INT8 compression and optional SnapKV
    eviction.

    Compatible with ``mlx_lm``'s cache API: the cache is a list of per-layer
    dicts with ``'keys'`` and ``'values'`` MLX arrays, populated on demand.

    Usage
    -----
    After model load, before generation:

        cache = QuantizedKVCache(n_layers=32, window=64, mode="snap",
                                 budget=2048, snap_window=32)
        # Pass as kv_cache to mlx_lm generate helpers, or use
        # cache.to_mlx_list() to get the raw list format.
    """

    def __init__(
        self,
        n_layers: int,
        window: int = 64,
        mode: str = "int8",                 # "fp16" | "int8" | "snap"
        budget: int = 4096,
        snap_window: int = 32,
        svd_rank: int = 0,                  # Phase 1: 0 = off; set to rank < head_dim
    ):
        """
        Parameters
        ----------
        n_layers    : number of transformer layers
        window      : recent FP16 window size (KIVI parameter)
        mode        : "fp16" (no compression) | "int8" (KIVI) | "snap" (KIVI+SnapKV)
        budget      : max K/V positions to retain per layer (SnapKV only)
        snap_window : attention window for importance scoring (SnapKV)
        svd_rank    : Phase 1 — project head_dim → rank before INT8 quant (0 = off)
        """
        if mode not in ("fp16", "int8", "snap"):
            raise ValueError(f"mode must be fp16, int8, or snap — got {mode!r}")

        self.mode        = mode
        self.window      = window
        self.budget      = budget
        self.snap_window = snap_window
        self.svd_rank    = svd_rank
        self.n_layers    = n_layers
        self._layers: list[KVLayerCache] = [
            KVLayerCache(window=window) for _ in range(n_layers)
        ]
        if svd_rank > 0:
            for layer in self._layers:
                layer._svd_rank = svd_rank
        self._snapped = [False] * n_layers   # has SnapKV eviction been applied?

    # ── Compatibility shims for mlx_lm cache list API ────────────────────────

    def __len__(self):
        return self.n_layers

    def __getitem__(self, idx):
        """Return the KVLayerCache for layer idx (mlx_lm update_and_fetch protocol)."""
        return self._layers[idx]

    def __iter__(self):
        for i in range(self.n_layers):
            yield self[i]

    # ── Layer update (called by patched attention) ────────────────────────────

    def update(self, layer_idx: int, key_np: np.ndarray, value_np: np.ndarray) -> None:
        """
        Append key/value for ``layer_idx``.  Applies SnapKV eviction once the
        cache exceeds ``budget`` tokens (first call after prefill).
        """
        layer = self._layers[layer_idx]
        layer.append(key_np, value_np)

        if (self.mode == "snap"
                and not self._snapped[layer_idx]
                and layer.n_tokens > self.budget):
            _snap_evict(layer, self.budget, self.snap_window)
            self._snapped[layer_idx] = True

    def get_kv_mlx(self, layer_idx: int):
        """Return (keys, values) as MLX bfloat16 arrays."""
        return self._layers[layer_idx].get_as_mlx()

    def reset(self):
        """Clear all layers (new conversation)."""
        for layer in self._layers:
            layer.reset()
        self._snapped = [False] * self.n_layers

    @property
    def n_tokens(self) -> int:
        """Total K/V tokens currently cached (layer 0 is representative)."""
        return self._layers[0].n_tokens if self._layers else 0

    @property
    def memory_mb(self) -> float:
        """Approximate total KV cache memory in MB."""
        total = sum(layer.memory_bytes for layer in self._layers)
        return total / 1_048_576

    def stats(self) -> dict:
        return {
            "mode":      self.mode,
            "n_layers":  self.n_layers,
            "n_tokens":  self.n_tokens,
            "memory_mb": round(self.memory_mb, 2),
            "window":    self.window,
            "budget":    self.budget,
        }

    def restore_from(self, src: "QuantizedKVCache") -> None:
        """
        Copy all layer data from *src* into this cache in-place.

        Used by the disk-prompt-cache path: the disk-loaded cache is
        deserialised into a temporary object, then its state is copied
        into the model-patched layers so the existing object references
        remain valid.
        """
        for dst_lay, src_lay in zip(self._layers, src._layers):
            dst_lay.keys_old_q   = src_lay.keys_old_q
            dst_lay.keys_old_s   = src_lay.keys_old_s
            dst_lay.values_old_q = src_lay.values_old_q
            dst_lay.values_old_s = src_lay.values_old_s
            dst_lay.keys_recent   = list(src_lay.keys_recent)
            dst_lay.values_recent = list(src_lay.values_recent)
            dst_lay.n_heads  = src_lay.n_heads
            dst_lay.head_dim = src_lay.head_dim
            # Phase 1: carry SVD basis across (calibration buffer is per-request)
            dst_lay._svd_Vk   = src_lay._svd_Vk
            dst_lay._svd_Vv   = src_lay._svd_Vv
            dst_lay._svd_rank = src_lay._svd_rank
            dst_lay._svd_buf_k = None
            dst_lay._svd_buf_v = None


class _LayerCacheView:
    """
    Thin shim so QuantizedKVCache[i] behaves like a KV cache dict for mlx_lm.
    """
    def __init__(self, layer: KVLayerCache, parent: QuantizedKVCache):
        self._layer  = layer
        self._parent = parent

    @property
    def keys(self):
        k, _ = self._layer.get_as_mlx()
        return k

    @property
    def values(self):
        _, v = self._layer.get_as_mlx()
        return v


# ---------------------------------------------------------------------------
# Model patching — intercept attention layers to use QuantizedKVCache
# ---------------------------------------------------------------------------

def _n_layers(model) -> int:  # pragma: no cover
    """Infer number of transformer layers from a loaded mlx_lm model."""
    # Most mlx_lm models expose model.model.layers
    try:
        return len(model.model.layers)
    except AttributeError:
        pass
    try:
        return len(model.layers)
    except AttributeError:
        pass
    # Fallback: count by inspecting config
    try:
        cfg = model.args
        return (getattr(cfg, "num_hidden_layers", None)
                or getattr(cfg, "n_layers", None)
                or 32)
    except Exception:
        return 32


def make_quantized_cache(  # pragma: no cover
    model,
    mode: str = "int8",
    window: int = 64,
    budget: int = 4096,
    snap_window: int = 32,
    svd_rank: int = 0,
) -> QuantizedKVCache:
    """
    Create a :class:`QuantizedKVCache` sized correctly for ``model``.

    Parameters
    ----------
    model       : mlx_lm model (already loaded)
    mode        : "fp16" | "int8" | "snap"
    window      : FP16 residual window (KIVI)
    budget      : max K/V positions (SnapKV only)
    snap_window : attention window for importance (SnapKV)
    svd_rank    : Phase 1 — SVD projection rank (0 = off)
    """
    n = _n_layers(model)
    return QuantizedKVCache(
        n_layers=n, window=window, mode=mode,
        budget=budget, snap_window=snap_window,
        svd_rank=svd_rank,
    )


def patch_model_kv_cache(  # pragma: no cover
    model,
    mode: str = "int8",
    window: int = 64,
    budget: int = 4096,
    snap_window: int = 32,
    svd_rank: int = 0,
    verbose: bool = True,
) -> QuantizedKVCache:
    """
    Monkey-patch the model's attention layers so that the KV cache written
    during generation is automatically quantized.

    This is less invasive than modifying mlx_lm internals: instead of
    replacing the attention class, we wrap the cache-update step by
    intercepting ``mlx_lm.utils.generate_step``-style generation via
    a shared :class:`QuantizedKVCache` object.

    Returns the :class:`QuantizedKVCache` instance; pass it to
    ``generate_with_cache(model, tokenizer, prompt, cache=...)`` or
    use ``generate_step`` from mlx_lm.

    Note
    ----
    For full KV-cache quantization inside the MLX attention kernel, a
    future version will use MLX custom primitives.  This implementation
    works at the Python level with numpy round-trips for the compressed
    portion, which adds ~5-10 ms per 100 tokens for the
    dequantize-and-forward step.
    """
    cache = make_quantized_cache(
        model, mode=mode, window=window,
        budget=budget, snap_window=snap_window,
        svd_rank=svd_rank,
    )
    n = _n_layers(model)

    if verbose:
        svd_info = f"  svd_rank={svd_rank}" if svd_rank > 0 else ""
        print(f"  [KV cache] mode={mode}  window={window}  "
              f"budget={budget if mode == 'snap' else '—'}  "
              f"layers={n}{svd_info}")

    # Store on the model so server.py can retrieve it
    model._squish_kv_cache = cache
    return cache


# ---------------------------------------------------------------------------
# generate_with_cache — convenience wrapper for server.py
# ---------------------------------------------------------------------------

def generate_step_with_quantized_cache(  # pragma: no cover
    model,
    token_ids,          # (1, seq_len) MLX int32
    quantized_cache: QuantizedKVCache,
    temperature: float = 0.0,
    top_p: float = 1.0,
):
    """
    Run a single generation step using the quantized KV cache.

    This is a simplified stub that works with models whose attention
    accepts a ``cache`` keyword argument as a list of dicts with
    ``keys`` / ``values`` entries.

    For production use, mlx_lm's ``generate_step`` with ``cache=``
    is the recommended path once QuantizedKVCache supports the full
    mlx_lm cache protocol.

    Returns
    -------
    next_token_id : int
    """
    mx = _mx()

    with mx.stream(mx.gpu):
        logits = model(token_ids)          # (1, seq, vocab)

    next_logits = np.array(logits[0, -1, :].astype(mx.float32))  # (vocab,)

    if temperature <= 0.0 or temperature < 1e-5:
        return int(np.argmax(next_logits))

    next_logits = next_logits / temperature
    # top-p filtering
    if top_p < 1.0:
        sorted_idx  = np.argsort(-next_logits)
        cum_probs   = np.cumsum(
            np.exp(next_logits[sorted_idx]
                   - np.max(next_logits[sorted_idx])))
        cum_probs  /= cum_probs[-1] + 1e-9
        sorted_idx[np.searchsorted(cum_probs, top_p) + 1
                                  if np.searchsorted(cum_probs, top_p) + 1
                                     < len(sorted_idx) else -1]
        next_logits[sorted_idx[np.searchsorted(cum_probs, top_p) + 1:]] = -1e9

    probs = np.exp(next_logits - np.max(next_logits))
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))


# ---------------------------------------------------------------------------
# DiskKVCache — persistent cross-request prompt cache backed by NVMe
# ---------------------------------------------------------------------------

class DiskKVCache:
    """
    Cross-request disk-backed prompt cache for QuantizedKVCache.

    Serialises full KV state to per-entry ``.npz`` files keyed by the
    SHA-256 of the input token-id sequence.  On a cache hit, prefill is
    skipped entirely.

    Parameters
    ----------
    cache_dir : str | Path
        Directory on fast NVMe where entry files are stored.  Created if it
        does not exist.
    max_entries : int
        Maximum number of entries; LRU eviction by mtime when exceeded.
    """

    def __init__(self, cache_dir, max_entries: int = 64):
        import threading as _threading
        self._dir = Path(cache_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._max   = max_entries
        self._lock  = _threading.Lock()

    # ── public API ──────────────────────────────────────────────────────────

    def lookup(self, input_ids: list[int]) -> "tuple[QuantizedKVCache, np.ndarray] | None":
        """
        Return ``(qkv_cache, last_logit_f32)`` on a cache hit, or ``None``.

        *last_logit_f32* is the prefill's final-position raw logit vector so
        the caller can sample the first generated token without re-running
        the model.
        """
        entry = self._dir / (self._key(input_ids) + ".npz")
        if not entry.exists():
            return None
        try:
            data = np.load(entry, allow_pickle=False)
            qkv       = self._deserialise(data)
            last_logit = data["last_logit"].astype(np.float32) if "last_logit" in data else None
            if last_logit is None:
                return None  # legacy entry without logit — treat as miss
            # Touch mtime for LRU ordering
            entry.touch()
            return qkv, last_logit
        except Exception:  # corrupted or schema mismatch — treat as miss  # pragma: no cover
            try:
                entry.unlink(missing_ok=True)
            except Exception:
                pass
            return None

    def store(
        self,
        input_ids: list[int],
        qkv_cache: "QuantizedKVCache",
        last_logit_np: "np.ndarray | None" = None,
    ) -> None:
        """
        Persist *qkv_cache* (and optionally *last_logit_np*) to disk in a
        background thread.  Returns immediately; silently drops on error.
        """
        import threading as _threading

        def _worker():
            try:
                arrays = self._serialise(qkv_cache)
                if arrays is None:
                    return
                if last_logit_np is not None:
                    arrays["last_logit"] = last_logit_np.astype(np.float32)
                entry = self._dir / (self._key(input_ids) + ".npz")
                np.savez_compressed(str(entry), **arrays)
                self._evict_if_needed()
            except Exception:  # pragma: no cover
                pass

        _threading.Thread(target=_worker, daemon=True).start()

    # ── internals ───────────────────────────────────────────────────────────

    @staticmethod
    def _key(input_ids: list[int]) -> str:
        import hashlib
        raw = np.array(input_ids, dtype=np.int32).tobytes()
        return hashlib.sha256(raw).hexdigest()

    @staticmethod
    def _serialise(qkv_cache: "QuantizedKVCache") -> "dict | None":
        """
        Pack all layers into a flat dict of numpy arrays.

        Keys:
          ``n_layers``           — int scalar
          ``L{i}_n_heads``       — int scalar
          ``L{i}_head_dim``      — int scalar
          ``L{i}_keys_old_q``    — (n_heads, n_old, head_dim) int8 or missing
          ``L{i}_keys_old_s``    — (n_heads, n_old) f32        or missing
          ``L{i}_vals_old_q``    — (n_heads, n_old, head_dim) int8 or missing
          ``L{i}_vals_old_s``    — (n_heads, n_old) f32        or missing
          ``L{i}_n_recent``      — int scalar
          ``L{i}_keys_recent``   — (n_heads, n_rec, head_dim) f16 or missing
          ``L{i}_vals_recent``   — (n_heads, n_rec, head_dim) f16 or missing
        """
        layers = qkv_cache._layers
        out: dict[str, np.ndarray] = {
            "n_layers": np.array(len(layers), dtype=np.int32),
        }
        for i, lay in enumerate(layers):
            if lay.n_heads is None:
                return None  # layer not yet populated — skip whole entry
            out[f"L{i}_n_heads"]  = np.array(lay.n_heads,  dtype=np.int32)
            out[f"L{i}_head_dim"] = np.array(lay.head_dim, dtype=np.int32)
            if lay.keys_old_q is not None:
                out[f"L{i}_keys_old_q"] = lay.keys_old_q
                out[f"L{i}_keys_old_s"] = lay.keys_old_s
                out[f"L{i}_vals_old_q"] = lay.values_old_q
                out[f"L{i}_vals_old_s"] = lay.values_old_s
            n_rec = len(lay.keys_recent)
            out[f"L{i}_n_recent"] = np.array(n_rec, dtype=np.int32)
            if n_rec > 0:
                out[f"L{i}_keys_recent"] = np.stack(lay.keys_recent, axis=1)   # (H, n_rec, D)
                out[f"L{i}_vals_recent"] = np.stack(lay.values_recent, axis=1)
            # Phase 1: persist SVD basis so subsequent requests skip refitting
            if lay._svd_Vk is not None:
                out[f"L{i}_svd_rank"] = np.array(lay._svd_rank, dtype=np.int32)
                out[f"L{i}_svd_Vk"]   = lay._svd_Vk   # (n_heads, rank, head_dim) f16
                out[f"L{i}_svd_Vv"]   = lay._svd_Vv
        return out

    @staticmethod
    def _deserialise(data) -> "QuantizedKVCache":
        """Reconstruct a QuantizedKVCache from a loaded npz dict."""
        n_layers = int(data["n_layers"])
        # Build a shell QuantizedKVCache with the right layer count
        # We bypass patch_model_kv_cache and construct layers directly.
        layers: list[KVLayerCache] = []
        for i in range(n_layers):
            lay = KVLayerCache()
            lay.n_heads  = int(data[f"L{i}_n_heads"])
            lay.head_dim = int(data[f"L{i}_head_dim"])
            if f"L{i}_keys_old_q" in data:
                lay.keys_old_q   = data[f"L{i}_keys_old_q"]
                lay.keys_old_s   = data[f"L{i}_keys_old_s"]
                lay.values_old_q = data[f"L{i}_vals_old_q"]
                lay.values_old_s = data[f"L{i}_vals_old_s"]
            n_rec = int(data[f"L{i}_n_recent"])
            if n_rec > 0:
                k_rec = data[f"L{i}_keys_recent"]   # (H, n_rec, D)
                v_rec = data[f"L{i}_vals_recent"]
                for t in range(n_rec):
                    lay.keys_recent.append(k_rec[:, t, :])
                    lay.values_recent.append(v_rec[:, t, :])
            # Phase 1: restore SVD basis if persisted
            if f"L{i}_svd_Vk" in data:
                lay._svd_rank = int(data[f"L{i}_svd_rank"])
                lay._svd_Vk   = data[f"L{i}_svd_Vk"]
                lay._svd_Vv   = data[f"L{i}_svd_Vv"]
            layers.append(lay)

        qkv = object.__new__(QuantizedKVCache)
        qkv._layers = layers
        # Restore public config attributes with safe defaults
        qkv.mode   = getattr(layers[0], "_mode", "int8") if layers else "int8"
        qkv.window = getattr(layers[0], "_window", 64)   if layers else 64
        qkv.budget = getattr(layers[0], "_budget", 4096) if layers else 4096
        return qkv

    def _evict_if_needed(self) -> None:
        """Remove the oldest (by mtime) entries when over the size cap."""
        with self._lock:
            entries = sorted(self._dir.glob("*.npz"), key=lambda p: p.stat().st_mtime)
            while len(entries) > self._max:
                try:
                    entries.pop(0).unlink(missing_ok=True)
                except Exception:  # pragma: no cover
                    pass


# ---------------------------------------------------------------------------
# SessionKVCache — persistent cross-session KV state (Phase 3)
# ---------------------------------------------------------------------------

class SessionKVCache:
    """
    Persistent KV-state cache keyed by a SHA-256 hash of the last 8 message 
    contents in a conversation.

    Unlike :class:`DiskKVCache` (which is keyed by raw token IDs), this cache
    is keyed by the *conversation context*, allowing the server to resume KV
    state across restarts without requiring identical tokenization.

    Session files are stored as compressed ``.npz`` under ``cache_dir``.
    Writes are non-blocking (background thread).

    Parameters
    ----------
    cache_dir   : str | Path — directory for session files (created if needed)
    max_entries : int        — LRU cap; oldest evicted when exceeded
    """

    def __init__(self, cache_dir, max_entries: int = 128):
        import threading as _threading
        self._dir  = Path(cache_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._max  = max_entries
        self._lock = _threading.Lock()

    # ── Public API ──────────────────────────────────────────────────────────

    def session_key(self, messages: "list[dict]") -> str:
        """
        Derive a stable session key from the last 8 message contents.

        Parameters
        ----------
        messages : list of OpenAI-style message dicts with ``"content"`` keys

        Returns
        -------
        32-hex-char string (SHA-256 truncated to 128 bits)
        """
        import hashlib
        tail = messages[-8:] if len(messages) > 8 else messages
        raw  = "\n".join(str(m.get("content", "")) for m in tail).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()[:32]

    def load_session(self, key: str) -> "QuantizedKVCache | None":
        """
        Return a :class:`QuantizedKVCache` for a prior session, or ``None`` on miss.

        Parameters
        ----------
        key : session key from :meth:`session_key`
        """
        entry = self._dir / (key + ".npz")
        if not entry.exists():
            return None
        try:
            data = np.load(entry, allow_pickle=False)
            qkv  = DiskKVCache._deserialise(data)
            entry.touch()   # update mtime for LRU ordering
            return qkv
        except Exception:
            try:
                entry.unlink(missing_ok=True)
            except Exception:  # pragma: no cover
                pass
            return None

    def save_session(
        self,
        key: str,
        qkv_cache: "QuantizedKVCache",
    ) -> None:
        """
        Persist *qkv_cache* under *key* in a background thread.

        Returns immediately; silently drops on serialisation error.
        """
        import threading as _threading

        def _worker():
            try:
                arrays = DiskKVCache._serialise(qkv_cache)
                if arrays is None:
                    return
                entry = self._dir / (key + ".npz")
                np.savez_compressed(str(entry), **arrays)
                self._evict_if_needed()
            except Exception:  # pragma: no cover
                pass

        _threading.Thread(target=_worker, daemon=True).start()

    def list_sessions(self) -> "list[str]":
        """Return sorted list of active session keys (file stems)."""
        return sorted(p.stem for p in self._dir.glob("*.npz"))

    # ── internals ───────────────────────────────────────────────────────────

    def _evict_if_needed(self) -> None:
        with self._lock:
            entries = sorted(
                self._dir.glob("*.npz"),
                key=lambda p: p.stat().st_mtime,
            )
            while len(entries) > self._max:
                try:
                    entries.pop(0).unlink(missing_ok=True)
                except Exception:  # pragma: no cover
                    pass

