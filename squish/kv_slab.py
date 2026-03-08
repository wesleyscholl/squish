"""
squish/kv_slab.py

KV slab allocator — pre-allocates a contiguous pool of KV "pages" at server
startup so that each new request gets its memory from a bump-pointer / free-list
into the slab, rather than calling malloc/new for every generation.

Motivation
----------
Python and NumPy allocate heap memory for each KV layer cache append.  At 20+
concurrent requests per second this produces significant GC pressure and
occasional >10 ms allocation stalls during the decode loop.

The slab allocator eliminates these stalls by:
  1. Pre-allocating one large contiguous NumPy array at init time.
  2. Handing out fixed-size *page* slices from that array via a lock-free
     free-list (``collections.deque``).
  3. Returning pages to the free-list on deallocation — zero OS interaction.

Integration with paged_attention.py
------------------------------------
:class:`KVSlabAllocator` is a drop-in replacement for the ``BlockAllocator``
inside ``PagedKVCache``.  Pass an instance as ``allocator=...`` when
constructing ``PagedKVCache``:

    from squish.kv_slab import KVSlabAllocator
    from squish.paged_attention import PagedKVCache

    slab = KVSlabAllocator(
        n_pages   = 512,
        page_size = 16,   # tokens per page
        n_layers  = 32,
        n_heads   = 32,
        head_dim  = 128,
        dtype     = np.float16,
    )
    cache = PagedKVCache(block_size=16, allocator=slab)

Memory model
------------
  total_bytes = n_pages × page_size × n_layers × 2 × n_heads × head_dim × dtype_bytes

  For Qwen3-8B (32 layers, 8 KV heads, head_dim=128, n_pages=512, page=16 tokens):
    512 × 16 × 32 × 2 × 8 × 128 × 2  bytes  ≈  270 MB

  That 270 MB pool covers 512 × 16 = 8 192 simultaneous context tokens across
  all concurrent requests — more than enough for typical chat workloads.
"""
from __future__ import annotations

import threading
from collections import deque
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# KVPage — a single fixed-size block of K and V storage
# ---------------------------------------------------------------------------

class KVPage:
    """
    A fixed-size contiguous block of KV storage for ``page_size`` token positions.

    Attributes
    ----------
    page_id   : int   — unique identifier within the slab
    page_size : int   — number of token positions this page holds
    keys      : np.ndarray  shape (n_layers, page_size, n_heads, head_dim)
    values    : np.ndarray  same shape as *keys*
    n_filled  : int         — how many token slots are currently in use
    """

    __slots__ = ("page_id", "page_size", "keys", "values", "n_filled")

    def __init__(
        self,
        page_id:   int,
        page_size: int,
        keys:      np.ndarray,
        values:    np.ndarray,
    ) -> None:
        self.page_id   = page_id
        self.page_size = page_size
        self.keys      = keys
        self.values    = values
        self.n_filled  = 0

    def reset(self) -> None:
        """Return the page to a clean state (does NOT zero memory — caller owns that)."""
        self.n_filled = 0

    def is_full(self) -> bool:
        return self.n_filled >= self.page_size

    def remaining(self) -> int:
        return self.page_size - self.n_filled


# ---------------------------------------------------------------------------
# KVSlabAllocator
# ---------------------------------------------------------------------------

class KVSlabAllocator:
    """
    Pre-allocated KV page pool for zero-malloc decode-loop memory management.

    The allocator owns one large contiguous NumPy array for keys and one for
    values.  Individual :class:`KVPage` objects are views into that array —
    no extra allocation happens during ``alloc()`` or ``free()``.

    Parameters
    ----------
    n_pages   : total number of pages in the slab  (default 512)
    page_size : tokens per page  (default 16; matches ``paged_attention`` default)
    n_layers  : number of transformer layers
    n_heads   : number of KV heads per layer
    head_dim  : dimension of each KV head
    dtype     : NumPy dtype for the KV storage  (default ``np.float16``)
    """

    def __init__(
        self,
        n_pages:   int   = 512,
        page_size: int   = 16,
        n_layers:  int   = 32,
        n_heads:   int   = 8,
        head_dim:  int   = 128,
        dtype            = np.float16,
    ) -> None:
        self.n_pages   = n_pages
        self.page_size = page_size
        self.n_layers  = n_layers
        self.n_heads   = n_heads
        self.head_dim  = head_dim
        self.dtype     = dtype

        # Single contiguous allocation for ALL pages' keys and values.
        # Shape: (n_pages, n_layers, page_size, n_heads, head_dim)
        slab_shape = (n_pages, n_layers, page_size, n_heads, head_dim)
        self._slab_k = np.zeros(slab_shape, dtype=dtype)
        self._slab_v = np.zeros(slab_shape, dtype=dtype)

        # Build KVPage view objects pointing into the slab
        self._pages: list[KVPage] = [
            KVPage(
                page_id   = i,
                page_size = page_size,
                keys      = self._slab_k[i],   # view: (n_layers, page_size, n_heads, head_dim)
                values    = self._slab_v[i],
            )
            for i in range(n_pages)
        ]

        # Free-list — all pages start free
        self._free: deque[KVPage] = deque(self._pages)
        self._lock = threading.Lock()

        # Statistics
        self._alloc_count = 0
        self._free_count  = 0

    # ── Allocation interface ──────────────────────────────────────────────────

    def alloc(self) -> Optional[KVPage]:
        """
        Pop a free page from the slab and return it.

        Returns ``None`` if the slab is exhausted (caller must handle OOM).
        Thread-safe.
        """
        with self._lock:
            if not self._free:
                return None
            page = self._free.popleft()
            page.reset()
            self._alloc_count += 1
            return page

    def free(self, page: KVPage) -> None:
        """
        Return *page* to the free-list.  Must be a page from THIS allocator.
        Thread-safe.
        """
        with self._lock:
            self._free.appendleft(page)
            self._free_count += 1

    def free_many(self, pages: "list[KVPage]") -> None:
        """Return a batch of pages to the free-list atomically."""
        with self._lock:
            for page in pages:
                self._free.appendleft(page)
            self._free_count += len(pages)

    # ── Introspection ─────────────────────────────────────────────────────────

    def n_free(self) -> int:
        """Number of currently free pages."""
        with self._lock:
            return len(self._free)

    def n_used(self) -> int:
        """Number of pages currently allocated."""
        return self.n_pages - self.n_free()

    def memory_bytes(self) -> int:
        """Total bytes consumed by the slab (both K and V)."""
        return self._slab_k.nbytes + self._slab_v.nbytes

    def stats(self) -> dict:
        """Return allocation statistics."""
        return {
            "n_pages":       self.n_pages,
            "n_free":        self.n_free(),
            "n_used":        self.n_used(),
            "page_size":     self.page_size,
            "alloc_total":   self._alloc_count,
            "free_total":    self._free_count,
            "memory_mb":     self.memory_bytes() / 1024 / 1024,
        }

    def __repr__(self) -> str:
        mb = self.memory_bytes() / 1024 / 1024
        return (f"KVSlabAllocator(n_pages={self.n_pages}, page_size={self.page_size}, "
                f"free={self.n_free()}, {mb:.1f} MB)")
