"""squish/kv_defrag.py

KVDefragmenter — Online KV cache defragmentation and in-place compaction.

In a paged KV cache, each sequence is assigned a list of physical pages
from a shared page pool.  As sequences are created and destroyed in a
multi-request serving loop, freed pages leave gaps in the pool.  This
external fragmentation has two costs:

1. **Memory inefficiency** — the highest-indexed allocated page determines
   the "active" pool range.  Free pages interspersed throughout that range
   cannot be returned to the OS without compaction.

2. **Locality loss** — scattered page layouts cause cache-unfriendly memory
   access patterns during attention computation.

:class:`KVDefragmenter` maintains a free-list of available physical pages and
a page table mapping each live sequence to its list of page indices.  Each
page stores the actual KV data for ``page_size`` tokens across all heads
(shape ``(2, n_heads, page_size, head_dim)`` where axis-0 is ``[keys, values]``
packed together).

The :meth:`defrag` operation compacts all allocated pages to the front of the
pool (indices ``0 .. n_allocated - 1``), updates every sequence's page-table
entries to reflect their new positions, and rebuilds the free list as the
contiguous suffix ``[n_allocated .. n_total_pages - 1]``.  The pool itself is
not shrunk (total page count is preserved), but the contiguous free region at
the end could be reclaimed by a higher-level allocator.

:attr:`fragmentation_ratio` measures the fraction of the pool's "active
range" (``[0 .. max_allocated_page]``) that is occupied by free pages.  After
a successful defrag this is guaranteed to be ``0.0``.

Example usage::

    import numpy as np
    from squish.kv_defrag import KVDefragmenter, DefragStats

    defrag = KVDefragmenter(page_size=16, n_heads=4, head_dim=32)

    pages_a = defrag.allocate(seq_id=1, n_tokens=40)  # 3 pages
    pages_b = defrag.allocate(seq_id=2, n_tokens=24)  # 2 pages
    defrag.free(seq_id=1)                             # creates gaps
    print(f"frag={defrag.fragmentation_ratio:.2f}")

    stats = defrag.defrag()
    print(stats)
    print(f"frag_after={defrag.fragmentation_ratio:.2f}")  # 0.0
"""

from __future__ import annotations

__all__ = ["DefragStats", "KVDefragmenter"]

import math
from collections import deque
from dataclasses import dataclass
from typing import Deque

import numpy as np


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------


@dataclass
class DefragStats:
    """Statistics returned by a single :meth:`~KVDefragmenter.defrag` call.

    Attributes:
        n_pages_before:       Total pages in the pool before defrag.
        n_pages_after:        Total pages in the pool after defrag
                              (unchanged; the pool is not shrunk).
        bytes_freed:          Conceptual bytes freed: the number of free pages
                              that were interspersed in the active pool range
                              before defrag, multiplied by the page byte size.
                              These are now consolidated into a contiguous free
                              suffix that could be returned to the OS.
        fragmentation_before: :attr:`~KVDefragmenter.fragmentation_ratio`
                              measured immediately before compaction.
        fragmentation_after:  :attr:`~KVDefragmenter.fragmentation_ratio`
                              measured immediately after compaction (always 0.0
                              when there is at least one allocated page).
    """

    n_pages_before:       int
    n_pages_after:        int
    bytes_freed:          int
    fragmentation_before: float
    fragmentation_after:  float


# ---------------------------------------------------------------------------
# Defragmenter
# ---------------------------------------------------------------------------


class KVDefragmenter:
    """Online KV cache page-table manager with defragmentation support.

    Pages are allocated lazily: the pool starts empty and grows on demand.
    Each page stores KV data for ``page_size`` token positions as a NumPy
    array of shape ``(2, n_heads, page_size, head_dim)`` (float32), where
    the first dimension encodes ``[keys, values]``.

    The free list is a :class:`collections.deque` of available page indices;
    :meth:`allocate` pops from the front and :meth:`free` appends to the back.
    When the free list is empty, :meth:`allocate` extends the pool by appending
    new zero-initialised pages.

    Args:
        page_size: Number of token slots per page.  Must be >= 1.
        n_heads:   Number of attention heads per page.  Must be >= 1.
        head_dim:  Head dimension.  Must be >= 1.
    """

    def __init__(
        self,
        page_size: int = 16,
        n_heads:   int = 4,
        head_dim:  int = 32,
    ) -> None:
        if page_size < 1:
            raise ValueError(f"page_size must be >= 1, got {page_size}")
        if n_heads < 1:
            raise ValueError(f"n_heads must be >= 1, got {n_heads}")
        if head_dim < 1:
            raise ValueError(f"head_dim must be >= 1, got {head_dim}")

        self._page_size = page_size
        self._n_heads   = n_heads
        self._head_dim  = head_dim

        # Bytes per page: 2 (K+V) × n_heads × page_size × head_dim × float32.
        self._page_bytes: int = 2 * n_heads * page_size * head_dim * 4

        # Physical page pool: list of np.ndarray, each (2, n_heads, page_size, head_dim).
        self._pages:      list[np.ndarray] = []
        # Free list of available page indices.
        self._free_list:  Deque[int] = deque()
        # Page table: seq_id → list of physical page indices.
        self._page_table: dict[int, list[int]] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def allocate(self, seq_id: int, n_tokens: int) -> list[int]:
        """Allocate pages for *seq_id* to cover *n_tokens* token slots.

        The number of pages allocated is ``ceil(n_tokens / page_size)``.
        If the sequence already has an allocation it is replaced; the old
        pages are returned to the free list first.

        Args:
            seq_id:   Integer identifier for the sequence.
            n_tokens: Number of token slots required.  Must be >= 1.

        Returns:
            List of physical page indices assigned to *seq_id*.

        Raises:
            ValueError: If *n_tokens* < 1.
        """
        if n_tokens < 1:
            raise ValueError(f"n_tokens must be >= 1, got {n_tokens}")

        # Release any existing pages for this sequence.
        if seq_id in self._page_table:
            self.free(seq_id)

        n_pages_needed = math.ceil(n_tokens / self._page_size)
        assigned: list[int] = []

        for _ in range(n_pages_needed):
            if self._free_list:
                page_idx = self._free_list.popleft()
            else:
                # Extend the pool.
                page_idx = len(self._pages)
                self._pages.append(
                    np.zeros(
                        (2, self._n_heads, self._page_size, self._head_dim),
                        dtype=np.float32,
                    )
                )
            assigned.append(page_idx)

        self._page_table[seq_id] = assigned
        return assigned

    def free(self, seq_id: int) -> None:
        """Free all pages belonging to *seq_id* and return them to the pool.

        Args:
            seq_id: Integer identifier of the sequence to free.

        Raises:
            KeyError: If *seq_id* is not currently allocated.
        """
        if seq_id not in self._page_table:
            raise KeyError(
                f"seq_id {seq_id!r} is not currently allocated."
            )
        for page_idx in self._page_table.pop(seq_id):
            # Zero the page so stale data is never accidentally exposed.
            self._pages[page_idx][:] = 0.0
            self._free_list.append(page_idx)

    def defrag(self) -> DefragStats:
        """Compact all allocated pages to the front of the pool.

        All currently allocated pages are remapped to indices
        ``0 .. n_allocated - 1`` in a deterministic order (sorted by their
        original page index).  The page table is updated in-place, physical
        KV data is copied to the new positions, and the free list is rebuilt
        as the contiguous suffix ``[n_allocated .. n_total_pages - 1]``.

        Returns:
            A :class:`DefragStats` instance describing the operation.

        Note:
            The pool is not shrunk.  ``n_pages_after == n_pages_before``.
        """
        n_total   = len(self._pages)
        frag_before = self.fragmentation_ratio

        # Compute bytes "trapped" in fragmented positions before defrag.
        fragmented_free_pages = self._count_fragmented_free_pages()
        bytes_freed = fragmented_free_pages * self._page_bytes

        # Collect all (seq_id, local_position, current_page_idx) triples,
        # sorted by current page index for deterministic new assignment.
        alloc_entries: list[tuple[int, int, int]] = []
        for seq_id, pages in self._page_table.items():
            for local_pos, page_idx in enumerate(pages):
                alloc_entries.append((seq_id, local_pos, page_idx))
        alloc_entries.sort(key=lambda t: t[2])

        n_alloc = len(alloc_entries)

        if n_alloc == 0:
            # Nothing to compact.
            self._free_list = deque(range(n_total))
            return DefragStats(
                n_pages_before=n_total,
                n_pages_after=n_total,
                bytes_freed=0,
                fragmentation_before=frag_before,
                fragmentation_after=0.0,
            )

        # Copy physical KV data into a temporary buffer indexed by new position.
        # Using a temporary list avoids clobbering source data during in-place moves.
        temp_data: list[np.ndarray] = [
            self._pages[old_idx].copy()
            for _, _, old_idx in alloc_entries
        ]

        # Write compacted data back into the pool at new indices 0..n_alloc-1.
        for new_idx, data in enumerate(temp_data):
            self._pages[new_idx] = data

        # Rebuild the page table with new indices.
        new_page_table: dict[int, list[int]] = {}
        for new_idx, (seq_id, local_pos, _) in enumerate(alloc_entries):
            if seq_id not in new_page_table:
                original_len = len(self._page_table[seq_id])
                new_page_table[seq_id] = [0] * original_len
            new_page_table[seq_id][local_pos] = new_idx
        self._page_table = new_page_table

        # Rebuild the free list as the contiguous suffix.
        self._free_list = deque(range(n_alloc, n_total))

        frag_after = self.fragmentation_ratio
        return DefragStats(
            n_pages_before=n_total,
            n_pages_after=n_total,
            bytes_freed=bytes_freed,
            fragmentation_before=frag_before,
            fragmentation_after=frag_after,
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def fragmentation_ratio(self) -> float:
        """Fraction of the active pool range occupied by free (wasted) pages.

        The active range is defined as ``[0, max_allocated_page_index]``.
        Free pages interspersed within this range are "fragmented".  The ratio
        equals ``n_fragmented_free_pages / (max_allocated_page_index + 1)``.

        Returns ``0.0`` when no pages are allocated or the pool is empty.
        """
        all_allocated = self._all_allocated_page_indices()
        if not all_allocated:
            return 0.0
        max_page = max(all_allocated)
        allocated_set = set(all_allocated)
        n_free_in_range = sum(
            1 for i in range(max_page + 1) if i not in allocated_set
        )
        return n_free_in_range / (max_page + 1)

    @property
    def utilization(self) -> float:
        """Fraction of total physical pages currently allocated to sequences.

        Returns ``0.0`` when the pool is empty.
        """
        n_total = len(self._pages)
        if n_total == 0:
            return 0.0
        n_alloc = sum(len(pages) for pages in self._page_table.values())
        return n_alloc / n_total

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _all_allocated_page_indices(self) -> list[int]:
        """Return a flat list of all page indices currently in the page table."""
        result: list[int] = []
        for pages in self._page_table.values():
            result.extend(pages)
        return result

    def _count_fragmented_free_pages(self) -> int:
        """Count free pages that lie within the active allocated range."""
        all_allocated = self._all_allocated_page_indices()
        if not all_allocated:
            return 0
        max_page      = max(all_allocated)
        allocated_set = set(all_allocated)
        return sum(
            1 for i in range(max_page + 1) if i not in allocated_set
        )
