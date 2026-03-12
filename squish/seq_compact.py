"""squish/seq_compact.py

SequenceCompactor — In-place KV sequence compaction after token pruning.

After token pruning (e.g. following eviction by a recency or importance policy)
the KV cache contains gaps: positions whose tokens have been discarded.  Those
gaps waste memory bandwidth on every subsequent attention step because the empty
slots are still iterated over.  SequenceCompactor closes those gaps by gathering
the kept positions contiguously at the front of the ``(n_heads, seq, head_dim)``
arrays without allocating a full-sized scratch buffer.

The compaction is performed by an explicit two-step gather:

1. Compute ``gathered = keys[:, keep_indices, :]`` — NumPy advanced indexing
   always returns a freshly allocated copy for integer-array indices, so there
   is no aliasing regardless of where the kept positions fall in the source.
2. Write the gathered slice back to ``keys[:, :n_kept, :]`` in-place, then
   return a view of that leading slice to the caller.

The returned views share the backing allocation of the caller's arrays, so no
large buffer is duplicated.

Example usage::

    import numpy as np
    from squish.seq_compact import SequenceCompactor

    compactor = SequenceCompactor(n_heads=8, head_dim=64)
    keys   = np.random.randn(8, 128, 64).astype(np.float32)
    values = np.random.randn(8, 128, 64).astype(np.float32)

    mask        = np.ones(128, dtype=bool)
    mask[::4]   = False                        # drop every 4th token
    ck, cv, stats = compactor.compact(keys, values, mask)
    print(stats)
    # CompactStats(n_tokens_before=128, n_tokens_after=96,
    #              bytes_saved=65536, compaction_ratio=0.75)
"""

from __future__ import annotations

__all__ = ["CompactStats", "SequenceCompactor"]

import dataclasses
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class CompactStats:
    """Statistics returned by :meth:`SequenceCompactor.compact`.

    Attributes:
        n_tokens_before:  Sequence length before compaction.
        n_tokens_after:   Sequence length after compaction (number of kept tokens).
        bytes_saved:      Bytes freed by eliminating gap positions, computed as
                          ``(n_tokens_before - n_tokens_after)
                          * n_heads * head_dim * element_bytes * 2``
                          (the factor of 2 accounts for both the key and value
                          tensors).
        compaction_ratio: ``n_tokens_after / n_tokens_before``.  A value of
                          ``1.0`` means nothing was removed.
    """

    n_tokens_before: int
    n_tokens_after: int
    bytes_saved: int
    compaction_ratio: float


# ---------------------------------------------------------------------------
# Compactor
# ---------------------------------------------------------------------------


class SequenceCompactor:
    """In-place KV sequence compactor for use after token-level pruning.

    Given a boolean *keep_mask* (or equivalently an explicit index array via
    :meth:`compact_indices`), gathers the retained KV entries contiguously at
    the front of the ``(n_heads, seq, head_dim)`` arrays.  The compacted slices
    are views into the original allocations, avoiding any large secondary copy.

    Args:
        n_heads:  Number of attention heads.  Must be >= 1.
        head_dim: Dimension of each attention head.  Must be >= 1.

    Raises:
        ValueError: If *n_heads* or *head_dim* are not positive integers.
    """

    def __init__(self, n_heads: int, head_dim: int) -> None:
        if n_heads < 1:
            raise ValueError(f"n_heads must be >= 1, got {n_heads}")
        if head_dim < 1:
            raise ValueError(f"head_dim must be >= 1, got {head_dim}")
        self._n_heads = n_heads
        self._head_dim = head_dim

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compact(
        self,
        keys: np.ndarray,
        values: np.ndarray,
        keep_mask: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, CompactStats]:
        """Compact *keys* and *values* by removing positions where *keep_mask*
        is ``False``.

        The leading ``n_kept`` entries of the original arrays are overwritten
        with the gathered result.  The returned arrays are views into those
        leading slices — no new large allocation is made.

        Args:
            keys:      Float32 array of shape ``(n_heads, seq_len, head_dim)``.
            values:    Float32 array of shape ``(n_heads, seq_len, head_dim)``.
            keep_mask: Boolean array of shape ``(seq_len,)``.  ``True``
                       positions are retained; ``False`` positions are dropped.

        Returns:
            A three-tuple ``(compacted_keys, compacted_values, stats)``:

            * ``compacted_keys``   — float32, shape ``(n_heads, n_kept, head_dim)``.
            * ``compacted_values`` — float32, same shape as *compacted_keys*.
            * ``stats``            — :class:`CompactStats` summary.

        Raises:
            ValueError: If shape or dtype constraints are violated.
        """
        keys      = np.asarray(keys,      dtype=np.float32)
        values    = np.asarray(values,    dtype=np.float32)
        keep_mask = np.asarray(keep_mask, dtype=bool)

        if keys.ndim != 3:
            raise ValueError(
                f"keys must be 3-D (n_heads, seq_len, head_dim), got {keys.shape}"
            )
        if values.ndim != 3:
            raise ValueError(
                f"values must be 3-D (n_heads, seq_len, head_dim), got {values.shape}"
            )
        if keys.shape != values.shape:
            raise ValueError(
                f"keys and values must have identical shapes; "
                f"got {keys.shape} vs {values.shape}"
            )

        n_heads, seq_len, head_dim = keys.shape
        if n_heads != self._n_heads:
            raise ValueError(
                f"keys n_heads={n_heads} does not match "
                f"configured n_heads={self._n_heads}"
            )
        if head_dim != self._head_dim:
            raise ValueError(
                f"keys head_dim={head_dim} does not match "
                f"configured head_dim={self._head_dim}"
            )
        if keep_mask.ndim != 1 or keep_mask.shape[0] != seq_len:
            raise ValueError(
                f"keep_mask must have shape ({seq_len},), got {keep_mask.shape}"
            )

        keep_indices = np.where(keep_mask)[0]  # int64 sorted positions
        n_kept = keep_indices.shape[0]

        if n_kept > 0:
            # Advanced indexing creates a copy, so the gather is aliasing-safe
            # even when some kept positions fall within the destination slice.
            gathered_keys   = keys[:, keep_indices, :]
            gathered_values = values[:, keep_indices, :]
            keys[:, :n_kept, :]   = gathered_keys
            values[:, :n_kept, :] = gathered_values

        compacted_keys   = keys[:, :n_kept, :]
        compacted_values = values[:, :n_kept, :]

        element_bytes = keys.dtype.itemsize
        bytes_saved   = (
            (seq_len - n_kept) * n_heads * head_dim * element_bytes * 2
        )
        compaction_ratio = float(n_kept) / float(seq_len) if seq_len > 0 else 1.0

        stats = CompactStats(
            n_tokens_before=seq_len,
            n_tokens_after=n_kept,
            bytes_saved=bytes_saved,
            compaction_ratio=compaction_ratio,
        )
        return compacted_keys, compacted_values, stats

    def compact_indices(
        self,
        seq_len: int,
        keep_indices: np.ndarray,
    ) -> np.ndarray:
        """Build a mapping from old sequence positions to their new compacted
        positions.

        After compaction the *i*-th retained token occupies position *i* in
        the output sequence.  This method returns a lookup array ``mapping`` of
        shape ``(seq_len,)`` where ``mapping[old_pos] = new_pos`` for kept
        positions and ``mapping[old_pos] = -1`` for discarded positions.

        Args:
            seq_len:      Original sequence length before compaction.
            keep_indices: 1-D int64 array of sorted retained positions in
                          ``[0, seq_len)``.

        Returns:
            int64 array of shape ``(seq_len,)`` with new 0-based positions for
            retained tokens and ``-1`` for discarded tokens.

        Raises:
            ValueError: If *keep_indices* is not 1-D, if any index is out of
                        range, or if *seq_len* is not positive.
        """
        if seq_len < 0:
            raise ValueError(f"seq_len must be >= 0, got {seq_len}")

        keep_indices = np.asarray(keep_indices, dtype=np.int64)
        if keep_indices.ndim != 1:
            raise ValueError(
                f"keep_indices must be 1-D, got shape {keep_indices.shape}"
            )
        if keep_indices.size > 0:
            if int(keep_indices[0]) < 0 or int(keep_indices[-1]) >= seq_len:
                raise ValueError(
                    f"keep_indices values must be in [0, {seq_len}); "
                    f"got min={int(keep_indices[0])}, max={int(keep_indices[-1])}"
                )

        mapping = np.full(seq_len, -1, dtype=np.int64)
        mapping[keep_indices] = np.arange(keep_indices.shape[0], dtype=np.int64)
        return mapping
