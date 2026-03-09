"""
squish/seq_packing.py

Sequence Packing — Barrel Effect Elimination for Batched Inference.

Problem
-------
Continuous batching reduces *inter-request* idle time but does not eliminate
*intra-batch* padding waste.  When requests of different lengths are batched
together, shorter sequences are padded to match the longest — wasting GPU
compute on padding tokens.  This is the "barrel effect": the shortest stave
(sequence) determines the height the barrel can hold; every stave is cut to
the same height regardless.

Example: batching a 128-token commit message with a 2048-token DevOps plan
yields 1920 wasted padding operations on the 128-token sequence — 93.75% waste.

Solution — Sequence Packing
----------------------------
Instead of padding short sequences, **concatenate** multiple sequences into
one "super-sequence" and modify the attention mask to prevent cross-sequence
attention.  The attention mask for a packed sequence is block-diagonal:

    [seq1 mask   0        0      ]
    [0        seq2 mask   0      ]
    [0        0        seq3 mask ]

One GPU forward pass processes multiple logical requests as one long sequence.
Padding is only needed when the packed super-sequence doesn't fill exactly to
*max_packed_length* — much smaller waste.

Implementations in HuggingFace Transformers, TRL, and mlx-lm already support
sequence packing.  This module provides the packing logic and efficiency stats.

Conflict notes
--------------
- **No conflict** with any inference technique — packing changes input layout,
  not the model itself.
- **Synergy with ForeLen / TRAIL**: length prediction enables the packer to
  choose which sequences can be packed together without exceeding the budget.
- **Synergy with BucketServe**: BucketServe buckets by output length; packing
  works within each bucket to eliminate within-bucket padding.

Provides
--------
  PackingConfig     — max_packed_length, alignment padding.
  SequencePacker    — packs a list of token sequences into PackedBatches.
  PackedBatch       — container with token_ids, block-diagonal mask, offsets.
  PackingStats      — efficiency counters.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

__all__ = [
    "PackingConfig",
    "SequencePacker",
    "PackedBatch",
    "PackingStats",
]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class PackingConfig:
    """Configuration for sequence packing.

    Parameters
    ----------
    max_packed_length:
        Maximum total tokens in one packed super-sequence.  Should match
        the model's context window or the KV budget configured for serving.
    pad_to_multiple:
        Pad the final packed length to a multiple of this value (for memory
        alignment).  8 is efficient for most hardware.
    allow_partial:
        If True, sequences longer than *max_packed_length* are included
        individually (possibly truncated).  If False, they are rejected.
    """

    max_packed_length: int = 8192
    pad_to_multiple: int = 8
    allow_partial: bool = True

    def __post_init__(self) -> None:
        if self.max_packed_length < 1:
            raise ValueError("max_packed_length must be >= 1")
        if self.pad_to_multiple < 1:
            raise ValueError("pad_to_multiple must be >= 1")


# ---------------------------------------------------------------------------
# PackedBatch
# ---------------------------------------------------------------------------

@dataclass
class PackedBatch:
    """A packed super-sequence ready for a single model forward pass.

    Attributes
    ----------
    token_ids:
        1-D array of token IDs for the packed sequence, shape ``(total_len,)``.
    attention_mask:
        2-D boolean array of shape ``(total_len, total_len)``.  Entry [i, j]
        is True iff token i can attend to token j (same original sequence AND
        i >= j for causal attention).
    sequence_offsets:
        Start position of each packed sequence in ``token_ids``.
        Length = number of sequences packed.
    sequence_lengths:
        Length of each original sequence (before padding).
    pad_token_id:
        ID used for padding tokens.
    """

    token_ids: np.ndarray
    attention_mask: np.ndarray
    sequence_offsets: list[int]
    sequence_lengths: list[int]
    pad_token_id: int = 0

    @property
    def n_sequences(self) -> int:
        """Number of sequences packed."""
        return len(self.sequence_offsets)

    @property
    def total_length(self) -> int:
        """Total tokens in the packed super-sequence (including padding)."""
        return len(self.token_ids)

    @property
    def content_length(self) -> int:
        """Total non-padding tokens."""
        return sum(self.sequence_lengths)

    @property
    def padding_tokens(self) -> int:
        return self.total_length - self.content_length

    @property
    def padding_ratio(self) -> float:
        """Fraction of tokens that are padding.  0.0 = perfect packing."""
        return self.padding_tokens / self.total_length if self.total_length > 0 else 0.0

    def is_valid(self) -> bool:
        """Basic integrity check."""
        if len(self.token_ids) != len(self.attention_mask):
            return False
        if self.attention_mask.shape[0] != self.attention_mask.shape[1]:
            return False
        return True


# ---------------------------------------------------------------------------
# SequencePacker
# ---------------------------------------------------------------------------

class SequencePacker:
    """Packs multiple token sequences into ``PackedBatch`` super-sequences.

    Uses a greedy first-fit decreasing (FFD) bin-packing strategy:
    sequences are sorted longest-first and packed into the first available
    bin with sufficient remaining capacity.

    Parameters
    ----------
    config:
        ``PackingConfig``.
    pad_token_id:
        Token ID used for padding.
    """

    def __init__(
        self,
        config: PackingConfig | None = None,
        pad_token_id: int = 0,
    ) -> None:
        self._cfg = config or PackingConfig()
        self._pad = pad_token_id

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_packed_batch(self, sequences: list[list[int]]) -> PackedBatch:
        """Build one ``PackedBatch`` from a list of sequences.

        The sequences are assumed to fit within *max_packed_length* in total.
        """
        lengths = [len(s) for s in sequences]
        total_content = sum(lengths)

        # Align to pad_to_multiple
        m = self._cfg.pad_to_multiple
        padded_total = ((total_content + m - 1) // m) * m

        token_ids = np.full(padded_total, self._pad, dtype=np.int64)
        offsets: list[int] = []
        cursor = 0
        for seq in sequences:
            offsets.append(cursor)
            token_ids[cursor : cursor + len(seq)] = seq
            cursor += len(seq)

        # Build block-diagonal causal attention mask
        mask = np.zeros((padded_total, padded_total), dtype=bool)
        cursor = 0
        for seq_len in lengths:
            end = cursor + seq_len
            for i in range(cursor, end):
                for j in range(cursor, i + 1):
                    mask[i, j] = True
            cursor = end

        return PackedBatch(
            token_ids=token_ids,
            attention_mask=mask,
            sequence_offsets=offsets,
            sequence_lengths=lengths,
            pad_token_id=self._pad,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def pack(self, sequences: list[list[int]]) -> list[PackedBatch]:
        """Pack a list of sequences into as few ``PackedBatch`` objects as possible.

        Long sequences (> max_packed_length) are placed alone in their own batch.

        Parameters
        ----------
        sequences:
            List of token ID lists, variable length.

        Returns
        -------
        List[PackedBatch]
            One or more packed batches covering all input sequences.
        """
        limit = self._cfg.max_packed_length

        # Sort by length descending (greedy FFD)
        indexed = sorted(enumerate(sequences), key=lambda x: len(x[1]), reverse=True)

        bins: list[list[list[int]]] = []     # list of bins; each bin = list of seq lists
        bin_used: list[int] = []             # tokens used per bin

        for _orig_idx, seq in indexed:
            n = len(seq)
            if n > limit:
                if self._cfg.allow_partial:
                    bins.append([seq[:limit]])
                    bin_used.append(min(n, limit))
                continue  # skip overlong sequences if allow_partial=False

            placed = False
            for b_idx, used in enumerate(bin_used):
                if used + n <= limit:
                    bins[b_idx].append(seq)
                    bin_used[b_idx] += n
                    placed = True
                    break
            if not placed:
                bins.append([seq])
                bin_used.append(n)

        return [self._build_packed_batch(bin_seqs) for bin_seqs in bins if bin_seqs]


# ---------------------------------------------------------------------------
# PackingStats
# ---------------------------------------------------------------------------

@dataclass
class PackingStats:
    """Efficiency counters for sequence packing.

    Attributes
    ----------
    total_sequences:
        Total sequences submitted for packing.
    total_tokens:
        Total non-padding content tokens across all sequences.
    padded_tokens:
        Total padding tokens added to fill alignment gaps.
    total_batches:
        Number of ``PackedBatch`` objects produced.
    """

    total_sequences: int = 0
    total_tokens: int = 0
    padded_tokens: int = 0
    total_batches: int = 0

    def record_batches(self, batches: list[PackedBatch]) -> None:
        """Accumulate statistics from a list of produced batches."""
        for b in batches:
            self.total_sequences += b.n_sequences
            self.total_tokens += b.content_length
            self.padded_tokens += b.padding_tokens
        self.total_batches += len(batches)

    @property
    def packing_efficiency(self) -> float:
        """Non-padding fraction of all packed tokens (1.0 = perfect)."""
        total = self.total_tokens + self.padded_tokens
        return self.total_tokens / total if total > 0 else 0.0

    @property
    def mean_sequences_per_batch(self) -> float:
        return self.total_sequences / self.total_batches if self.total_batches > 0 else 0.0

    @property
    def padding_ratio(self) -> float:
        total = self.total_tokens + self.padded_tokens
        return self.padded_tokens / total if total > 0 else 0.0
