"""
squish/streaming_sink.py

StreamingLLM Attention Sinks — Infinite Context Without Retraining.

Based on:
  "Efficient Streaming Language Models with Attention Sinks"
  Xiao et al., ICLR 2024 — MIT-HAN Lab
  github.com/mit-han-lab/streaming-llm

Problem
-------
LLMs cannot natively process sequences longer than their training context
window.  Naively extending by keeping a full rolling KV cache causes memory
to grow linearly with sequence length — eventually exhausting GPU VRAM.

Existing sliding-window approaches remove old KV entries to bound memory.
But removing the *very first* tokens causes catastrophic quality degradation,
even though those tokens may have low semantic importance.

The reason: LLMs develop "**attention sinks**" at initial tokens.  Because
the attention distribution must sum to 1.0, the model learns to route excess
probability mass to the very first tokens — regardless of their meaning.
These positions act as "dump sites" for attention probability.  When they are
evicted, the model is forced to route probability mass to semantically
irrelevant positions in the sliding window, producing incoherent output.

StreamingLLM's solution
-----------------------
Always keep the first ``num_sinks`` tokens' KV entries in cache regardless
of their recency.  Combine with a sliding window of recent tokens.  Total
cache size stays fixed at ``num_sinks + window_size``.

This enables truly infinite-length processing:
- Memory is bounded (no growth with sequence length)
- Quality is maintained (sinks absorb excess attention mass as intended)
- No retraining required

Squish integration
------------------
The ``SinkKVCache`` wraps a layer's KV storage.  The first
``num_sink_tokens=4`` entries are "pinned" — never evicted regardless of
cache pressure.  When the cache fills, the oldest *non-sink* token is evicted.

This directly prevents the pathological case where aggressive KV eviction
(from CommVQ hot-tier eviction or ShadowKV pressure) accidentally removes
system-prompt sink tokens, causing quality collapse on long sessions.

Conflict notes
--------------
- **Synergy with KV tier management** (CommVQ, ShadowKV, SpeContext): add
  ``SinkKVCache`` as the first-tier gate that prevents sink tokens from ever
  being nominated for eviction or offload.
- **No conflict**: StreamingLLM is a cache eviction policy; it does not touch
  quantization, speculation, or scheduling.
- **Attention mask**: the sink tokens + recent window must appear in the
  attention key matrix — their positions are non-contiguous relative to the
  current decode position, so the attention mask must correctly map them.

Provides
--------
  SinkConfig        — num_sinks and window_size.
  SinkKVCache       — KV cache with pinned sinks + sliding window.
  SinkStats         — eviction and preservation counters.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

__all__ = [
    "SinkConfig",
    "SinkKVCache",
    "SinkStats",
]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class SinkConfig:
    """Configuration for StreamingLLM attention sinks.

    Parameters
    ----------
    num_sinks:
        Number of initial tokens to pin permanently in the cache.
        The StreamingLLM paper finds 4 sinks sufficient for all models tested.
    window_size:
        Number of *recent* tokens to keep in the sliding window.
        Total cache capacity = num_sinks + window_size.
    head_dim:
        Dimension of each head's key/value vector.
    """

    num_sinks: int = 4
    window_size: int = 1024
    head_dim: int = 128

    def __post_init__(self) -> None:
        if self.num_sinks < 0:
            raise ValueError("num_sinks must be >= 0")
        if self.window_size < 1:
            raise ValueError("window_size must be >= 1")
        if self.head_dim < 1:
            raise ValueError("head_dim must be >= 1")

    @property
    def capacity(self) -> int:
        return self.num_sinks + self.window_size


# ---------------------------------------------------------------------------
# SinkKVCache
# ---------------------------------------------------------------------------

class SinkKVCache:
    """KV cache with pinned attention sinks and a sliding window.

    Internally maintains two separate stores:
    - **Sink store**: fixed-size buffer of the first *num_sinks* tokens.
      Once filled, it never changes.
    - **Window store**: circular buffer of the most recent *window_size*
      tokens.  Old entries are evicted as new tokens arrive.

    Parameters
    ----------
    config:
        ``SinkConfig``.
    """

    def __init__(self, config: SinkConfig | None = None) -> None:
        self._cfg = config or SinkConfig()
        # Sink store: fixed capacity, filled once
        self._sink_keys: list[np.ndarray] = []    # up to num_sinks
        self._sink_values: list[np.ndarray] = []
        self._sink_positions: list[int] = []       # original sequence position
        # Window store: circular buffer
        self._win_keys: list[np.ndarray] = []
        self._win_values: list[np.ndarray] = []
        self._win_positions: list[int] = []
        # Global sequence counter
        self._seq_pos: int = 0
        # Stats
        self._stats: SinkStats = SinkStats()

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def append(self, key: np.ndarray, value: np.ndarray) -> None:
        """Append one KV pair at the current sequence position.

        If we haven't filled the sink store yet, the token is pinned as a
        sink.  Otherwise it enters the sliding window, potentially evicting
        the oldest window token.

        Parameters
        ----------
        key:
            Shape ``(head_dim,)`` key vector.
        value:
            Shape ``(head_dim,)`` value vector.
        """
        k = np.asarray(key, dtype=np.float32).ravel()
        v = np.asarray(value, dtype=np.float32).ravel()
        pos = self._seq_pos
        self._seq_pos += 1
        self._stats.total_appended += 1

        if len(self._sink_keys) < self._cfg.num_sinks:
            # Pin as sink
            self._sink_keys.append(k)
            self._sink_values.append(v)
            self._sink_positions.append(pos)
            self._stats.sink_preserved += 1
        else:
            # Add to sliding window
            if len(self._win_keys) >= self._cfg.window_size:
                # Evict oldest window entry
                self._win_keys.pop(0)
                self._win_values.pop(0)
                self._win_positions.pop(0)
                self._stats.evictions += 1
            self._win_keys.append(k)
            self._win_values.append(v)
            self._win_positions.append(pos)

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def get_kv(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return all KV entries available: sinks + window.

        Returns
        -------
        keys:
            Shape ``(n_cached, head_dim)`` — sinks first, then window.
        values:
            Shape ``(n_cached, head_dim)``.
        positions:
            Shape ``(n_cached,)`` — original sequence positions (for RoPE
            or positional encoding re-injection).
        """
        all_keys = self._sink_keys + self._win_keys
        all_vals = self._sink_values + self._win_values
        all_pos = self._sink_positions + self._win_positions

        if not all_keys:
            empty = np.zeros((0, self._cfg.head_dim), dtype=np.float32)
            return empty, empty, np.array([], dtype=np.int64)

        keys = np.stack(all_keys)
        values = np.stack(all_vals)
        positions = np.array(all_pos, dtype=np.int64)
        return keys, values, positions

    def reset(self) -> None:
        """Clear the cache for a new request."""
        self._sink_keys.clear()
        self._sink_values.clear()
        self._sink_positions.clear()
        self._win_keys.clear()
        self._win_values.clear()
        self._win_positions.clear()
        self._seq_pos = 0
        self._stats = SinkStats()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def size(self) -> int:
        """Number of KV entries currently cached (sinks + window)."""
        return len(self._sink_keys) + len(self._win_keys)

    @property
    def n_sinks(self) -> int:
        """Number of tokens currently pinned as attention sinks."""
        return len(self._sink_keys)

    @property
    def n_window(self) -> int:
        """Number of tokens currently in the sliding window."""
        return len(self._win_keys)

    @property
    def is_full(self) -> bool:
        """True when both the sink store and window are at capacity."""
        return (
            len(self._sink_keys) >= self._cfg.num_sinks
            and len(self._win_keys) >= self._cfg.window_size
        )

    @property
    def stats(self) -> SinkStats:
        return self._stats


# ---------------------------------------------------------------------------
# SinkStats
# ---------------------------------------------------------------------------

@dataclass
class SinkStats:
    """Eviction and preservation counters for ``SinkKVCache``.

    Attributes
    ----------
    total_appended:
        Total KV pairs appended since last reset.
    evictions:
        Window entries evicted to make room for newer tokens.
    sink_preserved:
        Sink entries permanently retained (never evicted).
    """

    total_appended: int = 0
    evictions: int = 0
    sink_preserved: int = 0

    @property
    def window_tokens_total(self) -> int:
        """Total tokens that have passed through the window."""
        return max(0, self.total_appended - self.sink_preserved)

    @property
    def eviction_rate(self) -> float:
        """Fraction of window tokens that were evicted."""
        wt = self.window_tokens_total
        return self.evictions / wt if wt > 0 else 0.0

    @property
    def sink_preservation_rate(self) -> float:
        """Fraction of appended tokens that are permanently pinned as sinks."""
        return self.sink_preserved / self.total_appended if self.total_appended > 0 else 0.0
