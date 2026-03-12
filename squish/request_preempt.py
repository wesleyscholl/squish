"""squish/request_preempt.py

PreemptScheduler — Preemptive SRPT (Shortest Remaining Processing Time)
scheduler with KV cache save/restore for LLM inference.

Production LLM servers must support request preemption to maintain latency
SLOs under mixed-priority workloads.  When a higher-priority or shorter
request arrives, the scheduler can pause a running generation, optionally
saving its KV state, and later resume it.

Two preemption strategies are supported:

``swap``
    The KV cache for the preempted sequence is copied to host memory.  On
    resumption the KV is restored and decoding continues from the exact saved
    state.  Fastest resume, but requires additional memory proportional to
    the KV cache size.

``recompute``
    The KV cache is discarded.  On resumption the prefill phase must be re-run
    to reconstruct the KV, which means any tokens generated before preemption
    are lost and must be regenerated.  Zero memory overhead at the cost of
    extra compute.

:class:`PreemptStats` tracks how many preemptions and resumes have occurred
and averages the token loss (recompute-strategy only).

Reference:
    Influenced by the vLLM preemption design described in:
    Kwon et al., "Efficient Memory Management for Large Language Model Serving
    with PagedAttention", SOSP 2023.
    https://arxiv.org/abs/2309.06180

Example usage::

    import numpy as np
    from squish.request_preempt import PreemptScheduler, PreemptState

    scheduler = PreemptScheduler(n_heads=8, head_dim=64, n_layers=4)

    kv = np.random.randn(4, 2, 8, 32, 64).astype(np.float32)

    state = scheduler.preempt("req-1", current_kv=kv, strategy="swap")
    assert scheduler.can_resume("req-1")

    restored = scheduler.resume("req-1")
    assert restored is not None
    assert restored.saved_kv is not None

    print(scheduler.stats)
"""

from __future__ import annotations

__all__ = ["PreemptState", "PreemptStats", "PreemptScheduler"]

import dataclasses
from typing import Optional

import numpy as np


# Valid strategy identifiers.
_STRATEGY_SWAP:      str = "swap"
_STRATEGY_RECOMPUTE: str = "recompute"
_VALID_STRATEGIES: tuple[str, str] = (_STRATEGY_SWAP, _STRATEGY_RECOMPUTE)

# Index of the sequence dimension inside a KV tensor of shape
# (n_layers, 2, n_heads, seq, head_dim).
_SEQ_AXIS: int = 3


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class PreemptState:
    """Saved state for a preempted sequence.

    Attributes:
        seq_id:             Identifier of the preempted sequence.
        n_tokens_generated: Number of tokens in the KV cache at preemption
                            time (i.e. the sequence dimension of the saved KV,
                            or 0 when the strategy is ``"recompute"``).
        saved_kv:           KV tensor of shape
                            ``(n_layers, 2, n_heads, seq, head_dim)`` when the
                            strategy is ``"swap"``; ``None`` when the strategy
                            is ``"recompute"``.
        saved_at:           Monotonically increasing scheduler tick recorded
                            at the moment of preemption.
    """

    seq_id:             str
    n_tokens_generated: int
    saved_kv:           Optional[np.ndarray]
    saved_at:           int


@dataclasses.dataclass
class PreemptStats:
    """Aggregate preemption statistics.

    Attributes:
        n_preemptions:   Total number of :meth:`PreemptScheduler.preempt`
                         calls.
        n_resumes:       Total number of :meth:`PreemptScheduler.resume`
                         calls that returned a non-``None`` state.
        avg_tokens_lost: Mean number of tokens that had to be recomputed
                         across all ``"recompute"``-strategy preemptions.
                         0.0 if no recompute preemptions have occurred.
    """

    n_preemptions:   int
    n_resumes:       int
    avg_tokens_lost: float


# ---------------------------------------------------------------------------
# PreemptScheduler
# ---------------------------------------------------------------------------


class PreemptScheduler:
    """Preemptive SRPT scheduler with KV save/restore support.

    Maintains an internal monotonic tick that is incremented on every
    :meth:`preempt` call, giving each :class:`PreemptState` a unique
    ``saved_at`` timestamp.

    Args:
        n_heads:  Number of attention heads.  Must be >= 1.
        head_dim: Dimension of each attention head.  Must be >= 1.
        n_layers: Number of transformer layers.  Defaults to 4.
    """

    def __init__(
        self,
        n_heads:  int,
        head_dim: int,
        n_layers: int = 4,
    ) -> None:
        if n_heads < 1:
            raise ValueError(f"n_heads must be >= 1, got {n_heads}")
        if head_dim < 1:
            raise ValueError(f"head_dim must be >= 1, got {head_dim}")
        if n_layers < 1:
            raise ValueError(f"n_layers must be >= 1, got {n_layers}")

        self._n_heads:  int = n_heads
        self._head_dim: int = head_dim
        self._n_layers: int = n_layers

        # Active preemption states keyed by seq_id.
        self._saved: dict[str, PreemptState] = {}

        # Counters.
        self._tick:                  int = 0
        self._n_preemptions:         int = 0
        self._n_resumes:             int = 0
        self._total_tokens_lost:     int = 0
        self._recompute_preemptions: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def preempt(
        self,
        seq_id:     str,
        current_kv: Optional[np.ndarray] = None,
        strategy:   str = "swap",
    ) -> PreemptState:
        """Preempt a running sequence, optionally saving its KV state.

        Args:
            seq_id:     Unique identifier of the sequence to preempt.
            current_kv: KV tensor of shape
                        ``(n_layers, 2, n_heads, seq, head_dim)``, float32.
                        Must be provided when ``strategy="swap"``; may be
                        ``None`` when ``strategy="recompute"``.
            strategy:   ``"swap"`` — copy KV to host memory for fast resume.
                        ``"recompute"`` — discard KV; tokens must be
                        recomputed on resume.

        Returns:
            A :class:`PreemptState` recording the preemption.

        Raises:
            ValueError: If ``strategy`` is not a recognised value, or if
                        ``strategy="swap"`` is requested without ``current_kv``.
        """
        if strategy not in _VALID_STRATEGIES:
            raise ValueError(
                f"strategy must be one of {_VALID_STRATEGIES}, got '{strategy}'"
            )
        if strategy == _STRATEGY_SWAP and current_kv is None:
            raise ValueError(
                "current_kv must be provided when strategy='swap'"
            )

        self._tick += 1

        if strategy == _STRATEGY_SWAP:
            kv_copy = np.asarray(current_kv, dtype=np.float32).copy()
            # Extract the sequence-length dimension (axis 3 of 5-D tensor).
            n_tokens = int(kv_copy.shape[_SEQ_AXIS]) if kv_copy.ndim == 5 else 0
        else:
            kv_copy  = None
            # Count tokens that will be lost (must be recomputed on resume).
            if current_kv is not None:
                arr      = np.asarray(current_kv)
                n_tokens = int(arr.shape[_SEQ_AXIS]) if arr.ndim == 5 else 0
            else:
                n_tokens = 0
            self._total_tokens_lost     += n_tokens
            self._recompute_preemptions += 1

        state = PreemptState(
            seq_id=seq_id,
            n_tokens_generated=n_tokens,
            saved_kv=kv_copy,
            saved_at=self._tick,
        )

        self._saved[seq_id]   = state
        self._n_preemptions  += 1
        return state

    def resume(self, seq_id: str) -> Optional[PreemptState]:
        """Retrieve and remove the saved preempt state for ``seq_id``.

        Removes the state from the internal store so a subsequent call with
        the same ``seq_id`` will return ``None`` (unless it is preempted
        again).

        Args:
            seq_id: Sequence identifier to resume.

        Returns:
            The saved :class:`PreemptState`, or ``None`` if ``seq_id`` was
            not in the preempted set.
        """
        state = self._saved.pop(seq_id, None)
        if state is not None:
            self._n_resumes += 1
        return state

    def can_resume(self, seq_id: str) -> bool:
        """Return ``True`` if ``seq_id`` has a saved preemption state.

        Args:
            seq_id: Sequence identifier to query.
        """
        return seq_id in self._saved

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def stats(self) -> PreemptStats:
        """Return a snapshot of aggregate preemption statistics."""
        avg_lost = (
            self._total_tokens_lost / self._recompute_preemptions
            if self._recompute_preemptions > 0
            else 0.0
        )
        return PreemptStats(
            n_preemptions=self._n_preemptions,
            n_resumes=self._n_resumes,
            avg_tokens_lost=avg_lost,
        )
