# [Experimental] This module is part of Squish v6+ (Waves 19–26).
# Proof-of-concept quality: API and behaviour may change without notice.
# For stable core functionality see Waves 1–12 in MODULES.md.
#!/usr/bin/env python3
"""
squish/fault_tolerance.py

FaultTolerance — Graceful OOM degradation for inference servers.

Under memory pressure, the server should automatically execute an ordered
sequence of degradation actions to reclaim capacity and protect service
availability:

  1. **Evict KV** — evict expired or least-recently-used KV cache entries
     (cheapest; transparent to clients that have not yet decoded those tokens).
  2. **Disable draft** — turn off the speculative-decode draft model (eliminates
     its memory footprint and CPU/GPU overhead immediately).
  3. **Reduce batch** — shrink the active batch size toward ``min_batch_size``
     (trades throughput for lower peak working set).
  4. **Renegotiate SLO** — signal upstream that latency guarantees must be
     relaxed (last resort; visible to clients).

This module provides a pure policy engine.  It does not hold references to
model weights or KV caches; callers apply the returned actions to their own
subsystems.

Example usage::

    from squish.fault_tolerance import FaultHandler, FaultPolicy

    policy = FaultPolicy(
        evict_kv_at=0.85,
        disable_draft_at=0.90,
        reduce_batch_at=0.95,
        min_batch_size=1,
    )
    handler = FaultHandler(policy)

    actions = handler.evaluate(pressure=0.92, current_batch_size=8)
    print(actions)  # ["evict_kv", "disable_draft"]

    evicted = handler.apply_evict_kv(n_to_evict=32)
    print(f"Evicted {evicted} KV entries, stats={handler.stats}")
"""

from __future__ import annotations

__all__ = [
    "FaultPolicy",
    "FaultAction",
    "FaultHandler",
    "FaultStats",
    "mem_pressure_fraction",
]

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from squish.memory_governor import MemoryGovernor


# ---------------------------------------------------------------------------
# MemoryGovernor pressure → [0, 1] fraction
# ---------------------------------------------------------------------------

# Map MemoryGovernor integer pressure levels to normalised fractions.
# LEVEL_NORMAL=0, LEVEL_WARNING=1, LEVEL_URGENT=2, LEVEL_CRITICAL=4.
_PRESSURE_LEVEL_MAP: dict[int, float] = {
    0: 0.00,   # NORMAL   — no action needed
    1: 0.75,   # WARNING  — below default evict_kv_at=0.85; head-start signal
    2: 0.92,   # URGENT   — triggers evict_kv + disable_draft at defaults
    4: 1.00,   # CRITICAL — full degradation cascade
}


def mem_pressure_fraction(pressure_level: int) -> float:
    """Convert a MemoryGovernor integer pressure level to a [0, 1] fraction.

    Unknown levels are clamped to 1.0 so they always trigger full degradation.

    Args:
        pressure_level: Integer level (0, 1, 2, or 4) from
                        ``MemoryGovernor.pressure_level``.

    Returns:
        A float in [0, 1] suitable for passing to
        :meth:`FaultHandler.evaluate`.
    """
    return _PRESSURE_LEVEL_MAP.get(pressure_level, 1.0)


# ---------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------

@dataclass
class FaultPolicy:
    """Thresholds that govern when each degradation action is triggered.

    All pressure fractions are in the range [0.0, 1.0] where 1.0 represents
    fully-utilised memory.

    Attributes:
        evict_kv_at:      Memory-pressure fraction at which KV eviction begins.
        disable_draft_at: Pressure fraction at which the draft model is disabled.
        reduce_batch_at:  Pressure fraction at which batch size is reduced.
        min_batch_size:   Floor for batch-size reduction (>= 1).
    """

    evict_kv_at: float = 0.85
    disable_draft_at: float = 0.90
    reduce_batch_at: float = 0.95
    min_batch_size: int = 1

    def __post_init__(self) -> None:
        for name, value in [
            ("evict_kv_at", self.evict_kv_at),
            ("disable_draft_at", self.disable_draft_at),
            ("reduce_batch_at", self.reduce_batch_at),
        ]:
            if not (0.0 < value <= 1.0):
                raise ValueError(
                    f"{name} must be in (0, 1], got {value}"
                )
        if not (self.evict_kv_at <= self.disable_draft_at <= self.reduce_batch_at):
            raise ValueError(
                "Thresholds must satisfy "
                "evict_kv_at <= disable_draft_at <= reduce_batch_at, "
                f"got {self.evict_kv_at} / {self.disable_draft_at} / "
                f"{self.reduce_batch_at}"
            )
        if self.min_batch_size < 1:
            raise ValueError(
                f"min_batch_size must be >= 1, got {self.min_batch_size}"
            )


# ---------------------------------------------------------------------------
# Action identifiers
# ---------------------------------------------------------------------------

class FaultAction:
    """Enumeration of degradation actions returned by :class:`FaultHandler`.

    Attributes:
        EVICT_KV:        Evict least-recently-used / expired KV cache entries.
        DISABLE_DRAFT:   Disable the speculative-decode draft model.
        REDUCE_BATCH:    Reduce the active inference batch size.
        RENEGOTIATE_SLO: Relax client-facing latency SLOs.
    """

    EVICT_KV: str = "evict_kv"
    DISABLE_DRAFT: str = "disable_draft"
    REDUCE_BATCH: str = "reduce_batch"
    RENEGOTIATE_SLO: str = "renegotiate_slo"


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

@dataclass
class FaultStats:
    """Cumulative counters maintained by :class:`FaultHandler`.

    Attributes:
        total_evaluations:  Number of times :meth:`FaultHandler.evaluate`
                            has been called.
        kv_evictions:       Cumulative KV entries evicted via
                            :meth:`FaultHandler.apply_evict_kv`.
        draft_disables:     Number of evaluations that triggered
                            :attr:`FaultAction.DISABLE_DRAFT`.
        batch_reductions:   Number of evaluations that triggered
                            :attr:`FaultAction.REDUCE_BATCH`.
        last_governor_level: Most recent MemoryGovernor pressure level (int)
                            that was passed to
                            :meth:`FaultHandler.evaluate_from_governor`,
                            or ``None`` if that method has never been called.
    """

    total_evaluations: int = 0
    kv_evictions: int = 0
    draft_disables: int = 0
    batch_reductions: int = 0
    last_governor_level: "int | None" = None


# ---------------------------------------------------------------------------
# Handler
# ---------------------------------------------------------------------------

class FaultHandler:
    """Policy engine for graceful OOM degradation.

    Evaluates the current memory-pressure fraction against the configured
    :class:`FaultPolicy` thresholds and returns an ordered list of
    degradation actions the caller must apply to their subsystems.

    Actions are cumulative: if ``pressure >= reduce_batch_at`` then the
    returned list contains *all three* lower-severity actions as well.

    Args:
        policy: A :class:`FaultPolicy` instance describing action thresholds.
    """

    def __init__(self, policy: FaultPolicy) -> None:
        self._policy = policy
        self._stats = FaultStats()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(
        self,
        pressure: float,
        current_batch_size: int,
    ) -> list[str]:
        """Evaluate current memory pressure and return required actions.

        Actions are returned in severity order (cheapest first).  The
        caller is responsible for executing them against their subsystems.

        Args:
            pressure:           Current memory-pressure fraction in [0, 1].
            current_batch_size: Active batch size before any reduction.

        Returns:
            Ordered list of :class:`FaultAction` strings.  An empty list
            means no action is required.

        Raises:
            ValueError: if ``pressure`` is outside [0, 1] or
                        ``current_batch_size`` is < 1.
        """
        if not (0.0 <= pressure <= 1.0):
            raise ValueError(
                f"pressure must be in [0, 1], got {pressure}"
            )
        if current_batch_size < 1:
            raise ValueError(
                f"current_batch_size must be >= 1, got {current_batch_size}"
            )

        self._stats.total_evaluations += 1
        actions: list[str] = []
        policy = self._policy

        if pressure >= policy.evict_kv_at:
            actions.append(FaultAction.EVICT_KV)

        if pressure >= policy.disable_draft_at:
            actions.append(FaultAction.DISABLE_DRAFT)
            self._stats.draft_disables += 1

        if pressure >= policy.reduce_batch_at:
            actions.append(FaultAction.REDUCE_BATCH)
            if current_batch_size > policy.min_batch_size:
                self._stats.batch_reductions += 1

        # Renegotiate SLO when all lower-severity actions are already active
        # and the server is still at or above the highest threshold.
        if pressure >= 1.0:
            actions.append(FaultAction.RENEGOTIATE_SLO)

        return actions

    def apply_evict_kv(self, n_to_evict: int) -> int:
        """Simulate a KV-cache eviction and record the count in stats.

        This method does not hold a reference to any actual KV cache.
        Callers invoke this after performing the eviction in their own
        subsystem so that the fault handler can track cumulative evictions.

        Args:
            n_to_evict: Number of KV entries to evict (>= 0).

        Returns:
            The number of entries evicted (equal to ``n_to_evict``).

        Raises:
            ValueError: if ``n_to_evict`` is negative.
        """
        if n_to_evict < 0:
            raise ValueError(
                f"n_to_evict must be >= 0, got {n_to_evict}"
            )
        self._stats.kv_evictions += n_to_evict
        return n_to_evict

    def evaluate_from_governor(
        self,
        governor: "MemoryGovernor",
        current_batch_size: int,
    ) -> list[str]:
        """Evaluate using a live :class:`~squish.memory_governor.MemoryGovernor`.

        Reads ``governor.pressure_level``, converts it to a [0, 1] fraction
        via :func:`mem_pressure_fraction`, records the raw level in
        :attr:`FaultStats.last_governor_level`, then delegates to
        :meth:`evaluate`.

        Args:
            governor:           A started ``MemoryGovernor`` instance.
            current_batch_size: Active batch size before any reduction.

        Returns:
            Ordered list of :class:`FaultAction` strings.
        """
        level = governor.pressure_level
        self._stats.last_governor_level = level
        return self.evaluate(
            pressure=mem_pressure_fraction(level),
            current_batch_size=current_batch_size,
        )

    @property
    def stats(self) -> FaultStats:
        """Cumulative degradation statistics (updated in place)."""
        return self._stats
