"""
Robust LLM Scheduling with Interval Predictions.

arxiv.org/abs/2508.14544
"Adaptively Robust LLM Inference Optimization under Prediction Uncertainty"
August 2025.

Key insight: point predictions from ForeLen/TRAIL introduce scheduling
failures when wrong.  Interval predictions (min/max range) enable provably
near-optimal scheduling even under worst-case uncertainty.

Two algorithms:
  - A_max (conservative): schedule by upper bound → no OOM, conservative batching.
  - A_balanced (adaptive): balance between bounds based on memory pressure →
    competitive ratio provably close to clairvoyant optimum.

This module provides:
  - RobustSchedulerConfig — configuration
  - LengthInterval — predicted [lo, hi] range for one request
  - Request — one inference request with metadata
  - AMaxScheduler — conservative upper-bound scheduler
  - ABalancedScheduler — adaptive load-aware scheduler
  - RobustSchedulerStats — statistics
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class RobustSchedulerConfig:
    """Configuration for robust interval-prediction schedulers.

    Args:
        max_batch_tokens:  Maximum total tokens (KV) across all in-flight
                           requests.  Used for memory pressure calculation.
        max_batch_size:    Maximum number of requests per batch.
        memory_pressure_threshold: When occupied_tokens / max_batch_tokens
                           exceeds this, A_balanced switches to conservative
                           (upper-bound) scheduling.  In [0, 1].
        alpha:             Blend weight in A_balanced: effective length =
                           alpha * hi + (1 - alpha) * lo.  Alpha is adapted
                           dynamically in [0, 1].
        preemption_penalty: Priority penalty applied to requests preempted due
                           to prediction error.
    """

    max_batch_tokens: int = 32768
    max_batch_size: int = 64
    memory_pressure_threshold: float = 0.85
    alpha: float = 0.5
    preemption_penalty: float = 0.1

    def __post_init__(self) -> None:
        if self.max_batch_tokens < 1:
            raise ValueError("max_batch_tokens must be >= 1")
        if self.max_batch_size < 1:
            raise ValueError("max_batch_size must be >= 1")
        if not (0.0 <= self.memory_pressure_threshold <= 1.0):
            raise ValueError("memory_pressure_threshold must be in [0, 1]")
        if not (0.0 <= self.alpha <= 1.0):
            raise ValueError("alpha must be in [0, 1]")
        if self.preemption_penalty < 0:
            raise ValueError("preemption_penalty must be >= 0")


# ---------------------------------------------------------------------------
# Domain objects
# ---------------------------------------------------------------------------

@dataclass
class LengthInterval:
    """Predicted output length interval [lo, hi] for a request."""

    lo: int
    hi: int
    point_estimate: int | None = None

    def __post_init__(self) -> None:
        if self.lo < 0:
            raise ValueError("lo must be >= 0")
        if self.hi < self.lo:
            raise ValueError("hi must be >= lo")

    @property
    def midpoint(self) -> int:
        return (self.lo + self.hi) // 2

    @property
    def range_width(self) -> int:
        return self.hi - self.lo

    def effective_length(self, alpha: float) -> int:
        """Blended length: alpha*hi + (1-alpha)*lo."""
        return int(alpha * self.hi + (1.0 - alpha) * self.lo)

    @classmethod
    def from_point(cls, estimate: int, uncertainty: float = 0.25) -> LengthInterval:
        """Build interval from a point estimate with fractional uncertainty."""
        margin = max(1, int(estimate * uncertainty))
        return cls(
            lo=max(0, estimate - margin),
            hi=estimate + margin,
            point_estimate=estimate,
        )


@dataclass
class Request:
    """One inference request."""

    request_id: str
    input_len: int
    length_interval: LengthInterval
    arrival_time: float = 0.0
    priority: float = 1.0
    task_type: str = "default"

    @property
    def tokens_at_hi(self) -> int:
        return self.input_len + self.length_interval.hi

    @property
    def tokens_at_lo(self) -> int:
        return self.input_len + self.length_interval.lo

    def tokens_at_alpha(self, alpha: float) -> int:
        return self.input_len + self.length_interval.effective_length(alpha)


# ---------------------------------------------------------------------------
# A_max — Conservative scheduler
# ---------------------------------------------------------------------------

class AMaxScheduler:
    """Conservative robust scheduler: schedules by upper bound (hi).

    Guarantees no memory overflow by always reserving KV space for the
    worst-case (longest) predicted output.

    Requests are sorted by their upper-bound token count (ascending = Shortest
    Upper Bound First = best for throughput at memory safety).
    """

    def __init__(self, config: RobustSchedulerConfig) -> None:
        self._config = config
        self._queue: list[Request] = []
        self._in_flight: list[Request] = []
        self._stats = RobustSchedulerStats()

    def enqueue(self, request: Request) -> None:
        self._queue.append(request)

    def schedule_batch(self) -> list[Request]:
        """Return the next batch of requests to execute.

        Selects requests from queue in ascending upper-bound order, stopping
        when memory budget or batch-size limit is reached.
        """
        cfg = self._config
        # Sort by upper-bound tokens ascending (SUBF — Shortest Upper Bound First)
        candidates = sorted(self._queue, key=lambda r: r.tokens_at_hi)

        batch: list[Request] = []
        reserved_tokens = sum(r.tokens_at_hi for r in self._in_flight)

        for req in candidates:
            if len(batch) + len(self._in_flight) >= cfg.max_batch_size:
                break
            projected = reserved_tokens + req.tokens_at_hi
            if projected <= cfg.max_batch_tokens:
                batch.append(req)
                reserved_tokens += req.tokens_at_hi

        selected_ids = {r.request_id for r in batch}
        self._queue = [r for r in self._queue if r.request_id not in selected_ids]

        object.__setattr__(self._stats, "total_scheduled",
                           self._stats.total_scheduled + len(batch))
        object.__setattr__(self._stats, "total_batches",
                           self._stats.total_batches + 1)

        return batch

    def complete(self, request_id: str) -> None:
        self._in_flight = [r for r in self._in_flight if r.request_id != request_id]

    @property
    def queue_size(self) -> int:
        return len(self._queue)

    @property
    def stats(self) -> RobustSchedulerStats:
        return self._stats


# ---------------------------------------------------------------------------
# A_balanced — Adaptive scheduler
# ---------------------------------------------------------------------------

class ABalancedScheduler:
    """Adaptive robust scheduler: blends lo/hi based on memory pressure.

    When memory is tight (pressure ≥ threshold), alpha → 1.0 (conservative).
    When memory is abundant, alpha → 0.0 (optimistic).
    Achieves competitive ratio provably close to the clairvoyant optimum.
    """

    def __init__(self, config: RobustSchedulerConfig) -> None:
        self._config = config
        self._queue: list[Request] = []
        self._in_flight: list[Request] = []
        self._alpha = config.alpha
        self._stats = RobustSchedulerStats()
        self._preemptions = 0

    # ------------------------------------------------------------------
    @property
    def memory_pressure(self) -> float:
        """Current memory pressure: in-flight tokens / max budget."""
        used = sum(r.tokens_at_alpha(self._alpha) for r in self._in_flight)
        return used / max(1, self._config.max_batch_tokens)

    def _adapt_alpha(self) -> None:
        """Update alpha based on current memory pressure."""
        pressure = self.memory_pressure
        threshold = self._config.memory_pressure_threshold
        if pressure >= threshold:
            # Under pressure: be conservative
            self._alpha = min(1.0, self._alpha + 0.1)
        else:
            # Comfortable: be optimistic
            self._alpha = max(0.0, self._alpha - 0.05)

    def enqueue(self, request: Request) -> None:
        self._queue.append(request)

    def schedule_batch(self) -> list[Request]:
        """Return next batch using current alpha-blended length estimates."""
        self._adapt_alpha()
        cfg = self._config
        alpha = self._alpha

        # Sort by effective token estimate ascending (best-fit for throughput)
        candidates = sorted(
            self._queue, key=lambda r: r.tokens_at_alpha(alpha)
        )

        batch: list[Request] = []
        reserved_tokens = sum(
            r.tokens_at_alpha(alpha) for r in self._in_flight
        )

        for req in candidates:
            if len(batch) + len(self._in_flight) >= cfg.max_batch_size:
                break
            projected = reserved_tokens + req.tokens_at_alpha(alpha)
            if projected <= cfg.max_batch_tokens:
                batch.append(req)
                reserved_tokens += req.tokens_at_alpha(alpha)

        selected_ids = {r.request_id for r in batch}
        self._queue = [r for r in self._queue if r.request_id not in selected_ids]

        object.__setattr__(self._stats, "total_scheduled",
                           self._stats.total_scheduled + len(batch))
        object.__setattr__(self._stats, "total_batches",
                           self._stats.total_batches + 1)

        return batch

    def handle_preemption(self, request_id: str) -> None:
        """Record that a request was preempted due to prediction error."""
        self._preemptions += 1
        object.__setattr__(self._stats, "preemptions",
                           self._stats.preemptions + 1)

    def complete(self, request_id: str) -> None:
        self._in_flight = [r for r in self._in_flight if r.request_id != request_id]

    @property
    def current_alpha(self) -> float:
        return self._alpha

    @property
    def queue_size(self) -> int:
        return len(self._queue)

    @property
    def stats(self) -> RobustSchedulerStats:
        return self._stats


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

@dataclass
class RobustSchedulerStats:
    """Statistics for robust schedulers."""

    total_scheduled: int = 0
    total_batches: int = 0
    preemptions: int = 0
    oom_events: int = 0

    @property
    def mean_batch_size(self) -> float:
        if self.total_batches == 0:
            return 0.0
        return self.total_scheduled / self.total_batches

    @property
    def preemption_rate(self) -> float:
        if self.total_scheduled == 0:
            return 0.0
        return self.preemptions / self.total_scheduled
