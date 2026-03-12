"""squish/production_profiler.py

ProductionProfiler — APM-style continuous inference profiler with p50/p99/p999
latency percentiles per named operation.

In production LLM serving, understanding latency distributions per operation
is critical for meeting SLA targets.  Simple mean latency hides the long-tail
behaviour that degrades user experience: a p99 of 500 ms is often more
actionable than a mean of 50 ms because it reveals that 1 % of requests
experience unacceptable delays even when the average looks healthy.

ProductionProfiler maintains a fixed-size rolling window of samples per
operation name using ``collections.deque(maxlen=window_size)``.  When the
window is full the oldest sample is evicted automatically.  Percentile
statistics are computed lazily on demand with :func:`numpy.percentile`, so
:meth:`record` is O(1) and :meth:`stats` / :meth:`report` are O(n) where
``n <= window_size``.

The implementation is thread-safe for concurrent :meth:`record` calls because
Python's GIL guarantees atomicity of ``deque.append`` together with the
maxlen eviction.

Example usage::

    from squish.production_profiler import ProfilerWindow, ProductionProfiler

    profiler = ProductionProfiler(ProfilerWindow(window_size=500))
    profiler.record("prefill", 12.4)
    profiler.record("decode",   3.1)
    profiler.record("decode",   4.2)

    stats = profiler.stats("decode")
    print(f"p50={stats.p50_ms:.2f} ms  p99={stats.p99_ms:.2f} ms  p999={stats.p999_ms:.2f} ms")
    print(profiler.report())
"""

from __future__ import annotations

__all__ = ["ProfilerWindow", "OperationStats", "ProductionProfiler"]

import collections
from dataclasses import dataclass
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class ProfilerWindow:
    """Rolling-window configuration for :class:`ProductionProfiler`.

    Attributes:
        window_size: Maximum number of latency samples retained per operation.
                     Once full, the oldest sample is evicted on each new record.
    """

    window_size: int = 1000

    def __post_init__(self) -> None:
        if self.window_size < 1:
            raise ValueError(
                f"window_size must be >= 1, got {self.window_size}"
            )


# ---------------------------------------------------------------------------
# Stats output
# ---------------------------------------------------------------------------


@dataclass
class OperationStats:
    """Latency statistics for a single tracked operation.

    All time values are in milliseconds.

    Attributes:
        name:      Operation name.
        n_samples: Number of samples currently held in the rolling window.
        mean_ms:   Arithmetic mean latency.
        p50_ms:    50th-percentile (median) latency.
        p99_ms:    99th-percentile latency.
        p999_ms:   99.9th-percentile latency.
        min_ms:    Minimum observed latency in the window.
        max_ms:    Maximum observed latency in the window.
    """

    name: str
    n_samples: int
    mean_ms: float
    p50_ms: float
    p99_ms: float
    p999_ms: float
    min_ms: float
    max_ms: float


# ---------------------------------------------------------------------------
# Profiler
# ---------------------------------------------------------------------------


class ProductionProfiler:
    """Continuous APM-style per-operation latency tracker.

    Maintains one ``collections.deque(maxlen=window_size)`` per operation name.
    Percentiles are computed on demand using :func:`numpy.percentile`.

    Thread-safe for concurrent :meth:`record` calls.  Not safe for concurrent
    structural mutation (e.g. simultaneous :meth:`reset` and :meth:`record`
    for the same operation from separate threads without external locking).

    Args:
        config: A :class:`ProfilerWindow` controlling the rolling window size.
                Defaults to ``ProfilerWindow()`` (window_size=1000).
    """

    def __init__(self, config: Optional[ProfilerWindow] = None) -> None:
        self._cfg: ProfilerWindow = config if config is not None else ProfilerWindow()
        # Maps operation name to its rolling sample window.
        self._windows: dict[str, collections.deque] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record(self, operation: str, latency_ms: float) -> None:
        """Record a latency sample for *operation*.

        If this is the first sample for *operation* a new deque is created
        automatically.

        Args:
            operation:  Arbitrary operation name string.
            latency_ms: Observed latency in milliseconds.  Must be >= 0.

        Raises:
            ValueError: If *latency_ms* is negative.
        """
        if latency_ms < 0.0:
            raise ValueError(
                f"latency_ms must be >= 0, got {latency_ms}"
            )
        if operation not in self._windows:
            self._windows[operation] = collections.deque(
                maxlen=self._cfg.window_size
            )
        self._windows[operation].append(latency_ms)

    def stats(self, operation: str) -> OperationStats:
        """Return latency statistics for *operation*.

        Args:
            operation: Name of a previously recorded operation.

        Returns:
            An :class:`OperationStats` dataclass populated with percentile
            and summary statistics derived from the current rolling window.

        Raises:
            KeyError: If *operation* has never been recorded.
        """
        if operation not in self._windows:
            raise KeyError(f"Unknown operation: {operation!r}")
        return self._compute_stats(operation, self._windows[operation])

    def report(self) -> dict[str, OperationStats]:
        """Return latency statistics for all tracked operations.

        Returns:
            A dict mapping operation name to :class:`OperationStats`.
            Returns an empty dict if no operations have been recorded.
        """
        return {
            name: self._compute_stats(name, window)
            for name, window in self._windows.items()
        }

    def reset(self, operation: Optional[str] = None) -> None:
        """Clear recorded samples.

        Args:
            operation: If provided, clears only that operation's rolling
                       window while preserving all others.  If ``None``,
                       clears all windows entirely.

        Raises:
            KeyError: If a specific *operation* is given but has never been
                      recorded.
        """
        if operation is None:
            self._windows.clear()
        else:
            if operation not in self._windows:
                raise KeyError(f"Unknown operation: {operation!r}")
            self._windows[operation].clear()

    @property
    def operations(self) -> list[str]:
        """Sorted list of all tracked operation names."""
        return sorted(self._windows.keys())

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_stats(
        self,
        name: str,
        window: collections.deque,
    ) -> OperationStats:
        """Compute percentile statistics from the samples in *window*.

        When the window is empty all statistics are reported as 0.0 to
        allow callers to distinguish an empty window from a missing operation.
        """
        samples = np.array(window, dtype=np.float64)
        n = len(samples)
        if n == 0:
            return OperationStats(
                name=name,
                n_samples=0,
                mean_ms=0.0,
                p50_ms=0.0,
                p99_ms=0.0,
                p999_ms=0.0,
                min_ms=0.0,
                max_ms=0.0,
            )
        p50, p99, p999 = np.percentile(samples, [50.0, 99.0, 99.9])
        return OperationStats(
            name=name,
            n_samples=n,
            mean_ms=float(np.mean(samples)),
            p50_ms=float(p50),
            p99_ms=float(p99),
            p999_ms=float(p999),
            min_ms=float(np.min(samples)),
            max_ms=float(np.max(samples)),
        )
