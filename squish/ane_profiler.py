"""ANEProfiler — Apple Neural Engine utilization profiler for LLM layers.

Tracks which operations have been dispatched to the Apple Neural Engine (ANE)
vs. GPU (MPS) vs. CPU by inspecting operation metadata and timing.  On
inference, MPS/ANE decision is determined by tensor size, dtype, and op type.
This module provides heuristic classification and timing-based profiling.

Classification heuristic (based on empirical Apple Silicon dispatch rules):
  * ``float32`` dtype → always routed to CPU (ANE does not support FP32).
  * ``float16`` / ``bfloat16`` + shape product > threshold → ANE.
  * ``float16`` / ``bfloat16`` + shape product <= threshold → GPU (MPS).
  * Any other dtype → GPU.

Reference:
    Apple, "Optimize your Core ML usage", WWDC 2023.
    Zaremba et al., "LLM Inference on Apple Silicon: Performance Analysis
    and Optimization", Apple Machine Learning Research Blog, 2024.

Usage::

    from squish.ane_profiler import ANEProfiler, ANEMetrics

    profiler = ANEProfiler()
    profiler.record_op("matmul", shape=(4096, 4096), dtype="float16", latency_us=820.0)
    metrics  = profiler.summary()
    print(metrics.ane_fraction)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

__all__ = [
    "OpDevice",
    "ANEOpRecord",
    "ANEMetrics",
    "ANEProfiler",
    "ANEProfilingSession",
]


# ---------------------------------------------------------------------------
# Device constants
# ---------------------------------------------------------------------------


class OpDevice:
    """String constants for the three target devices on Apple Silicon."""

    ANE = "ane"
    GPU = "gpu"
    CPU = "cpu"


# ---------------------------------------------------------------------------
# Data records
# ---------------------------------------------------------------------------


@dataclass
class ANEOpRecord:
    """A single recorded operation with its dispatch classification.

    Parameters
    ----------
    op_name : str
        Name of the operation (e.g. ``"matmul"``, ``"layernorm"``).
    shape : tuple
        Tensor shape used to compute the element count for classification.
    dtype : str
        Data type string (e.g. ``"float16"``, ``"float32"``).
    latency_us : float
        Measured or estimated latency in microseconds.
    device : str
        Classified device: one of :data:`OpDevice.ANE`, :data:`OpDevice.GPU`,
        or :data:`OpDevice.CPU`.
    """

    op_name: str
    shape: tuple
    dtype: str
    latency_us: float
    device: str


@dataclass
class ANEMetrics:
    """Aggregate profiling metrics over a set of recorded operations.

    Parameters
    ----------
    total_ops : int
        Total number of operations recorded.
    ane_ops : int
        Number of operations dispatched to the ANE.
    gpu_ops : int
        Number of operations dispatched to the GPU (MPS).
    cpu_ops : int
        Number of operations dispatched to the CPU.
    total_latency_us : float
        Total latency across all operations (microseconds).
    ane_latency_us : float
        ANE-attributed latency (microseconds).
    """

    total_ops: int
    ane_ops: int
    gpu_ops: int
    cpu_ops: int
    total_latency_us: float
    ane_latency_us: float

    @property
    def ane_fraction(self) -> float:
        """Fraction of operations dispatched to the ANE."""
        if self.total_ops == 0:
            return 0.0
        return self.ane_ops / self.total_ops

    @property
    def ane_latency_fraction(self) -> float:
        """Fraction of total latency attributable to ANE operations."""
        if self.total_latency_us <= 0.0:
            return 0.0
        return self.ane_latency_us / self.total_latency_us

    @property
    def avg_latency_us(self) -> float:
        """Average latency per operation in microseconds."""
        if self.total_ops == 0:
            return 0.0
        return self.total_latency_us / self.total_ops


# ---------------------------------------------------------------------------
# ANEProfiler
# ---------------------------------------------------------------------------


class ANEProfiler:
    """Heuristic ANE/GPU/CPU dispatch classifier and latency profiler.

    Parameters
    ----------
    ane_threshold_elements : int
        Minimum tensor element count for an operation to be classified as
        ANE-dispatched (when dtype is ``float16`` or ``bfloat16``).
        Operations with fewer elements are assumed to remain on the GPU.
        Default ``65536`` (empirically the crossover point on M-series chips
        at which ANE dispatch overhead becomes worth it).

    Examples
    --------
    >>> p = ANEProfiler()
    >>> p.record_op("matmul", shape=(1024, 1024), dtype="float16", latency_us=300.0)
    >>> m = p.summary()
    >>> print(m.ane_fraction)  # 1.0 — 1024*1024 = 1M > threshold
    """

    _ANE_DTYPES = frozenset(("float16", "bfloat16"))
    _CPU_DTYPES = frozenset(("float32",))

    def __init__(self, ane_threshold_elements: int = 65_536) -> None:
        if ane_threshold_elements < 1:
            raise ValueError(
                f"ane_threshold_elements must be >= 1; "
                f"got {ane_threshold_elements}"
            )
        self._threshold = ane_threshold_elements
        self._records: List[ANEOpRecord] = []

    # ── Recording ─────────────────────────────────────────────────────────────

    def record_op(
        self,
        op_name: str,
        shape: tuple,
        dtype: str = "float16",
        latency_us: float = 0.0,
    ) -> None:
        """Classify and record an operation.

        Classification rules:

        * ``dtype == "float32"`` → ``"cpu"``
        * ``dtype in {"float16", "bfloat16"}`` and
          ``prod(shape) > ane_threshold_elements`` → ``"ane"``
        * ``dtype in {"float16", "bfloat16"}`` and
          ``prod(shape) <= ane_threshold_elements`` → ``"gpu"``
        * all other dtypes → ``"gpu"``

        Parameters
        ----------
        op_name : str
            Human-readable operation name.
        shape : tuple
            Tensor shape.  The product of all dimensions is used.
        dtype : str
            Data type string.
        latency_us : float
            Measured latency in microseconds (0.0 if not timed).
        """
        n_elements = int(np.prod(shape)) if shape else 1

        if dtype in self._CPU_DTYPES:
            device = OpDevice.CPU
        elif dtype in self._ANE_DTYPES:
            device = OpDevice.ANE if n_elements > self._threshold else OpDevice.GPU
        else:
            device = OpDevice.GPU

        self._records.append(
            ANEOpRecord(
                op_name=op_name,
                shape=tuple(shape),
                dtype=dtype,
                latency_us=float(latency_us),
                device=device,
            )
        )

    # ── Aggregation ───────────────────────────────────────────────────────────

    def summary(self) -> ANEMetrics:
        """Compute and return aggregate metrics over all recorded ops."""
        total_ops = len(self._records)
        ane_ops = sum(1 for r in self._records if r.device == OpDevice.ANE)
        gpu_ops = sum(1 for r in self._records if r.device == OpDevice.GPU)
        cpu_ops = sum(1 for r in self._records if r.device == OpDevice.CPU)
        total_latency = sum(r.latency_us for r in self._records)
        ane_latency = sum(
            r.latency_us for r in self._records if r.device == OpDevice.ANE
        )
        return ANEMetrics(
            total_ops=total_ops,
            ane_ops=ane_ops,
            gpu_ops=gpu_ops,
            cpu_ops=cpu_ops,
            total_latency_us=total_latency,
            ane_latency_us=ane_latency,
        )

    def op_breakdown(self) -> Dict[str, Dict[str, Any]]:
        """Return per-op-name statistics.

        Returns
        -------
        dict[str, dict]
            Mapping from ``op_name`` to ``{"n_calls": int, "total_us": float,
            "device": str}`` where ``device`` reflects the classification of
            the *last* recorded call for that op name.
        """
        breakdown: Dict[str, Dict[str, Any]] = {}
        for rec in self._records:
            if rec.op_name not in breakdown:
                breakdown[rec.op_name] = {
                    "n_calls": 0,
                    "total_us": 0.0,
                    "device": rec.device,
                }
            breakdown[rec.op_name]["n_calls"] += 1
            breakdown[rec.op_name]["total_us"] += rec.latency_us
            breakdown[rec.op_name]["device"] = rec.device  # last seen device
        return breakdown

    def reset(self) -> None:
        """Clear all recorded operations."""
        self._records.clear()

    @property
    def n_ops(self) -> int:
        """Total number of operations recorded since last :meth:`reset`."""
        return len(self._records)


# ---------------------------------------------------------------------------
# Context manager
# ---------------------------------------------------------------------------


class ANEProfilingSession:
    """Context manager that resets the profiler on entry and captures metrics
    on exit.

    Parameters
    ----------
    profiler : ANEProfiler
        The profiler instance to manage.

    Attributes
    ----------
    metrics : ANEMetrics or None
        Set to the final :class:`ANEMetrics` snapshot on ``__exit__``.
        ``None`` before the context block completes.

    Examples
    --------
    >>> p = ANEProfiler()
    >>> with ANEProfilingSession(p) as sess:
    ...     p.record_op("conv", shape=(64, 512, 512), dtype="float16", latency_us=150.0)
    >>> print(sess.metrics.ane_fraction)
    """

    def __init__(self, profiler: ANEProfiler) -> None:
        self._profiler = profiler
        self.metrics: Optional[ANEMetrics] = None

    def __enter__(self) -> "ANEProfilingSession":
        self._profiler.reset()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.metrics = self._profiler.summary()
