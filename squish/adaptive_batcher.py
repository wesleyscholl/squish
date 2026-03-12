"""squish/adaptive_batcher.py

AdaptiveBatchController — Throughput/latency-objective dynamic batch size
controller for LLM inference serving.

Static batch sizes are a poor fit for workloads with variable request rates
and heterogeneous sequence lengths.  A batch that is too small wastes GPU
compute; a batch that is too large breaches latency SLAs and blocks shorter
requests behind long ones.

AdaptiveBatchController solves this by maintaining a per-batch-size latency
model learned from live observations.  On each scheduling decision it applies
one of two policies:

* **Throughput mode** — fill the batch as much as possible:
  ``min(queue_depth, max_batch_size)``, maximising hardware utilisation.
* **Latency mode** — find the largest batch size whose *estimated* latency is
  within ``target_latency_ms``.  Falls back to ``min_batch_size`` when no
  candidate meets the target.

The latency model is updated incrementally with an exponential moving average
(alpha=0.3) so recent observations carry more weight.  For batch sizes with
no direct observation the model linearly interpolates between the two nearest
known data points, with flat extrapolation at the edges.

Example usage::

    from squish.adaptive_batcher import BatchObjective, AdaptiveBatchController

    obj  = BatchObjective(mode="latency", target_latency_ms=80.0, max_batch_size=16)
    ctrl = AdaptiveBatchController(obj)

    ctrl.record_observation(1,  20.0)
    ctrl.record_observation(8,  75.0)
    ctrl.record_observation(16, 160.0)

    decision = ctrl.next_batch(queue_depth=12)
    print(decision.batch_size, decision.reason, decision.estimated_latency_ms)
"""

from __future__ import annotations

__all__ = ["BatchObjective", "BatchDecision", "AdaptiveBatchController"]

from dataclasses import dataclass
from typing import Optional

import numpy as np


# EMA smoothing factor for latency model updates.
_EMA_ALPHA: float = 0.3


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class BatchObjective:
    """Optimisation objective for :class:`AdaptiveBatchController`.

    Attributes:
        mode:              ``"throughput"`` or ``"latency"``.
        target_latency_ms: SLA target in milliseconds.  Used only in latency
                           mode.
        max_batch_size:    Hard upper bound on the recommended batch size.
        min_batch_size:    Hard lower bound on the recommended batch size.
    """

    mode: str = "throughput"
    target_latency_ms: float = 100.0
    max_batch_size: int = 32
    min_batch_size: int = 1

    def __post_init__(self) -> None:
        if self.mode not in ("throughput", "latency"):
            raise ValueError(
                f"mode must be 'throughput' or 'latency', got {self.mode!r}"
            )
        if self.target_latency_ms <= 0.0:
            raise ValueError(
                f"target_latency_ms must be > 0, got {self.target_latency_ms}"
            )
        if self.max_batch_size < 1:
            raise ValueError(
                f"max_batch_size must be >= 1, got {self.max_batch_size}"
            )
        if self.min_batch_size < 1:
            raise ValueError(
                f"min_batch_size must be >= 1, got {self.min_batch_size}"
            )
        if self.min_batch_size > self.max_batch_size:
            raise ValueError(
                f"min_batch_size ({self.min_batch_size}) must be "
                f"<= max_batch_size ({self.max_batch_size})"
            )


# ---------------------------------------------------------------------------
# Decision output
# ---------------------------------------------------------------------------


@dataclass
class BatchDecision:
    """Output from :meth:`AdaptiveBatchController.next_batch`.

    Attributes:
        batch_size:           Recommended batch size for the next scheduling
                              cycle.
        reason:               Human-readable explanation of how the decision
                              was reached.
        estimated_latency_ms: Predicted end-to-end latency for this batch
                              size in milliseconds.  0.0 when the model has
                              no observations yet.
    """

    batch_size: int
    reason: str
    estimated_latency_ms: float


# ---------------------------------------------------------------------------
# Controller
# ---------------------------------------------------------------------------


class AdaptiveBatchController:
    """Throughput/latency-objective dynamic batching controller.

    Learns a latency-vs-batch-size model from live observations and uses it
    to select the optimal batch size on each scheduling call.

    Args:
        objective: A :class:`BatchObjective` defining the optimisation target.
    """

    def __init__(self, objective: BatchObjective) -> None:
        self._obj = objective
        # Maps batch_size (int) → EMA latency estimate (float ms).
        self._latency_model: dict[int, float] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def next_batch(self, queue_depth: int) -> BatchDecision:
        """Return the optimal batch size given the current queue depth.

        Args:
            queue_depth: Number of requests currently waiting to be batched.
                         Must be >= 0.  A depth of 0 still returns a valid
                         decision using ``min_batch_size``.

        Returns:
            A :class:`BatchDecision` with the recommended batch size,
            a human-readable reason string, and the estimated latency.

        Raises:
            ValueError: If *queue_depth* is negative.
        """
        if queue_depth < 0:
            raise ValueError(f"queue_depth must be >= 0, got {queue_depth}")
        if self._obj.mode == "throughput":
            return self._decide_throughput(queue_depth)
        return self._decide_latency(queue_depth)

    def record_observation(self, batch_size: int, latency_ms: float) -> None:
        """Update the latency model with a new observed data point.

        Uses an exponential moving average (alpha=0.3) so that recent
        observations are weighted more heavily than older ones.

        Args:
            batch_size:  The batch size that was executed.  Must be >= 1.
            latency_ms:  Observed end-to-end latency in milliseconds.
                         Must be >= 0.

        Raises:
            ValueError: If *batch_size* < 1 or *latency_ms* < 0.
        """
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")
        if latency_ms < 0.0:
            raise ValueError(f"latency_ms must be >= 0, got {latency_ms}")
        if batch_size in self._latency_model:
            prev = self._latency_model[batch_size]
            self._latency_model[batch_size] = (
                _EMA_ALPHA * latency_ms + (1.0 - _EMA_ALPHA) * prev
            )
        else:
            self._latency_model[batch_size] = latency_ms

    @property
    def latency_model(self) -> dict[int, float]:
        """Snapshot of the learned latency model as batch_size → estimated ms."""
        return dict(self._latency_model)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _decide_throughput(self, queue_depth: int) -> BatchDecision:
        """Select batch size to maximise hardware utilisation."""
        obj = self._obj
        raw = queue_depth if queue_depth > 0 else obj.min_batch_size
        bs = max(obj.min_batch_size, min(raw, obj.max_batch_size))
        est = self._estimate_latency(bs)
        reason = (
            f"throughput mode: batch_size=clamp(queue_depth={queue_depth}, "
            f"{obj.min_batch_size}, {obj.max_batch_size})={bs}"
        )
        return BatchDecision(batch_size=bs, reason=reason, estimated_latency_ms=est)

    def _decide_latency(self, queue_depth: int) -> BatchDecision:
        """Select the largest batch size whose estimated latency <= target."""
        obj = self._obj
        cap = min(queue_depth, obj.max_batch_size) if queue_depth > 0 else obj.max_batch_size
        cap = max(cap, obj.min_batch_size)

        # Probe candidates from largest to smallest; accept the first that fits.
        chosen_bs = obj.min_batch_size
        chosen_est = self._estimate_latency(obj.min_batch_size)
        found = False
        for bs in range(cap, obj.min_batch_size - 1, -1):
            est = self._estimate_latency(bs)
            if est <= obj.target_latency_ms:
                chosen_bs = bs
                chosen_est = est
                found = True
                break

        if found:
            reason = (
                f"latency mode: largest batch_size={chosen_bs} with "
                f"estimated_latency={chosen_est:.1f} ms "
                f"<= target={obj.target_latency_ms} ms"
            )
        else:
            reason = (
                f"latency mode: no candidate meets target={obj.target_latency_ms} ms; "
                f"falling back to min_batch_size={obj.min_batch_size}"
            )
        return BatchDecision(
            batch_size=chosen_bs,
            reason=reason,
            estimated_latency_ms=chosen_est,
        )

    def _estimate_latency(self, batch_size: int) -> float:
        """Estimate latency for *batch_size* using the learned model.

        Returns the EMA estimate directly when *batch_size* is known.
        Linearly interpolates between the two nearest bracketing points
        otherwise.  Uses flat extrapolation outside the known range.
        Returns 0.0 when no observations have been recorded yet.
        """
        if batch_size in self._latency_model:
            return self._latency_model[batch_size]
        if not self._latency_model:
            return 0.0

        known_sizes = sorted(self._latency_model.keys())
        known_lats = [self._latency_model[s] for s in known_sizes]

        # Flat extrapolation below the minimum known size.
        if batch_size <= known_sizes[0]:
            return float(known_lats[0])
        # Flat extrapolation above the maximum known size.
        if batch_size >= known_sizes[-1]:
            return float(known_lats[-1])

        # Linear interpolation between the two bracketing known points.
        idx = int(np.searchsorted(known_sizes, batch_size))
        lo_bs, hi_bs = known_sizes[idx - 1], known_sizes[idx]
        lo_lat, hi_lat = known_lats[idx - 1], known_lats[idx]
        frac = (batch_size - lo_bs) / (hi_bs - lo_bs)
        return lo_lat + frac * (hi_lat - lo_lat)
