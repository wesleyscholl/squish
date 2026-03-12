"""squish/infer_gateway.py

InferenceGateway — Smart front-door routing for a fleet of LLM inference
workers.

A production LLM serving cluster runs multiple worker processes (or pods),
each capable of handling a bounded number of concurrent requests.  Without a
centralised gateway, routing is static and unable to adapt to hot workers,
version heterogeneity, or failed nodes.

InferenceGateway implements three routing heuristics in priority order:

1. **Version affinity** — when ``required_version`` is provided, only workers
   running that model version are considered.  Among them, least-loaded wins.
2. **Least-loaded** — among all healthy workers (or the version-filtered
   subset), the worker with the lowest current utilisation
   (``active / capacity``) is chosen.
3. **Fallback** — if a version-filtered candidate set is entirely at capacity,
   the gateway widens the search to all healthy workers with the
   ``"fallback"`` reason.  A :class:`RuntimeError` is raised only when every
   registered worker is unhealthy.

The ``complete()`` method must be called once per completed request to
decrement the active count; forgetting to call it will cause the worker to
appear permanently busy.

Example usage::

    from squish.infer_gateway import InferenceGateway

    gw = InferenceGateway()
    gw.register("worker-0", capacity=4, model_version="v1")
    gw.register("worker-1", capacity=4, model_version="v2")

    result = gw.route("req-001", required_version="v1")
    print(result)   # RouteResult(worker_id='worker-0', reason='affinity', ...)

    gw.complete("worker-0")
    print(f"total load: {gw.total_load:.2%}")
"""

from __future__ import annotations

__all__ = ["WorkerInfo", "RouteResult", "InferenceGateway"]

import dataclasses
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class WorkerInfo:
    """Mutable state record for a single inference worker.

    Attributes:
        worker_id:     Unique identifier for the worker.
        capacity:      Maximum number of concurrent requests the worker can
                       handle.
        active:        Number of requests currently in-flight on this worker.
        healthy:       Whether the worker is accepting new requests.
        model_version: Model version string currently served by this worker.
    """

    worker_id:     str
    capacity:      int
    active:        int   = 0
    healthy:       bool  = True
    model_version: str   = "default"

    def __post_init__(self) -> None:
        if self.capacity < 1:
            raise ValueError(
                f"capacity must be >= 1, got {self.capacity} "
                f"for worker '{self.worker_id}'"
            )
        if self.active < 0:
            raise ValueError(
                f"active must be >= 0, got {self.active} "
                f"for worker '{self.worker_id}'"
            )


@dataclasses.dataclass
class RouteResult:
    """Outcome of a single routing decision.

    Attributes:
        worker_id:     Worker chosen to handle the request.
        reason:        Human-readable routing rationale:
                       ``"least-loaded"`` — standard least-load selection;
                       ``"affinity"`` — version affinity narrowed the pool;
                       ``"fallback"`` — version-filtered pool was all at
                       capacity so the search widened.
        load_fraction: Utilisation of the chosen worker **after** the new
                       request was counted (``active / capacity``).
    """

    worker_id:     str
    reason:        str
    load_fraction: float


# ---------------------------------------------------------------------------
# InferenceGateway
# ---------------------------------------------------------------------------


class InferenceGateway:
    """Smart routing gateway for an LLM inference worker fleet.

    Workers are registered with a maximum capacity and model version label.
    All routing decisions are health-aware; unhealthy workers are excluded from
    consideration.  The gateway maintains no request queues — it is purely a
    routing oracle, and the caller is responsible for managing request
    lifecycle.
    """

    def __init__(self) -> None:
        self._workers: dict[str, WorkerInfo] = {}

    # ------------------------------------------------------------------
    # Worker management
    # ------------------------------------------------------------------

    def register(
        self,
        worker_id:     str,
        capacity:      int,
        model_version: str = "default",
    ) -> None:
        """Register a new worker with the gateway.

        Args:
            worker_id:     Unique string identifier for the worker.
            capacity:      Maximum number of concurrent requests.
            model_version: Model version label served by this worker.

        Raises:
            ValueError: If ``worker_id`` is already registered or
                        ``capacity`` < 1.
        """
        if worker_id in self._workers:
            raise ValueError(
                f"Worker '{worker_id}' is already registered."
            )
        self._workers[worker_id] = WorkerInfo(
            worker_id=worker_id,
            capacity=capacity,
            active=0,
            healthy=True,
            model_version=model_version,
        )

    def deregister(self, worker_id: str) -> None:
        """Remove a worker from the routing pool.

        Args:
            worker_id: Identifier of the worker to remove.

        Raises:
            KeyError: If ``worker_id`` is not registered.
        """
        if worker_id not in self._workers:
            raise KeyError(f"Worker '{worker_id}' is not registered.")
        del self._workers[worker_id]

    def mark_unhealthy(self, worker_id: str) -> None:
        """Mark a worker as unhealthy so it receives no new routing decisions.

        Args:
            worker_id: Identifier of the worker.

        Raises:
            KeyError: If ``worker_id`` is not registered.
        """
        self._get_worker(worker_id).healthy = False

    def mark_healthy(self, worker_id: str) -> None:
        """Restore a previously unhealthy worker to the routing pool.

        Args:
            worker_id: Identifier of the worker.

        Raises:
            KeyError: If ``worker_id`` is not registered.
        """
        self._get_worker(worker_id).healthy = True

    # ------------------------------------------------------------------
    # Routing
    # ------------------------------------------------------------------

    def route(
        self,
        request_id:       str,
        required_version: Optional[str] = None,
    ) -> RouteResult:
        """Select a worker for a new request and increment its active count.

        Routing priority:

        1. If ``required_version`` is given, restrict the candidate pool to
           healthy workers of that version.  Among them choose the least loaded.
        2. If the version-filtered pool is empty (no healthy workers match the
           version), widen to all healthy workers and route with reason
           ``"fallback"``.
        3. If no version constraint is given, choose the globally least-loaded
           healthy worker with reason ``"least-loaded"``.

        Args:
            request_id:       Opaque identifier used only in error messages.
            required_version: If provided, prefer workers running this model
                              version.

        Returns:
            A :class:`RouteResult` describing the selected worker.

        Raises:
            RuntimeError: If there are no healthy workers in the registry.
        """
        healthy = [w for w in self._workers.values() if w.healthy]
        if not healthy:
            raise RuntimeError(
                f"Cannot route request '{request_id}': "
                "no healthy workers are registered."
            )

        candidates = healthy
        reason     = "least-loaded"

        if required_version is not None:
            versioned = [w for w in healthy if w.model_version == required_version]
            if versioned:
                candidates = versioned
                reason     = "affinity"
            else:
                # No healthy worker matches the required version — fall back to
                # all healthy workers so the request is not dropped.
                reason = "fallback"

        chosen = min(candidates, key=lambda w: w.active / w.capacity)
        chosen.active += 1

        return RouteResult(
            worker_id=chosen.worker_id,
            reason=reason,
            load_fraction=chosen.active / chosen.capacity,
        )

    def complete(self, worker_id: str) -> None:
        """Decrement the active request count for a worker after completion.

        This method must be called exactly once for every successful
        :meth:`route` call to keep utilisation metrics accurate.

        Args:
            worker_id: Identifier of the worker that finished a request.

        Raises:
            KeyError:   If ``worker_id`` is not registered.
            ValueError: If the active count would go below zero.
        """
        worker = self._get_worker(worker_id)
        if worker.active <= 0:
            raise ValueError(
                f"Cannot complete: worker '{worker_id}' has no active requests."
            )
        worker.active -= 1

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def workers(self) -> dict[str, WorkerInfo]:
        """Snapshot of the worker registry (shallow copy)."""
        return dict(self._workers)

    @property
    def total_load(self) -> float:
        """Overall utilisation fraction across all healthy workers.

        Computed as ``sum(active) / sum(capacity)`` for healthy workers.
        Returns 0.0 when no healthy workers are registered.
        """
        healthy = [w for w in self._workers.values() if w.healthy]
        if not healthy:
            return 0.0
        total_active   = sum(w.active   for w in healthy)
        total_capacity = sum(w.capacity for w in healthy)
        return total_active / total_capacity

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_worker(self, worker_id: str) -> WorkerInfo:
        """Return :class:`WorkerInfo` for ``worker_id`` or raise :class:`KeyError`."""
        if worker_id not in self._workers:
            raise KeyError(f"Worker '{worker_id}' is not registered.")
        return self._workers[worker_id]
