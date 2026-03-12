"""ContinuousBatching — Continuous batching scheduler with mid-generation insertion.

Standard static batching waits for all requests in a batch to finish before
accepting new ones.  Continuous batching (Yu et al., OSDI 2022) preempts the
current batch at each decode step to insert newly-arrived requests, maximising
GPU utilisation at any arrival rate.

Reference:
    Yu et al., "Orca: A Distributed Serving System for Transformer-Based
    Generative Models", OSDI 2022.  https://www.usenix.org/conference/osdi22

Usage::

    from squish.continuous_batching import CBScheduler, InFlightRequest

    sched = CBScheduler(CBConfig(max_batch_size=32, max_seq_len=2048))
    sched.submit(InFlightRequest(request_id="r1", prompt_tokens=[1,2,3], max_new_tokens=128))
    batch = sched.step_batch()
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Dict, List, Optional

__all__ = [
    "CBConfig",
    "RequestState",
    "InFlightRequest",
    "CBScheduler",
    "CBStats",
]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CBConfig:
    """Configuration for the continuous-batching scheduler.

    Parameters
    ----------
    max_batch_size : int
        Maximum number of requests that can run concurrently.  Must be >= 1.
    max_seq_len : int
        Maximum total sequence length (prompt + generated tokens) per request.
        Must be >= 1.
    priority_policy : str
        Scheduling policy for promoting waiting requests into the batch.
        ``"fifo"`` — first-in first-out (submission order).
        ``"sjf"``  — shortest-job-first (ascending ``max_new_tokens``).
    """

    max_batch_size: int = 32
    max_seq_len: int = 2048
    priority_policy: str = "fifo"

    def __post_init__(self) -> None:
        if self.max_batch_size < 1:
            raise ValueError(
                f"max_batch_size must be >= 1; got {self.max_batch_size}"
            )
        if self.max_seq_len < 1:
            raise ValueError(f"max_seq_len must be >= 1; got {self.max_seq_len}")
        if self.priority_policy not in ("fifo", "sjf"):
            raise ValueError(
                f"priority_policy must be 'fifo' or 'sjf'; "
                f"got {self.priority_policy!r}"
            )


# ---------------------------------------------------------------------------
# Request state
# ---------------------------------------------------------------------------


class RequestState(str, enum.Enum):
    """Lifecycle state of an in-flight request."""

    WAITING = "waiting"
    RUNNING = "running"
    FINISHED = "finished"
    PREEMPTED = "preempted"


# ---------------------------------------------------------------------------
# InFlightRequest
# ---------------------------------------------------------------------------


@dataclass
class InFlightRequest:
    """A single generation request tracked by the scheduler.

    Parameters
    ----------
    request_id : str
        Unique identifier for this request.
    prompt_tokens : list[int]
        Tokenised prompt (input context).
    max_new_tokens : int
        Maximum number of tokens to generate.
    generated_tokens : list[int]
        Tokens produced so far (mutated in-place by the scheduler).
    state : RequestState
        Current lifecycle state of the request.
    priority : int
        Application-level priority hint (higher = more urgent).  Not used
        by the built-in ``"fifo"``/``"sjf"`` policies but available for
        custom schedulers.
    """

    request_id: str
    prompt_tokens: List[int]
    max_new_tokens: int
    generated_tokens: List[int] = field(default_factory=list)
    state: RequestState = RequestState.WAITING
    priority: int = 0

    def __post_init__(self) -> None:
        if self.max_new_tokens < 1:
            raise ValueError(
                f"max_new_tokens must be >= 1; got {self.max_new_tokens}"
            )

    @property
    def tokens_remaining(self) -> int:
        """Number of tokens still to be generated."""
        return self.max_new_tokens - len(self.generated_tokens)

    @property
    def is_finished(self) -> bool:
        """``True`` when ``tokens_remaining == 0``."""
        return self.tokens_remaining <= 0


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


@dataclass
class CBStats:
    """Aggregate statistics from a :class:`CBScheduler` session.

    Parameters
    ----------
    total_submitted : int
        Requests submitted via :meth:`CBScheduler.submit`.
    total_completed : int
        Requests that reached the ``FINISHED`` state.
    total_preemptions : int
        Total times :meth:`CBScheduler.preempt` was called.
    total_tokens_generated : int
        Total tokens appended via :meth:`CBScheduler.complete_token`.
    n_steps : int
        Total :meth:`CBScheduler.step_batch` calls.
    """

    total_submitted: int
    total_completed: int
    total_preemptions: int
    total_tokens_generated: int
    n_steps: int

    @property
    def avg_batch_size(self) -> float:
        """Average tokens generated per scheduler step."""
        if self.n_steps == 0:
            return 0.0
        return self.total_tokens_generated / self.n_steps

    @property
    def completion_rate(self) -> float:
        """Fraction of submitted requests that completed (not preempted)."""
        if self.total_submitted == 0:
            return 0.0
        return self.total_completed / self.total_submitted


# ---------------------------------------------------------------------------
# CBScheduler
# ---------------------------------------------------------------------------


class CBScheduler:
    """Continuous-batching scheduler with mid-generation request insertion.

    Each call to :meth:`step_batch` returns the current active batch (up to
    ``max_batch_size`` running requests) after promoting as many waiting
    requests as there is capacity.  The caller is responsible for generating
    one token per running request and feeding it back via
    :meth:`complete_token`.

    Parameters
    ----------
    config : CBConfig
        Scheduler configuration.

    Examples
    --------
    >>> sched = CBScheduler(CBConfig(max_batch_size=4))
    >>> sched.submit(InFlightRequest("r1", [1, 2, 3], max_new_tokens=8))
    >>> batch = sched.step_batch()
    >>> for req in batch:
    ...     sched.complete_token(req.request_id, token=42)
    """

    def __init__(self, config: CBConfig) -> None:
        self._config = config
        # Authoritative store: request_id → request
        self._requests: Dict[str, InFlightRequest] = {}
        # Ordered waiting queue (insertion order preserved for FIFO)
        self._waiting: List[str] = []
        # Currently running request ids
        self._running: List[str] = []
        # Terminal-state request ids
        self._finished: List[str] = []
        self._preempted: List[str] = []
        # Counters
        self._n_steps: int = 0
        self._total_tokens_generated: int = 0
        self._total_submitted: int = 0
        self._total_completed: int = 0
        self._total_preemptions: int = 0

    # ── Submission ────────────────────────────────────────────────────────────

    def submit(self, request: InFlightRequest) -> None:
        """Add a new request to the waiting queue.

        Parameters
        ----------
        request : InFlightRequest
            The request to enqueue.  Its ``state`` is set to ``WAITING``.

        Raises
        ------
        ValueError
            If a request with the same ``request_id`` is already tracked.
        """
        if request.request_id in self._requests:
            raise ValueError(
                f"request_id {request.request_id!r} is already tracked."
            )
        request.state = RequestState.WAITING
        self._requests[request.request_id] = request
        self._waiting.append(request.request_id)
        self._total_submitted += 1

    # ── Step ─────────────────────────────────────────────────────────────────

    def step_batch(self) -> List[InFlightRequest]:
        """Advance the scheduler by one decode step.

        Promotes waiting requests into the running state up to
        ``max_batch_size`` total running slots, honouring the configured
        ``priority_policy``.

        Returns
        -------
        list[InFlightRequest]
            All currently running requests (the active batch).
        """
        cfg = self._config
        available_slots = cfg.max_batch_size - len(self._running)

        if available_slots > 0 and self._waiting:
            if cfg.priority_policy == "sjf":
                # Sort waiting by max_new_tokens ascending (shortest first).
                sorted_waiting = sorted(
                    self._waiting,
                    key=lambda rid: self._requests[rid].max_new_tokens,
                )
            else:
                # FIFO: maintain submission order.
                sorted_waiting = list(self._waiting)

            to_promote = sorted_waiting[:available_slots]
            promote_set = set(to_promote)
            self._waiting = [rid for rid in self._waiting if rid not in promote_set]

            for rid in to_promote:
                req = self._requests[rid]
                req.state = RequestState.RUNNING
                self._running.append(rid)

        self._n_steps += 1
        return [self._requests[rid] for rid in self._running]

    # ── Token completion ──────────────────────────────────────────────────────

    def complete_token(self, request_id: str, token: int) -> None:
        """Append a generated token to a running request.

        If the request reaches ``max_new_tokens`` after this token, it is
        automatically moved to the ``FINISHED`` state and removed from the
        active batch.

        Parameters
        ----------
        request_id : str
            ID of the running request.
        token : int
            The generated token id.

        Raises
        ------
        KeyError
            If ``request_id`` is not tracked.
        ValueError
            If the request is not in ``RUNNING`` state.
        """
        if request_id not in self._requests:
            raise KeyError(f"Unknown request_id {request_id!r}.")
        req = self._requests[request_id]
        if req.state != RequestState.RUNNING:
            raise ValueError(
                f"Request {request_id!r} is not running (state={req.state.value})."
            )
        req.generated_tokens.append(token)
        self._total_tokens_generated += 1

        if req.is_finished:
            req.state = RequestState.FINISHED
            self._running.remove(request_id)
            self._finished.append(request_id)
            self._total_completed += 1

    # ── Preemption ────────────────────────────────────────────────────────────

    def preempt(self, request_id: str) -> None:
        """Move a running request to the ``PREEMPTED`` state.

        The request is removed from the active batch.  It can be resubmitted
        later if desired.

        Parameters
        ----------
        request_id : str
            ID of the running request to preempt.

        Raises
        ------
        KeyError
            If ``request_id`` is not tracked.
        ValueError
            If the request is not currently running.
        """
        if request_id not in self._requests:
            raise KeyError(f"Unknown request_id {request_id!r}.")
        req = self._requests[request_id]
        if req.state != RequestState.RUNNING:
            raise ValueError(
                f"Request {request_id!r} is not running (state={req.state.value})."
            )
        req.state = RequestState.PREEMPTED
        self._running.remove(request_id)
        self._preempted.append(request_id)
        self._total_preemptions += 1

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def n_waiting(self) -> int:
        """Number of requests currently waiting for a batch slot."""
        return len(self._waiting)

    @property
    def n_running(self) -> int:
        """Number of requests currently in the active batch."""
        return len(self._running)

    @property
    def n_finished(self) -> int:
        """Number of requests that have completed generation."""
        return len(self._finished)

    @property
    def n_steps(self) -> int:
        """Total number of :meth:`step_batch` calls performed."""
        return self._n_steps

    @property
    def throughput(self) -> float:
        """Tokens generated per scheduler step (average across all steps)."""
        if self._n_steps == 0:
            return 0.0
        return self._total_tokens_generated / self._n_steps

    def scheduler_stats(self) -> CBStats:
        """Return a snapshot of cumulative scheduler statistics."""
        return CBStats(
            total_submitted=self._total_submitted,
            total_completed=self._total_completed,
            total_preemptions=self._total_preemptions,
            total_tokens_generated=self._total_tokens_generated,
            n_steps=self._n_steps,
        )
