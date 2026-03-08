"""
squish/swiftspec.py

SwiftSpec — Asynchronous Disaggregated Speculative Decoding.

Based on:
  "SwiftSpec: Asynchronous Disaggregated Speculative Decoding for
   Ultra-Low Latency LLM Serving" — arXiv:2506.11309

Key insight
-----------
In standard speculative decoding (EAGLE-3, QuantSpec, SWIFT, …) the draft
generation step is *on the critical path*: the target (verifier) model must
wait for the draft to complete before it can begin verifying.  The total
latency per decode round is:

    T_round = T_draft + T_verify

SwiftSpec removes the sequential dependency by making draft generation and
verification *fully overlapped*:

    Round N:  draft_N runs   ──────────────────→
    Round N:  verify_{N-1}          ──────────→

Under this pipelined scheme, steady-state latency per round approaches:

    T_round ≈ max(T_draft, T_verify)

In Python this is realised with :class:`concurrent.futures.ThreadPoolExecutor`
so that draft and verify run on separate threads *concurrently*.  On Apple
Silicon with MLX, the Metal command queues for draft and verify can be
dispatched simultaneously without additional library changes.

This module provides a simulator of the async pipeline using callable
``draft_fn`` and ``verify_fn`` that stand in for the actual model calls.

Provides
--------
  SwiftSpecConfig    — pipeline parameters.
  SwiftSpecStats     — throughput and acceptance counters.
  SwiftSpecDecoder   — async draft+verify decode loop.
"""

from __future__ import annotations

import concurrent.futures
from dataclasses import dataclass
from typing import Callable, List, Tuple

__all__ = [
    "SwiftSpecConfig",
    "SwiftSpecStats",
    "SwiftSpecDecoder",
]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class SwiftSpecConfig:
    """Parameters for the SwiftSpec async pipeline.

    Parameters
    ----------
    gamma : int
        Draft tokens requested per step (≥ 1).
    max_workers : int
        Thread-pool size.  2 is the natural choice (one draft thread, one
        verify thread).  Must be ≥ 1.
    """

    gamma:       int = 5
    max_workers: int = 2

    def __post_init__(self) -> None:
        if self.gamma < 1:
            raise ValueError("gamma must be ≥ 1")
        if self.max_workers < 1:
            raise ValueError("max_workers must be ≥ 1")


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

@dataclass
class SwiftSpecStats:
    """Per-generation counters returned by :class:`SwiftSpecDecoder`."""

    total_tokens:   int = 0
    draft_steps:    int = 0
    accepted_total: int = 0

    @property
    def mean_accepted_per_step(self) -> float:
        return self.accepted_total / self.draft_steps if self.draft_steps > 0 else 0.0


# ---------------------------------------------------------------------------
# Decoder
# ---------------------------------------------------------------------------

class SwiftSpecDecoder:
    """Asynchronous overlapped draft+verify speculative decoder.

    The key architectural invariant: while the verify step processes the
    current batch, the next batch's draft is already being computed in
    a separate thread.  Both futures are submitted to a shared thread pool.

    Parameters
    ----------
    draft_fn : callable
        ``draft_fn(ids: list[int], gamma: int) -> list[int]``
        Produces ``gamma`` draft token ids given the current context.
    verify_fn : callable
        ``verify_fn(ids: list[int], draft_ids: list[int])
            -> tuple[list[int], object]``
        Verifies *draft_ids* given *ids*, returns ``(accepted_ids, extras)``.
        ``accepted_ids`` is the list of tokens to append to the output;
        it may be shorter than *draft_ids* (partial acceptance) but must
        be non-empty (at least one correction or bonus token).
    config : SwiftSpecConfig
    """

    def __init__(
        self,
        draft_fn:  Callable[[List[int], int], List[int]],
        verify_fn: Callable[[List[int], List[int]], Tuple[List[int], object]],
        config:    SwiftSpecConfig,
    ) -> None:
        self._draft  = draft_fn
        self._verify = verify_fn
        self._cfg    = config

    # ------------------------------------------------------------------

    def generate(
        self,
        input_ids: List[int],
        max_new_tokens: int = 64,
    ) -> Tuple[List[int], SwiftSpecStats]:
        """Generate up to *max_new_tokens* tokens with async overlap.

        The pipeline:

        1. Submit draft for round 0.
        2. While tokens remain to generate:
           a. Retrieve the current draft result.
           b. Immediately submit the *next* draft concurrently.
           c. Verify the current draft (this overlaps with b).
           d. Accept results; update state.
           e. If done, cancel pending next draft and break.
           f. Otherwise, adopt the pending next draft as the new current.

        Parameters
        ----------
        input_ids : list[int]
        max_new_tokens : int

        Returns
        -------
        (output_ids, stats)
        """
        cfg       = self._cfg
        ids       = list(input_ids)
        generated = 0
        stats     = SwiftSpecStats()

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=cfg.max_workers
        ) as ex:
            # Pre-submit the first draft
            draft_future: concurrent.futures.Future = ex.submit(
                self._draft, list(ids), cfg.gamma
            )

            while generated < max_new_tokens:
                draft_ids = draft_future.result()

                # ── Submit next draft concurrently with verification ──────────
                next_draft_future: concurrent.futures.Future = ex.submit(
                    self._draft, list(ids) + draft_ids, cfg.gamma
                )

                # ── Verify current batch (runs while next draft is pending) ───
                accepted, _ = self._verify(list(ids), draft_ids)
                stats.draft_steps    += 1
                stats.accepted_total += len(accepted)
                ids.extend(accepted)
                generated += len(accepted)

                if generated >= max_new_tokens:
                    next_draft_future.cancel()
                    break

                draft_future = next_draft_future

        stats.total_tokens = generated
        return ids, stats
