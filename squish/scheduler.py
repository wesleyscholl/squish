#!/usr/bin/env python3
"""
squish/scheduler.py

Continuous batching scheduler for the Squish inference server.

Problem
-------
An autoregressive LLM generates one token per forward pass.  Each forward pass
on a 7B model takes ~50-70 ms on an M-series chip.  A single-threaded server
spends ~95% of that time with the GPU fully occupied — but *only for one request*.
Concurrent users queue behind that request and see latencies that scale linearly
with queue depth.

Solution: Static Batching with a Coalescing Window
---------------------------------------------------
Instead of processing one request at a time, the scheduler:

  1. Collects all requests that arrive within a ``batch_window_ms`` window
     (default 20 ms — typically catches 2–8 requests at moderate load).

  2. Tokenises each prompt and pads the batch to the longest sequence
     (same strategy as Phase 1.5 padded batch evaluation).

  3. Runs the padded batch through the model in ONE forward pass → gets logits
     for ALL requests simultaneously.

  4. Samples the next token per request (respecting each request's temperature,
     top_p, seed, and stop sequences).

  5. Removes completed requests (EOS / max_tokens) from the batch, yields their
     final tokens, and starts accepting new requests for the next window.

Throughput gain
---------------
For a batch of N requests, one forward pass costs roughly the same latency as a
single-request pass (memory-bandwidth-bound → throughput, not latency-bound on
Apple Silicon unified memory).  So N concurrent requests yield ~N× the token
throughput at similar per-request latency.

Limitations
-----------
- This is *static* batching (not continuous / paged attention).  New requests
  that arrive mid-batch must wait for the current batch to complete before joining.
  True continuous batching would require paged KV cache management (a larger
  future item).
- Maximum effective batch size is limited by unified memory; the scheduler
  enforces ``max_batch_size`` (default 8) to prevent OOM.
- Variable output lengths mean SOME requests will finish early while the batch
  continues.  Completed slots are freed and their prompt padding is removed for
  subsequent steps.

Usage
-----
In server.py:

    from squish.scheduler import BatchScheduler
    _scheduler = BatchScheduler(model, tokenizer, max_batch_size=8,
                                batch_window_ms=20)
    _scheduler.start()

    # In the chat/completion endpoint (async path):
    async for tok_text, finish_reason in _scheduler.submit(prompt, ...):
        yield tok_text

Standalone test:

    python3 -m squish.scheduler
"""
import asyncio
import dataclasses
import logging
import hashlib
import queue
import threading
import time
from collections.abc import AsyncIterator

import numpy as np

log = logging.getLogger(__name__)

# Sentinel value placed on a request's output queue when generation is done
_DONE = object()


# ---------------------------------------------------------------------------
# Per-request data
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class _Request:
    """Internal representation of one pending / active generation request."""
    request_id:  str
    input_ids:   list[int]            # tokenised prompt
    max_tokens:  int
    temperature: float
    top_p:       float
    stop_ids:    list[list[int]]
    seed:        int | None

    # Output: tokens are placed here by the worker thread
    # Items are (token_text: str, finish_reason: str | None)
    out_queue: queue.SimpleQueue = dataclasses.field(default_factory=queue.SimpleQueue)

    # Mutable state during generation
    generated_ids: list[int] = dataclasses.field(default_factory=list)
    stop_buf:      list[int] = dataclasses.field(default_factory=list)
    done:          bool      = False
    finish_reason: str       = "stop"


# ---------------------------------------------------------------------------
# Sampling helpers (numpy, CPU — fast enough for per-step sampling)
# ---------------------------------------------------------------------------

def _softmax_f32(logits: np.ndarray) -> np.ndarray:
    """Numerically stable softmax over the last axis (float32 input expected)."""
    shifted = logits - logits.max()
    exp_l   = np.exp(shifted)
    return exp_l / (exp_l.sum() + 1e-9)


def _top_p_filter(probs: np.ndarray, top_p: float) -> np.ndarray:
    """Zero out tokens below the top-p cumulative probability threshold."""
    if top_p >= 1.0:
        return probs
    sorted_idx  = np.argsort(-probs)         # descending
    sorted_p    = probs[sorted_idx]
    cum_p       = np.cumsum(sorted_p)
    # Keep tokens up to and including the one that pushes cum_p above top_p
    cutoff      = int(np.searchsorted(cum_p, top_p)) + 1
    mask        = np.zeros_like(probs)
    mask[sorted_idx[:max(1, cutoff)]] = 1.0
    filtered    = probs * mask
    total       = filtered.sum()
    return filtered / total if total > 0 else probs


def _sample_token(logits_row: np.ndarray, temperature: float, top_p: float,
                  rng: np.random.Generator) -> int:
    """Sample next token-id from logits for one request."""
    if temperature <= 0.0 or temperature < 1e-5:
        return int(np.argmax(logits_row))
    probs = _softmax_f32((logits_row / temperature).astype(np.float32))
    probs = _top_p_filter(probs, top_p)
    return int(rng.choice(len(probs), p=probs))


def _check_stop(req: _Request, next_id: int) -> bool:
    """Return True if the new token completes a stop sequence."""
    if not req.stop_ids:
        return False
    req.stop_buf.append(next_id)
    for seq in req.stop_ids:
        if req.stop_buf[-len(seq):] == seq:
            return True
    if len(req.stop_buf) > 64:
        req.stop_buf = req.stop_buf[-64:]
    return False


# ---------------------------------------------------------------------------
# BatchScheduler
# ---------------------------------------------------------------------------

class QueueFullError(RuntimeError):
    """Raised when the pending request queue has reached ``max_pending`` capacity.

    The server converts this to an HTTP 429 Too Many Requests response.
    """


class BatchScheduler:
    """
    Coalescing-window batch scheduler for autoregressive generation.

    Thread model:
    • Caller threads submit requests via :meth:`submit_sync` (blocking iterator)
      or :meth:`submit` (async iterator for FastAPI endpoints).
    • ONE background worker thread drains ``_pending_queue``, forms batches,
      and runs the generation loop.

    Parameters
    ----------
    model           : loaded mlx_lm model (already on Metal)
    tokenizer       : HuggingFace tokenizer matching the model
    max_batch_size  : hard cap on concurrent requests (default 8)
    max_pending     : maximum pending queue depth before HTTP 429 is raised
                      (0 = unlimited, default 64)
    batch_window_ms : how long to wait for more requests before starting
                      a batch (default 20 ms)
    pad_token_id    : padding token ID (auto-detected from tokenizer)
    """

    def __init__(
        self,
        model,
        tokenizer,
        max_batch_size:  int   = 8,
        batch_window_ms: float = 20.0,
        pad_token_id:    int | None = None,
        max_pending:     int   = 64,
    ):
        import mlx.core as mx  # noqa: F401 (validate import on init)

        self._model          = model
        self._tokenizer      = tokenizer
        self._max_batch      = max_batch_size
        self._window_ms      = batch_window_ms
        self._max_pending    = max_pending
        self._pad_id         = (pad_token_id
                                or getattr(tokenizer, "pad_token_id", None)
                                or getattr(tokenizer, "eos_token_id", None)
                                or 0)
        self._eos_id         = (getattr(tokenizer, "eos_token_id", None) or 151645)

        # Pending requests submitted by callers (thread-safe queue)
        self._pending: queue.Queue = queue.Queue()

        # ── Phase D: Double-buffer queues ─────────────────────────────────────
        # _prepare_worker fills prepared batches; _worker consumes them.
        self._prepared_queue: queue.Queue = queue.Queue(maxsize=1)
        self._prepare_thread: threading.Thread | None = None

        # Worker thread
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()

        # Per-request RNG (seeded lazily)
        self._rng = np.random.default_rng()

        # Metrics
        self.total_batches     = 0
        self.total_tokens_gen  = 0
        self.total_requests    = 0
        self._lock             = threading.Lock()

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def start(self) -> "BatchScheduler":
        """Start the prepare-worker and GPU-worker background threads."""
        if self._thread is not None and self._thread.is_alive():
            return self
        self._stop_event.clear()
        self._prepare_thread = threading.Thread(
            target=self._prepare_worker, daemon=True,
            name="squish-prepare-worker",
        )
        self._prepare_thread.start()
        self._thread = threading.Thread(target=self._worker, daemon=True,
                                        name="squish-batch-worker")
        self._thread.start()
        log.info("BatchScheduler started  (max_batch=%d  window=%.0fms)",
                 self._max_batch, self._window_ms)
        return self

    def stop(self, timeout: float = 5.0) -> None:
        """Signal both threads to stop and wait for them to finish."""
        self._stop_event.set()
        if self._prepare_thread is not None:
            self._prepare_thread.join(timeout=timeout)
        self._prepare_thread = None
        if self._thread is not None:
            self._thread.join(timeout=timeout)
        self._thread = None

    def is_running(self) -> bool:
        t1 = self._thread is not None and self._thread.is_alive()
        t2 = self._prepare_thread is not None and self._prepare_thread.is_alive()
        return t1 and t2

    # ── Request submission ────────────────────────────────────────────────────

    def _make_request(  # pragma: no cover
        self,
        request_id:  str,
        prompt:      str,
        max_tokens:  int,
        temperature: float,
        top_p:       float,
        stop_ids:    list[list[int]],
        seed:        int | None,
    ) -> _Request:
        """Tokenise prompt and build a _Request object."""
        try:
            input_ids = self._tokenizer.encode(prompt, add_special_tokens=True)
        except Exception:
            input_ids = self._tokenizer(prompt)["input_ids"]
        return _Request(
            request_id  = request_id,
            input_ids   = input_ids,
            max_tokens  = max_tokens,
            temperature = temperature,
            top_p       = top_p,
            stop_ids    = stop_ids,
            seed        = seed,
        )

    def submit_sync(  # pragma: no cover
        self,
        prompt:      str,
        max_tokens:  int   = 512,
        temperature: float = 0.7,
        top_p:       float = 0.9,
        stop_ids:    list[list[int]] | None = None,
        seed:        int | None = None,
        request_id:  str | None = None,
    ):
        """
        Submit a generation request and iterate over (token_text, finish_reason).

        Blocking Python iterator — suitable for use in a dedicated thread.
        """
        import uuid as _uuid
        rid = request_id or _uuid.uuid4().hex[:8]
        req = self._make_request(rid, prompt, max_tokens, temperature, top_p,
                                 stop_ids or [], seed)
        if self._max_pending and self._pending.qsize() >= self._max_pending:
            raise QueueFullError(
                f"Server is at capacity ({self._max_pending} pending requests). "
                "Retry after a moment."
            )
        self._pending.put(req)
        while True:
            item = req.out_queue.get()
            if item is _DONE:
                break
            yield item   # (tok_text, finish_reason_or_None)

    async def submit(  # pragma: no cover
        self,
        prompt:      str,
        max_tokens:  int   = 512,
        temperature: float = 0.7,
        top_p:       float = 0.9,
        stop_ids:    list[list[int]] | None = None,
        seed:        int | None = None,
        request_id:  str | None = None,
    ) -> AsyncIterator[tuple[str, str | None]]:
        """
        Submit a generation request and async-iterate (token_text, finish_reason).

        Uses ``asyncio.to_thread`` to bridge the sync ``out_queue`` to async callers.
        Compatible with FastAPI streaming responses.
        """
        import uuid as _uuid
        if self._max_pending and self._pending.qsize() >= self._max_pending:
            raise QueueFullError(
                f"Batch scheduler queue full ({self._max_pending} pending). "
                "Retry after a moment."
            )
        rid = request_id or _uuid.uuid4().hex[:8]
        req = self._make_request(rid, prompt, max_tokens, temperature, top_p,
                                 stop_ids or [], seed)
        self._pending.put(req)

        loop = asyncio.get_event_loop()
        while True:
            # Run the blocking get() in a thread pool to avoid blocking the event loop
            item = await loop.run_in_executor(None, req.out_queue.get)
            if item is _DONE:
                break
            yield item

    # ── Metrics ──────────────────────────────────────────────────────────────

    def stats(self) -> dict:
        with self._lock:
            return {
                "running":          self.is_running(),
                "total_batches":    self.total_batches,
                "total_tokens_gen": self.total_tokens_gen,
                "total_requests":   self.total_requests,
                "pending_queue":    self._pending.qsize(),
                "prepared_queue":   self._prepared_queue.qsize(),
                "max_batch_size":   self._max_batch,
                "batch_window_ms":  self._window_ms,
            }

    # ── Worker ────────────────────────────────────────────────────────────────

    def _prepare_worker(self) -> None:  # pragma: no cover
        """
        CPU-side prepare thread: collect + group + order batches, then hand
        them to the GPU worker via *_prepared_queue*.
        """
        log.debug("BatchScheduler prepare-worker started")
        while not self._stop_event.is_set():
            # Collect up to 2× max_batch_size requests for prefix grouping.
            pool = self._collect_batch(limit=self._max_batch * 2)
            if not pool:
                continue

            # D2: prefer same-prefix cohorts; put extras back for next window.
            batch, leftovers = self._group_by_prefix(pool)
            for req in leftovers:
                self._pending.put(req)

            # D3: decode requests execute before prefill in the same batch.
            batch = self._sort_decode_first(batch)

            # Hand off to GPU worker (blocks until worker consumes prev batch).
            while not self._stop_event.is_set():
                try:
                    self._prepared_queue.put(batch, timeout=0.05)
                    break
                except queue.Full:
                    continue

        log.debug("BatchScheduler prepare-worker stopped")

    def _worker(self) -> None:  # pragma: no cover
        """
        GPU thread: consume prepared batches from *_prepared_queue* and run
        the autoregressive generation loop.
        """
        import mlx.core as mx

        log.debug("BatchScheduler worker started")
        while not self._stop_event.is_set():
            # ── Consume a prepared batch ──────────────────────────────────────
            try:
                batch = self._prepared_queue.get(timeout=0.05)
            except queue.Empty:
                continue

            with self._lock:
                self.total_batches  += 1
                self.total_requests += len(batch)

            # Seed per-batch RNG from first request's seed (if any)
            seeds = [r.seed for r in batch if r.seed is not None]
            if seeds:
                self._rng = np.random.default_rng(seeds[0])

            # ── Run the generation loop for this batch ───────────────────────
            try:
                self._generate_batch(batch, mx)
            except Exception as exc:
                log.error("Batch generation error: %s", exc, exc_info=True)
                for req in batch:
                    if not req.done:
                        req.out_queue.put(("", "error"))
                        req.out_queue.put(_DONE)

        log.debug("BatchScheduler worker stopped")

    def _collect_batch(self, limit: int | None = None) -> list[_Request]:  # pragma: no cover
        """
        Block until at least one request arrives, then wait up to
        ``batch_window_ms`` for more — up to *limit* (default ``max_batch_size``).
        """
        _limit = limit if limit is not None else self._max_batch
        batch: list[_Request] = []
        deadline = None

        # Block waiting for the first request (with a short timeout so we can
        # check the stop event periodically).
        while not self._stop_event.is_set():
            try:
                req = self._pending.get(timeout=0.05)
            except queue.Empty:
                continue
            batch.append(req)
            deadline = time.perf_counter() + self._window_ms / 1000.0
            break

        if not batch:
            return batch

        # Collect additional requests until the window closes or batch is full
        while (len(batch) < _limit
               and time.perf_counter() < deadline):
            try:
                req = self._pending.get_nowait()
                batch.append(req)
            except queue.Empty:
                remaining_ms = (deadline - time.perf_counter()) * 1000
                if remaining_ms > 0:
                    time.sleep(min(remaining_ms / 1000, 0.002))

        return batch

    def _group_by_prefix(
        self, pool: list["_Request"]
    ) -> tuple[list["_Request"], list["_Request"]]:
        """Reorder *pool* to prefer batching requests with a shared 64-token prefix.

        Groups are formed by hashing the first 64 input-token IDs; the largest
        cohort is selected first to maximise KV-cache warm hits.  FIFO order is
        preserved within each group.  Requests that exceed ``max_batch_size``
        are returned as *leftovers* so the caller can re-enqueue them.

        Returns
        -------
        (selected, leftovers)
            *selected*  : up to ``max_batch_size`` requests, prefix-grouped.
            *leftovers* : remaining requests to return to the pending queue.
        """
        if len(pool) <= self._max_batch:
            return pool, []

        # Build per-prefix groups preserving original FIFO insertion order.
        groups: dict[str, list["_Request"]] = {}
        order:  list[str] = []
        for req in pool:
            key = hashlib.sha256(
                np.array(req.input_ids[:64], dtype=np.int32).tobytes()
            ).hexdigest()[:8]
            if key not in groups:
                groups[key] = []
                order.append(key)
            groups[key].append(req)

        # Sort groups by descending size — largest cohort fills the batch first.
        sorted_keys = sorted(order, key=lambda k: -len(groups[k]))

        selected:  list["_Request"] = []
        leftovers: list["_Request"] = []
        for key in sorted_keys:
            group = groups[key]
            remaining_slots = self._max_batch - len(selected)
            if remaining_slots > 0:
                selected.extend(group[:remaining_slots])
                leftovers.extend(group[remaining_slots:])
            else:
                leftovers.extend(group)

        return selected, leftovers

    def _sort_decode_first(self, batch: list["_Request"]) -> list["_Request"]:
        """Sort *batch* so decode requests (those with generated tokens) run first.

        Decode requests have ``generated_ids`` populated; prefill requests do not.
        Prioritising decode prevents GPU starvation by long prefill passes.
        """
        decode  = [r for r in batch if r.generated_ids]
        prefill = [r for r in batch if not r.generated_ids]
        return decode + prefill

    def _generate_batch(self, batch: list[_Request], mx) -> None:  # pragma: no cover
        """
        Run the autoregressive generation loop for a batch of requests.

        Each step:
        1. Build padded (B, max_len) input array from current sequence states.
        2. Forward pass → (B, max_len, vocab) logits.
        3. Extract last logit row per request → sample next token.
        4. Append token, check EOS/stop, stream to out_queue.
        5. Remove completed requests.
        """
        # We generate into each request's input_ids + generated_ids combined,
        # so the attention window always sees the full context.
        active    = list(batch)
        step      = 0
        max_steps = max(r.max_tokens for r in active)

        while active and step < max_steps:
            # ── Build padded batch ───────────────────────────────────────────
            seqs    = [r.input_ids + r.generated_ids for r in active]
            lengths = [len(s) for s in seqs]
            max_len = max(lengths)

            # Right-pad with pad_token_id
            padded = np.full((len(active), max_len), self._pad_id, dtype=np.int32)
            for i, seq in enumerate(seqs):
                padded[i, :len(seq)] = seq

            # ── Forward pass ─────────────────────────────────────────────────
            ids_batch = mx.array(padded, dtype=mx.int32)   # (B, max_len)
            logits_all = self._model(ids_batch)            # (B, max_len, vocab)
            mx.eval(logits_all)                            # materialise before numpy

            logits_np = np.array(logits_all.astype(mx.float32))  # CPU numpy

            # ── Sample + stream per request ──────────────────────────────────
            still_active: list[_Request] = []
            for i, req in enumerate(active):
                # Logits at the position of the LAST real token for this request
                last_pos      = lengths[i] - 1
                logit_row     = logits_np[i, last_pos, :]       # (vocab,)

                next_id       = _sample_token(logit_row, req.temperature,
                                              req.top_p, self._rng)
                tok_text      = self._tokenizer.decode([next_id])

                req.generated_ids.append(next_id)

                # Check EOS
                is_eos  = (next_id == self._eos_id)
                is_stop = _check_stop(req, next_id)
                is_max  = (len(req.generated_ids) >= req.max_tokens)

                with self._lock:
                    self.total_tokens_gen += 1

                if is_eos or is_stop:
                    req.out_queue.put((tok_text, "stop"))
                    req.out_queue.put(_DONE)
                    req.done = True
                elif is_max:
                    req.out_queue.put((tok_text, "length"))
                    req.out_queue.put(_DONE)
                    req.done = True
                else:
                    req.out_queue.put((tok_text, None))
                    still_active.append(req)

            active = still_active
            step  += 1

        # Any requests still running have hit the global max_steps cap
        for req in active:
            if not req.done:
                req.out_queue.put(("", "length"))
                req.out_queue.put(_DONE)
                req.done = True


# ---------------------------------------------------------------------------
# NestedWaitScheduler — continuous-batching with inter-step merge
# ---------------------------------------------------------------------------

class NestedWaitScheduler(BatchScheduler):
    """
    Nested-WAIT continuous-batching scheduler.

    Improvement over :class:`BatchScheduler` (legacy static batcher)
    ----------------------------------------------------------------
    The legacy scheduler collects up to ``max_batch_size`` requests in a
    coalescing window, processes the entire batch to completion, *then*
    picks up the next batch — leaving the GPU idle between batches.

    NestedWaitScheduler eliminates that gap by merging newly-prefilled
    requests **between decode steps**, so the GPU never waits for a full
    batch to finish before accepting new work.

    Algorithm
    ---------
    1. GPU-worker loop picks up a batch from ``_prepared_queue`` and calls
       :meth:`_generate_batch_nested`.
    2. After each decode step, the worker non-blocking-polls ``_prepared_queue``
       for more prepared requests ("Nested WAIT merge"):
       - If available, extend the active batch immediately (up to
         ``max_batch_size``; excess requests are re-queued).
       - Merge adds zero GPU idle time — it happens between kernel launches.
    3. Requests that finish early free their slot for the next merge cycle.

    The result is that throughput degrades gracefully under load (GPU
    utilisation stays ≥ 95%) while TTFT for the *first* request remains
    the same as single-request mode.

    Backward Compatibility
    ----------------------
    All :class:`BatchScheduler` public methods (``submit``, ``submit_sync``,
    ``start``, ``stop``, ``stats``) are preserved.  Use
    ``--scheduler=legacy`` to fall back to the static batcher.
    """

    def _worker(self) -> None:  # pragma: no cover
        """
        GPU thread: run continuous-batching decode with inter-step Nested WAIT merges.
        """
        import mlx.core as mx

        log.debug("NestedWaitScheduler worker started")
        while not self._stop_event.is_set():
            try:
                batch = self._prepared_queue.get(timeout=0.05)
            except queue.Empty:
                continue

            with self._lock:
                self.total_batches  += 1
                self.total_requests += len(batch)

            seeds = [r.seed for r in batch if r.seed is not None]
            if seeds:
                self._rng = np.random.default_rng(seeds[0])

            try:
                self._generate_batch_nested(batch, mx)
            except Exception as exc:
                log.error("NestedWait batch error: %s", exc, exc_info=True)
                for req in batch:
                    if not req.done:
                        req.out_queue.put(("", "error"))
                        req.out_queue.put(_DONE)

        log.debug("NestedWaitScheduler worker stopped")

    def _generate_batch_nested(  # pragma: no cover
        self, batch: list[_Request], mx
    ) -> None:
        """
        Autoregressive decode loop with per-step merge of new prepared requests.

        At each step boundary the worker non-blocking-polls ``_prepared_queue``.
        If a new prepared batch is available and there are free slots, the new
        requests are merged into ``active`` immediately — eliminating the
        inter-batch GPU idle gap characteristic of static batching.
        """
        active:   list[_Request] = list(batch)
        step:     int            = 0
        max_steps: int           = max(r.max_tokens for r in active)

        while active and step < max_steps:
            # ── Build padded batch ───────────────────────────────────────────
            seqs    = [r.input_ids + r.generated_ids for r in active]
            lengths = [len(s) for s in seqs]
            max_len = max(lengths)

            padded = np.full((len(active), max_len), self._pad_id, dtype=np.int32)
            for i, seq in enumerate(seqs):
                padded[i, :len(seq)] = seq

            # ── Forward pass ─────────────────────────────────────────────────
            ids_batch  = mx.array(padded, dtype=mx.int32)
            logits_all = self._model(ids_batch)
            mx.eval(logits_all)
            logits_np  = np.array(logits_all.astype(mx.float32))

            # ── Sample + stream per request ──────────────────────────────────
            still_active: list[_Request] = []
            for i, req in enumerate(active):
                last_pos  = lengths[i] - 1
                logit_row = logits_np[i, last_pos, :]
                next_id   = _sample_token(logit_row, req.temperature,
                                          req.top_p, self._rng)
                tok_text  = self._tokenizer.decode([next_id])

                req.generated_ids.append(next_id)

                is_eos  = (next_id == self._eos_id)
                is_stop = _check_stop(req, next_id)
                is_max  = (len(req.generated_ids) >= req.max_tokens)

                with self._lock:
                    self.total_tokens_gen += 1

                if is_eos or is_stop:
                    req.out_queue.put((tok_text, "stop"))
                    req.out_queue.put(_DONE)
                    req.done = True
                elif is_max:
                    req.out_queue.put((tok_text, "length"))
                    req.out_queue.put(_DONE)
                    req.done = True
                else:
                    req.out_queue.put((tok_text, None))
                    still_active.append(req)

            active = still_active
            step  += 1

            # ── Nested WAIT merge: pull in any newly-prepared requests ────────
            free_slots = self._max_batch - len(active)
            if free_slots > 0 and not self._stop_event.is_set():
                try:
                    new_batch = self._prepared_queue.get_nowait()
                    # Respect max_batch_size — re-queue excess requests
                    if len(new_batch) > free_slots:
                        for _excess in new_batch[free_slots:]:
                            self._pending.put(_excess)
                        new_batch = new_batch[:free_slots]
                    active.extend(new_batch)
                    # Extend max_steps if new requests need more tokens
                    if new_batch:
                        max_steps = max(max_steps,
                                        step + max(r.max_tokens for r in new_batch))
                        with self._lock:
                            self.total_requests += len(new_batch)
                        log.debug(
                            "NestedWait merged %d new request(s) at step %d "
                            "(active_batch=%d)",
                            len(new_batch), step, len(active),
                        )
                except queue.Empty:
                    pass  # no new work ready — continue with current batch

        # Drain any requests still running at the global step cap
        for req in active:
            if not req.done:
                req.out_queue.put(("", "length"))
                req.out_queue.put(_DONE)
                req.done = True



@dataclasses.dataclass
class BucketBounds:
    """Defines a single output-length bucket."""
    min_tokens: int
    max_tokens: int
    label:      str = ""

    def __post_init__(self) -> None:
        if self.min_tokens < 0:
            raise ValueError("min_tokens must be ≥ 0")
        if self.max_tokens < self.min_tokens:
            raise ValueError("max_tokens must be ≥ min_tokens")

    def contains(self, length: int) -> bool:
        return self.min_tokens <= length <= self.max_tokens


def build_default_buckets() -> list[BucketBounds]:
    """Return the default BucketServe output-length buckets."""
    return [
        BucketBounds(0,    63,   "xs"),
        BucketBounds(64,   127,  "s"),
        BucketBounds(128,  255,  "m"),
        BucketBounds(256,  511,  "l"),
        BucketBounds(512,  1023, "xl"),
        BucketBounds(1024, 4095, "xxl"),
    ]


def assign_bucket(
    predicted_length: int,
    buckets: list[BucketBounds] | None = None,
) -> BucketBounds:
    """
    Assign a request to the appropriate output-length bucket.

    Parameters
    ----------
    predicted_length : int — predicted output token count
    buckets          : list of BucketBounds (defaults to :func:`build_default_buckets`)

    Returns
    -------
    BucketBounds — the matching bucket (or the last bucket as fallback)
    """
    _buckets = buckets or build_default_buckets()
    for b in _buckets:
        if b.contains(predicted_length):
            return b
    return _buckets[-1]


# ---------------------------------------------------------------------------
# Argus — lightweight output-length predictor
# ---------------------------------------------------------------------------

class OutputLengthPredictor:
    """
    Lightweight linear regression model that predicts the output token count
    from request features.

    Based on:
      "Argus: Efficient LLM Serving via Output Length Prediction"
      (Systems for ML Workshop, NeurIPS 2024)

    Features used (all scalar, computed from the request):
      [1, prompt_length, log(prompt_length+1), task_type_id]

    where ``task_type_id`` is derived from a keyword scan of the prompt text.

    Parameters
    ----------
    default_output_length : int
        Fallback prediction when no prior data is available.
    alpha : float
        L2 regularisation factor for the online linear update.
    """

    _TASK_KEYWORDS: dict[str, int] = {
        "summarize":   0,
        "summarise":   0,
        "summary":     0,
        "compare":     1,
        "explain":     2,
        "translate":   3,
        "code":        4,
        "generate":    5,
        "list":        6,
        "answer":      7,
        "question":    7,
        "what":        7,
        "why":         7,
        "how":         7,
    }
    _N_FEATURES: int = 4   # bias, prompt_len, log_prompt_len, task_id_norm

    def __init__(
        self,
        default_output_length: int = 256,
        alpha:                 float = 1e-4,
    ) -> None:
        if default_output_length < 1:
            raise ValueError("default_output_length must be ≥ 1")
        self._default    = default_output_length
        self._alpha      = alpha
        # Weights: [w_bias, w_prompt_len, w_log_prompt_len, w_task_id]
        self._weights    = np.array([float(default_output_length), 0.0, 0.0, 0.0],
                                    dtype=np.float64)
        self._n_samples  = 0

    def _featurize(self, prompt: str) -> np.ndarray:
        """Convert a prompt string into a feature vector."""
        prompt_len     = len(prompt.split())
        log_prompt_len = float(np.log(prompt_len + 1))
        task_id        = self._detect_task(prompt)
        task_id_norm   = task_id / max(len(self._TASK_KEYWORDS), 1)
        return np.array([1.0, float(prompt_len), log_prompt_len, task_id_norm],
                        dtype=np.float64)

    @classmethod
    def _detect_task(cls, prompt: str) -> int:
        """Return an integer task-type ID based on keyword matching."""
        lower = prompt.lower()
        for kw, tid in cls._TASK_KEYWORDS.items():
            if kw in lower:
                return tid
        return 8   # "other"

    def predict(self, prompt: str) -> int:
        """
        Predict the output token count for *prompt*.

        Returns
        -------
        int — predicted token count (≥ 1)
        """
        features = self._featurize(prompt)
        raw      = float(features @ self._weights)
        return max(1, round(raw))

    def update(self, prompt: str, actual_length: int) -> None:
        """
        Online update (SGD + L2) given the observed output length.

        Parameters
        ----------
        prompt        : str — the original request text
        actual_length : int — observed number of output tokens
        """
        x        = self._featurize(prompt)
        y_hat    = float(x @ self._weights)
        error    = float(actual_length) - y_hat
        lr       = 1.0 / (self._n_samples + 1.0)
        self._weights = (
            self._weights * (1.0 - lr * self._alpha)
            + lr * error * x
        )
        self._n_samples += 1

    @property
    def n_samples(self) -> int:
        """Number of training samples seen so far."""
        return self._n_samples

    @property
    def weights(self) -> np.ndarray:
        """Current regression weights (copy)."""
        return self._weights.copy()


# ── ORCA: Iteration-level continuous batching scheduler ───────────────────────
#
# Based on:
#   "Orca: A Distributed Serving System for Transformer-Based Generative Models"
#   — Yu et al., OSDI 2022
#
# Key insight
# -----------
# Traditional LLM serving batches requests at the *request* level: all requests
# in a batch must finish before new ones can enter.  ORCA batches at the
# *iteration* level: at every single decode step, the scheduler can add new
# requests (if memory permits) or preempt overrunning ones.  This dramatically
# reduces head-of-line blocking and improves GPU utilisation.
#
# This module provides a simulation layer:
#   OrcaConfig            — token budget and preemption mode.
#   RequestState          — per-request tracking (prompt + generated so far).
#   IterationLevelScheduler — at each step, decides which requests to run.
#   SelectivePreemption   — selects which running request to preempt.

from dataclasses import dataclass as _orca_dc, field as _orca_field


@_orca_dc
class OrcaConfig:
    """Configuration for ORCA iteration-level continuous batching.

    Parameters
    ----------
    max_batch_tokens : int
        Maximum total tokens (prompt + generated so far) across all concurrently
        running requests.  New requests are admitted only when they fit.
    preemption_mode : str
        ``"swap"`` — preempted request state is swapped to CPU memory and can
        resume later.
        ``"recompute"`` — preempted request is evicted; if retried it is
        recomputed from scratch.
    max_waiting : int
        Maximum number of requests allowed in the waiting queue (0 = unlimited).
    """

    max_batch_tokens: int = 2048
    preemption_mode:  str = "swap"
    max_waiting:      int = 0

    def __post_init__(self) -> None:
        if self.max_batch_tokens < 1:
            raise ValueError("max_batch_tokens must be >= 1")
        if self.preemption_mode not in ("swap", "recompute"):
            raise ValueError("preemption_mode must be 'swap' or 'recompute'")
        if self.max_waiting < 0:
            raise ValueError("max_waiting must be >= 0")


@_orca_dc
class RequestState:
    """Track the current state of a single inference request.

    Parameters
    ----------
    request_id : str
    prompt_len : int
        Number of prompt tokens (fixed at admission).
    max_new_tokens : int
        Maximum tokens to generate.
    generated : int
        Tokens generated so far (starts at 0).
    preempted : bool
        True if this request was previously preempted (swap or recompute).
    """

    request_id:     str = ""
    prompt_len:     int = 0
    max_new_tokens: int = 128
    generated:      int = 0
    preempted:      bool = False

    @property
    def total_tokens(self) -> int:
        """Total tokens currently occupying KV cache."""
        return self.prompt_len + self.generated

    @property
    def is_finished(self) -> bool:
        return self.generated >= self.max_new_tokens


class SelectivePreemption:
    """Selects which running request to preempt when the batch is over-budget.

    Strategy: preempt the request with the *most* total tokens (longest KV
    cache footprint), as evicting it frees the most memory per preemption event.

    Parameters
    ----------
    mode : str — ``"swap"`` or ``"recompute"`` (informational; affects the
        ``preempted`` flag on the returned request).
    """

    def __init__(self, mode: str = "swap") -> None:
        if mode not in ("swap", "recompute"):
            raise ValueError("mode must be 'swap' or 'recompute'")
        self._mode = mode

    def select_victim(self, running: "list[RequestState]") -> "RequestState | None":
        """Return the request to preempt, or None if the list is empty."""
        if not running:
            return None
        victim = max(running, key=lambda r: r.total_tokens)
        return victim

    def preempt(
        self,
        victim: RequestState,
        running: "list[RequestState]",
        waiting: "list[RequestState]",
    ) -> None:
        """Remove ``victim`` from ``running`` and re-queue in ``waiting``.

        In ``'recompute'`` mode the generated count is reset to 0 (request
        starts over).  In ``'swap'`` mode progress is preserved.
        """
        running.remove(victim)
        if self._mode == "recompute":
            victim.generated  = 0
        victim.preempted = True
        waiting.insert(0, victim)   # re-queue at the front


class IterationLevelScheduler:
    """ORCA iteration-level scheduler.

    Maintains two queues:
    * ``running`` — requests currently occupying GPU memory (active decode step).
    * ``waiting`` — requests queued for admission.

    At each :meth:`step` the scheduler:
    1. Computes the current token budget consumption of all running requests.
    2. Admits waiting requests that fit within ``max_batch_tokens``.
    3. If a running request exceeds the budget (after a long prompt is added),
       invokes :class:`SelectivePreemption` to free space.
    4. Returns the current ``(to_run, to_admit, to_preempt)`` decision.

    Parameters
    ----------
    config : OrcaConfig
    """

    def __init__(self, config: OrcaConfig) -> None:
        self._cfg       = config
        self._preempt   = SelectivePreemption(config.preemption_mode)
        self._running:  list[RequestState] = []
        self._waiting:  list[RequestState] = []
        self._step_num: int                = 0

    @property
    def running(self) -> "list[RequestState]":
        return list(self._running)

    @property
    def waiting(self) -> "list[RequestState]":
        return list(self._waiting)

    def add_request(self, req: RequestState) -> None:
        """Enqueue a new request.  Rejects if waiting queue is full."""
        if self._cfg.max_waiting > 0 and len(self._waiting) >= self._cfg.max_waiting:
            raise RuntimeError(
                f"Waiting queue full ({self._cfg.max_waiting}); "
                f"cannot admit request {req.request_id!r}"
            )
        self._waiting.append(req)

    def _token_budget_used(self) -> int:
        return sum(r.total_tokens for r in self._running)

    def step(self) -> "tuple[list[RequestState], list[RequestState], list[RequestState]]":
        """Execute one iteration-level scheduling step.

        Returns
        -------
        ``(to_run, newly_admitted, preempted)``

        * ``to_run`` — all requests that should receive one decode step this
          iteration (running after admission/preemption).
        * ``newly_admitted`` — requests moved from waiting → running this step.
        * ``preempted`` — requests moved from running → waiting this step.
        """
        self._step_num += 1
        newly_admitted: list[RequestState] = []
        preempted:      list[RequestState] = []

        # Admit waiting requests that fit within the budget
        i = 0
        while i < len(self._waiting):
            req    = self._waiting[i]
            needed = req.total_tokens
            if self._token_budget_used() + needed <= self._cfg.max_batch_tokens:
                self._running.append(req)
                self._waiting.pop(i)
                newly_admitted.append(req)
            else:
                i += 1

        # If over budget, preempt until we fit
        while self._token_budget_used() > self._cfg.max_batch_tokens and self._running:
            victim = self._preempt.select_victim(self._running)
            if victim is None:  # pragma: no cover
                break
            preempted.append(victim)
            self._preempt.preempt(victim, self._running, self._waiting)

        # Mark finished requests for removal (they consumed their last step)
        finished = [r for r in self._running if r.is_finished]
        for r in finished:
            self._running.remove(r)

        to_run = list(self._running)
        return to_run, newly_admitted, preempted

    def tick(self, tokens_per_request: int = 1) -> None:
        """Advance all running requests by ``tokens_per_request`` generated tokens."""
        for r in self._running:
            r.generated = min(r.generated + tokens_per_request, r.max_new_tokens)

    @property
    def step_number(self) -> int:
        return self._step_num


# ---------------------------------------------------------------------------
# Standalone test / benchmark
# ---------------------------------------------------------------------------

def _demo():  # pragma: no cover
    """
    Quick demo without a real model — uses a toy identity model.

    Run:  python3 -m squish.scheduler
    """
    print("BatchScheduler — standalone demo (toy model, no GPU)\n")

    # Toy model that returns random logits
    class _ToyModel:
        def __call__(self, x):
            import mlx.core as mx
            B, T = x.shape
            return mx.array(
                np.random.randn(B, T, 1000).astype(np.float32)
            )

    class _ToyTokenizer:
        eos_token_id = 2
        pad_token_id = 0
        def encode(self, text, add_special_tokens=True):
            return [1, 1, 1, 1, 1][:5]
        def decode(self, ids):
            chars = "abcdefghijklmnopqrstuvwxyz "
            return "".join(chars[i % len(chars)] for i in ids)

    try:
        import mlx.core  # noqa: F401
    except ImportError:
        print("mlx not available — skipping demo (install mlx to test)")
        return

    scheduler = BatchScheduler(
        _ToyModel(), _ToyTokenizer(),
        max_batch_size=4, batch_window_ms=50
    )
    scheduler.start()

    results   = {}
    threads   = []
    prompts   = ["Hello", "What is 2+2?", "Explain gravity", "Tell me a joke"]

    def _run(p_idx, prompt):
        tokens = []
        for tok, fin in scheduler.submit_sync(prompt, max_tokens=10,
                                              temperature=0.7):
            tokens.append(tok)
            if fin:
                break
        results[p_idx] = "".join(tokens)

    for i, p in enumerate(prompts):
        t = threading.Thread(target=_run, args=(i, p))
        t.start()
        threads.append(t)

    for t in threads:
        t.join()

    for i, p in enumerate(prompts):
        print(f"  [{p[:20]}]  →  {results.get(i, '?')!r}")

    print(f"\nStats: {scheduler.stats()}")
    scheduler.stop()
    print("\nDemo complete.")


if __name__ == "__main__":
    import numpy as np  # ensure available for _demo
    logging.basicConfig(level=logging.INFO)
    _demo()
