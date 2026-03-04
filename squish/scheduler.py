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
        """Start the background worker thread."""
        if self._thread is not None and self._thread.is_alive():
            return self
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._worker, daemon=True,
                                        name="squish-batch-worker")
        self._thread.start()
        log.info("BatchScheduler started  (max_batch=%d  window=%.0fms)",
                 self._max_batch, self._window_ms)
        return self

    def stop(self, timeout: float = 5.0) -> None:
        """Signal the worker to stop and wait for it to finish."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
        self._thread = None

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    # ── Request submission ────────────────────────────────────────────────────

    def _make_request(
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

    def submit_sync(
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

    async def submit(
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
                "max_batch_size":   self._max_batch,
                "batch_window_ms":  self._window_ms,
            }

    # ── Worker ────────────────────────────────────────────────────────────────

    def _worker(self) -> None:
        """
        Background thread: drain pending queue → batch → generate → stream output.
        """
        import mlx.core as mx

        log.debug("BatchScheduler worker started")
        while not self._stop_event.is_set():
            # ── Collect a batch ──────────────────────────────────────────────
            batch = self._collect_batch()
            if not batch:
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

    def _collect_batch(self) -> list[_Request]:
        """
        Block until at least one request arrives, then wait up to
        ``batch_window_ms`` for more — up to ``max_batch_size``.
        """
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
        while (len(batch) < self._max_batch
               and time.perf_counter() < deadline):
            try:
                req = self._pending.get_nowait()
                batch.append(req)
            except queue.Empty:
                remaining_ms = (deadline - time.perf_counter()) * 1000
                if remaining_ms > 0:
                    time.sleep(min(remaining_ms / 1000, 0.002))

        return batch

    def _generate_batch(self, batch: list[_Request], mx) -> None:
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
# Standalone test / benchmark
# ---------------------------------------------------------------------------

def _demo():
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
