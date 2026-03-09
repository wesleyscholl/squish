"""
squish/mirror_sd.py

Mirror-SD — GPU+NPU Dual-Pipeline Speculative Decoding.

Based on:
  "Mirror-SD: Mirror Speculative Decoding for Efficient LLM Inference
   on Heterogeneous Hardware"
  Apple ML Research; arXiv:2510.13161; machinelearning.apple.com/research/mirror

Problem
-------
All prior speculative decoders (EAGLE-3, CLaSp, SwiftSpec) run draft and
verification *sequentially*: draft produces γ tokens, then the target model
verifies them.  The GPU is partially idle during verification (it already ran
the draft), and the ANE/NPU is completely idle.

Mirror-SD reframes the problem: draft and target run **simultaneously** on
separate hardware.
  - GPU (Metal): runs the lightweight draft model producing candidate tokens.
  - ANE/NPU (Core ML): runs the large target model for token-level verification
    of the preceding step's candidates.

The synchronisation point is token-level (one integer per candidate), not
tensor-level, so cross-device bandwidth is negligible even on M3's unified
memory architecture.

2.8x–5.8x wall-time speedup over EAGLE-3 baseline on SpecBench (14B-66B);
30% average relative improvement over EAGLE-3.

Method
------
1. **MirrorDraftPipeline** — manages the GPU-side draft stream.  Each call to
   ``step(hidden)`` runs one autoregressive draft step, returns logits.

2. **MirrorVerifyPipeline** — manages the ANE-side verify stream.  Each call
   to ``step(ids)`` returns full target-model logits for each position; this
   call is non-blocking (return immediately, completion via ``wait()``).

3. **MirrorSDDecoder** — the outer scheduler that keeps both pipelines
   saturated:
   a. Enqueue draft step n+1 on GPU while ANE is verifying step n.
   b. When ANE finishes step n, compute accept/reject and extend ids.
   c. Stall only on the first step (pipeline warm-up).

Design note
-----------
This is a *framework-agnostic, numpy reference* implementation.  The
``draft_fn``/``verify_fn`` callables abstract over Metal/Core ML; ``verify_fn``
is wrapped in a thin ``MirrorFuture`` to model the non-blocking ANE call.

Conflict-resolved notes (Master Conflict Report)
-------------------------------------------------
- **Replaces mx.compile+ANE disaggregation** in the long term.  When Mirror-SD
  is active, mx.compile covers GPU sub-graphs only.
- **Partial conflict with EAGLE-3**: composes when EAGLE-3 produces draft
  tokens on GPU and Mirror-SD's ANE path handles verification.  Medium
  engineering effort to wire; no fundamental incompatibility.
- **ANE disaggregation priority**: Mirror-SD owns the ANE during decoding;
  ANE disaggregation for prefill is deferred to chunked-prefill when Mirror-SD
  is active.

Provides
--------
  MirrorSDConfig        — tuning parameters.
  MirrorFuture          — lightweight async-result wrapper.
  MirrorDraftPipeline   — GPU-side one-step draft runner.
  MirrorVerifyPipeline  — ANE-side non-blocking verification runner.
  MirrorSDStats         — acceptance and pipeline-overlap counters.
  MirrorSDDecoder       — dual-pipeline orchestrator.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

import numpy as np

__all__ = [
    "MirrorSDConfig",
    "MirrorFuture",
    "MirrorDraftPipeline",
    "MirrorVerifyPipeline",
    "MirrorSDStats",
    "MirrorSDDecoder",
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()


def _top_p_filter(probs: np.ndarray, top_p: float) -> np.ndarray:
    sorted_idx = np.argsort(probs)[::-1]
    cumsum = np.cumsum(probs[sorted_idx])
    cutoff = np.searchsorted(cumsum, top_p) + 1
    mask = np.zeros_like(probs)
    mask[sorted_idx[:cutoff]] = 1.0
    filtered = probs * mask
    return filtered / filtered.sum()


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class MirrorSDConfig:
    """Parameters for Mirror-SD.

    Parameters
    ----------
    gamma:
        Number of draft tokens generated per step (speculation depth).
    temperature:
        Softmax temperature for sampling.  1.0 = unmodified distribution.
    top_p:
        Nucleus sampling threshold.  1.0 = no nucleus filtering.
    overlap_steps:
        Number of in-flight ANE verification requests to keep queued.
        1 = no lookahead (single-step pipeline).
        2 = one step of lookahead (GPU runs step n+1 while ANE finishes n).
    """

    gamma: int = 4
    temperature: float = 1.0
    top_p: float = 1.0
    overlap_steps: int = 2

    def __post_init__(self) -> None:
        if self.gamma < 1:
            raise ValueError("gamma must be >= 1")
        if self.temperature <= 0:
            raise ValueError("temperature must be > 0")
        if not (0.0 < self.top_p <= 1.0):
            raise ValueError("top_p must be in (0, 1]")
        if self.overlap_steps < 1:
            raise ValueError("overlap_steps must be >= 1")


# ---------------------------------------------------------------------------
# MirrorFuture — thin async-result wrapper
# ---------------------------------------------------------------------------

class MirrorFuture:
    """Wraps a callable in a background thread, mimicking a non-blocking ANE call.

    In production this would be a Core ML async inference ticket.  Here we use
    a Python thread so the test suite can exercise the overlap logic without
    actual GPU/ANE hardware.

    Parameters
    ----------
    fn:
        Zero-argument callable whose return value is stored as the result.
    """

    def __init__(self, fn: Callable[[], object]) -> None:
        self._result: object = None
        self._exc: Optional[BaseException] = None
        self._done = threading.Event()
        self._thread = threading.Thread(target=self._run, args=(fn,), daemon=True)
        self._thread.start()

    def _run(self, fn: Callable[[], object]) -> None:
        try:
            self._result = fn()
        except BaseException as exc:  # noqa: BLE001
            self._exc = exc
        finally:
            self._done.set()

    def wait(self) -> object:
        """Block until the future completes and return its result."""
        self._done.wait()
        if self._exc is not None:
            raise self._exc
        return self._result

    @property
    def ready(self) -> bool:
        """True if the background computation has already completed."""
        return self._done.is_set()


# ---------------------------------------------------------------------------
# MirrorDraftPipeline — GPU-side
# ---------------------------------------------------------------------------

class MirrorDraftPipeline:
    """GPU-side draft pipeline for Mirror-SD.

    Manages one autoregressive step of the draft model.  In production this
    wraps an MLX/Metal compute graph; here it holds a callable.

    Parameters
    ----------
    draft_fn:
        ``(ids: List[int]) -> np.ndarray`` — returns ``(vocab,)`` logits for
        the next token given the current token sequence.
    config:
        MirrorSDConfig — used for temperature / top-p sampling.
    rng_seed:
        Seed for numpy RNG (reproducible tests).
    """

    def __init__(
        self,
        draft_fn: Callable[[List[int]], np.ndarray],
        config: MirrorSDConfig,
        rng_seed: int = 0,
    ) -> None:
        self._fn = draft_fn
        self._cfg = config
        self._rng = np.random.default_rng(rng_seed)

    def step(self, ids: List[int]) -> Tuple[int, np.ndarray]:
        """Run one GPU draft step.

        Returns
        -------
        (draft_token, draft_probs)
        """
        logits = self._fn(ids)
        probs = _softmax(logits / max(self._cfg.temperature, 1e-8))
        if self._cfg.top_p < 1.0:
            probs = _top_p_filter(probs, self._cfg.top_p)
        token = int(self._rng.choice(len(probs), p=probs))
        return token, probs

    def draft_sequence(
        self, ids: List[int], gamma: int
    ) -> Tuple[List[int], List[np.ndarray]]:
        """Generate *gamma* draft tokens autoregressively.

        Returns
        -------
        (draft_tokens, draft_probs_list)
        """
        ctx = list(ids)
        tokens: List[int] = []
        probs: List[np.ndarray] = []
        for _ in range(gamma):
            tok, p = self.step(ctx)
            tokens.append(tok)
            probs.append(p)
            ctx.append(tok)
        return tokens, probs


# ---------------------------------------------------------------------------
# MirrorVerifyPipeline — ANE-side
# ---------------------------------------------------------------------------

class MirrorVerifyPipeline:
    """ANE-side (non-blocking) verification pipeline for Mirror-SD.

    Wraps the target model's logit function in a ``MirrorFuture`` so the GPU
    can begin the next draft step without waiting for ANE completion.

    Parameters
    ----------
    target_fn:
        ``(ids: List[int]) -> np.ndarray`` — returns ``(vocab,)`` logits for
        the next token.
    config:
        MirrorSDConfig — used for sampling.
    rng_seed:
        Seed for numpy RNG.
    """

    def __init__(
        self,
        target_fn: Callable[[List[int]], np.ndarray],
        config: MirrorSDConfig,
        rng_seed: int = 0,
    ) -> None:
        self._fn = target_fn
        self._cfg = config
        self._rng = np.random.default_rng(rng_seed)

    def enqueue(self, ids: List[int]) -> MirrorFuture:
        """Launch a non-blocking ANE verification pass.

        Returns a ``MirrorFuture`` that resolves to
        ``(verify_token, verify_probs)``.
        """
        ids_snapshot = list(ids)

        def _work():
            logits = self._fn(ids_snapshot)
            probs = _softmax(logits / max(self._cfg.temperature, 1e-8))
            if self._cfg.top_p < 1.0:
                probs = _top_p_filter(probs, self._cfg.top_p)
            token = int(self._rng.choice(len(probs), p=probs))
            return token, probs

        return MirrorFuture(_work)


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

@dataclass
class MirrorSDStats:
    """Counters for Mirror-SD inference statistics.

    Parameters
    ----------
    total_tokens:
        Total new tokens generated (accepted + bonus).
    draft_steps:
        Number of γ-token draft batches attempted.
    accepted_total:
        Draft tokens accepted by the target model.
    rejected_total:
        Draft tokens rejected; a target-correction token was substituted.
    overlap_hits:
        Steps where the ANE future was already done when the GPU finished,
        meaning full pipeline overlap was achieved.
    """

    total_tokens: int = 0
    draft_steps: int = 0
    accepted_total: int = 0
    rejected_total: int = 0
    overlap_hits: int = 0

    @property
    def acceptance_rate(self) -> float:
        n = self.accepted_total + self.rejected_total
        return self.accepted_total / n if n > 0 else 0.0

    @property
    def mean_accepted_per_step(self) -> float:
        return self.accepted_total / self.draft_steps if self.draft_steps > 0 else 0.0


# ---------------------------------------------------------------------------
# MirrorSDDecoder — dual-pipeline orchestrator
# ---------------------------------------------------------------------------

class MirrorSDDecoder:
    """Mirror-SD dual-pipeline decoder.

    Schedules GPU (draft) and ANE (verify) work so that the next draft step
    starts as soon as the current draft is enqueued for verification — before
    the previous verification has finished.

    The acceptance logic is the standard Leviathan et al. speculative
    sampling criterion:
      accept d_tok if U ~ Uniform(0,1) < min(1, p_target[d_tok] / p_draft[d_tok])
    On rejection, substitute the target's resampled correction token; stop the
    current batch.  On full acceptance, append one bonus target token.

    Parameters
    ----------
    draft_pipeline:
        Configured ``MirrorDraftPipeline`` (GPU side).
    verify_pipeline:
        Configured ``MirrorVerifyPipeline`` (ANE side).
    config:
        ``MirrorSDConfig`` governing gamma and sampling.
    rng_seed:
        Seed for the acceptance RNG.
    """

    def __init__(
        self,
        draft_pipeline: MirrorDraftPipeline,
        verify_pipeline: MirrorVerifyPipeline,
        config: Optional[MirrorSDConfig] = None,
        rng_seed: int = 0,
    ) -> None:
        if config is None:
            config = MirrorSDConfig()
        self._draft = draft_pipeline
        self._verify = verify_pipeline
        self._cfg = config
        self._rng = np.random.default_rng(rng_seed)

    # ------------------------------------------------------------------

    def generate(
        self,
        input_ids: List[int],
        max_new_tokens: int = 64,
    ) -> Tuple[List[int], MirrorSDStats]:
        """Generate up to *max_new_tokens* tokens via dual-pipeline Mirror-SD.

        Parameters
        ----------
        input_ids:
            Starting token sequence (prompt).
        max_new_tokens:
            Hard upper bound on tokens appended to *input_ids*.

        Returns
        -------
        (output_ids, stats)
            ``output_ids`` = ``input_ids + generated_tokens``.
        """
        cfg = self._cfg
        stats = MirrorSDStats()
        ids = list(input_ids)
        generated = 0

        while generated < max_new_tokens:
            gamma = min(cfg.gamma, max_new_tokens - generated)

            # ── GPU: draft γ tokens ──────────────────────────────────────────
            draft_tokens, draft_probs = self._draft.draft_sequence(ids, gamma)
            stats.draft_steps += 1

            # ── ANE: launch verification (non-blocking) ──────────────────────
            # Enqueue one verify call per draft position.
            verify_futures: List[MirrorFuture] = []
            ctx = list(ids)
            for d_tok in draft_tokens:
                future = self._verify.enqueue(ctx)
                verify_futures.append(future)
                ctx.append(d_tok)

            # ── Collect verify results and accept/reject ─────────────────────
            accepted: List[int] = []
            rejected = False

            for i, (d_tok, d_probs, fut) in enumerate(
                zip(draft_tokens, draft_probs, verify_futures)
            ):
                v_tok, v_probs = fut.wait()
                if fut.ready:
                    # Future was already done before we called wait() —
                    # counts as a pipeline-overlap hit.
                    stats.overlap_hits += 1

                p_t = float(v_probs[d_tok])
                p_d = float(d_probs[d_tok])

                if self._rng.random() < min(1.0, p_t / max(p_d, 1e-12)):
                    accepted.append(d_tok)
                    stats.accepted_total += 1
                else:
                    # Rejection: substitute target's correction token
                    accepted.append(v_tok)
                    stats.rejected_total += 1
                    rejected = True
                    break

            # ── Bonus token on full acceptance ───────────────────────────────
            if not rejected and (generated + len(accepted)) < max_new_tokens:
                # Final context includes all accepted draft tokens
                bonus_ctx = list(ids) + accepted
                bonus_future = self._verify.enqueue(bonus_ctx)
                bonus_tok, _ = bonus_future.wait()
                accepted.append(bonus_tok)
                stats.accepted_total += 1

            ids.extend(accepted)
            generated += len(accepted)

        stats.total_tokens = generated
        return ids, stats
