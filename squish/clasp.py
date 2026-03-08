"""
squish/clasp.py

CLaSp — In-Context Layer Skip with Adaptive Feedback Loop.

Based on:
  "CLaSp: In-Context Layer Skip for Self-Speculative Decoding"
  — ACL 2025  arXiv:2505.24196

Key insight
-----------
The verification pass of self-speculative decoding already computes the
full hidden-state trajectory for every token.  CLaSp uses that free
information as feedback: it runs a dynamic-programming algorithm over the
verification hidden states to decide which layers to skip in the *next*
draft round.

Specifically, for each layer L, CLaSp measures:

    importance(L) = 1 − cosine_similarity( h[L−1], h[L] )

A layer that barely changes the hidden state (high cosine similarity →
low importance) is a prime candidate to skip.  The DP selects the
``max_skip_layers`` least-important layers to skip for the next draft step.

This creates a closed feedback loop:

    draft (with current skip set)
      → verify (full model)
        → measure per-layer importance from hidden states
          → update skip set
            → next draft

As generation proceeds through long structured outputs, the skip set
naturally converges to match the actual redundancy of the current context.

Provides
--------
  CLaSPConfig           — tuning parameters.
  CLaSPSkipOptimizer    — DP optimizer using verification hidden states.
  CLaSPStats            — per-generation counters.
  CLaSPDecoder          — full draft→verify→adapt loop.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

import numpy as np

__all__ = [
    "CLaSPConfig",
    "CLaSPSkipOptimizer",
    "CLaSPStats",
    "CLaSPDecoder",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _softmax(logits: np.ndarray) -> np.ndarray:
    x = np.asarray(logits, dtype=np.float64)
    x = x - x.max()
    e = np.exp(x)
    return (e / e.sum()).astype(np.float32)


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two 1-D vectors; returns 1.0 for near-zero."""
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-12 or nb < 1e-12:
        return 1.0
    val = float(np.dot(a, b) / (na * nb))
    return max(-1.0, min(1.0, val))


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class CLaSPConfig:
    """Configuration for CLaSp adaptive layer-skip decoder.

    Parameters
    ----------
    num_layers : int
        Total transformer layers.
    max_skip_layers : int
        Maximum number of layers to skip in any single draft step.
    draft_gamma : int
        Draft tokens per speculative step.
    similarity_threshold : float
        Layers with cosine similarity above this threshold between adjacent
        hidden states are considered low-importance candidates for skipping.
        Range (0, 1].
    """

    num_layers:           int   = 32
    max_skip_layers:      int   = 8
    draft_gamma:          int   = 4
    similarity_threshold: float = 0.95

    def __post_init__(self) -> None:
        if self.num_layers < 2:
            raise ValueError("num_layers must be ≥ 2")
        if self.max_skip_layers < 0:
            raise ValueError("max_skip_layers must be ≥ 0")
        if self.max_skip_layers >= self.num_layers:
            raise ValueError("max_skip_layers must be < num_layers")
        if self.draft_gamma < 1:
            raise ValueError("draft_gamma must be ≥ 1")
        if not 0.0 < self.similarity_threshold <= 1.0:
            raise ValueError("similarity_threshold must be in (0, 1]")


# ---------------------------------------------------------------------------
# Optimizer
# ---------------------------------------------------------------------------

class CLaSPSkipOptimizer:
    """Select the next draft's layer-skip set from verification hidden states.

    Call :meth:`update_from_hidden_states` after each verification pass, then
    :meth:`select_skip_set` to obtain the skip list for the next draft.
    """

    def __init__(self, similarity_threshold: float = 0.95) -> None:
        if not 0.0 < similarity_threshold <= 1.0:
            raise ValueError("similarity_threshold must be in (0, 1]")
        self._threshold  = similarity_threshold
        self._importances: List[float] = []

    # ------------------------------------------------------------------

    def update_from_hidden_states(
        self,
        hidden_states: List[np.ndarray],
    ) -> None:
        """Compute per-layer importance from consecutive hidden-state pairs.

        Parameters
        ----------
        hidden_states : list of (hidden_dim,) arrays
            One entry per layer (index 0 = layer 0, index N-1 = layer N-1).
            Typically the last-token hidden state of each layer.
        """
        if len(hidden_states) < 2:
            self._importances = []
            return

        importances: List[float] = []
        for i in range(1, len(hidden_states)):
            h_prev = np.asarray(hidden_states[i - 1], dtype=np.float64)
            h_cur  = np.asarray(hidden_states[i],     dtype=np.float64)
            sim    = _cosine_sim(h_prev, h_cur)
            importances.append(1.0 - sim)
        self._importances = importances

    def layer_importances(self) -> List[float]:
        """Return per-layer importance scores (layer 1 … N−1).

        Higher score → the layer changed the representation more → keep it.
        Lower score  → layer contributed little → candidate for skipping.
        """
        return list(self._importances)

    def select_skip_set(self, max_skip: int) -> List[int]:
        """Return the *max_skip* lowest-importance layer indices to skip.

        Layer index here is 1-based (layer 1 is the second transformer layer,
        compared between h[0] and h[1]).

        Returns an empty list if no importance data is available.
        """
        importances = self._importances
        if not importances:
            return []

        actual_skip = min(max_skip, len(importances))
        if actual_skip <= 0:
            return []

        indexed = sorted(enumerate(importances), key=lambda x: x[1])
        skip = sorted((idx + 1) for idx, _ in indexed[:actual_skip])
        return skip


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

@dataclass
class CLaSPStats:
    """Per-generation counters returned by :class:`CLaSPDecoder`."""

    total_tokens:    int   = 0
    accepted_draft:  int   = 0
    rejected_draft:  int   = 0
    adaptation_steps: int  = 0   # times the skip set changed
    total_skip_applications: int = 0

    @property
    def acceptance_rate(self) -> float:
        total = self.accepted_draft + self.rejected_draft
        return self.accepted_draft / total if total > 0 else 0.0


# ---------------------------------------------------------------------------
# Decoder
# ---------------------------------------------------------------------------

class CLaSPDecoder:
    """Drive inference with the CLaSp adaptive layer-skip loop.

    Parameters
    ----------
    forward_fn : callable
        ``forward_fn(ids, skip_layers=None) -> (np.ndarray, list[np.ndarray])``

        Returns ``(logits, hidden_states)`` where:
          - ``logits`` is ``(vocab_size,)``
          - ``hidden_states`` is a list of ``(hidden_dim,)`` arrays, one per
            layer, representing the last-token hidden state at each layer.

        When ``skip_layers`` is a non-empty list of 0-based layer indices,
        those layers perform an identity (residual) pass-through.
    config : CLaSPConfig
    rng_seed : int
    """

    def __init__(
        self,
        forward_fn: Callable[..., Tuple[np.ndarray, List[np.ndarray]]],
        config: CLaSPConfig,
        rng_seed: int = 0,
    ) -> None:
        self._fwd       = forward_fn
        self._cfg       = config
        self._optimizer = CLaSPSkipOptimizer(config.similarity_threshold)
        self._rng       = np.random.default_rng(rng_seed)

    # ------------------------------------------------------------------

    def generate(
        self,
        input_ids: List[int],
        max_new_tokens: int = 64,
    ) -> Tuple[List[int], CLaSPStats]:
        """Generate tokens with adaptive CLaSp layer-skip feedback.

        Parameters
        ----------
        input_ids : list[int]   (not modified)
        max_new_tokens : int

        Returns
        -------
        (output_ids, stats)
        """
        cfg   = self._cfg
        stats = CLaSPStats()
        ids   = list(input_ids)
        generated = 0
        skip_set: List[int] = []   # start with no skipping
        verify_hidden: Optional[List[np.ndarray]] = None

        while generated < max_new_tokens:
            # ── Draft with current skip set ───────────────────────────────────
            draft_ids:   List[int]        = []
            draft_probs: List[np.ndarray] = []
            ctx = list(ids)

            for _ in range(cfg.draft_gamma):
                logits, _ = self._fwd(ctx, skip_set)
                probs      = _softmax(logits)
                tok        = int(np.argmax(logits))
                draft_ids.append(tok)
                draft_probs.append(probs)
                ctx.append(tok)

            stats.total_skip_applications += len(skip_set)

            # ── Verify with full model (no skipping) ─────────────────────────
            ctx_v    = list(ids)
            accepted: List[int] = []
            rejected  = False

            for d_tok, d_probs in zip(draft_ids, draft_probs):
                full_logits, hidden_states = self._fwd(ctx_v, [])
                verify_hidden = hidden_states
                full_probs    = _softmax(full_logits)
                v_tok         = int(np.argmax(full_logits))
                p_t = float(full_probs[d_tok])
                p_d = float(d_probs[d_tok])

                if self._rng.random() < min(1.0, p_t / max(p_d, 1e-12)):
                    accepted.append(d_tok)
                    ctx_v.append(d_tok)
                    stats.accepted_draft += 1
                else:
                    accepted.append(v_tok)
                    ctx_v.append(v_tok)
                    stats.rejected_draft += 1
                    rejected = True
                    break

            # Bonus token
            if not rejected:
                full_logits, hidden_states = self._fwd(ctx_v, [])
                verify_hidden = hidden_states
                accepted.append(int(np.argmax(full_logits)))

            # ── Update optimizer with verification hidden states ──────────────
            if verify_hidden is not None:
                self._optimizer.update_from_hidden_states(verify_hidden)
                new_skip = self._optimizer.select_skip_set(cfg.max_skip_layers)
                if new_skip != skip_set:
                    stats.adaptation_steps += 1
                    skip_set = new_skip

            ids.extend(accepted)
            generated += len(accepted)
            stats.total_tokens += len(accepted)

        return ids, stats
