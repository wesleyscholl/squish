"""
squish/del_decoder.py

DEL — Dynamic Exit Layer for Self-Speculative Decoding.

Based on:
  "DEL: Dynamic Exit Layer for Self-Speculative Decoding"
  — Presented at COLM 2025, arXiv:2504.05598
  GitHub: hoenza/DEL

Key innovations
---------------
Standard self-speculative decoding (LayerSkip / SWIFT) picks ONE fixed exit
layer and ONE fixed draft length (gamma) before inference begins.  DEL makes
both adaptive *per decode step*:

1. **Shadow Token Analysis** — using hidden states already computed by the
   previous verification pass, DEL estimates which exit layer will yield the
   best acceptance rate for the *current* decoding position.  No extra
   forward passes.

2. **Token-per-Layer (TPL) metric** — balances draft acceptance rate with
   compute cost:  ``TPL(L) = acceptance_rate(L) / (L / num_layers)``.
   Maximising TPL keeps the fastest-verifying layers active.

3. **Dynamic Draft Exiting** — the draft step stops early once the exit-layer
   confidence drops below ``confidence_threshold``, ensuring we never waste
   forward passes on tokens the verifier will reject anyway.

Provides
--------
  DELConfig     — model and tuning parameters.
  DELStats      — per-generation counters.
  DELDecoder    — drives the full draft → verify loop.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

import numpy as np

__all__ = [
    "DELConfig",
    "DELStats",
    "DELDecoder",
]


# ---------------------------------------------------------------------------
# Helpers (module-private)
# ---------------------------------------------------------------------------

def _softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically stable softmax over a 1-D float array."""
    x = np.asarray(logits, dtype=np.float64)
    x = x - x.max()
    e = np.exp(x)
    return (e / e.sum()).astype(np.float32)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class DELConfig:
    """Parameters for the DEL decoder.

    Parameters
    ----------
    num_layers : int
        Total transformer layers N.
    min_exit_layer : int
        Earliest candidate exit layer (≥ 1).
    max_exit_layer : int
        Latest candidate exit layer (≤ num_layers).
    gamma : int
        Maximum draft tokens per step.
    confidence_threshold : float
        Confidence (max-probability) below which dynamic draft exiting fires.
        Range (0, 1].
    """

    num_layers:            int   = 32
    min_exit_layer:        int   = 8
    max_exit_layer:        int   = 24
    gamma:                 int   = 5
    confidence_threshold:  float = 0.5

    def __post_init__(self) -> None:
        if self.num_layers < 2:
            raise ValueError("num_layers must be ≥ 2")
        if self.min_exit_layer < 1:
            raise ValueError("min_exit_layer must be ≥ 1")
        if self.max_exit_layer > self.num_layers:
            raise ValueError("max_exit_layer must be ≤ num_layers")
        if self.min_exit_layer > self.max_exit_layer:
            raise ValueError("min_exit_layer must be ≤ max_exit_layer")
        if self.gamma < 1:
            raise ValueError("gamma must be ≥ 1")
        if not 0.0 < self.confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be in (0, 1]")


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

@dataclass
class DELStats:
    """Per-generation counters returned by :class:`DELDecoder`."""

    total_tokens:        int = 0
    early_exits:         int = 0   # draft rounds that stopped before gamma
    shadow_analyses:     int = 0   # shadow-analysis calls
    accepted_draft:      int = 0
    rejected_draft:      int = 0
    exit_layer_counts:   dict = field(default_factory=dict)

    @property
    def acceptance_rate(self) -> float:
        total = self.accepted_draft + self.rejected_draft
        return self.accepted_draft / total if total > 0 else 0.0


# ---------------------------------------------------------------------------
# Decoder
# ---------------------------------------------------------------------------

class DELDecoder:
    """Drive inference with DEL dynamic exit-layer selection.

    Parameters
    ----------
    forward_fn : callable
        Signature: ``forward_fn(ids: list[int], layer_limit: int | None) -> np.ndarray``
        Returns ``(vocab_size,)`` logits.  When ``layer_limit`` is an integer
        ``L``, the model exits after layer ``L``; when ``None`` it runs all
        layers (full verification pass).
    config : DELConfig
    rng_seed : int, optional
        Seed for the accept/reject random draws.
    """

    def __init__(
        self,
        forward_fn: Callable[[List[int], Optional[int]], np.ndarray],
        config: DELConfig,
        rng_seed: int = 0,
    ) -> None:
        self._fwd  = forward_fn
        self._cfg  = config
        self._rng  = np.random.default_rng(rng_seed)
        # Running acceptance stats per exit layer (for TPL)
        self._layer_accepts: dict[int, int] = {}
        self._layer_rejects: dict[int, int] = {}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _shadow_analysis(self, ids: List[int]) -> int:
        """Run each candidate exit layer and return the one with best TPL.

        In a production system this is zero-overhead because intermediate
        hidden states are cached from the previous verify pass.  Here we call
        ``forward_fn`` once per candidate layer to simulate the same outcome.
        """
        cfg = self._cfg
        best_tpl:   float = -1.0
        best_layer: int   = cfg.min_exit_layer

        for L in range(cfg.min_exit_layer, cfg.max_exit_layer + 1):
            logits     = self._fwd(ids, L)
            probs      = _softmax(logits)
            confidence = float(probs.max())
            cost_frac  = L / cfg.num_layers
            tpl        = confidence / max(cost_frac, 1e-12)
            if tpl > best_tpl:
                best_tpl   = tpl
                best_layer = L

        return best_layer

    def _select_tpl_layer(self) -> int:
        """Return exit layer with best historical TPL metric.

        Falls back to ``min_exit_layer`` when no history exists.
        """
        cfg = self._cfg
        best_tpl:   float = -1.0
        best_layer: int   = cfg.min_exit_layer

        for L in range(cfg.min_exit_layer, cfg.max_exit_layer + 1):
            n_acc = self._layer_accepts.get(L, 0)
            n_rej = self._layer_rejects.get(L, 0)
            total       = n_acc + n_rej
            ar          = n_acc / total if total > 0 else 0.5  # uninformed prior
            cost_frac   = L / cfg.num_layers
            tpl         = ar / max(cost_frac, 1e-12)
            if tpl > best_tpl:
                best_tpl   = tpl
                best_layer = L

        return best_layer

    def _draft_dynamic(
        self,
        ids: List[int],
        exit_layer: int,
    ) -> Tuple[List[int], List[np.ndarray]]:
        """Draft up to *gamma* tokens, stopping early on low confidence.

        Returns
        -------
        (draft_ids, draft_probs) — token ids and corresponding probability
        distributions.  May be shorter than *gamma* if dynamic exit fires.
        """
        cfg       = self._cfg
        draft_ids:   List[int]        = []
        draft_probs: List[np.ndarray] = []
        ctx = list(ids)

        for _ in range(cfg.gamma):
            logits     = self._fwd(ctx, exit_layer)
            probs      = _softmax(logits)
            confidence = float(probs.max())
            tok        = int(np.argmax(logits))
            draft_ids.append(tok)
            draft_probs.append(probs)
            ctx.append(tok)
            if confidence < cfg.confidence_threshold:
                break  # dynamic draft exit

        return draft_ids, draft_probs

    # ------------------------------------------------------------------
    # Main generate loop
    # ------------------------------------------------------------------

    def generate(
        self,
        input_ids: List[int],
        max_new_tokens: int = 64,
    ) -> Tuple[List[int], DELStats]:
        """Generate up to *max_new_tokens* tokens using DEL.

        Parameters
        ----------
        input_ids : list[int]
            Prompt token ids (not modified).
        max_new_tokens : int

        Returns
        -------
        (output_ids, stats) where ``output_ids`` includes the prompt.
        """
        cfg   = self._cfg
        stats = DELStats()
        ids   = list(input_ids)
        generated = 0

        while generated < max_new_tokens:
            # Shadow analysis to pick the best exit layer for this position
            exit_layer = self._shadow_analysis(ids)
            stats.shadow_analyses += 1
            stats.exit_layer_counts[exit_layer] = (
                stats.exit_layer_counts.get(exit_layer, 0) + 1
            )

            # Draft with dynamic exit
            draft_ids, draft_probs = self._draft_dynamic(ids, exit_layer)
            if len(draft_ids) < cfg.gamma:
                stats.early_exits += 1

            # Verify each draft token with the full model
            ctx      = list(ids)
            accepted: List[int] = []
            rejected  = False

            for d_tok, d_probs in zip(draft_ids, draft_probs):
                full_logits = self._fwd(ctx, None)
                full_probs  = _softmax(full_logits)
                v_tok       = int(np.argmax(full_logits))
                p_t = float(full_probs[d_tok])
                p_d = float(d_probs[d_tok])

                if self._rng.random() < min(1.0, p_t / max(p_d, 1e-12)):
                    accepted.append(d_tok)
                    ctx.append(d_tok)
                    stats.accepted_draft += 1
                    self._layer_accepts[exit_layer] = (
                        self._layer_accepts.get(exit_layer, 0) + 1
                    )
                else:
                    accepted.append(v_tok)
                    ctx.append(v_tok)
                    stats.rejected_draft += 1
                    self._layer_rejects[exit_layer] = (
                        self._layer_rejects.get(exit_layer, 0) + 1
                    )
                    rejected = True
                    break

            # Bonus token when all draft tokens were accepted
            if not rejected:
                bonus_logits = self._fwd(ctx, None)
                bonus_tok    = int(np.argmax(bonus_logits))
                accepted.append(bonus_tok)

            ids.extend(accepted)
            generated += len(accepted)
            stats.total_tokens += len(accepted)

        return ids, stats
