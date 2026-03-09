"""
squish/token_swift.py

TokenSwift — Ultra-Long Text Generation with Multi-Token Draft Heads and
Partial KV Cache Reuse.

Based on:
  "TokenSwift: Accelerating LLM Inference in Ultra-Long Generation"
  ICML 2025 — arXiv:2502.18890   GitHub: bigai-nlco/TokenSwift

Problem
-------
Generating very long outputs (20K-100K tokens) with autoregressive decoding
has two compounding costs:

1. **KV cache growth** — every new token extends the KV cache.  For a 7B
   model at 100K tokens the KV cache alone can exceed 60 GB.

2. **Repeated attention over stale context** — most of the KV cache is
   occupied by the initial prompt and early output, which never changes.
   Re-attending over these frozen positions on every decode step wastes
   compute proportional to context length.

3. **Quality collapse** — vanilla long generation converges to repetitive
   phrase loops beyond ~20K tokens because nothing penalises already-
   generated patterns.

TokenSwift three-part solution
-------------------------------
1. **Multi-token draft heads** — K lightweight linear heads (offsets +1…+K)
   predict candidate tokens in parallel.  Like Medusa heads, they are trained
   on the model's own outputs in ~30 minutes.  Accepted tokens spread the
   verification cost across K positions at once.

2. **Partial KV cache reuse** — the initial prompt KV is computed once and
   frozen.  Only a rolling *dynamic window* (the most recent ``window_size``
   generated tokens) is updated at each step.  Positions outside the window
   are handled via landmark-based approximate attention, avoiding O(n²) cost.

3. **Contextual n-gram penalty** — at each step, candidate logits are
   penalised proportionally to the recency-weighted frequency of each token
   in the already-generated sequence.  This prevents the repetitive-pattern
   collapse seen in naïve ultra-long generation.

Performance
-----------
TokenSwift achieves over 3× speedup across 1.5B, 7B, 8B, and 14B models
(MHA and GQA architectures): the paper explicitly benchmarks Qwen2.5, making
results directly applicable to Qwen3.  A 100K-token generation that formerly
took ~5 hours completes in ~90 minutes.

Usage
-----
    from squish.token_swift import TokenSwiftConfig, MultiTokenHead, TokenSwiftDecoder

    cfg   = TokenSwiftConfig(n_heads=3, window_size=512, ngram_penalty=0.3)
    heads = MultiTokenHead(n_heads=3, hidden_size=4096, vocab_size=32000)
    dec   = TokenSwiftDecoder(
        target_fn=lambda ids: model_logits(ids),
        config=cfg,
        hidden_fn=lambda ids: model_hidden(ids),
        heads=heads,
    )
    output_ids, stats = dec.generate(prompt_ids, max_new_tokens=50_000)

Provides
--------
  TokenSwiftConfig     — configuration dataclass.
  MultiTokenHead       — K parallel draft prediction heads.
  PartialKVManager     — bookkeeping for frozen-prompt + dynamic-window KV.
  TokenSwiftStats      — per-generation counters.
  TokenSwiftDecoder    — full inference loop.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

__all__ = [
    "TokenSwiftConfig",
    "MultiTokenHead",
    "PartialKVManager",
    "TokenSwiftStats",
    "TokenSwiftDecoder",
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _softmax(logits: np.ndarray) -> np.ndarray:
    x = np.asarray(logits, dtype=np.float64)
    x = x - x.max()
    e = np.exp(x)
    return (e / e.sum()).astype(np.float32)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class TokenSwiftConfig:
    """Configuration for TokenSwift ultra-long generation.

    Parameters
    ----------
    n_heads : int
        Number of multi-token draft heads (predicts offsets +1..+n_heads).
        Must be ≥ 1.  The paper uses 3.
    window_size : int
        Size of the rolling dynamic KV window.  The most recent
        ``window_size`` generated tokens are fully updated each step.  The
        initial prompt KV is frozen.  Must be ≥ 1.
    ngram_penalty : float
        Repetition penalty weight applied to n-gram logit suppression.
        0 = disabled; 0.1–0.5 is the recommended range.  Must be ≥ 0.
    ngram_n : int
        Order of n-gram to track for the contextual penalty (default 3 =
        trigrams).  Must be ≥ 1.
    vocab_size : int
        Vocabulary size (≥ 2).  Only needed when ``ngram_penalty > 0``.
    temperature : float
        Sampling temperature (> 0).
    top_p : float
        Nucleus sampling probability (0, 1].
    gamma : int
        Speculation depth per round.  Should equal ``n_heads``; can be
        set lower to test partial-head drafting.  Clamped to ``n_heads``
        internally.
    """

    n_heads:       int   = 3
    window_size:   int   = 512
    ngram_penalty: float = 0.0
    ngram_n:       int   = 3
    vocab_size:    int   = 32000
    temperature:   float = 1.0
    top_p:         float = 1.0
    gamma:         int   = 3

    def __post_init__(self) -> None:
        if self.n_heads < 1:
            raise ValueError("n_heads must be ≥ 1")
        if self.window_size < 1:
            raise ValueError("window_size must be ≥ 1")
        if self.ngram_penalty < 0:
            raise ValueError("ngram_penalty must be ≥ 0")
        if self.ngram_n < 1:
            raise ValueError("ngram_n must be ≥ 1")
        if self.vocab_size < 2:
            raise ValueError("vocab_size must be ≥ 2")
        if self.temperature <= 0:
            raise ValueError("temperature must be > 0")
        if not 0.0 < self.top_p <= 1.0:
            raise ValueError("top_p must be in (0, 1]")
        if self.gamma < 1:
            raise ValueError("gamma must be ≥ 1")


# ---------------------------------------------------------------------------
# Multi-Token Draft Heads
# ---------------------------------------------------------------------------

class MultiTokenHead:
    """
    K lightweight linear draft heads for parallel multi-token prediction.

    Head *i* predicts the logit distribution for the token at offset +(i+1)
    from the current last position.  All heads take the same hidden-state
    vector as input and produce independent logit arrays.  They share no
    parameters.

    In the TokenSwift paper the heads are independent linear layers trained
    on 8K tokens of the target model's own outputs in ~30 minutes per model.
    Here they are randomly initialised; call :meth:`load_head_weights` to
    replace with trained weights.

    Parameters
    ----------
    n_heads : int
        Number of heads (= maximum token offset predicted).
    hidden_size : int
        Input hidden dimension.
    vocab_size : int
        Output vocabulary size.
    rng_seed : int
        Seed for random weight initialisation.
    """

    def __init__(
        self,
        n_heads:     int,
        hidden_size: int,
        vocab_size:  int,
        rng_seed:    int = 0,
    ) -> None:
        if n_heads < 1:
            raise ValueError("n_heads must be ≥ 1")
        if hidden_size < 1:
            raise ValueError("hidden_size must be ≥ 1")
        if vocab_size < 2:
            raise ValueError("vocab_size must be ≥ 2")

        rng   = np.random.default_rng(rng_seed)
        scale = float(np.sqrt(1.0 / hidden_size))
        self.weights: List[np.ndarray] = [
            (rng.standard_normal((vocab_size, hidden_size)) * scale).astype(np.float32)
            for _ in range(n_heads)
        ]
        self.biases: List[np.ndarray] = [
            np.zeros(vocab_size, dtype=np.float32)
            for _ in range(n_heads)
        ]
        self.n_heads     = n_heads
        self.hidden_size = hidden_size
        self.vocab_size  = vocab_size

    def predict(self, hidden: np.ndarray) -> List[np.ndarray]:
        """
        Run all K heads on *hidden* in a single pass.

        Parameters
        ----------
        hidden : np.ndarray
            Shape ``(hidden_size,)``.

        Returns
        -------
        list of K ``np.ndarray`` of shape ``(vocab_size,)`` — one logit array
        per head; head 0 → offset +1, head K-1 → offset +K.
        """
        h = hidden.astype(np.float32)
        return [h @ W.T + b for W, b in zip(self.weights, self.biases)]

    def load_head_weights(
        self,
        head_idx: int,
        W: np.ndarray,
        b: Optional[np.ndarray] = None,
    ) -> None:
        """Load pre-trained weights for head at *head_idx* (0-based)."""
        if not (0 <= head_idx < self.n_heads):
            raise IndexError(
                f"head_idx {head_idx} out of range [0, {self.n_heads})"
            )
        self.weights[head_idx] = W.astype(np.float32)
        if b is not None:
            self.biases[head_idx] = b.astype(np.float32)


# ---------------------------------------------------------------------------
# Partial KV Manager
# ---------------------------------------------------------------------------

class PartialKVManager:
    """
    Bookkeeping for TokenSwift's frozen-prompt + rolling-window KV policy.

    TokenSwift freezes the initial prompt's KV entries after the first
    prefill.  Only a rolling window of the most recently generated tokens
    (``window_size``) receives fresh attention writes each step.  All other
    generated positions are handled by approximate (landmark-based) attention
    rather than exact full-context attention.

    This class tracks position indices; the actual KV tensor operations are
    performed by the inference backend (caller's responsibility).

    Parameters
    ----------
    prompt_len : int
        Number of tokens in the initial prompt.  These positions are frozen.
    window_size : int
        Maximum number of *generated* positions in the dynamic update window.
    """

    def __init__(self, prompt_len: int, window_size: int) -> None:
        if prompt_len < 0:
            raise ValueError("prompt_len must be ≥ 0")
        if window_size < 1:
            raise ValueError("window_size must be ≥ 1")
        self.prompt_len  = prompt_len
        self.window_size = window_size
        self._gen_len    = 0

    def add_tokens(self, n: int) -> None:
        """Record that *n* new tokens have been appended to the sequence."""
        self._gen_len += max(0, n)

    @property
    def total_len(self) -> int:
        """Total sequence length (prompt + generated)."""
        return self.prompt_len + self._gen_len

    @property
    def window_start(self) -> int:
        """First position (inclusive) of the dynamic window."""
        return max(self.prompt_len, self.total_len - self.window_size)

    @property
    def window_end(self) -> int:
        """Last position (exclusive) of the dynamic window."""
        return self.total_len

    @property
    def window_positions(self) -> List[int]:
        """List of positions in the rolling dynamic window."""
        return list(range(self.window_start, self.window_end))

    @property
    def frozen_positions(self) -> List[int]:
        """Prompt positions that are permanently frozen (always attended)."""
        return list(range(self.prompt_len))

    def evict_fraction(self) -> float:
        """Fraction of generated tokens currently outside the window (0–1)."""
        if self._gen_len == 0:
            return 0.0
        evicted = max(0, self._gen_len - self.window_size)
        return evicted / self._gen_len


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

@dataclass
class TokenSwiftStats:
    """Per-generation counters returned by :class:`TokenSwiftDecoder`."""

    total_tokens:    int = 0
    draft_steps:     int = 0
    accepted_total:  int = 0
    rejected_total:  int = 0
    penalty_applied: int = 0   # logit computations where n-gram penalty fired

    @property
    def acceptance_rate(self) -> float:
        n = self.accepted_total + self.rejected_total
        return self.accepted_total / n if n > 0 else 0.0

    @property
    def mean_accepted_per_step(self) -> float:
        return self.accepted_total / self.draft_steps if self.draft_steps > 0 else 0.0


# ---------------------------------------------------------------------------
# Decoder
# ---------------------------------------------------------------------------

class TokenSwiftDecoder:
    """
    Ultra-long text generation loop combining multi-token draft heads,
    partial KV cache reuse, and contextual n-gram repetition penalty.

    Parameters
    ----------
    target_fn : callable
        ``target_fn(ids: list[int]) -> np.ndarray``
        Full-model forward pass returning ``(vocab_size,)`` logits for the
        last position.  For partial KV reuse the caller should only attend
        over ``frozen_positions + window_positions``; this module passes the
        full ``ids`` list and the caller may implement the KV shortcut.
    config : TokenSwiftConfig
    hidden_fn : callable, optional
        ``hidden_fn(ids: list[int]) -> np.ndarray``
        Returns ``(hidden_size,)`` last hidden state.  Required when
        ``heads`` is provided; if absent, falls back to sequential
        ``target_fn`` greedy drafting.
    heads : MultiTokenHead, optional
        Draft prediction heads.  If absent, the decoder uses the target
        model directly for drafting (slower, but keeps the n-gram penalty
        and partial KV machinery).
    rng_seed : int, optional
    """

    def __init__(
        self,
        target_fn: Callable[[List[int]], np.ndarray],
        config:    TokenSwiftConfig,
        hidden_fn: Optional[Callable[[List[int]], np.ndarray]] = None,
        heads:     Optional[MultiTokenHead] = None,
        rng_seed:  int = 0,
    ) -> None:
        self._target       = target_fn
        self._hidden       = hidden_fn
        self._heads        = heads
        self._cfg          = config
        self._rng          = np.random.default_rng(rng_seed)
        self._ngram_counts: Dict[Tuple[int, ...], int] = {}

    # ------------------------------------------------------------------
    # N-gram penalty
    # ------------------------------------------------------------------

    def _update_ngram(self, tok: int, ids: List[int]) -> None:
        """Increment the n-gram counter for the newly appended token *tok*."""
        n = self._cfg.ngram_n
        if len(ids) >= n - 1:
            key = tuple(ids[-(n - 1):]) + (tok,)
            self._ngram_counts[key] = self._ngram_counts.get(key, 0) + 1

    def _apply_ngram_penalty(
        self,
        logits: np.ndarray,
        ids: List[int],
    ) -> Tuple[np.ndarray, bool]:
        """
        Subtract a count-proportional penalty from logits for repeated
        n-grams.

        Returns ``(adjusted_logits, penalty_was_nonzero)``.
        """
        cfg = self._cfg
        if cfg.ngram_penalty == 0.0 or not self._ngram_counts:
            return logits, False

        n       = cfg.ngram_n
        penalty = np.zeros(cfg.vocab_size, dtype=np.float32)
        applied = False

        if len(ids) >= n - 1:
            prefix = tuple(ids[-(n - 1):])
            for vocab_id in range(min(cfg.vocab_size, len(logits))):
                cnt = self._ngram_counts.get(prefix + (vocab_id,), 0)
                if cnt > 0:
                    penalty[vocab_id] = cfg.ngram_penalty * cnt
                    applied = True

        return logits - penalty[: len(logits)], applied

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def _sample(self, logits: np.ndarray) -> int:
        probs = _softmax(logits / max(self._cfg.temperature, 1e-8))
        if self._cfg.top_p < 1.0:
            idx    = np.argsort(-probs)
            cumsum = np.cumsum(probs[idx])
            cutoff = int((cumsum < self._cfg.top_p).sum()) + 1
            mask   = np.zeros_like(probs)
            mask[idx[:max(1, cutoff)]] = 1.0
            s     = (probs * mask).sum()
            probs = probs * mask / (s + 1e-12)
        try:
            return int(self._rng.choice(len(probs), p=probs))
        except ValueError:
            return int(np.argmax(logits))

    # ------------------------------------------------------------------
    # Draft generation
    # ------------------------------------------------------------------

    def _draft_tokens(
        self,
        ids: List[int],
        gamma: int,
    ) -> Tuple[List[int], List[np.ndarray]]:
        """
        Produce up to *gamma* draft token ids and their probability arrays.

        If multi-token heads + hidden_fn are available, calls ``hidden_fn``
        once and applies all K heads in a single pass.  Otherwise falls back
        to γ sequential ``target_fn`` calls.
        """
        if self._heads is not None and self._hidden is not None:
            hidden      = self._hidden(ids)
            head_logits = self._heads.predict(hidden)
            draft_ids:   List[int]        = []
            draft_probs: List[np.ndarray] = []
            for hlogit in head_logits[:gamma]:
                adj, _ = self._apply_ngram_penalty(hlogit, ids + draft_ids)
                draft_ids.append(self._sample(adj))
                draft_probs.append(_softmax(adj))
            return draft_ids, draft_probs

        # Fallback: sequential target calls
        draft_ids   = []
        draft_probs = []
        ctx = list(ids)
        for _ in range(gamma):
            raw         = self._target(ctx)
            adj, _      = self._apply_ngram_penalty(raw, ctx)
            draft_ids.append(self._sample(adj))
            draft_probs.append(_softmax(adj))
            ctx.append(draft_ids[-1])
        return draft_ids, draft_probs

    # ------------------------------------------------------------------
    # Main generate loop
    # ------------------------------------------------------------------

    def generate(
        self,
        input_ids: List[int],
        max_new_tokens: int = 1024,
    ) -> Tuple[List[int], TokenSwiftStats]:
        """
        Generate up to *max_new_tokens* tokens using TokenSwift.

        Each round:
        1. Draft *gamma* tokens using multi-token heads (or fallback).
        2. Verify each draft token against the target model with the
           Leviathan accept/reject rule.  N-gram penalty is applied to
           all target-logit computations.
        3. Append accepted tokens, update n-gram table, advance KV manager.
        4. Add a bonus token on full acceptance.

        Parameters
        ----------
        input_ids : list[int]
        max_new_tokens : int

        Returns
        -------
        (output_ids, stats)
        """
        cfg    = self._cfg
        stats  = TokenSwiftStats()
        ids    = list(input_ids)
        kv_mgr = PartialKVManager(
            prompt_len  = len(input_ids),
            window_size = cfg.window_size,
        )
        generated = 0
        self._ngram_counts.clear()

        while generated < max_new_tokens:
            avail = max_new_tokens - generated
            gamma = min(cfg.gamma, min(cfg.n_heads, avail))

            # ── Draft phase ───────────────────────────────────────────────────
            draft_ids, draft_probs = self._draft_tokens(ids, gamma)
            if not draft_ids:
                break
            stats.draft_steps += 1

            # ── Verify phase ──────────────────────────────────────────────────
            ctx       = list(ids)
            accepted: List[int] = []
            rejected  = False

            for d_tok, d_probs in zip(draft_ids, draft_probs):
                t_raw         = self._target(ctx)
                t_adj, fired  = self._apply_ngram_penalty(t_raw, ctx)
                if fired:
                    stats.penalty_applied += 1
                t_probs = _softmax(t_adj)
                v_tok   = self._sample(t_adj)
                p_t     = float(t_probs[d_tok])
                p_d     = float(d_probs[d_tok])

                if self._rng.random() < min(1.0, p_t / max(p_d, 1e-12)):
                    accepted.append(d_tok)
                    ctx.append(d_tok)
                    stats.accepted_total += 1
                else:
                    accepted.append(v_tok)
                    ctx.append(v_tok)
                    stats.rejected_total += 1
                    rejected = True
                    break

            if not rejected and (generated + len(accepted)) < max_new_tokens:
                bonus_raw       = self._target(ctx)
                bonus_adj, fire = self._apply_ngram_penalty(bonus_raw, ctx)
                if fire:
                    stats.penalty_applied += 1
                bonus_tok = self._sample(bonus_adj)
                accepted.append(bonus_tok)
                stats.accepted_total += 1

            # Update n-gram table and advance KV manager
            for tok in accepted:
                self._update_ngram(tok, ids)
                ids.append(tok)
            kv_mgr.add_tokens(len(accepted))
            generated += len(accepted)

        stats.total_tokens = generated
        return ids, stats
