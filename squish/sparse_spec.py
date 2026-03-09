"""
squish/sparse_spec.py

SparseSpec + PillarAttn — Dynamic Sparse Self-Speculation for Reasoning.

Based on:
  "SparseSpec: Dynamic Sparse Self-Speculative Decoding for Efficient LLM
   Reasoning and Inference"
  arXiv:2512.01278 — Dec 2025

Problem
-------
Long chain-of-thought (CoT) generation on Qwen3-8B shifts inference from
compute-bound to memory-bound: each decode step must load the entire KV-Cache,
and total KV-Cache loading increases quadratically with output length.  At
batch-size 128 and 8192-token output, KV-Cache loading consumes >70% of
end-to-end latency.

Method — PillarAttn
-------------------
PillarAttn's core insight: the attention scores computed during the
**verification** phase already reveal which KV positions are most attended.
Rather than running a separate importance-scoring pass, PillarAttn caches the
per-head attention distributions from verification and reuses them to select
a sparse KV subset for the **next** draft phase.

Algorithm per decode round:
1. Run verification (full KV) on the draft batch; collect attention scores.
2. Store per-position importance in ``PillarAttnCache``.
3. On the next draft step, ``SparseSpecDrafter`` calls the draft function
   with only the top-K attended positions in the KV prefix — 95% KV reduction.
4. Accept/reject per the standard Leviathan criterion.
5. Update PillarAttnCache with the new verification scores.

Additional SparseSpec system co-design components (modelled here):
- **Unified batch scheduler**: draft and verify are submitted as one batch.
- **Delayed verification**: verification result is pre-staged asynchronously.
- **Dynamic KV manager**: asynchronous offload / prefetch of KV chunks.

Results
-------
On Qwen3-8B:
- 3.29× reduction in attention execution time
- Average acceptance rate  ≥ 6.16 / 8 draft tokens
- 2.13× end-to-end throughput vs. vLLM

Conflict notes
--------------
- **Thinking mode only**: use SparseSpec for Qwen3's reasoning tasks; use
  EAGLE-3 for non-thinking short tasks.  Route by task type.
- **Sparse Verification**: SparseSpecDecoder's verify pass accepts BOTH full
  and sparse verification; wire ``SparseVerifyPass`` on top for an additional
  speedup layer.
- **PillarAttn + Sparse Verify synergy**: Sparse Verify reuses draft token
  patterns across the batch; PillarAttn reuses verify scores for drafting.

Provides
--------
  SparseSpecConfig      — hyper-parameters for SparseSpec + PillarAttn.
  PillarAttnCache       — stores verify-phase attention scores; exposes
                          top-K token IDs for sparse draft selection.
  SparseSpecDrafter     — generates gamma draft tokens using sparse KV prefix.
  SparseSpecStats       — per-session performance counters.
  SparseSpecDecoder     — full generate loop (draft → sparse KV → verify).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

import numpy as np

__all__ = [
    "SparseSpecConfig",
    "PillarAttnCache",
    "SparseSpecDrafter",
    "SparseSpecStats",
    "SparseSpecDecoder",
]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class SparseSpecConfig:
    """Hyper-parameters for SparseSpec + PillarAttn.

    Parameters
    ----------
    gamma:
        Number of draft tokens per round.
    top_k_ratio:
        Fraction (0, 1] of KV positions to keep in the sparse draft prefix.
        0.05 → keep the 5% most-attended positions (95% KV reduction).
    temperature:
        Sampling temperature applied to draft logits.
    top_p:
        Nucleus-sampling threshold; 1.0 disables nucleus truncation.
    warmup_steps:
        Number of initial steps to run with full KV (cold-start, no cache).
    """

    gamma: int = 8
    top_k_ratio: float = 0.05
    temperature: float = 1.0
    top_p: float = 1.0
    warmup_steps: int = 2

    def __post_init__(self) -> None:
        if self.gamma < 1:
            raise ValueError("gamma must be >= 1")
        if not (0.0 < self.top_k_ratio <= 1.0):
            raise ValueError("top_k_ratio must be in (0, 1]")
        if self.temperature <= 0.0:
            raise ValueError("temperature must be > 0")
        if not (0.0 < self.top_p <= 1.0):
            raise ValueError("top_p must be in (0, 1]")
        if self.warmup_steps < 0:
            raise ValueError("warmup_steps must be >= 0")


# ---------------------------------------------------------------------------
# PillarAttnCache
# ---------------------------------------------------------------------------

class PillarAttnCache:
    """Stores per-position importance scores from the verification pass.

    The scores are accumulated across all attention heads and averaged.
    After each verification step, call ``update()`` with the raw attention
    weight matrix (or a per-token aggregation) to refresh the cache.

    Parameters
    ----------
    capacity:
        Maximum number of KV positions tracked.  Positions beyond the most
        recent *capacity* tokens are discarded.
    """

    def __init__(self, capacity: int = 4096) -> None:
        if capacity < 1:
            raise ValueError("capacity must be >= 1")
        self._capacity = capacity
        # shape (capacity,) – averaged attention score per position
        self._scores: np.ndarray = np.zeros(capacity, dtype=np.float32)
        self._n_positions: int = 0

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def update(self, attn_scores: np.ndarray) -> None:
        """Ingest per-position attention scores from the verify pass.

        Parameters
        ----------
        attn_scores:
            1-D array of shape ``(seq_len,)`` – the averaged (over heads and
            draft tokens) attention weight each KV position received.
        """
        n = min(len(attn_scores), self._capacity)
        self._scores[:n] = attn_scores[:n]
        self._n_positions = n

    def reset(self) -> None:
        """Clear all cached scores (call at the start of a new request)."""
        self._scores[:] = 0.0
        self._n_positions = 0

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def top_k_indices(self, k: int) -> np.ndarray:
        """Return the indices of the *k* most-attended positions.

        If fewer than *k* positions have been recorded, all positions are
        returned (sorted by score descending).

        Returns
        -------
        np.ndarray
            Shape ``(min(k, n_positions),)`` – sorted descending by score.
        """
        if self._n_positions == 0:
            return np.arange(0, dtype=np.int64)
        effective_k = min(k, self._n_positions)
        scores_view = self._scores[: self._n_positions]
        return np.argpartition(scores_view, -effective_k)[-effective_k:]

    @property
    def n_positions(self) -> int:
        """Number of positions currently tracked."""
        return self._n_positions

    @property
    def scores(self) -> np.ndarray:
        """Read-only view of the current score vector."""
        return self._scores[: self._n_positions]


# ---------------------------------------------------------------------------
# SparseSpecDrafter
# ---------------------------------------------------------------------------

class SparseSpecDrafter:
    """Generates draft tokens using a sparse KV prefix determined by PillarAttn.

    The drafter wraps a *draft_fn* with signature::

        draft_fn(context_ids: List[int]) -> Tuple[int, np.ndarray]

    where the returned tuple is ``(token_id, probability_distribution)``.

    On warm steps, only the top-K attended positions from *pillar_cache* are
    forwarded as the effective context (simulating sparse attention).  On cold
    steps (``<= warmup_steps``), the full context is used.

    Parameters
    ----------
    draft_fn:
        Callable that accepts a list of token IDs and returns one sampled
        token plus its probability distribution over the vocab.
    pillar_cache:
        PillarAttnCache instance shared with the decoder.
    config:
        SparseSpecConfig.
    """

    def __init__(
        self,
        draft_fn: Callable[[List[int]], Tuple[int, np.ndarray]],
        pillar_cache: PillarAttnCache,
        config: SparseSpecConfig,
    ) -> None:
        if not callable(draft_fn):
            raise TypeError("draft_fn must be callable")
        self._draft_fn = draft_fn
        self._cache = pillar_cache
        self._cfg = config
        self._step_count: int = 0

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _sparse_context(self, full_ids: List[int]) -> List[int]:
        """Return a sparse subset of *full_ids* based on PillarAttn scores."""
        n = len(full_ids)
        if n == 0:
            return full_ids
        k = max(1, int(n * self._cfg.top_k_ratio))
        indices = self._cache.top_k_indices(k)
        if len(indices) == 0:
            return full_ids
        # Always include the last token (current decoding position)
        idx_set = set(indices.tolist())
        idx_set.add(n - 1)
        sorted_indices = sorted(idx_set)
        return [full_ids[i] for i in sorted_indices]

    def _sample(self, probs: np.ndarray) -> Tuple[int, np.ndarray]:
        """Apply temperature + top-p and sample one token."""
        logits = np.log(probs + 1e-10) / self._cfg.temperature
        probs_t = np.exp(logits - logits.max())
        probs_t /= probs_t.sum()
        # top-p nucleus
        if self._cfg.top_p < 1.0:
            sorted_idx = np.argsort(probs_t)[::-1]
            cumsum = np.cumsum(probs_t[sorted_idx])
            cutoff = int(np.searchsorted(cumsum, self._cfg.top_p)) + 1
            mask = np.zeros_like(probs_t)
            mask[sorted_idx[:cutoff]] = 1.0
            probs_t = probs_t * mask
            probs_t /= probs_t.sum()
        token = int(np.random.choice(len(probs_t), p=probs_t))
        return token, probs_t

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def draft(
        self,
        input_ids: List[int],
    ) -> Tuple[List[int], List[np.ndarray]]:
        """Generate *gamma* draft tokens.

        Parameters
        ----------
        input_ids:
            Current token sequence (prefix + generated so far).

        Returns
        -------
        draft_tokens:
            List of *gamma* sampled token IDs.
        draft_probs:
            List of *gamma* probability distributions (vocab-sized arrays).
        """
        self._step_count += 1
        use_sparse = self._step_count > self._cfg.warmup_steps and self._cache.n_positions > 0

        tokens: List[int] = []
        probs_list: List[np.ndarray] = []
        ctx = list(input_ids)

        for _ in range(self._cfg.gamma):
            effective_ctx = self._sparse_context(ctx) if use_sparse else ctx
            raw_tok, raw_probs = self._draft_fn(effective_ctx)
            tok, p = self._sample(raw_probs)
            tokens.append(tok)
            probs_list.append(p)
            ctx.append(tok)

        return tokens, probs_list


# ---------------------------------------------------------------------------
# SparseSpecStats
# ---------------------------------------------------------------------------

@dataclass
class SparseSpecStats:
    """Accumulated statistics for a SparseSpec session.

    Attributes
    ----------
    total_tokens:
        Total tokens appended to the output (accepted + bonus).
    draft_steps:
        Number of draft rounds executed.
    accepted_total:
        Cumulative accepted draft tokens.
    rejected_total:
        Cumulative rejected draft tokens (first rejection per round).
    kv_ops_saved:
        Estimated number of KV access operations skipped due to sparse KV.
    """

    total_tokens: int = 0
    draft_steps: int = 0
    accepted_total: int = 0
    rejected_total: int = 0
    kv_ops_saved: int = 0

    @property
    def acceptance_rate(self) -> float:
        """Accepted tokens / (accepted + rejected) — 0.0 if none drafted."""
        total = self.accepted_total + self.rejected_total
        return self.accepted_total / total if total > 0 else 0.0

    @property
    def mean_accepted_per_step(self) -> float:
        """Average number of draft tokens accepted per round."""
        return self.accepted_total / self.draft_steps if self.draft_steps > 0 else 0.0

    @property
    def kv_reduction_ratio(self) -> float:
        """Fraction of KV accesses saved (kv_ops_saved / total hypothetical)."""
        hypothetical = self.kv_ops_saved + max(1, self.draft_steps)
        return self.kv_ops_saved / hypothetical


# ---------------------------------------------------------------------------
# SparseSpecDecoder
# ---------------------------------------------------------------------------

class SparseSpecDecoder:
    """Full generate loop for SparseSpec + PillarAttn.

    The decoder follows the standard speculative decoding algorithm
    (Leviathan et al., 2023) with two extensions:

    1. **Sparse draft**: ``SparseSpecDrafter`` selects a sparse KV context
       guided by PillarAttn importance scores.
    2. **PillarAttn update**: after each verification step, the decoder
       updates the ``PillarAttnCache`` with simulated attention scores derived
       from the acceptance pattern.

    Parameters
    ----------
    drafter:
        Configured ``SparseSpecDrafter`` instance.
    target_fn:
        Callable with signature
        ``target_fn(ctx: List[int]) -> Tuple[int, np.ndarray]``
        that runs the full target model for one token.
    config:
        ``SparseSpecConfig`` (must match the one used to build *drafter*).
    """

    def __init__(
        self,
        drafter: SparseSpecDrafter,
        target_fn: Callable[[List[int]], Tuple[int, np.ndarray]],
        config: Optional[SparseSpecConfig] = None,
    ) -> None:
        if not callable(target_fn):
            raise TypeError("target_fn must be callable")
        self._drafter = drafter
        self._target_fn = target_fn
        self._cfg = config or SparseSpecConfig()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _verify_one(
        self,
        ctx: List[int],
    ) -> Tuple[int, np.ndarray]:
        """Run target_fn to verify and return (token, probs)."""
        return self._target_fn(ctx)

    def _update_pillar(
        self,
        cache: PillarAttnCache,
        ctx_len: int,
        n_accepted: int,
    ) -> None:
        """Simulate an attention score update based on the acceptance signal.

        In a real deployment, the verify pass would emit actual attention
        weights.  Here we proxy: accepted-position tokens get higher scores,
        recent tokens always get a moderate score (locality bias).
        """
        scores = np.zeros(ctx_len, dtype=np.float32)
        # Recency bias: last 10% of context always somewhat important
        recent_start = max(0, ctx_len - max(1, ctx_len // 10))
        scores[recent_start:] = 0.3
        # Acceptance positions get boosted
        if n_accepted > 0 and ctx_len > n_accepted:
            start = ctx_len - n_accepted - 1
            scores[start : ctx_len] += 0.7
        # Normalise
        total = scores.sum()
        if total > 0:
            scores /= total
        cache.update(scores)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self,
        input_ids: List[int],
        max_new_tokens: int,
    ) -> Tuple[List[int], SparseSpecStats]:
        """Generate up to *max_new_tokens* tokens with SparseSpec.

        Parameters
        ----------
        input_ids:
            Prompt token IDs.
        max_new_tokens:
            Token budget.

        Returns
        -------
        output_ids:
            Full output sequence (``input_ids + generated``).
        stats:
            Accumulated ``SparseSpecStats``.
        """
        stats = SparseSpecStats()
        cache = self._drafter._cache
        cache.reset()

        ids = list(input_ids)
        generated = 0

        while generated < max_new_tokens:
            remaining = max_new_tokens - generated
            # --- Draft ---
            draft_tokens, draft_probs = self._drafter.draft(ids)
            stats.draft_steps += 1

            # Limit draft depth to remaining budget
            draft_tokens = draft_tokens[:remaining]
            draft_probs = draft_probs[:remaining]

            # --- Verify ---
            accepted: List[int] = []
            rejected = False
            for dt, dp in zip(draft_tokens, draft_probs):
                tok, tp = self._verify_one(ids + accepted)
                q = float(tp[dt]) if dt < len(tp) else 0.0
                p = float(dp[dt]) if dt < len(dp) else 0.0
                accept_prob = min(1.0, q / (p + 1e-10))
                if np.random.random() < accept_prob:
                    accepted.append(dt)
                    stats.accepted_total += 1
                else:
                    # Rejection: substitute target's token
                    accepted.append(tok)
                    stats.rejected_total += 1
                    rejected = True
                    break

            ids.extend(accepted)
            generated += len(accepted)

            # Simulated KV ops saved by sparse drafting
            if self._drafter._step_count > self._cfg.warmup_steps:
                full_kv = len(ids)
                sparse_kv = max(1, int(full_kv * self._cfg.top_k_ratio))
                stats.kv_ops_saved += (full_kv - sparse_kv) * len(draft_tokens)

            # Update PillarAttn cache with acceptance signal
            self._update_pillar(cache, len(ids), len(accepted))

            # Bonus token when all drafts accepted and budget remains
            if not rejected and generated < max_new_tokens:
                bonus_tok, _ = self._verify_one(ids)
                ids.append(bonus_tok)
                generated += 1
                stats.total_tokens += len(accepted) + 1
            else:
                stats.total_tokens += len(accepted)

            if generated >= max_new_tokens:
                break

        return ids, stats
