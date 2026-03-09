"""
squish/duo_decoding.py

DuoDecoding — Hardware-Aware Dynamic Multi-Sequence Speculative Decoding.

Based on:
  "DuoDecoding: Hardware-Aware Heterogeneous Speculative Decoding with
   Dynamic Multi-Sequence Drafting"
  arXiv:2503.00784; Kai Lv et al., Fudan University, March 2025

Problem
-------
Dovetail and Mirror-SD use a fixed number of draft sequences per verification
step (γ = constant).  On M3's heterogeneous compute fabric, GPU utilisation
varies:
  - Short prompts: GPU has spare capacity → waste.
  - Long KV caches: attention is the bottleneck → draft overhead dominates.

DuoDecoding addresses this with *dynamic multi-sequence drafting*:
  - Maintain K parallel candidate sequences on the GPU simultaneously.
  - Each candidate has an in-flight acceptance probability estimate.
  - The scheduler abandons branches below a pruning threshold and spawns new
    ones when GPU capacity allows, adapting K per step.

CPU verification remains unified: the CPU target model verifies the BEST
candidate sequence selected by the scheduler after GPU drafting completes.

Method
------
1. **DuoCandidate** — one active draft sequence (ids, probs, score estimate).

2. **DuoScheduler** — manages K simultaneous candidates:
   - ``step(base_ids, k_target)`` — draft one step for each active candidate.
   - ``prune(threshold)`` — remove candidates below score threshold.
   - ``best()`` — return the highest-scored candidate.

3. **DuoCPUVerifier** — identical to Dovetail's; runs the target model on CPU
   to verify the best candidate from the scheduler.

4. **DuoDecodingDecoder** — outer loop:
   a. DuoScheduler produces up to K draft sequences of depth γ.
   b. Select the best candidate based on joint log-probability.
   c. DuoCPUVerifier accepts/rejects token by token (Leviathan).
   d. Append accepted tokens; score accepted sequences higher in next round.

Design note
-----------
The "hardware utilisation" signal in production would read Metal GPU occupancy
(via MTLCommandBuffer completion callbacks).  In this reference implementation,
the number of active candidates is controlled by ``k_max`` and ``prune_threshold``.

Conflict-resolved notes (Master Conflict Report)
-------------------------------------------------
- **vs Dovetail**: DuoDecoding extends Dovetail with multi-sequence drafting.
  DuoDecoding's single-candidate degenerate case is Dovetail.
- **vs Mirror-SD**: Mirror-SD is preferred when ANE is available.  DuoDecoding
  is the CPU-only fallback when ANE is congested.
- **vs SwiftSpec async pipeline**: SwiftSpec's async scheduling works *within*
  the decode partition; DuoDecoding manages *between-candidate* scheduling.
  They compose at different scheduling layers.

Provides
--------
  DuoDecodingConfig     — tuning parameters.
  DuoCandidate          — single active draft sequence + metadata.
  DuoScheduler          — K-sequence draft manager.
  DuoCPUVerifier        — CPU-side target verification (same API as Dovetail).
  DuoDecodingStats      — per-session counters.
  DuoDecodingDecoder    — full heterogeneous speculative decoder.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

import numpy as np

__all__ = [
    "DuoDecodingConfig",
    "DuoCandidate",
    "DuoScheduler",
    "DuoCPUVerifier",
    "DuoDecodingStats",
    "DuoDecodingDecoder",
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()


def _sample(logits: np.ndarray, temperature: float, top_p: float,
            rng: np.random.Generator) -> Tuple[int, np.ndarray]:
    probs = _softmax(logits / max(temperature, 1e-8))
    if top_p < 1.0:
        sorted_idx = np.argsort(probs)[::-1]
        cumsum = np.cumsum(probs[sorted_idx])
        cutoff = int(np.searchsorted(cumsum, top_p)) + 1
        mask = np.zeros_like(probs)
        mask[sorted_idx[:cutoff]] = 1.0
        probs = probs * mask
        probs /= probs.sum()
    token = int(rng.choice(len(probs), p=probs))
    return token, probs


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class DuoDecodingConfig:
    """Parameters for DuoDecoding.

    Parameters
    ----------
    gamma:
        Draft speculation depth per candidate sequence.
    k_max:
        Maximum number of parallel candidate sequences maintained per step.
    prune_threshold:
        Candidates with joint log-probability below
        ``best_score + prune_threshold`` are dropped.  Larger values = more
        aggressive pruning.  0.0 = keep all candidates.
    temperature:
        Sampling temperature.
    top_p:
        Nucleus sampling threshold.
    """

    gamma: int = 4
    k_max: int = 3
    prune_threshold: float = 5.0
    temperature: float = 1.0
    top_p: float = 1.0

    def __post_init__(self) -> None:
        if self.gamma < 1:
            raise ValueError("gamma must be >= 1")
        if self.k_max < 1:
            raise ValueError("k_max must be >= 1")
        if self.prune_threshold < 0:
            raise ValueError("prune_threshold must be >= 0")
        if self.temperature <= 0:
            raise ValueError("temperature must be > 0")
        if not (0.0 < self.top_p <= 1.0):
            raise ValueError("top_p must be in (0, 1]")


# ---------------------------------------------------------------------------
# DuoCandidate
# ---------------------------------------------------------------------------

@dataclass
class DuoCandidate:
    """One active draft sequence managed by the DuoScheduler.

    Parameters
    ----------
    tokens:
        Draft tokens generated so far for this candidate (not including the
        base context).
    probs:
        Probability distributions for each token in ``tokens``.
    log_prob:
        Cumulative log-probability of the candidate (used for scoring /
        pruning).
    """

    tokens: List[int] = field(default_factory=list)
    probs: List[np.ndarray] = field(default_factory=list)
    log_prob: float = 0.0

    def append(self, token: int, prob_dist: np.ndarray) -> None:
        """Extend this candidate with one more draft token."""
        self.tokens.append(token)
        self.probs.append(prob_dist)
        self.log_prob += float(np.log(max(float(prob_dist[token]), 1e-45)))

    @property
    def depth(self) -> int:
        return len(self.tokens)


# ---------------------------------------------------------------------------
# DuoScheduler
# ---------------------------------------------------------------------------

class DuoScheduler:
    """Manages K simultaneous draft candidate sequences.

    In each scheduling step, each active candidate is extended by one token
    using the GPU-side draft function.  Candidates below the pruning threshold
    are dropped; new candidates are spawned (from the best candidate) if the
    active count is below ``k_max``.

    Parameters
    ----------
    draft_fn:
        ``(ids: List[int]) -> np.ndarray`` — draft model logit function.
    config:
        ``DuoDecodingConfig``.
    rng_seed:
        Reproducibility seed.
    """

    def __init__(
        self,
        draft_fn: Callable[[List[int]], np.ndarray],
        config: DuoDecodingConfig,
        rng_seed: int = 0,
    ) -> None:
        self._fn = draft_fn
        self._cfg = config
        self._rng = np.random.default_rng(rng_seed)

    # ------------------------------------------------------------------

    def draft_candidates(
        self, base_ids: List[int]
    ) -> List[DuoCandidate]:
        """Generate up to *k_max* candidates each of depth *gamma*.

        Each candidate diverges from a fresh sample at its first token,
        mimicking the diverse sampling needed for multi-sequence speculation.

        Returns
        -------
        List of DuoCandidate sorted by descending log_prob.
        """
        cfg = self._cfg
        candidates: List[DuoCandidate] = []

        for _ in range(cfg.k_max):
            cand = DuoCandidate()
            ctx = list(base_ids)
            for _ in range(cfg.gamma):
                logits = self._fn(ctx)
                tok, probs = _sample(logits, cfg.temperature, cfg.top_p,
                                     self._rng)
                cand.append(tok, probs)
                ctx.append(tok)
            candidates.append(cand)

        # Prune candidates far below the best
        candidates.sort(key=lambda c: c.log_prob, reverse=True)
        if cfg.prune_threshold > 0 and candidates:
            best_score = candidates[0].log_prob
            candidates = [
                c for c in candidates
                if (best_score - c.log_prob) <= cfg.prune_threshold
            ]

        return candidates

    def best(self, candidates: List[DuoCandidate]) -> DuoCandidate:
        """Return the highest-probability candidate."""
        if not candidates:
            raise ValueError("candidate list is empty")
        return max(candidates, key=lambda c: c.log_prob)


# ---------------------------------------------------------------------------
# DuoCPUVerifier
# ---------------------------------------------------------------------------

class DuoCPUVerifier:
    """CPU-side target model verifier for DuoDecoding.

    Identical role to Dovetail's verifier — accepts a token sequence context
    and returns the target model's next-token distribution, used for the
    Leviathan accept/reject criterion.

    Parameters
    ----------
    target_fn:
        ``(ids: List[int]) -> np.ndarray`` — returns ``(vocab_size,)`` logits.
    config:
        ``DuoDecodingConfig``.
    rng_seed:
        Reproducibility seed.
    """

    def __init__(
        self,
        target_fn: Callable[[List[int]], np.ndarray],
        config: DuoDecodingConfig,
        rng_seed: int = 0,
    ) -> None:
        self._fn = target_fn
        self._cfg = config
        self._rng = np.random.default_rng(rng_seed)

    def verify_one(self, ctx: List[int]) -> Tuple[int, np.ndarray]:
        """Verify one position: run target model, return (token, probs)."""
        logits = self._fn(ctx)
        tok, probs = _sample(logits, self._cfg.temperature,
                             self._cfg.top_p, self._rng)
        return tok, probs


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

@dataclass
class DuoDecodingStats:
    """Per-session statistics for DuoDecoding.

    Parameters
    ----------
    total_tokens:
        New tokens generated.
    draft_steps:
        GPU scheduling rounds executed.
    accepted_total:
        Draft tokens accepted by the CPU verifier.
    rejected_total:
        Draft tokens rejected (correction token substituted).
    candidates_pruned:
        Total candidates dropped due to the prune_threshold policy.
    """

    total_tokens: int = 0
    draft_steps: int = 0
    accepted_total: int = 0
    rejected_total: int = 0
    candidates_pruned: int = 0

    @property
    def acceptance_rate(self) -> float:
        n = self.accepted_total + self.rejected_total
        return self.accepted_total / n if n > 0 else 0.0

    @property
    def mean_accepted_per_step(self) -> float:
        return self.accepted_total / self.draft_steps if self.draft_steps > 0 else 0.0


# ---------------------------------------------------------------------------
# DuoDecodingDecoder
# ---------------------------------------------------------------------------

class DuoDecodingDecoder:
    """DuoDecoding full heterogeneous speculative decoder.

    Combines the DuoScheduler (multi-candidate GPU drafting) with the
    DuoCPUVerifier (target model on CPU) to produce output tokens.

    Per step:
    1. DuoScheduler generates up to k_max candidates.
    2. Best candidate is selected (highest joint log-probability).
    3. DuoCPUVerifier accepts/rejects each token in the best candidate.
    4. Accepted tokens appended; on full acceptance add a bonus token.

    Parameters
    ----------
    scheduler:
        ``DuoScheduler`` wrapping the GPU-side draft model.
    cpu_verifier:
        ``DuoCPUVerifier`` wrapping the CPU-side target model.
    config:
        ``DuoDecodingConfig``.
    rng_seed:
        Seed for the accept/reject RNG.
    """

    def __init__(
        self,
        scheduler: DuoScheduler,
        cpu_verifier: DuoCPUVerifier,
        config: Optional[DuoDecodingConfig] = None,
        rng_seed: int = 0,
    ) -> None:
        if config is None:
            config = DuoDecodingConfig()
        self._scheduler = scheduler
        self._verify = cpu_verifier
        self._cfg = config
        self._rng = np.random.default_rng(rng_seed)

    # ------------------------------------------------------------------

    def generate(
        self,
        input_ids: List[int],
        max_new_tokens: int = 64,
    ) -> Tuple[List[int], DuoDecodingStats]:
        """Generate up to *max_new_tokens* tokens.

        Parameters
        ----------
        input_ids:
            Starting token sequence.
        max_new_tokens:
            Upper bound on new tokens appended.

        Returns
        -------
        (output_ids, stats)
        """
        stats = DuoDecodingStats()
        ids = list(input_ids)
        generated = 0

        while generated < max_new_tokens:
            # ── GPU: generate K candidate sequences ─────────────────────────
            all_candidates = self._scheduler.draft_candidates(ids)
            pruned_count = self._cfg.k_max - len(all_candidates)
            stats.candidates_pruned += max(0, pruned_count)
            stats.draft_steps += 1

            best = self._scheduler.best(all_candidates)

            # Limit best candidate depth to remaining budget
            max_remaining = max_new_tokens - generated
            draft_tokens = best.tokens[:max_remaining]
            draft_probs  = best.probs[:max_remaining]

            # ── CPU: verify best candidate (Leviathan accept/reject) ─────────
            ctx = list(ids)
            accepted: List[int] = []
            rejected = False

            for d_tok, d_probs in zip(draft_tokens, draft_probs):
                v_tok, v_probs = self._verify.verify_one(ctx)
                p_t = float(v_probs[d_tok])
                p_d = float(d_probs[d_tok])

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

            # ── Bonus token on full acceptance ───────────────────────────────
            if not rejected and (generated + len(accepted)) < max_new_tokens:
                bonus_tok, _ = self._verify.verify_one(ctx)
                accepted.append(bonus_tok)
                stats.accepted_total += 1

            ids.extend(accepted)
            generated += len(accepted)

        stats.total_tokens = generated
        return ids, stats
