"""
squish/dovetail.py

Dovetail — CPU-Verification + GPU-Drafting Heterogeneous Speculative Decoding.

Based on:
  "Dovetail: Heterogeneous Speculative Decoding with CPU-GPU Collaboration"
  EMNLP 2025; arXiv:2412.18934

Problem
-------
Standard speculative decoding runs both draft and target models on the GPU.
On memory-constrained Apple Silicon (M3, 16 GB unified memory), loading a
7B/8B target model leaves little headroom for the KV cache and the 1.5B draft
model, causing frequent GPU out-of-memory pressuring and high memory pressure.

Dovetail **inverts** the compute assignment:
  - GPU (Metal): runs the small *draft* model (fast matrix multiply, short seq)
  - CPU (performance cores): runs the large *target* model (high parallelism,
    large L3 cache for BF16 verification passes)

On M3's unified memory, inter-device transfer is a zero-copy pointer cast.
CPU verification at BF16 on 4 performance cores (~3.7 GHz) is competitive with
GPU verification at INT4 for 8B-class models.

1.79x–10.1x speedup across devices (EMNLP 2025 benchmarks).

Method
------
1. **DovetailDraftRunner** — GPU-side draft; produces γ candidate tokens and
   their probability distributions in one sequential pass.

2. **DovetailCPUVerifier** — CPU-side target model verifier; called with the
   current context and each draft token; returns target logits per position.

3. **DovetailDecoder** — outer loop:
   a. GPU drafts γ tokens.
   b. CPU verifier pops one token at a time; apply Leviathan accept/reject.
   c. On full acceptance, emit one bonus target token from the CPU verifier.

Design note
-----------
In this reference implementation, ``draft_fn`` and ``verify_fn`` are plain
callables.  In production, ``draft_fn`` would be an MLX/Metal callable and
``verify_fn`` a BF16 llama.cpp-style function running on the CPU thread pool.

Conflict-resolved notes (Master Conflict Report)
-------------------------------------------------
- **vs EAGLE-3**: Dovetail uses a *separate* draft model on GPU (not EAGLE-3's
  hidden-state head on the same device).  Use Dovetail when EAGLE-3 weights
  are unavailable or GPU memory is critically constrained.
- **vs ShadowKV + SpeCache**: both may also use CPU memory bandwidth.  Resolved
  by using SpeCache to pre-stage value tensors so Dovetail's CPU verify pass
  reads from a fast-access buffer.
- **vs Mirror-SD**: Mirror-SD requires ANE availability; Dovetail runs on any
  CPU.  Use Dovetail as the fallback for non-M3 or ANE-congested scenarios.

Provides
--------
  DovetailConfig        — tuning parameters.
  DovetailDraftRunner   — GPU-side draft sequence generator.
  DovetailCPUVerifier   — CPU-side target verification.
  DovetailStats         — per-session acceptance and device-utilisation stats.
  DovetailDecoder       — heterogeneous CPU+GPU orchestrator.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

__all__ = [
    "DovetailConfig",
    "DovetailDraftRunner",
    "DovetailCPUVerifier",
    "DovetailStats",
    "DovetailDecoder",
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()


def _sample(logits: np.ndarray, temperature: float, top_p: float,
            rng: np.random.Generator) -> tuple[int, np.ndarray]:
    probs = _softmax(logits / max(temperature, 1e-8))
    if top_p < 1.0:
        sorted_idx = np.argsort(probs)[::-1]
        cumsum = np.cumsum(probs[sorted_idx])
        cutoff = int(np.searchsorted(cumsum, top_p)) + 1
        mask = np.zeros_like(probs)
        mask[sorted_idx[:cutoff]] = 1.0
        probs = probs * mask
        probs = probs / probs.sum()
    token = int(rng.choice(len(probs), p=probs))
    return token, probs


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class DovetailConfig:
    """Parameters for Dovetail.

    Parameters
    ----------
    gamma:
        Draft speculation depth (number of tokens per GPU draft step).
    temperature:
        Sampling temperature.
    top_p:
        Nucleus sampling threshold.
    """

    gamma: int = 4
    temperature: float = 1.0
    top_p: float = 1.0

    def __post_init__(self) -> None:
        if self.gamma < 1:
            raise ValueError("gamma must be >= 1")
        if self.temperature <= 0:
            raise ValueError("temperature must be > 0")
        if not (0.0 < self.top_p <= 1.0):
            raise ValueError("top_p must be in (0, 1]")


# ---------------------------------------------------------------------------
# DovetailDraftRunner — GPU side
# ---------------------------------------------------------------------------

class DovetailDraftRunner:
    """GPU-side draft model runner.

    Generates γ autoregressive draft tokens using the small draft model.

    Parameters
    ----------
    draft_fn:
        ``(ids: List[int]) -> np.ndarray`` — returns ``(vocab_size,)`` logits.
    config:
        ``DovetailConfig`` used for sampling parameters.
    rng_seed:
        Reproducibility seed.
    """

    def __init__(
        self,
        draft_fn: Callable[[list[int]], np.ndarray],
        config: DovetailConfig,
        rng_seed: int = 0,
    ) -> None:
        if not callable(draft_fn):
            raise TypeError("draft_fn must be callable")
        self._fn = draft_fn
        self._cfg = config
        self._rng = np.random.default_rng(rng_seed)

    def run(
        self, ids: list[int], gamma: int
    ) -> tuple[list[int], list[np.ndarray]]:
        """Produce *gamma* draft tokens autoregressively.

        Returns
        -------
        (tokens, probs_list)
            ``tokens[i]`` is the i-th draft token;
            ``probs_list[i]`` is its probability distribution over the vocab.
        """
        ctx = list(ids)
        tokens: list[int] = []
        probs: list[np.ndarray] = []
        for _ in range(gamma):
            logits = self._fn(ctx)
            tok, p = _sample(logits, self._cfg.temperature,
                             self._cfg.top_p, self._rng)
            tokens.append(tok)
            probs.append(p)
            ctx.append(tok)
        return tokens, probs


# ---------------------------------------------------------------------------
# DovetailCPUVerifier — CPU side
# ---------------------------------------------------------------------------

class DovetailCPUVerifier:
    """CPU-side target model verifier.

    Runs the large target model on the CPU thread pool to verify draft tokens
    produced by the GPU-side draft runner.

    Parameters
    ----------
    target_fn:
        ``(ids: List[int]) -> np.ndarray`` — returns ``(vocab_size,)`` logits.
        In production this function runs on the CPU thread pool.
    config:
        ``DovetailConfig``.
    rng_seed:
        Reproducibility seed.
    """

    def __init__(
        self,
        target_fn: Callable[[list[int]], np.ndarray],
        config: DovetailConfig,
        rng_seed: int = 0,
    ) -> None:
        if not callable(target_fn):
            raise TypeError("target_fn must be callable")
        self._fn = target_fn
        self._cfg = config
        self._rng = np.random.default_rng(rng_seed)

    def verify_one(
        self, ctx: list[int]
    ) -> tuple[int, np.ndarray]:
        """Verify one position: run target model on *ctx*, return (token, probs)."""
        logits = self._fn(ctx)
        tok, probs = _sample(logits, self._cfg.temperature,
                             self._cfg.top_p, self._rng)
        return tok, probs


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

@dataclass
class DovetailStats:
    """Per-session statistics for Dovetail.

    Parameters
    ----------
    total_tokens:
        Total new tokens emitted (accepted + corrections + bonus).
    draft_steps:
        Number of GPU draft batches dispatched.
    accepted_total:
        Draft tokens accepted by the CPU verifier.
    rejected_total:
        Draft tokens rejected; a correction was substituted from the target.
    """

    total_tokens: int = 0
    draft_steps: int = 0
    accepted_total: int = 0
    rejected_total: int = 0

    @property
    def acceptance_rate(self) -> float:
        n = self.accepted_total + self.rejected_total
        return self.accepted_total / n if n > 0 else 0.0

    @property
    def mean_accepted_per_step(self) -> float:
        return self.accepted_total / self.draft_steps if self.draft_steps > 0 else 0.0


# ---------------------------------------------------------------------------
# DovetailDecoder
# ---------------------------------------------------------------------------

class DovetailDecoder:
    """Dovetail heterogeneous CPU+GPU speculative decoder.

    Orchestrates the draft (GPU) and verify (CPU) halves:
    1. GPU draft runner generates γ candidate tokens.
    2. CPU verifier checks each draft token using Leviathan accept/reject.
    3. On full acceptance, the CPU verifier emits one bonus token.
    4. All accepted tokens are appended to the running sequence.

    Parameters
    ----------
    draft_runner:
        ``DovetailDraftRunner`` wrapping the GPU-side small model.
    cpu_verifier:
        ``DovetailCPUVerifier`` wrapping the CPU-side large model.
    config:
        ``DovetailConfig`` (gamma and sampling).
    rng_seed:
        Seed for the accept/reject RNG.
    """

    def __init__(
        self,
        draft_runner: DovetailDraftRunner,
        cpu_verifier: DovetailCPUVerifier,
        config: DovetailConfig | None = None,
        rng_seed: int = 0,
    ) -> None:
        if config is None:
            config = DovetailConfig()
        self._draft = draft_runner
        self._verify = cpu_verifier
        self._cfg = config
        self._rng = np.random.default_rng(rng_seed)

    # ------------------------------------------------------------------

    def generate(
        self,
        input_ids: list[int],
        max_new_tokens: int = 64,
    ) -> tuple[list[int], DovetailStats]:
        """Generate up to *max_new_tokens* tokens.

        Parameters
        ----------
        input_ids:
            Starting token sequence (prompt).
        max_new_tokens:
            Upper bound on new tokens appended.

        Returns
        -------
        (output_ids, stats)
        """
        cfg = self._cfg
        stats = DovetailStats()
        ids = list(input_ids)
        generated = 0

        while generated < max_new_tokens:
            gamma = min(cfg.gamma, max_new_tokens - generated)

            # ── GPU: draft γ tokens ──────────────────────────────────────────
            draft_tokens, draft_probs = self._draft.run(ids, gamma)
            stats.draft_steps += 1

            # ── CPU: verify each draft token (Leviathan criterion) ───────────
            ctx = list(ids)
            accepted: list[int] = []
            rejected = False

            for d_tok, d_probs in zip(draft_tokens, draft_probs, strict=False):
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
