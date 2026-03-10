"""ConfSpec — Confidence-Gated Step-Level Speculative Reasoning.

Implements the confidence-gated step verification mechanism from ConfSpec
(arXiv:2602.18447, Feb 2026) that extends SpecReason by using the draft
model's own confidence signal to avoid unnecessary large-model verification.

Key insight: if the draft model is highly confident about a reasoning step,
the target model will almost certainly agree — run the expensive target-model
verifier only when the draft is uncertain.

Routing rules:
  - confidence >= high_gate  → auto-accept (skip verification entirely)
  - low_gate <= confidence < high_gate → run lightweight verifier
  - confidence < low_gate   → run full target model

Metrics used for confidence:
  - "entropy": normalized entropy of next-token logit distribution (lower = more confident)
  - "top_prob": probability mass on highest-probability token (higher = more confident)
  - "margin": gap between top-1 and top-2 next-token probabilities

ConfSpec improves on SpecReason by eliminating target-model calls for easy
reasoning steps, reserving the full 8B forward pass for genuinely hard ones.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


class ConfidenceMetric(str):
    """Confidence metric name (type-safe string)."""
    ENTROPY = "entropy"
    TOP_PROB = "top_prob"
    MARGIN = "margin"


VALID_METRICS = {ConfidenceMetric.ENTROPY, ConfidenceMetric.TOP_PROB, ConfidenceMetric.MARGIN}


@dataclass
class ConfSpecConfig:
    """Configuration for ConfSpec confidence-gated verification.

    Args:
        high_gate: Confidence score above which auto-accept (no verification).
        low_gate: Confidence score below which full target model is called.
            Steps between low_gate and high_gate use a lightweight verifier.
        metric: Which confidence metric to compute ("entropy", "top_prob", "margin").
        vocab_size: Vocabulary size (needed for entropy normalization).
        ema_alpha: EMA factor for calibrating gate thresholds over time.
        auto_calibrate: If True, update high/low gates based on observed accept rates.
        target_accept_rate: Desired autonomous (no-verify) acceptance rate for calibration.
    """

    high_gate: float = 0.90
    low_gate: float = 0.50
    metric: str = ConfidenceMetric.TOP_PROB
    vocab_size: int = 32000
    ema_alpha: float = 0.05
    auto_calibrate: bool = False
    target_accept_rate: float = 0.60

    def __post_init__(self) -> None:
        if not 0.0 <= self.low_gate < self.high_gate <= 1.0:
            raise ValueError("Must have 0 <= low_gate < high_gate <= 1")
        if self.metric not in VALID_METRICS:
            raise ValueError(f"metric must be one of {VALID_METRICS}")
        if self.vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        if not 0 < self.ema_alpha <= 1.0:
            raise ValueError("ema_alpha must be in (0, 1]")
        if not 0.0 < self.target_accept_rate < 1.0:
            raise ValueError("target_accept_rate must be in (0, 1)")


def compute_confidence(
    logits: np.ndarray,
    metric: str,
    vocab_size: int | None = None,
) -> float:
    """Compute a confidence score in [0, 1] from next-token logits.

    Args:
        logits: (vocab_size,) raw unnormalized logits.
        metric: One of "entropy", "top_prob", "margin".
        vocab_size: Required for entropy normalization.

    Returns:
        Confidence score in [0, 1] (higher = more confident).
    """
    # Softmax
    logits_f = logits.astype(np.float32)
    logits_f -= logits_f.max()
    probs = np.exp(logits_f)
    probs /= probs.sum() + 1e-9

    if metric == ConfidenceMetric.TOP_PROB:
        return float(probs.max())

    if metric == ConfidenceMetric.MARGIN:
        sorted_p = np.sort(probs)[::-1]
        return float(sorted_p[0] - sorted_p[1]) if len(sorted_p) > 1 else 1.0

    if metric == ConfidenceMetric.ENTROPY:
        # Normalized entropy: 1 = fully confident (0 entropy), 0 = fully uncertain
        entropy = -np.sum(probs * np.log(probs + 1e-9))
        v = vocab_size if vocab_size and vocab_size > 1 else len(probs)
        max_entropy = math.log(v)
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0
        return 1.0 - float(normalized_entropy)  # flip: high = confident

    raise ValueError(f"Unknown metric: {metric}")


class VerificationRouting(str):
    """Routing decision for a step."""
    AUTO_ACCEPT = "auto_accept"
    LIGHTWEIGHT = "lightweight"
    FULL_TARGET = "full_target"


@dataclass
class ConfSpecDecision:
    """Decision record for one step verification.

    Attributes:
        confidence: Computed confidence score.
        routing: Which verification path was taken.
        accepted: Whether the step was accepted.
        score: Final semantic score from the verifier (if verification ran).
    """

    confidence: float
    routing: str
    accepted: bool
    score: float | None = None


@dataclass
class ConfSpecStats:
    """Runtime statistics for ConfSpec."""

    total_steps: int = 0
    auto_accepted: int = 0      # skipped verification; accepted
    lightweight_accepted: int = 0
    lightweight_rejected: int = 0
    full_target_accepted: int = 0
    full_target_rejected: int = 0
    calibration_updates: int = 0

    def record(self, decision: ConfSpecDecision) -> None:
        self.total_steps += 1
        if decision.routing == VerificationRouting.AUTO_ACCEPT:
            self.auto_accepted += 1
        elif decision.routing == VerificationRouting.LIGHTWEIGHT:
            if decision.accepted:
                self.lightweight_accepted += 1
            else:
                self.lightweight_rejected += 1
        else:
            if decision.accepted:
                self.full_target_accepted += 1
            else:
                self.full_target_rejected += 1

    @property
    def total_accepted(self) -> int:
        return self.auto_accepted + self.lightweight_accepted + self.full_target_accepted

    @property
    def auto_accept_rate(self) -> float:
        if self.total_steps == 0:
            return 0.0
        return self.auto_accepted / self.total_steps

    @property
    def full_target_rate(self) -> float:
        if self.total_steps == 0:
            return 0.0
        return (self.full_target_accepted + self.full_target_rejected) / self.total_steps

    @property
    def target_calls_saved_fraction(self) -> float:
        """Fraction of steps that did NOT need a full target-model call."""
        if self.total_steps == 0:
            return 0.0
        lightweight = self.lightweight_accepted + self.lightweight_rejected
        return (self.auto_accepted + lightweight) / self.total_steps

    @property
    def overall_acceptance_rate(self) -> float:
        if self.total_steps == 0:
            return 0.0
        return self.total_accepted / self.total_steps

    @property
    def estimated_speedup_vs_always_verify(self) -> float:
        """Speedup from skipping full target calls.

        Each auto-accepted step saves one 8B forward pass.
        Lightweight verifier assumed to be 10% of full call cost.
        """
        if self.total_steps == 0:
            return 1.0
        lightweight_count = self.lightweight_accepted + self.lightweight_rejected
        lightweight_frac = lightweight_count / self.total_steps
        auto_frac = self.auto_accepted / self.total_steps
        full_frac = self.full_target_rate
        # Cost model: auto=0, lightweight=0.1, full=1.0 (relative to one verify call)
        avg_cost = auto_frac * 0.0 + lightweight_frac * 0.1 + full_frac * 1.0
        return 1.0 / max(avg_cost, 1e-6) if avg_cost < 1.0 else 1.0


# Type alias for verifier callables
LightweightVerifierFn = Callable[[str, str], float]  # (step_text, context) → score
FullVerifierFn = Callable[[str, str], float]


def _jaccard_verifier(step_text: str, context: str) -> float:
    s = set(step_text.lower().split())
    c = set(context.lower().split())
    if not s and not c:
        return 1.0
    if not s or not c:
        return 0.0
    return len(s & c) / len(s | c)


class ConfSpecVerifier:
    """Confidence-gated verifier that routes steps to the right verification path.

    Args:
        config: ConfSpecConfig.
        lightweight_fn: Fast verifier (e.g., embedding similarity).
        full_fn: Full large-model verifier.
    """

    def __init__(
        self,
        config: ConfSpecConfig,
        lightweight_fn: LightweightVerifierFn | None = None,
        full_fn: FullVerifierFn | None = None,
    ) -> None:
        self.config = config
        self._lightweight = lightweight_fn or _jaccard_verifier
        self._full = full_fn or _jaccard_verifier
        self._stats = ConfSpecStats()
        self._recent_confidences: list[float] = []

    def verify_step(
        self,
        step_text: str,
        context: str,
        logits: np.ndarray,
    ) -> ConfSpecDecision:
        """Verify a draft step using confidence-gated routing.

        Args:
            step_text: The draft step text.
            context: Current reasoning context.
            logits: (vocab_size,) next-token logits from the draft model.

        Returns:
            ConfSpecDecision with routing, confidence, and acceptance result.
        """
        confidence = compute_confidence(logits, self.config.metric, self.config.vocab_size)
        self._recent_confidences.append(confidence)

        if confidence >= self.config.high_gate:
            decision = ConfSpecDecision(
                confidence=confidence,
                routing=VerificationRouting.AUTO_ACCEPT,
                accepted=True,
                score=confidence,  # use confidence as proxy score
            )
        elif confidence < self.config.low_gate:
            score = self._full(step_text, context)
            accepted = score >= 0.5  # full verifier threshold
            decision = ConfSpecDecision(
                confidence=confidence,
                routing=VerificationRouting.FULL_TARGET,
                accepted=accepted,
                score=score,
            )
        else:
            score = self._lightweight(step_text, context)
            accepted = score >= 0.4  # lightweight threshold (more lenient)
            decision = ConfSpecDecision(
                confidence=confidence,
                routing=VerificationRouting.LIGHTWEIGHT,
                accepted=accepted,
                score=score,
            )

        self._stats.record(decision)

        # Auto-calibrate gates based on observed auto-accept rate
        if self.config.auto_calibrate and len(self._recent_confidences) >= 20:
            self._calibrate_gates()

        return decision

    def _calibrate_gates(self) -> None:
        """Adjust high/low gates to approach target_accept_rate."""
        current_rate = self._stats.auto_accept_rate
        target = self.config.target_accept_rate
        if abs(current_rate - target) < 0.05:
            return  # close enough
        alpha = self.config.ema_alpha
        if current_rate < target:
            # Too few auto-accepts: lower the high gate
            self.config.high_gate = max(
                self.config.low_gate + 0.05,
                self.config.high_gate - alpha * 0.1,
            )
        else:
            # Too many auto-accepts: raise the high gate
            self.config.high_gate = min(
                1.0,
                self.config.high_gate + alpha * 0.1,
            )
        self._stats.calibration_updates += 1
        self._recent_confidences.clear()

    @property
    def stats(self) -> ConfSpecStats:
        return self._stats

    def reset(self) -> None:
        self._stats = ConfSpecStats()
        self._recent_confidences.clear()
