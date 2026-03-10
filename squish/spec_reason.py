"""SpecReason — Step-Level Speculative Reasoning for LRMs.

Implements the step-level speculative decoding framework from SpecReason
(NeurIPS 2025, arXiv:2504.07891) that operates at reasoning-step granularity
rather than token granularity.

Key insight: LRM chain-of-thought reasoning is semantically tolerant —
a step providing the correct logical insight is acceptable regardless of
exact token sequence.  The small model proposes complete reasoning steps;
the large model evaluates semantic correctness rather than doing exact
token matching.

Reported speedup: 1.4–3.0× over vanilla LRM inference, with additional
8.8–58.0% reduction over token-level speculation (e.g., EAGLE-3) alone.

Also improves accuracy 0.4–9.0% because small-model step attempts guide
the large model to better reasoning paths.

This module provides the orchestration layer — it does not re-implement
the neural models but defines the step proposal/verification protocol,
acceptance logic, and statistics tracking.
"""

from __future__ import annotations

import enum
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


class StepVerdict(enum.Enum):
    """Result of verifying a draft reasoning step."""

    ACCEPT = "accept"       # step semantically correct; accept wholesale
    REJECT = "reject"       # step semantically incorrect; target regenerates
    PARTIAL = "partial"     # step partially correct; accept prefix reasoning


@dataclass
class ReasoningStep:
    """A single reasoning step produced during chain-of-thought.

    Attributes:
        text: The step text (could be a sentence, sub-conclusion, or calculation).
        source: "draft" (small model) or "target" (large model).
        confidence: Draft model's confidence score for this step (0–1).
        tokens_used: Number of tokens this step consumed.
        step_idx: Sequential index within the current reasoning chain.
    """

    text: str
    source: str  # "draft" or "target"
    confidence: float
    tokens_used: int
    step_idx: int = 0

    def __post_init__(self) -> None:
        if self.source not in ("draft", "target"):
            raise ValueError("source must be 'draft' or 'target'")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be in [0, 1]")
        if self.tokens_used < 0:
            raise ValueError("tokens_used must be non-negative")


@dataclass
class SpecReasonConfig:
    """Configuration for SpecReason step-level speculation.

    Args:
        min_acceptance_score: Semantic similarity score threshold for acceptance.
            Steps with score >= threshold are automatically accepted.
        max_step_tokens: Maximum tokens allowed per reasoning step.
        max_draft_steps: Maximum number of consecutive draft steps before
            forcing a target model regeneration cycle.
        confidence_gate: Draft confidence above which verification is skipped
            (ConfSpec integration point — set 0.0 to always verify).
        target_regenerates_on_reject: Whether the target model regenerates
            the step on rejection (True) or simply discards it (False).
        domain: Task domain hint for acceptance calibration
            ("code", "math", "general", etc.).
    """

    min_acceptance_score: float = 0.75
    max_step_tokens: int = 256
    max_draft_steps: int = 4
    confidence_gate: float = 0.0  # 0.0 = always verify (pure SpecReason)
    target_regenerates_on_reject: bool = True
    domain: str = "general"

    def __post_init__(self) -> None:
        if not 0.0 <= self.min_acceptance_score <= 1.0:
            raise ValueError("min_acceptance_score must be in [0, 1]")
        if self.max_step_tokens <= 0:
            raise ValueError("max_step_tokens must be positive")
        if self.max_draft_steps <= 0:
            raise ValueError("max_draft_steps must be positive")
        if not 0.0 <= self.confidence_gate <= 1.0:
            raise ValueError("confidence_gate must be in [0, 1]")


# Type alias for semantic scoring function:  (draft_step, target_context) → float
SemanticScorerFn = Callable[[ReasoningStep, str], float]


def _cosine_sim_scorer(step: ReasoningStep, context: str) -> float:
    """Default semantic scorer: simple token overlap (Jaccard similarity).

    In a real deployment this would be replaced by embedding cosine similarity
    or a dedicated NLI model.  This implementation is intentionally simple to
    remain dependency-free.
    """
    s_tokens = set(step.text.lower().split())
    c_tokens = set(context.lower().split())
    if not s_tokens and not c_tokens:
        return 1.0
    if not s_tokens or not c_tokens:
        return 0.0
    intersection = s_tokens & c_tokens
    union = s_tokens | c_tokens
    return len(intersection) / len(union)


@dataclass
class SpecReasonStats:
    """Runtime statistics for SpecReason."""

    total_steps: int = 0
    draft_steps: int = 0
    target_steps: int = 0
    accepted_draft_steps: int = 0
    rejected_draft_steps: int = 0
    partial_accepted_steps: int = 0
    conf_gate_skipped: int = 0   # verification skipped due to confidence_gate

    total_draft_tokens: int = 0
    total_target_tokens: int = 0

    @property
    def draft_acceptance_rate(self) -> float:
        if self.draft_steps == 0:
            return 0.0
        return self.accepted_draft_steps / self.draft_steps

    @property
    def target_tokens_saved(self) -> int:
        """Tokens the target model did NOT have to generate (accepted from draft)."""
        return self.total_draft_tokens if self.accepted_draft_steps > 0 else 0

    @property
    def estimated_speedup(self) -> float:
        """Estimated speedup vs running only the target model.

        Models: speedup ≈ 1 / (1 - acceptance_rate + acceptance_rate / γ)
        where γ = draft_model_cost_ratio (~0.2 for 1.5B vs 8B).
        """
        acc = self.draft_acceptance_rate
        gamma = 0.2  # cost ratio of small to large model
        if acc >= 1.0:
            return 1.0 / gamma
        return 1.0 / (1.0 - acc + acc * gamma)


class SpecReasonOrchestrator:
    """Manages the SpecReason step-level speculation loop.

    In a real system the orchestrator would call into actual model inference.
    Here it works with callable stubs (draft_fn, target_fn, scorer_fn) so the
    logic can be tested independently of any ML framework.

    Args:
        config: SpecReasonConfig.
        draft_fn: Callable(context: str) → ReasoningStep.  Simulates small model.
        target_fn: Callable(context: str) → ReasoningStep.  Simulates large model.
        scorer_fn: Callable(draft_step, context) → float in [0, 1].
    """

    def __init__(
        self,
        config: SpecReasonConfig,
        draft_fn: Callable[[str], ReasoningStep],
        target_fn: Callable[[str], ReasoningStep],
        scorer_fn: SemanticScorerFn | None = None,
    ) -> None:
        self.config = config
        self._draft_fn = draft_fn
        self._target_fn = target_fn
        self._scorer_fn = scorer_fn or _cosine_sim_scorer
        self._stats = SpecReasonStats()
        self._accepted_steps: list[ReasoningStep] = []

    def generate_step(self, context: str) -> tuple[ReasoningStep, StepVerdict]:
        """Generate the next reasoning step using draft-then-verify.

        Args:
            context: Current reasoning chain context (accepted steps so far).

        Returns:
            Tuple of (accepted_step, verdict).
        """
        draft = self._draft_fn(context)
        draft = ReasoningStep(
            text=draft.text,
            source="draft",
            confidence=draft.confidence,
            tokens_used=draft.tokens_used,
            step_idx=self._stats.total_steps,
        )
        self._stats.draft_steps += 1
        self._stats.total_draft_tokens += draft.tokens_used

        # ConfSpec integration: high-confidence draft skips verification
        if draft.confidence >= self.config.confidence_gate > 0.0:
            self._stats.total_steps += 1
            self._stats.accepted_draft_steps += 1
            self._stats.conf_gate_skipped += 1
            self._accepted_steps.append(draft)
            return draft, StepVerdict.ACCEPT

        # Semantic verification
        score = self._scorer_fn(draft, context)

        if score >= self.config.min_acceptance_score:
            verdict = StepVerdict.ACCEPT
            self._stats.accepted_draft_steps += 1
            accepted = draft
        else:
            verdict = StepVerdict.REJECT
            self._stats.rejected_draft_steps += 1
            if self.config.target_regenerates_on_reject:
                accepted = self._target_fn(context)
                accepted = ReasoningStep(
                    text=accepted.text,
                    source="target",
                    confidence=accepted.confidence,
                    tokens_used=accepted.tokens_used,
                    step_idx=self._stats.total_steps,
                )
                self._stats.target_steps += 1
                self._stats.total_target_tokens += accepted.tokens_used
            else:
                accepted = draft  # keep draft anyway

        self._stats.total_steps += 1
        self._accepted_steps.append(accepted)
        return accepted, verdict

    def generate_chain(
        self, initial_context: str, max_steps: int = 16
    ) -> list[ReasoningStep]:
        """Generate a full reasoning chain up to max_steps.

        Args:
            initial_context: Starting prompt/context.
            max_steps: Cap on reasoning steps.

        Returns:
            List of accepted ReasoningStep objects.
        """
        context = initial_context
        for _ in range(max_steps):
            step, verdict = self.generate_step(context)
            context = context + "\n" + step.text
            if step.text.strip().upper().startswith("FINAL") or "∎" in step.text:
                break
        return self._accepted_steps.copy()

    @property
    def stats(self) -> SpecReasonStats:
        return self._stats

    @property
    def accepted_steps(self) -> list[ReasoningStep]:
        return list(self._accepted_steps)

    def reset(self) -> None:
        self._stats = SpecReasonStats()
        self._accepted_steps.clear()
