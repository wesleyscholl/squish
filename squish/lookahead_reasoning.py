"""LookaheadReasoning — Parallel Step Verification via Draft Model.

Implements the multi-step batched verification approach from
LookaheadReasoning (NeurIPS 2025, github.com/hao-ai-lab/LookaheadReasoning),
explicitly supporting the Qwen3 model family.

Key difference from SpecReason:
  - LookaheadReasoning generates N draft steps in parallel then verifies all N
    in a single batched target-model call.
  - SpecReason generates one step at a time and verifies sequentially.
  - LookaheadReasoning trades higher per-step latency for more parallelism.

The cyclical process:
  1. Draft model generates `lookahead_k` future reasoning steps.
  2. Target model processes all proposals in one batched forward pass.
  3. Accept the longest prefix of semantically correct steps.
  4. Advance context by accepted steps; repeat.

This module provides the orchestration layer and statistics tracking.
"""

from __future__ import annotations

import enum
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class LookaheadConfig:
    """Configuration for LookaheadReasoning.

    Args:
        lookahead_k: Number of draft steps to generate per lookahead cycle.
        min_acceptance_score: Semantic score threshold for step acceptance.
        max_step_tokens: Maximum tokens per step.
        greedy_prefix_accept: If True, accept the longest sequential prefix
            of good steps (stops at first rejection within the batch).
            If False, accept all individually good steps.
        model_family: Hint for acceptance calibration ("qwen3", "llama", etc.).
    """

    lookahead_k: int = 4
    min_acceptance_score: float = 0.70
    max_step_tokens: int = 256
    greedy_prefix_accept: bool = True
    model_family: str = "qwen3"

    def __post_init__(self) -> None:
        if self.lookahead_k <= 0:
            raise ValueError("lookahead_k must be positive")
        if not 0.0 <= self.min_acceptance_score <= 1.0:
            raise ValueError("min_acceptance_score must be in [0, 1]")
        if self.max_step_tokens <= 0:
            raise ValueError("max_step_tokens must be positive")


@dataclass
class LookaheadStep:
    """A single step candidate in a lookahead batch.

    Attributes:
        text: Step text.
        source: "draft" or "target".
        confidence: Draft model self-confidence.
        tokens_used: Token count.
        batch_position: Position within the lookahead batch (0-indexed).
        accepted: Whether this step was accepted.
        score: Semantic verification score (assigned after target verification).
    """

    text: str
    source: str
    confidence: float
    tokens_used: int
    batch_position: int = 0
    accepted: bool = False
    score: float = 0.0

    def __post_init__(self) -> None:
        if self.source not in ("draft", "target"):
            raise ValueError("source must be 'draft' or 'target'")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be in [0, 1]")


@dataclass
class LookaheadBatch:
    """A batch of N draft steps generated in one lookahead cycle."""

    steps: list[LookaheadStep]
    context_at_start: str
    n_accepted: int = 0

    @property
    def n_steps(self) -> int:
        return len(self.steps)

    @property
    def accepted_steps(self) -> list[LookaheadStep]:
        return [s for s in self.steps if s.accepted]

    @property
    def acceptance_rate(self) -> float:
        if not self.steps:
            return 0.0
        return self.n_accepted / len(self.steps)

    @property
    def total_tokens(self) -> int:
        return sum(s.tokens_used for s in self.steps)


@dataclass
class LookaheadStats:
    """Runtime statistics for LookaheadReasoning."""

    total_cycles: int = 0
    total_draft_steps: int = 0
    total_accepted_steps: int = 0
    total_rejected_steps: int = 0
    total_target_batches: int = 0
    total_draft_tokens: int = 0
    total_accepted_tokens: int = 0

    @property
    def steps_per_cycle(self) -> float:
        if self.total_cycles == 0:
            return 0.0
        return self.total_draft_steps / self.total_cycles

    @property
    def acceptance_rate(self) -> float:
        if self.total_draft_steps == 0:
            return 0.0
        return self.total_accepted_steps / self.total_draft_steps

    @property
    def batch_efficiency(self) -> float:
        """Accepted steps per target-model batch call."""
        if self.total_target_batches == 0:
            return 0.0
        return self.total_accepted_steps / self.total_target_batches

    @property
    def estimated_speedup(self) -> float:
        """Speedup over sequential target-model generation.

        Each lookahead cycle uses 1 batched target call to verify k steps.
        Accepted steps per cycle = acceptance_rate * k.
        Cost model: 1 target call per cycle regardless of k.
        Baseline: 1 target call per step.
        Speedup ≈ accepted_per_cycle.
        """
        if self.total_cycles == 0:
            return 1.0
        accepted_per_cycle = self.total_accepted_steps / self.total_cycles
        return max(1.0, accepted_per_cycle)


# Callable type aliases
DraftFn = Callable[[str], LookaheadStep]
BatchVerifyFn = Callable[[list[LookaheadStep], str], list[float]]


def _default_batch_verifier(
    steps: list[LookaheadStep], context: str
) -> list[float]:
    """Default batch verifier using token Jaccard similarity."""
    c_tokens = set(context.lower().split())
    scores = []
    for step in steps:
        s_tokens = set(step.text.lower().split())
        if not s_tokens and not c_tokens:
            scores.append(1.0)
        elif not s_tokens or not c_tokens:
            scores.append(0.0)
        else:
            union = s_tokens | c_tokens
            scores.append(len(s_tokens & c_tokens) / len(union))
    return scores


class LookaheadReasoningEngine:
    """Orchestrates the LookaheadReasoning cyclical draft-verify loop.

    Args:
        config: LookaheadConfig.
        draft_fn: Callable(context) → LookaheadStep.
        batch_verify_fn: Callable(steps, context) → List[float] (scores).
            Called once per cycle with all k draft steps.
    """

    def __init__(
        self,
        config: LookaheadConfig,
        draft_fn: DraftFn,
        batch_verify_fn: BatchVerifyFn | None = None,
    ) -> None:
        self.config = config
        self._draft_fn = draft_fn
        self._verify_fn = batch_verify_fn or _default_batch_verifier
        self._stats = LookaheadStats()
        self._all_accepted: list[LookaheadStep] = []

    def run_cycle(self, context: str) -> LookaheadBatch:
        """Run one lookahead cycle: generate k drafts, verify in batch, accept prefix.

        Args:
            context: Current reasoning chain as a string.

        Returns:
            LookaheadBatch with acceptance annotations.
        """
        # Step 1: Generate k draft steps
        draft_steps: list[LookaheadStep] = []
        ctx = context
        for pos in range(self.config.lookahead_k):
            step = self._draft_fn(ctx)
            draft_steps.append(
                LookaheadStep(
                    text=step.text,
                    source="draft",
                    confidence=step.confidence,
                    tokens_used=step.tokens_used,
                    batch_position=pos,
                )
            )
            # Extend context speculatively for next draft step
            ctx = ctx + "\n" + step.text

        # Step 2: Batch verification (one target model call)
        scores = self._verify_fn(draft_steps, context)

        # Step 3: Accept prefix (or all individually good steps)
        n_accepted = 0
        for step, score in zip(draft_steps, scores, strict=False):
            step.score = score
            if score >= self.config.min_acceptance_score:
                step.accepted = True
                n_accepted += 1
                if not self.config.greedy_prefix_accept:
                    continue
            else:
                # In greedy mode, stop accepting at first rejection
                if self.config.greedy_prefix_accept:
                    break

        batch = LookaheadBatch(
            steps=draft_steps,
            context_at_start=context,
            n_accepted=n_accepted,
        )

        # Update stats
        self._stats.total_cycles += 1
        self._stats.total_draft_steps += len(draft_steps)
        self._stats.total_accepted_steps += n_accepted
        self._stats.total_rejected_steps += len(draft_steps) - n_accepted
        self._stats.total_target_batches += 1
        self._stats.total_draft_tokens += batch.total_tokens
        self._stats.total_accepted_tokens += sum(
            s.tokens_used for s in batch.accepted_steps
        )
        self._all_accepted.extend(batch.accepted_steps)

        return batch

    def generate_chain(
        self, initial_context: str, max_steps: int = 32
    ) -> list[LookaheadStep]:
        """Generate a full reasoning chain.

        Args:
            initial_context: Starting prompt/context.
            max_steps: Total step cap (across all cycles).

        Returns:
            List of all accepted steps.
        """
        context = initial_context
        total_accepted = 0

        while total_accepted < max_steps:
            batch = self.run_cycle(context)
            for step in batch.accepted_steps:
                context = context + "\n" + step.text
                total_accepted += 1
                if step.text.strip().upper().startswith("FINAL") or "∎" in step.text:
                    return self._all_accepted.copy()
            # If no step was accepted this cycle, break to avoid infinite loop
            if batch.n_accepted == 0:
                break

        return self._all_accepted.copy()

    @property
    def stats(self) -> LookaheadStats:
        return self._stats

    @property
    def all_accepted_steps(self) -> list[LookaheadStep]:
        return list(self._all_accepted)

    def reset(self) -> None:
        self._stats = LookaheadStats()
        self._all_accepted.clear()
