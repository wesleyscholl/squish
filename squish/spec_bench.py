"""SpecBench — SpecBench CI evaluation harness for speculative decoding.

SpecBench (Xia et al., ICLR 2024) measures speculative decode performance
across six task categories: translation, summarisation, QA, math reasoning,
RAG, and code.  This module provides a lightweight runner that measures
acceptance rate, mean accepted tokens per step, and end-to-end throughput.

Reference:
    Xia et al., "SpecBench: A Comprehensive Benchmark for Speculative Decoding
    Systems", ICLR 2024.  https://arxiv.org/abs/2401.14401

Usage::

    from squish.spec_bench import SpecBenchRunner, SpecBenchTask

    runner = SpecBenchRunner()
    task   = SpecBenchTask("math", prompts=["Solve: 2+2=", "What is pi?"])
    result = runner.run_task(task, draft_fn=my_draft, target_fn=my_target)
    print(result.mean_accepted_per_step)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

__all__ = [
    "SpecBenchTask",
    "SpecBenchResult",
    "SpecBenchRunner",
    "SpecBenchStats",
]


# ---------------------------------------------------------------------------
# Task definition
# ---------------------------------------------------------------------------


@dataclass
class SpecBenchTask:
    """A named collection of prompts for one SpecBench task category.

    Parameters
    ----------
    task_name : str
        Unique name for this task (e.g. ``"math"``, ``"code"``).
    prompts : list[str]
        The prompt strings to evaluate.
    category : str
        High-level category label.  Defaults to ``"general"``.
    """

    task_name: str
    prompts: List[str]
    category: str = "general"

    def __post_init__(self) -> None:
        if not self.task_name:
            raise ValueError("task_name must not be empty.")
        if not self.prompts:
            raise ValueError("prompts must contain at least one entry.")

    @property
    def n_prompts(self) -> int:
        """Number of prompts in this task."""
        return len(self.prompts)


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------


@dataclass
class SpecBenchResult:
    """Measured performance for a single :class:`SpecBenchTask`.

    Parameters
    ----------
    task_name : str
        Name of the evaluated task.
    n_prompts : int
        Number of prompts evaluated.
    total_tokens : int
        Total draft tokens proposed (``gamma * n_prompts``).
    total_accepted : int
        Total draft tokens accepted by the target model.
    total_steps : int
        Total speculative steps executed (one per prompt in this harness).
    latency_ms_total : float
        Wall-clock time for the entire task evaluation in milliseconds.
    """

    task_name: str
    n_prompts: int
    total_tokens: int
    total_accepted: int
    total_steps: int
    latency_ms_total: float

    @property
    def acceptance_rate(self) -> float:
        """Fraction of draft tokens accepted by the target model."""
        if self.total_tokens == 0:
            return 0.0
        return self.total_accepted / self.total_tokens

    @property
    def mean_accepted_per_step(self) -> float:
        """Average number of tokens accepted per speculative step."""
        if self.total_steps == 0:
            return 0.0
        return self.total_accepted / self.total_steps

    @property
    def tokens_per_second(self) -> float:
        """End-to-end throughput: total draft tokens divided by elapsed time."""
        if self.latency_ms_total <= 0.0:
            return 0.0
        return self.total_tokens / (self.latency_ms_total / 1000.0)


# ---------------------------------------------------------------------------
# Aggregate statistics
# ---------------------------------------------------------------------------


@dataclass
class SpecBenchStats:
    """Summary statistics across a full SpecBench suite run.

    Parameters
    ----------
    tasks_run : int
        Number of tasks evaluated.
    total_prompts : int
        Total prompts evaluated across all tasks.
    overall_acceptance_rate : float
        Weighted acceptance rate across all tasks.
    """

    tasks_run: int
    total_prompts: int
    overall_acceptance_rate: float

    @property
    def grade(self) -> str:
        """Letter grade based on overall acceptance rate.

        Returns ``"A"`` (≥ 0.8), ``"B"`` (≥ 0.6), ``"C"`` (≥ 0.4),
        or ``"D"`` (< 0.4).
        """
        if self.overall_acceptance_rate >= 0.8:
            return "A"
        if self.overall_acceptance_rate >= 0.6:
            return "B"
        if self.overall_acceptance_rate >= 0.4:
            return "C"
        return "D"


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

# Type aliases for the draft and target callables.
DraftFn = Callable[[str], List[int]]
TargetFn = Callable[[str, List[int]], List[bool]]


class SpecBenchRunner:
    """Lightweight SpecBench CI harness for speculative decoding systems.

    Simulates speculative decoding by calling a user-supplied *draft function*
    to propose ``gamma`` tokens, then a *target function* to verify them.
    Results are aggregated per task into :class:`SpecBenchResult` objects.

    Parameters
    ----------
    gamma : int
        Number of draft tokens proposed per speculative step.  Must be >= 1.
    temperature : float
        Sampling temperature hint passed through to draft/target functions
        (not used internally; available for callables that inspect it via
        a closure).  Must be > 0.

    Examples
    --------
    >>> def draft(prompt):
    ...     return [42] * 4          # always draft token id 42
    >>> def target(prompt, draft_tokens):
    ...     return [True, True, False, False]  # accept first two
    >>> runner = SpecBenchRunner(gamma=4)
    >>> task   = SpecBenchTask("qa", prompts=["What is 2+2?"])
    >>> result = runner.run_task(task, draft_fn=draft, target_fn=target)
    >>> result.acceptance_rate   # 2 / 4 == 0.5
    0.5
    """

    def __init__(self, gamma: int = 4, temperature: float = 1.0) -> None:
        if gamma < 1:
            raise ValueError(f"gamma must be >= 1; got {gamma}")
        if temperature <= 0.0:
            raise ValueError(f"temperature must be > 0; got {temperature}")
        self._gamma = gamma
        self._temperature = temperature

    # ── Task evaluation ───────────────────────────────────────────────────────

    def run_task(
        self,
        task: SpecBenchTask,
        draft_fn: DraftFn,
        target_fn: TargetFn,
    ) -> SpecBenchResult:
        """Evaluate a single :class:`SpecBenchTask`.

        For each prompt in ``task.prompts``, this method:

        1. Calls ``draft_fn(prompt)`` to obtain a list of ``gamma`` draft
           token ids.
        2. Calls ``target_fn(prompt, draft_tokens)`` to obtain a list of
           ``gamma`` booleans indicating which tokens were accepted.
        3. Accumulates per-token accept/reject counts.

        Parameters
        ----------
        task : SpecBenchTask
            Task to evaluate.
        draft_fn : callable
            ``draft_fn(prompt: str) -> list[int]`` — returns exactly
            ``gamma`` draft token ids.
        target_fn : callable
            ``target_fn(prompt: str, draft_tokens: list[int]) -> list[bool]``
            — returns one boolean per draft token.

        Returns
        -------
        SpecBenchResult
            Aggregated metrics for this task.
        """
        total_tokens = 0
        total_accepted = 0
        total_steps = 0

        t_start = time.perf_counter()

        for prompt in task.prompts:
            draft_tokens: List[int] = draft_fn(prompt)
            accepted: List[bool] = target_fn(prompt, draft_tokens)

            n_draft = len(draft_tokens)
            n_accepted = sum(1 for a in accepted if a)

            total_tokens += n_draft
            total_accepted += n_accepted
            total_steps += 1

        latency_ms = (time.perf_counter() - t_start) * 1000.0

        return SpecBenchResult(
            task_name=task.task_name,
            n_prompts=task.n_prompts,
            total_tokens=total_tokens,
            total_accepted=total_accepted,
            total_steps=total_steps,
            latency_ms_total=latency_ms,
        )

    # ── Default task suite ────────────────────────────────────────────────────

    @staticmethod
    def default_tasks() -> List[SpecBenchTask]:
        """Return the canonical 6-task SpecBench suite with synthetic prompts.

        Returns
        -------
        list[SpecBenchTask]
            Six tasks: translation, summarization, qa, math, rag, code.
        """
        return [
            SpecBenchTask(
                task_name="translation",
                prompts=[
                    "Translate to French: The quick brown fox jumps over the lazy dog.",
                    "Translate to Spanish: Artificial intelligence is transforming the world.",
                ],
                category="translation",
            ),
            SpecBenchTask(
                task_name="summarization",
                prompts=[
                    (
                        "Summarize the following: Large language models have revolutionised "
                        "natural language processing by achieving state-of-the-art results "
                        "across many benchmarks."
                    ),
                    (
                        "Summarize the following: Speculative decoding uses a fast draft "
                        "model to propose tokens that are then verified by the target model, "
                        "achieving significant throughput gains."
                    ),
                ],
                category="summarization",
            ),
            SpecBenchTask(
                task_name="qa",
                prompts=[
                    "Question: What is the capital of France? Answer:",
                    "Question: Who wrote the theory of relativity? Answer:",
                ],
                category="qa",
            ),
            SpecBenchTask(
                task_name="math",
                prompts=[
                    "Solve step by step: If x^2 + 5x + 6 = 0, what are the values of x?",
                    "Compute the derivative of f(x) = 3x^3 - 2x^2 + 7x - 4.",
                ],
                category="math",
            ),
            SpecBenchTask(
                task_name="rag",
                prompts=[
                    (
                        "Context: The Eiffel Tower is located in Paris, France, "
                        "and was built in 1889. Question: Where is the Eiffel Tower? Answer:"
                    ),
                    (
                        "Context: Water boils at 100 degrees Celsius at sea level. "
                        "Question: At what temperature does water boil? Answer:"
                    ),
                ],
                category="rag",
            ),
            SpecBenchTask(
                task_name="code",
                prompts=[
                    "Write a Python function that reverses a string without using slicing.",
                    "Implement binary search in Python that returns the index of the target.",
                ],
                category="code",
            ),
        ]

    # ── Suite runner ──────────────────────────────────────────────────────────

    def run_suite(
        self,
        draft_fn: DraftFn,
        target_fn: TargetFn,
    ) -> Dict[str, SpecBenchResult]:
        """Run all default tasks and return results keyed by task name.

        Parameters
        ----------
        draft_fn : callable
            Draft token generator.
        target_fn : callable
            Target model verifier.

        Returns
        -------
        dict[str, SpecBenchResult]
            Mapping from task name to its :class:`SpecBenchResult`.
        """
        results: Dict[str, SpecBenchResult] = {}
        for task in self.default_tasks():
            results[task.task_name] = self.run_task(task, draft_fn, target_fn)
        return results

    @staticmethod
    def overall_acceptance_rate(results: Dict[str, SpecBenchResult]) -> float:
        """Compute the token-weighted mean acceptance rate across all results.

        Parameters
        ----------
        results : dict[str, SpecBenchResult]
            Output of :meth:`run_suite` or a subset thereof.

        Returns
        -------
        float
            Weighted mean acceptance rate, or ``0.0`` if no tokens were
            proposed.
        """
        total_tokens = sum(r.total_tokens for r in results.values())
        total_accepted = sum(r.total_accepted for r in results.values())
        if total_tokens == 0:
            return 0.0
        return total_accepted / total_tokens

    def suite_stats(
        self,
        results: Dict[str, SpecBenchResult],
    ) -> SpecBenchStats:
        """Build a :class:`SpecBenchStats` summary from a suite run.

        Parameters
        ----------
        results : dict[str, SpecBenchResult]
            Output of :meth:`run_suite`.

        Returns
        -------
        SpecBenchStats
        """
        return SpecBenchStats(
            tasks_run=len(results),
            total_prompts=sum(r.n_prompts for r in results.values()),
            overall_acceptance_rate=self.overall_acceptance_rate(results),
        )
