"""squish/parallel_sampler.py

ParallelSampler — Best-of-n token sampling with diversity scoring.

Standard single-sample temperature sampling can produce repetitive or
low-diversity outputs when the model's probability mass is spread across many
plausible tokens.  ParallelSampler draws ``n_samples`` candidate tokens from
the temperature-scaled softmax distribution and scores each one by::

    score(i) = log_prob(token_i) + diversity_weight * diversity(token_i)

where ``diversity(token_i)`` is the mean absolute distance (normalised by
``vocab_size``) between ``token_i`` and every other candidate drawn in the
same round.  The candidate with the highest combined score is returned as the
selected token.

This vocabulary-index distance is a simple, compute-free proxy for output
diversity.  It favours candidates that are spread across the vocabulary rather
than clustered in a single high-probability region.

Example usage::

    import numpy as np
    from squish.parallel_sampler import DiversityConfig, ParallelSampler

    cfg     = DiversityConfig(n_samples=8, temperature=0.9,
                              diversity_weight=0.2, seed=0)
    sampler = ParallelSampler(cfg)

    rng    = np.random.default_rng(1)
    logits = rng.standard_normal(32000).astype(np.float32)
    result = sampler.sample(logits)
    print(f"best_token={result.best_token}  diversity={result.diversity_score:.4f}")
"""

from __future__ import annotations

__all__ = ["DiversityConfig", "SampleResult", "ParallelSampler"]

import dataclasses
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class DiversityConfig:
    """Configuration for best-of-n parallel sampling.

    Attributes:
        n_samples:        Number of candidate tokens to draw per step.
                          Must be >= 1.
        temperature:      Softmax temperature applied to logits.  Values in
                          ``(0, 1)`` sharpen the distribution; values ``> 1``
                          flatten it.  Must be > 0.
        diversity_weight: Scalar weight for the diversity bonus in the combined
                          score.  Set to ``0.0`` to recover pure log-prob
                          ranking.  Must be >= 0.
        seed:             Optional integer seed for the internal RNG.  When
                          ``None`` a non-deterministic seed is used.
    """

    n_samples:        int            = 8
    temperature:      float          = 0.8
    diversity_weight: float          = 0.1
    seed:             Optional[int]  = None

    def __post_init__(self) -> None:
        if self.n_samples < 1:
            raise ValueError(
                f"n_samples must be >= 1, got {self.n_samples}"
            )
        if self.temperature <= 0.0:
            raise ValueError(
                f"temperature must be > 0, got {self.temperature}"
            )
        if self.diversity_weight < 0.0:
            raise ValueError(
                f"diversity_weight must be >= 0, got {self.diversity_weight}"
            )


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class SampleResult:
    """Result of a single :meth:`ParallelSampler.sample` call.

    Attributes:
        best_token:      Vocabulary index of the selected candidate.
        best_score:      Combined score
                         ``log_prob + diversity_weight * diversity``
                         of the selected candidate.
        all_tokens:      int32 array of shape ``(n_samples,)`` with all drawn
                         candidate token indices.
        all_probs:       float32 array of shape ``(n_samples,)`` with the
                         sampling probability of each candidate.
        diversity_score: Mean pairwise normalised absolute distance among the
                         candidates in ``[0, 1]``.
    """

    best_token:      int
    best_score:      float
    all_tokens:      np.ndarray
    all_probs:       np.ndarray
    diversity_score: float


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _softmax_1d(x: np.ndarray) -> np.ndarray:
    """Numerically stable softmax for a 1-D array."""
    shifted = x - np.max(x)
    exp_x   = np.exp(shifted)
    return exp_x / np.sum(exp_x)


# ---------------------------------------------------------------------------
# Sampler
# ---------------------------------------------------------------------------


class ParallelSampler:
    """Best-of-n sampler with diversity-augmented scoring.

    Draws *n_samples* candidates from the temperature-scaled softmax
    distribution and returns the one with the highest combined
    ``log_prob + diversity_weight * diversity_score``.

    Args:
        config: :class:`DiversityConfig` controlling sampling behaviour.
    """

    def __init__(self, config: DiversityConfig) -> None:
        self._cfg = config
        self._rng = np.random.default_rng(config.seed)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def sample(self, logits: np.ndarray) -> SampleResult:
        """Sample the best token from a single set of logits.

        Args:
            logits: Float32 array of shape ``(vocab_size,)`` containing raw
                    (unnormalised) model logits.

        Returns:
            A :class:`SampleResult` with the best token and diversity metrics.

        Raises:
            ValueError: If *logits* is not 1-D or if ``vocab_size`` is smaller
                        than ``n_samples``.
        """
        logits = np.asarray(logits, dtype=np.float32)
        if logits.ndim != 1:
            raise ValueError(
                f"logits must be 1-D (vocab_size,), got shape {logits.shape}"
            )
        vocab_size = logits.shape[0]
        if vocab_size < self._cfg.n_samples:
            raise ValueError(
                f"vocab_size ({vocab_size}) must be >= n_samples "
                f"({self._cfg.n_samples})"
            )

        # Temperature-scaled probabilities.
        scaled = logits.astype(np.float64) / self._cfg.temperature
        probs  = _softmax_1d(scaled)  # (vocab_size,)

        # Draw n_samples candidate token indices (with replacement).
        candidates = self._rng.choice(
            vocab_size,
            size=self._cfg.n_samples,
            replace=True,
            p=probs,
        ).astype(np.int32)

        # Probabilities and log-probabilities of the drawn candidates.
        candidate_probs = probs[candidates].astype(np.float32)
        # Add a small epsilon to guard against exact-zero probability entries.
        log_probs = np.log(candidate_probs.astype(np.float64) + 1e-45)

        # Per-candidate diversity: mean pairwise normalised absolute distance.
        diversity_per_candidate = self._per_candidate_diversity(
            candidates, vocab_size
        )
        mean_diversity = float(np.mean(diversity_per_candidate))

        # Combined score: log_prob + diversity_weight * diversity.
        scores     = log_probs + self._cfg.diversity_weight * diversity_per_candidate
        best_idx   = int(np.argmax(scores))
        best_token = int(candidates[best_idx])
        best_score = float(scores[best_idx])

        return SampleResult(
            best_token=best_token,
            best_score=best_score,
            all_tokens=candidates,
            all_probs=candidate_probs,
            diversity_score=mean_diversity,
        )

    def sample_batch(self, logits: np.ndarray) -> np.ndarray:
        """Sample the best token independently for each item in a batch.

        Args:
            logits: Float32 array of shape ``(batch, vocab_size)``.

        Returns:
            int32 array of shape ``(batch,)`` with the best token id per item.

        Raises:
            ValueError: If *logits* is not 2-D.
        """
        logits = np.asarray(logits, dtype=np.float32)
        if logits.ndim != 2:
            raise ValueError(
                f"logits must be 2-D (batch, vocab_size), got shape {logits.shape}"
            )
        batch = logits.shape[0]
        best_tokens = np.empty(batch, dtype=np.int32)
        for i in range(batch):
            result = self.sample(logits[i])
            best_tokens[i] = result.best_token
        return best_tokens

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _per_candidate_diversity(
        self,
        candidates: np.ndarray,
        vocab_size: int,
    ) -> np.ndarray:
        """Compute per-candidate mean pairwise normalised absolute distance.

        For candidate at index ``i``::

            diversity[i] = mean_{j ≠ i} |candidates[i] - candidates[j]|
                           / vocab_size

        Returns:
            float64 array of shape ``(n_samples,)``.
        """
        n = candidates.shape[0]
        if n == 1:
            return np.zeros(1, dtype=np.float64)

        cands = candidates.astype(np.float64)
        # Pairwise absolute differences: shape (n, n).
        diffs = np.abs(cands[:, None] - cands[None, :])
        # Zero out diagonal so self-distance does not inflate the mean.
        np.fill_diagonal(diffs, 0.0)
        row_sums  = diffs.sum(axis=1)                        # (n,)
        diversity = row_sums / float((n - 1) * vocab_size)
        return diversity
