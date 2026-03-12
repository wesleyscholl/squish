"""squish/safety_layer.py

SafetyClassifier — Inline token-sequence safety classifier for LLM inference.

Deploying generative models in production requires fast, low-overhead content
moderation that does not require a second full forward pass through the model.
SafetyClassifier implements a lightweight bag-of-tokens linear classifier that
operates directly on token IDs or on the model's output logit distribution.

The classifier maintains a single weight matrix ``W`` of shape
``(vocab_size, n_categories)`` initialised from a seeded RNG.  For a sequence
of token IDs the score is the mean of ``W[token_ids]`` across all positions,
followed by a softmax to produce per-category probabilities.  The first
category is always ``"safe"``; remaining categories represent unsafe content
types (``"violent"``, ``"sexual"``, ``"hate"``; additional categories are
labelled ``"category_N"``).

Two entry points are provided:

* :meth:`score` — operates on a sequence of integer token IDs.
* :meth:`score_logits` — operates on a next-token logit distribution,
  computing a probability-weighted average over the vocabulary so that
  likely-but-not-yet-sampled tokens also contribute to the safety decision.

Example usage::

    import numpy as np
    from squish.safety_layer import SafetyConfig, SafetyClassifier

    cfg        = SafetyConfig(vocab_size=32000, n_categories=4, threshold=0.5)
    classifier = SafetyClassifier(cfg)

    token_ids = np.array([101, 5432, 17, 8001], dtype=np.int32)
    result    = classifier.score(token_ids)
    print(result.is_safe, result.score, result.category)
"""

from __future__ import annotations

__all__ = ["SafetyConfig", "SafetyResult", "SafetyClassifier"]

from dataclasses import dataclass
from typing import Optional

import numpy as np


# Default category labels in slot order.
_DEFAULT_CATEGORIES: list[str] = ["safe", "violent", "sexual", "hate"]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class SafetyConfig:
    """Configuration for :class:`SafetyClassifier`.

    Attributes:
        vocab_size:   Size of the model vocabulary.
        n_categories: Number of safety categories.  The first is always
                      ``"safe"``; the rest represent unsafe content types.
                      Must be >= 2.
        threshold:    Probability threshold for the ``"safe"`` category.
                      A sequence is classified safe iff
                      ``safe_score >= threshold``.  Must be in ``(0, 1]``.
        seed:         RNG seed for reproducible weight initialisation.
    """

    vocab_size: int = 32000
    n_categories: int = 4
    threshold: float = 0.5
    seed: int = 42

    def __post_init__(self) -> None:
        if self.vocab_size < 1:
            raise ValueError(f"vocab_size must be >= 1, got {self.vocab_size}")
        if self.n_categories < 2:
            raise ValueError(
                f"n_categories must be >= 2 (safe + at least one unsafe), "
                f"got {self.n_categories}"
            )
        if not (0.0 < self.threshold <= 1.0):
            raise ValueError(
                f"threshold must be in (0, 1], got {self.threshold}"
            )


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------


@dataclass
class SafetyResult:
    """Output of a :class:`SafetyClassifier` evaluation.

    Attributes:
        is_safe:         ``True`` iff the safe-category score >= threshold.
        score:           Probability assigned to the ``"safe"`` category,
                         in ``[0, 1]``.
        category:        ``"safe"`` when safe; otherwise the dominant unsafe
                         category label (e.g. ``"violent"``).
        category_scores: Float32 array of shape ``(n_categories,)`` with
                         softmax probabilities for every category.
    """

    is_safe: bool
    score: float
    category: str
    category_scores: np.ndarray


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _softmax_1d(x: np.ndarray) -> np.ndarray:
    """Numerically stable softmax for a 1-D float64 array."""
    x_shifted = x - np.max(x)
    exp_x = np.exp(x_shifted)
    return exp_x / np.sum(exp_x)


def _build_category_labels(n_categories: int) -> list[str]:
    """Build a category label list of length *n_categories*.

    The first entry is always ``"safe"``.  Known unsafe labels fill positions
    1–3; any overflow positions are labelled ``"category_N"``.
    """
    labels: list[str] = []
    for i in range(n_categories):
        if i < len(_DEFAULT_CATEGORIES):
            labels.append(_DEFAULT_CATEGORIES[i])
        else:
            labels.append(f"category_{i}")
    return labels


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------


class SafetyClassifier:
    """Inline token-sequence safety classifier.

    Uses a bag-of-tokens linear projection followed by softmax to produce
    per-category safety scores.  No second forward pass through the main
    model is required.

    The weight matrix ``W`` has shape ``(vocab_size, n_categories)`` and is
    initialised with scale ``1 / sqrt(vocab_size)`` from a seeded RNG.

    Args:
        config: A :class:`SafetyConfig` controlling model dimensions and the
                classification threshold.
    """

    def __init__(self, config: SafetyConfig) -> None:
        self._cfg = config
        self._categories: list[str] = _build_category_labels(config.n_categories)

        # Weight matrix: each row is a safety embedding for one token ID.
        rng = np.random.default_rng(config.seed)
        scale = 1.0 / np.sqrt(float(config.vocab_size))
        self._W: np.ndarray = (
            rng.standard_normal((config.vocab_size, config.n_categories)).astype(np.float32)
            * scale
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def score(self, token_ids: np.ndarray) -> SafetyResult:
        """Classify a sequence of token IDs.

        The safety score is computed as the mean of ``W[token_ids]`` over the
        sequence, followed by a softmax over categories.

        Args:
            token_ids: 1-D integer array of shape ``(seq,)`` with values in
                       ``[0, vocab_size)``.  Must be non-empty.

        Returns:
            A :class:`SafetyResult` with per-category scores and a safety
            verdict based on the configured threshold.

        Raises:
            ValueError: If *token_ids* is not 1-D, is empty, or contains
                        values outside ``[0, vocab_size)``.
        """
        token_ids = np.asarray(token_ids, dtype=np.int32)
        if token_ids.ndim != 1:
            raise ValueError(
                f"token_ids must be 1-D, got shape {token_ids.shape}"
            )
        if token_ids.size == 0:
            raise ValueError("token_ids must be non-empty")
        mask = (token_ids < 0) | (token_ids >= self._cfg.vocab_size)
        if np.any(mask):
            bad = token_ids[mask]
            raise ValueError(
                f"token_ids contains out-of-range values: {bad[:5].tolist()}; "
                f"vocab_size={self._cfg.vocab_size}"
            )
        # Bag-of-tokens: mean embedding of all token positions.
        raw_scores = np.mean(self._W[token_ids], axis=0).astype(np.float64)
        return self._result_from_raw(raw_scores)

    def score_logits(self, logits: np.ndarray) -> SafetyResult:
        """Classify from a next-token logit distribution.

        Computes ``score = softmax(logits) @ W``, so tokens with high
        probability in the distribution contribute proportionally to the
        safety score.  When a 2-D input is provided the per-position
        probability distributions are averaged before projection.

        Args:
            logits: Float array of shape ``(vocab_size,)`` or
                    ``(seq, vocab_size)``.

        Returns:
            A :class:`SafetyResult`.

        Raises:
            ValueError: If *logits* is not 1-D or 2-D, or if the last
                        dimension does not equal ``vocab_size``.
        """
        logits = np.asarray(logits, dtype=np.float32)
        if logits.ndim == 1:
            logits = logits[np.newaxis, :]  # (1, vocab_size)
        if logits.ndim != 2:
            raise ValueError(
                f"logits must be 1-D or 2-D, got shape {logits.shape}"
            )
        if logits.shape[-1] != self._cfg.vocab_size:
            raise ValueError(
                f"logits last dim ({logits.shape[-1]}) must equal "
                f"vocab_size ({self._cfg.vocab_size})"
            )
        # Numerically stable softmax per position, then average over seq.
        shifted = logits - logits.max(axis=-1, keepdims=True)
        exp_l = np.exp(shifted)
        probs = exp_l / exp_l.sum(axis=-1, keepdims=True)  # (seq, vocab_size)
        mean_probs = np.mean(probs, axis=0)                 # (vocab_size,)

        raw_scores = (mean_probs @ self._W).astype(np.float64)  # (n_categories,)
        return self._result_from_raw(raw_scores)

    def update_threshold(self, threshold: float) -> None:
        """Update the safety classification threshold in place.

        Args:
            threshold: New threshold value in ``(0, 1]``.

        Raises:
            ValueError: If *threshold* is outside ``(0, 1]``.
        """
        if not (0.0 < threshold <= 1.0):
            raise ValueError(
                f"threshold must be in (0, 1], got {threshold}"
            )
        self._cfg.threshold = threshold

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _result_from_raw(self, raw_scores: np.ndarray) -> SafetyResult:
        """Convert raw linear scores to a :class:`SafetyResult`."""
        category_scores = _softmax_1d(raw_scores)
        safe_score = float(category_scores[0])
        is_safe = safe_score >= self._cfg.threshold
        if is_safe:
            category = "safe"
        else:
            # Dominant unsafe category: argmax over the non-safe slots.
            unsafe_idx = int(np.argmax(category_scores[1:])) + 1
            category = self._categories[unsafe_idx]
        return SafetyResult(
            is_safe=is_safe,
            score=safe_score,
            category=category,
            category_scores=category_scores.astype(np.float32),
        )
