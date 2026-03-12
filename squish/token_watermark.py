"""squish/token_watermark.py

TokenWatermarker — Statistical green-list token watermarking.

Implements the soft red-list / green-list watermarking scheme introduced by
Kirchenbauer et al. (2023).  At generation time, the vocabulary is
partitioned into a *green list* (tokens whose logits receive a positive
``delta`` boost) and a *red list* (tokens left unchanged).  This biases the
model toward green tokens without catastrophically altering the output
distribution.

At detection time, the fraction of green tokens in a sequence is measured and
converted to a Z-score under the null hypothesis that the sequence was
produced without a watermark (binomial with p = ``green_list_fraction``).  A
Z-score above the configured threshold yields a positive detection.

The green-list partition is derived from a seeded random permutation of the
vocabulary::

    rng_seed = mix(config.seed, context_token or 0)
    perm     = rng.permutation(vocab_size)
    green    = perm[:int(vocab_size * green_list_fraction)]

Using the preceding context token as part of the seed makes the partition
context-dependent, which improves statistical robustness against adversarial
token substitution while remaining fully reproducible given the shared seed.

Reference:
    Kirchenbauer et al., "A Watermark for Large Language Models", ICML 2023.
    https://arxiv.org/abs/2301.10226

Example usage::

    import numpy as np
    from squish.token_watermark import WatermarkConfig, TokenWatermarker

    cfg = WatermarkConfig(vocab_size=32000, green_list_fraction=0.5,
                          delta=2.0, seed=42)
    wm  = TokenWatermarker(cfg)

    rng    = np.random.default_rng(7)
    logits = rng.standard_normal(32000).astype(np.float32)

    # Mark logits for generation.
    biased = wm.mark(logits, context_token=1234)

    # Detect watermark in a generated sequence.
    token_ids = np.array([1, 5, 300, 12000, 7], dtype=np.int32)
    result    = wm.detect(token_ids)
    print(result)
"""

from __future__ import annotations

__all__ = ["WatermarkConfig", "DetectionResult", "TokenWatermarker"]

import dataclasses
import math
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Configuration and result types
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class WatermarkConfig:
    """Configuration for :class:`TokenWatermarker`.

    Attributes:
        vocab_size:           Size of the model vocabulary.  Must be >= 2.
        green_list_fraction:  Fraction of vocabulary tokens assigned to the
                              green list in ``(0, 1)``.
        delta:                Logit boost added to green-list tokens at
                              generation time.  Must be >= 0.
        seed:                 Base integer seed for reproducible green-list
                              partitioning.
        z_threshold:          Z-score threshold above which a sequence is
                              classified as watermarked.  Kirchenbauer et al.
                              recommend 4.0 for a false-positive rate of ~
                              3×10⁻⁵.
    """

    vocab_size:          int   = 32000
    green_list_fraction: float = 0.5
    delta:               float = 2.0
    seed:                int   = 42
    z_threshold:         float = 4.0

    def __post_init__(self) -> None:
        if self.vocab_size < 2:
            raise ValueError(
                f"vocab_size must be >= 2, got {self.vocab_size}"
            )
        if not (0.0 < self.green_list_fraction < 1.0):
            raise ValueError(
                f"green_list_fraction must be in (0, 1), "
                f"got {self.green_list_fraction}"
            )
        if self.delta < 0.0:
            raise ValueError(
                f"delta must be >= 0, got {self.delta}"
            )


@dataclasses.dataclass
class DetectionResult:
    """Result of a watermark detection test.

    Attributes:
        z_score:        Z-score of the one-sided binomial test.  Higher values
                        indicate stronger evidence of watermarking.
        p_value:        One-tailed p-value P(Z ≥ z_score) under the null
                        hypothesis (no watermark), computed via the normal
                        approximation using ``math.erfc``.
        n_green_tokens: Number of tokens in the sequence that fall in their
                        respective context-dependent green list.
        n_total_tokens: Total number of tokens evaluated.
        is_watermarked: ``True`` when ``z_score > config.z_threshold``.
    """

    z_score:        float
    p_value:        float
    n_green_tokens: int
    n_total_tokens: int
    is_watermarked: bool


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _hash_mix(a: int, b: int) -> int:
    """Combine two non-negative integers into a deterministic 32-bit seed.

    Uses Knuth's multiplicative hash and a finalisation step for good
    avalanche behaviour.
    """
    x = (a & 0xFFFFFFFF) ^ ((b * 0x9E3779B9) & 0xFFFFFFFF)
    x = ((x ^ (x >> 16)) * 0x45D9F3B) & 0xFFFFFFFF
    x = x ^ (x >> 16)
    return int(x)


# ---------------------------------------------------------------------------
# Watermarker
# ---------------------------------------------------------------------------


class TokenWatermarker:
    """Statistical green-list token watermarker (Kirchenbauer et al. 2023).

    Manages generation-time logit biasing and detection-time hypothesis
    testing for the soft-watermark scheme.

    Args:
        config: :class:`WatermarkConfig` controlling the partition and
                detection parameters.
    """

    def __init__(self, config: WatermarkConfig) -> None:
        self._cfg = config

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def mark(
        self,
        logits: np.ndarray,
        context_token: Optional[int] = None,
    ) -> np.ndarray:
        """Apply the green-list logit bias to *logits*.

        Green-list token positions receive an additive boost of
        ``config.delta``; all other positions are left unchanged.

        Args:
            logits:        Float32 array of shape ``(vocab_size,)``.
            context_token: Optional preceding token index used to derive a
                           context-dependent partition.  When ``None`` the
                           partition is derived from the base seed alone.

        Returns:
            A new float32 array of shape ``(vocab_size,)`` with the bias
            applied.

        Raises:
            ValueError: If *logits* has the wrong shape or if
                        *context_token* is out of range.
        """
        logits = np.asarray(logits, dtype=np.float32)
        if logits.shape != (self._cfg.vocab_size,):
            raise ValueError(
                f"logits must have shape ({self._cfg.vocab_size},), "
                f"got {logits.shape}"
            )
        if context_token is not None and not (
            0 <= context_token < self._cfg.vocab_size
        ):
            raise ValueError(
                f"context_token must be in [0, {self._cfg.vocab_size}), "
                f"got {context_token}"
            )

        mask   = self.green_list(context_token)
        result = logits.copy()
        result[mask] += np.float32(self._cfg.delta)
        return result

    def detect(self, token_ids: np.ndarray) -> DetectionResult:
        """Detect the watermark in a token sequence.

        For each position *i*, the green list is derived using the preceding
        token ``token_ids[i-1]`` as context (``None`` for position 0), then
        the token at position *i* is tested for membership.

        The Z-score is computed as::

            Z = (n_green - n * p) / sqrt(n * p * (1 - p))

        where ``p = config.green_list_fraction``.

        Args:
            token_ids: int32 array of shape ``(seq_len,)`` with vocabulary
                       indices in ``[0, vocab_size)``.

        Returns:
            A :class:`DetectionResult` with the test statistics.

        Raises:
            ValueError: If *token_ids* is not 1-D or contains out-of-range
                        indices.
        """
        token_ids = np.asarray(token_ids, dtype=np.int32)
        if token_ids.ndim != 1:
            raise ValueError(
                f"token_ids must be 1-D (seq_len,), got shape {token_ids.shape}"
            )
        n = token_ids.shape[0]

        if n > 0 and (
            int(token_ids.min()) < 0
            or int(token_ids.max()) >= self._cfg.vocab_size
        ):
            raise ValueError(
                f"All token_ids must be in [0, {self._cfg.vocab_size}); "
                f"got min={int(token_ids.min())}, max={int(token_ids.max())}"
            )

        n_green = 0
        for i in range(n):
            ctx_token = int(token_ids[i - 1]) if i > 0 else None
            mask      = self.green_list(ctx_token)
            if mask[int(token_ids[i])]:
                n_green += 1

        p        = self._cfg.green_list_fraction
        variance = float(n) * p * (1.0 - p)

        if n == 0 or variance < 1e-12:
            z_score = 0.0
            p_value = 0.5
        else:
            z_score = (float(n_green) - float(n) * p) / math.sqrt(variance)
            # One-tailed p-value via the complementary error function.
            p_value = 0.5 * math.erfc(z_score / math.sqrt(2.0))
            # Clamp to [0, 1] to guard against floating-point edge cases.
            p_value = max(0.0, min(1.0, p_value))

        return DetectionResult(
            z_score=z_score,
            p_value=p_value,
            n_green_tokens=n_green,
            n_total_tokens=n,
            is_watermarked=z_score > self._cfg.z_threshold,
        )

    def green_list(self, context_token: Optional[int] = None) -> np.ndarray:
        """Return a boolean mask of shape ``(vocab_size,)`` for green tokens.

        The partition is derived from a seeded random permutation of the
        vocabulary.  The RNG seed is::

            rng_seed = mix(config.seed, context_token or 0)

        so that different context tokens produce different (but reproducible)
        partitions.

        Args:
            context_token: Preceding token index, or ``None`` to use the
                           base seed without context mixing.

        Returns:
            Boolean array of shape ``(vocab_size,)`` where ``True`` indicates
            a green-list token.
        """
        ctx      = int(context_token) if context_token is not None else 0
        rng_seed = _hash_mix(self._cfg.seed, ctx)
        rng      = np.random.default_rng(rng_seed)

        n_green  = int(self._cfg.vocab_size * self._cfg.green_list_fraction)
        perm     = rng.permutation(self._cfg.vocab_size)

        mask            = np.zeros(self._cfg.vocab_size, dtype=bool)
        mask[perm[:n_green]] = True
        return mask
