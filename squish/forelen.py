"""
squish/forelen.py

ForeLen — Entropy-Guided Output Length Prediction.

Based on:
  "ForeLen: Better LLM Serving via Output Length Prediction"
  ICLR 2026 — arXiv:2602.11812

Problem
-------
Existing output-length predictors rely on auxiliary models (BERT, separate
fine-tuned classifiers) that:
  - Incur high overhead (a whole model call per request)
  - Generalise poorly across domains
  - Fail in stochastic "one-to-many" sampling scenarios where the same prompt
    can yield answers of vastly different lengths

ForeLen introduces two lightweight mechanisms that *reuse the main model's
own hidden states* — zero auxiliary model overhead:

1. EGTP (Entropy-Guided Token Pooling)
   - Reads per-token entropy of the model's attention (or logit) distribution
     from the existing prefill forward pass.
   - Token entropy directly correlates with how "uncertain" the model is, and
     uncertainty correlates with output length.
   - Pools entropy values over a configurable number of entropy bins and feeds
     the histogram to a linear probe → length bucket prediction.
   - Zero additional forward passes; the entropy is already computed.

2. PLP (Progressive Length Prediction)
   - Updates the length estimate at each decode step as new tokens are
     generated and new entropy values are observed.
   - Handles the stochastic case: if the model starts diverging from the
     initial prediction, PLP corrects in real time.

Results (vs. TRAIL baseline, ICLR 2025)
-----------------------------------------
- MAE reduced by 29.16% over best baseline
- In the Long Sequence workload, halves job completion time vs. TRAIL
- Padding ratio reduced to 0.18 (vs. TRAIL's 0.51) — ~3× improvement

Conflict notes
--------------
- **Supersedes TRAIL** when ForeLen code is available; TRAIL (trail.py) is
  the fallback deployed today.
- **Zero conflict with any inference method** — ForeLen is a *scheduler hint*;
  it only affects KV pre-allocation and BucketServe routing.
- **Synergy with BucketServe**: EGTP predicts the length bucket that
  BucketServe uses for batch composition.

Provides
--------
  ForelenConfig     — hyper-parameters for EGTP and PLP.
  EGTPPredictor     — single-shot length prediction from prefill entropy.
  PLPPredictor      — progressive per-step length correction.
  ForelenStats      — prediction accuracy tracking.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

__all__ = [
    "ForelenConfig",
    "EGTPPredictor",
    "PLPPredictor",
    "ForelenStats",
]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class ForelenConfig:
    """Hyper-parameters for ForeLen.

    Parameters
    ----------
    entropy_bins:
        Number of histogram bins used to represent the entropy distribution
        over input tokens.  A linear probe maps this histogram → length bucket.
    n_length_buckets:
        How many length buckets to predict.  Bucket boundaries are spaced
        log-uniformly between 1 and *max_length*.
    max_length:
        Maximum output length considered.  Requests predicted beyond this
        are clamped.
    plp_decay:
        Exponential decay factor [0, 1) for the progressive predictor.
        Higher decay = slower adaptation; lower decay = more reactive.
    plp_update_every:
        How many decode steps between each PLP update.
    """

    entropy_bins: int = 16
    n_length_buckets: int = 8
    max_length: int = 8192
    plp_decay: float = 0.9
    plp_update_every: int = 32

    def __post_init__(self) -> None:
        if self.entropy_bins < 2:
            raise ValueError("entropy_bins must be >= 2")
        if self.n_length_buckets < 2:
            raise ValueError("n_length_buckets must be >= 2")
        if self.max_length < 1:
            raise ValueError("max_length must be >= 1")
        if not (0.0 <= self.plp_decay < 1.0):
            raise ValueError("plp_decay must be in [0, 1)")
        if self.plp_update_every < 1:
            raise ValueError("plp_update_every must be >= 1")


# ---------------------------------------------------------------------------
# EGTPPredictor
# ---------------------------------------------------------------------------

class EGTPPredictor:
    """Entropy-Guided Token Pooling length predictor.

    Uses token entropy from the model's own prefill pass to predict a length
    bucket for the output.  A single linear probe (weight vector ``w`` and
    bias ``b``) maps the entropy histogram to a length bucket index.

    The probe is ``fit()`` offline on calibration examples.  If no weights
    have been fit, a heuristic based on mean entropy is used as fallback.

    Parameters
    ----------
    config:
        ``ForelenConfig``.
    """

    def __init__(self, config: Optional[ForelenConfig] = None) -> None:
        self._cfg = config or ForelenConfig()
        # Linear probe: w (entropy_bins,) + b scalar → bucket logit
        self._w: Optional[np.ndarray] = None
        self._b: float = 0.0
        # Log-uniform bucket boundaries
        self._boundaries: np.ndarray = np.unique(np.round(
            np.exp(
                np.linspace(
                    np.log(1),
                    np.log(self._cfg.max_length + 1),
                    self._cfg.n_length_buckets + 1,
                )
            )
        ).astype(np.int64))

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(
        self,
        entropy_histograms: np.ndarray,
        output_lengths: np.ndarray,
    ) -> None:
        """Fit the linear probe on calibration data.

        Parameters
        ----------
        entropy_histograms:
            Shape ``(n_examples, entropy_bins)`` — normalised histogram of
            per-token entropy values for each example's prefill pass.
        output_lengths:
            Shape ``(n_examples,)`` — observed output lengths.
        """
        X = np.asarray(entropy_histograms, dtype=np.float64)
        y = np.asarray(output_lengths, dtype=np.float64)
        if X.shape[0] != y.shape[0]:
            raise ValueError("entropy_histograms and output_lengths must have the same number of rows")
        if X.shape[1] != self._cfg.entropy_bins:
            raise ValueError(
                f"entropy_histograms must have {self._cfg.entropy_bins} columns"
            )
        # Least-squares fit: [X | 1] @ [w; b] = y
        X_aug = np.hstack([X, np.ones((len(X), 1))])
        result, _, _, _ = np.linalg.lstsq(X_aug, y, rcond=None)
        self._w = result[:-1]
        self._b = float(result[-1])

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def _compute_histogram(self, token_entropies: np.ndarray) -> np.ndarray:
        """Bin per-token entropy values into a normalised histogram."""
        h, _ = np.histogram(
            token_entropies,
            bins=self._cfg.entropy_bins,
            range=(0.0, float(np.log(max(2, len(token_entropies) + 1)))),
        )
        total = h.sum()
        return h.astype(np.float32) / max(1, total)

    def predict(self, token_entropies: np.ndarray) -> int:
        """Predict output length from per-token entropy values.

        Parameters
        ----------
        token_entropies:
            1-D array of entropy values, one per input token.  Entropy is
            ``-sum(p * log(p))`` over the token's attention or softmax
            distribution.

        Returns
        -------
        int
            Predicted output length (clamped to ``[1, max_length]``).
        """
        hist = self._compute_histogram(np.asarray(token_entropies, dtype=np.float32))
        if self._w is not None:
            raw = float(hist @ self._w) + self._b
        else:
            # Heuristic: higher mean entropy → longer output
            mean_entropy = float(np.mean(token_entropies)) if len(token_entropies) > 0 else 1.0
            max_ent = float(np.log(max(2, len(token_entropies) + 1)))
            ratio = min(1.0, mean_entropy / (max_ent + 1e-9))
            raw = 1.0 + ratio * self._cfg.max_length

        return int(np.clip(round(raw), 1, self._cfg.max_length))

    @property
    def is_fitted(self) -> bool:
        return self._w is not None


# ---------------------------------------------------------------------------
# PLPPredictor
# ---------------------------------------------------------------------------

class PLPPredictor:
    """Progressive Length Predictor — updates estimate during decoding.

    Starts from an initial EGTP estimate and corrects it as tokens are
    generated and per-step entropy observations arrive.

    Parameters
    ----------
    initial_prediction:
        Starting length estimate (from EGTPPredictor or prior heuristic).
    config:
        ``ForelenConfig``.
    """

    def __init__(
        self,
        initial_prediction: int,
        config: Optional[ForelenConfig] = None,
    ) -> None:
        self._cfg = config or ForelenConfig()
        self._estimate: float = float(max(1, initial_prediction))
        self._step: int = 0
        self._ema_entropy: float = 0.0
        self._updates: int = 0

    def update(self, current_len: int, step_entropy: float) -> int:
        """Observe one decode step and return updated length prediction.

        Parameters
        ----------
        current_len:
            Number of tokens generated so far.
        step_entropy:
            Entropy of the current step's logit distribution (a proxy for
            "how uncertain" the model is about ending soon).

        Returns
        -------
        int
            Revised remaining-length estimate.
        """
        self._step += 1
        # EMA of per-step entropy
        alpha = 1.0 - self._cfg.plp_decay
        self._ema_entropy = (
            self._cfg.plp_decay * self._ema_entropy + alpha * step_entropy
        )

        if self._step % self._cfg.plp_update_every == 0:
            # High entropy → longer; low entropy → shorter
            max_ent = float(np.log(max(2, self._cfg.max_length)))
            entropy_ratio = min(1.0, self._ema_entropy / (max_ent + 1e-9))
            correction = entropy_ratio * (self._cfg.max_length - current_len)
            self._estimate = float(current_len) + correction
            self._updates += 1

        remaining = max(1, int(self._estimate) - current_len)
        return min(remaining, self._cfg.max_length - current_len)

    @property
    def current_estimate(self) -> int:
        return max(1, int(round(self._estimate)))

    @property
    def n_updates(self) -> int:
        return self._updates


# ---------------------------------------------------------------------------
# ForelenStats
# ---------------------------------------------------------------------------

@dataclass
class ForelenStats:
    """Prediction accuracy tracker for ForeLen.

    Attributes
    ----------
    predictions_made:
        Number of EGTP predictions issued.
    total_abs_error:
        Cumulative sum of |predicted - actual| lengths.
    plp_corrections:
        Number of times PLP issued a significantly different estimate
        (|correction| > 10% of initial prediction).
    bucket_hits:
        Number of predictions whose bucket matched the actual output bucket.
    bucket_total:
        Total bucket comparisons made (denominator for bucket accuracy).
    """

    predictions_made: int = 0
    total_abs_error: float = 0.0
    plp_corrections: int = 0
    bucket_hits: int = 0
    bucket_total: int = 0

    def record(self, predicted: int, actual: int) -> None:
        self.predictions_made += 1
        self.total_abs_error += abs(predicted - actual)

    @property
    def mae(self) -> float:
        return self.total_abs_error / self.predictions_made if self.predictions_made > 0 else 0.0

    @property
    def bucket_accuracy(self) -> float:
        return self.bucket_hits / self.bucket_total if self.bucket_total > 0 else 0.0
