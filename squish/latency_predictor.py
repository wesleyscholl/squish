"""squish/latency_predictor.py

LatencyPredictor — Per-request latency prediction for online scheduling.

Accurate per-request latency estimates allow a serving scheduler to bin-pack
requests into batches that are unlikely to breach a latency SLO, set adaptive
timeouts, and balance work across multiple inference workers.

LatencyPredictor fits a linear model::

    latency_ms ≈ prefill_coeff  * n_prefill
               + decode_coeff   * n_decode
               + kv_coeff       * (n_heads * head_dim * (n_prefill + n_decode))
               + base_latency

from observed ``(n_prefill, n_decode, measured_ms)`` tuples using ordinary
least squares (NumPy ``linalg.lstsq``).  The ``kv_coeff`` term models the KV
cache read-bandwidth cost that grows with the total context size and the number
of head-dimension bytes.

Before sufficient calibration data has been collected (fewer than 3 samples)
the predictor returns a prediction with ``confidence = 0.0``.  Confidence
scales linearly from 0.0 at 3 samples to 1.0 at 10 samples.

Example usage::

    from squish.latency_predictor import LatencyPredictor

    predictor = LatencyPredictor(n_heads=8, head_dim=64)
    for n_p, n_d, ms in [(128, 32, 45.3), (256, 64, 92.1), (64, 16, 22.8)]:
        predictor.record(n_p, n_d, ms)
    predictor.fit()
    pred = predictor.predict(200, 50)
    print(f"{pred.total_ms:.1f} ms  (confidence={pred.confidence:.2f})")
"""

from __future__ import annotations

__all__ = ["LatencyModel", "LatencyPrediction", "LatencyPredictor"]

import dataclasses
from typing import Optional

import numpy as np


# Minimum samples required before a fit is considered reliable.
_MIN_RELIABLE_SAMPLES: int = 3

# Sample count at which confidence saturates to 1.0.
_CONFIDENCE_SATURATION: int = 10


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class LatencyModel:
    """Fitted latency model coefficients.

    Attributes:
        prefill_coeff: Marginal latency per prefill token (ms / token).
        decode_coeff:  Marginal latency per decode token (ms / token).
        kv_coeff:      Marginal latency per KV element
                       (ms / (n_heads * head_dim * context_len)).
        base_latency:  Constant overhead per request (ms).
    """

    prefill_coeff: float
    decode_coeff:  float
    kv_coeff:      float
    base_latency:  float


@dataclasses.dataclass
class LatencyPrediction:
    """Predicted latency for a single request.

    Attributes:
        prefill_ms:  Estimated prefill phase contribution (ms).
        decode_ms:   Estimated decode phase contribution (ms).
        total_ms:    Total estimated latency (ms), including KV-bandwidth and
                     base-overhead terms.
        confidence:  Fit confidence in ``[0, 1]``.  Values below ``0.3``
                     indicate the model has very few calibration samples and
                     predictions may be unreliable.
    """

    prefill_ms: float
    decode_ms:  float
    total_ms:   float
    confidence: float


# ---------------------------------------------------------------------------
# Predictor
# ---------------------------------------------------------------------------


class LatencyPredictor:
    """Per-request latency predictor using online ordinary-least-squares
    calibration.

    Records observed ``(n_prefill, n_decode, measured_ms)`` data points and
    fits a four-parameter linear model from which future latencies can be
    predicted.  The model is not automatically refitted on every
    :meth:`record` call; call :meth:`fit` explicitly after accumulating a
    new batch of data points.

    Args:
        n_heads:  Number of attention heads (used to construct the KV feature).
        head_dim: Dimension per attention head.

    Raises:
        ValueError: If *n_heads* or *head_dim* are not positive integers.
    """

    def __init__(self, n_heads: int, head_dim: int) -> None:
        if n_heads < 1:
            raise ValueError(f"n_heads must be >= 1, got {n_heads}")
        if head_dim < 1:
            raise ValueError(f"head_dim must be >= 1, got {head_dim}")

        self._n_heads  = n_heads
        self._head_dim = head_dim

        # Accumulated calibration rows: (n_prefill, n_decode, measured_ms).
        self._data: list[tuple[int, int, float]] = []

        # Last fitted model; coefficients start at zero (cold model).
        self._model = LatencyModel(
            prefill_coeff=0.0,
            decode_coeff=0.0,
            kv_coeff=0.0,
            base_latency=0.0,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record(self, n_prefill: int, n_decode: int, measured_ms: float) -> None:
        """Add a calibration data point.

        Args:
            n_prefill:   Number of prefill tokens for the request.  Must be
                         >= 0.
            n_decode:    Number of decode tokens generated.  Must be >= 0.
            measured_ms: Wall-clock latency actually observed (ms).  Must be
                         a non-negative finite value.

        Raises:
            ValueError: If any argument violates its constraint.
        """
        if n_prefill < 0:
            raise ValueError(f"n_prefill must be >= 0, got {n_prefill}")
        if n_decode < 0:
            raise ValueError(f"n_decode must be >= 0, got {n_decode}")
        if not np.isfinite(measured_ms) or measured_ms < 0.0:
            raise ValueError(
                f"measured_ms must be a non-negative finite value, "
                f"got {measured_ms}"
            )
        self._data.append((n_prefill, n_decode, float(measured_ms)))

    def fit(self) -> LatencyModel:
        """Refit the linear model from all recorded data points.

        Solves the ordinary least-squares problem::

            [n_prefill  n_decode  kv_feature  1] · [c0 c1 c2 c3]ᵀ = ms

        where ``kv_feature = n_heads * head_dim * (n_prefill + n_decode)``.

        Returns:
            The newly fitted :class:`LatencyModel`.  Also updates the
            internal model returned by :attr:`model`.

        Raises:
            RuntimeError: If fewer than 2 data points have been recorded
                          (underdetermined system).
        """
        n = len(self._data)
        if n < 2:
            raise RuntimeError(
                f"Need at least 2 recorded samples to fit; have {n}. "
                "Call record() first."
            )

        kv_scale = float(self._n_heads * self._head_dim)

        A_rows: list[list[float]] = []
        b_rows: list[float] = []
        for n_prefill, n_decode, ms in self._data:
            kv_feat = kv_scale * float(n_prefill + n_decode)
            A_rows.append([float(n_prefill), float(n_decode), kv_feat, 1.0])
            b_rows.append(ms)

        A = np.array(A_rows, dtype=np.float64)
        b = np.array(b_rows, dtype=np.float64)

        # lstsq returns (solution, residuals, rank, singular_values).
        coeffs, _, _, _ = np.linalg.lstsq(A, b, rcond=None)

        self._model = LatencyModel(
            prefill_coeff=float(coeffs[0]),
            decode_coeff=float(coeffs[1]),
            kv_coeff=float(coeffs[2]),
            base_latency=float(coeffs[3]),
        )
        return self._model

    def predict(self, n_prefill: int, n_decode: int) -> LatencyPrediction:
        """Predict request latency using the current fitted model.

        If fewer than :data:`_MIN_RELIABLE_SAMPLES` data points have been
        recorded, the prediction is returned with ``confidence=0.0``.
        Confidence rises linearly from ``0.0`` at 3 samples to ``1.0`` at
        :data:`_CONFIDENCE_SATURATION` samples.

        Args:
            n_prefill: Anticipated number of prefill tokens.  Must be >= 0.
            n_decode:  Anticipated number of decode tokens.  Must be >= 0.

        Returns:
            A :class:`LatencyPrediction` with component and total estimates.

        Raises:
            ValueError: If either count is negative.
        """
        if n_prefill < 0:
            raise ValueError(f"n_prefill must be >= 0, got {n_prefill}")
        if n_decode < 0:
            raise ValueError(f"n_decode must be >= 0, got {n_decode}")

        n = len(self._data)
        if n < _MIN_RELIABLE_SAMPLES:
            confidence = 0.0
        else:
            span = float(_CONFIDENCE_SATURATION - _MIN_RELIABLE_SAMPLES)
            confidence = min(1.0, float(n - _MIN_RELIABLE_SAMPLES) / span)

        m          = self._model
        kv_scale   = float(self._n_heads * self._head_dim)
        prefill_ms = m.prefill_coeff * float(n_prefill)
        decode_ms  = m.decode_coeff  * float(n_decode)
        kv_ms      = m.kv_coeff * kv_scale * float(n_prefill + n_decode)
        total_ms   = prefill_ms + decode_ms + kv_ms + m.base_latency

        return LatencyPrediction(
            prefill_ms=prefill_ms,
            decode_ms=decode_ms,
            total_ms=total_ms,
            confidence=confidence,
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def n_samples(self) -> int:
        """Number of calibration data points recorded so far."""
        return len(self._data)

    @property
    def model(self) -> LatencyModel:
        """The most recently fitted :class:`LatencyModel`.

        Returns a zeroed model if :meth:`fit` has never been called.
        """
        return self._model
