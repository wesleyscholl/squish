"""
squish/trail.py

TRAIL — Recycled LLM Embedding Length Predictor.

Based on:
  "TRAIL: Trace-Recycled LLM Length Prediction for Latency-Aware Scheduling"
  ICLR 2025 — proceedings.iclr.cc/paper/2025/9eb8b5ccb0de594a16548f7c058fdadf

Problem
-------
Prior output-length predictors use BERT (large overhead) or n-gram counting
(low accuracy).  TRAIL "recycles" the embeddings from an intermediate
transformer layer of the serving LLM — making a single forward pass do
double duty: generate the first token AND predict output length simultaneously.

Key results
-----------
- Predictions from layer-11 embeddings achieve 2.66× lower MAE vs. BERT
- Integration with a length-aware scheduler yields 1.66×–2.01× lower mean
  latency and 1.76×–24.07× lower mean TTFT
- SRPT (Shortest Remaining Processing Time) scheduling with limited
  preemptions: theoretically optimal for LLM inference under memory constraints

Relationship to ForeLen
-----------------------
TRAIL is the *currently deployable* predecessor to ForeLen (ICLR 2026).
ForeLen's EGTP uses token entropy rather than intermediate embeddings and
requires no probe fine-tuning.  Deploy TRAIL today; migrate to ForeLen when
its code releases.

Implementation notes
--------------------
``TrailLinearProbe`` is a single-layer linear regression / classification head
trained on (layer-N embedding, output_length) pairs from a calibration set.
In production, weights are persisted as numpy ``.npy`` files.

Conflict notes
--------------
- **No conflict** with any inference method — TRAIL is a scheduler hint only.
- **Synergy with BucketServe**: the predicted length bucket routes requests
  into homogeneous batches, reducing padding waste (barrel effect).
- **Superseded by ForeLen** when available; the API is intentionally similar
  to ease migration.

Provides
--------
  TrailConfig         — configuration (probe layer, hidden dim, max length).
  TrailLinearProbe    — linear probe: fit / predict / save / load.
  TrailPredictor      — high-level interface wrapping the probe.
  TrailStats          — prediction accuracy tracking.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import numpy as np

__all__ = [
    "TrailConfig",
    "TrailLinearProbe",
    "TrailPredictor",
    "TrailStats",
]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class TrailConfig:
    """Configuration for the TRAIL length predictor.

    Parameters
    ----------
    probe_layer:
        Index (0-based) of the transformer layer whose output embeddings are
        used as the prediction feature vector.  TRAIL's paper uses layer 11
        for a 32-layer model.
    hidden_dim:
        Embedding dimensionality (e.g., 4096 for Qwen3-8B).
    max_length:
        Maximum output length considered; predictions are clamped.
    n_buckets:
        Number of coarse length buckets for bucket-accuracy evaluation.
    """

    probe_layer: int = 11
    hidden_dim: int = 4096
    max_length: int = 8192
    n_buckets: int = 8

    def __post_init__(self) -> None:
        if self.probe_layer < 0:
            raise ValueError("probe_layer must be >= 0")
        if self.hidden_dim < 1:
            raise ValueError("hidden_dim must be >= 1")
        if self.max_length < 1:
            raise ValueError("max_length must be >= 1")
        if self.n_buckets < 2:
            raise ValueError("n_buckets must be >= 2")


# ---------------------------------------------------------------------------
# TrailLinearProbe
# ---------------------------------------------------------------------------

class TrailLinearProbe:
    """Lightweight linear regression probe: embedding → output length.

    Parameters
    ----------
    config:
        ``TrailConfig``.
    """

    def __init__(self, config: TrailConfig | None = None) -> None:
        self._cfg = config or TrailConfig()
        self._w: np.ndarray | None = None  # shape (hidden_dim,)
        self._b: float = 0.0

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(
        self,
        embeddings: np.ndarray,
        output_lengths: np.ndarray,
    ) -> None:
        """Fit the linear probe via least-squares regression.

        Parameters
        ----------
        embeddings:
            Shape ``(n_examples, hidden_dim)``.  Each row is the mean-pooled
            embedding of the prompt from the ``probe_layer``-th transformer
            layer.
        output_lengths:
            Shape ``(n_examples,)``.  Observed output lengths in tokens.
        """
        X = np.asarray(embeddings, dtype=np.float64)
        y = np.asarray(output_lengths, dtype=np.float64)
        if X.shape[0] != y.shape[0]:
            raise ValueError("embeddings and output_lengths row counts differ")
        if X.shape[1] != self._cfg.hidden_dim:
            raise ValueError(
                f"embedding dim {X.shape[1]} != config.hidden_dim {self._cfg.hidden_dim}"
            )
        X_aug = np.hstack([X, np.ones((len(X), 1))])
        result, _, _, _ = np.linalg.lstsq(X_aug, y, rcond=None)
        self._w = result[:-1]
        self._b = float(result[-1])

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, embedding: np.ndarray) -> int:
        """Predict output length for one example.

        Parameters
        ----------
        embedding:
            1-D array of shape ``(hidden_dim,)`` — the mean-pooled layer
            embedding for the prompt.

        Returns
        -------
        int
            Predicted output length, clamped to ``[1, max_length]``.
        """
        e = np.asarray(embedding, dtype=np.float64).ravel()
        if len(e) != self._cfg.hidden_dim:
            raise ValueError(
                f"embedding length {len(e)} != config.hidden_dim {self._cfg.hidden_dim}"
            )
        if self._w is not None:
            raw = float(e @ self._w) + self._b
        else:
            # Heuristic fallback: use L2 norm as a length proxy
            raw = float(np.linalg.norm(e))
        return int(np.clip(round(raw), 1, self._cfg.max_length))

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save weights to a ``.npz`` file at *path*."""
        if self._w is None:
            raise RuntimeError("Probe has not been fitted; nothing to save.")
        np.savez(path, w=self._w, b=np.array([self._b]))

    def load(self, path: str) -> None:
        """Load weights from a ``.npz`` file at *path*."""
        data = np.load(path)
        self._w = data["w"]
        self._b = float(data["b"][0])

    @property
    def is_fitted(self) -> bool:
        return self._w is not None


# ---------------------------------------------------------------------------
# TrailPredictor
# ---------------------------------------------------------------------------

class TrailPredictor:
    """High-level TRAIL interface.

    Wraps ``TrailLinearProbe`` with bucket-aware prediction and a one-step
    SRPT priority score.

    Parameters
    ----------
    config:
        ``TrailConfig``.
    """

    def __init__(self, config: TrailConfig | None = None) -> None:
        self._cfg = config or TrailConfig()
        self.probe = TrailLinearProbe(self._cfg)
        # Log-uniform bucket boundaries
        self._bucket_bounds: np.ndarray = np.unique(np.round(
            np.exp(
                np.linspace(
                    np.log(1),
                    np.log(self._cfg.max_length + 1),
                    self._cfg.n_buckets + 1,
                )
            )
        ).astype(np.int64))

    def predict(self, embedding: np.ndarray) -> int:
        """Predict output length for one example.

        Returns
        -------
        int
            Predicted output length in tokens.
        """
        return self.probe.predict(embedding)

    def predict_bucket(self, embedding: np.ndarray) -> int:
        """Predict length bucket index for BucketServe routing.

        Returns
        -------
        int
            0-based bucket index (larger → longer predicted output).
        """
        length = self.predict(embedding)
        bucket = int(np.searchsorted(self._bucket_bounds, length, side="right")) - 1
        return max(0, min(bucket, self._cfg.n_buckets - 1))

    def srpt_priority(self, embedding: np.ndarray, current_tokens: int = 0) -> float:
        """Compute SRPT priority score (lower → schedules first).

        SRPT schedules the request with the *shortest remaining processing
        time* first.  Remaining = predicted_length - current_tokens.

        Returns
        -------
        float
            Remaining token count estimate; schedule in ascending order.
        """
        predicted = self.predict(embedding)
        return max(0.0, float(predicted - current_tokens))


# ---------------------------------------------------------------------------
# TrailStats
# ---------------------------------------------------------------------------

@dataclass
class TrailStats:
    """Prediction accuracy tracker for TRAIL.

    Attributes
    ----------
    prediction_count:
        Number of length predictions issued.
    total_abs_error:
        Cumulative |predicted - actual| across all predictions.
    bucket_hits:
        Predictions where the bucket matched the actual output bucket.
    bucket_total:
        Total bucket comparisons.
    """

    prediction_count: int = 0
    total_abs_error: float = 0.0
    bucket_hits: int = 0
    bucket_total: int = 0

    def record(self, predicted: int, actual: int, predicted_bucket: int, actual_bucket: int) -> None:
        """Record one prediction result."""
        self.prediction_count += 1
        self.total_abs_error += abs(predicted - actual)
        self.bucket_total += 1
        if predicted_bucket == actual_bucket:
            self.bucket_hits += 1

    @property
    def mae(self) -> float:
        return self.total_abs_error / self.prediction_count if self.prediction_count > 0 else 0.0

    @property
    def bucket_accuracy(self) -> float:
        return self.bucket_hits / self.bucket_total if self.bucket_total > 0 else 0.0
