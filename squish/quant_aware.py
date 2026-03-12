"""QuantAware — Quantization-aware calibration for per-channel scale selection.

Standard post-training quantization (PTQ) uses simple min/max statistics to
determine scales.  Quantization-aware calibration (QAC) refines scales by
running a small calibration dataset through the model and minimizing the
quantization error.  This module implements:

* **MinMax** — per-channel scale from absolute max
* **Percentile** — per-channel scale from Nth percentile (reduces outlier impact)
* **MSE** — grid-search scale minimizing mean-squared quantization error

Reference:
    Nagel et al., "Up or Down? Adaptive Rounding for Post-Training Quantization",
    ICML 2020.  https://arxiv.org/abs/2004.10568

Usage::

    from squish.quant_aware import QuantAwareCalibrator, QAConfig

    cfg  = QAConfig(method="percentile", percentile=99.9, n_bits=8)
    cal  = QuantAwareCalibrator(cfg)
    cal.record(activation_batch)      # collect statistics
    scales = cal.compute_scales()     # per-channel float32 array
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

__all__ = [
    "QAConfig",
    "QuantAwareCalibrator",
    "QAStats",
]

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class QAConfig:
    """Configuration for quantization-aware calibration.

    Parameters
    ----------
    method : str
        Calibration method: ``"minmax"``, ``"percentile"``, or ``"mse"``.
    percentile : float
        Percentile value (0, 100] for the ``"percentile"`` method.
    n_bits : int
        Target quantization bit-width in [2, 16].
    per_channel : bool
        When True, compute one scale per channel; otherwise use a single
        global scale broadcast across all channels.
    mse_grid_steps : int
        Number of candidate scale values to evaluate during MSE grid search.
    """

    method: str = "percentile"
    percentile: float = 99.9
    n_bits: int = 8
    per_channel: bool = True
    mse_grid_steps: int = 32

    def __post_init__(self) -> None:
        valid_methods = ("minmax", "percentile", "mse")
        if self.method not in valid_methods:
            raise ValueError(
                f"method must be one of {valid_methods}; got '{self.method}'."
            )
        if not (0.0 < self.percentile <= 100.0):
            raise ValueError(
                f"percentile must be in (0, 100]; got {self.percentile}."
            )
        if not (2 <= self.n_bits <= 16):
            raise ValueError(
                f"n_bits must be in [2, 16]; got {self.n_bits}."
            )
        if self.mse_grid_steps < 1:
            raise ValueError(
                f"mse_grid_steps must be >= 1; got {self.mse_grid_steps}."
            )


# ---------------------------------------------------------------------------
# Stats dataclass
# ---------------------------------------------------------------------------


@dataclass
class QAStats:
    """Summary statistics produced by a completed calibration run.

    Parameters
    ----------
    n_batches : int
        Number of activation batches recorded.
    n_channels : int
        Number of channels detected.
    method : str
        Calibration method used.
    max_scale : float
        Maximum scale value across all channels.
    min_scale : float
        Minimum scale value across all channels.
    """

    n_batches: int
    n_channels: int
    method: str
    max_scale: float = 0.0
    min_scale: float = 0.0

    @property
    def dynamic_range_db(self) -> float:
        """Dynamic range in dB: ``20 * log10(max_scale / min_scale)``."""
        if self.min_scale <= 0.0 or self.max_scale <= 0.0:
            return 0.0
        return 20.0 * math.log10(self.max_scale / self.min_scale)


# ---------------------------------------------------------------------------
# Calibrator
# ---------------------------------------------------------------------------


class QuantAwareCalibrator:
    """Accumulates activation statistics and computes per-channel quant scales.

    Parameters
    ----------
    config : QAConfig
        Calibration hyperparameters.
    """

    def __init__(self, config: QAConfig) -> None:
        self._cfg = config
        self._n_batches: int = 0
        self._channels: Optional[int] = None
        # Per-channel running statistics.
        self._running_max: Optional[np.ndarray] = None   # shape (C,)
        self._percentile_buf: List[np.ndarray] = []      # list of (N*T, C) abs values
        self._sum_sq: Optional[np.ndarray] = None        # shape (C,)
        self._n_samples: int = 0  # total activation samples seen

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _flatten_to_2d(self, activations: np.ndarray) -> np.ndarray:
        """Return activations reshaped to (N, C)."""
        a = np.asarray(activations, dtype=np.float32)
        if a.ndim == 2:
            return a
        if a.ndim == 3:
            # (batch, seq_len, channels) -> (batch*seq_len, channels)
            b, t, c = a.shape
            return a.reshape(b * t, c)
        raise ValueError(
            f"activations must be 2-D (batch, channels) or 3-D "
            f"(batch, seq_len, channels); got shape {a.shape}."
        )

    def _init_buffers(self, n_channels: int) -> None:
        self._channels = n_channels
        self._running_max = np.zeros(n_channels, dtype=np.float32)
        self._sum_sq = np.zeros(n_channels, dtype=np.float32)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record(self, activations: np.ndarray) -> None:
        """Accumulate statistics from one activation batch.

        Parameters
        ----------
        activations : np.ndarray
            Shape ``(batch, channels)`` or ``(batch, seq_len, channels)``.
            Values are treated as floating-point activations.
        """
        flat = self._flatten_to_2d(activations)  # (N, C)
        n, c = flat.shape

        if self._channels is None:
            self._init_buffers(c)
        elif c != self._channels:
            raise ValueError(
                f"Channel count mismatch: expected {self._channels}, got {c}."
            )

        abs_flat = np.abs(flat)
        # Update running max.
        batch_max = abs_flat.max(axis=0)  # (C,)
        np.maximum(self._running_max, batch_max, out=self._running_max)  # type: ignore[arg-type]
        # Accumulate percentile buffer (store absolute values for percentile).
        if self._cfg.method in ("percentile", "mse"):
            self._percentile_buf.append(abs_flat)
        # Accumulate sum of squares for MSE method.
        self._sum_sq += (abs_flat ** 2).sum(axis=0)  # type: ignore[operator]
        self._n_samples += n
        self._n_batches += 1

    def compute_scales(self) -> np.ndarray:
        """Compute per-channel quantization scales.

        Returns
        -------
        np.ndarray
            Shape ``(n_channels,)`` float32 scale values.

        Raises
        ------
        RuntimeError
            If no data has been recorded yet.
        """
        if self._channels is None or self._n_batches == 0:
            raise RuntimeError(
                "No activations recorded.  Call record() before compute_scales()."
            )

        q_max = float(2 ** (self._cfg.n_bits - 1) - 1)  # e.g. 127 for INT8

        if self._cfg.method == "minmax":
            scales = self._running_max / q_max

        elif self._cfg.method == "percentile":
            # Concatenate all buffered absolute values and compute percentile.
            all_abs = np.concatenate(self._percentile_buf, axis=0)  # (N_total, C)
            pct_vals = np.percentile(all_abs, self._cfg.percentile, axis=0)  # (C,)
            scales = pct_vals.astype(np.float32) / q_max

        elif self._cfg.method == "mse":
            # Grid search: candidate scales from 0.5× to 1.5× of minmax scale.
            minmax_scales = self._running_max / q_max  # (C,)
            all_abs = np.concatenate(self._percentile_buf, axis=0)  # (N_total, C)
            best_scales = np.empty(self._channels, dtype=np.float32)
            low = 0.5 * minmax_scales
            high = 1.5 * minmax_scales
            steps = self._cfg.mse_grid_steps
            for c_idx in range(self._channels):
                candidates = np.linspace(
                    low[c_idx], high[c_idx], steps, dtype=np.float32
                )
                best_mse = np.inf
                best_s = minmax_scales[c_idx]
                channel_vals = all_abs[:, c_idx]
                for s in candidates:
                    if s <= 0.0:
                        continue
                    clipped = np.clip(channel_vals, 0.0, s * q_max)
                    quantized = np.round(clipped / s) * s
                    mse = float(np.mean((channel_vals - quantized) ** 2))
                    if mse < best_mse:
                        best_mse = mse
                        best_s = s
                best_scales[c_idx] = best_s
            scales = best_scales
        else:
            raise RuntimeError(f"Unknown method '{self._cfg.method}'.")

        # Ensure no zero scales.
        scales = np.where(scales > 0.0, scales, np.finfo(np.float32).tiny)
        if not self._cfg.per_channel:
            # Global scale: use the max across all channels.
            global_scale = float(scales.max())
            scales = np.full(self._channels, global_scale, dtype=np.float32)

        return scales.astype(np.float32)

    def reset(self) -> None:
        """Clear all accumulated statistics and reset to initial state."""
        self._n_batches = 0
        self._channels = None
        self._running_max = None
        self._percentile_buf = []
        self._sum_sq = None
        self._n_samples = 0

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def n_batches(self) -> int:
        """Number of activation batches recorded so far."""
        return self._n_batches

    @property
    def channels(self) -> Optional[int]:
        """Detected channel count, or ``None`` before the first :meth:`record`."""
        return self._channels

    def stats(self) -> QAStats:
        """Return a snapshot of calibration statistics.

        Returns
        -------
        QAStats
            Summary of recorded data and, if scales have been computed, their
            dynamic range.
        """
        if self._channels is None or self._n_batches == 0:
            return QAStats(
                n_batches=0,
                n_channels=0,
                method=self._cfg.method,
            )
        try:
            scales = self.compute_scales()
            max_s = float(scales.max())
            min_s = float(scales.min())
        except Exception:
            max_s = 0.0
            min_s = 0.0
        return QAStats(
            n_batches=self._n_batches,
            n_channels=self._channels,
            method=self._cfg.method,
            max_scale=max_s,
            min_scale=min_s,
        )
