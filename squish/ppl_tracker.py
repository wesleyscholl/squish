"""PPLTracker — Rolling perplexity tracker with sliding window and alerts.

Tracks per-batch perplexity (PPL) during inference to detect quality
degradation from quantisation drift, adapter conflicts, or KV eviction.
Maintains a sliding window of recent PPL values and fires alerts when the
rolling PPL exceeds a threshold relative to the baseline.

Perplexity is defined as:

.. math::

    \\text{PPL} = \\exp\\!\\left(\\frac{1}{N}\\sum_{i=1}^{N} -\\log p(x_i \\mid x_{<i})\\right)

Rolling perplexity uses the geometric mean of PPL values in the window
(equivalent to :math:`\\exp(\\text{mean of log-PPL})`), which is more
numerically stable and less sensitive to outlier spikes than the arithmetic
mean.

Usage::

    from squish.ppl_tracker import PPLTracker, PPLWindow

    tracker = PPLTracker(window_size=100, alert_threshold=1.5)
    for logits, target_ids in stream:
        tracker.record(logits, target_ids)
    print(tracker.rolling_ppl)
    print(tracker.is_degraded)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

__all__ = [
    "PPLWindow",
    "PPLAlert",
    "PPLTracker",
    "PPLStats",
]


# ---------------------------------------------------------------------------
# PPLWindow — sliding window of PPL values
# ---------------------------------------------------------------------------


@dataclass
class PPLWindow:
    """Fixed-capacity sliding window of perplexity values.

    Parameters
    ----------
    window_size : int
        Maximum number of values to retain.  When the window is full, the
        oldest value is evicted on :meth:`push`.
    values : list[float]
        Current window contents (most recent ``window_size`` values).
    """

    window_size: int
    values: List[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.window_size < 1:
            raise ValueError(f"window_size must be >= 1; got {self.window_size}")

    def push(self, value: float) -> None:
        """Append a new PPL value, evicting the oldest if the window is full.

        Parameters
        ----------
        value : float
            Perplexity value to append.
        """
        self.values.append(float(value))
        if len(self.values) > self.window_size:
            self.values = self.values[-self.window_size :]

    @property
    def mean(self) -> float:
        """Arithmetic mean of the current window values.

        Returns ``float('nan')`` when the window is empty.
        """
        if not self.values:
            return float("nan")
        return float(np.mean(self.values))

    @property
    def std(self) -> float:
        """Standard deviation of the current window values.

        Returns ``float('nan')`` when fewer than 2 values are present.
        """
        if len(self.values) < 2:
            return float("nan")
        return float(np.std(self.values, ddof=1))

    @property
    def min(self) -> float:
        """Minimum value in the current window.

        Returns ``float('nan')`` when the window is empty.
        """
        if not self.values:
            return float("nan")
        return float(np.min(self.values))

    @property
    def max(self) -> float:
        """Maximum value in the current window.

        Returns ``float('nan')`` when the window is empty.
        """
        if not self.values:
            return float("nan")
        return float(np.max(self.values))


# ---------------------------------------------------------------------------
# PPLAlert
# ---------------------------------------------------------------------------


@dataclass
class PPLAlert:
    """Record of a perplexity degradation alert.

    Parameters
    ----------
    step : int
        The record step at which the alert was fired.
    rolling_ppl : float
        Rolling perplexity at alert time.
    baseline_ppl : float
        Baseline perplexity used for comparison.
    ratio : float
        ``rolling_ppl / baseline_ppl`` at alert time.
    message : str
        Human-readable alert description.
    """

    step: int
    rolling_ppl: float
    baseline_ppl: float
    ratio: float
    message: str


# ---------------------------------------------------------------------------
# PPLStats
# ---------------------------------------------------------------------------


@dataclass
class PPLStats:
    """Aggregate statistics from a :class:`PPLTracker` session.

    Parameters
    ----------
    total_tokens : int
        Total tokens scored across all :meth:`PPLTracker.record` calls.
    total_steps : int
        Total :meth:`PPLTracker.record` calls.
    min_ppl : float
        Minimum per-step PPL observed.
    max_ppl : float
        Maximum per-step PPL observed.
    """

    total_tokens: int
    total_steps: int
    min_ppl: float
    max_ppl: float

    @property
    def range_ppl(self) -> float:
        """Range of observed PPL values (max - min)."""
        return self.max_ppl - self.min_ppl


# ---------------------------------------------------------------------------
# PPLTracker
# ---------------------------------------------------------------------------


class PPLTracker:
    """Rolling perplexity tracker with baseline comparison and alert firing.

    Parameters
    ----------
    window_size : int
        Number of recent PPL values retained for the rolling computation.
    alert_threshold : float
        Multiplier applied to ``baseline_ppl``.  An alert is raised when
        ``rolling_ppl > baseline_ppl * alert_threshold``.  Must be > 1.
    baseline_ppl : float, optional
        Baseline perplexity for degradation comparison.  Can also be set
        post-construction via :meth:`set_baseline`.

    Examples
    --------
    >>> tracker = PPLTracker(window_size=50, alert_threshold=1.5)
    >>> tracker.record(logits, target_ids)
    >>> if tracker.is_degraded:
    ...     print(f"PPL degraded: {tracker.rolling_ppl:.2f}")
    """

    def __init__(
        self,
        window_size: int = 100,
        alert_threshold: float = 1.5,
        baseline_ppl: Optional[float] = None,
    ) -> None:
        if window_size < 1:
            raise ValueError(f"window_size must be >= 1; got {window_size}")
        if alert_threshold <= 1.0:
            raise ValueError(
                f"alert_threshold must be > 1.0; got {alert_threshold}"
            )
        if baseline_ppl is not None and baseline_ppl <= 0.0:
            raise ValueError(
                f"baseline_ppl must be > 0; got {baseline_ppl}"
            )
        self._window = PPLWindow(window_size=window_size)
        self._alert_threshold = alert_threshold
        self._baseline_ppl: Optional[float] = baseline_ppl
        self._alerts: List[PPLAlert] = []
        self._step: int = 0
        # For PPLStats
        self._total_tokens: int = 0
        self._min_ppl: float = float("inf")
        self._max_ppl: float = float("-inf")

    # ── Recording ─────────────────────────────────────────────────────────────

    def record(self, logits: np.ndarray, target_ids: np.ndarray) -> None:
        """Score a sequence and update the rolling PPL window.

        Parameters
        ----------
        logits : np.ndarray
            Raw (unnormalised) logits of shape ``(seq_len, vocab_size)``,
            dtype ``float32``.
        target_ids : np.ndarray
            Ground-truth token ids of shape ``(seq_len,)``, dtype integer.

        Notes
        -----
        Log-softmax is computed in a numerically stable way (subtract row
        max before exponentiation).  PPL is ``exp(mean_nll)`` where
        ``mean_nll`` is the average negative log-likelihood over the
        sequence.
        """
        logits_f = np.asarray(logits, dtype=np.float64)
        ids = np.asarray(target_ids, dtype=np.int64)
        seq_len = logits_f.shape[0]

        if seq_len == 0:
            return

        # Numerically stable log-softmax.
        max_logits = logits_f.max(axis=-1, keepdims=True)
        shifted = logits_f - max_logits
        log_sum_exp = np.log(np.sum(np.exp(shifted), axis=-1, keepdims=True))
        log_softmax = shifted - log_sum_exp  # (seq_len, vocab_size)

        # Per-token NLL.
        row_idx = np.arange(seq_len)
        nll = -log_softmax[row_idx, ids]  # (seq_len,)
        mean_nll = float(np.mean(nll))
        ppl = float(np.exp(mean_nll))

        # Clamp to a sane range to guard against near-zero probability tokens.
        ppl = max(ppl, 1e-6)

        self._window.push(ppl)
        self._step += 1
        self._total_tokens += seq_len

        # Update global min/max.
        if ppl < self._min_ppl:
            self._min_ppl = ppl
        if ppl > self._max_ppl:
            self._max_ppl = ppl

        # Check for degradation alert.
        self._maybe_fire_alert()

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def rolling_ppl(self) -> float:
        """Geometric mean of PPL values in the sliding window.

        Computed as ``exp(mean(log(ppl_values)))`` for numerical stability.
        Returns ``float('nan')`` when the window is empty.
        """
        if not self._window.values:
            return float("nan")
        log_ppls = np.log(
            np.maximum(np.array(self._window.values, dtype=np.float64), 1e-10)
        )
        return float(np.exp(np.mean(log_ppls)))

    @property
    def is_degraded(self) -> bool:
        """``True`` when ``rolling_ppl > baseline_ppl * alert_threshold``.

        Returns ``False`` when no baseline has been set or the window is
        empty.
        """
        if self._baseline_ppl is None:
            return False
        rp = self.rolling_ppl
        if np.isnan(rp):
            return False
        return rp > self._baseline_ppl * self._alert_threshold

    @property
    def alerts(self) -> List[PPLAlert]:
        """All :class:`PPLAlert` instances fired since construction or last
        :meth:`reset`."""
        return list(self._alerts)

    @property
    def step(self) -> int:
        """Total number of :meth:`record` calls since construction or reset."""
        return self._step

    # ── Baseline management ───────────────────────────────────────────────────

    def set_baseline(self, ppl: Optional[float] = None) -> None:
        """Set the reference baseline PPL for degradation detection.

        Parameters
        ----------
        ppl : float, optional
            Explicit baseline value.  When ``None``, the current
            :attr:`rolling_ppl` is used as the baseline.

        Raises
        ------
        ValueError
            If ``ppl`` is provided but is not > 0.
        RuntimeError
            If ``ppl`` is ``None`` and the window is empty (no rolling PPL
            available).
        """
        if ppl is not None:
            if ppl <= 0.0:
                raise ValueError(f"baseline_ppl must be > 0; got {ppl}")
            self._baseline_ppl = ppl
        else:
            rp = self.rolling_ppl
            if np.isnan(rp):
                raise RuntimeError(
                    "Cannot set baseline from rolling_ppl: window is empty."
                )
            self._baseline_ppl = rp

    # ── Reset ─────────────────────────────────────────────────────────────────

    def reset(self) -> None:
        """Clear the window, alerts, and step counter.

        The baseline PPL and alert threshold are preserved.
        """
        self._window = PPLWindow(window_size=self._window.window_size)
        self._alerts.clear()
        self._step = 0
        self._total_tokens = 0
        self._min_ppl = float("inf")
        self._max_ppl = float("-inf")

    # ── Statistics ────────────────────────────────────────────────────────────

    def ppl_stats(self) -> PPLStats:
        """Return aggregate statistics accumulated since construction or reset.

        Returns
        -------
        PPLStats
        """
        min_ppl = self._min_ppl if self._min_ppl != float("inf") else float("nan")
        max_ppl = self._max_ppl if self._max_ppl != float("-inf") else float("nan")
        return PPLStats(
            total_tokens=self._total_tokens,
            total_steps=self._step,
            min_ppl=min_ppl,
            max_ppl=max_ppl,
        )

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _maybe_fire_alert(self) -> None:
        """Fire a :class:`PPLAlert` if degradation is detected."""
        if self._baseline_ppl is None:
            return
        rp = self.rolling_ppl
        if np.isnan(rp):
            return
        if rp > self._baseline_ppl * self._alert_threshold:
            ratio = rp / self._baseline_ppl
            alert = PPLAlert(
                step=self._step,
                rolling_ppl=rp,
                baseline_ppl=self._baseline_ppl,
                ratio=ratio,
                message=(
                    f"PPL degradation at step {self._step}: "
                    f"rolling={rp:.2f}, baseline={self._baseline_ppl:.2f}, "
                    f"ratio={ratio:.3f} > threshold={self._alert_threshold:.2f}"
                ),
            )
            self._alerts.append(alert)
