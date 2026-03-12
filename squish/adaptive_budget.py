"""AdaptiveBudget — SLO-aware adaptive inference compute budget controller.

Jointly manages KV cache budget and layer-skip rates to satisfy per-request
latency SLOs.  Uses a PI controller (Proportional-Integral) to adaptively
adjust the compute budget: when measured latency exceeds the SLO, it reduces
KV budget and increases layer-skip aggressiveness; when latency is well within
SLO, it relaxes constraints to improve quality.

Reference:
    Crankshaw et al., "Clipper: A Low-Latency Online Prediction Serving System",
    NSDI 2017.  https://www.usenix.org/conference/nsdi17/technical-sessions/presentation/crankshaw

Usage::

    from squish.adaptive_budget import AdaptiveBudgetController, BudgetConfig

    cfg   = BudgetConfig(target_latency_ms=150.0, kv_budget_min=512, kv_budget_max=4096)
    ctrl  = AdaptiveBudgetController(cfg)
    budget = ctrl.step(observed_latency_ms=180.0)
    print(budget.kv_tokens, budget.skip_layers())
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List

__all__ = [
    "BudgetConfig",
    "BudgetState",
    "AdaptiveBudgetController",
    "BudgetStats",
]

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class BudgetConfig:
    """Hyperparameters for the PI budget controller.

    Parameters
    ----------
    target_latency_ms : float
        Target per-request latency in milliseconds (SLO).
    kv_budget_min : int
        Minimum allowed KV cache token budget.
    kv_budget_max : int
        Maximum allowed KV cache token budget.
    max_skip_fraction : float
        Maximum fraction of transformer layers that may be skipped in [0, 1].
    kp : float
        Proportional gain of the PI controller (> 0).
    ki : float
        Integral gain of the PI controller (>= 0).
    """

    target_latency_ms: float = 150.0
    kv_budget_min: int = 512
    kv_budget_max: int = 4096
    max_skip_fraction: float = 0.5
    kp: float = 0.1
    ki: float = 0.01

    def __post_init__(self) -> None:
        if self.target_latency_ms <= 0.0:
            raise ValueError(
                f"target_latency_ms must be > 0; got {self.target_latency_ms}."
            )
        if self.kv_budget_min >= self.kv_budget_max:
            raise ValueError(
                f"kv_budget_min ({self.kv_budget_min}) must be strictly less "
                f"than kv_budget_max ({self.kv_budget_max})."
            )
        if self.kv_budget_min < 1:
            raise ValueError(
                f"kv_budget_min must be >= 1; got {self.kv_budget_min}."
            )
        if self.kp <= 0.0:
            raise ValueError(f"kp must be > 0; got {self.kp}.")
        if self.ki < 0.0:
            raise ValueError(f"ki must be >= 0; got {self.ki}.")
        if not (0.0 <= self.max_skip_fraction <= 1.0):
            raise ValueError(
                f"max_skip_fraction must be in [0, 1]; got {self.max_skip_fraction}."
            )


# ---------------------------------------------------------------------------
# BudgetState
# ---------------------------------------------------------------------------


@dataclass
class BudgetState:
    """Current compute budget issued to an inference request.

    Parameters
    ----------
    kv_tokens : int
        Maximum number of KV-cache tokens to retain.
    skip_fraction : float
        Fraction of transformer layers to skip in [0, max_skip_fraction].
    quality_mode : str
        Current operating mode: ``"performance"``, ``"balanced"``, or
        ``"quality"``.
    """

    kv_tokens: int
    skip_fraction: float
    quality_mode: str = "balanced"

    def __post_init__(self) -> None:
        if self.kv_tokens < 1:
            raise ValueError(
                f"kv_tokens must be >= 1; got {self.kv_tokens}."
            )
        if not (0.0 <= self.skip_fraction <= 1.0):
            raise ValueError(
                f"skip_fraction must be in [0, 1]; got {self.skip_fraction}."
            )
        valid_modes = ("performance", "balanced", "quality")
        if self.quality_mode not in valid_modes:
            raise ValueError(
                f"quality_mode must be one of {valid_modes}; "
                f"got '{self.quality_mode}'."
            )

    def skip_layers(self, total_layers: int = 32) -> int:
        """Return the integer number of layers to skip.

        Parameters
        ----------
        total_layers : int
            Total number of transformer layers in the model.

        Returns
        -------
        int
            ``round(skip_fraction * total_layers)`` layers to skip.
        """
        if total_layers < 1:
            raise ValueError(
                f"total_layers must be >= 1; got {total_layers}."
            )
        return round(self.skip_fraction * total_layers)


# ---------------------------------------------------------------------------
# BudgetStats
# ---------------------------------------------------------------------------


@dataclass
class BudgetStats:
    """Aggregate performance statistics for a :class:`AdaptiveBudgetController`.

    Parameters
    ----------
    n_steps : int
        Total number of :meth:`AdaptiveBudgetController.step` calls.
    mean_latency_ms : float
        Mean observed latency over all steps.
    slo_violations : int
        Number of steps where observed latency exceeded the target.
    """

    n_steps: int = 0
    mean_latency_ms: float = 0.0
    slo_violations: int = 0

    @property
    def violation_rate(self) -> float:
        """Fraction of steps that violated the latency SLO."""
        if self.n_steps == 0:
            return 0.0
        return self.slo_violations / self.n_steps


# ---------------------------------------------------------------------------
# AdaptiveBudgetController
# ---------------------------------------------------------------------------


class AdaptiveBudgetController:
    """PI controller that jointly tunes KV budget and layer-skip rate.

    The controller observes the most recent request latency, computes the
    signed error relative to the SLO, and updates the KV token budget via a
    discrete PI law::

        integral  += error
        kv_budget -= kp * error + ki * integral

    The KV budget is clamped to ``[kv_budget_min, kv_budget_max]``.  The
    skip fraction scales linearly from 0 at the minimum budget to
    ``max_skip_fraction`` at the maximum latency pressure.

    Parameters
    ----------
    config : BudgetConfig
        Controller configuration.
    """

    def __init__(self, config: BudgetConfig) -> None:
        self._cfg = config
        kv_mid = (config.kv_budget_min + config.kv_budget_max) // 2
        self._kv_budget: float = float(kv_mid)
        self._integral: float = 0.0
        self._latency_history: List[float] = []
        self._n_steps: int = 0
        self._slo_violations: int = 0

    # ------------------------------------------------------------------
    # Core update
    # ------------------------------------------------------------------

    def step(self, observed_latency_ms: float) -> BudgetState:
        """Update the budget based on the observed request latency.

        Parameters
        ----------
        observed_latency_ms : float
            Wall-clock latency of the most recent request in milliseconds.

        Returns
        -------
        BudgetState
            Updated budget to apply to the next request.
        """
        if observed_latency_ms < 0.0:
            raise ValueError(
                f"observed_latency_ms must be >= 0; got {observed_latency_ms}."
            )
        target = self._cfg.target_latency_ms
        error = observed_latency_ms - target

        # PI update.
        self._integral += error
        # Integral wind-up guard: clamp integral to ±10× budget range.
        budget_range = float(self._cfg.kv_budget_max - self._cfg.kv_budget_min)
        max_integral = 10.0 * budget_range
        self._integral = max(-max_integral, min(max_integral, self._integral))

        delta = self._cfg.kp * error + self._cfg.ki * self._integral
        self._kv_budget -= delta
        self._kv_budget = max(
            float(self._cfg.kv_budget_min),
            min(float(self._cfg.kv_budget_max), self._kv_budget),
        )

        # Compute skip fraction proportional to latency excess.
        # Map latency ratio [0, 2×target] → skip_fraction [0, max_skip_fraction].
        latency_ratio = observed_latency_ms / target if target > 0.0 else 1.0
        # Normalise: ratio=1.0 → 0 skip; ratio=2.0 → max_skip.
        raw_skip = (latency_ratio - 1.0) * self._cfg.max_skip_fraction
        skip_fraction = max(0.0, min(self._cfg.max_skip_fraction, raw_skip))

        # Quality mode determination.
        if observed_latency_ms > 1.2 * target:
            quality_mode = "performance"
        elif observed_latency_ms < 0.8 * target:
            quality_mode = "quality"
        else:
            quality_mode = "balanced"

        self._latency_history.append(float(observed_latency_ms))
        self._n_steps += 1
        if observed_latency_ms > target:
            self._slo_violations += 1

        return BudgetState(
            kv_tokens=int(round(self._kv_budget)),
            skip_fraction=round(skip_fraction, 6),
            quality_mode=quality_mode,
        )

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset the controller state to its initial (mid-range) budget."""
        kv_mid = (self._cfg.kv_budget_min + self._cfg.kv_budget_max) // 2
        self._kv_budget = float(kv_mid)
        self._integral = 0.0
        self._latency_history = []
        self._n_steps = 0
        self._slo_violations = 0

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def current_budget(self) -> BudgetState:
        """Current budget without applying a new step."""
        # Compute skip fraction based on last latency, or 0 if no steps.
        if self._latency_history:
            last = self._latency_history[-1]
            target = self._cfg.target_latency_ms
            latency_ratio = last / target if target > 0.0 else 1.0
            raw_skip = (latency_ratio - 1.0) * self._cfg.max_skip_fraction
            skip_fraction = max(0.0, min(self._cfg.max_skip_fraction, raw_skip))
        else:
            skip_fraction = 0.0
        return BudgetState(
            kv_tokens=int(round(self._kv_budget)),
            skip_fraction=round(skip_fraction, 6),
            quality_mode="balanced",
        )

    @property
    def latency_history(self) -> List[float]:
        """List of all observed latencies in order of arrival."""
        return list(self._latency_history)

    @property
    def n_steps(self) -> int:
        """Total number of :meth:`step` calls made."""
        return self._n_steps

    def stats(self) -> BudgetStats:
        """Return aggregate statistics."""
        mean_lat = (
            sum(self._latency_history) / len(self._latency_history)
            if self._latency_history
            else 0.0
        )
        return BudgetStats(
            n_steps=self._n_steps,
            mean_latency_ms=mean_lat,
            slo_violations=self._slo_violations,
        )
