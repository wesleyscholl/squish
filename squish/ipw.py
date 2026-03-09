"""
squish/ipw.py

Intelligence Per Watt (IPW) — Evaluation Framework for Local LLM Deployment.

Based on:
  "Local LLMs: State of the Union — An Empirical Study of Intelligence Per
   Watt for Battery-Powered Inference"
  arXiv:2511.07885 — Nov 2025

Problem
-------
Existing inference benchmarks optimise for *throughput* (tokens/second) or
*latency* (TTFT, e2e latency).  For battery-powered deployments (MacBook M3),
neither metric accounts for *energy cost*.  A configuration that generates
100 tok/s while drawing 40W is strictly worse than one that generates 80 tok/s
at 20W: the first depletes the battery twice as fast for only 25% more speed.

IPW
---
  IPW = quality_score / energy_joules

where *quality_score* is a normalised task-completion metric (0–1) and
*energy_joules* is the total energy consumed to complete the request.

The empirical study found:
- Local LMs answer 88.7% of single-turn queries successfully.
- The IPW-optimal configuration is often *not* the peak-throughput config.
- For simple tasks: smaller model + faster inference dominates IPW.
- For complex tasks: larger model + aggressive caching dominates IPW.
- IPW makes the model/config selection tradeoff quantitative.

Squish integration
------------------
- After each request, call ``IPWTracker.record()`` with the quality score
  (from automated eval or a proxy like acceptance-rate), energy consumed
  (from powermetrics or a hw power monitor), and tokens generated.
- Call ``tracker.summary()`` to get the IPW distribution by task type.
- Use the summary to auto-tune: route future tasks to the config with highest
  mean IPW for that task type.

Conflict notes
--------------
- **No conflict** with any inference technique — IPW is a *measurement layer*
  that wraps any configuration.
- **Synergy with EnergyAwareScheduler**: the energy field comes directly from
  ``PowerMonitor.read_joules()`` already wired in the scheduler.
- **Synergy with ForeLen / TRAIL**: predicted output length feeds the energy
  estimate before the request completes (early-exit IPW prediction).

Provides
--------
  IPWConfig         — configuration (energy unit, quality weight).
  IPWMeasurement    — single request measurement with derived metrics.
  IPWTracker        — accumulates and aggregates measurements.
  IPWSummary        — statistical summary of IPW across task types.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

__all__ = [
    "IPWConfig",
    "IPWMeasurement",
    "IPWTracker",
    "IPWSummary",
]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class IPWConfig:
    """Configuration for IPW measurement.

    Parameters
    ----------
    energy_unit:
        Unit for the energy field in ``IPWMeasurement``.
        ``"mJ"`` (millijoules) is recommended for per-request granularity;
        ``"J"`` for long sessions.
    quality_weight:
        Scalar multiplier applied to *quality_score* before dividing by
        energy. Default 1.0 (no re-weighting).
    min_energy_mj:
        Floor for the energy denominator to prevent division by zero for
        instant cache hits.  In mJ.
    """

    energy_unit: str = "mJ"
    quality_weight: float = 1.0
    min_energy_mj: float = 0.1

    def __post_init__(self) -> None:
        if self.energy_unit not in ("mJ", "J", "uJ"):
            raise ValueError("energy_unit must be 'mJ', 'J', or 'uJ'")
        if self.quality_weight <= 0.0:
            raise ValueError("quality_weight must be > 0")
        if self.min_energy_mj < 0.0:
            raise ValueError("min_energy_mj must be >= 0")


# ---------------------------------------------------------------------------
# IPWMeasurement
# ---------------------------------------------------------------------------

@dataclass
class IPWMeasurement:
    """A single request's quality and energy measurement.

    Attributes
    ----------
    quality_score:
        Task-completion metric in [0, 1].  For code tasks: tests passing / total.
        For chat: ROUGE-L or a proxy acceptance rate.  For structured output:
        schema validity flag.
    energy_mj:
        Energy consumed by the inference request in millijoules.
    time_ms:
        Wall-clock latency for the request in milliseconds.
    tokens_generated:
        Number of output tokens produced.
    task_type:
        Optional label for task routing (e.g., ``"git_commit"``,
        ``"devops_plan"``, ``"code_review"``).
    config_label:
        Optional label for the inference configuration used (e.g.,
        ``"qwen3-8b-int4"``, ``"qwen3-8b-bf16"``).
    """

    quality_score: float
    energy_mj: float
    time_ms: float
    tokens_generated: int
    task_type: str = "unknown"
    config_label: str = "default"

    def __post_init__(self) -> None:
        if not (0.0 <= self.quality_score <= 1.0):
            raise ValueError("quality_score must be in [0, 1]")
        if self.energy_mj < 0.0:
            raise ValueError("energy_mj must be >= 0")
        if self.time_ms < 0.0:
            raise ValueError("time_ms must be >= 0")
        if self.tokens_generated < 0:
            raise ValueError("tokens_generated must be >= 0")

    @property
    def ipw(self, min_energy_mj: float = 0.1) -> float:
        """IPW = quality_score / energy_mj (floored at min_energy_mj)."""
        return self.quality_score / max(self.energy_mj, min_energy_mj)

    @property
    def tokens_per_joule(self) -> float:
        energy_j = self.energy_mj / 1000.0
        return self.tokens_generated / energy_j if energy_j > 0 else 0.0

    @property
    def tokens_per_second(self) -> float:
        time_s = self.time_ms / 1000.0
        return self.tokens_generated / time_s if time_s > 0 else 0.0


# ---------------------------------------------------------------------------
# IPWTracker
# ---------------------------------------------------------------------------

class IPWTracker:
    """Accumulates IPW measurements and produces aggregate summaries.

    Parameters
    ----------
    config:
        ``IPWConfig``.
    """

    def __init__(self, config: IPWConfig | None = None) -> None:
        self._cfg = config or IPWConfig()
        self._measurements: list[IPWMeasurement] = []

    def record(self, measurement: IPWMeasurement) -> None:
        """Append a measurement to the tracker."""
        self._measurements.append(measurement)

    def record_values(
        self,
        quality_score: float,
        energy_mj: float,
        time_ms: float,
        tokens_generated: int,
        task_type: str = "unknown",
        config_label: str = "default",
    ) -> IPWMeasurement:
        """Convenience helper: build and record an ``IPWMeasurement``."""
        m = IPWMeasurement(
            quality_score=quality_score,
            energy_mj=energy_mj,
            time_ms=time_ms,
            tokens_generated=tokens_generated,
            task_type=task_type,
            config_label=config_label,
        )
        self.record(m)
        return m

    def summary(self) -> IPWSummary:
        """Compute a statistical summary across all recorded measurements."""
        return IPWSummary.from_measurements(self._measurements, self._cfg)

    def summary_by_task(self) -> dict[str, IPWSummary]:
        """Per-task-type summaries."""
        by_type: dict[str, list[IPWMeasurement]] = {}
        for m in self._measurements:
            by_type.setdefault(m.task_type, []).append(m)
        return {
            task: IPWSummary.from_measurements(ms, self._cfg)
            for task, ms in by_type.items()
        }

    @property
    def total_measurements(self) -> int:
        return len(self._measurements)

    def reset(self) -> None:
        self._measurements.clear()


# ---------------------------------------------------------------------------
# IPWSummary
# ---------------------------------------------------------------------------

@dataclass
class IPWSummary:
    """Statistical summary of IPW across a collection of measurements.

    Attributes
    ----------
    count:
        Number of measurements in this summary.
    mean_ipw:
        Mean Intelligence Per Watt.
    median_ipw:
        Median IPW.
    p90_ipw:
        90th-percentile IPW.
    mean_quality:
        Mean quality score.
    mean_energy_mj:
        Mean energy per request (mJ).
    mean_tokens_per_second:
        Mean throughput.
    best_config:
        Config label with the highest mean IPW (if config labels differ).
    """

    count: int
    mean_ipw: float
    median_ipw: float
    p90_ipw: float
    mean_quality: float
    mean_energy_mj: float
    mean_tokens_per_second: float
    best_config: str = "default"

    @staticmethod
    def from_measurements(
        measurements: list[IPWMeasurement],
        config: IPWConfig | None = None,
    ) -> IPWSummary:
        """Build an ``IPWSummary`` from a list of measurements."""
        cfg = config or IPWConfig()
        if not measurements:
            return IPWSummary(
                count=0,
                mean_ipw=0.0,
                median_ipw=0.0,
                p90_ipw=0.0,
                mean_quality=0.0,
                mean_energy_mj=0.0,
                mean_tokens_per_second=0.0,
            )

        ipw_vals = np.array([
            m.quality_score * cfg.quality_weight / max(m.energy_mj, cfg.min_energy_mj)
            for m in measurements
        ])
        qualities = np.array([m.quality_score for m in measurements])
        energies = np.array([m.energy_mj for m in measurements])
        tps_vals = np.array([m.tokens_per_second for m in measurements])

        # Best config by mean IPW
        config_ipw: dict[str, list[float]] = {}
        for m, ipw in zip(measurements, ipw_vals, strict=False):
            config_ipw.setdefault(m.config_label, []).append(float(ipw))
        best_config = max(config_ipw, key=lambda k: np.mean(config_ipw[k]))

        return IPWSummary(
            count=len(measurements),
            mean_ipw=float(np.mean(ipw_vals)),
            median_ipw=float(np.median(ipw_vals)),
            p90_ipw=float(np.percentile(ipw_vals, 90)),
            mean_quality=float(np.mean(qualities)),
            mean_energy_mj=float(np.mean(energies)),
            mean_tokens_per_second=float(np.mean(tps_vals)),
            best_config=best_config,
        )
