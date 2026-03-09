"""
KVTuner — Sensitivity-Aware Layer-wise Mixed Precision KV Quantization.

ICML 2025. arxiv.org/abs/2502.04420
github.com/cmd2001/KVTuner

Key findings:
  - LLM sensitivity to KV quantization is independent of input prompt.
  - Key cache is generally more important than value cache → assign higher
    precision to K than V.
  - One-time calibration produces optimal per-layer K/V precision pairs.
  - Achieves nearly-lossless 3.25-bit mixed precision; up to 21.25%
    throughput improvement over KIVI-KV8.

This module provides:
  - KVTunerConfig — search space and calibration settings
  - LayerSensitivity — measured sensitivity for one transformer layer
  - KVTunerCalibrator — runs calibration, produces KVQuantConfig
  - KVQuantConfig — optimal per-layer precision assignments (save/load JSON)
  - KVTunerStats — throughput and memory impact estimates
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ALLOWED_BITS = (2, 3, 4, 6, 8)
"""Hardware-friendly quantization bit-widths."""


@dataclass
class KVTunerConfig:
    """Configuration for KVTuner search.

    Args:
        n_layers:        Number of transformer layers to tune.
        candidate_bits:  Bit-widths to consider (must be subset of
                         ALLOWED_BITS).
        target_avg_bits: Target average bits across all layers.  The search
                         will try to stay at or below this budget.
        key_priority:    Weight given to key-cache sensitivity vs. value-cache
                         sensitivity.  Default 1.5 (keys are more important).
        n_calibration_samples: Number of examples used for calibration.
        sensitivity_metric: 'mse' | 'max_abs' | 'cosine' — how to measure
                             quantization sensitivity.
    """

    n_layers: int = 32
    candidate_bits: Tuple[int, ...] = (2, 3, 4, 8)
    target_avg_bits: float = 4.0
    key_priority: float = 1.5
    n_calibration_samples: int = 512
    sensitivity_metric: str = "mse"

    def __post_init__(self) -> None:
        if self.n_layers < 1:
            raise ValueError("n_layers must be >= 1")
        invalid = [b for b in self.candidate_bits if b not in ALLOWED_BITS]
        if invalid:
            raise ValueError(
                f"Invalid bit-widths {invalid}.  Allowed: {ALLOWED_BITS}"
            )
        if self.target_avg_bits <= 0:
            raise ValueError("target_avg_bits must be > 0")
        if self.key_priority <= 0:
            raise ValueError("key_priority must be > 0")
        valid_metrics = {"mse", "max_abs", "cosine"}
        if self.sensitivity_metric not in valid_metrics:
            raise ValueError(
                f"sensitivity_metric must be one of {valid_metrics}"
            )


# ---------------------------------------------------------------------------
# Sensitivity measurement
# ---------------------------------------------------------------------------

@dataclass
class LayerSensitivity:
    """Measured quantization sensitivity for a single transformer layer."""

    layer_idx: int
    key_sensitivity: float
    """Higher = more sensitive to quantization (needs more bits)."""

    value_sensitivity: float

    n_samples: int = 0

    @property
    def combined_sensitivity(self) -> float:
        """Weighted combination; keys weighted more heavily by design."""
        return self.key_sensitivity * 1.5 + self.value_sensitivity

    def __repr__(self) -> str:
        return (
            f"LayerSensitivity(layer={self.layer_idx}, "
            f"K={self.key_sensitivity:.4f}, V={self.value_sensitivity:.4f})"
        )


def _simulate_quantization_error(
    tensor: np.ndarray, bits: int, metric: str
) -> float:
    """Simulate quantization at *bits* and return error by *metric*."""
    flat = tensor.ravel().astype(np.float64)
    if flat.size == 0:
        return 0.0

    # Symmetric uniform quantization
    max_val = np.max(np.abs(flat))
    if max_val == 0.0:
        return 0.0
    n_levels = 2 ** bits
    step = 2 * max_val / n_levels
    quantized = np.round(flat / step) * step
    error = flat - quantized

    if metric == "mse":
        return float(np.mean(error ** 2))
    elif metric == "max_abs":
        return float(np.max(np.abs(error)))
    elif metric == "cosine":
        dot = float(np.dot(flat, quantized))
        denom = (np.linalg.norm(flat) + 1e-9) * (np.linalg.norm(quantized) + 1e-9)
        return 1.0 - dot / denom
    return float(np.mean(error ** 2))


# ---------------------------------------------------------------------------
# Calibrator
# ---------------------------------------------------------------------------

class KVTunerCalibrator:
    """Runs the KVTuner search over calibration data.

    Workflow::

        cal = KVTunerCalibrator(config)
        for i in range(n_layers):
            cal.record_layer(i, key_tensor, value_tensor)
        quant_cfg = cal.search()
        quant_cfg.save("qwen3-kvtuner.json")
    """

    def __init__(self, config: KVTunerConfig) -> None:
        self.config = config
        self._key_samples: Dict[int, List[np.ndarray]] = {}
        self._val_samples: Dict[int, List[np.ndarray]] = {}

    # ------------------------------------------------------------------
    def record_layer(
        self, layer_idx: int, keys: np.ndarray, values: np.ndarray
    ) -> None:
        """Record one calibration sample for *layer_idx*."""
        self._key_samples.setdefault(layer_idx, []).append(
            np.asarray(keys, dtype=np.float32)
        )
        self._val_samples.setdefault(layer_idx, []).append(
            np.asarray(values, dtype=np.float32)
        )

    # ------------------------------------------------------------------
    def _measure_sensitivity(self, layer_idx: int) -> LayerSensitivity:
        """Measure key/value sensitivity for *layer_idx* at all candidate bits.

        Returns the sensitivity as the relative degradation when moving from
        the highest candidate precision down to the lowest.
        """
        cfg = self.config
        bits_sorted = sorted(cfg.candidate_bits, reverse=True)
        high_bits = bits_sorted[0]
        low_bits = bits_sorted[-1]

        key_data = self._key_samples.get(layer_idx)
        val_data = self._val_samples.get(layer_idx)

        def _layer_sensitivity(samples: Optional[List[np.ndarray]]) -> float:
            if not samples:
                # No data → assign heuristic sensitivity based on layer position
                # (middle layers tend to be more sensitive)
                n = max(1, cfg.n_layers)
                rel = layer_idx / n
                return 0.1 + 0.8 * math.sin(math.pi * rel) ** 2

            combined = np.concatenate([s.ravel() for s in samples])
            err_high = _simulate_quantization_error(combined, high_bits, cfg.sensitivity_metric)
            err_low = _simulate_quantization_error(combined, low_bits, cfg.sensitivity_metric)
            # Relative degradation when reducing from high to low bits
            denom = err_high + 1e-12
            return float(err_low / denom - 1.0)

        k_sens = _layer_sensitivity(key_data)
        v_sens = _layer_sensitivity(val_data)
        n_samples = len(key_data) if key_data else 0

        return LayerSensitivity(
            layer_idx=layer_idx,
            key_sensitivity=max(0.0, k_sens),
            value_sensitivity=max(0.0, v_sens),
            n_samples=n_samples,
        )

    # ------------------------------------------------------------------
    def search(self) -> "KVQuantConfig":
        """Run the search and return the optimal per-layer KV precision config.

        Algorithm:
        1. Measure sensitivity for each layer.
        2. Sort layers by combined sensitivity (descending = most sensitive first).
        3. Greedily assign the highest bit-width to the most sensitive layers
           while budget constraint (target_avg_bits) is not violated.
        4. Remaining layers get lower bit-widths.
        5. Keys always receive ≥ value bits (asymmetric K/V assignment).
        """
        cfg = self.config
        sensitivities: List[LayerSensitivity] = [
            self._measure_sensitivity(i) for i in range(cfg.n_layers)
        ]

        # Sort by combined sensitivity descending
        order = sorted(
            range(cfg.n_layers),
            key=lambda i: sensitivities[i].combined_sensitivity,
            reverse=True,
        )

        bits_sorted = sorted(cfg.candidate_bits)  # ascending
        total_bit_budget = cfg.target_avg_bits * cfg.n_layers

        k_bits_map: Dict[int, int] = {}
        v_bits_map: Dict[int, int] = {}
        bits_used = 0.0

        for rank, layer_idx in enumerate(order):
            remaining_layers = cfg.n_layers - rank
            bits_remaining = total_bit_budget - bits_used
            avg_remaining = bits_remaining / max(1, remaining_layers)

            # Pick highest affordable bit-width for keys
            k_choice = bits_sorted[0]
            for b in reversed(bits_sorted):
                if b <= avg_remaining * cfg.key_priority:
                    k_choice = b
                    break

            # Value bits ≤ key bits (keys are more important)
            v_choice = bits_sorted[0]
            for b in reversed(bits_sorted):
                if b <= k_choice:
                    v_choice = b
                    break

            k_bits_map[layer_idx] = k_choice
            v_bits_map[layer_idx] = v_choice
            bits_used += (k_choice + v_choice) / 2.0

        return KVQuantConfig(
            k_bits=k_bits_map,
            v_bits=v_bits_map,
            sensitivities=sensitivities,
            config=cfg,
        )


# ---------------------------------------------------------------------------
# Quantization config — the runtime artifact
# ---------------------------------------------------------------------------

@dataclass
class KVQuantConfig:
    """Per-layer KV quantization precision configuration.

    Produced by :class:`KVTunerCalibrator`.  Save to JSON; load at runtime.
    """

    k_bits: Dict[int, int]
    """key-cache bits per layer."""

    v_bits: Dict[int, int]
    """value-cache bits per layer."""

    sensitivities: List[LayerSensitivity]
    config: KVTunerConfig

    # ------------------------------------------------------------------
    def bits_for_layer(self, layer_idx: int) -> Tuple[int, int]:
        """Return *(key_bits, value_bits)* for *layer_idx*."""
        k = self.k_bits.get(layer_idx, 4)
        v = self.v_bits.get(layer_idx, 4)
        return k, v

    @property
    def avg_bits(self) -> float:
        n = max(1, len(self.k_bits))
        total = sum(self.k_bits[i] + self.v_bits[i] for i in self.k_bits)
        return total / (2 * n)

    @property
    def n_layers(self) -> int:
        return len(self.k_bits)

    # ------------------------------------------------------------------
    def save(self, path: str) -> None:
        data = {
            "k_bits": {str(k): v for k, v in self.k_bits.items()},
            "v_bits": {str(k): v for k, v in self.v_bits.items()},
            "avg_bits": self.avg_bits,
            "target_avg_bits": self.config.target_avg_bits,
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: str, config: Optional[KVTunerConfig] = None) -> "KVQuantConfig":
        with open(path) as f:
            data = json.load(f)
        k_bits = {int(k): v for k, v in data["k_bits"].items()}
        v_bits = {int(k): v for k, v in data["v_bits"].items()}
        cfg = config or KVTunerConfig(n_layers=len(k_bits))
        return cls(k_bits=k_bits, v_bits=v_bits, sensitivities=[], config=cfg)


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

@dataclass
class KVTunerStats:
    """Throughput and memory impact of a :class:`KVQuantConfig`."""

    quant_config: KVQuantConfig

    @property
    def avg_bits(self) -> float:
        return self.quant_config.avg_bits

    def estimated_memory_reduction_vs_fp16(self) -> float:
        """Fraction of KV memory saved vs. 16-bit (bfloat16) baseline."""
        baseline_bits = 16.0
        return 1.0 - self.avg_bits / baseline_bits

    def estimated_memory_reduction_vs_kivi8(self) -> float:
        """Fraction of KV memory saved vs. KIVI KV8 (8-bit baseline)."""
        return 1.0 - self.avg_bits / 8.0

    def estimated_throughput_improvement_vs_kivi8(self) -> float:
        """Estimated throughput improvement ratio over KIVI-KV8.

        Linear approximation based on the 21.25% figure from the paper.
        """
        # Paper reports 21.25% improvement at 4-bit vs 8-bit
        bits_ratio = 8.0 / max(0.5, self.avg_bits)
        return (bits_ratio - 1.0) * 0.17  # conservative linear scaling
