# [Experimental] This module is part of Squish v10+ (Wave 10A).
# Proof-of-concept quality: API and behaviour may change without notice.
#!/usr/bin/env python3
"""
squish/neuron_profile.py

PowerInfer-style offline neuron profiling for Apple Silicon LLMs.

Profiles FFN activation frequency across calibration inputs and produces a
persistent ``NeuronProfile`` that ``NeuronRouter`` uses at inference time to
split each MLP layer's weight matrix into hot (frequently activated) and cold
(rarely activated) neuron sub-tensors, routing hot neurons to the GPU's on-chip
SRAM and cold neurons to CPU/DRAM.

Key insight (Song et al., SOSP 2024): ~20% of neurons in each MLP layer are
"hot" — active in >80% of calibration forward passes.  Loading only those rows
from DRAM reduces effective memory bandwidth per decode step significantly.

Usage::

    from squish.neuron_profile import NeuronProfileConfig, NeuronProfiler, load_profile

    config  = NeuronProfileConfig(hot_fraction=0.20, n_calib_samples=512)
    profiler = NeuronProfiler(config)

    # act_counts[i] is a numpy array of shape (ffn_dim,) holding per-neuron
    # activation counts across all calibration samples for layer i.
    profile = profiler.calibrate(act_counts_per_layer)
    profile.save("neuron_profile.json")

    # Later, at server startup:
    profile = load_profile("neuron_profile.json")
"""

from __future__ import annotations

__all__ = [
    "NeuronProfileConfig",
    "NeuronProfiler",
    "NeuronProfile",
    "load_profile",
]

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import numpy as np


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class NeuronProfileConfig:
    """Configuration for the offline neuron profiling pass.

    Attributes:
        n_calib_samples: Number of calibration texts to profile over
                         (default 512).
        hot_fraction:    Fraction of neurons per layer considered "hot"
                         (default 0.20).  Hot neurons are loaded first on
                         the GPU path.
        save_path:       Default filesystem path for ``NeuronProfile.save()``.
                         Empty string means caller must supply a path
                         explicitly.
    """

    n_calib_samples: int = 512
    hot_fraction: float = 0.20
    save_path: str = ""

    def __post_init__(self) -> None:
        if self.n_calib_samples < 1:
            raise ValueError(
                f"n_calib_samples must be >= 1, got {self.n_calib_samples}"
            )
        if not (0.0 < self.hot_fraction < 1.0):
            raise ValueError(
                f"hot_fraction must be in (0, 1), got {self.hot_fraction}"
            )


# ---------------------------------------------------------------------------
# NeuronProfile — per-layer hot/cold index store
# ---------------------------------------------------------------------------

@dataclass
class NeuronProfile:
    """Per-layer hot and cold neuron indices derived from calibration.

    Each element of ``hot_indices`` and ``cold_indices`` is a 1-D
    ``numpy.ndarray`` of dtype ``int64`` holding the column indices of the
    weight matrix rows in that tier for the corresponding layer.

    Attributes:
        hot_indices:  List of one array per layer; each array contains the
                      indices of the "hot" neurons sorted in descending order
                      of activation frequency.
        cold_indices: List of one array per layer; each array contains the
                      remaining "cold" neuron indices.
        hot_fraction: The hot fraction used when this profile was created.
    """

    hot_indices: List[np.ndarray] = field(default_factory=list)
    cold_indices: List[np.ndarray] = field(default_factory=list)
    hot_fraction: float = 0.20

    # ------------------------------------------------------------------
    # Derived properties
    # ------------------------------------------------------------------

    @property
    def layer_count(self) -> int:
        """Number of profiled layers."""
        return len(self.hot_indices)

    def n_hot(self, layer_idx: int) -> int:
        """Number of hot neurons in layer ``layer_idx``."""
        self._check_layer(layer_idx)
        return int(self.hot_indices[layer_idx].shape[0])

    def n_cold(self, layer_idx: int) -> int:
        """Number of cold neurons in layer ``layer_idx``."""
        self._check_layer(layer_idx)
        return int(self.cold_indices[layer_idx].shape[0])

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        """Serialise to a JSON-compatible dict (indices as plain lists)."""
        return {
            "hot_fraction": self.hot_fraction,
            "hot_indices":  [arr.tolist() for arr in self.hot_indices],
            "cold_indices": [arr.tolist() for arr in self.cold_indices],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "NeuronProfile":
        """Deserialise from a dict (inverse of :meth:`to_dict`)."""
        return cls(
            hot_fraction=float(d.get("hot_fraction", 0.20)),
            hot_indices=[
                np.asarray(arr, dtype=np.int64) for arr in d["hot_indices"]
            ],
            cold_indices=[
                np.asarray(arr, dtype=np.int64) for arr in d["cold_indices"]
            ],
        )

    def save(self, path: str) -> None:
        """Write profile to a JSON file.

        Args:
            path: Filesystem path; parent directories are created if needed.
        """
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: str) -> "NeuronProfile":
        """Load profile from a JSON file.

        Args:
            path: Filesystem path written by :meth:`save`.

        Raises:
            FileNotFoundError: if the file does not exist.
        """
        return cls.from_dict(json.loads(Path(path).read_text()))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _check_layer(self, layer_idx: int) -> None:
        if not (0 <= layer_idx < self.layer_count):
            raise IndexError(
                f"layer_idx {layer_idx} out of range for profile with "
                f"{self.layer_count} layers"
            )

    def __repr__(self) -> str:
        if self.layer_count == 0:
            return "NeuronProfile(layers=0)"
        n_hot_0 = self.n_hot(0)
        n_total = n_hot_0 + self.n_cold(0)
        return (
            f"NeuronProfile(layers={self.layer_count}  "
            f"hot_fraction={self.hot_fraction:.2f}  "
            f"hot_per_layer≈{n_hot_0}/{n_total})"
        )


# ---------------------------------------------------------------------------
# NeuronProfiler
# ---------------------------------------------------------------------------

class NeuronProfiler:
    """Offline profiler that produces a :class:`NeuronProfile`.

    The profiler operates on pre-computed activation-count arrays (one per
    layer) rather than raw model weights or tokenized inputs.  This makes
    the class hardware-agnostic and fully unit-testable without MLX.

    In a real workflow the caller would:

    1. Forward ``n_calib_samples`` prompts through the model, accumulating
       per-neuron activation counts for each MLP layer.
    2. Pass the resulting list of count arrays to :meth:`calibrate`.

    Args:
        config: A :class:`NeuronProfileConfig` governing the hot fraction and
                calibration sample count.
    """

    def __init__(self, config: NeuronProfileConfig | None = None) -> None:
        self.config = config or NeuronProfileConfig()

    def calibrate(
        self,
        act_counts_per_layer: list[np.ndarray],
    ) -> NeuronProfile:
        """Build a :class:`NeuronProfile` from per-layer activation counts.

        Args:
            act_counts_per_layer: List of 1-D float or int arrays; element
                ``i`` holds per-neuron activation frequencies / counts for
                MLP layer ``i``.

        Returns:
            A :class:`NeuronProfile` whose hot/cold indices are sorted by
            activation frequency (hot: most frequently active first; cold:
            least frequently active first).

        Raises:
            ValueError: If ``act_counts_per_layer`` is empty.
        """
        if not act_counts_per_layer:
            raise ValueError("act_counts_per_layer must not be empty")

        hot_indices: list[np.ndarray] = []
        cold_indices: list[np.ndarray] = []

        for counts in act_counts_per_layer:
            counts = np.asarray(counts, dtype=np.float32)
            n = counts.shape[0]
            n_hot = max(1, int(round(n * self.config.hot_fraction)))
            # argsort descending — highest frequency first
            order = np.argsort(-counts)
            hot_indices.append(order[:n_hot].astype(np.int64))
            cold_indices.append(order[n_hot:].astype(np.int64))

        return NeuronProfile(
            hot_indices=hot_indices,
            cold_indices=cold_indices,
            hot_fraction=self.config.hot_fraction,
        )


# ---------------------------------------------------------------------------
# Convenience loader
# ---------------------------------------------------------------------------

def load_profile(path: str) -> NeuronProfile:
    """Load a :class:`NeuronProfile` from a JSON file.

    Convenience wrapper around :meth:`NeuronProfile.load`.

    Args:
        path: Filesystem path written by :meth:`NeuronProfile.save`.

    Returns:
        A populated :class:`NeuronProfile`.

    Raises:
        FileNotFoundError: if the file does not exist.
    """
    return NeuronProfile.load(path)
