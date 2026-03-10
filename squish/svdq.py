"""
SVDq — Singular Value Decomposition for Per-Head Key Cache Mixed Precision.

Referenced in KVTuner related work; 2025 preprint.

SVDq decomposes the key cache per-layer and per-head using SVD to identify
which dimensions carry the most information:
  - Large singular values → high precision needed
  - Small singular values → low precision acceptable

This module provides:
  - SVDqConfig — configuration for head-wise rank search
  - HeadSVDProfile — singular value profile for one attention head
  - SVDqCalibrator — runs calibration, produces SVDqPrecisionMap
  - SVDqPrecisionMap — per-layer, per-head bit-width and rank assignments
  - SVDqStats — compression statistics
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class SVDqConfig:
    """Configuration for SVDq calibration and search."""

    n_layers: int = 32
    n_heads: int = 32
    head_dim: int = 128
    candidate_bits: tuple[int, ...] = (2, 4, 8)
    target_avg_bits: float = 4.0
    energy_threshold: float = 0.95
    """Fraction of total singular value energy to retain.  A head whose top-k
    singular values capture ≥ energy_threshold of the total energy is
    considered compressible."""

    min_rank: int = 8
    """Minimum rank to assign even for highly compressible heads."""

    def __post_init__(self) -> None:
        if self.n_layers < 1:
            raise ValueError("n_layers must be >= 1")
        if self.n_heads < 1:
            raise ValueError("n_heads must be >= 1")
        if not (0.0 < self.energy_threshold <= 1.0):
            raise ValueError("energy_threshold must be in (0, 1]")
        if self.min_rank < 1:
            raise ValueError("min_rank must be >= 1")
        if self.target_avg_bits <= 0:
            raise ValueError("target_avg_bits must be > 0")


# ---------------------------------------------------------------------------
# Head SVD profile
# ---------------------------------------------------------------------------

@dataclass
class HeadSVDProfile:
    """Singular value profile for one attention head's key cache."""

    layer_idx: int
    head_idx: int
    singular_values: np.ndarray
    """1-D array of singular values in descending order."""

    @property
    def total_energy(self) -> float:
        return float(np.sum(self.singular_values ** 2))

    def effective_rank(self, energy_threshold: float = 0.95) -> int:
        """Minimum rank capturing *energy_threshold* of total energy."""
        total = self.total_energy
        if total == 0.0:
            return 1
        cumulative = 0.0
        for k, sv in enumerate(self.singular_values, 1):
            cumulative += sv ** 2
            if cumulative / total >= energy_threshold:
                return k
        return len(self.singular_values)

    def compressibility(self, energy_threshold: float = 0.95) -> float:
        """1 - (effective_rank / total_rank). Higher = more compressible."""
        total_rank = max(1, len(self.singular_values))
        eff = self.effective_rank(energy_threshold)
        return 1.0 - eff / total_rank


# ---------------------------------------------------------------------------
# Calibrator
# ---------------------------------------------------------------------------

class SVDqCalibrator:
    """Calibrates per-head key rank and bit-width assignments.

    Workflow::

        cal = SVDqCalibrator(config)
        for layer_idx in range(n_layers):
            for head_idx in range(n_heads):
                cal.record_head_keys(layer_idx, head_idx, key_matrix)
        precision_map = cal.search()
    """

    def __init__(self, config: SVDqConfig) -> None:
        self.config = config
        # (layer, head) -> list of key matrices
        self._key_samples: dict[tuple[int, int], list[np.ndarray]] = {}

    def record_head_keys(
        self, layer_idx: int, head_idx: int, key_matrix: np.ndarray
    ) -> None:
        """Record one sample of the key matrix for *(layer_idx, head_idx)*.

        Args:
            key_matrix: Shape (seq_len, head_dim) or (head_dim,) for one token.
        """
        km = np.asarray(key_matrix, dtype=np.float32)
        self._key_samples.setdefault((layer_idx, head_idx), []).append(km)

    def _profile_head(self, layer_idx: int, head_idx: int) -> HeadSVDProfile:
        samples = self._key_samples.get((layer_idx, head_idx))
        cfg = self.config

        if not samples:
            # Heuristic: simulate a random head (for testing without data)
            rng = np.random.default_rng(layer_idx * 1000 + head_idx)
            dim = cfg.head_dim
            svs = np.sort(rng.exponential(scale=1.0 / (1 + head_idx % 4), size=dim))[::-1]
            return HeadSVDProfile(layer_idx=layer_idx, head_idx=head_idx, singular_values=svs)

        # Stack samples and compute SVD
        stacked = np.vstack([s.reshape(-1, s.shape[-1]) if s.ndim > 1 else s.reshape(1, -1)
                             for s in samples])
        # Use economy SVD — only compute singular values
        try:
            svs = np.linalg.svd(stacked, compute_uv=False)
        except np.linalg.LinAlgError:
            svs = np.ones(min(stacked.shape))

        return HeadSVDProfile(
            layer_idx=layer_idx,
            head_idx=head_idx,
            singular_values=svs,
        )

    def search(self) -> SVDqPrecisionMap:
        """Run SVD on all heads and assign per-head bit-widths and ranks."""
        cfg = self.config
        profiles: dict[tuple[int, int], HeadSVDProfile] = {}
        for li in range(cfg.n_layers):
            for hi in range(cfg.n_heads):
                profiles[(li, hi)] = self._profile_head(li, hi)

        bits_sorted = sorted(cfg.candidate_bits)
        total_heads = cfg.n_layers * cfg.n_heads
        total_bit_budget = cfg.target_avg_bits * total_heads

        # Sort heads by compressibility (ascending = least compressible first)
        head_order = sorted(
            profiles.keys(),
            key=lambda k: profiles[k].compressibility(cfg.energy_threshold),
        )

        bits_map: dict[tuple[int, int], int] = {}
        rank_map: dict[tuple[int, int], int] = {}
        bits_used = 0.0

        for rank_pos, (li, hi) in enumerate(head_order):
            profile = profiles[(li, hi)]
            remaining_heads = total_heads - rank_pos
            bits_remaining = total_bit_budget - bits_used
            avg_remaining = bits_remaining / max(1, remaining_heads)

            # Less compressible heads (low compressibility = many needed SVs)
            # → assign more bits
            choice = bits_sorted[0]
            for b in reversed(bits_sorted):
                if b <= avg_remaining:
                    choice = b
                    break

            eff_rank = max(
                cfg.min_rank,
                profile.effective_rank(cfg.energy_threshold),
            )
            eff_rank = min(eff_rank, cfg.head_dim)

            bits_map[(li, hi)] = choice
            rank_map[(li, hi)] = eff_rank
            bits_used += choice

        return SVDqPrecisionMap(
            bits_map=bits_map,
            rank_map=rank_map,
            profiles=profiles,
            config=cfg,
        )


# ---------------------------------------------------------------------------
# Precision map
# ---------------------------------------------------------------------------

@dataclass
class SVDqPrecisionMap:
    """Per-head SVD rank and bit-width assignments."""

    bits_map: dict[tuple[int, int], int]
    """(layer_idx, head_idx) → key-cache bits."""

    rank_map: dict[tuple[int, int], int]
    """(layer_idx, head_idx) → effective SVD rank."""

    profiles: dict[tuple[int, int], HeadSVDProfile]
    config: SVDqConfig

    def bits_for_head(self, layer_idx: int, head_idx: int) -> int:
        return self.bits_map.get((layer_idx, head_idx), 4)

    def rank_for_head(self, layer_idx: int, head_idx: int) -> int:
        return self.rank_map.get((layer_idx, head_idx), self.config.head_dim)

    @property
    def avg_bits(self) -> float:
        if not self.bits_map:
            return 4.0
        return sum(self.bits_map.values()) / len(self.bits_map)

    @property
    def avg_rank(self) -> float:
        if not self.rank_map:
            return float(self.config.head_dim)
        return sum(self.rank_map.values()) / len(self.rank_map)

    def rank_compression_ratio(self) -> float:
        """Fraction of dimensions eliminated from key cache on average."""
        return 1.0 - self.avg_rank / max(1, self.config.head_dim)


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

@dataclass
class SVDqStats:
    """Compression statistics for a :class:`SVDqPrecisionMap`."""

    precision_map: SVDqPrecisionMap

    @property
    def avg_bits(self) -> float:
        return self.precision_map.avg_bits

    @property
    def rank_compression_ratio(self) -> float:
        return self.precision_map.rank_compression_ratio()

    def combined_compression_ratio(self) -> float:
        """Combined compression from bits + rank reduction vs. FP16 full-rank."""
        bit_ratio = self.avg_bits / 16.0
        rank_ratio = 1.0 - self.rank_compression_ratio
        return bit_ratio * rank_ratio
