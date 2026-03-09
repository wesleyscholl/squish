"""
CLA — Cross-Layer Attention architecture module.

Brandon et al., 2024. arxiv.org/abs/2405.12981
Referenced in NAACL 2025 systematic study.

CLA bakes cross-layer KV sharing into model architecture at training time.
Certain transformer layers are "KV-generating" layers; others consume the KV
of the nearest upstream generator.  Reduces KV cache by 2× with competitive
performance.

This module provides:
  - CLAConfig — architecture config for designing CLA-style models
  - CLALayerSpec — specifies whether a layer generates or borrows KV
  - CLASchedule — a complete schedule for an N-layer model
  - CLAStats — memory and compute statistics for a given schedule
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class CLAConfig:
    """Architecture configuration for a CLA-style transformer.

    Args:
        n_layers:       Total number of transformer layers.
        sharing_factor: How many consecutive layers share one KV generator.
                        sharing_factor=2 → every other layer generates KV (2×
                        KV cache reduction). sharing_factor=4 → 4× reduction.
        generator_stride: Index stride for generator placement.
                          1 = first layer in each group is generator.
        allow_first_layer_borrow: Whether layer 0 may borrow (usually False —
                          the very first layer has no upstream generator).
    """

    n_layers: int = 32
    sharing_factor: int = 2
    generator_stride: int = 0
    """Offset of the generator within each group (0 = first in group)."""

    allow_first_layer_borrow: bool = False

    def __post_init__(self) -> None:
        if self.n_layers < 2:
            raise ValueError("n_layers must be >= 2")
        if self.sharing_factor < 1:
            raise ValueError("sharing_factor must be >= 1")
        if not (0 <= self.generator_stride < self.sharing_factor):
            raise ValueError(
                "generator_stride must be in [0, sharing_factor)"
            )


# ---------------------------------------------------------------------------
# Layer spec
# ---------------------------------------------------------------------------

@dataclass
class CLALayerSpec:
    """Specification for a single transformer layer in a CLA model."""

    layer_idx: int
    is_generator: bool
    """True → this layer computes and stores its own KV cache."""

    borrows_from: Optional[int]
    """Layer index to borrow from (None if this layer is a generator)."""

    @property
    def is_borrower(self) -> bool:
        return not self.is_generator

    def __repr__(self) -> str:
        if self.is_generator:
            return f"CLALayerSpec(layer={self.layer_idx}, GENERATOR)"
        return (
            f"CLALayerSpec(layer={self.layer_idx}, "
            f"BORROWER←{self.borrows_from})"
        )


# ---------------------------------------------------------------------------
# Schedule
# ---------------------------------------------------------------------------

@dataclass
class CLASchedule:
    """Complete CLA sharing schedule for an N-layer transformer.

    Build via :meth:`from_config`.
    """

    specs: List[CLALayerSpec]
    config: CLAConfig

    # ------------------------------------------------------------------
    @classmethod
    def from_config(cls, config: CLAConfig) -> "CLASchedule":
        """Build the schedule from *config*.

        Each group of `sharing_factor` consecutive layers has exactly one
        generator (at position `generator_stride` within the group); all
        other layers in the group borrow from it.  Layer 0 is always a
        generator when `allow_first_layer_borrow=False`.
        """
        specs: List[CLALayerSpec] = []
        sf = config.sharing_factor
        gs = config.generator_stride

        for i in range(config.n_layers):
            group = i // sf
            pos_in_group = i % sf

            is_gen = (pos_in_group == gs)

            # Protect the very first layer if configured
            if i == 0 and not config.allow_first_layer_borrow:
                is_gen = True

            if is_gen:
                spec = CLALayerSpec(
                    layer_idx=i, is_generator=True, borrows_from=None
                )
            else:
                # Find the generator in the same group
                generator_idx = group * sf + gs
                # Clamp to valid range
                generator_idx = max(0, min(generator_idx, config.n_layers - 1))
                spec = CLALayerSpec(
                    layer_idx=i,
                    is_generator=False,
                    borrows_from=generator_idx,
                )

            specs.append(spec)

        return cls(specs=specs, config=config)

    # ------------------------------------------------------------------
    def spec_for(self, layer_idx: int) -> CLALayerSpec:
        return self.specs[layer_idx]

    @property
    def generator_layers(self) -> List[int]:
        return [s.layer_idx for s in self.specs if s.is_generator]

    @property
    def borrower_layers(self) -> List[int]:
        return [s.layer_idx for s in self.specs if s.is_borrower]

    @property
    def n_generators(self) -> int:
        return len(self.generator_layers)

    @property
    def n_borrowers(self) -> int:
        return len(self.borrower_layers)

    def kv_cache_reduction_factor(self) -> float:
        """Ratio of KV slots needed vs. a standard N-layer model."""
        n = self.config.n_layers
        return self.n_generators / max(1, n)

    def summary(self) -> str:
        lines = [
            f"CLASchedule: {self.n_generators} generators, "
            f"{self.n_borrowers} borrowers "
            f"(KV cache {self.kv_cache_reduction_factor():.1%} of baseline)"
        ]
        for spec in self.specs:
            lines.append(f"  {spec}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

@dataclass
class CLAStats:
    """Memory and compute statistics for a CLA schedule."""

    schedule: CLASchedule
    head_dim: int = 128
    n_kv_heads: int = 8
    seq_len: int = 2048
    dtype_bytes: int = 2  # bfloat16

    @property
    def kv_bytes_standard(self) -> int:
        """KV bytes for a standard (non-CLA) transformer."""
        n = self.schedule.config.n_layers
        return (
            2 * n * self.n_kv_heads * self.seq_len * self.head_dim * self.dtype_bytes
        )

    @property
    def kv_bytes_cla(self) -> int:
        """KV bytes for the CLA model (only generators allocate KV)."""
        g = self.schedule.n_generators
        return (
            2 * g * self.n_kv_heads * self.seq_len * self.head_dim * self.dtype_bytes
        )

    @property
    def kv_memory_reduction_ratio(self) -> float:
        """1 - (CLA bytes / standard bytes)."""
        return 1.0 - self.kv_bytes_cla / max(1, self.kv_bytes_standard)

    @property
    def kv_cache_multiplier(self) -> float:
        """CLA / standard KV memory ratio (< 1.0 means CLA is smaller)."""
        return self.kv_bytes_cla / max(1, self.kv_bytes_standard)
