"""
YOCO — You Only Cache Once.

Sun et al., 2024. arxiv.org/abs/2405.05254

Architecture: the first half of the transformer computes full self-attention
and caches KV.  The second half reuses this single shared KV cache via
cross-attention — only ONE set of KV tensors is cached for the entire model,
halving total KV memory regardless of context length.

This module provides:
  - YOCOConfig — architecture config
  - YOCOKVStore — the single shared KV cache
  - YOCOLayerSpec — per-layer role (self-attn vs. cross-attn)
  - YOCOSchedule — complete role assignment for an N-layer model
  - YOCOStats — memory comparison vs. standard transformer
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class YOCOConfig:
    """Architecture configuration for a YOCO-style transformer.

    The first ``n_self_attn_layers`` layers compute standard self-attention
    and write to the shared KV store.  The remaining layers consume the
    shared KV via cross-attention.

    Setting ``n_self_attn_layers = n_layers // 2`` halves KV memory.
    """

    n_layers: int = 32
    n_self_attn_layers: int = 16
    """Number of layers in the self-attention prefix (writes to KV store)."""

    head_dim: int = 128
    n_kv_heads: int = 8

    def __post_init__(self) -> None:
        if self.n_layers < 2:
            raise ValueError("n_layers must be >= 2")
        if not (1 <= self.n_self_attn_layers < self.n_layers):
            raise ValueError(
                "n_self_attn_layers must be in [1, n_layers)"
            )
        if self.head_dim < 1:
            raise ValueError("head_dim must be >= 1")
        if self.n_kv_heads < 1:
            raise ValueError("n_kv_heads must be >= 1")

    @property
    def n_cross_attn_layers(self) -> int:
        return self.n_layers - self.n_self_attn_layers


# ---------------------------------------------------------------------------
# Layer spec
# ---------------------------------------------------------------------------

@dataclass
class YOCOLayerSpec:
    """Role of a single transformer layer in a YOCO model."""

    layer_idx: int
    role: str
    """'self' for self-attention layers, 'cross' for cross-attention layers."""

    @property
    def is_self_attn(self) -> bool:
        return self.role == "self"

    @property
    def is_cross_attn(self) -> bool:
        return self.role == "cross"

    def __repr__(self) -> str:
        return f"YOCOLayerSpec(layer={self.layer_idx}, {self.role.upper()}-ATTN)"


# ---------------------------------------------------------------------------
# Schedule
# ---------------------------------------------------------------------------

@dataclass
class YOCOSchedule:
    """Layer role assignment for a YOCO model.

    Build via :meth:`from_config`.
    """

    specs: List[YOCOLayerSpec]
    config: YOCOConfig

    @classmethod
    def from_config(cls, config: YOCOConfig) -> "YOCOSchedule":
        specs = []
        for i in range(config.n_layers):
            role = "self" if i < config.n_self_attn_layers else "cross"
            specs.append(YOCOLayerSpec(layer_idx=i, role=role))
        return cls(specs=specs, config=config)

    def spec_for(self, layer_idx: int) -> YOCOLayerSpec:
        return self.specs[layer_idx]

    @property
    def self_attn_layers(self) -> List[int]:
        return [s.layer_idx for s in self.specs if s.is_self_attn]

    @property
    def cross_attn_layers(self) -> List[int]:
        return [s.layer_idx for s in self.specs if s.is_cross_attn]

    def kv_cache_reduction_factor(self) -> float:
        """KV slots used / KV slots in a standard model."""
        # Standard: n_layers sets of KV.
        # YOCO: only n_self_attn_layers sets (shared by all cross-attn layers).
        return self.config.n_self_attn_layers / self.config.n_layers

    def summary(self) -> str:
        return (
            f"YOCOSchedule: {self.config.n_self_attn_layers} self-attn "
            f"+ {self.config.n_cross_attn_layers} cross-attn layers "
            f"({self.kv_cache_reduction_factor():.1%} of standard KV memory)"
        )


# ---------------------------------------------------------------------------
# Shared KV Store — the single cache written by self-attn, read by cross-attn
# ---------------------------------------------------------------------------

class YOCOKVStore:
    """The single shared KV cache for a YOCO model.

    Self-attention layers append to this store.
    Cross-attention layers read the full store (all tokens written so far).
    """

    def __init__(self, config: YOCOConfig) -> None:
        self._config = config
        self._keys: List[np.ndarray] = []    # each entry: (n_kv_heads, head_dim)
        self._values: List[np.ndarray] = []
        self._stats = YOCOStats()

    # ------------------------------------------------------------------
    def append(self, keys: np.ndarray, values: np.ndarray) -> None:
        """Append token KV (written by a self-attention layer).

        Args:
            keys:   Shape (n_kv_heads, head_dim) or (head_dim,) — one token.
            values: Same shape as keys.
        """
        self._keys.append(np.asarray(keys, dtype=np.float32))
        self._values.append(np.asarray(values, dtype=np.float32))
        object.__setattr__(
            self._stats, "tokens_written", self._stats.tokens_written + 1
        )

    def get_shared_kv(
        self,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return *(keys, values)* for all tokens written so far.

        Returns:
            keys:   (T, n_kv_heads, head_dim) or (T, head_dim) depending on
                    what was appended.
            values: Same shape.
        """
        if not self._keys:
            cfg = self._config
            empty_k = np.zeros(
                (0, cfg.n_kv_heads, cfg.head_dim), dtype=np.float32
            )
            return empty_k, empty_k.copy()
        object.__setattr__(
            self._stats, "reads", self._stats.reads + 1
        )
        return np.stack(self._keys), np.stack(self._values)

    def reset(self) -> None:
        self._keys.clear()
        self._values.clear()

    @property
    def size(self) -> int:
        return len(self._keys)

    @property
    def is_empty(self) -> bool:
        return len(self._keys) == 0

    @property
    def stats(self) -> "YOCOStats":
        return self._stats


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

@dataclass
class YOCOStats:
    """Runtime + memory statistics for YOCO."""

    tokens_written: int = 0
    reads: int = 0

    def kv_memory_bytes(
        self, n_kv_heads: int, head_dim: int, dtype_bytes: int = 2
    ) -> int:
        """Current KV memory usage in bytes (keys + values)."""
        return 2 * self.tokens_written * n_kv_heads * head_dim * dtype_bytes

    def standard_kv_memory_bytes(
        self, n_layers: int, n_kv_heads: int, head_dim: int, dtype_bytes: int = 2
    ) -> int:
        """Equivalent memory for a standard (non-YOCO) model."""
        return (
            2 * n_layers * self.tokens_written * n_kv_heads * head_dim * dtype_bytes
        )

    def kv_memory_reduction_ratio(
        self, n_layers: int, n_kv_heads: int, head_dim: int, dtype_bytes: int = 2
    ) -> float:
        """Fraction of KV memory saved vs. standard model."""
        yoco = self.kv_memory_bytes(n_kv_heads, head_dim, dtype_bytes)
        standard = self.standard_kv_memory_bytes(
            n_layers, n_kv_heads, head_dim, dtype_bytes
        )
        if standard == 0:
            return 0.0
        return 1.0 - yoco / standard
