"""DiffKV — Differentiated Memory Management with Parallel KV Compaction.

Implements the three-axis KV cache differentiation from DiffKV
(SOSP 2025, arXiv:2412.03131):

  Axis 1 — K vs. V asymmetry:
    Keys determine attention routing; values carry content.
    Assign different precision to K and V independently per token.

  Axis 2 — Token importance:
    Critical tokens → K=INT8, V=INT4.
    Marginal tokens → K=INT4, V=INT2.
    Unimportant tokens → evict.

  Axis 3 — Per-head dynamic sparsity:
    Each head gets an independent compression policy derived from its
    observed sparsity pattern.  High-sparsity heads → more aggressive.

The on-GPU parallel compaction that makes irregular memory layout
efficient is modelled here as a bulk re-packing operation.

Reported: 1.9–5.4× throughput, 2.7–5.7× KV memory reduction with
near-FP16 accuracy on thinking models (Qwen2.5-32B, QwQ-32B).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

# Supported precision levels (bits)
PRECISION_TIERS: tuple[int, ...] = (2, 4, 8)

# Token importance thresholds
IMPORTANCE_CRITICAL = 2
IMPORTANCE_MARGINAL = 1
IMPORTANCE_EVICT = 0


@dataclass
class DiffKVConfig:
    """Configuration for DiffKV differentiated KV management.

    Args:
        n_layers: Total transformer layers.
        n_heads: Attention heads per layer.
        critical_k_bits: K precision for critical tokens.
        critical_v_bits: V precision for critical tokens.
        marginal_k_bits: K precision for marginal tokens.
        marginal_v_bits: V precision for marginal tokens.
        critical_fraction: Top-f fraction of tokens classified as critical.
        marginal_fraction: Next-g fraction of tokens classified as marginal.
        head_sparsity_boost: Extra compression ratio applied to high-sparsity heads.
        sparsity_threshold: Head sparsity ratio above which boost applies.
        compact_block_size: Number of tokens per GPU compaction block.
    """

    n_layers: int = 32
    n_heads: int = 32
    critical_k_bits: int = 8
    critical_v_bits: int = 4
    marginal_k_bits: int = 4
    marginal_v_bits: int = 2
    critical_fraction: float = 0.20
    marginal_fraction: float = 0.40
    head_sparsity_boost: float = 1.5
    sparsity_threshold: float = 0.70
    compact_block_size: int = 16

    def __post_init__(self) -> None:
        if self.n_layers <= 0:
            raise ValueError("n_layers must be positive")
        if self.n_heads <= 0:
            raise ValueError("n_heads must be positive")
        if self.critical_k_bits not in PRECISION_TIERS:
            raise ValueError(f"critical_k_bits must be one of {PRECISION_TIERS}")
        if self.critical_v_bits not in PRECISION_TIERS:
            raise ValueError(f"critical_v_bits must be one of {PRECISION_TIERS}")
        if self.marginal_k_bits not in PRECISION_TIERS:
            raise ValueError(f"marginal_k_bits must be one of {PRECISION_TIERS}")
        if self.marginal_v_bits not in PRECISION_TIERS:
            raise ValueError(f"marginal_v_bits must be one of {PRECISION_TIERS}")
        if not 0 < self.critical_fraction < 1:
            raise ValueError("critical_fraction must be in (0, 1)")
        if not 0 < self.marginal_fraction < 1:
            raise ValueError("marginal_fraction must be in (0, 1)")
        if self.critical_fraction + self.marginal_fraction >= 1.0:
            raise ValueError("critical_fraction + marginal_fraction must be < 1")
        if not 0 < self.sparsity_threshold <= 1.0:
            raise ValueError("sparsity_threshold must be in (0, 1]")

    @property
    def evict_fraction(self) -> float:
        return 1.0 - self.critical_fraction - self.marginal_fraction


@dataclass
class HeadSparsityProfile:
    """Observed sparsity pattern for one attention head.

    Attributes:
        layer_idx: Layer index.
        head_idx: Head index.
        observed_sparsity: Fraction of near-zero attention values observed.
        n_samples: Number of forward passes sampled.
    """

    layer_idx: int
    head_idx: int
    observed_sparsity: float = 0.0
    n_samples: int = 0

    def update(self, attn_weights: np.ndarray, near_zero_threshold: float = 0.01) -> None:
        """Update sparsity estimate from attention weight matrix.

        Args:
            attn_weights: (seq_q, seq_k) attention probabilities.
            near_zero_threshold: Values below this are considered zero.
        """
        sparsity = float((attn_weights < near_zero_threshold).mean())
        # Running average
        self.observed_sparsity = (
            self.observed_sparsity * self.n_samples + sparsity
        ) / (self.n_samples + 1)
        self.n_samples += 1

    @property
    def is_high_sparsity(self) -> bool:
        return self.observed_sparsity > 0.0  # used with config threshold externally


@dataclass
class TokenImportanceTier:
    """Classification of tokens into critical/marginal/evict tiers."""

    token_indices: np.ndarray  # global token positions
    tier: int  # IMPORTANCE_CRITICAL, IMPORTANCE_MARGINAL, or IMPORTANCE_EVICT

    @property
    def n_tokens(self) -> int:
        return len(self.token_indices)


def classify_tokens(
    attn_scores: np.ndarray,
    config: DiffKVConfig,
) -> list[TokenImportanceTier]:
    """Classify tokens by importance using attention score aggregation.

    Args:
        attn_scores: (seq_len,) or (n_heads, seq_len) aggregated attention weights.
        config: DiffKVConfig.

    Returns:
        List of three TokenImportanceTier objects [critical, marginal, evict].
    """
    if attn_scores.ndim > 1:
        scores = attn_scores.mean(axis=0)
    else:
        scores = attn_scores.astype(np.float32)

    seq_len = scores.size
    n_critical = max(1, int(seq_len * config.critical_fraction))
    n_marginal = max(1, int(seq_len * config.marginal_fraction))
    # Ensure we don't exceed seq_len
    n_critical = min(n_critical, seq_len)
    n_marginal = min(n_marginal, seq_len - n_critical)

    sorted_idx = np.argsort(-scores)  # descending
    critical_idx = np.sort(sorted_idx[:n_critical])
    marginal_idx = np.sort(sorted_idx[n_critical : n_critical + n_marginal])
    evict_idx = np.sort(sorted_idx[n_critical + n_marginal :])

    return [
        TokenImportanceTier(token_indices=critical_idx, tier=IMPORTANCE_CRITICAL),
        TokenImportanceTier(token_indices=marginal_idx, tier=IMPORTANCE_MARGINAL),
        TokenImportanceTier(token_indices=evict_idx, tier=IMPORTANCE_EVICT),
    ]


def _bits_to_bytes_per_element(bits: int) -> float:
    return bits / 8.0


@dataclass
class DiffKVPolicy:
    """Compression policy for one (layer, head) pair.

    Attributes:
        layer_idx: Layer index.
        head_idx: Head index.
        critical_k_bits: K bits for critical tokens.
        critical_v_bits: V bits for critical tokens.
        marginal_k_bits: K bits for marginal tokens.
        marginal_v_bits: V bits for marginal tokens.
        sparsity_boost_active: Whether the per-head boost is applied.
    """

    layer_idx: int
    head_idx: int
    critical_k_bits: int
    critical_v_bits: int
    marginal_k_bits: int
    marginal_v_bits: int
    sparsity_boost_active: bool = False

    def effective_k_bits(self, tier: int) -> int:
        if tier == IMPORTANCE_CRITICAL:
            return self.critical_k_bits
        if tier == IMPORTANCE_MARGINAL:
            bits = self.marginal_k_bits
            return max(2, bits - 2) if self.sparsity_boost_active else bits
        return 2  # evicted tokens stored at minimum

    def effective_v_bits(self, tier: int) -> int:
        if tier == IMPORTANCE_CRITICAL:
            return self.critical_v_bits
        if tier == IMPORTANCE_MARGINAL:
            bits = self.marginal_v_bits
            return max(2, bits - 2) if self.sparsity_boost_active else bits
        return 2


class DiffKVPolicyManager:
    """Manages per-head DiffKV policies across all layers."""

    def __init__(self, config: DiffKVConfig) -> None:
        self.config = config
        self._profiles: dict[tuple[int, int], HeadSparsityProfile] = {}
        for layer_idx in range(config.n_layers):
            for head_idx in range(config.n_heads):
                self._profiles[(layer_idx, head_idx)] = HeadSparsityProfile(
                    layer_idx=layer_idx, head_idx=head_idx
                )

    def record_attention(
        self, layer_idx: int, head_idx: int, attn_weights: np.ndarray
    ) -> None:
        """Record observed attention pattern for a head."""
        key = (layer_idx, head_idx)
        if key in self._profiles:
            self._profiles[key].update(attn_weights)

    def get_policy(self, layer_idx: int, head_idx: int) -> DiffKVPolicy:
        """Derive a DiffKVPolicy for a head based on observed sparsity."""
        profile = self._profiles.get(
            (layer_idx, head_idx),
            HeadSparsityProfile(layer_idx=layer_idx, head_idx=head_idx),
        )
        boost = profile.observed_sparsity >= self.config.sparsity_threshold
        return DiffKVPolicy(
            layer_idx=layer_idx,
            head_idx=head_idx,
            critical_k_bits=self.config.critical_k_bits,
            critical_v_bits=self.config.critical_v_bits,
            marginal_k_bits=self.config.marginal_k_bits,
            marginal_v_bits=self.config.marginal_v_bits,
            sparsity_boost_active=boost,
        )

    def all_policies(self) -> list[DiffKVPolicy]:
        return [
            self.get_policy(l, h)
            for l in range(self.config.n_layers)
            for h in range(self.config.n_heads)
        ]


@dataclass
class CompactedKVSlot:
    """Memory-compact representation of KV cache for one (layer, head, request).

    Models the on-GPU parallel compaction result.
    """

    layer_idx: int
    head_idx: int
    n_critical: int
    n_marginal: int
    n_evicted: int
    head_dim: int
    policy: DiffKVPolicy

    @property
    def n_retained(self) -> int:
        return self.n_critical + self.n_marginal

    @property
    def bytes_used(self) -> float:
        """Estimated memory bytes for this slot."""
        critical_bytes = self.n_critical * self.head_dim * (
            _bits_to_bytes_per_element(self.policy.critical_k_bits)
            + _bits_to_bytes_per_element(self.policy.critical_v_bits)
        )
        marginal_bytes = self.n_marginal * self.head_dim * (
            _bits_to_bytes_per_element(self.policy.marginal_k_bits)
            + _bits_to_bytes_per_element(self.policy.marginal_v_bits)
        )
        return critical_bytes + marginal_bytes

    @property
    def bytes_fp16_equivalent(self) -> float:
        """Bytes if stored in full FP16 (2 bytes/element for K and V)."""
        return (self.n_critical + self.n_marginal) * self.head_dim * 4.0  # 2B K + 2B V

    @property
    def compression_ratio(self) -> float:
        if self.bytes_used <= 0:
            return 1.0
        return self.bytes_fp16_equivalent / self.bytes_used


def compact_kv(
    attn_scores: np.ndarray,
    layer_idx: int,
    head_idx: int,
    head_dim: int,
    policy: DiffKVPolicy,
    config: DiffKVConfig,
) -> CompactedKVSlot:
    """Simulate on-GPU parallel KV compaction for one (layer, head) slot.

    Args:
        attn_scores: (seq_len,) importance scores for tokens.
        layer_idx: Layer index.
        head_idx: Head index.
        head_dim: Head dimension.
        policy: DiffKVPolicy for this head.
        config: DiffKVConfig.

    Returns:
        CompactedKVSlot describing the compacted memory layout.
    """
    tiers = classify_tokens(attn_scores, config)
    n_critical = tiers[0].n_tokens
    n_marginal = tiers[1].n_tokens
    n_evicted = tiers[2].n_tokens
    return CompactedKVSlot(
        layer_idx=layer_idx,
        head_idx=head_idx,
        n_critical=n_critical,
        n_marginal=n_marginal,
        n_evicted=n_evicted,
        head_dim=head_dim,
        policy=policy,
    )


@dataclass
class DiffKVStats:
    """Aggregate DiffKV statistics across all heads/layers."""

    total_slots: int = 0
    total_critical_tokens: int = 0
    total_marginal_tokens: int = 0
    total_evicted_tokens: int = 0
    total_bytes_diffkv: float = 0.0
    total_bytes_fp16: float = 0.0
    high_sparsity_heads: int = 0

    def record_slot(self, slot: CompactedKVSlot) -> None:
        self.total_slots += 1
        self.total_critical_tokens += slot.n_critical
        self.total_marginal_tokens += slot.n_marginal
        self.total_evicted_tokens += slot.n_evicted
        self.total_bytes_diffkv += slot.bytes_used
        self.total_bytes_fp16 += slot.bytes_fp16_equivalent
        if slot.policy.sparsity_boost_active:
            self.high_sparsity_heads += 1

    @property
    def overall_compression_ratio(self) -> float:
        if self.total_bytes_diffkv <= 0:
            return 1.0
        return self.total_bytes_fp16 / self.total_bytes_diffkv

    @property
    def eviction_rate(self) -> float:
        total = (
            self.total_critical_tokens
            + self.total_marginal_tokens
            + self.total_evicted_tokens
        )
        if total == 0:
            return 0.0
        return self.total_evicted_tokens / total

    @property
    def estimated_throughput_multiplier(self) -> float:
        """Rough throughput estimate from paper's 1.9–5.4× range scaled to compression."""
        cr = self.overall_compression_ratio
        # Paper reports ~1.9× at ~2.7× compression, ~5.4× at ~5.7× compression
        # Linear interpolation in log space
        low_cr, high_cr = 2.7, 5.7
        low_tp, high_tp = 1.9, 5.4
        if cr <= low_cr:
            return low_tp
        if cr >= high_cr:
            return high_tp
        t = (cr - low_cr) / (high_cr - low_cr)
        return low_tp + t * (high_tp - low_tp)
