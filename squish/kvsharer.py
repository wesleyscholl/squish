"""
KVSharer — Cross-layer KV cache sharing with dissimilar-layer pairing.

ICLR 2025 submission. arxiv.org/abs/2410.18517
github.com/yangyifei729/KVSharer

Key insight: sharing *dissimilar* KV caches across layers better preserves
model performance than sharing similar adjacent layers. A one-time calibration
pass produces a static share map applied at runtime with zero overhead.

Result: 30% KV computation reduction, ≥1.3× generation acceleration.
Compatible with all intra-layer compression methods.
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
class KVSharerConfig:
    """Configuration for KVSharer calibration and runtime sharing."""

    n_layers: int = 32
    """Total number of transformer layers."""

    similarity_threshold: float = 0.95
    """Maximum output-representation cosine similarity allowed for a sharing
    pair.  Pairs above this threshold are rejected (too different in function).
    Set < 1.0 to preserve quality; default 0.95 is conservative."""

    max_share_fraction: float = 0.30
    """Maximum fraction of layers allowed to be shared (i.e. redirect their
    KV computation to a donor layer).  Limits quality degradation."""

    prefer_dissimilar: bool = True
    """When True (default), sort candidate pairs in *descending* distance order
    so the most dissimilar pairs are considered first — the counterintuitive
    finding that makes KVSharer work.  Set False for ablation."""

    def __post_init__(self) -> None:
        if not (0 < self.similarity_threshold <= 1.0):
            raise ValueError("similarity_threshold must be in (0, 1]")
        if not (0.0 < self.max_share_fraction < 1.0):
            raise ValueError("max_share_fraction must be in (0, 1)")
        if self.n_layers < 2:
            raise ValueError("n_layers must be >= 2")


# ---------------------------------------------------------------------------
# Calibration — build the share map
# ---------------------------------------------------------------------------

@dataclass
class KVSharerCalibrator:
    """Computes which layers should share KV caches.

    Workflow::

        cal = KVSharerCalibrator(config)
        for batch in calibration_data:
            cal.record_layer_kv(layer_idx, keys, values)      # shape (B, H, T, D)
        share_map = cal.compute_share_map()
        # share_map[donor] = [recipient, ...]
    """

    config: KVSharerConfig

    def __post_init__(self) -> None:
        # layer_idx -> list of KV mean vectors (one per calibration batch)
        self._kv_means: Dict[int, List[np.ndarray]] = {}

    # ------------------------------------------------------------------
    def record_layer_kv(
        self, layer_idx: int, keys: np.ndarray, values: np.ndarray
    ) -> None:
        """Record one calibration example for *layer_idx*.

        Args:
            layer_idx: Which transformer layer produced these tensors.
            keys:   Float array, any shape — will be flattened to a 1-D
                    mean vector per example.
            values: Float array, same shape as keys.
        """
        kv = np.concatenate([keys.ravel(), values.ravel()])
        kv_mean = kv.astype(np.float64)
        self._kv_means.setdefault(layer_idx, []).append(kv_mean)

    # ------------------------------------------------------------------
    def _layer_centroid(self, layer_idx: int) -> np.ndarray:
        """Return the average KV representation for *layer_idx*."""
        vecs = self._kv_means.get(layer_idx)
        if not vecs:
            # Fallback: random unit vector (for testing without calibration data)
            rng = np.random.default_rng(layer_idx)
            v = rng.standard_normal(64)
            return v / (np.linalg.norm(v) + 1e-9)
        stacked = np.vstack([v[:min(len(v), 64)] if len(v) >= 64 else
                             np.pad(v, (0, 64 - len(v))) for v in vecs])
        mean = stacked.mean(axis=0)
        norm = np.linalg.norm(mean)
        return mean / (norm + 1e-9)

    # ------------------------------------------------------------------
    @staticmethod
    def _euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
        size = min(len(a), len(b))
        return float(np.linalg.norm(a[:size] - b[:size]))

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        size = min(len(a), len(b))
        a, b = a[:size], b[:size]
        denom = (np.linalg.norm(a) + 1e-9) * (np.linalg.norm(b) + 1e-9)
        return float(np.dot(a, b) / denom)

    # ------------------------------------------------------------------
    def compute_share_map(self) -> "KVShareMap":
        """Run the calibration algorithm and return a :class:`KVShareMap`.

        Algorithm:
        1. Compute centroid for each observed layer.
        2. Enumerate all (donor, recipient) ordered pairs.
        3. Sort pairs by centroid distance (descending = most dissimilar first
           when prefer_dissimilar=True).
        4. Greedily assign pairs; reject if donor/recipient already assigned
           or output-similarity constraint would be violated.
        5. Stop when max_share_fraction reached.
        """
        cfg = self.config
        n = cfg.n_layers
        max_recipients = max(1, int(n * cfg.max_share_fraction))

        centroids: Dict[int, np.ndarray] = {
            i: self._layer_centroid(i) for i in range(n)
        }

        # All pairs (donor != recipient)
        pairs: List[Tuple[float, int, int]] = []
        for donor in range(n):
            for recipient in range(n):
                if donor == recipient:
                    continue
                dist = self._euclidean_distance(
                    centroids[donor], centroids[recipient]
                )
                pairs.append((dist, donor, recipient))

        # Sort most-dissimilar first or most-similar first
        pairs.sort(key=lambda x: x[0], reverse=cfg.prefer_dissimilar)

        share_map: Dict[int, int] = {}  # recipient -> donor
        donor_recipients: Dict[int, List[int]] = {}
        assigned_recipients: set = set()

        for dist, donor, recipient in pairs:
            if len(assigned_recipients) >= max_recipients:
                break
            if recipient in assigned_recipients:
                continue
            if donor in assigned_recipients:
                continue  # donor must compute its own KV

            # Similarity constraint: output representations must not be too
            # close — if they are, the pairing is likely to harm quality.
            sim = self._cosine_similarity(centroids[donor], centroids[recipient])
            if sim > cfg.similarity_threshold:
                continue

            share_map[recipient] = donor
            assigned_recipients.add(recipient)
            donor_recipients.setdefault(donor, []).append(recipient)

        return KVShareMap(
            share_map=share_map,
            donor_recipients=donor_recipients,
            n_layers=n,
            config=cfg,
        )


# ---------------------------------------------------------------------------
# Share Map — lightweight runtime artifact
# ---------------------------------------------------------------------------

@dataclass
class KVShareMap:
    """Immutable sharing map produced by :class:`KVSharerCalibrator`.

    ``share_map[recipient_layer] = donor_layer`` means the recipient layer
    should reuse the KV cache of the donor layer instead of computing its own.
    """

    share_map: Dict[int, int]
    """Maps recipient layer index → donor layer index."""

    donor_recipients: Dict[int, List[int]]
    """Maps donor layer index → list of layers that borrow from it."""

    n_layers: int
    config: KVSharerConfig

    # ------------------------------------------------------------------
    @property
    def n_shared(self) -> int:
        """Number of layers whose KV computation is eliminated."""
        return len(self.share_map)

    @property
    def share_fraction(self) -> float:
        """Fraction of total layers that are shared."""
        return self.n_shared / max(1, self.n_layers)

    @property
    def donor_layers(self) -> List[int]:
        """Layers that compute (and lend) their own KV cache."""
        return sorted(self.donor_recipients.keys())

    @property
    def recipient_layers(self) -> List[int]:
        """Layers whose KV computation is eliminated."""
        return sorted(self.share_map.keys())

    def donor_for(self, layer_idx: int) -> int:
        """Return the donor for *layer_idx* (itself if not shared)."""
        return self.share_map.get(layer_idx, layer_idx)

    def is_shared(self, layer_idx: int) -> bool:
        """True if *layer_idx* reuses another layer's KV cache."""
        return layer_idx in self.share_map

    def kv_ops_saved_fraction(self) -> float:
        """Estimated fraction of KV operations eliminated."""
        return self.share_fraction

    def summary(self) -> str:
        lines = [
            f"KVShareMap: {self.n_shared}/{self.n_layers} layers shared "
            f"({self.share_fraction:.1%} KV reduction)",
        ]
        for r in self.recipient_layers:
            d = self.share_map[r]
            lines.append(f"  layer {r:2d} → borrows KV from layer {d:2d}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Runtime KV Layer Cache — applies the share map
# ---------------------------------------------------------------------------

class KVLayerCache:
    """Minimal KV cache that redirects shared layers to their donor's slot.

    Usage::

        cache = KVLayerCache(share_map)
        cache.store(layer_idx, keys, values)   # stores at donor slot
        k, v = cache.retrieve(layer_idx)       # reads from donor slot
    """

    def __init__(self, share_map: KVShareMap) -> None:
        self._share_map = share_map
        self._store: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
        self._stats = KVSharerStats()

    # ------------------------------------------------------------------
    def store(self, layer_idx: int, keys: np.ndarray, values: np.ndarray) -> None:
        """Store KV for *layer_idx* (or its donor if shared)."""
        effective = self._share_map.donor_for(layer_idx)
        self._store[effective] = (keys, values)
        if self._share_map.is_shared(layer_idx):
            object.__setattr__(self._stats, "redirect_writes",
                               self._stats.redirect_writes + 1)

    def retrieve(
        self, layer_idx: int
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Retrieve KV for *layer_idx* (redirected to donor if shared)."""
        effective = self._share_map.donor_for(layer_idx)
        result = self._store.get(effective)
        if result is None:
            return None
        if self._share_map.is_shared(layer_idx):
            object.__setattr__(self._stats, "redirect_reads",
                               self._stats.redirect_reads + 1)
        return result

    def reset(self) -> None:
        self._store.clear()

    @property
    def stats(self) -> "KVSharerStats":
        return self._stats

    @property
    def n_cached_layers(self) -> int:
        return len(self._store)


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

@dataclass
class KVSharerStats:
    """Runtime statistics for KV sharing."""

    redirect_writes: int = 0
    """Number of KV write operations redirected (i.e., skipped for recipient)."""

    redirect_reads: int = 0
    """Number of KV read operations that served a shared layer from donor."""

    @property
    def total_redirects(self) -> int:
        return self.redirect_writes + self.redirect_reads

    @property
    def estimated_compute_savings(self) -> float:
        """Fraction of write ops saved (proxy for compute reduction)."""
        total = self.redirect_writes + max(1, self.redirect_reads)
        return self.redirect_writes / total
