"""squish/morph_attn.py

AttentionMorpher — Per-layer attention pattern selector.

Different sequence lengths place different demands on the attention mechanism.
For short sequences (≤ ``seq_len_full_threshold``) the quadratic cost of full
attention is negligible, and it should be used everywhere to maximise quality.
As sequences grow it becomes advantageous to switch higher (semantically
richer) layers to cheaper attention variants while keeping lower layers at
full quality, since lower layers aggregate syntactic and positional
information that is sensitive to masking.

Pattern selection rules
-----------------------
- ``seq ≤ full_threshold``  — **full** attention for every layer.
- ``full_threshold < seq ≤ sparse_threshold`` — **sparse** for layers in the
  upper half (``layer_id ≥ n_layers // 2``); **full** for the lower half.
- ``seq > sparse_threshold`` — **linear** for upper-half layers;
  **sparse** for mid layers (``0 < layer_id < n_layers // 2``);
  **full** only for layer 0.

FLOPs estimation
----------------
The :meth:`AttentionMorpher.estimate_flops_reduction` method models:

- full attention    ≈ 1.0 unit per layer (O(seq²))
- sparse attention  ≈ 0.5 units per layer (≈ 50 % of full)
- linear attention  ≈ ``1 / max(seq, 1)`` units per layer (O(seq) ÷ O(seq²))

and returns the fraction of FLOPs saved relative to an all-full baseline.

Example usage::

    from squish.morph_attn import MorphConfig, AttentionMorpher

    cfg    = MorphConfig(n_layers=32, seq_len_full_threshold=512,
                         seq_len_sparse_threshold=4096)
    morpher = AttentionMorpher(cfg)

    patterns = morpher.layer_patterns(seq_len=2048)
    print(patterns)
    # ['full', 'full', ..., 'sparse', 'sparse']

    print(f"FLOPs reduction: {morpher.estimate_flops_reduction(2048):.2%}")
"""

from __future__ import annotations

__all__ = ["MorphConfig", "AttentionMorpher"]

import dataclasses
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class MorphConfig:
    """Configuration for :class:`AttentionMorpher`.

    Attributes:
        n_layers:                  Total number of transformer layers.  Must
                                   be >= 1.
        seq_len_full_threshold:    Sequence lengths at or below this value use
                                   full attention for all layers.  Must be
                                   >= 1.
        seq_len_sparse_threshold:  Sequence lengths at or below this value
                                   (but above ``seq_len_full_threshold``) use
                                   sparse attention for upper-half layers.
                                   Must be >= ``seq_len_full_threshold``.
    """

    n_layers:                 int = 12
    seq_len_full_threshold:   int = 512
    seq_len_sparse_threshold: int = 4096

    def __post_init__(self) -> None:
        if self.n_layers < 1:
            raise ValueError(
                f"n_layers must be >= 1, got {self.n_layers}"
            )
        if self.seq_len_full_threshold < 1:
            raise ValueError(
                f"seq_len_full_threshold must be >= 1, "
                f"got {self.seq_len_full_threshold}"
            )
        if self.seq_len_sparse_threshold < self.seq_len_full_threshold:
            raise ValueError(
                f"seq_len_sparse_threshold ({self.seq_len_sparse_threshold}) "
                f"must be >= seq_len_full_threshold "
                f"({self.seq_len_full_threshold})"
            )


# ---------------------------------------------------------------------------
# Morpher
# ---------------------------------------------------------------------------

# Relative cost of each attention pattern vs full attention (which = 1.0).
_FULL_COST:   float = 1.0
_SPARSE_COST: float = 0.5   # ≈ 50 % of full — sparse / strided attention


class AttentionMorpher:
    """Per-layer attention pattern selector.

    Chooses between ``"full"``, ``"sparse"``, and ``"linear"`` attention for
    each layer based on the current sequence length and layer index according
    to the heuristics described in the module docstring.

    Args:
        config: :class:`MorphConfig` controlling thresholds and layer count.
    """

    def __init__(self, config: MorphConfig) -> None:
        self._cfg = config

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def select_pattern(self, layer_id: int, seq_len: int) -> str:
        """Return the attention pattern for a single ``(layer_id, seq_len)``
        combination.

        Args:
            layer_id: Zero-based layer index in ``[0, n_layers)``.
            seq_len:  Current sequence length.  Must be >= 0.

        Returns:
            One of ``"full"``, ``"sparse"``, or ``"linear"``.

        Raises:
            ValueError: If *layer_id* is outside ``[0, n_layers)`` or
                        *seq_len* is negative.
        """
        if not (0 <= layer_id < self._cfg.n_layers):
            raise ValueError(
                f"layer_id must be in [0, {self._cfg.n_layers}), "
                f"got {layer_id}"
            )
        if seq_len < 0:
            raise ValueError(f"seq_len must be >= 0, got {seq_len}")

        cfg         = self._cfg
        upper_start = cfg.n_layers // 2  # first layer in the "upper half"

        if seq_len <= cfg.seq_len_full_threshold:
            return "full"

        if seq_len <= cfg.seq_len_sparse_threshold:
            # Upper half → sparse; lower half → full.
            return "sparse" if layer_id >= upper_start else "full"

        # seq_len > sparse_threshold
        if layer_id >= upper_start:
            return "linear"
        if layer_id > 0:
            return "sparse"
        return "full"  # layer 0 always uses full attention

    def layer_patterns(self, seq_len: int) -> list[str]:
        """Return the attention pattern for every layer given *seq_len*.

        Args:
            seq_len: Current sequence length.  Must be >= 0.

        Returns:
            A list of length ``n_layers`` where each element is one of
            ``"full"``, ``"sparse"``, or ``"linear"``.

        Raises:
            ValueError: If *seq_len* is negative.
        """
        if seq_len < 0:
            raise ValueError(f"seq_len must be >= 0, got {seq_len}")
        return [
            self.select_pattern(layer_id, seq_len)
            for layer_id in range(self._cfg.n_layers)
        ]

    def estimate_flops_reduction(self, seq_len: int) -> float:
        """Estimate the FLOPs reduction relative to all-full-attention.

        Models each pattern's cost relative to full attention:

        * full:   1.0  (reference)
        * sparse: 0.5  (≈ 50 % of full)
        * linear: ``1.0 / max(seq_len, 1)``  (O(seq) ÷ O(seq²))

        Returns:
            Fraction of FLOPs saved in ``[0.0, 1.0)``.  A value of ``0.0``
            means every layer uses full attention (no savings).

        Raises:
            ValueError: If *seq_len* is negative.
        """
        if seq_len < 0:
            raise ValueError(f"seq_len must be >= 0, got {seq_len}")

        linear_cost    = _FULL_COST / float(max(seq_len, 1))
        pattern_to_cost = {
            "full":   _FULL_COST,
            "sparse": _SPARSE_COST,
            "linear": linear_cost,
        }

        patterns        = self.layer_patterns(seq_len)
        total_potential = float(self._cfg.n_layers) * _FULL_COST
        if total_potential <= 0.0:
            return 0.0

        total_actual = sum(pattern_to_cost[p] for p in patterns)
        saved        = total_potential - total_actual
        return max(0.0, min(1.0, saved / total_potential))
