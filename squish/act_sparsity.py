"""ActSparsity — ReLU/SwiGLU activation sparsity predictor + FFN skip gate.

Activation sparsity in transformer FFN layers is the phenomenon where a large
fraction of intermediate activations become near-zero after ReLU or SwiGLU.
DejaVu (Liu et al., NeurIPS 2023) and PowerInfer (Song et al., SOSP 2024)
exploit this to skip computation for near-zero neurons, achieving 2–4× FFN
speedup with minimal accuracy loss.

Reference:
    Liu et al., "DejaVu: Contextual Sparsity for Efficient LLMs at Inference
    Time", NeurIPS 2023. https://arxiv.org/abs/2310.17157

    Song et al., "PowerInfer: Fast Large Language Model Serving with a
    Consumer-grade GPU", SOSP 2024. https://arxiv.org/abs/2312.12456

Usage example::

    import numpy as np
    from squish.act_sparsity import SparsityConfig, ActSparsityPredictor, SparseFFNGate

    config = SparsityConfig(hidden_dim=4096, n_layers=32, threshold=0.01)
    predictor = ActSparsityPredictor(config)

    # Calibration phase
    for layer_idx in range(32):
        acts = np.random.randn(512, 4096).astype(np.float32)
        predictor.record(layer_idx, acts)

    sparsity_map = predictor.calibrate()

    # Inference phase
    gate = SparseFFNGate(config, layer_idx=0)
    activations = np.random.randn(16, 4096).astype(np.float32)
    masked = gate.apply(activations)
    ratio = gate.compression_ratio()
    print(f"Compression ratio (fraction kept): {ratio:.3f}")
"""

from __future__ import annotations

__all__ = [
    "SparsityConfig",
    "ActSparsityPredictor",
    "SparseFFNGate",
    "ActSparsityStats",
]

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np


@dataclass(frozen=True)
class SparsityConfig:
    """Configuration for activation-sparsity calibration and FFN gating.

    Attributes:
        hidden_dim: Width of the FFN hidden layer.
        n_layers: Total number of transformer layers.
        threshold: Absolute-value cutoff below which a neuron is considered
            near-zero (inactive).
        min_sparsity: Floor on the reported sparsity fraction (prevents
            over-optimistic gating on layers with no recorded activations).
        calibration_steps: Expected number of calibration forward passes;
            informational only, does not cap :meth:`ActSparsityPredictor.record`.
    """

    hidden_dim: int = 4096
    n_layers: int = 32
    threshold: float = 0.01
    min_sparsity: float = 0.0
    calibration_steps: int = 100

    def __post_init__(self) -> None:
        if self.hidden_dim <= 0:
            raise ValueError(
                f"hidden_dim must be a positive integer, got {self.hidden_dim}"
            )
        if self.n_layers <= 0:
            raise ValueError(
                f"n_layers must be a positive integer, got {self.n_layers}"
            )
        if self.threshold < 0.0:
            raise ValueError(
                f"threshold must be non-negative, got {self.threshold}"
            )
        if not (0.0 <= self.min_sparsity <= 1.0):
            raise ValueError(
                f"min_sparsity must be in [0.0, 1.0], got {self.min_sparsity}"
            )
        if self.calibration_steps <= 0:
            raise ValueError(
                f"calibration_steps must be a positive integer, "
                f"got {self.calibration_steps}"
            )


class ActSparsityPredictor:
    """Calibrates per-layer activation sparsity from observed activation tensors.

    Maintains running counters for total elements and near-zero elements per
    layer.  Call :meth:`record` during a calibration forward pass, then
    :meth:`calibrate` to retrieve the layer → sparsity mapping.
    """

    def __init__(self, config: SparsityConfig) -> None:
        self._config = config
        self._total: Dict[int, int] = {}
        self._zeros: Dict[int, int] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record(self, layer_idx: int, activations: np.ndarray) -> None:
        """Accumulate sparsity statistics for one activation tensor.

        Args:
            layer_idx: Zero-based layer index in ``[0, n_layers)``.
            activations: Float32 array of shape ``(seq_len, hidden_dim)``.

        Raises:
            ValueError: If *layer_idx* is out of range or *activations* is
                not 2-D.
        """
        if not (0 <= layer_idx < self._config.n_layers):
            raise ValueError(
                f"layer_idx must be in [0, {self._config.n_layers}), "
                f"got {layer_idx}"
            )
        if activations.ndim != 2:
            raise ValueError(
                f"activations must be 2-D (seq_len, hidden_dim), "
                f"got shape {activations.shape}"
            )
        n_total = int(activations.size)
        n_zeros = int(
            np.count_nonzero(np.abs(activations) < self._config.threshold)
        )
        self._total[layer_idx] = self._total.get(layer_idx, 0) + n_total
        self._zeros[layer_idx] = self._zeros.get(layer_idx, 0) + n_zeros

    def get_sparsity(self, layer_idx: int) -> float:
        """Return fraction of near-zero activations seen for *layer_idx*.

        Returns ``min_sparsity`` if no activations have been recorded for
        that layer.
        """
        total = self._total.get(layer_idx, 0)
        if total == 0:
            return self._config.min_sparsity
        raw = self._zeros.get(layer_idx, 0) / total
        return max(raw, self._config.min_sparsity)

    def should_skip(self, layer_idx: int) -> bool:
        """Return ``True`` if recorded sparsity for *layer_idx* exceeds 50%."""
        return self.get_sparsity(layer_idx) > 0.5

    def calibrate(self) -> Dict[int, float]:
        """Return ``{layer_idx: sparsity_fraction}`` for all recorded layers."""
        return {idx: self.get_sparsity(idx) for idx in sorted(self._total)}

    def reset(self) -> None:
        """Clear all accumulated statistics."""
        self._total.clear()
        self._zeros.clear()


class SparseFFNGate:
    """Applies a threshold-based sparsity mask to FFN activations.

    Values whose absolute value falls below ``config.threshold`` are zeroed
    out, effectively reducing the active neuron count and enabling downstream
    sparse-GEMM optimizations.
    """

    def __init__(self, config: SparsityConfig, layer_idx: int) -> None:
        if not (0 <= layer_idx < config.n_layers):
            raise ValueError(
                f"layer_idx must be in [0, {config.n_layers}), got {layer_idx}"
            )
        self._config = config
        self._layer_idx = layer_idx
        self._last_mask: Optional[np.ndarray] = None

    def apply(self, activations: np.ndarray) -> np.ndarray:
        """Return a masked copy of *activations* with sub-threshold values zeroed.

        Args:
            activations: Float32 array of any shape; the threshold is applied
                element-wise.

        Returns:
            Array of the same shape and dtype as *activations* with near-zero
            values replaced by 0.
        """
        mask = np.abs(activations) >= self._config.threshold
        self._last_mask = mask
        return activations * mask

    def compression_ratio(self) -> float:
        """Fraction of values preserved in the most recent :meth:`apply` call.

        Returns ``1.0`` if :meth:`apply` has not yet been called.
        """
        if self._last_mask is None:
            return 1.0
        return float(np.mean(self._last_mask))


@dataclass
class ActSparsityStats:
    """Aggregate statistics for an activation-sparsity session.

    Attributes:
        total_activations_seen: Cumulative count of individual float values
            processed by :meth:`ActSparsityPredictor.record`.
        total_zeros: Cumulative count of near-zero values among the above.
        total_skipped_layers: Count of whole-layer skip events triggered by
            :meth:`ActSparsityPredictor.should_skip`.
    """

    total_activations_seen: int = 0
    total_zeros: int = 0
    total_skipped_layers: int = 0

    @property
    def sparsity_rate(self) -> float:
        """Fraction of activation values that were near-zero."""
        if self.total_activations_seen == 0:
            return 0.0
        return self.total_zeros / self.total_activations_seen

    @property
    def skip_rate(self) -> float:
        """Fraction of total observed events that were whole-layer skips.

        The denominator is ``total_skipped_layers + total_activations_seen``.
        This is most meaningful when ``total_activations_seen`` counts layer
        evaluation events (not individual float values); callers should
        document their accumulation convention accordingly.
        """
        total = self.total_skipped_layers + self.total_activations_seen
        if total == 0:
            return 0.0
        return self.total_skipped_layers / total
