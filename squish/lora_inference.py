"""LoRAInference — Zero-copy LoRA delta inference (no weight merge required).

Low-Rank Adaptation (LoRA) decomposes weight updates into two small matrices
A (in × r) and B (r × out) with r << min(in, out).  At inference time the
delta ``x @ A @ B * scaling`` can be added to the base-model output without
merging weights into the base checkpoint, enabling instant adapter switching
and multi-adapter serving with a single resident base model.

Reference:
    Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models",
    ICLR 2022. https://arxiv.org/abs/2106.09685

Usage example::

    import numpy as np
    from squish.lora_inference import LoRAConfig, LoRALayer, LoRAInferenceAdapter

    config = LoRAConfig(rank=16, alpha=32.0)
    adapter = LoRAInferenceAdapter(config)

    rng = np.random.default_rng(0)
    in_f, out_f, r = 4096, 4096, 16
    A = rng.standard_normal((in_f, r)).astype(np.float32) * 0.02
    B = np.zeros((r, out_f), dtype=np.float32)
    adapter.add_layer("q_proj", in_f, out_f, A, B)

    x = rng.standard_normal((8, in_f)).astype(np.float32)
    base_out = rng.standard_normal((8, out_f)).astype(np.float32)
    result = adapter.apply("q_proj", x, base_out)
    print(result.shape)  # (8, 4096)
"""

from __future__ import annotations

__all__ = [
    "LoRAConfig",
    "LoRALayer",
    "LoRAInferenceAdapter",
    "LoRAStats",
]

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class LoRAConfig:
    """Configuration for LoRA inference adapters.

    Attributes:
        rank: Inner dimension of the decomposition (r).
        alpha: Scaling numerator; effective scale = alpha / rank.
        dropout: Dropout probability applied during *training* — stored for
            completeness but not applied during inference.
        target_modules: Names of modules that have LoRA adapters.  Defaults
            to ``("q_proj", "v_proj")``.
    """

    rank: int = 16
    alpha: float = 32.0
    dropout: float = 0.0
    target_modules: Optional[Tuple[str, ...]] = None

    def __post_init__(self) -> None:
        if self.rank <= 0:
            raise ValueError(f"rank must be a positive integer, got {self.rank}")
        if self.alpha <= 0.0:
            raise ValueError(f"alpha must be positive, got {self.alpha}")
        if not (0.0 <= self.dropout < 1.0):
            raise ValueError(
                f"dropout must be in [0.0, 1.0), got {self.dropout}"
            )
        if self.target_modules is None:
            object.__setattr__(self, "target_modules", ("q_proj", "v_proj"))


@dataclass(eq=False)
class LoRALayer:
    """A single LoRA adapter for one linear projection.

    Attributes:
        module_name: Name of the base module this adapter targets.
        in_features: Input feature dimension.
        out_features: Output feature dimension.
        A: Low-rank matrix of shape ``(in_features, rank)``.
        B: Low-rank matrix of shape ``(rank, out_features)``.
        scaling: Effective scale factor ``alpha / rank``.
    """

    module_name: str
    in_features: int
    out_features: int
    A: np.ndarray  # (in_features, rank)
    B: np.ndarray  # (rank, out_features)
    scaling: float

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Compute the LoRA delta for input *x*.

        Args:
            x: Input array of shape ``(batch, in_features)``.

        Returns:
            Delta array of shape ``(batch, out_features)`` equal to
            ``x @ A @ B * scaling``.
        """
        return (x @ self.A @ self.B) * self.scaling

    @property
    def rank(self) -> int:
        """Rank of this adapter, inferred from the A matrix."""
        return int(self.A.shape[1])

    @property
    def n_params(self) -> int:
        """Total number of trainable parameters in this adapter."""
        return int(self.A.size + self.B.size)


class LoRAInferenceAdapter:
    """Manages a collection of :class:`LoRALayer` instances and applies them
    as additive deltas during inference.

    Thread-safety: not thread-safe; wrap with a lock for concurrent use.
    """

    def __init__(self, config: LoRAConfig) -> None:
        self._config = config
        self._layers: Dict[str, LoRALayer] = {}
        self._stats = LoRAStats()

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def add_layer(
        self,
        module_name: str,
        in_features: int,
        out_features: int,
        A: np.ndarray,
        B: np.ndarray,
    ) -> None:
        """Register a LoRA adapter for *module_name*.

        Args:
            module_name: Identifier matching the base model module name.
            in_features: Input feature dimension.
            out_features: Output feature dimension.
            A: Matrix of shape ``(in_features, rank)``.
            B: Matrix of shape ``(rank, out_features)``.

        Raises:
            ValueError: If matrix shapes are inconsistent.
        """
        if A.ndim != 2 or A.shape[0] != in_features:
            raise ValueError(
                f"A must have shape (in_features={in_features}, rank), "
                f"got {A.shape}"
            )
        if B.ndim != 2 or B.shape[1] != out_features:
            raise ValueError(
                f"B must have shape (rank, out_features={out_features}), "
                f"got {B.shape}"
            )
        if A.shape[1] != B.shape[0]:
            raise ValueError(
                f"A rank ({A.shape[1]}) must equal B rank ({B.shape[0]})"
            )
        scaling = self._config.alpha / A.shape[1]
        self._layers[module_name] = LoRALayer(
            module_name=module_name,
            in_features=in_features,
            out_features=out_features,
            A=A.astype(np.float32, copy=False),
            B=B.astype(np.float32, copy=False),
            scaling=scaling,
        )

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def apply(
        self,
        module_name: str,
        x: np.ndarray,
        base_output: np.ndarray,
    ) -> np.ndarray:
        """Add the LoRA delta to *base_output*.

        If *module_name* is not registered, returns *base_output* unchanged.

        Args:
            module_name: Name of the module being computed.
            x: Input to the linear layer, shape ``(batch, in_features)``.
            base_output: Output of the frozen base layer, same batch size.

        Returns:
            ``base_output + delta`` if adapter registered, else ``base_output``.
        """
        if module_name not in self._layers:
            return base_output
        layer = self._layers[module_name]
        delta = layer.forward(x)
        delta_norm = float(np.linalg.norm(delta))
        self._stats.n_forward_calls += 1
        self._stats.total_delta_norm_sum += delta_norm
        return base_output + delta

    # ------------------------------------------------------------------
    # Merge
    # ------------------------------------------------------------------

    def merge_into(self, weights_dict: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Return a new weight dict with each LoRA delta permanently merged.

        For each registered adapter the weight delta ``A @ B * scaling`` (shape
        ``(in_features, out_features)``) is added to ``weights_dict[module_name]``
        when present, or stored as a standalone entry otherwise.  The original
        *weights_dict* is never modified.

        Args:
            weights_dict: Mapping from module name to base weight matrix.

        Returns:
            New dict with merged weights; unregistered keys are preserved
            unchanged.
        """
        result: Dict[str, np.ndarray] = dict(weights_dict)
        for name, layer in self._layers.items():
            # Weight delta: equivalent to applying LoRA to an identity input
            delta = layer.A @ layer.B * layer.scaling  # (in_features, out_features)
            if name in result:
                result[name] = result[name] + delta
            else:
                result[name] = delta
        return result

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def adapter_names(self) -> List[str]:
        """Names of all registered LoRA modules."""
        return list(self._layers.keys())

    @property
    def total_params(self) -> int:
        """Total number of LoRA parameters across all registered adapters."""
        return sum(layer.n_params for layer in self._layers.values())

    @property
    def stats(self) -> "LoRAStats":
        """Running inference statistics."""
        return self._stats


@dataclass
class LoRAStats:
    """Running statistics for a :class:`LoRAInferenceAdapter` session.

    Attributes:
        n_forward_calls: Number of times :meth:`LoRAInferenceAdapter.apply`
            triggered an actual delta computation.
        total_delta_norm_sum: Cumulative L2 norm of all delta tensors.
    """

    n_forward_calls: int = 0
    total_delta_norm_sum: float = 0.0

    @property
    def avg_delta_norm(self) -> float:
        """Mean L2 norm of LoRA deltas across all forward calls."""
        if self.n_forward_calls == 0:
            return 0.0
        return self.total_delta_norm_sum / self.n_forward_calls
