"""FusedRMSNorm — Fused RMSNorm + residual add + scale in a single numpy pass.

Fusing the residual add with normalization is a key optimization used in
FlashAttention-2, vLLM, and TensorRT-LLM.  Reading x and residual together
and writing the summed result once halves memory bandwidth compared to
separate add + norm kernels.

Reference:
    Zhang & Sennrich, "Root Mean Square Layer Normalization", NeurIPS 2019.
    https://arxiv.org/abs/1910.07467

    Dao et al., "FlashAttention-2: Faster Attention with Better Parallelism
    and Work Partitioning", ICLR 2024. https://arxiv.org/abs/2307.08691

Usage example::

    import numpy as np
    from squish.fused_rmsnorm import FusedNormConfig, FusedRMSNorm, fused_add_rms_norm

    config = FusedNormConfig(hidden_dim=4096, eps=1e-6)
    norm = FusedRMSNorm(config)

    # Sequence of shape (seq_len, hidden_dim)
    x = np.random.randn(512, 4096).astype(np.float32)
    residual = np.zeros_like(x)
    out, new_residual = norm.forward(x, residual)
    print(out.shape, new_residual.shape)  # (512, 4096) (512, 4096)

    # Module-level fused function
    out2, res2 = fused_add_rms_norm(x, residual, norm.weight, eps=1e-6)
"""

from __future__ import annotations

__all__ = [
    "FusedNormConfig",
    "FusedRMSNorm",
    "FusedLayerNorm",
    "fused_add_rms_norm",
]

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class FusedNormConfig:
    """Configuration shared by :class:`FusedRMSNorm` and :class:`FusedLayerNorm`.

    Attributes:
        hidden_dim: Feature dimension (last axis of the input tensor).
        eps: Small constant added to the denominator for numerical stability.
        add_residual: When ``True``, ``forward`` adds the residual to ``x``
            before normalizing and returns the summed tensor as
            ``residual_out``.
        elementwise_scale: When ``True``, the normalized output is multiplied
            element-wise by the learned weight vector.
    """

    hidden_dim: int = 4096
    eps: float = 1e-6
    add_residual: bool = True
    elementwise_scale: bool = True

    def __post_init__(self) -> None:
        if self.hidden_dim <= 0:
            raise ValueError(
                f"hidden_dim must be a positive integer, got {self.hidden_dim}"
            )
        if self.eps <= 0.0:
            raise ValueError(f"eps must be positive, got {self.eps}")


class FusedRMSNorm:
    """Fused RMSNorm with optional residual add and element-wise scale.

    Performs a single read of ``x`` (and optionally ``residual``), computes
    the RMS denominator, normalizes, and applies the learned scale in one
    sequential pass — equivalent to what a fused CUDA kernel would do.
    """

    def __init__(
        self,
        config: FusedNormConfig,
        weight: Optional[np.ndarray] = None,
    ) -> None:
        """
        Args:
            config: Normalization configuration.
            weight: Scale tensor of shape ``(hidden_dim,)`` and dtype
                ``float32``.  If ``None``, initialized to all-ones.
        """
        self._config = config
        if weight is not None:
            if weight.shape != (config.hidden_dim,):
                raise ValueError(
                    f"weight must have shape ({config.hidden_dim},), "
                    f"got {weight.shape}"
                )
            self._weight = weight.astype(np.float32, copy=False)
        else:
            self._weight = np.ones(config.hidden_dim, dtype=np.float32)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def weight(self) -> np.ndarray:
        """The element-wise scale vector of shape ``(hidden_dim,)``."""
        return self._weight

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        x: np.ndarray,
        residual: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Fused residual add + RMS normalization + optional element-wise scale.

        Args:
            x: Input tensor of shape ``(..., hidden_dim)``.
            residual: Optional residual of the same shape as ``x``.  Only
                used when ``config.add_residual`` is ``True``.

        Returns:
            A tuple ``(norm_output, residual_out)`` where:
            - ``norm_output`` is the normalized (and scaled) tensor of the
              same shape as ``x``.
            - ``residual_out`` is the summed pre-norm tensor when
              ``add_residual`` is ``True`` and *residual* is provided;
              otherwise ``None``.
        """
        if self._config.add_residual and residual is not None:
            x_proc = x + residual
            residual_out: Optional[np.ndarray] = x_proc
        else:
            x_proc = x
            residual_out = None

        # RMS over the last axis — shape (..., 1)
        rms = np.sqrt(
            np.mean(x_proc ** 2, axis=-1, keepdims=True) + self._config.eps
        )
        x_norm = x_proc / rms

        if self._config.elementwise_scale:
            x_norm = x_norm * self._weight

        return x_norm, residual_out


class FusedLayerNorm:
    """Fused standard LayerNorm (mean + variance) with optional residual add.

    Follows the same fused-residual API as :class:`FusedRMSNorm` for
    drop-in substitution in architectures that use full LayerNorm rather
    than RMSNorm.
    """

    def __init__(
        self,
        config: FusedNormConfig,
        weight: Optional[np.ndarray] = None,
        bias: Optional[np.ndarray] = None,
    ) -> None:
        """
        Args:
            config: Normalization configuration.
            weight: Scale (gamma) tensor of shape ``(hidden_dim,)``; defaults
                to ones.
            bias: Shift (beta) tensor of shape ``(hidden_dim,)``; defaults to
                zeros when ``elementwise_scale`` is ``True``, otherwise
                ignored.
        """
        self._config = config
        if weight is not None:
            if weight.shape != (config.hidden_dim,):
                raise ValueError(
                    f"weight must have shape ({config.hidden_dim},), "
                    f"got {weight.shape}"
                )
            self._weight = weight.astype(np.float32, copy=False)
        else:
            self._weight = np.ones(config.hidden_dim, dtype=np.float32)

        if bias is not None:
            if bias.shape != (config.hidden_dim,):
                raise ValueError(
                    f"bias must have shape ({config.hidden_dim},), "
                    f"got {bias.shape}"
                )
            self._bias: Optional[np.ndarray] = bias.astype(
                np.float32, copy=False
            )
        else:
            self._bias = np.zeros(config.hidden_dim, dtype=np.float32)

    @property
    def weight(self) -> np.ndarray:
        """The element-wise scale (gamma) vector."""
        return self._weight

    @property
    def bias(self) -> Optional[np.ndarray]:
        """The element-wise shift (beta) vector, or ``None`` if not set."""
        return self._bias

    def forward(
        self,
        x: np.ndarray,
        residual: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Fused residual add + LayerNorm + optional element-wise affine.

        Args:
            x: Input tensor of shape ``(..., hidden_dim)``.
            residual: Optional residual of the same shape as ``x``.

        Returns:
            ``(norm_output, residual_out)`` following the same convention
            as :meth:`FusedRMSNorm.forward`.
        """
        if self._config.add_residual and residual is not None:
            x_proc = x + residual
            residual_out: Optional[np.ndarray] = x_proc
        else:
            x_proc = x
            residual_out = None

        mean = np.mean(x_proc, axis=-1, keepdims=True)
        var = np.mean((x_proc - mean) ** 2, axis=-1, keepdims=True)
        x_norm = (x_proc - mean) / np.sqrt(var + self._config.eps)

        if self._config.elementwise_scale:
            x_norm = x_norm * self._weight
            if self._bias is not None:
                x_norm = x_norm + self._bias

        return x_norm, residual_out


# ---------------------------------------------------------------------------
# Module-level fused function
# ---------------------------------------------------------------------------

def fused_add_rms_norm(
    x: np.ndarray,
    residual: np.ndarray,
    weight: np.ndarray,
    eps: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Fused residual add + RMS normalization (standalone function).

    Computes ``new_residual = x + residual``, then normalizes by the
    per-vector RMS and applies the element-wise scale ``weight``.

    Args:
        x: Input tensor of shape ``(..., hidden_dim)``.
        residual: Residual tensor of the same shape as ``x``.
        weight: Scale vector of shape ``(hidden_dim,)``.
        eps: Numerical stability epsilon.

    Returns:
        ``(norm_output, new_residual)`` — the normalized+scaled output and
        the summed pre-norm tensor (which becomes the residual for the next
        layer).
    """
    new_residual = x + residual
    rms = np.sqrt(np.mean(new_residual ** 2, axis=-1, keepdims=True) + eps)
    norm_output = (new_residual / rms) * weight
    return norm_output, new_residual
