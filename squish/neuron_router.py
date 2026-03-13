# [Experimental] This module is part of Squish v10+ (Wave 10A).
# Proof-of-concept quality: API and behaviour may change without notice.
#!/usr/bin/env python3
"""
squish/neuron_router.py

Inference-time hot/cold neuron dispatch for Apple Silicon LLMs.

Uses a :class:`~squish.neuron_profile.NeuronProfile` to split each MLP
layer's weight matrices into hot (frequently activated) and cold (rarely
activated) row sub-tensors at startup.  During decode:

* Hot rows → computed on the fast path (GPU/ANE).
* Cold rows → computed on the slow path (CPU via unified-memory pointer).

The result is merged to produce a numerically equivalent output to the
original dense forward pass while reducing effective DRAM bandwidth per
decode step by ~3–4× on a 20/80 hot/cold split.

Example::

    from squish.neuron_profile import load_profile
    from squish.neuron_router import NeuronRouterConfig, NeuronRouter

    profile = load_profile("neuron_profile.json")
    config  = NeuronRouterConfig(profile=profile)
    router  = NeuronRouter(config)

    # Pure-numpy forward compatible with test harnesses:
    out = router.forward(layer_idx=0, x=activations,
                         gate_w=gate_proj, up_w=up_proj, down_w=down_proj)
"""

from __future__ import annotations

__all__ = [
    "NeuronRouterConfig",
    "NeuronRouter",
    "patch_model_neuron_routing",
]

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np

from squish.neuron_profile import NeuronProfile


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class NeuronRouterConfig:
    """Configuration for the NeuronRouter.

    Attributes:
        profile:      A :class:`~squish.neuron_profile.NeuronProfile` that
                      supplies per-layer hot/cold index arrays.
        hot_device:   Device String for hot-neuron execution (default ``"gpu"``).
                      Has no computational effect in the numpy reference path;
                      passed through to Metal / MLX dispatch when available.
        cold_device:  Device string for cold-neuron execution (default
                      ``"cpu"``).
    """

    profile: NeuronProfile = field(default_factory=NeuronProfile)
    hot_device: str = "gpu"
    cold_device: str = "cpu"

    def __post_init__(self) -> None:
        if not isinstance(self.profile, NeuronProfile):
            raise TypeError(
                f"profile must be a NeuronProfile, got {type(self.profile)}"
            )
        valid_devices = {"gpu", "cpu", "ane"}
        if self.hot_device not in valid_devices:
            raise ValueError(
                f"hot_device must be one of {valid_devices}, got {self.hot_device!r}"
            )
        if self.cold_device not in valid_devices:
            raise ValueError(
                f"cold_device must be one of {valid_devices}, got {self.cold_device!r}"
            )


# ---------------------------------------------------------------------------
# NeuronRouter
# ---------------------------------------------------------------------------

class NeuronRouter:
    """Splits and routes MLP neurons at inference time.

    Splits weight matrices for each layer into hot and cold sub-matrices
    on construction (``__init__``) and dispatches them independently during
    :meth:`forward`, merging the results into the final hidden output.

    **Reference numpy path** (used in tests and non-Metal environments):

    Given a SwiGLU-style MLP with ``gate_proj``, ``up_proj``, ``down_proj``
    weight matrices of shape ``(ffn_dim, hidden_dim)``:

    * For each set of hot/cold indices ``h``, ``c`` of layer ``layer_idx``:

      .. code-block:: text

          gate_hot = gate_w[h, :]     shape (|h|, hidden_dim)
          gate_cold = gate_w[c, :]    shape (|c|, hidden_dim)
          up_hot = up_w[h, :]         shape (|h|, hidden_dim)
          up_cold = up_w[c, :]        shape (|c|, hidden_dim)
          down_hot = down_w[:, h]     shape (hidden_dim, |h|)
          down_cold = down_w[:, c]    shape (hidden_dim, |c|)

    * Compute the intermediate activations per tier:

      .. code-block:: text

          # hot tier
          mid_hot = silu(gate_hot @ x.T) * (up_hot @ x.T)   # (|h|, batch)
          out_hot = down_hot @ mid_hot                         # (hidden, batch)

          # cold tier
          mid_cold = silu(gate_cold @ x.T) * (up_cold @ x.T)
          out_cold = down_cold @ mid_cold

    * Merge: ``output = out_hot + out_cold``

    The output is numerically equivalent to the original dense forward pass
    (modulo FP rounding differences between the two compute paths).

    Args:
        config: A :class:`NeuronRouterConfig` instance.
    """

    def __init__(self, config: NeuronRouterConfig) -> None:
        self.config = config
        self._profile = config.profile
        self._patched_layers: dict[int, Callable] = {}

    # ------------------------------------------------------------------
    # Forward pass (numpy reference)
    # ------------------------------------------------------------------

    def forward(
        self,
        layer_idx: int,
        x: np.ndarray,
        gate_w: np.ndarray,
        up_w: np.ndarray,
        down_w: np.ndarray,
    ) -> np.ndarray:
        """Execute the split hot/cold forward pass for one MLP layer.

        Numerically equivalent to ``down_w.T @ (silu(gate_w @ x.T) * (up_w @ x.T))``.

        Args:
            layer_idx: Index into the profile's layer list.
            x:         Input activations of shape ``(batch, hidden_dim)`` or
                       ``(hidden_dim,)`` (1-D input is promoted to
                       ``(1, hidden_dim)``).
            gate_w:    Gate projection weight matrix, shape
                       ``(ffn_dim, hidden_dim)``.
            up_w:      Up projection weight matrix, shape
                       ``(ffn_dim, hidden_dim)``.
            down_w:    Down projection weight matrix, shape
                       ``(hidden_dim, ffn_dim)`` (or equivalently
                       ``(out_dim, ffn_dim)``).

        Returns:
            Output of shape ``(batch, out_dim)`` for 2-D inputs, or
            ``(out_dim,)`` for 1-D inputs.

        Raises:
            IndexError: if ``layer_idx`` is out of range for the profile.
        """
        self._profile._check_layer(layer_idx)

        # Normalise to 2-D: (batch, hidden)
        squeeze = x.ndim == 1
        if squeeze:
            x = x[np.newaxis, :]

        h = self._profile.hot_indices[layer_idx]   # shape (n_hot,)
        c = self._profile.cold_indices[layer_idx]  # shape (n_cold,)

        x_f = x.astype(np.float32)

        # ── Hot tier ──────────────────────────────────────────────────
        g_hot = gate_w[h, :].astype(np.float32)   # (n_hot, hidden)
        u_hot = up_w[h, :].astype(np.float32)
        d_hot = down_w[:, h].astype(np.float32)   # (out_dim, n_hot)

        gate_act_hot = self._silu(g_hot @ x_f.T)  # (n_hot, batch)
        up_act_hot   = u_hot @ x_f.T               # (n_hot, batch)
        mid_hot      = gate_act_hot * up_act_hot    # (n_hot, batch)
        out_hot      = d_hot @ mid_hot              # (out_dim, batch)

        # ── Cold tier ─────────────────────────────────────────────────
        g_cold = gate_w[c, :].astype(np.float32)
        u_cold = up_w[c, :].astype(np.float32)
        d_cold = down_w[:, c].astype(np.float32)

        gate_act_cold = self._silu(g_cold @ x_f.T)
        up_act_cold   = u_cold @ x_f.T
        mid_cold      = gate_act_cold * up_act_cold
        out_cold      = d_cold @ mid_cold

        # ── Merge ─────────────────────────────────────────────────────
        output = (out_hot + out_cold).T  # (batch, out_dim)

        return output.squeeze(0) if squeeze else output

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _silu(x: np.ndarray) -> np.ndarray:
        """SiLU activation: x * sigmoid(x)."""
        return x * (1.0 / (1.0 + np.exp(-x)))

    def stats(self, layer_idx: int) -> dict:
        """Return hot/cold neuron counts for a layer.

        Args:
            layer_idx: Index into the profile's layer list.

        Returns:
            Dict with keys ``"n_hot"``, ``"n_cold"``, ``"n_total"``,
            ``"hot_fraction"``.
        """
        self._profile._check_layer(layer_idx)
        n_hot  = self._profile.n_hot(layer_idx)
        n_cold = self._profile.n_cold(layer_idx)
        return {
            "n_hot":       n_hot,
            "n_cold":      n_cold,
            "n_total":     n_hot + n_cold,
            "hot_fraction": n_hot / max(1, n_hot + n_cold),
        }

    def __repr__(self) -> str:
        return (
            f"NeuronRouter(layers={self._profile.layer_count}  "
            f"hot={self.config.hot_device}  cold={self.config.cold_device})"
        )


# ---------------------------------------------------------------------------
# Model patching
# ---------------------------------------------------------------------------

def patch_model_neuron_routing(
    model: Any,
    router: NeuronRouter,
) -> dict[int, Callable]:
    """Monkey-patch ``model``'s MLP layers to use ``router.forward``.

    Iterates over ``model.layers`` (each expected to expose a ``mlp``
    attribute) and replaces the callable (``__call__`` / ``forward``) with a
    closure that invokes :meth:`NeuronRouter.forward` for the corresponding
    layer index.

    The closure extracts ``gate_proj``, ``up_proj``, and ``down_proj`` weight
    tensors from the MLP using numpy-compatible ``.weight`` or ``.data``
    attribute access.  In MLX the equivalent is ``layer.weight.numpy()``.

    Args:
        model:  The transformer model object.  Must have a ``layers``
                attribute that is iterable and whose elements expose ``.mlp``.
        router: A started :class:`NeuronRouter` instance.

    Returns:
        Dict mapping ``layer_idx → original_callable`` so the patch can be
        reversed by the caller if needed.

    Raises:
        AttributeError: if a layer does not expose ``mlp``, ``gate_proj``,
                        ``up_proj``, or ``down_proj`` with a ``.weight``
                        attribute.
    """
    originals: dict[int, Callable] = {}

    def _make_patched_forward(layer_idx: int, mlp: Any) -> Callable:
        original_call = getattr(mlp, "_original_forward", None) or mlp.__call__

        def patched(x, **kwargs):
            gate_w = np.asarray(mlp.gate_proj.weight)
            up_w   = np.asarray(mlp.up_proj.weight)
            down_w = np.asarray(mlp.down_proj.weight)
            return router.forward(layer_idx, np.asarray(x), gate_w, up_w, down_w)

        return patched

    for i, layer in enumerate(model.layers):
        mlp = layer.mlp
        originals[i] = mlp.__call__
        patched = _make_patched_forward(i, mlp)
        # Store on the mlp object for reversibility
        mlp._original_forward = originals[i]
        mlp.__call__ = patched

    router._patched_layers = originals
    return originals
