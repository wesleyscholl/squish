"""LoRACompose — Dynamic multi-LoRA adapter composition with mixture weights.

Allows blending multiple LoRA adapters at inference time using a convex
combination of their deltas — without merging weights into the base model.
Useful for multi-domain serving where different users need different adapters.

Reference:
    Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models",
    ICLR 2022.  https://arxiv.org/abs/2106.09685

Usage::

    from squish.lora_compose import LoRAComposer, AdapterStack

    composer = LoRAComposer(hidden_dim=4096)
    composer.add_adapter("code", A_code, B_code, scale=1.0)
    composer.add_adapter("chat", A_chat, B_chat, scale=0.8)
    out = composer.forward(x, weights={"code": 0.7, "chat": 0.3})
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

__all__ = [
    "AdapterConfig",
    "AdapterStack",
    "LoRAComposer",
    "CompositionStats",
]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AdapterConfig:
    """Configuration for constructing a single LoRA adapter from scratch.

    Parameters
    ----------
    rank : int
        Inner dimension of the low-rank factorisation (``r`` in the paper).
        Must be > 0.
    alpha : float
        Scaling hyper-parameter.  The effective delta scale is
        ``alpha / rank``.  Must be > 0.
    hidden_dim : int
        Input feature dimension (``d_model``).  Must be > 0.
    out_dim : int, optional
        Output feature dimension.  Defaults to ``hidden_dim``.  Must be > 0.
    """

    rank: int = 16
    alpha: float = 32.0
    hidden_dim: int = 4096
    out_dim: Optional[int] = None
    # Computed field — not accepted as a constructor argument.
    scaling: float = field(init=False, default=0.0)

    def __post_init__(self) -> None:
        if self.rank <= 0:
            raise ValueError(f"rank must be > 0; got {self.rank}")
        if self.alpha <= 0.0:
            raise ValueError(f"alpha must be > 0; got {self.alpha}")
        if self.hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be > 0; got {self.hidden_dim}")

        resolved_out = self.out_dim if self.out_dim is not None else self.hidden_dim
        if resolved_out <= 0:
            raise ValueError(f"out_dim must be > 0; got {resolved_out}")

        object.__setattr__(self, "out_dim", resolved_out)
        object.__setattr__(self, "scaling", self.alpha / self.rank)


# ---------------------------------------------------------------------------
# AdapterStack — single loaded LoRA
# ---------------------------------------------------------------------------


@dataclass
class AdapterStack:
    """A single registered LoRA adapter ready for inference.

    Parameters
    ----------
    name : str
        Unique identifier for this adapter within a :class:`LoRAComposer`.
    A : np.ndarray
        Down-projection matrix of shape ``(hidden_dim, rank)``.
    B : np.ndarray
        Up-projection matrix of shape ``(rank, out_dim)``.
    scaling : float
        Per-adapter scalar multiplier applied to the output delta
        ``x @ A @ B``.  Typically set to ``alpha / rank`` from
        :class:`AdapterConfig` or to a user-supplied ``scale`` value.
    """

    name: str
    A: np.ndarray  # (hidden_dim, rank)
    B: np.ndarray  # (rank, out_dim)
    scaling: float

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Compute the LoRA delta :math:`(x A B) \\cdot \\text{scaling}`.

        Parameters
        ----------
        x : np.ndarray
            Input tensor of shape ``(batch, hidden_dim)`` or
            ``(hidden_dim,)``.

        Returns
        -------
        np.ndarray
            Delta of shape ``(batch, out_dim)`` or ``(out_dim,)``, dtype
            ``float32``.
        """
        return (x.astype(np.float32) @ self.A @ self.B) * self.scaling

    @property
    def n_params(self) -> int:
        """Total number of parameters in this adapter (A + B elements)."""
        return int(self.A.size + self.B.size)


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


@dataclass
class CompositionStats:
    """Statistics accumulated by :class:`LoRAComposer`.

    Parameters
    ----------
    n_forward_calls : int
        Number of ``forward()`` calls performed.
    adapters_used_total : int
        Total adapter usages across all forward calls.  If 3 adapters are
        active in each of 10 calls, this is 30.
    """

    n_forward_calls: int
    adapters_used_total: int

    @property
    def avg_adapters_per_call(self) -> float:
        """Average number of adapters active per forward call."""
        if self.n_forward_calls == 0:
            return 0.0
        return self.adapters_used_total / self.n_forward_calls


# ---------------------------------------------------------------------------
# LoRAComposer
# ---------------------------------------------------------------------------


class LoRAComposer:
    """Runtime multi-LoRA adapter composition engine.

    Maintains a registry of named adapters and computes their weighted delta
    sum at inference time without altering the base model weights.

    Parameters
    ----------
    hidden_dim : int
        Input feature dimension expected for all registered adapters.
    out_dim : int, optional
        Output feature dimension.  Defaults to ``hidden_dim``.

    Examples
    --------
    >>> composer = LoRAComposer(hidden_dim=4096)
    >>> composer.add_adapter("code", A_code, B_code, scale=1.0)
    >>> composer.add_adapter("chat", A_chat, B_chat, scale=0.8)
    >>> delta = composer.forward(x, weights={"code": 0.7, "chat": 0.3})
    """

    def __init__(self, hidden_dim: int, out_dim: Optional[int] = None) -> None:
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be > 0; got {hidden_dim}")
        self._hidden_dim = hidden_dim
        self._out_dim = out_dim if out_dim is not None else hidden_dim
        self._adapters: Dict[str, AdapterStack] = {}
        self._n_forward_calls: int = 0
        self._adapters_used_total: int = 0

    # ── Adapter registry ─────────────────────────────────────────────────────

    def add_adapter(
        self,
        name: str,
        A: np.ndarray,
        B: np.ndarray,
        scale: float = 1.0,
    ) -> None:
        """Register a new LoRA adapter.

        Parameters
        ----------
        name : str
            Unique name.  Raises ``ValueError`` if already registered.
        A : np.ndarray
            Down-projection matrix, shape ``(hidden_dim, rank)``.
        B : np.ndarray
            Up-projection matrix, shape ``(rank, out_dim)``.
        scale : float
            Scaling factor applied to ``x @ A @ B``.  Use ``alpha / rank``
            from :class:`AdapterConfig` when following the LoRA paper.

        Raises
        ------
        ValueError
            If ``name`` is already registered, or if ``A``/``B`` shapes are
            incompatible with this composer's dimensions.
        """
        if name in self._adapters:
            raise ValueError(
                f"Adapter {name!r} is already registered; call remove_adapter() first."
            )
        if A.ndim != 2 or A.shape[0] != self._hidden_dim:
            raise ValueError(
                f"A must have shape (hidden_dim={self._hidden_dim}, rank); "
                f"got {A.shape}"
            )
        if B.ndim != 2 or B.shape[1] != self._out_dim:
            raise ValueError(
                f"B must have shape (rank, out_dim={self._out_dim}); "
                f"got {B.shape}"
            )
        if A.shape[1] != B.shape[0]:
            raise ValueError(
                f"rank mismatch: A.shape[1]={A.shape[1]} != B.shape[0]={B.shape[0]}"
            )
        self._adapters[name] = AdapterStack(
            name=name,
            A=A.astype(np.float32),
            B=B.astype(np.float32),
            scaling=float(scale),
        )

    def remove_adapter(self, name: str) -> None:
        """Remove a registered adapter.

        Raises
        ------
        KeyError
            If ``name`` is not currently registered.
        """
        if name not in self._adapters:
            raise KeyError(
                f"Adapter {name!r} not found; registered: {list(self._adapters)}"
            )
        del self._adapters[name]

    # ── Forward pass ─────────────────────────────────────────────────────────

    def forward(
        self,
        x: np.ndarray,
        weights: Optional[Dict[str, float]] = None,
    ) -> np.ndarray:
        """Compute the weighted sum of adapter deltas for input ``x``.

        Parameters
        ----------
        x : np.ndarray
            Input tensor of shape ``(batch, hidden_dim)`` or
            ``(hidden_dim,)``.
        weights : dict[str, float], optional
            Mixture weights keyed by adapter name.  Need not sum to 1;
            arbitrary scaling is permitted.  Defaults to equal weight
            ``1 / n_adapters`` for all registered adapters.

        Returns
        -------
        np.ndarray
            Weighted adapter delta with the same leading shape as ``x`` and
            output dimension ``out_dim``.  dtype is ``float32``.

        Raises
        ------
        ValueError
            If ``weights`` references an unregistered adapter name.
        """
        if not self._adapters:
            out_shape = x.shape[:-1] + (self._out_dim,)
            return np.zeros(out_shape, dtype=np.float32)

        if weights is None:
            n = len(self._adapters)
            weights = {name: 1.0 / n for name in self._adapters}

        unknown = set(weights) - set(self._adapters)
        if unknown:
            raise ValueError(
                f"Unknown adapter name(s) {sorted(unknown)}; "
                f"registered: {list(self._adapters)}"
            )

        out_shape = x.shape[:-1] + (self._out_dim,)
        delta = np.zeros(out_shape, dtype=np.float32)

        for name, w in weights.items():
            delta = delta + float(w) * self._adapters[name].forward(x)

        self._n_forward_calls += 1
        self._adapters_used_total += len(weights)
        return delta

    # ── Properties ───────────────────────────────────────────────────────────

    @property
    def adapter_names(self) -> List[str]:
        """Names of all currently registered adapters (insertion order)."""
        return list(self._adapters.keys())

    @property
    def total_params(self) -> int:
        """Total parameter count across all registered adapters."""
        return sum(adapter.n_params for adapter in self._adapters.values())

    def composition_stats(self) -> CompositionStats:
        """Return forward-pass composition statistics accumulated so far."""
        return CompositionStats(
            n_forward_calls=self._n_forward_calls,
            adapters_used_total=self._adapters_used_total,
        )
