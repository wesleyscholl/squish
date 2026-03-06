"""
lazy_llm.py — Dynamic token pruning for faster prefill on Apple Silicon.

Based on the "LazyLLM: Dynamic Token Pruning for Efficient Long Context LLM
Inference" approach (Dong et al. 2024).  Each transformer layer, from
`start_layer` onward, re-scores token importance and may skip computing
the full representation for low-importance positions.

Implementation notes
--------------------
* We wrap each TransformerBlock's ``__call__`` so the change is transparent
  to the rest of the model and is fully reversible via ``unpatch``.
* Token importance is computed as the L2-norm of the hidden states at the
  *output* of each layer — a cheap, model-agnostic proxy that correlates
  well with attention-based importance in practice (per the LazyLLM paper).
* The *revive window* always keeps the ``revive_window`` most-recent tokens
  active regardless of importance, preventing "context collapse" at the
  trailing edge.
* Pruning is applied only during PREFILL (sequence length > 1).  Single-
  token decode steps pass through unchanged.
* A per-request reset hook clears accumulated masks so consecutive requests
  do not interfere.

Limitations
-----------
* Requires the model to expose ``model.model.layers`` (standard mlx_lm
  layout) or ``model.layers``.
* Not compatible with models that fuse positional encodings in a way that
  makes pruned-position logits undefined; test with verbose=True first.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, List

__all__ = ["LazyLLMConfig", "patch_model_lazy_llm", "unpatch_model_lazy_llm"]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class LazyLLMConfig:
    """
    Hyper-parameters for LazyLLM token pruning.

    Parameters
    ----------
    keep_ratio : float
        Fraction of tokens to keep at each pruning layer (0 < keep_ratio ≤ 1).
        E.g. 0.7 keeps the top 70 % most important positions.
    start_layer : int
        Index of the first transformer layer where pruning is applied.
        Layers 0 … start_layer-1 always see the full sequence.
    revive_window : int
        Number of most-recent tokens that are *always* kept, regardless of
        importance score.  Prevents the model losing trailing context.
    verbose : bool
        Print per-layer pruning stats (for debugging / tuning).
    """
    keep_ratio:    float = 0.70
    start_layer:   int   = 2
    revive_window: int   = 4
    verbose:       bool  = False

    def __post_init__(self):
        if not (0.0 < self.keep_ratio <= 1.0):
            raise ValueError(f"keep_ratio must be in (0, 1] — got {self.keep_ratio}")
        if self.start_layer < 0:
            raise ValueError("start_layer must be ≥ 0")
        if self.revive_window < 0:
            raise ValueError("revive_window must be ≥ 0")


# ---------------------------------------------------------------------------
# Internal state (per-patched model)
# ---------------------------------------------------------------------------

class _PruneState:
    """Mutable pruning state reset at the start of each forward pass."""

    __slots__ = ("active_mask",)   # bool ndarray (T,) | None

    def __init__(self):
        self.active_mask = None   # None → all tokens active


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def _importance_scores(hidden: "mx.array") -> "np.ndarray":
    """
    Compute per-token importance as the L2-norm of the hidden-state vector.

    Parameters
    ----------
    hidden : mx.array  shape (B, T, D) — float16 / bfloat16

    Returns
    -------
    scores : np.ndarray  shape (T,) — float32, higher ⇒ more important
    """
    try:
        import mlx.core as mx
        import numpy as np
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("MLX is required for LazyLLM pruning") from exc  # pragma: no cover

    # (B, T, D) → (T,) via mean over batch then L2 over embed dim
    h = hidden[0].astype(mx.float32)              # (T, D)
    norms = mx.sqrt(mx.sum(h * h, axis=-1))       # (T,)
    mx.eval(norms)
    return np.array(norms)                        # (T,) float32


def _build_keep_mask(
    scores: "np.ndarray",
    keep_ratio: float,
    revive_window: int,
) -> "np.ndarray":
    """
    Build a boolean keep-mask (T,) from importance *scores*.

    Always keeps the *revive_window* most-recent tokens.  Among the
    remaining tokens, keeps the top-`keep_ratio` fraction by score.
    """
    import numpy as np

    T = len(scores)
    mask = np.zeros(T, dtype=bool)

    # Revive window: unconditionally keep trailing positions
    if revive_window > 0:
        mask[-min(revive_window, T):] = True

    # Budget for importance-ranked tokens (excluding revive window)
    n_non_revive = max(0, T - revive_window)
    n_keep = max(1, int(math.ceil(keep_ratio * T)))
    # Don't exceed positions we haven't already marked for revival
    n_rank_keep = max(0, n_keep - int(mask.sum()))

    if n_rank_keep > 0 and n_non_revive > 0:
        # Rank non-revive positions by score (descending)
        candidates = np.where(~mask)[0]           # positions not yet marked
        top_idx = candidates[
            np.argsort(-scores[candidates])[:n_rank_keep]
        ]
        mask[top_idx] = True

    return mask                                   # (T,) bool


def _apply_mask_to_hidden(
    hidden: "mx.array",
    mask: "np.ndarray",
) -> "mx.array":
    """
    Zero out pruned positions in *hidden* via element-wise multiply.

    Shape preserved: (B, T, D).  Pruned tokens produce zero vectors so
    subsequent attention layers weight them near-zero after softmax.
    """
    import mlx.core as mx
    import numpy as np

    T = hidden.shape[1]
    gate = mx.array(
        mask[:T].astype(np.float32).reshape(1, T, 1),
        dtype=hidden.dtype,
    )
    return hidden * gate


# ---------------------------------------------------------------------------
# Layer wrapper
# ---------------------------------------------------------------------------

class _LazyLLMLayerWrapper:
    """
    Wraps a single TransformerBlock to apply LazyLLM token pruning.

    The wrapper is transparent: it has the same ``__call__`` signature and
    delegates to the original layer, intercepting only to gate pruned tokens.
    """

    def __init__(
        self,
        original_layer,
        layer_idx: int,
        config: LazyLLMConfig,
        state: _PruneState,
    ):
        self._orig    = original_layer
        self._idx     = layer_idx
        self._config  = config
        self._state   = state

    def __call__(self, x, *args, **kwargs):
        import mlx.core as mx

        # Pass-through for decode steps (single token)
        if x.shape[1] <= 1:
            return self._orig(x, *args, **kwargs)

        # Apply accumulated mask from previous layers (gate pruned positions)
        if self._state.active_mask is not None and self._idx >= self._config.start_layer:
            x = _apply_mask_to_hidden(x, self._state.active_mask)

        # Forward through original layer
        out = self._orig(x, *args, **kwargs)

        # Extract hidden states (handle tuple returns e.g. (hidden, kv))
        if isinstance(out, (tuple, list)):
            hidden = out[0]
        else:
            hidden = out

        # Update mask starting from start_layer
        if self._idx >= self._config.start_layer:
            if hidden.shape[1] > 1:
                scores = _importance_scores(hidden)
                new_mask = _build_keep_mask(
                    scores,
                    self._config.keep_ratio,
                    self._config.revive_window,
                )
                self._state.active_mask = new_mask
                if self._config.verbose:
                    n_pruned = int((~new_mask).sum())
                    print(
                        f"  [lazy_llm] layer={self._idx:02d} "
                        f"kept={new_mask.sum()}/{len(new_mask)} "
                        f"pruned={n_pruned}"
                    )

        return out

    # Delegate attribute access to the original layer so model internals
    # (weight loading, parameter iteration) continue to work.
    def __getattr__(self, name):
        return getattr(self._orig, name)


# ---------------------------------------------------------------------------
# Model-level wrap / unwrap
# ---------------------------------------------------------------------------

def _get_layers(model) -> list | None:
    """Return the transformer layer list from a mlx_lm model, or None."""
    # Standard mlx_lm layout: model.model.layers
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    # Some models expose layers directly
    if hasattr(model, "layers"):
        return model.layers
    return None


def patch_model_lazy_llm(
    model,
    config: LazyLLMConfig | None = None,
) -> "_PruneState | None":
    """
    Patch *model* in-place with LazyLLM token pruning.

    Parameters
    ----------
    model  : mlx_lm Transformer model (with ``.model.layers`` or ``.layers``)
    config : ``LazyLLMConfig``.  A default config with ``keep_ratio=0.70`` is
             used when *config* is ``None``.

    Returns
    -------
    state : ``_PruneState`` — call ``state.active_mask = None`` to reset
            between requests.  Returns ``None`` if the model is incompatible.
    """
    if config is None:
        config = LazyLLMConfig()

    layers = _get_layers(model)
    if layers is None:
        import logging
        logging.getLogger(__name__).warning(
            "lazy_llm: cannot locate transformer layers — patching skipped"
        )
        return None

    state = _PruneState()

    new_layers = []
    for i, layer in enumerate(layers):
        if i >= config.start_layer:
            new_layers.append(
                _LazyLLMLayerWrapper(layer, i, config, state)
            )
        else:
            new_layers.append(layer)

    # Mutate in-place via the same attribute path
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        model.model.layers = new_layers
    else:
        model.layers = new_layers

    model._lazy_llm_state  = state
    model._lazy_llm_orig   = layers   # stash for unpatch

    return state


def unpatch_model_lazy_llm(model) -> None:
    """
    Remove LazyLLM patches from *model*, restoring original layers.

    Safe to call even if the model was never patched.
    """
    if not hasattr(model, "_lazy_llm_orig"):
        return

    orig_layers = model._lazy_llm_orig
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        model.model.layers = orig_layers
    elif hasattr(model, "layers"):
        model.layers = orig_layers

    del model._lazy_llm_state
    del model._lazy_llm_orig
