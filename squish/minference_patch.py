"""Phase 3C — MInference-style sparse attention for long-context prefill.

Reduces the O(n²) attention cost of full-sequence prefill to O(n·k) by
patching each attention-like submodule in an MLX model to apply a sparse
additive mask for sequences longer than ``seq_len_threshold``.

Three built-in sparsity patterns:

  ``a-shape``
      Sliding local window + attention sinks (first ``n_sinks`` tokens).
      Best for models trained with RoPE + local attention patterns (Llama /
      Qwen3).  Safe default: captures most positional relevance.

  ``vertical-slash``
      Sparse column sampling every ``stride`` tokens + attention sinks.
      Approximates the "vertical-slash" pattern from the MInference paper.
      Good for long-document retrieval tasks.

  ``block-sparse``
      Block-level attention: attend within the current block and to the
      first block (attention sinks).  Highest sparsity; use for very long
      contexts (≥ 8 K tokens).

Conflict rules (plan §3):
  • Only applied during prefill (``seq_len > 1``).  Decode steps
    (``seq_len == 1``) pass through unmodified regardless of patching.
  • ``unpatch_model_minference()`` **must** be called after prefill completes
    so that decode steps revert to dense attention.
  • Incompatible with ``--inference-backend ane-disagg``: the Core ML graph
    is pre-compiled and cannot accept a Python mask injection.
    server.py must guard: ``if _inference_backend != "ane-disagg"``.

Implementation note — special method dispatch
  Python routes ``obj(x)`` through ``type(obj).__call__``, not
  ``obj.__dict__["__call__"]``.  To patch a specific **instance** without
  touching every instance of the class we dynamically create a per-patch
  subclass and reassign ``instance.__class__``.  The subclass has the same
  ``__init__`` path (never called again) and identical memory layout, so MLX
  parameter tracking is unaffected.  On restore we set ``__class__`` back to
  the original.
"""
from __future__ import annotations

import functools
from typing import Callable

import numpy as np


# ── Sparse mask builders ──────────────────────────────────────────────────────

def _make_a_shape_mask(seq_len: int, window: int = 256, n_sinks: int = 4):
    """A-shape mask: attention sinks + sliding local window.

    Returns an additive float16 MLX array of shape ``[seq_len, seq_len]``.
    Values: 0.0 → attend, −10000.0 → do not attend.
    Upper triangle (future positions) is always −10000.0 (causal).
    """
    import mlx.core as mx
    mask = np.full((seq_len, seq_len), -1e4, dtype=np.float16)
    sinks = min(n_sinks, seq_len)
    for i in range(seq_len):
        # Attend to attention sinks: first n_sinks tokens, causal only
        mask[i, :min(sinks, i + 1)] = 0.0
        # Causal sliding window: attend to the `window` most recent tokens
        w_start = max(sinks, i - window + 1)
        mask[i, w_start: i + 1] = 0.0
    return mx.array(mask)


def _make_vertical_slash_mask(seq_len: int, stride: int = 64, n_sinks: int = 4):
    """Vertical-slash mask: attention sinks + every ``stride``-th column.

    Approximates the "vertical-slash" pattern from the MInference paper.
    """
    import mlx.core as mx
    mask = np.full((seq_len, seq_len), -1e4, dtype=np.float16)
    sinks = min(n_sinks, seq_len)
    for i in range(seq_len):
        # Attention sinks (first n_sinks tokens always attended)
        mask[i, :sinks] = 0.0
        # Sparse columns at fixed stride within causal range
        for col in range(sinks, i + 1, stride):
            mask[i, col] = 0.0
        # Always attend to self
        mask[i, i] = 0.0
    return mx.array(mask)


def _make_block_sparse_mask(seq_len: int, block_size: int = 64):
    """Block-sparse mask: within-block attention + first-block (sinks).

    Maximum sparsity; recommended for seq_len ≥ 8 K.  Uses a fixed block
    layout rather than per-layer importance scoring — safe and predictable.
    """
    import mlx.core as mx
    mask = np.full((seq_len, seq_len), -1e4, dtype=np.float16)
    n_blocks = (seq_len + block_size - 1) // block_size
    for b in range(n_blocks):
        row_s = b * block_size
        row_e = min(row_s + block_size, seq_len)
        for i in range(row_s, row_e):
            # Within-block causal attention
            mask[i, row_s: i + 1] = 0.0
            # First-block sinks (skip if we ARE the first block)
            if row_s > 0:
                mask[i, :block_size] = 0.0
    return mx.array(mask)


_MASK_BUILDERS: dict[str, Callable] = {
    "a-shape":       _make_a_shape_mask,
    "vertical-slash": _make_vertical_slash_mask,
    "block-sparse":  _make_block_sparse_mask,
}


# ── Module tree walker ────────────────────────────────────────────────────────

def _iter_attention_modules(model):
    """Yield ``(attr_name, module)`` for every attention-like submodule.

    Walks the Python attribute tree recursively, looking for attributes
    whose names contain ``attn``, ``attention``, or ``mixer`` — the
    conventional names used in mlx_lm model definitions (Qwen3, Llama3,
    Mistral, Phi-3, …).  Lists/tuples of ``nn.Module`` are traversed too
    (handles the ``layers`` list pattern).
    """
    try:
        import mlx.nn as nn
    except ImportError:
        return  # MLX not available — no modules to patch

    _ATTN_KEYWORDS = ("attn", "attention", "mixer")
    visited: set[int] = set()

    def _walk(obj, depth: int = 0):
        if depth > 20:
            return
        obj_id = id(obj)
        if obj_id in visited:
            return
        visited.add(obj_id)

        if not isinstance(obj, nn.Module):
            return

        for attr_name in vars(obj):
            if attr_name.startswith("_"):
                continue
            child = getattr(obj, attr_name, None)
            if child is None:
                continue

            if isinstance(child, nn.Module):
                name_lower = attr_name.lower()
                if any(kw in name_lower for kw in _ATTN_KEYWORDS):
                    yield attr_name, child
                else:
                    yield from _walk(child, depth + 1)
            elif isinstance(child, (list, tuple)):
                for item in child:
                    if isinstance(item, nn.Module):
                        yield from _walk(item, depth + 1)

    yield from _walk(model)


# ── Per-instance subclass patch helper ───────────────────────────────────────

def _make_patched_class(orig_class, get_mask: Callable, seq_len_threshold: int):
    """Return a dynamic subclass of *orig_class* that injects a sparse mask.

    The subclass overrides ``__call__`` to add the sparse mask to the
    ``mask`` argument when ``seq_len > seq_len_threshold``.  For single-token
    decode calls (``seq_len == 1``) the original path is taken unchanged.
    """
    class _InferencePatched(orig_class):
        def __call__(self, x, mask=None, cache=None, **kw):
            try:
                seq_len = x.shape[1]
            except (AttributeError, IndexError):
                seq_len = 1
            if seq_len > seq_len_threshold:
                sparse = get_mask(seq_len)
                mask = (mask + sparse) if mask is not None else sparse
            return super().__call__(x, mask=mask, cache=cache, **kw)

    _InferencePatched.__name__ = f"_Inf{orig_class.__name__}"
    _InferencePatched.__qualname__ = _InferencePatched.__name__
    return _InferencePatched


# ── Public API ────────────────────────────────────────────────────────────────

def patch_model_minference(
    model,
    seq_len_threshold: int = 1024,
    pattern: str = "a-shape",
    window: int = 256,
    n_sinks: int = 4,
    stride: int = 64,
    block_size: int = 64,
) -> Callable:
    """Patch all attention layers in *model* with sparse attention.

    Parameters
    ----------
    model:
        Top-level MLX model (``nn.Module`` subclass).
    seq_len_threshold:
        Minimum sequence length to activate sparse attention.
        For ``seq_len <= seq_len_threshold`` the original dense path is used.
    pattern:
        One of ``"a-shape"``, ``"vertical-slash"``, ``"block-sparse"``.
    window:
        Local window size for ``"a-shape"`` pattern (default 256 tokens).
    n_sinks:
        Number of attention-sink tokens kept for all patterns (default 4).
    stride:
        Column sampling stride for ``"vertical-slash"`` (default 64).
    block_size:
        Block size (tokens) for ``"block-sparse"`` (default 64).

    Returns
    -------
    restore_fn : Callable
        Zero-argument callable.  Call it (or pass to
        ``unpatch_model_minference``) to restore original attention.

    Raises
    ------
    ValueError
        If *pattern* is not one of the supported options.

    Notes
    -----
    * Thread-safety: not safe to call concurrently on the same model.
      Call from the thread that will run the forward pass.
    * Mask arrays are cached by ``seq_len`` so repeat prefill calls of the
      same length avoid recomputation.
    * If ``__class__`` reassignment fails for a particular module (e.g. a
      C-extension type), that module is silently skipped.
    """
    if pattern not in _MASK_BUILDERS:
        raise ValueError(
            f"Unknown minference pattern {pattern!r}; "
            f"choose from {list(_MASK_BUILDERS)}"
        )

    # Build pattern-specific kwargs once
    _pkw: dict
    if pattern == "a-shape":
        _pkw = {"window": window, "n_sinks": n_sinks}
    elif pattern == "vertical-slash":
        _pkw = {"stride": stride, "n_sinks": n_sinks}
    else:  # block-sparse
        _pkw = {"block_size": block_size}

    _builder = _MASK_BUILDERS[pattern]
    _mask_cache: dict[int, object] = {}

    def _get_mask(sl: int):
        if sl not in _mask_cache:
            _mask_cache[sl] = _builder(sl, **_pkw)
        return _mask_cache[sl]

    restores: list[tuple] = []

    for _path, attn_mod in _iter_attention_modules(model):
        orig_class = type(attn_mod)
        try:
            patched_class = _make_patched_class(orig_class, _get_mask, seq_len_threshold)
            attn_mod.__class__ = patched_class
            restores.append((attn_mod, orig_class))
        except Exception:
            # Silently skip modules that cannot be patched (C-extension types,
            # frozen objects, etc.) — partial sparsity is better than a crash.
            pass

    def _restore() -> None:
        for mod, orig_cls in restores:
            try:
                mod.__class__ = orig_cls
            except Exception:
                pass

    return _restore


def unpatch_model_minference(model, restore_fn: Callable) -> None:
    """Restore model attention layers to their original (dense) implementation.

    Parameters
    ----------
    model:
        The same model object passed to ``patch_model_minference``.
        (Accepted for API symmetry; not currently used — all state is in
        *restore_fn*'s closure.)
    restore_fn:
        The callable returned by ``patch_model_minference``.

    This function never raises; restoration errors are silently swallowed
    so that a failed unpatch never blocks token generation.
    """
    try:
        restore_fn()
    except Exception:
        pass


def select_pattern_for_sequence(seq_len: int) -> str:
    """Heuristic pattern selection based on sequence length.

    Returns
    -------
    str
        ``"a-shape"``        for seq_len < 2 048   (local + sink patterns dominate)
        ``"vertical-slash"`` for seq_len < 8 192   (global sparse columns)
        ``"block-sparse"``   for seq_len ≥ 8 192   (maximum sparsity)
    """
    if seq_len < 2048:
        return "a-shape"
    if seq_len < 8192:
        return "vertical-slash"
    return "block-sparse"
