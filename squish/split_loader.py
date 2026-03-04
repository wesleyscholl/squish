#!/usr/bin/env python3
"""
squish/split_loader.py

CPU/GPU split layer loading for models that exceed the Metal memory budget.

On Apple Silicon, RAM is unified — there is no separate VRAM.  However, the
Metal allocator has a configurable ceiling (see Phase 0.1).  When a model's
weight footprint exceeds that ceiling, this module:

  1. Profiles each transformer layer's total weight size (bytes).
  2. Greedily assigns layers to GPU (Metal allocator) until 90% of the budget
     is consumed — the "GPU partition".
  3. Stores the remaining layers' weights as CPU-resident NumPy float16 arrays,
     *outside* the Metal allocator's tracked heap.
  4. Wraps each CPU-resident layer in an OffloadedLayer that materialises
     numpy→MLX on each forward call, computes, then immediately releases the
     transiently created MLX arrays so Metal heap space is freed.

This lets a 32B model at AWQ+ZipNN (~13-15 GB) load on a 14.4 GB Metal budget:
typically only 2-4 of the 64 layers need to be offloaded, at a throughput cost
of ~1-2 tok/s per offloaded layer due to numpy↔MLX round-trips.

Usage
-----
    from squish.split_loader import SplitLayerLoader

    model, tokenizer = load_compressed_model(model_dir, compressed_dir)

    loader   = SplitLayerLoader(model, target_fraction=0.90)
    info     = loader.apply()          # mutates model.layers in-place

    print(info)
    # SplitInfo(gpu_layers=60, cpu_layers=4, gpu_gb=14.1, cpu_gb=0.9)

    # Generation proceeds normally — no API changes
    tokens = generate(model, tokenizer, prompt)

Notes
-----
* The Metal budget reported by ``mx.metal.get_memory_limit()`` is used as the
  ceiling.  If MLX is unavailable (tests), a synthetic 16 GB budget is used.
* ``target_fraction`` (default 0.90) applies a safety margin so the Metal
  allocator retains headroom for activations and the KV cache.
* Calling ``loader.restore()`` moves all CPU layers back to GPU (undoes apply).
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Lazy MLX import (module imported in unit tests without Metal GPU)
# ---------------------------------------------------------------------------

def _mx():
    import mlx.core as mx
    return mx


def _nn():
    import mlx.nn as nn
    return nn


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_metal_limit_bytes() -> int:
    """Return the Metal allocator ceiling in bytes.  Falls back to 16 GB."""
    try:
        import mlx.core as mx
        return mx.metal.get_memory_limit()
    except Exception:
        return 16 * 1024 ** 3


def _total_ram_bytes() -> int:
    """Return total physical memory in bytes (macOS / Linux)."""
    try:
        import ctypes
        lib = ctypes.CDLL("libSystem.dylib")
        mem = ctypes.c_uint64(0)
        sz  = ctypes.c_size_t(8)
        lib.sysctlbyname(b"hw.memsize", ctypes.byref(mem), ctypes.byref(sz), None, 0)
        return mem.value
    except Exception:
        pass
    try:
        import psutil
        return psutil.virtual_memory().total
    except Exception:
        return 16 * 1024 ** 3


def _layer_weight_bytes(layer) -> int:
    """
    Return the total byte footprint of all parameters in ``layer``.

    Walks the parameter tree returned by ``layer.parameters()``, handling
    nested dicts (e.g. {"self_attn": {"q_proj": {"weight": arr}}}).
    """
    try:
        import mlx.core as mx
        params = layer.parameters()
    except Exception:
        return 0

    def _walk(obj) -> int:
        if isinstance(obj, mx.array):
            return obj.nbytes
        if isinstance(obj, dict):
            return sum(_walk(v) for v in obj.values())
        if isinstance(obj, (list, tuple)):
            return sum(_walk(v) for v in obj)
        return 0

    return _walk(params)


def _flatten_params(layer) -> list[tuple[str, np.ndarray]]:
    """
    Flatten all MLX parameters of ``layer`` into (dotted_name, numpy_array)
    pairs — ready for CPU storage.

    Uses ``layer.parameters()`` which returns nested dicts, and walks them
    recursively to produce flat dotted paths.
    """
    import mlx.core as mx

    results: list[tuple[str, np.ndarray]] = []

    def _walk(obj: Any, prefix: str) -> None:
        if isinstance(obj, mx.array):
            results.append((prefix, np.array(obj, dtype=np.float16)))
        elif isinstance(obj, dict):
            for k, v in obj.items():
                _walk(v, f"{prefix}.{k}" if prefix else k)
        elif isinstance(obj, (list, tuple)):
            for i, v in enumerate(obj):
                _walk(v, f"{prefix}.{i}" if prefix else str(i))

    _walk(layer.parameters(), "")
    return results


def _restore_params(layer, flat_params: list[tuple[str, np.ndarray]]) -> None:
    """Re-load CPU numpy params back into ``layer`` as MLX bfloat16 arrays."""
    import mlx.core as mx
    mlx_weights = [(name, mx.array(arr).astype(mx.bfloat16)) for name, arr in flat_params]
    layer.load_weights(mlx_weights, strict=False)
    mx.eval(layer.parameters())


# ---------------------------------------------------------------------------
# OffloadedLayer — CPU-resident wrapper
# ---------------------------------------------------------------------------

class OffloadedLayer:
    """
    Wraps a single transformer layer whose weights live in CPU numpy arrays.

    On each forward call:
      1. Materialise: numpy float16 → MLX bfloat16  (Metal allocation)
      2. Run: layer forward pass
      3. Evaluate activation outputs (force Metal compute)
      4. Release: discard MLX weight arrays → Metal heap freed
         (Python GC + mx.clear_cache() ensures prompt release)

    The wrapped layer's ``__call__`` signature is forwarded unchanged — callers
    see no difference from a fully-GPU layer.
    """

    def __init__(self, layer, cpu_params: list[tuple[str, np.ndarray]]) -> None:
        self._layer      = layer
        self._cpu_params = cpu_params          # [(name, float16 ndarray)]
        self._lock       = threading.Lock()    # one forward at a time per layer

    # ---- forward --------------------------------------------------------

    def __call__(self, *args, **kwargs):
        import mlx.core as mx

        with self._lock:
            # 1. Materialise weights onto Metal
            mlx_w = [(n, mx.array(a).astype(mx.bfloat16)) for n, a in self._cpu_params]
            self._layer.load_weights(mlx_w, strict=False)
            mx.eval(self._layer.parameters())

            # 2. Forward pass
            out = self._layer(*args, **kwargs)

            # 3. Evaluate outputs — important: force compute BEFORE we release weights
            if isinstance(out, mx.array):
                mx.eval(out)
            elif isinstance(out, (list, tuple)):
                for x in out:
                    if isinstance(x, mx.array):
                        mx.eval(x)

            # 4. Release Metal weights — zero the parameter references
            zero_w = [(n, mx.zeros_like(v)) for n, v in mlx_w]
            self._layer.load_weights(zero_w, strict=False)
            mx.eval(self._layer.parameters())

            # Encourage Metal allocator to reclaim freed pages
            try:
                mx.clear_cache()
            except AttributeError:
                try:
                    mx.metal.clear_cache()  # older MLX API
                except Exception:
                    pass

            del mlx_w, zero_w

        return out

    # ---- delegation (for attribute access on the wrapped layer) ---------

    def __getattr__(self, name: str):
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self._layer, name)

    @property
    def cpu_bytes(self) -> int:
        """Total bytes held in numpy CPU arrays."""
        return sum(a.nbytes for _, a in self._cpu_params)


# ---------------------------------------------------------------------------
# SplitInfo dataclass
# ---------------------------------------------------------------------------

@dataclass
class SplitInfo:
    """Result of SplitLayerLoader.apply()."""
    gpu_layers:   list[int]  = field(default_factory=list)
    cpu_layers:   list[int]  = field(default_factory=list)
    gpu_bytes:    int        = 0
    cpu_bytes:    int        = 0
    metal_limit:  int        = 0
    target_bytes: int        = 0

    # ---- convenience properties ----------------------------------------

    @property
    def gpu_gb(self) -> float:
        return self.gpu_bytes / 1024 ** 3

    @property
    def cpu_gb(self) -> float:
        return self.cpu_bytes / 1024 ** 3

    @property
    def gpu_count(self) -> int:
        return len(self.gpu_layers)

    @property
    def cpu_count(self) -> int:
        return len(self.cpu_layers)

    def __str__(self) -> str:
        util_pct = 100 * self.gpu_bytes / max(self.metal_limit, 1)
        return (
            f"SplitInfo("
            f"gpu_layers={self.gpu_count}, cpu_layers={self.cpu_count}, "
            f"gpu={self.gpu_gb:.2f}GB/{self.metal_limit/1024**3:.1f}GB "
            f"({util_pct:.0f}%), cpu={self.cpu_gb:.2f}GB)"
        )


# ---------------------------------------------------------------------------
# SplitLayerLoader
# ---------------------------------------------------------------------------

class SplitLayerLoader:
    """
    Applies CPU/GPU layer splitting to a loaded MLX model.

    Parameters
    ----------
    model
        Loaded MLX nn.Module with a ``layers`` attribute (list of transformer
        layers).  Typically the output of ``load_compressed_model()``.
    target_fraction : float
        Fraction of the Metal memory limit to use for GPU-resident layers.
        Default 0.90 — keeps 10% headroom for activations and KV cache.
    force_cpu_layers : list[int] | None
        Override: force specific layer indices to CPU regardless of budget.
        Useful for testing or fine-grained control.
    verbose : bool
        Print a layer-by-layer allocation report during apply().
    """

    def __init__(
        self,
        model,
        target_fraction: float = 0.90,
        force_cpu_layers: list[int] | None = None,
        verbose: bool = True,
    ) -> None:
        self._model          = model
        self._target_fraction = target_fraction
        self._force_cpu      = set(force_cpu_layers or [])
        self._verbose        = verbose
        self._info: SplitInfo | None = None
        # Store originals for restore()
        self._original_layers: dict[int, Any] = {}
        self._cpu_param_store: dict[int, list[tuple[str, np.ndarray]]] = {}

    # ---- public API -------------------------------------------------------

    def apply(self) -> SplitInfo:
        """
        Profile layers, decide split point, offload CPU layers, return SplitInfo.

        Mutates ``model.layers`` in-place: CPU layer slots are replaced with
        ``OffloadedLayer`` wrappers.  GPU layers are left unchanged.
        """
        metal_limit  = _get_metal_limit_bytes()
        target_bytes = int(metal_limit * self._target_fraction)

        layers = self._model.layers
        n = len(layers)

        if self._verbose:
            print(f"\n[split_loader] {n} layers  |  "
                  f"Metal limit {metal_limit/1024**3:.1f} GB  |  "
                  f"target {target_bytes/1024**3:.1f} GB "
                  f"({self._target_fraction*100:.0f}%)")

        # Profile layer sizes
        sizes = [_layer_weight_bytes(layers[i]) for i in range(n)]
        sum(sizes)

        info = SplitInfo(metal_limit=metal_limit, target_bytes=target_bytes)

        cumulative = 0
        for i, sz in enumerate(sizes):
            if i in self._force_cpu:
                decision = "cpu (forced)"
                on_cpu = True
            elif cumulative + sz <= target_bytes:
                decision = "gpu"
                on_cpu   = False
            else:
                decision = "cpu (budget)"
                on_cpu   = True

            if self._verbose:
                print(f"  layer {i:3d}  {sz/1024**2:6.1f} MB  →  {decision}")

            if on_cpu:
                # Save original reference, flatten params to numpy
                self._original_layers[i] = layers[i]
                flat = _flatten_params(layers[i])
                self._cpu_param_store[i] = flat
                # Replace layer slot with offloaded wrapper
                layers[i] = OffloadedLayer(layers[i], flat)
                info.cpu_layers.append(i)
                info.cpu_bytes += sz
            else:
                info.gpu_layers.append(i)
                info.gpu_bytes += cumulative + sz - cumulative  # = sz
                cumulative     += sz

        if self._verbose:
            print(f"\n[split_loader] {info}")

        self._info = info
        return info

    def restore(self) -> None:
        """
        Undo apply() — reload all CPU weights back to GPU, remove wrappers.
        After calling restore(), model.layers is identical to before apply().
        """
        layers = self._model.layers
        for i, original in self._original_layers.items():
            flat = self._cpu_param_store.get(i, [])
            if flat:
                _restore_params(original, flat)
            layers[i] = original

        self._original_layers.clear()
        self._cpu_param_store.clear()
        self._info = None

    @property
    def info(self) -> SplitInfo | None:
        return self._info

    # ---- class-level convenience -----------------------------------------

    @classmethod
    def auto_split(
        cls,
        model,
        target_fraction: float = 0.90,
        verbose: bool = True,
    ) -> SplitInfo | None:
        """
        Apply CPU/GPU split only if needed (total weight > target budget).

        Returns ``SplitInfo`` if splitting was applied, ``None`` if the model
        fit entirely within the Metal budget.

        Example::

            info = SplitLayerLoader.auto_split(model)
            if info:
                print(f"Offloaded {info.cpu_count} layers to CPU")
        """
        metal_limit  = _get_metal_limit_bytes()
        target_bytes = int(metal_limit * target_fraction)

        layers = getattr(model, "layers", [])
        total  = sum(_layer_weight_bytes(lyr) for lyr in layers)

        if total <= target_bytes:
            if verbose:
                print(f"[split_loader] Model fits in Metal budget "
                      f"({total/1024**3:.2f} GB <= {target_bytes/1024**3:.2f} GB) "
                      f"— no split needed")
            return None

        loader = cls(model, target_fraction=target_fraction, verbose=verbose)
        return loader.apply()


# ---------------------------------------------------------------------------
# Standalone profiling utility
# ---------------------------------------------------------------------------

def profile_model_layers(model) -> list[dict]:
    """
    Return per-layer memory statistics without modifying the model.

    Returns a list of dicts:
        [{"index": i, "bytes": N, "mb": f, "name": "..."}, ...]
    """
    layers = getattr(model, "layers", [])
    results = []
    for i, layer in enumerate(layers):
        nb  = _layer_weight_bytes(layer)
        results.append({
            "index":   i,
            "bytes":   nb,
            "mb":      nb / 1024 ** 2,
            "name":    type(layer).__name__,
        })
    return results


def print_layer_profile(model) -> None:
    """Print a formatted layer-by-layer memory table."""
    rows   = profile_model_layers(model)
    total  = sum(r["bytes"] for r in rows)
    limit  = _get_metal_limit_bytes()

    print(f"\nLayer memory profile ({len(rows)} layers):")
    print(f"  {'idx':>4}  {'MB':>8}  {'cumul GB':>10}  type")
    print("  " + "-" * 50)

    cumul = 0
    for r in rows:
        cumul += r["bytes"]
        print(f"  {r['index']:4d}  {r['mb']:8.1f}  "
              f"{cumul/1024**3:10.3f}  {r['name']}")

    print(f"\n  Total: {total/1024**3:.3f} GB  |  "
          f"Metal limit: {limit/1024**3:.1f} GB  |  "
          f"{'FITS' if total <= limit else 'EXCEEDS BUDGET'}")
