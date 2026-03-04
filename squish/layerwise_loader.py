#!/usr/bin/env python3
"""
squish/layerwise_loader.py

AirLLM-style layer-streaming inference for 70B+ models on Apple Silicon.

Standard MLX inference loads the entire model into Metal memory at startup.
For a 70B model in bfloat16 that is ~140 GB — far beyond any M-series Mac's
unified memory.  Layer-streaming solves this by keeping **only a handful of
layers in Metal memory at once**, loading the rest from disk on demand.

Architecture
────────────
  Disk  →  LayerCache (LRU, N layers in Metal)  →  Forward pass

Each transformer layer is saved as a *shard directory*::

    squish_layered/
        layer_000/
            weight.npy           (float16 numpy)
            layer_meta.json      {"index": 0, "bytes": 52428800, "class": "Qwen2DecoderLayer"}
        layer_001/ ...
        model_meta.json          {"n_layers": 80, "format": "squish_layered_v1"}

At inference time ``LayerwiseLoader.forward(hidden, layer_idx)`` either:
  * Finds the layer in ``LayerCache`` → free Metal hit
  * Loads from disk → numpy→MLX, evicts LRU layer from Metal if cache is full

Background prefetch
────────────────────
A ``prefetch_thread`` runs ``LayerwiseLoader.prefetch(layer_idx + 1)`` while
layer ``layer_idx`` is being executed, so disk I/O overlaps compute.

Typical usage
─────────────
    from squish.layerwise_loader import shard_model, LayerwiseLoader

    # One-time prep (saves to disk):
    shard_model(model, "/path/to/squish_layered/", verbose=True)

    # Inference via streaming:
    loader = LayerwiseLoader("/path/to/squish_layered/", cache_size=8)
    for layer_idx in range(loader.n_layers):
        hidden = loader.forward(hidden, layer_idx)

Performance guidance
─────────────────────
    cache_size=2   → minimal Metal use, ~100% NVMe time
    cache_size=8   → good balance for 70B on 32 GB Mac
    cache_size=20  → near-zero disk overhead for 13B on 32 GB Mac

Notes
─────
* Sharding is format-independent.  Weights are stored as float16 numpy to
  minimise disk size; on load they are cast to bfloat16 in Metal.
* For very large models the shard step itself may require the model weights to
  be loaded layer-by-layer (via mlx-lm's lazy-eval) — this module handles the
  case where ``model.layers`` is a Python list of already-materialised layers.
* ``LayerCache`` is thread-safe for multi-request servers.
"""

from __future__ import annotations

import json
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Lazy MLX import
# ---------------------------------------------------------------------------

def _mx():
    import mlx.core as mx
    return mx


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_FORMAT_VERSION = "squish_layered_v1"
_MODEL_META_FILE = "model_meta.json"
_LAYER_META_FILE = "layer_meta.json"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _layer_dir(shard_root: Path, layer_idx: int) -> Path:
    return shard_root / f"layer_{layer_idx:03d}"


def _flatten_params(layer) -> list[tuple[str, np.ndarray]]:
    """
    Recursively walk layer.parameters() (nested dict) and extract all arrays.

    Returns a list of (dotted_key_path, float16_numpy_array) pairs, e.g.::

        [("self_attn.q_proj.weight", array(..., dtype=float16)), ...]
    """
    def _walk(d: dict, prefix: str) -> list[tuple[str, np.ndarray]]:
        pairs = []
        for k, v in d.items():
            full_key = f"{prefix}.{k}" if prefix else k
            if isinstance(v, dict):
                pairs.extend(_walk(v, full_key))
            else:
                try:
                    arr = np.array(v, dtype=np.float16)
                    pairs.append((full_key, arr))
                except Exception:
                    pass
        return pairs
    return _walk(layer.parameters(), "")


def _restore_params(layer, flat_params: list[tuple[str, np.ndarray]]) -> None:
    """
    Reload numpy arrays back into a layer as bfloat16 MLX arrays.

    Uses ``layer.load_weights([(name, mx_array)], strict=False)``.
    """
    mx = _mx()
    mlx_pairs = [(name, mx.array(arr).astype(mx.bfloat16)) for name, arr in flat_params]
    layer.load_weights(mlx_pairs, strict=False)


def _layer_bytes(layer) -> int:
    """Return total parameter bytes for ``layer`` (MLX arrays)."""
    total = 0
    def _walk(d: dict) -> None:
        nonlocal total
        for v in d.values():
            if isinstance(v, dict):
                _walk(v)
            else:
                try:
                    total += v.nbytes
                except AttributeError:
                    pass
    _walk(layer.parameters())
    return total


# ---------------------------------------------------------------------------
# LayerCache — thread-safe LRU cache backed by Metal memory
# ---------------------------------------------------------------------------

class LayerCache:
    """
    LRU cache of at most ``capacity`` transformer layers in Metal memory.

    Each entry is a *loaded* MLX ``nn.Module`` whose weights are allocated on
    the Metal device.  When capacity is exceeded the least-recently-used layer
    is evicted (weights zeroed, Metal cache flushed).

    Thread-safe via a single reentrant lock.
    """

    def __init__(self, capacity: int = 8) -> None:
        if capacity < 1:
            raise ValueError(f"capacity must be ≥ 1, got {capacity}")
        self.capacity = capacity
        self._cache: OrderedDict[int, Any] = OrderedDict()
        self._lock = threading.RLock()

    def get(self, layer_idx: int) -> Any | None:
        with self._lock:
            if layer_idx in self._cache:
                self._cache.move_to_end(layer_idx)
                return self._cache[layer_idx]
        return None

    def put(self, layer_idx: int, layer) -> list[int]:
        """
        Insert ``layer`` at ``layer_idx``.

        Returns the list of evicted layer indices.
        """
        evicted = []
        with self._lock:
            if layer_idx in self._cache:
                self._cache.move_to_end(layer_idx)
                return evicted
            self._cache[layer_idx] = layer
            while len(self._cache) > self.capacity:
                old_idx, old_layer = self._cache.popitem(last=False)
                _zero_layer_weights(old_layer)
                evicted.append(old_idx)
        return evicted

    def __contains__(self, layer_idx: int) -> bool:
        with self._lock:
            return layer_idx in self._cache

    def __len__(self) -> int:
        with self._lock:
            return len(self._cache)

    @property
    def cached_indices(self) -> list[int]:
        with self._lock:
            return list(self._cache.keys())


def _zero_layer_weights(layer) -> None:
    """Zero out layer weights and release Metal memory."""
    try:
        mx = _mx()
        params = layer.parameters()
        def _walk(d: dict, prefix: str = "") -> list[tuple[str, Any]]:
            out = []
            for k, v in d.items():
                fp = f"{prefix}.{k}" if prefix else k
                if isinstance(v, dict):
                    out.extend(_walk(v, fp))
                else:
                    out.append((fp, v))
            return out
        flat = _walk(params)
        zeroed = [(name, mx.zeros_like(arr)) for name, arr in flat]
        layer.load_weights(zeroed, strict=False)
        mx.eval(dict(zeroed))
        try:
            mx.clear_cache()
        except AttributeError:
            try:
                mx.metal.clear_cache()
            except AttributeError:
                pass
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Shard I/O
# ---------------------------------------------------------------------------

def shard_model(
    model,
    output_dir: str | Path,
    compress: bool = False,
    verbose: bool = True,
) -> Path:
    """
    Serialize each transformer layer to its own directory under ``output_dir``.

    Format::

        output_dir/
            model_meta.json
            layer_000/
                layer_meta.json
                <param_name>.npy   e.g. self_attn.q_proj.weight.npy
            layer_001/ ...

    Parameters
    ----------
    model
        Loaded MLX model with a ``model.layers`` attribute.
    output_dir
        Destination directory.  Created if it does not exist.
    compress
        If True, use ``np.savez_compressed`` instead of ``np.save``.
        Roughly 2–3× savings for bfloat16 weights; slower to load.
    verbose
        Print progress.

    Returns
    -------
    Path
        Absolute path to ``output_dir``.
    """
    out = Path(output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    layers = getattr(model, "layers", [])
    if not layers:
        raise ValueError("model.layers is empty — nothing to shard")

    n_layers = len(layers)

    if verbose:
        print(f"[layerwise] Sharding {n_layers} layers → {out}")

    total_bytes = 0
    t0 = time.perf_counter()

    for idx, layer in enumerate(layers):
        layer_path = _layer_dir(out, idx)
        layer_path.mkdir(exist_ok=True)

        flat = _flatten_params(layer)
        bytes_this = sum(a.nbytes for _, a in flat)
        total_bytes += bytes_this

        for name, arr in flat:
            # Use dots in key name but safe filename: replace '.' with '__'
            safe_name = name.replace(".", "__")
            fpath = layer_path / f"{safe_name}.npy"
            if compress:
                np.savez_compressed(fpath, data=arr)
            else:
                np.save(fpath, arr)

        meta = {
            "index":    idx,
            "bytes":    bytes_this,
            "n_params": len(flat),
            "class":    type(layer).__name__,
            "format":   _FORMAT_VERSION,
        }
        (layer_path / _LAYER_META_FILE).write_text(json.dumps(meta, indent=2))

        if verbose and (idx % 5 == 0 or idx == n_layers - 1):
            elapsed = time.perf_counter() - t0
            print(
                f"  layer {idx:3d}/{n_layers-1}  "
                f"{bytes_this / 1024**2:6.1f} MB  "
                f"({elapsed:.1f}s elapsed)"
            )

    model_meta = {
        "n_layers":    n_layers,
        "total_bytes": total_bytes,
        "format":      _FORMAT_VERSION,
        "compressed":  compress,
    }
    (out / _MODEL_META_FILE).write_text(json.dumps(model_meta, indent=2))

    elapsed = time.perf_counter() - t0
    if verbose:
        print(
            f"[layerwise] Done — {n_layers} layers, "
            f"{total_bytes / 1024**3:.2f} GB in {elapsed:.1f}s → {out}"
        )

    return out


def _load_layer_from_shard(shard_root: Path, layer_idx: int, template_layer) -> Any:
    """
    Load a single layer from its shard directory.

    Clones ``template_layer``'s weights from disk into a fresh copy.
    Since we cannot deep-copy MLX modules portably, we reload weights
    into the *same* template object (caller is responsible for thread locking).

    Returns a dict mapping param_name → np.ndarray (float16).
    This dict is ready to be passed to ``_restore_params``.
    """
    layer_path = _layer_dir(shard_root, layer_idx)
    if not layer_path.exists():
        raise FileNotFoundError(f"Layer shard not found: {layer_path}")

    flat: list[tuple[str, np.ndarray]] = []
    for npy_file in sorted(layer_path.glob("*.npy")):
        safe_name = npy_file.stem
        param_name = safe_name.replace("__", ".")
        arr = np.load(npy_file).astype(np.float16)
        flat.append((param_name, arr))

    return flat


# ---------------------------------------------------------------------------
# LayerwiseLoader
# ---------------------------------------------------------------------------

@dataclass
class LoadStats:
    """Runtime statistics for a LayerwiseLoader session."""
    cache_hits:   int = 0
    cache_misses: int = 0
    total_loaded_bytes: int = 0
    total_load_time_s:  float = 0.0
    prefetch_hits: int = 0

    @property
    def hit_rate(self) -> float:
        total = self.cache_hits + self.cache_misses
        return self.cache_hits / total if total else 0.0

    def __str__(self) -> str:
        return (
            f"LoadStats(hits={self.cache_hits}, misses={self.cache_misses}, "
            f"hit_rate={self.hit_rate:.1%}, "
            f"loaded={self.total_loaded_bytes/1024**3:.2f}GB, "
            f"load_time={self.total_load_time_s:.1f}s)"
        )


class LayerwiseLoader:
    """
    Stream transformer layers from a squish_layered shard directory.

    Parameters
    ----------
    shard_root
        Directory produced by ``shard_model()``.
    template_model
        A loaded (possibly tiny) model used as the layer template.
        Weights from disk are loaded into copies of its layers.
    cache_size
        Number of layers to keep in Metal memory simultaneously.
    prefetch
        If True, start loading layer N+1 in a background thread
        while layer N is being computed.
    """

    def __init__(
        self,
        shard_root: str | Path,
        template_model=None,
        cache_size: int = 8,
        prefetch: bool = True,
    ) -> None:
        self._root = Path(shard_root).expanduser().resolve()
        if not self._root.exists():
            raise FileNotFoundError(f"Shard directory not found: {self._root}")

        meta_path = self._root / _MODEL_META_FILE
        if not meta_path.exists():
            raise FileNotFoundError(f"model_meta.json not found in {self._root}")
        self._meta = json.loads(meta_path.read_text())
        self._n_layers = self._meta["n_layers"]

        self._template_model = template_model
        self._cache = LayerCache(capacity=cache_size)
        self._prefetch_enabled = prefetch
        self._prefetch_future: dict[int, threading.Thread] = {}
        self._prefetch_lock = threading.Lock()
        self._stats = LoadStats()

    @property
    def n_layers(self) -> int:
        return self._n_layers

    @property
    def stats(self) -> LoadStats:
        return self._stats

    def _template_layer(self, layer_idx: int):
        """Return the template layer object for ``layer_idx``."""
        if self._template_model is None:
            raise RuntimeError(
                "template_model is required to load shards — "
                "pass the (CPU-resident) base model to LayerwiseLoader()"
            )
        return self._template_model.layers[layer_idx]

    def _load_layer(self, layer_idx: int):
        """Load ``layer_idx`` from disk, inject into template, return layer."""
        t0 = time.perf_counter()
        flat = _load_layer_from_shard(self._root, layer_idx, None)
        layer = self._template_layer(layer_idx)
        _restore_params(layer, flat)
        mx = _mx()
        mx.eval(layer.parameters())
        dt = time.perf_counter() - t0
        self._stats.total_load_time_s += dt
        self._stats.total_loaded_bytes += sum(a.nbytes for _, a in flat)
        return layer

    def _ensure_loaded(self, layer_idx: int):
        """Ensure layer is in cache; load from disk if needed."""
        layer = self._cache.get(layer_idx)
        if layer is not None:
            self._stats.cache_hits += 1
            return layer

        self._stats.cache_misses += 1

        # Wait for prefetch thread if one is running
        with self._prefetch_lock:
            t = self._prefetch_future.pop(layer_idx, None)
        if t is not None:
            t.join()
            layer = self._cache.get(layer_idx)
            if layer is not None:
                self._stats.prefetch_hits += 1
                return layer

        layer = self._load_layer(layer_idx)
        self._cache.put(layer_idx, layer)
        return layer

    def _start_prefetch(self, layer_idx: int) -> None:
        """Start a background thread to prefetch ``layer_idx``."""
        if not self._prefetch_enabled:
            return
        if layer_idx >= self._n_layers or layer_idx < 0:
            return
        if layer_idx in self._cache:
            return
        with self._prefetch_lock:
            if layer_idx in self._prefetch_future:
                return
            t = threading.Thread(
                target=self._ensure_loaded,
                args=(layer_idx,),
                daemon=True,
                name=f"squish-prefetch-{layer_idx}",
            )
            self._prefetch_future[layer_idx] = t
            t.start()

    def forward(self, hidden_states, layer_idx: int, **kwargs):
        """
        Run ``layer_idx`` on ``hidden_states``.

        Loads the layer from cache (or disk) and immediately starts
        prefetching the next layer.

        Parameters
        ----------
        hidden_states
            MLX array of shape (batch, seq_len, hidden_dim).
        layer_idx
            Which transformer layer to execute (0-indexed).
        **kwargs
            Forwarded to the layer's ``__call__``.

        Returns
        -------
        MLX array
            Output hidden states.
        """
        layer = self._ensure_loaded(layer_idx)
        # Prefetch next layer while this one runs
        self._start_prefetch(layer_idx + 1)
        return layer(hidden_states, **kwargs)

    def warmup(self, n_layers: int | None = None) -> None:
        """
        Pre-load the first ``n_layers`` layers into the cache.

        Useful to fill the cache before starting inference to avoid
        cold-start latency on the first few tokens.
        """
        n = min(n_layers or self._cache.capacity, self._n_layers)
        print(f"[layerwise] Warming up {n} layers ...")
        for i in range(n):
            self._ensure_loaded(i)
        print(f"[layerwise] Warmup complete — {len(self._cache)} layers in Metal cache")

    def clear_cache(self) -> None:
        """Evict all layers from Metal cache."""
        with self._cache._lock:
            for layer in self._cache._cache.values():
                _zero_layer_weights(layer)
            self._cache._cache.clear()

    def print_stats(self) -> None:
        """Print a human-readable stats summary."""
        s = self._stats
        print(
            f"[layerwise] Cache stats:\n"
            f"  hits={s.cache_hits}  misses={s.cache_misses}  "
            f"hit_rate={s.hit_rate:.1%}\n"
            f"  prefetch_hits={s.prefetch_hits}\n"
            f"  loaded={s.total_loaded_bytes/1024**3:.3f} GB  "
            f"load_time={s.total_load_time_s:.1f}s"
        )


# ---------------------------------------------------------------------------
# Convenience: auto_chunk_70b
# ---------------------------------------------------------------------------

def recommend_cache_size(
    total_model_gb: float,
    n_layers: int,
    available_metal_gb: float,
    safety_factor: float = 0.80,
) -> int:
    """
    Recommend a ``cache_size`` for ``LayerwiseLoader`` given hardware limits.

    Parameters
    ----------
    total_model_gb
        Total model size in GB (e.g. 140.0 for 70B bfloat16).
    n_layers
        Number of transformer layers (e.g. 80 for Llama-70B).
    available_metal_gb
        Available Metal memory in GB.
    safety_factor
        Fraction of available memory to use.  0.80 = leave 20% overhead.

    Returns
    -------
    int
        Recommended cache_size (minimum 2, maximum n_layers).
    """
    per_layer_gb = total_model_gb / n_layers
    usable_gb = available_metal_gb * safety_factor
    cache_size = max(2, min(n_layers, int(usable_gb / per_layer_gb)))
    return cache_size
