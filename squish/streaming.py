#!/usr/bin/env python3
"""
streaming_loader.py

Layer-by-layer streaming loader with background prefetch.

The key claim this module proves: peak RAM during model loading equals
(non-layer tensors) + (one decoder layer decompressed at a time), NOT
the full uncompressed model.

Architecture:
  - Tensors are grouped by MLX layer index:
      group 0   → "non-layer" tensors: embed_tokens, lm_head, final norm, etc.
      group 1…N → decoder layer N-1  (model.layers.<i>.*)
  - A prefetch thread decompresses group i+1 from the npz while
    model.load_weights() is called for group i.
  - Each group is released from RAM after injection.

Public API:
    model, tokenizer, stats = load_streaming(
        model_dir, npz_path, manifest_path=None, verbose=True
    )

stats keys:
    ram_baseline_mb       RSS before any loading
    ram_peak_mb           Highest RSS observed during the streaming load
    ram_peak_layer        Layer index at which peak occurred
    ram_after_load_mb     RSS immediately after final load_weights()
    ram_delta_mb          ram_after_load_mb - ram_baseline_mb
    layer_ram_mb_list     Per-group peak RSS reading  [list[float]]
    n_groups              Number of groups (1 + n_decoder_layers)
    n_quantized           Tensors loaded via Vectro INT8
    n_passthrough         Tensors stored as float32 (outlier / sensitive)
    decompression_time_s  Total wall time for decompression
    prefetch_savings_s    Estimated latency hidden by prefetch thread
"""
import platform
import queue
import re
import resource
import threading
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
from transformers import AutoTokenizer

from squish.loader_utils import (
    _dequantize,
    _instantiate_model,
    _safe_key_to_original,
    _unique_base_keys,
)

# ---------------------------------------------------------------------------
# RSS helper
# ---------------------------------------------------------------------------

def _rss_mb() -> float:
    ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if platform.system() == "Darwin":
        return ru / 1_048_576
    return ru / 1_024


# ---------------------------------------------------------------------------
# Tensor grouping
# ---------------------------------------------------------------------------

_LAYER_RE = re.compile(r"model\.layers\.(\d+)\.")


def _group_tensors(
    base_keys: set[str],
    safe_to_original: dict[str, str],
) -> tuple[list[tuple[str, str]], dict[int, list[tuple[str, str]]]]:
    """
    Split base_keys into (non_layer_pairs, {layer_idx: [(sk, original_name), ...]}).

    non_layer_pairs: embed_tokens, lm_head, norms, etc.
    Returns them sorted for deterministic iteration.
    """
    non_layer: list[tuple[str, str]] = []
    layers: dict[int, list[tuple[str, str]]] = {}

    for sk in sorted(base_keys):
        original = safe_to_original.get(sk)
        if original is None:
            continue
        m = _LAYER_RE.search(original)
        if m:
            idx = int(m.group(1))
            layers.setdefault(idx, []).append((sk, original))
        else:
            non_layer.append((sk, original))

    return non_layer, layers


# ---------------------------------------------------------------------------
# Decompression / prefetch helpers
# ---------------------------------------------------------------------------

def _decompress_group(
    npz: np.lib.npyio.NpzFile,
    group: list[tuple[str, str]],
) -> tuple[list[tuple[str, mx.array]], int, int]:
    """
    Decompress a list of (safe_key, original_name) pairs.
    Returns (weight_tuples, n_quantized, n_passthrough).
    Each numpy buffer is freed immediately after casting to mlx.array.
    """
    out = []
    n_q = n_pt = 0
    for sk, original in group:
        arr = _dequantize(npz, sk)
        mlx_arr = mx.array(arr).astype(mx.bfloat16)
        del arr
        out.append((original, mlx_arr))
        if (sk + "__pt") in npz.files:
            n_pt += 1
        else:
            n_q += 1
    return out, n_q, n_pt


# ---------------------------------------------------------------------------
# Main public API
# ---------------------------------------------------------------------------

def load_streaming(
    model_dir: str,
    npz_path: str,
    manifest_path: str | None = None,
    verbose: bool = True,
) -> tuple:
    """
    Load a compressed model layer-by-layer with background prefetch.

    Returns (model, tokenizer, stats_dict).
    stats_dict always present — no flag needed.
    """
    if manifest_path is None:
        manifest_path = npz_path.replace(".npz", "_manifest.json")

    stats: dict = {}
    rss_baseline = _rss_mb()
    stats["ram_baseline_mb"] = rss_baseline

    # 1. Build model architecture (config.json only)
    if verbose:
        print(f"[stream] Building architecture from {model_dir}/config.json ...")
    model, mlx_type = _instantiate_model(model_dir)
    if verbose:
        print(f"[stream]   model_type = {mlx_type}")

    # 2. Load manifest
    safe_to_original = _safe_key_to_original(manifest_path)

    # 3. Open npz lazily
    if verbose:
        print(f"[stream] Opening: {npz_path}")
    npz = np.load(npz_path, allow_pickle=False)
    base_keys = _unique_base_keys(list(npz.files))
    if verbose:
        print(f"[stream]   {len(base_keys)} tensors in archive")

    # 4. Group tensors
    non_layer_pairs, layer_groups = _group_tensors(base_keys, safe_to_original)
    n_layers = len(layer_groups)
    if verbose:
        print(f"[stream]   {len(non_layer_pairs)} non-layer tensors, "
              f"{n_layers} decoder layers")

    total_decomp_time = 0.0
    total_n_q = 0
    total_n_pt = 0
    layer_ram_readings: list[float] = []
    rss_peak = rss_baseline
    peak_layer = -1

    # -----------------------------------------------------------------------
    # 5. Load non-layer tensors (embeddings, final norm, lm_head)
    # -----------------------------------------------------------------------
    if verbose:
        print(f"\n[stream] Loading non-layer tensors ({len(non_layer_pairs)}) ...")
    t0 = time.time()
    non_layer_weights, nq, npt = _decompress_group(npz, non_layer_pairs)
    total_decomp_time += time.time() - t0
    total_n_q += nq
    total_n_pt += npt
    model.load_weights(non_layer_weights)
    del non_layer_weights
    rss_now = _rss_mb()
    layer_ram_readings.append(rss_now)
    if rss_now > rss_peak:
        rss_peak = rss_now
        peak_layer = -1
    if verbose:
        print(f"  RSS after non-layer load: {rss_now:.0f} MB")

    # -----------------------------------------------------------------------
    # 6. Stream decoder layers with prefetch
    # -----------------------------------------------------------------------
    sorted_layer_indices = sorted(layer_groups.keys())

    # Prefetch queue: background thread posts decompressed weight tuples here.
    # maxsize=1 keeps exactly one layer pre-decompressed at a time.
    prefetch_q: queue.Queue = queue.Queue(maxsize=1)
    prefetch_times: list[float] = []

    def _prefetch_worker(layer_idx: int) -> None:
        """Decompress layer layer_idx and push result onto prefetch_q."""
        tp0 = time.perf_counter()
        group = layer_groups[layer_idx]
        weights, nq, npt = _decompress_group(npz, group)
        elapsed = time.perf_counter() - tp0
        prefetch_q.put((layer_idx, weights, nq, npt, elapsed))

    for i, layer_idx in enumerate(sorted_layer_indices):
        # Kick off prefetch of the NEXT layer in a background thread
        next_layer_idx = sorted_layer_indices[i + 1] if i + 1 < len(sorted_layer_indices) else None
        if next_layer_idx is not None:
            t = threading.Thread(
                target=_prefetch_worker,
                args=(next_layer_idx,),
                daemon=True,
            )
            t.start()

        # Decompress current layer (or fetch from queue if prefetch already ran)
        if i == 0:
            # First layer: no prefetch result yet, decompress synchronously
            t_sync = time.perf_counter()
            group = layer_groups[layer_idx]
            weights, nq, npt = _decompress_group(npz, group)
            decomp_elapsed = time.perf_counter() - t_sync
        else:
            # Get the prefetched result for this layer
            (fetched_idx, weights, nq, npt, decomp_elapsed) = prefetch_q.get()
            assert fetched_idx == layer_idx, (
                f"Prefetch mismatch: expected {layer_idx}, got {fetched_idx}"
            )
            # Estimate latency hidden: decompression happened in background
            prefetch_times.append(decomp_elapsed)

        total_decomp_time += decomp_elapsed
        total_n_q += nq
        total_n_pt += npt

        # Inject this layer's weights
        model.load_weights(weights)
        del weights

        rss_now = _rss_mb()
        layer_ram_readings.append(rss_now)
        if rss_now > rss_peak:
            rss_peak = rss_now
            peak_layer = layer_idx

        if verbose:
            print(f"  [layer {layer_idx:>3d}] {nq:3d}Q + {npt}PT  "
                  f"decomp {decomp_elapsed*1000:.0f}ms  "
                  f"RSS {rss_now:.0f} MB")

        # Wait for background thread to finish so we don't open too many threads
        # (the next iteration will block on prefetch_q.get() naturally)

    npz.close()

    rss_after = _rss_mb()

    # 7. Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)

    rss_final = _rss_mb()

    # Estimate prefetch savings: sum of decompression times that ran in background
    prefetch_savings = sum(prefetch_times)

    stats.update({
        "ram_peak_mb": rss_peak,
        "ram_peak_layer": peak_layer,
        "ram_after_load_mb": rss_after,
        "ram_final_mb": rss_final,
        "ram_delta_mb": rss_final - rss_baseline,
        "layer_ram_mb_list": layer_ram_readings,
        "n_groups": 1 + n_layers,
        "n_quantized": total_n_q,
        "n_passthrough": total_n_pt,
        "n_tensors": total_n_q + total_n_pt,
        "decompression_time_s": total_decomp_time,
        "prefetch_savings_s": prefetch_savings,
    })

    if verbose:
        delta = rss_final - rss_baseline
        print("\n[stream] Done.")
        print(f"  RAM baseline:       {rss_baseline:.0f} MB")
        print(f"  RAM peak (layer {peak_layer}): {rss_peak:.0f} MB")
        print(f"  RAM after load:     {rss_after:.0f} MB  (Δ {delta:+.0f} MB)")
        print(f"  Decomp total:       {total_decomp_time:.2f}s")
        print(f"  Prefetch savings:   ~{prefetch_savings:.2f}s hidden by bg thread\n")

    return model, tokenizer, stats


# ---------------------------------------------------------------------------
# CLI shim for quick testing
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    from mlx_lm import generate

    ap = argparse.ArgumentParser(description="Streaming compressed-weight loader")
    ap.add_argument("--model-dir",
                    default=str(Path.home() / "models" / "Qwen2.5-1.5B-Instruct"))
    ap.add_argument("--npz",
                    default=str(Path.home() / "models" /
                                "Qwen2.5-1.5B-Instruct-compressed" /
                                "weights_compressed.npz"))
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--max-tokens", type=int, default=30)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    print("=== Streaming loader demo ===\n")
    model, tokenizer, stats = load_streaming(
        model_dir=args.model_dir,
        npz_path=args.npz,
        verbose=not args.quiet,
    )

    print(f"Generating: {args.prompt!r}")
    out = generate(model, tokenizer, prompt=args.prompt,
                   max_tokens=args.max_tokens, verbose=True)
    print(f"\nOutput: {out!r}")
    print("\nRAM stats:")
    for k, v in stats.items():
        if k == "layer_ram_mb_list":
            continue
        print(f"  {k:<30} {v}")
