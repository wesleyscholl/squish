#!/usr/bin/env python3
"""
compressed_loader.py

Loads a Vectro-compressed model (weights_compressed.npz) into an MLX graph
without ever opening the original .safetensors files.

Public API (same contract as mlx_lm.load):
    model, tokenizer = load_compressed_model(model_dir, npz_path, ...)

How it works:
  1. Parse config.json to build the MLX model architecture (no weights yet).
  2. Open the npz lazily — arrays are not decompressed until indexed.
  3. For each tensor: read the int8+scale block, call Vectro reconstruct, cast
     to bfloat16 mlx.array.
  4. Inject all tensors via model.load_weights(list_of_tuples).
  5. Return (model, tokenizer) just like mlx_lm.load().

Peak RAM behaviour:
  - The npz zipfile index is held in RAM (~kilobytes for 700 keys).
  - Each tensor is decompressed → reconstructed → converted → loaded into MLX
    one at a time; the numpy buffer is released before the next tensor.
  - The full uncompressed model is never simultaneously in RAM.
"""
import sys
import json
import time
import dataclasses
import importlib
import os
import resource
import threading
import queue
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# Optional Rust squish_quant extension — enables INT4 load path
# ---------------------------------------------------------------------------
_squish_quant = None
try:
    import squish_quant as _squish_quant
except ImportError:
    pass

_INT4_READY      = ".squish_int4_ready"   # sentinel: INT4 dir is complete
_INT4_GROUP_SIZE = 64                     # nibble-pack group size (must match save)

# ---------------------------------------------------------------------------
# Optional zstd decompression — transparent .npy.zst reading
# ---------------------------------------------------------------------------
_zstd_dctx = None   # lazily initialised on first use


def _get_zstd_dctx():
    """Lazily create a shared ZstdDecompressor.  Returns None if zstandard not installed."""
    global _zstd_dctx
    if _zstd_dctx is None:
        try:
            import zstandard as _zstd
            _zstd_dctx = _zstd.ZstdDecompressor()
        except ImportError:
            _zstd_dctx = False       # sentinel: unavailable
    return _zstd_dctx if _zstd_dctx is not False else None


def _load_npy_path(path: Path, mmap_mode: str | None = "r") -> np.ndarray:
    """
    Load a ``.npy`` or ``.npy.zst`` file.

    * If ``path`` exists as-is → ``np.load(path, mmap_mode=mmap_mode)``
    * If ``path`` does not exist but ``path + '.zst'`` does → decompress via
      zstandard into a BytesIO buffer and call ``np.load`` on the buffer.
      mmap is not supported for compressed files (decompression is streaming).

    Raises ``FileNotFoundError`` if neither variant exists.
    Raises ``RuntimeError`` if the .zst file exists but zstandard is not installed.
    """
    import io
    if path.exists():
        return np.load(str(path), mmap_mode=mmap_mode)
    zst_path = Path(str(path) + ".zst")
    if zst_path.exists():
        dctx = _get_zstd_dctx()
        if dctx is None:
            raise RuntimeError(
                f"Found {zst_path} but 'zstandard' is not installed. "
                "Run: pip install zstandard"
            )
        with open(zst_path, "rb") as f:
            buf = io.BytesIO(dctx.decompress(f.read()))
        return np.load(buf, allow_pickle=False)
    raise FileNotFoundError(f"Neither {path} nor {zst_path} found")

# ---------------------------------------------------------------------------
# RAM watermark helpers (macOS: ru_maxrss is bytes; Linux: kilobytes)
# ---------------------------------------------------------------------------
import platform as _platform

def _rss_mb() -> float:
    """Current process RSS in megabytes."""
    ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # macOS returns bytes; Linux returns kilobytes
    if _platform.system() == "Darwin":
        return ru / 1_048_576
    return ru / 1_024

import os as _os

# squish.quantizer is the self-contained replacement for vectro/python/interface.py
from squish.quantizer import reconstruct_embeddings, QuantizationResult

import mlx.core as mx
import mlx.nn as nn
from transformers import AutoTokenizer

# ---------------------------------------------------------------------------
# Phase 0.1 — Metal memory budget
# ---------------------------------------------------------------------------
# MLX defaults to 75 % of unified RAM for the Metal allocator.  On a 24 GB
# machine that leaves ~6 GB unused that could hold model weights or KV cache.
# Raising the ceiling to 90 % (configurable via SQUISH_METAL_FRACTION) unlocks
# roughly +2 GB on 16 GB or +3.6 GB on 24 GB machines — enough to hold the
# 7B model's KV cache at batch 4 or to accelerate the 14B model without OOM.
# The `relaxed=True` flag means MLX will still reuse allocations above the
# limit before raising an error, preventing spurious OOM on bursty batches.
def _configure_metal_memory() -> None:
    """Raise the MLX Metal allocator ceiling to SQUISH_METAL_FRACTION of total RAM."""
    try:
        fraction = float(os.environ.get("SQUISH_METAL_FRACTION", "0.90"))
        if not (0.5 <= fraction <= 0.99):
            return
        # macOS: hw.memsize sysctl gives total physical bytes
        import ctypes
        libc      = ctypes.CDLL("libSystem.dylib")
        memsize   = ctypes.c_uint64(0)
        size_ptr  = ctypes.c_size_t(ctypes.sizeof(memsize))
        ret = libc.sysctlbyname(b"hw.memsize",
                                ctypes.byref(memsize), ctypes.byref(size_ptr),
                                None, 0)
        if ret != 0:
            return
        limit = int(memsize.value * fraction)
        mx.metal.set_memory_limit(limit, relaxed=True)
    except Exception:
        pass   # non-fatal: non-Apple hardware or old MLX build

_configure_metal_memory()


# ---------------------------------------------------------------------------
# Model architecture instantiation
# ---------------------------------------------------------------------------

# Map HuggingFace model_type values to mlx_lm module names where they differ
_HF_TO_MLX_TYPE = {
    "qwen2": "qwen2",
    "mistral": "mistral",
    "llama": "llama",
    "phi3": "phi3",
    "phi": "phi",
    "gemma": "gemma",
    "gemma2": "gemma2",
    "starcoder2": "starcoder2",
    "cohere": "cohere",
    "falcon": "falcon",
    "mpt": "mpt",
    "gpt2": "gpt2",
    "gpt_neox": "gpt_neox",
    "olmo": "olmo",
    "openelm": "openelm",
}


def _build_model_args(ModelArgs, config: dict):
    """Construct ModelArgs from config, keeping only recognized fields."""
    if hasattr(ModelArgs, "from_dict"):
        return ModelArgs.from_dict(config)
    valid = {f.name for f in dataclasses.fields(ModelArgs)}
    return ModelArgs(**{k: v for k, v in config.items() if k in valid})


def _instantiate_model(model_dir: str):
    """
    Build the MLX model object from config.json alone — no weights loaded.
    Returns (model, mlx_model_type_str).
    """
    config_path = Path(model_dir) / "config.json"
    with open(config_path) as f:
        config = json.load(f)

    hf_type = config.get("model_type", "").lower()
    mlx_type = _HF_TO_MLX_TYPE.get(hf_type, hf_type)

    try:
        module = importlib.import_module(f"mlx_lm.models.{mlx_type}")
    except ImportError as e:
        raise ValueError(
            f"Cannot find mlx_lm.models.{mlx_type} for model_type={hf_type!r}. "
            f"Supported: {sorted(_HF_TO_MLX_TYPE.values())}"
        ) from e

    ModelArgs = module.ModelArgs
    Model = module.Model

    args = _build_model_args(ModelArgs, config)
    model = Model(args)
    return model, mlx_type


# ---------------------------------------------------------------------------
# Decompression helpers
# ---------------------------------------------------------------------------

def _safe_key_to_original(manifest_path: str) -> dict:
    """Load manifest and return inverted dict: safe_key -> original_name."""
    with open(manifest_path) as f:
        manifest = json.load(f)
    return {v: k for k, v in manifest.items()}


def _unique_base_keys(npz_files: list) -> set:
    """
    Given the list of array names inside a npz, extract the unique base keys
    (everything before the last __q / __s / __pt / __shape suffix).
    """
    suffixes = ("__q", "__s", "__pt", "__shape")
    base_keys = set()
    for fname in npz_files:
        for suf in suffixes:
            if fname.endswith(suf):
                base_keys.add(fname[: -len(suf)])
                break
    return base_keys


def _dequantize(npz, sk: str) -> np.ndarray:
    """
    Reconstruct one tensor from the npz.
    Returns a float32 numpy array with the original shape.
    """
    has_shape = (sk + "__shape") in npz.files

    if (sk + "__pt") in npz.files:
        # Passthrough: stored as float32 directly.
        # __shape may be absent if the archive was written by an older version of
        # convert_weights.py — in that case the pt array already has the right shape.
        if has_shape:
            original_shape = tuple(int(x) for x in npz[sk + "__shape"].tolist())
        else:
            original_shape = npz[sk + "__pt"].shape
        return npz[sk + "__pt"].reshape(original_shape)

    # Quantized path always requires __shape (needed to undo the 2D reshape).
    original_shape = tuple(int(x) for x in npz[sk + "__shape"].tolist())

    # Quantized path: int8 + float32 scales
    q = npz[sk + "__q"]   # shape (n_rows, n_cols)
    s = npz[sk + "__s"]   # shape (n_rows,)

    result = QuantizationResult(
        quantized=q,
        scales=s,
        dims=q.shape[1],
        n=q.shape[0],
    )
    arr_f32 = reconstruct_embeddings(result)   # (n_rows, n_cols)
    return arr_f32.reshape(original_shape)


# ---------------------------------------------------------------------------
# npy-dir: memory-mapped loader (no zlib, near-zero decomp overhead)
# ---------------------------------------------------------------------------

_FINALIZED_DIR    = "finalized"            # sub-dir: per-tensor float16 .npy files
_MLX_CACHE_FILE   = "squish_weights.safetensors"  # combined bf16 MLX safetensors (fastest)
_MLX_CACHE_READY  = ".squish_ready"         # sentinel alongside the safetensors file


def _save_finalized_cache(dir_path: Path, base_keys: list[str],
                          tensor_dir: Path, safe_to_original: dict,
                          verbose: bool = True) -> None:
    """
    Save the reconstructed float16 tensors to a fast-load finalized cache.

    After the first npy-dir load (which runs Vectro), the results are re-saved
    as simple float16 .npy files.  Subsequent loads read these directly via
    mmap — no Vectro reconstruction needed → ~3-4× faster load.

    Layout:  {dir_path}/finalized/{safe_key}.npy   (float16)
    Sentinel: {dir_path}/finalized/.ready           (written last)
    """
    finalized_dir = dir_path / _FINALIZED_DIR
    finalized_dir.mkdir(exist_ok=True)

    ready_flag = finalized_dir / ".ready"
    if ready_flag.exists():
        return   # already saved from a previous run

    if verbose:
        print(f"\n  Saving finalized f16 cache → {finalized_dir} ...")
    t0 = time.perf_counter()
    bytes_written = 0
    for sk in base_keys:
        original_name = safe_to_original.get(sk)
        if original_name is None:
            continue
        arr_f32 = _dequantize_npy_dir(tensor_dir, sk)
        arr_f16 = arr_f32.astype(np.float16)
        del arr_f32
        fkey = original_name.replace(".", "__")
        out_path = finalized_dir / f"{fkey}.npy"
        np.save(str(out_path), arr_f16)
        bytes_written += out_path.stat().st_size
    ready_flag.touch()   # mark cache as complete / valid
    elapsed = time.perf_counter() - t0
    if verbose:
        print(f"  Finalized cache saved in {elapsed:.1f}s  "
              f"({bytes_written / 1e6:.0f} MB)  → next load will skip Vectro")


def _load_finalized_cache(
    dir_path: Path,
    model_dir: str,
    verbose: bool = True,
    return_stats: bool = False,
):
    """
    Fast load from finalized f16 cache — no Vectro decompression.
    Returns same API as load_from_npy_dir.
    """
    finalized_dir = dir_path / _FINALIZED_DIR
    stats: dict = {}
    rss_baseline = _rss_mb()
    stats["ram_baseline_mb"] = rss_baseline

    if verbose:
        print(f"Loading finalized cache (f16, no Vectro) from {finalized_dir} ...")
    model, mlx_type = _instantiate_model(model_dir)
    if verbose:
        print(f"  model_type = {mlx_type}")

    npy_files = sorted(finalized_dir.glob("*.npy"))
    if verbose:
        print(f"  {len(npy_files)} cached f16 tensors (mmap)")

    t0 = time.perf_counter()
    rss_peak = rss_baseline
    weight_tuples = []

    for p in npy_files:
        # Reverse the safe_key encoding: __ → .
        original_name = p.stem.replace("__", ".")
        arr_f16 = np.load(str(p), mmap_mode='r')
        mlx_arr = mx.array(arr_f16).astype(mx.bfloat16)   # mmap → Metal, no CPU copy
        del arr_f16
        weight_tuples.append((original_name, mlx_arr))
        cur_rss = _rss_mb()
        if cur_rss > rss_peak:
            rss_peak = cur_rss

    decomp_time = time.perf_counter() - t0
    if verbose:
        print(f"  Loaded {len(weight_tuples)} tensors in {decomp_time:.2f}s")
        print("Calling model.load_weights() ...")

    model.load_weights(weight_tuples)
    del weight_tuples

    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    rss_final = _rss_mb()

    stats.update({
        "ram_peak_mb":          rss_peak,
        "ram_delta_mb":         rss_final - rss_baseline,
        "decompression_time_s": decomp_time,
        "decomp_workers":       1,
        "loader":               "finalized-f16",
    })
    if verbose:
        print(f"RAM Δ: {rss_final - rss_baseline:+.0f} MB  |  Model ready.\n")

    if return_stats:
        return model, tokenizer, stats
    return model, tokenizer


def _load_mlx_cache(
    dir_path: Path,
    model_dir: str,
    verbose: bool = True,
    return_stats: bool = False,
):
    """
    Fastest load path: read squish_weights.safetensors → mx.load() → load_weights().

    This bypasses numpy entirely.  mx.load() on a safetensors file maps the
    bytes directly to MLX arrays on the Metal device — near-identical speed
    to mlx_lm.load() loading the original safetensors.

    Expected load time: ≈ reference model load time (1.5-2.5s).
    """
    cache_path = dir_path / _MLX_CACHE_FILE
    stats: dict = {}
    rss_baseline = _rss_mb()
    stats["ram_baseline_mb"] = rss_baseline

    if verbose:
        sz_mb = cache_path.stat().st_size / 1e6
        print(f"Loading Squish MLX cache ({sz_mb:.0f} MB) from {cache_path} ...")
    model, mlx_type = _instantiate_model(model_dir)
    if verbose:
        print(f"  model_type = {mlx_type}")

    t0 = time.perf_counter()
    weights = mx.load(str(cache_path))          # dict: {name → mx.array}
    load_time = time.perf_counter() - t0

    # mx.load returns a flat dict; convert to list[tuple] for load_weights
    weight_list = list(weights.items())
    del weights

    rss_peak = _rss_mb()
    if verbose:
        print(f"  {len(weight_list)} tensors loaded in {load_time:.2f}s")
        print("Calling model.load_weights() ...")

    model.load_weights(weight_list)
    del weight_list

    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    rss_final = _rss_mb()

    stats.update({
        "ram_peak_mb":          rss_peak,
        "ram_delta_mb":         rss_final - rss_baseline,
        "decompression_time_s": load_time,
        "decomp_workers":       1,
        "loader":               "squish-mlx",
    })
    if verbose:
        print(f"RAM Δ: {rss_final - rss_baseline:+.0f} MB  |  Model ready.\n")

    if return_stats:
        return model, tokenizer, stats
    return model, tokenizer


def _npy_exists(path: Path) -> bool:
    """True if ``path`` or ``path + '.zst'`` exists (transparent zstd support)."""
    return path.exists() or Path(str(path) + ".zst").exists()


def _dequantize_npy_dir(tensor_dir: Path, sk: str) -> np.ndarray:
    """
    Reconstruct one tensor from a npy-dir format directory.
    Uses mmap_mode='r' so the OS avoids upfront reads; only the bytes
    actually touched when building the MLX array are paged in.

    Priority:
      1. INT4 packed (``__q4.npy`` + ``__s4.npy``)  — 50 % disk vs INT8, Rust deq
      2. INT8 quantized (``__q.npy`` + ``__s.npy``)  — Vectro / NumPy
      3. Passthrough float16 (``__pt.npy``)

    All paths support transparent zstandard decompression: if ``foo.npy`` is
    absent but ``foo.npy.zst`` exists it is decompressed on-the-fly via the
    module-level ``_zstd_dctx`` (installed with ``pip install zstandard``).
    """
    shape_path = tensor_dir / f"{sk}__shape.npy"

    # ── Tier 0: INT4 nibble-packed (requires squish_quant Rust extension) ────
    q4_path = tensor_dir / f"{sk}__q4.npy"
    s4_path = tensor_dir / f"{sk}__s4.npy"
    if _npy_exists(q4_path) and _npy_exists(s4_path) and _squish_quant is not None:
        packed = np.ascontiguousarray(
            _load_npy_path(q4_path), dtype=np.uint8)       # (n, d//2)
        scales = np.ascontiguousarray(
            _load_npy_path(s4_path), dtype=np.float32)    # (n, n_groups)
        original_shape = tuple(_load_npy_path(shape_path).tolist())
        arr_f32 = _squish_quant.dequantize_int4_grouped(
            packed, scales, _INT4_GROUP_SIZE,
        )
        return arr_f32.reshape(original_shape)

    pt_path = tensor_dir / f"{sk}__pt.npy"
    if _npy_exists(pt_path):
        arr = _load_npy_path(pt_path)                      # float16, possibly mmap'd
        if _npy_exists(shape_path):
            original_shape = tuple(_load_npy_path(shape_path).tolist())
        else:
            original_shape = arr.shape
        return np.array(arr, dtype=np.float32).reshape(original_shape)

    # Quantized path: int8 + float32 scales
    q = _load_npy_path(tensor_dir / f"{sk}__q.npy")
    s = _load_npy_path(tensor_dir / f"{sk}__s.npy")
    original_shape = tuple(_load_npy_path(tensor_dir / f"{sk}__shape.npy").tolist())

    result = QuantizationResult(
        quantized=np.array(q),   # bring int8 into RAM for Vectro
        scales=np.array(s),
        dims=q.shape[1],
        n=q.shape[0],
    )
    arr_f32 = reconstruct_embeddings(result)
    return arr_f32.reshape(original_shape)


def save_int4_npy_dir(
    npy_dir: str,
    group_size: int = _INT4_GROUP_SIZE,
    verbose: bool = True,
) -> dict:
    """
    Convert an existing INT8 npy-dir tensors/ directory to INT4 packed format.

    Reads each ``{sk}__q.npy`` + ``{sk}__s.npy`` pair, dequantizes to float32,
    then re-quantizes with INT4 grouped packing (50 % disk vs INT8).  Passthrough
    float16 tensors are skipped — they are already near-lossless.

    Writes ``{sk}__q4.npy`` and ``{sk}__s4.npy`` alongside the existing files.
    Touches ``.squish_int4_ready`` in the npy-dir root when complete.

    Requires ``squish_quant`` Rust extension.  Run once per model; subsequent
    loads auto-detect the INT4 files and use them without re-quantizing.

    Args:
        npy_dir:    Path to the npy-dir root (contains manifest.json + tensors/).
        group_size: Nibble-pack group width (default 64 — must stay constant
                    between save and load).
        verbose:    Print per-tensor progress.

    Returns:
        dict with 'n_converted', 'n_skipped', 'bytes_before', 'bytes_after',
        'savings_pct', 'elapsed_s'.
    """
    if _squish_quant is None:
        raise RuntimeError(
            "squish_quant Rust extension required.  "
            "Run: cd /path/to/poc/squish_quant_rs && python3 -m maturin build --release"
        )

    root       = Path(npy_dir)
    tensor_dir = root / "tensors"
    ready_flag = root / _INT4_READY

    if ready_flag.exists():
        if verbose:
            print(f"INT4 cache already exists in {root} — skipping conversion")
        return {"skipped": True}

    if not tensor_dir.exists():
        raise FileNotFoundError(f"tensors/ directory not found in {npy_dir}")

    import re as _re
    suffix_re = _re.compile(r'__(q|s|shape|pt)\.npy$')
    base_keys = sorted({
        suffix_re.sub('', p.name)
        for p in tensor_dir.glob("*.npy")
        if suffix_re.search(p.name)
    } | {
        suffix_re.sub('', p.name[:-4])
        for p in tensor_dir.glob("*.npy.zst")
        if suffix_re.search(p.name[:-4])
    })

    if verbose:
        print(f"Converting {len(base_keys)} tensors to INT4  "
              f"(group_size={group_size}) → {tensor_dir}")

    t0 = time.perf_counter()
    n_converted = n_skipped = 0
    bytes_before = bytes_after = 0

    for sk in base_keys:
        q_path = tensor_dir / f"{sk}__q.npy"
        s_path = tensor_dir / f"{sk}__s.npy"
        pt_path = tensor_dir / f"{sk}__pt.npy"

        if pt_path.exists() and not q_path.exists():
            # Passthrough tensor — lossless float16, no benefit quantizing to INT4
            n_skipped += 1
            if verbose:
                print(f"  [SKIP-PT] {sk}")
            continue

        if not _npy_exists(q_path):
            n_skipped += 1
            if verbose:
                print(f"  [SKIP-MISSING] {sk}")
            continue

        q8  = np.array(_load_npy_path(q_path), dtype=np.int8)
        s8  = np.array(_load_npy_path(s_path), dtype=np.float32)
        original_shape = tuple(_load_npy_path(
            tensor_dir / f"{sk}__shape.npy"
        ).tolist())

        # Dequantize INT8 → float32 (flat 2-D view is what reconstruct expects)
        result_obj = QuantizationResult(
            quantized=q8, scales=s8, dims=q8.shape[1], n=q8.shape[0]
        )
        arr_f32 = reconstruct_embeddings(result_obj)     # (n_rows, n_cols)

        # Re-quantize to INT4 nibble-packed
        packed, scales4 = _squish_quant.quantize_int4_grouped(
            np.ascontiguousarray(arr_f32, dtype=np.float32), group_size
        )
        del arr_f32

        bytes_before += q8.nbytes + s8.nbytes
        bytes_after  += packed.nbytes + scales4.nbytes

        np.save(str(tensor_dir / f"{sk}__q4.npy"), packed)
        np.save(str(tensor_dir / f"{sk}__s4.npy"), scales4)
        n_converted += 1

        if verbose:
            pct = packed.nbytes / q8.nbytes * 100
            print(f"  [INT4] {sk}: {q8.shape} → packed{packed.shape}  "
                  f"({pct:.0f}% of INT8 size)")

    ready_flag.touch()
    elapsed = time.perf_counter() - t0
    savings = (1 - bytes_after / bytes_before) * 100 if bytes_before else 0.0

    summary = {
        "n_converted":  n_converted,
        "n_skipped":    n_skipped,
        "bytes_before": bytes_before,
        "bytes_after":  bytes_after,
        "savings_pct":  savings,
        "elapsed_s":    elapsed,
    }

    if verbose:
        print(f"\n  INT4 conversion complete in {elapsed:.1f}s")
        print(f"  Converted: {n_converted}  Skipped (PT): {n_skipped}")
        print(f"  Size: {bytes_before / 1e6:.1f} MB → {bytes_after / 1e6:.1f} MB  "
              f"({savings:.0f}% savings)")
        print(f"  Sentinel written: {ready_flag}")

    return summary


# ---------------------------------------------------------------------------
# Phase 1.1 — ZipNN/zstd weight compression
# ---------------------------------------------------------------------------
# After producing an INT8 or INT4 npy-dir the files are raw numpy binary —
# no entropy coding.  Zstandard level 3 (fast) typically saves 20-35 % on
# model weights with near-instant decompression (~1-3 GB/s on Apple Silicon).
# The resulting .npy.zst files are transparently loaded by _load_npy_path().

def compress_npy_dir(
    npy_dir: str,
    level: int  = 3,
    threads: int = -1,
    verbose: bool = True,
    skip_existing: bool = True,
) -> dict:
    """
    Compress all ``.npy`` files in a npy-dir with Zstandard.

    Each ``foo.npy`` is compressed to ``foo.npy.zst``.  If the compressed file
    is smaller the original ``.npy`` is removed; otherwise the ``.zst`` is
    discarded (some tensors are incompressible).

    The finalized/ cache and .squish_* sentinel files are left unchanged so
    the load path continues to work without modification.

    Args:
        npy_dir:        Path to the npy-dir root (parent of tensors/).
        level:          Zstd compression level 1-22 (default 3 — fast + good ratio).
        threads:        Zstd worker threads.  -1 = use all CPUs.
        verbose:        Print per-file progress.
        skip_existing:  Skip files where the .npy.zst already exists.

    Returns:
        dict: n_compressed, n_skipped, bytes_before, bytes_after, savings_pct, elapsed_s
    """
    try:
        import zstandard as _zstd
    except ImportError:
        raise ImportError(
            "zstandard is required for weight compression.  "
            "Install with: pip install zstandard"
        )

    root       = Path(npy_dir)
    # Compress in tensors/ and any sub-dirs (e.g. finalized/)
    npy_files  = sorted(root.rglob("*.npy"))
    if not npy_files:
        return {"n_compressed": 0, "n_skipped": 0,
                "bytes_before": 0, "bytes_after": 0, "savings_pct": 0.0, "elapsed_s": 0.0}

    cctx = _zstd.ZstdCompressor(level=level,
                                  threads=threads if threads != -1 else 0)
    t0 = time.perf_counter()
    bytes_before = n_compressed = n_skipped = 0
    bytes_after  = 0

    for fpath in npy_files:
        zst_path = Path(str(fpath) + ".zst")

        if skip_existing and zst_path.exists():
            bytes_before += fpath.stat().st_size
            bytes_after  += zst_path.stat().st_size
            n_skipped    += 1
            continue

        raw = fpath.read_bytes()
        compressed = cctx.compress(raw)
        bytes_before += len(raw)

        if len(compressed) < len(raw):
            zst_path.write_bytes(compressed)
            bytes_after += len(compressed)
            if verbose:
                ratio = len(compressed) / len(raw) * 100
                print(f"  [zstd] {fpath.name}  {len(raw)//1024} KB → "
                      f"{len(compressed)//1024} KB  ({ratio:.0f}%)")
            fpath.unlink()   # remove uncompressed original
            n_compressed += 1
        else:
            # Not compressible — keep original
            bytes_after += len(raw)
            n_skipped   += 1
            if verbose:
                print(f"  [skip] {fpath.name}  incompressible")

    elapsed  = time.perf_counter() - t0
    savings  = (1.0 - bytes_after / bytes_before) * 100 if bytes_before else 0.0

    if verbose:
        print(f"\n  zstd compression complete in {elapsed:.1f}s")
        print(f"  Compressed: {n_compressed}  Skipped: {n_skipped}")
        print(f"  {bytes_before / 1e6:.1f} MB → {bytes_after / 1e6:.1f} MB  "
              f"({savings:.1f}% savings)")

    return {
        "n_compressed": n_compressed,
        "n_skipped":    n_skipped,
        "bytes_before": bytes_before,
        "bytes_after":  bytes_after,
        "savings_pct":  savings,
        "elapsed_s":    elapsed,
    }


def _decomp_task(
    tensor_dir: Path, sk: str
) -> tuple[str, "np.ndarray", str]:
    """
    Worker function for parallel decompression.
    Returns (safe_key, arr_f32, mode_label) where mode_label is 'PT', 'INT4', or 'Q8'.
    Runs in a ThreadPoolExecutor thread — reconstruct_embeddings is a native
    C extension that releases the GIL, so threads run in true parallel.
    """
    q4_path = tensor_dir / f"{sk}__q4.npy"
    pt_path = tensor_dir / f"{sk}__pt.npy"
    if _npy_exists(q4_path) and _squish_quant is not None:
        mode = "INT4"
    elif _npy_exists(pt_path):
        mode = "PT"
    else:
        mode = "Q8"
    arr = _dequantize_npy_dir(tensor_dir, sk)
    return sk, arr, mode


def load_from_npy_dir(
    dir_path: str,
    model_dir: str,
    verbose: bool = True,
    return_stats: bool = False,
    workers: int = 0,
    auto_quantize_bits: int | None = None,
):
    """
    Load a Vectro-compressed model from a npy-dir directory
    (produced by convert_weights.py --format npy-dir).

    First call:  Vectro decompression + saves a finalized f16 cache for future runs.
    Subsequent:  Loads directly from the f16 cache — no Vectro, ~3-4s load time.

    Args:
        workers: decompression threads.  0 or 1 (default) = serial, which is
                 faster for GIL-bound Vectro.  >1 = I/O prefetch pipeline.

    Returns:
        (model, tokenizer)               when return_stats=False
        (model, tokenizer, stats_dict)   when return_stats=True
    """
    # Serial is faster: Vectro is GIL-bound; threading adds overhead with no benefit.
    if workers <= 0:
        workers = 1
    dir_path   = Path(dir_path)
    tensor_dir = dir_path / "tensors"
    manifest_path_obj = dir_path / "manifest.json"

    # ── Tier 0a: directory IS a native MLX/HF model (config.json present, no manifest) ──
    # Handles mlx-community 4-bit models passed directly as model_dir, e.g.
    #   squish run ~/.squish/models/llama3.1-8b-4bit
    # These have config.json + model.safetensors but no manifest.json/tensors/.
    if (dir_path / "config.json").exists() and not manifest_path_obj.exists():
        if verbose:
            _sf = list(dir_path.glob("*.safetensors"))
            _sz = sum(f.stat().st_size for f in _sf) / 1e9
            print(f"  → Native MLX model detected ({_sz:.1f} GB safetensors) "
                  f"— loading via mlx_lm.load()")
        import mlx_lm as _mlx_lm_native
        _rss0 = _rss_mb()
        _t0   = time.perf_counter()
        _model, _tok = _mlx_lm_native.load(str(dir_path))
        _load_s = time.perf_counter() - _t0
        _rss1   = _rss_mb()
        if verbose:
            print(f"  Model loaded in {_load_s:.2f}s  (RAM Δ {_rss1 - _rss0:+.0f} MB)")
        _stats = {
            "loader":               "mlx-native",
            "decompression_time_s": _load_s,
            "ram_delta_mb":         _rss1 - _rss0,
            "ram_baseline_mb":      _rss0,
        }
        if return_stats:
            return _model, _tok, _stats
        return _model, _tok

    # ── Tier 0: 4-bit MLX model dir (built once by mlx_lm.convert) ──────────
    # Check this FIRST — before manifest/tensors guards — so models that were
    # compressed with --large-only (no Q8 npy-dir step) still load correctly.
    # For models where Q8→bf16 expansion > ~10 GB (i.e. 7B+) the bf16 load
    # path would OOM on a 16 GB device.  mlx_lm.convert creates a proper 4-bit
    # safetensors model (4-5 GB for 7B) which loads via mlx_lm.load() in <2s
    # and stays well within the Metal budget.
    _four_bit_dir   = dir_path / "squish_4bit"
    _four_bit_ready = dir_path / ".squish_4bit_ready"
    if _four_bit_ready.exists() and (_four_bit_dir / "config.json").exists():
        if verbose:
            _sz = sum(f.stat().st_size for f in _four_bit_dir.rglob("*")
                      if f.is_file()) / 1e9
            print(f"  → 4-bit model cache found ({_sz:.1f} GB) "
                  f"— loading via mlx_lm.load()")
        import mlx_lm as _mlx_lm_4bit
        _rss0 = _rss_mb()
        _t0   = time.perf_counter()
        _model, _tok = _mlx_lm_4bit.load(str(_four_bit_dir))
        _load_s = time.perf_counter() - _t0
        _rss1   = _rss_mb()
        if verbose:
            print(f"  4-bit model loaded in {_load_s:.2f}s  "
                  f"(RAM Δ {_rss1 - _rss0:+.0f} MB)")
        _stats = {
            "loader":               "squish-4bit",
            "decompression_time_s": _load_s,
            "ram_delta_mb":         _rss1 - _rss0,
            "ram_baseline_mb":      _rss0,
        }
        if return_stats:
            return _model, _tok, _stats
        return _model, _tok

    # ── Tier 0 not found — verify npy-dir exists before proceeding ───────────
    if not manifest_path_obj.exists():
        raise FileNotFoundError(
            f"manifest.json not found in {dir_path}\n"
            f"Tip: run pull_model.py to build the compressed model first."
        )
    if not tensor_dir.exists():
        raise FileNotFoundError(
            f"tensors/ subdirectory not found in {dir_path}\n"
            f"Tip: run pull_model.py to build the compressed model first."
        )

    # ── Safety check: refuse to load large models as bf16 (would OOM/crash) ──
    _q8_gb      = sum(f.stat().st_size for f in tensor_dir.rglob("*") if f.is_file()) / 1e9
    _est_bf16_gb = _q8_gb * 2.0
    _MAX_BF16_GB = 10.0
    if _est_bf16_gb > _MAX_BF16_GB and auto_quantize_bits is None:
        raise RuntimeError(
            f"Model Q8→bf16 expansion ({_est_bf16_gb:.1f} GB) would exceed safe Metal "
            f"limit ({_MAX_BF16_GB:.0f} GB) on a 16 GB device.\n"
            f"Run pull_model.py to build the 4-bit cache first:\n"
            f"  python3 pull_model.py <MODEL_ID> --skip-download --skip-compress"
        )

    # ── Tier 1: MLX safetensors fast cache (bf16, sub-2s loads) ─────────────
    finalized_dir   = dir_path / _FINALIZED_DIR
    mlx_cache_path  = dir_path / _MLX_CACHE_FILE
    mlx_cache_ready = dir_path / _MLX_CACHE_READY
    if mlx_cache_ready.exists() and mlx_cache_path.exists() and auto_quantize_bits is None:
        if verbose:
            print(f"  → MLX safetensors cache found — loading at reference speed")
        return _load_mlx_cache(dir_path, model_dir,
                               verbose=verbose, return_stats=return_stats)

    # ── Tier 2: Finalized f16 .npy cache (4-5s loads) ─────────────────────
    ready_flag    = finalized_dir / ".ready"
    if ready_flag.exists() and auto_quantize_bits is None:
        if verbose:
            print(f"  → Finalized cache found — loading f16 weights (no Vectro)")
        return _load_finalized_cache(dir_path, model_dir,
                                     verbose=verbose, return_stats=return_stats)

    # ── First (Vectro) load ───────────────────────────────────────────────────
    stats: dict = {}
    rss_baseline = _rss_mb()
    stats["ram_baseline_mb"] = rss_baseline

    if verbose:
        print(f"Building model architecture from {model_dir}/config.json ...")
    model, mlx_type = _instantiate_model(model_dir)
    if verbose:
        print(f"  model_type = {mlx_type}")

    with open(manifest_path_obj) as f:
        manifest = json.load(f)
    safe_to_original = {v: k for k, v in manifest.items()}

    suffix_re = __import__("re").compile(r'__(q|s|shape|pt)\.npy$')
    base_keys = sorted({
        suffix_re.sub('', p.name)
        for p in tensor_dir.glob("*.npy")
        if suffix_re.search(p.name)
    } | {
        # Also include zstd-compressed tensors (.npy.zst → strip .zst first)
        suffix_re.sub('', p.name[:-4])   # p.name[:-4] strips ".zst"
        for p in tensor_dir.glob("*.npy.zst")
        if suffix_re.search(p.name[:-4])
    })

    # Check whether INT4 packed files are available (written by save_int4_npy_dir)
    _int4_ready = (dir_path / _INT4_READY).exists() and _squish_quant is not None
    mode_label = f"pipeline ({workers}T)" if workers > 1 else "serial"
    quant_label = "INT4 Rust" if _int4_ready else "INT8 Vectro"
    if verbose:
        print(f"  {len(base_keys)} tensors  →  {quant_label} decomp ({mode_label})")
        if _int4_ready:
            print(f"  ⚡ INT4 nibble-packed cache active (50% disk vs INT8)")

    # ── Prepare finalized cache directory ─────────────────────────────────────
    # Skip for large models that will be 4-bit quantized — the f16 cache
    # would still require bf16 expansion on the next load.
    finalized_dir_obj = dir_path / _FINALIZED_DIR
    try:
        finalized_dir_obj.mkdir(exist_ok=True)
        save_finalized = auto_quantize_bits is None   # skip for 4-bit models
    except OSError:
        save_finalized = False

    t0 = time.perf_counter()
    rss_peak = rss_baseline
    weight_tuples = []
    n_q = n_pt = 0

    if workers == 1:
        # ── Fast serial path ──────────────────────────────────────────────────
        for sk in base_keys:
            original_name = safe_to_original.get(sk)
            if original_name is None:
                if verbose:
                    print(f"  WARNING: no manifest entry for '{sk}' — skipping")
                continue
            arr_f32 = _dequantize_npy_dir(tensor_dir, sk)

            # Save f16 to finalized cache while we have arr_f32 in hand
            if save_finalized:
                fkey = original_name.replace(".", "__")
                np.save(str(finalized_dir_obj / f"{fkey}.npy"),
                        arr_f32.astype(np.float16))

            mlx_arr = mx.array(arr_f32).astype(mx.bfloat16)
            del arr_f32
            weight_tuples.append((original_name, mlx_arr))
            cur_rss = _rss_mb()
            if cur_rss > rss_peak:
                rss_peak = cur_rss
            # Determine mode label for verbose output
            if _npy_exists(tensor_dir / f"{sk}__q4.npy") and _squish_quant is not None:
                mode_str = "INT4"
                n_q += 1
            elif _npy_exists(tensor_dir / f"{sk}__pt.npy"):
                mode_str = "PT"
                n_pt += 1
            else:
                mode_str = "Q8"
                n_q += 1
            if verbose:
                print(f"  [{mode_str}] {original_name}: {tuple(mlx_arr.shape)}  RSS {cur_rss:.0f} MB")
    else:
        # ── Streaming pipeline: I/O-prefetch via thread pool ─────────────────
        # Futures consumed in submission order; numpy buffers freed immediately.
        with ThreadPoolExecutor(max_workers=workers) as pool:
            ordered_futures = [
                (sk, pool.submit(_decomp_task, tensor_dir, sk))
                for sk in base_keys
            ]
            for sk, fut in ordered_futures:
                original_name = safe_to_original.get(sk)
                if original_name is None:
                    if verbose:
                        print(f"  WARNING: no manifest entry for '{sk}' — skipping")
                    fut.result()
                    continue
                _sk_done, arr_f32, mode_str = fut.result()
                if save_finalized:
                    fkey = original_name.replace(".", "__")
                    np.save(str(finalized_dir_obj / f"{fkey}.npy"),
                            arr_f32.astype(np.float16))
                mlx_arr = mx.array(arr_f32).astype(mx.bfloat16)
                del arr_f32
                weight_tuples.append((original_name, mlx_arr))
                cur_rss = _rss_mb()
                if cur_rss > rss_peak:
                    rss_peak = cur_rss
                if mode_str == "PT":
                    n_pt += 1
                else:
                    n_q += 1
                if verbose:
                    print(f"  [{mode_str}] {original_name}: {tuple(mlx_arr.shape)}  RSS {cur_rss:.0f} MB")

    decompression_time = time.perf_counter() - t0
    if verbose:
        print(f"\nLoaded {len(weight_tuples)} tensors "
              f"({n_q} Q8, {n_pt} PT-f16) in {decompression_time:.2f}s")
        print("Calling model.load_weights() ...")

    # ── Save MLX safetensors fast cache (skip for large models — building the
    #    dict of ALL bf16 arrays at once would OOM on 16 GB before we can quantize)
    mlx_cache_path  = dir_path / _MLX_CACHE_FILE
    mlx_cache_ready = dir_path / _MLX_CACHE_READY
    if auto_quantize_bits is None:
        t_cache = time.perf_counter()
        try:
            weight_dict = {name: arr for name, arr in weight_tuples}
            mx.save_safetensors(str(mlx_cache_path), weight_dict)
            del weight_dict
            mlx_cache_ready.touch()
            save_cache_s = time.perf_counter() - t_cache
            if verbose:
                cache_mb = mlx_cache_path.stat().st_size / 1e6
                print(f"  MLX safetensors cache saved ({cache_mb:.0f} MB, {save_cache_s:.1f}s)"
                      f" → next load will be reference-speed")
        except Exception as _e:
            if verbose:
                print(f"  WARNING: could not save MLX cache: {_e}")

    model.load_weights(weight_tuples)
    del weight_tuples

    # ── Post-load 4-bit quantization (for large models that won't fit in Metal as bf16)
    # MLX is lazy: nn.quantize transforms the computation graph so that Metal
    # only ever materializes 4-bit weights (~50% of Q8 size) instead of bf16.
    loader_tag = "npy-dir-int4" if _int4_ready else "npy-dir"
    if auto_quantize_bits is not None:
        import mlx.nn as _nn
        t_q = time.perf_counter()
        if verbose:
            print(f"  → Quantizing to {auto_quantize_bits}-bit via nn.quantize() …")
        # Run quantize graph on CPU so the bf16 intermediate stays in system RAM,
        # not Metal — preventing the OOM that would otherwise occur during eval.
        with mx.stream(mx.cpu):
            _nn.quantize(model, bits=auto_quantize_bits, group_size=64)
        mx.eval(model.parameters())
        q_s = time.perf_counter() - t_q
        loader_tag = f"npy-dir-{auto_quantize_bits}bit"
        if verbose:
            print(f"  → {auto_quantize_bits}-bit quantization complete ({q_s:.1f}s)")

    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)

    rss_final = _rss_mb()
    stats.update({
        "ram_peak_mb":          rss_peak,
        "ram_after_load_mb":    rss_final,
        "ram_final_mb":         rss_final,
        "ram_delta_mb":         rss_final - rss_baseline,
        "n_quantized":          n_q,
        "n_passthrough":        n_pt,
        "n_tensors":            n_q + n_pt,
        "decompression_time_s": decompression_time,
        "decomp_workers":       workers,
        "loader":               loader_tag,
    })

    if verbose:
        print(f"RAM baseline: {rss_baseline:.0f} MB  →  final: {rss_final:.0f} MB  "
              f"(Δ {rss_final - rss_baseline:+.0f} MB)\nModel ready.\n")

    # ── Write finalized f16 .npy cache sentinel ───────────────────────────────
    if save_finalized:
        (finalized_dir_obj / ".ready").touch()
        if verbose:
            cache_mb = sum(p.stat().st_size for p in finalized_dir_obj.glob("*.npy")) / 1e6
            print(f"  Finalized f16 cache written ({cache_mb:.0f} MB) → fallback fast path")

    if return_stats:
        return model, tokenizer, stats
    return model, tokenizer




def load_compressed_model(
    model_dir: str,
    npz_path: str,
    manifest_path: Optional[str] = None,
    verbose: bool = True,
    return_stats: bool = False,
    workers: int = 0,
):
    """
    Unified entry point — auto-detects npy-dir vs npz format.

    - If ``npz_path`` is a *directory*  → :func:`load_from_npy_dir` (mmap, fast)
    - If ``npz_path`` ends in ``.npz``  → legacy zlib-npz loader

    Arguments:
        model_dir     -- HuggingFace model directory (needs config.json + tokenizer)
        npz_path      -- path to weights_compressed.npz  -OR-  the npy-dir directory
        manifest_path -- npz only: path to _manifest.json (default: derived from npz_path)
        verbose       -- print per-tensor progress
        return_stats  -- return (model, tokenizer, stats_dict) instead of (model, tokenizer)
        workers       -- npy-dir only: parallel decompression threads (0 = auto)
    """
    # ── Auto-detect format ────────────────────────────────────────────────
    if Path(npz_path).is_dir():
        return load_from_npy_dir(npz_path, model_dir,
                                 verbose=verbose, return_stats=return_stats,
                                 workers=workers)

    if manifest_path is None:
        manifest_path = npz_path.replace(".npz", "_manifest.json")

    stats: dict = {}
    rss_baseline = _rss_mb()
    stats["ram_baseline_mb"] = rss_baseline

    # 1. Build the model architecture from config — no weights, very low RAM
    if verbose:
        print(f"Building model architecture from {model_dir}/config.json ...")
    model, mlx_type = _instantiate_model(model_dir)
    if verbose:
        print(f"  model_type = {mlx_type}")

    # 2. Load manifest
    safe_to_original = _safe_key_to_original(manifest_path)

    # 3. Open npz lazily
    if verbose:
        print(f"Opening compressed weights: {npz_path}")
    npz = np.load(npz_path, allow_pickle=False)
    base_keys = _unique_base_keys(list(npz.files))

    if verbose:
        print(f"  {len(base_keys)} tensors found in archive")

    # 4. Dequantize each tensor one at a time and accumulate (name, mlx_array) pairs
    t0 = time.time()
    rss_peak_during = rss_baseline
    weight_tuples = []
    n_q = 0
    n_pt = 0

    for sk in sorted(base_keys):
        original_name = safe_to_original.get(sk)
        if original_name is None:
            if verbose:
                print(f"  WARNING: no manifest entry for '{sk}' — skipping")
            continue

        arr_f32 = _dequantize(npz, sk)

        # Cast to bfloat16 to match MLX-LM's expected dtype
        mlx_arr = mx.array(arr_f32).astype(mx.bfloat16)
        del arr_f32  # release the numpy buffer immediately

        weight_tuples.append((original_name, mlx_arr))

        # track peak RSS as we load tensors one by one
        cur_rss = _rss_mb()
        if cur_rss > rss_peak_during:
            rss_peak_during = cur_rss

        is_pt = (sk + "__pt") in npz.files
        if is_pt:
            n_pt += 1
        else:
            n_q += 1

        if verbose:
            mode = "PT" if is_pt else "Q8"
            print(f"  [{mode}] {original_name}: {tuple(mlx_arr.shape)}"
                  f"  RSS {cur_rss:.0f} MB")

    npz.close()

    decompression_time = time.time() - t0
    if verbose:
        print(f"\nDecompressed {len(weight_tuples)} tensors "
              f"({n_q} quantized, {n_pt} passthrough) in {decompression_time:.2f}s")

    # 5. Inject weights into the model
    if verbose:
        print("Calling model.load_weights() ...")
    model.load_weights(weight_tuples)
    del weight_tuples  # allow GC after injection

    rss_after_load = _rss_mb()

    # 6. Load tokenizer from model_dir
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)

    rss_final = _rss_mb()
    stats.update({
        "ram_peak_mb": rss_peak_during,
        "ram_after_load_mb": rss_after_load,
        "ram_final_mb": rss_final,
        "ram_delta_mb": rss_final - rss_baseline,
        "n_quantized": n_q,
        "n_passthrough": n_pt,
        "n_tensors": n_q + n_pt,
        "decompression_time_s": decompression_time,
        "loader": "batch-npz",
    })

    if verbose:
        print(f"RAM baseline:   {rss_baseline:.0f} MB")
        print(f"RAM peak:       {rss_peak_during:.0f} MB")
        print(f"RAM after load: {rss_after_load:.0f} MB  (Δ {rss_after_load - rss_baseline:+.0f} MB)")
        print("Model ready.\n")

    if return_stats:
        return model, tokenizer, stats
    return model, tokenizer
