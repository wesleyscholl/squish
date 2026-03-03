#!/usr/bin/env python3
"""
squish/loader_utils.py

Model architecture instantiation and npy/npz decompression helpers.

Previously these lived in the root compressed_loader.py and were imported via
a sys.path hack.  Moving them here makes the squish package self-contained so
that `pip install squish` works without a sibling ~/vectro or ~/squish checkout.

Public helpers (used by streaming.py and the npy-dir loader):
    _instantiate_model(model_dir)            → (model, mlx_type_str)
    _safe_key_to_original(manifest_path)     → dict[safe_key → original_name]
    _unique_base_keys(npz_files)             → set[str]
    _dequantize(npz, sk)                     → np.ndarray (float32)
    _dequantize_npy(tensor_dir, sk)          → np.ndarray (float32)
"""
from __future__ import annotations

import dataclasses
import importlib
import json
import io
from pathlib import Path
from typing import Any

import numpy as np

from squish.quantizer import reconstruct_embeddings, QuantizationResult, dequantize_int4

# ---------------------------------------------------------------------------
# Optional zstd decompression
# ---------------------------------------------------------------------------
_zstd_dctx = None


def _get_zstd_dctx():
    global _zstd_dctx
    if _zstd_dctx is None:
        try:
            import zstandard as _zstd
            _zstd_dctx = _zstd.ZstdDecompressor()
        except ImportError:
            _zstd_dctx = False
    return _zstd_dctx if _zstd_dctx is not False else None


def _load_npy_path(path: Path, mmap_mode: str | None = "r") -> np.ndarray:
    """Load a .npy or .npy.zst file with transparent decompression."""
    if path.exists():
        return np.load(str(path), mmap_mode=mmap_mode)
    zst_path = Path(str(path) + ".zst")
    if zst_path.exists():
        dctx = _get_zstd_dctx()
        if dctx is None:
            raise RuntimeError(
                f"Found {zst_path} but 'zstandard' is not installed.\n"
                "  Run: pip install zstandard"
            )
        with open(zst_path, "rb") as f:
            buf = io.BytesIO(dctx.decompress(f.read()))
        return np.load(buf, allow_pickle=False)
    raise FileNotFoundError(f"Neither {path} nor {zst_path} found")


# ---------------------------------------------------------------------------
# Model architecture map  (HuggingFace model_type → mlx_lm module name)
# ---------------------------------------------------------------------------
_HF_TO_MLX_TYPE: dict[str, str] = {
    "qwen2":       "qwen2",
    "mistral":     "mistral",
    "llama":       "llama",
    "phi3":        "phi3",
    "phi":         "phi",
    "gemma":       "gemma",
    "gemma2":      "gemma2",
    "starcoder2":  "starcoder2",
    "cohere":      "cohere",
    "falcon":      "falcon",
    "mpt":         "mpt",
    "gpt2":        "gpt2",
    "gpt_neox":    "gpt_neox",
    "olmo":        "olmo",
    "openelm":     "openelm",
}


def _build_model_args(ModelArgs: Any, config: dict) -> Any:
    """Construct ModelArgs from config, keeping only recognised fields."""
    if hasattr(ModelArgs, "from_dict"):
        return ModelArgs.from_dict(config)
    valid = {f.name for f in dataclasses.fields(ModelArgs)}
    return ModelArgs(**{k: v for k, v in config.items() if k in valid})


def _instantiate_model(model_dir: str) -> tuple:
    """Build the MLX model object from config.json alone — no weights loaded.

    Returns:
        (model, mlx_model_type_str)
    """
    config_path = Path(model_dir) / "config.json"
    with open(config_path) as f:
        config = json.load(f)

    hf_type  = config.get("model_type", "").lower()
    mlx_type = _HF_TO_MLX_TYPE.get(hf_type, hf_type)

    try:
        module = importlib.import_module(f"mlx_lm.models.{mlx_type}")
    except ImportError as e:
        raise ValueError(
            f"Cannot find mlx_lm.models.{mlx_type} for model_type={hf_type!r}. "
            f"Supported: {sorted(_HF_TO_MLX_TYPE.values())}"
        ) from e

    ModelArgs = module.ModelArgs
    Model     = module.Model
    args      = _build_model_args(ModelArgs, config)
    model     = Model(args)
    return model, mlx_type


# ---------------------------------------------------------------------------
# NPZ decompression helpers  (legacy .npz path — npz holds int8+scales)
# ---------------------------------------------------------------------------

def _safe_key_to_original(manifest_path: str) -> dict[str, str]:
    """Load manifest.json and return inverted dict: safe_key → original_name."""
    with open(manifest_path) as f:
        manifest = json.load(f)
    return {v: k for k, v in manifest.items()}


def _unique_base_keys(npz_files: list[str]) -> set[str]:
    """Extract unique base keys from an npz file list.

    e.g. ['model__embed__q', 'model__embed__s', 'model__embed__shape']
         → {'model__embed'}
    """
    suffixes = ("__q", "__s", "__pt", "__shape")
    base_keys: set[str] = set()
    for fname in npz_files:
        for suf in suffixes:
            if fname.endswith(suf):
                base_keys.add(fname[: -len(suf)])
                break
    return base_keys


def _dequantize(npz, sk: str) -> np.ndarray:
    """Reconstruct one tensor from an open NpzFile.

    Args:
        npz: open numpy NpzFile (lazy-loaded).
        sk:  safe_key (dots replaced with double-underscore).

    Returns:
        float32 numpy array with the original tensor shape.
    """
    has_shape = (sk + "__shape") in npz.files

    if (sk + "__pt") in npz.files:
        if has_shape:
            original_shape = tuple(int(x) for x in npz[sk + "__shape"].tolist())
        else:
            original_shape = npz[sk + "__pt"].shape
        return npz[sk + "__pt"].astype(np.float32).reshape(original_shape)

    original_shape = tuple(int(x) for x in npz[sk + "__shape"].tolist())
    q = npz[sk + "__q"]   # int8  (n_rows, n_cols)
    s = npz[sk + "__s"]   # float32 scales

    result = QuantizationResult(
        quantized=q,
        scales=s,
        dims=q.shape[1],
        n=q.shape[0],
    )
    arr_f32 = reconstruct_embeddings(result)
    return arr_f32.reshape(original_shape)


# ---------------------------------------------------------------------------
# npy-dir decompression helper  (current preferred path)
# ---------------------------------------------------------------------------

def _dequantize_npy(tensor_dir: Path, sk: str) -> np.ndarray:
    """Reconstruct one tensor from an npy-dir.

    Supports per-row INT8, per-group INT8 (scales 2-D), and float16
    passthrough files.  Also recognises .npy.zst compressed variants.

    Args:
        tensor_dir: directory containing {sk}__q.npy, {sk}__s.npy, etc.
        sk:         safe_key for the tensor.

    Returns:
        float32 numpy array with the original shape.
    """
    pt_path = tensor_dir / f"{sk}__pt.npy"
    q_path  = tensor_dir / f"{sk}__q.npy"
    q4_path = tensor_dir / f"{sk}__q4.npy"
    s4_path = tensor_dir / f"{sk}__s4.npy"

    # ── INT4 nibble-packed (highest compression, requires squish_quant) ───────
    q4_exists = q4_path.exists() or Path(str(q4_path) + ".zst").exists()
    s4_exists = s4_path.exists() or Path(str(s4_path) + ".zst").exists()
    if q4_exists and s4_exists:
        packed = np.ascontiguousarray(_load_npy_path(q4_path, mmap_mode=None), dtype=np.uint8)
        scales = np.ascontiguousarray(_load_npy_path(s4_path, mmap_mode=None), dtype=np.float32)
        return dequantize_int4(packed, scales, group_size=64)

    # ── Passthrough (float16) ─────────────────────────────────────────────
    pt_exists = pt_path.exists() or Path(str(pt_path) + ".zst").exists()
    if pt_exists:
        arr = _load_npy_path(pt_path, mmap_mode="r")
        return arr.astype(np.float32)

    # ── Quantized (INT8 per-row or per-group) ─────────────────────────────
    q = _load_npy_path(q_path, mmap_mode="r")
    s = _load_npy_path(tensor_dir / f"{sk}__s.npy", mmap_mode="r")

    result = QuantizationResult(
        quantized=np.asarray(q),
        scales=np.asarray(s),
        dims=q.shape[1],
        n=q.shape[0],
    )
    return reconstruct_embeddings(result)
