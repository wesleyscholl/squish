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
import io
import json
from pathlib import Path
from typing import Any

import numpy as np

from squish.quantizer import QuantizationResult, dequantize_int4, reconstruct_embeddings


# ---------------------------------------------------------------------------
# Lazy loaders for optional compression backends
# ---------------------------------------------------------------------------
def _get_dequantize_nf4():
    from squish.nf4_quant import dequantize_nf4
    return dequantize_nf4


def _get_dfloat11():
    from squish.dfloat11 import DFloat11Compressor, DFloat11Config
    return DFloat11Config, DFloat11Compressor


def _get_vptq():
    from squish.vptq import VPTQCodebook, VPTQConfig, VPTQLayer, VPTQQuantizer
    return VPTQConfig, VPTQCodebook, VPTQLayer, VPTQQuantizer

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
        except ImportError:  # pragma: no cover
            _zstd_dctx = False
    return _zstd_dctx if _zstd_dctx is not False else None


def _load_npy_path(path: Path, mmap_mode: str | None = "r") -> np.ndarray:
    """Load a .npy, .npy.zst, or .npy.br file with transparent decompression."""
    if path.exists():
        return np.load(str(path), mmap_mode=mmap_mode)
    zst_path = Path(str(path) + ".zst")
    if zst_path.exists():
        dctx = _get_zstd_dctx()
        if dctx is None:  # pragma: no cover
            raise RuntimeError(
                f"Found {zst_path} but 'zstandard' is not installed.\n"
                "  Run: pip install zstandard"
            )
        with open(zst_path, "rb") as f:
            buf = io.BytesIO(dctx.decompress(f.read()))
        return np.load(buf, allow_pickle=False)
    br_path = Path(str(path) + ".br")
    if br_path.exists():
        try:
            import brotli as _brotli
        except ImportError:  # pragma: no cover
            raise RuntimeError(
                f"Found {br_path} but 'brotli' is not installed.\n"
                "  Run: pip install brotli"
            )
        with open(br_path, "rb") as f:
            buf = io.BytesIO(_brotli.decompress(f.read()))
        return np.load(buf, allow_pickle=False)
    raise FileNotFoundError(f"Neither {path} nor {zst_path} nor {br_path} found")


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


def _instantiate_model(model_dir: str) -> tuple:  # pragma: no cover
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
    pt_path      = tensor_dir / f"{sk}__pt.npy"
    q_path       = tensor_dir / f"{sk}__q.npy"
    q4_path      = tensor_dir / f"{sk}__q4.npy"
    s4_path      = tensor_dir / f"{sk}__s4.npy"
    nf4_path     = tensor_dir / f"{sk}__nf4.npy"
    s_nf4_path   = tensor_dir / f"{sk}__s_nf4.npy"
    vq_idx_path  = tensor_dir / f"{sk}__vq_idx.npy"
    pt_df11_path = tensor_dir / f"{sk}__pt_df11.npy"
    s4_df11_path = tensor_dir / f"{sk}__s4_df11.npy"

    # ── DFloat11-compressed passthrough ──────────────────────────────────────
    pt_df11_exists = pt_df11_path.exists() or Path(str(pt_df11_path) + ".zst").exists() or Path(str(pt_df11_path) + ".br").exists()
    if pt_df11_exists:
        import pickle
        _, DFloat11Compressor = _get_dfloat11()
        blob = _load_npy_path(pt_df11_path, mmap_mode=None).tobytes()
        blocks = pickle.loads(blob)
        comp = DFloat11Compressor()
        arr = comp.decompress_array(blocks)
        shape_path = tensor_dir / f"{sk}__shape.npy"
        if shape_path.exists() or Path(str(shape_path) + ".zst").exists():
            original_shape = tuple(int(x) for x in _load_npy_path(shape_path, mmap_mode=None).tolist())
            return arr.astype(np.float32).reshape(original_shape)
        return arr.astype(np.float32)

    # ── NF4 nibble-packed ─────────────────────────────────────────────────────
    nf4_exists   = nf4_path.exists() or Path(str(nf4_path) + ".zst").exists()
    s_nf4_exists = s_nf4_path.exists() or Path(str(s_nf4_path) + ".zst").exists()
    if nf4_exists and s_nf4_exists:
        dequantize_nf4 = _get_dequantize_nf4()
        packed = np.ascontiguousarray(_load_npy_path(nf4_path, mmap_mode=None), dtype=np.uint8)
        scales = np.ascontiguousarray(_load_npy_path(s_nf4_path, mmap_mode=None), dtype=np.float32)
        arr = dequantize_nf4(packed, scales, group_size=64)
        shape_path = tensor_dir / f"{sk}__shape.npy"
        if shape_path.exists() or Path(str(shape_path) + ".zst").exists():
            original_shape = tuple(int(x) for x in _load_npy_path(shape_path, mmap_mode=None).tolist())
            return arr.reshape(original_shape)
        return arr

    # ── VPTQ vector-quantized ─────────────────────────────────────────────────
    vq_idx_exists = vq_idx_path.exists() or Path(str(vq_idx_path) + ".zst").exists()
    if vq_idx_exists:
        VPTQConfig, VPTQCodebook, VPTQLayer, VPTQQuantizer = _get_vptq()
        idx       = _load_npy_path(vq_idx_path, mmap_mode=None).astype(np.int64)
        cb_data   = _load_npy_path(tensor_dir / f"{sk}__vq_cb.npy", mmap_mode=None).astype(np.float32)
        res_data  = _load_npy_path(tensor_dir / f"{sk}__vq_res.npy", mmap_mode=None).astype(np.int64)
        rescb_data = _load_npy_path(tensor_dir / f"{sk}__vq_rescb.npy", mmap_mode=None).astype(np.float32)
        col_scales_path = tensor_dir / f"{sk}__vq_cols.npy"
        meta_path = tensor_dir / f"{sk}__vq_meta.npy"
        meta      = _load_npy_path(meta_path, mmap_mode=None).astype(np.int64)
        n_rows, n_cols, group_size, n_cb, n_res_cb = int(meta[0]), int(meta[1]), int(meta[2]), int(meta[3]), int(meta[4])

        # Reconstruct primary codebook
        cb_centroids = cb_data[:-3].reshape(-1, group_size).astype(np.float32)
        primary_cb   = VPTQCodebook.__new__(VPTQCodebook)
        primary_cb._centroids          = cb_centroids
        primary_cb.n_codebook_entries  = n_cb
        primary_cb.group_size          = group_size
        primary_cb.n_fit_iters         = int(cb_data[-1])

        # Reconstruct residual codebook if present
        residual_cb  = None
        residual_idx = None
        if n_res_cb > 0 and res_data.size > 1:
            rescb_centroids = rescb_data[:-3].reshape(-1, group_size).astype(np.float32)
            residual_cb = VPTQCodebook.__new__(VPTQCodebook)
            residual_cb._centroids          = rescb_centroids
            residual_cb.n_codebook_entries  = n_res_cb
            residual_cb.group_size          = group_size
            residual_cb.n_fit_iters         = int(rescb_data[-1])
            residual_idx = res_data

        col_scales = None
        if col_scales_path.exists() or Path(str(col_scales_path) + ".zst").exists():
            cs = _load_npy_path(col_scales_path, mmap_mode=None).astype(np.float32)
            if cs.size > 0:
                col_scales = cs

        layer = VPTQLayer(
            primary_indices  = idx,
            residual_indices = residual_idx,
            primary_cb       = primary_cb,
            residual_cb      = residual_cb,
            original_shape   = (n_rows, n_cols),
            col_scales       = col_scales,
        )
        quant = VPTQQuantizer.__new__(VPTQQuantizer)
        quant.config = VPTQConfig(
            n_codebook_entries=n_cb,
            group_size=group_size,
            n_residual_entries=n_res_cb,
        )
        arr = quant.decompress(layer)
        shape_path = tensor_dir / f"{sk}__shape.npy"
        if shape_path.exists() or Path(str(shape_path) + ".zst").exists():
            original_shape = tuple(int(x) for x in _load_npy_path(shape_path, mmap_mode=None).tolist())
            return arr.reshape(original_shape)
        return arr

    # ── INT4 nibble-packed (highest compression, requires squish_quant) ───────
    q4_exists = q4_path.exists() or Path(str(q4_path) + ".zst").exists()
    s4_exists = s4_path.exists() or Path(str(s4_path) + ".zst").exists()
    # Also check for DFloat11-compressed scales
    s4_df11_exists = s4_df11_path.exists() or Path(str(s4_df11_path) + ".zst").exists()
    if q4_exists and (s4_exists or s4_df11_exists):
        packed = np.ascontiguousarray(_load_npy_path(q4_path, mmap_mode=None), dtype=np.uint8)
        if s4_df11_exists:
            import pickle
            _, DFloat11Compressor = _get_dfloat11()
            blob   = _load_npy_path(s4_df11_path, mmap_mode=None).tobytes()
            blocks = pickle.loads(blob)
            comp   = DFloat11Compressor()
            # decompress_array returns a flat 1-D array; reshape to 2-D (N, n_groups)
            # so dequantize_int4_grouped (Rust) receives PyReadonlyArray2 as required.
            scales = np.ascontiguousarray(
                comp.decompress_array(blocks).astype(np.float32).reshape(packed.shape[0], -1)
            )
        else:
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
