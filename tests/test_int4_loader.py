#!/usr/bin/env python3
"""
Smoke test: save_int4_npy_dir + INT4 dequantize path in compressed_loader.

Creates a minimal synthetic npy-dir, runs save_int4_npy_dir(), then calls
_dequantize_npy_dir() and verifies INT4 round-trip cosine quality.
"""
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

# Vectro optional dependency — set VECTRO_DIR env var or place at ~/vectro
_vectro = Path(os.environ.get("VECTRO_DIR", Path.home() / "vectro"))
if _vectro.exists():
    sys.path.insert(0, str(_vectro))

from squish.compressed_loader import (  # noqa: E402
    _INT4_READY,
    _dequantize_npy_dir,
    save_int4_npy_dir,
)


# ── Build a tiny synthetic npy-dir ──────────────────────────────────────────
def make_synthetic_npy_dir(root: Path, n: int = 64, d: int = 512) -> list[str]:
    """Save two INT8 tensors and one passthrough tensor."""
    tensor_dir = root / "tensors"
    tensor_dir.mkdir(parents=True)

    rng = np.random.default_rng(42)
    keys = []

    for name in ("tensor_a", "tensor_b"):
        # 2D weight tensor (n × d)
        arr = rng.standard_normal((n, d)).astype(np.float32)
        # INT8 per-row quantize (simple)
        row_max = np.abs(arr).max(axis=1, keepdims=True)
        scales = (row_max / 127.0).astype(np.float32).squeeze()
        q = np.clip(np.round(arr / scales[:, None]), -127, 127).astype(np.int8)
        np.save(str(tensor_dir / f"{name}__q.npy"), q)
        np.save(str(tensor_dir / f"{name}__s.npy"), scales)
        np.save(str(tensor_dir / f"{name}__shape.npy"), np.array(arr.shape))
        keys.append(name)

    # Passthrough float16 tensor
    bias = rng.standard_normal((d,)).astype(np.float16)
    np.save(str(tensor_dir / "bias__pt.npy"), bias)
    np.save(str(tensor_dir / "bias__shape.npy"), np.array(bias.shape))
    keys.append("bias")

    # Minimal manifest
    manifest = {f"model.{k}": k for k in keys}
    with open(root / "manifest.json", "w") as f:
        json.dump(manifest, f)

    return keys


with tempfile.TemporaryDirectory() as tmp:
    root = Path(tmp)
    print(f"Synthetic npy-dir: {root}")
    keys = make_synthetic_npy_dir(root)

    # ── Step 1: convert INT8 → INT4 ──────────────────────────────────────────
    print("\n--- save_int4_npy_dir() ---")
    result = save_int4_npy_dir(str(root), verbose=True)
    print(f"\nSummary: {result}")
    assert (root / _INT4_READY).exists(), "Sentinel not written"
    assert result["n_converted"] == 2, f"Expected 2 converted, got {result['n_converted']}"
    assert result["n_skipped"]   == 1, f"Expected 1 skipped (bias), got {result['n_skipped']}"
    savings = result["savings_pct"]
    print(f"\nDisk savings: {savings:.1f}%  (expected ~50%)")
    assert 40 < savings < 60, f"Expected ~50% savings, got {savings:.1f}%"

    # ── Step 2: verify INT4 round-trip quality ────────────────────────────────
    print("\n--- _dequantize_npy_dir() round-trip ---")
    tensor_dir = root / "tensors"
    rng = np.random.default_rng(42)
    for name in ("tensor_a", "tensor_b"):
        n, d = 64, 512
        arr_orig = rng.standard_normal((n, d)).astype(np.float32)

        arr_rec = _dequantize_npy_dir(tensor_dir, name)
        assert arr_rec.shape == (n, d), f"Shape mismatch: {arr_rec.shape}"

        cos = np.mean([
            np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
            for a, b in zip(arr_orig[:8], arr_rec[:8], strict=False)
        ])
        max_err = np.abs(arr_orig - arr_rec).max()
        print(f"  {name}: mean_cosine={cos:.5f}  max_err={max_err:.4f}")
        assert cos > 0.98, f"Cosine too low for INT4: {cos:.5f}"

    print("\nAll checks passed ✓")
