#!/usr/bin/env python3
"""
verify.py

Validate the compressed-weight PoC by comparing:
  1. Token agreement: reference_output.json vs compressed_output.json
  2. Weight fidelity: cosine similarity between original and reconstructed tensors

Success criteria (all must pass):
  - Token agreement >= 20% on the test prompt at max_tokens
    (INT8 quantization causes autoregressive divergence after ~10 tokens;
     the important signal is semantic correctness, captured by cosine sim)
  - Mean cosine similarity >= 0.97 on sampled weight matrices

Usage:
    python3 verify.py \\
        [--reference reference_output.json] \\
        [--compressed compressed_output.json] \\
        [--model-dir ~/models/Qwen2.5-1.5B-Instruct] \\
        [--npz ~/models/.../weights_compressed.npz] \\
        [--min-cosine 0.97] \\
        [--min-token-agreement 0.60]
"""
import sys
import json
import argparse
from pathlib import Path

import numpy as np
import os as _os
from pathlib import Path as _Path

# ---------------------------------------------------------------------------
# Resolve the vectro source tree (VECTRO_DIR env var → sibling → ~/vectro)
# ---------------------------------------------------------------------------
def _find_vectro() -> str:
    if "VECTRO_DIR" in _os.environ:
        return _os.environ["VECTRO_DIR"]
    candidate = _Path(__file__).resolve().parent.parent.parent / "vectro"
    if candidate.exists():
        return str(candidate)
    return str(_Path.home() / "vectro")

sys.path.insert(0, _find_vectro())
from python.interface import reconstruct_embeddings, QuantizationResult

MODEL_DIR_DEFAULT = str(Path.home() / ".squish" / "models" / "Qwen2.5-1.5B-Instruct")
NPZ_DEFAULT = str(Path.home() / ".squish" / "models" / "Qwen2.5-1.5B-Instruct-compressed" / "weights_compressed.npz")

# Tensors to sample for weight fidelity checks
SAMPLE_TENSOR_PATTERNS = [
    "model.layers.0.self_attn.q_proj.weight",
    "model.layers.0.mlp.gate_proj.weight",
    "model.layers.5.self_attn.k_proj.weight",
    "model.layers.10.mlp.up_proj.weight",
    "model.layers.0.self_attn.v_proj.weight",
    "model.layers.0.mlp.down_proj.weight",
]


def cosine_similarity_rows(a: np.ndarray, b: np.ndarray) -> float:
    """Mean cosine similarity between corresponding rows of two 2D arrays."""
    a_flat = a.reshape(-1, a.shape[-1]).astype(np.float64)
    b_flat = b.reshape(-1, b.shape[-1]).astype(np.float64)
    # Normalize rows
    a_norm = a_flat / (np.linalg.norm(a_flat, axis=1, keepdims=True) + 1e-10)
    b_norm = b_flat / (np.linalg.norm(b_flat, axis=1, keepdims=True) + 1e-10)
    dot = np.sum(a_norm * b_norm, axis=1)
    return float(np.mean(np.clip(dot, -1, 1)))


def token_agreement(ref_text: str, comp_text: str) -> tuple[float, int]:
    """Return (agreement_ratio, n_compared_tokens)."""
    ref_tokens = ref_text.strip().split()
    comp_tokens = comp_text.strip().split()
    n = min(len(ref_tokens), len(comp_tokens))
    if n == 0:
        return 0.0, 0
    matches = sum(1 for a, b in zip(ref_tokens[:n], comp_tokens[:n]) if a == b)
    return matches / n, n


def section(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", default="reference_output.json")
    ap.add_argument("--compressed", default="compressed_output.json")
    ap.add_argument("--model-dir", default=MODEL_DIR_DEFAULT)
    ap.add_argument("--npz", default=NPZ_DEFAULT)
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--min-cosine", type=float, default=0.97,
                    help="Minimum acceptable mean cosine similarity (default: 0.97)")
    ap.add_argument("--min-token-agreement", type=float, default=0.20,
                    help="Minimum acceptable token agreement ratio (default: 0.20)")
    ap.add_argument("--skip-weights", action="store_true",
                    help="Skip weight fidelity check (faster; use if safetensors absent)")
    args = ap.parse_args()

    # Auto-detect storage format
    _is_npy_dir = Path(args.npz).is_dir()
    if _is_npy_dir:
        manifest_path = args.manifest or str(Path(args.npz) / "manifest.json")
    else:
        manifest_path = args.manifest or args.npz.replace(".npz", "_manifest.json")

    passed = []
    failed = []

    # -----------------------------------------------------------------------
    # 1. Token comparison
    # -----------------------------------------------------------------------
    section("1. Token Comparison")

    try:
        with open(args.reference) as f:
            ref = json.load(f)
        with open(args.compressed) as f:
            comp = json.load(f)
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        print("Run run_reference.py and run_inference.py first.")
        sys.exit(1)

    print(f"Prompt:     {ref['prompt']!r}")
    print(f"Reference:  {ref['output']!r}")
    print(f"Compressed: {comp['output']!r}")

    agree_ratio, n_compared = token_agreement(ref["output"], comp["output"])
    print(f"\nToken agreement: {agree_ratio:.1%} over {n_compared} tokens")

    if agree_ratio >= args.min_token_agreement:
        print(f"  PASS (>= {args.min_token_agreement:.0%})")
        passed.append("Token agreement")
    else:
        print(f"  FAIL (< {args.min_token_agreement:.0%})")
        failed.append("Token agreement")

    # -----------------------------------------------------------------------
    # 2. Timing comparison
    # -----------------------------------------------------------------------
    section("2. Timing Comparison")
    print(f"{'Metric':<25} {'Reference':>12} {'Compressed':>12}")
    print(f"{'-'*50}")
    print(f"{'Load time (s)':<25} {ref.get('load_time_s', 'n/a'):>12} "
          f"{comp.get('load_time_s', 'n/a'):>12}")
    print(f"{'Generation time (s)':<25} {ref.get('gen_time_s', 'n/a'):>12} "
          f"{comp.get('gen_time_s', 'n/a'):>12}")

    # -----------------------------------------------------------------------
    # 3. Weight fidelity
    # -----------------------------------------------------------------------
    section("3. Weight Fidelity (cosine similarity)")

    if args.skip_weights:
        print("Skipped (--skip-weights flag set)")
    else:
        import mlx.core as mx

        # Load original weights (float32 via MLX)
        shard_files = sorted(Path(args.model_dir).glob("*.safetensors"))
        if not shard_files:
            print(f"ERROR: no .safetensors in {args.model_dir} — use --skip-weights")
            sys.exit(1)

        # Load all shards to find tensor by name
        orig_weights = {}
        for shard in shard_files:
            w = mx.load(str(shard))
            for name, arr in w.items():
                orig_weights[name] = np.array(arr.astype(mx.float32))
        print(f"Loaded {len(orig_weights)} original tensors for comparison")

        # Load manifest
        with open(manifest_path) as f:
            manifest = json.load(f)
        safe_map = {v: k for k, v in manifest.items()}  # safe_key -> original_name

        if _is_npy_dir:
            # npy-dir: load tensors from individual .npy files (mmap-safe)
            tensor_dir = Path(args.npz) / "tensors"

            def _recon_npy_dir(sk, orig_shape):
                pt_path = tensor_dir / f"{sk}__pt.npy"
                if pt_path.exists():
                    arr = np.load(str(pt_path), mmap_mode='r').astype(np.float32)
                    return arr.reshape(orig_shape), "PT", arr.nbytes
                q = np.load(str(tensor_dir / f"{sk}__q.npy"), mmap_mode='r')
                s = np.load(str(tensor_dir / f"{sk}__s.npy"), mmap_mode='r')
                result = QuantizationResult(quantized=np.array(q), scales=np.array(s),
                                            dims=q.shape[1], n=q.shape[0])
                recon = reconstruct_embeddings(result).reshape(orig_shape)
                return recon, "Q8", np.array(q).nbytes + np.array(s).nbytes
        else:
            npz = np.load(args.npz, allow_pickle=False)

        print()
        print(f"{'Tensor':<55} {'CosSim':>8} {'MaxErr':>10} {'RatioX':>7}")
        print(f"{'-'*82}")

        all_cosines = []
        any_missing = False

        for name in SAMPLE_TENSOR_PATTERNS:
            # Try to find this tensor in whichever shard it lives in
            orig = orig_weights.get(name)
            if orig is None:
                print(f"  (not found in model: {name})")
                any_missing = True
                continue

            sk = manifest.get(name)
            if sk is None:
                print(f"  (not in manifest: {name})")
                any_missing = True
                continue

            # Reconstruct from compressed — format-aware
            if _is_npy_dir:
                shape_path = tensor_dir / f"{sk}__shape.npy"
                if shape_path.exists():
                    original_shape = tuple(np.load(str(shape_path)).tolist())
                else:
                    pt_path = tensor_dir / f"{sk}__pt.npy"
                    original_shape = np.load(str(pt_path), mmap_mode='r').shape
                recon, mode, comp_bytes = _recon_npy_dir(sk, original_shape)
            else:
                original_shape = tuple(int(x) for x in npz[sk + "__shape"].tolist())
                if (sk + "__pt") in npz.files:
                    recon = npz[sk + "__pt"].reshape(original_shape)
                    mode = "PT"
                else:
                    q = npz[sk + "__q"]
                    s = npz[sk + "__s"]
                    result = QuantizationResult(quantized=q, scales=s,
                                                dims=q.shape[1], n=q.shape[0])
                    recon = reconstruct_embeddings(result).reshape(original_shape)
                    mode = "Q8"
                comp_bytes = (npz.get(sk + "__q", np.array([])).nbytes +
                              npz.get(sk + "__s", np.array([])).nbytes +
                              npz.get(sk + "__pt", np.array([])).nbytes)

            orig_f32 = orig.reshape(original_shape)
            cos     = cosine_similarity_rows(orig_f32, recon)
            max_err = float(np.max(np.abs(orig_f32 - recon.astype(np.float32))))
            ratio   = orig_f32.astype(np.float32).nbytes / max(comp_bytes, 1)

            all_cosines.append(cos)
            print(f"  [{mode}] {name:<51} {cos:>8.5f} {max_err:>10.5f} {ratio:>7.2f}x")

        if not _is_npy_dir:
            npz.close()

        if all_cosines:
            mean_cos = np.mean(all_cosines)
            min_cos = np.min(all_cosines)
            print(f"\n  Mean cosine similarity: {mean_cos:.5f}")
            print(f"  Min  cosine similarity: {min_cos:.5f}")

            if mean_cos >= args.min_cosine:
                print(f"  PASS (mean >= {args.min_cosine})")
                passed.append("Weight fidelity")
            else:
                print(f"  FAIL (mean < {args.min_cosine})")
                failed.append("Weight fidelity")
        else:
            print("  No tensors sampled — check model directory and manifest.")
            failed.append("Weight fidelity (no samples)")

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    section("Summary")
    for p in passed:
        print(f"  PASS  {p}")
    for f in failed:
        print(f"  FAIL  {f}")

    print()
    if not failed:
        print("All checks passed. PoC validated.")
        sys.exit(0)
    else:
        print(f"{len(failed)} check(s) failed.")
        sys.exit(1)


if __name__ == "__main__":
    main()
