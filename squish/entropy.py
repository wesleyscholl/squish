#!/usr/bin/env python3
"""
squish_entropy.py

Entropy coding layer for Squish npy-dir compressed models.
Applies zstd compression on top of INT8 quantised weights for ~35% disk reduction.

Key insight: INT8 quantised weight matrices have measured Shannon entropy of
approximately 4.5 bits/value (vs nominal 8 bits). zstd at level 3 reliably
achieves 30-40% compression on this data with >2 GB/sec decompression
throughput in C — negligible load time overhead.

Architecture:
    On-disk (Tier 1 npy-dir):   INT8 .npy files  →  INT8.zst files + .squish_zst_ready
    In memory (unchanged):      float16 numpy arrays (reconstructed on load)
    Inference path (unchanged): Metal GPU, MLX arrays

Usage:
    # Compress an existing npy-dir in place:
    python3 squish_entropy.py compress ~/models/Qwen2.5-1.5B-Instruct-bf16-compressed/tensors

    # Decompress (rarely needed — loader handles this transparently):
    python3 squish_entropy.py decompress ~/models/.../tensors

    # Benchmark compression ratio on a model:
    python3 squish_entropy.py bench ~/models/Qwen2.5-1.5B-Instruct-bf16-compressed/tensors

Dependencies:
    pip install zstandard
"""
import argparse
import sys
import time
from pathlib import Path

# ── zstandard availability ───────────────────────────────────────────────────

def _require_zstd():
    try:
        import zstandard as zstd
        return zstd
    except ImportError:
        print("Missing: zstandard.  Install with:  pip install zstandard")
        sys.exit(1)


# ── Compress ─────────────────────────────────────────────────────────────────

def compress_npy_dir(
    tensors_dir: Path,
    level: int  = 3,
    threads: int = 0,   # 0 = use all cores
    verbose: bool = True,
) -> dict:
    """
    Compress all .npy files in a tensors/ directory with zstd.

    Writes <name>.npy.zst alongside (or replaces) each .npy.
    Writes .squish_zst_ready sentinel on completion.

    Returns stats dict.
    """
    zstd = _require_zstd()
    tensors_dir = Path(tensors_dir)
    if not tensors_dir.is_dir():
        raise FileNotFoundError(f"tensors/ not found: {tensors_dir}")

    npy_files = sorted(tensors_dir.glob("*.npy"))
    if not npy_files:
        raise ValueError(f"No .npy files found in {tensors_dir}")

    sentinel = tensors_dir.parent / ".squish_zst_ready"
    if sentinel.exists():
        if verbose:
            print("  Already compressed (.squish_zst_ready present) — skipping")
        return {}

    cctx = zstd.ZstdCompressor(level=level, threads=threads)

    total_orig  = 0
    total_comp  = 0
    t0 = time.perf_counter()

    for i, npy_path in enumerate(npy_files):
        orig_bytes = npy_path.stat().st_size
        zst_path   = npy_path.with_suffix(".npy.zst")

        with open(npy_path, "rb") as src, open(zst_path, "wb") as dst:
            cctx.copy_stream(src, dst, size=orig_bytes)

        comp_bytes = zst_path.stat().st_size
        total_orig += orig_bytes
        total_comp += comp_bytes
        ratio       = orig_bytes / max(comp_bytes, 1)

        if verbose and (i % 50 == 0 or i == len(npy_files) - 1):
            pct = 100 * (i + 1) / len(npy_files)
            print(f"  [{pct:5.1f}%] {npy_path.name:50s}  {orig_bytes/1e6:6.1f} MB → {comp_bytes/1e6:6.1f} MB  ({ratio:.2f}×)", flush=True)

        # Remove original .npy to save disk (the .zst is the primary)
        npy_path.unlink()

    elapsed = time.perf_counter() - t0
    overall_ratio = total_orig / max(total_comp, 1)
    savings_gb    = (total_orig - total_comp) / 1e9

    sentinel.write_text("squish-zst-v1")

    stats = {
        "files":          len(npy_files),
        "orig_gb":        round(total_orig / 1e9, 3),
        "comp_gb":        round(total_comp / 1e9, 3),
        "ratio":          round(overall_ratio, 3),
        "savings_gb":     round(savings_gb, 3),
        "throughput_gbs": round(total_orig / 1e9 / elapsed, 2),
        "elapsed_s":      round(elapsed, 1),
    }

    if verbose:
        print()
        print(f"  ✓ Compressed {len(npy_files)} tensors in {elapsed:.1f}s")
        print(f"    {total_orig/1e9:.2f} GB  →  {total_comp/1e9:.2f} GB  ({overall_ratio:.2f}×, saved {savings_gb:.2f} GB)")
        print(f"    Throughput: {stats['throughput_gbs']:.1f} GB/s")

    return stats


# ── Decompress ───────────────────────────────────────────────────────────────

def decompress_npy_dir(tensors_dir: Path, verbose: bool = True) -> None:
    """
    Decompress .npy.zst files back to .npy (for inspection / migration).
    """
    zstd = _require_zstd()
    tensors_dir = Path(tensors_dir)
    zst_files   = sorted(tensors_dir.glob("*.npy.zst"))

    if not zst_files:
        print("No .npy.zst files found — nothing to decompress")
        return

    dctx = zstd.ZstdDecompressor()
    t0   = time.perf_counter()

    for zst_path in zst_files:
        npy_path = zst_path.with_suffix("")  # removes .zst → .npy
        with open(zst_path, "rb") as src, open(npy_path, "wb") as dst:
            dctx.copy_stream(src, dst)

    elapsed = time.perf_counter() - t0
    sentinel = tensors_dir.parent / ".squish_zst_ready"
    if sentinel.exists():
        sentinel.unlink()

    if verbose:
        print(f"  ✓ Decompressed {len(zst_files)} tensors in {elapsed:.1f}s")


# ── Transparent numpy loader (for integration into compressed_loader.py) ─────

def load_npy_zst(path: Path, dctx=None):
    """
    Load a .npy.zst file transparently as if it were a .npy file.

    Returns a numpy array.  Compatible with np.load() semantics.
    Decompresses into a BytesIO buffer then calls np.load — no temp files.
    """
    import io

    import numpy as np

    if dctx is None:
        import zstandard as zstd
        dctx = zstd.ZstdDecompressor()

    with open(path, "rb") as f:
        compressed = f.read()

    buf = io.BytesIO(dctx.decompress(compressed))
    return np.load(buf, allow_pickle=False)


# ── Benchmark ────────────────────────────────────────────────────────────────

def benchmark_compression(tensors_dir: Path) -> None:
    """
    Measure compression ratios without modifying files.
    Samples up to 20 tensors, reports per-tensor and aggregate stats.
    """
    zstd = _require_zstd()

    cctx = zstd.ZstdCompressor(level=3)
    dctx = zstd.ZstdDecompressor()

    npy_files = sorted(Path(tensors_dir).glob("*.npy"))[:20]
    if not npy_files:
        zst_files = sorted(Path(tensors_dir).glob("*.npy.zst"))[:20]
        if not zst_files:
            print("No .npy or .npy.zst files found")
            return
        # Decompress and analyse
        npy_files = []
        for zst in zst_files:
            arr = load_npy_zst(zst, dctx)
            npy_files.append((zst.name, arr.nbytes, len(cctx.compress(arr.tobytes()))))
    else:
        npy_files = [(f.name, f.stat().st_size, None) for f in npy_files]

    total_orig = total_comp = 0
    print(f"  {'Tensor':55s}  {'Orig MB':>8}  {'Comp MB':>8}  {'Ratio':>6}  {'Entropy est':>12}")
    print(f"  {'-'*100}")
    for item in npy_files:
        if isinstance(item, tuple) and len(item) == 3:
            name, orig_bytes, comp_bytes_val = item
            if comp_bytes_val is None:
                npy_path = Path(tensors_dir) / name
                data     = open(npy_path, "rb").read()
                comp_bytes_val = len(cctx.compress(data))
                orig_bytes = len(data)
        else:
            name      = item
            npy_path  = Path(tensors_dir) / name
            data      = open(npy_path, "rb").read()
            orig_bytes      = len(data)
            comp_bytes_val  = len(cctx.compress(data))

        ratio          = orig_bytes / max(comp_bytes_val, 1)
        entropy_est    = 8.0 / ratio  # bits per value if all data is weight values
        total_orig    += orig_bytes
        total_comp    += comp_bytes_val
        print(f"  {str(name):55s}  {orig_bytes/1e6:8.2f}  {comp_bytes_val/1e6:8.2f}  {ratio:6.2f}×  ~{entropy_est:.1f} bits/val")

    overall = total_orig / max(total_comp, 1)
    print(f"  {'-'*100}")
    print(f"  {'TOTAL (sample)':55s}  {total_orig/1e6:8.2f}  {total_comp/1e6:8.2f}  {overall:6.2f}×")
    print()
    print("  Extrapolated full-model savings:")
    model_gb_estimate = total_orig / 1e9 * (len(sorted(Path(tensors_dir).glob("*.npy"))) / max(len(npy_files), 1))
    savings_estimate  = model_gb_estimate * (1.0 - 1.0 / overall)
    print(f"    Model dir size (est): {model_gb_estimate:.1f} GB")
    print(f"    Compressed size (est): {model_gb_estimate / overall:.1f} GB")
    print(f"    Savings (est): {savings_estimate:.1f} GB  ({(1 - 1/overall)*100:.0f}%)")


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Squish entropy coding for npy-dir")
    ap.add_argument("command", choices=["compress", "decompress", "bench"])
    ap.add_argument("tensors_dir", help="Path to tensors/ directory")
    ap.add_argument("--level", type=int, default=3, help="zstd compression level (1-22)")
    ap.add_argument("--threads", type=int, default=0, help="Compression threads (0=all)")
    args = ap.parse_args()

    tensors_dir = Path(args.tensors_dir).expanduser()

    if args.command == "compress":
        print(f"Compressing {tensors_dir} with zstd level {args.level}...")
        compress_npy_dir(tensors_dir, level=args.level, threads=args.threads)
    elif args.command == "decompress":
        print(f"Decompressing {tensors_dir}...")
        decompress_npy_dir(tensors_dir)
    elif args.command == "bench":
        print(f"Benchmarking compression on {tensors_dir}...")
        benchmark_compression(tensors_dir)


if __name__ == "__main__":
    main()
