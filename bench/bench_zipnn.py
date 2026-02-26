#!/usr/bin/env python3
"""
bench/bench_zipnn.py

Phase 0.3 — Measure Zstandard compression ratios on an existing model weights
directory without modifying any files.

Produces a table showing per-tensor-type savings so you can decide whether
ZipNN / zstd integration is worth the decompression overhead for your workflow.

Usage:
    python3 bench/bench_zipnn.py --dir ~/models/squish_7b/tensors
    python3 bench/bench_zipnn.py --dir ~/models/squish_7b --levels 1 3 9
    python3 bench/bench_zipnn.py --dir ~/models/squish_4bit --sample 50
"""

import argparse
import io
import time
from pathlib import Path

import numpy as np


def _try_import_zstd():
    try:
        import zstandard as zstd
        return zstd
    except ImportError:
        print("ERROR: Install zstandard first:  pip install zstandard")
        import sys; sys.exit(1)


def _classify(name: str) -> str:
    """Return a short tensor-type label based on filename suffix."""
    n = name.lower()
    if n.endswith("__q4.npy"):  return "INT4-q"
    if n.endswith("__s4.npy"):  return "INT4-s"
    if n.endswith("__q.npy"):   return "INT8-q"
    if n.endswith("__s.npy"):   return "INT8-s"
    if n.endswith("__pt.npy"):  return "FP16-pt"
    if n.endswith("__shape.npy"): return "shape"
    return "other"


def run(dir_path: Path, levels: list, sample: int, verbose: bool) -> None:
    zstd = _try_import_zstd()

    npy_files = sorted(dir_path.rglob("*.npy"))
    if not npy_files:
        print(f"No .npy files found under {dir_path}")
        return

    if sample and sample < len(npy_files):
        import random
        random.seed(42)
        npy_files = random.sample(npy_files, sample)
        print(f"Sampling {sample}/{len(npy_files)} files for speed.\n")

    print(f"Scanning {len(npy_files)} .npy files in {dir_path}")
    print(f"{'Level':>5}  {'Type':>8}  {'Files':>6}  "
          f"{'Raw MB':>8}  {'Comp MB':>8}  {'Ratio':>7}  {'MB/s':>7}")
    print("─" * 60)

    for level in sorted(levels):
        cctx = zstd.ZstdCompressor(level=level, threads=0)
        by_type: dict = {}

        t0 = time.perf_counter()
        for fpath in npy_files:
            label   = _classify(fpath.name)
            raw     = fpath.read_bytes()
            comp    = cctx.compress(raw)
            entry   = by_type.setdefault(label, {"raw": 0, "comp": 0, "n": 0})
            entry["raw"]  += len(raw)
            entry["comp"] += len(comp)
            entry["n"]    += 1
        elapsed = time.perf_counter() - t0

        total_raw  = sum(e["raw"]  for e in by_type.values())
        total_comp = sum(e["comp"] for e in by_type.values())
        total_n    = sum(e["n"]    for e in by_type.values())
        mb_per_s   = total_raw / elapsed / 1e6

        for label, e in sorted(by_type.items()):
            ratio = e["comp"] / e["raw"] * 100 if e["raw"] else 0
            print(f"  {level:>3}   {label:>8}   {e['n']:>5}  "
                  f"{e['raw']/1e6:>8.1f}  {e['comp']/1e6:>8.1f}  "
                  f"{ratio:>6.1f}%  {'':>7}")
        overall_ratio = total_comp / total_raw * 100 if total_raw else 0
        savings = 100 - overall_ratio
        print(f"  {level:>3}   {'TOTAL':>8}   {total_n:>5}  "
              f"{total_raw/1e6:>8.1f}  {total_comp/1e6:>8.1f}  "
              f"{overall_ratio:>6.1f}%  {mb_per_s:>7.0f}")
        print(f"        → level={level}: {savings:.1f}% size savings  "
              f"({total_raw/1e6:.0f} MB → {total_comp/1e6:.0f} MB)")
        print()

    # Decompression speed estimate
    print("Decompression speed estimate:")
    sample_file = npy_files[0]
    raw   = sample_file.read_bytes()
    cctx2 = zstd.ZstdCompressor(level=3, threads=0)
    comp  = cctx2.compress(raw)
    dctx  = zstd.ZstdDecompressor()
    N = max(1, min(20, len(npy_files)))

    t0 = time.perf_counter()
    for f in npy_files[:N]:
        c = cctx2.compress(f.read_bytes())
        dctx.decompress(c)
    elapsed = time.perf_counter() - t0
    total_bytes = sum(f.stat().st_size for f in npy_files[:N])
    print(f"  ~{total_bytes / elapsed / 1e9:.2f} GB/s raw throughput  "
          f"(over {N} files, level=3)")
    print()
    print("Verdict:")
    cctx3 = zstd.ZstdCompressor(level=3, threads=0)
    all_raw   = sum(f.stat().st_size for f in npy_files)
    all_comp  = sum(len(cctx3.compress(f.read_bytes())) for f in npy_files)
    sav_pct   = (1 - all_comp / all_raw) * 100 if all_raw else 0
    decom_gbs = total_bytes / elapsed / 1e9
    load_time_without = all_raw / 2e9   # assume 2 GB/s NVMe/SSD
    load_time_with    = all_comp / 2e9 + all_raw / decom_gbs / 1e9
    print(f"  Disk space :  {all_raw/1e6:.0f} MB → {all_comp/1e6:.0f} MB  "
          f"({sav_pct:.0f}% saved)")
    print(f"  Load time  :  ~{load_time_without:.1f}s uncompressed  "
          f"vs  ~{load_time_with:.1f}s with zstd level=3")
    print()
    if sav_pct >= 15:
        print("  RECOMMENDATION: Enable zstd compression "
              f"(run: python3 -m squish.tools.compress_weights --dir {dir_path})")
    else:
        print("  RECOMMENDATION: Compression savings too small — keep as-is.")


def main():
    ap = argparse.ArgumentParser(
        description="Measure zstd compression ratios on a Squish npy-dir (dry run)"
    )
    ap.add_argument("--dir",     required=True, help="Path containing .npy files")
    ap.add_argument("--levels",  nargs="+", type=int, default=[1, 3, 9],
                    help="Zstd levels to benchmark (default: 1 3 9)")
    ap.add_argument("--sample",  type=int, default=0,
                    help="Random sample N files (0=all)")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    run(Path(args.dir).expanduser(), args.levels, args.sample, args.verbose)


if __name__ == "__main__":
    main()
