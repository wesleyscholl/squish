#!/usr/bin/env python3
"""
tools/compress_weights.py

CLI tool to zstd-compress an existing Squish npy-dir — Phase 1.1.

After running this tool the directory is smaller on disk.  All subsequent
calls to load_from_npy_dir() transparently decompress .npy.zst files at
load time using the _load_npy_path() helper (which already handles .zst).

Usage:
    python3 tools/compress_weights.py --dir ~/models/squish_7b
    python3 tools/compress_weights.py --dir ~/models/squish_7b --level 6 --dry-run
    python3 tools/compress_weights.py --dir ~/models/squish_7b --undo

Compression levels:
    1   Fastest (~2s for 7B),  ~15-25% savings
    3   Default (~5s for 7B),  ~20-30% savings  ← recommended
    9   Slow (~60s for 7B),    ~22-32% savings
    19  Maximum (~10min),      ~24-35% savings
"""

import argparse
import sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(
        description="Compress or decompress a Squish npy-dir weights directory"
    )
    ap.add_argument("--dir",     required=True,
                    help="Path to npy-dir root (contains tensors/ or .npy files directly)")
    ap.add_argument("--level",   type=int, default=3,
                    help="Zstd compression level 1-22 (default 3)")
    ap.add_argument("--threads", type=int, default=-1,
                    help="CPU threads for compression (-1=all, default -1)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print what would happen without touching any files")
    ap.add_argument("--undo",    action="store_true",
                    help="Decompress .npy.zst files back to .npy (reverses compression)")
    ap.add_argument("--verbose", action="store_true", default=True)
    args = ap.parse_args()

    target = Path(args.dir).expanduser().resolve()
    if not target.exists():
        print(f"ERROR: directory not found: {target}")
        sys.exit(1)

    # ── Decompression mode ────────────────────────────────────────────────────
    if args.undo:
        try:
            import zstandard as zstd
        except ImportError:
            print("ERROR: pip install zstandard")
            sys.exit(1)

        zst_files = sorted(target.rglob("*.npy.zst"))
        if not zst_files:
            print(f"No .npy.zst files found under {target}")
            sys.exit(0)

        print(f"Decompressing {len(zst_files)} files under {target} …")
        dctx     = zstd.ZstdDecompressor()
        n_done   = 0
        bytes_in = bytes_out = 0

        for f in zst_files:
            npy_path = Path(str(f)[:-4])  # strip .zst
            if npy_path.exists() and not args.dry_run:
                print(f"  [skip] {f.name} (original already exists)")
                continue
            if args.dry_run:
                print(f"  [dry] would decompress {f.name}")
                continue
            raw = dctx.decompress(f.read_bytes())
            npy_path.write_bytes(raw)
            bytes_in  += f.stat().st_size
            bytes_out += len(raw)
            f.unlink()
            n_done += 1

        if not args.dry_run:
            print(f"\nDecompressed {n_done} files  "
                  f"({bytes_in/1e6:.0f} MB → {bytes_out/1e6:.0f} MB)")
        return

    # ── Compression mode ──────────────────────────────────────────────────────
    if args.dry_run:
        # Dry run: measure only, don't write anything
        try:
            import zstandard as zstd
        except ImportError:
            print("ERROR: pip install zstandard")
            sys.exit(1)

        npy_files = sorted(target.rglob("*.npy"))
        if not npy_files:
            print(f"No .npy files found under {target}")
            sys.exit(0)

        cctx     = zstd.ZstdCompressor(level=args.level, threads=0)
        total_raw = total_comp = 0

        for f in npy_files:
            raw  = f.read_bytes()
            comp = cctx.compress(raw)
            total_raw  += len(raw)
            total_comp += len(comp)
            if args.verbose:
                ratio = len(comp) / len(raw) * 100
                print(f"  {f.name}  {len(raw)//1024} KB → {len(comp)//1024} KB  "
                      f"({ratio:.0f}%)")

        savings = (1 - total_comp / total_raw) * 100 if total_raw else 0
        print(f"\n[DRY RUN] level={args.level}: "
              f"{total_raw/1e6:.0f} MB → {total_comp/1e6:.0f} MB  "
              f"({savings:.0f}% savings)")
        print("Rerun without --dry-run to apply.")
        return

    # Real compression via squish.compressed_loader.compress_npy_dir()
    try:
        from squish.compressed_loader import compress_npy_dir
    except ImportError as e:
        print(f"ERROR: could not import squish.compressed_loader: {e}")
        sys.exit(1)

    print(f"Compressing {target}  (level={args.level}) …")
    stats = compress_npy_dir(
        npy_dir  = str(target),
        level    = args.level,
        threads  = args.threads,
        verbose  = args.verbose,
    )

    print(f"\nDone.")
    print(f"  Compressed : {stats['n_compressed']}")
    print(f"  Skipped    : {stats['n_skipped']}")
    print(f"  Before     : {stats['bytes_before']/1e6:.1f} MB")
    print(f"  After      : {stats['bytes_after']/1e6:.1f} MB")
    print(f"  Savings    : {stats['savings_pct']:.1f}%")
    print(f"  Time       : {stats['elapsed_s']:.1f}s")


if __name__ == "__main__":
    main()
