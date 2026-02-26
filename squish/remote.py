#!/usr/bin/env python3
"""squish_remote_manifest.py — Generate manifest.json for remote streaming.

Given a squish_4bit compressed model directory, generate a manifest.json
that allows clients to stream individual tensors via HTTP range requests.

Usage:
    python3 squish_remote_manifest.py \\
        --model-dir ~/models/Qwen2.5-7B-compressed \\
        --base-url  https://pub-ABC.r2.dev/qwen25-7b \\
        --output    /tmp/manifest.json

Then upload everything to your CDN:
    aws s3 cp --recursive ~/models/Qwen2.5-7B-compressed s3://bucket/qwen25-7b/ \\
        --endpoint-url https://ACCOUNT.r2.cloudflarestorage.com
"""

from __future__ import annotations
import argparse
import hashlib
import json
import os
import struct
import sys
from pathlib import Path
from typing import Generator


# ─── safetensors header parsing ──────────────────────────────────────────────

def _read_safetensors_metadata(path: Path) -> tuple[dict, int]:
    """Return (tensor_metadata_dict, header_length_bytes) for a safetensors file."""
    with open(path, "rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        header_json = f.read(header_len).decode("utf-8")
    meta = json.loads(header_json)
    return meta, 8 + header_len  # 8 bytes for the length prefix


def _safetensors_tensors(path: Path, base_url_file: str) -> Generator[dict, None, None]:
    """Yield per-tensor entries with HTTP-ready byte offsets for manifest."""
    meta, header_end = _read_safetensors_metadata(path)
    for name, info in meta.items():
        if name == "__metadata__":
            continue
        offsets = info.get("data_offsets", None)
        if offsets is None:
            continue
        abs_start = header_end + offsets[0]
        abs_end   = header_end + offsets[1]
        yield {
            "name":         name,
            "dtype":        info.get("dtype", "F32"),
            "shape":        info.get("shape", []),
            "data_offsets": [abs_start, abs_end],
        }


# ─── npy-dir scanning ────────────────────────────────────────────────────────

def _npy_dir_tensors(tensors_dir: Path, base_url_dir: str) -> list[dict]:
    """Build a virtual 'file' entry for all .npy files in a npy-dir."""
    entries = []
    for npy_file in sorted(tensors_dir.glob("*.npy")):
        arr_bytes = npy_file.stat().st_size
        # Each .npy has a small header; data starts right after
        with open(npy_file, "rb") as f:
            magic = f.read(6)
            if magic != b"\x93NUMPY":
                continue
            f.read(2)  # version
            header_len = struct.unpack("<H", f.read(2))[0]
            data_start = 10 + header_len
        entries.append({
            "filename": npy_file.name,
            "url":      f"{base_url_dir}/{npy_file.name}",
            "size_bytes": arr_bytes,
            "data_start": data_start,
            "data_end":   arr_bytes,
        })
    return entries


# ─── sha256 helper ───────────────────────────────────────────────────────────

def _sha256(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


# ─── main manifest builder ───────────────────────────────────────────────────

def build_manifest(
    model_dir: Path,
    base_url: str,
    include_sha256: bool = True,
) -> dict:
    """Build a manifest.json for the compressed model at model_dir."""
    base_url = base_url.rstrip("/")
    files = []
    total_bytes = 0

    # Detect format
    safetensors_files = sorted(model_dir.glob("*.safetensors"))
    tensors_dir = model_dir / "tensors"

    if safetensors_files:
        # squish_4bit / MLX safetensors
        for sf in safetensors_files:
            print(f"  scanning {sf.name} ...", end=" ", flush=True)
            size = sf.stat().st_size
            total_bytes += size
            tensors = list(_safetensors_tensors(sf, f"{base_url}/{sf.name}"))
            entry = {
                "filename":   sf.name,
                "url":        f"{base_url}/{sf.name}",
                "size_bytes": size,
                "tensors":    tensors,
            }
            if include_sha256:
                entry["sha256"] = _sha256(sf)
                print(f"{len(tensors)} tensors, {size/1e6:.0f} MB ✓")
            else:
                print(f"{len(tensors)} tensors, {size/1e6:.0f} MB")
            files.append(entry)

    elif tensors_dir.exists():
        # npy-dir format (INT8 quantized)
        npy_files = sorted(tensors_dir.glob("*.npy"))
        print(f"  scanning {len(npy_files)} .npy files ...")
        npy_entries = _npy_dir_tensors(tensors_dir, f"{base_url}/tensors")
        for entry in npy_entries:
            sz = (model_dir / "tensors" / entry["filename"]).stat().st_size
            total_bytes += sz
            if include_sha256:
                entry["sha256"] = _sha256(model_dir / "tensors" / entry["filename"])
        files.append({
            "filename": "tensors/",
            "url":      f"{base_url}/tensors/",
            "format":   "npy_dir",
            "items":    npy_entries,
        })

    else:
        print("ERROR: no safetensors files or tensors/ directory found", file=sys.stderr)
        sys.exit(1)

    # Collect tokenizer / config URLs
    ancillary = []
    for name in ["tokenizer.json", "tokenizer_config.json", "config.json",
                 "generation_config.json", "special_tokens_map.json",
                 "tokenizer.model", ".squish_ready", "manifest.json"]:
        p = model_dir / name
        if p.exists():
            ancillary.append({"filename": name, "url": f"{base_url}/{name}"})

    model_name = model_dir.name
    manifest = {
        "squish_version":  "0.2",
        "model_name":      model_name,
        "total_size_bytes": total_bytes,
        "files":           files,
        "ancillary":       ancillary,
    }
    return manifest


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-dir",  required=True, type=Path, help="Path to squish compressed model dir")
    p.add_argument("--base-url",   required=True, help="Public base URL where files will be hosted")
    p.add_argument("--output",     default=None,  help="Output manifest path (default: <model-dir>/manifest.json)")
    p.add_argument("--no-sha256",  action="store_true", help="Skip SHA-256 hashing (faster, less secure)")
    args = p.parse_args()

    model_dir = args.model_dir.expanduser().resolve()
    if not model_dir.is_dir():
        print(f"ERROR: {model_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    output = Path(args.output).expanduser().resolve() if args.output else model_dir / "manifest.json"

    print(f"Building manifest for: {model_dir}")
    print(f"Base URL: {args.base_url}")
    print(f"SHA-256: {'disabled' if args.no_sha256 else 'enabled'}")
    print()

    manifest = build_manifest(model_dir, args.base_url, include_sha256=not args.no_sha256)

    with open(output, "w") as f:
        json.dump(manifest, f, indent=2)

    n_tensors = sum(len(fi.get("tensors", [])) for fi in manifest["files"])
    print(f"\nManifest written to: {output}")
    print(f"  {n_tensors} tensors across {len(manifest['files'])} file(s)")
    print(f"  Total size: {manifest['total_size_bytes'] / 1e9:.2f} GB")
    print()
    print("Next steps:")
    print(f'  aws s3 cp --recursive "{model_dir}" "s3://BUCKET/PATH/" --endpoint-url https://ACCOUNT.r2.cloudflarestorage.com')
    print(f'  python3 squish_server.py --manifest {args.base_url}/manifest.json')


if __name__ == "__main__":
    main()
