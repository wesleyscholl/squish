#!/usr/bin/env python3
"""Reconstruct _manifest.json from existing .npz keys (no re-conversion needed)."""
import numpy as np
import json
import re
import sys
from pathlib import Path

npz_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.home() / "models/Qwen2.5-1.5B-Instruct-bf16-compressed/weights_compressed.npz"

print(f"Loading keys from {npz_path.name} ...")
npz = np.load(str(npz_path), allow_pickle=False)
keys = list(npz.files)
print(f"  {len(keys)} arrays in archive")

suffix_re = re.compile(r'__(q|s|shape|pt)$')
safe_keys = sorted({suffix_re.sub('', k) for k in keys if suffix_re.search(k)})
print(f"  {len(safe_keys)} unique tensors")

manifest = {sk.replace("__", "."): sk for sk in safe_keys}

out_path = str(npz_path).replace(".npz", "_manifest.json")
with open(out_path, "w") as f:
    json.dump(manifest, f, indent=2)

print(f"  Written: {out_path}")
for orig, sk in list(manifest.items())[:4]:
    print(f"    {orig}  →  {sk}")
print("  ✓ Done")
