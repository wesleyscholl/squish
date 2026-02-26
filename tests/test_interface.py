#!/usr/bin/env python3
"""Quick round-trip test for interface.py Rust paths + INT4 API."""
import sys
sys.path.insert(0, '/Users/wscholl/vectro')

import numpy as np
from python.interface import (
    quantize_embeddings, reconstruct_embeddings,
    quantize_int4, dequantize_int4, get_backend_info,
)

print("Backends:", get_backend_info())
print()

emb = np.random.randn(512, 768).astype(np.float32)

# INT8 round-trip (Rust quant + Rust dequant via squish_quant backend)
r = quantize_embeddings(emb)
rec = reconstruct_embeddings(r)
cos = np.mean([
    np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    for a, b in zip(emb[:10], rec[:10])
])
print(f"INT8 Rust quant+dequant  mean_cosine={cos:.7f}  max_err={np.abs(emb - rec).max():.5f}")

# INT4 round-trip
packed, scales = quantize_int4(emb, group_size=64)
rec4 = dequantize_int4(packed, scales, group_size=64)
cos4 = np.mean([
    np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    for a, b in zip(emb[:10], rec4[:10])
])
print(f"INT4 Rust quant+dequant  mean_cosine={cos4:.7f}  max_err={np.abs(emb - rec4).max():.5f}")
print(f"INT4 disk vs INT8:  {packed.nbytes:,} B vs {r.quantized.nbytes:,} B  ({packed.nbytes/r.quantized.nbytes:.0%})")
print()
print("All OK")
