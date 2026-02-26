#!/usr/bin/env python3
"""Benchmark current vectro quantizer vs vectorized numpy."""
import sys, time
import numpy as np
sys.path.insert(0, '/Users/wscholl/vectro')
from python.interface import quantize_embeddings, reconstruct_embeddings, get_backend_info

print("Backend:", get_backend_info())

# Large weight tensor typical of 14B MLP layer
shapes = [
    (13824, 5120, "14B mlp weight"),
    (4096, 4096, "7B attention"),
    (1024, 5120, "14B k_proj"),
]

for n, d, label in shapes:
    data = np.random.randn(n, d).astype(np.float32)

    # Current backend
    t0 = time.perf_counter()
    result = quantize_embeddings(data, backend='mojo')
    t1 = time.perf_counter()

    # Fully vectorized numpy
    t2 = time.perf_counter()
    scales_v = np.abs(data).max(axis=1, keepdims=True) / 127.0
    scales_v = np.where(scales_v == 0, 1.0, scales_v)
    q_v = np.clip(np.round(data / scales_v), -127, 127).astype(np.int8)
    t3 = time.perf_counter()

    print(f"\n{label} ({n}x{d}):")
    print(f"  Current (mojo-loop):       quantize={t1-t0:.3f}s")
    print(f"  Vectorized numpy:          quantize={t3-t2:.3f}s")
    print(f"  Speedup:                   {(t1-t0)/(max(t3-t2,1e-9)):.1f}x")

    # Reconstruct comparison
    t4 = time.perf_counter()
    r_curr = reconstruct_embeddings(result)
    t5 = time.perf_counter()

    t6 = time.perf_counter()
    r_vec = (q_v.astype(np.float32)) * scales_v
    t7 = time.perf_counter()

    print(f"  Current reconstruct:       {t5-t4:.3f}s")
    print(f"  Vectorized reconstruct:    {t7-t6:.3f}s")
    print(f"  Reconstruct speedup:       {(t5-t4)/(max(t7-t6,1e-9)):.1f}x")

# Group-wise INT4 quantization potential
print("\n=== INT4 group-wise (q_bits=4, group_size=64) potential ===")
n, d = 13824, 5120
data = np.random.randn(n, d).astype(np.float32)
# INT4 = 0.5 bytes/param vs INT8 = 1 byte/param => 2x smaller
t0 = time.perf_counter()
groups = d // 64
data_r = data.reshape(n * groups, 64)
scales4 = np.abs(data_r).max(axis=1, keepdims=True) / 7.0
scales4 = np.where(scales4 == 0, 1.0, scales4)
q4 = np.clip(np.round(data_r / scales4), -7, 7).astype(np.int8)  # 4-bit values in int8
t1 = time.perf_counter()
print(f"  INT4 group-wise quantize: {t1-t0:.3f}s  (2x smaller than INT8)")
print(f"  Disk: {q4.nbytes / 1e6:.1f} MB vs {data.nbytes / 1e6:.1f} MB (f32)")
