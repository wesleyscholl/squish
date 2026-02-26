"""Benchmark the vectorized quantizer against a simulated 7B weight matrix."""
import sys, time
import numpy as np
sys.path.insert(0, "/Users/wscholl/vectro")
from python.interface import quantize_embeddings, mean_cosine_similarity

rng = np.random.default_rng(42)

shapes = [
    ("7B MLP gate (13824x4096)", 13824, 4096),
    ("7B attn q_proj (4096x4096)", 4096, 4096),
    ("14B MLP gate (13824x5120)", 13824, 5120),
]

for label, R, C in shapes:
    mat = rng.standard_normal((R, C)).astype(np.float32)
    t0 = time.perf_counter()
    result = quantize_embeddings(mat)
    elapsed = time.perf_counter() - t0
    recon = result.quantized.astype(np.float32) * result.scales[:, None]
    cs = mean_cosine_similarity(mat, recon)
    gb = mat.nbytes / 1e9
    throughput = gb / elapsed
    print(f"{label}")
    print(f"  shape={mat.shape}  elapsed={elapsed:.3f}s  cosine={cs:.6f}  throughput={throughput:.1f} GB/s")
    print()

# Estimate full 14B compression time
# 14B original is 29.6GB, all tensors combined
total_gb = 29.6
# use the throughput from the largest tensor as estimate
mat = rng.standard_normal((13824, 5120)).astype(np.float32)
t0 = time.perf_counter()
for _ in range(3):
    quantize_embeddings(mat)
tp = mat.nbytes / ((time.perf_counter() - t0) / 3) / 1e9
est_s = total_gb / tp
print(f"14B full model estimate: ~{est_s:.0f}s  (previous: ~580s)")
print(f"Speedup: ~{580 / est_s:.0f}x")
