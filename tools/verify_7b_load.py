#!/usr/bin/env python3
"""Verify 7B model loads correctly after tensors/ deletion."""
import time
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

t0 = time.perf_counter()
from compressed_loader import load_from_npy_dir

result = load_from_npy_dir(
    '/Users/wscholl/models/Qwen2.5-7B-Instruct-bf16-compressed',
    model_dir='/Users/wscholl/models/Qwen2.5-7B-Instruct-bf16',
    verbose=True,
    return_stats=True,
)
elapsed = time.perf_counter() - t0

model, tok, stats = result

print(f"\n{'═'*50}")
print(f"  Load time : {elapsed:.2f}s")
print(f"  Loader    : {stats.get('loader', 'unknown')}")
print(f"{'═'*50}")

assert stats.get('loader') == 'squish-4bit', f"Expected squish-4bit loader, got: {stats.get('loader')}"
print("  ✓ Tier 0 (squish_4bit) confirmed — tensors/ deletion did not break load")

