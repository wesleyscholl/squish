#!/usr/bin/env python3
"""Verify 7B model loads correctly after tensors/ deletion."""
import time
import sys
import os
import argparse
from pathlib import Path

_MODELS_DIR = Path.home() / ".squish" / "models"

ap = argparse.ArgumentParser(description="Verify 7B model load")
ap.add_argument("--model", default=str(_MODELS_DIR / "Qwen2.5-7B-Instruct-bf16"))
ap.add_argument("--compressed", default=None)
args = ap.parse_args()

t0 = time.perf_counter()
from squish.compressed_loader import load_from_npy_dir

comp_dir = args.compressed or (args.model + "-compressed")
result = load_from_npy_dir(
    comp_dir,
    model_dir=args.model,
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

