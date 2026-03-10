"""Verify 7B still loads after deleting tensors/ dir (Tier 0 should catch it)."""
import sys, time, argparse
from pathlib import Path
from squish.compressed_loader import load_from_npy_dir

_MODELS_DIR = Path.home() / ".squish" / "models"

ap = argparse.ArgumentParser(description="Verify 7B model loads without tensors/ dir")
ap.add_argument("--model", default=str(_MODELS_DIR / "Qwen2.5-7B-Instruct-bf16"), help="Path to bf16 model dir")
ap.add_argument("--compressed", default=None, help="Path to compressed dir (default: <model>-compressed)")
args = ap.parse_args()

MODEL_DIR = args.model
COMP_DIR  = args.compressed or (MODEL_DIR + "-compressed")

t0 = time.perf_counter()
model, tok, stats = load_from_npy_dir(COMP_DIR, model_dir=MODEL_DIR, verbose=True, return_stats=True)
load_s = time.perf_counter() - t0

import mlx_lm
resp = mlx_lm.generate(model, tok, prompt='The capital of France is', max_tokens=12, verbose=False)
print(f'\nloader={stats.get("loader")}  load_s={load_s:.2f}s')
print(f'response: {resp!r}')
