"""Verify 7B still loads after deleting tensors/ dir (Tier 0 should catch it)."""
import sys, time
sys.path.insert(0, '/Users/wscholl/poc')
from compressed_loader import load_from_npy_dir

COMP_DIR  = '/Users/wscholl/models/Qwen2.5-7B-Instruct-bf16-compressed'
MODEL_DIR = '/Users/wscholl/models/Qwen2.5-7B-Instruct-bf16'

t0 = time.perf_counter()
model, tok, stats = load_from_npy_dir(COMP_DIR, model_dir=MODEL_DIR, verbose=True, return_stats=True)
load_s = time.perf_counter() - t0

import mlx_lm
resp = mlx_lm.generate(model, tok, prompt='The capital of France is', max_tokens=12, verbose=False)
print(f'\nloader={stats.get("loader")}  load_s={load_s:.2f}s')
print(f'response: {resp!r}')
