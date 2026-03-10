#!/usr/bin/env python3
"""Test if the 14B eval EPERM error occurs in a clean lm_eval call."""
import sys
import argparse
from pathlib import Path

_MODELS_DIR = Path.home() / ".squish" / "models"

ap = argparse.ArgumentParser()
ap.add_argument("--model", default=str(_MODELS_DIR / "Qwen2.5-14B-Instruct-bf16"))
ap.add_argument("--compressed", default=None)
args = ap.parse_args()

MODEL_DIR = args.model
COMP_DIR  = args.compressed or (MODEL_DIR + "-compressed")

import traceback
from squish.squish_lm_eval import SquishLM  # noqa — registers "squish" model type
from lm_eval import simple_evaluate
try:
    results = simple_evaluate(
        model='squish',
        model_args=f'model_dir={MODEL_DIR},compressed_dir={COMP_DIR},verbose=False',
        tasks=['winogrande'],
        limit=5,
        log_samples=False,
        verbosity='WARNING',
        random_seed=42,
        numpy_random_seed=42,
    )
    print("SUCCESS!")
    for task, v in results.get('results', {}).items():
        print(f"  {task}: {v}")
except Exception as e:
    traceback.print_exc()
    print(f"\nFAILED: {type(e).__name__}: {e}")
