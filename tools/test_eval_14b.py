#!/usr/bin/env python3
"""Test if the 14B eval EPERM error occurs in a clean lm_eval call."""
import sys
sys.path.insert(0, '/Users/wscholl/poc')

import traceback
import lm_eval

# Register squish model type 
import squish_lm_eval  # noqa

from lm_eval import simple_evaluate

MODEL_DIR = '/Users/wscholl/models/Qwen2.5-14B-Instruct-bf16'
COMP_DIR = '/Users/wscholl/models/Qwen2.5-14B-Instruct-bf16-compressed'

print("Testing 14B eval with limit=5...")
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
