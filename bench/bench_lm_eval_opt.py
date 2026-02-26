#!/usr/bin/env python3
"""
bench_lm_eval_opt.py

Benchmark and correctness test for the optimised _forward_selected_logprobs
vs the legacy _forward_logprobs method.

Verifies:
1. log-prob values are identical (within float32 precision)
2. is_greedy values are identical
3. Optimised method is faster (less numpy overhead)

Usage:
    python3 bench_lm_eval_opt.py
"""
import time
import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

def main():
    print("Loading squish_lm_eval...")
    from squish_lm_eval import SquishCompressedLM

    # Use the 7B model (smaller, faster to load for benchmarking)
    comp_dir = "/Users/wscholl/models/Qwen2.5-7B-Instruct-bf16-compressed"
    model_dir = "/Users/wscholl/models/Qwen2.5-7B-Instruct-bf16"

    print(f"Loading model from {comp_dir}...")
    t0 = time.perf_counter()
    lm = SquishCompressedLM(
        compressed_dir=comp_dir,
        model_dir=model_dir,
        verbose=True,
    )
    print(f"Model loaded in {time.perf_counter() - t0:.2f}s\n")

    # Test cases: (context, continuation)
    test_cases = [
        ("The capital of France is", " Paris"),
        ("Two plus two equals", " four"),
        ("The sky is", " blue"),
        ("Machine learning is a type of", " artificial intelligence"),
        ("In Python, you can get the length of a list with the function", " len"),
    ]

    print(f"{'='*60}")
    print(f"  Correctness test ({len(test_cases)} examples)")
    print(f"{'='*60}")

    legacy_lp = []
    opt_lp    = []
    legacy_times = []
    opt_times    = []

    for ctx, cont in test_cases:
        ctx_tokens  = lm.tok_encode(ctx)
        cont_tokens = lm.tok_encode(cont)
        all_tokens  = ctx_tokens + cont_tokens

        # Legacy method
        t0 = time.perf_counter()
        log_probs   = lm._forward_logprobs(all_tokens)
        cont_start  = len(ctx_tokens) - 1
        cont_end    = cont_start + len(cont_tokens)
        cont_lp     = log_probs[cont_start:cont_end]
        lp_sum_old  = float(sum(cont_lp[i, cont_tokens[i]] for i in range(len(cont_tokens))))
        is_greedy_old = all(int(np.argmax(cont_lp[i])) == cont_tokens[i] for i in range(len(cont_tokens)))
        legacy_times.append(time.perf_counter() - t0)
        legacy_lp.append((lp_sum_old, is_greedy_old))

        # Optimised method
        t0 = time.perf_counter()
        lp_list_new, is_greedy_list_new = lm._forward_selected_logprobs(all_tokens, cont_tokens)
        lp_sum_new  = float(sum(lp_list_new))
        is_greedy_new = all(is_greedy_list_new)
        opt_times.append(time.perf_counter() - t0)
        opt_lp.append((lp_sum_new, is_greedy_new))

        lp_diff = abs(lp_sum_old - lp_sum_new)
        match = "✓" if lp_diff < 1e-4 and is_greedy_old == is_greedy_new else "✗"
        print(f"  {match}  ctx='{ctx[:30]}...' cont='{cont}'")
        print(f"     lp_sum: legacy={lp_sum_old:.4f}  opt={lp_sum_new:.4f}  diff={lp_diff:.2e}")
        print(f"     is_greedy: legacy={is_greedy_old}  opt={is_greedy_new}")

    all_pass = all(
        abs(legacy_lp[i][0] - opt_lp[i][0]) < 1e-4 and legacy_lp[i][1] == opt_lp[i][1]
        for i in range(len(test_cases))
    )
    print(f"\n{'='*60}")
    print(f"  Correctness: {'ALL PASS ✓' if all_pass else 'FAIL ✗'}")
    print(f"  Legacy mean time : {sum(legacy_times)/len(legacy_times):.3f}s")
    print(f"  Optimised mean   : {sum(opt_times)/len(opt_times):.3f}s")
    if all(legacy_times) and all(opt_times):
        speedup = sum(legacy_times) / sum(opt_times)
        print(f"  Speedup          : {speedup:.2f}×")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
