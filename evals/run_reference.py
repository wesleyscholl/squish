#!/usr/bin/env python3
"""
run_reference.py

Establish a baseline: standard mlx_lm generation with unmodified weights.
Saves reference_output.json for later comparison against compressed inference.

Usage:
    python3 run_reference.py [--model-dir PATH] [--prompt TEXT] [--max-tokens N]
    python3 run_reference.py --prompts-file prompts.json --output results.json
"""
import json
import time
import argparse

MODEL_DIR_DEFAULT = "/Users/wscholl/models/Qwen2.5-1.5B-Instruct"
PROMPT_DEFAULT = "The capital of France is"
MAX_TOKENS_DEFAULT = 20


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=MODEL_DIR_DEFAULT)
    ap.add_argument("--prompt", default=PROMPT_DEFAULT)
    ap.add_argument("--max-tokens", type=int, default=MAX_TOKENS_DEFAULT)
    ap.add_argument("--output", default="reference_output.json")
    ap.add_argument("--prompts-file", default=None,
                    help="JSON file [{category, prompt, max_tokens}...] — loads model once, "
                         "generates all prompts, saves a JSON array to --output")
    args = ap.parse_args()

    from mlx_lm import load, generate

    print(f"Loading model from {args.model_dir} ...")
    t0 = time.time()
    model, tokenizer = load(args.model_dir)
    load_time = time.time() - t0
    print(f"Loaded in {load_time:.2f}s")

    # ── Batch / test-suite mode ───────────────────────────────────────────
    if args.prompts_file:
        with open(args.prompts_file) as f:
            items = json.load(f)

        results = []
        for item in items:
            prompt = item["prompt"]
            mt = item.get("max_tokens", args.max_tokens)
            print(f"\n  [{item.get('category', '?')}] {prompt!r}  (max_tokens={mt})")
            t1 = time.time()
            out = generate(model, tokenizer, prompt=prompt, max_tokens=mt, verbose=False)
            gen_time = time.time() - t1
            print(f"    → {out!r}  ({gen_time:.2f}s)")
            results.append({
                "category":    item.get("category", ""),
                "prompt":      prompt,
                "output":      out,
                "load_time_s": load_time,
                "gen_time_s":  gen_time,
                "max_tokens":  mt,
                "model_dir":   args.model_dir,
            })

        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved {len(results)} results → {args.output}")
        return

    # ── Single-prompt mode (original behaviour) ───────────────────────────
    print(f"\nGenerating (prompt: {args.prompt!r}) ...")
    t1 = time.time()
    output = generate(model, tokenizer, prompt=args.prompt, max_tokens=args.max_tokens, verbose=True)
    gen_time = time.time() - t1

    print(f"\nGeneration time: {gen_time:.2f}s")
    print(f"Output: {output!r}")

    result = {
        "prompt": args.prompt,
        "output": output,
        "load_time_s": load_time,
        "gen_time_s": gen_time,
        "max_tokens": args.max_tokens,
        "model_dir": args.model_dir,
    }
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
