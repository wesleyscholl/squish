#!/usr/bin/env python3
"""
run_inference.py

Run text generation using compressed (Vectro INT8) weights loaded via
compressed_loader.py.  The original .safetensors files are NOT needed —
move them away to confirm this works without them.

Produces compressed_output.json which verify.py compares against
reference_output.json from run_reference.py.

Usage:
    python3 run_inference.py \\
        [--model-dir ~/models/Qwen2.5-1.5B-Instruct] \\
        [--npz ~/models/Qwen2.5-1.5B-Instruct-compressed/weights_compressed.npz] \\
        [--prompt "The capital of France is"] \\
        [--max-tokens 50]
"""
import sys
import json
import time
import argparse
from pathlib import Path

_DEFAULT_MODELS_DIR = Path.home() / ".squish" / "models"
MODEL_DIR_DEFAULT = str(_DEFAULT_MODELS_DIR / "Qwen2.5-1.5B-Instruct")
NPZ_DEFAULT = str(_DEFAULT_MODELS_DIR / "Qwen2.5-1.5B-Instruct-compressed" / "weights_compressed.npz")
PROMPT_DEFAULT = "The capital of France is"
MAX_TOKENS_DEFAULT = 50


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=MODEL_DIR_DEFAULT)
    ap.add_argument("--npz", default=NPZ_DEFAULT,
                    help="Path to weights_compressed.npz  -OR-  path to a npy-dir "
                         "directory produced by convert_weights.py --format npy-dir")
    ap.add_argument("--manifest", default=None,
                    help="npz format only: path to manifest JSON (default: derived from --npz)")
    ap.add_argument("--prompt", default=PROMPT_DEFAULT)
    ap.add_argument("--max-tokens", type=int, default=MAX_TOKENS_DEFAULT)
    ap.add_argument("--output", default="compressed_output.json")
    ap.add_argument("--quiet", action="store_true",
                    help="Suppress per-tensor loading output")
    ap.add_argument("--streaming", action="store_true",
                    help="Use layer-by-layer streaming loader instead of batch loader")
    ap.add_argument("--workers", type=int, default=0,
                    help="Parallel decomp threads for npy-dir loader. "
                         "0 (default) = auto (min(cpu_count, 8)). "
                         "Set 1 to disable parallelism.")
    ap.add_argument("--prompts-file", default=None,
                    help="JSON [{category, prompt, max_tokens}...] — loads model once, "
                         "generates all prompts, saves JSON array to --output")
    args = ap.parse_args()

    # Add poc/ to path so loaders are importable
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    from mlx_lm import generate

    import os as _os
    fmt_label = "npy-dir" if _os.path.isdir(args.npz) else "npz"
    print(f"Loading compressed model ({fmt_label}) ...")
    print(f"  model-dir : {args.model_dir}")
    print(f"  weights   : {args.npz}")
    print()

    t0 = time.time()

    if args.streaming:
        from streaming_loader import load_streaming
        model, tokenizer, loader_stats = load_streaming(
            model_dir=args.model_dir,
            npz_path=args.npz,
            manifest_path=args.manifest,
            verbose=not args.quiet,
        )
    else:
        from squish.compressed_loader import load_compressed_model
        model, tokenizer, loader_stats = load_compressed_model(
            model_dir=args.model_dir,
            npz_path=args.npz,
            manifest_path=args.manifest,
            verbose=not args.quiet,
            return_stats=True,
            workers=args.workers,
        )

    load_time = time.time() - t0
    print(f"Load time: {load_time:.2f}s\n")

    # ── Batch / test-suite mode ───────────────────────────────────────────
    if args.prompts_file:
        with open(args.prompts_file) as f:
            items = json.load(f)
        results = []
        for item in items:
            prompt = item["prompt"]
            mt = item.get("max_tokens", args.max_tokens)
            print(f"  [{item.get('category', '?')}] {prompt!r}  (max_tokens={mt})")
            t1 = time.time()
            out = generate(model, tokenizer, prompt=prompt, max_tokens=mt, verbose=False)
            gen_time = time.time() - t1
            print(f"    \u2192 {out!r}  ({gen_time:.2f}s)")
            results.append({
                "category":            item.get("category", ""),
                "prompt":              prompt,
                "output":              out,
                "load_time_s":         load_time,
                "gen_time_s":          gen_time,
                "max_tokens":          mt,
                "loader":              loader_stats.get("loader", "unknown"),
                "ram_delta_mb":        loader_stats.get("ram_delta_mb"),
                "decompression_time_s": loader_stats.get("decompression_time_s"),
            })
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved {len(results)} results \u2192 {args.output}")
        return

    # ── Single-prompt mode ────────────────────────────────────────────────
    print(f"Generating (prompt: {args.prompt!r}, max_tokens={args.max_tokens}) ...")
    t1 = time.time()
    output = generate(
        model, tokenizer,
        prompt=args.prompt,
        max_tokens=args.max_tokens,
        verbose=True,
    )
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
        "npz_path": args.npz,
        "loader": loader_stats.get("loader", "streaming" if args.streaming else "batch"),
        # RAM stats from loader (both batch and streaming provide these)
        "ram_baseline_mb":   loader_stats.get("ram_baseline_mb"),
        "ram_peak_mb":       loader_stats.get("ram_peak_mb"),
        "ram_delta_mb":      loader_stats.get("ram_delta_mb"),
        "n_quantized":       loader_stats.get("n_quantized"),
        "n_passthrough":     loader_stats.get("n_passthrough"),
        "decompression_time_s": loader_stats.get("decompression_time_s"),
        "decomp_workers":    loader_stats.get("decomp_workers"),
    }
    if args.streaming:
        result["prefetch_savings_s"] = loader_stats.get("prefetch_savings_s")
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
