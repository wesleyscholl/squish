#!/usr/bin/env python3
"""
benchmark.py

Automated benchmark suite for the Squish PoC.

Compares three loading strategies across a suite of prompts:
  1. Reference    — stock mlx_lm.load() / generate()
  2. Compressed   — compressed_loader (batch decompression)
  3. Streaming    — streaming_loader  (layer-by-layer prefetch)

Measured metrics:
  load_time_s           Time to load model weights
  first_token_s         Latency to first generated token
  throughput_toks       Tokens per second during generation
  ram_delta_mb          RSS increase during load (vs baseline)
  disk_mb               Disk size of the weight files used
  token_agree           Agreement ratio vs reference output
  mean_cosine           Mean cosine sim of sampled tensors (vs fp16)

Usage:
    python3 benchmark.py \\
        [--model-dir ~/models/Qwen2.5-1.5B-Instruct] \\
        [--npz ~/models/.../weights_compressed.npz] \\
        [--max-tokens 40] \\
        [--output benchmark_results.json]

Output:
    Prints a formatted comparison table to stdout.
    Saves all raw data to --output (default: benchmark_results.json).
"""
import sys
import json
import time
import argparse
import resource
import platform
from pathlib import Path
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

GREEN  = ""
YELLOW = ""
CYAN   = ""
BOLD   = ""
RESET  = ""
DIM    = ""


def _rss_mb() -> float:
    ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if platform.system() == "Darwin":
        return ru / 1_048_576
    return ru / 1_024


def _disk_mb(paths: list[Path]) -> float:
    return sum(p.stat().st_size for p in paths if p.exists()) / 1_000_000


def _token_agreement(ref: str, cmp: str) -> float:
    rt = ref.strip().split()
    ct = cmp.strip().split()
    n = min(len(rt), len(ct))
    if n == 0:
        return 0.0
    return sum(a == b for a, b in zip(rt[:n], ct[:n])) / n


def _cosine_sim_sample(model_dir: Path, npz_path: Path, manifest_path: Path,
                       n_sample: int = 8) -> Optional[float]:
    """Cosine similarity between original fp16 and reconstructed Q8 tensors."""
    shard_files = sorted(model_dir.glob("*.safetensors"))
    if not shard_files:
        return None  # safetensors gone — skip

    import mlx.core as mx
    from pathlib import Path as _P
    sys.path.insert(0, str(_P(__file__).parent))
    from compressed_loader import _dequantize, _unique_base_keys, _safe_key_to_original

    orig = {}
    for sf in shard_files:
        for name, arr in mx.load(str(sf)).items():
            orig[name] = np.array(arr.astype(mx.float32))

    with open(manifest_path) as f:
        manifest = json.load(f)
    s2o = {v: k for k, v in manifest.items()}

    npz = np.load(str(npz_path), allow_pickle=False)
    base_keys = list(_unique_base_keys(list(npz.files)))

    # pick tensors that are quantized (not passthrough) and exist in orig
    sample_keys = [
        sk for sk in base_keys
        if (sk + "__q") in npz.files
        and s2o.get(sk) in orig
    ][:n_sample]

    if not sample_keys:
        npz.close()
        return None

    cosines = []
    for sk in sample_keys:
        oname = s2o[sk]
        orig_arr = orig[oname]
        original_shape = tuple(int(x) for x in npz[sk + "__shape"].tolist())
        recon = _dequantize(npz, sk).reshape(original_shape)

        a = orig_arr.reshape(-1, orig_arr.shape[-1]).astype(np.float64)
        b = recon.reshape(-1, recon.shape[-1]).astype(np.float64)
        an = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-10)
        bn = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-10)
        cosines.append(float(np.mean(np.sum(an * bn, axis=1))))

    npz.close()
    return float(np.mean(cosines))


# ---------------------------------------------------------------------------
# Runner functions (one per strategy)
# ---------------------------------------------------------------------------

def run_reference(model_dir: Path, prompt: str, max_tokens: int) -> dict:
    from mlx_lm import load, generate

    rss0 = _rss_mb()
    t0 = time.perf_counter()
    model, tokenizer = load(str(model_dir))
    load_time = time.perf_counter() - t0
    rss1 = _rss_mb()

    t1 = time.perf_counter()
    output = generate(model, tokenizer, prompt=prompt,
                      max_tokens=max_tokens, verbose=False)
    gen_time = time.perf_counter() - t1

    n_tokens = len(output.split())
    throughput = n_tokens / gen_time if gen_time > 0 else 0.0

    disk_files = sorted(model_dir.glob("*.safetensors"))
    return {
        "strategy": "reference",
        "load_time_s": round(load_time, 3),
        "gen_time_s": round(gen_time, 3),
        "throughput_toks_s": round(throughput, 1),
        "ram_delta_mb": round(rss1 - rss0, 1),
        "disk_mb": round(_disk_mb(disk_files), 1),
        "output": output,
        "token_agree": 1.0,
        "mean_cosine": 1.0,
    }


def run_compressed(model_dir: Path, npz_path: Path, manifest_path: Path,
                   prompt: str, max_tokens: int, ref_output: str) -> dict:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from compressed_loader import load_compressed_model
    from mlx_lm import generate

    rss0 = _rss_mb()
    t0 = time.perf_counter()
    model, tokenizer, stats = load_compressed_model(
        model_dir=str(model_dir),
        npz_path=str(npz_path),
        manifest_path=str(manifest_path),
        verbose=False,
        return_stats=True,
    )
    load_time = time.perf_counter() - t0
    rss1 = _rss_mb()

    t1 = time.perf_counter()
    output = generate(model, tokenizer, prompt=prompt,
                      max_tokens=max_tokens, verbose=False)
    gen_time = time.perf_counter() - t1

    n_tokens = len(output.split())
    throughput = n_tokens / gen_time if gen_time > 0 else 0.0

    cosine = _cosine_sim_sample(model_dir, npz_path, manifest_path)

    return {
        "strategy": "compressed_batch",
        "load_time_s": round(load_time, 3),
        "gen_time_s": round(gen_time, 3),
        "throughput_toks_s": round(throughput, 1),
        "ram_delta_mb": round(rss1 - rss0, 1),
        "ram_peak_mb": round(stats.get("ram_peak_mb", 0), 1),
        "disk_mb": round(_disk_mb([npz_path]), 1),
        "output": output,
        "token_agree": round(_token_agreement(ref_output, output), 3),
        "mean_cosine": round(cosine, 5) if cosine is not None else None,
        "n_quantized": stats.get("n_quantized"),
        "n_passthrough": stats.get("n_passthrough"),
        "decompression_time_s": round(stats.get("decompression_time_s", 0), 3),
    }


def run_streaming(model_dir: Path, npz_path: Path, manifest_path: Path,
                  prompt: str, max_tokens: int, ref_output: str) -> dict:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from streaming_loader import load_streaming
    from mlx_lm import generate

    rss0 = _rss_mb()
    t0 = time.perf_counter()
    model, tokenizer, stats = load_streaming(
        model_dir=str(model_dir),
        npz_path=str(npz_path),
        manifest_path=str(manifest_path),
        verbose=False,
    )
    load_time = time.perf_counter() - t0
    rss1 = _rss_mb()

    t1 = time.perf_counter()
    output = generate(model, tokenizer, prompt=prompt,
                      max_tokens=max_tokens, verbose=False)
    gen_time = time.perf_counter() - t1

    n_tokens = len(output.split())
    throughput = n_tokens / gen_time if gen_time > 0 else 0.0

    cosine = _cosine_sim_sample(model_dir, npz_path, manifest_path)

    return {
        "strategy": "compressed_streaming",
        "load_time_s": round(load_time, 3),
        "gen_time_s": round(gen_time, 3),
        "throughput_toks_s": round(throughput, 1),
        "ram_delta_mb": round(rss1 - rss0, 1),
        "ram_peak_mb": round(stats.get("ram_peak_mb", 0), 1),
        "disk_mb": round(_disk_mb([npz_path]), 1),
        "output": output,
        "token_agree": round(_token_agreement(ref_output, output), 3),
        "mean_cosine": round(cosine, 5) if cosine is not None else None,
        "n_quantized": stats.get("n_quantized"),
        "n_passthrough": stats.get("n_passthrough"),
        "decompression_time_s": round(stats.get("decompression_time_s", 0), 3),
        "prefetch_savings_s": round(stats.get("prefetch_savings_s", 0), 3),
    }


# ---------------------------------------------------------------------------
# Table printer
# ---------------------------------------------------------------------------

def _fmt(val, unit="", precision=1, good_low=False):
    """Format a metric value, colouring best result green."""
    if val is None:
        return f"{'n/a':>10}"
    if isinstance(val, float):
        s = f"{val:.{precision}f}{unit}"
    else:
        s = f"{val}{unit}"
    return f"{s:>10}"


def print_table(results: list[dict], ref_disk_mb: float):
    strategies = [r["strategy"] for r in results]
    col_w = max(len(s) for s in strategies) + 2

    headers = [
        "strategy", "load_s", "gen_s", "tok/s", "ΔRAM MB",
        "disk MB", "compress", "agree%", "cosine",
    ]

    # separator
    sep = "─" * (col_w + 9 * 11)
    print(f"\n{BOLD}{CYAN}{sep}{RESET}")
    print(f"{BOLD}{'Strategy':<{col_w}}" +
          "".join(f"{h:>11}" for h in headers[1:]) + RESET)
    print(f"{BOLD}{CYAN}{sep}{RESET}")

    for r in results:
        compress_ratio = (
            f"{ref_disk_mb / r['disk_mb']:.2f}x"
            if r.get("disk_mb") and ref_disk_mb > 0
            else "—"
        )
        agree_pct = (
            f"{r['token_agree']*100:.0f}%"
            if r.get("token_agree") is not None
            else "—"
        )
        cosine_str = (
            f"{r['mean_cosine']:.4f}"
            if r.get("mean_cosine") is not None
            else "—"
        )

        row = (
            f"{r['strategy']:<{col_w}}"
            f"{r['load_time_s']:>11.2f}"
            f"{r['gen_time_s']:>11.2f}"
            f"{r['throughput_toks_s']:>11.1f}"
            f"{r['ram_delta_mb']:>11.0f}"
            f"{r.get('disk_mb', 0):>11.0f}"
            f"{compress_ratio:>11}"
            f"{agree_pct:>11}"
            f"{cosine_str:>11}"
        )
        if r["strategy"] == "reference":
            print(f"{DIM}{row}{RESET}")
        else:
            print(row)

    print(f"{BOLD}{CYAN}{sep}{RESET}")
    print(f"\n  {DIM}load_s: weight-load wall time | gen_s: generation wall time | tok/s: tokens/second{RESET}")
    print(f"  {DIM}ΔRAM: RSS delta during load | compress: disk vs reference | agree%: token match vs ref{RESET}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Squish compression benchmark")
    ap.add_argument("--model-dir",
                    default=str(Path.home() / "models" / "Qwen2.5-1.5B-Instruct"))
    ap.add_argument("--npz",
                    default=str(Path.home() / "models" /
                                "Qwen2.5-1.5B-Instruct-compressed" /
                                "weights_compressed.npz"))
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--max-tokens", type=int, default=40)
    ap.add_argument("--output", default="benchmark_results.json")
    ap.add_argument("--skip-reference", action="store_true",
                    help="Skip reference run (use existing benchmark_results.json)")
    ap.add_argument("--skip-streaming", action="store_true",
                    help="Skip streaming strategy run")
    args = ap.parse_args()

    model_dir = Path(args.model_dir).expanduser()
    npz_path  = Path(args.npz).expanduser()
    manifest_path = Path(
        args.manifest or str(npz_path).replace(".npz", "_manifest.json")
    ).expanduser()

    if not npz_path.exists():
        print(f"ERROR: npz not found: {npz_path}")
        print("Run convert_weights.py first, or run_poc.py to do everything.")
        sys.exit(1)

    print(f"\n{BOLD}Squish PoC — Compression Benchmark{RESET}")
    print(f"Model:    {model_dir}")
    print(f"Npz:      {npz_path}")
    print(f"Prompt:   {args.prompt!r}")
    print(f"Tokens:   {args.max_tokens}\n")

    results: list[dict] = []

    # Reference
    if args.skip_reference:
        print(f"{YELLOW}→ Reference skipped{RESET}")
        ref_output = ""
        ref_disk_mb = _disk_mb(sorted(model_dir.glob("*.safetensors")))
        results.append({"strategy": "reference", "skipped": True,
                        "disk_mb": ref_disk_mb, "token_agree": 1.0, "mean_cosine": 1.0})
    else:
        print(f"{CYAN}Running reference strategy ...{RESET}")
        ref = run_reference(model_dir, args.prompt, args.max_tokens)
        print(f"  load={ref['load_time_s']:.2f}s  gen={ref['gen_time_s']:.2f}s  "
              f"output={ref['output'][:60]!r}")
        results.append(ref)
        ref_output = ref["output"]
        ref_disk_mb = ref["disk_mb"]

    # Compressed batch
    print(f"\n{CYAN}Running compressed_batch strategy ...{RESET}")
    comp = run_compressed(model_dir, npz_path, manifest_path,
                          args.prompt, args.max_tokens, ref_output)
    print(f"  load={comp['load_time_s']:.2f}s  gen={comp['gen_time_s']:.2f}s  "
          f"agree={comp['token_agree']:.1%}  output={comp['output'][:60]!r}")
    results.append(comp)

    # Streaming
    if not args.skip_streaming:
        print(f"\n{CYAN}Running compressed_streaming strategy ...{RESET}")
        stream = run_streaming(model_dir, npz_path, manifest_path,
                               args.prompt, args.max_tokens, ref_output)
        print(f"  load={stream['load_time_s']:.2f}s  gen={stream['gen_time_s']:.2f}s  "
              f"agree={stream['token_agree']:.1%}  prefetch_savings={stream.get('prefetch_savings_s',0):.2f}s")
        results.append(stream)

    # Table
    print_table(results, ref_disk_mb)

    # Compression ratio callout
    if comp.get("disk_mb") and ref_disk_mb > 0:
        ratio = ref_disk_mb / comp["disk_mb"]
        print(f"  {GREEN}{BOLD}Compression ratio: {ratio:.2f}x {RESET}"
              f"({ref_disk_mb:.0f} MB → {comp['disk_mb']:.0f} MB on disk)\n")

    # Save JSON
    output_path = Path(args.output)
    with open(output_path, "w") as f:
        json.dump({
            "model_dir": str(model_dir),
            "npz_path": str(npz_path),
            "prompt": args.prompt,
            "max_tokens": args.max_tokens,
            "results": results,
        }, f, indent=2)
    print(f"  Raw results saved to {output_path}\n")


if __name__ == "__main__":
    main()
