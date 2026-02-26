#!/usr/bin/env python3
"""
pull_model.py  —  Download, compress, and cache any HuggingFace model for Squish.

Phases:
  1. Resolve model → mlx-community bf16 variant (fastest) or raw HF → convert
  2. Vectro INT8 compression  (creates npy-dir: ~50% disk of bf16)
  3. Tier 1 cache warmup      (pre-loads finalized/ MLX arrays, sub-5s subsequent loads)
  4. Tier 2 cache build       (squish_weights.safetensors, sub-1s loads)
  5. Validation               (token agreement + cosine similarity vs mlx_lm reference)
  6. Benchmark snapshot       (load-time comparison + token/s)

Usage:
    python3 pull_model.py Qwen/Qwen2.5-7B-Instruct
    python3 pull_model.py Qwen/Qwen2.5-14B-Instruct --skip-validation
    python3 pull_model.py mlx-community/Qwen2.5-7B-Instruct-bf16  --model-dir ~/models/my7b

Options:
    MODEL_ID              HuggingFace model ID (required)
    --model-dir DIR       Where to store the bf16 model  [~/models/<name>-bf16]
    --compressed-dir DIR  Where to store Vectro output   [<model-dir>-compressed]
    --skip-download       Model already present, skip download
    --skip-compress       Compressed dir already exists, skip Vectro step
    --skip-cache          Skip tier 2 (squish_weights.safetensors) build
    --skip-validation     Skip reference vs compressed token agreement check
    --skip-benchmark      Skip load-time comparison
    --passthrough NAMES   Space-separated layer names to keep as float16
    --outlier-threshold F Vectro outlier threshold (default 20.0)
    --prompt TEXT         Validation/benchmark prompt
    --max-tokens N        Tokens for validation inference (default 32)
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

POC_DIR = Path(__file__).resolve().parent.parent.resolve()
sys.path.insert(0, str(POC_DIR))

GREEN  = ""
RED    = ""
YELLOW = ""
CYAN   = ""
MAGENTA = ""
BOLD   = ""
DIM    = ""
RESET  = ""

MODELS_ROOT = Path.home() / "models"


def banner(phase, title):
    print(f"\n{BOLD}{CYAN}{'─'*62}{RESET}")
    print(f"{BOLD}{CYAN}  Phase {phase}: {title}{RESET}")
    print(f"{BOLD}{CYAN}{'─'*62}{RESET}\n")

def ok(msg):   print(f"  {GREEN}✓{RESET}  {msg}")
def fail(msg): print(f"  {RED}✗{RESET}  {msg}"); sys.exit(1)
def info(msg): print(f"  {CYAN}→{RESET}  {msg}")
def warn(msg): print(f"  {YELLOW}⚠{RESET}  {msg}")
def sep():     print(f"\n{DIM}{'─'*62}{RESET}\n")


# ─────────────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────────────

def du_gb(path: Path) -> float:
    """Disk usage of a directory in gigabytes."""
    total = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    return total / 1e9


def run(cmd: list, desc: str, check: bool = True, env: dict | None = None) -> int:
    info(f"{desc} …")
    e = {**os.environ, **(env or {})}
    result = subprocess.run(cmd, env=e)
    if check and result.returncode != 0:
        fail(f"Command failed (exit {result.returncode}): {' '.join(str(c) for c in cmd)}")
    return result.returncode


def mlx_community_id(model_id: str) -> str:
    """
    Best-effort: find the mlx-community bf16 variant for a model.
    e.g. 'Qwen/Qwen2.5-7B-Instruct' → 'mlx-community/Qwen2.5-7B-Instruct-bf16'
    """
    short = model_id.split("/")[-1]
    return f"mlx-community/{short}-bf16"


# ─────────────────────────────────────────────────────────────────────────────
#  Phase 1 — Download / convert to bf16
# ─────────────────────────────────────────────────────────────────────────────

def phase_download(model_id: str, model_dir: Path, skip: bool):
    banner(1, f"Download  [{model_id}]")

    safetensors = list(model_dir.glob("*.safetensors")) if model_dir.exists() else []
    config      = model_dir / "config.json"

    if skip or (safetensors and config.exists()):
        ok(f"Model already at {model_dir}  ({du_gb(model_dir):.1f} GB)")
        return

    model_dir.mkdir(parents=True, exist_ok=True)

    from huggingface_hub import snapshot_download
    import requests

    # ── Try mlx-community bf16 first (no conversion needed) ──────────────────
    mlx_id = mlx_community_id(model_id)

    # Quick existence check via HF API (no download)
    exists = False
    try:
        from huggingface_hub import model_info
        model_info(mlx_id)
        exists = True
    except Exception:
        pass

    if exists:
        info(f"Downloading {mlx_id}  (MLX bf16, no conversion needed) …")
        t0 = time.perf_counter()
        snapshot_download(
            repo_id         = mlx_id,
            local_dir       = str(model_dir),
            ignore_patterns = ["*.pt", "*.bin", "original/", "*.gguf"],
        )
        elapsed = time.perf_counter() - t0
        sz = du_gb(model_dir)
        ok(f"Downloaded  {sz:.1f} GB  in {elapsed:.0f}s  →  {model_dir}")
        return

    # ── Fall back: download raw model, then convert with mlx_lm.convert ──────
    warn(f"mlx-community/{mlx_id.split('/')[-1]} not found — downloading {model_id} and converting")

    raw_dir = model_dir.parent / (model_dir.name.replace("-bf16", "") + "-raw")
    raw_dir.mkdir(parents=True, exist_ok=True)

    info(f"Downloading {model_id}  (raw HF safetensors) …")
    t0 = time.perf_counter()
    snapshot_download(
        repo_id         = model_id,
        local_dir       = str(raw_dir),
        ignore_patterns = ["*.pt", "*.bin", "original/", "*.gguf"],
    )
    elapsed = time.perf_counter() - t0
    ok(f"Raw model downloaded  in {elapsed:.0f}s")

    info(f"Converting to bf16 MLX format …")
    t1 = time.perf_counter()
    ret = subprocess.run([
        sys.executable, "-m", "mlx_lm.convert",
        "--hf-path",    str(raw_dir),
        "--mlx-path",   str(model_dir),
        "--dtype",      "bfloat16",
    ])
    if ret.returncode != 0:
        fail("mlx_lm.convert failed")
    ok(f"Converted to bf16  in {time.perf_counter() - t1:.0f}s  →  {model_dir}")


# ─────────────────────────────────────────────────────────────────────────────
#  Phase 2 — Vectro INT8 compression
# ─────────────────────────────────────────────────────────────────────────────

def phase_compress(model_dir: Path, comp_dir: Path, skip: bool,
                   passthrough: list[str], outlier_threshold: float):
    banner(2, "Vectro INT8 Compression")

    sentinel = comp_dir / "tensors" / ".manifest_ready"
    if skip or sentinel.exists():
        ok(f"Compressed npy-dir already at {comp_dir}  ({du_gb(comp_dir):.1f} GB)")
        return

    # ── Large-model early exit ─────────────────────────────────────────────────
    # For models > _METAL_TIER2_LIMIT_GB the Q8 npy-dir would never be used for
    # inference (phase_build_cache uses mlx_lm.convert → squish_4bit instead).
    # Skipping Q8 saves:  7B → skip 579s + 8.7 GB waste | 14B → skip 580s + 28.4 GB waste
    model_size_gb = du_gb(model_dir)
    if model_size_gb > _METAL_TIER2_LIMIT_GB:
        ok(
            f"Large model ({model_size_gb:.1f} GB > {_METAL_TIER2_LIMIT_GB:.0f} GB limit) "
            f"— skipping Q8 npy-dir phase (4-bit cache will be built instead, saves disk + time)"
        )
        # Write a minimal manifest so loader doesn't fail on old code paths
        comp_dir.mkdir(parents=True, exist_ok=True)
        import json as _json
        (comp_dir / "manifest.json").write_text(_json.dumps({"_large_model": "_large_model"}))
        return

    comp_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, str(POC_DIR / "squish" / "convert.py"),
        "--model-dir",        str(model_dir),
        "--output",           str(comp_dir),
        "--format",           "npy-dir",
        "--outlier-threshold", str(outlier_threshold),
        "--verbose",
    ]
    if passthrough:
        cmd += ["--passthrough"] + passthrough

    t0 = time.perf_counter()
    ret = subprocess.run(cmd)
    elapsed = time.perf_counter() - t0

    if ret.returncode != 0:
        fail("Vectro compression failed")

    original_gb   = du_gb(model_dir)
    compressed_gb = du_gb(comp_dir)
    ratio = original_gb / max(compressed_gb, 0.01)
    ok(f"Compressed in {elapsed:.0f}s")
    ok(f"Original:   {original_gb:.1f} GB")
    ok(f"Compressed: {compressed_gb:.1f} GB  ({ratio:.2f}× smaller on disk)")


# ─────────────────────────────────────────────────────────────────────────────
#  Phase 3+4 — Tier 1 + Tier 2 cache build
# ─────────────────────────────────────────────────────────────────────────────

# Metal memory budget for tier2 bf16 cache (leave headroom for OS + KV cache)
# Models above this estimated bf16 size skip the tier2 safetensors build (which
# requires all weights bf16 in memory at once).  Load path falls back to npy-dir
# with automatic 4-bit post-load quantization, which fits any 7B/14B model in 16 GB.
_METAL_TIER2_LIMIT_GB = 14.0   # skip tier2 safetensors build if est. bf16 > this

def _comp_dir_bf16_gb(comp_dir: Path, model_dir: Path | None = None) -> float:
    """
    Estimate the Metal memory footprint for a fully-materialized bf16 model
    loaded from a Q8 npy-dir.  Q8 stores 1 byte/element; bf16 is 2 bytes, so
    the Metal allocation is roughly 2 × the on-disk Q8 tensor size.

    We measure only the `tensors/` subdir (raw Q8 data), ignoring tier2
    safetensors, finalized caches, etc., to avoid inflated estimates.

    Falls back to model_dir size (= bf16 bytes directly) when tensors/ is absent —
    happens for large models that skipped the Q8 phase entirely.
    """
    tensors_dir = comp_dir / "tensors"
    if tensors_dir.is_dir():
        return du_gb(tensors_dir) * 2.0
    # No Q8 dir — use bhf16 model_dir directly (already in bf16 bytes)
    if model_dir is not None and model_dir.is_dir():
        return du_gb(model_dir)
    return du_gb(comp_dir) * 2.0


def phase_build_cache(model_dir: Path, comp_dir: Path, skip: bool, prompt: str, max_tokens: int):
    banner("3+4", "Build Tier 1 + Tier 2 cache  (squish_weights.safetensors)")

    tier2_path  = comp_dir / "squish_weights.safetensors"
    ready_flag  = comp_dir / ".squish_ready"

    if skip:
        ok("Build Tier 1 + Tier 2 cache — skipping (--skip-cache)")
        return

    if tier2_path.exists() and ready_flag.exists():
        sz = tier2_path.stat().st_size / 1e9
        ok(f"Tier 2 cache already built  ({sz:.1f} GB)  — skipping")
        return

    # ── For large models: build a 4-bit MLX model via mlx_lm.convert() ───────
    # mlx_lm.convert streams each bf16 shard, quantizes, writes one layer at a
    # time — peak RAM ≈ 1 shard (~5 GB for 7B), never the full 18 GB bf16 model.
    # The resulting 4-bit dir (~4 GB) is then loaded via mlx_lm.load() in <2s.
    est_bf16_gb = _comp_dir_bf16_gb(comp_dir, model_dir)
    four_bit_dir   = comp_dir / "squish_4bit"
    four_bit_ready = comp_dir / ".squish_4bit_ready"

    if est_bf16_gb > _METAL_TIER2_LIMIT_GB:
        if four_bit_ready.exists() and (four_bit_dir / "config.json").exists():
            sz = sum(f.stat().st_size for f in four_bit_dir.rglob("*") if f.is_file()) / 1e9
            ok(f"4-bit cache already built  ({sz:.1f} GB)  → {four_bit_dir}")
            ready_flag.write_text("ok")
            return

        info(
            f"Model too large for bf16 cache ({est_bf16_gb:.1f} GB > {_METAL_TIER2_LIMIT_GB:.0f} GB limit)"
        )
        info("Building 4-bit MLX model via mlx_lm.convert()  (one-time, ~5-10 min) …")
        import mlx_lm as _mlx_lm_convert
        import shutil as _shutil
        # Remove any partial/empty dir from a previous crashed run
        if four_bit_dir.exists() and not (four_bit_dir / "config.json").exists():
            _shutil.rmtree(four_bit_dir)
            info(f"Removed incomplete {four_bit_dir.name}/ (no config.json)")
        t0 = time.perf_counter()
        _mlx_lm_convert.convert(
            hf_path      = str(model_dir),
            mlx_path     = str(four_bit_dir),
            quantize     = True,
            q_bits       = 4,
            q_group_size = 64,
        )
        elapsed = time.perf_counter() - t0
        sz = sum(f.stat().st_size for f in four_bit_dir.rglob("*") if f.is_file()) / 1e9
        ok(f"4-bit cache written in {elapsed:.0f}s  ({sz:.1f} GB)  → {four_bit_dir}")

        info("Verifying 4-bit model loads correctly …")
        t_load = time.perf_counter()
        model_4bit, tok_4bit = _mlx_lm_convert.load(str(four_bit_dir))
        load_s = time.perf_counter() - t_load
        ok(f"4-bit model loaded in {load_s:.2f}s")
        try:
            resp = _mlx_lm_convert.generate(
                model_4bit, tok_4bit,
                prompt=prompt, max_tokens=max_tokens, verbose=False,
            )
            ok(f'Generated: "{resp.strip()[:80]}"')
        except Exception as _e:
            warn(f"Generation test: {_e}")
        del model_4bit, tok_4bit

        four_bit_ready.touch()
        ready_flag.write_text("ok")
        ok(".squish_4bit_ready and .squish_ready sentinels written")
        return

    info("Loading compressed model (first load builds caches) …")
    info("This takes 1-5 minutes for a 7B model — subsequent loads will be <1s")

    from compressed_loader import load_from_npy_dir

    t0 = time.perf_counter()
    model, tokenizer, stats = load_from_npy_dir(
        str(comp_dir),
        model_dir   = str(model_dir),
        verbose     = True,
        return_stats= True,
    )
    elapsed = time.perf_counter() - t0

    loader = stats.get("loader", "unknown")
    ram_mb = stats.get("ram_delta_mb", 0)

    ok(f"Loaded in {elapsed:.2f}s  (loader: {loader})")
    ok(f"RAM delta: {ram_mb:.0f} MB")

    # Quick generation test
    info(f"Running inference test: {repr(prompt)}")
    try:
        import mlx_lm
        response = mlx_lm.generate(
            model, tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
            verbose=False,
        )
        ok(f'Generated: "{response.strip()[:80]}"')
    except Exception as e:
        warn(f"Generation test error: {e}")

    if tier2_path.exists():
        sz = tier2_path.stat().st_size / 1e9
        ok(f"Tier 2 cache: {sz:.1f} GB  →  {tier2_path}")
        # Write sentinel so subsequent loads use the fast Tier 2 path
        ready_flag.write_text("ok")
        ok(f".squish_ready sentinel written")
    else:
        warn("Tier 2 cache not written yet — run again to build it")

    del model


# ─────────────────────────────────────────────────────────────────────────────
#  Phase 5 — Validation (token agreement + cosine similarity)
# ─────────────────────────────────────────────────────────────────────────────

def phase_validate(model_dir: Path, comp_dir: Path, skip: bool, prompt: str, max_tokens: int):
    banner(5, "Validation  (token agreement + weight cosine similarity)")

    if skip:
        info("Skipped (--skip-validation)")
        return

    import numpy as np

    # For large models the bf16 reference load itself OOMs — skip reference,
    # only validate that the squish compressed model generates coherent output.
    est_bf16_gb = _comp_dir_bf16_gb(comp_dir)
    ref_too_large = est_bf16_gb > _METAL_TIER2_LIMIT_GB
    if ref_too_large:
        warn(
            f"Reference bf16 load skipped ({est_bf16_gb:.1f} GB > {_METAL_TIER2_LIMIT_GB:.0f} GB Metal limit) "
            f"— validating squish compressed model only"
        )

    PROMPTS = [
        prompt,
        "The capital of France is",
        "def fibonacci(n):",
        "The chemical formula for water is",
        "In machine learning, gradient descent",
    ][:3]  # Keep fast — 3 prompts is enough for a quick validation

    # ── Reference (mlx_lm) ───────────────────────────────────────────────────
    ref_outputs: dict = {}
    ref_load_s: float = 0.0
    if not ref_too_large:
        info("Loading reference model (mlx_lm) …")
        try:
            import mlx_lm
            t0 = time.perf_counter()
            ref_model, ref_tok = mlx_lm.load(str(model_dir))
            ref_load_s = time.perf_counter() - t0
            ok(f"Reference loaded in {ref_load_s:.2f}s")
            for p in PROMPTS:
                try:
                    text = mlx_lm.generate(ref_model, ref_tok, prompt=p,
                                           max_tokens=max_tokens, verbose=False)
                    ref_outputs[p] = text.strip()
                except Exception as e:
                    warn(f"Reference generation failed for prompt {p!r}: {e}")
                    ref_outputs[p] = ""
            del ref_model
        except Exception as e:
            warn(f"Reference load failed: {e}")

    # ── Compressed ───────────────────────────────────────────────────────────
    info("Loading compressed model …")
    from compressed_loader import load_from_npy_dir
    import mlx_lm as _mlx_lm_comp
    t0 = time.perf_counter()
    comp_model, comp_tok, stats = load_from_npy_dir(
        str(comp_dir), model_dir=str(model_dir), verbose=False, return_stats=True
    )
    comp_load_s = time.perf_counter() - t0
    ok(f"Compressed loaded in {comp_load_s:.2f}s  (loader: {stats.get('loader')})")

    comp_outputs = {}
    for p in PROMPTS:
        try:
            text = _mlx_lm_comp.generate(comp_model, comp_tok, prompt=p,
                                         max_tokens=max_tokens, verbose=False)
            comp_outputs[p] = text.strip()
        except Exception as e:
            warn(f"Compressed generation failed for prompt {p!r}: {e}")
            comp_outputs[p] = ""
    del comp_model

    # ── Compare (text similarity) ─────────────────────────────────────────────
    print()
    agreements = []
    for p in PROMPTS:
        ref_text  = ref_outputs.get(p, "")
        comp_text = comp_outputs.get(p, "")
        short_p   = p[:40]
        if ref_too_large or not ref_text:
            # No reference — just show squish output
            ok(f"  {short_p!r:<42}  (squish output)")
            print(f"       squish: {comp_text[:80]!r}")
        else:
            # Character-level overlap as proxy for agreement
            n     = min(len(ref_text), len(comp_text))
            agree = sum(r == c for r, c in zip(ref_text[:n], comp_text[:n]))
            pct   = agree / max(n, 1) * 100
            agreements.append(pct)
            status  = GREEN + "✓" + RESET if pct >= 60 else RED + "✗" + RESET
            print(f"  {status}  {short_p!r:<42}  agree={pct:.0f}%")
            print(f"       ref:  {ref_text[:60]!r}")
            print(f"       comp: {comp_text[:60]!r}")
        print()

    if agreements:
        mean_agree = sum(agreements) / len(agreements)
        status     = GREEN + "✓" + RESET if mean_agree >= 60 else RED + "✗" + RESET
        print(f"  {status}  Mean text overlap: {mean_agree:.1f}%")
    if ref_load_s > 0:
        print(f"  Load speedup: {ref_load_s / max(comp_load_s, 0.001):.1f}×  "
              f"({ref_load_s:.2f}s ref  vs  {comp_load_s:.2f}s squish)")


# ─────────────────────────────────────────────────────────────────────────────
#  Phase 6 — Benchmark snapshot
# ─────────────────────────────────────────────────────────────────────────────

def phase_benchmark(model_dir: Path, comp_dir: Path, skip: bool, prompt: str, max_tokens: int):
    banner(6, "Benchmark  (load time × 3 + throughput)")

    if skip:
        info("Skipped (--skip-benchmark)")
        return

    est_bf16_gb = _comp_dir_bf16_gb(comp_dir)
    ref_too_large = est_bf16_gb > _METAL_TIER2_LIMIT_GB
    if ref_too_large:
        warn(f"Reference bf16 benchmark skipped ({est_bf16_gb:.1f} GB > "
             f"{_METAL_TIER2_LIMIT_GB:.0f} GB Metal limit) — squish only")

    N_RUNS = 3

    def _time_load(loader_fn, label):
        times = []
        for i in range(N_RUNS):
            t0 = time.perf_counter()
            model, tok, _ = loader_fn()
            elapsed = time.perf_counter() - t0

            # Throughput
            import mlx_lm as _mlx_lm_bench
            t1 = time.perf_counter()
            try:
                resp = _mlx_lm_bench.generate(model, tok, prompt=prompt,
                                              max_tokens=max_tokens, verbose=False)
                gen_s = time.perf_counter() - t1
                n_gen = len(tok.encode(resp, add_special_tokens=False))
                tps   = n_gen / max(gen_s, 0.001)
            except Exception:
                tps = 0.0
            del model

            times.append((elapsed, tps))
            info(f"  [{label}] run {i+1}: load={elapsed:.2f}s  gen={tps:.1f} tok/s")

        load_times = [t[0] for t in times]
        tps_vals   = [t[1] for t in times]
        return (
            sum(load_times) / len(load_times),
            min(load_times),
            sum(tps_vals) / len(tps_vals),
        )

    from compressed_loader import load_from_npy_dir
    def _comp_load():
        return load_from_npy_dir(str(comp_dir), model_dir=str(model_dir),
                                 verbose=False, return_stats=True)

    ref_mean = ref_min = ref_tps = None
    if not ref_too_large:
        def _ref_load():
            import mlx_lm
            model, tok = mlx_lm.load(str(model_dir))
            return model, tok, {}
        ref_mean, ref_min, ref_tps = _time_load(_ref_load, "reference")
        sep()

    comp_mean, comp_min, comp_tps = _time_load(_comp_load, "squish")

    speedup    = ref_mean / max(comp_mean, 0.001) if ref_mean else None
    tps_delta  = ((comp_tps - ref_tps) / max(ref_tps, 0.001)) * 100 if ref_tps else None

    sep()
    model_name = model_dir.name
    ref_mean_s = f"{ref_mean:.2f}s" if ref_mean is not None else "N/A (too large)"
    ref_min_s  = f"{ref_min:.2f}s"  if ref_mean is not None else "N/A"
    ref_tps_s  = f"{ref_tps:.1f}"   if ref_tps  is not None else "N/A"
    spd_s      = f"{speedup:.1f}×"  if speedup  is not None else "N/A"
    tps_s      = f"{tps_delta:+.1f}%" if tps_delta is not None else "N/A"
    print(f"\n{BOLD}{'─'*62}{RESET}")
    print(f"{BOLD}  Benchmark Results — {model_name}{RESET}")
    print(f"{'─'*62}")
    print(f"  {'Metric':<30}  {'Reference':>12}  {'Squish':>12}")
    print(f"  {'─'*28}  {'─'*12}  {'─'*12}")
    print(f"  {'Load time (mean)' :<30}  {ref_mean_s:>12}  {comp_mean:>10.2f}s")
    print(f"  {'Load time (best)' :<30}  {ref_min_s:>12}  {comp_min:>10.2f}s")
    print(f"  {'Throughput (tok/s)':<30}  {ref_tps_s:>12}   {comp_tps:>10.1f}")
    print(f"  {'Load speedup'      :<30}  {'':>12}  {spd_s:>12}")
    print(f"  {'Throughput delta'  :<30}  {'':>12}  {tps_s:>12}")
    print(f"{'─'*62}\n")

    # Write JSON snapshot for RESULTS.md
    snap = {
        "model":           model_name,
        "date":            time.strftime("%Y-%m-%d"),
        "ref_load_s":      round(ref_mean, 3) if ref_mean is not None else None,
        "comp_load_s":     round(comp_mean, 3),
        "speedup":         round(speedup, 1)   if speedup   is not None else None,
        "ref_tps":         round(ref_tps, 1)   if ref_tps   is not None else None,
        "comp_tps":        round(comp_tps, 1),
        "tps_delta_pct":   round(tps_delta, 1) if tps_delta is not None else None,
    }
    snap_path = POC_DIR / f"benchmark_{model_name}.json"
    with open(snap_path, "w") as f:
        json.dump(snap, f, indent=2)
    ok(f"Benchmark snapshot saved → {snap_path.name}")

    return snap


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Download, compress, and benchmark any HF model for Squish",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("model_id",
                    help="HuggingFace model ID, e.g. Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--model-dir",
                    help="Local path for the bf16 model")
    ap.add_argument("--compressed-dir",
                    help="Local path for the Vectro compressed output")
    ap.add_argument("--skip-download",    action="store_true")
    ap.add_argument("--skip-compress",    action="store_true")
    ap.add_argument("--skip-cache",       action="store_true")
    ap.add_argument("--skip-validation",  action="store_true")
    ap.add_argument("--skip-benchmark",   action="store_true")
    ap.add_argument("--passthrough",      nargs="*",   default=[],
                    help="Tensor names to keep as float16 (e.g. embed_tokens lm_head)")
    ap.add_argument("--outlier-threshold", type=float, default=20.0)
    ap.add_argument("--prompt",           default="The capital of France is",
                    help="Prompt for validation + benchmark inference")
    ap.add_argument("--max-tokens",       type=int,    default=32,
                    help="Max tokens for validation/benchmark generation")
    args = ap.parse_args()

    short_name = args.model_id.split("/")[-1]
    # Normalise: strip existing -bf16 suffix so we don't double it
    base_name  = short_name.rstrip("-bf16")
    bf16_name  = base_name + "-bf16"

    model_dir  = Path(args.model_dir).expanduser()  if args.model_dir  else MODELS_ROOT / bf16_name
    comp_dir   = Path(args.compressed_dir).expanduser() if args.compressed_dir else MODELS_ROOT / (bf16_name + "-compressed")

    print(f"\n{BOLD}{CYAN}{'═'*62}{RESET}")
    print(f"{BOLD}{CYAN}  Squish — Pull & Compress  [{args.model_id}]{RESET}")
    print(f"{BOLD}{CYAN}{'═'*62}{RESET}")
    print(f"  Model dir:      {model_dir}")
    print(f"  Compressed dir: {comp_dir}")
    print()

    t_total = time.perf_counter()

    phase_download(args.model_id,  model_dir, args.skip_download)
    phase_compress(model_dir, comp_dir, args.skip_compress,
                   args.passthrough, args.outlier_threshold)
    phase_build_cache(model_dir, comp_dir, args.skip_cache,
                      args.prompt, args.max_tokens)
    phase_validate(model_dir, comp_dir, args.skip_validation,
                   args.prompt, args.max_tokens)
    snap = phase_benchmark(model_dir, comp_dir, args.skip_benchmark,
                           args.prompt, args.max_tokens)

    elapsed = time.perf_counter() - t_total
    print(f"\n{BOLD}{GREEN}{'═'*62}{RESET}")
    print(f"{BOLD}{GREEN}  Done!  Total time: {elapsed/60:.1f} min{RESET}")
    if snap:
        print(f"  {snap['model']}: {snap['speedup']}× faster load  "
              f"({snap['ref_load_s']}s → {snap['comp_load_s']}s)")
    print(f"{BOLD}{GREEN}{'═'*62}{RESET}\n")
    print(f"  Next steps:")
    print(f"  {CYAN}squish serve --model {base_name}{RESET}")
    print(f"  {CYAN}squish run \"What is attention in transformers?\"{RESET}")
    print(f"  {CYAN}python3 run_eval.py --model-type squish{RESET}")
    print()


if __name__ == "__main__":
    main()
