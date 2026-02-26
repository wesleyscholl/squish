#!/usr/bin/env python3
"""
benchmark_comparison.py

Side-by-side benchmark comparison across multiple model sizes:
  1.5B → 7B → 14B → 32B  (as each becomes available locally)

For each model, measures:
  - Reference load time   (mlx_lm native safetensors)
  - Squish load time      (Tier 2 squish_weights.safetensors)
  - Load speedup          (×)
  - Throughput ref        (tok/s)
  - Throughput squish     (tok/s)
  - Model size on disk    (GB)
  - Compressed size       (GB)
  - lm-eval accuracy (optional, takes hours — use --no-bench-accuracy to skip)

Usage:
    # Quick run — just load times + throughput, no accuracy eval
    python3 benchmark_comparison.py

    # Full run with lm-eval accuracy on all available models
    python3 benchmark_comparison.py --accuracy --tasks arc_easy,hellaswag --limit 200

    # Specific models only
    python3 benchmark_comparison.py --models 7b,14b

Output:
    benchmark_multi_model.json   — machine-readable results
    benchmark_multi_model.md     — publication-quality comparison table
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

POC_DIR = Path(__file__).resolve().parent.parent.resolve()
sys.path.insert(0, str(POC_DIR))

GREEN   = ""
RED     = ""
YELLOW  = ""
CYAN    = ""
MAGENTA = ""
BOLD    = ""
DIM     = ""
RESET   = ""

MODELS_ROOT = Path.home() / "models"

# ── Model registry ─────────────────────────────────────────────────────────────
# Each entry: (label, bf16_dir_name, tag_for_cli)
MODEL_CONFIGS = [
    ("Qwen2.5-1.5B", "Qwen2.5-1.5B-Instruct-bf16",  "1.5b"),
    ("Qwen2.5-3B",   "Qwen2.5-3B-Instruct-bf16",    "3b"),
    ("Qwen2.5-7B",   "Qwen2.5-7B-Instruct-bf16",    "7b"),
    ("Qwen2.5-14B",  "Qwen2.5-14B-Instruct-bf16",   "14b"),
    ("Qwen2.5-32B",  "Qwen2.5-32B-Instruct-bf16",   "32b"),
    ("Llama3.2-3B",  "Llama-3.2-3B-Instruct-bf16",  "llama3b"),
    ("Llama3.1-8B",  "Meta-Llama-3.1-8B-Instruct-bf16", "llama8b"),
]

DEFAULT_PROMPT = (
    "Explain the transformer attention mechanism in one concise paragraph."
)
N_TIMING_RUNS  = 1   # 1 run per strategy to avoid Metal memory accumulation
                     # (increase to 3+ only if Metal memory is fully reclaimed between runs)
MAX_TOKENS     = 64
# bf16 models above this size (GB) OOM-abort on 16 GB Apple Silicon Metal;
# reference timing is auto-skipped for them (Squish-only timing still runs).
REF_MAX_GB     = 12.0


def _du_gb(path: Path) -> float:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1e9


def _available_models(filter_tags: list[str] | None) -> list[tuple[str, Path, Path]]:
    """
    Return list of (label, model_dir, comp_dir) for models that are
    present locally (have both bf16 safetensors and a compressed npy-dir).
    """
    available = []
    for label, bf16_name, tag in MODEL_CONFIGS:
        if filter_tags and tag not in filter_tags:
            continue
        model_dir = MODELS_ROOT / bf16_name
        comp_dir  = MODELS_ROOT / (bf16_name + "-compressed")

        has_model = model_dir.exists() and list(model_dir.glob("*.safetensors"))
        has_comp  = comp_dir.exists()  and (comp_dir / "tensors").exists()

        if has_model and has_comp:
            available.append((label, model_dir, comp_dir))

    return available


def _time_load_ref(model_dir: Path) -> tuple[float | None, float | None]:
    """(mean_load_s, mean_tps)  — each run isolated in a subprocess for clean Metal state."""
    import json
    import subprocess

    _SCRIPT = """
import time, json, sys
sys.path.insert(0, {poc_dir!r})
import mlx_lm
t0 = time.perf_counter()
model, tok = mlx_lm.load({model_dir!r})
load_s = time.perf_counter() - t0
resp = mlx_lm.generate(model, tok, prompt={prompt!r}, max_tokens={max_tok}, verbose=False)
gen_s = time.perf_counter() - (t0 + load_s)
n_gen = len(tok.encode(resp, add_special_tokens=False))
tps = n_gen / max(gen_s, 0.001)
print(json.dumps({{"load_s": load_s, "tps": tps}}))
""".format(poc_dir=str(POC_DIR), model_dir=str(model_dir),
           prompt=DEFAULT_PROMPT, max_tok=MAX_TOKENS)

    times: list[tuple[float, float]] = []
    for run in range(N_TIMING_RUNS):
        try:
            result = subprocess.run(
                [sys.executable, "-c", _SCRIPT],
                capture_output=True, text=True, timeout=300,
            )
            if result.returncode != 0:
                # OOM: libc++abi abort (returncode 134 / SIGABRT)
                return None, None
            data = json.loads(result.stdout.strip())
            times.append((data["load_s"], data["tps"]))
        except (subprocess.TimeoutExpired, json.JSONDecodeError, Exception):
            return None, None

    if not times:
        return None, None

    return (
        sum(t[0] for t in times) / len(times),
        sum(t[1] for t in times) / len(times),
    )


def _time_load_squish(model_dir: Path, comp_dir: Path) -> tuple[float | None, float | None, str]:
    """(mean_load_s, mean_tps_or_None, loader_name)  — each run isolated in a subprocess.
    tps is None when the model is too large to run inference on this machine.
    Returns (None, None, '<reason>') when the Q8 → bf16 expansion would exceed Metal budget.
    """
    import json
    import subprocess

    # ── Metal size gate: raise limit since compressed_loader auto-quantizes
    # to 4-bit for large models, keeping Metal usage to ~Q8_size/2 (~4.7 GB for 7B).
    # Only block models where even 4-bit won't fit (est. bf16 > ~58 GB ≈ 14+ GB 4-bit).
    _METAL_LIMIT_GB = 58.0
    # Measure only the tensors/ subdir (raw Q8 data), not tier2 or finalized caches
    _tensors_dir = comp_dir / "tensors"
    comp_q8_gb   = _du_gb(_tensors_dir if _tensors_dir.is_dir() else comp_dir)
    est_bf16_gb  = comp_q8_gb * 2.1
    if est_bf16_gb > _METAL_LIMIT_GB:
        return None, None, f"oom-gated ({est_bf16_gb:.1f} GB bf16 > {_METAL_LIMIT_GB:.0f} GB Metal limit)"

    # Only attempt generation when model can fit comfortably with KV cache overhead.
    # With 4-bit auto-quantize, 7B uses ~4 GB weights → generation is fine up to ~14B.
    tier2 = comp_dir / "squish_weights.safetensors"
    tier2_gb = tier2.stat().st_size / 1e9 if tier2.exists() else 0
    # Use Q8 tensors dir as proxy when no tier2 cache (large models use npy-dir + 4-bit)
    _tdir = comp_dir / "tensors"
    _q8_gb = _du_gb(_tdir if _tdir.is_dir() else comp_dir)
    run_gen = (_q8_gb * 2.0) < 30.0   # allow gen for models where bf16 est. < 30 GB (≤ 7B)

    _SCRIPT = """
import time, json, sys, os
sys.path.insert(0, {poc_dir!r})
from compressed_loader import load_from_npy_dir
t0 = time.perf_counter()
model, tok, stats = load_from_npy_dir(
    {comp_dir!r},
    model_dir={model_dir!r},
    verbose=False,
    return_stats=True,
)
load_s = time.perf_counter() - t0
loader = stats.get("loader", "?")
tps = None
if {run_gen!r}:
    try:
        import mlx_lm
        resp = mlx_lm.generate(model, tok, prompt={prompt!r},
                               max_tokens={max_tok}, verbose=False)
        gen_s = time.perf_counter() - (t0 + load_s)
        n_gen = len(tok.encode(resp, add_special_tokens=False))
        tps = n_gen / max(gen_s, 0.001)
    except Exception:
        tps = None
print(json.dumps({{"load_s": load_s, "tps": tps, "loader": loader}}))
""".format(poc_dir=str(POC_DIR), model_dir=str(model_dir), comp_dir=str(comp_dir),
           prompt=DEFAULT_PROMPT, max_tok=MAX_TOKENS, run_gen=run_gen)

    times: list[tuple[float, float | None]] = []
    loader = "?"
    for run in range(N_TIMING_RUNS):
        try:
            result = subprocess.run(
                [sys.executable, "-c", _SCRIPT],
                capture_output=True, text=True, timeout=900,   # 15 min: 7B first decomp ~5 min
            )
            # Parse whatever the subprocess printed before any abort
            stdout = result.stdout.strip()
            if stdout:
                try:
                    data = json.loads(stdout)
                    loader = data.get("loader", "?")
                    times.append((data["load_s"], data.get("tps")))
                    continue
                except json.JSONDecodeError:
                    pass
            if result.returncode != 0:
                stderr_snippet = result.stderr[:300]
                raise RuntimeError(f"Squish timing subprocess failed (exit {result.returncode}): {stderr_snippet}")
        except subprocess.TimeoutExpired:
            raise RuntimeError("Squish timing subprocess timed out (900s)")

    if not times:
        raise RuntimeError("No timing results collected from squish subprocess")

    mean_load = sum(t[0] for t in times) / len(times)
    valid_tps = [t[1] for t in times if t[1] is not None]
    mean_tps  = sum(valid_tps) / len(valid_tps) if valid_tps else None
    return mean_load, mean_tps, loader


def _run_quick_accuracy(label: str, model_dir: Path, comp_dir: Path,
                        tasks: list[str], limit: int) -> dict:
    """Run lm-eval on a single model, return {task: {ref_acc, comp_acc}}."""
    import squish_lm_eval   # registers model types
    from run_eval import run_lm_eval, _extract_metric, POC_DIR as _

    out_dir = POC_DIR / "eval_output" / label.replace(" ", "_")
    out_dir.mkdir(parents=True, exist_ok=True)

    ref_args  = f"model_dir={model_dir}"
    comp_args = (
        f"model_dir={model_dir},"
        f"compressed_dir={comp_dir},"
        f"verbose=False"
    )
    ref_r  = run_lm_eval("squish-reference", ref_args,  tasks, limit, {},
                          out_dir / "ref.json",  f"{label}-ref")
    comp_r = run_lm_eval("squish",            comp_args, tasks, limit, {},
                          out_dir / "comp.json", f"{label}-comp")

    result = {}
    for task in tasks:
        r_acc  = _extract_metric(ref_r,  task) if ref_r  else None
        c_acc  = _extract_metric(comp_r, task) if comp_r else None
        delta  = (c_acc - r_acc) if (r_acc and c_acc) else None
        result[task] = {"ref": r_acc, "comp": c_acc, "delta": delta}
    return result


def _banner(msg: str):
    print(f"\n{BOLD}{CYAN}{'═'*68}{RESET}")
    print(f"{BOLD}{CYAN}  {msg}{RESET}")
    print(f"{BOLD}{CYAN}{'═'*68}{RESET}\n")


def main():
    ap = argparse.ArgumentParser(
        description="Multi-model Squish benchmark comparison",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--models",   default=None,
                    help="Comma-separated model tags to test (e.g. 1.5b,7b,14b). "
                         "Default: all available.")
    ap.add_argument("--accuracy", action="store_true",
                    help="Also run lm-eval accuracy benchmarks (slow: 10-60 min/model)")
    ap.add_argument("--tasks",    default="arc_easy,hellaswag",
                    help="lm-eval tasks for accuracy (only used with --accuracy)")
    ap.add_argument("--limit",    type=int, default=200,
                    help="Examples per task for lm-eval")
    ap.add_argument("--skip-ref", action="store_true",
                    help="Skip reference (mlx_lm) load timing (faster, squish-only)")
    ap.add_argument("--output",   default=str(POC_DIR / "benchmark_multi_model.json"))
    args = ap.parse_args()

    filter_tags = [t.strip() for t in args.models.split(",")] if args.models else None
    models = _available_models(filter_tags)

    if not models:
        print(f"\n{RED}No models found in {MODELS_ROOT}.{RESET}")
        print(f"Download one with:")
        print(f"  {CYAN}python3 pull_model.py Qwen/Qwen2.5-7B-Instruct{RESET}")
        sys.exit(1)

    print(f"\n{BOLD}Squish Multi-Model Benchmark{RESET}")
    print(f"Models found: {', '.join(lbl for lbl, _, _ in models)}")
    print(f"Timing runs:  {N_TIMING_RUNS} per strategy")
    print(f"Accuracy:     {'yes — ' + args.tasks if args.accuracy else 'skipped (use --accuracy)'}\n")

    all_results = []

    for label, model_dir, comp_dir in models:
        _banner(f"{label}")
        row: dict = {"model": label, "timestamp": time.strftime("%Y-%m-%d %H:%M")}

        # Disk sizes
        row["model_gb"]   = round(_du_gb(model_dir), 1)
        row["comp_gb"]    = round(_du_gb(comp_dir),  1)
        row["size_ratio"] = round(row["model_gb"] / max(row["comp_gb"], 0.01), 2)
        print(f"  Model:      {row['model_gb']:.1f} GB")
        print(f"  Compressed: {row['comp_gb']:.1f} GB  ({row['size_ratio']:.2f}× smaller)\n")

        # Reference timing
        # Auto-skip for models that exceed Metal memory (OOM-abort on 16 GB M-series)
        model_too_large = row["model_gb"] > REF_MAX_GB
        if not args.skip_ref and not model_too_large:
            print(f"  {CYAN}\u25b8 Reference (mlx_lm) timing \u2026{RESET}")
            t0 = time.perf_counter()
            ref_load_s, ref_tps = _time_load_ref(model_dir)
            if ref_load_s is None:
                print(f"    {YELLOW}Reference OOM (model too large for 16 GB Metal){RESET}")
                row["ref_load_s"] = None
                row["ref_tps"]    = None
            else:
                print(f"    load: {ref_load_s:.2f}s   throughput: {ref_tps:.1f} tok/s")
                row["ref_load_s"]  = round(ref_load_s, 3)
                row["ref_tps"]     = round(ref_tps,    1)
        elif model_too_large and not args.skip_ref:
            print(f"  {YELLOW}\u25b8 Reference timing skipped  "
                  f"(model {row['model_gb']:.1f} GB > {REF_MAX_GB} GB Metal limit){RESET}")
            row["ref_load_s"] = None
            row["ref_tps"]    = None
        else:
            row["ref_load_s"]  = None
            row["ref_tps"]     = None

        # Squish timing
        print(f"  Squish timing ...")
        comp_load_s, comp_tps, loader = _time_load_squish(model_dir, comp_dir)
        if comp_load_s is None:
            print(f"    {YELLOW}Squish load skipped: {loader}{RESET}")
            row["comp_load_s"] = None
            row["comp_tps"]    = None
            row["loader"]      = loader
            all_results.append(row)
            print()
            continue
        row["comp_load_s"] = round(comp_load_s, 3)
        row["comp_tps"]    = round(comp_tps, 1) if comp_tps is not None else None
        row["loader"]      = loader
        tps_str = f"{comp_tps:.1f} tok/s" if comp_tps is not None else "N/A (inference skipped — model too large)"
        print(f"    load: {comp_load_s:.2f}s   throughput: {tps_str}   [{loader}]")

        if row["ref_load_s"]:
            speedup = row["ref_load_s"] / max(row["comp_load_s"], 0.001)
            row["speedup"] = round(speedup, 1)
            if comp_tps is not None and ref_tps is not None:
                tps_pct = ((comp_tps - ref_tps) / max(ref_tps, 0.001)) * 100
                row["tps_delta_pct"] = round(tps_pct, 1)
                print(f"\n  Load speedup: {speedup:.1f}x  Throughput delta: {tps_pct:+.1f}%")
            else:
                row["tps_delta_pct"] = None
                print(f"\n  Load speedup: {speedup:.1f}x")

        # Optional accuracy
        if args.accuracy:
            print(f"\n  {CYAN}▸ lm-eval accuracy ({args.tasks}, {args.limit}/task) …{RESET}")
            tasks_list = [t.strip() for t in args.tasks.split(",")]
            try:
                acc = _run_quick_accuracy(label, model_dir, comp_dir, tasks_list, args.limit)
                row["accuracy"] = acc
                for task, vals in acc.items():
                    r, c, d = vals.get("ref"), vals.get("comp"), vals.get("delta")
                    sym = GREEN+"✓"+RESET if d is not None and abs(d) <= 0.03 else YELLOW+"⚠"+RESET
                    r_s = f"{r*100:.1f}%" if r else "—"
                    c_s = f"{c*100:.1f}%" if c else "—"
                    d_s = f"{d*100:+.1f}%" if d is not None else "—"
                    print(f"    {sym} {task:<20} ref={r_s}  squish={c_s}  delta={d_s}")
            except Exception as e:
                print(f"    {YELLOW}Accuracy eval failed: {e}{RESET}")

        all_results.append(row)
        print()

    # ── Print final table ──────────────────────────────────────────────────────
    _banner("Results Summary")
    has_ref  = any(r.get("ref_load_s") for r in all_results)
    has_tps  = any(r.get("ref_tps") for r in all_results)

    # Header
    c0, c1, c2, c3, c4, c5, c6 = 16, 10, 10, 10, 10, 7, 7
    hdr = (f"  {'Model':<{c0}}  {'Size(GB)':>{c1}}  {'Comp(GB)':>{c2}}"
           + (f"  {'Ref load':>{c3}}" if has_ref else "")
           + f"  {'Sq load':>{c4}}"
           + (f"  {'Speedup':>{c5}}" if has_ref else "")
           + (f"  {'Tok/s Δ':>{c6}}" if has_tps else ""))
    print(hdr)
    print("  " + "─" * (len(hdr) - 2))

    for row in all_results:
        ref_s   = f"{row['ref_load_s']:.2f}s"  if row.get("ref_load_s") else "—"
        comp_s  = f"{row['comp_load_s']:.2f}s"  if row.get("comp_load_s") else "—"
        spd_s   = f"{row['speedup']:.1f}×"     if row.get("speedup")    else "—"
        tps_s   = f"{row['tps_delta_pct']:+.1f}%" if row.get("tps_delta_pct") is not None else "—"
        line = (f"  {row['model']:<{c0}}  {row['model_gb']:>{c1}.1f}  {row['comp_gb']:>{c2}.1f}"
                + (f"  {ref_s:>{c3}}"  if has_ref  else "")
                + f"  {comp_s:>{c4}}"
                + (f"  {spd_s:>{c5}}" if has_ref  else "")
                + (f"  {tps_s:>{c6}}" if has_tps  else ""))
        print(line)

    print()

    # ── Save JSON ──────────────────────────────────────────────────────────────
    out_path = Path(args.output)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"  {GREEN}✓{RESET}  Results saved → {out_path}")

    # ── Write Markdown ────────────────────────────────────────────────────────
    md_path = out_path.with_suffix(".md")
    _write_markdown(all_results, md_path, args.accuracy)
    print(f"  {GREEN}✓{RESET}  Markdown  saved → {md_path}\n")


def _write_markdown(results: list[dict], path: Path, include_accuracy: bool):
    lines = [
        "# Squish — Multi-Model Benchmark Results",
        "",
        "## Load Time & Throughput",
        "",
        "| Model | Size (GB) | Compressed (GB) | Ref Load | Squish Load | Speedup | Tok/s Δ |",
        "|-------|-----------|-----------------|----------|-------------|---------|---------|",
    ]
    for r in results:
        ref_s  = f"{r['ref_load_s']:.2f}s"   if r.get("ref_load_s") else "—"
        comp_s = f"{r['comp_load_s']:.2f}s"  if r.get("comp_load_s") else "—"
        spd_s  = f"**{r['speedup']:.1f}×**"  if r.get("speedup")    else "—"
        tps_s  = f"{r['tps_delta_pct']:+.1f}%" if r.get("tps_delta_pct") is not None else "—"
        lines.append(
            f"| {r['model']} | {r.get('model_gb','—')} | {r.get('comp_gb','—')} "
            f"| {ref_s} | {comp_s} | {spd_s} | {tps_s} |"
        )

    if include_accuracy:
        lines += ["", "## Accuracy (lm-eval)", ""]
        # Collect all tasks
        all_tasks: set[str] = set()
        for r in results:
            all_tasks.update((r.get("accuracy") or {}).keys())
        if all_tasks:
            task_list = sorted(all_tasks)
            hdr = "| Model | " + " | ".join(
                f"{t} ref | {t} squish | {t} Δ" for t in task_list
            ) + " |"
            lines.append(hdr)
            lines.append("|" + "---|" * (1 + 3 * len(task_list)))
            for r in results:
                acc = r.get("accuracy") or {}
                cols = []
                for t in task_list:
                    v = acc.get(t, {})
                    rv = f"{v['ref']*100:.1f}%"  if v.get("ref")  else "—"
                    cv = f"{v['comp']*100:.1f}%" if v.get("comp") else "—"
                    dv = f"{v['delta']*100:+.1f}%" if v.get("delta") is not None else "—"
                    cols += [rv, cv, dv]
                lines.append(f"| {r['model']} | " + " | ".join(cols) + " |")

    lines += [
        "",
        "## Notes",
        "",
        "- **Squish load** uses Tier 2 cache (`squish_weights.safetensors`) — sub-second on warm runs",
        "- **Tok/s Δ** = throughput change vs reference (Vectro INT8 has near-zero quality impact)",
        "- Benchmarks run on Apple M3 16GB unified memory",
        "- lm-eval harness: EleutherAI lm-evaluation-harness v0.4.x",
        "",
    ]
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
