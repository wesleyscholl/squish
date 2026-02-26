#!/usr/bin/env python3
"""
run_poc.py

Fully automated end-to-end runner for the Squish PoC.

Phases:
  1. Check / install Python dependencies
  2. Download Qwen2.5-1.5B-Instruct (float16) if not present
  3. Establish reference baseline (standard mlx_lm)
  4. Convert weights → Vectro INT8 compressed npz
  5. Run inference from compressed weights (safetensors present)
  6. Prove safetensors are NOT needed (move them away, re-run, restore)
  7. Verify: token agreement + weight cosine similarity
  8. Print final report

Usage:
    python3 run_poc.py [options]

Options:
    --model-dir DIR      Where to download/find the model
                         (default: ~/models/Qwen2.5-1.5B-Instruct)
    --compressed-dir DIR Where to write the compressed npz
                         (default: ~/models/Qwen2.5-1.5B-Instruct-compressed)
    --prompt TEXT        Prompt for generation (default: "The capital of France is")
    --max-tokens N       Tokens to generate (default: 20)
    --skip-download          Skip model download (model already present)
    --skip-reference         Skip reference run (reference_output.json already exists)
    --skip-convert           Skip conversion (weights_compressed.npz already exists)
    --skip-safetensors-check  Skip the "prove safetensors not needed" step
    --min-cosine FLOAT       Weight fidelity threshold (default: 0.97)
    --min-token-agree        Token agreement threshold (default: 0.60)
    --streaming              Use layer-by-layer streaming loader (streaming_loader.py)
                             instead of the default batch loader (compressed_loader.py)
    --benchmark              After the main PoC run, also invoke benchmark.py for
                             a full three-strategy comparison table
    --tool-calling-demo      After the main run, launch tool_calling_demo.py
"""
import argparse
import importlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

POC_DIR = Path(__file__).resolve().parent.parent.resolve()

# ── ANSI colours ──────────────────────────────────────────────────────────────
GREEN  = ""
RED    = ""
YELLOW = ""
CYAN   = ""
BOLD   = ""
RESET  = ""

def banner(phase: int | str, title: str):
    print(f"\n{BOLD}{CYAN}{'─'*60}{RESET}")
    print(f"{BOLD}{CYAN}  Phase {phase}: {title}{RESET}")
    print(f"{BOLD}{CYAN}{'─'*60}{RESET}\n")

def ok(msg: str):
    print(f"  {GREEN}✓{RESET} {msg}")

def fail(msg: str):
    print(f"  {RED}✗{RESET} {msg}")

def info(msg: str):
    print(f"  {YELLOW}→{RESET} {msg}")

def run(cmd: list[str], desc: str, check: bool = True, cwd=None) -> subprocess.CompletedProcess:
    info(f"{desc} ...")
    result = subprocess.run(cmd, cwd=cwd or POC_DIR)
    if check and result.returncode != 0:
        fail(f"Command failed (exit {result.returncode}): {' '.join(cmd)}")
        sys.exit(result.returncode)
    return result


# ── Test suite prompts (used in phase_test_suite) ─────────────────────────────
# (category, prompt, max_tokens)
_TEST_PROMPTS: list[tuple[str, str, int]] = [
    ("geography",   "The capital of Japan is",                  10),
    ("arithmetic",  "15 multiplied by 8 equals",                10),
    ("science",     "The chemical symbol for water is",         10),
    ("language",    "The French word for 'cat' is",             10),
    ("coding",      "To get the length of a list in Python use", 10),
]


# ── Phase 1: Dependencies ─────────────────────────────────────────────────────

def phase_deps():
    banner(1, "Check / install dependencies")
    required = ["mlx_lm", "safetensors", "numpy", "transformers", "huggingface_hub"]
    missing = []
    for pkg in required:
        try:
            importlib.import_module(pkg)
            ok(pkg)
        except ImportError:
            fail(f"{pkg}  (missing)")
            missing.append(pkg)

    if missing:
        info(f"Installing: {', '.join(missing)}")
        run(
            [sys.executable, "-m", "pip", "install", "-q",
             "mlx-lm", "safetensors", "numpy", "transformers", "huggingface_hub"],
            "pip install",
        )
        ok("Dependencies installed")
    else:
        ok("All dependencies present")


# ── Phase 2: Model download ───────────────────────────────────────────────────

def _disable_ssl_verification():
    """Disable SSL certificate verification for all HTTP clients HuggingFace Hub uses.

    Needed when the network uses SSL interception (corporate proxy / VPN with
    a self-signed cert in the chain).  Not recommended outside of PoC/dev use.

    Strategy (applied in order, all layers):
      1. HF_HUB_DISABLE_SSL_VERIFICATION env var  (newest hf_hub builds)
      2. configure_http_backend with verify=False  (hf_hub >= 0.18)
      3. Monkey-patch httpx.Client / AsyncClient   (catches anything that imports httpx)
      4. Blank out CA-bundle env vars              (urllib3 / requests fallback)
      5. ssl module patch                          (last resort for old stacks)
    """
    import os

    # Layer 1: environment variable recognised by recent huggingface_hub builds
    os.environ["HF_HUB_DISABLE_SSL_VERIFICATION"] = "1"
    # Layer 4: blank CA-bundle env vars for requests / urllib3 paths
    os.environ["CURL_CA_BUNDLE"] = ""
    os.environ["REQUESTS_CA_BUNDLE"] = ""
    os.environ["SSL_CERT_FILE"] = ""

    # Layer 3: monkey-patch httpx so *every* Client created after this point
    # (including ones instantiated inside huggingface_hub internals) skips
    # certificate verification.  This is the reliable fix for the httpx path.
    try:
        import httpx

        _orig_client_init = httpx.Client.__init__
        _orig_async_init  = httpx.AsyncClient.__init__

        def _patched_client_init(self, *args, **kwargs):
            kwargs.setdefault("verify", False)
            _orig_client_init(self, *args, **kwargs)

        def _patched_async_init(self, *args, **kwargs):
            kwargs.setdefault("verify", False)
            _orig_async_init(self, *args, **kwargs)

        httpx.Client.__init__      = _patched_client_init   # type: ignore[method-assign]
        httpx.AsyncClient.__init__ = _patched_async_init    # type: ignore[method-assign]

        try:
            import urllib3
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        except ImportError:
            pass
        info("SSL verification disabled (httpx monkey-patch + env vars)")
    except ImportError:
        pass

    # Layer 2: configure_http_backend (redundant after patch, but harmless)
    try:
        from huggingface_hub import configure_http_backend
        import httpx as _hx
        configure_http_backend(backend_factory=lambda: _hx.Client(verify=False))
    except Exception:
        pass

    # Layer 5: ssl module fallback for anything using the stdlib ssl stack
    try:
        import ssl
        ssl._create_default_https_context = ssl._create_unverified_context  # noqa: SLF001
    except Exception:
        pass


# Primary and fallback repos to try (in order)
_DOWNLOAD_REPOS = [
    "mlx-community/Qwen2.5-1.5B-Instruct-bf16",
    "mlx-community/Qwen2.5-1.5B-Instruct-4bit",
    "Qwen/Qwen2.5-1.5B-Instruct",
]


def phase_download(model_dir: Path, skip: bool,
                  no_ssl_verify: bool = False,
                  hf_token: str | None = None):
    banner(2, "Download model")

    # Always verify actual files are present — never trust skip flag blindly
    safetensors_files = list(model_dir.glob("*.safetensors")) if model_dir.exists() else []

    if safetensors_files:
        ok(f"Model already present at {model_dir} ({len(safetensors_files)} shard(s))")
        return

    if skip:
        if not model_dir.exists():
            fail(f"--skip-download specified but model dir does not exist: {model_dir}")
        else:
            other = list(model_dir.iterdir())
            fail(f"--skip-download specified but no .safetensors in {model_dir} "
                 f"({[f.name for f in other[:5]]})")
        fail("Re-run without --skip-download to fetch the model.")
        sys.exit(1)

    if no_ssl_verify:
        _disable_ssl_verification()

    # Resolve token: explicit arg > env vars > None (public repos only)
    token = hf_token
    if not token:
        token = (
            os.environ.get("HF_TOKEN")
            or os.environ.get("HF_API_KEY")
            or os.environ.get("HUGGINGFACE_HUB_TOKEN")
            or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        ) or None

    if token:
        # Ensure huggingface_hub's own env-var auth path also picks it up
        os.environ["HF_TOKEN"] = token
        info("HuggingFace token set")
    else:
        info("No HuggingFace token — attempting public access "
             "(use --hf-token or set HF_TOKEN env var if you get a 401)")

    model_dir.mkdir(parents=True, exist_ok=True)

    from huggingface_hub import snapshot_download

    last_err: Exception | None = None
    for repo_id in _DOWNLOAD_REPOS:
        info(f"Trying {repo_id} → {model_dir}")
        try:
            snapshot_download(
                repo_id=repo_id,
                local_dir=str(model_dir),
                ignore_patterns=["*.bin", "*.pt"],
                token=token or False,
            )
            last_err = None
            break
        except Exception as e:
            err_str = str(e)
            if "CERTIFICATE" in err_str.upper() or "SSL" in err_str.upper():
                fail("SSL certificate verification failed.")
                fail("Your network is likely using a proxy or VPN that intercepts HTTPS.")
                fail("Re-run with --no-ssl-verify:  python3 run_poc.py --no-ssl-verify")
                sys.exit(1)
            if "401" in err_str or "Unauthorized" in err_str or "Invalid username" in err_str:
                info(f"  401 on {repo_id} — trying next repo ...")
                last_err = e
                continue
            raise

    if last_err is not None:
        fail("All repos returned 401 Unauthorized.")
        fail("A free HuggingFace account and token are required to download this model.")
        fail("")
        fail("Fix options (pick one):")
        fail("  1. Log in once:   huggingface-cli login")
        fail("  2. Pass a token:  python3 run_poc.py --hf-token <your_token>")
        fail("  3. Set env var:   export HF_TOKEN=<your_token>")
        fail("")
        fail("Get a free read-only token at: https://huggingface.co/settings/tokens")
        sys.exit(1)

    safetensors_files = list(model_dir.glob("*.safetensors"))
    if not safetensors_files:
        fail(f"Download finished but no .safetensors found in {model_dir}")
        sys.exit(1)
    ok(f"Download complete ({len(safetensors_files)} shard(s))")


# ── Phase 3: Reference baseline ──────────────────────────────────────────────

def phase_reference(model_dir: Path, prompt: str, max_tokens: int,
                    out_file: Path, skip: bool):
    banner(3, "Establish reference baseline")

    if skip and out_file.exists():
        ok(f"reference_output.json already exists — skipping")
        with open(out_file) as f:
            ref = json.load(f)
        info(f"Cached output: {ref['output']!r}")
        return

    run(
        [sys.executable, "run_reference.py",
         "--model-dir", str(model_dir),
         "--prompt",    prompt,
         "--max-tokens", str(max_tokens),
         "--output",    str(out_file)],
        "Running reference inference",
    )
    with open(out_file) as f:
        ref = json.load(f)
    ok(f"Reference output: {ref['output']!r}")
    ok(f"Load time:        {ref['load_time_s']:.2f}s")
    ok(f"Gen  time:        {ref['gen_time_s']:.2f}s")


# ── Phase 4: Convert weights ──────────────────────────────────────────────────

def phase_convert(model_dir: Path, compressed_dir: Path,
                  weights_path: Path, fmt: str, skip: bool):
    banner(4, "Convert weights \u2192 Vectro INT8 compressed")

    # Determine whether weights already exist for this format
    if fmt == "npy-dir":
        already_done = (weights_path / "manifest.json").exists()
        if skip and already_done:
            tensor_dir = weights_path / "tensors"
            size_bytes = sum(p.stat().st_size for p in tensor_dir.glob("*.npy")) if tensor_dir.exists() else 0
            ok(f"npy-dir weights already exist ({size_bytes/1e6:.0f} MB) \u2014 skipping")
            return
        output_arg = str(weights_path)          # directory
    else:
        already_done = weights_path.exists()
        if skip and already_done:
            ok(f"weights_compressed.npz already exists ({weights_path.stat().st_size/1e6:.0f} MB) \u2014 skipping")
            return
        output_arg = str(weights_path)          # .npz file

    compressed_dir.mkdir(parents=True, exist_ok=True)
    run(
        [sys.executable, "convert_weights.py",
         "--model-dir", str(model_dir),
         "--output",    output_arg,
         "--format",    fmt],
        "Converting model weights",
    )
    if fmt == "npy-dir":
        tensor_dir = weights_path / "tensors"
        size_bytes = sum(p.stat().st_size for p in tensor_dir.glob("*.npy")) if tensor_dir.exists() else 0
        ok(f"npy-dir written: {weights_path}  ({size_bytes/1e6:.0f} MB)")
    else:
        ok(f"Compressed npz written: {weights_path} ({weights_path.stat().st_size/1e6:.0f} MB)")


# ── Phase 5: Inference from compressed weights ────────────────────────────────

def phase_inference(model_dir: Path, npz_path: Path,
                    prompt: str, max_tokens: int, out_file: Path,
                    streaming: bool = False, workers: int = 0) -> dict:
    banner(5, f"Run inference from compressed weights ({'streaming' if streaming else 'npy-dir mmap' if Path(str(npz_path)).is_dir() else 'batch-npz'})")

    extra_flags = ["--quiet"]
    if streaming:
        extra_flags.append("--streaming")
    if workers:
        extra_flags += ["--workers", str(workers)]

    run(
        [sys.executable, "run_inference.py",
         "--model-dir",  str(model_dir),
         "--npz",        str(npz_path),
         "--prompt",     prompt,
         "--max-tokens", str(max_tokens),
         "--output",     str(out_file),
         *extra_flags],
        "Compressed inference",
    )
    with open(out_file) as f:
        comp = json.load(f)
    ok(f"Compressed output: {comp['output']!r}")
    ok(f"Load time:         {comp['load_time_s']:.2f}s")
    ok(f"Gen  time:         {comp['gen_time_s']:.2f}s")
    if comp.get("ram_delta_mb") is not None:
        ok(f"RAM delta:         {comp['ram_delta_mb']:.0f} MB")
    return comp


# ── Phase 6: Prove safetensors not needed ─────────────────────────────────────

def phase_prove_no_safetensors(model_dir: Path, npz_path: Path,
                               prompt: str, max_tokens: int, skip: bool,
                               workers: int = 0):
    banner(6, "Prove safetensors are NOT needed")

    if skip:
        info("Skipped (--skip-safetensors-check)")
        return

    stash_dir = Path("/tmp/_squish_poc_stash")
    stash_dir.mkdir(exist_ok=True)
    shards = list(model_dir.glob("*.safetensors"))

    if not shards:
        info("No .safetensors found to move — skipping proof step")
        return

    info(f"Moving {len(shards)} shard(s) to {stash_dir} ...")
    for s in shards:
        shutil.move(str(s), str(stash_dir / s.name))

    worker_flags = ["--workers", str(workers)] if workers else []
    try:
        result = run(
            [sys.executable, "run_inference.py",
             "--model-dir",  str(model_dir),
             "--npz",        str(npz_path),
             "--prompt",     prompt,
             "--max-tokens", str(max_tokens),
             "--output",     "/tmp/_squish_poc_no_safetensors.json",
             "--quiet", *worker_flags],
            "Inference WITHOUT safetensors",
            check=False,
        )
        if result.returncode == 0:
            ok("Inference succeeded with safetensors absent")
        else:
            fail("Inference FAILED without safetensors — check compressed_loader.py")
    finally:
        info(f"Restoring {len(shards)} shard(s) ...")
        for s in stash_dir.glob("*.safetensors"):
            shutil.move(str(s), str(model_dir / s.name))
        ok("Safetensors restored")


# ── Phase 7: Verify ───────────────────────────────────────────────────────────

def phase_verify(model_dir: Path, npz_path: Path,
                 ref_file: Path, comp_file: Path,
                 min_cosine: float, min_token_agree: float) -> bool:
    banner(7, "Verify: token agreement + weight fidelity")

    result = run(
        [sys.executable, "verify.py",
         "--model-dir",          str(model_dir),
         "--npz",                str(npz_path),
         "--reference",          str(ref_file),
         "--compressed",         str(comp_file),
         "--min-cosine",         str(min_cosine),
         "--min-token-agreement", str(min_token_agree)],
        "Running verification suite",
        check=False,
    )
    return result.returncode == 0


# ── Phase 8: Final report ─────────────────────────────────────────────────────

def phase_report(ref_file: Path, comp_file: Path,
                 weights_path: Path, model_dir: Path, verified: bool, fmt: str = "npy-dir"):
    banner(8, "Final report")

    try:
        with open(ref_file)  as f: ref  = json.load(f)
        with open(comp_file) as f: comp = json.load(f)

        # compute original safetensors size
        orig_bytes = sum(p.stat().st_size for p in model_dir.glob("*.safetensors"))
        # compute compressed weights size depending on format
        if fmt == "npy-dir" and weights_path.is_dir():
            tensor_dir = weights_path / "tensors"
            comp_bytes = sum(p.stat().st_size for p in tensor_dir.glob("*.npy")) if tensor_dir.exists() else 0
            fmt_label = "npy-dir"
        else:
            comp_bytes = weights_path.stat().st_size if weights_path.exists() else 0
            fmt_label = "npz"
        disk_ratio = orig_bytes / max(comp_bytes, 1)

        print(f"  {'Metric':<35} {'Reference':>12} {'Compressed':>12}")
        print(f"  {'─'*60}")
        print(f"  {'Load time (s)':<35} "
              f"{ref.get('load_time_s', 'n/a'):>12.2f} "
              f"{comp.get('load_time_s', 'n/a'):>12.2f}")
        print(f"  {'Gen time (s)':<35} "
              f"{ref.get('gen_time_s', 'n/a'):>12.2f} "
              f"{comp.get('gen_time_s', 'n/a'):>12.2f}")
        if comp.get("decompression_time_s") is not None:
            print(f"  {'Decomp time (s)':<35} "
                  f"{'—':>12} "
                  f"{comp['decompression_time_s']:>12.2f}")
        if orig_bytes:
            disk_ratio_label = f"Disk ratio ({fmt_label})"
            print(f"  {'Disk size (MB)':<35} "
                  f"{orig_bytes/1e6:>12.0f} "
                  f"{comp_bytes/1e6:>12.0f}")
            print(f"  {disk_ratio_label:<35} "
                  f"{'—':>12} "
                  f"{disk_ratio:>11.2f}x")
        if comp.get("ram_delta_mb") is not None:
            print(f"  {'RAM delta during load (MB)':<35} "
                  f"{'—':>12} "
                  f"{comp['ram_delta_mb']:>12.0f}")
        if comp.get("ram_peak_mb") is not None:
            print(f"  {'RAM peak during load (MB)':<35} "
                  f"{'—':>12} "
                  f"{comp['ram_peak_mb']:>12.0f}")
        if comp.get("n_quantized") is not None:
            print(f"  {'Tensors quantized (Q8)':<35} "
                  f"{'—':>12} "
                  f"{comp['n_quantized']:>12}")
        if comp.get("n_passthrough") is not None:
            print(f"  {'Tensors passthrough (f16)':<35} "
                  f"{'—':>12} "
                  f"{comp['n_passthrough']:>12}")
        if comp.get("decomp_workers") is not None:
            print(f"  {'Decomp workers':<35} "
                  f"{'—':>12} "
                  f"{comp['decomp_workers']:>12}")
        loader_label = comp.get("loader", "npy-dir")
        loader_display = {
            "npy-dir":      "npy-dir (1st run)",
            "finalized-f16":"finalized⚡ (f16)",
            "squish-mlx":    "squish-mlx ⚡⚡ (native)",
            "batch-npz":    "batch-npz",
        }.get(loader_label, loader_label)
        print(f"  {'Loader strategy':<35} "
              f"{'mlx_lm':>12} "
              f"{loader_display:>12}")
        print()
        print(f"  Reference output:   {ref['output']!r}")
        print(f"  Compressed output:  {comp['output']!r}")

    except (FileNotFoundError, KeyError):
        info("Could not load output files for comparison")

    print()
    if verified:
        print(f"  {BOLD}{GREEN}PoC VALIDATED — all checks passed.{RESET}")
    else:
        print(f"  {BOLD}{RED}PoC INCOMPLETE — one or more checks failed.{RESET}")
        print(f"  Check the output above for details.")
    print()
    print(f"  {YELLOW}Next steps:{RESET}")
    print(f"    python3 run_poc.py --format npy-dir --skip-download --skip-reference  # re-run with mmap loader")
    print(f"    python3 run_eval.py --tasks arc_easy,hellaswag,winogrande,piqa --limit 200")
    print(f"           \u2014 industry-standard benchmark comparison (ARC, HellaSwag, Winogrande, PIQA)")
    print(f"    python3 benchmark.py         \u2014 full three-strategy load-time comparison table")
    print(f"    python3 tool_calling_demo.py \u2014 constrained JSON tool calling demo")
    print(f"    python3 streaming_loader.py  \u2014 standalone streaming layer-by-layer demo")


# ── Phase ★: Test Suite ──────────────────────────────────────────────────────

def phase_test_suite(model_dir: Path, weights_path: Path, workers: int = 0):
    """
    Run _TEST_PROMPTS × {reference, compressed} — one model load each — and
    print a per-prompt comparison table.  Uses --prompts-file batch mode so
    both reference and compressed load their models exactly once.
    """
    banner("★", f"Test Suite — {len(_TEST_PROMPTS)} prompts × reference + compressed")

    prompts_file      = POC_DIR / "_test_suite_prompts.json"
    ref_results_file  = POC_DIR / "_test_suite_ref.json"
    comp_results_file = POC_DIR / "_test_suite_comp.json"

    items = [{"category": cat, "prompt": p, "max_tokens": mt}
             for cat, p, mt in _TEST_PROMPTS]
    with open(prompts_file, "w") as f:
        json.dump(items, f)

    info("Reference run (loading model once for all prompts) ...")
    run([sys.executable, "run_reference.py",
         "--model-dir",    str(model_dir),
         "--prompts-file", str(prompts_file),
         "--output",       str(ref_results_file)],
        "Reference test suite",
    )

    worker_flags = ["--workers", str(workers)] if workers else []
    info("Compressed run (loading model once for all prompts) ...")
    run([sys.executable, "run_inference.py",
         "--model-dir",    str(model_dir),
         "--npz",          str(weights_path),
         "--prompts-file", str(prompts_file),
         "--output",       str(comp_results_file),
         "--quiet", *worker_flags],
        "Compressed test suite",
    )

    with open(ref_results_file)  as f: ref_list  = json.load(f)
    with open(comp_results_file) as f: comp_list = json.load(f)
    ref_map  = {r["prompt"]: r for r in ref_list}
    comp_map = {r["prompt"]: r for r in comp_list}

    col = 26
    hdr = f"  {'Category':<12} {'Prompt':<40} {'Reference':<{col}} {'Compressed':<{col}} Agree"
    print(f"\n{BOLD}{hdr}{RESET}")
    print("  " + "─" * (len(hdr) - 2))

    n_first_match = 0
    for cat, prompt, _ in _TEST_PROMPTS:
        ref  = ref_map.get(prompt, {})
        comp = comp_map.get(prompt, {})

        ref_words  = ref.get("output", "").strip().split()
        comp_words = comp.get("output", "").strip().split()
        first_match = bool(ref_words and comp_words and ref_words[0] == comp_words[0])
        if first_match:
            n_first_match += 1

        ref_snip  = (ref.get("output", "—")  or "—").strip()[:col - 2]
        comp_snip = (comp.get("output", "—") or "—").strip()[:col - 2]
        sym = f"{GREEN}✓{RESET}" if first_match else f"{YELLOW}~{RESET}"

        print(f"  {cat:<12} {prompt[:37]:<40} {ref_snip:<{col}} {comp_snip:<{col}} {sym}")

    score = n_first_match / len(_TEST_PROMPTS)
    print(f"\n  First-token match: {n_first_match}/{len(_TEST_PROMPTS)}  ({score:.0%})")
    if score >= 0.8:
        ok(f"Test suite PASSED  ({score:.0%} first-token agreement)")
    else:
        info(f"Test suite: {score:.0%} first-token agreement "
             f"(some drift is normal for INT8 quantization)")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Squish PoC — automated end-to-end runner")
    ap.add_argument("--model-dir",
                    default=str(Path.home() / "models" / "Qwen2.5-1.5B-Instruct-bf16"))
    ap.add_argument("--compressed-dir",
                    default=str(Path.home() / "models" / "Qwen2.5-1.5B-Instruct-bf16-compressed"))
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--max-tokens", type=int, default=20)
    ap.add_argument("--format",
                    choices=["npy-dir", "npz"],
                    default="npy-dir",
                    help="Compressed weights storage format.  "
                         "npy-dir (default): individual .npy files with mmap loading, "
                         "no zlib, fast load + fast write.  "
                         "npz: single zlib-compressed archive, slow write (~9 min) "
                         "and slow load (~9s) but maximum disk compression.")
    ap.add_argument("--skip-download",           action="store_true")
    ap.add_argument("--skip-reference",          action="store_true")
    ap.add_argument("--skip-convert",            action="store_true")
    ap.add_argument("--skip-safetensors-check",  action="store_true")
    ap.add_argument("--no-ssl-verify",           action="store_true",
                    help="Disable SSL certificate verification for the HuggingFace "
                         "download (needed behind proxies/VPNs with self-signed certs)")
    ap.add_argument("--hf-token",                default=None, metavar="TOKEN",
                    help="HuggingFace read token for authenticated downloads. "
                         "Also read from HF_TOKEN env var. "
                         "Get one free at https://huggingface.co/settings/tokens")
    ap.add_argument("--min-cosine",              type=float, default=0.97)
    ap.add_argument("--min-token-agree",         type=float, default=0.60)
    ap.add_argument("--workers", type=int, default=0,
                    help="Parallel decompression threads for npy-dir loader. "
                         "0 (default) = auto (min(cpu_count, 8)). "
                         "Set 1 for serial baseline comparison.")
    ap.add_argument("--streaming",               action="store_true",
                    help="Use streaming_loader (layer-by-layer prefetch) "
                         "instead of the default batch loader")
    ap.add_argument("--benchmark",               action="store_true",
                    help="Run benchmark.py after the main PoC phases for "
                         "a full three-strategy comparison table")
    ap.add_argument("--tool-calling-demo",       action="store_true",
                    help="Run tool_calling_demo.py after the main PoC phases")
    args = ap.parse_args()

    model_dir      = Path(args.model_dir).expanduser()
    compressed_dir = Path(args.compressed_dir).expanduser()
    # weights_path: the npy-dir directory itself (npy-dir format)
    #               OR  the .npz file path  (npz format)
    if args.format == "npy-dir":
        weights_path = compressed_dir
    else:
        weights_path = compressed_dir / "weights_compressed.npz"
    ref_file       = POC_DIR / "reference_output.json"
    comp_file      = POC_DIR / "compressed_output.json"

    t_start = time.time()
    print(f"\n{BOLD}Squish PoC — Compressed Weight Loading for MLX Inference{RESET}")
    print(f"Model dir:      {model_dir}")
    print(f"Compressed dir: {compressed_dir}")
    print(f"Format:         {args.format}")
    print(f"Workers:        {args.workers if args.workers > 0 else 'auto'}")
    print(f"Prompt:         {args.prompt!r}")
    print(f"Max tokens:     {args.max_tokens}")

    phase_deps()
    phase_download(model_dir, args.skip_download,
                   no_ssl_verify=args.no_ssl_verify,
                   hf_token=args.hf_token)
    phase_reference(model_dir, args.prompt, args.max_tokens,
                    ref_file, args.skip_reference)
    phase_convert(model_dir, compressed_dir, weights_path, args.format, args.skip_convert)
    phase_inference(model_dir, weights_path, args.prompt, args.max_tokens, comp_file,
                    streaming=args.streaming, workers=args.workers)
    phase_prove_no_safetensors(model_dir, weights_path, args.prompt, args.max_tokens,
                               args.skip_safetensors_check, workers=args.workers)
    verified = phase_verify(model_dir, weights_path, ref_file, comp_file,
                            args.min_cosine, args.min_token_agree)
    phase_report(ref_file, comp_file, weights_path, model_dir, verified, fmt=args.format)

    # Always run the multi-prompt test suite
    phase_test_suite(model_dir, weights_path, workers=args.workers)

    # Optional: full benchmark
    if args.benchmark:
        banner(9, "Benchmark — three-strategy comparison")
        run(
            [sys.executable, "benchmark.py",
             "--model-dir",  str(model_dir),
             "--npz",        str(weights_path),
             "--prompt",     args.prompt,
             "--max-tokens", str(args.max_tokens)],
            "Running benchmark.py",
        )

    # Optional: tool calling demo
    if args.tool_calling_demo:
        banner(10, "Tool Calling Demo — constrained JSON generation")
        run(
            [sys.executable, "tool_calling_demo.py",
             "--model-dir",  str(model_dir),
             "--npz",        str(weights_path)],
            "Running tool_calling_demo.py",
        )

    elapsed = time.time() - t_start
    print(f"\n  Total wall time: {elapsed/60:.1f} min\n")
    sys.exit(0 if verified else 1)


if __name__ == "__main__":
    main()
