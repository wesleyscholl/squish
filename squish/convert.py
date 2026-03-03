#!/usr/bin/env python3
"""
convert_weights.py

Convert a float16/bfloat16 safetensors model to Vectro INT8 compressed format.
Stores everything in a single .npz archive with a companion _manifest.json.

Storage layout inside the npz:
  {safe_key}__q      → int8 array     quantized weights (shape: n_rows × n_cols)
  {safe_key}__s      → float32 array  per-row scale factors (shape: n_rows)
  {safe_key}__shape  → int64 array    original tensor shape (for reshape on load)
  {safe_key}__pt     → float32 array  passthrough tensors stored unquantized

safe_key = tensor name with '.' replaced by '__'
Companion file: {output}_manifest.json  maps original_name -> safe_key

Usage:
    python3 convert_weights.py \\
        --model-dir ~/models/Qwen2.5-1.5B-Instruct \\
        --output ~/models/Qwen2.5-1.5B-Instruct-compressed/weights_compressed.npz \\
        [--passthrough embed_tokens lm_head] \\
        [--outlier-threshold 20.0] \\
        [--verbose]
"""
import sys
import json
import time
import threading
import argparse
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# AWQ scale application (imported lazily so convert.py works without awq.py)
# ---------------------------------------------------------------------------
def _apply_awq_single(name: str, arr_f32: np.ndarray, awq_scales: dict) -> np.ndarray:
    """
    Apply AWQ scale to a single weight tensor in-place (returns modified copy).
    Wraps squish.awq.apply_awq_to_weights for single-tensor use.
    """
    if not awq_scales:
        return arr_f32
    try:
        from squish.awq import apply_awq_to_weights
        tmp = {name: arr_f32}
        apply_awq_to_weights(tmp, awq_scales, verbose=False)
        return tmp[name]
    except ImportError:
        return arr_f32


from squish.quantizer import quantize_embeddings, quantize_int4, QuantizationResult


# ---------------------------------------------------------------------------
# ─── TTY-safe line-clear helper ─────────────────────────────────────────────
def _clear_line() -> None:
    """Overwrite the current terminal line.  No-op when stdout is not a TTY."""
    if sys.stdout.isatty():
        sys.stdout.write("\r" + " " * 80 + "\r")
        sys.stdout.flush()


# Spinner
# ---------------------------------------------------------------------------

class Spinner:
    """
    Background-thread snake spinner.
    Usage:
        with Spinner("Writing weights_compressed.npz"):
            slow_operation()
    Also supports manual updates:
        sp = Spinner("Quantizing")
        sp.update("layer 5/28")
        sp.stop()
    When stdout is not a TTY (e.g. piped to tee), all \r-based overwriting is
    suppressed to avoid rendering artifacts in VS Code / other terminals.
    """
    _FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

    def __init__(self, label: str = "", interval: float = 0.1):
        self._label    = label
        self._interval = interval
        self._suffix   = ""
        self._stop_evt = threading.Event()
        self._is_tty   = sys.stdout.isatty()
        self._thread   = threading.Thread(target=self._spin, daemon=True)

    def _spin(self):
        if not self._is_tty:
            # Non-TTY: just block until stopped, no output from the spin thread
            self._stop_evt.wait()
            return
        i = 0
        while not self._stop_evt.is_set():
            frame = self._FRAMES[i % len(self._FRAMES)]
            line  = f"\r  {frame}  {self._label}  {self._suffix}"
            sys.stdout.write(line)
            sys.stdout.flush()
            self._stop_evt.wait(self._interval)
            i += 1
        # clear the spinner line
        sys.stdout.write("\r" + " " * (len(self._label) + len(self._suffix) + 10) + "\r")
        sys.stdout.flush()

    def update(self, suffix: str):
        self._suffix = suffix

    def start(self):
        self._thread.start()
        return self

    def stop(self, final_msg: str = ""):
        self._stop_evt.set()
        self._thread.join()
        if final_msg:
            print(f"  ✓  {final_msg}")

    def __enter__(self):
        return self.start()

    def __exit__(self, *_):
        self.stop()


def safe_key(tensor_name: str) -> str:
    """Convert dot-notation tensor name to a valid npz key (no dots)."""
    return tensor_name.replace(".", "__")


def has_outliers(arr_f32: np.ndarray, threshold: float) -> bool:
    """Return True if the tensor has strong outlier rows (per-row max/mean > threshold)."""
    row_max = np.max(np.abs(arr_f32), axis=-1)
    row_mean = np.mean(np.abs(arr_f32), axis=-1) + 1e-8
    ratio = row_max / row_mean
    return float(ratio.max()) > threshold


def quantize_tensor(
    name: str,
    arr_f32: np.ndarray,
    outlier_threshold: float,
    passthrough_patterns: list[str],
    use_int4: bool = False,
) -> dict:
    """
    Quantize a single float32 tensor.

    Returns a dict of file suffixes → numpy arrays:
      INT8 (default):  __q, __s, __shape
      INT4 (use_int4): __q4, __s4, __shape
      passthrough:     __pt, __shape
    """
    original_shape = arr_f32.shape

    # --- decide whether to pass through ---
    skip = any(p in name for p in passthrough_patterns)
    if not skip and arr_f32.ndim >= 2:
        flat = arr_f32.reshape(-1, arr_f32.shape[-1])
        skip = has_outliers(flat, outlier_threshold)
        if skip:
            print(f"    [outlier passthrough] {name}")

    shape_arr = np.array(original_shape, dtype=np.int64)

    if skip:
        return {
            "__pt": arr_f32,
            "__shape": shape_arr,
        }

    # --- reshape to 2D for per-row quantization ---
    if arr_f32.ndim == 0:
        flat = arr_f32.reshape(1, 1)
    elif arr_f32.ndim == 1:
        flat = arr_f32.reshape(1, -1)
    else:
        flat = arr_f32.reshape(-1, arr_f32.shape[-1])

    result: QuantizationResult = quantize_embeddings(flat, group_size=64)
    if use_int4:
        # INT4 nibble-packed: ~50 % disk vs INT8, requires squish_quant Rust ext.
        # Reconstruct from INT8 first, then pack to INT4.
        from squish.quantizer import reconstruct_embeddings
        reconstructed = reconstruct_embeddings(result)
        packed, scales4 = quantize_int4(reconstructed, group_size=64)
        return {
            "__q4":    packed,   # uint8 nibble-packed  (n, d//2)
            "__s4":    scales4,  # float32              (n, d//64)
            "__shape": shape_arr,
        }
    return {
        "__q": result.quantized,   # int8  (grouped-64 per default)
        "__s": result.scales,      # float32 (n_rows, n_groups) or (n_rows,)
        "__shape": shape_arr,
    }


def load_mlx_weights_shard(shard_path: Path) -> dict:
    """
    Load a single safetensors shard as float32 numpy arrays.
    Uses safetensors.numpy directly (CPU only — no Metal, no MLX).
    Avoids the Metal GPU timeout that occurs when loading 7B+ models.
    """
    try:
        from safetensors.numpy import load_file as st_load_numpy
        raw = st_load_numpy(str(shard_path))
        # safetensors.numpy returns {name: np.ndarray} — may be float16 or bfloat16
        out = {}
        for name, arr in raw.items():
            # Convert to float32 for Vectro quantization
            out[name] = arr.astype(np.float32)
        return out
    except Exception:
        # Fallback: use mlx but only for this one shard, never the whole model
        import mlx.core as mx
        shard_weights = mx.load(str(shard_path))
        return {
            name: np.array(arr.astype(mx.float32))
            for name, arr in shard_weights.items()
        }


def load_mlx_weights(model_dir: Path) -> dict:
    """
    Load all weights from safetensors shards as float32 numpy arrays.

    ⚠ NOTE: This loads the ENTIRE model into RAM as float32.  For models >3B this
    can easily exceed available memory.  Use process_weights_streaming() instead,
    which processes one shard at a time and writes output as it goes.

    Kept for backward-compatibility with existing callers.
    """
    shard_files = sorted(model_dir.glob("*.safetensors"))
    if not shard_files:
        raise FileNotFoundError(f"No .safetensors files in {model_dir}")

    weights = {}
    for shard in shard_files:
        print(f"  Loading {shard.name} ...")
        weights.update(load_mlx_weights_shard(shard))
    return weights


def process_weights_streaming(
    model_dir: Path,
    output_path: Path,
    passthrough_patterns: list[str],
    outlier_threshold: float,
    verbose: bool,
    awq_scales: dict | None = None,
    use_int4: bool = False,
) -> dict:
    """
    Streaming shard-by-shard compression — works for any model size.

    Processes one .safetensors shard at a time:
      1. Load shard (CPU numpy, no MLX/Metal)
      2. Quantize each tensor
      3. Write .npy files immediately
      4. Free the shard from RAM

    Peak RAM ≈ 2× the size of one shard (typically 2-5 GB for 7B models,
    ~2 GB for sharded 7B), regardless of total model size.
    """
    shard_files = sorted(model_dir.glob("*.safetensors"))
    if not shard_files:
        raise FileNotFoundError(f"No .safetensors files in {model_dir}")

    tensor_dir = output_path / "tensors"
    tensor_dir.mkdir(parents=True, exist_ok=True)

    manifest  = {}   # original_name → safe_key
    stats     = {"n_quantized": 0, "n_passthrough": 0,
                 "orig_f32_bytes": 0, "compressed_bytes": 0}
    total_tensors = 0

    print(f"\n  Processing {len(shard_files)} shard(s) …  (streaming — peak RAM ≈ 1 shard)")

    for shard_idx, shard in enumerate(shard_files, 1):
        print(f"\n  [{shard_idx}/{len(shard_files)}] {shard.name}")
        shard_weights = load_mlx_weights_shard(shard)
        shard_tensors = len(shard_weights)

        sp = Spinner(f"Shard {shard_idx}/{len(shard_files)}  ({shard_tensors} tensors)").start()
        for tensor_idx, (name, arr_f32) in enumerate(shard_weights.items(), 1):
            sp.update(f"{tensor_idx}/{shard_tensors}  {name}")
            sk = safe_key(name)
            manifest[name] = sk

            # ── Phase 1.2: Apply AWQ scales before quantization ───────────
            if awq_scales:
                arr_f32 = _apply_awq_single(name, arr_f32, awq_scales)

            sub = quantize_tensor(name, arr_f32, outlier_threshold, passthrough_patterns, use_int4=use_int4)

            # Write immediately — don't accumulate in RAM
            for suffix, data in sub.items():
                out_arr = data.astype(np.float16) if suffix == "__pt" else data
                np.save(str(tensor_dir / f"{sk}{suffix}.npy"), out_arr)

            orig_bytes = arr_f32.nbytes
            comp_bytes = sum(
                (tensor_dir / f"{sk}{sfx}.npy").stat().st_size
                for sfx in sub
                if not sfx.endswith("__shape")
            )
            stats["orig_f32_bytes"]   += orig_bytes
            stats["compressed_bytes"] += comp_bytes

            if "__pt" in sub:
                stats["n_passthrough"] += 1
            else:
                stats["n_quantized"] += 1

            if verbose:
                ratio = orig_bytes / max(comp_bytes, 1)
                mode  = "PT" if "__pt" in sub else ("Q4" if use_int4 else "Q8")
                _clear_line()
                print(f"  [{mode}] {name}: {arr_f32.shape} ratio={ratio:.2f}x")

            total_tensors += 1
            del arr_f32

        sp.stop(f"Shard {shard_idx} done  ({shard_tensors} tensors written)")
        del shard_weights

    # Write manifest
    with open(output_path / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    # Sentinel for consistency with old npy-dir reader
    (tensor_dir / ".manifest_ready").touch()

    print(f"\n  Total tensors: {total_tensors}")
    return stats


def write_npy_dir(output_dir: Path, npz_dict: dict, manifest: dict) -> int:
    """
    Write tensors as individual uncompressed .npy files for memory-mapped loading.

    Layout::
        {output_dir}/
            manifest.json          # original_name → safe_key
            tensors/
                {safe_key}__q.npy     # int8 quantized weights
                {safe_key}__s.npy     # float32 per-row scales
                {safe_key}__shape.npy # int64 original shape
                {safe_key}__pt.npy    # float16 passthrough weights

    Passthrough tensors are stored as float16:
      - Original model was bfloat16.  bf16 → f32 is lossless; f32 → f16 has MORE
        mantissa bits than bf16 (10 vs 7), so all precision from the source model
        is preserved.  Saves ~50%% disk vs float32.
    No zlib compression: the OS can mmap individual .npy files for near-zero
    decompression overhead when loading.

    Returns total bytes written.
    """
    tensor_dir = output_dir / "tensors"
    tensor_dir.mkdir(parents=True, exist_ok=True)
    total_bytes = 0
    for key, arr in npz_dict.items():
        out_arr = arr.astype(np.float16) if key.endswith("__pt") else arr
        path = tensor_dir / f"{key}.npy"
        np.save(str(path), out_arr)
        total_bytes += path.stat().st_size
    with open(output_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    return total_bytes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument(
        "--format",
        choices=["npz", "npy-dir"],
        default="npy-dir",
        help="Storage format: npy-dir (default, fast mmap loading) or npz (zlib-compressed)",
    )
    ap.add_argument(
        "--passthrough",
        nargs="*",
        default=[],
        metavar="PATTERN",
        help="Tensor name substrings to store as float32 without quantizing",
    )
    ap.add_argument(
        "--outlier-threshold",
        type=float,
        default=20.0,
        help="Auto-passthrough if row max/mean ratio exceeds this (default: 20)",
    )
    ap.add_argument(
        "--awq-scales",
        metavar="DIR",
        default=None,
        help="Directory of .awq.npy scale files produced by 'python3 -m squish.awq'. "
             "When provided, AWQ scales are applied to each weight tensor before "
             "quantization for improved INT8 accuracy.",
    )
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument(
        "--int4",
        action="store_true",
        default=False,
        help="Use INT4 nibble-packed quantization instead of INT8.  Halves disk usage "
             "(~1.5 GB for 1.5B vs ~2.9 GB INT8) at ≤2%% accuracy delta.  "
             "Requires squish_quant Rust extension (built with maturin).  "
             "Recommended for 1.5B models where every GB matters.",
    )
    args = ap.parse_args()

    model_dir = Path(args.model_dir)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # For npy-dir, output_path IS the directory; manifest lives inside it.
    # For npz, manifest is a sibling file.
    if args.format == "npy-dir":
        manifest_path = str(output_path / "manifest.json")
    else:
        manifest_path = str(output_path).replace(".npz", "_manifest.json")

    if args.format == "npy-dir":
        # ── Streaming path (7B+): load one shard → quantize → write → free ─────
        # Keeps peak RAM to ~1 shard (~2 GB for 7B) instead of loading the full
        # model into GPU memory all at once (which triggers Metal GPU timeout on
        # 16 GB unified-memory machines like M3 MacBook Pro).
        print(f"\nStreaming quantization → {output_path}/tensors/")
        print(f"  (CPU-only shard loading — no Metal GPU, works for any model size)")

        # ── Load AWQ scales if provided ────────────────────────────────────
        awq_scales: dict = {}
        if args.awq_scales:
            try:
                from squish.awq import load_awq_scales
                awq_scales = load_awq_scales(args.awq_scales)
                n_awq = len(awq_scales)
                print(f"  [AWQ] Loaded {n_awq} layer scales from {args.awq_scales}")
            except Exception as e:
                print(f"  [AWQ] Warning: could not load scales: {e}  (continuing without AWQ)")

        t0 = time.time()
        stats = process_weights_streaming(
            model_dir,
            output_path,
            args.passthrough,
            args.outlier_threshold,
            args.verbose,
            awq_scales=awq_scales,
            use_int4=args.int4,
        )
        elapsed = time.time() - t0

        tensor_dir = output_path / "tensors"
        disk_bytes = sum(p.stat().st_size for p in tensor_dir.glob("*.npy"))
        disk_mb = disk_bytes / 1e6
        orig_gb = stats["orig_f32_bytes"] / 1e9
        comp_gb = stats["compressed_bytes"] / 1e9
        ratio = stats["orig_f32_bytes"] / max(stats["compressed_bytes"], 1)
        disk_ratio = stats["orig_f32_bytes"] / max(disk_bytes, 1)
        n_total = stats["n_quantized"] + stats["n_passthrough"]

        print(f"\n{'='*50}")
        print(f"  Format:           npy-dir (streaming)")
        print(f"  Quantization:     {'INT4 nibble-packed (group-64)' if args.int4 else 'INT8 per-group-64'}")
        print(f"  Tensors:          {n_total} total")
        print(f"    Quantized (Q8): {stats['n_quantized']}")
        print(f"    Passthrough (f16 on disk): {stats['n_passthrough']}")
        print(f"  Original (f32):   {orig_gb:.3f} GB")
        print(f"  Quantized raw:    {comp_gb:.3f} GB  ({ratio:.2f}x ratio)")
        print(f"  On-disk (npy-dir): {disk_mb:.1f} MB  ({disk_ratio:.2f}x ratio)")
        print(f"  Total time:       {elapsed:.1f}s")
        print(f"  Manifest:         {output_path / 'manifest.json'}")
        print(f"{'='*50}")

    else:
        # ── Legacy batch path (NPZ / small models) ────────────────────────────
        print(f"Loading weights from {model_dir} ...")
        weights = load_mlx_weights(model_dir)
        print(f"  {len(weights)} tensors loaded")

        print(f"\nQuantizing {len(weights)} tensors ...")
        npz_dict = {}
        manifest = {}  # original_name -> safe_key
        stats = {
            "n_quantized": 0,
            "n_passthrough": 0,
            "orig_f32_bytes": 0,
            "compressed_bytes": 0,
        }

        t0 = time.time()
        total = len(weights)
        sp = Spinner("Quantizing").start()
        for idx, (name, arr_f32) in enumerate(weights.items(), 1):
            sp.update(f"{idx}/{total}  {name}")
            sk = safe_key(name)
            manifest[name] = sk

            sub = quantize_tensor(name, arr_f32, args.outlier_threshold, args.passthrough)

            for suffix, data in sub.items():
                npz_dict[sk + suffix] = data

            orig_bytes = arr_f32.nbytes
            comp_bytes = sum(d.nbytes for k, d in sub.items() if not k.endswith("__shape"))
            stats["orig_f32_bytes"] += orig_bytes
            stats["compressed_bytes"] += comp_bytes

            if "__pt" in sub:
                stats["n_passthrough"] += 1
            else:
                stats["n_quantized"] += 1

            if args.verbose:
                ratio = orig_bytes / max(comp_bytes, 1)
                mode = "PT" if "__pt" in sub else "Q8"
                _clear_line()
                print(f"  [{mode}] {name}: {arr_f32.shape} ratio={ratio:.2f}x")

        sp.stop(f"Quantization done  ({stats['n_quantized']}Q + {stats['n_passthrough']}PT)")

        t1 = time.time()
        print(f"\nWriting {output_path} ...")
        print(f"  (this is the slow step — zlib single-threaded compression)")
        with Spinner(f"savez_compressed  →  {output_path.name}"):
            np.savez_compressed(str(output_path), **npz_dict)
        write_time = time.time() - t1
        print(f"  Written in {write_time:.1f}s")
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

        elapsed = time.time() - t0
        orig_gb = stats["orig_f32_bytes"] / 1e9
        comp_gb = stats["compressed_bytes"] / 1e9
        ratio = stats["orig_f32_bytes"] / max(stats["compressed_bytes"], 1)
        disk_bytes = output_path.stat().st_size
        disk_mb = disk_bytes / 1e6
        disk_ratio = stats["orig_f32_bytes"] / max(disk_bytes, 1)

        print(f"\n{'='*50}")
        print(f"  Format:           npz")
        print(f"  Tensors:          {len(weights)} total")
        print(f"    Quantized (Q8): {stats['n_quantized']}")
        print(f"    Passthrough (f16 on disk): {stats['n_passthrough']}")
        print(f"  Original (f32):   {orig_gb:.3f} GB")
        print(f"  Quantized raw:    {comp_gb:.3f} GB  ({ratio:.2f}x ratio)")
        print(f"  On-disk (zlib):   {disk_mb:.1f} MB  ({disk_ratio:.2f}x ratio)")
        print(f"  Write time:       {write_time:.1f}s")
        print(f"  Total time:       {elapsed:.1f}s")
        print(f"  Manifest:         {manifest_path}")
        print(f"{'='*50}")


if __name__ == "__main__":
    main()
