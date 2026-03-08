"""
squish/life_model.py

LIFE (Latency Inference Framework Estimator) — an analytical performance model
for local LLM inference on Apple Silicon.

The model decomposes inference latency into three analytically tractable terms:

  TTFT (Time-to-First-Token, prefill latency):
    TTFT = Compute_prefill / (flops_per_second × efficiency)
          + Memory_reads / memory_bandwidth

  TPOT (Time-per-Output-Token, decode latency):
    TPOT ≈ Model_memory_bytes / memory_bandwidth   (memory-bandwidth-bound)

  Throughput:
    tok/s = 1 / TPOT   (single stream)
    tok/s = batch_size / max(TTFT/tokens, TPOT)   (batched)

Key insight for Apple Silicon unified memory
--------------------------------------------
Decode on M-series chips is almost always *memory-bandwidth-bound*, not
compute-bound, because:
  - Each decode step reads ALL model weights (7B × 2 bytes ≈ 14 GB/s at 1B/s BW)
  - Compute is minimal: batch_size × 1 token × ffn_dim × 2 flops per weight

For a batch_size > ``compute_crossover_batch``, the regime flips to
compute-bound and TPOT stops improving with larger batch.

Usage
-----
    from squish.life_model import predict

    result = predict(
        model_dir  = "~/squish/models/Qwen3-8B-mlx-int4",
        batch_size = 1,
        seq_len    = 512,
        output_len = 128,
    )
    print(result)
    # {
    #   "ttft_ms": 42.5,
    #   "tpot_ms": 12.3,
    #   "tokens_per_sec": 81.3,
    #   "kv_memory_gb": 0.21,
    #   "bottleneck": "memory-bandwidth",
    #   "model_params_b": 8.0,
    #   "effective_bw_gb_s": 77.0,
    #   "hardware": "Apple M3 Pro",
    # }

    # Or from CLI:
    squish predict qwen3:8b --seq-len 512 --output-len 128
"""
from __future__ import annotations

import json
import os
import subprocess
import platform
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Hardware profiling helpers
# ---------------------------------------------------------------------------

# Approximate unified-memory bandwidth by chip family (GB/s)
# Source: Apple Silicon specs + mlx-benchmark measurements
_CHIP_BANDWIDTH_GB_S: dict[str, float] = {
    "apple m1":        68.0,
    "apple m1 pro":    200.0,
    "apple m1 max":    400.0,
    "apple m1 ultra":  800.0,
    "apple m2":        100.0,
    "apple m2 pro":    200.0,
    "apple m2 max":    400.0,
    "apple m2 ultra":  800.0,
    "apple m3":        100.0,
    "apple m3 pro":    150.0,
    "apple m3 max":    300.0,
    "apple m3 ultra":  600.0,
    "apple m4":        120.0,
    "apple m4 pro":    273.0,
    "apple m4 max":    546.0,
}

# Approximate ANE + GPU TFLOPS per chip (float16 peak)
_CHIP_TFLOPS: dict[str, float] = {
    "apple m1":        11.0,
    "apple m1 pro":    16.0,
    "apple m1 max":    21.0,
    "apple m1 ultra":  42.0,
    "apple m2":        15.8,
    "apple m2 pro":    19.0,
    "apple m2 max":    27.2,
    "apple m2 ultra":  54.4,
    "apple m3":        18.0,
    "apple m3 pro":    18.0,
    "apple m3 max":    42.0,
    "apple m3 ultra":  84.0,
    "apple m4":        38.0,
    "apple m4 pro":    62.0,
    "apple m4 max":   124.0,
}

_DEFAULT_BW_GB_S:  float = 100.0
_DEFAULT_TFLOPS:   float = 15.0
_EFFICIENCY:       float = 0.70   # empirical Metal + mlx overhead factor


def _detect_chip() -> tuple[str, float, float]:
    """
    Return ``(chip_name, bw_gb_s, tflops)``.

    Uses ``sysctl hw.model`` on macOS; falls back to defaults on other
    platforms or when the chip is not in the lookup table.
    """
    if platform.system() != "Darwin":
        return ("unknown", _DEFAULT_BW_GB_S, _DEFAULT_TFLOPS)

    try:
        raw = subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            stderr=subprocess.DEVNULL, text=True,
        ).strip().lower()
    except Exception:
        raw = ""

    if not raw:
        try:
            raw = subprocess.check_output(
                ["sysctl", "-n", "hw.model"], stderr=subprocess.DEVNULL, text=True,
            ).strip().lower()
        except Exception:
            raw = ""

    for key, bw in _CHIP_BANDWIDTH_GB_S.items():
        if key in raw:
            return (key.title(), bw, _CHIP_TFLOPS.get(key, _DEFAULT_TFLOPS))

    return ("Apple Silicon (unknown variant)", _DEFAULT_BW_GB_S, _DEFAULT_TFLOPS)


# ---------------------------------------------------------------------------
# Model parameter counting
# ---------------------------------------------------------------------------

def _count_model_params(model_dir: Optional[str]) -> float:
    """
    Return the model's total parameter count in billions.

    Reads ``config.json`` from *model_dir* (``num_parameters`` or
    computed from hidden sizes).  Falls back to the directory name heuristic.
    """
    if model_dir is None:
        return 7.0  # reasonable default

    config_path = Path(model_dir).expanduser() / "config.json"
    if config_path.exists():
        try:
            cfg = json.loads(config_path.read_text())
            # Direct count
            if "num_parameters" in cfg:
                return cfg["num_parameters"] / 1e9
            # Compute from architecture
            h   = cfg.get("hidden_size", 4096)
            ffn = cfg.get("intermediate_size", h * 4)
            n   = cfg.get("num_hidden_layers", 32)
            v   = cfg.get("vocab_size", 32000)
            # Rough: 12 × h² × n (attention + FFN) + vocab embedding
            params = (12 * h * h * n + 3 * ffn * h * n + v * h)
            return params / 1e9
        except Exception:
            pass

    # Fallback: guess from directory name
    name = Path(model_dir).name.lower() if model_dir else ""
    for size, b in (("72b", 72.0), ("32b", 32.0), ("14b", 14.0),
                    ("8b", 8.0), ("7b", 7.0), ("3b", 3.0),
                    ("1.5b", 1.5), ("0.5b", 0.5)):
        if size in name:
            return b
    return 7.0


def _quant_bytes_per_param(model_dir: Optional[str]) -> float:
    """
    Return effective bytes-per-parameter based on the quantization detected
    in *model_dir*.  Uses the ``quantization`` key in ``config.json`` or
    directory name heuristics.

    - fp16  / bf16 : 2.0
    - int8         : 1.0
    - int4  / q4   : 0.5  (typical 4-bit with group scales)
    - int2         : 0.25
    """
    if model_dir is None:
        return 2.0

    config_path = Path(model_dir).expanduser() / "config.json"
    if config_path.exists():
        try:
            cfg  = json.loads(config_path.read_text())
            bits = (cfg.get("quantization", {}).get("bits", None)
                    or cfg.get("num_bits", None))
            if bits is not None:
                return int(bits) / 8.0
        except Exception:
            pass

    name = Path(model_dir).name.lower() if model_dir else ""
    if any(x in name for x in ("int4", "q4", "4bit", "mlx-int4")):
        return 0.5
    if any(x in name for x in ("int8", "q8", "8bit", "mlx-int8")):
        return 1.0
    return 2.0


# ---------------------------------------------------------------------------
# KV cache memory estimation
# ---------------------------------------------------------------------------

def _kv_memory_gb(
    model_dir:  Optional[str],
    seq_len:    int,
    output_len: int,
    batch_size: int,
) -> float:
    """
    Estimate KV cache memory in GB for one generation request.

    KV size = 2 × n_layers × n_kv_heads × (seq_len + output_len) × head_dim × 2 bytes (fp16)
    """
    cfg: dict = {}
    if model_dir:
        config_path = Path(model_dir).expanduser() / "config.json"
        if config_path.exists():
            try:
                cfg = json.loads(config_path.read_text())
            except Exception:
                pass

    n_layers  = cfg.get("num_hidden_layers", 32)
    n_kv_h    = cfg.get("num_key_value_heads", cfg.get("num_attention_heads", 32))
    head_dim  = cfg.get("head_dim",
                        cfg.get("hidden_size", 4096) // cfg.get("num_attention_heads", 32))
    total_tok = (seq_len + output_len) * batch_size
    bytes_kv  = 2 * n_layers * n_kv_h * total_tok * head_dim * 2   # 2 = K+V, 2 = fp16
    return bytes_kv / (1024 ** 3)


# ---------------------------------------------------------------------------
# Main prediction function
# ---------------------------------------------------------------------------

def predict(
    model_dir:  Optional[str] = None,
    batch_size: int   = 1,
    seq_len:    int   = 512,
    output_len: int   = 128,
) -> dict:
    """
    Compute analytical TTFT / TPOT / throughput estimates using the LIFE model.

    Parameters
    ----------
    model_dir  : path to the model directory (or None to use defaults)
    batch_size : number of concurrent requests
    seq_len    : number of input (prefill) tokens
    output_len : expected number of output (decode) tokens

    Returns
    -------
    dict with keys:
        ttft_ms          : float — predicted time-to-first-token in ms
        tpot_ms          : float — predicted time-per-output-token in ms
        tokens_per_sec   : float — single-stream decode throughput
        kv_memory_gb     : float — KV cache memory estimate in GB
        model_memory_gb  : float — model weights memory in GB
        bottleneck       : str   — "memory-bandwidth" | "compute"
        model_params_b   : float — estimated parameter count in billions
        effective_bw_gb_s: float — effective memory bandwidth used
        hardware         : str   — detected chip name
    """
    chip_name, bw_gb_s, tflops = _detect_chip()
    eff_bw   = bw_gb_s * _EFFICIENCY    # effective bandwidth after overhead
    eff_tops = tflops  * _EFFICIENCY * 1e12   # effective FLOPS

    params_b         = _count_model_params(model_dir)
    bytes_per_param  = _quant_bytes_per_param(model_dir)
    model_bytes      = params_b * 1e9 * bytes_per_param
    model_gb         = model_bytes / (1024 ** 3)

    # ── TTFT: prefill latency ─────────────────────────────────────────────────
    # Approximate FLOPs for prefill: 2 × params × seq_len (matmul dominates)
    prefill_flops = 2.0 * params_b * 1e9 * seq_len * batch_size
    ttft_compute  = prefill_flops / eff_tops           # seconds, compute-bound
    # Memory reads during prefill: read all model weights once
    ttft_memory   = model_bytes / (eff_bw * 1e9)       # seconds, BW-bound
    ttft_s        = max(ttft_compute, ttft_memory)

    # ── TPOT: decode latency ─────────────────────────────────────────────────
    # Decode reads all model weights once per token per batch element.
    # Memory-bandwidth bound when batch_size < compute_crossover_batch.
    decode_flops_per_tok  = 2.0 * params_b * 1e9 * batch_size
    tpot_compute  = decode_flops_per_tok / eff_tops    # seconds, compute-bound
    tpot_memory   = model_bytes / (eff_bw * 1e9)       # seconds, BW-bound
    tpot_s        = max(tpot_compute, tpot_memory)

    bottleneck = "memory-bandwidth" if tpot_memory >= tpot_compute else "compute"

    # ── Throughput ─────────────────────────────────────────────────────────────
    tokens_per_sec = batch_size / tpot_s if tpot_s > 0 else 0.0

    # ── KV cache ─────────────────────────────────────────────────────────────
    kv_gb = _kv_memory_gb(model_dir, seq_len, output_len, batch_size)

    return {
        "ttft_ms":           round(ttft_s * 1000, 2),
        "tpot_ms":           round(tpot_s * 1000, 3),
        "tokens_per_sec":    round(tokens_per_sec, 1),
        "kv_memory_gb":      round(kv_gb, 3),
        "model_memory_gb":   round(model_gb, 2),
        "bottleneck":        bottleneck,
        "model_params_b":    round(params_b, 2),
        "effective_bw_gb_s": round(eff_bw, 1),
        "hardware":          chip_name,
    }
