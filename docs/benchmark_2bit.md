# Squish — 2-bit Quantization Comparison Benchmark

> Phase 9C — sub-2-bit weight compression: INT4 vs VPTQ vs AQLM vs QuIP#
> Weight-reconstruction metrics from synthetic 64×64 weight matrices.
> Perplexity + TPS metrics require `--model-dir` (run on real hardware).

---

## Overview

This benchmark evaluates four near-2-bit weight-compression methods on the
same weight matrix and reports:

- **BPW** — bits per weight after compression (including index + scale overhead)
- **SNR (dB)** — weight-reconstruction signal-to-noise ratio vs. original
- **Compress ms** — wall-clock time for the offline compression step
- **Decompress ms** — wall-clock time to reconstruct weights from compressed form
- **Perplexity** — wikitext-2 perplexity (requires `--model-dir`, see below)
- **TPS** — tokens/second during generation (requires `--model-dir`)

---

## Stage 1 — Weight Reconstruction (Synthetic 64×64 matrix)

> **Note:** The synthetic weight matrix (64×64 = 4096 parameters at σ=0.02)
> matches the scale of a small linear layer.  BPW results are representative
> of the compression scheme; SNR is a proxy for perplexity — higher is better.
> Compress time for VPTQ reflects its k-means++ calibration cost (reduced
> config for benchmark speed; production use targets k=256, group=8).

| Method | BPW | SNR (dB) | Compress (ms) | Decompress (ms) | Backend |
|--------|----:|---------:|--------------:|----------------:|---------|
| INT4 nibble (baseline) | 5.00 | 21.0 | 0.8 | 0.14 | numpy |
| VPTQ (NeurIPS 2025) | 1.25 | 3.7 | 111.1 | 0.02 | vptq-numpy |
| AQLM 2-bit (Phase 9A) | — | — | — | — | *not yet implemented* |
| QuIP# 2-bit (Phase 9B) | 3.00 | 6.8 | 0.7 | 0.02 | quip-numpy |

### Notes on BPW

| Method | BPW formula | Notes |
|--------|-------------|-------|
| INT4 nibble | 4 + 32/group_size + 32/group_size = 5.0 bpw | Asymmetric; 2× float32 overhead per group (scale + zero_point). Rust symmetric path: 4.5 bpw |
| VPTQ | (log₂(k_primary) + log₂(k_residual)) / group_size + scale_overhead | Benchmark config: k=16 primary + k=4 residual, group=8 → 0.75 bpw indices + 0.5 bpw col-scales = **1.25 bpw**. Production config (k=256 + k=16) → ~1.5–1.75 bpw |
| AQLM | (M × log₂(codebook_size)) / group_size | *Not implemented yet (Phase 9A)* |
| QuIP# | 8 bits (E8 index) + 16 bits (residual scale) per 8-D chunk = **3.0 bpw** | Excludes rotation matrix (per-matrix one-time cost; negligible for large layers) |

### Notes on SNR

The SNR values on the 64×64 synthetic matrix reflect the quantization
error for a small random Gaussian weight distribution (σ=0.02).  On a real
language model, the true quality metric is wikitext-2 perplexity.  The
expected perplexity degradation relative to FP16 is:

| Method | Expected Δ PPL (Qwen2.5-1.5B, wikitext-2) |
|--------|-------------------------------------------|
| INT4 nibble | ~+0.3–0.8 nats above FP16 |
| VPTQ (k=256, group=8) | ~within 1 nat of INT4 |
| AQLM 2-bit | ~within 0.5 nat of INT4 |
| QuIP# 2-bit | ~within 0.3 nat of FP16 |

---

## Stage 2 — Model Evaluation (requires `--model-dir`)

Model-level perplexity and TPS have not been collected yet.
To collect them, run with a downloaded model:

```bash
python3 dev/benchmarks/bench_2bit.py \
    --model-dir models/Qwen2.5-1.5B \
    --ppl-tokens 2048 \
    --tps-tokens 128 \
    --output dev/results/quant_2bit_comparison.json
```

This evaluates FP16 perplexity and generation throughput via `mlx_lm`.
Quantized model evaluation will be added when AQLM (Phase 9A) is complete.

---

## Benchmark Configuration

The benchmark uses reduced settings for fast CI execution (< 15 s per run).
For a high-fidelity comparison on large weight matrices, increase:

```python
# In dev/benchmarks/bench_2bit.py:
BENCH_ROWS      = 4096   # realistic linear layer height
BENCH_COLS      = 4096   # realistic linear layer width
VPTQ_N_PRIMARY  = 256    # full codebook (standard NeurIPS 2025 config)
VPTQ_N_RESIDUAL = 16
VPTQ_ITERS      = 20     # full k-means iterations
```

---

## Running the Benchmark

```bash
# Stage 1 only (no model required, < 15 s):
python3 dev/benchmarks/bench_2bit.py --dry-run

# Stage 1 + Stage 2 (requires mlx_lm and a downloaded model):
python3 dev/benchmarks/bench_2bit.py --model-dir models/Qwen2.5-1.5B

# With Markdown table output:
python3 dev/benchmarks/bench_2bit.py --markdown

# Custom output path:
python3 dev/benchmarks/bench_2bit.py --output /path/to/out.json
```

Results are written to `dev/results/quant_2bit_comparison.json`.

---

## Implementation Status

| Phase | Method | Module | Status |
|-------|--------|--------|--------|
| Baseline | INT4 nibble | `squish/quantizer.py` | ✅ Complete |
| Phase 7 | VPTQ | `squish/vptq.py` | ✅ Complete |
| Phase 9A | AQLM 2-bit | `squish/aqlm.py` | ⏳ Not yet implemented |
| Phase 9B | QuIP# 2-bit | `squish/quip_sharp.py` | ✅ Complete |
| Phase 9C | This benchmark | `dev/benchmarks/bench_2bit.py` | ✅ Complete |

---

## See Also

- [`PLAN.md`](../PLAN.md) — Phase 9 implementation roadmap
- [`squish/vptq.py`](../squish/vptq.py) — VPTQ implementation
- [`squish/quip_sharp.py`](../squish/quip_sharp.py) — QuIP# implementation
- [`dev/results/quant_2bit_comparison.json`](../dev/results/quant_2bit_comparison.json) — Raw results
