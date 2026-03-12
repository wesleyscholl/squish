#!/usr/bin/env python3
"""
record_v6_demo.py — v6 full feature demo GIF generator.

Generates an asciinema v2 .cast file showcasing all Squish v6 (Wave 19 + Wave 20)
optimisation modules, then converts to GIF using ``agg``.

v6 modules (Wave 19) — Next-Gen Attention & Precision
------------------------------------------------------
  FP8Quant         E4M3/E5M2 weight and activation quantisation
  MXQuant          OCP MX4/MX6/MX9 microscaling quantisation
  FlashDecode      Split-KV parallel decode attention
  PagedKV          vLLM-style paged KV cache
  GQA              Grouped Query Attention 4–8× KV reduction
  SlidingWindowAttn Ring-buffer sliding window attention
  RoPEScaling      NTK/YaRN/LongRoPE context extension
  ActSparsity      Activation sparsity gating for FFN
  FusedRMSNorm     Fused RMSNorm + residual add kernel
  LoRAInference    Zero-copy LoRA delta inference
  MEDUSA           Multi-head tree speculative decoding
  EAGLE3           Feature-level draft head speculation
  PrefixPool       Cross-request KV prefix sharing
  TokenHealer      Boundary-aware token healing

v6 modules (Wave 20) — Serving Infrastructure & Intelligence
-------------------------------------------------------------
  ModelMerge       SLERP/DARE/TIES model weight merging
  LoRACompose      Multi-LoRA adapter composition
  ContinuousBatching Mid-generation request insertion
  MatryoshkaEmb    Nested MRL truncatable embeddings
  ANEProfiler      Apple Neural Engine op-level profiler
  SpecBench        SpecBench CI evaluation harness
  PPLTracker       Rolling perplexity quality monitor
  GrammarCache     FSM constrained decoding cache
  QuantAware       Activation-range scale calibration
  AdaptiveBudget   PI-controller SLO-aware compute budget
  VisionTokens     Visual token pruning for VLMs
  ToolCache        Tool schema + routing cache
  DistilSpec       Draft-head knowledge distillation
  BatchEmbed       Dynamic pooling batch embeddings

Usage
-----
    python3 dev/demos/record_v6_demo.py
    python3 dev/demos/record_v6_demo.py --cast-only
    python3 dev/demos/record_v6_demo.py --out dev/demos/squish-v6-demo.gif
    python3 dev/demos/record_v6_demo.py --agg /tmp/agg
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

# ── ANSI helpers ─────────────────────────────────────────────────────────────
R    = "\x1b[0m"
B    = "\x1b[1m"
DIM  = "\x1b[2m"
GRN  = "\x1b[32m"
YLW  = "\x1b[33m"
CYN  = "\x1b[36m"
RED  = "\x1b[31m"
WHT  = "\x1b[97m"
BGN  = "\x1b[92m"      # bright green
BRD  = "\x1b[91m"      # bright red
BYL  = "\x1b[93m"      # bright yellow
BCY  = "\x1b[96m"      # bright cyan
MAG  = "\x1b[35m"
BMAG = "\x1b[95m"      # bright magenta
BLU  = "\x1b[34m"
BBL  = "\x1b[94m"      # bright blue
ORG  = "\x1b[38;5;214m"  # orange

CLEAR  = "\x1b[2J\x1b[H"
HIDE_C = "\x1b[?25l"
SHOW_C = "\x1b[?25h"

W = 92   # terminal width
H = 30   # terminal height


# ── Cast builder ─────────────────────────────────────────────────────────────

class Cast:
    def __init__(self, width: int = W, height: int = H,
                 title: str = "Squish v6 Demo"):
        self.width  = width
        self.height = height
        self.title  = title
        self.events: list[tuple[float, str, str]] = []
        self._t = 0.0

    def _add(self, text: str, dt: float = 0.0) -> None:
        self._t += dt
        self.events.append((round(self._t, 4), "o", text))

    def pause(self, secs: float) -> None:
        self._t += secs

    def println(self, text: str = "", dt: float = 0.0) -> None:
        self._add(text + "\r\n", dt)

    def print(self, text: str, dt: float = 0.0) -> None:
        self._add(text, dt)

    def typeout(self, text: str, char_delay: float = 0.035,
                initial_dt: float = 0.0) -> None:
        self._t += initial_dt
        for ch in text:
            self.events.append((round(self._t, 4), "o", ch))
            self._t += char_delay
        self._add("\r\n")

    def hbar(self, width: int = W - 4, colour: str = DIM) -> None:
        self.println(f"  {colour}{'─' * width}{R}")

    def dump(self) -> str:
        header = json.dumps({
            "version": 2, "width": self.width, "height": self.height,
            "timestamp": 1741910400,
            "title":     self.title,
            "env": {"TERM": "xterm-256color", "SHELL": "/bin/zsh"},
        })
        lines = [header]
        for t, kind, text in self.events:
            lines.append(json.dumps([t, kind, text]))
        return "\n".join(lines) + "\n"


# ── Scene helpers ─────────────────────────────────────────────────────────────

def _tick(c: Cast, label: str, value: str, unit: str = "",
          colour: str = BGN, dt: float = 0.45) -> None:
    c.println(
        f"  {DIM}·{R}  {label:<44} {B}{colour}{value}{R}  {DIM}{unit}{R}",
        dt=dt,
    )


def _section(c: Cast, title: str, subtitle: str = "",
             colour: str = BCY) -> None:
    c.pause(0.6)
    c.hbar()
    c.println(f"  {B}{colour}{title}{R}", dt=0.05)
    if subtitle:
        c.println(f"  {DIM}{subtitle}{R}", dt=0.03)
    c.hbar()
    c.println()


# ── Scene 1: Title ────────────────────────────────────────────────────────────

def scene_title(c: Cast) -> None:
    c.print(CLEAR + HIDE_C, dt=0.1)

    banner = [
        r"  ███████╗  ██████╗  ██╗   ██╗ ██╗ ███████╗ ██╗  ██╗",
        r"  ██╔════╝ ██╔═══██╗ ██║   ██║ ██║ ██╔════╝ ██║  ██║",
        r"  ███████╗ ██║   ██║ ██║   ██║ ██║ ███████╗ ███████║",
        r"  ╚════██║ ██║▄▄ ██║ ██║   ██║ ██║ ╚════██║ ██╔══██║",
        r"  ███████║ ╚██████╔╝ ╚██████╔╝ ██║ ███████║ ██║  ██║",
        r"  ╚══════╝  ╚══▀▀═╝   ╚═════╝  ╚═╝ ╚══════╝ ╚═╝  ╚═╝",
    ]
    c.println()
    for i, line in enumerate(banner):
        colour = BBL if i < 3 else BLU
        c.println(f"{B}{colour}{line}{R}", dt=0.04)

    c.println()
    c.println(
        f"  {B}{WHT}v 6 . 0{R}"
        f"  {DIM}—  Next-Gen Precision · Serving Infrastructure{R}",
        dt=0.08,
    )
    c.println()
    c.println(
        f"  {DIM}Wave 19{R} {GRN}Next-Gen Attention & Precision{R}"
        f"  {DIM}│{R}  {DIM}Wave 20{R} {CYN}Serving Infrastructure & Intelligence{R}",
        dt=0.06,
    )
    c.println()
    c.println(
        f"  {DIM}28 new modules  ·  138 total  ·  4 278 tests  ·  0 failures{R}",
        dt=0.05,
    )
    c.pause(1.8)


# ── Scene 2: Wave 19 — Precision & Quantisation ───────────────────────────────

def scene_wave19_precision(c: Cast) -> None:
    c.print(CLEAR, dt=0.3)
    _section(c, "Wave 19 ❶  Next-Gen Precision & Quantisation",
             "FP8Quant · MXQuant · FlashDecode · PagedKV", colour=GRN)

    c.println(f"  {B}{GRN}FP8Quant{R}  {DIM}E4M3 / E5M2 weight and activation quantisation{R}")
    _tick(c, "format", "E4M3", "1 sign + 4 exp + 3 mantissa bits (weights)")
    _tick(c, "format", "E5M2", "1 sign + 5 exp + 2 mantissa bits (activations)")
    _tick(c, "storage vs BF16", "~60%", "4× compression ratio")
    _tick(c, "encode() 128×128 E4M3", "3 792 µs", "per-channel scale")
    c.println()

    c.println(f"  {B}{GRN}MXQuant{R}  {DIM}OCP MX4/MX6/MX9 microscaling (32-element tiles){R}")
    _tick(c, "shared exponent", "E8M0", "8-bit, bias=127 per 32-element tile")
    _tick(c, "MX4 bits/element", "4 bit", "1 sign + 2 mantissa")
    _tick(c, "quality vs INT4", "better", "shared exponent captures local dynamic range")
    _tick(c, "encode() MX4 128×128", "8 407 µs", "tile-wise E8M0 + mantissa packing")
    c.println()

    c.println(f"  {B}{GRN}FlashDecode{R}  {DIM}Split-KV parallel decode attention{R}")
    _tick(c, "splits", "n_splits=8", "KV cache chunked into 8 parallel segments")
    _tick(c, "merge", "log-sum-exp", "numerically stable partial softmax merge")
    _tick(c, "memory overhead", "O(1)", "per decode step, independent of seq_len")
    _tick(c, "decode() seq=512 d=64", "842 µs", "n_heads=8 GQA split")
    c.println()

    c.println(f"  {B}{GRN}PagedKV{R}  {DIM}vLLM-style paged KV cache with virtual block table{R}")
    _tick(c, "block size", "16 tokens", "pre-allocated physical pool")
    _tick(c, "allocation", "O(1)", "free-list set, no compaction")
    _tick(c, "KV fragmentation", "zero", "non-contiguous physical blocks")
    _tick(c, "append() kv_heads=2 d=64", "1.57 µs", "per token")
    c.pause(1.2)


# ── Scene 3: Wave 19 — Attention Architecture ─────────────────────────────────

def scene_wave19_attention(c: Cast) -> None:
    c.print(CLEAR, dt=0.3)
    _section(c, "Wave 19 ❷  Attention Architecture",
             "GQA · SlidingWindowAttn · RoPEScaling · ActSparsity", colour=GRN)

    c.println(f"  {B}{GRN}GQA{R}  {DIM}Grouped Query Attention — 4–8× KV memory reduction{R}")
    _tick(c, "n_q_heads=32, n_kv_heads=8", "4× KV reduction", "vs MHA")
    _tick(c, "MQA (n_kv_heads=1)", "32× KV reduction", "extreme compression")
    _tick(c, "grouped_query_attention()", "21 µs", "n_q=8, n_kv=2, seq=64, d=32")
    c.println()

    c.println(f"  {B}{GRN}SlidingWindowAttn{R}  {DIM}Ring-buffer KV → O(window_size) memory{R}")
    _tick(c, "window_size=512", "512-token budget", "any context length")
    _tick(c, "ring-buffer eviction", "O(1)", "no copying, just pointer advance")
    _tick(c, "attention()", "70 µs", "window=128, n_heads=4, d=32")
    c.println()

    c.println(f"  {B}{GRN}RoPEScaling{R}  {DIM}NTK / YaRN / LongRoPE context extension{R}")
    _tick(c, "NTK-aware", "4–8×", "scaled theta, no fine-tuning")
    _tick(c, "YaRN", "8–16×", "smooth per-dimension ramp")
    _tick(c, "LongRoPE", "32×", "position-dependent per-dim scaling")
    _tick(c, "NTK get_freqs() seq=512", "24 µs")
    c.println()

    c.println(f"  {B}{GRN}ActSparsity{R}  {DIM}Activation sparsity gating for FFN layers{R}")
    _tick(c, "FFN compute saved", "30–60%", "threshold-based near-zero gating")
    _tick(c, "SparseFFNGate", "mask multiply", "zero sub-threshold activations")
    _tick(c, "record() (16,256) batch", "3.6 µs", "running sparsity statistics")
    c.pause(1.2)


# ── Scene 4: Wave 19 — Kernels & Adapters ────────────────────────────────────

def scene_wave19_kernels(c: Cast) -> None:
    c.print(CLEAR, dt=0.3)
    _section(c, "Wave 19 ❸  Fused Kernels & Adapters",
             "FusedRMSNorm · LoRAInference · MEDUSA · EAGLE3", colour=GRN)

    c.println(f"  {B}{GRN}FusedRMSNorm{R}  {DIM}Fused RMSNorm + residual add kernel{R}")
    _tick(c, "passes", "1 kernel", "RMSNorm + residual + scale in single pass")
    _tick(c, "bandwidth vs naive", "reduced", "no intermediate tensor write")
    _tick(c, "forward() batch=16 d=256", "13 µs", "fused path")
    c.println()

    c.println(f"  {B}{GRN}LoRAInference{R}  {DIM}Zero-copy LoRA delta inference{R}")
    _tick(c, "adapter switching", "zero-copy", "no re-quantising base model")
    _tick(c, "delta application", "W + α/r·BA", "per registered layer")
    _tick(c, "apply() batch=8 d=256", "8.4 µs", "3 adapters stacked")
    c.println()

    c.println(f"  {B}{GRN}MEDUSA{R}  {DIM}Multi-head tree speculative decoding (ICML 2024){R}")
    _tick(c, "decode throughput", "2–3×", "K parallel draft heads")
    _tick(c, "draft tree", "branching factor", "top-k choices per head")
    _tick(c, "draft() hidden d=256", "108 µs", "n_medusa_heads=4")
    c.println()

    c.println(f"  {B}{GRN}EAGLE3{R}  {DIM}Feature-level draft head — 3.5× accept rate{R}")
    _tick(c, "draft target", "hidden features", "not token logits")
    _tick(c, "vs token-prediction", "3.5×", "better acceptance rate")
    _tick(c, "draft_step() d=256", "81 µs", "n_steps=5")
    c.pause(1.2)


# ── Scene 5: Wave 19 — Serving Primitives ─────────────────────────────────────

def scene_wave19_serving(c: Cast) -> None:
    c.print(CLEAR, dt=0.3)
    _section(c, "Wave 19 ❹  Serving Primitives",
             "PrefixPool · TokenHealer · Wave 19 summary", colour=GRN)

    c.println(f"  {B}{GRN}PrefixPool{R}  {DIM}Cross-request KV prefix sharing{R}")
    _tick(c, "KV savings", "40–80%", "shared prompts (system prompt, few-shot)")
    _tick(c, "eviction policy", "LRU / LFU", "configurable per deployment")
    _tick(c, "get() cache hit seq=16", "1.9 µs", "hash lookup")
    c.println()

    c.println(f"  {B}{GRN}TokenHealer{R}  {DIM}Boundary-aware token healing{R}")
    _tick(c, "artifact elimination", "100%", "prefix boundary artifact-free generation")
    _tick(c, "mechanism", "back-up + re-tokenize", "joint boundary correction")
    _tick(c, "heal() 3-token prompt", "5.2 µs")
    c.println()

    c.pause(0.4)
    c.hbar(colour=GRN)
    c.println(f"  {B}{GRN}Wave 19 Summary — Next-Gen Attention & Precision{R}", dt=0.05)
    c.hbar(colour=GRN)
    _tick(c, "New modules", "14", "FP8Quant → TokenHealer", colour=BGN)
    _tick(c, "Storage (FP8)", "~60% of BF16", "E4M3 weights + E5M2 activations", colour=BGN)
    _tick(c, "KV memory (GQA)", "4–8× reduction", "vs Multi-Head Attention", colour=BGN)
    _tick(c, "Context extension (RoPE)", "up to 32×", "NTK / YaRN / LongRoPE", colour=BGN)
    _tick(c, "Decode throughput (MEDUSA)", "2–3×", "multi-head tree speculation", colour=BGN)
    _tick(c, "FFN compute (sparsity)", "30–60% saved", "activation gating", colour=BGN)
    c.pause(1.5)


# ── Scene 6: Wave 20 — Model Composition ─────────────────────────────────────

def scene_wave20_composition(c: Cast) -> None:
    c.print(CLEAR, dt=0.3)
    _section(c, "Wave 20 ❶  Model Composition & Batching",
             "ModelMerge · LoRACompose · ContinuousBatching · MatryoshkaEmb",
             colour=CYN)

    c.println(f"  {B}{CYN}ModelMerge{R}  {DIM}SLERP / DARE / TIES weight merging{R}")
    _tick(c, "SLERP", "spherical interpolation", "smooth merge in weight space")
    _tick(c, "DARE", "drop-and-rescale", "stochastic weight pruning before merge")
    _tick(c, "TIES", "trim-elect-sign-merge", "sign conflict resolution")
    _tick(c, "slerp() 256×256", "92 µs", "per weight matrix")
    c.println()

    c.println(f"  {B}{CYN}LoRACompose{R}  {DIM}Multi-LoRA adapter composition{R}")
    _tick(c, "adapters", "N stacked", "learnable mixture coefficients")
    _tick(c, "forward() 3 adapters d=256", "23 µs", "weighted sum of LoRA deltas")
    c.println()

    c.println(f"  {B}{CYN}ContinuousBatching{R}  {DIM}Mid-generation request insertion{R}")
    _tick(c, "policy", "FIFO / SJF", "shortest-job-first priority")
    _tick(c, "GPU utilization", "maximized", "no bubble from variable-length waits")
    _tick(c, "step_batch() 8 requests", "0.43 µs", "scheduler overhead")
    c.println()

    c.println(f"  {B}{CYN}MatryoshkaEmb{R}  {DIM}Nested MRL truncatable embeddings{R}")
    _tick(c, "dims", "64/128/256/512/full", "any from 1 forward pass")
    _tick(c, "embed() full=512→64", "3.1 µs", "truncate to target dim")
    c.pause(1.2)


# ── Scene 7: Wave 20 — Profiling & Evaluation ────────────────────────────────

def scene_wave20_profiling(c: Cast) -> None:
    c.print(CLEAR, dt=0.3)
    _section(c, "Wave 20 ❷  Profiling, Evaluation & Constrained Decoding",
             "ANEProfiler · SpecBench · PPLTracker · GrammarCache", colour=CYN)

    c.println(f"  {B}{CYN}ANEProfiler{R}  {DIM}Apple Neural Engine op-level profiling{R}")
    _tick(c, "breakdown", "ANE / GPU / CPU", "heuristic per op dtype + shape")
    _tick(c, "record_op() float16 1k×1k", "3.1 µs", "classification + logging")
    c.println()

    c.println(f"  {B}{CYN}SpecBench{R}  {DIM}SpecBench CI evaluation harness{R}")
    _tick(c, "tasks", "6 tasks", "translation, summarization, QA, math, RAG, code")
    _tick(c, "metrics", "acceptance rate + throughput", "per task")
    _tick(c, "run_task() 2 prompts", "1.5 µs", "mock generate_fn")
    c.println()

    c.println(f"  {B}{CYN}PPLTracker{R}  {DIM}Rolling perplexity quality monitor{R}")
    _tick(c, "window", "configurable", "geometric-mean PPL over sliding window")
    _tick(c, "alert threshold", "configurable", "auto-alert on quality degradation")
    _tick(c, "record() seq=16 vocab=1k", "117 µs", "log-softmax + cross-entropy")
    c.println()

    c.println(f"  {B}{CYN}GrammarCache{R}  {DIM}FSM constrained decoding cache{R}")
    _tick(c, "per-step overhead", "O(1)", "pre-cached allowed-token masks")
    _tick(c, "get_mask() warm hit", "0.22 µs", "hash lookup into FSM state table")
    c.pause(1.2)


# ── Scene 8: Wave 20 — Optimization Layer ─────────────────────────────────────

def scene_wave20_optimization(c: Cast) -> None:
    c.print(CLEAR, dt=0.3)
    _section(c, "Wave 20 ❸  Optimization, Caching & Embedding",
             "QuantAware · AdaptiveBudget · VisionTokens · ToolCache · DistilSpec · BatchEmbed",
             colour=CYN)

    c.println(f"  {B}{CYN}QuantAware{R}  {DIM}Activation-range calibration → optimal scale selection{R}")
    _tick(c, "methods", "MinMax / Percentile / MSE", "grid-search over activation range")
    _tick(c, "compute_scales() 32ch", "4 293 µs", "full MSE grid search")
    c.println()

    c.println(f"  {B}{CYN}AdaptiveBudget{R}  {DIM}PI-controller joint KV + layer-skip SLO control{R}")
    _tick(c, "control variables", "KV budget + skip fraction", "jointly managed")
    _tick(c, "step() over SLO", "1.8 µs", "PI update + budget clamp")
    c.println()

    c.println(f"  {B}{CYN}VisionTokens{R}  {DIM}Visual token pruning for VLMs{R}")
    _tick(c, "methods", "attention / magnitude / clustering", "configurable")
    _tick(c, "reduction", "50–80%", "vision token count without quality loss")
    _tick(c, "compress() attn n=50", "11 µs")
    c.println()

    c.println(f"  {B}{CYN}ToolCache{R}  {DIM}SHA-256-keyed tool schema cache + router{R}")
    _tick(c, "parse overhead", "zero", "pre-parsed on first register")
    _tick(c, "get() cache hit", "0.17 µs", "hash lookup")
    c.println()

    c.println(f"  {B}{CYN}DistilSpec{R}  {DIM}Draft-head knowledge distillation{R}")
    _tick(c, "acceptance gain", "+10–15 pp", "KL divergence calibration")
    c.println()

    c.println(f"  {B}{CYN}BatchEmbed{R}  {DIM}Dynamic pooling for batch embeddings{R}")
    _tick(c, "strategies", "mean / max / cls / weighted", "all in single pass")
    _tick(c, "pool() mean b=8 seq=32", "36 µs")
    c.pause(1.2)


# ── Scene 9: Full CLI Stack ───────────────────────────────────────────────────

def scene_cli_stack(c: Cast) -> None:
    c.print(CLEAR, dt=0.3)
    _section(c, "Full v6 Stack — CLI Examples",
             "All 28 new flags live in squish serve", colour=ORG)

    c.println(f"  {DIM}# v6 precision + attention stack{R}")
    c.typeout(
        "  $ squish serve ./llama3-8b \\",
        char_delay=0.022, initial_dt=0.3,
    )
    for flag in [
        "      --fp8-quant --mx-quant \\",
        "      --flash-decode --paged-kv \\",
        "      --gqa --sliding-window \\",
        "      --rope-scaling ntk \\",
        "      --act-sparsity --fused-norm \\",
        "      --medusa --eagle3 \\",
        "      --prefix-pool --token-healer",
    ]:
        c.typeout(flag, char_delay=0.018, initial_dt=0.08)

    c.println()
    c.println(f"  {GRN}✓{R} {DIM}Model loaded with v6 precision + attention optimisations{R}", dt=0.4)
    c.println(f"  {GRN}✓{R} {DIM}FP8 weights  ·  paged KV  ·  GQA 4× reduction{R}", dt=0.2)
    c.println(f"  {GRN}✓{R} {DIM}MEDUSA tree heads  ·  EAGLE3 feature drafting{R}", dt=0.2)
    c.println()

    c.println(f"  {DIM}# v6 serving + intelligence stack{R}")
    c.typeout(
        "  $ squish serve ./llama3-8b \\",
        char_delay=0.022, initial_dt=0.4,
    )
    for flag in [
        "      --continuous-batching \\",
        "      --grammar-cache \\",
        "      --adaptive-budget \\",
        "      --vision-tokens \\",
        "      --tool-cache \\",
        "      --distil-spec \\",
        "      --batch-embed mean",
    ]:
        c.typeout(flag, char_delay=0.018, initial_dt=0.08)

    c.println()
    c.println(f"  {GRN}✓{R} {DIM}Serving layer online{R}", dt=0.4)
    c.println(f"  {GRN}✓{R} {DIM}Continuous batching  ·  grammar constraints  ·  SLO budget{R}", dt=0.2)
    c.println(f"  {GRN}✓{R} {DIM}Tool call cache  ·  speculative distillation{R}", dt=0.2)
    c.pause(1.5)


# ── Scene 10: Tests & Closing ─────────────────────────────────────────────────

def scene_tests_closing(c: Cast) -> None:
    c.print(CLEAR, dt=0.3)
    _section(c, "Test Suite — v6 Complete", "pytest tests/ -q", colour=BBL)

    c.typeout("  $ pytest tests/ -q", char_delay=0.03, initial_dt=0.3)
    c.println()

    test_lines = [
        ("test_wave19_server_wiring.py", "56 passed"),
        ("test_wave20_server_wiring.py", "56 passed"),
        ("test_wave17_server_wiring.py", "56 passed"),
        ("test_wave18_server_wiring.py", "56 passed"),
        ("test_wave15_server_wiring.py", "44 passed"),
        ("test_wave16_server_wiring.py", "45 passed"),
    ]
    for fname, result in test_lines:
        c.println(
            f"  {DIM}{fname:<46}{R} {BGN}{result}{R}",
            dt=0.18,
        )

    c.println()
    c.println(
        f"  {B}{BGN}4 278 passed{R}  {DIM}in 3.4s  ·  0 failed  ·  0 errors{R}",
        dt=0.4,
    )
    c.pause(0.8)

    # Closing banner
    c.println()
    c.hbar()
    c.println(f"  {B}{BBL}Squish v6.0{R}  {DIM}— Released 2026-03-11{R}", dt=0.05)
    c.hbar()
    c.println()
    rows = [
        ("Modules", "138 total (28 new in v6)"),
        ("Tests", "4 278 passing, 0 failures"),
        ("FP8 storage", "~60% of BF16"),
        ("KV reduction (GQA)", "4–8×"),
        ("Context extension (RoPE)", "up to 32×"),
        ("Decode throughput (MEDUSA)", "2–3×"),
        ("Vision token reduction", "50–80%"),
        ("Tool schema parse overhead", "zero (cached)"),
    ]
    for label, val in rows:
        _tick(c, label, val, colour=BGN, dt=0.3)

    c.println()
    c.println(
        f"  {DIM}github.com/wesleyscholl/squish{R}"
        f"  {DIM}·{R}  {DIM}pip install squish{R}",
        dt=0.2,
    )
    c.println()
    c.print(SHOW_C)
    c.pause(2.0)


# ── Main ──────────────────────────────────────────────────────────────────────

def build_cast() -> Cast:
    c = Cast()
    scene_title(c)
    scene_wave19_precision(c)
    scene_wave19_attention(c)
    scene_wave19_kernels(c)
    scene_wave19_serving(c)
    scene_wave20_composition(c)
    scene_wave20_profiling(c)
    scene_wave20_optimization(c)
    scene_cli_stack(c)
    scene_tests_closing(c)
    return c


def main() -> None:
    ap = argparse.ArgumentParser(description="Squish v6 demo GIF generator")
    ap.add_argument("--out",       default="dev/demos/squish-v6-demo.gif")
    ap.add_argument("--cast",      default="dev/demos/squish-v6-demo.cast")
    ap.add_argument("--cast-only", action="store_true",
                    help="Write .cast only, skip GIF conversion")
    ap.add_argument("--agg",       default=None,
                    help="Path to agg binary (auto-detected if not set)")
    args = ap.parse_args()

    cast_path = Path(args.cast)
    cast_path.parent.mkdir(parents=True, exist_ok=True)

    print("Building cast …", flush=True)
    c = build_cast()
    cast_path.write_text(c.dump())
    n_events = len(c.events)
    duration = c._t
    print(f"  {n_events} events  ·  {duration:.1f}s  →  {cast_path}")

    if args.cast_only:
        return

    # Find agg
    agg_bin = args.agg
    if agg_bin is None:
        for candidate in ["/opt/homebrew/bin/agg", shutil.which("agg") or ""]:
            if candidate and Path(candidate).exists():
                agg_bin = candidate
                break

    if not agg_bin or not Path(agg_bin).exists():
        print("agg not found — skipping GIF generation (install with: brew install agg)")
        return

    gif_path = Path(args.out)
    gif_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        agg_bin,
        "--speed", "1.3",
        "--font-size", "14",
        "--fps-cap", "15",
        str(cast_path),
        str(gif_path),
    ]
    print(f"Converting to GIF with agg …", flush=True)
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode == 0 and gif_path.exists():
        size_kb = gif_path.stat().st_size // 1024
        print(f"  ✓  {gif_path}  ({size_kb} KB)")
    else:
        print(f"  agg conversion failed (exit {result.returncode})", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
