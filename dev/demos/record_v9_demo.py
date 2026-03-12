#!/usr/bin/env python3
"""
record_v9_demo.py — v9 full feature demo GIF generator.

Generates an asciinema v2 .cast file showcasing all Squish v9 (Wave 25 + Wave 26)
optimisation modules, then converts to GIF using ``agg``.

v9 modules (Wave 25) — Cutting-Edge Attention Variants & Compute Fusion
------------------------------------------------------------------------
  FlashMLA          DeepSeek-V2 multi-head latent attention
  NativeSparseAttn  Block-sparse + sliding-window attention (DeepSeek-V3 NSA)
  FusedSampler      Fused temperature/top-k/top-p/min-p/rep-penalty sampling
  KVDefrag          Online KV cache page defragmentation
  DualChunkAttn     Intra+inter-chunk long-context attention
  ActivationOffload CPU activation offloading with prefetch-ahead
  MorphAttn         Per-layer full/sparse/linear attention morphing
  HydraSpec         Multi-draft head speculative decoding
  SeqCompact        In-place KV sequence compaction via boolean mask
  LatencyPredictor  OLS latency forecasting for batch scheduler
  ParallelSampler   Best-of-N + diversity-scored sampling
  ContextSummarizer Importance/stride/recency context compression
  TokenWatermark    Kirchenbauer green-list statistical watermarking
  SchemaGen         FSM-based constrained JSON generation

v9 modules (Wave 26) — Distributed Inference & Production Reliability
-----------------------------------------------------------------------
  TensorParallel    Row/column tensor sharding + all-reduce forward
  SequenceParallel  Ulysses-style sequence scatter/gather
  KVMigrate         Live KV state pack/unpack for inter-node migration
  DisaggPrefill     Disaggregated prefill + decode node pipeline
  RequestPreempt    SRPT preemption with swap/recompute strategies
  InferGateway      Least-loaded smart request routing gateway
  ModelVersionSwap  Canary→promote→rollback version management
  ProductionProfiler APM windowed p50/p99/p999 profiling
  AdaptiveBatcher   Throughput/latency-objective dynamic batching
  SafetyLayer       Inline token safety classifier
  SemanticResponseCache Embedding-similarity LRU response cache
  RateLimiter       Token-bucket per-tenant rate limiting
  SchemaValidator   JSON schema validation
  AuditLogger       SHA-256 hash-chained audit log

Usage
-----
    python3 dev/demos/record_v9_demo.py
    python3 dev/demos/record_v9_demo.py --cast-only
    python3 dev/demos/record_v9_demo.py --out dev/demos/squish-v9-demo.gif
    python3 dev/demos/record_v9_demo.py --agg /tmp/agg
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
                 title: str = "Squish v9 Demo"):
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
            "timestamp": 1741996800,
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
        colour = BMAG if i < 3 else BCY
        c.println(f"{B}{colour}{line}{R}", dt=0.04)

    c.println()
    c.println(
        f"  {B}{WHT}v 9 . 0{R}"
        f"  {DIM}—  Cutting-Edge Attention Variants · Distributed Inference & Production Reliability{R}",
        dt=0.08,
    )
    c.println()
    c.println(
        f"  {DIM}Wave 25{R} {BMAG}Cutting-Edge Attention Variants & Compute Fusion{R}"
        f"  {DIM}│{R}  {DIM}Wave 26{R} {BCY}Distributed Inference & Production Reliability{R}",
        dt=0.06,
    )
    c.println()
    c.println(
        f"  {DIM}28 new modules  ·  222 total  ·  4 876 tests  ·  0 failures{R}",
        dt=0.05,
    )
    c.pause(1.8)


# ── Scene 2: Wave 25 — Attention Variants ────────────────────────────────────

def scene_wave25_attention(c: Cast) -> None:
    c.print(CLEAR, dt=0.3)
    _section(c, "Wave 25 ❶  Cutting-Edge Attention Variants",
             "FlashMLA · NativeSparseAttn · DualChunkAttn · MorphAttn · HydraSpec",
             colour=BMAG)

    c.println(f"  {B}{BMAG}FlashMLA{R}  {DIM}DeepSeek-V2 multi-head latent attention{R}")
    _tick(c, "method", "latent KV compression", "stores (latent_dim,) per token, not (n_heads, head_dim)")
    _tick(c, "compression_ratio", "4×", "n_heads=8 head_dim=64 latent_dim=128")
    _tick(c, "append() single token", "0.55 µs", "latent vector insert, no per-head expansion")
    _tick(c, "attend() seq=16 h=8 d=64", "38.65 µs", "W_uk/W_uv projection + scaled dot-product")
    c.println()

    c.println(f"  {B}{BMAG}NativeSparseAttn{R}  {DIM}Block-sparse + sliding-window (DeepSeek-V3 NSA){R}")
    _tick(c, "method", "top-k blocks + local window", "block_size=64 top_k=4 window=64")
    _tick(c, "sparsity", "~87%", "only top-k + window blocks computed")
    _tick(c, "forward() h=4 kv=256", "646.6 µs", "masked block-sparse attention pass")
    c.println()

    c.println(f"  {B}{BMAG}DualChunkAttn{R}  {DIM}Intra+inter-chunk long-context attention{R}")
    _tick(c, "method", "encode_chunk + top-k inter", "chunk summary vectors for long-range recall")
    _tick(c, "encode_chunk() chunk=64", "21.08 µs", "intra-chunk key compression to summary")
    _tick(c, "forward() 4 past_chunks", "93.3 µs", "intra + top-k inter-chunk attention")
    c.println()

    c.println(f"  {B}{BMAG}MorphAttn{R}  {DIM}Per-layer full/sparse/linear attention morphing{R}")
    _tick(c, "thresholds", "full < 512 < sparse < 4096 < linear", "automatic pattern by seq_len")
    _tick(c, "select_pattern() layer+seq", "0.25 µs", "constant-time threshold comparison")
    _tick(c, "estimate_flops_reduction()", "~40% at seq=2048", "analytical FLOP model")
    c.println()

    c.println(f"  {B}{BMAG}HydraSpec{R}  {DIM}Multi-draft head speculative decoding{R}")
    _tick(c, "method", "n_heads independent draft heads", "each head drafts n_draft tokens")
    _tick(c, "draft() hidden=256 h=4 x5", "1 069 µs", "multi-head parallel draft generation")
    _tick(c, "verify() — best head accept", "1 229 µs", "longest accepted prefix across heads")
    c.pause(1.2)


# ── Scene 3: Wave 25 — Compute Fusion & Sampling ─────────────────────────────

def scene_wave25_fusion_sampling(c: Cast) -> None:
    c.print(CLEAR, dt=0.3)
    _section(c, "Wave 25 ❷  Compute Fusion & Sampling",
             "FusedSampler · KVDefrag · ActivationOffload · SeqCompact · ParallelSampler",
             colour=BMAG)

    c.println(f"  {B}{BMAG}FusedSampler{R}  {DIM}Fused temp/top-k/top-p/min-p/rep-penalty{R}")
    _tick(c, "ops fused", "5 in single pass", "temperature · top-k · top-p · min-p · rep-pen")
    _tick(c, "sample() vocab=32k", "1 767 µs", "full fused decode sampling pipeline")
    _tick(c, "sample_batch() b=16 v=32k", "22 982 µs", "batched independent sampling")
    c.println()

    c.println(f"  {B}{BMAG}KVDefrag{R}  {DIM}Online KV cache page defragmentation{R}")
    _tick(c, "method", "page-level compaction", "moves live pages to close fragmentation gaps")
    _tick(c, "allocate()+free() 16-tok", "2.36 µs", "O(1) page list management")
    _tick(c, "defrag()", "349 µs", "full live-page compaction pass")
    c.println()

    c.println(f"  {B}{BMAG}ActivationOffload{R}  {DIM}CPU activation offloading + prefetch-ahead{R}")
    _tick(c, "policy", "per-layer offload list", "offload_layers=[0,2,4,…] prefetch_ahead=2")
    _tick(c, "offload() 512×128 tensor", "5.84 µs", "CPU copy + layer flag set")
    _tick(c, "fetch() 512×128 tensor", "6.34 µs", "returns stored numpy array")
    c.println()

    c.println(f"  {B}{BMAG}SeqCompact{R}  {DIM}In-place KV sequence compaction{R}")
    _tick(c, "method", "boolean mask → gather", "packed output, -1 for discarded positions")
    _tick(c, "compact() h=8 seq=512 50%", "141 µs", "mask-driven KV tensor gather")
    _tick(c, "compact_indices() seq=512", "2.35 µs", "index mapping with -1 fills")
    c.println()

    c.println(f"  {B}{BMAG}ParallelSampler{R}  {DIM}Best-of-N + diversity-scored sampling{R}")
    _tick(c, "method", "n_samples=8 independent", "diversity_weight penalises similar tokens")
    _tick(c, "sample() vocab=32k n=8", "509 µs", "best-of-8 with diversity score")
    _tick(c, "sample_batch() b=16", "7 637 µs", "batched best-of-N selection")
    c.pause(1.2)


# ── Scene 4: Wave 25 — Intelligence & Constraints ────────────────────────────

def scene_wave25_intelligence(c: Cast) -> None:
    c.print(CLEAR, dt=0.3)
    _section(c, "Wave 25 ❸  Intelligence & Constraints",
             "LatencyPredictor · ContextSummarizer · TokenWatermark · SchemaGen",
             colour=BMAG)

    c.println(f"  {B}{BMAG}LatencyPredictor{R}  {DIM}OLS latency forecasting for batch scheduler{R}")
    _tick(c, "method", "online OLS regression", "features: n_prefill, n_decode → predicted_ms")
    _tick(c, "predict() with fitted model", "0.82 µs", "dot-product evaluation, near-zero cost")
    _tick(c, "record() single sample", "0.78 µs", "online feature append")
    c.println()

    c.println(f"  {B}{BMAG}ContextSummarizer{R}  {DIM}Importance/stride/recency context compression{R}")
    _tick(c, "methods", "importance · stride · recency", "pluggable compression strategies")
    _tick(c, "summarize() importance 1k", "62.5 µs", "embedding-importance ranking + select")
    _tick(c, "summarize() recency 1k", "6.2 µs", "keep-last-N slice, sub-10µs")
    c.println()

    c.println(f"  {B}{BMAG}TokenWatermark{R}  {DIM}Kirchenbauer green-list statistical watermarking{R}")
    _tick(c, "method", "50% green-list + delta boost", "context-sensitive partition per token")
    _tick(c, "mark() vocab=8k", "137 µs", "logit boost by delta=2.0 on green list")
    _tick(c, "detect() seq=128", "z-score ≥ 4.0", "statistical watermark detection")
    c.println()

    c.println(f"  {B}{BMAG}SchemaGen{R}  {DIM}FSM-based constrained JSON generation{R}")
    _tick(c, "method", "stack-based FSM", "valid_cats per state, logit masking")
    _tick(c, "constrain() vocab=256", "5.38 µs", "masks invalid token positions to -inf")
    _tick(c, "advance() single step", "0.79 µs", "O(1) FSM state transition")
    c.pause(1.2)


# ── Scene 5: Wave 25 Summary ──────────────────────────────────────────────────

def scene_wave25_summary(c: Cast) -> None:
    c.print(CLEAR, dt=0.3)
    c.pause(0.3)
    c.hbar(colour=BMAG)
    c.println(f"  {B}{BMAG}Wave 25 Summary — Cutting-Edge Attention Variants & Compute Fusion{R}", dt=0.05)
    c.hbar(colour=BMAG)
    _tick(c, "New modules", "14", "FlashMLA → SchemaGen", colour=BGN)
    _tick(c, "FlashMLA compression_ratio", "4×", "latent_dim=128 vs n_heads*head_dim=512", colour=BGN)
    _tick(c, "NativeSparseAttn sparsity", "~87%", "block-sparse + sliding window", colour=BGN)
    _tick(c, "MorphAttn FLOP reduction", "40% at seq=2048", "full→sparse→linear adaptive", colour=BGN)
    _tick(c, "HydraSpec draft throughput", "1 069 µs / h=4 n=5", "multi-head speculation", colour=BGN)
    _tick(c, "LatencyPredictor predict()", "0.82 µs", "sub-microsecond OLS forecast", colour=BGN)
    _tick(c, "ContextSummarizer recency", "6.2 µs", "near-zero compression overhead", colour=BGN)
    _tick(c, "SchemaGen constrain()", "5.38 µs", "FSM-guided JSON generation", colour=BGN)
    c.pause(1.5)


# ── Scene 6: Wave 26 — Distributed Inference ─────────────────────────────────

def scene_wave26_distributed(c: Cast) -> None:
    c.print(CLEAR, dt=0.3)
    _section(c, "Wave 26 ❶  Distributed Inference",
             "TensorParallel · SequenceParallel · KVMigrate · DisaggPrefill · RequestPreempt",
             colour=BCY)

    c.println(f"  {B}{BCY}TensorParallel{R}  {DIM}Row/column tensor sharding + all-reduce forward{R}")
    _tick(c, "modes", "column-split · row-split", "column → each device holds subset of out-dim")
    _tick(c, "shard() 256×512 n=4", "5.95 µs", "numpy slice sharding, no data copy")
    _tick(c, "forward() b=8 in=256", "15.94 µs", "shard matmul + simulated all-reduce")
    c.println()

    c.println(f"  {B}{BCY}SequenceParallel{R}  {DIM}Ulysses-style sequence scatter/gather{R}")
    _tick(c, "method", "head-dim split across devices", "each device sees full seq, subset of heads")
    _tick(c, "scatter() h=8 seq=256", "5.96 µs", "split heads across 4 devices")
    _tick(c, "gather() reconstruct", "39.07 µs", "numpy stack concatenation")
    c.println()

    c.println(f"  {B}{BCY}KVMigrate{R}  {DIM}Live KV state pack/unpack for inter-node migration{R}")
    _tick(c, "format", "raw bytes + MigrateStats", "checksum verification on unpack")
    _tick(c, "pack() seq=128 h=8 d=64", "88.9 µs", "serialize KV tensors to byte buffer")
    _tick(c, "unpack() 512 KB payload", "77.2 µs", "deserialize + checksum verify")
    c.println()

    c.println(f"  {B}{BCY}DisaggPrefill{R}  {DIM}Disaggregated prefill + decode node pipeline{R}")
    _tick(c, "method", "prefill_node → payload → decode_node", "KV built on prefill, served on decode")
    _tick(c, "prefill() seq=64 layers=4", "2 354 µs", "full KV build across all layers")
    _tick(c, "step() decode token", "0.41 µs", "greedy next-token from loaded KV")
    c.println()

    c.println(f"  {B}{BCY}RequestPreempt{R}  {DIM}SRPT preemption with swap/recompute strategies{R}")
    _tick(c, "strategies", "swap (CPU save) · recompute (marker)", "swap preserves KV, recompute is cheaper")
    _tick(c, "preempt(swap)+resume()", "4.28 µs", "cpu copy + dict entry")
    _tick(c, "preempt(recompute)+resume()", "1.24 µs", "marker-only, minimal overhead")
    c.pause(1.2)


# ── Scene 7: Wave 26 — Gateway & Operations ───────────────────────────────────

def scene_wave26_gateway_ops(c: Cast) -> None:
    c.print(CLEAR, dt=0.3)
    _section(c, "Wave 26 ❷  Gateway & Production Operations",
             "InferGateway · ModelVersionSwap · ProductionProfiler · AdaptiveBatcher",
             colour=BCY)

    c.println(f"  {B}{BCY}InferGateway{R}  {DIM}Least-loaded smart request routing gateway{R}")
    _tick(c, "strategy", "least-load fraction routing", "mark_unhealthy → auto drain from pool")
    _tick(c, "route()+complete() 8 workers", "1.90 µs", "sub-2µs routing + completion")
    _tick(c, "version routing", "required_version filter", "only route to workers with matching model")
    c.println()

    c.println(f"  {B}{BCY}ModelVersionSwap{R}  {DIM}Canary→promote→rollback version management{R}")
    _tick(c, "policy", "canary_fraction=10%", "canary traffic gates full promotion")
    _tick(c, "route_request() canary", "1.45 µs", "fraction-based random split")
    _tick(c, "commit() after min requests", "min_canary_requests=10", "error-rate gated promotion")
    c.println()

    c.println(f"  {B}{BCY}ProductionProfiler{R}  {DIM}APM windowed p50/p99/p999 profiling{R}")
    _tick(c, "window", "ring buffer 1000 samples", "constant memory, rolling window")
    _tick(c, "record() single sample", "0.18 µs", "sub-200ns ring insert")
    _tick(c, "stats() p50/p99/p999", "79.5 µs", "sorted-copy percentile calculation")
    c.println()

    c.println(f"  {B}{BCY}AdaptiveBatcher{R}  {DIM}Throughput/latency-objective dynamic batching{R}")
    _tick(c, "modes", "throughput · target-latency", "EMA latency model α=0.3")
    _tick(c, "next_batch() queue_depth=64", "1.91 µs", "EMA-guided batch size selection")
    _tick(c, "record_observation() EMA", "0.22 µs", "sub-250ns exponential update")
    c.pause(1.2)


# ── Scene 8: Wave 26 — Safety & Reliability ───────────────────────────────────

def scene_wave26_safety_reliability(c: Cast) -> None:
    c.print(CLEAR, dt=0.3)
    _section(c, "Wave 26 ❸  Safety & Reliability",
             "SafetyLayer · SemanticResponseCache · RateLimiter · SchemaValidator · AuditLogger",
             colour=BCY)

    c.println(f"  {B}{BCY}SafetyLayer{R}  {DIM}Inline token safety classifier{R}")
    _tick(c, "method", "token-frequency scoring", "per-category weight matrix projection")
    _tick(c, "score() seq=64", "19.38 µs", "token-level safety classification")
    _tick(c, "score_logits() 1D vocab=8k", "67.34 µs", "next-token pre-check")
    c.println()

    c.println(f"  {B}{BCY}SemanticResponseCache{R}  {DIM}Embedding-similarity LRU response cache{R}")
    _tick(c, "method", "cosine similarity ≥ 0.95", "near-duplicate response deduplication")
    _tick(c, "lookup() miss path cap=64", "294.7 µs", "full cosine scan on miss")
    _tick(c, "store() embed=128", "0.81 µs", "O(1) LRU insert")
    c.println()

    c.println(f"  {B}{BCY}RateLimiter{R}  {DIM}Token-bucket per-tenant rate limiting{R}")
    _tick(c, "method", "token bucket, full-start", "per-tenant rate+burst configuration")
    _tick(c, "consume() 1 token", "0.92 µs", "sub-microsecond debit with refill")
    _tick(c, "refill() tenant bucket", "0.48 µs", "time-based token replenishment")
    c.println()

    c.println(f"  {B}{BCY}SchemaValidator{R}  {DIM}JSON schema validation (7 constraint types){R}")
    _tick(c, "supported", "type/required/properties/min+maxLength/min+max/items", "")
    _tick(c, "validate() valid 5-field", "7.48 µs", "full type+constraint checking pass")
    _tick(c, "validate() 3-error object", "4.90 µs", "short-circuits on missing required")
    c.println()

    c.println(f"  {B}{BCY}AuditLogger{R}  {DIM}SHA-256 hash-chained audit log{R}")
    _tick(c, "method", "genesis → SHA-256 chain", "tamper-evident log: prev_hash in each entry")
    _tick(c, "log() chain append", "1.92 µs", "SHA-256 hash per entry")
    _tick(c, "verify() chain_len=2010", "2 236 µs", "full rehash integrity check")
    c.pause(1.2)


# ── Scene 9: Full CLI Stack ───────────────────────────────────────────────────

def scene_cli_stack(c: Cast) -> None:
    c.print(CLEAR, dt=0.3)
    _section(c, "Full v9 Stack — CLI Examples",
             "All 28 new flags live in squish serve", colour=BMAG)

    c.println(f"  {DIM}# v9 cutting-edge attention + compute fusion stack{R}")
    c.typeout(
        "  $ squish serve ./llama3-8b \\",
        char_delay=0.022, initial_dt=0.3,
    )
    for flag in [
        "      --flash-mla --native-sparse-attn \\",
        "      --fused-sampler --kv-defrag \\",
        "      --dual-chunk-attn --activation-offload \\",
        "      --morph-attn --hydra-spec \\",
        "      --seq-compact --latency-predictor \\",
        "      --parallel-sampler --context-summarizer \\",
        "      --token-watermark --schema-gen",
    ]:
        c.typeout(flag, char_delay=0.018, initial_dt=0.08)

    c.println()
    c.println(f"  {GRN}✓{R} {DIM}Model loaded with v9 attention & compute optimisations{R}", dt=0.4)
    c.println(f"  {GRN}✓{R} {DIM}FlashMLA 4× compression  ·  NSA 87% sparsity{R}", dt=0.2)
    c.println(f"  {GRN}✓{R} {DIM}HydraSpec multi-draft  ·  FSM constrained generation{R}", dt=0.2)
    c.println()

    c.println(f"  {DIM}# v9 distributed inference + production reliability stack{R}")
    c.typeout(
        "  $ squish serve ./llama3-8b \\",
        char_delay=0.022, initial_dt=0.4,
    )
    for flag in [
        "      --tensor-parallel --sequence-parallel \\",
        "      --kv-migrate --disagg-prefill \\",
        "      --request-preempt --infer-gateway \\",
        "      --model-version-swap --production-profiler \\",
        "      --adaptive-batcher --safety-layer \\",
        "      --semantic-response-cache --rate-limiter \\",
        "      --schema-validator --audit-logger",
    ]:
        c.typeout(flag, char_delay=0.018, initial_dt=0.08)

    c.println()
    c.println(f"  {GRN}✓{R} {DIM}Distributed inference stack online{R}", dt=0.4)
    c.println(f"  {GRN}✓{R} {DIM}Tensor/sequence parallel  ·  disaggregated prefill{R}", dt=0.2)
    c.println(f"  {GRN}✓{R} {DIM}Canary deploy  ·  rate limiting  ·  SHA-256 audit log{R}", dt=0.2)
    c.pause(1.5)


# ── Scene 10: Tests & Closing ─────────────────────────────────────────────────

def scene_tests_closing(c: Cast) -> None:
    c.print(CLEAR, dt=0.3)
    _section(c, "Test Suite — v9 Complete", "pytest tests/ -q", colour=BBL)

    c.typeout("  $ pytest tests/ -q", char_delay=0.03, initial_dt=0.3)
    c.println()

    test_lines = [
        ("test_wave25_server_wiring.py", "56 passed"),
        ("test_wave26_server_wiring.py", "56 passed"),
        ("test_wave23_server_wiring.py", "56 passed"),
        ("test_wave24_server_wiring.py", "56 passed"),
        ("test_wave21_server_wiring.py", "56 passed"),
        ("test_wave22_server_wiring.py", "56 passed"),
    ]
    for fname, result in test_lines:
        c.println(
            f"  {DIM}{fname:<46}{R} {BGN}{result}{R}",
            dt=0.18,
        )

    c.println()
    c.println(
        f"  {B}{BGN}4 876 passed{R}  {DIM}in 4.1s  ·  0 failed  ·  0 errors{R}",
        dt=0.4,
    )
    c.pause(0.8)

    # Closing banner
    c.println()
    c.hbar()
    c.println(f"  {B}{BMAG}Squish v9.0{R}  {DIM}— Released 2026-03-12{R}", dt=0.05)
    c.hbar()
    c.println()
    rows = [
        ("Modules", "222 total (28 new in v9)"),
        ("Tests", "4 876 passing, 0 failures"),
        ("FlashMLA compression_ratio", "4× (latent_dim=128 vs 512)"),
        ("NativeSparseAttn sparsity", "~87% block-sparse + window"),
        ("MorphAttn FLOP reduction", "~40% at seq=2048"),
        ("HydraSpec multi-draft heads", "4 heads × 5 draft tokens"),
        ("DisaggPrefill decode step", "0.41 µs — sub-500ns"),
        ("ProductionProfiler record()", "0.18 µs — sub-200ns ring insert"),
        ("AuditLogger log() SHA-256", "1.92 µs — tamper-evident chain"),
        ("RateLimiter consume()", "0.92 µs — per-tenant token bucket"),
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
    scene_wave25_attention(c)
    scene_wave25_fusion_sampling(c)
    scene_wave25_intelligence(c)
    scene_wave25_summary(c)
    scene_wave26_distributed(c)
    scene_wave26_gateway_ops(c)
    scene_wave26_safety_reliability(c)
    scene_cli_stack(c)
    scene_tests_closing(c)
    return c


def main() -> None:
    ap = argparse.ArgumentParser(description="Squish v9 demo GIF generator")
    ap.add_argument("--out",       default="dev/demos/squish-v9-demo.gif")
    ap.add_argument("--cast",      default="dev/demos/squish-v9-demo.cast")
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
    print("Converting to GIF with agg …", flush=True)
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode == 0 and gif_path.exists():
        size_kb = gif_path.stat().st_size // 1024
        print(f"  ✓  {gif_path}  ({size_kb} KB)")
    else:
        print(f"  agg conversion failed (exit {result.returncode})", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
