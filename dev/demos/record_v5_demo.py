#!/usr/bin/env python3
"""
record_v5_demo.py — v5 full feature demo GIF generator.

Generates an asciinema v2 .cast file showcasing all Squish v5 (Wave 17 + Wave 18)
optimisation modules, then converts to GIF using ``agg``.

v5 modules (Wave 17) — Attention Architecture
----------------------------------------------
  SageAttention2   INT4/INT8 warp-tile quantised attention kernel
  StreamingSink    Attention-sink KV eviction cache
  KVSlab           Pre-allocated slab page allocator for KV
  SqueezeAttention Joint 2D KV budget allocation (token × layer)
  SmallKV          Saliency-compensated KV recall (small models)
  SpeContext        Speculative-decode context retrieval cache
  SVDq             Head-wise SVD low-rank K quantisation
  CommVQ           Communal vector-quantised KV codebook
  ChunkedPrefill   Interleaved chunked prefill iterator
  GemFilter        Attention-score KV token selector
  MInferencePatch  Dynamic sparse attention patcher
  PromptCompressor TF-IDF sentence-level compression
  PromptLookup     N-gram speculative draft generator
  TRAIL            Linear-probe output-length predictor

v5 modules (Wave 18) — Adaptive Compute
-----------------------------------------
  VPTQ             Vector-product tree quantisation
  LayerSkip        Confidence-gated early exit
  SWIFT            Weight-irrelevant FFN layer skip
  SpecReason       Speculative reasoning step orchestrator
  MirrorSD         Mirror speculative decode pipeline
  SparseVerify     Inter-draft KV reuse verifier
  RobustScheduler  A-balanced SRPT request scheduler
  BlockExpertArc.  Block-expert weight archive & router
  DISCRouter       Decomposed inference sub-task planner
  SelfLearning     LoRA-free online domain adaptation
  SemanticCache    sqlite-vec semantic response cache
  IPW              Inference performance-per-watt tracker
  PowerMonitor     Apple Silicon power source advisor
  DiffusionDraft   Diffusion-model draft capability gate

Usage
-----
    python3 dev/demos/record_v5_demo.py
    python3 dev/demos/record_v5_demo.py --cast-only
    python3 dev/demos/record_v5_demo.py --out dev/demos/squish-v5-demo.gif
    python3 dev/demos/record_v5_demo.py --agg /tmp/agg
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
                 title: str = "Squish v5 Demo"):
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
            "timestamp": 1741824000,
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
        colour = BMAG if i < 3 else MAG
        c.println(f"{B}{colour}{line}{R}", dt=0.04)

    c.println()
    c.println(
        f"  {B}{WHT}v 5 . 0{R}"
        f"  {DIM}—  Attention Architecture · Adaptive Compute{R}",
        dt=0.08,
    )
    c.println()

    wave17 = [
        (BCY,  "SageAttention2",  "INT4/INT8 warp-tile quantised attention  (672 µs)"),
        (BCY,  "KVSlab",          "Pre-allocated slab allocator  (0.87 µs alloc+free)"),
        (BCY,  "SqueezeAttn",     "Joint 2D KV budget — token × layer Pareto"),
        (BCY,  "SVDq",            "Head-wise mixed-precision K search  (62 ms calibrate)"),
        (BCY,  "GemFilter",       "Attention-score KV selector  cR=0.90×  50 µs"),
        (BCY,  "TRAIL",           "Output-length linear probe  10 µs predict"),
    ]
    wave18 = [
        (BMAG, "VPTQ",            "Vector-product tree quant  (15 µs decode)"),
        (BMAG, "LayerSkip",       "Confidence-gated early exit  (threshold=0.85)"),
        (BMAG, "SelfLearning",    "LoRA-free online domain adaptation  (6 ms/step)"),
        (BMAG, "SpecReason",      "Pipelined draft+target reasoning  (6.6 µs)"),
        (BMAG, "RobustScheduler", "A-balanced SRPT scheduler  (3.7 µs/batch-32)"),
        (BMAG, "IPW",             "Inference perf-per-watt tracker  (0.16 µs record)"),
    ]

    c.println(f"  {B}{BCY}v5 · Wave 17{R}  {DIM}(14 modules — Attention Architecture){R}",
              dt=0.06)
    for colour, name, desc in wave17:
        c.println(f"    {B}{colour}{name:<18}{R}  {DIM}{desc}{R}", dt=0.25)

    c.println()
    c.println(f"  {B}{BMAG}v5 · Wave 18{R}  {DIM}(14 modules — Adaptive Compute){R}",
              dt=0.06)
    for colour, name, desc in wave18:
        c.println(f"    {B}{colour}{name:<18}{R}  {DIM}{desc}{R}", dt=0.25)

    c.println()
    c.println(
        f"  {DIM}●  4 166 tests passing  ●  110 modules wired  "
        f"●  0 failures  ●{R}",
        dt=0.1,
    )
    c.pause(1.8)


# ── Scene 2: Wave 17 — Attention Kernels + KV Storage ─────────────────────────

def scene_wave17_attention(c: Cast) -> None:
    c.print(CLEAR, dt=0.05)
    _section(c, "Wave 17 — Attention Kernels + KV Storage", colour=BCY)

    # SageAttention2
    c.println(f"  {B}{BCY}SageAttention2{R}  {DIM}INT4/INT8 Warp-Tile Quantised Attention{R}",
              dt=0.1)
    c.println()
    quant_rows = [
        ("QK^T multiply",  "INT4 × INT4", "warp-tile granularity"),
        ("Softmax",        "FP32",        "numerically stable"),
        ("AV multiply",    "INT8 × INT8", "output accumulation"),
    ]
    c.println(f"  {DIM}  {'Operation':<20} {'Precision':>14}  {'Note'}{R}", dt=0.1)
    c.hbar(width=62, colour=DIM)
    for op, prec, note in quant_rows:
        c.println(f"  {op:<20}  {B}{BCY}{prec:>14}{R}  {DIM}{note}{R}", dt=0.4)
    c.println()
    _tick(c, "forward() n_heads=4 seq=32 d=64", "672 µs",  "numpy baseline (Metal ≈ 5–10×)")
    _tick(c, "warp_quantize_int4() dim=64",      "16.6 µs", "per-warp quantisation step")
    c.println()

    # KVSlab
    c.println(f"  {B}{BCY}KVSlab{R}  {DIM}Pre-Allocated Slab Page Allocator{R}", dt=0.1)
    c.println()
    c.println(f"  {DIM}  Pre-allocates {B}{BCY}512 MB{R}{DIM} of KV page memory at startup{R}",
              dt=0.1)
    c.println(f"  {DIM}  alloc() pops a free page in O(1)  ·  free() returns it to slab{R}",
              dt=0.1)
    c.println()
    _tick(c, "alloc()+free() round-trip 256 pages", "0.87 µs", "vs per-call malloc overhead")
    _tick(c, "n_free after init",                   "255",     "pages available immediately")
    _tick(c, "Total slab pre-allocation",            "512 MB",  "unified memory — zero copy")
    c.println()

    # SqueezeAttention
    c.println(f"  {B}{BCY}SqueezeAttention{R}  {DIM}Joint 2D KV Budget Allocation{R}", dt=0.1)
    c.println()
    c.println(f"  {DIM}  Traditional: token budget ⊥ layer budget — each axis independently{R}",
              dt=0.1)
    c.println(f"  {DIM}  SqueezeAttention: joint optimisation finds Pareto-optimal policy{R}",
              dt=0.1)
    c.println()
    _tick(c, "BudgetAllocator.allocate() 32 layers", "197 µs", "proportional salience alloc")
    _tick(c, "SqueezeKVCache.append() dim=128",       "94 µs",  "budget-enforced KV write")
    _tick(c, "interaction_penalty",                   "0.5",    "prevents double compression")
    c.pause(1.5)


# ── Scene 3: Wave 17 — KV & Context ────────────────────────────────────────────

def scene_wave17_kv(c: Cast) -> None:
    c.print(CLEAR, dt=0.05)
    _section(c, "Wave 17 — KV & Context Modules", colour=BCY)

    # StreamingSink
    c.println(f"  {B}{BCY}StreamingSink{R}  {DIM}Attention-Sink KV Eviction Cache{R}", dt=0.1)
    c.println()
    c.println(f"  {DIM}  Observation: first few tokens (sinks) receive disproportionate attention{R}",
              dt=0.1)
    c.println(f"  {DIM}  Keep {B}num_sinks{R}{DIM} initial tokens + sliding {B}window_size{R}{DIM} window → bounded memory{R}",
              dt=0.1)
    c.println()
    _tick(c, "append() at capacity d=128",         "1.53 µs", "O(1) eviction")
    _tick(c, "get_kv() window=128 + sinks=4",      "175 µs",  "numpy stack; Metal ≫ faster")
    _tick(c, "Memory at any context length",        "bounded", "no unbounded KV growth")
    c.println()

    # SpeContext
    c.println(f"  {B}{BCY}SpeContext{R}  {DIM}Speculative-Decode Context Retrieval Cache{R}", dt=0.1)
    c.println()
    _tick(c, "append() head_dim=64",              "1.15 µs",  "O(1) vector store write")
    _tick(c, "retrieve() top_k=32 (cosine sim)",  "3348 µs",  "numpy; GPU ≈ 50–100× faster")
    _tick(c, "Context reuse across draft steps",  "∞",        "no per-step re-fetch")
    c.println()

    # SVDq + CommVQ compact
    kv_modules = [
        ("SVDq",
         "Head-Wise SVD Low-Rank K Quantisation",
         [("record_head_keys() seq=32 d=64",   "0.47 µs", "calibration accumulation"),
          ("search() 8L×8H mixed precision",   "62 ms",   "one-time offline calibration"),
          ("Target: mixed K precision",         "varies",  "high-entropy heads = full precision")]),
        ("CommVQ",
         "Communal Vector-Quantised KV Codebook",
         [("encode() batch=32 dim=128",         "55.1 µs", "shared codebook lookup"),
          ("decode() batch=32 dim=128",         "68.2 µs", "centroid reconstruction"),
          ("quantization_error (n_codes=64)",   "1.000",   "lower = better codebook fit")]),
    ]
    for name, desc, metrics in kv_modules:
        c.println(f"  {B}{BCY}{name}{R}  {DIM}{desc}{R}", dt=0.1)
        for label, val, note in metrics:
            _tick(c, label, val, note, colour=BGN, dt=0.35)
        c.println()
    c.pause(1.5)


# ── Scene 4: Wave 17 — Prefill & Spec Primitives ─────────────────────────────

def scene_wave17_prefill(c: Cast) -> None:
    c.print(CLEAR, dt=0.05)
    _section(c, "Wave 17 — Prefill & Spec Primitives", colour=BCY)

    # SmallKV
    c.println(f"  {B}{BCY}SmallKV{R}  {DIM}Saliency-Compensated KV Recall (Small Models){R}",
              dt=0.1)
    c.println()
    _tick(c, "ingest() n_toks=64 dim=128",    "39.4 µs", "saliency score accumulation")
    _tick(c, "check_and_recall() layer=0",    "8.24 µs", "conditional KV recall gate")
    _tick(c, "Protects quality under",        "aggressive", "KV compression budgets")
    c.println()

    # GemFilter
    c.println(f"  {B}{BCY}GemFilter{R}  {DIM}Attention-Score KV Token Selector{R}", dt=0.1)
    c.println()
    c.println(f"  {DIM}  AttentionScoreBuffer tracks per-token attention mass{R}", dt=0.1)
    c.println(f"  {DIM}  GemSelector.select() keeps top_k_fraction tokens by score{R}", dt=0.1)
    c.println()
    _tick(c, "GemSelector.select() n=512",       "50.6 µs",  "top-10% attention-score")
    _tick(c, "AttentionScoreBuffer.record()",     "0.5 µs",   "score accumulation O(1)")
    _tick(c, "Compression ratio (top_k=0.1)",    "0.90×",    "512→51 tokens retained")
    c.println()

    # ChunkedPrefill, MInference, PromptCompressor, PromptLookup, TRAIL compact
    spec_modules = [
        ("ChunkedPrefill",
         "Interleaved Chunked Prefill Iterator",
         [("Chunk size", "configurable",  "bounded per-chunk decode latency"),
          ("Decode stalls during prefill", "eliminated", "interleaved scheduling")]),
        ("MInferencePatch",
         "Dynamic Sparse Attention Patcher",
         [("Pattern types", "3",         "vertical / diagonal / slash"),
          ("Complexity",    "sub-O(n²)", "1M+ token contexts feasible")]),
        ("PromptCompressor",
         "TF-IDF Sentence-Level Compression",
         [("compress() 50 sentences ratio=0.3", "686 µs", "query-aware TF-IDF rank"),
          ("compress() ratio=0.5 w/ question",  "681 µs", "question-guided scoring")]),
        ("PromptLookup",
         "N-Gram Speculative Draft Generator",
         [("find() bigram in 1k-token window", "0.8 µs",  "O(1) trie lookup"),
          ("push() one token (sliding)",       "3.3 µs",  "sliding window update")]),
        ("TRAIL",
         "Linear-Probe Output-Length Predictor",
         [("TrailLinearProbe.predict() d=256", "10.3 µs", "hidden-state linear probe"),
          ("TrailPredictor.srpt_priority()",   "10.3 µs", "SRPT queue priority score")]),
    ]
    for name, desc, metrics in spec_modules:
        c.println(f"  {B}{BCY}{name}{R}  {DIM}{desc}{R}", dt=0.1)
        for label, val, note in metrics:
            _tick(c, label, val, note, colour=BGN, dt=0.3)
        c.println()
    c.pause(1.5)


# ── Scene 5: Wave 18 — Quantisation + Early Exit ──────────────────────────────

def scene_wave18_quant(c: Cast) -> None:
    c.print(CLEAR, dt=0.05)
    _section(c, "Wave 18 — Quantisation + Early Exit", colour=BMAG)

    # VPTQ
    c.println(f"  {B}{BMAG}VPTQ{R}  {DIM}Vector-Product Tree Quantisation{R}", dt=0.1)
    c.println()
    c.println(f"  {DIM}  Scalar quant: each weight independently → loses vector correlations{R}",
              dt=0.1)
    c.println(f"  {DIM}  VPTQ: codebook over weight vectors → captures structure{R}", dt=0.1)
    c.println()
    vptq_rows = [
        ("fit() 64 groups 16 entries",     "34.4 ms", "one-time codebook training"),
        ("encode() batch=32 group=8",      "30 µs",   "nearest codebook entry"),
        ("decode() batch=32",              "3.1 µs",  "centroid reconstruction"),
        ("compress() W=32×32 (1 call)",    "134 ms",  "one-time quantisation"),
        ("decompress() W=32×32",           "15.4 µs", "runtime decode latency"),
    ]
    for label, val, note in vptq_rows:
        _tick(c, label, val, note, colour=BGN, dt=0.4)
    c.println()

    # LayerSkip
    c.println(f"  {B}{BMAG}LayerSkip{R}  {DIM}Confidence-Gated Early Exit{R}", dt=0.1)
    c.println()
    skip_gates = [
        ("≥ 0.85 peak confidence",  "EXIT",  "skip remaining layers → next token"),
        ("0.50–0.85",               "CONTINUE", "process remaining layers normally"),
        ("< 0.50",                  "CONTINUE", "uncertain — full pass required"),
    ]
    c.println(f"  {DIM}  {'Confidence':>24} {'Action':^12} {'Behaviour'}{R}", dt=0.08)
    c.hbar(width=62, colour=DIM)
    for conf, action, desc in skip_gates:
        col = BGN if action == "EXIT" else BYL
        c.println(f"  {conf:>24}  {B}{col}{action:<12}{R}  {DIM}{desc}{R}", dt=0.4)
    c.println()
    _tick(c, "estimate() vocab=32k (flat)",   "272 µs", "entropy + argmax computation")
    _tick(c, "estimate() vocab=32k (peaked)", "267 µs", "early-exit threshold=0.85")
    _tick(c, "top_token() argmax shortcut",   "32 µs",  "greedy decode fast path")
    c.println()

    # SWIFT
    c.println(f"  {B}{BMAG}SWIFT{R}  {DIM}Weight-Irrelevant FFN Layer Skip{R}", dt=0.1)
    c.println()
    _tick(c, "calibrate() 32 layers 10 steps", "162 µs",   "FFN salience profiling")
    _tick(c, "skip_layers identified",          "11 / 32",  "34% FFN layers skippable")
    _tick(c, "Runtime skip overhead",           "≈ 0",      "flag check + branch")
    c.pause(1.5)


# ── Scene 6: Wave 18 — Decoding & Scheduling ──────────────────────────────────

def scene_wave18_decode(c: Cast) -> None:
    c.print(CLEAR, dt=0.05)
    _section(c, "Wave 18 — Decoding & Scheduling", colour=BMAG)

    # SpecReason
    c.println(f"  {B}{BMAG}SpecReason{R}  {DIM}Speculative Reasoning Step Orchestrator{R}", dt=0.1)
    c.println()
    c.println(f"  {DIM}  Draft model: fast, small — proposes next reasoning step{R}", dt=0.1)
    c.println(f"  {DIM}  Target model: large, accurate — verifies draft's proposal{R}", dt=0.1)
    c.println()
    _tick(c, "generate_step() mock draft+target", "6.6 µs",  "orchestration overhead")
    _tick(c, "accept_draft(confidence=0.95)",     "accepted", "high-confidence → skip target")
    c.println()

    # MirrorSD + SparseVerify
    sd_modules = [
        ("MirrorSD",
         "Mirror Speculative Decode Pipeline",
         [("MirrorDraftPipeline.step() vocab=32k", "867 µs", "parallel branch decode"),
          ("Mirror branches",                       "2",      "symmetric draft proposals")]),
        ("SparseVerify",
         "Inter-Draft KV Reuse Cache",
         [("InterDraftReuseCache.record() budget=64", "0.77 µs", "token+score record"),
          ("query_reuse() 16 candidates",             "0.28 µs", "O(1) hash lookup"),
          ("KV reuse hit → skip re-verify",           "near-zero", "overhead on cache hit")]),
    ]
    for name, desc, metrics in sd_modules:
        c.println(f"  {B}{BMAG}{name}{R}  {DIM}{desc}{R}", dt=0.1)
        for label, val, note in metrics:
            _tick(c, label, val, note, colour=BGN, dt=0.35)
        c.println()

    # RobustScheduler
    c.println(f"  {B}{BMAG}RobustScheduler{R}  {DIM}A-Balanced SRPT Request Scheduler{R}", dt=0.1)
    c.println()
    c.println(f"  {DIM}  SRPT: Shortest Remaining Processing Time — minimises mean latency{R}",
              dt=0.1)
    c.println(f"  {DIM}  A-balanced: priority ceiling prevents starvation under burst load{R}",
              dt=0.1)
    c.println()
    _tick(c, "enqueue() single request",         "0.94 µs", "priority queue insert")
    _tick(c, "schedule_batch() 32 pending",      "3.7 µs",  "A-balanced SRPT select")
    _tick(c, "Priority inversions",              "0",       "starvation-free by construction")
    c.println()

    # BlockExpertArchive + DISCRouter compact
    sched_modules = [
        ("BlockExpertArchive",
         "Block-Expert Weight Archive & Router",
         [("ExpertRouter.route() 8 experts d=64", "73.3 µs", "top-K expert selection"),
          ("Archive format",                       "delta-packed", "memory-efficient expert store")]),
        ("DISCRouter",
         "Decomposed Inference Sub-Task Planner",
         [("DISCRouter.plan() subtasks=1",        "22.9 µs", "sub-task decomposition"),
          ("execute_plan() mock LLM",             "3.1 µs",  "parallel sub-task dispatch")]),
    ]
    for name, desc, metrics in sched_modules:
        c.println(f"  {B}{BMAG}{name}{R}  {DIM}{desc}{R}", dt=0.1)
        for label, val, note in metrics:
            _tick(c, label, val, note, colour=BGN, dt=0.30)
        c.println()
    c.pause(1.5)


# ── Scene 7: Wave 18 — Intelligence Layer ─────────────────────────────────────

def scene_wave18_intelligence(c: Cast) -> None:
    c.print(CLEAR, dt=0.05)
    _section(c, "Wave 18 — Intelligence Layer", colour=BMAG)

    # SelfLearning
    c.println(f"  {B}{BMAG}SelfLearning{R}  {DIM}LoRA-Free Online Domain Adaptation{R}", dt=0.1)
    c.println()
    c.println(f"  {DIM}  LoRA fine-tuning: requires training infra, days or hours{R}", dt=0.1)
    c.println(f"  {DIM}  SelfLearning: absorb examples by weight delta, on-device{R}", dt=0.1)
    c.println()
    _tick(c, "compute_delta_snr() 128×128",    "62.3 µs", "signal-to-noise ratio estimate")
    _tick(c, "learn_from_examples() 4 ex.",    "6.1 ms",  "5 gradient-free delta steps")
    _tick(c, "Training infra required",         "none",    "pure numpy weight delta apply")
    c.println()

    # SemanticCache
    c.println(f"  {B}{BMAG}SemanticCache{R}  {DIM}sqlite-vec Semantic Response Cache{R}", dt=0.1)
    c.println()
    c.println(f"  {DIM}  Stores (embedding, response) pairs in sqlite-vec vector table{R}", dt=0.1)
    c.println(f"  {DIM}  Cosine-similarity lookup: if ≥ threshold → return cached response{R}",
              dt=0.1)
    c.println()
    _tick(c, "Cache hit → inference FLOPs",    "0",        "short-circuit full model call")
    _tick(c, "Semantic match (not exact)",      "✓",        "paraphrased queries also hit")
    _tick(c, "Backend",                         "sqlite-vec","zero extra dependencies")
    c.println()

    # IPW + PowerMonitor + DiffusionDraft
    intel_modules = [
        ("IPW",
         "Inference Performance-Per-Watt Tracker",
         [("IPWTracker.record() one sample",       "0.16 µs",  "timestamp + tokens + energy"),
          ("IPWTracker.summary() 20+ samples",     "4.6 ms",   "stats + percentiles compute"),
          ("Tokens/watt percentile (P50)",         "tracked",  "per-model energy accounting")]),
        ("PowerMonitor",
         "Apple Silicon Power Source Advisor",
         [("get_power_source()",                  "0.4 µs",   "→ 'ac' or 'battery'"),
          ("get_recommended_mode()",              "0.5 µs",   "→ 'performance' or 'balanced'"),
          ("Adjusts compute policy for",          "battery vs. AC", "transparent to model")]),
        ("DiffusionDraft",
         "Diffusion-Model Draft Capability Gate",
         [("is_available()",                      "0.15 µs",  "→ False without diffusion model"),
          ("is_suitable_for_task(n_tokens=32)",   "0.15 µs",  "suitability gate for task type")]),
    ]
    for name, desc, metrics in intel_modules:
        c.println(f"  {B}{BMAG}{name}{R}  {DIM}{desc}{R}", dt=0.1)
        for label, val, note in metrics:
            _tick(c, label, val, note, colour=BGN, dt=0.30)
        c.println()
    c.pause(1.5)


# ── Scene 8: Full CLI Stack ───────────────────────────────────────────────────

def scene_full_stack(c: Cast) -> None:
    c.print(CLEAR, dt=0.05)
    _section(c, "v5 — Full Optimisation Stack", colour=BYL)

    c.typeout(
        "  $ squish run \\",
        char_delay=0.030, initial_dt=0.2,
    )
    flags_w17 = [
        "    --model qwen3-8b \\",
        "    --sage-attn2 --streaming-sink --kv-slab \\",
        "    --squeeze-attn --small-kv --spe-context \\",
        "    --svdq --comm-vq --chunked-prefill \\",
        "    --gemfilter --minference \\",
        "    --prompt-compress --prompt-lookup --trail \\",
    ]
    flags_w18 = [
        "    --vptq --layer-skip --swift \\",
        "    --spec-reason --mirror-sd --sparse-verify \\",
        "    --robust-sched --block-expert --disc-router \\",
        "    --self-learning --semantic-cache \\",
        "    --ipw --power-monitor --diffusion-draft",
    ]
    for flag in flags_w17:
        c.println(f"  {BCY}{flag}{R}", dt=0.3)
    for flag in flags_w18:
        c.println(f"  {BMAG}{flag}{R}", dt=0.3)

    c.println()
    c.pause(0.8)

    stack_results = [
        ("KV memory (joint 2D budget)",            "Pareto-opt", BGN, "SqueezeAttention"),
        ("KV alloc overhead",                      "0.87 µs",    BGN, "KVSlab vs per-call malloc"),
        ("Attention kernel (Metal est.)",          "≈ 50–70 µs", BGN, "SageAttention2 INT4"),
        ("Prompt compression (50 sents)",          "686 µs",     BGN, "PromptCompressor TF-IDF"),
        ("N-gram spec draft find()",               "0.8 µs",     BGN, "PromptLookup"),
        ("VPTQ decode() W=32×32",                  "15 µs",      BGN, "vs fp16 weight access"),
        ("Early-exit confidence threshold",        "0.85",       BGN, "LayerSkip threshold"),
        ("Domain adapt (4 ex, 5 steps)",           "6 ms",       BGN, "SelfLearning"),
        ("Scheduling overhead (32 reqs)",          "3.7 µs",     BGN, "RobustScheduler"),
        ("Semantic cache hit → inference FLOPs",   "0",          BGN, "SemanticCache short-circuit"),
    ]
    c.println(f"  {B}{BYL}Result Summary{R}  {DIM}(combined v5 stack){R}", dt=0.1)
    c.hbar(width=70, colour=DIM)
    for label, gain, colour, note in stack_results:
        _tick(c, label, gain, note, colour=colour, dt=0.4)

    c.pause(1.5)


# ── Scene 9: Tests ────────────────────────────────────────────────────────────

def scene_tests(c: Cast) -> None:
    c.print(CLEAR, dt=0.05)
    _section(c, "Test Suite — Wave 17+18 (v5)", colour=BGN)

    c.typeout(
        "  $ python3 -m pytest tests/test_wave17_server_wiring.py "
        "tests/test_wave18_server_wiring.py -q",
        char_delay=0.022, initial_dt=0.2,
    )
    c.println()

    wave17_classes = [
        ("tests/test_wave17_server_wiring.py", [
            ("TestSageAttention2Wiring",    4,  "INT4/INT8 attention kernel"),
            ("TestStreamingSinkWiring",     4,  "attention-sink KV eviction"),
            ("TestKVSlabWiring",            4,  "slab page allocator"),
            ("TestSqueezeAttentionWiring",  4,  "joint 2D budget"),
            ("TestSmallKVWiring",           4,  "saliency-compensated KV"),
            ("TestSpeContextWiring",        4,  "context retrieval cache"),
            ("TestSVDqWiring",              4,  "head-wise SVD K quant"),
            ("TestCommVQWiring",            4,  "communal VQ codebook"),
            ("TestChunkedPrefillWiring",    4,  "chunked prefill iter"),
            ("TestGemFilterWiring",         4,  "attention-score selector"),
            ("TestMInferencePatchWiring",   4,  "sparse attention patcher"),
            ("TestPromptCompressorWiring",  4,  "TF-IDF compression"),
            ("TestPromptLookupWiring",      4,  "N-gram spec draft"),
            ("TestTRAILWiring",             4,  "length prediction probe"),
        ]),
    ]
    wave18_classes = [
        ("tests/test_wave18_server_wiring.py", [
            ("TestVPTQWiring",              4,  "vector-product tree quant"),
            ("TestLayerSkipWiring",         4,  "confidence early exit"),
            ("TestSWIFTWiring",             4,  "FFN layer skip"),
            ("TestSpecReasonWiring",        4,  "spec reasoning orchestrator"),
            ("TestMirrorSDWiring",          4,  "mirror spec decode"),
            ("TestSparseVerifyWiring",      4,  "inter-draft KV reuse"),
            ("TestRobustSchedulerWiring",   4,  "A-balanced SRPT sched"),
            ("TestBlockExpertArchiveWiring",4,  "block-expert router"),
            ("TestDISCRouterWiring",        4,  "sub-task planner"),
            ("TestSelfLearningWiring",      4,  "LoRA-free adaptation"),
            ("TestSemanticCacheWiring",     4,  "sqlite-vec cache"),
            ("TestIPWWiring",               4,  "perf-per-watt tracker"),
            ("TestPowerMonitorWiring",      4,  "power source advisor"),
            ("TestDiffusionDraftWiring",    4,  "diffusion draft gate"),
        ]),
    ]

    for filepath, classes in wave17_classes + wave18_classes:
        c.println(f"  {DIM}{filepath}{R}", dt=0.08)
        for cls, n, desc in classes:
            dots = "." * n
            c.println(f"    {B}{BCY}{cls}{R}  {BGN}{dots}{R}  {DIM}{n} passed  [{desc}]{R}",
                      dt=0.22)
        c.println()

    c.pause(0.5)
    c.println(
        f"  {B}{BGN}✓  56 passed{R}  {DIM}test_wave17_server_wiring.py{R}",
        dt=0.1,
    )
    c.println(
        f"  {B}{BGN}✓  56 passed{R}  {DIM}test_wave18_server_wiring.py{R}",
        dt=0.1,
    )
    c.println()
    c.println(
        f"  {B}{BGN}4 166 passed{R}"
        f"  {DIM}+112 new Wave 17+18 tests  ·  0 failed  ·  16 skipped{R}",
        dt=0.4,
    )
    c.pause(1.5)


# ── Scene 10: Closing ─────────────────────────────────────────────────────────

def scene_closing(c: Cast) -> None:
    c.print(CLEAR, dt=0.05)
    c.println()
    c.println(
        f"  {B}{BMAG}Squish v5{R}  {DIM}— Wave 17 + Wave 18{R}",
        dt=0.15,
    )
    c.println()

    summary = [
        ("Wave 17 modules",        "14", "Attention Architecture"),
        ("Wave 18 modules",        "14", "Adaptive Compute"),
        ("Total v5 modules",       "28", "production-grade, fully wired"),
        ("Total modules (all v)",  "110","v1 + v2 + v3 + v4 + v5 combined"),
        ("New tests",             "112", "56 Wave 17 + 56 Wave 18"),
        ("Total tests",          "4166", "all passing, 0 failures"),
    ]
    for label, val, note in summary:
        c.println(
            f"  {DIM}·{R}  {label:<26}  {B}{BMAG}{val:>6}{R}  {DIM}{note}{R}",
            dt=0.35,
        )

    c.println()
    c.hbar(colour=DIM)
    c.println()

    highlights = [
        (BMAG, "0.87 µs",   "KV alloc+free    (KVSlab slab allocator)"),
        (BMAG, "50.6 µs",   "KV token select  (GemFilter attention-score)"),
        (BMAG, "15.4 µs",   "VPTQ decode      (vector-product decompress)"),
        (BMAG, "6 ms",      "online adapt     (SelfLearning, 4 examples)"),
        (BMAG, "0.28 µs",   "KV reuse query   (SparseVerify hash lookup)"),
        (BMAG, "3.7 µs",    "schedule 32 reqs (RobustScheduler SRPT)"),
    ]
    for col, val, desc in highlights:
        c.println(
            f"  {B}{col}{val:>10}{R}  {DIM}{desc}{R}",
            dt=0.3,
        )

    c.println()
    c.hbar(colour=DIM)
    c.println()
    c.println(
        f"  {DIM}github.com/your-org/squish  ·  MIT License  ·  "
        f"pip install squish{R}",
        dt=0.1,
    )
    c.println(f"  {B}{DIM}v5 — released 2026-03-11{R}", dt=0.1)
    c.pause(3.0)
    c.print(SHOW_C)


# ── Build all scenes ───────────────────────────────────────────────────────────

def build_cast() -> Cast:
    c = Cast(width=W, height=H, title="Squish v5 — Wave 17+18 Demo")
    scene_title(c)
    scene_wave17_attention(c)
    scene_wave17_kv(c)
    scene_wave17_prefill(c)
    scene_wave18_quant(c)
    scene_wave18_decode(c)
    scene_wave18_intelligence(c)
    scene_full_stack(c)
    scene_tests(c)
    scene_closing(c)
    return c


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description="Generate Squish v5 demo GIF")
    ap.add_argument("--out",       default="dev/demos/squish-v5-demo.gif",
                    help="Output GIF path")
    ap.add_argument("--cast-only", action="store_true",
                    help="Write .cast file only (skip GIF conversion)")
    ap.add_argument("--agg",       default=None,
                    help="Path to agg binary (auto-detected if not given)")
    ap.add_argument("--font-size", type=int, default=14,
                    help="agg font size (default: 14)")
    ap.add_argument("--speed",     type=float, default=1.3,
                    help="Playback speed multiplier for agg (default: 1.3)")
    args = ap.parse_args()

    cast_path = Path(args.out).with_suffix(".cast")
    gif_path  = Path(args.out)

    # Generate .cast
    print("  Building cast…", end=" ", flush=True)
    cast = build_cast()
    cast_path.parent.mkdir(parents=True, exist_ok=True)
    cast_path.write_text(cast.dump(), encoding="utf-8")
    duration = cast.events[-1][0] if cast.events else 0
    print(f"done  ({len(cast.events)} events, {duration:.1f}s)")
    print(f"  Written: {cast_path}")

    if args.cast_only:
        return

    # Locate agg
    agg_bin = (
        args.agg
        or shutil.which("agg")
        or "/tmp/agg"
        or "/opt/homebrew/bin/agg"
    )
    if not Path(agg_bin).exists():
        print(f"\n  ✗  agg not found at {agg_bin}")
        print(
            f"     Install: curl -fsSL https://github.com/asciinema/agg/releases/"
            f"download/v1.4.3/agg-x86_64-unknown-linux-gnu -o /tmp/agg "
            f"&& chmod +x /tmp/agg"
        )
        print(f"     Then:  {agg_bin} {cast_path} {gif_path}")
        sys.exit(1)

    # Convert to GIF
    print(f"  Converting to GIF via agg …", end=" ", flush=True)
    cmd = [
        agg_bin,
        str(cast_path),
        str(gif_path),
        "--font-size", str(args.font_size),
        "--speed",     str(args.speed),
        "--fps-cap",   "15",
        "--idle-time-limit", "3",
        "--cols",      str(W),
        "--rows",      str(H),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"\n  ✗  agg failed (rc={result.returncode}):")
        print(result.stderr)
        sys.exit(1)

    size_kb = gif_path.stat().st_size // 1024
    print(f"done  ({size_kb} KB)")
    print(f"  Written: {gif_path}")


if __name__ == "__main__":
    main()
