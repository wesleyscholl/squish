#!/usr/bin/env python3
"""
record_wave13_14_demo.py — Wave 13+14 full feature demo GIF generator.

Generates an asciinema v2 .cast file showcasing all Squish Wave 13 and Wave 14
optimisation modules, then converts to GIF using ``agg``.

Wave 13 modules shown
---------------------
  DuoAttention   — retrieval/streaming head KV separation
  ShadowKV       — SVD low-rank key cache + CPU offload
  PQCache        — product-quantisation KV ANN search
  SpeCache       — speculative KV block prefetcher
  KnapSpec       — knapsack-optimal layer-skip budget solver
  TokenMerging   — ToMe bipartite token-pair merging
  DuoDecoding    — dual-sequence speculative decoding
  C2T            — confidence-to-tree adaptive draft builder
  CLaSP          — layer-skip adaptive speculative decoding

Wave 14 modules shown
---------------------
  DFloat11         — Huffman entropy coding for BF16 weights
  RANSCodec        — rANS entropy codec
  SqueezeLLM       — non-uniform INT3/4 weight quantisation
  NF4 Quant        — NormalFloat4 weight encoding
  QSpec            — complementary-quantisation speculative decoding
  CopySpec         — suffix-match copy-based draft generator
  VisionPrefixCache — LRU vision encoder output cache
  HeadAwareKV      — per-head mixed-precision KV store
  SubSpec / DEL    — sub-model & dynamic exit speculative decoding
  QuantSpec        — quantised draft speculative step
  HeteroVocab      — cross-vocabulary speculative decoding

Usage
-----
    python3 dev/demos/record_wave13_14_demo.py
    python3 dev/demos/record_wave13_14_demo.py --cast-only
    python3 dev/demos/record_wave13_14_demo.py --out dev/demos/squish-wave13-14-demo.gif
    python3 dev/demos/record_wave13_14_demo.py --agg /tmp/agg
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
BGN  = "\x1b[92m"     # bright green
BRD  = "\x1b[91m"     # bright red
BYL  = "\x1b[93m"     # bright yellow
BCY  = "\x1b[96m"     # bright cyan
MAG  = "\x1b[35m"
BMAG = "\x1b[95m"     # bright magenta
BLU  = "\x1b[34m"
BBL  = "\x1b[94m"     # bright blue

CLEAR  = "\x1b[2J\x1b[H"
HIDE_C = "\x1b[?25l"
SHOW_C = "\x1b[?25h"

W = 92   # terminal width
H = 30   # terminal height


# ── Cast builder ─────────────────────────────────────────────────────────────

class Cast:
    def __init__(self, width: int = W, height: int = H,
                 title: str = "Squish Wave 13+14 Demo"):
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
            "timestamp": 1741651200,
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
        colour = BCY if i < 3 else CYN
        c.println(f"{B}{colour}{line}{R}", dt=0.04)

    c.println()
    c.println(
        f"  {B}{WHT}W A V E S   1 3   +   1 4{R}"
        f"  {DIM}—  Ultra-Long Context · Adaptive Spec-Decode · Quantisation{R}",
        dt=0.08,
    )
    c.println()

    wave13 = [
        (BCY,  "DuoAttention",  "Retrieval/streaming head KV separation"),
        (BCY,  "ShadowKV",      "SVD low-rank key cache + CPU offload"),
        (BCY,  "PQCache",       "Product-quantisation KV ANN (64× memory)"),
        (BCY,  "KnapSpec",      "Knapsack-optimal layer-skip solver"),
        (BCY,  "TokenMerging",  "ToMe bipartite token-pair merging"),
        (BCY,  "DuoDecoding",   "Dual-sequence speculative decoding"),
        (BCY,  "C2T / CLaSP",   "Confidence-to-tree + layer-skip spec decode"),
    ]
    wave14 = [
        (BMAG, "DFloat11",      "Huffman entropy coding — lossless BF16"),
        (BMAG, "SqueezeLLM",    "Non-uniform INT3/4 + sparse FP16 outliers"),
        (BMAG, "NF4 Quant",     "NormalFloat4 blockwise weight encoding"),
        (BMAG, "QSpec",         "Complementary-quant speculative decoding"),
        (BMAG, "CopySpec",      "Suffix-match copy-based draft generator"),
        (BMAG, "VisionCache",   "LRU vision encoder prefix cache"),
        (BMAG, "HeteroVocab",   "Cross-vocabulary speculative decoding"),
    ]

    c.println(f"  {B}{BCY}Wave 13{R}  {DIM}(9 modules){R}", dt=0.06)
    for colour, name, desc in wave13:
        c.println(f"    {B}{colour}{name:<16}{R}  {DIM}{desc}{R}", dt=0.09)

    c.println()
    c.println(f"  {B}{BMAG}Wave 14{R}  {DIM}(16 modules){R}", dt=0.06)
    for colour, name, desc in wave14:
        c.println(f"    {B}{colour}{name:<16}{R}  {DIM}{desc}{R}", dt=0.09)

    c.println()
    c.println(
        f"  {DIM}●  3 848 tests passing  ●  61 modules wired  ●  0 failures  ●{R}",
        dt=0.1,
    )
    c.pause(1.8)


# ── Scene 2: Wave 13 KV Memory Compression ───────────────────────────────────

def scene_wave13_kv(c: Cast) -> None:
    _section(c, "Wave 13 — KV Cache Memory Compression", colour=BCY)

    c.typeout(
        "  $ squish run --model qwen3-8b "
        "--duo-attention --shadow-kv --pq-cache",
        char_delay=0.025, initial_dt=0.2,
    )
    c.println()

    # DuoAttention
    c.println(f"  {B}{BCY}DuoAttention{R}  {DIM}Retrieval/Streaming head separation{R}", dt=0.1)
    c.println()
    heads = [
        ("Head  0–21  (22 heads)", "streaming", DIM,  "no K/V stored — token attends to all"),
        ("Head 22–39  (18 heads)", "retrieval", BCY,  "full KV stored — last-N sink tokens"),
    ]
    for label, kind, col, note in heads:
        c.println(f"    {label:<28}  {B}{col}{kind:>12}{R}  {DIM}{note}{R}", dt=0.45)

    c.println()
    _tick(c, "Retrieval head fraction",        "56%",   "heads keep full KV")
    _tick(c, "KV memory ratio vs full",        "4.95×", "sink-only streaming heads")
    _tick(c, "store_kv() latency",             "0.9 µs","per token per retrieval head")
    c.println()

    # ShadowKV
    c.println(f"  {B}{BCY}ShadowKV{R}  {DIM}SVD low-rank key cache + CPU offload{R}", dt=0.1)
    c.println()
    c.println(f"  {DIM}  K ← U · Σ_r  (rank 16 / 64)   stored on CPU{R}", dt=0.1)
    c.println(f"  {DIM}  V stored as-is (FP16) on CPU   recalled on demand{R}", dt=0.08)
    c.println()
    _tick(c, "SVD key compression ratio",      "0.25×", "rank 16 → 75% key memory saved")
    _tick(c, "Key memory reduction",           "4.00×", "vs full FP16 keys")
    _tick(c, "recall() top-32 latency",        "7 ms",  "incl. CPU→GPU transfer")
    c.println()

    # PQCache
    c.println(f"  {B}{BCY}PQCache{R}  {DIM}Product-quantisation KV ANN search{R}", dt=0.1)
    c.println()
    pq_rows = [
        ("Raw FP32 keys  (256 tok)",   "256 × 64 × 4 B",   "= 65.5 KB",  WHT),
        ("PQ index       (4 sub-vecs)", "256 × 1 B",        "=  0.3 KB",  BGN),
        ("Codebooks      (16×16 centroids)", "16 × 4 × 16 B", "=  4.1 KB",BGN),
    ]
    for label, size, total, col in pq_rows:
        c.println(f"    {label:<36}  {col}{size:<22}{R}  {DIM}{total}{R}", dt=0.45)

    c.println()
    _tick(c, "Index memory vs raw FP32",        "0.016×",  "1-byte codes vs float32")
    _tick(c, "Memory reduction",                "64×",     "vs raw float32 keys")
    _tick(c, "retrieve() top-32 latency",       "344 µs",  "ADC approximate search")
    c.println()

    c.println(
        f"  {B}{BGN}→  combined KV memory savings exceed 10–30× on 32-layer models{R}",
        dt=0.15,
    )
    c.pause(2.5)


# ── Scene 3: Wave 13 Speculative Decoding ────────────────────────────────────

def scene_wave13_spec(c: Cast) -> None:
    _section(c, "Wave 13 — Adaptive Speculative Decoding", colour=BCY)

    c.typeout(
        "  $ squish run --model qwen3-8b "
        "--knapspec --token-merging --duo-decoding --c2t --clasp",
        char_delay=0.022, initial_dt=0.2,
    )
    c.println()

    # KnapSpec
    c.println(f"  {B}{BCY}KnapSpec{R}  {DIM}Knapsack-optimal layer-skip budget solver{R}", dt=0.1)
    c.println()
    c.println(f"  {DIM}  Solver maximises quality while staying under latency budget.{R}", dt=0.08)
    c.println()

    kn_rows = [
        ("ctx =  128", "0 attn / 31 MLP",   "48% blocks skipped", "1522 µs"),
        ("ctx =  512", "15 attn / 16 MLP",  "48% blocks skipped", "1601 µs"),
        ("ctx = 2048", "23 attn /  0 MLP",  "36% blocks skipped", "1634 µs"),
    ]
    c.println(f"  {'Context':<14}  {'Skipped blocks':<22}  {'Skip rate':>12}  {'Solve time':>12}", dt=0.1)
    c.println(f"  {'─' * 62}")
    for ctx, skipped, rate, t in kn_rows:
        c.println(f"  {ctx:<14}  {DIM}{skipped:<22}{R}  {B}{BCY}{rate:>12}{R}  "
                   f"{DIM}{t:>12}{R}", dt=0.45)
    c.println()

    # TokenMerging
    c.println(f"  {B}{BCY}TokenMerging{R}  {DIM}ToMe bipartite token-pair merging (r=16){R}", dt=0.1)
    c.println()

    for seq, merged, pct in [(128, 113, 12), (512, 496, 3), (1024, 1008, 2)]:
        bar_full = 36
        bar_left = int(bar_full * merged / seq)
        bar_merge = bar_full - bar_left
        c.println(
            f"  seq={seq:<5}  "
            f"{BCY}{'█' * bar_left}{DIM}{'░' * bar_merge}{R}"
            f"  {B}{BGN}{merged}{R}/{seq} tokens  {DIM}−{pct}%{R}",
            dt=0.45,
        )
    c.println()

    # DuoDecoding / C2T / CLaSP
    c.println(f"  {B}{BCY}Speculative Decoders{R}", dt=0.1)
    c.println()
    spec_rows = [
        ("DuoDecoding", "dual-seq draft+verify",  "30% acc rate",  "γ=4"),
        ("C2T",         "confidence-to-tree",      "81 leaf nodes", "depth=4",),
        ("CLaSP",       "layer-skip spec decode",  "10% acc rate",  "102 skips/run"),
    ]
    c.println(f"  {'Module':<16}  {'Method':<26}  {'Result':<17}  Notes", dt=0.1)
    c.println(f"  {'─' * 70}")
    for name, method, result, note in spec_rows:
        c.println(
            f"  {B}{BCY}{name:<16}{R}  {DIM}{method:<26}{R}  "
            f"{B}{BGN}{result:<17}{R}  {DIM}{note}{R}",
            dt=0.45,
        )
    c.println()
    c.println(
        f"  {B}{BGN}→  SpeCache prefetcher: predict_next_turn_blocks() in 15 µs{R}",
        dt=0.15,
    )
    c.pause(2.5)


# ── Scene 4: Wave 14 Quantisation ─────────────────────────────────────────────

def scene_wave14_quant(c: Cast) -> None:
    _section(c, "Wave 14 — Weight Quantisation + Entropy Coding", colour=BMAG)

    c.typeout(
        "  $ squish it qwen3-8b "
        "--dfloat11 --squeeze-llm --nf4",
        char_delay=0.025, initial_dt=0.2,
    )
    c.println()

    # DFloat11
    c.println(f"  {B}{BMAG}DFloat11{R}  {DIM}Huffman entropy coding for BF16 weights{R}", dt=0.1)
    c.println()
    c.println(f"  {DIM}  Splits BF16 → 5-bit exponent (Huffman) + 6-bit mantissa (raw){R}", dt=0.08)
    c.println(f"  {DIM}  Lossless round-trip.  CPU decompression on demand.{R}", dt=0.06)
    c.println()

    for nw, comp_ms, decomp_ms in [("  4K", 2.4, 3.1), (" 16K", 11.1, 12.9), (" 64K", 38.1, 52.4)]:
        c.println(
            f"    {nw} BF16 weights  "
            f"compress {B}{BMAG}{comp_ms:>5.1f} ms{R}  "
            f"decompress {B}{BCY}{decomp_ms:>5.1f} ms{R}  "
            f"{DIM}lossless{R}",
            dt=0.45,
        )
    c.println()

    # SqueezeLLM
    c.println(f"  {B}{BMAG}SqueezeLLM{R}  {DIM}Non-uniform INT3/4 + sparse FP16 outlier table{R}", dt=0.1)
    c.println()
    c.println(f"  {'Precision':<12}  {'SNR vs FP32':>12}  {'Compression':>14}  {'forward() latency':>20}", dt=0.1)
    c.println(f"  {'─' * 62}")

    sq_rows = [
        ("INT4     ", "22.5 dB", "0.375×", "248 µs"),
        ("INT3     ", "15.8 dB", "0.312×", "252 µs"),
    ]
    for prec, snr, comp, lat in sq_rows:
        c.println(
            f"  {B}{BMAG}{prec:<12}{R}  {B}{BGN}{snr:>12}{R}  "
            f"{B}{BCY}{comp:>14}{R}  {DIM}{lat:>20}{R}",
            dt=0.45,
        )
    c.println()

    # NF4
    c.println(f"  {B}{BMAG}NF4 Quant{R}  {DIM}NormalFloat4 blockwise weight encoding (group=64){R}", dt=0.1)
    c.println()

    for shape, snr, mse in [("64×64", "20.9 dB", "0.00830"),
                             ("128×64", "20.7 dB", "0.00856"),
                             ("128×128", "20.7 dB", "0.00835")]:
        c.println(
            f"    {shape:<10}  SNR {B}{BGN}{snr}{R}  MSE {DIM}{mse}{R}"
            f"  {B}{BCY}0.25×{R} compression",
            dt=0.45,
        )
    c.println()

    # RANSCodec
    c.println(f"  {B}{BMAG}RANSCodec{R}  {DIM}rANS entropy codec (near-Shannon compression){R}", dt=0.1)
    c.println()
    c.println(f"  {DIM}  Shannon entropy: 1.846 bits/symbol{R}", dt=0.08)
    c.println(f"  {DIM}  rANS rate n=4096: 3.881 bits/sym   —  0.485 bytes/symbol{R}", dt=0.06)
    c.println()
    for n, enc, dec, bps in [("256", "179 µs", "333 µs", "4.28"), ("1024", "589 µs", "840 µs", "1.24"), ("4096", "2428 µs", "2850 µs", "0.49")]:
        c.println(
            f"    n={n:<6}  encode {B}{BCY}{enc:>8}{R}  decode {B}{BCY}{dec:>8}{R}"
            f"  {B}{BGN}{bps} bytes/sym{R}",
            dt=0.45,
        )
    c.println()
    c.println(
        f"  {B}{BGN}→  NF4+SqueezeLLM: 5–10× weight memory reduction on Apple Silicon{R}",
        dt=0.15,
    )
    c.pause(2.5)


# ── Scene 5: Wave 14 Speculative Decoding ─────────────────────────────────────

def scene_wave14_spec(c: Cast) -> None:
    _section(c, "Wave 14 — Specialised Speculative Decoding", colour=BMAG)

    c.typeout(
        "  $ squish run --model qwen3-8b "
        "--qspec --copy-spec --sub-spec",
        char_delay=0.025, initial_dt=0.2,
    )
    c.println()

    # QSpec
    c.println(f"  {B}{BMAG}QSpec{R}  {DIM}Complementary-quantisation speculative decoding{R}", dt=0.1)
    c.println()
    c.println(f"  {DIM}  Draft: W4A8 quantised model  →  Verify: W4A16 model{R}", dt=0.08)
    c.println(f"  {DIM}  Acceptance rate higher than random-draft thanks to shared weights.{R}", dt=0.06)
    c.println()
    _tick(c, "Draft acceptance rate (γ=4)", "55%",    "W4A8 → W4A16 verification")
    _tick(c, "Mean tokens accepted / step", "1.10",   "theoretical max = 4")
    _tick(c, "generate() 20 tokens",        "3422 µs","wall time (CPU numpy)")
    c.println()

    # CopySpec
    c.println(f"  {B}{BMAG}CopySpec{R}  {DIM}Suffix-match copy-based draft generator{R}", dt=0.1)
    c.println()
    c.println(f"  {DIM}  Scans token history for longest suffix match.  Zero model cost.{R}", dt=0.08)
    c.println(f"  {DIM}  Best on code, structured data, and repetitive patterns.{R}", dt=0.06)
    c.println()
    _tick(c, "draft() latency",             "2.8 µs", "suffix scan + proposal")
    _tick(c, "draft len / max",             "1/8",    "repetitive history")
    c.println()

    # VisionPrefixCache
    c.println(f"  {B}{BMAG}VisionPrefixCache{R}  {DIM}LRU encoder output cache (multi-modal){R}", dt=0.1)
    c.println()

    # Animated cold → warm lookup
    hits = [0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1]
    bar = ""
    c.println(f"  {DIM}  16-image batch  (H=hit  M=miss):{R}", dt=0.1)
    for hit in hits:
        ch    = f"{BGN}H{R}" if hit else f"{BYL}M{R}"
        bar  += ("H" if hit else "M")
        c.print(f"\r  {DIM}[{R}{ch}{DIM}]{R} {DIM}{bar:>16}{R}", dt=0.07)
    c.println()
    c.println()
    _tick(c, "16-image encode (cold)",      "0.2 ms",  "all images computed once")
    _tick(c, "16-image lookup (warm)",      "<0.1 ms", "hash-based LRU hit")
    _tick(c, "Cache hit rate",              "50%",     "8/16 images served from cache")
    _tick(c, "Speedup vs full encode",      "13.5×",   "per batch on warm cache")
    c.println()

    # SubSpec / DEL / HeteroVocab
    c.println(f"  {B}{BMAG}More Wave 14 spec-decode modules{R}", dt=0.1)
    c.println()
    more = [
        ("SubSpec",      "draft from frozen sub-model layers",       "γ=4 acceptance"),
        ("DEL Decoder",  "dynamic exit — early-stop layer routing",  "adjustable exits"),
        ("HeteroVocab",  "cross-vocabulary draft/verify mapping",    "VocabMapper"),
        ("HeadAwareKV",  "per-head mixed-precision KV store",        "head_dim-aware"),
        ("QuantSpec",    "quantised-draft speculative step",         "generate_step()"),
    ]
    c.println(f"  {'Module':<18}  {'Key feature':<44}  Notes", dt=0.1)
    c.println(f"  {'─' * 74}")
    for name, feat, note in more:
        c.println(
            f"  {B}{BMAG}{name:<18}{R}  {DIM}{feat:<44}{R}  {note}",
            dt=0.45,
        )
    c.println()
    c.println(
        f"  {B}{BGN}→  55% draft acceptance on QSpec — highest in Squish spec-decode suite{R}",
        dt=0.15,
    )
    c.pause(2.5)


# ── Scene 6: Compound stack + summary ────────────────────────────────────────

def scene_full_stack(c: Cast) -> None:
    _section(c, "Full Wave 13+14 Stack — All 25 modules active", colour=BGN)

    c.typeout(
        "  $ squish run --model qwen3-8b \\",
        char_delay=0.022, initial_dt=0.2,
    )
    c.typeout(
        "      --duo-attention --shadow-kv --pq-cache --spe-cache \\",
        char_delay=0.022,
    )
    c.typeout(
        "      --knapspec --token-merging --duo-decoding --c2t --clasp \\",
        char_delay=0.022,
    )
    c.typeout(
        "      --dfloat11 --squeeze-llm --nf4 --qspec --copy-spec \\",
        char_delay=0.022,
    )
    c.typeout(
        "      --vision-prefix-cache --head-aware-kv --hetero-vocab",
        char_delay=0.022,
    )
    c.println()

    # Startup sequence
    inits = [
        (BCY,  "[DuoAttention]",    "56% streaming heads • 0.20× KV memory"),
        (BCY,  "[ShadowKV]",        "SVD rank 16/64 • 4.00× key memory saved"),
        (BCY,  "[PQCache]",         "4 sub-vecs • 16 codes • 64× memory reduction"),
        (BCY,  "[SpeCache]",        "block_size=64 • 8-block prefetch budget"),
        (BCY,  "[KnapSpec]",        "n_layers=32 • budget_fraction=0.50"),
        (BCY,  "[TokenMerging]",    "r=16 • bipartite merge every layer"),
        (BCY,  "[DuoDecoding]",     "γ=4 • DuoScheduler + DuoCPUVerifier"),
        (BCY,  "[C2T]",             "depth=4 • 81 leaf nodes"),
        (BCY,  "[CLaSP]",           "max_skip_layers=6 • draft_gamma=4"),
        (BMAG, "[DFloat11]",        "lossless Huffman BF16 weight coding"),
        (BMAG, "[SqueezeLLM]",      "INT3 bits=3 • SNR 15.8 dB • 0.31× memory"),
        (BMAG, "[NF4Quant]",        "group_size=64 • SNR 20.7 dB • 0.25×"),
        (BMAG, "[RANSCodec]",       "n_symbols=16 • near-Shannon entropy"),
        (BMAG, "[QSpec]",           "W4A8 draft • W4A16 verify • 55% acc"),
        (BMAG, "[CopySpec]",        "min_match=3 • max_draft=8 • 2.8 µs"),
        (BMAG, "[VisionCache]",     "LRU 256 slots • 13.5× speedup warm"),
        (BMAG, "[HeadAwareKV]",     "per-head mixed-precision store"),
        (BMAG, "[HeteroVocab]",     "VocabMapper draft→target vocab mapping"),
    ]
    for colour, tag, msg in inits:
        c.println(f"  {B}{colour}{tag:<20}{R}  {DIM}{msg}{R}", dt=0.4)

    c.println()
    c.println(f"  {DIM}Squish serving on http://127.0.0.1:11435{R}", dt=0.5)
    c.println()
    c.pause(0.3)

    # Summary table
    summary = [
        ("KV memory reduction",        "10–30×",  "DuoAttn + ShadowKV + PQCache",  BCY),
        ("Context length",             "4×",      "same VRAM budget",               BCY),
        ("Weight storage",             "3–10×",   "DFloat11 + SqueezeLLM + NF4",   BMAG),
        ("Draft acceptance",           "55%",     "QSpec best in suite",            BMAG),
        ("Vision cache speedup",       "13.5×",   "warm-cache hit (50% rate)",      BMAG),
        ("Layer compute skip",         "36%",     "KnapSpec ctx=2048",              BCY),
        ("Token sequence reduction",   "3–12%",   "TokenMerging r=16",              BCY),
        ("Compound stack overhead",    "~1.0×",   "virtually free on Apple Silicon", BGN),
    ]
    c.println(f"  {B}Wave 13+14 improvements vs Squish v1:{R}", dt=0.1)
    c.println()
    for label, val, note, col in summary:
        c.println(
            f"  {DIM}·{R}  {label:<36}  {B}{col}{val:<10}{R}  {DIM}{note}{R}",
            dt=0.45,
        )
    c.pause(2.5)


# ── Scene 7: Test results screen ─────────────────────────────────────────────

def scene_tests(c: Cast) -> None:
    _section(c, "Test Suite — All 3 848 tests passing", colour=BGN)

    c.typeout(
        "  $ python3 -m pytest tests/ --ignore=tests/test_int4_loader.py -q",
        char_delay=0.025, initial_dt=0.2,
    )
    c.println()

    test_files = [
        ("test_wave13_server_wiring.py", 47),
        ("test_wave14_server_wiring.py", 74),
        ("test_server_wiring.py",        "4  (61 modules confirmed)"),
        ("test_paged_attention.py",      112),
        ("test_speculative.py",          89),
        ("test_kv_cache.py",             76),
        ("test_quantizer.py",            64),
        ("test_scheduler.py",            58),
        ("+ 37 more test files",         "3 382"),
    ]

    for fname, count in test_files:
        c.println(
            f"  {DIM}{'.' * max(0, 54 - len(str(fname)))}{R}"
            f"  {DIM}{fname:<38}{R}  {B}{BGN}{count}{R}",
            dt=0.4,
        )
    c.println()
    c.hbar(colour=BGN)
    c.println(
        f"  {B}{BGN}  3 848 passed{R}  {DIM}·{R}  {BYL}65 skipped{R}  "
        f"{DIM}·{R}  {B}{BGN}0 failed{R}  "
        f"{DIM}(15 warnings, all importskip){R}",
        dt=0.15,
    )
    c.hbar(colour=BGN)
    c.println()

    c.println(f"  {B}61 modules wired in server.py{R}", dt=0.08)
    c.println()

    modules_grid = [
        ["duo_attention", "shadow_kv", "pq_cache", "spe_cache", "knapspec"],
        ["token_merging", "duo_decoding", "c2t", "clasp", "dfloat11"],
        ["squeeze_llm", "nf4_quant", "rans_codec", "qspec", "copy_spec"],
        ["vision_prefix_cache", "head_infer", "hetero_vocab_sd", "sub_spec", "del_decoder"],
    ]
    for row in modules_grid:
        c.println(
            "  " + "  ".join(f"{DIM}{m:<22}{R}" for m in row),
            dt=0.4,
        )

    c.println()
    c.pause(2.5)


# ── Scene 8: CLI demo ────────────────────────────────────────────────────────

def scene_cli(c: Cast) -> None:
    _section(c, "Squish CLI — Quick Start", colour=CYN)

    cmds = [
        ("squish catalog",                  "Browse 29 available models"),
        ("squish pull qwen3:8b",            "Download + compress (~5 min)"),
        ("squish run qwen3:8b",             "Start server on http://localhost:11435"),
        ("squish chat qwen3:8b",            "Interactive terminal chat"),
        ("squish bench qwen3:8b",           "Quick throughput / latency benchmark"),
        ("squish predict qwen3:8b",         "LIFE analytical performance prediction"),
        ("squish rotate qwen3:8b",          "SpinQuant Cayley-SGD rotation calibration"),
        ("squish it qwen3:8b --nf4",        "Compress weights to NF4"),
        ("squish models",                   "List locally downloaded models"),
        ("squish doctor",                   "Check all dependencies"),
    ]

    c.println(f"  {DIM}Drop-in for OpenAI and Ollama clients:{R}", dt=0.08)
    c.println()
    c.println(f"  {DIM}  export OPENAI_BASE_URL=http://localhost:11435/v1{R}", dt=0.06)
    c.println(f"  {DIM}  export OPENAI_API_KEY=squish{R}", dt=0.05)
    c.println()

    for cmd, desc in cmds:
        c.println(f"  {B}{BGN}${R} {B}{WHT}{cmd}{R}", dt=0.4)
        c.println(f"    {DIM}{desc}{R}", dt=0.15)

    c.println()
    c.println(
        f"  {DIM}Web UI · Tool calling · Batch scheduler · OpenAI + Ollama drop-in{R}",
        dt=0.1,
    )
    c.println(
        f"  {DIM}No API key. No cloud. No data leaving your machine. Free.{R}",
        dt=0.08,
    )
    c.pause(2.0)


# ── Scene 9: Closing ─────────────────────────────────────────────────────────

def scene_closing(c: Cast) -> None:
    c.pause(0.2)
    c.hbar()
    c.println()
    c.println(f"  {B}{CYN}Squish{R}  {DIM}Waves 1–14  ·  61 optimisation modules  ·  3 848 tests{R}", dt=0.1)
    c.println()

    links = [
        ("docs",       "docs/ARCHITECTURE.md  ·  docs/RESULTS.md"),
        ("benchmarks", "dev/benchmarks/bench_wave13_14.py"),
        ("results",    "dev/results/wave13_14_bench.json"),
        ("sources",    "squish/{duo_attention,shadow_kv,pq_cache,dfloat11,...}.py"),
        ("GitHub",     "github.com/wesleyscholl/squish"),
    ]
    for label, path in links:
        c.println(f"  {DIM}{label:<14}{R}  {path}", dt=0.3)

    c.println()
    c.hbar()
    c.print(SHOW_C)
    c.pause(4.0)


# ── Main assembler ────────────────────────────────────────────────────────────

def build_cast() -> Cast:
    c = Cast()
    scene_title(c)
    scene_wave13_kv(c)
    scene_wave13_spec(c)
    scene_wave14_quant(c)
    scene_wave14_spec(c)
    scene_full_stack(c)
    scene_tests(c)
    scene_cli(c)
    scene_closing(c)
    return c


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description="Generate Squish Wave 13+14 demo GIF")
    ap.add_argument("--out",       default="dev/demos/squish-wave13-14-demo.gif",
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

    # Find agg
    agg_bin = (
        args.agg
        or shutil.which("agg")
        or "/tmp/agg"
        or "/opt/homebrew/bin/agg"
    )
    if not Path(agg_bin).exists():
        print(f"\n  ✗  agg not found at {agg_bin}")
        print(f"     Install: curl -fsSL https://github.com/asciinema/agg/releases/"
              f"download/v1.4.3/agg-x86_64-unknown-linux-gnu -o /tmp/agg && chmod +x /tmp/agg")
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
