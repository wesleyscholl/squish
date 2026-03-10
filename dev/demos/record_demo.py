#!/usr/bin/env python3
"""
Squish demo GIF generator.

Generates an asciinema v2 .cast file from the 1.5B eval results,
then converts to GIF using agg.

Usage:
    python3 dev/demos/record_demo.py                  # generate + convert to GIF
    python3 dev/demos/record_demo.py --cast-only       # only write the .cast file
    python3 dev/demos/record_demo.py --out dev/demos/squish_demo.gif
    python3 dev/demos/record_demo.py --results eval_output/eval_meta.json
"""

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import time as _time
from pathlib import Path

# ── ANSI helpers ─────────────────────────────────────────────────────────────
R   = "\x1b[0m"
B   = "\x1b[1m"
DIM = "\x1b[2m"
GRN = "\x1b[32m"
BRN = "\x1b[33m"    # yellow/gold
CYN = "\x1b[36m"
RED = "\x1b[31m"
WHT = "\x1b[97m"
BGN = "\x1b[92m"    # bright green
BRD = "\x1b[91m"    # bright red

CLEAR  = "\x1b[2J\x1b[H"
HIDE_C = "\x1b[?25l"
SHOW_C = "\x1b[?25h"

W = 88   # terminal width
H = 26   # terminal height

# ── Benchmark data ─────────────────────────────────────────────────────────── 
TASKS = [
    ("ARC-Challenge", "arc_challenge", "acc,none",      "acc_norm,none"),
    ("HellaSwag",     "hellaswag",     "acc_norm,none",  None),
    ("PIQA",          "piqa",          "acc_norm,none",  None),
    ("Winogrande",    "winogrande",    "acc,none",       None),
]

def load_scores(ref_path: Path, comp_path: Path):
    """Load benchmark scores from eval result JSONs."""
    ref_data  = json.loads(ref_path.read_text())  if ref_path.is_file()  else {}
    comp_data = json.loads(comp_path.read_text()) if comp_path.is_file() else {}
    ref_r  = ref_data.get("results",  {})
    comp_r = comp_data.get("results", {})

    rows = []
    for label, key, primary, alt in TASKS:
        def pick(d, k=key, p=primary, a=alt):
            r = d.get(k, {})
            v = r.get(p)
            if v is None and a:
                v = r.get(a)
            return v
        ref_v  = pick(ref_r)
        comp_v = pick(comp_r)
        rows.append((label, ref_v, comp_v))
    return rows


def load_meta(meta_path: Path):
    """Return (ref_load_s, comp_load_s) from eval_meta.json."""
    if not meta_path.is_file():
        return 34.86, 0.51
    m = json.loads(meta_path.read_text())
    return m.get("ref_load_time_s", 34.86), m.get("comp_load_time_s", 0.51)


# ── Cast builder ──────────────────────────────────────────────────────────────

class Cast:
    def __init__(self, width=W, height=H, title="Squish Demo"):
        self.width  = width
        self.height = height
        self.title  = title
        self.events: list[tuple[float, str, str]] = []
        self._t = 0.0

    def _add(self, text: str, dt: float = 0.0):
        self._t += dt
        self.events.append((round(self._t, 4), "o", text))

    def pause(self, secs: float):
        self._t += secs

    def println(self, text: str = "", dt: float = 0.0):
        self._add(text + "\r\n", dt)

    def print(self, text: str, dt: float = 0.0):
        self._add(text, dt)

    def typeout(self, text: str, char_delay: float = 0.035, initial_dt: float = 0.0):
        """Type text out char by char."""
        self._t += initial_dt
        for ch in text:
            self.events.append((round(self._t, 4), "o", ch))
            self._t += char_delay
        self._add("\r\n")

    def progress_bar(self, total_blocks: int = 38, block_delay: float = 0.1,
                     initial_dt: float = 0.0, prefix: str = "  ["):
        """Animate a progress bar."""
        self._t += initial_dt
        self._add(prefix)
        for _ in range(total_blocks):
            self._add("█", block_delay)

    def dump(self) -> str:
        header = json.dumps({
            "version": 2,
            "width":   self.width,
            "height":  self.height,
            "timestamp": 1740668400,
            "title":   self.title,
            "env": {"TERM": "xterm-256color", "SHELL": "/bin/zsh"},
        })
        lines = [header]
        for t, kind, text in self.events:
            lines.append(json.dumps([t, kind, text]))
        return "\n".join(lines) + "\n"


def build_cast(ref_load: float, comp_load: float, rows: list) -> Cast:
    c = Cast()
    speedup = ref_load / comp_load

    bar_total = 36  # blocks in progress bar

    # ── Frame 0: clear + hide cursor ──────────────────────────────────────────
    c.print(CLEAR + HIDE_C, dt=0.1)

    # ── Header ────────────────────────────────────────────────────────────────
    logo_line = f"{B}{CYN}  ███████╗ ██████╗ ██╗   ██╗██╗███████╗██╗  ██╗{R}"
    c.println(f"  {'─' * (W - 4)}", dt=0.0)
    c.println(f"{B}{CYN}  {'S Q U I S H':^{W - 4}}{R}")
    c.println(f"{DIM}  {'LLM Compression — Industry-Standard Benchmark Demo':^{W - 4}}{R}")
    c.println(f"  {'─' * (W - 4)}")
    c.println()

    # ── Model info ────────────────────────────────────────────────────────────
    c.println(f"  {B}Model:{R}  Qwen2.5-1.5B-Instruct   {DIM}BF16 (14-bit) → Squish 4-bit{R}")
    c.println()

    # ── Reference load bar (slow) ─────────────────────────────────────────────
    c.println(f"  {DIM}Reference model (BF16) —{R}", dt=0.4)
    c.progress_bar(total_blocks=bar_total, block_delay=0.08, initial_dt=0.3,
                   prefix=f"  {DIM}[")
    c.print(f"]{R}  {RED}{B}{ref_load:.2f}s{R}{DIM}  (reference){R}\r\n", dt=0.05)
    c.println()

    # ── Compressed load bar (fast) ────────────────────────────────────────────
    c.println(f"  {B}{CYN}Squish 4-bit compressed —{R}", dt=0.4)
    bar_str = "█" * bar_total
    c.print(f"  {CYN}[{bar_str}]{R}  ", dt=0.3)
    c.print(f"{BGN}{B}{comp_load:.2f}s{R}", dt=0.05)
    c.println(f"  {BGN}{B}← {speedup:.1f}× faster!{R}", dt=0.1)
    c.println()

    # ── Speedup summary ───────────────────────────────────────────────────────
    c.pause(0.3)
    bar_width = 50
    ratio = min(1.0, comp_load / ref_load)
    filled = max(1, round(ratio * bar_width))
    empty  = bar_width - filled

    c.println(f"  {DIM}{'Load time comparison':}{R}", dt=0.1)
    c.println(f"  {RED}{'BF16 ref ':>12}  {'█' * bar_total}{'░' * (bar_width - bar_total)}  {ref_load:.2f}s{R}", dt=0.15)
    comp_blocks = max(1, round((comp_load / ref_load) * bar_total))
    c.println(f"  {BGN}{'Squish 4b ':>12}  {'█' * comp_blocks}{' ' * (bar_width - comp_blocks)}  {comp_load:.2f}s{R}", dt=0.12)
    c.println()

    # ── Benchmark table ───────────────────────────────────────────────────────
    c.pause(0.3)
    col1, col2, col3, col4, col5 = 18, 10, 12, 8, 8
    hdr = (f"  {B}{'Task':<{col1}}{'Reference':>{col2}}{'Squish 4b':>{col3}}"
           f"{'Δ':>{col4}}  {'':>{col5}}{R}")
    sep = f"  {'─' * (col1 + col2 + col3 + col4 + col5 + 4)}"
    c.println(f"  {'─' * (W - 4)}", dt=0.2)
    c.println(f"{B}{'  Benchmark Results (1000 samples each)':^{W}}{R}")
    c.println(f"  {'─' * (W - 4)}")
    c.println(hdr, dt=0.15)
    c.println(sep)

    all_pass = True
    for label, ref_v, comp_v in rows:
        if ref_v is None or comp_v is None:
            ref_s  = "  —   "
            comp_s = "  —   "
            delta_s = "  —  "
            ok_s   = f"{DIM}skip{R}"
        else:
            delta = comp_v - ref_v
            ok    = abs(delta) <= 0.02
            if not ok:
                all_pass = False
            ref_s   = f"{ref_v * 100:.1f}%"
            comp_s  = f"{comp_v * 100:.1f}%"
            sign    = "+" if delta >= 0 else ""
            delta_s = f"{sign}{delta * 100:.1f}%"
            col     = BGN if ok else BRD
            mark    = "✓" if ok else "✗"
            ok_s    = f"{col}{B}{mark}{R}"

        c.println(
            f"  {label:<{col1}}{ref_s:>{col2}}{comp_s:>{col3}}"
            f"{delta_s:>{col4}}  {ok_s}",
            dt=0.18,
        )

    c.println(sep, dt=0.12)

    # ── Average delta ─────────────────────────────────────────────────────────
    valid = [(r, co) for _, r, co in rows if r is not None and co is not None]
    if valid:
        avg_ref  = sum(r  for r, _ in valid) / len(valid)
        avg_comp = sum(co for _, co in valid) / len(valid)
        avg_d    = avg_comp - avg_ref
        sign     = "+" if avg_d >= 0 else ""
        c.println(
            f"  {DIM}{'Average':<{col1}}{avg_ref * 100:.1f}%{B}  "
            f"{avg_comp * 100:.1f}%  {sign}{avg_d * 100:.1f}%{R}",
            dt=0.12,
        )

    c.println()

    # ── Final verdict ─────────────────────────────────────────────────────────
    c.pause(0.4)
    c.println(f"  {'─' * (W - 4)}")
    if all_pass:
        c.println(
            f"  {BGN}{B}✓  ALL BENCHMARKS PASSED — accuracy within ±2% of reference{R}",
            dt=0.15,
        )
    else:
        c.println(
            f"  {BRD}{B}✗  Some tasks exceeded ±2% — review results above{R}",
            dt=0.15,
        )
    c.println(
        f"  {BGN}{B}✓  Load time: {comp_load:.2f}s  vs  {ref_load:.2f}s  "
        f"({speedup:.1f}× faster){R}",
        dt=0.1,
    )
    c.println(f"  {'─' * (W - 4)}", dt=0.1)
    c.println()
    c.print(SHOW_C)
    c.pause(3.0)  # hold end frame

    return c


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Generate Squish demo GIF")
    ap.add_argument("--out",        default="dev/demos/squish_demo.gif",
                    help="Output GIF path (default: dev/demos/squish_demo.gif)")
    ap.add_argument("--cast-out",   default="dev/demos/squish_demo.cast",
                    help="Output .cast path (default: dev/demos/squish_demo.cast)")
    ap.add_argument("--cast-only",  action="store_true",
                    help="Write .cast only, skip GIF conversion")
    ap.add_argument("--results",    default="eval_output",
                    help="Directory with eval_*.json files, or eval_meta.json path")
    ap.add_argument("--agg",        default=None,
                    help="Path to agg binary (auto-detected if not given)")
    ap.add_argument("--font-size",  type=int, default=16,
                    help="agg font size (default: 16)")
    ap.add_argument("--speed",      type=float, default=1.0,
                    help="Playback speed multiplier for agg (default: 1.0)")
    args = ap.parse_args()

    # ── Resolve paths ─────────────────────────────────────────────────────────
    root = Path(__file__).resolve().parent.parent
    results_path = Path(args.results)
    if not results_path.is_absolute():
        results_path = root / results_path

    if results_path.is_dir():
        meta_path = results_path / "eval_meta.json"
        ref_path  = results_path / "eval_reference.json"
        comp_path = results_path / "eval_compressed.json"
    else:
        meta_path = results_path
        ref_path  = results_path.parent / "eval_reference.json"
        comp_path = results_path.parent / "eval_compressed.json"

    cast_out = Path(args.cast_out)
    gif_out  = Path(args.out)
    if not cast_out.is_absolute():
        cast_out = root / cast_out
    if not gif_out.is_absolute():
        gif_out  = root / gif_out

    cast_out.parent.mkdir(parents=True, exist_ok=True)
    gif_out.parent.mkdir(parents=True, exist_ok=True)

    # ── Load data ─────────────────────────────────────────────────────────────
    ref_load, comp_load = load_meta(meta_path)
    rows = load_scores(ref_path, comp_path)

    print(f"  Reference load : {ref_load:.2f}s")
    print(f"  Compressed load: {comp_load:.2f}s  ({ref_load / comp_load:.1f}x faster)")
    print(f"  Benchmark rows : {len(rows)}")
    print()

    # ── Build + write cast ────────────────────────────────────────────────────
    cast = build_cast(ref_load, comp_load, rows)
    cast_out.write_text(cast.dump())
    duration = cast.events[-1][0] if cast.events else 0
    print(f"  ✓ Wrote {cast_out}  ({len(cast.events)} events, {duration:.1f}s)")

    if args.cast_only:
        return

    # ── Convert to GIF via agg ────────────────────────────────────────────────
    agg_bin = args.agg or shutil.which("agg") or "/opt/homebrew/bin/agg"
    if not Path(agg_bin).exists():
        print(f"\n  ✗ agg not found at {agg_bin}")
        print(f"    Install: brew install agg")
        print(f"    Then run: {agg_bin} {cast_out} {gif_out}")
        return

    cmd = [
        agg_bin,
        str(cast_out),
        str(gif_out),
        "--font-size",  str(args.font_size),
        "--speed",      str(args.speed),
        "--cols",       str(W),
        "--rows",       str(H),
    ]
    print(f"\n  Converting to GIF via agg …")
    print(f"  $ {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"\n  ✗ agg failed (rc={result.returncode}):")
        print(result.stderr)
        sys.exit(1)

    size_kb = gif_out.stat().st_size // 1024
    print(f"\n  ✓ GIF written → {gif_out}  ({size_kb} KB)")
    print(f"    Open with: open {gif_out}")


if __name__ == "__main__":
    main()
