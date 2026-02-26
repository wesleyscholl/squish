#!/usr/bin/env python3
"""Live monitor for the 7B eval process. Run: python3 monitor_eval.py"""
import subprocess, time, os, sys, shutil
from datetime import datetime

PID        = 16476
LOG        = "/tmp/squish_7b_full.log"
OUTPUT_DIR = "/Users/wscholl/poc/eval_output_7b_full"
INTERVAL   = 1

# ── ANSI colours ────────────────────────────────────────────────────────────
R  = "\033[0m"
B  = "\033[1m"          # bold
DIM= "\033[2m"
CY = "\033[96m"         # cyan
GR = "\033[92m"         # green
YL = "\033[93m"         # yellow
RD = "\033[91m"         # red
MG = "\033[95m"         # magenta
BL = "\033[94m"         # blue
BG = "\033[100m"        # dark grey bg
WH = "\033[97m"         # white

SPINNER = ["⠋","⠙","⠹","⠸","⠼","⠴","⠦","⠧","⠇","⠏"]

def clear():
    print("\033[H\033[J", end="")

def bar(fraction, width=30, color=GR):
    filled = int(fraction * width)
    empty  = width - filled
    return f"{color}{'█' * filled}{DIM}{'░' * empty}{R}"

def fmt_bytes(kb):
    gb = kb / 1048576
    if gb >= 1:
        return f"{gb:.2f} GB"
    return f"{kb/1024:.0f} MB"

def get_proc():
    try:
        out = subprocess.check_output(
            ["ps", "-p", str(PID), "-o", "pid,etime,cputime,pcpu,rss,stat"],
            text=True, stderr=subprocess.DEVNULL
        ).splitlines()
        if len(out) < 2:
            return None
        parts = out[1].split()
        return {
            "pid":     parts[0],
            "elapsed": parts[1],
            "cputime": parts[2],
            "pcpu":    float(parts[3]),
            "rss_kb":  int(parts[4]),
            "stat":    parts[5],
        }
    except Exception:
        return None

def get_disk():
    try:
        out = subprocess.check_output(["df", "-k", os.path.expanduser("~")],
                                      text=True).splitlines()
        parts = out[1].split()
        used_kb  = int(parts[2])
        avail_kb = int(parts[3])
        total_kb = used_kb + avail_kb
        pct      = used_kb / total_kb if total_kb else 0
        return used_kb, avail_kb, total_kb, pct
    except Exception:
        return None

def get_output_files():
    try:
        files = os.listdir(OUTPUT_DIR)
        return sorted(files)
    except Exception:
        return []

def log_tail(n=5):
    try:
        out = subprocess.check_output(["tail", f"-{n}", LOG],
                                      text=True, stderr=subprocess.DEVNULL)
        return out.strip().splitlines()
    except Exception:
        return []

def log_size():
    try:
        return os.path.getsize(LOG)
    except Exception:
        return 0

tick = 0
prev_log_size = 0

try:
    while True:
        spin  = SPINNER[tick % len(SPINNER)]
        now   = datetime.now().strftime("%H:%M:%S")
        proc  = get_proc()
        disk  = get_disk()
        files = get_output_files()
        lines = log_tail(4)
        lsize = log_size()
        log_growing = lsize != prev_log_size
        prev_log_size = lsize

        clear()

        # ── Header ──────────────────────────────────────────────────────────
        cols = shutil.get_terminal_size().columns
        title = f" {MG}{B}◈ Squish 7B Eval Monitor{R}  {DIM}{now}{R} "
        pad = "─" * max(0, (cols - len(title) + len(MG)+len(B)+len(R)+len(DIM)+len(R)) // 2)
        print(f"{BL}{pad}{R}{title}{BL}{pad}{R}")
        print()

        # ── Process ─────────────────────────────────────────────────────────
        if proc:
            cpu_color = GR if proc["pcpu"] > 5 else YL if proc["pcpu"] > 1 else RD
            alive_color = GR
            alive_label = f"{GR}{B}RUNNING{R}"
        else:
            cpu_color = RD
            alive_label = f"{RD}{B}DEAD / FINISHED{R}"

        print(f"  {CY}{B}Process{R}  {DIM}PID {PID}{R}   {alive_label}   {CY}{spin}{R}")
        print()

        if proc:
            ram_gb    = proc["rss_kb"] / 1048576
            ram_frac  = min(ram_gb / 16.0, 1.0)
            ram_color = GR if ram_gb < 10 else YL if ram_gb < 14 else RD

            print(f"  {WH}Elapsed   {R}{CY}{proc['elapsed']:<14}{R}  {WH}CPU time  {R}{GR}{proc['cputime']}{R}")
            print(f"  {WH}CPU now   {R}{cpu_color}{proc['pcpu']:>5.1f}%{R}         {WH}State     {R}{YL}{proc['stat']}{R}")
            print()
            print(f"  {WH}RAM (RSS) {R}  {bar(ram_frac, 28, ram_color)}  {ram_color}{B}{ram_gb:.2f} GB{R} / 16 GB")
        else:
            print(f"  {RD}Process not found — eval may have finished or crashed.{R}")
        print()

        # ── Disk ────────────────────────────────────────────────────────────
        if disk:
            used_kb, avail_kb, total_kb, pct = disk
            disk_color = GR if pct < 0.75 else YL if pct < 0.90 else RD
            print(f"  {WH}Disk      {R}  {bar(pct, 28, disk_color)}  "
                  f"{disk_color}{B}{pct*100:.1f}%{R}  "
                  f"{DIM}used {fmt_bytes(used_kb)}  free {fmt_bytes(avail_kb)}{R}")
        print()

        # ── Output files ────────────────────────────────────────────────────
        print(f"  {CY}{B}Output files{R}  {DIM}{OUTPUT_DIR}{R}")
        if files:
            for f in files:
                path = os.path.join(OUTPUT_DIR, f)
                size = os.path.getsize(path)
                mtime = datetime.fromtimestamp(os.path.getmtime(path)).strftime("%H:%M:%S")
                icon = f"{GR}✓{R}" if f.endswith(".json") else f"{YL}·{R}"
                print(f"    {icon}  {WH}{f:<40}{R}  {DIM}{size:>8,} B   {mtime}{R}")
            if any(f == "eval_compressed.json" for f in files):
                print(f"\n  {GR}{B}🎉 eval_compressed.json written — eval COMPLETE!{R}")
        else:
            print(f"    {DIM}(no result files yet){R}")
        print()

        # ── Log tail ────────────────────────────────────────────────────────
        log_icon = f"{GR}↑ growing{R}" if log_growing else f"{DIM}static{R}"
        print(f"  {CY}{B}Log tail{R}  {DIM}{LOG}  {log_icon}  {lsize:,} bytes{R}")
        for line in lines:
            # Colour tqdm progress lines differently
            if "100%|" in line:
                print(f"    {GR}{line}{R}")
            elif "✓" in line:
                print(f"    {YL}{line}{R}")
            elif "WARNING" in line or "ERROR" in line:
                print(f"    {RD}{line}{R}")
            else:
                print(f"    {DIM}{line}{R}")
        print()

        # ── Footer ──────────────────────────────────────────────────────────
        print(f"  {DIM}Refreshes every {INTERVAL}s  ·  Ctrl-C to quit{R}")

        tick += 1
        time.sleep(INTERVAL)

except KeyboardInterrupt:
    print(f"\n{DIM}Monitor exited.{R}\n")
