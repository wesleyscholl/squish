"""
squish/_term.py

Shared ANSI true-colour terminal utilities used by cli.py and server.py.

24-bit RGB escape codes (\033[38;2;R;G;Bm) bypass the host terminal's colour
theme entirely — themes only remap the 16 named ANSI palette indices, not
direct RGB.  Squish brand colours therefore render correctly on any true-colour
terminal regardless of theme, and fall back to no colour on basic TTYs.

Public API
----------
    has_truecolor(fd=1)      → bool   (True when fd is a true-colour TTY)
    C                        → _Palette  (brand colour constants)
    gradient(text, stops)    → str    (left-to-right RGB gradient text)
    LOGO_GRAD                → list   (purple → pink → teal stop list)
"""
from __future__ import annotations

import os
import sys

__all__ = ["has_truecolor", "C", "gradient", "LOGO_GRAD"]


def has_truecolor(fd: int = 1) -> bool:
    """Return True when file-descriptor *fd* (1=stdout, 2=stderr) is a
    true-colour TTY and NO_COLOR is not set."""
    try:
        is_tty = os.isatty(fd)
    except Exception:
        return False
    return (
        is_tty
        and "NO_COLOR" not in os.environ
        and (
            os.environ.get("COLORTERM", "").lower() in ("truecolor", "24bit")
            or os.environ.get("TERM_PROGRAM", "") in (
                "iTerm.app", "WezTerm", "Ghostty", "Hyper", "vscode", "warp",
                "Apple_Terminal",
            )
            or "kitty" in os.environ.get("TERM", "")
            or "direct" in os.environ.get("TERM", "")
            or bool(os.environ.get("FORCE_COLOR", ""))
        )
    )


# Palette — computed once at import time against stdout
_TC = has_truecolor(sys.stdout.fileno() if hasattr(sys.stdout, "fileno") else 1)


def _k(s: str) -> str:
    """Return *s* only when stdout is a true-colour TTY."""
    return s if _TC else ""


class _Palette:
    """ANSI 24-bit colour constants aligned to the Squish brand palette."""
    DP  = _k("\033[38;2;88;28;135m")    # deep purple  #581C87
    P   = _k("\033[38;2;124;58;237m")   # purple       #7C3AED
    V   = _k("\033[38;2;139;92;246m")   # violet       #8B5CF6
    L   = _k("\033[38;2;167;139;250m")  # lilac        #A78BFA
    MG  = _k("\033[38;2;192;132;252m")  # med-purple   #C084FC
    PK  = _k("\033[38;2;236;72;153m")   # pink         #EC4899
    LPK = _k("\033[38;2;249;168;212m")  # light pink   #F9A8D4
    T   = _k("\033[38;2;34;211;238m")   # teal         #22D3EE
    LT  = _k("\033[38;2;165;243;252m")  # light teal   #A5F3FC
    G   = _k("\033[38;2;52;211;153m")   # mint green   #34D399
    W   = _k("\033[38;2;248;250;252m")  # near-white   #F8FAFC
    SIL = _k("\033[38;2;180;185;210m")  # silver       #B4B9D2
    DIM = _k("\033[38;2;100;116;139m")  # dim slate    #64748B
    B   = _k("\033[1m")                 # bold
    R   = _k("\033[0m")                 # reset all


C = _Palette()

# Purple → pink → teal gradient used for the big logo and accent lines.
LOGO_GRAD: list[tuple[int, int, int]] = [
    ( 88,  28, 135),   # deep purple
    (124,  58, 237),   # purple
    (139,  92, 246),   # violet
    (192, 100, 220),   # lavender-pink
    (236,  72, 153),   # pink
    ( 34, 211, 238),   # teal
]


def gradient(text: str, stops: list[tuple[int, int, int]]) -> str:
    """Interpolate a left-to-right RGB gradient across *text*.

    Only emits escape codes when stdout is a true-colour TTY; otherwise
    returns *text* unchanged so plain-terminal output stays readable.
    """
    if not _TC or not text:
        return text
    n = len(text)
    k = len(stops) - 1
    out: list[str] = []
    for i, ch in enumerate(text):
        t = i / max(n - 1, 1)
        seg = min(int(t * k), k - 1)
        frac = t * k - seg
        r1, g1, b1 = stops[seg]
        r2, g2, b2 = stops[seg + 1]
        r = int(r1 + (r2 - r1) * frac)
        g = int(g1 + (g2 - g1) * frac)
        b = int(b1 + (b2 - b1) * frac)
        out.append(f"\033[38;2;{r};{g};{b}m{ch}")
    out.append("\033[0m")
    return "".join(out)
