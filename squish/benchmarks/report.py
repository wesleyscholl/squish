# squish/benchmarks/report.py
"""Unified benchmark report generator.

Merges Track A–E outputs into docs/benchmark_<date>.md.
"""
from __future__ import annotations

__all__ = ["ReportConfig", "ReportGenerator"]

import datetime
import platform
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from squish.benchmarks.base import ResultRecord


@dataclass
class ReportConfig:
    """Configuration for unified report generation."""
    input_dir: str = "eval_output"
    output_dir: str = "docs"
    hardware_info: str = ""   # auto-detected if empty
    squish_version: str = ""  # auto-detected if empty


class ReportGenerator:
    """Merges all Track A–E result records into a single markdown report."""

    TRACK_LABELS = {
        "quality": "Track A — Quality / Normal Text",
        "code":    "Track B — Code Generation",
        "tools":   "Track C — Tool Use / Function Calling",
        "agent":   "Track D — Agentic Tasks",
        "perf":    "Track E — Performance / Speed",
    }

    def __init__(self, config: ReportConfig) -> None:
        self.config = config

    def _detect_hardware(self) -> str:
        if self.config.hardware_info:
            return self.config.hardware_info
        try:
            import subprocess
            result = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True, text=True, timeout=3
            )
            chip = result.stdout.strip() or platform.processor()
        except Exception:
            chip = platform.processor() or "unknown"
        return f"{chip} / {sys.platform}"

    def _detect_version(self) -> str:
        if self.config.squish_version:
            return self.config.squish_version
        try:
            from importlib.metadata import version
            return version("squish")
        except Exception:
            return "dev"

    def load_results(self, date_prefix: str = "") -> List[ResultRecord]:
        results = []
        for p in Path(self.config.input_dir).glob("*.json"):
            if p.name.startswith("comparison_") or p.name == "eval_meta.json":
                continue
            try:
                r = ResultRecord.load(p)
                if date_prefix and not r.timestamp.startswith(date_prefix):
                    continue
                results.append(r)
            except Exception:  # noqa: BLE001
                continue
        return sorted(results, key=lambda r: (r.track, r.engine, r.model))

    def generate(
        self,
        results: Optional[List[ResultRecord]] = None,
        *,
        write_file: bool = True,
    ) -> str:
        if results is None:
            results = self.load_results()

        hw = self._detect_hardware()
        ver = self._detect_version()
        now = datetime.datetime.utcnow()

        lines = [
            f"# Squish Benchmark Report",
            f"",
            f"**Date**: {now.strftime('%Y-%m-%d %H:%M UTC')}  ",
            f"**Squish version**: {ver}  ",
            f"**Hardware**: {hw}  ",
            f"",
            f"---",
            f"",
            f"## Summary",
            f"",
        ]

        # Headline numbers per engine
        engines = sorted({r.engine for r in results})
        if engines:
            lines += ["| Engine | Quality | TPS | Tool exact-match | Agent completion |",
                      "|--------|---------|-----|-----------------|-----------------|"]
            for eng in engines:
                eng_results = [r for r in results if r.engine == eng]
                def _get(track, key, fmt="{:.1%}"):
                    vals = [r.metrics.get(key) for r in eng_results if r.track == track and r.metrics.get(key) is not None]
                    return fmt.format(vals[0]) if vals else "—"
                quality = _get("quality", "mmlu_acc", "{:.1%}")
                tps = _get("perf", "tps_mean", "{:.1f}")
                tool_em = _get("tools", "exact_match_pct", "{:.1%}")
                agent_cr = _get("agent", "completion_rate", "{:.1%}")
                lines.append(f"| {eng} | {quality} | {tps} | {tool_em} | {agent_cr} |")
            lines.append("")

        lines += ["---", ""]

        # Per-track sections
        for track_key, track_label in self.TRACK_LABELS.items():
            track_results = [r for r in results if r.track == track_key]
            if not track_results:
                continue
            lines += [f"## {track_label}", ""]
            for r in track_results:
                lines.append(f"**{r.engine}** / `{r.model}`")
                if r.metrics:
                    lines.append("")
                    lines.append("| Metric | Value |")
                    lines.append("|--------|-------|")
                    for k, v in r.metrics.items():
                        lines.append(f"| {k} | {v} |")
                lines.append("")

        lines += [
            "---",
            "",
            "## Methodology",
            "",
            f"- **Hardware**: {hw}",
            f"- **Squish version**: {ver}",
            "- All quality benchmarks use lm-eval with fixed random seed 42",
            "- Performance metrics are median of 5 runs (1 warmup discarded)",
            "- Tool benchmarks use `data/tool_schemas.json` canonical 20-schema set",
            "- Agent benchmarks use fixture replay (no live API calls)",
            "",
            "> Generated by `squish bench --report`",
        ]

        report = "\n".join(lines)

        if write_file:
            ts = now.strftime("%Y%m%d")
            out_dir = Path(self.config.output_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / f"benchmark_{ts}.md").write_text(report)

        return report
