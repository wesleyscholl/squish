# squish/benchmarks/compare.py
"""Cross-engine result table generator.

Reads ResultRecord JSON files from eval_output/ and builds:
- A markdown comparison table (docs/comparison_<date>.md)
- A CSV file (eval_output/comparison_<date>.csv)
"""
from __future__ import annotations

__all__ = ["CompareConfig", "ResultComparator"]

import csv
import datetime
import io
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from squish.benchmarks.base import ResultRecord


@dataclass
class CompareConfig:
    """Configuration for the comparison table generator."""
    input_dir: str = "eval_output"
    output_dir: str = "docs"
    tracks: List[str] = field(default_factory=lambda: ["quality", "code", "tools", "agent", "perf"])
    engines: List[str] = field(default_factory=list)   # empty = all found
    date_filter: str = ""                               # prefix-match on timestamp


class ResultComparator:
    """Loads ResultRecord files and generates comparison tables."""

    def __init__(self, config: CompareConfig) -> None:
        self.config = config

    def load_results(self) -> List[ResultRecord]:
        """Load all JSON result files from input_dir."""
        results = []
        for p in Path(self.config.input_dir).glob("*.json"):
            try:
                r = ResultRecord.load(p)
                if self.config.tracks and r.track not in self.config.tracks:
                    continue
                if self.config.engines and r.engine not in self.config.engines:
                    continue
                if self.config.date_filter and not r.timestamp.startswith(self.config.date_filter):
                    continue
                results.append(r)
            except Exception:  # noqa: BLE001
                continue
        return results

    def to_markdown(self, results: List[ResultRecord]) -> str:
        """Build a markdown comparison table from results."""
        if not results:
            return "<!-- No benchmark results found -->\n"

        lines = [
            f"# Squish Cross-Engine Benchmark Comparison",
            f"",
            f"Generated: {datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}",
            f"",
        ]

        for track in self.config.tracks:
            track_results = [r for r in results if r.track == track]
            if not track_results:
                continue
            lines.append(f"## Track {track.title()}")
            lines.append("")

            # Build column headers from all metric keys
            all_keys: list = []
            for r in track_results:
                for k in r.metrics:
                    if k not in all_keys:
                        all_keys.append(k)

            header = "| engine | model | " + " | ".join(all_keys) + " |"
            sep = "|--------|-------|" + "|---------|" * len(all_keys)
            lines.append(header)
            lines.append(sep)
            for r in track_results:
                vals = [str(r.metrics.get(k, "—")) for k in all_keys]
                lines.append(f"| {r.engine} | {r.model} | " + " | ".join(vals) + " |")
            lines.append("")

        return "\n".join(lines)

    def to_csv(self, results: List[ResultRecord]) -> str:
        """Build a CSV string from results."""
        if not results:
            return "track,engine,model,timestamp\n"

        all_metric_keys: list = []
        for r in results:
            for k in r.metrics:
                if k not in all_metric_keys:
                    all_metric_keys.append(k)

        out = io.StringIO()
        writer = csv.writer(out)
        writer.writerow(["track", "engine", "model", "timestamp"] + all_metric_keys)
        for r in results:
            row = [r.track, r.engine, r.model, r.timestamp]
            row += [r.metrics.get(k, "") for k in all_metric_keys]
            writer.writerow(row)
        return out.getvalue()

    def generate(self, *, write_files: bool = True) -> Dict[str, str]:
        """Load results and generate comparison outputs.

        Returns dict with 'markdown' and 'csv' keys.
        """
        results = self.load_results()
        md = self.to_markdown(results)
        csv_str = self.to_csv(results)

        if write_files:
            ts = datetime.datetime.utcnow().strftime("%Y%m%d")
            out_dir = Path(self.config.output_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / f"comparison_{ts}.md").write_text(md)
            eval_dir = Path(self.config.input_dir)
            eval_dir.mkdir(parents=True, exist_ok=True)
            (eval_dir / f"comparison_{ts}.csv").write_text(csv_str)

        return {"markdown": md, "csv": csv_str}
