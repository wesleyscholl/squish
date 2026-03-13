# squish/benchmarks/quality_bench.py
"""Track A — Quality / Normal Text benchmark.

Wraps the squish_lm_eval.py backend to run MMLU, ARC, HellaSwag,
WinoGrande, TruthfulQA, and GSM8K against a running Squish server.
"""
from __future__ import annotations

__all__ = ["QualityBenchConfig", "QualityBenchRunner"]

import datetime
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from squish.benchmarks.base import BenchmarkRunner, EngineConfig, ResultRecord


QUALITY_TASKS = {
    "mmlu":           {"n_shot": 5,  "metric": "acc"},
    "arc_challenge":  {"n_shot": 25, "metric": "acc_norm"},
    "hellaswag":      {"n_shot": 10, "metric": "acc_norm"},
    "winogrande":     {"n_shot": 5,  "metric": "acc"},
    "truthfulqa_mc1": {"n_shot": 0,  "metric": "acc"},
    "gsm8k":          {"n_shot": 8,  "metric": "exact_match"},
}


@dataclass
class QualityBenchConfig:
    """Configuration for Track A quality benchmark."""
    tasks: List[str] = field(default_factory=lambda: list(QUALITY_TASKS.keys()))
    limit: Optional[int] = None          # per-task sample cap (None = full)
    output_dir: str = "eval_output"
    seed: int = 42
    batch_size: int = 1


class QualityBenchRunner(BenchmarkRunner):
    """Track A: runs quality benchmarks via lm-eval backend."""

    def __init__(self, config: QualityBenchConfig) -> None:
        self._config = config

    @property
    def track_name(self) -> str:
        return "quality"

    def run(
        self,
        engine: EngineConfig,
        model: str,
        *,
        limit: Optional[int] = None,
    ) -> ResultRecord:
        """Run quality benchmarks; returns ResultRecord with per-task metrics."""
        effective_limit = limit if limit is not None else self._config.limit
        metrics: Dict[str, Any] = {}

        for task in self._config.tasks:
            try:
                task_metrics = self._run_task(engine, model, task, effective_limit)
                metrics.update(task_metrics)
            except Exception as exc:  # noqa: BLE001
                metrics[f"{task}_error"] = str(exc)

        return ResultRecord(
            track=self.track_name,
            engine=engine.name,
            model=model,
            metrics=metrics,
            metadata={
                "tasks": self._config.tasks,
                "limit": effective_limit,
                "seed": self._config.seed,
                "lm_eval_backend": "squish_lm_eval",
            },
        )

    def _run_task(
        self,
        engine: EngineConfig,
        model: str,
        task: str,
        limit: Optional[int],
    ) -> Dict[str, Any]:
        """Run a single lm-eval task; returns {task_metric_name: value}."""
        try:
            import lm_eval  # noqa: PLC0415

            # Build evaluator kwargs
            eval_kwargs: Dict[str, Any] = {
                "model": "local-completions",
                "model_args": (
                    f"base_url={engine.base_url}/v1,"
                    f"model={model},"
                    f"tokenized_requests=False"
                ),
                "tasks": [task],
                "num_fewshot": QUALITY_TASKS.get(task, {}).get("n_shot", 0),
                "batch_size": self._config.batch_size,
                "seed": self._config.seed,
            }
            if limit:
                eval_kwargs["limit"] = limit

            results = lm_eval.simple_evaluate(**eval_kwargs)
            task_results = results.get("results", {}).get(task, {})
            metric_key = QUALITY_TASKS.get(task, {}).get("metric", "acc")
            value = task_results.get(metric_key) or task_results.get(metric_key + ",none", 0.0)
            return {f"{task}_{metric_key}": round(float(value), 4)}

        except ImportError:
            return {f"{task}_error": "lm_eval not installed"}

    def output_path_for(self, engine: str, model: str) -> Path:
        ts = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        safe_model = model.replace("/", "_").replace(":", "_")
        return Path(self._config.output_dir) / f"quality_{safe_model}_{engine}_{ts}.json"
