# squish/benchmarks/code_bench.py
"""Track B — Code Generation benchmark.

Runs HumanEval and MBPP via lm-eval (pass@1 metric).
Code execution requires explicit --sandbox opt-in for safety.
"""
from __future__ import annotations

__all__ = ["CodeBenchConfig", "CodeBenchRunner", "SANDBOX_WARNING"]

import datetime
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from squish.benchmarks.base import BenchmarkRunner, EngineConfig, ResultRecord

SANDBOX_WARNING = (
    "Code generation benchmarks produce output to JSON but will not execute "
    "generated code. Pass --sandbox to run HumanEval/MBPP execution."
)

CODE_TASKS = {
    "humaneval": {"n_shot": 0, "metric": "pass@1"},
    "mbpp":      {"n_shot": 3, "metric": "pass@1"},
}


@dataclass
class CodeBenchConfig:
    """Configuration for Track B code generation benchmark."""
    tasks: List[str] = field(default_factory=lambda: list(CODE_TASKS.keys()))
    sandbox: bool = False          # must be explicitly True to execute code
    limit: Optional[int] = None
    output_dir: str = "eval_output"
    seed: int = 42


class CodeBenchRunner(BenchmarkRunner):
    """Track B: runs code generation benchmarks (HumanEval, MBPP)."""

    def __init__(self, config: CodeBenchConfig) -> None:
        self._config = config

    @property
    def track_name(self) -> str:
        return "code"

    def run(
        self,
        engine: EngineConfig,
        model: str,
        *,
        limit: Optional[int] = None,
    ) -> ResultRecord:
        effective_limit = limit if limit is not None else self._config.limit

        if not self._config.sandbox:
            import warnings
            warnings.warn(SANDBOX_WARNING, UserWarning, stacklevel=2)

        metrics: Dict[str, Any] = {
            "sandbox_enabled": self._config.sandbox,
        }

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
                "sandbox": self._config.sandbox,
                "limit": effective_limit,
                "seed": self._config.seed,
            },
        )

    def _run_task(
        self,
        engine: EngineConfig,
        model: str,
        task: str,
        limit: Optional[int],
    ) -> Dict[str, Any]:
        """Run a single code generation task via lm-eval."""
        try:
            import lm_eval  # noqa: PLC0415

            eval_kwargs: Dict[str, Any] = {
                "model": "local-completions",
                "model_args": (
                    f"base_url={engine.base_url}/v1,"
                    f"model={model},"
                    f"tokenized_requests=False"
                ),
                "tasks": [task],
                "num_fewshot": CODE_TASKS.get(task, {}).get("n_shot", 0),
                "seed": self._config.seed,
            }
            if limit:
                eval_kwargs["limit"] = limit
            # Only allow code execution in sandbox mode
            if not self._config.sandbox:
                eval_kwargs["predict_only"] = True

            results = lm_eval.simple_evaluate(**eval_kwargs)
            task_results = results.get("results", {}).get(task, {})
            metric_key = CODE_TASKS.get(task, {}).get("metric", "pass@1")
            value = task_results.get(metric_key) or task_results.get(metric_key + ",none", 0.0)
            return {f"{task}_{metric_key.replace('@', '_at_')}": round(float(value), 4)}

        except ImportError:
            return {f"{task}_error": "lm_eval not installed"}
