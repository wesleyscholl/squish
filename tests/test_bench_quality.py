# tests/test_bench_quality.py
"""Unit tests for Track A quality_bench.py (8+ tests)."""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from squish.benchmarks.base import EngineConfig, ResultRecord, SQUISH_ENGINE
from squish.benchmarks.quality_bench import (
    QUALITY_TASKS,
    QualityBenchConfig,
    QualityBenchRunner,
)


class TestQualityBenchConfig:
    def test_default_tasks_are_all_quality_tasks(self):
        cfg = QualityBenchConfig()
        assert set(cfg.tasks) == set(QUALITY_TASKS.keys())

    def test_limit_defaults_to_none(self):
        cfg = QualityBenchConfig()
        assert cfg.limit is None

    def test_custom_tasks(self):
        cfg = QualityBenchConfig(tasks=["mmlu", "gsm8k"])
        assert cfg.tasks == ["mmlu", "gsm8k"]

    def test_seed_default(self):
        cfg = QualityBenchConfig()
        assert cfg.seed == 42


class TestQualityBenchRunner:
    def _runner(self) -> QualityBenchRunner:
        return QualityBenchRunner(QualityBenchConfig(tasks=["mmlu"]))

    def test_track_name(self):
        assert self._runner().track_name == "quality"

    def test_run_returns_result_record(self):
        runner = self._runner()
        with patch.object(runner, "_run_task", return_value={"mmlu_acc": 0.55}):
            result = runner.run(SQUISH_ENGINE, "qwen2.5:1.5b")
        assert isinstance(result, ResultRecord)
        assert result.track == "quality"
        assert result.engine == "squish"
        assert result.model == "qwen2.5:1.5b"

    def test_run_includes_task_metrics(self):
        runner = self._runner()
        with patch.object(runner, "_run_task", return_value={"mmlu_acc": 0.60}):
            result = runner.run(SQUISH_ENGINE, "qwen2.5:1.5b")
        assert "mmlu_acc" in result.metrics

    def test_run_handles_task_error_gracefully(self):
        runner = self._runner()
        with patch.object(runner, "_run_task", side_effect=RuntimeError("lm_eval missing")):
            result = runner.run(SQUISH_ENGINE, "test_model")
        assert "mmlu_error" in result.metrics

    def test_limit_override_in_run(self):
        runner = self._runner()
        captured = {}
        def _fake_run_task(engine, model, task, limit):
            captured["limit"] = limit
            return {"mmlu_acc": 0.50}
        with patch.object(runner, "_run_task", side_effect=_fake_run_task):
            runner.run(SQUISH_ENGINE, "test_model", limit=50)
        assert captured["limit"] == 50

    def test_run_task_missing_lm_eval_returns_error_key(self):
        runner = self._runner()
        with patch.dict("sys.modules", {"lm_eval": None}):
            result = runner._run_task(SQUISH_ENGINE, "test_model", "mmlu", None)
        assert "mmlu_error" in result

    def test_output_path_contains_track_name(self):
        runner = self._runner()
        path = runner.output_path_for("squish", "qwen2.5:1.5b")
        assert "quality" in str(path)

    def test_metadata_includes_tasks_and_seed(self):
        runner = self._runner()
        with patch.object(runner, "_run_task", return_value={"mmlu_acc": 0.55}):
            result = runner.run(SQUISH_ENGINE, "test_model")
        assert "tasks" in result.metadata
        assert "seed" in result.metadata
