# tests/test_bench_code.py
"""Unit tests for Track B code_bench.py (6+ tests)."""
from __future__ import annotations

import warnings
from unittest.mock import MagicMock, patch

import pytest

from squish.benchmarks.base import EngineConfig, ResultRecord, SQUISH_ENGINE
from squish.benchmarks.code_bench import (
    SANDBOX_WARNING,
    CODE_TASKS,
    CodeBenchConfig,
    CodeBenchRunner,
)


class TestCodeBenchConfig:
    def test_default_tasks(self):
        cfg = CodeBenchConfig()
        assert set(cfg.tasks) == set(CODE_TASKS.keys())

    def test_sandbox_defaults_false(self):
        cfg = CodeBenchConfig()
        assert cfg.sandbox is False

    def test_sandbox_enabled(self):
        cfg = CodeBenchConfig(sandbox=True)
        assert cfg.sandbox is True


class TestCodeBenchRunner:
    def _runner(self, sandbox: bool = False) -> CodeBenchRunner:
        return CodeBenchRunner(CodeBenchConfig(tasks=["humaneval"], sandbox=sandbox))

    def test_track_name(self):
        assert self._runner().track_name == "code"

    def test_run_without_sandbox_issues_warning(self):
        runner = self._runner(sandbox=False)
        with patch.object(runner, "_run_task", return_value={"humaneval_pass_at_1": 0.30}):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                runner.run(SQUISH_ENGINE, "qwen2.5:1.5b")
            assert any(SANDBOX_WARNING in str(warning.message) for warning in w)

    def test_run_with_sandbox_no_warning(self):
        runner = self._runner(sandbox=True)
        with patch.object(runner, "_run_task", return_value={"humaneval_pass_at_1": 0.30}):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                runner.run(SQUISH_ENGINE, "qwen2.5:1.5b")
        sandbox_warnings = [x for x in w if SANDBOX_WARNING in str(x.message)]
        assert len(sandbox_warnings) == 0

    def test_run_returns_result_record(self):
        runner = self._runner()
        with patch.object(runner, "_run_task", return_value={"humaneval_pass_at_1": 0.35}):
            with warnings.catch_warnings(record=True):
                warnings.simplefilter("always")
                result = runner.run(SQUISH_ENGINE, "test_model")
        assert isinstance(result, ResultRecord)
        assert result.track == "code"

    def test_sandbox_flag_in_metrics(self):
        runner = self._runner(sandbox=True)
        with patch.object(runner, "_run_task", return_value={"humaneval_pass_at_1": 0.35}):
            result = runner.run(SQUISH_ENGINE, "test_model")
        assert result.metrics["sandbox_enabled"] is True

    def test_run_task_missing_lm_eval_returns_error_key(self):
        runner = self._runner()
        with patch.dict("sys.modules", {"lm_eval": None}):
            result = runner._run_task(SQUISH_ENGINE, "test_model", "humaneval", None)
        assert "humaneval_error" in result

    def test_limit_passed_to_run_task(self):
        runner = self._runner()
        captured = {}
        def _fake(engine, model, task, limit):
            captured["limit"] = limit
            return {"humaneval_pass_at_1": 0.3}
        with patch.object(runner, "_run_task", side_effect=_fake):
            with warnings.catch_warnings(record=True):
                warnings.simplefilter("always")
                runner.run(SQUISH_ENGINE, "m", limit=25)
        assert captured["limit"] == 25
