# tests/test_bench_tool.py
"""Unit tests for Track C tool_bench.py (10+ tests)."""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from squish.benchmarks.base import EngineConfig, ResultRecord, SQUISH_ENGINE
from squish.benchmarks.tool_bench import (
    ToolBenchConfig,
    ToolBenchRunner,
    ToolEvaluator,
)


def _make_response(tool_name: str, args: dict, valid: bool = True) -> dict:
    """Build a synthetic chat completion response with a tool call."""
    if not valid:
        return {"choices": [{"message": {"content": "I cannot help with that."}}]}
    return {
        "choices": [{
            "message": {
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": tool_name,
                        "arguments": json.dumps(args),
                    }
                }]
            }
        }]
    }


class TestToolEvaluatorSchemaCompliance:
    def test_valid_tool_call_is_compliant(self):
        resp = _make_response("calculator", {"a": 1, "b": 2, "operation": "add"})
        assert ToolEvaluator.schema_compliance(resp) is True

    def test_missing_tool_calls_not_compliant(self):
        resp = _make_response("", {}, valid=False)
        assert ToolEvaluator.schema_compliance(resp) is False

    def test_empty_choices_not_compliant(self):
        assert ToolEvaluator.schema_compliance({"choices": []}) is False

    def test_invalid_json_args_not_compliant(self):
        resp = {"choices": [{"message": {"tool_calls": [{"function": {"name": "f", "arguments": "not-json"}}]}}]}
        assert ToolEvaluator.schema_compliance(resp) is False


class TestToolEvaluatorNameMatch:
    def test_correct_name_matches(self):
        resp = _make_response("calculator", {"a": 1})
        assert ToolEvaluator.function_name_match(resp, "calculator") is True

    def test_wrong_name_no_match(self):
        resp = _make_response("calculator", {"a": 1})
        assert ToolEvaluator.function_name_match(resp, "file_read") is False

    def test_empty_choices_no_match(self):
        assert ToolEvaluator.function_name_match({"choices": []}, "any") is False


class TestToolEvaluatorArgumentMatch:
    def test_all_required_args_present(self):
        resp = _make_response("calc", {"a": 1, "b": 2, "operation": "add"})
        matched, total = ToolEvaluator.argument_match(resp, {"a": 1, "b": 2, "operation": "add"})
        assert matched == 3
        assert total == 3

    def test_partial_args(self):
        resp = _make_response("calc", {"a": 1})
        matched, total = ToolEvaluator.argument_match(resp, {"a": 1, "b": 2})
        assert matched == 1
        assert total == 2


class TestToolEvaluatorExactMatch:
    def test_exact_match_all_args(self):
        resp = _make_response("calc", {"a": 1, "b": 2})
        assert ToolEvaluator.exact_match(resp, {"name": "calc", "arguments": {"a": 1, "b": 2}}) is True

    def test_no_exact_match_wrong_name(self):
        resp = _make_response("calc", {"a": 1, "b": 2})
        assert ToolEvaluator.exact_match(resp, {"name": "other", "arguments": {"a": 1, "b": 2}}) is False


class TestToolBenchRunner:
    def test_track_name(self):
        runner = ToolBenchRunner(ToolBenchConfig())
        assert runner.track_name == "tools"

    def test_run_returns_result_record(self):
        runner = ToolBenchRunner(ToolBenchConfig())
        with patch.object(runner, "_run_canonical", return_value=[
            {"schema_compliance": True, "name_match": True, "arg_match_ratio": 1.0, "exact_match": True},
        ]):
            result = runner.run(SQUISH_ENGINE, "test_model")
        assert isinstance(result, ResultRecord)
        assert result.track == "tools"

    def test_metrics_include_pct_keys(self):
        runner = ToolBenchRunner(ToolBenchConfig())
        with patch.object(runner, "_run_canonical", return_value=[
            {"schema_compliance": True, "name_match": True, "arg_match_ratio": 1.0, "exact_match": True},
        ]):
            result = runner.run(SQUISH_ENGINE, "test_model")
        assert "schema_compliance_pct" in result.metrics
        assert "exact_match_pct" in result.metrics
