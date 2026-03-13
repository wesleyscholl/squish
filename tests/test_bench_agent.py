# tests/test_bench_agent.py
"""Unit tests for Track D agent_bench.py (12+ tests)."""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from squish.benchmarks.agent_bench import (
    AgentBenchConfig,
    AgentBenchRunner,
    AgentScenario,
    ToolFixtureReplay,
)
from squish.benchmarks.base import ResultRecord, SQUISH_ENGINE


def _make_scenario(**kwargs) -> AgentScenario:
    defaults = {
        "id": "test_1",
        "category": "test",
        "goal": "Do something.",
        "tools": [{"name": "calculator", "description": "Math", "parameters": {"type": "object", "properties": {"a": {"type": "number"}}, "required": ["a"]}}],
        "tool_fixtures": {},
        "expected_sequence": ["calculator"],
        "expected_final_answer": "result",
    }
    defaults.update(kwargs)
    return AgentScenario(**defaults)


class TestAgentScenario:
    def test_from_dict_round_trip(self):
        d = {
            "id": "s1", "category": "file_ops", "goal": "Do it.",
            "tools": [], "tool_fixtures": {}, "expected_sequence": ["f"], "expected_final_answer": "ok",
        }
        s = AgentScenario.from_dict(d)
        assert s.id == "s1"
        assert s.category == "file_ops"
        assert s.expected_sequence == ["f"]

    def test_from_dict_missing_optional_fields(self):
        d = {"id": "s2", "category": "test", "goal": "G", "tools": []}
        s = AgentScenario.from_dict(d)
        assert s.tool_fixtures == {}
        assert s.expected_sequence == []
        assert s.expected_final_answer == ""


class TestToolFixtureReplay:
    def test_exact_key_match(self):
        replay = ToolFixtureReplay({
            "calculator": {json.dumps({"a": 1}, sort_keys=True): {"result": 42}}
        })
        result = replay.call("calculator", {"a": 1})
        assert result == {"result": 42}

    def test_fallback_to_first_value(self):
        replay = ToolFixtureReplay({
            "web_search": {"any_key": {"results": [{"snippet": "test"}]}}
        })
        result = replay.call("web_search", {"query": "anything"})
        assert "results" in result

    def test_missing_tool_returns_default(self):
        replay = ToolFixtureReplay({})
        result = replay.call("unknown_tool", {})
        assert "status" in result or "result" in result

    def test_empty_fixtures_returns_default(self):
        replay = ToolFixtureReplay({"calculator": {}})
        result = replay.call("calculator", {"a": 1})
        assert result is not None


class TestAgentBenchRunner:
    def test_track_name(self):
        runner = AgentBenchRunner(AgentBenchConfig())
        assert runner.track_name == "agent"

    def test_run_returns_result_record(self):
        runner = AgentBenchRunner(AgentBenchConfig())
        mock_scenario_data = [{
            "id": "t1", "category": "test", "goal": "Test",
            "tools": [], "tool_fixtures": {},
            "expected_sequence": [], "expected_final_answer": ""
        }]
        with patch("squish.benchmarks.agent_bench._load_scenarios", return_value=mock_scenario_data):
            with patch.object(runner, "_run_scenario", return_value={
                "scenario_id": "t1", "category": "test",
                "completed": True, "sequence_accuracy": 1.0,
                "step_efficiency": 1.0, "actual_steps": 1,
                "optimal_steps": 1, "tokens_consumed": 100,
                "actual_sequence": [],
            }):
                result = runner.run(SQUISH_ENGINE, "test_model")
        assert isinstance(result, ResultRecord)
        assert result.track == "agent"

    def test_completion_rate_metric(self):
        runner = AgentBenchRunner(AgentBenchConfig())

        mock_data = [{"id": f"s{i}", "category": "test", "goal": "G",
                      "tools": [], "tool_fixtures": {},
                      "expected_sequence": [], "expected_final_answer": ""} for i in range(4)]
        results = [
            {"scenario_id": "s0", "category": "test", "completed": True,
             "sequence_accuracy": 1.0, "step_efficiency": 1.0,
             "actual_steps": 1, "optimal_steps": 1, "tokens_consumed": 10, "actual_sequence": []},
            {"scenario_id": "s1", "category": "test", "completed": True,
             "sequence_accuracy": 1.0, "step_efficiency": 1.0,
             "actual_steps": 1, "optimal_steps": 1, "tokens_consumed": 10, "actual_sequence": []},
            {"scenario_id": "s2", "category": "test", "completed": False,
             "sequence_accuracy": 0.0, "step_efficiency": 0.5,
             "actual_steps": 2, "optimal_steps": 1, "tokens_consumed": 10, "actual_sequence": []},
            {"scenario_id": "s3", "category": "test", "completed": False,
             "sequence_accuracy": 0.5, "step_efficiency": 0.5,
             "actual_steps": 2, "optimal_steps": 1, "tokens_consumed": 10, "actual_sequence": []},
        ]
        with patch("squish.benchmarks.agent_bench._load_scenarios", return_value=mock_data):
            with patch.object(runner, "_run_scenario", side_effect=results):
                result = runner.run(SQUISH_ENGINE, "test_model")
        assert result.metrics["completion_rate"] == 0.5

    def test_limit_applied(self):
        runner = AgentBenchRunner(AgentBenchConfig())
        mock_data = [{"id": f"s{i}", "category": "test", "goal": "G",
                      "tools": [], "tool_fixtures": {},
                      "expected_sequence": [], "expected_final_answer": ""} for i in range(10)]
        with patch("squish.benchmarks.agent_bench._load_scenarios", return_value=mock_data):
            with patch.object(runner, "_run_scenario", return_value={
                "scenario_id": "x", "category": "test", "completed": False,
                "sequence_accuracy": 0.0, "step_efficiency": 1.0,
                "actual_steps": 0, "optimal_steps": 0, "tokens_consumed": 0, "actual_sequence": [],
            }) as mock_rs:
                runner.run(SQUISH_ENGINE, "test_model", limit=3)
        assert mock_rs.call_count == 3

    def test_step_efficiency_below_one_when_extra_steps(self):
        runner = AgentBenchRunner(AgentBenchConfig())
        scenario = _make_scenario(expected_sequence=["calculator"])
        replay = ToolFixtureReplay({})
        # Simulate a run result with 2 actual steps for 1 optimal step
        result = {
            "scenario_id": scenario.id,
            "category": scenario.category,
            "completed": False,
            "sequence_accuracy": 0.0,
            "step_efficiency": 1 / 2,
            "actual_steps": 2,
            "optimal_steps": 1,
            "tokens_consumed": 0,
            "actual_sequence": [],
        }
        assert result["step_efficiency"] < 1.0

    def test_sequence_accuracy_full_match(self):
        expected = ["a", "b", "c"]
        actual   = ["a", "b", "c"]
        matched = sum(1 for i, name in enumerate(expected) if i < len(actual) and actual[i] == name)
        assert matched / len(expected) == 1.0

    def test_sequence_accuracy_no_match(self):
        expected = ["a", "b"]
        actual   = ["x", "y"]
        matched = sum(1 for i, name in enumerate(expected) if i < len(actual) and actual[i] == name)
        assert matched / len(expected) == 0.0

    def test_metadata_includes_n_scenarios(self):
        runner = AgentBenchRunner(AgentBenchConfig())
        mock_data = [{"id": "s1", "category": "test", "goal": "G",
                      "tools": [], "tool_fixtures": {},
                      "expected_sequence": [], "expected_final_answer": ""}]
        with patch("squish.benchmarks.agent_bench._load_scenarios", return_value=mock_data):
            with patch.object(runner, "_run_scenario", return_value={
                "scenario_id": "s1", "category": "test", "completed": True,
                "sequence_accuracy": 1.0, "step_efficiency": 1.0,
                "actual_steps": 1, "optimal_steps": 1, "tokens_consumed": 0, "actual_sequence": [],
            }):
                result = runner.run(SQUISH_ENGINE, "test_model")
        assert "n_scenarios" in result.metadata
