# squish/benchmarks/agent_bench.py
"""Track D — Agentic Tasks benchmark.

Runs a full agent loop against 20 hand-authored scenarios with
fixture replay (no live API calls or filesystem side effects).
"""
from __future__ import annotations

__all__ = [
    "AgentBenchConfig",
    "AgentScenario",
    "ToolFixtureReplay",
    "AgentBenchRunner",
]

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from squish.benchmarks.base import BenchmarkRunner, EngineClient, EngineConfig, ResultRecord


def _load_scenarios(path: Optional[str] = None) -> List[Dict[str, Any]]:
    if path is None:
        path = str(Path(__file__).parent / "data" / "agent_scenarios.json")
    with open(path) as f:
        return json.load(f)


@dataclass
class AgentScenario:
    """A single agentic task scenario."""
    id: str
    category: str
    goal: str
    tools: List[Dict[str, Any]]
    tool_fixtures: Dict[str, Any]      # {tool_name: {call_sig_tuple: response}}
    expected_sequence: List[str]       # ordered list of expected tool names
    expected_final_answer: str         # regex or substring to match

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "AgentScenario":
        return cls(
            id=d["id"],
            category=d["category"],
            goal=d["goal"],
            tools=d["tools"],
            tool_fixtures=d.get("tool_fixtures", {}),
            expected_sequence=d.get("expected_sequence", []),
            expected_final_answer=d.get("expected_final_answer", ""),
        )


class ToolFixtureReplay:
    """Replays tool call responses from a fixture dict."""

    def __init__(self, fixtures: Dict[str, Any]) -> None:
        self._fixtures = fixtures

    def call(self, tool_name: str, args: Dict[str, Any]) -> Any:
        """Return the fixture response for (tool_name, args), or a default."""
        tool_fixtures = self._fixtures.get(tool_name, {})
        # Try exact arg match first
        args_key = json.dumps(args, sort_keys=True)
        if args_key in tool_fixtures:
            return tool_fixtures[args_key]
        # Fall back to first available response
        if tool_fixtures:
            return next(iter(tool_fixtures.values()))
        return {"status": "ok", "result": f"[fixture not found for {tool_name}]"}


@dataclass
class AgentBenchConfig:
    """Configuration for Track D agent benchmark."""
    scenarios_path: Optional[str] = None
    max_turns: int = 10
    max_tokens_per_turn: int = 512
    limit: Optional[int] = None
    output_dir: str = "eval_output"


class AgentBenchRunner(BenchmarkRunner):
    """Track D: agentic task benchmark with fixture replay."""

    def __init__(self, config: AgentBenchConfig) -> None:
        self._config = config

    @property
    def track_name(self) -> str:
        return "agent"

    def run(
        self,
        engine: EngineConfig,
        model: str,
        *,
        limit: Optional[int] = None,
    ) -> ResultRecord:
        scenarios_data = _load_scenarios(self._config.scenarios_path)
        effective_limit = limit if limit is not None else self._config.limit
        if effective_limit:
            scenarios_data = scenarios_data[:effective_limit]

        scenarios = [AgentScenario.from_dict(d) for d in scenarios_data]
        client = EngineClient(engine)

        task_results = []
        for scenario in scenarios:
            result = self._run_scenario(client, model, scenario)
            task_results.append(result)

        n_total = len(task_results)
        n_complete = sum(1 for r in task_results if r.get("completed", False))
        seq_accuracies = [r.get("sequence_accuracy", 0.0) for r in task_results]
        efficiencies = [r.get("step_efficiency", 1.0) for r in task_results if r.get("actual_steps", 0) > 0]
        total_tokens = sum(r.get("tokens_consumed", 0) for r in task_results)

        metrics = {
            "total_scenarios": n_total,
            "completion_rate": round(n_complete / max(n_total, 1), 4),
            "sequence_accuracy": round(sum(seq_accuracies) / max(n_total, 1), 4),
            "mean_step_efficiency": round(sum(efficiencies) / max(len(efficiencies), 1), 4),
            "total_tokens_consumed": total_tokens,
        }

        return ResultRecord(
            track=self.track_name,
            engine=engine.name,
            model=model,
            metrics=metrics,
            metadata={
                "n_scenarios": n_total,
                "max_turns": self._config.max_turns,
                "limit": effective_limit,
            },
        )

    def _run_scenario(
        self,
        client: EngineClient,
        model: str,
        scenario: AgentScenario,
    ) -> Dict[str, Any]:
        """Run one scenario; returns result dict."""
        replay = ToolFixtureReplay(scenario.tool_fixtures)
        messages = [
            {"role": "system", "content": "You are a helpful assistant. Use the available tools to complete the task."},
            {"role": "user", "content": scenario.goal},
        ]
        tools = [{"type": "function", "function": t} for t in scenario.tools]

        actual_sequence: List[str] = []
        tokens_consumed = 0
        final_answer = ""
        completed = False

        for _turn in range(self._config.max_turns):
            try:
                resp = client.chat(
                    model=model,
                    messages=messages,
                    tools=tools,
                    max_tokens=self._config.max_tokens_per_turn,
                    temperature=0.0,
                )
            except Exception:  # noqa: BLE001
                break

            usage = resp.get("usage", {})
            tokens_consumed += usage.get("total_tokens", 0)

            choices = resp.get("choices", [])
            if not choices:
                break

            msg = choices[0].get("message", {})
            finish_reason = choices[0].get("finish_reason", "")
            tool_calls = msg.get("tool_calls", [])

            if tool_calls:
                # Execute each tool call via fixture replay
                messages.append({"role": "assistant", "content": None, "tool_calls": tool_calls})
                for tc in tool_calls:
                    fn = tc.get("function", {})
                    name = fn.get("name", "")
                    args_raw = fn.get("arguments", "{}")
                    try:
                        args = json.loads(args_raw)
                    except (json.JSONDecodeError, TypeError):
                        args = {}
                    actual_sequence.append(name)
                    result = replay.call(name, args)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.get("id", ""),
                        "content": json.dumps(result),
                    })
            else:
                # No tool call — this is the final answer
                final_answer = msg.get("content", "") or ""
                completed = True
                break

            if finish_reason == "stop":
                final_answer = msg.get("content", "") or ""
                completed = True
                break

        # Score the run
        expected = scenario.expected_sequence
        if expected:
            matched = sum(
                1 for i, name in enumerate(expected)
                if i < len(actual_sequence) and actual_sequence[i] == name
            )
            seq_accuracy = matched / len(expected)
        else:
            seq_accuracy = 1.0

        optimal = max(len(expected), 1)
        actual_steps = max(len(actual_sequence), 1)
        step_efficiency = optimal / actual_steps  # <= 1.0 is efficient

        # Check final answer
        if scenario.expected_final_answer and final_answer:
            import re  # noqa: PLC0415
            answer_match = bool(re.search(
                scenario.expected_final_answer, final_answer, re.IGNORECASE
            ))
        else:
            answer_match = completed

        return {
            "scenario_id": scenario.id,
            "category": scenario.category,
            "completed": answer_match,
            "sequence_accuracy": round(seq_accuracy, 4),
            "step_efficiency": round(step_efficiency, 4),
            "actual_steps": len(actual_sequence),
            "optimal_steps": len(expected),
            "tokens_consumed": tokens_consumed,
            "actual_sequence": actual_sequence,
        }
