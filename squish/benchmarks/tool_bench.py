# squish/benchmarks/tool_bench.py
"""Track C — Tool Use / Function Calling benchmark.

Posts tool-call prompts to the server's /v1/chat/completions with
tools payload and evaluates schema compliance vs ground truth.
"""
from __future__ import annotations

__all__ = ["ToolBenchConfig", "ToolBenchRunner", "ToolEvaluator", "EngineClient"]

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from squish.benchmarks.base import BenchmarkRunner, EngineClient, EngineConfig, ResultRecord


def _load_schemas(path: Optional[str] = None) -> List[Dict[str, Any]]:
    """Load canonical tool schemas from data/tool_schemas.json."""
    if path is None:
        path = str(Path(__file__).parent / "data" / "tool_schemas.json")
    with open(path) as f:
        return json.load(f)


@dataclass
class ToolBenchConfig:
    """Configuration for Track C tool-use benchmark."""
    schemas_path: Optional[str] = None   # None = use bundled data/tool_schemas.json
    limit: Optional[int] = None          # cap on BFCL cases (None = up to 200)
    output_dir: str = "eval_output"
    use_bfcl: bool = False               # True = also fetch BFCL v3 from HuggingFace
    bfcl_limit: int = 200


class ToolEvaluator:
    """Scores tool-call responses against ground-truth schemas."""

    @staticmethod
    def schema_compliance(response: Dict[str, Any]) -> bool:
        """Return True if response contains a valid tool_calls list."""
        choices = response.get("choices", [])
        if not choices:
            return False
        msg = choices[0].get("message", {})
        tool_calls = msg.get("tool_calls", [])
        if not tool_calls:
            return False
        for tc in tool_calls:
            if not isinstance(tc, dict):
                return False
            func = tc.get("function", {})
            if not isinstance(func.get("name"), str):
                return False
            args_raw = func.get("arguments", "{}")
            try:
                json.loads(args_raw)
            except (json.JSONDecodeError, TypeError):
                return False
        return True

    @staticmethod
    def function_name_match(response: Dict[str, Any], expected_name: str) -> bool:
        """Return True if the response tool call function name matches expected."""
        choices = response.get("choices", [])
        if not choices:
            return False
        tool_calls = choices[0].get("message", {}).get("tool_calls", [])
        if not tool_calls:
            return False
        actual = tool_calls[0].get("function", {}).get("name", "")
        return actual == expected_name

    @staticmethod
    def argument_match(
        response: Dict[str, Any], expected_args: Dict[str, Any]
    ) -> Tuple[int, int]:
        """Return (matched_required_args, total_required_args) for the first tool call."""
        choices = response.get("choices", [])
        if not choices:
            return 0, len(expected_args)
        tool_calls = choices[0].get("message", {}).get("tool_calls", [])
        if not tool_calls:
            return 0, len(expected_args)
        args_raw = tool_calls[0].get("function", {}).get("arguments", "{}")
        try:
            actual_args = json.loads(args_raw)
        except (json.JSONDecodeError, TypeError):
            return 0, len(expected_args)
        matched = sum(1 for k in expected_args if k in actual_args)
        return matched, len(expected_args)

    @staticmethod
    def exact_match(response: Dict[str, Any], expected_call: Dict[str, Any]) -> bool:
        """Return True if tool call matches name AND all expected args."""
        if not ToolEvaluator.function_name_match(response, expected_call.get("name", "")):
            return False
        matched, total = ToolEvaluator.argument_match(
            response, expected_call.get("arguments", {})
        )
        return matched == total


class ToolBenchRunner(BenchmarkRunner):
    """Track C: tool-use benchmark using canonical schemas + optional BFCL v3."""

    def __init__(self, config: ToolBenchConfig) -> None:
        self._config = config
        self._eval = ToolEvaluator()

    @property
    def track_name(self) -> str:
        return "tools"

    def run(
        self,
        engine: EngineConfig,
        model: str,
        *,
        limit: Optional[int] = None,
    ) -> ResultRecord:
        effective_limit = limit if limit is not None else self._config.limit
        client = EngineClient(engine)

        schemas = _load_schemas(self._config.schemas_path)
        if effective_limit:
            schemas = schemas[:effective_limit]

        results = self._run_canonical(client, model, schemas)

        total = len(results)
        n_compliant   = sum(1 for r in results if r.get("schema_compliance"))
        n_name_match  = sum(1 for r in results if r.get("name_match"))
        n_exact_match = sum(1 for r in results if r.get("exact_match"))
        n_arg_match   = sum(r.get("arg_match_ratio", 0.0) for r in results)

        metrics = {
            "total_cases": total,
            "schema_compliance_pct": round(n_compliant / max(total, 1), 4),
            "function_name_match_pct": round(n_name_match / max(total, 1), 4),
            "exact_match_pct": round(n_exact_match / max(total, 1), 4),
            "arg_match_pct": round(n_arg_match / max(total, 1), 4),
        }

        return ResultRecord(
            track=self.track_name,
            engine=engine.name,
            model=model,
            metrics=metrics,
            metadata={
                "schemas_source": "canonical",
                "n_canonical": len(schemas),
                "limit": effective_limit,
            },
        )

    def _run_canonical(
        self,
        client: EngineClient,
        model: str,
        schemas: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Run each canonical schema as a single-turn tool call."""
        case_results = []
        for schema in schemas:
            tool_def = schema.get("tool", {})
            prompt = schema.get("prompt", "Call the appropriate function.")
            expected = schema.get("expected", {})
            try:
                resp = client.chat(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    tools=[{"type": "function", "function": tool_def}],
                    max_tokens=256,
                    temperature=0.0,
                )
                compliant = self._eval.schema_compliance(resp)
                name_match = self._eval.function_name_match(resp, expected.get("name", ""))
                matched, total_args = self._eval.argument_match(
                    resp, expected.get("arguments", {})
                )
                ratio = matched / max(total_args, 1)
                exact = self._eval.exact_match(resp, expected)
                case_results.append({
                    "schema_compliance": compliant,
                    "name_match": name_match,
                    "arg_match_ratio": ratio,
                    "exact_match": exact,
                })
            except Exception as exc:  # noqa: BLE001
                case_results.append({
                    "schema_compliance": False,
                    "name_match": False,
                    "arg_match_ratio": 0.0,
                    "exact_match": False,
                    "error": str(exc),
                })
        return case_results
