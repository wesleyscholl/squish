"""
tests/test_tool_calling_ext.py

Extended coverage for squish/tool_calling.py:
  - _extract_json_objects with escaped quotes    (lines 123-128)
  - _is_tool_call with non-dict/non-list         (line 161)
  - format_tool_calls with non-dict/non-str args (line 232)
"""
from __future__ import annotations

import json

import pytest

from squish.tool_calling import _extract_json_objects, _is_tool_call, build_tool_calls_response


# ── _extract_json_objects with escape sequences ───────────────────────────────

class TestExtractJsonObjectsEscapes:
    def test_escaped_quote_inside_string(self):
        """Backslash-escaped quote inside JSON string (lines 123-128)."""
        text = '{"key": "val\\"ue"}'
        results = _extract_json_objects(text)
        assert len(results) >= 1
        # Parse it to confirm it's valid
        parsed = json.loads(results[0])
        assert parsed["key"] == 'val"ue'

    def test_escaped_backslash_inside_string(self):
        """Double backslash inside a JSON string."""
        text = '{"path": "C:\\\\Users\\\\test"}'
        results = _extract_json_objects(text)
        assert len(results) >= 1

    def test_escaped_quote_in_nested_string(self):
        """Nested object with escaped quote."""
        text = '{"outer": {"inner": "he said \\"hello\\""}, "x": 1}'
        results = _extract_json_objects(text)
        assert len(results) >= 1

    def test_backslash_only_inside_string_not_outside(self):
        """Backslash outside a string is not treated as escape."""
        text = '{"key": "normal"}\\extra'
        results = _extract_json_objects(text)
        assert len(results) >= 1
        assert results[0] == '{"key": "normal"}'

    def test_string_with_multiple_escapes(self):
        """Multiple escape sequences in same string."""
        text = '{"msg": "\\"hello\\" and \\"world\\""}'
        results = _extract_json_objects(text)
        assert len(results) >= 1

    def test_empty_string_returns_empty(self):
        assert _extract_json_objects("") == []

    def test_plain_text_no_json(self):
        assert _extract_json_objects("hello world") == []


# ── _is_tool_call with various types ──────────────────────────────────────────

class TestIsToolCall:
    def test_returns_false_for_string(self):
        """Non-dict non-list → False (line 161)."""
        assert _is_tool_call("hello") is False

    def test_returns_false_for_int(self):
        assert _is_tool_call(42) is False

    def test_returns_false_for_none(self):
        assert _is_tool_call(None) is False

    def test_returns_true_for_valid_dict(self):
        assert _is_tool_call({"name": "foo", "arguments": {}}) is True

    def test_returns_false_for_dict_missing_arguments(self):
        assert _is_tool_call({"name": "foo"}) is False

    def test_returns_true_for_valid_list(self):
        calls = [{"name": "foo", "arguments": {}}]
        assert _is_tool_call(calls) is True

    def test_returns_false_for_empty_list(self):
        """Empty list → False (len(obj) > 0 fails)."""
        assert _is_tool_call([]) is False

    def test_returns_false_for_list_with_non_dicts(self):
        assert _is_tool_call(["hello"]) is False


# ── build_tool_calls_response with various args types ───────────────────────────────

class TestFormatToolCallsArgTypes:
    def test_dict_args(self):
        """dict args → JSON string (line 228)."""
        raw = [{"name": "my_func", "arguments": {"x": 1}}]
        result = build_tool_calls_response(raw)
        assert len(result) == 1
        args = json.loads(result[0]["function"]["arguments"])
        assert args["x"] == 1

    def test_string_args_passthrough(self):
        """str args → already serialized, pass through (line 230)."""
        raw = [{"name": "my_func", "arguments": '{"x": 2}'}]
        result = build_tool_calls_response(raw)
        assert result[0]["function"]["arguments"] == '{"x": 2}'

    def test_non_dict_non_str_args(self):
        """list args → json.dumps (line 232)."""
        raw = [{"name": "my_func", "arguments": [1, 2, 3]}]
        result = build_tool_calls_response(raw)
        args_str = result[0]["function"]["arguments"]
        parsed = json.loads(args_str)
        assert parsed == [1, 2, 3]

    def test_int_args(self):
        """int args → json.dumps (line 232)."""
        raw = [{"name": "my_func", "arguments": 42}]
        result = build_tool_calls_response(raw)
        args_str = result[0]["function"]["arguments"]
        assert json.loads(args_str) == 42

    def test_result_has_id_type_function(self):
        raw = [{"name": "test", "arguments": {}}]
        result = build_tool_calls_response(raw)
        assert result[0]["type"] == "function"
        assert result[0]["id"].startswith("call_")
        assert result[0]["function"]["name"] == "test"


# ── parse_tool_calls fenced JSON loop continues (branch [181, 179]) ───────────

class TestParseFencedJsonLoop:
    def test_fenced_non_tool_call_then_valid_tool_call(self):
        """First fenced block parses but isn't a tool call → loop continues (line 181→179)."""
        from squish.tool_calling import parse_tool_calls

        text = (
            "Here is some context:\n"
            "```json\n"
            '{"some_key": "not_a_tool_call"}\n'
            "```\n"
            "And the actual call:\n"
            "```json\n"
            '{"name": "get_weather", "arguments": {"city": "NYC"}}\n'
            "```\n"
        )
        result = parse_tool_calls(text)
        assert result is not None
        assert result[0]["name"] == "get_weather"

    def test_fenced_non_tool_call_only_returns_none(self):
        """Text with fenced JSON that parses but is not a tool call and no other match."""
        from squish.tool_calling import parse_tool_calls

        text = '```json\n{"key": "value", "not_a_call": true}\n```'
        result = parse_tool_calls(text)
        # No valid tool call found anywhere
        assert result is None
