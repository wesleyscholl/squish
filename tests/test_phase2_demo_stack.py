"""
tests/test_phase2_demo_stack.py

Unit tests for Phase 2.2 — local demo stack:
  - squish.tool_calling  (pure Python, no MLX needed)
  - squish.ollama_compat (FastAPI TestClient, no model needed)
  - squish.scheduler     (data-path only, no live model)

Run with:
    pytest tests/test_phase2_demo_stack.py -v
"""
import json
import queue
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

# ─────────────────────────────────────────────────────────────────────────────
# 1.  squish.tool_calling — format + parse + build
# ─────────────────────────────────────────────────────────────────────────────

from squish.tool_calling import (
    format_tools_prompt,
    parse_tool_calls,
    build_tool_calls_response,
)

WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Return current weather for a city",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name"},
                "units": {"type": "string", "enum": ["celsius", "fahrenheit"]},
            },
            "required": ["city"],
        },
    },
}

CALC_TOOL = {
    "type": "function",
    "function": {
        "name": "calculator",
        "description": "Evaluate a math expression",
        "parameters": {
            "type": "object",
            "properties": {"expression": {"type": "string"}},
            "required": ["expression"],
        },
    },
}


class TestFormatToolsPrompt:
    def test_no_existing_system(self):
        msgs = [{"role": "user", "content": "What is the weather in Paris?"}]
        result = format_tools_prompt(msgs, [WEATHER_TOOL])
        # Should prepend a system message
        assert result[0]["role"] == "system"
        assert "get_weather" in result[0]["content"]
        # Raw {schemas} format placeholder should be replaced, not literal
        assert "{schemas}" not in result[0]["content"]
        assert result[1]["role"] == "user"

    def test_existing_system_prepended(self):
        msgs = [
            {"role": "system", "content": "You are a helpful bot."},
            {"role": "user", "content": "Help me"},
        ]
        result = format_tools_prompt(msgs, [WEATHER_TOOL])
        assert result[0]["role"] == "system"
        assert "get_weather" in result[0]["content"]
        assert "You are a helpful bot." in result[0]["content"]

    def test_empty_tools_returns_unchanged(self):
        msgs = [{"role": "user", "content": "hi"}]
        result = format_tools_prompt(msgs, [])
        assert result == msgs

    def test_multiple_tools_all_present(self):
        msgs = [{"role": "user", "content": "calc + weather?"}]
        result = format_tools_prompt(msgs, [WEATHER_TOOL, CALC_TOOL])
        sys_content = result[0]["content"]
        assert "get_weather" in sys_content
        assert "calculator" in sys_content

    def test_does_not_mutate_original(self):
        msgs = [{"role": "user", "content": "hi"}]
        original_id = id(msgs)
        result = format_tools_prompt(msgs, [WEATHER_TOOL])
        assert id(result) != original_id  # new list
        assert len(msgs) == 1             # original unchanged


class TestParseToolCalls:
    def test_bare_json_object(self):
        text = '{"name": "get_weather", "arguments": {"city": "Paris"}}'
        result = parse_tool_calls(text)
        assert result is not None
        assert len(result) == 1
        assert result[0]["name"] == "get_weather"
        assert result[0]["arguments"]["city"] == "Paris"

    def test_fenced_json_block(self):
        text = '```json\n{"name": "calculator", "arguments": {"expression": "2+2"}}\n```'
        result = parse_tool_calls(text)
        assert result is not None
        assert result[0]["name"] == "calculator"

    def test_json_array_of_calls(self):
        data = [
            {"name": "get_weather", "arguments": {"city": "Tokyo"}},
            {"name": "calculator",  "arguments": {"expression": "3*7"}},
        ]
        text = json.dumps(data)
        result = parse_tool_calls(text)
        assert result is not None
        assert len(result) == 2
        assert result[0]["name"] == "get_weather"
        assert result[1]["name"] == "calculator"

    def test_plain_text_returns_none(self):
        text = "Sure, the weather in Paris is sunny and 22°C today."
        result = parse_tool_calls(text)
        assert result is None

    def test_partial_json_returns_none(self):
        text = '{"name": "get_weather"}'  # missing "arguments"
        result = parse_tool_calls(text)
        assert result is None

    def test_json_embedded_in_prose(self):
        text = 'I will call this function:\n{"name": "get_weather", "arguments": {"city": "Berlin"}}\nDone.'
        result = parse_tool_calls(text)
        assert result is not None
        assert result[0]["arguments"]["city"] == "Berlin"

    def test_already_string_arguments(self):
        """Model may return arguments as pre-serialised JSON string."""
        text = '{"name": "get_weather", "arguments": "{\\"city\\": \\"Oslo\\"}"}'
        result = parse_tool_calls(text)
        # arguments is a string here — parse_tool_calls should still detect it
        # because 'name' and 'arguments' are present
        assert result is not None


class TestBuildToolCallsResponse:
    def test_single_call_shape(self):
        raw = [{"name": "get_weather", "arguments": {"city": "Paris"}}]
        tc  = build_tool_calls_response(raw)
        assert len(tc) == 1
        assert tc[0]["type"] == "function"
        assert tc[0]["function"]["name"] == "get_weather"
        assert tc[0]["id"].startswith("call_")
        # arguments must be a JSON string
        args = json.loads(tc[0]["function"]["arguments"])
        assert args["city"] == "Paris"

    def test_multiple_calls(self):
        raw = [
            {"name": "get_weather", "arguments": {"city": "Paris"}},
            {"name": "calculator",  "arguments": {"expression": "1+1"}},
        ]
        tc = build_tool_calls_response(raw)
        assert len(tc) == 2
        assert tc[0]["function"]["name"] == "get_weather"
        assert tc[1]["function"]["name"] == "calculator"

    def test_string_arguments_passed_through(self):
        """If arguments is already a JSON string, it should be preserved."""
        raw = [{"name": "calc", "arguments": '{"expression": "4*4"}'}]
        tc  = build_tool_calls_response(raw)
        assert tc[0]["function"]["arguments"] == '{"expression": "4*4"}'

    def test_unique_ids(self):
        raw = [
            {"name": "a", "arguments": {}},
            {"name": "b", "arguments": {}},
        ]
        tc = build_tool_calls_response(raw)
        assert tc[0]["id"] != tc[1]["id"]


# ─────────────────────────────────────────────────────────────────────────────
# 2.  squish.ollama_compat — endpoint shape tests (no real model)
# ─────────────────────────────────────────────────────────────────────────────

from fastapi import FastAPI
from fastapi.testclient import TestClient
from squish.ollama_compat import mount_ollama


def _make_mock_state(loaded: bool = False):
    state = MagicMock()
    state.model      = MagicMock() if loaded else None
    state.model_name = "squish-test"
    state.tokenizer  = None
    return state


def _make_mock_generate():
    """Returns a function that yields a short canned response."""
    def _generate(prompt, max_tokens, temperature, top_p, stop, seed):
        for word in ["Hello", " world", "!"]:
            yield (word, None)
        yield ("", "stop")
    return _generate


def _make_app(loaded: bool = True) -> tuple[FastAPI, TestClient]:
    app   = FastAPI()
    state = _make_mock_state(loaded)
    gen   = _make_mock_generate()

    # Tokenizer stub with apply_chat_template
    tok = MagicMock()
    tok.apply_chat_template = lambda msgs, **kw: " ".join(m["content"] for m in msgs)
    state.tokenizer = tok

    mount_ollama(
        app,
        get_state     = lambda: state,
        get_generate  = lambda: gen,
        get_tokenizer = lambda: tok,
        models_dir    = Path("/nonexistent"),  # skip real scan
    )
    client = TestClient(app, raise_server_exceptions=False)
    return app, client


class TestOllamaVersion:
    def test_returns_version(self):
        _, client = _make_app()
        r = client.get("/api/version")
        assert r.status_code == 200
        data = r.json()
        assert "version" in data


class TestOllamaTags:
    def test_returns_models_list(self):
        _, client = _make_app()
        r = client.get("/api/tags")
        assert r.status_code == 200
        data = r.json()
        assert "models" in data
        assert isinstance(data["models"], list)

    def test_model_has_required_fields(self):
        _, client = _make_app()
        r = client.get("/api/tags")
        for m in r.json()["models"]:
            assert "name"    in m
            assert "size"    in m
            assert "details" in m


class TestOllamaShow:
    def test_returns_details(self):
        _, client = _make_app()
        r = client.post("/api/show", json={"name": "squish-test"})
        assert r.status_code == 200
        data = r.json()
        assert "details" in data
        assert "modelfile" in data


class TestOllamaPull:
    def test_returns_stream(self):
        _, client = _make_app()
        r = client.post("/api/pull", json={"name": "some-model:latest"}, timeout=5)
        assert r.status_code == 200
        # Each line should be valid JSON
        for line in r.text.strip().splitlines():
            obj = json.loads(line)
            assert "status" in obj


class TestOllamaGenerate:
    def test_stream_false(self):
        _, client = _make_app(loaded=True)
        r = client.post("/api/generate", json={
            "model": "squish-test", "prompt": "Say hello", "stream": False,
        })
        assert r.status_code == 200
        data = r.json()
        assert "response" in data
        assert data["done"] is True
        assert data["response"] == "Hello world!"

    def test_stream_true_ndjson(self):
        _, client = _make_app(loaded=True)
        r = client.post("/api/generate", json={
            "model": "squish-test", "prompt": "hi", "stream": True,
        })
        assert r.status_code == 200
        lines = [l for l in r.text.strip().splitlines() if l.strip()]
        assert len(lines) >= 2
        # All lines after first should parse cleanly
        for line in lines:
            obj = json.loads(line)
            assert "model" in obj
        # Last line should be done=True
        last = json.loads(lines[-1])
        assert last["done"] is True

    def test_model_not_loaded_503(self):
        _, client = _make_app(loaded=False)
        r = client.post("/api/generate", json={"prompt": "hi"})
        assert r.status_code == 503

    def test_empty_prompt_400(self):
        _, client = _make_app(loaded=True)
        r = client.post("/api/generate", json={"prompt": ""})
        assert r.status_code == 400


class TestOllamaChat:
    def test_stream_false(self):
        _, client = _make_app(loaded=True)
        r = client.post("/api/chat", json={
            "model": "squish-test",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": False,
        })
        assert r.status_code == 200
        data = r.json()
        assert data["done"] is True
        assert "message" in data
        assert data["message"]["role"] == "assistant"
        assert "Hello" in data["message"]["content"]

    def test_stream_true_message_format(self):
        _, client = _make_app(loaded=True)
        r = client.post("/api/chat", json={
            "model": "squish-test",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        })
        assert r.status_code == 200
        lines = [l for l in r.text.strip().splitlines() if l.strip()]
        for line in lines:
            obj = json.loads(line)
            if not obj.get("done"):
                assert "message" in obj
                assert obj["message"]["role"] == "assistant"

    def test_empty_messages_400(self):
        _, client = _make_app(loaded=True)
        r = client.post("/api/chat", json={"messages": []})
        assert r.status_code == 400


# ─────────────────────────────────────────────────────────────────────────────
# 3.  squish.scheduler — data path (no model inference)
# ─────────────────────────────────────────────────────────────────────────────

from squish.scheduler import _Request


class TestRequestDataclass:
    def test_defaults(self):
        q = queue.SimpleQueue()
        r = _Request(
            request_id="req-001",
            input_ids=[1, 2, 3],
            max_tokens=64,
            temperature=0.7,
            top_p=0.9,
            stop_ids=[],
            seed=None,
            out_queue=q,
        )
        assert r.generated_ids == []
        assert r.done is False

    def test_out_queue_get_put(self):
        """Verify the output queue mechanism works end-to-end without a model."""
        q = queue.SimpleQueue()
        r = _Request(
            request_id="req-002",
            input_ids=[1, 2, 3], max_tokens=8, temperature=0.7, top_p=0.9,
            stop_ids=[], seed=None, out_queue=q,
        )
        # Simulate scheduler putting tokens
        q.put(("hello ", None))
        q.put((" world", None))
        q.put(("", "stop"))

        collected = []
        while True:
            tok, finish = q.get()
            collected.append(tok)
            if finish is not None:
                break

        assert "".join(collected).strip() == "hello  world"


class TestBatchSchedulerInit:
    def test_init_stores_params(self):
        from squish.scheduler import BatchScheduler
        mock_model = MagicMock()
        mock_tok   = MagicMock()
        mock_tok.convert_tokens_to_ids = MagicMock(return_value=[])
        sched = BatchScheduler(mock_model, mock_tok, max_batch_size=4, batch_window_ms=10)
        s = sched.stats()
        assert s["max_batch_size"] == 4
        assert not sched.is_running()

    def test_stats_before_start(self):
        from squish.scheduler import BatchScheduler
        mock_model = MagicMock()
        mock_tok   = MagicMock()
        sched = BatchScheduler(mock_model, mock_tok)
        s = sched.stats()
        assert s["total_requests"] == 0
        assert s["total_batches"]  == 0


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Integration smoke — tool_calling round-trip
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# 5.  /v1/embeddings — last-hidden-state fix (pre-launch patch)
# ─────────────────────────────────────────────────────────────────────────────

class TestEmbeddingsEndpoint:
    """
    Tests for the /v1/embeddings endpoint's last-hidden-state fix.

    The fix changed the embedding extraction order to:
      1. model.model(x)       — last hidden state (preferred)
      2. model.model.embed_tokens(x) — input token embeddings (fallback)
      3. model(x)             — logits mean-pool (last resort)
    """

    def _make_server_state(self, hidden_dim: int = 64, seq_len: int = 5):
        """Return (mock_model, mock_tokenizer) that produce real MLX tensors."""
        mx = pytest.importorskip("mlx.core")
        from unittest.mock import MagicMock

        mock_model = MagicMock()
        # model.model(x) returns (1, seq, hidden_dim) — non-zero so norm > 0
        mock_model.model.return_value = mx.ones([1, seq_len, hidden_dim])

        mock_tok = MagicMock()
        mock_tok.encode.return_value = list(range(1, seq_len + 1))
        return mock_model, mock_tok

    def test_happy_path_shape_and_normalization(self):
        """Preferred path: last hidden state → normalized embedding of correct dim."""
        import squish.server as _srv
        from fastapi.testclient import TestClient

        mock_model, mock_tok = self._make_server_state(hidden_dim=64)
        orig = _srv._state
        _srv._state = _srv._ModelState()
        _srv._state.model      = mock_model
        _srv._state.tokenizer  = mock_tok
        _srv._state.model_name = "test-embed-model"
        try:
            client = TestClient(_srv.app, raise_server_exceptions=False)
            resp = client.post(
                "/v1/embeddings",
                json={"input": "hello world", "model": "test"},
            )
            assert resp.status_code == 200, f"Got {resp.status_code}: {resp.text[:200]}"
            data = resp.json()
            assert data["object"] == "list"
            assert len(data["data"]) == 1
            emb = data["data"][0]["embedding"]
            assert len(emb) == 64, f"Expected dim=64, got {len(emb)}"
            # L2-normalized: sum of squares ≈ 1.0
            import math
            norm_sq = sum(v * v for v in emb)
            assert abs(norm_sq - 1.0) < 1e-4, f"Embedding not L2-normalized: norm²={norm_sq:.6f}"
        finally:
            _srv._state = orig

    def test_batch_input(self):
        """Multiple strings in input → multiple embeddings returned."""
        import squish.server as _srv
        from fastapi.testclient import TestClient

        mock_model, mock_tok = self._make_server_state(hidden_dim=32)
        orig = _srv._state
        _srv._state = _srv._ModelState()
        _srv._state.model      = mock_model
        _srv._state.tokenizer  = mock_tok
        _srv._state.model_name = "test-model"
        try:
            client = TestClient(_srv.app, raise_server_exceptions=False)
            resp = client.post(
                "/v1/embeddings",
                json={"input": ["hello", "world", "foo"], "model": "test"},
            )
            assert resp.status_code == 200
            data = resp.json()
            assert len(data["data"]) == 3
            indices = [d["index"] for d in data["data"]]
            assert indices == [0, 1, 2]
        finally:
            _srv._state = orig

    def test_no_model_returns_503(self):
        """Endpoint returns 503 when no model is loaded."""
        import squish.server as _srv
        from fastapi.testclient import TestClient

        orig = _srv._state
        _srv._state = _srv._ModelState()   # model=None
        try:
            client = TestClient(_srv.app, raise_server_exceptions=False)
            resp = client.post(
                "/v1/embeddings",
                json={"input": "test", "model": "test"},
            )
            assert resp.status_code == 503
        finally:
            _srv._state = orig


class TestToolCallingRoundTrip:
    """End-to-end: format → (simulated model output) → parse → build."""

    def test_full_cycle(self):
        tools = [WEATHER_TOOL]
        messages = [{"role": "user", "content": "What is the weather in Tokyo?"}]

        # 1. Format: inject tool schema
        aug_msgs = format_tools_prompt(messages, tools)
        assert aug_msgs[0]["role"] == "system"
        assert "get_weather" in aug_msgs[0]["content"]

        # 2. Simulate model output (a JSON tool call)
        model_output = '{"name": "get_weather", "arguments": {"city": "Tokyo", "units": "celsius"}}'

        # 3. Parse
        raw = parse_tool_calls(model_output)
        assert raw is not None
        assert raw[0]["name"] == "get_weather"

        # 4. Build OpenAI response format
        tc = build_tool_calls_response(raw)
        assert tc[0]["type"] == "function"
        args = json.loads(tc[0]["function"]["arguments"])
        assert args["city"] == "Tokyo"
        assert args["units"] == "celsius"

    def test_plain_text_skips_tool_path(self):
        """If model output has no JSON, all steps gracefully produce None."""
        output = "The weather in Tokyo is lovely and sunny today, around 24°C."
        assert parse_tool_calls(output) is None
